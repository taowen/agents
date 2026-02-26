/**
 * SessionContext: encapsulates all per-session state and initialization logic.
 *
 * Replaces the 15+ private fields and scattered "if (!this.userId) { getUserId(); doInitBash(); }"
 * guard pattern that previously lived in ChatAgentBase. Now all callers use:
 *
 *   await this.session.ensureReady(request?);
 *
 * No public API or storage format changes — pure internal refactor.
 */

import type { Bash, MountableFs } from "just-bash";
import type { ToolSet } from "ai";
import { initBash, doFstabMount } from "vfs";
import type { FsBindings } from "vfs";
import {
  createSessionsCommand,
  createSearchCommand,
  createWebFetchCommand
} from "./session-commands";
import {
  getCachedLlmConfig,
  getLlmModel,
  type LlmConfigCache
} from "./llm-config";
import {
  checkQuota,
  archiveSessionUsage,
  type QuotaCache
} from "./usage-tracker";

export class UserFacingError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UserFacingError";
  }
}

export class SessionContext {
  // ---- Identity ----
  userId: string | null = null;
  sessionUuid: string | null = null;

  // ---- Bash / Filesystem ----
  bash!: Bash;
  mountableFs!: MountableFs;
  private mounted = false;
  private mountPromise: Promise<void> | null = null;

  // ---- LLM config ----
  cachedLlmConfig: LlmConfigCache = null;
  cachedLlmModel: ReturnType<typeof getLlmModel> | null = null;
  cachedModelId: string | null = null;

  // ---- Prompt / tools cache ----
  cachedSystemPrompt: string | null = null;
  cachedDynamicContext: string | null = null;
  cachedTools: ToolSet | null = null;
  mcpServersLoaded = false;

  // ---- Quota ----
  private quotaCheckCache: QuotaCache | null = null;

  constructor(
    private ctx: DurableObjectState,
    private env: Env
  ) {}

  // ---- FsBindings ----

  get fsBindings(): FsBindings {
    return {
      db: this.env.DB,
      r2: this.env.R2,
      googleClientId: this.env.GOOGLE_CLIENT_ID,
      googleClientSecret: this.env.GOOGLE_CLIENT_SECRET
    };
  }

  /** Short session directory name derived from the DO ID. */
  get sessionDir(): string {
    return this.ctx.id.toString().slice(0, 12);
  }

  // ---- Unified initialization entry point ----

  /**
   * Ensure userId, bash, and sessionUuid are ready.
   * Safe to call multiple times — idempotent after first initialization.
   * Replaces the scattered guard blocks in onChatMessage, executeScheduledTask, etc.
   */
  async ensureReady(request?: Request): Promise<void> {
    if (!this.userId) {
      const uid = await this.getUserId(request);
      this.doInitBash(uid);
    }
    if (!this.sessionUuid) {
      await this.getSessionUuid(request);
    }
  }

  // ---- UserId ----

  async getUserId(request?: Request): Promise<string> {
    if (request) {
      const uid = request.headers.get("x-user-id");
      if (uid) {
        await this.ctx.storage.put("userId", uid);
        return uid;
      }
    }
    const stored = await this.ctx.storage.get<string>("userId");
    if (stored) return stored;
    throw new Error("No userId available");
  }

  // ---- SessionUuid ----

  async getSessionUuid(request?: Request): Promise<string | null> {
    if (request) {
      const sid = request.headers.get("x-session-id");
      if (sid) {
        await this.ctx.storage.put("sessionUuid", sid);
        this.sessionUuid = sid;
        return sid;
      }
    }
    if (this.sessionUuid) return this.sessionUuid;
    const stored = await this.ctx.storage.get<string>("sessionUuid");
    if (stored) {
      this.sessionUuid = stored;
      return stored;
    }
    return null;
  }

  // ---- Timezone ----

  async getTimezone(): Promise<string> {
    const stored = await this.ctx.storage.get<string>("timezone");
    return stored || "UTC";
  }

  // ---- Bash init ----

  doInitBash(userId: string): void {
    if (this.bash && this.userId === userId) return;
    this.userId = userId;
    const { bash, mountableFs } = initBash({
      bindings: this.fsBindings,
      userId,
      customCommands: [
        createSessionsCommand(this.env.DB, userId, this.env.ChatAgent),
        createSearchCommand(this.env),
        createWebFetchCommand(this.env)
      ]
    });
    this.bash = bash;
    this.mountableFs = mountableFs;
    this.mounted = false;
    this.mountPromise = null;
  }

  // ---- Fstab mount ----

  async ensureMounted(): Promise<void> {
    if (this.mounted) return;
    if (!this.mountPromise) {
      this.mountPromise = doFstabMount(
        this.mountableFs,
        this.fsBindings,
        this.userId!
      ).then(
        () => {
          this.mounted = true;
        },
        (err) => {
          console.error("ensureMounted: fstab mount failed:", err);
          this.mountPromise = null;
          throw err;
        }
      );
    }
    await this.mountPromise;
  }

  /** Invalidate cached LLM config so the next request re-reads /etc/llm.json. */
  invalidateLlmConfigCache(): void {
    this.cachedLlmConfig = null;
  }

  // ---- Quota + LLM model resolution ----

  async resolveQuotaAndModel(): Promise<{
    llmModel: ReturnType<typeof getLlmModel>;
    apiKeyType: "builtin" | "custom";
    isBuiltinKey: boolean;
    modelId: string;
  }> {
    const { data: llmConfig, cache } = await getCachedLlmConfig(
      this.env.DB,
      this.userId!,
      this.cachedLlmConfig
    );
    this.cachedLlmConfig = cache;
    const isBuiltinKey = llmConfig === null;

    if (isBuiltinKey) {
      this.quotaCheckCache = await checkQuota(
        this.env.DB,
        this.userId!,
        this.quotaCheckCache
      );
      if (this.quotaCheckCache.exceeded) {
        throw new UserFacingError(
          "You have exceeded the builtin API key usage quota. " +
            "Please configure your own API key in Settings to continue using the service."
        );
      }
    }

    const llmModel = getLlmModel(this.env, llmConfig);
    const modelId = llmConfig?.model ?? this.env.BUILTIN_LLM_MODEL;
    this.cachedModelId = modelId;
    return {
      llmModel,
      apiKeyType: isBuiltinKey ? "builtin" : "custom",
      isBuiltinKey,
      modelId
    };
  }

  // ---- Usage archiver factory ----

  createUsageArchiver(): () => Promise<void> {
    const { DB } = this.env;
    const sql = this.ctx.storage.sql;
    const userId = this.userId!;
    const sessionId = this.sessionUuid;
    return async () => {
      try {
        await archiveSessionUsage(DB, sql, userId, sessionId);
      } catch (e) {
        console.error("onFinish usage_archive write failed:", e);
      }
    };
  }
}
