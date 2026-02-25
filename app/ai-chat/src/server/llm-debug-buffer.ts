import * as Sentry from "@sentry/cloudflare";

// Each LLM streamText call
export interface LlmInteractionEntry {
  type: "llm";
  timestamp: string;
  traceId: string;
  spanId: string;
  request: {
    systemPrompt: string;
    dynamicContext: string;
    messages: unknown[];
    toolNames: string[];
    modelId: string;
  };
  response: {
    text: string;
    finishReason: string;
    usage: { inputTokens: number; outputTokens: number };
    stepCount: number;
    steps: Array<{
      text: string;
      finishReason: string;
      toolCalls: Array<{ toolName: string; args: unknown }>;
      toolResults: Array<{ toolName: string; output: unknown }>;
      usage: { inputTokens: number; outputTokens: number };
    }>;
  } | null;
  error?: string;
}

// Each cross-DO call (send_to_device → dispatch-task)
export interface DoCallEntry {
  type: "do_call";
  timestamp: string;
  traceId: string;
  spanId: string;
  direction: "outbound" | "inbound";
  targetDoId?: string;
  endpoint: string;
  request: unknown;
  response: unknown;
  durationMs: number;
  error?: string;
}

export interface ScheduleEntry {
  type: "schedule";
  timestamp: string;
  action: "create" | "cancel" | "execute" | "session_deleted";
  taskId: string;
  description: string;
  cron?: string;
  scheduledAt?: string;
  error?: string;
}

export interface DeviceConnectionEntry {
  type: "device_connection";
  timestamp: string;
  event: "connect" | "ready" | "disconnect" | "error";
  deviceName?: string;
  deviceId?: string;
}

export type DebugEntry =
  | LlmInteractionEntry
  | DoCallEntry
  | ScheduleEntry
  | DeviceConnectionEntry;

/**
 * SQLite-backed ring buffer that stores the last N debug entries.
 * Persists across DO hibernation via `ctx.storage.sql`.
 */
export class DebugRingBuffer {
  private sql: SqlStorage;
  private maxSize: number;
  private initialized = false;

  constructor(maxSize: number = 20, sql: SqlStorage) {
    this.maxSize = maxSize;
    this.sql = sql;
  }

  private ensureTable(): void {
    if (this.initialized) return;
    this.sql.exec(
      `CREATE TABLE IF NOT EXISTS debug_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT NOT NULL
      )`
    );
    this.initialized = true;
  }

  /** Push an entry and return its row id. */
  push(entry: DebugEntry): number {
    this.ensureTable();
    const cursor = this.sql.exec(
      `INSERT INTO debug_entries (data) VALUES (?)`,
      JSON.stringify(entry)
    );
    const rowId = Number(
      cursor.rowsWritten > 0
        ? this.sql.exec(`SELECT last_insert_rowid() as id`).one().id
        : 0
    );
    // Trim old rows beyond maxSize
    this.sql.exec(
      `DELETE FROM debug_entries WHERE id NOT IN (SELECT id FROM debug_entries ORDER BY id DESC LIMIT ?)`,
      this.maxSize
    );
    return rowId;
  }

  /** Update the response field of an LLM entry by row id. */
  updateResponse(
    rowId: number,
    response: LlmInteractionEntry["response"]
  ): void {
    this.ensureTable();
    this.sql.exec(
      `UPDATE debug_entries SET data = json_set(data, '$.response', json(?)) WHERE id = ?`,
      JSON.stringify(response),
      rowId
    );
  }

  /** Update the error field of an LLM entry by row id. */
  updateError(rowId: number, error: string): void {
    this.ensureTable();
    this.sql.exec(
      `UPDATE debug_entries SET data = json_set(data, '$.error', ?) WHERE id = ?`,
      error,
      rowId
    );
  }

  getAll(): DebugEntry[] {
    this.ensureTable();
    const rows = this.sql
      .exec(`SELECT data FROM debug_entries ORDER BY id ASC`)
      .toArray();
    return rows.map((row) => JSON.parse(row.data as string) as DebugEntry);
  }

  get size(): number {
    this.ensureTable();
    return Number(
      this.sql.exec(`SELECT COUNT(*) as cnt FROM debug_entries`).one().cnt
    );
  }
}

/** Build a request snapshot for an LLM interaction entry. */
export function buildRequestSnapshot(
  systemPrompt: string,
  dynamicContext: string,
  messages: unknown[],
  tools: Record<string, unknown>,
  modelId: string
): LlmInteractionEntry["request"] {
  return {
    systemPrompt,
    dynamicContext,
    messages,
    toolNames: Object.keys(tools),
    modelId
  };
}

/** Build a response snapshot from the AI SDK onFinish event. */
export function buildResponseSnapshot(event: {
  text?: string;
  finishReason?: string;
  usage?: { inputTokens?: number; outputTokens?: number };
  steps?: Array<{
    text?: string;
    finishReason?: string;
    toolCalls?: Array<{ toolName?: string; args?: unknown }>;
    toolResults?: Array<{ toolName?: string; result?: unknown }>;
    usage?: { inputTokens?: number; outputTokens?: number };
  }>;
}): LlmInteractionEntry["response"] {
  return {
    text: event.text ?? "",
    finishReason: event.finishReason ?? "unknown",
    usage: {
      inputTokens: event.usage?.inputTokens ?? 0,
      outputTokens: event.usage?.outputTokens ?? 0
    },
    stepCount: event.steps?.length ?? 0,
    steps: (event.steps ?? []).map((step) => ({
      text: step.text ?? "",
      finishReason: step.finishReason ?? "unknown",
      toolCalls: (step.toolCalls ?? []).map((tc) => ({
        toolName: tc.toolName ?? "",
        args: tc.args
      })),
      toolResults: (step.toolResults ?? []).map((tr) => ({
        toolName: tr.toolName ?? "",
        output: tr.result
      })),
      usage: {
        inputTokens: step.usage?.inputTokens ?? 0,
        outputTokens: step.usage?.outputTokens ?? 0
      }
    }))
  };
}

/** Get Sentry trace context from the active span, or empty strings if none. */
export function getSentryTraceContext(): { traceId: string; spanId: string } {
  try {
    const span = Sentry.getActiveSpan();
    if (span) {
      const ctx = span.spanContext();
      return { traceId: ctx.traceId || "", spanId: ctx.spanId || "" };
    }
  } catch {
    // Sentry not initialized or no active span
  }
  return { traceId: "", spanId: "" };
}

/**
 * Instrument an LLM call with debug buffer tracking.
 * Returns no-op callbacks when debugBuffer is undefined.
 */
export function instrumentLlmCall(
  debugBuffer: DebugRingBuffer | undefined,
  request: LlmInteractionEntry["request"]
): {
  onResponse: (event: Parameters<typeof buildResponseSnapshot>[0]) => void;
  onError: (error: string) => void;
  hookAbort: (signal?: AbortSignal) => void;
} {
  if (!debugBuffer) {
    return { onResponse: () => {}, onError: () => {}, hookAbort: () => {} };
  }

  const { traceId, spanId } = getSentryTraceContext();
  const entryRowId = debugBuffer.push({
    type: "llm",
    timestamp: new Date().toISOString(),
    traceId,
    spanId,
    request,
    response: null
  });

  const onResponse = (event: Parameters<typeof buildResponseSnapshot>[0]) => {
    try {
      debugBuffer.updateResponse(entryRowId, buildResponseSnapshot(event));
    } catch (e) {
      console.error("debug buffer updateResponse failed:", e);
    }
  };

  const onError = (error: string) => {
    try {
      debugBuffer.updateError(entryRowId, error);
    } catch (e) {
      console.error("debug buffer updateError failed:", e);
    }
  };

  const hookAbort = (signal?: AbortSignal) => {
    if (signal) {
      signal.addEventListener("abort", () => onError("aborted"));
    }
  };

  return { onResponse, onError, hookAbort };
}
