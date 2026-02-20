/**
 * Agent class — manages state, queues, and drives the agent loop.
 * Uses AI SDK's LanguageModel interface instead of pi-ai's Model.
 */

import type { LanguageModel, ModelMessage } from "ai";
import {
  agentLoop,
  agentLoopContinue,
  convertToModelMessages
} from "./agent-loop.js";
import type {
  AgentContext,
  AgentEvent,
  AgentLoopConfig,
  AgentMessage,
  AgentState,
  AgentTool,
  ImageContent,
  TextContent
} from "./types.js";

export interface AgentOptions {
  initialState?: Partial<AgentState>;

  /** Converts AgentMessage[] to AI SDK ModelMessage[] before each LLM call. */
  convertToLlm?: (
    messages: AgentMessage[]
  ) => ModelMessage[] | Promise<ModelMessage[]>;

  /** Optional transform applied to context before convertToLlm. */
  transformContext?: (
    messages: AgentMessage[],
    signal?: AbortSignal
  ) => Promise<AgentMessage[]>;

  /** Steering mode: "all" = send all at once, "one-at-a-time" = one per turn. */
  steeringMode?: "all" | "one-at-a-time";

  /** Follow-up mode: "all" = send all at once, "one-at-a-time" = one per turn. */
  followUpMode?: "all" | "one-at-a-time";

  temperature?: number;
  maxTokens?: number;
  providerOptions?: Record<string, Record<string, unknown>>;
}

export class Agent {
  private _state: AgentState = {
    systemPrompt: "",
    model: null,
    tools: [],
    messages: [],
    isStreaming: false,
    streamMessage: null,
    pendingToolCalls: new Set<string>(),
    error: undefined
  };

  private listeners = new Set<(e: AgentEvent) => void>();
  private abortController?: AbortController;
  private convertToLlm: (
    messages: AgentMessage[]
  ) => ModelMessage[] | Promise<ModelMessage[]>;
  private transformContext?: (
    messages: AgentMessage[],
    signal?: AbortSignal
  ) => Promise<AgentMessage[]>;
  private steeringQueue: AgentMessage[] = [];
  private followUpQueue: AgentMessage[] = [];
  private steeringMode: "all" | "one-at-a-time";
  private followUpMode: "all" | "one-at-a-time";
  private runningPrompt?: Promise<void>;
  private resolveRunningPrompt?: () => void;
  private _temperature?: number;
  private _maxTokens?: number;
  private _providerOptions?: Record<string, Record<string, unknown>>;

  constructor(opts: AgentOptions = {}) {
    this._state = { ...this._state, ...opts.initialState };
    this.convertToLlm = opts.convertToLlm || convertToModelMessages;
    this.transformContext = opts.transformContext;
    this.steeringMode = opts.steeringMode || "one-at-a-time";
    this.followUpMode = opts.followUpMode || "one-at-a-time";
    this._temperature = opts.temperature;
    this._maxTokens = opts.maxTokens;
    this._providerOptions = opts.providerOptions;
  }

  // ── Accessors ───────────────────────────────────────────────────────────

  get state(): AgentState {
    return this._state;
  }

  get temperature(): number | undefined {
    return this._temperature;
  }
  set temperature(v: number | undefined) {
    this._temperature = v;
  }

  get maxTokens(): number | undefined {
    return this._maxTokens;
  }
  set maxTokens(v: number | undefined) {
    this._maxTokens = v;
  }

  get providerOptions(): Record<string, Record<string, unknown>> | undefined {
    return this._providerOptions;
  }
  set providerOptions(v: Record<string, Record<string, unknown>> | undefined) {
    this._providerOptions = v;
  }

  // ── Subscriptions ───────────────────────────────────────────────────────

  subscribe(fn: (e: AgentEvent) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  // ── State Mutators ──────────────────────────────────────────────────────

  setSystemPrompt(v: string) {
    this._state.systemPrompt = v;
  }

  setModel(m: LanguageModel) {
    this._state.model = m;
  }

  setTools(t: AgentTool<any>[]) {
    this._state.tools = t;
  }

  replaceMessages(ms: AgentMessage[]) {
    this._state.messages = ms.slice();
  }

  appendMessage(m: AgentMessage) {
    this._state.messages = [...this._state.messages, m];
  }

  clearMessages() {
    this._state.messages = [];
  }

  // ── Steering & Follow-Up ────────────────────────────────────────────────

  setSteeringMode(mode: "all" | "one-at-a-time") {
    this.steeringMode = mode;
  }
  getSteeringMode(): "all" | "one-at-a-time" {
    return this.steeringMode;
  }

  setFollowUpMode(mode: "all" | "one-at-a-time") {
    this.followUpMode = mode;
  }
  getFollowUpMode(): "all" | "one-at-a-time" {
    return this.followUpMode;
  }

  /** Queue a steering message to interrupt the agent mid-run. */
  steer(m: AgentMessage) {
    this.steeringQueue.push(m);
  }

  /** Queue a follow-up message for after the agent finishes. */
  followUp(m: AgentMessage) {
    this.followUpQueue.push(m);
  }

  clearSteeringQueue() {
    this.steeringQueue = [];
  }
  clearFollowUpQueue() {
    this.followUpQueue = [];
  }
  clearAllQueues() {
    this.steeringQueue = [];
    this.followUpQueue = [];
  }

  hasQueuedMessages(): boolean {
    return this.steeringQueue.length > 0 || this.followUpQueue.length > 0;
  }

  private dequeueSteeringMessages(): AgentMessage[] {
    if (this.steeringMode === "one-at-a-time") {
      if (this.steeringQueue.length > 0) {
        const first = this.steeringQueue[0];
        this.steeringQueue = this.steeringQueue.slice(1);
        return [first];
      }
      return [];
    }
    const steering = this.steeringQueue.slice();
    this.steeringQueue = [];
    return steering;
  }

  private dequeueFollowUpMessages(): AgentMessage[] {
    if (this.followUpMode === "one-at-a-time") {
      if (this.followUpQueue.length > 0) {
        const first = this.followUpQueue[0];
        this.followUpQueue = this.followUpQueue.slice(1);
        return [first];
      }
      return [];
    }
    const followUp = this.followUpQueue.slice();
    this.followUpQueue = [];
    return followUp;
  }

  // ── Lifecycle ───────────────────────────────────────────────────────────

  abort() {
    this.abortController?.abort();
  }

  waitForIdle(): Promise<void> {
    return this.runningPrompt ?? Promise.resolve();
  }

  reset() {
    this._state.messages = [];
    this._state.isStreaming = false;
    this._state.streamMessage = null;
    this._state.pendingToolCalls = new Set<string>();
    this._state.error = undefined;
    this.steeringQueue = [];
    this.followUpQueue = [];
  }

  // ── Prompting ───────────────────────────────────────────────────────────

  async prompt(message: AgentMessage | AgentMessage[]): Promise<void>;
  async prompt(input: string, images?: ImageContent[]): Promise<void>;
  async prompt(
    input: string | AgentMessage | AgentMessage[],
    images?: ImageContent[]
  ) {
    if (this._state.isStreaming) {
      throw new Error(
        "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
      );
    }

    const model = this._state.model;
    if (!model) throw new Error("No model configured");

    let msgs: AgentMessage[];

    if (Array.isArray(input)) {
      msgs = input;
    } else if (typeof input === "string") {
      const content: Array<TextContent | ImageContent> = [
        { type: "text", text: input }
      ];
      if (images && images.length > 0) {
        content.push(...images);
      }
      msgs = [{ role: "user", content, timestamp: Date.now() }];
    } else {
      msgs = [input];
    }

    await this._runLoop(msgs);
  }

  /**
   * Continue from current context (used for retries and resuming queued messages).
   */
  async continue() {
    if (this._state.isStreaming) {
      throw new Error(
        "Agent is already processing. Wait for completion before continuing."
      );
    }

    const messages = this._state.messages;
    if (messages.length === 0) {
      throw new Error("No messages to continue from");
    }

    if (messages[messages.length - 1].role === "assistant") {
      const queuedSteering = this.dequeueSteeringMessages();
      if (queuedSteering.length > 0) {
        await this._runLoop(queuedSteering, { skipInitialSteeringPoll: true });
        return;
      }

      const queuedFollowUp = this.dequeueFollowUpMessages();
      if (queuedFollowUp.length > 0) {
        await this._runLoop(queuedFollowUp);
        return;
      }

      throw new Error("Cannot continue from message role: assistant");
    }

    await this._runLoop(undefined);
  }

  // ── Private Loop ────────────────────────────────────────────────────────

  private async _runLoop(
    messages?: AgentMessage[],
    options?: { skipInitialSteeringPoll?: boolean }
  ) {
    const model = this._state.model;
    if (!model) throw new Error("No model configured");

    this.runningPrompt = new Promise<void>((resolve) => {
      this.resolveRunningPrompt = resolve;
    });

    this.abortController = new AbortController();
    this._state.isStreaming = true;
    this._state.streamMessage = null;
    this._state.error = undefined;

    const context: AgentContext = {
      systemPrompt: this._state.systemPrompt,
      messages: this._state.messages.slice(),
      tools: this._state.tools
    };

    let skipInitialSteeringPoll = options?.skipInitialSteeringPoll === true;

    const config: AgentLoopConfig = {
      model,
      temperature: this._temperature,
      maxTokens: this._maxTokens,
      providerOptions: this._providerOptions,
      convertToLlm: this.convertToLlm,
      transformContext: this.transformContext,
      getSteeringMessages: async () => {
        if (skipInitialSteeringPoll) {
          skipInitialSteeringPoll = false;
          return [];
        }
        return this.dequeueSteeringMessages();
      },
      getFollowUpMessages: async () => this.dequeueFollowUpMessages()
    };

    let partial: AgentMessage | null = null;

    try {
      const eventStream = messages
        ? agentLoop(messages, context, config, this.abortController.signal)
        : agentLoopContinue(context, config, this.abortController.signal);

      for await (const event of eventStream) {
        switch (event.type) {
          case "message_start":
            partial = event.message;
            this._state.streamMessage = event.message;
            break;

          case "message_update":
            partial = event.message;
            this._state.streamMessage = event.message;
            break;

          case "message_end":
            partial = null;
            this._state.streamMessage = null;
            this.appendMessage(event.message);
            break;

          case "tool_execution_start": {
            const s = new Set(this._state.pendingToolCalls);
            s.add(event.toolCallId);
            this._state.pendingToolCalls = s;
            break;
          }

          case "tool_execution_end": {
            const s = new Set(this._state.pendingToolCalls);
            s.delete(event.toolCallId);
            this._state.pendingToolCalls = s;
            break;
          }

          case "turn_end":
            if (
              event.message.role === "assistant" &&
              (event.message as any).errorMessage
            ) {
              this._state.error = (event.message as any).errorMessage;
            }
            break;

          case "agent_end":
            this._state.isStreaming = false;
            this._state.streamMessage = null;
            break;
        }

        this.emit(event);
      }

      // Handle any remaining partial message
      if (
        partial &&
        partial.role === "assistant" &&
        "content" in partial &&
        Array.isArray(partial.content) &&
        partial.content.length > 0
      ) {
        const onlyEmpty = !partial.content.some(
          (c: any) =>
            (c.type === "thinking" && c.thinking.trim().length > 0) ||
            (c.type === "text" && c.text.trim().length > 0) ||
            (c.type === "toolCall" && c.name.trim().length > 0)
        );
        if (!onlyEmpty) {
          this.appendMessage(partial);
        } else if (this.abortController?.signal.aborted) {
          throw new Error("Request was aborted");
        }
      }
    } catch (err: any) {
      const errorMsg: AgentMessage = {
        role: "assistant",
        content: [{ type: "text", text: "" }],
        usage: {},
        stopReason: this.abortController?.signal.aborted ? "aborted" : "error",
        errorMessage: err?.message || String(err),
        timestamp: Date.now()
      };

      this.appendMessage(errorMsg);
      this._state.error = err?.message || String(err);
      this.emit({ type: "agent_end", messages: [errorMsg] });
    } finally {
      this._state.isStreaming = false;
      this._state.streamMessage = null;
      this._state.pendingToolCalls = new Set<string>();
      this.abortController = undefined;
      this.resolveRunningPrompt?.();
      this.runningPrompt = undefined;
      this.resolveRunningPrompt = undefined;
    }
  }

  private emit(e: AgentEvent) {
    for (const listener of this.listeners) {
      listener(e);
    }
  }
}
