import type { LanguageModel, ModelMessage } from "ai";
import type { z } from "zod";

// Content block types

export interface TextContent {
  type: "text";
  text: string;
}

export interface ThinkingContent {
  type: "thinking";
  thinking: string;
}

export interface ImageContent {
  type: "image";
  data: string;
  mimeType: string;
}

export interface ToolCall {
  type: "toolCall";
  id: string;
  name: string;
  arguments: Record<string, any>;
}

// Usage (simplified — no cost tracking)

export interface Usage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}

// Stop reasons

export type StopReason = "stop" | "length" | "toolUse" | "error" | "aborted";

// Message types

export interface UserMessage {
  role: "user";
  content: string | (TextContent | ImageContent)[];
  timestamp: number;
}

export interface AssistantMessage {
  role: "assistant";
  content: (TextContent | ThinkingContent | ToolCall)[];
  usage: Usage;
  stopReason: StopReason;
  errorMessage?: string;
  timestamp: number;
}

export interface ToolResultMessage<TDetails = any> {
  role: "toolResult";
  toolCallId: string;
  toolName: string;
  content: (TextContent | ImageContent)[];
  details?: TDetails;
  isError: boolean;
  timestamp: number;
}

export type Message = UserMessage | AssistantMessage | ToolResultMessage;

// Extensible custom messages (apps extend via declaration merging)

export interface CustomAgentMessages {
  // Empty by default — extend like:
  // declare module "pi" {
  //   interface CustomAgentMessages {
  //     artifact: ArtifactMessage;
  //   }
  // }
}

export type AgentMessage =
  | Message
  | CustomAgentMessages[keyof CustomAgentMessages];

// Tool types

export interface AgentToolResult<T> {
  content: (TextContent | ImageContent)[];
  details: T;
}

export type AgentToolUpdateCallback<T = any> = (
  partialResult: AgentToolResult<T>
) => void;

export interface AgentTool<
  TParams extends z.ZodType = z.ZodType,
  TDetails = any
> {
  name: string;
  description: string;
  label: string;
  parameters: TParams;
  execute: (
    toolCallId: string,
    params: z.infer<TParams>,
    signal?: AbortSignal,
    onUpdate?: AgentToolUpdateCallback<TDetails>
  ) => Promise<AgentToolResult<TDetails>>;
}

// Context

export interface AgentContext {
  systemPrompt: string;
  messages: AgentMessage[];
  tools?: AgentTool<any>[];
}

// Loop config

export interface AgentLoopConfig {
  model: LanguageModel;
  temperature?: number;
  maxTokens?: number;
  providerOptions?: Record<string, Record<string, unknown>>;

  /** Converts AgentMessage[] to AI SDK ModelMessage[] before each LLM call. */
  convertToLlm: (
    messages: AgentMessage[]
  ) => ModelMessage[] | Promise<ModelMessage[]>;

  /** Optional transform applied to context before convertToLlm. */
  transformContext?: (
    messages: AgentMessage[],
    signal?: AbortSignal
  ) => Promise<AgentMessage[]>;

  /** Returns steering messages to inject mid-run (checked after each tool execution). */
  getSteeringMessages?: () => Promise<AgentMessage[]>;

  /** Returns follow-up messages to process after the agent would otherwise stop. */
  getFollowUpMessages?: () => Promise<AgentMessage[]>;
}

// Agent state

export interface AgentState {
  systemPrompt: string;
  model: LanguageModel | null;
  tools: AgentTool<any>[];
  messages: AgentMessage[];
  isStreaming: boolean;
  streamMessage: AgentMessage | null;
  pendingToolCalls: Set<string>;
  error?: string;
}

// Agent events

export type AgentEvent =
  | { type: "agent_start" }
  | { type: "agent_end"; messages: AgentMessage[] }
  | { type: "turn_start" }
  | {
      type: "turn_end";
      message: AgentMessage;
      toolResults: ToolResultMessage[];
    }
  | { type: "message_start"; message: AgentMessage }
  | { type: "message_update"; message: AgentMessage }
  | { type: "message_end"; message: AgentMessage }
  | {
      type: "tool_execution_start";
      toolCallId: string;
      toolName: string;
      args: any;
    }
  | {
      type: "tool_execution_update";
      toolCallId: string;
      toolName: string;
      args: any;
      partialResult: any;
    }
  | {
      type: "tool_execution_end";
      toolCallId: string;
      toolName: string;
      result: any;
      isError: boolean;
    };
