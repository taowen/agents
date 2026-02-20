/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to CoreMessage[] only at the LLM call boundary via AI SDK's streamText.
 */

import { streamText, type ModelMessage } from "ai";
import { EventStream } from "./event-stream.js";
import type {
  AgentContext,
  AgentEvent,
  AgentLoopConfig,
  AgentMessage,
  AgentTool,
  AgentToolResult,
  AssistantMessage,
  ImageContent,
  Message,
  StopReason,
  TextContent,
  ThinkingContent,
  ToolCall,
  ToolResultMessage,
  UserMessage
} from "./types.js";

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Start an agent loop with new prompt messages.
 * The prompts are added to the context and events are emitted for them.
 */
export function agentLoop(
  prompts: AgentMessage[],
  context: AgentContext,
  config: AgentLoopConfig,
  signal?: AbortSignal
): EventStream<AgentEvent, AgentMessage[]> {
  const stream = createAgentStream();

  (async () => {
    const newMessages: AgentMessage[] = [...prompts];
    const currentContext: AgentContext = {
      ...context,
      messages: [...context.messages, ...prompts]
    };

    stream.push({ type: "agent_start" });
    stream.push({ type: "turn_start" });
    for (const prompt of prompts) {
      stream.push({ type: "message_start", message: prompt });
      stream.push({ type: "message_end", message: prompt });
    }

    await runLoop(currentContext, newMessages, config, signal, stream);
  })();

  return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries — context already has user message or tool results.
 */
export function agentLoopContinue(
  context: AgentContext,
  config: AgentLoopConfig,
  signal?: AbortSignal
): EventStream<AgentEvent, AgentMessage[]> {
  if (context.messages.length === 0) {
    throw new Error("Cannot continue: no messages in context");
  }

  if (context.messages[context.messages.length - 1].role === "assistant") {
    throw new Error("Cannot continue from message role: assistant");
  }

  const stream = createAgentStream();

  (async () => {
    const newMessages: AgentMessage[] = [];
    const currentContext: AgentContext = { ...context };

    stream.push({ type: "agent_start" });
    stream.push({ type: "turn_start" });

    await runLoop(currentContext, newMessages, config, signal, stream);
  })();

  return stream;
}

// ── Internals ───────────────────────────────────────────────────────────────

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
  return new EventStream<AgentEvent, AgentMessage[]>(
    (event: AgentEvent) => event.type === "agent_end",
    (event: AgentEvent) => (event.type === "agent_end" ? event.messages : [])
  );
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 */
async function runLoop(
  currentContext: AgentContext,
  newMessages: AgentMessage[],
  config: AgentLoopConfig,
  signal: AbortSignal | undefined,
  stream: EventStream<AgentEvent, AgentMessage[]>
): Promise<void> {
  let firstTurn = true;
  let pendingMessages: AgentMessage[] =
    (await config.getSteeringMessages?.()) || [];

  // Outer loop: continues when queued follow-up messages arrive after agent would stop
  while (true) {
    let hasMoreToolCalls = true;
    let steeringAfterTools: AgentMessage[] | null = null;

    // Inner loop: process tool calls and steering messages
    while (hasMoreToolCalls || pendingMessages.length > 0) {
      if (!firstTurn) {
        stream.push({ type: "turn_start" });
      } else {
        firstTurn = false;
      }

      // Process pending messages (inject before next assistant response)
      if (pendingMessages.length > 0) {
        for (const message of pendingMessages) {
          stream.push({ type: "message_start", message });
          stream.push({ type: "message_end", message });
          currentContext.messages.push(message);
          newMessages.push(message);
        }
        pendingMessages = [];
      }

      // Stream assistant response
      const message = await streamAssistantResponse(
        currentContext,
        config,
        signal,
        stream
      );
      newMessages.push(message);

      if (message.stopReason === "error" || message.stopReason === "aborted") {
        stream.push({ type: "turn_end", message, toolResults: [] });
        stream.push({ type: "agent_end", messages: newMessages });
        stream.end(newMessages);
        return;
      }

      // Check for tool calls
      const toolCalls = message.content.filter(
        (c): c is ToolCall => c.type === "toolCall"
      );
      hasMoreToolCalls = toolCalls.length > 0;

      const toolResults: ToolResultMessage[] = [];
      if (hasMoreToolCalls) {
        const toolExecution = await executeToolCalls(
          currentContext.tools,
          message,
          signal,
          stream,
          config.getSteeringMessages
        );
        toolResults.push(...toolExecution.toolResults);
        steeringAfterTools = toolExecution.steeringMessages ?? null;

        for (const result of toolResults) {
          currentContext.messages.push(result);
          newMessages.push(result);
        }
      }

      stream.push({ type: "turn_end", message, toolResults });

      // Get steering messages after turn completes
      if (steeringAfterTools && steeringAfterTools.length > 0) {
        pendingMessages = steeringAfterTools;
        steeringAfterTools = null;
      } else {
        pendingMessages = (await config.getSteeringMessages?.()) || [];
      }
    }

    // Agent would stop here. Check for follow-up messages.
    const followUpMessages = (await config.getFollowUpMessages?.()) || [];
    if (followUpMessages.length > 0) {
      pendingMessages = followUpMessages;
      continue;
    }

    break;
  }

  stream.push({ type: "agent_end", messages: newMessages });
  stream.end(newMessages);
}

// ── LLM Integration (AI SDK) ───────────────────────────────────────────────

/**
 * Stream an assistant response from the LLM using AI SDK's streamText.
 */
async function streamAssistantResponse(
  context: AgentContext,
  config: AgentLoopConfig,
  signal: AbortSignal | undefined,
  stream: EventStream<AgentEvent, AgentMessage[]>
): Promise<AssistantMessage> {
  // Apply context transform if configured
  let messages = context.messages;
  if (config.transformContext) {
    messages = await config.transformContext(messages, signal);
  }

  // Convert to LLM-compatible messages
  const llmMessages = await config.convertToLlm(messages);

  // Convert tools to AI SDK ToolSet (without execute — we handle execution ourselves)
  const toolSet = context.tools?.length
    ? convertToolsToToolSet(context.tools)
    : undefined;

  const assistantMessage: AssistantMessage = {
    role: "assistant",
    content: [],
    usage: {},
    stopReason: "stop",
    timestamp: Date.now()
  };

  stream.push({ type: "message_start", message: { ...assistantMessage } });

  try {
    const result = streamText({
      model: config.model,
      system: context.systemPrompt || undefined,
      messages: llmMessages,
      tools: toolSet,
      temperature: config.temperature,
      maxOutputTokens: config.maxTokens,
      providerOptions: config.providerOptions as any,
      abortSignal: signal
    });

    let currentTextIndex: number | null = null;
    let currentThinkingIndex: number | null = null;

    for await (const part of result.fullStream) {
      switch (part.type) {
        case "text-delta": {
          if (currentTextIndex === null) {
            assistantMessage.content.push({ type: "text", text: "" });
            currentTextIndex = assistantMessage.content.length - 1;
          }
          (assistantMessage.content[currentTextIndex] as TextContent).text +=
            part.text;
          stream.push({
            type: "message_update",
            message: { ...assistantMessage }
          });
          break;
        }

        case "reasoning-delta": {
          if (currentThinkingIndex === null) {
            assistantMessage.content.push({ type: "thinking", thinking: "" });
            currentThinkingIndex = assistantMessage.content.length - 1;
          }
          (
            assistantMessage.content[currentThinkingIndex] as ThinkingContent
          ).thinking += part.text;
          stream.push({
            type: "message_update",
            message: { ...assistantMessage }
          });
          break;
        }

        case "tool-call": {
          // New content block — reset text/thinking tracking
          currentTextIndex = null;
          currentThinkingIndex = null;

          assistantMessage.content.push({
            type: "toolCall",
            id: part.toolCallId,
            name: part.toolName,
            arguments: part.input as Record<string, any>
          });
          stream.push({
            type: "message_update",
            message: { ...assistantMessage }
          });
          break;
        }

        case "finish": {
          const usage = (part as any).usage;
          assistantMessage.usage = {
            inputTokens: usage?.promptTokens,
            outputTokens: usage?.completionTokens,
            totalTokens: usage?.totalTokens
          };
          assistantMessage.stopReason = mapFinishReason(
            (part as any).finishReason
          );
          break;
        }

        case "error": {
          assistantMessage.stopReason = "error";
          const err = (part as any).error;
          assistantMessage.errorMessage =
            err instanceof Error ? err.message : String(err);
          break;
        }
      }
    }
  } catch (err) {
    if (signal?.aborted) {
      assistantMessage.stopReason = "aborted";
    } else {
      assistantMessage.stopReason = "error";
      assistantMessage.errorMessage =
        err instanceof Error ? err.message : String(err);
    }
  }

  context.messages.push(assistantMessage);
  stream.push({ type: "message_end", message: assistantMessage });
  return assistantMessage;
}

// ── Tool Execution ──────────────────────────────────────────────────────────

/**
 * Execute tool calls from an assistant message sequentially,
 * checking for steering interrupts between each call.
 */
async function executeToolCalls(
  tools: AgentTool<any>[] | undefined,
  assistantMessage: AssistantMessage,
  signal: AbortSignal | undefined,
  stream: EventStream<AgentEvent, AgentMessage[]>,
  getSteeringMessages?: AgentLoopConfig["getSteeringMessages"]
): Promise<{
  toolResults: ToolResultMessage[];
  steeringMessages?: AgentMessage[];
}> {
  const toolCalls = assistantMessage.content.filter(
    (c): c is ToolCall => c.type === "toolCall"
  );
  const results: ToolResultMessage[] = [];
  let steeringMessages: AgentMessage[] | undefined;

  for (let index = 0; index < toolCalls.length; index++) {
    const toolCall = toolCalls[index];
    const tool = tools?.find((t) => t.name === toolCall.name);

    stream.push({
      type: "tool_execution_start",
      toolCallId: toolCall.id,
      toolName: toolCall.name,
      args: toolCall.arguments
    });

    let result: AgentToolResult<any>;
    let isError = false;

    try {
      if (!tool) throw new Error(`Tool ${toolCall.name} not found`);

      const validatedArgs = validateToolArguments(tool, toolCall.arguments);

      result = await tool.execute(
        toolCall.id,
        validatedArgs,
        signal,
        (partialResult) => {
          stream.push({
            type: "tool_execution_update",
            toolCallId: toolCall.id,
            toolName: toolCall.name,
            args: toolCall.arguments,
            partialResult
          });
        }
      );
    } catch (e) {
      result = {
        content: [
          { type: "text", text: e instanceof Error ? e.message : String(e) }
        ],
        details: {}
      };
      isError = true;
    }

    stream.push({
      type: "tool_execution_end",
      toolCallId: toolCall.id,
      toolName: toolCall.name,
      result,
      isError
    });

    const toolResultMessage: ToolResultMessage = {
      role: "toolResult",
      toolCallId: toolCall.id,
      toolName: toolCall.name,
      content: result.content,
      details: result.details,
      isError,
      timestamp: Date.now()
    };

    results.push(toolResultMessage);
    stream.push({ type: "message_start", message: toolResultMessage });
    stream.push({ type: "message_end", message: toolResultMessage });

    // Check for steering messages — skip remaining tools if user interrupted
    if (getSteeringMessages) {
      const steering = await getSteeringMessages();
      if (steering.length > 0) {
        steeringMessages = steering;
        const remainingCalls = toolCalls.slice(index + 1);
        for (const skipped of remainingCalls) {
          results.push(skipToolCall(skipped, stream));
        }
        break;
      }
    }
  }

  return { toolResults: results, steeringMessages };
}

function skipToolCall(
  toolCall: ToolCall,
  stream: EventStream<AgentEvent, AgentMessage[]>
): ToolResultMessage {
  const result: AgentToolResult<any> = {
    content: [{ type: "text", text: "Skipped due to queued user message." }],
    details: {}
  };

  stream.push({
    type: "tool_execution_start",
    toolCallId: toolCall.id,
    toolName: toolCall.name,
    args: toolCall.arguments
  });
  stream.push({
    type: "tool_execution_end",
    toolCallId: toolCall.id,
    toolName: toolCall.name,
    result,
    isError: true
  });

  const toolResultMessage: ToolResultMessage = {
    role: "toolResult",
    toolCallId: toolCall.id,
    toolName: toolCall.name,
    content: result.content,
    details: {},
    isError: true,
    timestamp: Date.now()
  };

  stream.push({ type: "message_start", message: toolResultMessage });
  stream.push({ type: "message_end", message: toolResultMessage });

  return toolResultMessage;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Validate tool arguments using the tool's zod schema.
 */
function validateToolArguments(
  tool: AgentTool<any>,
  args: Record<string, any>
): any {
  const result = tool.parameters.safeParse(args);
  if (!result.success) {
    throw new Error(
      `Invalid arguments for tool ${tool.name}: ${(result as any).error?.message ?? "validation failed"}`
    );
  }
  return (result as any).data;
}

/**
 * Convert AgentTool[] to AI SDK ToolSet (tools without execute — we handle execution).
 */
function convertToolsToToolSet(tools: AgentTool<any>[]): Record<string, any> {
  const toolSet: Record<string, any> = {};
  for (const t of tools) {
    toolSet[t.name] = {
      description: t.description,
      inputSchema: t.parameters
    };
  }
  return toolSet;
}

/**
 * Default convertToLlm implementation.
 * Converts standard AgentMessage types to AI SDK CoreMessage[].
 * Custom message types are skipped.
 */
export function convertToModelMessages(
  messages: AgentMessage[]
): ModelMessage[] {
  const result: ModelMessage[] = [];

  for (const msg of messages) {
    if (!("role" in msg)) continue;

    switch ((msg as Message).role) {
      case "user": {
        const userMsg = msg as UserMessage;
        if (typeof userMsg.content === "string") {
          result.push({ role: "user", content: userMsg.content });
        } else {
          result.push({
            role: "user",
            content: userMsg.content.map((c) => {
              if (c.type === "text")
                return { type: "text" as const, text: c.text };
              return {
                type: "image" as const,
                image: c.data,
                mimeType: c.mimeType
              };
            })
          });
        }
        break;
      }

      case "assistant": {
        const assistantMsg = msg as AssistantMessage;
        result.push({
          role: "assistant",
          content: assistantMsg.content.map((c) => {
            if (c.type === "text")
              return { type: "text" as const, text: c.text };
            if (c.type === "thinking")
              return { type: "reasoning" as const, text: c.thinking };
            return {
              type: "tool-call" as const,
              toolCallId: c.id,
              toolName: c.name,
              input: c.arguments
            };
          })
        } as ModelMessage);
        break;
      }

      case "toolResult": {
        const toolMsg = msg as ToolResultMessage;
        const textOutput = toolMsg.content
          .map((c) => (c.type === "text" ? c.text : ""))
          .filter(Boolean)
          .join("\n");
        result.push({
          role: "tool",
          content: [
            {
              type: "tool-result" as const,
              toolCallId: toolMsg.toolCallId,
              toolName: toolMsg.toolName,
              output: { type: "text" as const, value: textOutput || "OK" }
            }
          ]
        } as ModelMessage);
        const images = toolMsg.content.filter(
          (c): c is ImageContent => c.type === "image"
        );
        if (images.length > 0) {
          result.push({
            role: "user",
            content: images.map((c) => ({
              type: "image" as const,
              image: c.data,
              mimeType: c.mimeType
            }))
          } as ModelMessage);
        }
        break;
      }
    }
  }

  return result;
}

function mapFinishReason(reason: string): StopReason {
  switch (reason) {
    case "stop":
      return "stop";
    case "length":
      return "length";
    case "tool-calls":
      return "toolUse";
    default:
      return "stop";
  }
}
