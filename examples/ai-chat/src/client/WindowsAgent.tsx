import { useCallback, useEffect, useRef, useState } from "react";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import type { LanguageModel } from "ai";
import type { ModelMessage } from "ai";
import { streamText, tool } from "ai";
import { Bash, InMemoryFs } from "just-bash";
import { z } from "zod";
import { ModeToggle } from "@cloudflare/agents-ui";
import type { LlmConfig } from "../server/llm-proxy";
import "./windows-agent.css";

// ---- Types ----

interface ScreenControlParams {
  action: string;
  x?: number;
  y?: number;
  text?: string;
  key?: string;
  modifiers?: string[];
  button?: string;
  doubleClick?: boolean;
  direction?: string;
  amount?: number;
}

interface ScreenControlResult {
  success: boolean;
  error?: string;
  width?: number;
  height?: number;
  base64?: string;
  action?: string;
  [key: string]: unknown;
}

interface WorkWithWindows {
  ping: () => string;
  platform: string;
  screenControl: (params: ScreenControlParams) => Promise<ScreenControlResult>;
}

declare global {
  interface Window {
    workWithWindows?: WorkWithWindows;
  }
}

// ---- Agent logic (runs in-browser) ----

interface AgentCallbacks {
  onTextDelta: (delta: string) => void;
  onToolCall: (toolName: string, input: Record<string, unknown>) => void;
  onToolResult: (toolName: string, result: unknown) => void;
}

const fs = new InMemoryFs();
const bash = new Bash({ fs, cwd: "/home" });

const bashToolDef = tool({
  description: "Execute a bash command in the browser-side virtual filesystem",
  parameters: z.object({
    command: z.string().describe("The bash command to execute")
  })
});

const screenToolDef = tool({
  description:
    "Control the Windows desktop screen. Take screenshots, click, type, press keys, move mouse, and scroll. " +
    "Use 'screenshot' first to see what's on screen, then interact with elements by their pixel coordinates. " +
    "Coordinates are in physical pixels from top-left (0,0).",
  parameters: z.object({
    action: z
      .enum([
        "screenshot",
        "click",
        "mouse_move",
        "type",
        "key_press",
        "scroll"
      ])
      .describe("The screen action to perform"),
    x: z
      .number()
      .optional()
      .describe("X coordinate in pixels (for click, mouse_move, scroll)"),
    y: z
      .number()
      .optional()
      .describe("Y coordinate in pixels (for click, mouse_move, scroll)"),
    text: z.string().optional().describe("Text to type (for type action)"),
    key: z
      .string()
      .optional()
      .describe(
        "Key name to press, e.g. 'enter', 'tab', 'a', 'f5' (for key_press action)"
      ),
    modifiers: z
      .array(z.string())
      .optional()
      .describe(
        "Modifier keys to hold, e.g. ['ctrl', 'shift'] (for key_press action)"
      ),
    button: z
      .enum(["left", "right", "middle"])
      .optional()
      .describe("Mouse button (for click action, default: left)"),
    doubleClick: z
      .boolean()
      .optional()
      .describe("Double-click (for click action, default: false)"),
    direction: z
      .enum(["up", "down"])
      .optional()
      .describe("Scroll direction (for scroll action, default: down)"),
    amount: z
      .number()
      .optional()
      .describe("Scroll amount in notches (for scroll action, default: 3)")
  })
});

async function executeBash(command: string) {
  const result = await bash.exec(command);
  return {
    stdout: result.stdout,
    stderr: result.stderr,
    exitCode: result.exitCode
  };
}

async function executeScreen(
  input: ScreenControlParams
): Promise<ScreenControlResult> {
  if (!window.workWithWindows?.screenControl) {
    return {
      success: false,
      error: "Screen control is only available in the Electron desktop app",
      action: input.action
    };
  }
  const result = await window.workWithWindows.screenControl(input);
  return { ...result, action: input.action };
}

const history: ModelMessage[] = [];

let cachedModel: LanguageModel | null = null;

async function getModel(): Promise<LanguageModel> {
  if (cachedModel) return cachedModel;

  const resp = await fetch("/api/llm/config");
  if (!resp.ok) {
    throw new Error(`Failed to fetch LLM config: ${resp.status}`);
  }
  const config = (await resp.json()) as LlmConfig;

  if (config.provider === "google") {
    cachedModel = createGoogleGenerativeAI({
      baseURL: config.baseURL,
      apiKey: config.apiKey
    })(config.model);
  } else {
    cachedModel = createOpenAICompatible({
      name: "llm",
      baseURL: config.baseURL,
      apiKey: config.apiKey
    })(config.model);
  }

  return cachedModel;
}

async function chat(userMessage: string, callbacks: AgentCallbacks) {
  history.push({ role: "user", content: userMessage });

  const model = await getModel();

  const hasScreenControl = !!window.workWithWindows?.screenControl;

  const systemPrompt =
    "You are a helpful assistant running inside an Electron desktop application on Windows. " +
    "You have access to a bash shell running in a virtual in-memory filesystem (working directory: /home). " +
    (hasScreenControl
      ? "You also have access to the Windows desktop screen. You can take screenshots to see what's on screen, " +
        "then click, type, press keys, move the mouse, and scroll to interact with any application. " +
        "Always take a screenshot first to understand the current screen state before performing actions. " +
        "Use pixel coordinates from the screenshot to target UI elements."
      : "");

  const tools = { bash: bashToolDef, screen: screenToolDef };

  for (let step = 0; step < 10; step++) {
    // Single-step streamText (no stopWhen / maxSteps)
    const result = streamText({
      model,
      system: systemPrompt,
      messages: history,
      tools
    });

    // Collect tool calls from the stream
    const toolCalls: Array<{
      toolCallId: string;
      toolName: string;
      args: Record<string, unknown>;
    }> = [];

    for await (const event of result.fullStream) {
      switch (event.type) {
        case "text-delta":
          callbacks.onTextDelta(event.text);
          break;
        case "tool-call": {
          const tcArgs = (event as any).input ?? (event as any).args ?? {};
          console.log(
            "[agent] tool-call:",
            event.toolName,
            JSON.stringify(tcArgs)
          );
          toolCalls.push({
            toolCallId: event.toolCallId,
            toolName: event.toolName,
            args: tcArgs as Record<string, unknown>
          });
          callbacks.onToolCall(
            event.toolName,
            tcArgs as Record<string, unknown>
          );
          break;
        }
      }
    }

    // Push assistant message(s) from this step
    const response = await result.response;
    history.push(...response.messages);

    // No tool calls → model is done
    if (toolCalls.length === 0) break;

    // Execute each tool call and push results into history
    console.log(
      "[agent] step",
      step,
      "toolCalls:",
      JSON.stringify(
        toolCalls.map((t) => ({
          id: t.toolCallId,
          name: t.toolName,
          args: t.args
        }))
      )
    );
    for (const tc of toolCalls) {
      let toolResultContent: string;

      if (tc.toolName === "bash") {
        const bashResult = await executeBash(
          (tc.args as { command: string }).command
        );
        callbacks.onToolResult(tc.toolName, bashResult);
        toolResultContent = JSON.stringify(bashResult);
      } else if (tc.toolName === "screen") {
        const screenResult = await executeScreen(
          tc.args as ScreenControlParams
        );
        callbacks.onToolResult(tc.toolName, screenResult);

        // Build text summary (without base64)
        const lines: string[] = [];
        lines.push(
          `Action: ${screenResult.action ?? "screenshot"} | Success: ${screenResult.success}`
        );
        if (screenResult.error) lines.push(`Error: ${screenResult.error}`);
        if (screenResult.width && screenResult.height)
          lines.push(`Screen: ${screenResult.width}x${screenResult.height}`);
        toolResultContent = lines.join("\n");

        // Inject screenshot as a user message so the model can see the image
        if (screenResult.base64) {
          history.push({
            role: "tool" as const,
            content: [
              {
                type: "tool-result" as const,
                toolCallId: tc.toolCallId,
                toolName: tc.toolName,
                output: { type: "text", value: toolResultContent }
              }
            ]
          });
          history.push({
            role: "user",
            content: [
              { type: "text", text: "Here is the screenshot:" },
              {
                type: "image",
                image: screenResult.base64,
                mediaType: "image/png"
              }
            ]
          });
          continue; // skip the default tool-result push below
        }
      } else {
        toolResultContent = "Unknown tool";
      }

      history.push({
        role: "tool" as const,
        content: [
          {
            type: "tool-result" as const,
            toolCallId: tc.toolCallId,
            toolName: tc.toolName,
            output: { type: "text", value: toolResultContent }
          }
        ]
      });
    }
  }
}

function resetAgent() {
  history.length = 0;
  cachedModel = null;
}

// ---- UI ----

interface BashInvocation {
  type: "bash";
  command: string;
  result?: { stdout: string; stderr: string; exitCode: number };
}

interface ScreenInvocation {
  type: "screen";
  action: string;
  params: Record<string, unknown>;
  result?: ScreenControlResult;
}

type ToolInvocation = BashInvocation | ScreenInvocation;

interface Message {
  role: "user" | "assistant";
  text: string;
  toolInvocations: ToolInvocation[];
}

export function WindowsAgent() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [electronStatus, setElectronStatus] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentTextRef = useRef("");
  const currentToolsRef = useRef<ToolInvocation[]>([]);
  const rafIdRef = useRef<number>(0);

  useEffect(() => {
    if (window.workWithWindows) {
      const pong = window.workWithWindows.ping();
      const platform = window.workWithWindows.platform;
      const hasScreen = !!window.workWithWindows.screenControl;
      setElectronStatus(
        `Electron (${platform}) - bridge: ${pong}${hasScreen ? " + screen" : ""}`
      );
    }
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const flushToState = useCallback(() => {
    setMessages((prev) => {
      const updated = [...prev];
      const last = updated[updated.length - 1];
      if (last && last.role === "assistant") {
        updated[updated.length - 1] = {
          ...last,
          text: currentTextRef.current,
          toolInvocations: [...currentToolsRef.current]
        };
      }
      return updated;
    });
    scrollToBottom();
    rafIdRef.current = 0;
  }, [scrollToBottom]);

  const scheduleFlush = useCallback(() => {
    if (!rafIdRef.current) {
      rafIdRef.current = requestAnimationFrame(flushToState);
    }
  }, [flushToState]);

  const handleSubmit = useCallback(async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    setInput("");
    setIsStreaming(true);

    const userMsg: Message = { role: "user", text, toolInvocations: [] };
    const assistantMsg: Message = {
      role: "assistant",
      text: "",
      toolInvocations: []
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    currentTextRef.current = "";
    currentToolsRef.current = [];

    try {
      await chat(text, {
        onTextDelta(delta) {
          currentTextRef.current += delta;
          scheduleFlush();
        },
        onToolCall(toolName, toolInput) {
          console.log(
            "[UI] onToolCall:",
            toolName,
            "input:",
            JSON.stringify(toolInput)
          );
          if (toolName === "bash") {
            currentToolsRef.current.push({
              type: "bash",
              command: (toolInput as { command: string }).command
            });
          } else if (toolName === "screen") {
            const { action, ...params } = toolInput as {
              action: string;
              [k: string]: unknown;
            };
            currentToolsRef.current.push({
              type: "screen",
              action,
              params
            });
          }
          scheduleFlush();
        },
        onToolResult(toolName, result) {
          const tools = currentToolsRef.current;
          for (let i = tools.length - 1; i >= 0; i--) {
            const inv = tools[i];
            if (toolName === "bash" && inv.type === "bash" && !inv.result) {
              inv.result = result as {
                stdout: string;
                stderr: string;
                exitCode: number;
              };
              break;
            }
            if (toolName === "screen" && inv.type === "screen" && !inv.result) {
              inv.result = result as ScreenControlResult;
              break;
            }
          }
          scheduleFlush();
        }
      });
    } catch (err) {
      currentTextRef.current += `\n\n[Error: ${err instanceof Error ? err.message : String(err)}]`;
      scheduleFlush();
    }

    flushToState();
    setIsStreaming(false);
  }, [input, isStreaming, scheduleFlush, flushToState]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  const handleReset = useCallback(() => {
    resetAgent();
    setMessages([]);
  }, []);

  return (
    <div className="wa-app">
      <header className="wa-header">
        <div className="wa-header-left">
          <h1 className="wa-title">Windows Agent</h1>
          {electronStatus ? (
            <span className="wa-badge wa-badge-electron">{electronStatus}</span>
          ) : (
            <span className="wa-badge wa-badge-browser">Browser mode</span>
          )}
        </div>
        <div className="wa-header-right">
          <ModeToggle />
          <button
            className="wa-btn-reset"
            onClick={handleReset}
            disabled={isStreaming}
          >
            Reset
          </button>
        </div>
      </header>

      <div className="wa-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`wa-message wa-message-${msg.role}`}>
            <div className="wa-message-role">
              {msg.role === "user" ? "You" : "Assistant"}
            </div>
            {msg.toolInvocations.map((invocation, j) => (
              <div key={j} className="wa-tool">
                {invocation.type === "bash" ? (
                  <>
                    <div className="wa-tool-command">
                      $ {invocation.command}
                    </div>
                    {invocation.result ? (
                      <pre className="wa-tool-result">
                        {invocation.result.stdout}
                        {invocation.result.stderr && (
                          <span className="wa-tool-stderr">
                            {invocation.result.stderr}
                          </span>
                        )}
                        {invocation.result.exitCode !== 0 && (
                          <span className="wa-tool-exitcode">
                            exit code: {invocation.result.exitCode}
                          </span>
                        )}
                      </pre>
                    ) : (
                      <div className="wa-tool-running">Running...</div>
                    )}
                  </>
                ) : (
                  <>
                    <div className="wa-tool-command">
                      screen: {invocation.action}
                      {Object.keys(invocation.params).length > 0 && (
                        <span className="wa-tool-params">
                          {" "}
                          (
                          {Object.entries(invocation.params)
                            .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                            .join(", ")}
                          )
                        </span>
                      )}
                    </div>
                    {invocation.result ? (
                      <div className="wa-tool-result">
                        {invocation.result.success ? (
                          <>
                            {invocation.result.base64 && (
                              <img
                                src={`data:image/png;base64,${invocation.result.base64}`}
                                alt="Screenshot"
                                className="wa-screenshot"
                              />
                            )}
                            {invocation.result.width &&
                              invocation.result.height && (
                                <div className="wa-tool-info">
                                  Screen: {invocation.result.width}x
                                  {invocation.result.height}
                                </div>
                              )}
                          </>
                        ) : (
                          <span className="wa-tool-stderr">
                            Error: {invocation.result.error}
                          </span>
                        )}
                      </div>
                    ) : (
                      <div className="wa-tool-running">Running...</div>
                    )}
                  </>
                )}
              </div>
            ))}
            {msg.text && <div className="wa-message-text">{msg.text}</div>}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="wa-input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message... (Enter to send)"
          disabled={isStreaming}
          rows={2}
        />
        <button
          className="wa-btn-send"
          onClick={handleSubmit}
          disabled={isStreaming || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}
