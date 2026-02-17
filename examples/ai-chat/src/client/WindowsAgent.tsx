import { useCallback, useEffect, useRef, useState } from "react";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import type { LanguageModel } from "ai";
import type { ModelMessage } from "ai";
import { streamText, stepCountIs, tool } from "ai";
import { Bash, InMemoryFs } from "just-bash";
import { z } from "zod";
import { ModeToggle } from "@cloudflare/agents-ui";
import type { LlmConfig } from "../server/llm-proxy";
import "./windows-agent.css";

// ---- Agent logic (runs in-browser) ----

interface AgentCallbacks {
  onTextDelta: (delta: string) => void;
  onToolCall: (command: string) => void;
  onToolResult: (result: {
    stdout: string;
    stderr: string;
    exitCode: number;
  }) => void;
}

const fs = new InMemoryFs();
const bash = new Bash({ fs, cwd: "/home" });

const bashTool = tool({
  description: "Execute a bash command in the browser-side virtual filesystem",
  inputSchema: z.object({
    command: z.string().describe("The bash command to execute")
  }),
  execute: async ({ command }) => {
    const result = await bash.exec(command);
    return {
      stdout: result.stdout,
      stderr: result.stderr,
      exitCode: result.exitCode
    };
  }
});

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

  const result = streamText({
    model,
    system:
      "You are a helpful assistant with access to a bash shell running in a virtual filesystem. " +
      "The filesystem is an in-memory virtual filesystem. You can create files, run commands, etc. " +
      "The working directory is /home. " +
      "You are running inside an Electron desktop application on Windows.",
    messages: history,
    tools: { bash: bashTool },
    stopWhen: stepCountIs(10)
  });

  for await (const event of result.fullStream) {
    switch (event.type) {
      case "text-delta":
        callbacks.onTextDelta(event.text);
        break;
      case "tool-call":
        if (event.toolName === "bash") {
          callbacks.onToolCall((event.input as { command: string }).command);
        }
        break;
      case "tool-result":
        if (event.toolName === "bash") {
          callbacks.onToolResult(
            event.output as {
              stdout: string;
              stderr: string;
              exitCode: number;
            }
          );
        }
        break;
    }
  }

  const response = await result.response;
  history.push(...response.messages);
}

function resetAgent() {
  history.length = 0;
  cachedModel = null;
}

// ---- UI ----

interface ToolInvocation {
  command: string;
  result?: { stdout: string; stderr: string; exitCode: number };
}

interface Message {
  role: "user" | "assistant";
  text: string;
  toolInvocations: ToolInvocation[];
}

/** Detect the Electron preload bridge */
interface WorkWithWindows {
  ping: () => string;
  platform: string;
}

declare global {
  interface Window {
    workWithWindows?: WorkWithWindows;
  }
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
      setElectronStatus(`Electron (${platform}) - bridge: ${pong}`);
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
        onToolCall(command) {
          currentToolsRef.current.push({ command });
          scheduleFlush();
        },
        onToolResult(result) {
          const tools = currentToolsRef.current;
          for (let i = tools.length - 1; i >= 0; i--) {
            if (!tools[i].result) {
              tools[i].result = result;
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
            <span className="wa-badge wa-badge-electron">
              {electronStatus}
            </span>
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
                <div className="wa-tool-command">$ {invocation.command}</div>
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
