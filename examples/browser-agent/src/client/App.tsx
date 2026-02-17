import { useCallback, useRef, useState } from "react";
import { chat, resetAgent } from "./agent.ts";

interface ToolInvocation {
  command: string;
  result?: { stdout: string; stderr: string; exitCode: number };
}

interface Message {
  role: "user" | "assistant";
  text: string;
  toolInvocations: ToolInvocation[];
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  // Use refs for accumulating streaming state, flush via requestAnimationFrame
  const currentTextRef = useRef("");
  const currentToolsRef = useRef<ToolInvocation[]>([]);
  const rafIdRef = useRef<number>(0);

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

    // Add user message
    const userMsg: Message = { role: "user", text, toolInvocations: [] };
    // Add placeholder assistant message
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
          // Attach result to the last tool invocation without a result
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

    // Final flush
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
    <div className="app">
      <header>
        <h1>Browser Agent</h1>
        <button onClick={handleReset} disabled={isStreaming}>
          Reset
        </button>
      </header>
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-role">
              {msg.role === "user" ? "You" : "Assistant"}
            </div>
            {msg.toolInvocations.map((tool, j) => (
              <div key={j} className="tool-invocation">
                <div className="tool-command">$ {tool.command}</div>
                {tool.result ? (
                  <pre className="tool-result">
                    {tool.result.stdout}
                    {tool.result.stderr && (
                      <span className="stderr">{tool.result.stderr}</span>
                    )}
                    {tool.result.exitCode !== 0 && (
                      <span className="exit-code">
                        exit code: {tool.result.exitCode}
                      </span>
                    )}
                  </pre>
                ) : (
                  <div className="tool-running">Running...</div>
                )}
              </div>
            ))}
            {msg.text && <div className="message-text">{msg.text}</div>}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message... (Enter to send)"
          disabled={isStreaming}
          rows={2}
        />
        <button onClick={handleSubmit} disabled={isStreaming || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
