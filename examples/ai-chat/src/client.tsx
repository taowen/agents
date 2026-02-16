import { Suspense, useCallback, useState, useEffect, useRef } from "react";
import { useAgent } from "agents/react";
import { useAgentChat } from "@cloudflare/ai-chat/react";
import { isToolUIPart, getToolName } from "ai";
import type { UIMessage } from "ai";
import {
  Button,
  Badge,
  InputArea,
  Empty,
  Surface,
  Text
} from "@cloudflare/kumo";
import {
  ConnectionIndicator,
  ModeToggle,
  PoweredByAgents,
  type ConnectionStatus
} from "@cloudflare/agents-ui";
import {
  PaperPlaneRightIcon,
  TrashIcon,
  CheckCircleIcon,
  XCircleIcon,
  GearIcon,
  CloudSunIcon
} from "@phosphor-icons/react";
import { LoginPage } from "./LoginPage";
import { SessionSidebar, type SessionInfo } from "./SessionSidebar";
import { SettingsPage } from "./SettingsPage";

function getMessageText(message: UIMessage): string {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => (part as { type: "text"; text: string }).text)
    .join("");
}

interface UserInfo {
  id: string;
  email: string;
  name: string | null;
  picture: string | null;
}

function Chat({
  sessionId,
  onFirstMessage
}: {
  sessionId: string;
  onFirstMessage: (text: string) => void;
}) {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("connecting");
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const firstMessageSent = useRef(false);

  const agent = useAgent({
    agent: "ChatAgent",
    name: sessionId,
    onOpen: useCallback(() => setConnectionStatus("connected"), []),
    onClose: useCallback(() => setConnectionStatus("disconnected"), []),
    onError: useCallback(
      (error: Event) => console.error("WebSocket error:", error),
      []
    )
  });

  const {
    messages,
    sendMessage,
    clearHistory,
    addToolApprovalResponse,
    status
  } = useAgentChat({
    agent,
    body: {
      clientVersion: "1.0.0"
    },
    onToolCall: async ({ toolCall, addToolOutput }) => {
      if (toolCall.toolName === "getUserTimezone") {
        addToolOutput({
          toolCallId: toolCall.toolCallId,
          output: {
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            localTime: new Date().toLocaleTimeString()
          }
        });
      }
    }
  });

  const isStreaming = status === "streaming";
  const isConnected = connectionStatus === "connected";

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || isStreaming) return;
    setInput("");
    sendMessage({ role: "user", parts: [{ type: "text", text }] });
    if (!firstMessageSent.current) {
      firstMessageSent.current = true;
      onFirstMessage(text);
    }
  }, [input, isStreaming, sendMessage, onFirstMessage]);

  return (
    <div className="flex flex-col h-screen bg-kumo-elevated">
      {/* Header */}
      <header className="px-5 py-4 bg-kumo-base border-b border-kumo-line">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold text-kumo-default">AI Chat</h1>
            <Badge variant="secondary">
              <CloudSunIcon size={12} weight="bold" className="mr-1" />
              Tools + Approval
            </Badge>
          </div>
          <div className="flex items-center gap-3">
            <ConnectionIndicator status={connectionStatus} />
            <ModeToggle />
            <Button
              variant="secondary"
              icon={<TrashIcon size={16} />}
              onClick={clearHistory}
            >
              Clear
            </Button>
          </div>
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-5 py-6 space-y-5">
          {messages.length === 0 && (
            <Empty
              icon={<CloudSunIcon size={32} />}
              title="Start a conversation"
              description='Try "What is the weather in London?" or "What timezone am I in?" or ask to explore files with bash'
            />
          )}

          {messages.map((message, index) => {
            const isUser = message.role === "user";
            const isLastAssistant =
              message.role === "assistant" && index === messages.length - 1;

            return (
              <div key={message.id} className="space-y-2">
                {isUser ? (
                  <div className="flex justify-end">
                    <div className="max-w-[85%] px-4 py-2.5 rounded-2xl rounded-br-md bg-kumo-contrast text-kumo-inverse leading-relaxed">
                      {getMessageText(message)}
                    </div>
                  </div>
                ) : (
                  <div className="flex justify-start">
                    <div className="max-w-[85%] px-4 py-2.5 rounded-2xl rounded-bl-md bg-kumo-base text-kumo-default leading-relaxed">
                      <div className="whitespace-pre-wrap">
                        {getMessageText(message)}
                        {isLastAssistant && isStreaming && (
                          <span className="inline-block w-0.5 h-[1em] bg-kumo-brand ml-0.5 align-text-bottom animate-blink-cursor" />
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {message.parts
                  .filter((part) => isToolUIPart(part))
                  .map((part) => {
                    if (!isToolUIPart(part)) return null;
                    const toolName = getToolName(part);

                    if (part.state === "output-available") {
                      const isBash = toolName === "bash";
                      const isBrowser = toolName === "browser";
                      const bashInput = part.input as
                        | { command?: string }
                        | undefined;
                      const bashOutput = part.output as
                        | {
                            stdout?: string;
                            stderr?: string;
                            exitCode?: number;
                          }
                        | undefined;
                      const browserInput = part.input as
                        | {
                            action?: string;
                            url?: string;
                            selector?: string;
                            text?: string;
                            direction?: string;
                          }
                        | undefined;
                      const browserOutput = part.output as
                        | {
                            action?: string;
                            success?: boolean;
                            url?: string;
                            title?: string;
                            text?: string;
                            screenshot?: string;
                            error?: string;
                          }
                        | undefined;

                      if (isBash) {
                        const cmd = bashInput?.command || "";
                        const cmdShort =
                          cmd.length > 80 ? cmd.slice(0, 80) + "…" : cmd;
                        const output = bashOutput?.stdout || "";
                        const outputShort =
                          output.split("\n")[0]?.slice(0, 80) || "";
                        const exitOk = bashOutput?.exitCode === 0;

                        return (
                          <div
                            key={part.toolCallId}
                            className="flex justify-start"
                          >
                            <div className="max-w-[85%] space-y-1">
                              {/* Request (command) — collapsed by default */}
                              <details className="group">
                                <summary className="cursor-pointer list-none flex items-center gap-2 px-3 py-1.5 rounded-lg bg-kumo-base ring ring-kumo-line hover:bg-kumo-elevated transition-colors">
                                  <GearIcon
                                    size={12}
                                    className="text-kumo-inactive shrink-0"
                                  />
                                  <span className="font-mono text-xs text-kumo-secondary truncate">
                                    $ {cmdShort}
                                  </span>
                                </summary>
                                <div className="mt-1 px-3 py-2 rounded-lg bg-kumo-base ring ring-kumo-line font-mono text-xs whitespace-pre-wrap overflow-x-auto max-h-[300px] overflow-y-auto">
                                  <Text size="xs" variant="secondary">
                                    {cmd}
                                  </Text>
                                </div>
                              </details>
                              {/* Response (output) — collapsed by default */}
                              <details className="group">
                                <summary className="cursor-pointer list-none flex items-center gap-2 px-3 py-1.5 rounded-lg bg-kumo-base ring ring-kumo-line hover:bg-kumo-elevated transition-colors">
                                  {exitOk ? (
                                    <CheckCircleIcon
                                      size={12}
                                      className="text-kumo-inactive shrink-0"
                                    />
                                  ) : (
                                    <XCircleIcon
                                      size={12}
                                      className="text-kumo-inactive shrink-0"
                                    />
                                  )}
                                  {exitOk ? (
                                    <span className="text-xs text-kumo-secondary truncate">
                                      {outputShort || "OK"}
                                    </span>
                                  ) : (
                                    <span className="text-xs text-kumo-secondary truncate">
                                      Exit {bashOutput?.exitCode}
                                      {bashOutput?.stderr
                                        ? `: ${bashOutput.stderr.split("\n")[0]?.slice(0, 60)}`
                                        : ""}
                                    </span>
                                  )}
                                </summary>
                                <div className="mt-1 px-3 py-2 rounded-lg bg-kumo-base ring ring-kumo-line font-mono text-xs whitespace-pre-wrap overflow-x-auto max-h-[400px] overflow-y-auto">
                                  {bashOutput?.stdout && (
                                    <Text size="xs" variant="secondary">
                                      {bashOutput.stdout}
                                    </Text>
                                  )}
                                  {bashOutput?.stderr && (
                                    <Text size="xs" variant="error">
                                      {bashOutput.stderr}
                                    </Text>
                                  )}
                                </div>
                              </details>
                            </div>
                          </div>
                        );
                      }

                      if (isBrowser && browserOutput) {
                        const action =
                          browserOutput.action || browserInput?.action || "";
                        const url =
                          browserOutput.url || browserInput?.url || "";
                        const summaryText = `${action}${url ? " " + url : ""}`;
                        const summaryShort =
                          summaryText.length > 70
                            ? summaryText.slice(0, 70) + "…"
                            : summaryText;

                        return (
                          <div
                            key={part.toolCallId}
                            className="flex justify-start"
                          >
                            <details className="max-w-[85%]">
                              <summary className="cursor-pointer list-none flex items-center gap-2 px-3 py-1.5 rounded-lg bg-kumo-base ring ring-kumo-line hover:bg-kumo-elevated transition-colors">
                                <GearIcon
                                  size={12}
                                  className="text-kumo-inactive shrink-0"
                                />
                                <span className="font-mono text-xs text-kumo-secondary truncate">
                                  {summaryShort}
                                </span>
                                {browserOutput.success ? (
                                  <Badge variant="secondary">OK</Badge>
                                ) : (
                                  <Badge variant="destructive">Failed</Badge>
                                )}
                              </summary>
                              <div className="mt-1 px-3 py-2 rounded-lg bg-kumo-base ring ring-kumo-line space-y-2">
                                {browserOutput.error && (
                                  <Text size="xs" variant="error">
                                    {browserOutput.error}
                                  </Text>
                                )}
                                {browserOutput.url && browserOutput.title && (
                                  <Text size="xs" variant="secondary">
                                    {browserOutput.title} — {browserOutput.url}
                                  </Text>
                                )}
                                {browserOutput.screenshot && (
                                  <img
                                    src={`data:image/png;base64,${browserOutput.screenshot}`}
                                    alt="Browser screenshot"
                                    className="rounded border border-kumo-line max-w-full"
                                  />
                                )}
                                {browserOutput.text && (
                                  <pre className="p-2 bg-kumo-elevated rounded overflow-x-auto text-xs text-kumo-secondary max-h-[300px] overflow-y-auto">
                                    {browserOutput.text}
                                  </pre>
                                )}
                              </div>
                            </details>
                          </div>
                        );
                      }

                      return (
                        <div
                          key={part.toolCallId}
                          className="flex justify-start"
                        >
                          <details className="max-w-[85%]">
                            <summary className="cursor-pointer list-none flex items-center gap-2 px-3 py-1.5 rounded-lg bg-kumo-base ring ring-kumo-line hover:bg-kumo-elevated transition-colors">
                              <GearIcon
                                size={12}
                                className="text-kumo-inactive shrink-0"
                              />
                              <Text size="xs" variant="secondary" bold>
                                {toolName}
                              </Text>
                              <Badge variant="secondary">Done</Badge>
                            </summary>
                            <div className="mt-1 px-3 py-2 rounded-lg bg-kumo-base ring ring-kumo-line font-mono text-xs whitespace-pre-wrap">
                              <Text size="xs" variant="secondary">
                                {JSON.stringify(part.output, null, 2)}
                              </Text>
                            </div>
                          </details>
                        </div>
                      );
                    }

                    if (
                      "approval" in part &&
                      part.state === "approval-requested"
                    ) {
                      const approvalId = (part.approval as { id?: string })?.id;
                      return (
                        <div
                          key={part.toolCallId}
                          className="flex justify-start"
                        >
                          <Surface className="max-w-[85%] px-4 py-3 rounded-xl ring-2 ring-kumo-warning">
                            <div className="flex items-center gap-2 mb-2">
                              <GearIcon
                                size={14}
                                className="text-kumo-warning"
                              />
                              <Text size="sm" bold>
                                Approval needed: {toolName}
                              </Text>
                            </div>
                            <div className="font-mono mb-3">
                              <Text size="xs" variant="secondary">
                                {JSON.stringify(part.input, null, 2)}
                              </Text>
                            </div>
                            <div className="flex gap-2">
                              <Button
                                variant="primary"
                                size="sm"
                                icon={<CheckCircleIcon size={14} />}
                                onClick={() => {
                                  if (approvalId) {
                                    addToolApprovalResponse({
                                      id: approvalId,
                                      approved: true
                                    });
                                  }
                                }}
                              >
                                Approve
                              </Button>
                              <Button
                                variant="secondary"
                                size="sm"
                                icon={<XCircleIcon size={14} />}
                                onClick={() => {
                                  if (approvalId) {
                                    addToolApprovalResponse({
                                      id: approvalId,
                                      approved: false
                                    });
                                  }
                                }}
                              >
                                Reject
                              </Button>
                            </div>
                          </Surface>
                        </div>
                      );
                    }

                    if (
                      part.state === "input-available" ||
                      part.state === "input-streaming"
                    ) {
                      return (
                        <div
                          key={part.toolCallId}
                          className="flex justify-start"
                        >
                          <Surface className="max-w-[85%] px-4 py-2.5 rounded-xl ring ring-kumo-line">
                            <div className="flex items-center gap-2">
                              <GearIcon
                                size={14}
                                className="text-kumo-inactive animate-spin"
                              />
                              <Text size="xs" variant="secondary">
                                Running {toolName}...
                              </Text>
                            </div>
                          </Surface>
                        </div>
                      );
                    }

                    return null;
                  })}
              </div>
            );
          })}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-kumo-line bg-kumo-base">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
          className="max-w-3xl mx-auto px-5 py-4"
        >
          <div className="flex items-end gap-3 rounded-xl border border-kumo-line bg-kumo-base p-3 shadow-sm focus-within:ring-2 focus-within:ring-kumo-ring focus-within:border-transparent transition-shadow">
            <InputArea
              value={input}
              onValueChange={setInput}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="Try: What's the weather in Paris?"
              disabled={!isConnected || isStreaming}
              rows={2}
              className="flex-1 !ring-0 focus:!ring-0 !shadow-none !bg-transparent !outline-none"
            />
            <Button
              type="submit"
              variant="primary"
              shape="square"
              aria-label="Send message"
              disabled={!input.trim() || !isConnected || isStreaming}
              icon={<PaperPlaneRightIcon size={18} />}
              loading={isStreaming}
              className="mb-0.5"
            />
          </div>
        </form>
        <div className="flex justify-center pb-3">
          <PoweredByAgents />
        </div>
      </div>
    </div>
  );
}

function AuthenticatedApp({ user }: { user: UserInfo }) {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [view, setView] = useState<"chat" | "settings">("chat");

  // Load sessions on mount
  useEffect(() => {
    fetch("/api/sessions")
      .then((res) => res.json())
      .then((data: SessionInfo[]) => {
        setSessions(data);
        if (data.length > 0) {
          setActiveSessionId(data[0].id);
        }
      })
      .catch(console.error);
  }, []);

  const handleNewSession = async () => {
    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      });
      const session = (await res.json()) as SessionInfo;
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      setView("chat");
    } catch (e) {
      console.error("Failed to create session:", e);
    }
  };

  const handleDeleteSession = async (id: string) => {
    try {
      await fetch(`/api/sessions/${id}`, { method: "DELETE" });
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (activeSessionId === id) {
        const remaining = sessions.filter((s) => s.id !== id);
        setActiveSessionId(remaining.length > 0 ? remaining[0].id : null);
      }
    } catch (e) {
      console.error("Failed to delete session:", e);
    }
  };

  const handleFirstMessage = async (text: string) => {
    if (!activeSessionId) return;
    const title = text.length > 50 ? text.slice(0, 50) + "..." : text;
    try {
      await fetch(`/api/sessions/${activeSessionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title })
      });
      setSessions((prev) =>
        prev.map((s) => (s.id === activeSessionId ? { ...s, title } : s))
      );
    } catch (e) {
      console.error("Failed to update session title:", e);
    }
  };

  // Auto-create first session if none exist
  useEffect(() => {
    if (sessions.length === 0 && activeSessionId === null) {
      handleNewSession();
    }
  }, [sessions.length, activeSessionId]);

  return (
    <div className="flex h-screen">
      <SessionSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        user={user}
        onNewSession={handleNewSession}
        onSelectSession={(id) => {
          setActiveSessionId(id);
          setView("chat");
        }}
        onDeleteSession={handleDeleteSession}
        onOpenSettings={() => setView("settings")}
      />
      <div className="flex-1">
        {view === "settings" ? (
          <SettingsPage onBack={() => setView("chat")} />
        ) : activeSessionId ? (
          <Chat
            key={activeSessionId}
            sessionId={activeSessionId}
            onFirstMessage={handleFirstMessage}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-kumo-inactive">
            Loading...
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [authState, setAuthState] = useState<
    "loading" | "unauthenticated" | "authenticated"
  >("loading");
  const [user, setUser] = useState<UserInfo | null>(null);

  useEffect(() => {
    fetch("/auth/status")
      .then((res) => res.json())
      .then((data: { authenticated: boolean; user?: UserInfo }) => {
        if (data.authenticated && data.user) {
          setUser(data.user);
          setAuthState("authenticated");
        } else {
          setAuthState("unauthenticated");
        }
      })
      .catch(() => setAuthState("unauthenticated"));
  }, []);

  if (authState === "loading") {
    return (
      <div className="flex items-center justify-center h-screen text-kumo-inactive">
        Loading...
      </div>
    );
  }

  if (authState === "unauthenticated") {
    return <LoginPage />;
  }

  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-screen text-kumo-inactive">
          Loading...
        </div>
      }
    >
      <AuthenticatedApp user={user!} />
    </Suspense>
  );
}
