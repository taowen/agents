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
  CloudSunIcon,
  GithubLogoIcon,
  XIcon,
  ArrowSquareOutIcon
} from "@phosphor-icons/react";

function getMessageText(message: UIMessage): string {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => (part as { type: "text"; text: string }).text)
    .join("");
}

function GitHubSetupModal({
  open,
  onClose
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [clientId, setClientId] = useState("");
  const [clientSecret, setClientSecret] = useState("");
  const [configured, setConfigured] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (open) {
      setError("");
      fetch("/oauth/github/config?agent_id=default")
        .then((res) => res.json())
        .then((data: { clientId?: string; configured?: boolean }) => {
          if (data.clientId) setClientId(data.clientId);
          setConfigured(!!data.configured);
        })
        .catch(() => {});
    }
  }, [open]);

  const callbackUrl =
    typeof window !== "undefined"
      ? `${window.location.origin}/oauth/github/callback`
      : "";

  const handleSaveAndConnect = async () => {
    if (!clientId.trim()) {
      setError("Client ID is required");
      return;
    }
    if (!clientSecret.trim() && !configured) {
      setError("Client Secret is required");
      return;
    }
    setSaving(true);
    setError("");
    try {
      const body: Record<string, string> = { clientId: clientId.trim() };
      if (clientSecret.trim()) body.clientSecret = clientSecret.trim();
      const res = await fetch("/oauth/github/config?agent_id=default", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!res.ok) throw new Error(await res.text());
      window.location.href = "/oauth/github?agent_id=default";
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
      setSaving(false);
    }
  };

  if (!open) return null;

  const inputClass =
    "w-full px-3 py-2 rounded-lg border border-kumo-line bg-kumo-elevated text-kumo-default text-sm focus:outline-none focus:ring-2 focus:ring-kumo-ring font-mono";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg mx-4 rounded-xl ring ring-kumo-line overflow-hidden bg-kumo-base"
        onClick={(e: React.MouseEvent) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-kumo-line">
          <div className="flex items-center gap-2">
            <GithubLogoIcon size={20} className="text-kumo-default" />
            <span className="text-sm font-semibold text-kumo-default">
              Connect GitHub
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-kumo-elevated text-kumo-inactive hover:text-kumo-default transition-colors"
          >
            <XIcon size={18} />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4">
          {/* Instructions */}
          <div className="space-y-2 text-sm text-kumo-secondary">
            <span className="text-xs font-semibold text-kumo-default">
              Setup Instructions
            </span>
            <ol className="list-decimal list-inside space-y-1.5 text-kumo-secondary">
              <li>
                <a
                  href="https://github.com/settings/applications/new"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-kumo-brand hover:underline inline-flex items-center gap-1"
                >
                  Create a GitHub OAuth App
                  <ArrowSquareOutIcon size={12} />
                </a>
              </li>
              <li>
                Set <strong>Homepage URL</strong> to:{" "}
                <code className="px-1 py-0.5 rounded bg-kumo-elevated text-xs">
                  {typeof window !== "undefined" ? window.location.origin : ""}
                </code>
              </li>
              <li>
                Set <strong>Authorization callback URL</strong> to:
                <div className="mt-1">
                  <code className="block px-2 py-1 rounded bg-kumo-elevated text-xs break-all select-all">
                    {callbackUrl}
                  </code>
                </div>
              </li>
              <li>
                Copy the <strong>Client ID</strong> and generate a{" "}
                <strong>Client Secret</strong> below
              </li>
            </ol>
          </div>

          {/* Form */}
          <div className="space-y-3">
            <div>
              <label className="block text-xs font-medium text-kumo-secondary mb-1">
                Client ID
              </label>
              <input
                type="text"
                value={clientId}
                onChange={(e) => setClientId(e.target.value)}
                placeholder="Ov23li..."
                className={inputClass}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-kumo-secondary mb-1">
                Client Secret
                {configured && (
                  <span className="ml-1 text-kumo-inactive font-normal">
                    (leave blank to keep current)
                  </span>
                )}
              </label>
              <input
                type="password"
                value={clientSecret}
                onChange={(e) => setClientSecret(e.target.value)}
                placeholder={configured ? "********" : "Enter client secret"}
                className={inputClass}
              />
            </div>
          </div>

          {/* Error */}
          {error && <p className="text-xs text-red-500">{error}</p>}

          {/* Actions */}
          <div className="flex items-center justify-between pt-2">
            {configured && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  window.location.href = "/oauth/github?agent_id=default";
                }}
              >
                Connect with existing config
              </Button>
            )}
            <div className={configured ? "" : "ml-auto"}>
              <Button
                variant="primary"
                size="sm"
                icon={<GithubLogoIcon size={14} />}
                onClick={handleSaveAndConnect}
                loading={saving}
              >
                {configured ? "Save & Reconnect" : "Save & Connect"}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Chat() {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("connecting");
  const [input, setInput] = useState("");
  const [showGitHubSetup, setShowGitHubSetup] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const agent = useAgent({
    agent: "ChatAgent",
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
    // Custom data sent with every request (available in options.body on server)
    body: {
      clientVersion: "1.0.0"
    },
    // Handle client-side tools (tools without server execute function)
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
  }, [input, isStreaming, sendMessage]);

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
              icon={<GithubLogoIcon size={16} />}
              onClick={() => setShowGitHubSetup(true)}
            >
              GitHub
            </Button>
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
              description='Try "What is the weather in London?" or "What timezone am I in?" or "Calculate 150 * 3 (amount: $450)"'
            />
          )}

          {messages.map((message, index) => {
            const isUser = message.role === "user";
            const isLastAssistant =
              message.role === "assistant" && index === messages.length - 1;

            return (
              <div key={message.id} className="space-y-2">
                {/* Text content */}
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

                {/* Tool parts */}
                {message.parts
                  .filter((part) => isToolUIPart(part))
                  .map((part) => {
                    if (!isToolUIPart(part)) return null;
                    const toolName = getToolName(part);

                    // Tool completed
                    if (part.state === "output-available") {
                      const isBash = toolName === "bash";
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

                      return (
                        <div
                          key={part.toolCallId}
                          className="flex justify-start"
                        >
                          <Surface className="max-w-[85%] px-4 py-2.5 rounded-xl ring ring-kumo-line">
                            <div className="flex items-center gap-2 mb-1">
                              <GearIcon
                                size={14}
                                className="text-kumo-inactive"
                              />
                              <Text size="xs" variant="secondary" bold>
                                {toolName}
                              </Text>
                              {isBash && bashOutput ? (
                                bashOutput.exitCode === 0 ? (
                                  <Badge variant="secondary">Done</Badge>
                                ) : (
                                  <Badge variant="destructive">
                                    Exit {bashOutput.exitCode}
                                  </Badge>
                                )
                              ) : (
                                <Badge variant="secondary">Done</Badge>
                              )}
                            </div>
                            {isBash && bashInput?.command && (
                              <div className="font-mono bg-kumo-elevated rounded px-2 py-1 mb-1">
                                <Text size="xs" variant="secondary">
                                  $ {bashInput.command}
                                </Text>
                              </div>
                            )}
                            <div className="font-mono whitespace-pre-wrap">
                              {isBash && bashOutput ? (
                                <>
                                  {bashOutput.stdout && (
                                    <Text size="xs" variant="secondary">
                                      {bashOutput.stdout}
                                    </Text>
                                  )}
                                  {bashOutput.stderr && (
                                    <Text size="xs" variant="error">
                                      {bashOutput.stderr}
                                    </Text>
                                  )}
                                </>
                              ) : (
                                <Text size="xs" variant="secondary">
                                  {JSON.stringify(part.output, null, 2)}
                                </Text>
                              )}
                            </div>
                          </Surface>
                        </div>
                      );
                    }

                    // Tool needs approval
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

                    // Tool executing
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

      {/* GitHub Setup Modal */}
      <GitHubSetupModal
        open={showGitHubSetup}
        onClose={() => setShowGitHubSetup(false)}
      />
    </div>
  );
}

export default function App() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-screen text-kumo-inactive">
          Loading...
        </div>
      }
    >
      <Chat />
    </Suspense>
  );
}
