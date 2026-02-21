import { useCallback, useState, useEffect, useRef } from "react";
import { useParams, useOutletContext } from "react-router";
import { useAgent } from "agents/react";
import { useAgentChat } from "@cloudflare/ai-chat/react";
import { isToolUIPart } from "ai";
import type { UIMessage, FileUIPart } from "ai";
import { Button, Badge, InputArea, Empty, Text } from "@cloudflare/kumo";
import { useQuotaStatus, useInitialMessages } from "./api";
import type { AuthLayoutContext } from "./AuthLayout";
import {
  ConnectionIndicator,
  ModeToggle,
  type ConnectionStatus
} from "@cloudflare/agents-ui";
import {
  PaperPlaneRightIcon,
  StopIcon,
  CloudSunIcon,
  ListIcon,
  FolderIcon,
  BugIcon
} from "@phosphor-icons/react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ToolOutput } from "./ToolOutput";
import { FileManagerPanel } from "./FileManagerPanel";
import {
  IMAGE_EXTENSIONS,
  MIME_TYPES,
  getExtension
} from "../shared/file-utils";
const AT_REF_REGEX = /@(\/[\w./-]+\.\w+)/g;

async function fetchImageAsFileUIPart(
  path: string
): Promise<FileUIPart | null> {
  const ext = getExtension(path);
  if (!IMAGE_EXTENSIONS.has(ext)) return null;
  const mediaType = MIME_TYPES[ext] || "application/octet-stream";
  try {
    const res = await fetch(
      `/api/files/content?path=${encodeURIComponent(path)}`
    );
    if (!res.ok) return null;
    const blob = await res.blob();
    const dataUrl = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.readAsDataURL(blob);
    });
    const filename = path.split("/").pop() || path;
    return { type: "file", mediaType, url: dataUrl, filename };
  } catch {
    return null;
  }
}

function getMessageText(message: UIMessage): string {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => (part as { type: "text"; text: string }).text)
    .join("");
}

function ChatSkeleton({ onOpenSidebar }: { onOpenSidebar: () => void }) {
  return (
    <div className="flex flex-col h-screen bg-kumo-elevated">
      <header className="px-4 md:px-5 py-3 md:py-4 bg-kumo-base border-b border-kumo-line">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={onOpenSidebar}
              className="md:hidden p-1.5 -ml-1.5 rounded-lg hover:bg-kumo-elevated text-kumo-secondary hover:text-kumo-default transition-colors"
            >
              <ListIcon size={20} />
            </button>
            <h1 className="text-sm sm:text-lg font-semibold text-kumo-default">
              Work With Your Agent
            </h1>
          </div>
          <div className="flex items-center gap-3">
            <div className="h-5 w-20 rounded bg-kumo-line animate-pulse" />
            <div className="h-8 w-16 rounded bg-kumo-line animate-pulse" />
          </div>
        </div>
      </header>
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 md:px-5 py-6 space-y-5">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className={`flex ${i % 2 === 1 ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] px-4 py-2.5 rounded-2xl ${i % 2 === 1 ? "rounded-br-md bg-kumo-contrast/10" : "rounded-bl-md bg-kumo-base"}`}
              >
                <div className="h-4 w-48 rounded bg-kumo-line animate-pulse" />
                {i % 2 === 0 && (
                  <div className="h-4 w-32 rounded bg-kumo-line animate-pulse mt-2" />
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="border-t border-kumo-line bg-kumo-base">
        <div className="max-w-3xl mx-auto px-4 md:px-5 py-3 md:py-4">
          <div className="flex items-end gap-3 rounded-xl border border-kumo-line bg-kumo-base p-3">
            <div className="flex-1 h-12 rounded bg-kumo-line animate-pulse" />
            <div className="h-9 w-9 rounded bg-kumo-line animate-pulse mb-0.5" />
          </div>
        </div>
      </div>
    </div>
  );
}

export function Chat() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const { messages: initialMessages, isLoading } =
    useInitialMessages(sessionId);
  const { onFirstMessage, onOpenSidebar, onOpenBugReport } =
    useOutletContext<AuthLayoutContext>();

  if (isLoading) {
    return <ChatSkeleton onOpenSidebar={onOpenSidebar} />;
  }

  return (
    <ChatInner
      key={sessionId}
      sessionId={sessionId!}
      initialMessages={initialMessages}
      onFirstMessage={onFirstMessage}
      onOpenSidebar={onOpenSidebar}
      onOpenBugReport={onOpenBugReport}
    />
  );
}

function ChatInner({
  sessionId,
  initialMessages,
  onFirstMessage,
  onOpenSidebar,
  onOpenBugReport
}: {
  sessionId: string;
  initialMessages: UIMessage[];
  onFirstMessage: (text: string) => void;
  onOpenSidebar: () => void;
  onOpenBugReport: () => void;
}) {
  const isDeviceSession = sessionId.startsWith("device-");
  const deviceName = isDeviceSession ? sessionId.slice("device-".length) : null;

  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("connecting");
  const [input, setInput] = useState("");
  const inputRef = useRef(input);
  inputRef.current = input;
  const [fileManagerOpen, setFileManagerOpen] = useState(false);
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

  const { messages, sendMessage, addToolApprovalResponse, status, stop } =
    useAgentChat({
      agent,
      getInitialMessages: null,
      messages: initialMessages,
      body: {
        clientVersion: "1.0.0",
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
      }
    });

  const { quota } = useQuotaStatus();

  const isStreaming = status === "streaming";
  const isConnected = connectionStatus === "connected";

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = useCallback(async () => {
    const text = inputRef.current.trim();
    if (!text) return;
    inputRef.current = "";
    setInput("");

    // Collect file parts from @references in text
    const refPaths: string[] = [];
    for (const m of text.matchAll(AT_REF_REGEX)) {
      refPaths.push(m[1]);
    }
    const refResults = await Promise.all(
      refPaths.map((p) => fetchImageAsFileUIPart(p))
    );
    const fileParts = refResults.filter(Boolean) as FileUIPart[];

    const parts: Array<{ type: "text"; text: string } | FileUIPart> = [
      { type: "text", text }
    ];
    for (const fp of fileParts) parts.push(fp);

    if (isStreaming) {
      stop();
    }

    sendMessage({ role: "user", parts });
    if (!firstMessageSent.current) {
      firstMessageSent.current = true;
      onFirstMessage(text);
    }
  }, [isStreaming, sendMessage, stop, onFirstMessage]);

  const handleInsertFile = useCallback((path: string) => {
    setInput((prev) => {
      const prefix = prev.length > 0 && !prev.endsWith(" ") ? " " : "";
      return prev + prefix + "@" + path;
    });
  }, []);

  return (
    <div className="flex flex-col h-screen bg-kumo-elevated">
      {/* Header */}
      <header className="px-4 md:px-5 py-3 md:py-4 bg-kumo-base border-b border-kumo-line">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={onOpenSidebar}
              className="md:hidden p-1.5 -ml-1.5 rounded-lg hover:bg-kumo-elevated text-kumo-secondary hover:text-kumo-default transition-colors"
            >
              <ListIcon size={20} />
            </button>
            <h1 className="text-sm sm:text-lg font-semibold text-kumo-default">
              {deviceName ? `Device: ${deviceName}` : "Work With Your Agent"}
            </h1>
          </div>
          <div className="flex items-center gap-3">
            <ConnectionIndicator status={connectionStatus} />
            {!isDeviceSession && (
              <Button
                variant="secondary"
                icon={<FolderIcon size={16} />}
                onClick={() => setFileManagerOpen(true)}
                title="Files"
              >
                Files
              </Button>
            )}
            <ModeToggle />
            <Button
              variant="secondary"
              shape="square"
              aria-label="Report bug"
              icon={<BugIcon size={16} />}
              onClick={onOpenBugReport}
              title="Report Bug"
            />
          </div>
        </div>
      </header>

      {/* Quota exceeded banner */}
      {quota?.exceeded && (
        <div className="px-4 md:px-5 py-2 bg-amber-50 dark:bg-amber-950/30 border-b border-amber-200 dark:border-amber-800">
          <div className="max-w-3xl mx-auto">
            <Text size="sm" variant="secondary">
              <span className="text-amber-700 dark:text-amber-400">
                Builtin API key quota exceeded. Please configure your own API
                key in Settings to continue.
              </span>
            </Text>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 md:px-5 py-6 space-y-5">
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
                      {message.parts
                        .filter((p) => p.type === "file")
                        .map((p, i) => (
                          <img
                            key={i}
                            src={(p as FileUIPart).url}
                            alt={(p as FileUIPart).filename || "attachment"}
                            className="max-w-full rounded mt-2"
                          />
                        ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex justify-start">
                    <div className="max-w-[85%] px-4 py-2.5 rounded-2xl rounded-bl-md bg-kumo-base text-kumo-default leading-relaxed">
                      <div className="prose prose-sm max-w-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {getMessageText(message)}
                        </ReactMarkdown>
                        {isLastAssistant && isStreaming && (
                          <span className="inline-block w-0.5 h-[1em] bg-kumo-brand ml-0.5 align-text-bottom animate-blink-cursor" />
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {!isUser &&
                  (message as any).metadata?.usage &&
                  (() => {
                    const u = (message as any).metadata.usage;
                    const cached = u.cacheReadTokens || 0;
                    const uncached = (u.inputTokens || 0) - cached;
                    const output = u.outputTokens || 0;
                    return (
                      <div className="text-[11px] text-kumo-tertiary mt-1 ml-1 font-mono">
                        <span className="text-green-500">{cached}</span>
                        <span className="opacity-50"> cached </span>
                        <span className="opacity-50">| </span>
                        <span>{uncached}</span>
                        <span className="opacity-50"> input </span>
                        <span className="opacity-50">| </span>
                        <span className="text-blue-500">{output}</span>
                        <span className="opacity-50"> output</span>
                      </div>
                    );
                  })()}

                {message.parts
                  .filter((part) => isToolUIPart(part))
                  .map((part) => {
                    if (!isToolUIPart(part)) return null;
                    return (
                      <ToolOutput
                        key={part.toolCallId}
                        part={part}
                        addToolApprovalResponse={addToolApprovalResponse}
                      />
                    );
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
          className="max-w-3xl mx-auto px-4 md:px-5 py-3 md:py-4"
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
              placeholder={
                isStreaming
                  ? "Type to interrupt..."
                  : "Try: What's the weather in Paris?"
              }
              disabled={!isConnected}
              rows={2}
              className="flex-1 !ring-0 focus:!ring-0 !shadow-none !bg-transparent !outline-none"
            />
            {isStreaming && (
              <Button
                type="button"
                variant="secondary"
                shape="square"
                aria-label="Stop generation"
                icon={<StopIcon size={18} />}
                onClick={() => stop()}
                className="mb-0.5"
              />
            )}
            <Button
              type="submit"
              variant="primary"
              shape="square"
              aria-label="Send message"
              disabled={!input.trim() || !isConnected}
              icon={<PaperPlaneRightIcon size={18} />}
              className="mb-0.5"
            />
          </div>
        </form>
      </div>

      <FileManagerPanel
        open={fileManagerOpen}
        onClose={() => setFileManagerOpen(false)}
        onInsertFile={handleInsertFile}
      />
    </div>
  );
}
