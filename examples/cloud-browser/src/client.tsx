import { useState, useRef, useCallback, useEffect } from "react";
import { ModeToggle } from "@cloudflare/agents-ui";
import { Button, Text, Surface } from "@cloudflare/kumo";
import {
  PaperPlaneRight,
  Stop,
  Globe,
  Robot,
  Bug,
  Copy,
  X
} from "@phosphor-icons/react";

interface EventEntry {
  id: number;
  type: string;
  data: any;
}

export default function App() {
  const [task, setTask] = useState("");
  const [events, setEvents] = useState<EventEntry[]>([]);
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const eventIdRef = useRef(0);
  const currentMsgIdRef = useRef<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const [agentSentryTrace, setAgentSentryTrace] = useState<string | null>(null);
  const [showReportModal, setShowReportModal] = useState(false);
  const [reportDescription, setReportDescription] = useState("");
  const [reportId, setReportId] = useState<string | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [events]);

  const submitTask = useCallback(async () => {
    if (!task.trim() || running) return;

    setRunning(true);
    setEvents([]);
    setAgentSentryTrace(null);
    eventIdRef.current = 0;
    currentMsgIdRef.current = null;

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: task.trim() }),
        signal: controller.signal
      });

      if (!response.ok) {
        const err = await response.text();
        setEvents([
          {
            id: 0,
            type: "error",
            data: { message: `HTTP ${response.status}: ${err}` }
          }
        ]);
        setRunning(false);
        return;
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let currentEventType = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEventType = line.slice(7).trim();
          } else if (line.startsWith("data: ") && currentEventType) {
            try {
              const data = JSON.parse(line.slice(6));
              if (currentEventType === "trace_context") {
                setAgentSentryTrace(data.sentryTrace ?? null);
              } else if (
                currentEventType === "message_start" ||
                currentEventType === "message_update" ||
                currentEventType === "message_end"
              ) {
                const type = currentEventType;
                if (currentMsgIdRef.current === null) {
                  const id = eventIdRef.current++;
                  currentMsgIdRef.current = id;
                  setEvents((prev) => [...prev, { id, type, data }]);
                } else {
                  const msgId = currentMsgIdRef.current;
                  setEvents((prev) =>
                    prev.map((e) => (e.id === msgId ? { ...e, type, data } : e))
                  );
                }
                if (type === "message_end") {
                  currentMsgIdRef.current = null;
                }
              } else {
                const id = eventIdRef.current++;
                const type = currentEventType;
                setEvents((prev) => [...prev, { id, type, data }]);
              }
            } catch {
              // skip malformed JSON
            }
            currentEventType = "";
          }
        }
      }
    } catch (err: any) {
      if (err.name !== "AbortError") {
        const id = eventIdRef.current++;
        setEvents((prev) => [
          ...prev,
          { id, type: "error", data: { message: err.message } }
        ]);
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }, [task, running]);

  const stopAgent = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const submitReport = useCallback(async () => {
    setReportLoading(true);
    setReportError(null);
    try {
      // Strip screenshots from events to avoid huge payloads
      const strippedEvents = events.map((e) => {
        if (e.data?.result?.details?.screenshot) {
          const { screenshot, ...rest } = e.data.result.details;
          return {
            ...e,
            data: { ...e.data, result: { ...e.data.result, details: rest } }
          };
        }
        return e;
      });
      const res = await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          description: reportDescription,
          task,
          events: strippedEvents.slice(-50),
          ...(agentSentryTrace ? { sentryTrace: agentSentryTrace } : {})
        })
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      }
      const data = (await res.json()) as { reportId: string };
      setReportId(data.reportId);
    } catch (err: any) {
      setReportError(err.message || "Failed to submit report");
    } finally {
      setReportLoading(false);
    }
  }, [reportDescription, task, events, agentSentryTrace]);

  return (
    <div className="flex flex-col h-screen bg-kumo-elevated">
      <header className="px-5 py-4 bg-kumo-base border-b border-kumo-line">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Globe size={24} weight="duotone" className="text-kumo-accent" />
            <span className="text-kumo-default text-lg font-semibold">
              Cloud Browser Agent
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              onClick={() => {
                setShowReportModal(true);
                setReportId(null);
                setReportDescription("");
                setReportError(null);
              }}
            >
              <Bug size={18} />
              Report Bug
            </Button>
            <ModeToggle />
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-4 space-y-3" ref={scrollRef}>
        {events.length === 0 && !running && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-kumo-subtle">
            <Robot size={48} weight="duotone" />
            <Text variant="body">Enter a task to start browsing the web</Text>
          </div>
        )}

        {events.map((entry) => (
          <EventCard key={entry.id} entry={entry} />
        ))}

        {running && events.length > 0 && (
          <div className="flex items-center gap-2 text-kumo-subtle px-2">
            <div className="animate-spin h-4 w-4 border-2 border-kumo-accent border-t-transparent rounded-full" />
            <Text variant="secondary">Agent is working...</Text>
          </div>
        )}
      </div>

      <div className="px-4 pb-4 bg-kumo-elevated">
        <Surface className="flex gap-2 p-3 rounded-lg border border-kumo-line">
          <textarea
            className="flex-1 bg-transparent text-kumo-default placeholder-kumo-subtle resize-none outline-none text-sm"
            rows={2}
            placeholder="Describe a browsing task, e.g. 'Go to news.ycombinator.com and tell me the top 3 stories'"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submitTask();
              }
            }}
            disabled={running}
          />
          {running ? (
            <Button
              variant="secondary"
              onClick={stopAgent}
              className="self-end"
            >
              <Stop size={18} />
              Stop
            </Button>
          ) : (
            <Button
              variant="primary"
              onClick={submitTask}
              disabled={!task.trim()}
              className="self-end"
            >
              <PaperPlaneRight size={18} />
              Run
            </Button>
          )}
        </Surface>
      </div>

      {showReportModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Surface className="w-full max-w-md rounded-lg border border-kumo-line p-5 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-kumo-default font-semibold">
                Report Bug
              </span>
              <button
                className="text-kumo-subtle hover:text-kumo-default"
                onClick={() => setShowReportModal(false)}
              >
                <X size={20} />
              </button>
            </div>

            {reportId ? (
              <div className="space-y-3">
                <Text variant="body">Report submitted successfully.</Text>
                <div className="flex items-center gap-2 bg-kumo-elevated rounded p-3 border border-kumo-line">
                  <code className="flex-1 text-sm text-kumo-default font-mono break-all">
                    {reportId}
                  </code>
                  <button
                    className="text-kumo-subtle hover:text-kumo-default"
                    onClick={() => navigator.clipboard.writeText(reportId)}
                  >
                    <Copy size={18} />
                  </button>
                </div>
                <Button
                  variant="primary"
                  onClick={() => setShowReportModal(false)}
                  className="w-full"
                >
                  Done
                </Button>
              </div>
            ) : (
              <div className="space-y-3">
                <textarea
                  className="w-full bg-kumo-elevated text-kumo-default placeholder-kumo-subtle rounded border border-kumo-line p-3 resize-none outline-none text-sm"
                  rows={4}
                  placeholder="Describe what went wrong..."
                  value={reportDescription}
                  onChange={(e) => setReportDescription(e.target.value)}
                />
                {reportError && <Text variant="error">{reportError}</Text>}
                <Button
                  variant="primary"
                  onClick={submitReport}
                  disabled={reportLoading || !reportDescription.trim()}
                  className="w-full"
                >
                  {reportLoading ? "Submitting..." : "Submit Report"}
                </Button>
              </div>
            )}
          </Surface>
        </div>
      )}
    </div>
  );
}

function EventCard({ entry }: { entry: EventEntry }) {
  const { type, data } = entry;

  switch (type) {
    case "message_start":
    case "message_update":
      return (
        <Surface className="p-3 rounded-lg border border-kumo-line">
          <div className="text-kumo-subtle text-xs mb-1">Assistant</div>
          <div className="text-kumo-default whitespace-pre-wrap">
            {data.text}
          </div>
        </Surface>
      );

    case "message_end":
      return (
        <Surface className="p-3 rounded-lg border border-kumo-line bg-kumo-base">
          <div className="text-kumo-subtle text-xs mb-1">Assistant</div>
          <div className="text-kumo-default whitespace-pre-wrap">
            {data.text}
          </div>
        </Surface>
      );

    case "tool_execution_start":
      return (
        <Surface className="p-3 rounded-lg border border-kumo-line border-l-4 border-l-kumo-accent">
          <div className="text-kumo-accent text-sm">
            Calling browser.{data.args?.action ?? "unknown"}
            {data.args?.url ? ` \u2192 ${data.args.url}` : ""}
            {data.args?.selector ? ` on "${data.args.selector}"` : ""}
          </div>
        </Surface>
      );

    case "tool_execution_end": {
      const result = data.result;
      if (!result) return null;

      const details = result.details;
      const hasScreenshot = details?.screenshot;
      const textContent = result.content
        ?.filter((c: any) => c.type === "text")
        .map((c: any) => c.text)
        .join("\n");

      return (
        <Surface className="p-3 rounded-lg border border-kumo-line space-y-2">
          {textContent && (
            <div className="text-kumo-subtle text-xs font-mono whitespace-pre-wrap">
              {textContent.length > 500
                ? textContent.slice(0, 500) + "..."
                : textContent}
            </div>
          )}
          {hasScreenshot && (
            <img
              src={`data:image/png;base64,${details.screenshot}`}
              alt="Browser screenshot"
              className="rounded border border-kumo-line max-w-full"
            />
          )}
        </Surface>
      );
    }

    case "agent_end":
      return (
        <Surface className="p-3 rounded-lg border border-kumo-line">
          <Text variant="success">Task complete</Text>
        </Surface>
      );

    case "error":
      return (
        <Surface className="p-3 rounded-lg border border-red-400">
          <Text variant="error">Error: {data.message}</Text>
        </Surface>
      );

    default:
      return null;
  }
}
