/**
 * Bug report submission logic — D1 query, DO debug context, R2 upload, Sentry.
 * Extracted from api.ts to keep route handlers thin and allow independent testing.
 */

import * as Sentry from "@sentry/cloudflare";

interface SubmitBugReportParams {
  db: D1Database;
  r2: R2Bucket | undefined;
  chatAgentNs: DurableObjectNamespace;
  userId: string;
  sessionId: string;
  description: string;
}

export async function submitBugReport({
  db,
  r2,
  chatAgentNs,
  userId,
  sessionId,
  description
}: SubmitBugReportParams): Promise<{ reportId: string }> {
  const reportId = `BUG-${Date.now().toString(36).toUpperCase()}-${Array.from(
    crypto.getRandomValues(new Uint8Array(2))
  )
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .toUpperCase()}`;

  // Compute sessionDir from the DO ID (first 12 hex chars)
  const isolatedName = encodeURIComponent(`${userId}:${sessionId}`);
  const doId = chatAgentNs.idFromName(isolatedName);
  const sessionDir = doId.toString().slice(0, 12);

  // Query D1 for recent chat messages
  const chatPrefix = `/home/user/.chat/${sessionDir}/`;
  const rows = await db
    .prepare(
      `SELECT path, CAST(content AS TEXT) as content FROM files
       WHERE user_id = ? AND parent_path = ? AND is_directory = 0
       ORDER BY mtime DESC LIMIT 10`
    )
    .bind(userId, chatPrefix.slice(0, -1))
    .all<{ path: string; content: string }>();

  const recentMessages = rows.results.map((r) => {
    const text = r.content || "";
    return `${r.path}: ${text.slice(0, 200)}`;
  });

  // Collect full debug context from the DO (ring buffer + full messages)
  let debugContext: {
    debugEntries?: Array<{ type: string }>;
  } | null = null;
  try {
    const doStub = chatAgentNs.get(doId);
    const res = await doStub.fetch(
      new Request("http://agent/collect-debug-context")
    );
    if (res.ok) debugContext = await res.json();
  } catch (e) {
    console.error("collect-debug-context failed:", e);
  }

  // Write full payload to R2 for later investigation
  const r2Key = `bug-reports/${reportId}.json`;
  const fullPayload = JSON.stringify({
    reportId,
    description,
    userId,
    sessionId,
    recentMessages,
    debugContext,
    capturedAt: new Date().toISOString()
  });
  if (r2) {
    try {
      await r2.put(r2Key, fullPayload, {
        httpMetadata: { contentType: "application/json" }
      });
    } catch (e) {
      console.error("R2 bug-report write failed:", e);
    }
  }

  Sentry.withScope((scope) => {
    scope.setUser({ id: userId });
    scope.setTag("report_id", reportId);
    scope.setTag("user_id", userId);
    scope.setTag("session_uuid", sessionId);
    scope.setTag("r2_debug_key", r2Key);
    scope.setContext("bug_report", {
      description,
      reportId,
      sessionUuid: sessionId,
      userId,
      hasLlmDebug: !!debugContext,
      llmInteractionCount:
        debugContext?.debugEntries?.filter((e) => e.type === "llm").length ?? 0,
      doCallCount:
        debugContext?.debugEntries?.filter((e) => e.type === "do_call")
          .length ?? 0,
      r2DebugKey: r2Key
    });
    scope.setContext("recent_messages", { messages: recentMessages });
    Sentry.captureMessage(`[Bug Report ${reportId}] ${description}`, "warning");
  });

  return { reportId };
}
