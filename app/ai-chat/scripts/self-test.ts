// Usage: npx tsx app/ai-chat/scripts/self-test.ts [base-url]
//
// Authenticates via device auth flow, then tests API endpoints.
// Token is cached in scripts/.self-test-token for reuse across runs.
// Delete that file to force re-authentication.

import { readFileSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import WebSocket from "ws";

const BASE = process.argv[2] || "https://ai.connect-screen.com";
const TOKEN_FILE = join(
  dirname(fileURLToPath(import.meta.url)),
  ".self-test-token"
);

async function getToken(): Promise<string> {
  // Try cached token
  try {
    const cached = readFileSync(TOKEN_FILE, "utf-8").trim();
    if (cached) {
      const res = await fetch(
        `${BASE}/api/usage?start=2024-01-01T00&end=2024-01-01T01`,
        {
          headers: { Authorization: `Bearer ${cached}` }
        }
      );
      if (res.ok) {
        console.log("Using cached token.");
        return cached;
      }
      console.log("Cached token expired, re-authenticating...");
    }
  } catch {}

  // Device auth flow
  const startRes = await fetch(`${BASE}/auth/device/start`, { method: "POST" });
  if (!startRes.ok) {
    console.error(
      "Failed to start device auth:",
      startRes.status,
      await startRes.text()
    );
    process.exit(1);
  }
  const { code } = (await startRes.json()) as { code: string };

  console.log("");
  console.log("========================================");
  console.log(`  DEVICE CODE: ${code}`);
  console.log(`  Approve at:  ${BASE}/device`);
  console.log("========================================");
  console.log("");

  // Poll until approved
  while (true) {
    await new Promise((r) => setTimeout(r, 2000));
    const res = await fetch(`${BASE}/auth/device/check?code=${code}`);
    const data = (await res.json()) as { status: string; token?: string };
    if (data.status === "approved" && data.token) {
      writeFileSync(TOKEN_FILE, data.token);
      console.log("\nApproved! Token cached to .self-test-token\n");
      return data.token;
    }
    if (data.status === "expired") {
      console.error("\nCode expired. Run again to get a new code.");
      process.exit(1);
    }
    process.stdout.write(".");
  }
}

const token = await getToken();

// Create a proper D1 session so the scheduled task's D1 existence check passes
const createRes = await fetch(`${BASE}/api/sessions`, {
  method: "POST",
  headers: {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({ title: "self-test-schedule" })
});
if (!createRes.ok) {
  console.error(
    "Failed to create session:",
    createRes.status,
    await createRes.text()
  );
  process.exit(1);
}
const { id: SESSION } = (await createRes.json()) as { id: string };
console.log("Created D1 session:", SESSION);

// Helper: send a chat message via WebSocket and collect the full streamed response
async function sendChat(session: string, userContent: string): Promise<string> {
  const wsUrl = BASE.replace(/^http/, "ws") + `/agents/chat-agent/${session}`;
  console.log("Connecting to:", wsUrl);
  const ws = new WebSocket(wsUrl, {
    headers: { Authorization: `Bearer ${token}` }
  });

  return new Promise<string>((resolve, reject) => {
    const timeout = setTimeout(() => {
      ws.close();
      reject(new Error("WebSocket timeout after 90s"));
    }, 90_000);

    let collected = "";

    ws.on("open", () => {
      console.log("WebSocket connected!");
      const reqId = Math.random().toString(36).slice(2, 10);
      const payload = {
        id: reqId,
        type: "cf_agent_use_chat_request",
        init: {
          method: "POST",
          body: JSON.stringify({
            messages: [{ role: "user", content: userContent }]
          })
        }
      };
      ws.send(JSON.stringify(payload));
      console.log("Sent:", userContent.slice(0, 100));
    });

    ws.on("message", (data: Buffer) => {
      const raw = data.toString();
      try {
        const msg = JSON.parse(raw);
        if (msg.type === "cf_agent_use_chat_response") {
          if (msg.body) collected += msg.body;
          if (msg.done) {
            clearTimeout(timeout);
            ws.close();
            resolve(collected);
          }
        }
      } catch {}
    });

    ws.on("error", (err: Error) => {
      clearTimeout(timeout);
      reject(err);
    });
  });
}

// Helper: get messages from the DO
async function getMessages(session: string): Promise<any[]> {
  const wsUrl = BASE.replace(/^http/, "ws") + `/agents/chat-agent/${session}`;
  const ws = new WebSocket(wsUrl, {
    headers: { Authorization: `Bearer ${token}` }
  });
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      ws.terminate();
      reject(new Error("getMessages timeout"));
    }, 15_000);
    let resolved = false;
    ws.on("message", (data: Buffer) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.type === "cf_agent_chat_messages" && !resolved) {
          resolved = true;
          clearTimeout(timeout);
          const messages = msg.messages || [];
          // Close the WS after a short delay to let the server process the close
          setTimeout(() => ws.terminate(), 100);
          resolve(messages);
        }
      } catch {}
    });
    ws.on("error", (err: Error) => {
      if (!resolved) {
        clearTimeout(timeout);
        reject(err);
      }
    });
  });
}

console.log(`\nUsing session: ${SESSION}`);

// ---- Step 1: Ask the agent to schedule a task 10s from now ----
console.log("\n--- Step 1: Ask agent to schedule a delayed task ---");
const chatResult = await sendChat(
  SESSION,
  'Schedule a one-time task to run in 10 seconds from now. The task description should be "self-test-delayed" and the prompt should be "reply with exactly: SCHEDULED_TASK_OK". Use the schedule_task tool with a delay. Do NOT use bash.'
);
console.log("Agent response stream length:", chatResult.length);

// Verify the schedule_task tool was called
if (chatResult.includes("schedule_task") || chatResult.includes("schedule")) {
  console.log("✅ schedule_task tool was invoked in the response.");
} else {
  console.log(
    "⚠️ schedule_task tool may not have been invoked. Checking anyway..."
  );
}

// ---- Step 2: Verify schedule was created ----
console.log("\n--- Step 2: Check schedules ---");
const schedRes = await fetch(
  `${BASE}/agents/chat-agent/${SESSION}/get-schedules`,
  {
    headers: { Authorization: `Bearer ${token}` }
  }
);
const schedules = await schedRes.json();
console.log("Schedules:", JSON.stringify(schedules, null, 2));

// ---- Step 3: Wait for the scheduled task to fire ----
console.log("\n--- Step 3: Waiting 20s for scheduled task to execute ---");
await new Promise((r) => setTimeout(r, 20_000));

// ---- Step 4: Read messages and verify ----
console.log("\n--- Step 4: Check messages for [Scheduled Task] ---");
const messages = await getMessages(SESSION);
console.log(`Total messages: ${messages.length}`);

let foundScheduledUser = false;
let foundScheduledAssistant = false;
for (const m of messages) {
  for (const p of m.parts || []) {
    if (p.type === "text" && typeof p.text === "string") {
      if (p.text.includes("[Scheduled Task]") && m.role === "user") {
        foundScheduledUser = true;
        console.log(
          "Found [Scheduled Task] user message:",
          p.text.slice(0, 120)
        );
      }
      if (
        m.role === "assistant" &&
        foundScheduledUser &&
        !foundScheduledAssistant
      ) {
        foundScheduledAssistant = true;
        console.log("Found assistant response:", p.text.slice(0, 200));
      }
    }
    // Also check for tool invocations in the scheduled task response
    if (p.type === "tool-invocation") {
      console.log(`  Tool call: ${p.toolName} (state: ${p.state})`);
    }
  }
}

if (foundScheduledUser && foundScheduledAssistant) {
  console.log("\n✅ Scheduled task executed via unified chat path!");
  console.log("   - [Scheduled Task] user message present");
  console.log(
    "   - Assistant response present (went through streamText + onChatMessage)"
  );
} else if (foundScheduledUser) {
  console.log(
    "\n⚠️ Scheduled task user message found but no assistant response yet."
  );
  console.log("   The task may still be processing or failed.");
} else {
  console.log(
    "\n❌ No [Scheduled Task] message found. The task may not have fired yet."
  );
  console.log(
    "   Messages found:",
    messages.map(
      (m: any) => `${m.role}: ${(m.parts?.[0]?.text || "").slice(0, 60)}`
    )
  );
}

// ---- Cleanup ----
console.log("\n--- Cleanup ---");
await fetch(`${BASE}/agents/chat-agent/${SESSION}/cancel-all-schedules`, {
  method: "POST",
  headers: { Authorization: `Bearer ${token}` }
});
await fetch(`${BASE}/api/sessions/${SESSION}`, {
  method: "DELETE",
  headers: { Authorization: `Bearer ${token}` }
});
await fetch(`${BASE}/agents/chat-agent/${SESSION}/destroy`, {
  method: "POST",
  headers: { Authorization: `Bearer ${token}` }
});
console.log("Session and schedules cleaned up.");
