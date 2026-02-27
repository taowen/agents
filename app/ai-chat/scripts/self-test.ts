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

console.log("Token ready, starting tests...");

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
      reject(new Error("WebSocket timeout after 180s"));
    }, 180_000);

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

// Helper: create a session, send one message, dump messages, cleanup
async function testStep(
  label: string,
  userMsg: string,
  checkFn: (parts: any[]) => boolean
) {
  console.log(`\n--- ${label} ---`);
  const res = await fetch(`${BASE}/api/sessions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ title: `self-test-${label}` })
  });
  const { id: sid } = (await res.json()) as { id: string };
  console.log(`Session: ${sid}`);

  await sendChat(sid, userMsg);
  console.log("Chat stream done.");

  const msgs = await getMessages(sid);
  const allParts: any[] = [];
  for (const m of msgs) {
    console.log(`[${m.role}]`);
    for (const p of m.parts || []) {
      allParts.push(p);
      if (p.type === "text") {
        console.log(`  text: ${p.text.slice(0, 300)}`);
      } else if (p.type?.startsWith("tool-")) {
        const inputStr =
          typeof p.input === "string" ? p.input : JSON.stringify(p.input);
        console.log(`  ${p.type} (${p.state}) input=${inputStr.slice(0, 200)}`);
        if (p.output) {
          const outStr =
            typeof p.output === "string" ? p.output : JSON.stringify(p.output);
          console.log(`  output: ${outStr.slice(0, 300)}`);
        }
      } else if (p.type === "reasoning") {
        console.log("  (reasoning)");
      } else {
        console.log(`  ${p.type}`);
      }
    }
  }

  const pass = checkFn(allParts);
  console.log(pass ? `✅ PASS: ${label}` : `❌ FAIL: ${label}`);

  // cleanup
  await fetch(`${BASE}/agents/chat-agent/${sid}/cancel-all-schedules`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` }
  }).catch(() => {});
  await fetch(`${BASE}/api/sessions/${sid}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${token}` }
  }).catch(() => {});
  await fetch(`${BASE}/agents/chat-agent/${sid}/destroy`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` }
  }).catch(() => {});
  return pass;
}

function hasBashWebFetch(parts: any[], withPrompt: boolean) {
  for (const p of parts) {
    if (p.type?.startsWith("tool-") && p.toolName === "bash") {
      const inputStr =
        typeof p.input === "string" ? p.input : JSON.stringify(p.input);
      if (inputStr.includes("web-fetch")) {
        if (!withPrompt) return true;
        if (inputStr.includes("extract")) return true;
      }
    }
  }
  return false;
}

const r1 = await testStep(
  "web-fetch (raw)",
  'Run: bash({command: "web-fetch https://example.com"})',
  (parts) => hasBashWebFetch(parts, false)
);

const r2 = await testStep(
  "web-fetch (with prompt)",
  "Run this exact command: bash({command: 'web-fetch https://example.com \"extract the page title and main heading\"'}). " +
    "The prompt must be in double quotes. Do NOT split into separate commands.",
  (parts) => hasBashWebFetch(parts, true)
);

const r3 = await testStep(
  "web-fetch (pipe+redirect)",
  "Run this exact bash command: web-fetch https://example.com > /home/user/test-fetch.txt && cat /home/user/test-fetch.txt | head -5",
  (parts) => {
    let hasRedirect = false;
    let hasExampleDomain = false;
    for (const p of parts) {
      if (p.type?.startsWith("tool-") && p.toolName === "bash") {
        const inputStr =
          typeof p.input === "string" ? p.input : JSON.stringify(p.input);
        if (
          inputStr.includes("web-fetch") &&
          (inputStr.includes("> /home/user/") ||
            inputStr.includes(">/home/user/"))
        ) {
          hasRedirect = true;
        }
        const outStr = p.output
          ? typeof p.output === "string"
            ? p.output
            : JSON.stringify(p.output)
          : "";
        if (outStr.includes("Example Domain")) {
          hasExampleDomain = true;
        }
      }
    }
    if (!hasRedirect) console.log("  (missing: redirect to file)");
    if (!hasExampleDomain)
      console.log("  (missing: 'Example Domain' in output)");
    return hasRedirect && hasExampleDomain;
  }
);

console.log(
  `\n=== Summary: ${r1 && r2 && r3 ? "ALL PASS" : "SOME FAILED"} ===`
);

// (cleanup is done per-step inside testStep)
