// Usage: npx tsx examples/ai-chat/scripts/self-test.ts [base-url]
//
// Authenticates via device auth flow, then tests API endpoints.
// Token is cached in scripts/.self-test-token for reuse across runs.
// Delete that file to force re-authentication.

import { readFileSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

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

// Test /api/usage
const now = new Date();
const start = now.toISOString().slice(0, 13); // e.g. "2025-02-20T00"
const end = now.toISOString().slice(0, 13); // current hour
const usageRes = await fetch(`${BASE}/api/usage?start=${start}&end=${end}`, {
  headers: { Authorization: `Bearer ${token}` }
});
if (!usageRes.ok) {
  console.error("Usage API error:", usageRes.status, await usageRes.text());
  process.exit(1);
}
const usage = await usageRes.json();
console.log("GET /api/usage response:");
console.log(JSON.stringify(usage, null, 2));
