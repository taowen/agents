#!/usr/bin/env node
/**
 * Standalone Windows Agent CLI — runs the agent loop directly on Node.js.
 *
 * Usage:
 *   npx tsx standalone.ts "Take a screenshot and describe what you see"
 *
 * Environment variables:
 *   LLM_PROVIDER  - "openai-compatible" (default) or "google"
 *   LLM_BASE_URL  - API base URL (required for openai-compatible)
 *   LLM_API_KEY   - API key (required)
 *   LLM_MODEL     - Model name (required)
 */
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { screenControl } from "./win-automation.ts";
import { createAgentLoop } from "../src/shared/agent-loop.ts";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { NodeFsAdapter } from "./node-fs-adapter.ts";
import { detectDrives } from "./detect-drives.ts";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import type { LanguageModel } from "ai";

// ---- Parse CLI args ----

const prompt = process.argv.slice(2).join(" ").trim();
if (!prompt) {
  console.error("Usage: npx tsx standalone.ts <prompt>");
  console.error(
    '  e.g. npx tsx standalone.ts "Take a screenshot and describe what you see"'
  );
  process.exit(1);
}

// ---- Build LLM model from env vars ----

const provider = process.env.LLM_PROVIDER || "openai-compatible";
const baseURL = process.env.LLM_BASE_URL;
const apiKey = process.env.LLM_API_KEY;
const modelName = process.env.LLM_MODEL;

if (!apiKey) {
  console.error("Error: LLM_API_KEY environment variable is required");
  process.exit(1);
}
if (!modelName) {
  console.error("Error: LLM_MODEL environment variable is required");
  process.exit(1);
}

let model: LanguageModel;
if (provider === "google") {
  model = createGoogleGenerativeAI({
    baseURL: baseURL || undefined,
    apiKey
  })(modelName);
} else {
  if (!baseURL) {
    console.error(
      "Error: LLM_BASE_URL environment variable is required for openai-compatible provider"
    );
    process.exit(1);
  }
  model = createOpenAICompatible({
    name: "llm",
    baseURL,
    apiKey
  })(modelName);
}

// ---- Always-on log directory ----

const logDir = path.join(os.tmpdir(), "windows-agent-standalone", "logs");
fs.mkdirSync(logDir, { recursive: true });
// Clean previous run
for (const f of fs.readdirSync(logDir)) fs.unlinkSync(path.join(logDir, f));

const logFile = path.join(logDir, "agent.log");

function log(msg: string) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  process.stderr.write(line + "\n");
  fs.appendFileSync(logFile, line + "\n");
}

function saveScreenshot(step: number, action: string, base64: string) {
  const filename = `step-${String(step).padStart(2, "0")}-${action}.png`;
  fs.writeFileSync(path.join(logDir, filename), Buffer.from(base64, "base64"));
  log(`screenshot saved: ${filename}`);
}

// ---- Exit when parent dies (stdin pipe closes) ----
process.stdin.resume();
process.stdin.on("end", () => {
  log("[standalone] stdin closed, parent died — exiting");
  process.exit(0);
});

// ---- Run agent ----

const mountableFs = new MountableFs({ base: new InMemoryFs() });
for (const drive of detectDrives()) {
  mountableFs.mount(drive.mountPoint, new NodeFsAdapter(drive.root), "winfs");
}
const bash = new Bash({ fs: mountableFs, cwd: "/home" });

const agent = createAgentLoop({
  getModel: () => model,
  executeBash: (cmd) => bash.exec(cmd),
  executeScreenControl: screenControl,
  maxSteps: 20
});

log(`[standalone] Provider: ${provider}`);
log(`[standalone] Model: ${modelName}`);
log(`[standalone] Prompt: ${prompt}`);
log(`[standalone] Log dir: ${logDir}`);
log("");

try {
  const response = await agent.runAgent(prompt, {
    onLog: log,
    onScreenshot: saveScreenshot
  });
  log("");
  log("[standalone] Done.");
  console.log(response);
} catch (err) {
  const msg = err instanceof Error ? err.message : String(err);
  console.error(`[standalone] Fatal error: ${msg}`);
  process.exit(1);
}
