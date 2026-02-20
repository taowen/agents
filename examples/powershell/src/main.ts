#!/usr/bin/env node
/**
 * Standalone Windows Agent CLI — runs the agent loop directly on Node.js.
 *
 * Usage:
 *   npx tsx src/main.ts "Take a screenshot and describe what you see"
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
import { fileURLToPath } from "node:url";

// ---- Load .env file ----
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const envPath = path.join(__dirname, "..", ".env");
if (fs.existsSync(envPath)) {
  for (const line of fs.readFileSync(envPath, "utf-8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 0) continue;
    const key = trimmed.slice(0, eq).trim();
    const val = trimmed.slice(eq + 1).trim();
    if (!process.env[key]) process.env[key] = val;
  }
}

import { screenControl, runPowerShellCommand } from "./win-automation.js";
import { createDesktopAgent } from "./agent-loop.js";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import type { LanguageModel } from "ai";
import type { AssistantMessage, TextContent } from "pi";

// ---- Parse CLI args ----

const prompt = process.argv.slice(2).join(" ").trim();
if (!prompt) {
  console.error("Usage: npx tsx src/main.ts <prompt>");
  console.error(
    '  e.g. npx tsx src/main.ts "Take a screenshot and describe what you see"'
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
  })(modelName) as any as LanguageModel;
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
  })(modelName) as any as LanguageModel;
}

// ---- Log directory ----

const logDir = path.join(os.tmpdir(), "powershell-agent", "logs");
fs.mkdirSync(logDir, { recursive: true });
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

function saveText(step: number, label: string, text: string) {
  const filename = `step-${String(step).padStart(2, "0")}-${label}.txt`;
  fs.writeFileSync(path.join(logDir, filename), text, "utf-8");
  log(`text saved: ${filename} (${text.length} chars)`);
}

// ---- Create agent ----

const maxSteps = 20;

const agent = createDesktopAgent({
  model,
  executeScreenControl: screenControl,
  executePowerShell: runPowerShellCommand,
  onLog: log,
  onScreenshot: saveScreenshot,
  onText: saveText
});

log(`[standalone] Provider: ${provider}`);
log(`[standalone] Model: ${modelName}`);
log(`[standalone] Prompt: ${prompt}`);
log(`[standalone] Log dir: ${logDir}`);
log("");

// ---- Subscribe for logging + maxSteps ----

let stepCount = 0;
let finalText = "";

const unsubscribe = agent.subscribe((event) => {
  if (event.type === "turn_start") {
    stepCount++;
    log(`[agent] step ${stepCount}...`);
    if (stepCount > maxSteps) {
      log(`[agent] max steps (${maxSteps}) reached, aborting`);
      agent.abort();
    }
  }
  if (event.type === "message_end" && event.message.role === "assistant") {
    const assistantMsg = event.message as AssistantMessage;
    const textParts = assistantMsg.content.filter((c) => c.type === "text");
    const text = textParts.map((c) => (c as TextContent).text).join("");
    if (text) {
      log(`[agent] LLM text: ${text}`);
      finalText = text;
    }
    const toolCalls = assistantMsg.content.filter((c) => c.type === "toolCall");
    log(
      `[agent] step ${stepCount} LLM done, tools: ${toolCalls.length}, text: ${text.length} chars`
    );
  }
  if (event.type === "tool_execution_start") {
    log(
      `[agent] tool: ${event.toolName}(${JSON.stringify(event.args).slice(0, 100)})`
    );
  }
});

// ---- Run agent ----

try {
  await agent.prompt(prompt);
  await agent.waitForIdle();

  // If no text output, request a summary
  if (!finalText) {
    log("[agent] requesting summary...");
    await agent.prompt("Summarize what you did and the result.");
    await agent.waitForIdle();
    // Extract final text from the last assistant message
    const msgs = agent.state.messages;
    for (let i = msgs.length - 1; i >= 0; i--) {
      const m = msgs[i];
      if (m.role === "assistant") {
        const text = (m as AssistantMessage).content
          .filter((c) => c.type === "text")
          .map((c) => (c as TextContent).text)
          .join("");
        if (text) {
          finalText = text;
          break;
        }
      }
    }
  }

  log("");
  log("[standalone] Done.");
  console.log(finalText || "[Agent completed without text output]");
} catch (err) {
  const msg = err instanceof Error ? err.message : String(err);
  console.error(`[standalone] Fatal error: ${msg}`);
  process.exit(1);
} finally {
  unsubscribe();
}
