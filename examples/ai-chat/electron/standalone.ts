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
 *   DEBUG_DIR     - Directory for focus-transition logs and screenshot PNGs (optional)
 */
import fs from "node:fs";
import path from "node:path";
import { screenControl } from "./win-automation.ts";
import { createAgent } from "./agent-core.ts";
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

// ---- Debug mode ----

const debugDir = process.env.DEBUG_DIR;
if (debugDir) {
  fs.mkdirSync(debugDir, { recursive: true });
}

// ---- Run agent ----

const agent = createAgent({ screenControlFn: screenControl, model, debugDir });

function log(msg: string) {
  process.stderr.write(msg + "\n");
}

log(`[standalone] Provider: ${provider}`);
log(`[standalone] Model: ${modelName}`);
log(`[standalone] Prompt: ${prompt}`);
if (debugDir) log(`[standalone] Debug dir: ${debugDir}`);
log("");

try {
  const response = await agent.runAgent(prompt, log);
  log("");
  log("[standalone] Done.");
  console.log(response);

  if (debugDir) {
    const focusLog = path.join(debugDir, "focus-log.txt");
    if (fs.existsSync(focusLog)) {
      log("");
      log("[standalone] === Focus transition log ===");
      log(fs.readFileSync(focusLog, "utf-8"));
    }
    const pngs = fs
      .readdirSync(debugDir)
      .filter((f: string) => f.endsWith(".png"));
    if (pngs.length > 0) {
      log(`[standalone] Saved ${pngs.length} screenshot(s) to ${debugDir}/`);
      for (const png of pngs) log(`  - ${png}`);
    }
  }
} catch (err) {
  const msg = err instanceof Error ? err.message : String(err);
  console.error(`[standalone] Fatal error: ${msg}`);
  process.exit(1);
}
