# Cloud Browser Agent

A browser automation agent that uses Cloudflare Browser Rendering to browse the web. Submit a task, and the agent drives a headless Chrome via the `pi` framework, streaming results (including screenshots) back via SSE.

## Setup

1. Install dependencies from the repo root:
   ```
   npm install
   ```

2. Copy `.env.example` to `.env` and fill in your LLM credentials:
   ```
   cp .env.example .env
   ```

3. Start the dev server:
   ```
   npm run start
   ```

## Deploy

1. Build and deploy:
   ```
   npm run deploy
   ```

2. Set secrets (only needed once, or when credentials change):
   ```
   echo "https://your-api-base-url" | npx wrangler secret put LLM_BASE_URL
   echo "your-api-key" | npx wrangler secret put LLM_API_KEY
   echo "your-model-name" | npx wrangler secret put LLM_MODEL
   ```

Note: `.env` is for local dev only. For production, use `wrangler secret put` to set env vars as encrypted secrets.

## Required env vars

- `LLM_BASE_URL` — OpenAI-compatible API base URL
- `LLM_API_KEY` — API key
- `LLM_MODEL` — Model name (e.g. `gpt-4o`, `doubao-seed-2.0-code`)

## How it works

- **No Durable Objects** — the entire agent loop runs within a single POST request
- The pi `Agent` drives an LLM that calls a `browser` tool (goto, click, type, screenshot, scroll, extract, set_cookies, close)
- Agent events are streamed to the client as SSE, including inline base64 screenshots
- Browser session is cleaned up when the agent finishes
- Max 30 turns guard prevents runaway agent loops

## Architecture

```
Client (React)              Worker (Cloudflare)              Browser Rendering
     |                           |                               |
     |-- POST /api/agent ------->|                               |
     |    { task: "..." }        |-- creates pi Agent            |
     |                           |-- agent.prompt(task)          |
     |<-- SSE stream ------------|                               |
     |   event: message_update   |-- LLM call ------------------>|
     |   event: tool_exec_start  |<-- "call browser goto"        |
     |                           |-- puppeteer page.goto() ----->|
     |   event: tool_exec_end    |<-- screenshot + text ---------|
     |   ...repeats...           |                               |
     |   event: agent_end        |-- closeBrowser() ------------>|
```

## SSE event types

| Event | Payload | Description |
|-------|---------|-------------|
| `message_update` | `{ role, text }` | Streaming assistant text (partial) |
| `message_end` | `{ role, text }` | Final assistant message for a turn |
| `tool_execution_start` | `{ toolCallId, toolName, args }` | Browser tool invoked |
| `tool_execution_end` | `{ toolCallId, toolName, result, isError }` | Tool result with screenshot in `result.details.screenshot` (base64) |
| `agent_end` | `{}` | Agent finished all turns |
| `error` | `{ message }` | Error occurred |

## Deployment notes

- `wrangler.jsonc` must have `"main": "src/server.ts"` — the Cloudflare Vite plugin needs this to find the worker entry point
- Browser Rendering binding is configured as `"browser": { "binding": "MYBROWSER" }` in `wrangler.jsonc`
- The `@ai-sdk/openai-compatible` provider works with any OpenAI-compatible API (OpenAI, Azure, Doubao, etc.)
- Worker size is ~1MB due to `@cloudflare/puppeteer` — this is normal
