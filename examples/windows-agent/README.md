# Windows Agent

Electron desktop agent that combines a Cloudflare Worker LLM proxy with a browser-side AI chat UI. The React app runs inside an Electron `BrowserWindow`, with a preload bridge (`window.workWithWindows`) ready for future desktop capabilities (screenshots, shell commands, etc.).

## How to run

1. Install dependencies:

```bash
npm install
```

2. Copy `.env.example` to `.env` and fill in your LLM credentials.

3. Start the web server (Terminal 1):

```bash
npm run dev:web
```

4. Start Electron (Terminal 2):

```bash
npm run dev:electron
```

The Electron window loads the Vite dev server. You should see a green "Electron (win32) - bridge: pong" badge in the header, confirming the preload bridge is working.

## Required env vars

| Variable | Description |
|---|---|
| `LLM_BASE_URL` | Base URL of the LLM API (e.g. `https://api.openai.com`) |
| `LLM_MODEL` | Model ID (e.g. `gpt-4o`) |
| `LLM_API_KEY` | API key for the LLM provider |

## Architecture

```
Electron (main.js + preload.js)
  └─ BrowserWindow loads http://localhost:5173
       └─ React chat UI + just-bash virtual shell
            └─ LLM calls proxied via Cloudflare Worker (/api/v1/*)
```

The `window.workWithWindows` preload bridge is the extensibility point for adding real desktop tools (desktopCapturer screenshots, child_process commands, etc.).
