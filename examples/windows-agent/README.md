# Windows Agent

Electron desktop shell that loads the [AI Chat](../ai-chat/) web app at `https://ai.connect-screen.com/agent`. Authentication (Google OAuth) and LLM configuration are provided by the AI Chat service — no local API keys or Worker deployment needed.

The `window.workWithWindows` preload bridge is the extensibility point for adding real desktop tools (desktopCapturer screenshots, child_process commands, etc.).

## How to run

1. Install dependencies:

```bash
npm install
```

2. Start Electron:

```bash
npm run dev
```

The Electron window loads `https://ai.connect-screen.com/agent`. You'll be prompted to log in with Google OAuth on first launch.

For local development against the ai-chat dev server:

```bash
AGENT_URL=http://localhost:5173/agent npm run dev
```

## Architecture

```
Electron (main.js + preload.cjs)
  └─ BrowserWindow loads https://ai.connect-screen.com/agent
       └─ React chat UI + just-bash virtual shell (runs in browser)
            └─ LLM calls go directly to the configured provider
               (config fetched from /api/llm/config, auth via session cookie)
```

The backend is provided by the `examples/ai-chat` Worker deployed at `ai.connect-screen.com`. It handles:

- **Authentication** — Google OAuth + email-based login
- **LLM configuration** — per-user provider/model/key settings via `/api/llm/config`
- **No proxy needed** — the browser calls the LLM provider directly with the returned API key
