# AI Chat Example

A chat application built with `@cloudflare/ai-chat` featuring a sandboxed bash environment with persistent storage (D1 + R2), read-write Git repository mounting, and an optional Electron shell or standalone CLI for controlling Windows desktops.

## What it demonstrates

**Server (`src/server/`):**

- `toUIMessageStreamResponse()` -- the simplest streaming pattern
- Sandboxed bash tool via `just-bash` (ls, grep, awk, sed, curl, jq, etc.)
- Persistent `/home/user` and `/etc` directories backed by D1 (via `D1FsAdapter`)
- Persistent `/data` directory backed by R2 object storage (via `R2FsAdapter`, suitable for large files)
- In-memory filesystem for everything outside persistent mounts (with pre-created `/mnt` for mount points)
- `MountableFs` to combine persistent and ephemeral filesystems
- Read-write Git repository mounting via `mount -t git <url> /mnt/<repo-name>` (auto commit & push, auto-persisted to `/etc/fstab`)
- Browser tool for interacting with JavaScript-rendered web pages
- Scheduled/recurring task execution
- Persistent memory system in `/home/user/.memory/`
- `pruneMessages()` for managing LLM context in long conversations
- `maxPersistedMessages` for storage management
- `BridgeManager` Durable Object for relaying messages between ChatAgent and remote desktop agents

**Client (`src/client/`):**

- `useAgentChat` for chat interaction
- Structured bash tool output rendering (command, stdout, stderr, exit code badge)
- Tool part rendering (executing, completed, approval requested)
- Interrupt: type and send during streaming to stop current response and start new one
- Abort: stop button to cancel current response without sending
- Kumo design system components
- `useBridge` — connects Electron clients as remote desktop agents via WebSocket
- `useBridgeViewer` — shows connected devices and activity logs in the sidebar
- Capability-based Electron detection (`!!window.workWithWindows`) instead of URL path checks

**Shared (`src/shared/`):**

- `agent-loop.ts` — unified agent loop with tool definitions (bash, screen_control, window_control), shared between Electron bridge and standalone CLI
- `action-aliases.ts` — normalized mouse/keyboard action handling
- `screen-control-types.ts` — type definitions for screen automation
- `file-utils.ts` — shared file utilities

**Electron (`electron/`):**

- `main.ts` — Electron main process (TypeScript, run via tsx). IPC handlers for Win32 screen/window control via PowerShell, filesystem operations delegated to `NodeFsAdapter`
- `preload.cjs` — `contextBridge` exposing `window.workWithWindows` to the renderer
- `node-fs-adapter.ts` — `IFileSystem` implementation backed by `node:fs/promises`, with unix-to-windows path translation (used by both Electron IPC and standalone)
- `detect-drives.ts` — detects available Windows drives (A:-Z:) and WSL distros at startup
- `standalone.ts` — headless Node.js CLI entry point, mounts real Windows drives via `MountableFs` + `NodeFsAdapter`
- `win-automation.ts` — PowerShell automation module (zero Electron dependencies)

## Architecture

```
Browser (any device)                    Cloudflare Worker
  App.tsx                               (ai.connect-screen.com)
    |                                     |
    +- useAgentChat ---- WebSocket ----> ChatAgent DO
    |                                     |  +- bash tool (just-bash)
    +- useBridgeViewer - WebSocket ----> BridgeManager DO
    |  (device list, activity logs)       |  (relays messages)
    |                                     |
Electron (Windows)                        |
  App.tsx                                 |
    +- useAgentChat ---- WebSocket ----> ChatAgent DO
    |                                     |  sends screen_control requests
    +- useBridge ------- WebSocket ----> BridgeManager DO
    |  (registers as device,              |  (routes requests to device)
    |   executes screen_control           |
    |   via window.workWithWindows)       |
    |                                     |
  preload.cjs                             |
    +- window.workWithWindows ------> main.ts (IPC -> NodeFsAdapter / PowerShell)

Standalone (Windows, no Electron)
  standalone.ts
    +- createAgentLoop (src/shared/agent-loop.ts)
    |    +- bash tool (just-bash + MountableFs + NodeFsAdapter)
    |    +- screen_control (win-automation.ts -> PowerShell)
    +- detectDrives() -> mounts /mnt/c, /mnt/d, /mnt/wsl, ...
```

When `window.workWithWindows` is detected, the client automatically registers as a remote desktop device via `useBridge`. Any client can view connected devices and agent activity via `useBridgeViewer` in the sidebar.

## Storage architecture

| Mount point | Backend | Best for |
|-------------|---------|----------|
| `/home/user` | D1 (SQL) | Config, scripts, small text files |
| `/etc` | D1 (SQL) | System config (fstab, git-credentials) |
| `/data` | R2 (object storage) | Large files, binary data |
| `/mnt/*` | Git / in-memory | Git repos, temporary files |
| `/mnt/c`, `/mnt/d`, ... | NodeFsAdapter (local disk) | Standalone/Electron: real Windows drives |

All storage is scoped per user — D1 uses a `user_id` column, R2 uses a `{userId}/` key prefix.

## Running

```bash
npm install
npm run dev
```

Requires `BUILTIN_LLM_*` secrets for the built-in model (`BUILTIN_LLM_PROVIDER`, `BUILTIN_LLM_BASE_URL`, `BUILTIN_LLM_API_KEY`, `BUILTIN_LLM_MODEL`). Users can override this by configuring `/etc/llm.json` in the Settings UI.

### Cloudflare resources

- **D1 database** — created automatically or via `wrangler d1 create ai-chat-db`
- **R2 bucket** — requires R2 enabled on your Cloudflare account, then `wrangler r2 bucket create ai-chat-files`

## Running with Electron (Windows Agent)

### One-click build, deploy & launch (from WSL)

`run-on-windows.sh` runs four steps in sequence:

1. **Build just-bash** — ai-chat depends on its `dist/` output
2. **Build workspace packages** — `agents`, `@cloudflare/ai-chat`, etc.
3. **Deploy ai-chat** — `vite build && wrangler deploy` to Cloudflare
4. **Launch Electron** — via PowerShell on the Windows side

```bash
bash examples/ai-chat/run-on-windows.sh
```

The Electron step copies `electron/` to a Windows temp directory (`%TEMP%\windows-agent-build`) with a minimal `package.json` (includes tsx for TypeScript support) to avoid UNC path and monorepo workspace issues.

### Launch Electron only (skip deploy)

If ai-chat is already deployed, just start Electron:

```bash
npm run electron:dev
```

### Local development (no deploy)

Point Electron at a local ai-chat dev server:

1. Start ai-chat locally: `npm run start` (runs on `http://localhost:5173`)
2. Launch Electron with local URL:

```bash
AGENT_URL=http://localhost:5173 npm run electron:dev
```

### Debugging

Electron renderer `console.log` output is captured to a file by `main.ts`:

```
%TEMP%\windows-agent-build\electron\renderer.log
```

Read from WSL:

```bash
cat "/mnt/c/Users/$USER/AppData/Local/Temp/windows-agent-build/electron/renderer.log"
```

## Standalone Windows Agent (no Electron, no deploy)

A lightweight Node.js CLI that runs the screenshot/automation agent directly on Windows, without Electron or Cloudflare Worker deployment. Useful for quick local testing.

The standalone agent mounts real Windows drives (`/mnt/c`, `/mnt/d`, etc.) via `NodeFsAdapter`, giving the bash environment full read/write access to the local filesystem.

```
standalone.ts -> MountableFs + NodeFsAdapter (local drives)
              -> agent-loop.ts (shared with Electron bridge)
              -> win-automation.ts -> PowerShell scripts
```

### Prerequisites

- Windows (or WSL with `powershell.exe` accessible)
- Node.js 18+
- LLM API access (OpenAI-compatible or Google)

### Quick start (from WSL)

```bash
# Set LLM config
export LLM_PROVIDER=google          # or "openai-compatible"
export LLM_API_KEY=your-key
export LLM_MODEL=gemini-2.5-flash   # or your model name
# export LLM_BASE_URL=https://...   # required for openai-compatible

# Build just-bash and launch
bash examples/ai-chat/run-standalone-on-windows.sh "Take a screenshot and describe what you see"
```

### Quick start (from Windows PowerShell)

If `just-bash` is published to npm, you can run directly on the Windows side:

```powershell
$env:LLM_PROVIDER = "google"
$env:LLM_API_KEY = "your-key"
$env:LLM_MODEL = "gemini-2.5-flash"

powershell -ExecutionPolicy Bypass -File run-standalone.ps1 "List all visible windows"
```

### Example commands

```bash
# Screenshot test
bash run-standalone-on-windows.sh "Take a screenshot and describe what you see"

# Window list
bash run-standalone-on-windows.sh "List all visible windows"

# Multi-step
bash run-standalone-on-windows.sh "Open notepad and type hello"

# Local filesystem access
bash run-standalone-on-windows.sh "Run ls /mnt/c to see what's on the C drive"
```

### Logs and screenshots

Every run saves structured logs and screenshots to a fixed directory:

- **Windows**: `%TEMP%\windows-agent-standalone\logs\`
- **WSL**: `/mnt/c/Users/$USER/AppData/Local/Temp/windows-agent-standalone/logs/`

Contents:
- `agent.log` — timestamped log of every step, tool call, and result (including coordinate translations)
- `step-NN-<action>.png` — screenshots the agent captured during normal operation
- `step-NN-annotate.png` — annotated screenshots with red crosshair showing where the model intends to click

### Debugging with logs and screenshots

When the agent misbehaves, follow this workflow to diagnose the issue:

**1. Read `agent.log` and trace the coordinate chain**

Each click/move/annotate log line shows the full coordinate translation:

```
[agent] screen: click norm(250,230)->pixel(282,151)->desktop(1200,293) -> success
[agent] screen: annotate norm(250,230)->pixel(282,151) -> 1126x655
```

- `norm(250,230)` — what the model output (0-1000 normalized range)
- `pixel(282,151)` — converted using `norm / 1000 * screenshot_size`
- `desktop(1200,293)` — pixel + window offset (from last `window_screenshot`)

If the pixel or desktop coords look wrong, check whether the screenshot dimensions changed between steps.

**2. Check annotate screenshots to verify targeting**

`step-NN-annotate.png` shows a red crosshair at the exact pixel position where the agent will click. The label shows the normalized coordinates (e.g., `(norm: 250, 230)`). Compare the crosshair position against the UI element the agent is trying to target. If the crosshair is off, the coordinate conversion may be wrong; if the crosshair is on-target but the action fails, the issue is elsewhere.

**3. Compare before/after screenshots to verify click effects**

Walk through the screenshots in order:
- `step-02-window_screenshot.png` — state before click
- `step-03-annotate.png` — where the agent plans to click (crosshair)
- `step-05-window_screenshot.png` — state after click

If the UI didn't change as expected despite correct crosshair placement, common causes include:
- **Double-click toggling**: The model clicks the same target twice (first click selects, second click deselects). This happens when the model doesn't trust its first click and retries.
- **Focus stealing**: `focus_window` followed immediately by `click` — the focus event may consume the click. Adding a delay or taking a screenshot between focus and click helps.
- **Stale coordinates**: The model reuses coordinates from an earlier screenshot after the window has moved or resized.

**4. Quick commands**

```bash
# Check the log
cat /mnt/c/Users/$USER/AppData/Local/Temp/windows-agent-standalone/logs/agent.log

# Open screenshots (from Windows)
explorer.exe "C:\Users\%USERNAME%\AppData\Local\Temp\windows-agent-standalone\logs"
```

### How it works

- `electron/win-automation.ts` — PowerShell automation module (zero Electron dependencies)
- `src/shared/agent-loop.ts` — platform-agnostic agent loop with dependency injection (shared with Electron bridge)
- `electron/node-fs-adapter.ts` — `IFileSystem` backed by `node:fs/promises` (used by both standalone and Electron main process)
- `electron/detect-drives.ts` — detects Windows drives and WSL distros, returns mount points
- `electron/standalone.ts` — CLI entry point that wires everything together
- `run-standalone.ps1` — Windows-side launcher (copies files to temp dir, npm install, runs tsx)
- `run-standalone-on-windows.sh` — WSL-side launcher (builds just-bash, packs tarball, calls PS1)

## Sentry error monitoring

The project integrates `@sentry/cloudflare` for error tracking and performance tracing. Set the `SENTRY_DSN` secret via `wrangler secret put SENTRY_DSN` to enable it.

### Setup

Create an auth token at **Sentry Settings > Auth Tokens**, then save it locally:

```bash
# examples/ai-chat/.env.sentry (git-ignored)
SENTRY_AUTH_TOKEN=sntryu_xxx
SENTRY_ORG=txom
SENTRY_PROJECT=cloudflare-worker
SENTRY_BASE_URL=https://us.sentry.io
```

### Diagnostic workflow

When you see a 500 error or unexpected behavior:

**Step 1: List recent unresolved issues**

```bash
source examples/ai-chat/.env.sentry
curl -s "https://us.sentry.io/api/0/projects/txom/cloudflare-worker/issues/?query=is:unresolved&sort=date&limit=10" \
  -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" | jq '.[] | {id, title, lastSeen, count}'
```

**Step 2: Get the latest event for a specific issue**

Copy the `id` from step 1 and fetch the full event:

```bash
curl -s "https://us.sentry.io/api/0/issues/{issue_id}/events/latest/" \
  -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" | jq '{
    message: .message,
    exception: [.entries[] | select(.type == "exception") | .data.values[] | {type, value}],
    request: .request,
    tags: [.tags[] | select(.key == "url" or .key == "transaction" or .key == "user_id" or .key == "session_uuid")]
  }'
```

**Step 3: Check breadcrumbs for context**

```bash
curl -s "https://us.sentry.io/api/0/issues/{issue_id}/events/latest/" \
  -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" | jq '[.entries[] | select(.type == "breadcrumbs") | .data.values[-10:][]]'
```

### Common patterns

| Symptom | Likely cause |
|---------|-------------|
| `SqlError: Durable Object reset because its code was updated` | Transient — DOs restart after deploy, safe to ignore |
| 500 on API routes with no Sentry issue | Error not captured — add `try/catch` + `Sentry.captureException(e)` to the handler |
| `tryN.baseDelayMs.baseDelayMs` | DO retry loop hitting the reset window after deploy |

### Limitations

- **Worker-level Hono route errors** may not appear in Sentry unless explicitly captured with `Sentry.captureException()` inside a `try/catch` block. The Sentry middleware wraps the top-level fetch but uncaught exceptions in sub-routers can sometimes be swallowed by Hono's error handling.
- **DO errors** are instrumented via `instrumentDurableObjectWithSentry` and appear automatically.
- **`wrangler tail`** (`npx wrangler tail --format json`) shows real-time `console.log`/`console.error` output. Useful when Sentry doesn't capture the error.

### Key fields in the event response

| Field | What it tells you |
|-------|-------------------|
| `entries[type=exception].data.values` | Exception chain with stack traces |
| `entries[type=breadcrumbs].data.values` | Chronological log of console, fetch, and schedule events leading up to the error |
| `contexts.trace` | Distributed trace ID for correlating spans |
| `tags` | Environment, handler status, runtime info |

## Try it

- "List files in /home/user" -- bash command execution
- "Create a file called notes.txt with some content" -- files in /home/user persist across sessions
- "Save a large dataset to /data/export.csv" -- R2-backed storage for big files
- "Fetch https://example.com and save it" -- curl support with network access
- "Show me the disk usage of /home/user" -- persistent storage inspection
- "Mount https://github.com/user/repo and list its files" -- read-write Git repo browsing
- Run `mount` -- see all active filesystem mounts (d1fs, r2fs, git)
- Have a long conversation -- old tool calls are pruned from LLM context automatically
