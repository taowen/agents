# AI Chat Example

A chat application built with `@cloudflare/ai-chat` featuring a sandboxed bash environment with persistent storage (D1 + R2) and read-write Git repository mounting.

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

**Client (`src/client/`):**

- `useAgentChat` for chat interaction
- Structured bash tool output rendering (command, stdout, stderr, exit code badge)
- Tool part rendering (executing, completed, approval requested)
- Interrupt: type and send during streaming to stop current response and start new one
- Abort: stop button to cancel current response without sending
- Kumo design system components

**Shared (`src/shared/`):**

- `file-utils.ts` — shared file utilities

## Architecture

```
Browser                                 Cloudflare Worker
  App.tsx                               (ai.connect-screen.com)
    |                                     |
    +- useAgentChat ---- WebSocket ----> ChatAgent DO
                                          |  +- bash tool (just-bash)
                                          |  +- device_agent tool
                                          |
Android Agent                              |
  DeviceConnection.java                   |
    +- OkHttp WS -- /device-connect ---->-+
    |  (ping/pong keepalive)
    +- receives task dispatches
    +- proxies LLM requests (blocking)
    +- sends task results
```

### Device connection flow

1. The Android agent authenticates via the Device Authorization flow (`POST /auth/device/start` → 6-char code → user approves on web)
2. Once authorized, the agent opens a WebSocket directly to the user's ChatAgent Durable Object at `/agents/chat-agent/{session}/device-connect`
3. The ChatAgent tags this WebSocket as `["device"]` and starts a heartbeat alarm
4. The web user can dispatch tasks to the device via the `device_agent` tool in chat — the ChatAgent forwards these over the device WebSocket
5. The Android agent executes tasks locally (via Hermes + accessibility APIs) and sends results back

## Storage architecture

| Mount point | Backend | Best for |
|-------------|---------|----------|
| `/home/user` | D1 (SQL) | Config, scripts, small text files |
| `/etc` | D1 (SQL) | System config (fstab, git-credentials) |
| `/data` | R2 (object storage) | Large files, binary data |
| `/mnt/*` | Git / in-memory | Git repos, temporary files |

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

## D1 schema management

`schema.sql` is automatically applied to the remote D1 database as part of `npm run deploy` (runs `wrangler d1 execute` before `wrangler deploy`). All statements use `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`, so re-running is idempotent.

To drop a deprecated table, add `DROP TABLE IF EXISTS <name>;` before any CREATE statements.

To verify the current remote schema:

```bash
npx wrangler d1 execute ai-chat-db --remote --command "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
```

## Sentry error monitoring

Server: `@sentry/cloudflare` (Worker + DO). Client: `@sentry/react`. Enable by setting `SENTRY_DSN` via `wrangler secret put SENTRY_DSN`.

### Setup

Save an auth token locally for API queries:

```bash
# app/ai-chat/.env.sentry (git-ignored)
SENTRY_AUTH_TOKEN=sntryu_xxx
```

All `curl` examples below assume `SENTRY_AUTH_TOKEN` is set. Shell-based usage:

```bash
export SENTRY_AUTH_TOKEN=$(grep SENTRY_AUTH_TOKEN app/ai-chat/.env.sentry | cut -d= -f2)
```

The Sentry API base for all commands is:

```
https://us.sentry.io/api/0/projects/txom/cloudflare-worker
```

### Look up a bug report

In-app "Report Bug" creates two Sentry issues per report — one server-side (`@sentry/cloudflare`, has full context) and one client-side (`@sentry/react`, for correlation only). Both are tagged with `report_id`.

**Important**: Sentry issue search does **not** support free text. You must use the `report_id:` tag prefix:

```bash
# Find issues for a bug report
curl -s "https://us.sentry.io/api/0/projects/txom/cloudflare-worker/issues/?query=report_id:BUG-MLYD2QVH-928A" \
  -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" | jq '.[] | {id, title, lastSeen}'
```

Get the full event (use `/events/latest/`, not `/events/` which only returns summaries):

```bash
curl -s "https://us.sentry.io/api/0/issues/{issue_id}/events/latest/" \
  -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" | jq '{
    bug_report: .contexts.bug_report,
    recent_messages: .contexts.recent_messages.messages,
    tags: [.tags[] | select(.key == "report_id" or .key == "user_id" or .key == "session_uuid")]
  }'
```

Server-side bug report event structure:

| Location | Content |
|----------|---------|
| `contexts.bug_report` | `{description, reportId, sessionUuid, userId}` |
| `contexts.recent_messages.messages` | Last 10 D1 chat messages (path + text preview) |
| `entries[type=request]` | HTTP request (method, url, headers) |
| `tags` | `report_id`, `user_id`, `session_uuid`, `url`, `browser`, `os` |

`recent_messages` comes from D1 (`files` table, `/.chat/` prefix). The authoritative message store is the DO's internal SQLite (`cf_ai_chat_agent_messages`), read via `/get-messages`.

### Retrieve R2 debug payload

Bug reports include a full debug payload uploaded to R2. Use `wrangler r2 object get` with the **`--remote` flag** (without it you'll hit the local dev bucket, which is empty):

```bash
# Download the debug payload for a bug report
npx wrangler r2 object get ai-chat-files/bug-reports/BUG-XXXXXXXX-XXXX.json \
  --remote > /tmp/bug-payload.json

# Pretty-print the full payload
cat /tmp/bug-payload.json | jq .
```

R2 payload structure:

| Field | Content |
|-------|---------|
| `reportId` | Bug report ID (matches Sentry `report_id` tag) |
| `debugContext.debugEntries[]` | Ring buffer of recent LLM interactions and cross-DO calls |
| `debugContext.messages[]` | Full DO message history at time of report |
| `debugContext.bufferSize` | Number of debug entries captured |
| `debugContext.messageCount` | Number of messages in DO |
| `debugContext.doId` | Durable Object ID |
| `capturedAt` | ISO timestamp of report submission |

Debug entry types:

**`type: "llm"`** — each LLM `streamText`/`generateText` call:

| Field | Content |
|-------|---------|
| `request.systemPrompt` | System prompt sent to the model |
| `request.dynamicContext` | Dynamic context (memory, MCP servers) |
| `request.messages` | Conversation messages sent to the model |
| `request.toolNames` | Available tool names |
| `request.modelId` | Model identifier |
| `response.text` | Model response text |
| `response.steps[]` | Individual steps with tool calls/results and per-step usage |
| `response.finishReason` | How the model stopped (`stop`, `tool-calls`, etc.) |
| `response.usage` | Token usage (`inputTokens`, `outputTokens`) |
| `error` | Error string if the call failed or was aborted (`"aborted"` for cancelled streams) |

**`type: "do_call"`** — cross-DO calls (e.g. `send_to_device` → `dispatch-task`):

| Field | Content |
|-------|---------|
| `direction` | `"outbound"` or `"inbound"` |
| `endpoint` | Target endpoint path |
| `request` / `response` | Request and response payloads |
| `durationMs` | Round-trip duration |
| `error` | Error string if the call failed |

### End-to-end diagnostics flow

1. **Find the bug report in Sentry** — search by `report_id:BUG-XXXXXXXX-XXXX` to get the Sentry issue with user description and session metadata.

2. **Download R2 debug payload** — this contains the actual LLM request/response data that Sentry events don't include:
   ```bash
   npx wrangler r2 object get ai-chat-files/bug-reports/BUG-XXXXXXXX-XXXX.json \
     --remote > /tmp/bug-payload.json
   ```

3. **Inspect LLM interactions** — use `jq` to drill into specific entries:
   ```bash
   # List all debug entries with type and timestamp
   cat /tmp/bug-payload.json | jq '.debugContext.debugEntries[] | {type, timestamp, error}'

   # Show the last LLM request's system prompt and messages
   cat /tmp/bug-payload.json | jq '[.debugContext.debugEntries[] | select(.type == "llm")] | last | .request'

   # Show the last LLM response text and finish reason
   cat /tmp/bug-payload.json | jq '[.debugContext.debugEntries[] | select(.type == "llm")] | last | .response | {text, finishReason, usage}'

   # Show all tool calls across all steps
   cat /tmp/bug-payload.json | jq '[.debugContext.debugEntries[] | select(.type == "llm")] | last | .response.steps[] | .toolCalls[]'
   ```

4. **Cross-reference with Sentry exceptions** — use the `session_uuid` tag from the bug report to find related error events:
   ```bash
   curl -s "https://us.sentry.io/api/0/projects/txom/cloudflare-worker/issues/?query=session_uuid:SESSION_UUID_HERE" \
     -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" | jq '.[] | {id, title, lastSeen}'
   ```

### Data distribution

| Data | Location |
|------|----------|
| Bug metadata, user description | Sentry event (`contexts.bug_report`) |
| Full LLM request/response | R2 payload (`debugContext.debugEntries[]`) |
| DO full message history | R2 payload (`debugContext.messages[]`) |
| Exception stack traces, breadcrumbs | Sentry event (`entries[]`) |

### Investigate errors

List recent unresolved issues:

```bash
curl -s "https://us.sentry.io/api/0/projects/txom/cloudflare-worker/issues/?query=is:unresolved&sort=date&limit=10" \
  -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" | jq '.[] | {id, title, level, lastSeen, count}'
```

Get exception + stack trace for an issue:

```bash
curl -s "https://us.sentry.io/api/0/issues/{issue_id}/events/latest/" \
  -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" | jq '[.entries[] | select(.type == "exception") | .data.values[] | {type, value}]'
```

Get breadcrumbs (console logs, fetch calls leading up to the error):

```bash
curl -s "https://us.sentry.io/api/0/issues/{issue_id}/events/latest/" \
  -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" | jq '[.entries[] | select(.type == "breadcrumbs") | .data.values[-10:][]]'
```

Error events have `entries` of type `exception`, `breadcrumbs`, `debugmeta`. Useful tags on DO errors: `do_id`, `user_id`, `session_uuid`.

### Instrumentation coverage

| Source | How | Notes |
|--------|-----|-------|
| DO errors | `instrumentDurableObjectWithSentry` | Automatic — exceptions, breadcrumbs, traces |
| Worker Hono routes | `Sentry.withSentry` top-level wrapper | Uncaught exceptions in sub-routers may be swallowed by Hono — add explicit `Sentry.captureException(e)` in `try/catch` |
| Client | `@sentry/react` init + `ErrorBoundary` | Browser errors + manual `captureMessage` for bug reports |
| Real-time logs | `npx wrangler tail --format json` | Useful when Sentry doesn't capture the error |

### Common issues

| Symptom | Cause |
|---------|-------|
| `Durable Object reset because its code was updated` | Transient after deploy, safe to ignore |
| 500 with no Sentry issue | Error swallowed by Hono — add `try/catch` + `Sentry.captureException(e)` |
| `D1_ERROR: no such table` | Schema not applied — run `npx wrangler d1 execute ai-chat-db --remote --file schema.sql` |

## Self-testing API endpoints

The app has a device auth flow (similar to OAuth2 Device Authorization) designed for the companion Android device agent (`app/android-device/`) to link with a user's account. The self-test script reuses this flow to obtain a Bearer token and call authenticated API endpoints directly.

### Run the self-test script

```bash
npx tsx app/ai-chat/scripts/self-test.ts
```

On first run, the script will:
1. `POST /auth/device/start` to get a 6-character code
2. Print the code prominently — ask the user to approve it at `https://ai.connect-screen.com/device`
3. Poll `/auth/device/check` until approved, then cache the Bearer token to `scripts/.self-test-token`
4. Call `GET /api/usage` with the token and print the response

On subsequent runs, the cached token is reused automatically (no user interaction needed). Delete `scripts/.self-test-token` to force re-authentication.

Pass a custom base URL for local dev:

```bash
npx tsx app/ai-chat/scripts/self-test.ts http://localhost:5173
```

### Claude Code workflow

When debugging authenticated API endpoints, Claude Code should:

1. Run the script **in background**: `npx tsx app/ai-chat/scripts/self-test.ts` (with `run_in_background: true`)
2. After ~3 seconds, read the output file to check for the device code or cached token result
3. If a `DEVICE CODE: XXXXXX` appears, tell the user: "Please approve code **XXXXXX** at https://ai.connect-screen.com/device"
4. Wait for the script to complete — it will call the API and print the response
5. On subsequent runs the cached token is reused, so no user interaction is needed

### Combine with wrangler tail for full observability

In a separate terminal, stream live Worker logs:

```bash
npx wrangler tail --format json
```

Then run the self-test script. You'll see the `[usage]` and auth logs fire in real time alongside the script output.

### Extending

The token returned by the script works with any authenticated endpoint. To test a different route, add a fetch call with `Authorization: Bearer ${token}` after step 3 in `scripts/self-test.ts`.

## Build & Deploy

### ai-chat server

```bash
cd app/ai-chat
npm install
npm run deploy        # builds + deploys to Cloudflare
npm run dev           # local development
```

### Android Device Agent

```bash
cd app/android-device/android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n ai.connct_screen.rn/.MainActivity
```

Check device connection logs:

```bash
adb logcat -d | grep DeviceConn
```

## Try it

- "List files in /home/user" -- bash command execution
- "Create a file called notes.txt with some content" -- files in /home/user persist across sessions
- "Save a large dataset to /data/export.csv" -- R2-backed storage for big files
- "Fetch https://example.com and save it" -- curl support with network access
- "Show me the disk usage of /home/user" -- persistent storage inspection
- "Mount https://github.com/user/repo and list its files" -- read-write Git repo browsing
- Run `mount` -- see all active filesystem mounts (d1fs, r2fs, git)
- Have a long conversation -- old tool calls are pruned from LLM context automatically
