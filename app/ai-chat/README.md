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
```

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

D1 does **not** auto-apply `schema.sql` on deploy. When you add or alter tables in `schema.sql`, you must manually apply the changes to the remote database:

```bash
npx wrangler d1 execute ai-chat-db --remote --file schema.sql
```

All statements use `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`, so re-running is safe. Verify with:

```bash
npx wrangler d1 execute ai-chat-db --remote --command "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
```

> **Past incident (BUG-MLUSHN3O-0311):** `/api/usage` returned 500 because `usage_archive` and `device_messages` tables existed in `schema.sql` but were never created on the remote D1 database. Fixed by running the command above.

## Sentry error monitoring

The project integrates `@sentry/cloudflare` for error tracking and performance tracing. Set the `SENTRY_DSN` secret via `wrangler secret put SENTRY_DSN` to enable it.

### Setup

Create an auth token at **Sentry Settings > Auth Tokens**, then save it locally:

```bash
# app/ai-chat/.env.sentry (git-ignored)
SENTRY_AUTH_TOKEN=sntryu_xxx
SENTRY_ORG=txom
SENTRY_PROJECT=cloudflare-worker
SENTRY_BASE_URL=https://us.sentry.io
```

### Diagnostic workflow

When you see a 500 error or unexpected behavior:

**Step 1: List recent unresolved issues**

```bash
source app/ai-chat/.env.sentry
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
| `D1_ERROR: no such table: <name>: SQLITE_ERROR` | D1 schema out of sync — new table in `schema.sql` but never applied to remote. Fix: `npx wrangler d1 execute ai-chat-db --remote --file schema.sql` |

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

## Self-testing API endpoints

The app has a device auth flow (similar to OAuth2 Device Authorization) designed for the companion React Native agent (`app/rn/`) to link with a user's account. The self-test script reuses this flow to obtain a Bearer token and call authenticated API endpoints directly.

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

## Try it

- "List files in /home/user" -- bash command execution
- "Create a file called notes.txt with some content" -- files in /home/user persist across sessions
- "Save a large dataset to /data/export.csv" -- R2-backed storage for big files
- "Fetch https://example.com and save it" -- curl support with network access
- "Show me the disk usage of /home/user" -- persistent storage inspection
- "Mount https://github.com/user/repo and list its files" -- read-write Git repo browsing
- Run `mount` -- see all active filesystem mounts (d1fs, r2fs, git)
- Have a long conversation -- old tool calls are pruned from LLM context automatically
