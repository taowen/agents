# AI Chat Example

A chat application built with `@cloudflare/ai-chat` featuring a sandboxed bash environment with persistent storage and read-only Git repository mounting.

## What it demonstrates

**Server (`src/server.ts`):**

- `toUIMessageStreamResponse()` -- the simplest streaming pattern
- Sandboxed bash tool via `just-bash` (ls, grep, awk, sed, curl, jq, etc.)
- Persistent `/home/user` directory backed by Durable Object SQLite (via `agentfs-sdk`)
- In-memory filesystem for everything outside `/home/user` (with pre-created `/mnt` for mount points)
- `MountableFs` to combine persistent and ephemeral filesystems
- Read-only Git repository mounting via `mount -t git <url> /mnt/<repo-name>`
- `pruneMessages()` for managing LLM context in long conversations
- `maxPersistedMessages` for storage management

**Client (`src/client.tsx`):**

- `useAgentChat` for chat interaction
- Structured bash tool output rendering (command, stdout, stderr, exit code badge)
- Tool part rendering (executing, completed, approval requested)
- Kumo design system components

## Running

```bash
npm install
npm run dev
```

Requires a `GOOGLE_AI_API_KEY` environment variable for Gemini 3 Flash.

## Try it

- "List files in /home/user" -- bash command execution
- "Create a file called notes.txt with some content" -- files in /home/user persist across sessions
- "Fetch https://example.com and save it" -- curl support with network access
- "Show me the disk usage of /home/user" -- persistent storage inspection
- "Mount https://github.com/user/repo and list its files" -- read-only Git repo browsing
- Have a long conversation -- old tool calls are pruned from LLM context automatically
