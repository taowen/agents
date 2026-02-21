# just-bash

A simulated bash environment with an in-memory virtual filesystem, written in TypeScript.

Designed for AI agents that need a secure bash environment. Browser-compatible.

Supports optional network access via `curl` with secure-by-default URL filtering.

**Note**: This is beta software. Use at your own risk and please provide feedback.

## Table of Contents

- [Security model](#security-model)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic API](#basic-api)
  - [Configuration](#configuration)
  - [Custom Commands](#custom-commands)
  - [Filesystem Options](#filesystem-options)
  - [AI SDK Tool](#ai-sdk-tool)
- [Supported Commands](#supported-commands)
- [Shell Features](#shell-features)
- [Default Layout](#default-layout)
- [Network Access](#network-access)
- [Execution Protection](#execution-protection)
- [AST Transform Plugins](#ast-transform-plugins)
- [Development](#development)

## Security model

- The shell only has access to the provided file system.
- Execution is protected against infinite loops or recursion. However, Bash is not fully robust against DOS from input. If you need to be robust against this, use process isolation at the OS level.
- Binaries or even WASM are inherently unsupported (Use [Vercel Sandbox](https://vercel.com/docs/vercel-sandbox) or a similar product if a full VM is needed).
- There is no network access by default.
- Network access can be enabled, but requests are checked against URL prefix allow-lists and HTTP-method allow-lists. See [network access](#network-access) for details

## Installation

```bash
npm install just-bash
```

## Usage

### Basic API

```typescript
import { Bash } from "just-bash";

const env = new Bash();
await env.exec('echo "Hello" > greeting.txt');
const result = await env.exec("cat greeting.txt");
console.log(result.stdout); // "Hello\n"
console.log(result.exitCode); // 0
console.log(result.env); // Final environment after execution
```

Each `exec()` is isolated—env vars, functions, and cwd don't persist across calls (filesystem does).

### Configuration

```typescript
const env = new Bash({
  files: { "/data/file.txt": "content" }, // Initial files
  env: { MY_VAR: "value" }, // Initial environment
  cwd: "/app", // Starting directory (default: /home/user)
  executionLimits: { maxCallDepth: 50 }, // See "Execution Protection"
});

// Per-exec overrides
await env.exec("echo $TEMP", { env: { TEMP: "value" }, cwd: "/tmp" });
```

### Custom Commands

Extend just-bash with your own TypeScript commands using `defineCommand`:

```typescript
import { Bash, defineCommand } from "just-bash";

const hello = defineCommand("hello", async (args, ctx) => {
  const name = args[0] || "world";
  return { stdout: `Hello, ${name}!\n`, stderr: "", exitCode: 0 };
});

const upper = defineCommand("upper", async (args, ctx) => {
  return { stdout: ctx.stdin.toUpperCase(), stderr: "", exitCode: 0 };
});

const bash = new Bash({ customCommands: [hello, upper] });

await bash.exec("hello Alice"); // "Hello, Alice!\n"
await bash.exec("echo 'test' | upper"); // "TEST\n"
```

Custom commands receive the full `CommandContext` with access to `fs`, `cwd`, `env`, `stdin`, and `exec` for running subcommands.

### Filesystem Options

Two filesystem implementations are available:

**InMemoryFs** (default) - Pure in-memory filesystem, no disk access:

```typescript
import { Bash } from "just-bash";
const env = new Bash(); // Uses InMemoryFs by default
```

**MountableFs** - Mount multiple filesystems at different paths. Combines filesystems into a unified namespace:

```typescript
import { Bash, MountableFs, InMemoryFs } from "just-bash";

const base = new InMemoryFs();
const workspace = new InMemoryFs();

const fs = new MountableFs({ base });
fs.mount("/workspace", workspace);

const bash = new Bash({ fs, cwd: "/workspace" });

await bash.exec("ls /"); // sees both base and mounted filesystems
await bash.exec('echo "notes" > notes.txt'); // writes to workspace mount
```

You can also configure mounts in the constructor:

```typescript
import { MountableFs, InMemoryFs } from "just-bash";

const fs = new MountableFs({
  base: new InMemoryFs(),
  mounts: [
    { mountPoint: "/data", filesystem: new InMemoryFs() },
    { mountPoint: "/workspace", filesystem: new InMemoryFs() },
  ],
});
```

### AI SDK Tool

For AI agents, use [`bash-tool`](https://github.com/vercel-labs/bash-tool) which is optimized for just-bash and provides a ready-to-use [AI SDK](https://ai-sdk.dev/) tool:

```bash
npm install bash-tool
```

```typescript
import { createBashTool } from "bash-tool";
import { generateText } from "ai";

const bashTool = createBashTool({
  files: { "/data/users.json": '[{"name": "Alice"}, {"name": "Bob"}]' },
});

const result = await generateText({
  model: "anthropic/claude-sonnet-4",
  tools: { bash: bashTool },
  prompt: "Count the users in /data/users.json",
});
```

See the [bash-tool documentation](https://github.com/vercel-labs/bash-tool) for more details and examples.

## Supported Commands

### File Operations

`cat`, `cp`, `df`, `file`, `ln`, `ls`, `mkdir`, `mv`, `readlink`, `rm`, `rmdir`, `split`, `stat`, `touch`, `tree`

### Text Processing

`awk`, `base64`, `column`, `comm`, `cut`, `diff`, `expand`, `fold`, `grep` (+ `egrep`, `fgrep`), `head`, `join`, `md5sum`, `nl`, `od`, `paste`, `printf`, `rev`, `rg`, `sed`, `sha1sum`, `sha256sum`, `sort`, `strings`, `tac`, `tail`, `tr`, `unexpand`, `uniq`, `wc`, `xargs`

### Data Processing

`jq` (JSON)

### Navigation & Environment

`basename`, `cd`, `dirname`, `du`, `echo`, `env`, `export`, `find`, `hostname`, `printenv`, `pwd`, `tee`

### Shell Utilities

`alias`, `bash`, `chmod`, `clear`, `date`, `expr`, `false`, `help`, `history`, `id`, `seq`, `sh`, `sleep`, `time`, `timeout`, `true`, `uname`, `unalias`, `uptime`, `which`, `whoami`

### Network Commands

`curl`

All commands support `--help` for usage information.

## Shell Features

- **Pipes**: `cmd1 | cmd2`
- **Redirections**: `>`, `>>`, `2>`, `2>&1`, `<`
- **Command chaining**: `&&`, `||`, `;`
- **Variables**: `$VAR`, `${VAR}`, `${VAR:-default}`
- **Positional parameters**: `$1`, `$2`, `$@`, `$#`
- **Glob patterns**: `*`, `?`, `[...]`
- **If statements**: `if COND; then CMD; elif COND; then CMD; else CMD; fi`
- **Functions**: `function name { ... }` or `name() { ... }`
- **Local variables**: `local VAR=value`
- **Loops**: `for`, `while`, `until`
- **Symbolic links**: `ln -s target link`
- **Hard links**: `ln target link`

## Default Layout

When created without options, Bash provides a Unix-like directory structure:

- `/home/user` - Default working directory (and `$HOME`)
- `/bin` - Contains stubs for all built-in commands
- `/usr/bin` - Additional binary directory
- `/tmp` - Temporary files directory

Commands can be invoked by path (e.g., `/bin/ls`) or by name.

## Network Access

Network access (and the `curl` command) is disabled by default for security. To enable it, configure the `network` option:

```typescript
// Allow specific URLs with GET/HEAD only (safest)
const env = new Bash({
  network: {
    allowedUrlPrefixes: [
      "https://api.github.com/repos/myorg/",
      "https://api.example.com",
    ],
  },
});

// Allow specific URLs with additional methods
const env = new Bash({
  network: {
    allowedUrlPrefixes: ["https://api.example.com"],
    allowedMethods: ["GET", "HEAD", "POST"], // Default: ["GET", "HEAD"]
  },
});

// Allow all URLs and methods (use with caution)
const env = new Bash({
  network: { dangerouslyAllowFullInternetAccess: true },
});
```

**Note:** The `curl` command only exists when network is configured. Without network configuration, `curl` returns "command not found".

### Allow-List Security

The allow-list enforces:

- **Origin matching**: URLs must match the exact origin (scheme + host + port)
- **Path prefix**: Only paths starting with the specified prefix are allowed
- **HTTP method restrictions**: Only GET and HEAD by default (configure `allowedMethods` for more)
- **Redirect protection**: Redirects to non-allowed URLs are blocked

### Using curl

```bash
# Fetch and process data
curl -s https://api.example.com/data | grep pattern

# POST JSON data
curl -X POST -H "Content-Type: application/json" \
  -d '{"key":"value"}' https://api.example.com/endpoint
```

## Execution Protection

Bash protects against infinite loops and deep recursion with configurable limits:

```typescript
const env = new Bash({
  executionLimits: {
    maxCallDepth: 100, // Max function recursion depth
    maxCommandCount: 10000, // Max total commands executed
    maxLoopIterations: 10000, // Max iterations per loop
    maxAwkIterations: 10000, // Max iterations in awk programs
    maxSedIterations: 10000, // Max iterations in sed scripts
  },
});
```

All limits have sensible defaults. Error messages include hints on which limit to increase. Feel free to increase if your scripts intentionally go beyond them.

## AST Transform Plugins

Parse bash scripts into an AST, run transform plugins, and serialize back to executable bash. Useful for instrumenting scripts (e.g., capturing per-command stdout/stderr) or analyzing them (e.g., extracting command names) before execution.

```typescript
import { Bash, BashTransformPipeline, TeePlugin, CommandCollectorPlugin } from "just-bash";

// Standalone pipeline — output can be run by any shell
const pipeline = new BashTransformPipeline()
  .use(new TeePlugin({ outputDir: "/tmp/logs" }))
  .use(new CommandCollectorPlugin());
const result = pipeline.transform("echo hello | grep hello");
result.script;             // transformed bash string
result.metadata.commands;  // ["echo", "grep", "tee"]

// Integrated API — exec() auto-applies transforms and returns metadata
const bash = new Bash();
bash.registerTransformPlugin(new CommandCollectorPlugin());
const execResult = await bash.exec("echo hello | grep hello");
execResult.metadata?.commands; // ["echo", "grep"]
```

See [src/transform/README.md](src/transform/README.md) for the full API, built-in plugins, and how to write custom plugins.

## Development

```bash
npm test         # Run tests in watch mode
npm run test:run # Run tests once
npm run typecheck # Type check without emitting
npm run build    # Build TypeScript
```

## AI Agent Instructions

For AI agents, we recommend using [`bash-tool`](https://github.com/vercel-labs/bash-tool) which is optimized for just-bash and provides additional guidance in its `AGENTS.md`:

```bash
cat node_modules/bash-tool/dist/AGENTS.md
```

## License

Apache-2.0
