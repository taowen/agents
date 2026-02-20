# PowerShell Agent — Windows Desktop Automation via LLM

A standalone CLI agent that uses LLM to drive Windows desktop automation (screenshots, clicks, typing, window management) and PowerShell command execution.

## Quick Start

```powershell
cd examples\powershell
npm install
```

Create a `.env` file (see `.env.example`):

```
LLM_PROVIDER=openai-compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=gpt-4o
```

Run:

```powershell
npx tsx src/main.ts "Take a screenshot and describe what you see"
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | No | `openai-compatible` | `openai-compatible` or `google` |
| `LLM_BASE_URL` | For openai-compatible | — | API base URL |
| `LLM_API_KEY` | Yes | — | API key |
| `LLM_MODEL` | Yes | — | Model name |

The `.env` file in the project root is auto-loaded. Environment variables set in the shell take precedence.

## Architecture

The agent loop is built on the `pi` framework (`packages/pi`), which provides the `Agent` class, structured tool definitions (`AgentTool`), event streaming, and message history management. Domain logic (coordinate conversion, auto-screenshot, accessibility tree handling) lives in tool `execute` callbacks.

```
src/
├── main.ts            # CLI entry: loads .env, parses args, builds LLM model, runs agent
├── agent-loop.ts      # createDesktopAgent factory: tools + pi Agent (no custom convertToLlm)
├── win-automation.ts  # Windows desktop automation: dispatches to PowerShell scripts
├── types.ts           # ScreenControlParams, ScreenControlResult, BashResult
└── action-aliases.ts  # Normalizes LLM action names (e.g. "press_key" → "key_press")

scripts/               # PowerShell scripts for desktop automation
├── common.ps1         # Shared C# interop (WinInput, WinWindow), DPI awareness
├── screen-*.ps1       # Mouse/keyboard operations
├── window-*.ps1       # Window management operations
└── annotate-image.ps1 # Screenshot annotation (debug)
```

### pi Framework Integration

`agent-loop.ts` exports `createDesktopAgent(config)` which returns a `pi` `Agent` directly — no custom `convertToLlm` or `transformContext`. The pi framework's default `convertToModelMessages` handles everything:

- **ImageContent in tool results** → pi automatically splits into a follow-up user message (LLMs need images in user messages, not tool results)
- **Accessibility trees** → returned as `TextContent` with a label prefix directly in the tool result (no `details` hack)
- **No old screenshot stripping** — all messages are kept as-is in pi's native format

`main.ts` uses the Agent API directly:
- `agent.subscribe()` drives logging (step counts, tool calls, LLM text output) and enforces `maxSteps` by calling `agent.abort()`
- `agent.prompt()` + `agent.waitForIdle()` to run the agent

Key dependency note: `z` (zod) must be imported from `"pi"` (not `"zod"` directly) to ensure the same zod instance is used for tool parameter schemas across all packages.

### Agent Tools

The LLM has access to two tools:

**`desktop`** — Screen and window operations:
- `list_windows` — Discover windows and their handles
- `window_screenshot` — Capture a window (supports pixel and accessibility tree modes)
- `click`, `mouse_move`, `scroll` — Mouse operations (0-999 normalized coordinates)
- `type`, `key_press` — Keyboard operations
- `focus_window`, `resize_window`, `minimize_window`, `maximize_window`, `restore_window`

**`powershell`** — Run arbitrary PowerShell commands (system admin, file ops, registry, etc.)

### Coordinate System

All coordinates use a 0-999 grid mapped to the screenshot:
- (0,0) = top-left, (999,999) = bottom-right, (500,500) = center
- Automatically converted to pixel coordinates based on the last screenshot dimensions

### Auto-Screenshot

After every interactive action (click, type, scroll, etc.), the agent automatically takes a follow-up screenshot so the LLM sees the result without an extra call.

## Logs

Each run saves logs and screenshots to a temp directory:
- Windows: `%TEMP%\powershell-agent\logs\`
- WSL: `/tmp/powershell-agent/logs/`

Contents:
- `agent.log` — Full agent log (also streamed to stderr)
- `step-XX-<action>.png` — Screenshots at each step
- `step-XX-<label>.txt` — Accessibility trees

## Running from WSL

This agent works from WSL (Windows Subsystem for Linux). The key challenge is that WSL doesn't automatically pass custom environment variables to Windows executables like `powershell.exe`. The agent handles this by auto-setting `WSLENV` for any custom env vars passed to PowerShell scripts.

Tested configuration:
- WSL2 (Ubuntu) calling `powershell.exe`
- `list_windows`, `window_screenshot` (pixel + accessibility), `focus_window`, `click`, `type`, `key_press`, `scroll`, `powershell` commands all work
- Screenshots are captured via `PrintWindow` API (works even for background windows)

## Tested LLM Providers

| Provider | Config |
|----------|--------|
| Doubao (ByteDance) | `LLM_PROVIDER=openai-compatible`, `LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3` |
| OpenAI | `LLM_PROVIDER=openai-compatible`, `LLM_BASE_URL=https://api.openai.com/v1` |
| Google Gemini | `LLM_PROVIDER=google` (no base URL needed) |

Any OpenAI-compatible API endpoint should work.

## Example Session

```
$ npx tsx src/main.ts "列出当前打开的窗口"

[standalone] Provider: openai-compatible
[standalone] Model: doubao-seed-2.0-code
[agent] step 1...
[agent] tool: desktop({"action":"list_windows"})
[agent] desktop: list_windows → 7 windows
[agent] step 2...

当前打开的窗口列表：
1. Visual Studio Code - ".env - cloudflare-agents [WSL: Ubuntu]"
2. Microsoft Edge - "Grok / X and 3 more pages"
3. 微信 (Weixin)
4. Windows Input Experience
5. 其他后台窗口
```

```
$ npx tsx src/main.ts "对VS Code窗口截图，描述你看到的内容"

[agent] tool: desktop({"action":"list_windows"})
[agent] desktop: list_windows → 7 windows
[agent] tool: desktop({"action":"window_screenshot","handle":67076,"mode":"pixel"})
[agent] desktop: window_screenshot [pixel] handle=67076 → 1821x972

我看到了一个Visual Studio Code窗口...
- 左侧面板：资源管理器，显示项目结构
- 中间区域：.env 文件内容和代码编辑器
- 右侧面板：聊天/终端界面
```
