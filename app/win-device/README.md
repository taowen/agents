# Win-Device — PowerShell Windows Device Agent

A pure PowerShell agent that connects to the ai-chat server, allowing the LLM to control a Windows desktop via PowerShell code execution.

## Prerequisites

- Windows 10/11
- PowerShell 5.1+ (built into Windows)
- .NET Framework 4.5+ (built into Windows)

## First-time setup

```powershell
.\connect.ps1 -Server https://ai.connect-screen.com
```

This will:
1. Start the device authorization flow
2. Display a 6-character code
3. Open `https://ai.connect-screen.com/device` in your browser and enter the code
4. Once approved, the token is saved to `%APPDATA%\win-device\config.json`

## Subsequent runs

```powershell
.\connect.ps1
```

The saved token is reused automatically. If the token expires, re-authentication happens automatically.

## How it works

1. The agent connects via WebSocket to the ai-chat server's ChatAgent Durable Object
2. It sends a `ready` message declaring `execType: "exec_ps"` and an `execute_ps` tool
3. When a user sends a chat message in the device session, the LLM generates PowerShell code
4. The server sends `exec_ps` messages; the agent executes the code and returns results + screenshots

## Available automation functions

| Function | Description |
|----------|-------------|
| `take_screenshot` | Capture full screen |
| `click <x> <y>` | Left click at coordinates |
| `right_click <x> <y>` | Right click |
| `double_click <x> <y>` | Double click |
| `move_mouse <x> <y>` | Move cursor |
| `type_text <text>` | Type text via clipboard paste |
| `key_press <key> [<modifiers>]` | Press key combo (e.g. `key_press "A" "Ctrl"`) |
| `scroll <direction> [<amount>]` | Scroll up/down |
| `list_windows` | List visible windows |
| `focus_window <handleOrTitle>` | Bring window to front |
| `window_screenshot <handle>` | Capture specific window |
| `get_accessibility_tree <handle>` | UI Automation tree |
| `resize_window <handle> <x> <y> <w> <h>` | Move/resize window |
| `minimize_window <handle>` | Minimize |
| `maximize_window <handle>` | Maximize |
| `restore_window <handle>` | Restore |
| `sleep_ms <ms>` | Wait |

The LLM can also run arbitrary PowerShell commands for file operations, system info, etc.

## Config file

Stored at `%APPDATA%\win-device\config.json`:

```json
{
  "server": "https://ai.connect-screen.com",
  "token": "eyJ..."
}
```

Delete this file to force re-authentication.
