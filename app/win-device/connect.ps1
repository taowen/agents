# connect.ps1 — Windows Device Agent
# Connects to ai-chat server via device auth + WebSocket, receives PowerShell
# code to execute, and sends back results + screenshots.
#
# Usage:
#   .\connect.ps1 -Server https://ai.connect-screen.com   # first time
#   .\connect.ps1                                           # subsequent runs

param(
    [string]$Server
)

$ErrorActionPreference = "Stop"

# ---- Config management ----

$ConfigDir = "$env:APPDATA\win-device"
$ConfigFile = "$ConfigDir\config.json"

function Load-Config {
    if (Test-Path $ConfigFile) {
        return Get-Content $ConfigFile -Raw | ConvertFrom-Json
    }
    return $null
}

function Save-Config($cfg) {
    if (-not (Test-Path $ConfigDir)) {
        New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null
    }
    $cfg | ConvertTo-Json | Set-Content $ConfigFile
}

# Load or create config
$config = Load-Config
if ($Server) {
    # First-time setup or server change
    $Server = $Server.TrimEnd("/")
    if (-not $config) {
        $config = [PSCustomObject]@{ server = $Server; token = $null }
    } else {
        $config.server = $Server
    }
    Save-Config $config
} elseif (-not $config -or -not $config.server) {
    Write-Host "No server configured. Run with -Server parameter first:" -ForegroundColor Red
    Write-Host "  .\connect.ps1 -Server https://ai.connect-screen.com" -ForegroundColor Yellow
    exit 1
}

$serverUrl = $config.server

# ---- Device auth flow ----

function Do-DeviceAuth {
    Write-Host "Starting device authorization..." -ForegroundColor Cyan

    # Start auth flow
    $startResp = Invoke-RestMethod -Method POST -Uri "$serverUrl/auth/device/start" -ContentType "application/json"
    $code = $startResp.code

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Device Code: $code" -ForegroundColor White -BackgroundColor DarkGreen
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Approve this code at: $serverUrl/device" -ForegroundColor Yellow
    Write-Host "Waiting for approval..." -ForegroundColor Gray

    # Poll for approval
    while ($true) {
        Start-Sleep -Seconds 2
        try {
            $checkResp = Invoke-RestMethod -Uri "$serverUrl/auth/device/check?code=$code"
            if ($checkResp.status -eq "approved" -and $checkResp.token) {
                Write-Host "Device approved!" -ForegroundColor Green
                return $checkResp.token
            } elseif ($checkResp.status -eq "expired") {
                Write-Host "Code expired. Please try again." -ForegroundColor Red
                exit 1
            }
            # "pending" — keep polling
        } catch {
            # Network error — retry
            Write-Host "." -NoNewline
        }
    }
}

# Ensure we have a token
if (-not $config.token) {
    $config.token = Do-DeviceAuth
    Save-Config $config
}

$token = $config.token

# ---- System prompt ----

$systemPrompt = @"
You are a Windows desktop automation assistant. You control the user's Windows PC by executing PowerShell code.

Available functions:
- take_screenshot — capture full screen, returns "screenshot captured (WxH)"
- click <x> <y> — left click at pixel coordinates
- right_click <x> <y> — right click
- double_click <x> <y> — double click
- move_mouse <x> <y> — move cursor
- type_text <text> — type text via clipboard paste
- key_press <key> [<modifiers>] — press key combo (e.g. key_press "A" "Ctrl" for Ctrl+A)
- scroll <direction> [<amount>] — scroll "up" or "down", amount defaults to 3
- list_windows — list visible windows with handles, titles, positions
- focus_window <handleOrTitle> — bring window to front by handle number or title substring
- window_screenshot <handle> — capture specific window
- get_accessibility_tree <handle> — get UI Automation tree for a window
- resize_window <handle> <x> <y> <w> <h> — move/resize window
- minimize_window <handle> — minimize window
- maximize_window <handle> — maximize window
- restore_window <handle> — restore window
- sleep_ms <ms> — wait

Workflow:
1. Always start by taking a screenshot to see the current screen state.
2. Use list_windows to find application windows and their handles.
3. Use get_accessibility_tree with a window handle to understand UI elements.
4. Use click/type_text/key_press to interact with the UI.
5. Take screenshots after actions to verify results.

You can also run arbitrary PowerShell commands for file operations, system info, etc.
Coordinates are in screen pixels. Use window bounds from list_windows or accessibility tree to calculate click targets.
"@

# ---- Ready message ----

$deviceName = $env:COMPUTERNAME
$readyMsg = @{
    type = "ready"
    deviceName = $deviceName
    deviceId = $deviceName
    execType = "exec_ps"
    systemPrompt = $systemPrompt
    tools = @(
        @{
            type = "function"
            function = @{
                name = "execute_ps"
                description = "Execute PowerShell code on the Windows device. Available functions: take_screenshot, click, right_click, double_click, move_mouse, type_text, key_press, scroll, list_windows, focus_window, window_screenshot, get_accessibility_tree, resize_window, minimize_window, maximize_window, restore_window, sleep_ms. You can also run any arbitrary PowerShell commands."
            }
        }
    )
} | ConvertTo-Json -Depth 5 -Compress

# ---- WebSocket helpers ----

function Send-WsMessage($ws, $msg) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($msg)
    $segment = New-Object System.ArraySegment[byte] -ArgumentList @(,$bytes)
    $ws.SendAsync($segment, [System.Net.WebSockets.WebSocketMessageType]::Text, $true, [System.Threading.CancellationToken]::None).GetAwaiter().GetResult() | Out-Null
}

function Receive-WsMessage($ws) {
    $buffer = New-Object byte[] 65536
    $result = $null
    $ms = New-Object System.IO.MemoryStream

    do {
        $segment = New-Object System.ArraySegment[byte] -ArgumentList @(,$buffer)
        $result = $ws.ReceiveAsync($segment, [System.Threading.CancellationToken]::None).GetAwaiter().GetResult()
        if ($result.MessageType -eq [System.Net.WebSockets.WebSocketMessageType]::Close) {
            return $null
        }
        $ms.Write($buffer, 0, $result.Count)
    } while (-not $result.EndOfMessage)

    $ms.Position = 0
    $reader = New-Object System.IO.StreamReader($ms)
    $text = $reader.ReadToEnd()
    $reader.Dispose()
    $ms.Dispose()
    return $text
}

# ---- Load helper functions ----

. "$PSScriptRoot\functions.ps1"

# ---- Execute PowerShell code ----

function Execute-Code($code) {
    $script:capturedScreenshots = @()
    $result = $null
    $errorMsg = $null

    try {
        $sb = [ScriptBlock]::Create($code)
        $output = $sb.Invoke()
        if ($output) {
            $result = ($output | Out-String).Trim()
        } else {
            $result = ""
        }
    } catch {
        $errorMsg = $_.Exception.Message
        $result = "Error: $errorMsg"
    }

    return @{
        result = $result
        screenshots = $script:capturedScreenshots
    }
}

# ---- Main connection loop ----

function Connect-And-Run {
    # Build WebSocket URL
    $wsScheme = if ($serverUrl.StartsWith("https")) { "wss" } else { "ws" }
    $host_ = $serverUrl -replace "^https?://", ""
    $wsUrl = "${wsScheme}://${host_}/agents/chat-agent/device-${deviceName}/device-connect?token=${token}"

    Write-Host "Connecting to $serverUrl as '$deviceName'..." -ForegroundColor Cyan

    $ws = New-Object System.Net.WebSockets.ClientWebSocket
    # Disable .NET protocol-level WebSocket pings — Cloudflare Workers don't respond
    # to them, causing .NET to consider the connection dead. The server sends
    # application-level {"type":"ping"} messages instead.
    $ws.Options.KeepAliveInterval = [TimeSpan]::Zero

    try {
        $ws.ConnectAsync([Uri]$wsUrl, [System.Threading.CancellationToken]::None).GetAwaiter().GetResult() | Out-Null
    } catch {
        $err = $_.Exception.Message
        if ($err -match "401|403|Unauthorized") {
            Write-Host "Token rejected. Re-authenticating..." -ForegroundColor Yellow
            $config.token = $null
            Save-Config $config
            $config.token = Do-DeviceAuth
            Save-Config $config
            $script:token = $config.token
            return  # will reconnect on next iteration
        }
        throw
    }

    Write-Host "Connected! Sending ready message..." -ForegroundColor Green

    # Send ready
    Send-WsMessage $ws $readyMsg
    Write-Host "Ready. Waiting for commands..." -ForegroundColor Green
    Write-Host ""

    # Message loop
    while ($ws.State -eq [System.Net.WebSockets.WebSocketState]::Open) {
        $msg = Receive-WsMessage $ws
        if ($null -eq $msg) {
            Write-Host "WebSocket closed by server." -ForegroundColor Yellow
            break
        }

        try {
            $data = $msg | ConvertFrom-Json
        } catch {
            Write-Host "Malformed message, skipping." -ForegroundColor DarkGray
            continue
        }

        switch ($data.type) {
            "ping" {
                $pong = '{"type":"pong"}'
                Send-WsMessage $ws $pong
            }
            "exec_ps" {
                $execId = $data.execId
                $code = $data.code
                Write-Host "--- Executing (${execId}) ---" -ForegroundColor Cyan
                Write-Host $code -ForegroundColor DarkGray

                $execResult = Execute-Code $code

                Write-Host "Result: $($execResult.result)" -ForegroundColor White
                if ($execResult.screenshots.Count -gt 0) {
                    Write-Host "  ($($execResult.screenshots.Count) screenshot(s) captured)" -ForegroundColor DarkCyan
                }
                Write-Host ""

                $response = @{
                    type = "exec_result"
                    execId = $execId
                    result = $execResult.result
                    screenshots = $execResult.screenshots
                } | ConvertTo-Json -Depth 3 -Compress

                Send-WsMessage $ws $response
            }
            "task_done" {
                Write-Host "=== Task Complete ===" -ForegroundColor Green
                if ($data.result) {
                    Write-Host $data.result -ForegroundColor White
                }
                Write-Host ""
            }
            default {
                # Ignore unknown message types
            }
        }
    }

    # Clean up
    if ($ws.State -eq [System.Net.WebSockets.WebSocketState]::Open) {
        try {
            $ws.CloseAsync([System.Net.WebSockets.WebSocketCloseStatus]::NormalClosure, "", [System.Threading.CancellationToken]::None).GetAwaiter().GetResult() | Out-Null
        } catch {}
    }
    $ws.Dispose()
}

# ---- Auto-reconnect loop ----

Write-Host "Win-Device Agent starting..." -ForegroundColor White
Write-Host "Device: $deviceName" -ForegroundColor White
Write-Host "Server: $serverUrl" -ForegroundColor White
Write-Host ""

while ($true) {
    try {
        Connect-And-Run
    } catch {
        Write-Host "Connection error: $($_.Exception.Message)" -ForegroundColor Red
    }
    Write-Host "Reconnecting in 5 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
