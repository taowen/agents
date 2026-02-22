# functions.ps1 — Helper functions preloaded into LLM eval scope.
# Each function wraps a script from scripts/ and captures output.

$script:capturedScreenshots = @()
$script:ScriptsDir = "$PSScriptRoot\scripts"

function take_screenshot {
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\screen-screenshot.ps1"
    # First line is resolution (e.g. "1920x1080"), rest is base64
    $lines = $output -split "`n"
    $resolution = $lines[0].Trim()
    $base64 = ($lines[1..($lines.Length - 1)] -join "").Trim()
    $script:capturedScreenshots += $base64
    return "screenshot captured ($resolution)"
}

function click($x, $y) {
    $env:X = $x; $env:Y = $y
    $env:DOWN_FLAG = 0x0002  # MOUSEEVENTF_LEFTDOWN
    $env:UP_FLAG = 0x0004    # MOUSEEVENTF_LEFTUP
    $env:CLICK_COUNT = 1
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\screen-click.ps1"
    Remove-Item Env:\X, Env:\Y, Env:\DOWN_FLAG, Env:\UP_FLAG, Env:\CLICK_COUNT -ErrorAction SilentlyContinue
    return $output
}

function right_click($x, $y) {
    $env:X = $x; $env:Y = $y
    $env:DOWN_FLAG = 0x0008  # MOUSEEVENTF_RIGHTDOWN
    $env:UP_FLAG = 0x0010    # MOUSEEVENTF_RIGHTUP
    $env:CLICK_COUNT = 1
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\screen-click.ps1"
    Remove-Item Env:\X, Env:\Y, Env:\DOWN_FLAG, Env:\UP_FLAG, Env:\CLICK_COUNT -ErrorAction SilentlyContinue
    return $output
}

function double_click($x, $y) {
    $env:X = $x; $env:Y = $y
    $env:DOWN_FLAG = 0x0002  # MOUSEEVENTF_LEFTDOWN
    $env:UP_FLAG = 0x0004    # MOUSEEVENTF_LEFTUP
    $env:CLICK_COUNT = 2
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\screen-click.ps1"
    Remove-Item Env:\X, Env:\Y, Env:\DOWN_FLAG, Env:\UP_FLAG, Env:\CLICK_COUNT -ErrorAction SilentlyContinue
    return $output
}

function move_mouse($x, $y) {
    $env:X = $x; $env:Y = $y
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\screen-move.ps1"
    Remove-Item Env:\X, Env:\Y -ErrorAction SilentlyContinue
    return $output
}

function type_text($text) {
    $env:TEXT = $text
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\screen-type.ps1"
    Remove-Item Env:\TEXT -ErrorAction SilentlyContinue
    return $output
}

function key_press($key, $modifiers) {
    # Build keybd_event script from key name and modifiers.
    # $key: virtual key name (e.g. "Enter", "Tab", "A")
    # $modifiers: comma-separated modifier names (e.g. "Ctrl", "Alt", "Shift")
    $vkMap = @{
        "Backspace"=0x08; "Tab"=0x09; "Enter"=0x0D; "Return"=0x0D; "Escape"=0x1B; "Esc"=0x1B
        "Space"=0x20; "PageUp"=0x21; "PageDown"=0x22; "End"=0x23; "Home"=0x24
        "Left"=0x25; "Up"=0x26; "Right"=0x27; "Down"=0x28
        "Delete"=0x2E; "Insert"=0x2D; "PrintScreen"=0x2C
        "F1"=0x70; "F2"=0x71; "F3"=0x72; "F4"=0x73; "F5"=0x74; "F6"=0x75
        "F7"=0x76; "F8"=0x77; "F9"=0x78; "F10"=0x79; "F11"=0x7A; "F12"=0x7B
        "A"=0x41; "B"=0x42; "C"=0x43; "D"=0x44; "E"=0x45; "F"=0x46; "G"=0x47
        "H"=0x48; "I"=0x49; "J"=0x4A; "K"=0x4B; "L"=0x4C; "M"=0x4D; "N"=0x4E
        "O"=0x4F; "P"=0x50; "Q"=0x51; "R"=0x52; "S"=0x53; "T"=0x54; "U"=0x55
        "V"=0x56; "W"=0x57; "X"=0x58; "Y"=0x59; "Z"=0x5A
        "0"=0x30; "1"=0x31; "2"=0x32; "3"=0x33; "4"=0x34
        "5"=0x35; "6"=0x36; "7"=0x37; "8"=0x38; "9"=0x39
    }
    $modMap = @{ "Ctrl"=0xA2; "Alt"=0xA4; "Shift"=0xA0; "Win"=0x5B }

    $script = ""
    # Press modifiers down
    if ($modifiers) {
        foreach ($mod in ($modifiers -split ",")) {
            $mod = $mod.Trim()
            if ($modMap.ContainsKey($mod)) {
                $vk = $modMap[$mod]
                $script += "[WinInput]::keybd_event($vk, 0, 0, [IntPtr]::Zero)`n"
            }
        }
    }
    # Press key
    $vk = if ($vkMap.ContainsKey($key)) { $vkMap[$key] } else { [int][char]$key.ToUpper() }
    $script += "[WinInput]::keybd_event($vk, 0, 0, [IntPtr]::Zero)`n"
    $script += "[WinInput]::keybd_event($vk, 0, [WinInput]::KEYEVENTF_KEYUP, [IntPtr]::Zero)`n"
    # Release modifiers
    if ($modifiers) {
        foreach ($mod in ($modifiers -split ",")) {
            $mod = $mod.Trim()
            if ($modMap.ContainsKey($mod)) {
                $vk = $modMap[$mod]
                $script += "[WinInput]::keybd_event($vk, 0, [WinInput]::KEYEVENTF_KEYUP, [IntPtr]::Zero)`n"
            }
        }
    }

    $env:PRESS_SCRIPT = $script
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\screen-keypress.ps1"
    Remove-Item Env:\PRESS_SCRIPT -ErrorAction SilentlyContinue
    return $output
}

function scroll($direction, $amount) {
    if (-not $amount) { $amount = 3 }
    # direction: "up", "down", "left", "right"
    $delta = switch ($direction) {
        "up"    {  120 * $amount }
        "down"  { -120 * $amount }
        default {  120 * $amount }
    }
    $env:DELTA = $delta
    $env:X = ""; $env:Y = ""
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\screen-scroll.ps1"
    Remove-Item Env:\DELTA, Env:\X, Env:\Y -ErrorAction SilentlyContinue
    return $output
}

function list_windows {
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\window-list.ps1"
    return $output
}

function focus_window($handleOrTitle) {
    if ($handleOrTitle -match '^\d+$') {
        $env:HWND = $handleOrTitle
        $env:TITLE = ""
    } else {
        $env:HWND = ""
        $env:TITLE = $handleOrTitle
    }
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\window-focus.ps1"
    Remove-Item Env:\HWND, Env:\TITLE -ErrorAction SilentlyContinue
    return $output
}

function window_screenshot($handle) {
    $env:HWND = $handle
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\window-screenshot.ps1"
    Remove-Item Env:\HWND -ErrorAction SilentlyContinue
    $lines = $output -split "`n"
    $geometry = $lines[0].Trim()
    $base64 = ($lines[1..($lines.Length - 1)] -join "").Trim()
    $script:capturedScreenshots += $base64
    return "window screenshot captured ($geometry)"
}

function get_accessibility_tree($handle) {
    $env:HWND = $handle
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\window-accessibility.ps1"
    Remove-Item Env:\HWND -ErrorAction SilentlyContinue
    return $output
}

function resize_window($handle, $x, $y, $w, $h) {
    $env:HWND = $handle
    $env:X = $x; $env:Y = $y; $env:W = $w; $env:H = $h
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\window-resize.ps1"
    Remove-Item Env:\HWND, Env:\X, Env:\Y, Env:\W, Env:\H -ErrorAction SilentlyContinue
    return $output
}

function minimize_window($handle) {
    $env:HWND = $handle
    $env:SW_CMD = 6  # SW_MINIMIZE
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\window-set-state.ps1"
    Remove-Item Env:\HWND, Env:\SW_CMD -ErrorAction SilentlyContinue
    return $output
}

function maximize_window($handle) {
    $env:HWND = $handle
    $env:SW_CMD = 3  # SW_MAXIMIZE
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\window-set-state.ps1"
    Remove-Item Env:\HWND, Env:\SW_CMD -ErrorAction SilentlyContinue
    return $output
}

function restore_window($handle) {
    $env:HWND = $handle
    $env:SW_CMD = 9  # SW_RESTORE
    $output = & powershell.exe -NoProfile -File "$script:ScriptsDir\window-set-state.ps1"
    Remove-Item Env:\HWND, Env:\SW_CMD -ErrorAction SilentlyContinue
    return $output
}

function sleep_ms($ms) {
    Start-Sleep -Milliseconds $ms
    return "slept ${ms}ms"
}
