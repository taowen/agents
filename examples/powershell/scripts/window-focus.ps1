. "$PSScriptRoot\common.ps1"

function Focus-Window($hwnd) {
    [WinWindow]::ShowWindow($hwnd, [WinWindow]::SW_RESTORE) | Out-Null
    Start-Sleep -Milliseconds 50
    # Simulate Alt press/release to bypass SetForegroundWindow restriction.
    # Windows only allows the current foreground process to change foreground;
    # a brief Alt key event tricks the OS into allowing it from any process.
    [WinInput]::keybd_event(0xA4, 0, 0, [IntPtr]::Zero)     # Alt down
    [WinInput]::keybd_event(0xA4, 0, [WinInput]::KEYEVENTF_KEYUP, [IntPtr]::Zero)  # Alt up
    [WinWindow]::SetForegroundWindow($hwnd) | Out-Null
    Start-Sleep -Milliseconds 100
}

if ($env:HWND) {
    $hwnd = [IntPtr][long]$env:HWND
    Focus-Window $hwnd
    Write-Output "focused handle $($env:HWND)"
} elseif ($env:TITLE) {
    $proc = Get-Process | Where-Object {
        $_.MainWindowHandle -ne [IntPtr]::Zero -and $_.MainWindowTitle -like "*$($env:TITLE)*"
    } | Select-Object -First 1
    if ($proc) {
        $hwnd = $proc.MainWindowHandle
        Focus-Window $hwnd
        Write-Output "focused $($proc.MainWindowTitle)"
    } else {
        [Console]::Error.WriteLine("No window found matching '$($env:TITLE)'")
        exit 1
    }
} else {
    [Console]::Error.WriteLine("Provide HWND or TITLE")
    exit 1
}
