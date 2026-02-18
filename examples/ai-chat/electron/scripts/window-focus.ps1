. "$PSScriptRoot\common.ps1"

if ($env:HWND) {
    $hwnd = [IntPtr][long]$env:HWND
    [WinWindow]::ShowWindow($hwnd, [WinWindow]::SW_RESTORE) | Out-Null
    Start-Sleep -Milliseconds 50
    [WinWindow]::SetForegroundWindow($hwnd) | Out-Null
    Write-Output "focused handle $($env:HWND)"
} elseif ($env:TITLE) {
    $proc = Get-Process | Where-Object {
        $_.MainWindowHandle -ne [IntPtr]::Zero -and $_.MainWindowTitle -like "*$($env:TITLE)*"
    } | Select-Object -First 1
    if ($proc) {
        $hwnd = $proc.MainWindowHandle
        [WinWindow]::ShowWindow($hwnd, [WinWindow]::SW_RESTORE) | Out-Null
        Start-Sleep -Milliseconds 50
        [WinWindow]::SetForegroundWindow($hwnd) | Out-Null
        Write-Output "focused $($proc.MainWindowTitle)"
    } else {
        [Console]::Error.WriteLine("No window found matching '$($env:TITLE)'")
        exit 1
    }
} else {
    [Console]::Error.WriteLine("Provide HWND or TITLE")
    exit 1
}
