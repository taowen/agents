. "$PSScriptRoot\common.ps1"

if ($env:X -ne "" -and $env:Y -ne "") {
    [WinInput]::SetCursorPos([int]$env:X, [int]$env:Y)
    Start-Sleep -Milliseconds 10
}

[WinInput]::mouse_event([WinInput]::MOUSEEVENTF_WHEEL, 0, 0, [int]$env:DELTA, [IntPtr]::Zero)
Write-Output "scrolled"
