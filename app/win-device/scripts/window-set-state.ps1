. "$PSScriptRoot\common.ps1"

$hwnd = [IntPtr][long]$env:HWND
[WinWindow]::ShowWindow($hwnd, [int]$env:SW_CMD) | Out-Null
Write-Output "done"
