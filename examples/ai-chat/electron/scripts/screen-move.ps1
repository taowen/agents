. "$PSScriptRoot\common.ps1"

[WinInput]::SetCursorPos([int]$env:X, [int]$env:Y)
Write-Output "moved"
