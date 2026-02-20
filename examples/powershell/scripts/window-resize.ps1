. "$PSScriptRoot\common.ps1"

$hwnd = [IntPtr][long]$env:HWND
# Restore first if minimized/maximized
[WinWindow]::ShowWindow($hwnd, [WinWindow]::SW_RESTORE) | Out-Null
Start-Sleep -Milliseconds 50

$rect = New-Object WinWindow+RECT
[WinWindow]::GetWindowRect($hwnd, [ref]$rect) | Out-Null

$newX = if ($env:X) { [int]$env:X } else { $rect.Left }
$newY = if ($env:Y) { [int]$env:Y } else { $rect.Top }
$newW = if ($env:W) { [int]$env:W } else { $rect.Right - $rect.Left }
$newH = if ($env:H) { [int]$env:H } else { $rect.Bottom - $rect.Top }

[WinWindow]::MoveWindow($hwnd, $newX, $newY, $newW, $newH, $true) | Out-Null
Write-Output "moved to $newX,$newY size $newW x $newH"
