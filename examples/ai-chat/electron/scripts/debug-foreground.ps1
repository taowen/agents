# debug-foreground.ps1 — report current foreground window handle and title
. "$PSScriptRoot\common.ps1"
$fg = [WinWindow]::GetForegroundWindow()
$sb = New-Object System.Text.StringBuilder 256
[WinWindow]::GetWindowText($fg, $sb, 256) | Out-Null
Write-Output "$($fg.ToInt64())|$($sb.ToString())"
