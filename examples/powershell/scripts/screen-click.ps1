. "$PSScriptRoot\common.ps1"

[WinInput]::SetCursorPos([int]$env:X, [int]$env:Y)
Start-Sleep -Milliseconds 10

$downFlag = [uint32]$env:DOWN_FLAG
$upFlag = [uint32]$env:UP_FLAG
$count = [int]$env:CLICK_COUNT

for ($i = 0; $i -lt $count; $i++) {
    [WinInput]::mouse_event($downFlag, 0, 0, 0, [IntPtr]::Zero)
    [WinInput]::mouse_event($upFlag, 0, 0, 0, [IntPtr]::Zero)
    if ($i -lt ($count - 1)) {
        Start-Sleep -Milliseconds 50
    }
}

Write-Output "clicked"
