. "$PSScriptRoot\common.ps1"

if ($env:X -ne "" -and $env:Y -ne "") {
    [WinInput]::SetCursorPos([int]$env:X, [int]$env:Y)
    Start-Sleep -Milliseconds 10
}

$totalDelta = [int]$env:DELTA
$step = if ($totalDelta -gt 0) { 360 } else { -360 }
$remaining = [Math]::Abs($totalDelta)

while ($remaining -gt 0) {
    $chunk = [Math]::Min($remaining, 360)
    $actualDelta = if ($totalDelta -gt 0) { $chunk } else { -$chunk }
    [WinInput]::mouse_event([WinInput]::MOUSEEVENTF_WHEEL, 0, 0, $actualDelta, [IntPtr]::Zero)
    $remaining -= $chunk
    if ($remaining -gt 0) { Start-Sleep -Milliseconds 10 }
}
Write-Output "scrolled"
