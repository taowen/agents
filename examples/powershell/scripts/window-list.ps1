. "$PSScriptRoot\common.ps1"

$results = @()
Get-Process | Where-Object { $_.MainWindowHandle -ne [IntPtr]::Zero } | ForEach-Object {
    $hwnd = $_.MainWindowHandle
    if ([WinWindow]::IsWindowVisible($hwnd)) {
        $rect = New-Object WinWindow+RECT
        [WinWindow]::GetWindowRect($hwnd, [ref]$rect) | Out-Null
        $results += [PSCustomObject]@{
            handle = $hwnd.ToInt64()
            title = $_.MainWindowTitle
            processName = $_.ProcessName
            pid = $_.Id
            x = $rect.Left
            y = $rect.Top
            width = $rect.Right - $rect.Left
            height = $rect.Bottom - $rect.Top
            isMinimized = [WinWindow]::IsIconic($hwnd)
            isMaximized = [WinWindow]::IsZoomed($hwnd)
        }
    }
}
$results | ConvertTo-Json -Compress
