. "$PSScriptRoot\common.ps1"
Add-Type -AssemblyName System.Drawing

$hwnd = [IntPtr][long]$env:HWND

# Get window dimensions
$rect = New-Object WinWindow+RECT
[WinWindow]::GetWindowRect($hwnd, [ref]$rect) | Out-Null
$w = $rect.Right - $rect.Left
$h = $rect.Bottom - $rect.Top

if ($w -le 0 -or $h -le 0) {
    [Console]::Error.WriteLine("Window has zero size (may be minimized)")
    exit 1
}

# Create bitmap and capture window content via PrintWindow
$bmp = New-Object System.Drawing.Bitmap($w, $h)
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $gfx.GetHdc()
[WinWindow]::PrintWindow($hwnd, $hdc, [WinWindow]::PW_RENDERFULLCONTENT) | Out-Null
$gfx.ReleaseHdc($hdc)
$gfx.Dispose()

$base64 = Resize-AndEncode $bmp

Write-Output "$($rect.Left),$($rect.Top),${w}x${h}"
Write-Output $base64
