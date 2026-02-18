. "$PSScriptRoot\common.ps1"
Add-Type -AssemblyName System.Drawing

$imagePath = $env:IMAGE_PATH
$x = [int]$env:X
$y = [int]$env:Y

$bmp = [System.Drawing.Bitmap]::FromFile($imagePath)

$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$gfx.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

$pen = New-Object System.Drawing.Pen([System.Drawing.Color]::Red, 2)

# Crosshair lines (40px arms)
$arm = 40
$gfx.DrawLine($pen, ($x - $arm), $y, ($x + $arm), $y)
$gfx.DrawLine($pen, $x, ($y - $arm), $x, ($y + $arm))

# Circle around the point (radius 12)
$r = 12
$gfx.DrawEllipse($pen, ($x - $r), ($y - $r), ($r * 2), ($r * 2))

# Label text
$label = "($x, $y)"
$font = New-Object System.Drawing.Font("Arial", 12, [System.Drawing.FontStyle]::Bold)
$brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Red)
$bgBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(200, 255, 255, 255))

$size = $gfx.MeasureString($label, $font)
$labelX = $x + $r + 4
$labelY = $y - $r - 4

# Keep label within image bounds
if (($labelX + $size.Width) -gt $bmp.Width) { $labelX = $x - $r - 4 - $size.Width }
if ($labelY -lt 0) { $labelY = $y + $r + 4 }

$gfx.FillRectangle($bgBrush, $labelX, $labelY, $size.Width, $size.Height)
$gfx.DrawString($label, $font, $brush, $labelX, $labelY)

$pen.Dispose()
$font.Dispose()
$brush.Dispose()
$bgBrush.Dispose()
$gfx.Dispose()

$w = $bmp.Width
$h = $bmp.Height
$base64 = Encode-Bitmap $bmp

Write-Output "${w}x${h}"
Write-Output $base64
