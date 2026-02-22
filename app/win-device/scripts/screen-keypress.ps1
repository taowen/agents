. "$PSScriptRoot\common.ps1"

# PRESS_SCRIPT contains pre-generated keybd_event calls from JS
Invoke-Expression $env:PRESS_SCRIPT

Write-Output "pressed"
