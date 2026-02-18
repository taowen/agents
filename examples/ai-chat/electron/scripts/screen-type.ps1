. "$PSScriptRoot\common.ps1"
Add-Type -AssemblyName System.Windows.Forms

# Save current clipboard
$saved = $null
try { $saved = [System.Windows.Forms.Clipboard]::GetText() } catch {}

# Set text to clipboard and paste
[System.Windows.Forms.Clipboard]::SetText($env:TEXT)
Start-Sleep -Milliseconds 50

# Ctrl+V
[WinInput]::keybd_event(0xA2, 0, 0, [IntPtr]::Zero)
[WinInput]::keybd_event(0x56, 0, 0, [IntPtr]::Zero)
[WinInput]::keybd_event(0x56, 0, [WinInput]::KEYEVENTF_KEYUP, [IntPtr]::Zero)
[WinInput]::keybd_event(0xA2, 0, [WinInput]::KEYEVENTF_KEYUP, [IntPtr]::Zero)

Start-Sleep -Milliseconds 50

# Restore clipboard
if ($saved -ne $null) {
    [System.Windows.Forms.Clipboard]::SetText($saved)
} else {
    [System.Windows.Forms.Clipboard]::Clear()
}

Write-Output "typed text"
