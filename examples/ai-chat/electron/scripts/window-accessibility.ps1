. "$PSScriptRoot\common.ps1"
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

# MSAA helper for fallback name retrieval on owner-drawn controls
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class OleAccHelper {
    [DllImport("oleacc.dll")]
    public static extern int AccessibleObjectFromWindow(
        IntPtr hwnd, uint dwObjectID, ref Guid riid,
        [MarshalAs(UnmanagedType.Interface)] ref object ppvObject);
    public const uint OBJID_CLIENT = 0xFFFFFFFC;
}
'@ -ErrorAction SilentlyContinue

# GetWindowText helper for fallback name retrieval on Win32 controls with own HWND
Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public class WinText {
    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
    [DllImport("user32.dll")]
    public static extern int GetWindowTextLength(IntPtr hWnd);
}
'@ -ErrorAction SilentlyContinue

$hwnd = [IntPtr][long]$env:HWND

# Get window rect for geometry header and coordinate offset
$rect = New-Object WinWindow+RECT
[WinWindow]::GetWindowRect($hwnd, [ref]$rect) | Out-Null
$w = $rect.Right - $rect.Left
$h = $rect.Bottom - $rect.Top

if ($w -le 0 -or $h -le 0) {
    [Console]::Error.WriteLine("Window has zero size (may be minimized)")
    exit 1
}

$winLeft = $rect.Left
$winTop  = $rect.Top

# Output geometry header (same format as window-screenshot.ps1)
Write-Output "$winLeft,$winTop,${w}x${h}"
# Output window size line for LLM
Write-Output "Window: ${w}x${h}"

# Get automation element from handle
$element = [System.Windows.Automation.AutomationElement]::FromHandle($hwnd)
if (-not $element) {
    [Console]::Error.WriteLine("Cannot get AutomationElement for HWND")
    exit 1
}

$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker

$MAX_NODES = 500
$MAX_DEPTH = 15
$nodeCount = 0

function Get-InteractionPatterns($el) {
    $patterns = @()
    try {
        if ($el.GetCurrentPropertyValue([System.Windows.Automation.AutomationElement]::IsInvokePatternAvailableProperty)) {
            $patterns += "Invoke"
        }
    } catch {}
    try {
        if ($el.GetCurrentPropertyValue([System.Windows.Automation.AutomationElement]::IsTogglePatternAvailableProperty)) {
            $patterns += "Toggle"
            try {
                $toggle = $el.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern)
                $state = $toggle.Current.ToggleState
                if ($state -ne [System.Windows.Automation.ToggleState]::Indeterminate) {
                    $patterns += "ToggleState=$state"
                }
            } catch {}
        }
    } catch {}
    try {
        if ($el.GetCurrentPropertyValue([System.Windows.Automation.AutomationElement]::IsSelectionItemPatternAvailableProperty)) {
            $patterns += "SelectionItem"
        }
    } catch {}
    try {
        if ($el.GetCurrentPropertyValue([System.Windows.Automation.AutomationElement]::IsExpandCollapsePatternAvailableProperty)) {
            $patterns += "ExpandCollapse"
        }
    } catch {}
    try {
        if ($el.GetCurrentPropertyValue([System.Windows.Automation.AutomationElement]::IsScrollPatternAvailableProperty)) {
            $patterns += "Scroll"
        }
    } catch {}
    return $patterns
}

function Traverse-Element($el, $depth) {
    if ($script:nodeCount -ge $MAX_NODES) { return }
    if ($depth -gt $MAX_DEPTH) { return }
    if (-not $el) { return }

    $script:nodeCount++

    $indent = "  " * $depth
    $controlType = $el.Current.ControlType.ProgrammaticName -replace '^ControlType\.', ''
    $name = $el.Current.Name
    $boundingRect = $el.Current.BoundingRectangle

    # Name fallback chain: AutomationId → GetWindowText → MSAA accName

    # Fallback 1: GetWindowText (works for Win32 controls with own HWND)
    if (-not $name -and $el.Current.NativeWindowHandle -ne 0) {
        try {
            $nativeHwnd = [IntPtr]$el.Current.NativeWindowHandle
            $len = [WinText]::GetWindowTextLength($nativeHwnd)
            if ($len -gt 0) {
                $sb = New-Object System.Text.StringBuilder ($len + 1)
                [WinText]::GetWindowText($nativeHwnd, $sb, $sb.Capacity) | Out-Null
                $name = $sb.ToString()
            }
        } catch {}
    }

    # Fallback 2: MSAA (AccessibleObjectFromWindow + accName)
    # Handles owner-drawn Win32 controls (e.g. old Calculator buttons).
    if (-not $name -and $el.Current.NativeWindowHandle -ne 0) {
        try {
            $nativeHwnd = [IntPtr]$el.Current.NativeWindowHandle
            $iid = [Guid]'{618736E0-3C3D-11CF-810C-00AA00389B71}'
            $accObj = $null
            $hr = [OleAccHelper]::AccessibleObjectFromWindow($nativeHwnd, [OleAccHelper]::OBJID_CLIENT, [ref]$iid, [ref]$accObj)
            if ($hr -eq 0 -and $accObj) {
                try { $name = $accObj.accName($null) } catch {}
            }
        } catch {}
    }

    # Fix misidentified control type: Win32 Button class reported as Pane
    if ($controlType -eq 'Pane' -and $el.Current.ClassName -eq 'Button') {
        $controlType = 'Button'
    }

    # Diagnostics: log unnamed interactive elements to stderr for debugging
    if (-not $name -and ($controlType -eq 'Button' -or $controlType -eq 'MenuItem' -or $controlType -eq 'CheckBox')) {
        $hwndVal = $el.Current.NativeWindowHandle
        $dbgAutoId = try { $el.Current.AutomationId } catch { '' }
        if (-not $boundingRect.IsEmpty) {
            $dbgBounds = "$([int]($boundingRect.Left - $winLeft)),$([int]($boundingRect.Top - $winTop)),$([int]($boundingRect.Right - $winLeft)),$([int]($boundingRect.Bottom - $winTop))"
        } else {
            $dbgBounds = 'empty'
        }
        [Console]::Error.WriteLine("DEBUG: unnamed $controlType hwnd=$hwndVal autoId='$dbgAutoId' bounds=$dbgBounds")
    }

    # Build element line
    $line = "${indent}[${controlType}]"

    if ($name) {
        $line += " Name=`"$name`""
    }

    # Get Value if available
    try {
        if ($el.GetCurrentPropertyValue([System.Windows.Automation.AutomationElement]::IsValuePatternAvailableProperty)) {
            $valPattern = $el.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
            $val = $valPattern.Current.Value
            if ($val) {
                if ($val.Length -gt 100) {
                    $val = $val.Substring(0, 100) + "..."
                }
                $line += " Value=`"$val`""
            }
        }
    } catch {}

    # Bounds as window-relative coordinates
    if (-not $boundingRect.IsEmpty) {
        $bLeft   = [int]($boundingRect.Left   - $winLeft)
        $bTop    = [int]($boundingRect.Top    - $winTop)
        $bRight  = [int]($boundingRect.Right  - $winLeft)
        $bBottom = [int]($boundingRect.Bottom - $winTop)
        $line += " bounds=[$bLeft,$bTop][$bRight,$bBottom]"
    }

    # Interaction patterns
    $patterns = Get-InteractionPatterns $el
    if ($patterns.Count -gt 0) {
        $line += " " + ($patterns -join " ")
    }

    # Enabled/disabled state
    if (-not $el.Current.IsEnabled) {
        $line += " disabled"
    }

    # Focused state
    try {
        if ($el.Current.HasKeyboardFocus) {
            $line += " focused"
        }
    } catch {}

    Write-Output $line

    # Traverse children
    $child = $walker.GetFirstChild($el)
    while ($child -and $script:nodeCount -lt $MAX_NODES) {
        Traverse-Element $child ($depth + 1)
        $child = $walker.GetNextSibling($child)
    }
}

Traverse-Element $element 0

if ($script:nodeCount -ge $MAX_NODES) {
    Write-Output "# WARNING: tree truncated at $MAX_NODES nodes"
}
