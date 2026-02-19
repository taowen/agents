# debug-calc-buttons.ps1 — Dump all UIA properties for Calculator elements
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File debug-calc-buttons.ps1
#
# Tries every known method to get element names

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public class WinTextDbg {
    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
    [DllImport("user32.dll")]
    public static extern int GetWindowTextLength(IntPtr hWnd);
    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, IntPtr wParam, StringBuilder lParam);
}
'@ -ErrorAction SilentlyContinue

Add-Type @'
using System;
using System.Runtime.InteropServices;
public class OleAccDbg {
    [DllImport("oleacc.dll")]
    public static extern int AccessibleObjectFromWindow(
        IntPtr hwnd, uint dwObjectID, ref Guid riid,
        [MarshalAs(UnmanagedType.Interface)] ref object ppvObject);
    public const uint OBJID_CLIENT = 0xFFFFFFFC;
    public const uint OBJID_SELF = 0x00000000;
}
'@ -ErrorAction SilentlyContinue

# Find Calculator window
$root = [System.Windows.Automation.AutomationElement]::RootElement
$cond = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::NameProperty, "Calculator"
)
$calcWin = $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $cond)

if (-not $calcWin) {
    # Try partial match
    $allWins = $root.FindAll([System.Windows.Automation.TreeScope]::Children,
        [System.Windows.Automation.Condition]::TrueCondition)
    foreach ($w in $allWins) {
        if ($w.Current.Name -like "*Calc*" -or $w.Current.Name -like "*计算*") {
            $calcWin = $w
            break
        }
    }
}

if (-not $calcWin) {
    Write-Host "ERROR: Calculator window not found. Please open Calculator first."
    exit 1
}

$hwnd = [IntPtr]$calcWin.Current.NativeWindowHandle
Write-Host "=== Calculator found: hwnd=$hwnd Name='$($calcWin.Current.Name)' ==="
Write-Host ""

$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
$elementCount = 0
$buttonLikeCount = 0

function Dump-Element($el, $depth) {
    if (-not $el) { return }
    if ($depth -gt 10) { return }

    $script:elementCount++
    $indent = "  " * $depth
    $ct = $el.Current.ControlType.ProgrammaticName -replace '^ControlType\.', ''
    $className = $el.Current.ClassName
    $uiaName = $el.Current.Name
    $autoId = $el.Current.AutomationId
    $nativeHwnd = $el.Current.NativeWindowHandle
    $rect = $el.Current.BoundingRectangle

    # Detect button-like elements (Pane with ClassName=Button, or actual Button)
    $isButtonLike = ($ct -eq 'Button') -or ($ct -eq 'Pane' -and $className -eq 'Button')

    if ($isButtonLike) {
        $script:buttonLikeCount++
        Write-Host "${indent}=== Button-like #$($script:buttonLikeCount) (UIA: $ct, Class: $className) ==="
        Write-Host "${indent}  UIA Name        : '$uiaName'"
        Write-Host "${indent}  AutomationId    : '$autoId'"
        Write-Host "${indent}  ClassName       : '$className'"
        Write-Host "${indent}  NativeHwnd      : $nativeHwnd"

        if (-not $rect.IsEmpty) {
            Write-Host "${indent}  Bounds          : [$([int]$rect.Left),$([int]$rect.Top)][$([int]$rect.Right),$([int]$rect.Bottom)]"
        }

        # HelpText
        try {
            $helpText = $el.Current.HelpText
            if ($helpText) { Write-Host "${indent}  HelpText        : '$helpText'" }
        } catch {}

        # AccessKey
        try {
            $accessKey = $el.Current.AccessKey
            if ($accessKey) { Write-Host "${indent}  AccessKey       : '$accessKey'" }
        } catch {}

        # LegacyIAccessible pattern (ID = 10018)
        try {
            $legacyPatternId = [System.Windows.Automation.AutomationPattern]::LookupById(10018)
            if ($legacyPatternId) {
                $legacyPattern = $el.GetCurrentPattern($legacyPatternId)
                if ($legacyPattern) {
                    try { $v = $legacyPattern.Current.Name;        if ($v) { Write-Host "${indent}  Legacy.Name     : '$v'" } } catch {}
                    try { $v = $legacyPattern.Current.Description; if ($v) { Write-Host "${indent}  Legacy.Desc     : '$v'" } } catch {}
                    try { $v = $legacyPattern.Current.Value;       if ($v) { Write-Host "${indent}  Legacy.Value    : '$v'" } } catch {}
                    try { $v = $legacyPattern.Current.DefaultAction; if ($v) { Write-Host "${indent}  Legacy.Action   : '$v'" } } catch {}
                }
            }
        } catch {
            Write-Host "${indent}  LegacyIAcc      : not available ($_)"
        }

        # GetWindowText (if has HWND)
        if ($nativeHwnd -ne 0) {
            try {
                $h = [IntPtr]$nativeHwnd
                $len = [WinTextDbg]::GetWindowTextLength($h)
                if ($len -gt 0) {
                    $sb = New-Object System.Text.StringBuilder ($len + 1)
                    [WinTextDbg]::GetWindowText($h, $sb, $sb.Capacity) | Out-Null
                    Write-Host "${indent}  GetWindowText   : '$($sb.ToString())'"
                } else {
                    Write-Host "${indent}  GetWindowText   : (empty)"
                }
            } catch {
                Write-Host "${indent}  GetWindowText   : (error: $_)"
            }

            # WM_GETTEXT
            try {
                $h = [IntPtr]$nativeHwnd
                $sb = New-Object System.Text.StringBuilder 256
                [WinTextDbg]::SendMessage($h, 0x000D, [IntPtr]256, $sb) | Out-Null
                $wmText = $sb.ToString()
                if ($wmText) { Write-Host "${indent}  WM_GETTEXT      : '$wmText'" }
            } catch {}
        }

        # MSAA accName (if has HWND)
        if ($nativeHwnd -ne 0) {
            try {
                $h = [IntPtr]$nativeHwnd
                $iid = [Guid]'{618736E0-3C3D-11CF-810C-00AA00389B71}'
                $accObj = $null
                $hr = [OleAccDbg]::AccessibleObjectFromWindow($h, [OleAccDbg]::OBJID_CLIENT, [ref]$iid, [ref]$accObj)
                if ($hr -eq 0 -and $accObj) {
                    try { $v = $accObj.accName($null);        if ($v) { Write-Host "${indent}  MSAA.Name       : '$v'" } } catch {}
                    try { $v = $accObj.accDescription($null);  if ($v) { Write-Host "${indent}  MSAA.Desc       : '$v'" } } catch {}
                    try { $v = $accObj.accValue($null);        if ($v) { Write-Host "${indent}  MSAA.Value      : '$v'" } } catch {}
                    try { $v = $accObj.accDefaultAction($null); if ($v) { Write-Host "${indent}  MSAA.Action     : '$v'" } } catch {}
                } else {
                    Write-Host "${indent}  MSAA            : (hr=$hr)"
                }
            } catch {
                Write-Host "${indent}  MSAA            : (error: $_)"
            }
        } else {
            Write-Host "${indent}  MSAA/GetWinText : (skipped, NativeHwnd=0)"
        }

        # Interaction patterns
        $patterns = @()
        try { if ($el.GetCurrentPropertyValue([System.Windows.Automation.AutomationElement]::IsInvokePatternAvailableProperty)) { $patterns += "Invoke" } } catch {}
        try { if ($el.GetCurrentPropertyValue([System.Windows.Automation.AutomationElement]::IsTogglePatternAvailableProperty)) { $patterns += "Toggle" } } catch {}
        try { if ($el.GetCurrentPropertyValue([System.Windows.Automation.AutomationElement]::IsValuePatternAvailableProperty)) { $patterns += "Value" } } catch {}
        if ($patterns.Count -gt 0) {
            Write-Host "${indent}  Patterns        : $($patterns -join ', ')"
        }

        Write-Host ""
    } else {
        # Brief summary for non-button elements
        $brief = "${indent}[$ct]"
        if ($className) { $brief += " cls='$className'" }
        if ($uiaName) { $brief += " Name='$uiaName'" }
        if ($autoId) { $brief += " AutoId='$autoId'" }
        Write-Host $brief
    }

    # Traverse children
    $child = $walker.GetFirstChild($el)
    while ($child) {
        Dump-Element $child ($depth + 1)
        $child = $walker.GetNextSibling($child)
    }
}

Dump-Element $calcWin 0
Write-Host ""
Write-Host "=== Total: $elementCount elements, $buttonLikeCount button-like ==="
