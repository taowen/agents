# debug-calc-parent-msaa.ps1 — Use UIA to find button HWNDs, then try MSAA from parent

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;

public class WinApi2 {
    [DllImport("user32.dll")]
    public static extern IntPtr GetParent(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern int GetClassName(IntPtr hWnd, StringBuilder lpClassName, int nMaxCount);

    [DllImport("oleacc.dll")]
    public static extern int AccessibleObjectFromWindow(
        IntPtr hwnd, uint dwObjectID, ref Guid riid,
        [MarshalAs(UnmanagedType.Interface)] ref object ppvObject);

    [DllImport("oleacc.dll")]
    public static extern int AccessibleChildren(
        [MarshalAs(UnmanagedType.Interface)] object paccContainer,
        int iChildStart, int cChildren,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 2)] object[] rgvarChildren,
        out int pcObtained);

    [DllImport("user32.dll")]
    public static extern int GetDlgCtrlID(IntPtr hWnd);

    [DllImport("user32.dll", SetLastError = true)]
    public static extern IntPtr GetWindow(IntPtr hWnd, uint uCmd);

    public const uint OBJID_CLIENT = 0xFFFFFFFC;
    public const uint GW_CHILD = 5;
    public const uint GW_HWNDNEXT = 2;
}
'@ -ErrorAction SilentlyContinue

# Find Calculator via UIA
$root = [System.Windows.Automation.AutomationElement]::RootElement
$cond = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::NameProperty, "Calculator"
)
$calcWin = $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $cond)
if (-not $calcWin) { Write-Host "ERROR: Calculator not found"; exit 1 }

$calcHwnd = [IntPtr]$calcWin.Current.NativeWindowHandle
Write-Host "=== Calculator hwnd=$calcHwnd ==="

# ---- Step 1: Enumerate child windows of CalcFrame ----
Write-Host ""
Write-Host "=== Child window tree ==="
function Enum-Children($parentHwnd, $depth) {
    $child = [WinApi2]::GetWindow($parentHwnd, [WinApi2]::GW_CHILD)
    while ($child -ne [IntPtr]::Zero) {
        $sb = New-Object System.Text.StringBuilder 256
        [WinApi2]::GetClassName($child, $sb, 256) | Out-Null
        $cls = $sb.ToString()
        $ctrlId = [WinApi2]::GetDlgCtrlID($child)
        $indent = "  " * $depth
        Write-Host "${indent}hwnd=$child cls='$cls' ctrlId=$ctrlId"

        if ($depth -lt 3) {
            Enum-Children $child ($depth + 1)
        }
        $child = [WinApi2]::GetWindow($child, [WinApi2]::GW_HWNDNEXT)
    }
}
Enum-Children $calcHwnd 1

# ---- Step 2: Use UIA to find first button, get parent hwnd, try MSAA ----
Write-Host ""
Write-Host "=== UIA button → parent HWND → MSAA ==="

$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker

# Find first button-like element
function Find-FirstButton($el) {
    if (-not $el) { return $null }
    $ct = $el.Current.ControlType.ProgrammaticName -replace '^ControlType\.', ''
    $cls = $el.Current.ClassName
    if ($ct -eq 'Pane' -and $cls -eq 'Button') { return $el }
    if ($ct -eq 'Button') { return $el }

    $child = $walker.GetFirstChild($el)
    while ($child) {
        $result = Find-FirstButton $child
        if ($result) { return $result }
        $child = $walker.GetNextSibling($child)
    }
    return $null
}

$firstButton = Find-FirstButton $calcWin
if ($firstButton) {
    $btnHwnd = [IntPtr]$firstButton.Current.NativeWindowHandle
    $btnAutoId = $firstButton.Current.AutomationId
    Write-Host "  First button: hwnd=$btnHwnd autoId=$btnAutoId"

    $parentHwnd = [WinApi2]::GetParent($btnHwnd)
    $sb = New-Object System.Text.StringBuilder 256
    [WinApi2]::GetClassName($parentHwnd, $sb, 256) | Out-Null
    Write-Host "  Parent: hwnd=$parentHwnd class='$($sb.ToString())'"

    # MSAA from parent
    $iid = [Guid]'{618736E0-3C3D-11CF-810C-00AA00389B71}'
    $parentAcc = $null
    $hr = [WinApi2]::AccessibleObjectFromWindow($parentHwnd, [WinApi2]::OBJID_CLIENT, [ref]$iid, [ref]$parentAcc)
    Write-Host "  Parent MSAA hr=$hr"

    if ($hr -eq 0 -and $parentAcc) {
        $childCount = $parentAcc.accChildCount
        Write-Host "  Parent child count: $childCount"

        if ($childCount -gt 0 -and $childCount -le 100) {
            $children = New-Object object[] $childCount
            $obtained = 0
            [WinApi2]::AccessibleChildren($parentAcc, 0, $childCount, $children, [ref]$obtained) | Out-Null
            Write-Host "  Obtained: $obtained children"
            Write-Host ""

            for ($i = 0; $i -lt $obtained; $i++) {
                $child = $children[$i]
                if ($child -is [int]) {
                    try {
                        $cName = $parentAcc.accName($child)
                        $cRole = $parentAcc.accRole($child)
                        $cAction = try { $parentAcc.accDefaultAction($child) } catch { '' }
                        Write-Host "  [$i] childId=$child role=$cRole name='$cName' action='$cAction'"
                    } catch {
                        Write-Host "  [$i] childId=$child error: $_"
                    }
                } else {
                    try {
                        $cName = $child.accName($null)
                        $cRole = $child.accRole($null)
                        $cChildCount = $child.accChildCount
                        $cAction = try { $child.accDefaultAction($null) } catch { '' }
                        Write-Host "  [$i] IAccessible role=$cRole name='$cName' action='$cAction' children=$cChildCount"
                    } catch {
                        Write-Host "  [$i] IAccessible error: $_"
                    }
                }
            }
        }
    }
} else {
    Write-Host "  No button found via UIA"
}

# ---- Step 3: Try grandparent too ----
Write-Host ""
Write-Host "=== Grandparent MSAA ==="
if ($firstButton) {
    $btnHwnd2 = [IntPtr]$firstButton.Current.NativeWindowHandle
    $parentHwnd2 = [WinApi2]::GetParent($btnHwnd2)
    $grandParentHwnd = [WinApi2]::GetParent($parentHwnd2)
    $sb2 = New-Object System.Text.StringBuilder 256
    [WinApi2]::GetClassName($grandParentHwnd, $sb2, 256) | Out-Null
    Write-Host "  GrandParent: hwnd=$grandParentHwnd class='$($sb2.ToString())'"

    $iid2 = [Guid]'{618736E0-3C3D-11CF-810C-00AA00389B71}'
    $gpAcc = $null
    $hr2 = [WinApi2]::AccessibleObjectFromWindow($grandParentHwnd, [WinApi2]::OBJID_CLIENT, [ref]$iid2, [ref]$gpAcc)
    Write-Host "  GrandParent MSAA hr=$hr2"
    if ($hr2 -eq 0 -and $gpAcc) {
        $gpCC = $gpAcc.accChildCount
        Write-Host "  GrandParent child count: $gpCC"
        if ($gpCC -gt 0 -and $gpCC -le 100) {
            $gpChildren = New-Object object[] $gpCC
            $gpObt = 0
            [WinApi2]::AccessibleChildren($gpAcc, 0, $gpCC, $gpChildren, [ref]$gpObt) | Out-Null
            for ($i = 0; $i -lt $gpObt; $i++) {
                $child = $gpChildren[$i]
                if ($child -is [int]) {
                    try {
                        $n = $gpAcc.accName($child)
                        $r = $gpAcc.accRole($child)
                        Write-Host "  GP[$i] childId=$child role=$r name='$n'"
                    } catch {}
                } else {
                    try {
                        $n = $child.accName($null)
                        $r = $child.accRole($null)
                        $cc = $child.accChildCount
                        Write-Host "  GP[$i] role=$r name='$n' children=$cc"
                    } catch {
                        Write-Host "  GP[$i] error: $_"
                    }
                }
            }
        }
    }
}
