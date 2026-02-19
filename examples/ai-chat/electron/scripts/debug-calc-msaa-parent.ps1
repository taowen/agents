# debug-calc-msaa-parent.ps1 — Try MSAA parent enumeration + other approaches
# Instead of getting IAccessible from each button HWND individually,
# get it from the PARENT dialog and enumerate children.

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;

public class A11yHelper {
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
    public static extern IntPtr FindWindowEx(IntPtr hwndParent, IntPtr hwndChildAfter,
        string lpszClass, string lpszWindow);

    [DllImport("user32.dll")]
    public static extern int GetDlgCtrlID(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, IntPtr wParam, StringBuilder lParam);

    public const uint OBJID_CLIENT  = 0xFFFFFFFC;
    public const uint OBJID_WINDOW  = 0x00000000;
    public const uint OBJID_SELF    = 0x00000000;
}
'@ -ErrorAction SilentlyContinue

# Find Calculator window
$root = [System.Windows.Automation.AutomationElement]::RootElement
$cond = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::NameProperty, "Calculator"
)
$calcWin = $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $cond)
if (-not $calcWin) {
    Write-Host "ERROR: Calculator not found"
    exit 1
}
$calcHwnd = [IntPtr]$calcWin.Current.NativeWindowHandle
Write-Host "=== Calculator hwnd=$calcHwnd ==="
Write-Host ""

# ---- Approach 1: MSAA from top-level window, enumerate children ----
Write-Host "=== Approach 1: MSAA from top-level Calculator window ==="
$iid = [Guid]'{618736E0-3C3D-11CF-810C-00AA00389B71}'
$accObj = $null
$hr = [A11yHelper]::AccessibleObjectFromWindow($calcHwnd, [A11yHelper]::OBJID_CLIENT, [ref]$iid, [ref]$accObj)
Write-Host "  AccessibleObjectFromWindow hr=$hr"
if ($hr -eq 0 -and $accObj) {
    try {
        $childCount = $accObj.accChildCount
        Write-Host "  Child count: $childCount"
        if ($childCount -gt 0) {
            $children = New-Object object[] $childCount
            $obtained = 0
            [A11yHelper]::AccessibleChildren($accObj, 0, $childCount, $children, [ref]$obtained) | Out-Null
            Write-Host "  Obtained: $obtained children"
            for ($i = 0; $i -lt $obtained; $i++) {
                $child = $children[$i]
                if ($child -is [int]) {
                    # Simple child element (identified by ID)
                    try {
                        $cName = $accObj.accName($child)
                        $cRole = $accObj.accRole($child)
                        Write-Host "  Child[$i] id=$child role=$cRole name='$cName'"
                    } catch {
                        Write-Host "  Child[$i] id=$child (error: $_)"
                    }
                } else {
                    # IAccessible child object
                    try {
                        $cName = $child.accName($null)
                        $cRole = $child.accRole($null)
                        $cChildCount = $child.accChildCount
                        Write-Host "  Child[$i] role=$cRole name='$cName' children=$cChildCount"

                        # Enumerate sub-children
                        if ($cChildCount -gt 0 -and $cChildCount -le 50) {
                            $subChildren = New-Object object[] $cChildCount
                            $subObtained = 0
                            [A11yHelper]::AccessibleChildren($child, 0, $cChildCount, $subChildren, [ref]$subObtained) | Out-Null
                            for ($j = 0; $j -lt $subObtained; $j++) {
                                $sub = $subChildren[$j]
                                if ($sub -is [int]) {
                                    try {
                                        $sName = $child.accName($sub)
                                        $sRole = $child.accRole($sub)
                                        Write-Host "    SubChild[$j] id=$sub role=$sRole name='$sName'"
                                    } catch {
                                        Write-Host "    SubChild[$j] id=$sub (error: $_)"
                                    }
                                } else {
                                    try {
                                        $sName = $sub.accName($null)
                                        $sRole = $sub.accRole($null)
                                        Write-Host "    SubChild[$j] role=$sRole name='$sName'"
                                    } catch {
                                        Write-Host "    SubChild[$j] (error: $_)"
                                    }
                                }
                            }
                        }
                    } catch {
                        Write-Host "  Child[$i] (IAccessible error: $_)"
                    }
                }
            }
        }
    } catch {
        Write-Host "  Error enumerating: $_"
    }
}

Write-Host ""

# ---- Approach 2: Find the #32770 dialog child, get MSAA from it ----
Write-Host "=== Approach 2: MSAA from #32770 dialog children ==="
$dialogHwnd = [A11yHelper]::FindWindowEx($calcHwnd, [IntPtr]::Zero, '#32770', $null)
Write-Host "  First #32770 dialog: $dialogHwnd"
if ($dialogHwnd -ne [IntPtr]::Zero) {
    # Find second #32770 (the button panel)
    $dialog2Hwnd = [A11yHelper]::FindWindowEx($calcHwnd, $dialogHwnd, '#32770', $null)
    Write-Host "  Second #32770 dialog: $dialog2Hwnd"

    $targetDialog = if ($dialog2Hwnd -ne [IntPtr]::Zero) { $dialog2Hwnd } else { $dialogHwnd }
    Write-Host "  Using dialog: $targetDialog"

    $accDialog = $null
    $hr2 = [A11yHelper]::AccessibleObjectFromWindow($targetDialog, [A11yHelper]::OBJID_CLIENT, [ref]$iid, [ref]$accDialog)
    Write-Host "  AccessibleObjectFromWindow hr=$hr2"
    if ($hr2 -eq 0 -and $accDialog) {
        $dChildCount = $accDialog.accChildCount
        Write-Host "  Dialog child count: $dChildCount"
        if ($dChildCount -gt 0) {
            $dChildren = New-Object object[] $dChildCount
            $dObtained = 0
            [A11yHelper]::AccessibleChildren($accDialog, 0, $dChildCount, $dChildren, [ref]$dObtained) | Out-Null
            Write-Host "  Obtained: $dObtained"
            for ($i = 0; $i -lt $dObtained; $i++) {
                $child = $dChildren[$i]
                if ($child -is [int]) {
                    try {
                        $cName = $accDialog.accName($child)
                        $cRole = $accDialog.accRole($child)
                        Write-Host "  DlgChild[$i] id=$child role=$cRole name='$cName'"
                    } catch {
                        Write-Host "  DlgChild[$i] id=$child (error: $_)"
                    }
                } else {
                    try {
                        $cName = $child.accName($null)
                        $cRole = $child.accRole($null)
                        Write-Host "  DlgChild[$i] role=$cRole name='$cName'"
                    } catch {
                        Write-Host "  DlgChild[$i] (IAccessible error: $_)"
                    }
                }
            }
        }
    }
}

Write-Host ""

# ---- Approach 3: FindWindowEx to enumerate Button children, try GetDlgCtrlID ----
Write-Host "=== Approach 3: EnumChildWindows + GetDlgCtrlID ==="
$targetDlg = if ($dialog2Hwnd -ne [IntPtr]::Zero) { $dialog2Hwnd } else {
    [A11yHelper]::FindWindowEx($calcHwnd, [IntPtr]::Zero, '#32770', $null)
}
if ($targetDlg -ne [IntPtr]::Zero) {
    $btnHwnd = [IntPtr]::Zero
    $count = 0
    do {
        $btnHwnd = [A11yHelper]::FindWindowEx($targetDlg, $btnHwnd, 'Button', $null)
        if ($btnHwnd -ne [IntPtr]::Zero) {
            $count++
            $ctrlId = [A11yHelper]::GetDlgCtrlID($btnHwnd)

            # Try WM_GETTEXT on this button
            $sb = New-Object System.Text.StringBuilder 256
            [A11yHelper]::SendMessage($btnHwnd, 0x000D, [IntPtr]256, $sb) | Out-Null
            $wmText = $sb.ToString()

            # Try MSAA on this button directly
            $btnAcc = $null
            $btnHr = [A11yHelper]::AccessibleObjectFromWindow($btnHwnd, [A11yHelper]::OBJID_CLIENT, [ref]$iid, [ref]$btnAcc)
            $btnName = ''
            if ($btnHr -eq 0 -and $btnAcc) {
                try { $btnName = $btnAcc.accName($null) } catch {}
            }

            Write-Host "  Button[$count] hwnd=$btnHwnd ctrlId=$ctrlId wmText='$wmText' msaaName='$btnName'"
        }
    } while ($btnHwnd -ne [IntPtr]::Zero)
    Write-Host "  Total buttons found: $count"
}

Write-Host ""

# ---- Approach 4: Try OBJID_WINDOW instead of OBJID_CLIENT on a button ----
Write-Host "=== Approach 4: MSAA OBJID_WINDOW on first button ==="
if ($targetDlg -ne [IntPtr]::Zero) {
    $firstBtn = [A11yHelper]::FindWindowEx($targetDlg, [IntPtr]::Zero, 'Button', $null)
    if ($firstBtn -ne [IntPtr]::Zero) {
        # OBJID_WINDOW = 0
        $acc0 = $null
        $hr0 = [A11yHelper]::AccessibleObjectFromWindow($firstBtn, 0, [ref]$iid, [ref]$acc0)
        Write-Host "  OBJID_WINDOW hr=$hr0"
        if ($hr0 -eq 0 -and $acc0) {
            try { Write-Host "  Name: '$($acc0.accName($null))'" } catch { Write-Host "  Name: error" }
            try { Write-Host "  Role: $($acc0.accRole($null))" } catch {}
            try { Write-Host "  ChildCount: $($acc0.accChildCount)" } catch {}
            $cc = $acc0.accChildCount
            if ($cc -gt 0) {
                for ($c = 1; $c -le $cc; $c++) {
                    try {
                        $cn = $acc0.accName($c)
                        $cr = $acc0.accRole($c)
                        Write-Host "  ChildId=$c role=$cr name='$cn'"
                    } catch {}
                }
            }
        }
    }
}
