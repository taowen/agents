# Standalone Windows Agent — 开发指南

## 1. 项目概览

Standalone 模式是一个 **无 Electron、无浏览器** 的纯 CLI agent，直接在 Windows 上通过 Node.js 运行。它接收一句自然语言指令，调用 LLM 驱动一个 agent loop，自动操控 Windows 桌面（截图、点击、键盘输入、窗口管理、PowerShell 命令）来完成任务。

### 运行模式对比

| 模式 | 入口 | 依赖 | UI |
|------|------|------|----|
| **Standalone** (本文档) | `standalone.ts` → `run-standalone.ps1` | Node.js + tsx，无 Electron | 纯 CLI，stderr 输出日志 |
| **Electron** | `main.ts` → `run-windows.ps1` | Electron + Cloudflare 部署 | 浏览器窗口 |

### 架构层次

```
WSL (bash)                          Windows (PowerShell / Node.js)
┌─────────────────────┐             ┌──────────────────────────────────┐
│ run-standalone-      │  wslpath   │  run-standalone.ps1              │
│ on-windows.sh        │ ────────► │  ├─ 拷贝源码到 %TEMP%            │
│ ├─ build just-bash   │           │  ├─ 生成 package.json            │
│ ├─ npm pack tarball  │           │  ├─ npm install                  │
│ └─ 启动 PowerShell   │           │  └─ npx tsx standalone.ts <args> │
└─────────────────────┘             └──────────────┬───────────────────┘
                                                   │
                                    ┌──────────────▼───────────────────┐
                                    │  standalone.ts                    │
                                    │  ├─ 构建 LLM model (ai-sdk)      │
                                    │  ├─ createAgentLoop()             │
                                    │  └─ runAgent(prompt)              │
                                    └──────────────┬───────────────────┘
                                                   │
                                    ┌──────────────▼───────────────────┐
                                    │  agent-loop.ts (src/shared/)      │
                                    │  ├─ streamText() 多步循环         │
                                    │  ├─ tool: desktop → screenControl│
                                    │  └─ tool: powershell → psExecutor│
                                    └──────────────┬───────────────────┘
                                                   │
                                    ┌──────────────▼───────────────────┐
                                    │  win-automation.ts (electron/)    │
                                    │  ├─ screenControl() 分发          │
                                    │  ├─ runPowerShell() 调 .ps1 脚本  │
                                    │  └─ runPowerShellCommand() 任意命令│
                                    └──────────────┬───────────────────┘
                                                   │
                                    ┌──────────────▼───────────────────┐
                                    │  electron/scripts/*.ps1           │
                                    │  ├─ screen-screenshot.ps1        │
                                    │  ├─ screen-click.ps1             │
                                    │  ├─ screen-type.ps1              │
                                    │  ├─ window-list.ps1              │
                                    │  └─ ...                          │
                                    └──────────────────────────────────┘
```

## 2. 文件清单与职责

### 启动脚本

| 文件 | 运行环境 | 职责 |
|------|---------|------|
| `run-standalone-on-windows.sh` | WSL bash | 编排入口：build just-bash → npm pack → 启动 PowerShell → 进程清理 |
| `run-standalone.ps1` | Windows PowerShell | 准备工作目录、拷贝源码、生成 package.json、npm install、启动 standalone.ts |

### TypeScript 核心

| 文件 | 职责 |
|------|------|
| `electron/standalone.ts` | CLI 入口，解析参数和环境变量，构建 LLM model，创建 agent loop 并运行 |
| `src/shared/agent-loop.ts` | **核心 agent 循环**，多步 tool-use 对话，坐标转换，自动截图，历史管理 |
| `electron/win-automation.ts` | Windows 桌面自动化统一入口，调用 PowerShell 脚本执行操作 |
| `src/shared/screen-control-types.ts` | 类型定义：ScreenControlParams, ScreenControlResult, BashResult |
| `src/shared/action-aliases.ts` | action 名称别名映射（处理 LLM 输出的各种拼写变体） |

### PowerShell 脚本 (`electron/scripts/`)

| 脚本 | 功能 |
|------|------|
| `common.ps1` | 共享 C# interop 定义（WinInput, WinWindow），DPI 感知，Bitmap 编码 |
| `screen-screenshot.ps1` | 全屏截图，输出 `WxH\n<base64>` |
| `screen-click.ps1` | 鼠标点击 (通过环境变量 X, Y, DOWN_FLAG, UP_FLAG, CLICK_COUNT) |
| `screen-type.ps1` | 键盘输入文本 |
| `screen-keypress.ps1` | 按键+修饰键 |
| `screen-move.ps1` | 鼠标移动 |
| `screen-scroll.ps1` | 鼠标滚轮 |
| `window-list.ps1` | 列出可见窗口 (JSON) |
| `window-focus.ps1` | 聚焦窗口 |
| `window-resize.ps1` | 移动/调整窗口 |
| `window-set-state.ps1` | 最小化/最大化/还原 |
| `window-screenshot.ps1` | 单窗口截图，输出 `left,top,WxH\n<base64>` |
| `annotate-image.ps1` | 在截图上画红色十字标记，标签显示归一化坐标（通过 `NORM_X`/`NORM_Y` 环境变量） |
| `cloud-drive.ps1` | 挂载 cloud:\ PSDrive（可选） |

## 3. 开发环境准备

### 前置条件

- **WSL2** (Ubuntu) — 用于编译 just-bash 和作为启动入口
- **Windows** — Node.js (≥18)、npm、PowerShell 5.1+
- **LLM API** — 需要一个兼容 OpenAI 或 Google 的 API endpoint

### 环境变量配置

创建或编辑 `examples/ai-chat/.env.standalone`：

```bash
LLM_PROVIDER=openai-compatible   # 或 "google"
LLM_BASE_URL=https://your-api-endpoint/v1
LLM_API_KEY=your-api-key
LLM_MODEL=your-model-name

# 可选：cloud 文件存储
CLOUD_URL=https://your-cloud-server.com
CLOUD_COOKIE=your-session-cookie
```

支持的 Provider 配置：

| Provider | 必填环境变量 |
|----------|-------------|
| `openai-compatible` (默认) | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` |
| `google` | `LLM_API_KEY`, `LLM_MODEL`（`LLM_BASE_URL` 可选） |

## 4. 启动与运行

### 标准启动（从 WSL）

```bash
# 从 repo 根目录
bash examples/ai-chat/run-standalone-on-windows.sh "截个屏看看桌面上有什么"
```

这会自动：
1. 在 WSL 中 build just-bash (`npm run build`)
2. npm pack 生成 tarball
3. 在 Windows 上启动 `run-standalone.ps1`

### 手动启动（直接在 Windows PowerShell）

适用于跳过 just-bash 重新编译的场景：

```powershell
# 在 Windows PowerShell 中
cd C:\path\to\cloudflare-agents\examples\ai-chat

# 使用 npm registry 版本的 just-bash
.\run-standalone.ps1 "Take a screenshot"

# 使用本地 tarball
.\run-standalone.ps1 -JustBashTarball "C:\path\to\just-bash-2.10.0.tgz" "Take a screenshot"

# 指定源码目录（默认通过 wslpath 自动检测）
.\run-standalone.ps1 -ProjectDir "\\wsl$\Ubuntu\home\user\cloudflare-agents\examples\ai-chat" "Take a screenshot"
```

### 仅修改 TypeScript 后的快速迭代

如果只修改了 `standalone.ts`、`agent-loop.ts`、`win-automation.ts` 等 TypeScript 文件（未改动 just-bash）：

```bash
# 在 WSL 中，跳过 just-bash build，直接重新运行
# run-standalone.ps1 每次都会重新拷贝源码，所以直接重新运行即可
bash examples/ai-chat/run-standalone-on-windows.sh "你的测试指令"
```

或者更快：直接在 Windows 的工作目录中手动拷贝改动的文件再运行：

```powershell
# 工作目录位于
# %TEMP%\windows-agent-standalone\
# 直接替换文件后：
cd $env:TEMP\windows-agent-standalone
npx tsx electron/standalone.ts "你的测试指令"
```

## 5. 调试技巧

### 日志输出

Standalone 模式的日志同时输出到 **stderr** 和 **文件**：

- **实时日志**：运行时的 stderr 输出
- **日志文件**：`%TEMP%\windows-agent-standalone\logs\agent.log`
- **截图文件**：`%TEMP%\windows-agent-standalone\logs\step-XX-<action>.png`
- **文本文件**：`%TEMP%\windows-agent-standalone\logs\step-XX-<label>.txt`（a11y tree 等）

从 WSL 查看日志：

```bash
# 查看日志文件
cat /mnt/c/Users/$USER/AppData/Local/Temp/windows-agent-standalone/logs/agent.log

# 查看截图
ls /mnt/c/Users/$USER/AppData/Local/Temp/windows-agent-standalone/logs/*.png
```

### 调试 agent-loop

在 `agent-loop.ts` 中，`onLog` 回调会输出每一步的详细信息：

```
[agent] step 1...
[agent] tool: desktop({"action":"screenshot"})
[agent] desktop: screenshot → 1920x1080
[agent] step 1 LLM: 3200ms, tools: 1, text: 0 chars
[agent] autoScreenshot: screenshot → 1920x1080
[agent] step 2...
[agent] tool: desktop({"action":"click","x":500,"y":300})
[agent] desktop: click norm(500,300)→pixel(960,324)→desktop(960,324) → success
```

关键日志信息：
- `step N LLM: Xms` — LLM 调用耗时
- `tools: N` — 本步骤调用了几个 tool
- `[agent] LLM text: ...` — LLM 的推理文本输出（非 tool call 部分），用于理解 LLM 为何做出某个决策（如切换到 pixel 模式）
- `[agent] a11y: auto: tree accepted (47 elements with names)` — a11y tree 诊断，显示 `isTreeUseful` 的判断结果和原因
- `[agent] a11y: auto: tree rejected (only 2 elements), fell back to pixel` — tree 被拒绝时显示原因
- `norm(x,y)→pixel(x,y)→desktop(x,y)` — 坐标转换链路（归一化→像素→桌面绝对坐标）

### 调试 PowerShell 脚本

单独测试某个 PowerShell 脚本：

```powershell
# 测试截图
powershell -NoProfile -ExecutionPolicy Bypass -File electron\scripts\screen-screenshot.ps1

# 测试点击（需要设置环境变量）
$env:X = "500"; $env:Y = "500"; $env:DOWN_FLAG = "2"; $env:UP_FLAG = "4"; $env:CLICK_COUNT = "1"
powershell -NoProfile -ExecutionPolicy Bypass -File electron\scripts\screen-click.ps1

# 测试窗口列表
powershell -NoProfile -ExecutionPolicy Bypass -File electron\scripts\window-list.ps1
```

### 调试坐标系统

agent-loop 使用 **0-999 归一化坐标系**：
- LLM 输出 `(x=500, y=300)` 表示屏幕/窗口的 50% 水平，30% 垂直位置
- `normToPixel()` 将其转换为实际像素坐标
- `toDesktopCoords()` 在窗口截图模式下加上窗口偏移量

**annotate 标签坐标**：`annotate` action 在截图上画十字线时，标签文字显示的是 **归一化坐标**（0-999），而非 pixel 坐标。这通过 `ScreenControlParams.normX/normY` 字段传递到 `annotate-image.ps1` 的 `NORM_X`/`NORM_Y` 环境变量实现。这样 LLM 从图片上读到的坐标和它应该发送给 click 的坐标一致，避免坐标泄露导致的二次转换错误。

调试坐标问题时：
1. 查看日志中 `norm→pixel→desktop` 的转换结果
2. 查看 `logs/` 目录下的截图文件，确认截图内容是否正确
3. 查看 `step-XX-click+annotate.png` 文件（坐标操作时自动生成），验证十字标记是否在目标位置
4. 确认 annotate 截图上的标签数字是归一化坐标（如 `(633, 926)`），而非 pixel 坐标

### 调试进程生命周期

`run-standalone-on-windows.sh` 使用 `(sleep 999999) | powershell.exe ...` 模式保持 stdin 管道打开。`standalone.ts` 监听 stdin 关闭来检测父进程死亡：

```typescript
// standalone.ts:92-97
process.stdin.resume();
process.stdin.on("end", () => {
  log("[standalone] stdin closed, parent died — exiting");
  process.exit(0);
});
```

如果 agent 不退出，检查：
- WSL 端的 `cleanup()` 是否正确触发
- PID 文件 `%TEMP%\windows-agent-standalone.pid` 是否存在
- 使用 `taskkill.exe /PID <pid> /T /F` 手动终止

## 6. 关键数据流

### 一次完整的 agent 执行流程

```
1. standalone.ts: 解析 CLI 参数和环境变量
2. standalone.ts: 创建 LLM model (ai-sdk)
3. standalone.ts: createAgentLoop({ getModel, executePowerShell, executeScreenControl })
4. standalone.ts: agent.runAgent(prompt)
   │
   ├─ agent-loop.ts: 将 prompt 加入 history
   ├─ agent-loop.ts: streamText({ model, system, messages, tools })
   │   └─ LLM 返回 tool_call: desktop({ action: "screenshot" })
   ├─ agent-loop.ts: executeScreenControl({ action: "screenshot" })
   │   └─ win-automation.ts: runPowerShell("screen-screenshot.ps1")
   │       └─ PowerShell: 截屏 → base64
   ├─ agent-loop.ts: 将截图加入 history (image message)
   ├─ agent-loop.ts: 下一步 streamText()
   │   └─ LLM 看到截图，返回 tool_call: desktop({ action: "click", x: 700, y: 500 })
   ├─ agent-loop.ts: normToPixel(700, 500) → pixel(1344, 540)
   ├─ agent-loop.ts: executeScreenControl({ action: "click", x: 1344, y: 540 })
   │   └─ win-automation.ts: toDesktopCoords() + runPowerShell("screen-click.ps1")
   ├─ agent-loop.ts: autoScreenshot() → 自动截屏确认结果
   ├─ ... 循环直到 maxSteps 或 LLM 输出纯文本 ...
   │
   └─ agent-loop.ts: 返回最终文本回复
```

### tool 注册表

| Tool 名 | agent-loop 中的处理 | 实际执行 |
|---------|-------------------|---------|
| `desktop` | 坐标转换 + 窗口管理 + 自动截图 | `executeScreenControl()` → `win-automation.ts` → `.ps1` |
| `powershell` | 直接执行 | `executePowerShell()` → `runPowerShellCommand()` |

## 7. 常见修改场景

### 添加新的 screen action

1. 在 `electron/scripts/` 下创建新的 `.ps1` 脚本
2. 在 `win-automation.ts` 的 `screenControl()` switch 中添加分支
3. 如果需要，在 `agent-loop.ts` 的 tool schema 的 `action` enum 中添加选项
4. 在 `action-aliases.ts` 中添加常见别名

### 修改 LLM system prompt

编辑 `src/shared/agent-loop.ts` 中的 `SYSTEM_PROMPT_BASE` 或 `CLOUD_DRIVE_PROMPT`。

### 添加新的 LLM provider

编辑 `electron/standalone.ts` 中 model 创建部分，新增 `if (provider === "xxx")` 分支。

### 修改 agent 的 tool 定义

编辑 `src/shared/agent-loop.ts` 中的 `desktopToolDef`（统一的屏幕+窗口工具）或 `powershellToolDef`。

### 调整 npm 依赖

`run-standalone.ps1` 会生成一个精简的 `package.json`（第 54-68 行）。如果需要新依赖，在这里添加。同时在 `run-standalone-on-windows.sh` 不需要改动（它只负责 build 和传参）。

## 8. run-standalone.ps1 逐段解读

```powershell
# 参数：源码目录、just-bash tarball 路径、传给 standalone.ts 的剩余参数
param(
    [string]$ProjectDir,
    [string]$JustBashTarball,
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$AgentArgs
)
```

**工作目录**：`%TEMP%\windows-agent-standalone\`。避免了 UNC 路径和 monorepo workspace 的问题。

**文件拷贝**：将 `electron/` 和 `src/shared/` 拷贝到工作目录。standalone.ts 用相对路径 import 这两个目录下的文件。

**just-bash 依赖**：
- 有 tarball → `"file:just-bash-2.10.0.tgz"`（本地开发用最新代码）
- 无 tarball → `"^2.10.0"`（使用 npm registry 版本）

**BOM 问题**：PowerShell 5.1 的 `-Encoding UTF8` 会写入 BOM，导致 tsx 解析失败，所以用 `[System.IO.File]::WriteAllText()` 写无 BOM 的 UTF-8。

**环境变量加载**：从 `.env.standalone` 读取，但只在 `LLM_API_KEY` 未设置时加载（避免覆盖已有环境变量）。

**PID 文件**：写入 `%TEMP%\windows-agent-standalone.pid`，供 WSL 端的 cleanup() 使用 taskkill 终止整个进程树。

## 9. 故障排查

| 问题 | 排查方向 |
|------|---------|
| `npm install failed` | 检查 Windows 上的 Node.js/npm 版本；检查生成的 package.json 格式 |
| `LLM_API_KEY is required` | 确认 `.env.standalone` 存在且格式正确 |
| 截图全黑 | DPI 缩放问题，确认 `common.ps1` 中 `SetProcessDPIAware()` 被调用 |
| 坐标点击偏移 | 检查 `lastWindowOffset`；确认 window_screenshot 的 header 格式正确；确认 annotate 标签显示的是归一化坐标而非 pixel 坐标 |
| agent 超时不退出 | 检查 `maxSteps` 设置；检查 LLM 是否一直在调用 tool 不输出文本 |
| `Cannot determine project directory` | 手动传 `-ProjectDir` 或确认 WSL 的 `wslpath` 可用 |
| tsx 报错 BOM | 确认 package.json 写入时使用了无 BOM 的 UTF-8 编码 |
| 进程树残留 | 手动执行 `taskkill.exe /PID <pid> /T /F`，或删除 PID 文件后重启 |

## 10. Accessibility Tree 模式

### 概述

`window_screenshot` 默认使用 accessibility tree（文本）而非像素截图。tree 通过 `window-accessibility.ps1` 调用 Windows UI Automation 获取，列出窗口内所有控件的类型、名称、bounds 和交互模式。LLM 可以直接从 bounds 计算点击坐标，无需视觉识别。

### 名称回退机制

**问题**：部分控件在 UIA 中 `Name` 为空。常见场景：
- Win32 owner-drawn 控件（如旧版计算器按钮）：`ControlType` 也可能被错报为 `Pane`
- UWP/XAML 控件（如 Windows 10/11 计算器按钮）：UIA `Name` 为空但 `AutomationId` 有值（如 `num7Button`）
- 虚拟 UIA 元素（`NativeWindowHandle = 0`）：MSAA 和 GetWindowText 都无法工作

**名称回退优先级**（`window-accessibility.ps1`）：

1. **UIA Name**（默认）：`$el.Current.Name`
2. **AutomationId**：`$el.Current.AutomationId`——UWP/XAML 应用通常在此暴露控件标识（如 `num7Button`、`plusButton`）
3. **GetWindowText**：通过 `user32.dll` 的 `GetWindowText` 读取窗口标题文本。仅在 `NativeWindowHandle != 0` 时尝试
4. **MSAA accName**：通过 `oleacc.dll` 的 `AccessibleObjectFromWindow` 获取 `IAccessible` 对象，调用 `accName`。仅在 `NativeWindowHandle != 0` 时尝试。MSAA 是比 UIA 更老的辅助功能接口，许多 Win32 控件仍然暴露 MSAA 信息

**ControlType 修正**：当 `ClassName` 是 `Button` 但 UIA 报告 `ControlType.Pane` 时，将显示类型改为 `Button`。

**stderr 诊断**：当 Button/MenuItem/CheckBox 经过所有回退仍无 Name 时，输出 DEBUG 行到 stderr（不影响 stdout tree），例如：
```
DEBUG: unnamed Button hwnd=0 autoId='' bounds=27,173,78,214
```
这帮助排查哪些控件无法获取名称以及原因（hwnd=0 表示虚拟元素，autoId 为空表示开发者未设置）。

修复前后对比：
```
# 修复前（无法使用）
[Pane] bounds=[27,268][78,309]
[Pane] bounds=[27,318][78,359]

# 修复后（MSAA 回退成功）
[Button] Name="7" bounds=[27,268][78,309] Invoke
[Button] Name="4" bounds=[27,318][78,359] Invoke

# 修复后（AutomationId 回退，UWP 计算器）
[Button] Name="num7Button" bounds=[27,268][78,309]
[Button] Name="num4Button" bounds=[27,318][78,359]
```

### autoScreenshot 与坐标验证

**已解决问题 1：mode 粘住**
早期 `autoScreenshot()` 使用 `lastScreenshotMode` 跟随 LLM 的显式 mode 选择。LLM 一旦切换到 pixel，所有后续 autoScreenshot 都变成 pixel，a11y 优势丧失。
**修复**：删除 `lastScreenshotMode`，autoScreenshot 始终传 `mode: "auto"`，每次由后端 `isTreeUseful()` 决定。

**已解决问题 2：annotate 与 a11y 冲突**
annotate 工具需要 pixel base64，但 a11y 模式下没有。LLM 被迫浪费 2-3 步切换到 pixel 再 annotate。
**修复**：移除 annotate 作为 LLM 工具。坐标操作（click/scroll/mouse_move）时自动生成带标注的截图保存到日志目录（`step-XX-click+annotate.png`），仅供开发者调试，不发给 LLM。a11y 模式下会临时取一次 pixel 截图仅用于标注，不发给 LLM（每次 ~200ms 开销）。

**当前行为**：
- a11y 模式下：LLM 从 bounds 直接算坐标，直接 click；自动取 pixel 截图生成标注
- pixel 模式下：LLM 从截图估算坐标，直接 click
- 日志目录：**每次**坐标操作都保存带十字线标注的截图（包括 a11y 模式下的点击）

### isTreeUseful 判断逻辑

`win-automation.ts` 中的 `isTreeUseful()` 在 `auto` 模式下决定返回 tree 还是回退到像素截图：

1. 至少包含 **3 个**元素行（含 `[SomeType]` 的行）
2. **交互元素检查**：如果存在 ≥3 个交互元素（Button/MenuItem/CheckBox/RadioButton/TabItem/ListItem/TreeItem/ComboBox），则至少有 **1 个**交互元素包含非空 `Name`。否则 tree 虽然存在但对交互完全无用
3. 至少有 **1 个**元素包含非空 `Name`（`Name="..."` 不为空）

不满足任一条件时，`auto` 模式会回退到像素截图。名称回退机制的引入使更多元素获得 Name，减少了不必要的回退。

`isTreeUseful` 返回诊断原因字符串，通过 `ScreenControlResult.a11yDiagnostics` 传到 agent-loop，记录到日志：
- `auto: tree accepted (37 elements (28 interactive) with names)` — tree 被接受
- `auto: tree rejected (only 2 elements), fell back to pixel` — 元素太少
- `auto: tree rejected (28 interactive elements but none has Name), fell back to pixel` — 交互元素全部匿名
- `auto: tree rejected (15 elements but none has Name), fell back to pixel` — 无命名元素

tree 超过 `MAX_NODES=500` 时，`window-accessibility.ps1` 会在输出末尾追加 `# WARNING: tree truncated at 500 nodes`。

### 调试 a11y tree

单独运行 `window-accessibility.ps1` 检查某个窗口的 tree 输出：

```powershell
# 先获取窗口句柄
powershell -NoProfile -ExecutionPolicy Bypass -File electron\scripts\window-list.ps1

# 用句柄运行 a11y tree
$env:HWND = "123456"
powershell -NoProfile -ExecutionPolicy Bypass -File electron\scripts\window-accessibility.ps1
```

输出第一行是 `left,top,WxH` 的窗口几何信息，第二行是 `Window: WxH`，后续为缩进的 tree。检查关键控件是否有 `Name` 和正确的 `ControlType`。

### 日志目录完整文件列表

运行一次 agent 后，`%TEMP%\windows-agent-standalone\logs\` 包含：

| 文件模式 | 来源 | 说明 |
|---------|------|------|
| `agent.log` | 全程 | 所有 `[agent]` 日志行 |
| `step-XX-<action>+auto_screenshot.png` | autoScreenshot pixel | 操作后的自动截图（pixel 模式） |
| `step-XX-<action>+annotate.png` | auto-annotate | 带红色十字标记的截图，每次坐标操作都有 |
| `step-XX-window_screenshot.png` | 显式截图 | LLM 请求的 pixel 截图 |
| `step-XX-<action>+auto_a11y.txt` | autoScreenshot a11y | 操作后自动获取的 a11y tree 文本 |
| `step-XX-window_screenshot+a11y.txt` | 显式截图 a11y | LLM 请求 window_screenshot 返回的 a11y tree |

### a11y 调试工作流

当 a11y 模式效果不佳（点击未命中、LLM 频繁切换到 pixel）时：

1. **查看 `agent.log`**：
   - `[agent] a11y: auto: tree accepted/rejected (reason)` — 确认 tree 是否被采用
   - `[agent] LLM text: ...` — 查看 LLM 为什么切换到 pixel 或做出某个决策
   - `step N LLM: Xms` — 监控 LLM 响应时间是否异常增长

2. **查看 a11y tree 文件** (`step-XX-*+a11y.txt`)：
   - 确认目标控件是否在 tree 中
   - 检查 bounds 是否合理
   - 检查 Name 是否正确
   - 检查末尾是否有 `# WARNING: tree truncated at 500 nodes`

3. **查看 annotate 截图** (`step-XX-*+annotate.png`)：
   - 红色十字是否在目标控件上
   - 如果偏移，可能是 bounds 解析错误或坐标转换问题

4. **对比 a11y tree 与实际 UI**：
   - 手动运行 `window-accessibility.ps1` 获取 tree
   - 将 tree 中的 bounds 与窗口截图对比

### 计算器测试经验

测试 "打开计算器，用鼠标点击操作1+2=" 时发现的问题和修复：

**第一轮问题：owner-drawn 控件无名称**
- 旧版计算器的 owner-drawn 按钮在 UIA 中 Name 为空且 ControlType 被误报为 Pane
- **修复**：添加 MSAA accName 回退 + ControlType 修正

**第二轮问题：UWP 计算器按钮无名称 + isTreeUseful 误判**
- Windows 10/11 UWP 计算器的 28 个按钮全部无 UIA Name，且 `NativeWindowHandle = 0`（虚拟 UIA 元素），MSAA 回退无法工作
- `isTreeUseful` 仅检查"至少 1 个元素有 Name"，Window 标题和显示区 Pane 满足条件 → tree 被误判为有用
- 后果：a11y/pixel 乒乓——autoScreenshot 返回 a11y tree → LLM 看不懂 → 请求 pixel → 点击 → 又收到 a11y → 每次操作浪费一步，20 步仅完成约 10 次有效操作
- **修复 1**：`isTreeUseful` 新增交互元素检查——存在 ≥3 个 Button 但没有一个有 Name → tree 被拒绝 → 直接用 pixel，无乒乓
- **修复 2**：名称回退链新增 AutomationId（优先级高于 GetWindowText 和 MSAA）——UWP 按钮通常有 `AutomationId`（如 `num7Button`）

**排查清单（当 a11y 效果不理想时）**：
1. 查看 `agent.log` 中 `[agent] a11y:` 行，确认 tree 是被 accepted 还是 rejected
2. 查看 `step-XX-*+a11y.txt`，检查目标控件是否有 Name
3. 如果控件无 Name，检查 stderr 的 `DEBUG: unnamed Button` 行：
   - `hwnd=0` → 虚拟元素，GetWindowText 和 MSAA 无法工作，依赖 AutomationId
   - `autoId=''` → 开发者未设置 AutomationId，所有回退都无法获取名称
   - `hwnd=非零, autoId=''` → MSAA 或 GetWindowText 应该能工作，检查是否有异常
4. 如果 tree 被 accepted 但 LLM 仍频繁切换 pixel，查看 `[agent] LLM text:` 了解原因
5. 确认 annotate 截图中红色十字位置是否正确
