# Android Accessibility Agent — 调试指南

## 1. 项目概览

Android 模式通过 **Accessibility Service** 控制手机，接收自然语言指令后调用 LLM 驱动 agent loop，自动操作手机界面（读取 accessibility tree、点击、滚动、输入文字）来完成任务。

### 架构层次

```
Host (WSL/Mac/Linux)                      Android 设备
┌──────────────────────┐                  ┌──────────────────────────────────┐
│ adb shell am broadcast│   ───adb───►   │ TaskReceiver (BroadcastReceiver) │
│   --es task '...'     │                 │   └─ new Thread → AgentLoop     │
│   --es api_url '...'  │                 │       ├─ LlmClient (HTTP)       │
│   --es api_key '...'  │                 │       ├─ JsEngine (Rhino)       │
│   --es model '...'    │                 │       │   └─ SelectToSpeakService│
└──────────────────────┘                  │       └─ log() → files/         │
                                          └──────────────────────────────────┘
```

### 核心文件

| 文件 | 职责 |
|------|------|
| `TaskReceiver.java` | BroadcastReceiver，接收 adb 广播，启动 AgentLoop 线程 |
| `AgentLoop.java` | Agent 循环核心：LLM 对话、tool 调用、文件日志 |
| `JsEngine.java` | Mozilla Rhino JS 引擎封装，注册全局函数供 LLM 生成的 JS 调用 |
| `LlmClient.java` | OpenAI 兼容 API 的 HTTP 客户端 |
| `SelectToSpeakService.java` | Accessibility Service，提供 getAccessibilityTree / click / scroll 等操作 |

## 2. JS 引擎架构（Rhino）

### 为什么引入 JS 引擎

旧架构下每个原子操作（click、get_screen）都需要一次 LLM 往返，计算器 1+2 需要 25 步。引入 Rhino JS 引擎后，LLM 可以在一次 tool call 中批量执行多个操作，大幅减少往返次数。

### 交互流程对比

```
旧: LLM → tool_call(click,"1") → result → LLM → tool_call(click,"+") → result → LLM → ...  (25步)
新: LLM → execute_js("click('1'); click({desc:'加'}); click('2'); click({desc:'等于'}); sleep(500); get_screen()") → 批量结果 → LLM  (6步)
```

### 实测效果：计算器 1+2 任务

| 指标 | 旧（多 tool） | 新（JS 引擎） |
|------|-------------|--------------|
| 总步数 | 25 | 6 |
| 计算过程 | 每个按钮 1 次 LLM 往返 | 4 个按钮 + 验证 = 1 次 LLM 往返 |
| 总耗时 | ~2 分钟 | ~25 秒 |
| 结果 | 正确 (3) | 正确 (3) |

实测日志（2025-02-19）：
```
[20:20:48] [TASK] Received task: Go to the home screen, find and open the Calculator app, compute 1+2, tell me the result.
[20:20:52] Step 1: press_home + sleep + get_screen (batched in one JS)
[20:20:56] Step 2: click("小工具") + get_screen
[20:21:01] Step 3: scroll("right") + get_screen
[20:21:05] Step 4: click("计算器") + get_screen
[20:21:09] Step 5: click("1") + click({desc:"加"}) + click("2") + click({desc:"等于"}) + sleep + get_screen (ALL batched!)
[20:21:13] [DONE] Task completed: The result of 1+2 is 3.
```

### Rhino 技术要点

- **依赖**: `org.mozilla:rhino:1.7.15`（纯 Java，~1.2MB JAR）
- **必须设置**: `cx.setOptimizationLevel(-1)` — Android 不支持 Rhino 字节码生成，需用解释模式
- **语言支持**: 仅 ES5 — 不支持箭头函数(=>)、模板字符串(`` ` ``)、let/const、解构、展开运算符。system prompt 中已强制说明
- **Scope 持久化**: `ScriptableObject scope` 在 JsEngine 实例内保持，`globalThis` 跨 execute_js 调用共享

### 超时保护（两层防护）

JsEngine 对每次 `execute_js` 调用设置 **30 秒超时**，采用两层防护：

**第一层：Rhino 指令观察器（TimeLimitContextFactory）**
- 自定义 `ContextFactory` 子类，设置 `setInstructionObserverThreshold(10_000)`
- 每 10,000 条字节码指令触发一次 `observeInstructionCount()`，检查 `System.currentTimeMillis() > deadline`
- 超时抛出 `Error`（不是 `Exception`），`execute()` 中用 `catch (Throwable)` 捕获
- **解决纯 JS 死循环**：`while(true){}` 这类不调用任何全局函数的代码也能被终止

**第二层：Thread.interrupt() watchdog（兜底）**
- `ScheduledExecutorService` 30 秒后 `Thread.interrupt()` 执行线程
- 每个全局函数入口调用 `checkTimeout()`，检测到中断则抛 RuntimeException
- `sleep()` 被中断时同样抛出超时错误
- **解决阻塞 I/O**：全局函数内的网络/文件操作可被中断

执行完毕后清除中断状态，不影响后续调用。超时后返回的错误格式：`[JS Error] Script execution timeout (30s)`

### get_screen 调用限制

每次 `execute_js` 调用中，`get_screen()` 最多调用 **5 次**（`MAX_GET_SCREEN_PER_EXEC = 5`）。超限时抛出 RuntimeException 并提示 LLM 返回结果后在下一次 execute_js 中继续。

这是为了防止 LLM 在单次 execute_js 中写出 `while` 循环反复 scroll + get_screen，导致产生大量快照（实测曾出现单步 140+ 快照、耗时 2 分钟的情况）。

只限制 get_screen，不限制 scroll/click — 后者开销小，且短序列（3 次 scroll + 1 次 get_screen）是合理用法。

### Screen 快照

每次调用 `get_screen()` 时，JsEngine 会自动将 accessibility tree 保存到 `files/screens/screen_NNN.txt`（NNN 从 001 递增）。每次新任务开始时目录会被清空、计数器重置。

查看快照：
```bash
adb shell run-as ai.connct_screen.com ls files/screens/
adb shell run-as ai.connct_screen.com cat files/screens/screen_001.txt
```

### JS 全局函数 API

LLM 通过唯一的 `execute_js` tool 执行 JavaScript，可用全局函数：

```javascript
// 屏幕信息
get_screen()              // → string (accessibility tree)

// 点击 - 三种模式
click("text")             // 匹配 text 或 contentDescription
click({desc: "加"})       // 仅匹配 contentDescription
click({x: 720, y: 2556}) // 坐标点击

// 长按 - 同上三种模式
long_click("text")
long_click({desc: "确定"})
long_click({x: 720, y: 2556})

// 滚动
scroll("up"|"down"|"left"|"right")

// 文本输入
type_text("hello")

// 导航
press_home()
press_back()
press_recents()           // 打开最近任务列表（多应用切换）
show_notifications()      // 下拉通知栏

// 工具
sleep(ms)                 // 等待 UI 稳定
log(message)              // 写入 agent_log.txt
```

### globalThis 持久化

`globalThis` 在多次 `execute_js` 调用间共享，LLM 可以存储上下文：

```javascript
// Step 1: 存储按钮坐标
let screen = get_screen();
globalThis.plusBtn = {x: 1236, y: 2556};
globalThis.eqBtn  = {x: 1236, y: 2900};

// Step 2: 后续直接用，不需要再解析 tree
click("1");
click(globalThis.plusBtn);
click("2");
click(globalThis.eqBtn);
```

## 3. LLM 配置

配置保存在 `android/llm-config.json`（已 gitignore）：

```json
{
  "provider": "openai-compatible",
  "baseURL": "https://your-api-endpoint/v3",
  "apiKey": "your-api-key",
  "model": "your-model"
}
```

发送任务时通过 `--es` 传入这些参数（见下文）。

## 4. 构建与安装

```bash
cd examples/ai-chat/android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

安装后需要在设备的 **设置 → 无障碍** 中启用 "Select to Speak" 服务。
重新安装 APK 后通常不需要重新启用，但如果服务不响应可以关闭再开启。

## 5. 发送任务

### adb shell 引号陷阱（重要！）

`adb shell` 传递含空格的字符串时，**必须用双引号包裹整个 shell 命令**，内部用单引号包裹参数值：

```bash
# 正确 — 双引号包裹整条命令，单引号包裹参数
adb shell "am broadcast -n ai.connct_screen.com/.TaskReceiver \
  --es task 'describe what you see on screen' \
  --es api_url 'https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions' \
  --es api_key 'your-key' \
  --es model 'doubao-seed-2.0-code'"

# 错误 — 外层单引号被本地 shell 消费，adb shell 拿到的是裸的多个 token
adb shell am broadcast -n ai.connct_screen.com/.TaskReceiver \
  --es task 'describe what you see on screen' \
  --es api_url '...' ...
```

错误写法的症状：broadcast intent 会显示 `pkg=what`（把 task 中的第二个单词当成了包名），任务参数被截断，agent 拿到的 task 只有第一个单词。

### 验证广播格式

正确的 broadcast 输出：
```
Broadcasting: Intent { flg=0x400000 cmp=ai.connct_screen.com/.TaskReceiver (has extras) }
```

错误的 broadcast 输出（注意多出的 `pkg=what`）：
```
Broadcasting: Intent { act=android.intent.action.MAIN cat=[android.intent.category.LAUNCHER] flg=0x400000 pkg=what cmp=ai.connct_screen.com/.TaskReceiver (has extras) }
```

## 6. 日志系统

### 为什么用文件日志而不是 logcat

Android 对每个进程有 logcat 输出频率限制（`chatty: uid=... identical N lines` 或 `LOGS OVER PROC QUOTA`）。Accessibility Service 在每个 UI 事件上触发，如果在 `onAccessibilityEvent` 中打印 accessibility tree 到 logcat，会迅速耗尽配额，导致后续 agent 的关键日志被静默丢弃。

当前方案：
- **Agent 日志** → 写入 `files/agent_log.txt`（每次任务开始时清空），同时 best-effort 写 logcat
- **onAccessibilityEvent** → 仅保存到内存 `logEntries`（供 MainActivity 显示），不写 logcat

### 读取日志

```bash
# Agent 完整日志（不受 logcat 限制）
adb shell run-as ai.connct_screen.com cat files/agent_log.txt
```

### 日志格式

```
[20:20:48] [TASK] Received task: Go to the home screen, find and open the Calculator app...
[20:20:48] [STEP 1] Calling LLM...
[20:20:52] [STEP 1] LLM returned tool_calls: execute_js
[20:20:52] [TOOL] execute_js -> [press_home] -> true
[sleep] 500ms
[get_screen] 21723 chars
[Script returned] ... (21826 chars)
[20:20:52] [STEP 2] Calling LLM...
...
[20:21:13] [DONE] Task completed: The result of 1+2 is 3.
```

关键标记：
- `[TASK]` — 任务开始
- `[STEP N]` — LLM 调用
- `[TOOL] execute_js` — JS 脚本执行，内含各函数调用日志
- `[DONE]` — 任务完成
- `[ERROR]` — 出错
- `[JS Error]` — JS 脚本执行出错

## 7. 故障排查

| 问题 | 排查方向 |
|------|---------|
| `agent_log.txt: No such file or directory` | Service 未运行或 broadcast 格式有误。检查 `adb shell settings get secure enabled_accessibility_services` |
| Agent 无响应，logcat 也没输出 | 先确认 `adb shell pidof ai.connct_screen.com` 有进程，再检查 broadcast 引号格式 |
| `LOGS OVER PROC QUOTA` | 正常——logcat 有限额，看 `files/agent_log.txt` 才是完整的 |
| LLM 返回错误 | 检查 `agent_log.txt` 中的 `[ERROR]`，常见：URL 拼错、API key 无效、model 名称错误 |
| 点击不生效 | 查看 get_screen 返回的 bounds 坐标，确认 accessibility tree 是否包含目标元素 |
| 安装后 service 不工作 | 去设置 → 无障碍中关闭再开启 Select to Speak 服务 |
| `run-as: Package not debuggable` | 确认安装的是 debug APK (`assembleDebug`)，不是 release |
| `[JS Error]` in log | Rhino JS 执行出错，检查 LLM 生成的 JS 代码语法（Rhino 仅支持 ES5 + 部分 ES6） |
| click("text") 匹配到错误元素 | `click("text")` 会同时匹配 text 和 desc。用 `click({desc:"X"})` 限定只匹配 contentDescription |

## 8. 实测记录

### 测试 1：计算器 1+2（简单任务）

**日期**: 2025-02-19 | **模型**: doubao-seed-2.0-code

| 指标 | 旧（多 tool） | 新（JS 引擎） |
|------|-------------|--------------|
| 总步数 | 25 | 6 |
| 总耗时 | ~2 分钟 | ~25 秒 |
| 结果 | 正确 (3) | 正确 (3) |

### 测试 2：设置→关于手机→读取设备信息（复杂任务）

**日期**: 2026-02-19 | **模型**: doubao-seed-2.0-code

**任务**: `打开设置，找到 关于手机 页面，告诉我手机的 设备名称 和 处理器 信息`

| 指标 | 值 |
|------|-----|
| 总步数 | 17（Step 1 语法错误 + 16 有效步） |
| 总耗时 | ~4 分 23 秒 |
| Screen 快照 | 199 个文件（保存正常） |
| 超时卡死 | 无 |
| 结果 | 成功：设备名称=一加 13，处理器=骁龙®8至尊版移动平台 |

**执行流程**:
```
Step 1:  [JS Error] missing ; before statement — LLM 生成了 Rhino 不支持的语法
Step 2:  press_home → click("设置") → get_screen → scroll → get_screen  ✓ 进入设置
Step 3-4: 在设置列表中反复滚动找"关于手机"，未找到（实际菜单名是"关于本机"）
Step 4:  尝试搜索框 click(desc="搜索编辑框") → false（搜索框在设置首页不可点击）
Step 5-6: 重新进入设置，点击搜索区域坐标 → type_text("关于手机") → 搜索成功
Step 7:  点击搜索结果"关于手机" → 进入了一个页面，但发现是"关于本机"
Step 8-9: 探索页面，遇到弹窗 → click("取消")
Step 10-14: 在设置各层级反复滚动探索，单步内大量 scroll+get_screen（产生 ~140 个快照）
Step 14: 尝试 系统 → 关于本机 路径，执行了 2 分钟的大量操作
Step 15-16: 回到正确路径，系统 → 关于本机
Step 17: [DONE] 成功读取设备名称和处理器
```

**发现的问题与改进方向**:

| 问题 | 影响 | 改进方向 |
|------|------|---------|
| LLM 首次生成 ES6 语法 | 浪费 1 步 | system prompt 中强调 ES5 only |
| 单步内循环 scroll+get_screen | 199 个快照，步骤 14 耗时 2 分钟 | 限制单次 execute_js 内 get_screen 调用次数，或在 system prompt 中提示避免循环 |
| 任务说"关于手机"但实际菜单是"关于本机" | agent 找不到精确匹配 | 这是 LLM 理解能力问题，不是工具问题 |
| 搜索功能发现但使用不顺 | 先 click desc 失败，后用坐标成功 | 设置搜索框的 accessibility 属性可能需要先点击才出现 |

**Screen 快照验证**:
```bash
adb shell run-as ai.connct_screen.com ls files/screens/ | wc -l  # 199 个文件
adb shell run-as ai.connct_screen.com cat files/screens/screen_199.txt  # 关于本机页面，包含完整硬件信息
```

screen_199.txt 内容摘要（关于本机页面 accessibility tree）：
```
[TextView] text="关于本机"
[TextView] text="一加 13"
[TextView] text="16.0.3"
[TextView] text="处理器"     → "骁龙®8至尊版移动平台"
[TextView] text="电池"       → "6000 mAh (典型值)"
[TextView] text="运行内存"   → "24.0 GB + 12.0 GB"
[TextView] text="存储空间"   → "540 GB / 1.00 TB"
```

### 测试 3：设置→关于手机（改进后重测）

**日期**: 2026-02-19 | **模型**: doubao-seed-2.0-code

**任务**: 同测试 2 — `打开设置，找到 关于手机 页面，告诉我手机的 设备名称 和 处理器 信息`

**改进措施**（本次测试前实施）:
1. **TimeLimitContextFactory** — Rhino 指令观察器防纯 JS 死循环
2. **get_screen 调用限制** — 每次 execute_js 最多 5 次
3. **System prompt 改进** — ES5 语法警告、反循环指导、设置搜索提示

**改进前后对比**:

| 指标 | 测试 2（改进前） | 测试 3（改进后） | 提升 |
|------|----------------|----------------|------|
| Step 1 语法错误 | 有（ES6 → Rhino 报错） | 无 | 消除 |
| 总步数 | 17 | 8 | 53% |
| Screen 快照 | 199 | 7 | 96% |
| 总耗时 | ~4 分 23 秒 | ~34 秒 | 87% |
| 结果 | 成功 | 成功 | — |

**执行流程**:
```
[20:52:23] Step 1: press_home + click("设置") + get_screen            → 进入设置
[20:52:36] Step 2: scroll("down") + get_screen                        → 未找到关于手机
[20:52:39] Step 3: scroll("down") + get_screen                        → 仍未找到
[20:52:42] Step 4: click(desc="搜索编辑框") + type_text("关于手机") + get_screen  → 搜索成功！
[20:52:46] Step 5: click("关于本机") + get_screen                     → 进入了结果页
[20:52:51] Step 6: click coords(720,759) + get_screen                 → 导航中
[20:52:53] Step 7: click coords(256,918) + get_screen                 → 到达关于本机页面
[20:52:57] Step 8: [DONE] 设备名称=一加 13，处理器=骁龙®8至尊版移动平台
```

**关键改进效果分析**:

1. **ES5 警告生效** — Step 1 直接使用了 ES5 语法（`var`、`function(){}`），没有语法错误
2. **反循环指导生效** — 每步只做 1-2 个操作 + 1 次 get_screen，不再出现单步内循环
3. **设置搜索提示生效** — 滚动 2 次未找到后，Step 4 主动使用搜索功能（改进前滚动了 10+ 次才尝试搜索）
4. **快照数从 199 降到 7** — 完全在 get_screen 限制（5 次/步）以内，实际未触发限制

### 测试 4：时钟闹钟（launch_app + scroll_element 验证）

**日期**: 2026-02-19 | **模型**: doubao-seed-2.0-code

**任务**: `打开时钟应用，创建一个7:30的闹钟，然后返回桌面`

**背景**: 新增 `launch_app()`、`list_apps()`、`scroll_element()` 三个 JS 全局函数。首次测试时发现 agent 从未调用这些函数，仍用老方法导航桌面 10 步才打开时钟——根因是 `adb install -r` 不会重启 accessibility service 进程，旧代码一直在运行。`force-stop` 后重新启用服务，新代码立即生效。

| 指标 | 旧代码（测试 3 同期） | 新代码（本次） |
|------|---------------------|-------------|
| 打开时钟 | 10+ 步导航桌面 | 1 步 `launch_app("时钟")` |
| 调整时间 | 反复 `scroll up/down` | `scroll_element` 精准控制 NumberPicker |
| 总步数 | 27 | **9** |
| 总耗时 | ~3 分钟 | **~58 秒** |
| Screen 快照 | 24 | **8** |
| 结果 | 成功 | 成功 |

**执行流程**:
```
[21:41:20] Step 1: launch_app("时钟") → Launched 时钟 (com.coloros.alarmclock) + get_screen
[21:41:29] Step 2: click(desc="添加闹钟") + get_screen
[21:41:34] Step 3: scroll_element("21时", "up") + get_screen → 小时开始调整
[21:41:38] Step 4: 继续 scroll_element 小时（20→...→8时）
[21:41:45] Step 5: 继续 scroll_element 小时（13→...→7时）+ get_screen 确认
[21:41:51] Step 6: scroll_element("41分", "down") → 分钟调整开始（部分 "No scrollable element found"）
[21:41:59] Step 7: scroll_element 分钟（40→...→30分）+ get_screen 确认
[21:42:05] Step 8: click("完成") + press_home + get_screen
[21:42:18] [DONE] 成功创建 7:30 闹钟并返回桌面
```

**发现的问题**:

| 问题 | 影响 | 改进方向 |
|------|------|---------|
| `scroll_element` 找不到滚动后的新文本 | Step 6 中 `"40分"` 找不到（滚动后文本已变） | LLM 需要先 get_screen 再用新文本；或改为按 index/bounds 定位 |
| 小时调整步数较多 | 从 21 到 7 需要 14 次 scroll | 考虑支持直接设置值，或一次滚动多步 |

## 9. 关键教训

### APK 热更新不等于代码生效

Android accessibility service 是长驻进程，`adb install -r` **只替换磁盘上的 APK 文件，不会重启正在运行的进程**。每次安装新 APK 后必须重启进程：

```bash
# 方法 1：force-stop（推荐）
adb shell am force-stop ai.connct_screen.com
# 然后在设置 → 无障碍中重新启用服务

# 方法 2：如果 force-stop 后服务自动重启（取决于设备）
# 无需额外操作
```

验证新代码已加载：
```bash
adb logcat -d -s A11yAgent:D | grep "connected"
# 应看到: SelectToSpeakService connected - agent ready
```

### Force-stop 后广播的注意事项

`force-stop` 会让 app 进入 "stopped state"，此时**隐式广播**（`-a action`）不会被接收。必须使用**显式组件名**（`-n`）发送广播：

```bash
# 正确 — 显式组件名
adb shell "am broadcast -n ai.connct_screen.com/.TaskReceiver --es task '...' ..."

# 错误 — 隐式 action，force-stop 后不生效
adb shell am broadcast -a ai.connct_screen.com.EXECUTE_TASK --es task '...' ...
```

## 10. 完整调试流程示例

```bash
# 1. 构建安装
cd examples/ai-chat/android && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 2. 重启进程（关键！）
adb shell am force-stop ai.connct_screen.com
# 然后在设置 → 无障碍中重新启用 Select to Speak 服务

# 3. 确认 service 在运行
adb shell settings get secure enabled_accessibility_services
# 应包含: ai.connct_screen.com/com.google.android.accessibility.selecttospeak.SelectToSpeakService

# 4. 发送任务（注意引号！用 -n 显式组件名！）
adb shell "am broadcast -n ai.connct_screen.com/.TaskReceiver \
  --es task 'Go to the home screen, find and open the Calculator app, compute 1+2, tell me the result.' \
  --es api_url 'https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions' \
  --es api_key 'your-key' \
  --es model 'doubao-seed-2.0-code'"

# 5. 等待完成后读日志
adb shell run-as ai.connct_screen.com cat files/agent_log.txt

# 6. 查看 screen 快照
adb shell run-as ai.connct_screen.com ls files/screens/

# 7. 验证日志完整性：应包含 [TASK] [STEP] [TOOL] [DONE]
```
