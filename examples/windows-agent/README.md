# Windows Agent

Electron desktop shell that loads the [AI Chat](../ai-chat/) web app at `https://ai.connect-screen.com/windows-agent`。本项目自身**不包含任何后端或 UI 代码**，所有功能由 `examples/ai-chat` 提供。

The `window.workWithWindows` preload bridge is the extensibility point for adding real desktop tools (desktopCapturer screenshots, child_process commands, etc.).

## 与 ai-chat 的关系

本项目是一个纯 Electron 壳，依赖 `examples/ai-chat` 部署在 `ai.connect-screen.com` 的 Cloudflare Worker 提供：

- **整个 UI** — Electron 窗口直接加载远程 URL `/windows-agent`，本地不打包任何前端资源
- **认证** — Google OAuth 登录流程由 ai-chat 的 `/auth/*` 路由处理
- **LLM 配置** — `WindowsAgent` 组件通过 `fetch("/api/llm/config")` 获取 API key 和模型信息，然后**从浏览器直接调用 LLM 提供商**（不经过服务端代理）
- **虚拟 Shell** — 使用 `just-bash` + `InMemoryFs` 在浏览器中运行 bash 工具

注意：ai-chat 的 `/windows-agent` 路由渲染的是独立的 `<WindowsAgent />` 组件（简化版单会话聊天 UI），**不使用** Durable Object WebSocket 连接。

```
examples/windows-agent/                    examples/ai-chat/
(Electron shell, 无后端)                   (Cloudflare Worker + SPA)
                                           部署在 ai.connect-screen.com
electron/main.js
  加载 URL ─────────────────────────────> /windows-agent (SPA 路由)
                                            │
electron/preload.cjs                        ├─ App.tsx: pathname==="/windows-agent"
  暴露 window.workWithWindows ───────────>  │  渲染 <WindowsAgent />
  (ping, platform)                          │
                                            ├─ /api/llm/config → 返回 LLM 配置
                                            ├─ /auth/*         → Google OAuth
                                            └─ /agents/*       → ChatAgent DO
                                                                 (WindowsAgent 不使用)
```

## 一键构建部署（从 WSL）

`run-on-windows.sh` 会依次完成以下四步：

1. **构建 just-bash** — ai-chat 依赖其 `dist/` 产物
2. **构建 workspace 包** — `agents`、`@cloudflare/ai-chat` 等库
3. **部署 ai-chat** — `vite build && wrangler deploy` 到 Cloudflare
4. **启动 Electron** — 通过 PowerShell 在 Windows 侧运行

```bash
bash examples/windows-agent/run-on-windows.sh
```

构建依赖链：

```
just-bash (npm run build)
    ↓
agents, @cloudflare/ai-chat, ... (npm run build at root)
    ↓
examples/ai-chat (vite build → wrangler deploy → ai.connect-screen.com)
    ↓
examples/windows-agent (Electron 加载远程 URL)
```

Electron 部分会将 `package.json` 和 `electron/` 复制到 Windows 临时目录（`%TEMP%\windows-agent-build`），避免 UNC 路径和 monorepo workspace 的兼容性问题。

## 仅启动 Electron（不重新部署）

如果 ai-chat 已经部署好，只想启动 Electron：

```bash
cd examples/windows-agent
npm install
npm run dev
```

## 本地联合开发（不部署，使用本地 ai-chat）

如果需要修改 ai-chat 的 UI 或后端逻辑，可以在本地运行 ai-chat 而不部署：

1. 在 `examples/ai-chat/` 中配置 `.dev.vars`（需要 `AUTH_SECRET`、`GOOGLE_CLIENT_ID`、`GOOGLE_CLIENT_SECRET`，Google OAuth 需将 `http://localhost:5173` 加入授权回调地址）

2. 启动 ai-chat 本地开发服务器：

```bash
cd examples/ai-chat
npm run start    # vite dev，启动在 http://localhost:5173
```

3. 将 windows-agent 指向本地服务器：

```bash
AGENT_URL=http://localhost:5173/windows-agent npm run dev
```

## 调试技巧

### 查看 renderer 日志

Electron renderer 进程的 `console.log` 无法直接从 WSL 终端看到。`main.js` 通过 `webContents.on("console-message")` 捕获 renderer 日志并写入文件：

```
%TEMP%\windows-agent-build\electron\renderer.log
```

从 WSL 读取：

```bash
cat "/mnt/c/Users/taowen/AppData/Local/Temp/windows-agent-build/electron/renderer.log"
```

### AI SDK v6 注意事项

- `fullStream` 中 `tool-call` 事件的参数字段名为 **`input`**（不是 v4 的 `args`）
- 事件完整字段：`type, toolCallId, toolName, input, providerExecuted, providerMetadata, title`
- 手动构造 `history` 消息时需严格匹配 `ModelMessage[]` schema，否则会报 `AI_InvalidPromptError`

### 手动 agentic 循环中的 history 构造

不使用 SDK 内置的多步执行（`stopWhen` / `maxSteps`）时，需要手动构造 tool result 消息。注意：
- `response.messages` 已包含 assistant 的 tool-call 消息，直接 push 到 history
- tool result 消息格式必须符合 AI SDK 的 `ModelMessage[]` schema
- 截图等图片不能放在 tool result 中（`@ai-sdk/openai-compatible` 会将整个对象 JSON.stringify 为文本），需作为 user message 注入
