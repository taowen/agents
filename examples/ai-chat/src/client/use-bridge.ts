import { useCallback, useEffect, useRef, useState } from "react";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import type { LanguageModel, ModelMessage } from "ai";
import { streamText, tool } from "ai";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { WindowsFsAdapter } from "./windows-fs-adapter";
import { z } from "zod";
import type { LlmConfig } from "../server/llm-proxy";

// ---- Types ----

interface ScreenControlParams {
  action: string;
  x?: number;
  y?: number;
  text?: string;
  key?: string;
  modifiers?: string[];
  button?: string;
  doubleClick?: boolean;
  direction?: string;
  amount?: number;
}

interface ScreenControlResult {
  success: boolean;
  error?: string;
  width?: number;
  height?: number;
  base64?: string;
  action?: string;
  [key: string]: unknown;
}

interface FsOpResult {
  ok: boolean;
  result?: unknown;
  error?: string;
  code?: string;
}

interface DriveInfo {
  mountPoint: string;
  root: string;
}

interface WorkWithWindows {
  ping: () => string;
  platform: string;
  screenControl: (params: ScreenControlParams) => Promise<ScreenControlResult>;
  fileSystem?: (params: Record<string, unknown>) => Promise<FsOpResult>;
  detectDrives?: () => Promise<DriveInfo[]>;
}

declare global {
  interface Window {
    workWithWindows?: WorkWithWindows;
  }
}

// ---- Local agent logic (runs in-browser, triggered by bridge messages) ----

const mountableFs = new MountableFs({ base: new InMemoryFs() });
const bash = new Bash({ fs: mountableFs, cwd: "/home" });

let windowsMountsInitialized = false;
let mountedDriveDescriptions: string[] = [];

async function initWindowsMounts() {
  if (windowsMountsInitialized) return;
  if (!window.workWithWindows?.detectDrives) return;
  windowsMountsInitialized = true;

  try {
    const drives = await window.workWithWindows.detectDrives();
    const ipcFn = window.workWithWindows.fileSystem!;
    for (const drive of drives) {
      const adapter = new WindowsFsAdapter(drive.root, ipcFn);
      mountableFs.mount(drive.mountPoint, adapter, "winfs");
      mountedDriveDescriptions.push(
        `${drive.mountPoint} (${drive.root.replace(/\\$/, "")})`
      );
    }
  } catch (err) {
    console.error("Failed to init Windows mounts:", err);
  }
}

const bashToolDef = tool({
  description: "Execute a bash command in the browser-side virtual filesystem",
  parameters: z.object({
    command: z.string().describe("The bash command to execute")
  })
});

const screenToolDef = tool({
  description:
    "Control the Windows desktop screen. Take screenshots, click, type, press keys, move mouse, and scroll. " +
    "Use 'screenshot' first to see what's on screen, then interact with elements by their pixel coordinates. " +
    "Coordinates are in physical pixels from top-left (0,0).",
  parameters: z.object({
    action: z
      .enum([
        "screenshot",
        "click",
        "mouse_move",
        "type",
        "key_press",
        "scroll"
      ])
      .describe("The screen action to perform"),
    x: z.number().optional().describe("X coordinate in pixels"),
    y: z.number().optional().describe("Y coordinate in pixels"),
    text: z.string().optional().describe("Text to type"),
    key: z.string().optional().describe("Key name to press"),
    modifiers: z.array(z.string()).optional().describe("Modifier keys"),
    button: z
      .enum(["left", "right", "middle"])
      .optional()
      .describe("Mouse button"),
    doubleClick: z.boolean().optional().describe("Double-click"),
    direction: z.enum(["up", "down"]).optional().describe("Scroll direction"),
    amount: z.number().optional().describe("Scroll amount")
  })
});

const windowToolDef = tool({
  description:
    "Manage windows on the Windows desktop. List visible windows, focus/activate, " +
    "move/resize, minimize/maximize/restore, or take a screenshot of a single window. " +
    "Use list_windows first to discover windows and their handles. " +
    "Use window_screenshot to capture just one window (smaller image than full desktop screenshot).",
  parameters: z.object({
    action: z.enum([
      "list_windows",
      "focus_window",
      "resize_window",
      "minimize_window",
      "maximize_window",
      "restore_window",
      "window_screenshot"
    ]),
    handle: z.number().optional().describe("Window handle from list_windows"),
    title: z
      .string()
      .optional()
      .describe("Window title substring (for focus_window)"),
    x: z.number().optional().describe("X position for resize_window"),
    y: z.number().optional().describe("Y position for resize_window"),
    width: z.number().optional().describe("Width for resize_window"),
    height: z.number().optional().describe("Height for resize_window")
  })
});

async function executeBash(command: string) {
  const result = await bash.exec(command);
  return {
    stdout: result.stdout,
    stderr: result.stderr,
    exitCode: result.exitCode
  };
}

async function executeScreen(
  input: ScreenControlParams
): Promise<ScreenControlResult> {
  if (!window.workWithWindows?.screenControl) {
    return {
      success: false,
      error: "Screen control is only available in the Electron desktop app",
      action: input.action
    };
  }
  const result = await window.workWithWindows.screenControl(input);
  return { ...result, action: input.action };
}

async function executeWindow(
  input: ScreenControlParams
): Promise<ScreenControlResult> {
  if (!window.workWithWindows?.screenControl) {
    return {
      success: false,
      error: "Window control is only available in the Electron desktop app",
      action: input.action
    };
  }
  const result = await window.workWithWindows.screenControl(input);
  return { ...result, action: input.action };
}

// Persistent local agent session — maintains conversation context across bridge messages
const agentHistory: ModelMessage[] = [];
let cachedModel: LanguageModel | null = null;

async function getModel(): Promise<LanguageModel> {
  if (cachedModel) return cachedModel;

  const resp = await fetch("/api/llm/config");
  if (!resp.ok) {
    throw new Error(`Failed to fetch LLM config: ${resp.status}`);
  }
  const config = (await resp.json()) as LlmConfig;

  if (config.provider === "google") {
    cachedModel = createGoogleGenerativeAI({
      baseURL: config.baseURL,
      apiKey: config.apiKey
    })(config.model);
  } else {
    cachedModel = createOpenAICompatible({
      name: "llm",
      baseURL: config.baseURL,
      apiKey: config.apiKey
    })(config.model);
  }

  return cachedModel;
}

/**
 * Run the local agent loop for a bridge message.
 * Appends the message to the persistent session, runs the agent, returns assistant text.
 */
async function runLocalAgent(
  userMessage: string,
  onLog?: (msg: string) => void
): Promise<string> {
  await initWindowsMounts();
  agentHistory.push({ role: "user", content: userMessage });

  const model = await getModel();
  const hasScreenControl = !!window.workWithWindows?.screenControl;

  const fsDescription =
    mountedDriveDescriptions.length > 0
      ? "You have access to a bash shell with the host file system mounted: " +
        mountedDriveDescriptions.join(", ") +
        ". You can ls, cat, grep, find files on the Windows machine. " +
        "File writes are real — they modify the host filesystem. Be cautious with rm and destructive operations. "
      : "You have access to a bash shell (virtual in-memory filesystem). ";

  const systemPrompt =
    "You are a remote desktop agent running on a Windows machine. " +
    "You receive instructions from a central AI assistant and execute them on the local desktop. " +
    fsDescription +
    (hasScreenControl
      ? "You also have the Windows desktop screen. You can take screenshots, click, type, press keys, move the mouse, and scroll. " +
        "You also have a 'win' tool for window management: list visible windows, focus/activate, " +
        "move/resize, minimize/maximize/restore, and take a screenshot of a single window. " +
        "Use win({ action: 'list_windows' }) to discover windows and their handles. " +
        "Use win({ action: 'window_screenshot', handle: ... }) to capture just one window (smaller than full desktop screenshot). " +
        "Always take a screenshot first to understand the current screen state before performing actions. " +
        "Use pixel coordinates from the screenshot to target UI elements. "
      : "") +
    "After completing the task, provide a concise text summary of what you did and the result. " +
    "Do NOT include base64 image data in your text response.";

  const tools = {
    bash: bashToolDef,
    screen: screenToolDef,
    win: windowToolDef
  };
  let finalText = "";

  for (let step = 0; step < 10; step++) {
    onLog?.(`[agent] step ${step + 1}...`);

    const result = streamText({
      model,
      system: systemPrompt,
      messages: agentHistory,
      tools
    });

    const toolCalls: Array<{
      toolCallId: string;
      toolName: string;
      args: Record<string, unknown>;
    }> = [];

    let stepText = "";
    for await (const event of result.fullStream) {
      if (event.type === "text-delta") {
        stepText += event.text;
      } else if (event.type === "tool-call") {
        const tcArgs = (event as any).input ?? (event as any).args ?? {};
        toolCalls.push({
          toolCallId: event.toolCallId,
          toolName: event.toolName,
          args: tcArgs as Record<string, unknown>
        });
        onLog?.(
          `[agent] tool: ${event.toolName}(${JSON.stringify(tcArgs).slice(0, 100)})`
        );
      }
    }

    const response = await result.response;
    agentHistory.push(...response.messages);

    if (stepText) finalText = stepText;

    if (toolCalls.length === 0) break;

    for (const tc of toolCalls) {
      let toolResultContent: string;

      if (tc.toolName === "bash") {
        const bashResult = await executeBash(
          (tc.args as { command: string }).command
        );
        onLog?.(
          `[agent] bash result: exit=${bashResult.exitCode} stdout=${bashResult.stdout.slice(0, 100)}`
        );
        toolResultContent = JSON.stringify(bashResult);
      } else if (tc.toolName === "screen") {
        const screenResult = await executeScreen(
          tc.args as ScreenControlParams
        );
        onLog?.(
          `[agent] screen: ${screenResult.action} success=${screenResult.success}`
        );

        const lines: string[] = [];
        lines.push(
          `Action: ${screenResult.action ?? "screenshot"} | Success: ${screenResult.success}`
        );
        if (screenResult.error) lines.push(`Error: ${screenResult.error}`);
        if (screenResult.width && screenResult.height)
          lines.push(`Screen: ${screenResult.width}x${screenResult.height}`);
        toolResultContent = lines.join("\n");

        if (screenResult.base64) {
          agentHistory.push({
            role: "tool" as const,
            content: [
              {
                type: "tool-result" as const,
                toolCallId: tc.toolCallId,
                toolName: tc.toolName,
                output: { type: "text", value: toolResultContent }
              }
            ]
          });
          agentHistory.push({
            role: "user",
            content: [
              { type: "text", text: "Here is the screenshot:" },
              {
                type: "image",
                image: screenResult.base64,
                mediaType: "image/png"
              }
            ]
          });
          continue;
        }
      } else if (tc.toolName === "win") {
        const winResult = await executeWindow(tc.args as ScreenControlParams);
        onLog?.(
          `[agent] win: ${winResult.action} success=${winResult.success}`
        );

        if (winResult.action === "list_windows") {
          const windows = (winResult as any).windows || [];
          toolResultContent = JSON.stringify(windows, null, 2);
        } else if (
          winResult.action === "window_screenshot" &&
          winResult.base64
        ) {
          const lines: string[] = [];
          lines.push(
            `Action: window_screenshot | Success: ${winResult.success}`
          );
          if (winResult.width && winResult.height)
            lines.push(`Window size: ${winResult.width}x${winResult.height}`);
          toolResultContent = lines.join("\n");

          agentHistory.push({
            role: "tool" as const,
            content: [
              {
                type: "tool-result" as const,
                toolCallId: tc.toolCallId,
                toolName: tc.toolName,
                output: { type: "text", value: toolResultContent }
              }
            ]
          });
          agentHistory.push({
            role: "user",
            content: [
              { type: "text", text: "Here is the window screenshot:" },
              {
                type: "image",
                image: winResult.base64,
                mediaType: "image/png"
              }
            ]
          });
          continue;
        } else {
          const lines: string[] = [];
          lines.push(
            `Action: ${winResult.action} | Success: ${winResult.success}`
          );
          if (winResult.error) lines.push(`Error: ${winResult.error}`);
          if ((winResult as any).message)
            lines.push((winResult as any).message);
          toolResultContent = lines.join("\n");
        }
      } else {
        toolResultContent = "Unknown tool";
      }

      agentHistory.push({
        role: "tool" as const,
        content: [
          {
            type: "tool-result" as const,
            toolCallId: tc.toolCallId,
            toolName: tc.toolName,
            output: { type: "text", value: toolResultContent }
          }
        ]
      });
    }
  }

  return finalText || "[Agent completed without text output]";
}

function resetLocalAgent() {
  agentHistory.length = 0;
  cachedModel = null;
}

// ---- Bridge WebSocket hook ----

export type BridgeStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "running";

export interface BridgeLog {
  time: string;
  message: string;
}

export function useBridge(deviceName: string) {
  const [status, setStatus] = useState<BridgeStatus>("disconnected");
  const [logs, setLogs] = useState<BridgeLog[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const addLog = useCallback((message: string) => {
    const time = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev.slice(-99), { time, message }]);
    // Also relay the log upstream to BridgeManager for viewer broadcast
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: "cf_agent_bridge_log", message })
      );
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(
      `${proto}//${location.host}/agents/bridge-manager/default`
    );
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "cf_agent_bridge_register",
          deviceName
        })
      );
      addLog(`Registering as "${deviceName}"...`);
    };

    ws.onmessage = async (event) => {
      let data: any;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }

      if (data.type === "cf_agent_bridge_registered") {
        setStatus("connected");
        addLog(`Registered as "${data.deviceName}"`);
        return;
      }

      if (data.type === "cf_agent_bridge_message") {
        setStatus("running");
        addLog(`Received task: ${data.content.slice(0, 100)}`);
        try {
          const response = await runLocalAgent(data.content, addLog);
          addLog(`Task complete. Response: ${response.slice(0, 100)}`);
          ws.send(
            JSON.stringify({
              type: "cf_agent_bridge_response",
              messageId: data.messageId,
              content: response
            })
          );
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          addLog(`Error: ${errMsg}`);
          ws.send(
            JSON.stringify({
              type: "cf_agent_bridge_response",
              messageId: data.messageId,
              content: `[Error] Local agent failed: ${errMsg}`
            })
          );
        }
        setStatus("connected");
        return;
      }
    };

    ws.onclose = () => {
      setStatus("disconnected");
      addLog("Disconnected. Reconnecting in 3s...");
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      addLog("WebSocket error");
    };
  }, [deviceName, addLog]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const resetAgent = useCallback(() => {
    resetLocalAgent();
    addLog("Local agent session reset");
  }, [addLog]);

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  return { status, logs, resetAgent, clearLogs };
}
