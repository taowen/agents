import { useCallback, useEffect, useRef, useState } from "react";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import type { LanguageModel } from "ai";
import { Bash, InMemoryFs, MountableFs, defineCommand } from "just-bash";
import { WindowsFsAdapter } from "./windows-fs-adapter";
import { HttpFsAdapter } from "./http-fs-adapter";
import type { LlmConfig } from "../server/llm-proxy";
import type {
  ScreenControlParams,
  ScreenControlResult,
  BashResult
} from "../shared/screen-control-types";
import { createAgentLoop } from "../shared/agent-loop";
import type { AgentLoop } from "../shared/agent-loop";

// ---- Types ----

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
  executePowerShell?: (params: { command: string }) => Promise<BashResult>;
  fileSystem?: (params: Record<string, unknown>) => Promise<FsOpResult>;
  detectDrives?: () => Promise<DriveInfo[]>;
}

declare global {
  interface Window {
    workWithWindows?: WorkWithWindows;
  }
}

// ---- Filesystem setup ----

const mountableFs = new MountableFs({ base: new InMemoryFs() });

const mountCmd = defineCommand("mount", async (args) => {
  if (args.length > 0) {
    return {
      stdout: "",
      stderr: "mount: read-only (use the cloud agent to manage mounts)\n",
      exitCode: 1
    };
  }
  const mounts = mountableFs.getMounts();
  const lines = mounts.map(
    (m) => `${m.fsType || "unknown"} on ${m.mountPoint}`
  );
  return {
    stdout: lines.length ? lines.join("\n") + "\n" : "",
    stderr: "",
    exitCode: 0
  };
});

const bash = new Bash({
  fs: mountableFs,
  cwd: "/home",
  customCommands: [mountCmd]
});

let windowsMountsInitialized = false;

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
    }
  } catch (err) {
    console.error("Failed to init Windows mounts:", err);
  }
}

let cloudMountInitialized = false;

function initCloudMount() {
  if (cloudMountInitialized) return;
  cloudMountInitialized = true;

  const adapter = new HttpFsAdapter();
  mountableFs.mount("/cloud", adapter, "http");
}

// ---- Model resolution ----

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

// ---- Screen control bridge ----

async function executeScreenControl(
  params: ScreenControlParams
): Promise<ScreenControlResult> {
  if (!window.workWithWindows?.screenControl) {
    return {
      success: false,
      error: "Screen control is only available in the Electron desktop app"
    };
  }
  return window.workWithWindows.screenControl(params);
}

// ---- PowerShell bridge ----

async function executePowerShell(command: string): Promise<BashResult> {
  if (!window.workWithWindows?.executePowerShell) {
    return {
      stdout: "",
      stderr: "PowerShell is only available in the Electron desktop app",
      exitCode: 1
    };
  }
  return window.workWithWindows.executePowerShell({ command });
}

// ---- Agent instance ----

let agentInstance: AgentLoop | null = null;

function getOrCreateAgent(): AgentLoop {
  if (agentInstance) return agentInstance;
  const hasPowerShell = !!window.workWithWindows?.executePowerShell;
  agentInstance = createAgentLoop({
    getModel,
    executeBash: (cmd) => bash.exec(cmd),
    executeScreenControl,
    ...(hasPowerShell ? { executePowerShell } : {}),
    maxSteps: 10
  });
  return agentInstance;
}

async function runLocalAgent(
  userMessage: string,
  onLog?: (msg: string) => void,
  abortSignal?: AbortSignal
): Promise<string> {
  await initWindowsMounts();
  initCloudMount();
  const agent = getOrCreateAgent();
  return agent.runAgent(userMessage, { onLog, abortSignal });
}

function resetLocalAgent() {
  if (agentInstance) {
    agentInstance.reset();
  }
  agentInstance = null;
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
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined
  );
  const agentAbortRef = useRef<AbortController | null>(null);

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
        const abortController = new AbortController();
        agentAbortRef.current = abortController;
        try {
          const response = await runLocalAgent(
            data.content,
            addLog,
            abortController.signal
          );
          addLog(`Task complete. Response: ${response.slice(0, 100)}`);
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(
              JSON.stringify({
                type: "cf_agent_bridge_response",
                messageId: data.messageId,
                content: response
              })
            );
          }
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          addLog(`Error: ${errMsg}`);
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(
              JSON.stringify({
                type: "cf_agent_bridge_response",
                messageId: data.messageId,
                content: `[Error] Local agent failed: ${errMsg}`
              })
            );
          }
        } finally {
          agentAbortRef.current = null;
        }
        setStatus("connected");
        return;
      }
    };

    ws.onclose = () => {
      // Abort any running agent when the connection drops
      if (agentAbortRef.current) {
        agentAbortRef.current.abort();
        agentAbortRef.current = null;
        addLog("Agent aborted (connection closed)");
      }
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
