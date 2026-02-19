import { useCallback, useEffect, useRef, useState } from "react";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import type { LanguageModel } from "ai";
import type { LlmConfig } from "../server/llm-proxy";
import type {
  ScreenControlParams,
  ScreenControlResult,
  BashResult
} from "../shared/screen-control-types";
import { createAgentLoop } from "../shared/agent-loop";
import type { AgentLoop } from "../shared/agent-loop";

// ---- Types ----

interface DebugTaskData {
  taskId: string;
  prompt: string;
}

interface WorkWithWindows {
  ping: () => string;
  platform: string;
  screenControl: (params: ScreenControlParams) => Promise<ScreenControlResult>;
  executePowerShell?: (params: { command: string }) => Promise<BashResult>;
  onDebugTask?: (callback: (data: DebugTaskData) => void) => void;
  sendDebugResult?: (data: {
    taskId: string;
    response?: string;
    error?: string;
  }) => void;
}

declare global {
  interface Window {
    workWithWindows?: WorkWithWindows;
  }
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
  agentInstance = createAgentLoop({
    getModel,
    executePowerShell,
    executeScreenControl,
    hasCloudDrive: true,
    maxSteps: 10
  });
  return agentInstance;
}

async function runLocalAgent(
  userMessage: string,
  onLog?: (msg: string) => void,
  abortSignal?: AbortSignal
): Promise<string> {
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
    console.log(`[bridge] ${message}`);
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

  // Listen for debug tasks from the Electron main process (HTTP debug server)
  useEffect(() => {
    if (!window.workWithWindows?.onDebugTask) return;
    window.workWithWindows.onDebugTask(async (data) => {
      addLog(
        `[debug] Received task ${data.taskId}: ${data.prompt.slice(0, 100)}`
      );
      try {
        const response = await runLocalAgent(data.prompt, addLog);
        addLog(`[debug] Task ${data.taskId} complete`);
        window.workWithWindows?.sendDebugResult?.({
          taskId: data.taskId,
          response
        });
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        addLog(`[debug] Task ${data.taskId} error: ${errMsg}`);
        window.workWithWindows?.sendDebugResult?.({
          taskId: data.taskId,
          error: errMsg
        });
      }
    });
  }, [addLog]);

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
