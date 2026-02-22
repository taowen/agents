import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Linking,
  AppState,
  DeviceEventEmitter
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import AccessibilityBridge from "./NativeAccessibilityBridge";
import type { LlmConfig } from "./types";

const STORAGE_KEY = "llm_config";
const SERVER_URL = "https://ai.connect-screen.com";
const DEVICE_POLL_INTERVAL = 2000;

function App(): React.JSX.Element {
  const [task, setTask] = useState("");
  const [logs, setLogs] = useState<string[]>([]);
  const [serviceRunning, setServiceRunning] = useState(false);
  const [config, setConfig] = useState<LlmConfig>({
    baseURL: "",
    apiKey: "",
    model: ""
  });
  const [configLoaded, setConfigLoaded] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [deviceCode, setDeviceCode] = useState<string | null>(null);
  const [loginStatus, setLoginStatus] = useState<
    "idle" | "polling" | "approved" | "error"
  >("idle");

  const [cloudConnected, setCloudConnected] = useState(false);

  const scrollRef = useRef<ScrollView>(null);
  const devicePollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isRunningRef = useRef(false);

  const checkService = useCallback(async () => {
    try {
      const running = await AccessibilityBridge.isServiceRunning();
      setServiceRunning(running);
    } catch {
      setServiceRunning(false);
    }
  }, []);

  // Load config on mount — only trust AsyncStorage (written after device login)
  useEffect(() => {
    (async () => {
      try {
        const stored = await AsyncStorage.getItem(STORAGE_KEY);
        if (stored) {
          setConfig(JSON.parse(stored));
          setConfigLoaded(true);
        }
      } catch {}
    })();
  }, []);

  // Check service status on mount and when app comes to foreground
  useEffect(() => {
    checkService();
    const sub = AppState.addEventListener("change", (state) => {
      if (state === "active") checkService();
    });
    return () => sub.remove();
  }, [checkService]);

  // Cloud connection via Java DeviceConnection (OkHttp WebSocket)
  // Handles both LLM proxying and task dispatch
  useEffect(() => {
    if (!configLoaded || !config.apiKey) return;

    const name = AccessibilityBridge.getDeviceName();
    const wsUrl = `wss://ai.connect-screen.com/agents/chat-agent/device-${encodeURIComponent(name)}/device-connect?token=${encodeURIComponent(config.apiKey)}`;
    AccessibilityBridge.connectCloud(wsUrl, name);

    // Listen for task events pushed from Java DeviceConnection
    const sub = DeviceEventEmitter.addListener("DeviceTask", async (data) => {
      if (isRunningRef.current) {
        AccessibilityBridge.sendTaskResult(data.taskId, "Device busy", false);
        return;
      }
      setLogs((prev) => [
        ...prev,
        `[${formatTime()}] [CLOUD] Task: ${data.description}`
      ]);
      setIsRunning(true);
      isRunningRef.current = true;
      try {
        const configJson = JSON.stringify(config);
        const agentResult = await AccessibilityBridge.runAgentTask(
          data.description,
          configJson
        );
        AccessibilityBridge.sendTaskResult(
          data.taskId,
          agentResult || "done",
          true
        );
        setLogs((prev) => [...prev, `[${formatTime()}] [CLOUD] Done`]);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        AccessibilityBridge.sendTaskResult(
          data.taskId,
          msg || "Task failed",
          false
        );
        setLogs((prev) => [...prev, `[${formatTime()}] [CLOUD] Error: ${msg}`]);
      } finally {
        setIsRunning(false);
        isRunningRef.current = false;
      }
    });

    return () => {
      sub.remove();
      AccessibilityBridge.disconnectCloud();
      setCloudConnected(false);
    };
  }, [configLoaded, config.apiKey]);

  // Listen for real WebSocket connection status from Java DeviceConnection
  useEffect(() => {
    setCloudConnected(AccessibilityBridge.isCloudConnected());
    const sub = DeviceEventEmitter.addListener(
      "DeviceConnectionStatus",
      (data: { connected: boolean }) => {
        setCloudConnected(data.connected);
      }
    );
    return () => sub.remove();
  }, []);

  const stopDevicePoll = useCallback(() => {
    if (devicePollRef.current) {
      clearInterval(devicePollRef.current);
      devicePollRef.current = null;
    }
  }, []);

  const startDeviceLogin = useCallback(async () => {
    setLoginStatus("idle");
    setDeviceCode(null);
    try {
      const res = await fetch(`${SERVER_URL}/auth/device/start`, {
        method: "POST"
      });
      const data = (await res.json()) as { code: string };
      setDeviceCode(data.code);
      setLoginStatus("polling");

      // Poll for approval
      devicePollRef.current = setInterval(async () => {
        try {
          const checkRes = await fetch(
            `${SERVER_URL}/auth/device/check?code=${data.code}`
          );
          const checkData = (await checkRes.json()) as {
            status: string;
            token?: string;
            baseURL?: string;
            model?: string;
          };
          if (checkData.status === "approved" && checkData.token) {
            stopDevicePoll();
            const newConfig: LlmConfig = {
              baseURL: checkData.baseURL || "",
              apiKey: checkData.token,
              model: checkData.model || ""
            };
            setConfig(newConfig);
            await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(newConfig));
            await AccessibilityBridge.saveConfig(
              newConfig.baseURL,
              newConfig.apiKey,
              newConfig.model
            );
            setConfigLoaded(true);
            setLoginStatus("approved");
            setDeviceCode(null);
          } else if (checkData.status === "expired") {
            stopDevicePoll();
            setLoginStatus("error");
          }
        } catch {}
      }, DEVICE_POLL_INTERVAL);
    } catch {
      setLoginStatus("error");
    }
  }, [stopDevicePoll]);

  const cancelDeviceLogin = useCallback(() => {
    stopDevicePoll();
    setDeviceCode(null);
    setLoginStatus("idle");
  }, [stopDevicePoll]);

  const addLog = useCallback((msg: string) => {
    setLogs((prev) => [...prev, msg]);
  }, []);

  const handleSend = useCallback(async () => {
    const trimmed = task.trim();
    if (!trimmed || !configLoaded || isRunning) return;

    setLogs([]);
    AccessibilityBridge.clearLogFile();
    setTask("");
    setIsRunning(true);
    isRunningRef.current = true;

    try {
      const configJson = JSON.stringify(config);
      addLog(`[${formatTime()}] [TASK] Starting: ${trimmed}`);
      await AccessibilityBridge.runAgentTask(trimmed, configJson);
      addLog(`[${formatTime()}] [DONE] Agent finished`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      addLog(`[${formatTime()}] [ERROR] ${msg}`);
    } finally {
      setIsRunning(false);
      isRunningRef.current = false;
    }
  }, [task, config, configLoaded, isRunning, addLog]);

  if (!configLoaded) {
    // Login screen
    return (
      <View style={styles.container}>
        <View style={styles.loginScreen}>
          <Text style={styles.loginTitle}>RN Agent</Text>

          {loginStatus === "polling" && deviceCode ? (
            <View style={styles.deviceCodePanel}>
              <Text style={styles.deviceCodeLabel}>
                Open ai.connect-screen.com/device and enter:
              </Text>
              <Text style={styles.deviceCodeText}>{deviceCode}</Text>
              <Text style={styles.deviceCodeHint}>Waiting for approval...</Text>
              <TouchableOpacity onPress={cancelDeviceLogin}>
                <Text style={styles.cancelText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          ) : loginStatus === "error" ? (
            <View style={styles.deviceCodePanel}>
              <Text style={styles.errorText}>
                Code expired or failed. Try again.
              </Text>
              <TouchableOpacity onPress={startDeviceLogin}>
                <Text style={styles.retryText}>Retry</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <TouchableOpacity
              style={styles.loginBtn}
              onPress={startDeviceLogin}
            >
              <Text style={styles.loginBtnText}>Login</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    );
  }

  // Task screen (logged in)
  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>RN Agent</Text>
        <View style={styles.headerRight}>
          <TouchableOpacity
            style={styles.headerRight}
            onPress={() => {
              if (!cloudConnected) {
                addLog(`[${formatTime()}] [CLOUD] Reconnecting...`);
                AccessibilityBridge.reconnectCloud();
              }
            }}
          >
            <View
              style={[
                styles.statusDot,
                { backgroundColor: cloudConnected ? "#2196F3" : "#666" }
              ]}
            />
            <Text style={styles.statusText}>
              {cloudConnected ? "Cloud" : "Offline"}
            </Text>
          </TouchableOpacity>
          <View
            style={[
              styles.statusDot,
              {
                backgroundColor: serviceRunning ? "#4CAF50" : "#F44336",
                marginLeft: 8
              }
            ]}
          />
          <Text style={styles.statusText}>
            {serviceRunning ? "Service" : "No Svc"}
          </Text>
        </View>
      </View>

      {/* Buttons row */}
      <View style={styles.buttonRow}>
        <TouchableOpacity
          style={styles.smallBtn}
          onPress={() =>
            Linking.sendIntent("android.settings.ACCESSIBILITY_SETTINGS")
          }
        >
          <Text style={styles.smallBtnText}>Accessibility</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.smallBtn}
          onPress={() => {
            checkService();
          }}
        >
          <Text style={styles.smallBtnText}>Refresh</Text>
        </TouchableOpacity>
      </View>

      {/* Task input */}
      <View style={styles.inputRow}>
        <TextInput
          style={styles.taskInput}
          placeholder="Enter task..."
          placeholderTextColor="#999"
          value={task}
          onChangeText={setTask}
          multiline
          editable={!isRunning}
        />
        {isRunning ? (
          <TouchableOpacity style={styles.abortBtn} disabled>
            <Text style={styles.sendBtnText}>Running</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={[styles.sendBtn, !task.trim() && styles.disabledBtn]}
            onPress={handleSend}
            disabled={!task.trim()}
          >
            <Text style={styles.sendBtnText}>Send</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Logs */}
      <ScrollView
        ref={scrollRef}
        style={styles.logScroll}
        onContentSizeChange={() =>
          scrollRef.current?.scrollToEnd({ animated: true })
        }
      >
        {logs.length === 0 ? (
          <Text style={styles.logPlaceholder}>Enter a task and tap Send</Text>
        ) : (
          logs.map((line, i) => (
            <Text key={i} style={styles.logLine}>
              {line}
            </Text>
          ))
        )}
      </ScrollView>
    </View>
  );
}

function formatTime(): string {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#1a1a2e",
    padding: 12,
    paddingTop: 40
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8
  },
  title: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#eee"
  },
  headerRight: {
    flexDirection: "row",
    alignItems: "center"
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 6
  },
  statusText: {
    color: "#ccc",
    fontSize: 13
  },
  buttonRow: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 8
  },
  smallBtn: {
    backgroundColor: "#16213e",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: "#0f3460"
  },
  smallBtnText: {
    color: "#e0e0e0",
    fontSize: 13
  },
  inputRow: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 8
  },
  taskInput: {
    flex: 1,
    backgroundColor: "#16213e",
    color: "#eee",
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    maxHeight: 80,
    borderWidth: 1,
    borderColor: "#0f3460"
  },
  sendBtn: {
    backgroundColor: "#e94560",
    borderRadius: 8,
    paddingHorizontal: 16,
    justifyContent: "center"
  },
  abortBtn: {
    backgroundColor: "#666",
    borderRadius: 8,
    paddingHorizontal: 16,
    justifyContent: "center"
  },
  disabledBtn: {
    opacity: 0.4
  },
  sendBtnText: {
    color: "#fff",
    fontWeight: "bold",
    fontSize: 15
  },
  logScroll: {
    flex: 1,
    backgroundColor: "#0a0a1a",
    borderRadius: 8,
    padding: 8
  },
  loginScreen: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center"
  },
  loginTitle: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#eee",
    marginBottom: 40
  },
  loginBtn: {
    backgroundColor: "#e94560",
    borderRadius: 8,
    paddingHorizontal: 40,
    paddingVertical: 14
  },
  loginBtnText: {
    color: "#fff",
    fontWeight: "bold",
    fontSize: 18
  },
  deviceCodePanel: {
    backgroundColor: "#16213e",
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: "#0f3460",
    alignItems: "center"
  },
  deviceCodeLabel: {
    color: "#ccc",
    fontSize: 12,
    marginBottom: 8,
    textAlign: "center"
  },
  deviceCodeText: {
    color: "#fff",
    fontSize: 28,
    fontWeight: "bold",
    fontFamily: "monospace",
    letterSpacing: 6,
    marginBottom: 8
  },
  deviceCodeHint: {
    color: "#999",
    fontSize: 12,
    marginBottom: 4
  },
  cancelText: {
    color: "#e94560",
    fontSize: 13,
    marginTop: 4
  },
  approvedText: {
    color: "#4CAF50",
    fontSize: 14,
    fontWeight: "bold"
  },
  errorText: {
    color: "#F44336",
    fontSize: 13,
    marginBottom: 4
  },
  retryText: {
    color: "#e94560",
    fontSize: 13,
    fontWeight: "bold"
  },
  logPlaceholder: {
    color: "#666",
    fontStyle: "italic",
    textAlign: "center",
    marginTop: 20
  },
  logLine: {
    color: "#b0b0b0",
    fontFamily: "monospace",
    fontSize: 11,
    lineHeight: 16
  }
});

export default App;
