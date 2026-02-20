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
import { AgentLoop } from "./AgentLoop";
import type { LlmConfig } from "./types";

const STORAGE_KEY = "llm_config";

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
  const [showConfig, setShowConfig] = useState(false);

  const agentRef = useRef(new AgentLoop());
  const scrollRef = useRef<ScrollView>(null);

  const checkService = useCallback(async () => {
    try {
      const running = await AccessibilityBridge.isServiceRunning();
      setServiceRunning(running);
    } catch {
      setServiceRunning(false);
    }
  }, []);

  // Load config on mount
  useEffect(() => {
    (async () => {
      try {
        const stored = await AsyncStorage.getItem(STORAGE_KEY);
        if (stored) {
          setConfig(JSON.parse(stored));
          setConfigLoaded(true);
          return;
        }
      } catch {}

      // Fallback to bundled asset
      try {
        const assetJson = await AccessibilityBridge.readAssetConfig();
        const parsed = JSON.parse(assetJson);
        const cfg: LlmConfig = {
          baseURL: parsed.baseURL || "",
          apiKey: parsed.apiKey || "",
          model: parsed.model || "gpt-4o"
        };
        setConfig(cfg);
        setConfigLoaded(cfg.baseURL !== "" && cfg.apiKey !== "");
      } catch {
        setConfigLoaded(false);
      }
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

  // Listen for broadcast-triggered tasks (from TaskReceiver)
  useEffect(() => {
    const sub = DeviceEventEmitter.addListener("onTaskReceived", (event) => {
      console.log("[onTaskReceived]", JSON.stringify(event));
      const broadcastTask = event.task;
      if (!broadcastTask) return;

      // Use broadcast config overrides if provided, otherwise use stored config
      const taskConfig: LlmConfig = {
        baseURL: event.apiUrl || config.baseURL,
        apiKey: event.apiKey || config.apiKey,
        model: event.model || config.model
      };

      if (!taskConfig.baseURL || !taskConfig.apiKey) return;
      if (agentRef.current.isRunning) return;

      setLogs([]);
      agentRef.current.execute(broadcastTask, taskConfig, addLog);
    });
    return () => sub.remove();
  }, [config, addLog]);

  const saveConfig = useCallback(async () => {
    try {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(config));
      setConfigLoaded(config.baseURL !== "" && config.apiKey !== "");
      setShowConfig(false);
    } catch {}
  }, [config]);

  const addLog = useCallback((msg: string) => {
    setLogs((prev) => [...prev, msg]);
  }, []);

  const handleSend = useCallback(async () => {
    const trimmed = task.trim();
    if (!trimmed) return;
    if (!configLoaded) return;
    if (agentRef.current.isRunning) return;

    setLogs([]);
    setTask("");

    await agentRef.current.execute(trimmed, config, addLog);
  }, [task, config, configLoaded, addLog]);

  const handleAbort = useCallback(() => {
    agentRef.current.abort();
  }, []);

  const isRunning = agentRef.current.isRunning;

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>RN Agent</Text>
        <View style={styles.headerRight}>
          <View
            style={[
              styles.statusDot,
              { backgroundColor: serviceRunning ? "#4CAF50" : "#F44336" }
            ]}
          />
          <Text style={styles.statusText}>
            {serviceRunning ? "Service ON" : "Service OFF"}
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
          onPress={() => setShowConfig(!showConfig)}
        >
          <Text style={styles.smallBtnText}>
            {showConfig ? "Hide Config" : "LLM Config"}
          </Text>
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

      {/* Config panel */}
      {showConfig && (
        <View style={styles.configPanel}>
          <TextInput
            style={styles.configInput}
            placeholder="Base URL"
            placeholderTextColor="#999"
            value={config.baseURL}
            onChangeText={(t) => setConfig((prev) => ({ ...prev, baseURL: t }))}
            autoCapitalize="none"
          />
          <TextInput
            style={styles.configInput}
            placeholder="API Key"
            placeholderTextColor="#999"
            value={config.apiKey}
            onChangeText={(t) => setConfig((prev) => ({ ...prev, apiKey: t }))}
            autoCapitalize="none"
            secureTextEntry
          />
          <TextInput
            style={styles.configInput}
            placeholder="Model (e.g. gpt-4o)"
            placeholderTextColor="#999"
            value={config.model}
            onChangeText={(t) => setConfig((prev) => ({ ...prev, model: t }))}
            autoCapitalize="none"
          />
          <TouchableOpacity style={styles.saveBtn} onPress={saveConfig}>
            <Text style={styles.saveBtnText}>Save</Text>
          </TouchableOpacity>
        </View>
      )}

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
          <TouchableOpacity style={styles.abortBtn} onPress={handleAbort}>
            <Text style={styles.sendBtnText}>Stop</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={[
              styles.sendBtn,
              (!configLoaded || !task.trim()) && styles.disabledBtn
            ]}
            onPress={handleSend}
            disabled={!configLoaded || !task.trim()}
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
          <Text style={styles.logPlaceholder}>
            {configLoaded
              ? "Enter a task and tap Send"
              : "Configure LLM settings first"}
          </Text>
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
  configPanel: {
    backgroundColor: "#16213e",
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: "#0f3460"
  },
  configInput: {
    backgroundColor: "#0f3460",
    color: "#eee",
    borderRadius: 6,
    padding: 8,
    marginBottom: 6,
    fontSize: 14
  },
  saveBtn: {
    backgroundColor: "#e94560",
    borderRadius: 6,
    padding: 8,
    alignItems: "center"
  },
  saveBtnText: {
    color: "#fff",
    fontWeight: "bold"
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
    backgroundColor: "#F44336",
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
