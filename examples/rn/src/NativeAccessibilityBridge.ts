import { NativeModules } from "react-native";

interface AccessibilityBridgeInterface {
  // Sync methods
  getDeviceName(): string;
  appendLogLine(line: string): boolean;
  clearLogFile(): boolean;

  // Async methods
  isServiceRunning(): Promise<boolean>;
  saveConfig(baseURL: string, apiKey: string, model: string): Promise<void>;
  runAgentTask(task: string, configJson: string): Promise<string>;
}

const { AccessibilityBridge } = NativeModules;

export default AccessibilityBridge as AccessibilityBridgeInterface;
