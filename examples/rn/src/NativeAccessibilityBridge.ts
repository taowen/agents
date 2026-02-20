import { NativeModules } from "react-native";

interface AccessibilityBridgeInterface {
  // Sync methods (log file)
  appendLogLine(line: string): boolean;
  clearLogFile(): boolean;

  // Async methods
  isServiceRunning(): Promise<boolean>;
  readAssetConfig(): Promise<string>;
  saveConfig(baseURL: string, apiKey: string, model: string): Promise<void>;
  runAgentTask(task: string, configJson: string): Promise<string>;
}

const { AccessibilityBridge } = NativeModules;

export default AccessibilityBridge as AccessibilityBridgeInterface;
