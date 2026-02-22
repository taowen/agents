import { NativeModules } from "react-native";

interface AccessibilityBridgeInterface {
  // Sync methods
  getDeviceName(): string;
  appendLogLine(line: string): boolean;
  clearLogFile(): boolean;
  isCloudConnected(): boolean;

  // Async methods
  isServiceRunning(): Promise<boolean>;
  reconnectCloud(): Promise<void>;
  saveConfig(baseURL: string, apiKey: string, model: string): Promise<void>;
  sendUserTask(text: string): Promise<void>;
  connectCloud(url: string, deviceName: string): Promise<void>;
  disconnectCloud(): Promise<void>;
}

const { AccessibilityBridge } = NativeModules;

export default AccessibilityBridge as AccessibilityBridgeInterface;
