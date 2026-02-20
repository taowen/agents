import { NativeModules } from "react-native";

interface AccessibilityBridgeInterface {
  // Sync methods (blocking, called from Hermes eval)
  getScreen(): string;
  clickByText(text: string): boolean;
  clickByDesc(desc: string): boolean;
  clickByCoords(x: number, y: number): boolean;
  longClickByText(text: string): boolean;
  longClickByDesc(desc: string): boolean;
  longClickByCoords(x: number, y: number): boolean;
  scrollScreen(direction: string): boolean;
  scrollElement(text: string, direction: string): string;
  typeText(text: string): boolean;
  pressHome(): boolean;
  pressBack(): boolean;
  pressRecents(): boolean;
  showNotifications(): boolean;
  sleepMs(ms: number): boolean;
  launchApp(name: string): string;
  listApps(): string;
  appendLogLine(line: string): boolean;
  clearLogFile(): boolean;

  // Async methods
  isServiceRunning(): Promise<boolean>;
  resetScreens(): Promise<void>;
  readAssetConfig(): Promise<string>;
}

const { AccessibilityBridge } = NativeModules;

export default AccessibilityBridge as AccessibilityBridgeInterface;
