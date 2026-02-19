/**
 * Shared types for screen/window control used across
 * the Electron main process, preload bridge, and browser agent.
 */

export interface ScreenControlParams {
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
  base64?: string;
  handle?: number;
  title?: string;
  width?: number;
  height?: number;
  normX?: number;
  normY?: number;
  mode?: "auto" | "accessibility" | "pixel";
}

export interface ScreenControlResult {
  success: boolean;
  error?: string;
  width?: number;
  height?: number;
  base64?: string;
  accessibilityTree?: string;
  action?: string;
  windows?: Array<Record<string, unknown>>;
  message?: string;
  a11yDiagnostics?: string;
  [key: string]: unknown;
}

export interface BashResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}
