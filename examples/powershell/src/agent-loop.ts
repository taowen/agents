/**
 * Agent loop for Windows desktop automation — built on the `pi` framework.
 * Standalone CLI version — no cloud drive, no Electron dependencies.
 */
import type { LanguageModel } from "ai";
import { z, Agent, type AgentTool, type AgentToolResult } from "pi";

import type {
  ScreenControlParams,
  ScreenControlResult,
  BashResult
} from "./types.js";
import { normalizeAction, isDoubleClickAlias } from "./action-aliases.js";

// ---- Shared desktop state ----

interface DesktopState {
  lastScreenshotBase64: string | null;
  lastScreenshotWidth: number;
  lastScreenshotHeight: number;
  lastScreenshotIsWindow: boolean;
  lastWindowHandle: number | null;
  stepCounter: number;
}

// ---- Callbacks passed to tools ----

interface ToolCallbacks {
  onLog?: (msg: string) => void;
  onScreenshot?: (step: number, action: string, base64: string) => void;
  onText?: (step: number, label: string, text: string) => void;
}

// ---- System prompt ----

const SYSTEM_PROMPT =
  "You are a remote desktop agent running on a Windows machine.\n" +
  "You have a `powershell` tool for running PowerShell commands on the Windows host.\n" +
  "You have a `desktop` tool for all screen and window operations: screenshots, click, type, press keys, move mouse, scroll, " +
  "and window management (list_windows, focus_window, resize_window, minimize/maximize/restore_window, window_screenshot).\n" +
  "\n" +
  "COORDINATE SYSTEM:\n" +
  "All coordinates use a 0-999 grid mapped to the screenshot image.\n" +
  "(0,0) = top-left corner, (999,999) = bottom-right corner, (500,500) = center.\n" +
  "Examples: top-right corner = (999,0), bottom-left corner = (0,999).\n" +
  "A button at roughly 70% from left and 85% from top → x=700, y=850.\n" +
  "\n" +
  "WORKFLOW RULES:\n" +
  "1. Every action (click, type, key_press, scroll, mouse_move) automatically includes a follow-up screenshot — you see the result immediately without calling screenshot again.\n" +
  "2. Start by calling desktop({ action: 'list_windows' }) to discover windows and their handles, then use desktop({ action: 'window_screenshot', handle }) to see a specific window. Only call window_screenshot explicitly when you need to see the screen for the FIRST time or after switching windows.\n" +
  "3. If wrong window is in front, use desktop({ action: 'focus_window' }) then the follow-up screenshot will show the new state.\n" +
  "4. Use coordinates from the MOST RECENT screenshot only.\n" +
  "5. Use coordinates from the MOST RECENT screenshot or accessibility tree only.\n" +
  "6. Do NOT include base64 image data in your text response.\n" +
  "7. When scrolling, ALWAYS provide x,y coordinates pointing to the CENTER of the area you want to scroll (e.g. the chat message area, not the sidebar). Use a small amount (3-5) so you don't skip content.\n" +
  "\n" +
  "ACCESSIBILITY TREE MODE:\n" +
  "window_screenshot returns a text accessibility tree by default.\n" +
  "The tree uses indentation for parent-child hierarchy. Each element line:\n" +
  '  [ControlType] Name="..." Value="..." bounds=[left,top][right,bottom] <patterns>\n' +
  "The first line is 'Window: WxH' — use W,H in the coordinate formula below.\n" +
  "Patterns indicate available interactions: Invoke (clickable), Toggle (checkbox/switch), " +
  "Scroll (scrollable area), ExpandCollapse (dropdown/tree node), SelectionItem.\n" +
  "\n" +
  "To click an element, convert its bounds to 0-999 coords:\n" +
  "  normX = round((left+right)/2 * 999 / W), normY = round((top+bottom)/2 * 999 / H)\n" +
  "Then desktop({ action: 'click', x: normX, y: normY }).\n" +
  "\n" +
  "MODE SELECTION (a11y vs pixel):\n" +
  "- a11y: faster (no image transfer), saves tokens. Best for standard UI.\n" +
  "- pixel: needed for visual content (images, colors, layout verification), canvas/games, or elements without names.\n" +
  "Use window_screenshot with mode='pixel' or mode='accessibility' to force a specific mode for that call.\n" +
  "Follow-up auto-screenshots always use 'auto' mode — the backend picks the best format based on content.\n" +
  "In a11y mode, coordinates from element bounds are exact — click directly.\n" +
  "In pixel mode, visually estimate coordinates from the screenshot.";

// ---- Logging helpers ----

function formatDesktopLog(
  result: ScreenControlResult,
  params: ScreenControlParams,
  normX?: number,
  normY?: number
): string {
  const action = result.action ?? params.action;
  if (!result.success)
    return `[agent] desktop: ${action} → error: ${result.error}`;
  if (action === "click" || action === "mouse_move") {
    const norm = normX !== undefined ? `norm(${normX},${normY})→` : "";
    const pixel = `pixel(${result.x},${result.y})`;
    const desktop =
      result.desktopX !== undefined
        ? `→desktop(${result.desktopX},${result.desktopY})`
        : "";
    return `[agent] desktop: ${action} ${norm}${pixel}${desktop} → success`;
  }
  if (action === "scroll") {
    const norm = normX !== undefined ? `norm(${normX},${normY})→` : "";
    return `[agent] desktop: scroll ${result.direction} ${result.amount}${norm ? ` at ${norm}` : ""}${result.desktopX !== undefined ? `desktop(${result.desktopX},${result.desktopY})` : ""} → success`;
  }
  if (action === "list_windows") {
    const count = result.windows?.length ?? 0;
    return `[agent] desktop: list_windows → ${count} windows`;
  }
  if (action === "window_screenshot") {
    const handle = params.handle ?? "?";
    const size = result.width ? `${result.width}x${result.height}` : "?";
    const pos =
      result.windowLeft !== undefined
        ? ` at (${result.windowLeft},${result.windowTop})`
        : "";
    const tag = result.accessibilityTree !== undefined ? "[a11y]" : "[pixel]";
    return `[agent] desktop: window_screenshot ${tag} handle=${handle} → ${size}${pos}`;
  }
  if (action === "focus_window") {
    const handle = params.handle ?? params.title ?? "?";
    return `[agent] desktop: focus_window handle=${handle} → success`;
  }
  return `[agent] desktop: ${action} → success`;
}

// ---- Desktop tool ----

const AUTO_SCREENSHOT_ACTIONS = new Set([
  "click",
  "scroll",
  "type",
  "key_press",
  "mouse_move",
  "focus_window",
  "resize_window",
  "maximize_window",
  "restore_window"
]);

function createDesktopTool(
  state: DesktopState,
  executeScreenControl: (
    params: ScreenControlParams
  ) => Promise<ScreenControlResult>,
  callbacks: ToolCallbacks
): AgentTool {
  return {
    name: "desktop",
    description:
      "Control the Windows desktop: screenshots, click, type, press keys, move mouse, scroll, " +
      "and window management (list, focus, resize, minimize, maximize, restore, window screenshot). " +
      "Each action automatically returns a follow-up screenshot showing the result. " +
      "Coordinates use a 0-999 grid: (0,0)=top-left corner, (999,999)=bottom-right corner, (500,500)=center. " +
      "Use list_windows to discover windows and their handles. " +
      "Use window_screenshot to capture just one window (smaller image than full desktop screenshot).",
    label: "Desktop Control",
    parameters: z.object({
      action: z
        .enum([
          "click",
          "mouse_move",
          "type",
          "key_press",
          "scroll",
          "list_windows",
          "focus_window",
          "resize_window",
          "minimize_window",
          "maximize_window",
          "restore_window",
          "window_screenshot"
        ])
        .describe("The desktop action to perform"),
      x: z
        .number()
        .optional()
        .describe(
          "X position in 0-999 range for screen actions, or pixel position for resize_window"
        ),
      y: z
        .number()
        .optional()
        .describe(
          "Y position in 0-999 range for screen actions, or pixel position for resize_window"
        ),
      text: z.string().optional().describe("Text to type"),
      key: z.string().optional().describe("Key name to press"),
      modifiers: z.array(z.string()).optional().describe("Modifier keys"),
      button: z
        .enum(["left", "right", "middle"])
        .optional()
        .describe("Mouse button"),
      doubleClick: z.boolean().optional().describe("Double-click"),
      direction: z.enum(["up", "down"]).optional().describe("Scroll direction"),
      amount: z
        .number()
        .optional()
        .describe(
          "Scroll wheel notches (default 3, use 30-50 to scroll through long pages or chat history)"
        ),
      handle: z.number().optional().describe("Window handle from list_windows"),
      title: z
        .string()
        .optional()
        .describe("Window title substring (for focus_window)"),
      width: z.number().optional().describe("Width for resize_window"),
      height: z.number().optional().describe("Height for resize_window"),
      mode: z
        .enum(["auto", "accessibility", "pixel"])
        .optional()
        .describe(
          "window_screenshot mode: 'auto' (default, prefers a11y tree), 'accessibility' (tree only), 'pixel' (image only). Only affects this call — auto-screenshots always use 'auto'."
        )
    }),
    execute: async (_toolCallId, params): Promise<AgentToolResult<any>> => {
      const { onLog, onScreenshot, onText } = callbacks;
      const currentStep = state.stepCounter++;
      const args = { ...(params as any) } as ScreenControlParams;

      // Normalize action + double-click alias
      const originalAction = args.action;
      args.action = normalizeAction(args.action);
      if (isDoubleClickAlias(originalAction)) {
        args.doubleClick = true;
      }

      // Coordinate conversion (0-999 → pixel)
      let savedNormX: number | undefined;
      let savedNormY: number | undefined;
      const COORD_ACTIONS = new Set(["click", "mouse_move", "scroll"]);
      if (
        COORD_ACTIONS.has(args.action) &&
        args.x !== undefined &&
        args.y !== undefined &&
        state.lastScreenshotWidth > 0 &&
        state.lastScreenshotHeight > 0
      ) {
        savedNormX = args.x;
        savedNormY = args.y;

        if (
          savedNormX < 0 ||
          savedNormX > 999 ||
          savedNormY < 0 ||
          savedNormY > 999
        ) {
          return {
            content: [
              {
                type: "text",
                text: `Error: coordinates out of range. x=${savedNormX}, y=${savedNormY}. Use values 0-999.`
              }
            ],
            details: {}
          };
        }

        args.x = Math.round((savedNormX / 1000) * state.lastScreenshotWidth);
        args.y = Math.round((savedNormY / 1000) * state.lastScreenshotHeight);
        args.normX = savedNormX;
        args.normY = savedNormY;
      }

      // Auto-annotate for logging
      if (savedNormX !== undefined && savedNormY !== undefined) {
        let screenshotForAnnotation = state.lastScreenshotBase64;
        if (
          !screenshotForAnnotation &&
          state.lastScreenshotIsWindow &&
          state.lastWindowHandle
        ) {
          try {
            const pixelShot = await executeScreenControl({
              action: "window_screenshot",
              handle: state.lastWindowHandle,
              mode: "pixel"
            });
            if (pixelShot.success && pixelShot.base64) {
              screenshotForAnnotation = pixelShot.base64;
            }
          } catch {}
        }
        if (screenshotForAnnotation) {
          try {
            const annotateResult = await executeScreenControl({
              action: "annotate",
              x: args.x!,
              y: args.y!,
              normX: savedNormX,
              normY: savedNormY,
              base64: screenshotForAnnotation
            });
            if (annotateResult.success && annotateResult.base64) {
              onScreenshot?.(
                currentStep,
                args.action + "+annotate",
                annotateResult.base64
              );
            }
          } catch {}
        }
      }

      // Execute the action
      const result = await executeScreenControl(args);
      const resultWithAction = { ...result, action: args.action };
      onLog?.(formatDesktopLog(resultWithAction, args, savedNormX, savedNormY));
      if (resultWithAction.a11yDiagnostics) {
        onLog?.(`[agent] a11y: ${resultWithAction.a11yDiagnostics}`);
      }

      // Build tool result text
      let toolResultText: string;
      if (resultWithAction.action === "list_windows") {
        toolResultText = JSON.stringify(
          resultWithAction.windows || [],
          null,
          2
        );
      } else {
        const lines: string[] = [];
        lines.push(
          `Action: ${resultWithAction.action} | Success: ${resultWithAction.success}`
        );
        if (resultWithAction.error)
          lines.push(`Error: ${resultWithAction.error}`);
        if (resultWithAction.message) lines.push(resultWithAction.message);
        toolResultText = lines.join("\n");
      }

      // Auto-screenshot after interactive actions
      if (
        AUTO_SCREENSHOT_ACTIONS.has(resultWithAction.action) &&
        resultWithAction.success
      ) {
        if (resultWithAction.action === "focus_window" && args.handle) {
          state.lastWindowHandle = args.handle as number;
          state.lastScreenshotIsWindow = true;
        }
        const autoResult = await autoScreenshot(
          state,
          executeScreenControl,
          onLog
        );
        if (autoResult?.a11yDiagnostics) {
          onLog?.(`[agent] a11y: ${autoResult.a11yDiagnostics}`);
        }

        if (autoResult?.success && autoResult.accessibilityTree) {
          state.lastScreenshotBase64 = null;
          if (autoResult.width && autoResult.height) {
            state.lastScreenshotWidth = autoResult.width;
            state.lastScreenshotHeight = autoResult.height;
          }
          onText?.(
            currentStep,
            resultWithAction.action + "+auto_a11y",
            autoResult.accessibilityTree
          );
          return {
            content: [
              {
                type: "text",
                text:
                  toolResultText +
                  "\nHere is the updated accessibility tree after the action:\n" +
                  autoResult.accessibilityTree
              }
            ],
            details: {}
          };
        }

        if (autoResult?.success && autoResult.base64) {
          state.lastScreenshotBase64 = autoResult.base64;
          if (autoResult.width && autoResult.height) {
            state.lastScreenshotWidth = autoResult.width;
            state.lastScreenshotHeight = autoResult.height;
          }
          onScreenshot?.(
            currentStep,
            resultWithAction.action + "+auto_screenshot",
            autoResult.base64
          );
          return {
            content: [
              { type: "text", text: toolResultText },
              { type: "image", data: autoResult.base64, mimeType: "image/png" }
            ],
            details: {}
          };
        }
      }

      // Handle accessibility tree from window_screenshot
      if (resultWithAction.accessibilityTree !== undefined) {
        state.lastScreenshotBase64 = null;
        state.lastScreenshotIsWindow = true;
        state.lastWindowHandle = (args.handle as number) ?? null;
        if (resultWithAction.width && resultWithAction.height) {
          state.lastScreenshotWidth = resultWithAction.width;
          state.lastScreenshotHeight = resultWithAction.height;
        }
        onText?.(
          currentStep,
          "window_screenshot+a11y",
          resultWithAction.accessibilityTree
        );
        return {
          content: [
            {
              type: "text",
              text:
                toolResultText +
                "\nHere is the window accessibility tree:\n" +
                resultWithAction.accessibilityTree
            }
          ],
          details: {}
        };
      }

      // Handle pixel screenshot from window_screenshot
      if (resultWithAction.base64) {
        if (resultWithAction.action === "window_screenshot") {
          state.lastScreenshotBase64 = resultWithAction.base64;
          state.lastScreenshotIsWindow = true;
          state.lastWindowHandle = (args.handle as number) ?? null;
        }
        if (resultWithAction.width && resultWithAction.height) {
          state.lastScreenshotWidth = resultWithAction.width;
          state.lastScreenshotHeight = resultWithAction.height;
        }
        onScreenshot?.(
          currentStep,
          resultWithAction.action,
          resultWithAction.base64
        );
        return {
          content: [
            { type: "text", text: toolResultText },
            {
              type: "image",
              data: resultWithAction.base64,
              mimeType: "image/png"
            }
          ],
          details: {}
        };
      }

      // Plain text result (no screenshot/a11y)
      return {
        content: [{ type: "text", text: toolResultText }],
        details: {}
      };
    }
  };
}

// ---- Auto-screenshot helper ----

async function autoScreenshot(
  state: DesktopState,
  executeScreenControl: (
    params: ScreenControlParams
  ) => Promise<ScreenControlResult>,
  onLog?: (msg: string) => void
): Promise<ScreenControlResult | null> {
  await new Promise((r) => setTimeout(r, 300));
  try {
    if (state.lastScreenshotIsWindow && state.lastWindowHandle) {
      const result = await executeScreenControl({
        action: "window_screenshot",
        handle: state.lastWindowHandle,
        mode: "auto"
      });
      const tag = result.accessibilityTree ? "[a11y]" : "[pixel]";
      onLog?.(
        `[agent] autoScreenshot: window_screenshot ${tag} handle=${state.lastWindowHandle} → ${result.success ? `${result.width}x${result.height}` : result.error}`
      );
      return result;
    }
    onLog?.(`[agent] autoScreenshot: no window handle, skipping`);
    return null;
  } catch (e) {
    onLog?.(`[agent] autoScreenshot: error → ${e}`);
    return null;
  }
}

// ---- PowerShell tool ----

function createPowershellTool(
  executePowerShell: (command: string) => Promise<BashResult>,
  onLog?: (msg: string) => void
): AgentTool {
  return {
    name: "powershell",
    description:
      "Run an arbitrary PowerShell command on the Windows host. " +
      "Use for system administration, registry access, process management, file operations, " +
      "installing software, or any task that requires native Windows capabilities. " +
      "Returns stdout, stderr, and exitCode.",
    label: "PowerShell",
    parameters: z.object({
      command: z
        .string()
        .describe("The PowerShell command or script block to execute")
    }),
    execute: async (_toolCallId, params): Promise<AgentToolResult<any>> => {
      const { command } = params as { command: string };
      const psResult = await executePowerShell(command);
      onLog?.(
        `[agent] powershell: exit=${psResult.exitCode} stdout=${psResult.stdout.slice(0, 100)}`
      );
      return {
        content: [{ type: "text", text: JSON.stringify(psResult) }],
        details: psResult
      };
    }
  };
}

// ---- Agent factory ----

export interface DesktopAgentConfig {
  model: LanguageModel;
  executeScreenControl: (
    params: ScreenControlParams
  ) => Promise<ScreenControlResult>;
  executePowerShell: (command: string) => Promise<BashResult>;
  onLog?: (msg: string) => void;
  onScreenshot?: (step: number, action: string, base64: string) => void;
  onText?: (step: number, label: string, text: string) => void;
}

export function createDesktopAgent(config: DesktopAgentConfig): Agent {
  const state: DesktopState = {
    lastScreenshotBase64: null,
    lastScreenshotWidth: 0,
    lastScreenshotHeight: 0,
    lastScreenshotIsWindow: false,
    lastWindowHandle: null,
    stepCounter: 0
  };

  return new Agent({
    initialState: {
      systemPrompt: SYSTEM_PROMPT,
      model: config.model,
      tools: [
        createDesktopTool(state, config.executeScreenControl, config),
        createPowershellTool(config.executePowerShell, config.onLog)
      ]
    }
  });
}
