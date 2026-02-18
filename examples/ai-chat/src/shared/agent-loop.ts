/**
 * Unified agent loop for Windows desktop automation.
 * Used by both the Electron standalone CLI and the browser-side bridge agent.
 * No Electron or browser globals — all I/O via dependency injection.
 */
import type { LanguageModel, ModelMessage } from "ai";
import { streamText, tool } from "ai";
import { z } from "zod";

import type {
  ScreenControlParams,
  ScreenControlResult,
  BashResult
} from "./screen-control-types.ts";
import { normalizeAction, isDoubleClickAlias } from "./action-aliases.ts";

// ---- Tool definitions ----

const bashToolDef = tool({
  description: "Execute a bash command in a virtual in-memory filesystem",
  inputSchema: z.object({
    command: z.string().describe("The bash command to execute")
  })
});

const screenToolDef = tool({
  description:
    "Control the Windows desktop screen. Take screenshots, click, type, press keys, move mouse, and scroll. " +
    "Use 'screenshot' first to see what's on screen, then interact with elements using normalized 0-1000 coordinates. " +
    "Coordinates use a normalized 0-1000 range: the image is divided into a 1000x1000 grid, top-left is (0,0), bottom-right is (999,999). " +
    "The system automatically converts these to actual pixel positions. " +
    "Use 'annotate' with x,y to draw a red crosshair on the last screenshot — verify your target before clicking.",
  inputSchema: z.object({
    action: z
      .enum([
        "screenshot",
        "click",
        "mouse_move",
        "type",
        "key_press",
        "scroll",
        "annotate"
      ])
      .describe("The screen action to perform"),
    x: z
      .number()
      .optional()
      .describe("X coordinate in normalized 0-1000 range"),
    y: z
      .number()
      .optional()
      .describe("Y coordinate in normalized 0-1000 range"),
    text: z.string().optional().describe("Text to type"),
    key: z.string().optional().describe("Key name to press"),
    modifiers: z.array(z.string()).optional().describe("Modifier keys"),
    button: z
      .enum(["left", "right", "middle"])
      .optional()
      .describe("Mouse button"),
    doubleClick: z.boolean().optional().describe("Double-click"),
    direction: z.enum(["up", "down"]).optional().describe("Scroll direction"),
    amount: z.number().optional().describe("Scroll amount")
  })
});

const windowToolDef = tool({
  description:
    "Manage windows on the Windows desktop. List visible windows, focus/activate, " +
    "move/resize, minimize/maximize/restore, or take a screenshot of a single window. " +
    "Use list_windows first to discover windows and their handles. " +
    "Use window_screenshot to capture just one window (smaller image than full desktop screenshot).",
  inputSchema: z.object({
    action: z.enum([
      "list_windows",
      "focus_window",
      "resize_window",
      "minimize_window",
      "maximize_window",
      "restore_window",
      "window_screenshot"
    ]),
    handle: z.number().optional().describe("Window handle from list_windows"),
    title: z
      .string()
      .optional()
      .describe("Window title substring (for focus_window)"),
    x: z.number().optional().describe("X position for resize_window"),
    y: z.number().optional().describe("Y position for resize_window"),
    width: z.number().optional().describe("Width for resize_window"),
    height: z.number().optional().describe("Height for resize_window")
  })
});

// ---- System prompt ----

const SYSTEM_PROMPT =
  "You are a remote desktop agent running on a Windows machine.\n" +
  "You have access to a bash shell. Use `mount` to see available filesystems.\n" +
  "You have the Windows desktop screen. You can take screenshots, click, type, press keys, move the mouse, and scroll.\n" +
  "You have a 'win' tool for window management: list_windows, focus/activate, move/resize, minimize/maximize/restore, window_screenshot.\n" +
  "\n" +
  "COORDINATE SYSTEM:\n" +
  "All coordinates use a normalized 0-1000 range. Top-left is (0,0), bottom-right is (999,999).\n" +
  "\n" +
  "WORKFLOW RULES:\n" +
  "1. Take a screenshot before every action. Never assume previous action succeeded.\n" +
  "2. After each click or key_press, take another screenshot to verify.\n" +
  "3. If wrong window is in front, use win({ action: 'focus_window' }) then screenshot again.\n" +
  "4. Use coordinates from the MOST RECENT screenshot only.\n" +
  "5. Before clicking, use screen({ action: 'annotate', x, y }) to verify target coords.\n" +
  "6. Do NOT include base64 image data in your text response.";

// ---- Logging helpers ----

function formatScreenLog(
  result: ScreenControlResult,
  normX?: number,
  normY?: number
): string {
  const action = result.action ?? "screenshot";
  if (!result.success)
    return `[agent] screen: ${action} → error: ${result.error}`;
  if (action === "click" || action === "mouse_move") {
    const norm = normX !== undefined ? `norm(${normX},${normY})→` : "";
    const pixel = `pixel(${result.x},${result.y})`;
    const desktop =
      result.desktopX !== undefined
        ? `→desktop(${result.desktopX},${result.desktopY})`
        : "";
    return `[agent] screen: ${action} ${norm}${pixel}${desktop} → success`;
  }
  if (action === "screenshot" && result.width) {
    return `[agent] screen: screenshot → ${result.width}x${result.height}`;
  }
  if (action === "annotate" && result.width) {
    const norm = normX !== undefined ? `norm(${normX},${normY})→` : "";
    const pixel = `pixel(${result.x ?? "?"},${result.y ?? "?"})`;
    return `[agent] screen: annotate ${norm}${pixel} → ${result.width}x${result.height}`;
  }
  if (action === "scroll") {
    return `[agent] screen: scroll ${result.direction} ${result.amount}${result.desktopX !== undefined ? ` at desktop(${result.desktopX},${result.desktopY})` : ""} → success`;
  }
  return `[agent] screen: ${action} → success`;
}

function formatWinLog(
  result: ScreenControlResult,
  params: ScreenControlParams
): string {
  const action = result.action ?? params.action;
  if (!result.success) return `[agent] win: ${action} → error: ${result.error}`;
  if (action === "list_windows") {
    const count = result.windows?.length ?? 0;
    return `[agent] win: list_windows → ${count} windows`;
  }
  if (action === "window_screenshot") {
    const handle = params.handle ?? "?";
    const size = result.width ? `${result.width}x${result.height}` : "?";
    const pos =
      result.windowLeft !== undefined
        ? ` at (${result.windowLeft},${result.windowTop})`
        : "";
    return `[agent] win: window_screenshot handle=${handle} → ${size}${pos}`;
  }
  if (action === "focus_window") {
    const handle = params.handle ?? params.title ?? "?";
    return `[agent] win: focus_window handle=${handle} → success`;
  }
  return `[agent] win: ${action} → success`;
}

// ---- Agent factory ----

export interface AgentLoopConfig {
  getModel: () => LanguageModel | Promise<LanguageModel>;
  executeBash: (command: string) => Promise<BashResult>;
  executeScreenControl: (
    params: ScreenControlParams
  ) => Promise<ScreenControlResult>;
  maxSteps?: number;
}

export interface AgentLoopCallbacks {
  onLog?: (msg: string) => void;
  onScreenshot?: (step: number, action: string, base64: string) => void;
}

export interface AgentLoop {
  runAgent(
    userMessage: string,
    callbacks?: AgentLoopCallbacks
  ): Promise<string>;
  reset(): void;
}

export function createAgentLoop(config: AgentLoopConfig): AgentLoop {
  const { getModel, executeBash, executeScreenControl, maxSteps = 20 } = config;

  const history: ModelMessage[] = [];
  let stepCounter = 0;
  let lastScreenshotBase64: string | null = null;
  let lastScreenshotWidth = 0;
  let lastScreenshotHeight = 0;

  function normToPixel(normX: number, normY: number): { x: number; y: number } {
    return {
      x: Math.round((normX / 1000) * lastScreenshotWidth),
      y: Math.round((normY / 1000) * lastScreenshotHeight)
    };
  }

  async function runAgent(
    userMessage: string,
    callbacks?: AgentLoopCallbacks
  ): Promise<string> {
    const onLog = callbacks?.onLog;
    const onScreenshot = callbacks?.onScreenshot;

    history.push({ role: "user", content: userMessage });
    stepCounter = 0;

    const model = await getModel();
    const tools = {
      bash: bashToolDef,
      screen: screenToolDef,
      win: windowToolDef
    };
    let finalText = "";

    for (let step = 0; step < maxSteps; step++) {
      onLog?.(`[agent] step ${step + 1}...`);

      const result = streamText({
        model,
        system: SYSTEM_PROMPT,
        messages: history,
        tools
      });

      const toolCalls: Array<{
        toolCallId: string;
        toolName: string;
        args: Record<string, unknown>;
      }> = [];

      let stepText = "";
      for await (const event of result.fullStream) {
        if (event.type === "text-delta") {
          stepText += event.text;
        } else if (event.type === "tool-call") {
          const tcArgs = (event as any).input ?? (event as any).args ?? {};
          toolCalls.push({
            toolCallId: event.toolCallId,
            toolName: event.toolName,
            args: tcArgs as Record<string, unknown>
          });
          onLog?.(
            `[agent] tool: ${event.toolName}(${JSON.stringify(tcArgs).slice(0, 100)})`
          );
        }
      }

      const response = await result.response;
      history.push(...response.messages);

      if (stepText) finalText = stepText;
      if (toolCalls.length === 0) break;

      for (const tc of toolCalls) {
        let toolResultContent: string;
        const currentStep = stepCounter++;

        if (tc.toolName === "bash") {
          const bashResult = await executeBash(
            (tc.args as { command: string }).command
          );
          onLog?.(
            `[agent] bash: exit=${bashResult.exitCode} stdout=${bashResult.stdout.slice(0, 100)}`
          );
          toolResultContent = JSON.stringify(bashResult);
        } else if (tc.toolName === "screen") {
          const screenArgs = tc.args as unknown as ScreenControlParams;
          let logNormX: number | undefined;
          let logNormY: number | undefined;

          // Normalize action aliases
          const originalAction = screenArgs.action;
          screenArgs.action = normalizeAction(screenArgs.action);
          if (isDoubleClickAlias(originalAction)) {
            screenArgs.doubleClick = true;
          }

          // Convert normalized 0-1000 coords to pixel coords
          const coordActions = ["click", "mouse_move", "scroll", "annotate"];
          if (
            coordActions.includes(screenArgs.action) &&
            screenArgs.x !== undefined &&
            screenArgs.y !== undefined &&
            lastScreenshotWidth > 0 &&
            lastScreenshotHeight > 0
          ) {
            logNormX = screenArgs.x;
            logNormY = screenArgs.y;
            const pixel = normToPixel(screenArgs.x, screenArgs.y);
            screenArgs.x = pixel.x;
            screenArgs.y = pixel.y;
          }

          // Inject stored screenshot for annotate action
          if (screenArgs.action === "annotate" && lastScreenshotBase64) {
            screenArgs.base64 = lastScreenshotBase64;
            if (logNormX !== undefined) {
              screenArgs.normX = logNormX;
              screenArgs.normY = logNormY;
            }
          }

          const screenResult = await executeScreenControl(screenArgs);
          const resultWithAction = {
            ...screenResult,
            action: screenArgs.action
          };
          onLog?.(formatScreenLog(resultWithAction, logNormX, logNormY));

          const lines: string[] = [];
          lines.push(
            `Action: ${resultWithAction.action ?? "screenshot"} | Success: ${resultWithAction.success}`
          );
          if (resultWithAction.error)
            lines.push(`Error: ${resultWithAction.error}`);
          if (resultWithAction.width && resultWithAction.height)
            lines.push(
              `Screen: ${resultWithAction.width}x${resultWithAction.height}`
            );
          toolResultContent = lines.join("\n");

          if (resultWithAction.base64) {
            // Store for future annotate/coord conversion (but not from annotate itself)
            if (resultWithAction.action !== "annotate") {
              lastScreenshotBase64 = resultWithAction.base64;
              if (resultWithAction.width && resultWithAction.height) {
                lastScreenshotWidth = resultWithAction.width;
                lastScreenshotHeight = resultWithAction.height;
              }
            }

            onScreenshot?.(
              currentStep,
              resultWithAction.action ?? "screenshot",
              resultWithAction.base64
            );

            history.push({
              role: "tool" as const,
              content: [
                {
                  type: "tool-result" as const,
                  toolCallId: tc.toolCallId,
                  toolName: tc.toolName,
                  output: { type: "text", value: toolResultContent }
                }
              ]
            });
            history.push({
              role: "user",
              content: [
                {
                  type: "text",
                  text:
                    resultWithAction.action === "annotate"
                      ? "Here is the annotated screenshot with crosshair:"
                      : "Here is the screenshot:"
                },
                {
                  type: "image",
                  image: resultWithAction.base64,
                  mediaType: "image/png"
                }
              ]
            });
            continue;
          }
        } else if (tc.toolName === "win") {
          const winArgs = tc.args as unknown as ScreenControlParams;

          // Normalize action aliases for win tool too
          const originalAction = winArgs.action;
          winArgs.action = normalizeAction(winArgs.action);
          if (isDoubleClickAlias(originalAction)) {
            winArgs.doubleClick = true;
          }

          const winResult = await executeScreenControl(winArgs);
          const resultWithAction = { ...winResult, action: winArgs.action };
          onLog?.(formatWinLog(resultWithAction, winArgs));

          if (resultWithAction.action === "list_windows") {
            const windows = resultWithAction.windows || [];
            toolResultContent = JSON.stringify(windows, null, 2);
          } else if (
            resultWithAction.action === "window_screenshot" &&
            resultWithAction.base64
          ) {
            const lines: string[] = [];
            lines.push(
              `Action: window_screenshot | Success: ${resultWithAction.success}`
            );
            if (resultWithAction.width && resultWithAction.height)
              lines.push(
                `Window size: ${resultWithAction.width}x${resultWithAction.height}`
              );
            toolResultContent = lines.join("\n");

            // Store for future annotate/coord conversion
            lastScreenshotBase64 = resultWithAction.base64;
            if (resultWithAction.width && resultWithAction.height) {
              lastScreenshotWidth = resultWithAction.width;
              lastScreenshotHeight = resultWithAction.height;
            }

            onScreenshot?.(
              currentStep,
              "window_screenshot",
              resultWithAction.base64
            );

            history.push({
              role: "tool" as const,
              content: [
                {
                  type: "tool-result" as const,
                  toolCallId: tc.toolCallId,
                  toolName: tc.toolName,
                  output: { type: "text", value: toolResultContent }
                }
              ]
            });
            history.push({
              role: "user",
              content: [
                { type: "text", text: "Here is the window screenshot:" },
                {
                  type: "image",
                  image: resultWithAction.base64,
                  mediaType: "image/png"
                }
              ]
            });
            continue;
          } else {
            const lines: string[] = [];
            lines.push(
              `Action: ${resultWithAction.action} | Success: ${resultWithAction.success}`
            );
            if (resultWithAction.error)
              lines.push(`Error: ${resultWithAction.error}`);
            if (resultWithAction.message) lines.push(resultWithAction.message);
            toolResultContent = lines.join("\n");
          }
        } else {
          toolResultContent = "Unknown tool";
        }

        history.push({
          role: "tool" as const,
          content: [
            {
              type: "tool-result" as const,
              toolCallId: tc.toolCallId,
              toolName: tc.toolName,
              output: { type: "text", value: toolResultContent }
            }
          ]
        });
      }
    }

    // Final summary step without tools
    history.push({
      role: "user",
      content: "Summarize what you did and the result."
    });
    const summary = streamText({
      model,
      system: SYSTEM_PROMPT,
      messages: history
    });
    finalText = "";
    for await (const event of summary.fullStream) {
      if (event.type === "text-delta") {
        finalText += event.text;
      }
    }
    const summaryResp = await summary.response;
    history.push(...summaryResp.messages);

    return finalText || "[Agent completed without text output]";
  }

  function reset() {
    history.length = 0;
    lastScreenshotBase64 = null;
    lastScreenshotWidth = 0;
    lastScreenshotHeight = 0;
  }

  return { runAgent, reset };
}
