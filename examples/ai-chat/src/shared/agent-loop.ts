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
    "Each action automatically returns a follow-up screenshot showing the result. " +
    "Coordinates use a 0-999 grid: (0,0)=top-left corner, (999,999)=bottom-right corner, (500,500)=center. " +
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
      .describe("X position in 0-999 range (0=left edge, 999=right edge)"),
    y: z
      .number()
      .optional()
      .describe("Y position in 0-999 range (0=top edge, 999=bottom edge)"),
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
      )
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
  "All coordinates use a 0-999 grid mapped to the screenshot image.\n" +
  "(0,0) = top-left corner, (999,999) = bottom-right corner, (500,500) = center.\n" +
  "Examples: top-right corner = (999,0), bottom-left corner = (0,999).\n" +
  "A button at roughly 70% from left and 85% from top → x=700, y=850.\n" +
  "\n" +
  "SCREENSHOT STRATEGY:\n" +
  "PREFER win({ action: 'window_screenshot', handle }) over screen({ action: 'screenshot' }).\n" +
  "Full desktop screenshots are very large and waste tokens. Only use screen({ action: 'screenshot' }) when:\n" +
  "- You don't know which window to target yet (first step)\n" +
  "- You need to see the taskbar or desktop icons\n" +
  "For all other cases, use win({ action: 'list_windows' }) to find the target window handle,\n" +
  "then use win({ action: 'window_screenshot', handle }) to capture just that window.\n" +
  "\n" +
  "WORKFLOW RULES:\n" +
  "1. Every action (click, type, key_press, scroll, mouse_move) automatically includes a follow-up screenshot — you see the result immediately without calling screenshot again.\n" +
  "2. Only call screenshot/window_screenshot explicitly when you need to see the screen for the FIRST time or after switching windows.\n" +
  "3. If wrong window is in front, use win({ action: 'focus_window' }) then the follow-up screenshot will show the new state.\n" +
  "4. Use coordinates from the MOST RECENT screenshot only.\n" +
  "5. Before clicking, use screen({ action: 'annotate', x, y }) to verify target coords.\n" +
  "6. Do NOT include base64 image data in your text response.\n" +
  "7. When scrolling, ALWAYS provide x,y coordinates pointing to the CENTER of the area you want to scroll (e.g. the chat message area, not the sidebar). Use a small amount (3-5) so you don't skip content.";

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
    const norm = normX !== undefined ? `norm(${normX},${normY})→` : "";
    return `[agent] screen: scroll ${result.direction} ${result.amount}${norm ? ` at ${norm}` : ""}${result.desktopX !== undefined ? `desktop(${result.desktopX},${result.desktopY})` : ""} → success`;
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
  abortSignal?: AbortSignal;
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
  let lastScreenshotIsWindow = false;
  let lastWindowHandle: number | null = null;

  /** Take a follow-up screenshot after an action, reusing the last screenshot mode. */
  async function autoScreenshot(
    onLog?: (msg: string) => void
  ): Promise<ScreenControlResult | null> {
    await new Promise((r) => setTimeout(r, 300)); // wait for UI update
    try {
      if (lastScreenshotIsWindow && lastWindowHandle) {
        const result = await executeScreenControl({
          action: "window_screenshot",
          handle: lastWindowHandle
        });
        onLog?.(
          `[agent] autoScreenshot: window_screenshot handle=${lastWindowHandle} → ${result.success ? `${result.width}x${result.height}` : result.error}`
        );
        return result;
      }
      const result = await executeScreenControl({ action: "screenshot" });
      onLog?.(
        `[agent] autoScreenshot: screenshot → ${result.success ? `${result.width}x${result.height}` : result.error}`
      );
      return result;
    } catch (e) {
      onLog?.(`[agent] autoScreenshot: error → ${e}`);
      return null;
    }
  }

  /** Remove old screenshot images from history to prevent token bloat. */
  function stripOldImages(): void {
    for (const msg of history) {
      if (msg.role !== "user" || typeof msg.content === "string") continue;
      const parts = msg.content as unknown as Array<{
        type: string;
        [k: string]: unknown;
      }>;
      for (let i = parts.length - 1; i >= 0; i--) {
        if (parts[i].type === "image") {
          parts[i] = {
            type: "text",
            text: "[Previous screenshot removed]"
          } as any;
        }
      }
    }
  }

  /** Convert 0-999 normalized coordinates to pixel coordinates. */
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
    const abortSignal = callbacks?.abortSignal;

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
      if (abortSignal?.aborted) {
        onLog?.("[agent] aborted before step " + (step + 1));
        return finalText || "[Agent aborted]";
      }
      onLog?.(`[agent] step ${step + 1}...`);

      const result = streamText({
        model,
        system: SYSTEM_PROMPT,
        messages: history,
        tools,
        abortSignal
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
        if (abortSignal?.aborted) {
          onLog?.("[agent] aborted before tool execution");
          return finalText || "[Agent aborted]";
        }
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

          // Normalize action aliases
          const originalAction = screenArgs.action;
          screenArgs.action = normalizeAction(screenArgs.action);
          if (isDoubleClickAlias(originalAction)) {
            screenArgs.doubleClick = true;
          }

          // Coordinate conversion: 0-999 normalized → pixel
          let savedNormX: number | undefined;
          let savedNormY: number | undefined;
          const coordActions = ["click", "mouse_move", "scroll", "annotate"];
          if (
            coordActions.includes(screenArgs.action) &&
            screenArgs.x !== undefined &&
            screenArgs.y !== undefined &&
            lastScreenshotWidth > 0 &&
            lastScreenshotHeight > 0
          ) {
            savedNormX = screenArgs.x;
            savedNormY = screenArgs.y;

            // Range check: normalized coordinates must be 0-999
            if (
              savedNormX < 0 ||
              savedNormX > 999 ||
              savedNormY < 0 ||
              savedNormY > 999
            ) {
              const errMsg = `Error: coordinates out of range. x=${savedNormX}, y=${savedNormY}. Use values 0-999.`;
              onLog?.(
                `[agent] screen: ${screenArgs.action} → REJECTED out-of-range`
              );
              history.push({
                role: "tool" as const,
                content: [
                  {
                    type: "tool-result" as const,
                    toolCallId: tc.toolCallId,
                    toolName: tc.toolName,
                    output: { type: "text", value: errMsg }
                  }
                ]
              });
              continue;
            }

            // Convert to pixel coordinates
            const pixel = normToPixel(savedNormX, savedNormY);
            screenArgs.x = pixel.x;
            screenArgs.y = pixel.y;
          }

          // Inject stored screenshot for annotate action
          if (screenArgs.action === "annotate" && lastScreenshotBase64) {
            screenArgs.base64 = lastScreenshotBase64;
          }

          const screenResult = await executeScreenControl(screenArgs);
          const resultWithAction = {
            ...screenResult,
            action: screenArgs.action
          };
          onLog?.(formatScreenLog(resultWithAction, savedNormX, savedNormY));

          const lines: string[] = [];
          lines.push(
            `Action: ${resultWithAction.action ?? "screenshot"} | Success: ${resultWithAction.success}`
          );
          if (resultWithAction.error)
            lines.push(`Error: ${resultWithAction.error}`);
          if (
            resultWithAction.action === "annotate" &&
            savedNormX !== undefined
          ) {
            lines.push(`Annotated at (${savedNormX}, ${savedNormY}).`);
            lines.push(
              `If the crosshair is on the correct target, click with x=${savedNormX}, y=${savedNormY} (use these EXACT numbers).`
            );
          }
          toolResultContent = lines.join("\n");

          // Auto-screenshot after interactive actions
          const autoScreenshotActions = [
            "click",
            "scroll",
            "type",
            "key_press",
            "mouse_move"
          ];
          if (
            autoScreenshotActions.includes(resultWithAction.action) &&
            resultWithAction.success
          ) {
            const autoResult = await autoScreenshot(onLog);
            if (autoResult?.success && autoResult.base64) {
              // Update stored screenshot state
              lastScreenshotBase64 = autoResult.base64;
              if (autoResult.width && autoResult.height) {
                lastScreenshotWidth = autoResult.width;
                lastScreenshotHeight = autoResult.height;
              }

              onScreenshot?.(
                currentStep,
                resultWithAction.action + "+auto_screenshot",
                autoResult.base64
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
              stripOldImages();
              history.push({
                role: "user",
                content: [
                  {
                    type: "text",
                    text: "Here is the screenshot after the action:"
                  },
                  {
                    type: "image",
                    image: autoResult.base64,
                    mediaType: "image/png"
                  }
                ]
              });
              continue;
            }
          }

          if (resultWithAction.base64) {
            // Store for future annotate/coord conversion (but not from annotate itself)
            if (resultWithAction.action !== "annotate") {
              lastScreenshotBase64 = resultWithAction.base64;
              lastScreenshotIsWindow = false;
              lastWindowHandle = null;
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
            stripOldImages();
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
            toolResultContent = `Action: window_screenshot | Success: ${resultWithAction.success}`;

            // Store for future annotate/coord conversion
            lastScreenshotBase64 = resultWithAction.base64;
            lastScreenshotIsWindow = true;
            lastWindowHandle = (winArgs.handle as number) ?? null;
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
            stripOldImages();
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

            // Auto-screenshot after window management actions
            const winAutoActions = [
              "focus_window",
              "resize_window",
              "maximize_window",
              "restore_window"
            ];
            if (
              winAutoActions.includes(resultWithAction.action) &&
              resultWithAction.success
            ) {
              // If focus_window was called with a handle, use it for window_screenshot
              if (
                resultWithAction.action === "focus_window" &&
                winArgs.handle
              ) {
                lastWindowHandle = winArgs.handle as number;
                lastScreenshotIsWindow = true;
              }
              const autoResult = await autoScreenshot(onLog);
              if (autoResult?.success && autoResult.base64) {
                lastScreenshotBase64 = autoResult.base64;
                if (autoResult.width && autoResult.height) {
                  lastScreenshotWidth = autoResult.width;
                  lastScreenshotHeight = autoResult.height;
                }

                onScreenshot?.(
                  currentStep,
                  resultWithAction.action + "+auto_screenshot",
                  autoResult.base64
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
                stripOldImages();
                history.push({
                  role: "user",
                  content: [
                    {
                      type: "text",
                      text: "Here is the screenshot after the action:"
                    },
                    {
                      type: "image",
                      image: autoResult.base64,
                      mediaType: "image/png"
                    }
                  ]
                });
                continue;
              }
            }
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
    if (abortSignal?.aborted) {
      return finalText || "[Agent aborted]";
    }
    history.push({
      role: "user",
      content: "Summarize what you did and the result."
    });
    const summary = streamText({
      model,
      system: SYSTEM_PROMPT,
      messages: history,
      abortSignal
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
    lastScreenshotIsWindow = false;
    lastWindowHandle = null;
  }

  return { runAgent, reset };
}
