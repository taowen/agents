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

const desktopToolDef = tool({
  description:
    "Control the Windows desktop: screenshots, click, type, press keys, move mouse, scroll, " +
    "and window management (list, focus, resize, minimize, maximize, restore, window screenshot). " +
    "Each action automatically returns a follow-up screenshot showing the result. " +
    "Coordinates use a 0-999 grid: (0,0)=top-left corner, (999,999)=bottom-right corner, (500,500)=center. " +
    "Use list_windows to discover windows and their handles. " +
    "Use window_screenshot to capture just one window (smaller image than full desktop screenshot).",
  inputSchema: z.object({
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
  })
});

const powershellToolDef = tool({
  description:
    "Run an arbitrary PowerShell command on the Windows host. " +
    "Use for system administration, registry access, process management, file operations, " +
    "installing software, or any task that requires native Windows capabilities. " +
    "Returns stdout, stderr, and exitCode.",
  inputSchema: z.object({
    command: z
      .string()
      .describe("The PowerShell command or script block to execute")
  })
});

// ---- System prompt ----

const SYSTEM_PROMPT_BASE =
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
  "- a11y: faster (no image transfer), saves tokens, gives exact element bounds. Best for standard UI.\n" +
  "- pixel: needed for visual content (images, colors, layout verification), canvas/games, or elements without names.\n" +
  "Use window_screenshot with mode='pixel' or mode='accessibility' to force a specific mode for that call.\n" +
  "Follow-up auto-screenshots always use 'auto' mode — the backend picks the best format based on content.\n" +
  "In a11y mode, coordinates from element bounds are exact — click directly.\n" +
  "In pixel mode, visually estimate coordinates from the screenshot.";

const CLOUD_DRIVE_PROMPT =
  "\n\nCLOUD FILE STORAGE:\n" +
  "You have a cloud file storage mounted as the cloud:\\ PowerShell drive. " +
  "This drive is backed by the user's cloud bash environment — they share the same filesystem:\n" +
  "  cloud:\\home\\user  = /home/user  (persistent user files)\n" +
  "  cloud:\\data       = /data       (persistent large file storage)\n" +
  "  cloud:\\etc        = /etc        (persistent config)\n\n" +
  "Use standard PowerShell cmdlets:\n" +
  "- Get-ChildItem cloud:\\           # list root (home, data, etc, mnt)\n" +
  "- Get-ChildItem cloud:\\home\\user  # list user's files\n" +
  "- Get-Content cloud:\\home\\user\\file.txt     # read file\n" +
  '- Set-Content cloud:\\home\\user\\file.txt -Value "hello"  # write file\n' +
  "- Copy-Item cloud:\\home\\user\\file.txt C:\\Users\\...\\Desktop\\  # download to Windows\n" +
  "- Copy-Item C:\\Users\\...\\report.pdf cloud:\\home\\user\\       # upload from Windows\n" +
  "- Remove-Item cloud:\\home\\user\\old.txt      # delete\n" +
  "- New-Item cloud:\\home\\user\\subdir -ItemType Directory  # mkdir\n\n" +
  "IMPORTANT: The user (or the main cloud agent) may ask you to transfer files between Windows and cloud. " +
  "Use Copy-Item to move files in either direction.";

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

// ---- Agent factory ----

export interface AgentLoopConfig {
  getModel: () => LanguageModel | Promise<LanguageModel>;
  executePowerShell: (command: string) => Promise<BashResult>;
  executeScreenControl: (
    params: ScreenControlParams
  ) => Promise<ScreenControlResult>;
  hasCloudDrive?: boolean;
  maxSteps?: number;
}

export interface AgentLoopCallbacks {
  onLog?: (msg: string) => void;
  onScreenshot?: (step: number, action: string, base64: string) => void;
  onText?: (step: number, label: string, text: string) => void;
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
  const {
    getModel,
    executePowerShell,
    executeScreenControl,
    hasCloudDrive = false,
    maxSteps = 20
  } = config;

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
          handle: lastWindowHandle,
          mode: "auto"
        });
        const tag = result.accessibilityTree ? "[a11y]" : "[pixel]";
        onLog?.(
          `[agent] autoScreenshot: window_screenshot ${tag} handle=${lastWindowHandle} → ${result.success ? `${result.width}x${result.height}` : result.error}`
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

  /** Remove old screenshot images and accessibility trees from history to prevent token bloat. */
  function stripOldScreenshots(): void {
    for (const msg of history) {
      if (msg.role !== "user" || typeof msg.content === "string") continue;
      const parts = msg.content as unknown as Array<{
        type: string;
        text?: string;
        [k: string]: unknown;
      }>;
      for (let i = parts.length - 1; i >= 0; i--) {
        if (parts[i].type === "image") {
          parts[i] = {
            type: "text",
            text: "[Previous screenshot removed]"
          } as any;
        } else if (
          parts[i].type === "text" &&
          (parts[i].text?.startsWith(
            "Here is the window accessibility tree:"
          ) ||
            parts[i].text?.startsWith(
              "Here is the updated accessibility tree after the action:"
            ))
        ) {
          parts[i] = {
            type: "text",
            text: "[Previous accessibility tree removed]"
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
    const onText = callbacks?.onText;
    const abortSignal = callbacks?.abortSignal;

    history.push({ role: "user", content: userMessage });
    stepCounter = 0;

    const model = await getModel();
    const tools: Record<string, ReturnType<typeof tool>> = {
      powershell: powershellToolDef,
      desktop: desktopToolDef
    };

    const systemPrompt = hasCloudDrive
      ? SYSTEM_PROMPT_BASE + CLOUD_DRIVE_PROMPT
      : SYSTEM_PROMPT_BASE;
    let finalText = "";

    for (let step = 0; step < maxSteps; step++) {
      if (abortSignal?.aborted) {
        onLog?.("[agent] aborted before step " + (step + 1));
        return finalText || "[Agent aborted]";
      }
      const stepStart = Date.now();
      onLog?.(`[agent] step ${step + 1}...`);

      const result = streamText({
        model,
        system: systemPrompt,
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

      const llmMs = Date.now() - stepStart;
      onLog?.(
        `[agent] step ${step + 1} LLM: ${llmMs}ms, tools: ${toolCalls.length}, text: ${stepText.length} chars`
      );
      if (stepText) {
        onLog?.(`[agent] LLM text: ${stepText}`);
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

        if (tc.toolName === "powershell") {
          const psResult = await executePowerShell(
            (tc.args as { command: string }).command
          );
          onLog?.(
            `[agent] powershell: exit=${psResult.exitCode} stdout=${psResult.stdout.slice(0, 100)}`
          );
          toolResultContent = JSON.stringify(psResult);
        } else if (tc.toolName === "desktop") {
          const args = tc.args as unknown as ScreenControlParams;

          // Normalize action aliases
          const originalAction = args.action;
          args.action = normalizeAction(args.action);
          if (isDoubleClickAlias(originalAction)) {
            args.doubleClick = true;
          }

          // Coordinate conversion: 0-999 normalized → pixel for coordinate-based actions
          let savedNormX: number | undefined;
          let savedNormY: number | undefined;
          const COORD_ACTIONS = new Set(["click", "mouse_move", "scroll"]);
          if (
            COORD_ACTIONS.has(args.action) &&
            args.x !== undefined &&
            args.y !== undefined &&
            lastScreenshotWidth > 0 &&
            lastScreenshotHeight > 0
          ) {
            savedNormX = args.x;
            savedNormY = args.y;

            // Range check: normalized coordinates must be 0-999
            if (
              savedNormX < 0 ||
              savedNormX > 999 ||
              savedNormY < 0 ||
              savedNormY > 999
            ) {
              const errMsg = `Error: coordinates out of range. x=${savedNormX}, y=${savedNormY}. Use values 0-999.`;
              onLog?.(
                `[agent] desktop: ${args.action} → REJECTED out-of-range`
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
            args.x = pixel.x;
            args.y = pixel.y;
            args.normX = savedNormX;
            args.normY = savedNormY;
          }

          // Auto-annotate for logging (not sent to LLM)
          if (savedNormX !== undefined && savedNormY !== undefined) {
            let screenshotForAnnotation = lastScreenshotBase64;
            // In a11y mode lastScreenshotBase64 is null — take a temporary pixel screenshot for annotation only
            if (
              !screenshotForAnnotation &&
              lastScreenshotIsWindow &&
              lastWindowHandle
            ) {
              try {
                const pixelShot = await executeScreenControl({
                  action: "window_screenshot",
                  handle: lastWindowHandle,
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
                  x: args.x,
                  y: args.y,
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

          const result = await executeScreenControl(args);
          const resultWithAction = { ...result, action: args.action };
          onLog?.(
            formatDesktopLog(resultWithAction, args, savedNormX, savedNormY)
          );
          if (resultWithAction.a11yDiagnostics) {
            onLog?.(`[agent] a11y: ${resultWithAction.a11yDiagnostics}`);
          }

          // Build toolResultContent based on action type
          if (resultWithAction.action === "list_windows") {
            const windows = resultWithAction.windows || [];
            toolResultContent = JSON.stringify(windows, null, 2);
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

          // Auto-screenshot after interactive actions
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
          if (
            AUTO_SCREENSHOT_ACTIONS.has(resultWithAction.action) &&
            resultWithAction.success
          ) {
            // If focus_window was called with a handle, use it for window_screenshot
            if (resultWithAction.action === "focus_window" && args.handle) {
              lastWindowHandle = args.handle as number;
              lastScreenshotIsWindow = true;
            }
            const autoResult = await autoScreenshot(onLog);
            if (autoResult?.a11yDiagnostics) {
              onLog?.(`[agent] a11y: ${autoResult.a11yDiagnostics}`);
            }
            if (autoResult?.success && autoResult.accessibilityTree) {
              // Auto-screenshot returned accessibility tree
              lastScreenshotBase64 = null;
              if (autoResult.width && autoResult.height) {
                lastScreenshotWidth = autoResult.width;
                lastScreenshotHeight = autoResult.height;
              }
              onText?.(
                currentStep,
                resultWithAction.action + "+auto_a11y",
                autoResult.accessibilityTree
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
              stripOldScreenshots();
              history.push({
                role: "user",
                content: [
                  {
                    type: "text",
                    text:
                      "Here is the updated accessibility tree after the action:\n" +
                      autoResult.accessibilityTree
                  }
                ]
              });
              continue;
            } else if (autoResult?.success && autoResult.base64) {
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
              stripOldScreenshots();
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

          // Handle accessibility tree results from window_screenshot
          if (resultWithAction.accessibilityTree !== undefined) {
            lastScreenshotBase64 = null;
            lastScreenshotIsWindow = true;
            lastWindowHandle = (args.handle as number) ?? null;
            if (resultWithAction.width && resultWithAction.height) {
              lastScreenshotWidth = resultWithAction.width;
              lastScreenshotHeight = resultWithAction.height;
            }
            onText?.(
              currentStep,
              "window_screenshot+a11y",
              resultWithAction.accessibilityTree
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
            stripOldScreenshots();
            history.push({
              role: "user",
              content: [
                {
                  type: "text",
                  text:
                    "Here is the window accessibility tree:\n" +
                    resultWithAction.accessibilityTree
                }
              ]
            });
            continue;
          }

          // Handle direct screenshot results (screenshot, window_screenshot, annotate)
          if (resultWithAction.base64) {
            if (resultWithAction.action === "window_screenshot") {
              lastScreenshotBase64 = resultWithAction.base64;
              lastScreenshotIsWindow = true;
              lastWindowHandle = (args.handle as number) ?? null;
            }
            // Update dimensions (for all screenshot types including annotate)
            if (resultWithAction.width && resultWithAction.height) {
              lastScreenshotWidth = resultWithAction.width;
              lastScreenshotHeight = resultWithAction.height;
            }

            onScreenshot?.(
              currentStep,
              resultWithAction.action,
              resultWithAction.base64
            );

            const imageLabel = "Here is the window screenshot:";

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
            stripOldScreenshots();
            history.push({
              role: "user",
              content: [
                { type: "text", text: imageLabel },
                {
                  type: "image",
                  image: resultWithAction.base64,
                  mediaType: "image/png"
                }
              ]
            });
            continue;
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

    // Final summary step — only if the model never produced text output
    if (abortSignal?.aborted) {
      return finalText || "[Agent aborted]";
    }
    if (finalText) {
      onLog?.("[agent] skipping summary (already have text output)");
      return finalText;
    }

    onLog?.("[agent] requesting summary...");
    const summaryStart = Date.now();
    history.push({
      role: "user",
      content: "Summarize what you did and the result."
    });
    const summary = streamText({
      model,
      system: systemPrompt,
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
    onLog?.(`[agent] summary LLM: ${Date.now() - summaryStart}ms`);

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
