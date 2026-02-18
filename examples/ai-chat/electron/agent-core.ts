/**
 * Platform-agnostic agent loop for Windows desktop automation.
 * Uses dependency injection — no Electron, no browser globals.
 */
import type { LanguageModel, ModelMessage } from "ai";
import { streamText, tool } from "ai";
import { Bash, InMemoryFs } from "just-bash";
import { z } from "zod";

import type {
  ScreenControlParams,
  ScreenControlResult,
  ScreenControlOptions
} from "./win-automation.ts";
import { getForegroundWindow } from "./win-automation.ts";

// ---- Tool definitions ----

const bashToolDef = tool({
  description: "Execute a bash command in a virtual in-memory filesystem",
  parameters: z.object({
    command: z.string().describe("The bash command to execute")
  })
});

const screenToolDef = tool({
  description:
    "Control the Windows desktop screen. Take screenshots, click, type, press keys, move mouse, and scroll. " +
    "Use 'screenshot' first to see what's on screen, then interact with elements by their pixel coordinates. " +
    "Coordinates are in physical pixels from top-left (0,0).",
  parameters: z.object({
    action: z
      .enum([
        "screenshot",
        "click",
        "mouse_move",
        "type",
        "key_press",
        "scroll"
      ])
      .describe("The screen action to perform"),
    x: z.number().optional().describe("X coordinate in pixels"),
    y: z.number().optional().describe("Y coordinate in pixels"),
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
  parameters: z.object({
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

// ---- Agent factory ----

export interface AgentDeps {
  screenControlFn: (
    params: ScreenControlParams,
    options?: ScreenControlOptions
  ) => Promise<ScreenControlResult>;
  model: LanguageModel;
  debugDir?: string;
}

export function createAgent(deps: AgentDeps) {
  const fs = new InMemoryFs();
  const bash = new Bash({ fs, cwd: "/home" });
  const history: ModelMessage[] = [];
  let debugStepCounter = 0;

  async function executeBash(command: string) {
    const result = await bash.exec(command);
    return {
      stdout: result.stdout,
      stderr: result.stderr,
      exitCode: result.exitCode
    };
  }

  async function executeScreen(
    input: ScreenControlParams
  ): Promise<ScreenControlResult> {
    const step = debugStepCounter++;
    const debugOpts: ScreenControlOptions | undefined = deps.debugDir
      ? { debugDir: deps.debugDir, debugStep: step }
      : undefined;
    const result = await deps.screenControlFn(input, debugOpts);
    return { ...result, action: input.action };
  }

  async function executeWindow(
    input: ScreenControlParams
  ): Promise<ScreenControlResult> {
    const step = debugStepCounter++;
    const debugOpts: ScreenControlOptions | undefined = deps.debugDir
      ? { debugDir: deps.debugDir, debugStep: step }
      : undefined;
    const result = await deps.screenControlFn(input, debugOpts);
    return { ...result, action: input.action };
  }

  async function runAgent(
    userMessage: string,
    onLog?: (msg: string) => void
  ): Promise<string> {
    history.push({ role: "user", content: userMessage });

    const systemPrompt =
      "You are a remote desktop agent running on a Windows machine. " +
      "You receive instructions and execute them on the local desktop. " +
      "You have access to a bash shell (virtual in-memory filesystem) " +
      "and the Windows desktop screen. You can take screenshots, click, type, press keys, move the mouse, and scroll. " +
      "You also have a 'win' tool for window management: list visible windows, focus/activate, " +
      "move/resize, minimize/maximize/restore, and take a screenshot of a single window. " +
      "Use win({ action: 'list_windows' }) to discover windows and their handles. " +
      "Use win({ action: 'window_screenshot', handle: ... }) to capture just one window (smaller image, and coordinates in subsequent click/move/scroll are automatically translated to desktop coordinates). " +
      "\n\nIMPORTANT workflow rules:\n" +
      "1. BEFORE every action (click, type, key_press), take a screenshot or window_screenshot first to see the current state. Never assume a previous action succeeded — always verify visually.\n" +
      "2. After each click or key_press, take another screenshot to confirm the expected result happened.\n" +
      "3. If a screenshot shows the wrong window is in front, use win({ action: 'focus_window' }) to bring the target window back, then take another screenshot to confirm.\n" +
      "4. Use pixel coordinates from the MOST RECENT screenshot to target UI elements. Never reuse coordinates from an earlier screenshot.\n" +
      "5. After completing the task, provide a concise text summary of what you did and the result. Do NOT include base64 image data in your text response.";

    const tools = {
      bash: bashToolDef,
      screen: screenToolDef,
      win: windowToolDef
    };
    let finalText = "";

    for (let step = 0; step < 20; step++) {
      onLog?.(`[agent] step ${step + 1}...`);

      const result = streamText({
        model: deps.model,
        system: systemPrompt,
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

        if (deps.debugDir) {
          const fg = await getForegroundWindow();
          onLog?.(
            `[debug] BEFORE ${tc.toolName}: foreground = ${fg.handle} "${fg.title}"`
          );
        }

        if (tc.toolName === "bash") {
          const bashResult = await executeBash(
            (tc.args as { command: string }).command
          );
          onLog?.(
            `[agent] bash result: exit=${bashResult.exitCode} stdout=${bashResult.stdout.slice(0, 100)}`
          );
          toolResultContent = JSON.stringify(bashResult);
        } else if (tc.toolName === "screen") {
          const screenResult = await executeScreen(
            tc.args as ScreenControlParams
          );
          onLog?.(
            `[agent] screen: ${screenResult.action} success=${screenResult.success}`
          );

          const lines: string[] = [];
          lines.push(
            `Action: ${screenResult.action ?? "screenshot"} | Success: ${screenResult.success}`
          );
          if (screenResult.error) lines.push(`Error: ${screenResult.error}`);
          if (screenResult.width && screenResult.height)
            lines.push(`Screen: ${screenResult.width}x${screenResult.height}`);
          toolResultContent = lines.join("\n");

          if (screenResult.base64) {
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
                { type: "text", text: "Here is the screenshot:" },
                {
                  type: "image",
                  image: screenResult.base64,
                  mediaType: "image/png"
                }
              ]
            });
            if (deps.debugDir) {
              const fg = await getForegroundWindow();
              onLog?.(
                `[debug] AFTER  ${tc.toolName}: foreground = ${fg.handle} "${fg.title}"`
              );
            }
            continue;
          }
        } else if (tc.toolName === "win") {
          const winResult = await executeWindow(tc.args as ScreenControlParams);
          onLog?.(
            `[agent] win: ${winResult.action} success=${winResult.success}`
          );

          if (winResult.action === "list_windows") {
            const windows = winResult.windows || [];
            toolResultContent = JSON.stringify(windows, null, 2);
          } else if (
            winResult.action === "window_screenshot" &&
            winResult.base64
          ) {
            const lines: string[] = [];
            lines.push(
              `Action: window_screenshot | Success: ${winResult.success}`
            );
            if (winResult.width && winResult.height)
              lines.push(`Window size: ${winResult.width}x${winResult.height}`);
            toolResultContent = lines.join("\n");

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
                  image: winResult.base64,
                  mediaType: "image/png"
                }
              ]
            });
            if (deps.debugDir) {
              const fg = await getForegroundWindow();
              onLog?.(
                `[debug] AFTER  ${tc.toolName}: foreground = ${fg.handle} "${fg.title}"`
              );
            }
            continue;
          } else {
            const lines: string[] = [];
            lines.push(
              `Action: ${winResult.action} | Success: ${winResult.success}`
            );
            if (winResult.error) lines.push(`Error: ${winResult.error}`);
            if (winResult.message) lines.push(winResult.message);
            toolResultContent = lines.join("\n");
          }
        } else {
          toolResultContent = "Unknown tool";
        }

        if (deps.debugDir) {
          const fg = await getForegroundWindow();
          onLog?.(
            `[debug] AFTER  ${tc.toolName}: foreground = ${fg.handle} "${fg.title}"`
          );
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

    return finalText || "[Agent completed without text output]";
  }

  function reset() {
    history.length = 0;
  }

  return { runAgent, reset };
}
