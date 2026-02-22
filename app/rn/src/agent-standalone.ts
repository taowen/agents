/**
 * agent-standalone.ts
 *
 * Self-contained agent loop that runs in a standalone Hermes runtime
 * (inside the AccessibilityService process). Host function metadata lives
 * in host-api.ts; prompt & tool definitions are generated in prompt.ts.
 */

import { SYSTEM_PROMPT, TOOLS } from "./prompt";
import type { ChatMessage, LlmConfig, ToolCall } from "./types";
import { callLLM } from "./llm-client";
import { trimMessages, compactConversation } from "./conversation-manager";

// Declare globals provided by C++ host functions (for TypeScript only)
declare global {
  var get_screen: () => string;
  var take_screenshot: () => string;
  var click: (
    target: string | { desc?: string; x?: number; y?: number }
  ) => boolean;
  var long_click: (
    target: string | { desc?: string; x?: number; y?: number }
  ) => boolean;
  var scroll: (direction: string) => boolean;
  var scroll_element: (text: string, direction: string) => string;
  var type_text: (text: string) => boolean;
  var press_home: () => boolean;
  var press_back: () => boolean;
  var press_recents: () => boolean;
  var show_notifications: () => boolean;
  var launch_app: (name: string) => string;
  var list_apps: () => string;
  var sleep: (ms: number) => void;
  var log: (msg: string) => void;
  var update_status: (text: string) => void;
  var ask_user: (question: string) => string;
  var hide_overlay: () => void;
  var runAgent: (task: string, configJson: string) => string;
  var executeCodeForServer: (code: string) => {
    result: string;
    screenshots?: string[];
  };
  var __DEVICE_PROMPT__: { systemPrompt: string; tools: unknown[] };
}

const MAX_STEPS = 30;
const MAX_GET_SCREEN_PER_EXEC = 5;
const EXEC_TIMEOUT_MS = 30_000;

function formatTime(): string {
  const d = new Date();
  const pad = (n: number) => (n < 10 ? "0" : "") + n;
  return (
    pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds())
  );
}

function agentLog(msg: string): void {
  const line = "[" + formatTime() + "] " + msg;
  log(line);
}

/**
 * Save and wrap all host functions with update_status + safety limits.
 * Returns a cleanup function that restores originals.
 */
function wrapHostFunctions(opts: {
  deadline: number;
  maxGetScreen: number;
  onGetScreen?: (tree: string) => void;
  onScreenshot: (b64: string) => void;
}): { getScreenCount: () => number; restore: () => void } {
  let getScreenCount = 0;

  const origGetScreen = get_screen;
  const origTakeScreenshot = take_screenshot;
  const origClick = click;
  const origLongClick = long_click;
  const origTypeText = type_text;
  const origLaunchApp = launch_app;
  const origScroll = scroll;
  const origScrollElement = scroll_element;
  const origPressHome = press_home;
  const origPressBack = press_back;
  const origPressRecents = press_recents;
  const origShowNotifications = show_notifications;

  globalThis.get_screen = function (): string {
    if (Date.now() > opts.deadline) throw new Error("Script execution timeout");
    getScreenCount++;
    if (getScreenCount > opts.maxGetScreen) {
      throw new Error(
        "get_screen() called " +
          getScreenCount +
          " times. Max is " +
          opts.maxGetScreen +
          ". Return and plan next actions in a new execute_js call."
      );
    }
    update_status("read screen");
    const tree = origGetScreen();
    if (opts.onGetScreen) opts.onGetScreen(tree);
    return tree;
  };

  globalThis.take_screenshot = function (): string {
    if (Date.now() > opts.deadline) throw new Error("Script execution timeout");
    update_status("take screenshot");
    const b64 = origTakeScreenshot();
    if (b64.startsWith("ERROR:")) {
      return b64;
    }
    opts.onScreenshot(b64);
    return "screenshot captured - image will be sent to you";
  };

  globalThis.click = function (
    target: string | { desc?: string; x?: number; y?: number }
  ): boolean {
    const desc =
      typeof target === "string"
        ? target
        : target.desc || "(" + target.x + "," + target.y + ")";
    update_status("click " + desc);
    return origClick(target);
  };

  globalThis.long_click = function (
    target: string | { desc?: string; x?: number; y?: number }
  ): boolean {
    const desc =
      typeof target === "string"
        ? target
        : target.desc || "(" + target.x + "," + target.y + ")";
    update_status("long click " + desc);
    return origLongClick(target);
  };

  globalThis.type_text = function (text: string): boolean {
    update_status("type " + text.substring(0, 15));
    return origTypeText(text);
  };

  globalThis.launch_app = function (name: string): string {
    update_status("launch " + name);
    return origLaunchApp(name);
  };

  globalThis.scroll = function (direction: string): boolean {
    update_status("scroll " + direction);
    return origScroll(direction);
  };

  globalThis.scroll_element = function (
    text: string,
    direction: string
  ): string {
    update_status("scroll " + text + " " + direction);
    return origScrollElement(text, direction);
  };

  globalThis.press_home = function (): boolean {
    update_status("press home");
    return origPressHome();
  };

  globalThis.press_back = function (): boolean {
    update_status("press back");
    return origPressBack();
  };

  globalThis.press_recents = function (): boolean {
    update_status("press recents");
    return origPressRecents();
  };

  globalThis.show_notifications = function (): boolean {
    update_status("show notifications");
    return origShowNotifications();
  };

  return {
    getScreenCount: () => getScreenCount,
    restore() {
      globalThis.get_screen = origGetScreen;
      globalThis.take_screenshot = origTakeScreenshot;
      globalThis.click = origClick;
      globalThis.long_click = origLongClick;
      globalThis.type_text = origTypeText;
      globalThis.launch_app = origLaunchApp;
      globalThis.scroll = origScroll;
      globalThis.scroll_element = origScrollElement;
      globalThis.press_home = origPressHome;
      globalThis.press_back = origPressBack;
      globalThis.press_recents = origPressRecents;
      globalThis.show_notifications = origShowNotifications;
    }
  };
}

function executeCode(
  code: string,
  thinking?: string
): { text: string; screenshot?: string } {
  if (thinking) {
    agentLog("[THINK] " + thinking);
  }
  agentLog("[CODE] " + code);

  let lastGetScreenResult: string | null = null;
  let capturedScreenshot: string | undefined;

  const wrapped = wrapHostFunctions({
    deadline: Date.now() + EXEC_TIMEOUT_MS,
    maxGetScreen: MAX_GET_SCREEN_PER_EXEC,
    onGetScreen(tree) {
      lastGetScreenResult = tree;
    },
    onScreenshot(b64) {
      capturedScreenshot = b64;
    }
  });

  try {
    let result = (0, eval)(code);
    if (result === undefined && lastGetScreenResult !== null) {
      result = lastGetScreenResult;
    }
    const text = result === undefined ? "undefined" : String(result);
    return { text, screenshot: capturedScreenshot };
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    const text = "[JS Error] " + msg;
    return { text, screenshot: capturedScreenshot };
  } finally {
    wrapped.restore();
  }
}

/**
 * executeCodeForServer — called from DeviceConnection.java for cloud exec_js.
 * Same host function wrapping (with update_status) but returns { result, screenshots[] }.
 */
function executeCodeForServer(code: string): {
  result: string;
  screenshots: string[];
} {
  const capturedScreenshots: string[] = [];
  let lastGetScreenResult: string | null = null;

  const wrapped = wrapHostFunctions({
    deadline: Date.now() + EXEC_TIMEOUT_MS,
    maxGetScreen: MAX_GET_SCREEN_PER_EXEC,
    onGetScreen(tree) {
      lastGetScreenResult = tree;
    },
    onScreenshot(b64) {
      capturedScreenshots.push(b64);
    }
  });

  try {
    let result = (0, eval)(code);
    if (result === undefined && lastGetScreenResult !== null) {
      result = lastGetScreenResult;
    }
    return {
      result: result === undefined ? "undefined" : String(result),
      screenshots: capturedScreenshots
    };
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return {
      result: "[JS Error] " + msg,
      screenshots: capturedScreenshots
    };
  } finally {
    wrapped.restore();
  }
}

globalThis.executeCodeForServer = executeCodeForServer;

// Global conversation buffer — persists across runAgent calls
let conversationMessages: ChatMessage[] | null = null;

// Main entry point — called from Java via nativeEvaluateJS
function runAgent(task: string, configJson: string): string {
  const config: LlmConfig = JSON.parse(configJson);

  agentLog("[TASK] Received task: " + task);
  update_status("task started");

  if (!conversationMessages) {
    conversationMessages = [{ role: "system", content: SYSTEM_PROMPT }];
  }
  conversationMessages.push({ role: "user", content: task });
  compactConversation(conversationMessages, config, agentLog);

  const messages = conversationMessages;

  for (let step = 1; step <= MAX_STEPS; step++) {
    update_status("step " + step + "/" + MAX_STEPS + ": thinking…");
    agentLog("[STEP " + step + "] Calling LLM...");
    trimMessages(messages);

    let response: { content: string | null; toolCalls: ToolCall[] | null };
    try {
      response = callLLM(messages, TOOLS, config);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      update_status("error: " + msg);
      agentLog("[ERROR] LLM call failed: " + msg);
      return "Error: " + msg;
    }

    if (response.toolCalls) {
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: response.content,
        tool_calls: response.toolCalls
      };
      messages.push(assistantMsg);

      const toolNames = response.toolCalls
        .map((tc: ToolCall) => tc.function.name)
        .join(", ");
      agentLog("[STEP " + step + "] LLM returned tool_calls: " + toolNames);

      update_status("step " + step + "/" + MAX_STEPS + ": executing…");

      for (const toolCall of response.toolCalls) {
        let resultText: string;
        let screenshot: string | undefined;
        if (toolCall.function.name === "execute_js") {
          const args = JSON.parse(toolCall.function.arguments);
          const execResult = executeCode(
            args.code,
            response.content ?? undefined
          );
          resultText = execResult.text;
          screenshot = execResult.screenshot;
        } else {
          resultText = "Unknown tool: " + toolCall.function.name;
        }

        const logResult =
          resultText.length > 200
            ? resultText.substring(0, 200) +
              "... (" +
              resultText.length +
              " chars)"
            : resultText;
        agentLog("[TOOL] " + toolCall.function.name + " -> " + logResult);

        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: resultText
        });

        // Inject screenshot as a user vision message
        if (screenshot) {
          messages.push({
            role: "user",
            content: [
              {
                type: "image_url",
                image_url: { url: "data:image/jpeg;base64," + screenshot }
              }
            ]
          });
        }
      }
    } else {
      const finalContent = response.content || "(no response)";
      agentLog("[DONE] Task completed: " + finalContent);
      return finalContent;
    }
  }

  update_status("max steps reached");
  agentLog(
    "[ERROR] Reached max steps (" + MAX_STEPS + ") without completing task"
  );
  return "Error: reached max steps";
}

// Expose runAgent on globalThis so it can be called from Java
globalThis.runAgent = runAgent;
