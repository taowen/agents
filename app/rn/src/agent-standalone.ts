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
declare function get_screen(): string;
declare function take_screenshot(): string;
declare function click(
  target: string | { desc?: string; x?: number; y?: number }
): boolean;
declare function long_click(
  target: string | { desc?: string; x?: number; y?: number }
): boolean;
declare function scroll(direction: string): boolean;
declare function scroll_element(text: string, direction: string): string;
declare function type_text(text: string): boolean;
declare function press_home(): boolean;
declare function press_back(): boolean;
declare function press_recents(): boolean;
declare function show_notifications(): boolean;
declare function launch_app(name: string): string;
declare function list_apps(): string;
declare function sleep(ms: number): void;
declare function log(msg: string): void;

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

function executeCode(
  code: string,
  thinking?: string
): { text: string; screenshot?: string } {
  if (thinking) {
    agentLog("[THINK] " + thinking);
  }
  agentLog("[CODE] " + code);

  const actionLog: string[] = [];
  let getScreenCount = 0;
  let capturedScreenshot: string | undefined;
  const deadline = Date.now() + EXEC_TIMEOUT_MS;

  // Wrap host functions with logging and limits
  const origGetScreen = get_screen;
  (globalThis as any).get_screen = function (): string {
    if (Date.now() > deadline) throw new Error("Script execution timeout");
    getScreenCount++;
    if (getScreenCount > MAX_GET_SCREEN_PER_EXEC) {
      throw new Error(
        "get_screen() called " +
          getScreenCount +
          " times. Max is " +
          MAX_GET_SCREEN_PER_EXEC +
          ". Return and plan next actions in a new execute_js call."
      );
    }
    const tree = origGetScreen();
    actionLog.push("[get_screen] (" + tree.length + " chars)");
    return tree;
  };

  const origTakeScreenshot = take_screenshot;
  (globalThis as any).take_screenshot = function (): string {
    if (Date.now() > deadline) throw new Error("Script execution timeout");
    const b64 = origTakeScreenshot();
    if (b64.startsWith("ERROR:")) {
      actionLog.push("[take_screenshot] " + b64);
      return b64;
    }
    capturedScreenshot = b64;
    actionLog.push(
      "[take_screenshot] captured (" + b64.length + " chars base64)"
    );
    return "screenshot captured - image will be sent to you";
  };

  try {
    // Use indirect eval to run in global scope
    const result = (0, eval)(code);
    const resultStr = result === undefined ? "undefined" : String(result);
    let text: string;
    if (actionLog.length > 0) {
      actionLog.push("[Script returned] " + resultStr);
      text = actionLog.join("\n");
    } else {
      text = resultStr;
    }
    return { text, screenshot: capturedScreenshot };
  } catch (e: any) {
    const error = "[JS Error] " + (e.message || String(e));
    let text: string;
    if (actionLog.length > 0) {
      actionLog.push(error);
      text = actionLog.join("\n");
    } else {
      text = error;
    }
    return { text, screenshot: capturedScreenshot };
  } finally {
    // Restore originals
    (globalThis as any).get_screen = origGetScreen;
    (globalThis as any).take_screenshot = origTakeScreenshot;
  }
}

// Global conversation buffer — persists across runAgent calls
let conversationMessages: ChatMessage[] | null = null;

// Main entry point — called from Java via nativeEvaluateJS
function runAgent(task: string, configJson: string): string {
  const config: LlmConfig = JSON.parse(configJson);

  agentLog("[TASK] Received task: " + task);

  if (!conversationMessages) {
    conversationMessages = [{ role: "system", content: SYSTEM_PROMPT }];
  }
  conversationMessages.push({ role: "user", content: task });
  compactConversation(conversationMessages, config, agentLog);

  const messages = conversationMessages;

  for (let step = 1; step <= MAX_STEPS; step++) {
    agentLog("[STEP " + step + "] Calling LLM...");
    trimMessages(messages);

    let response: { content: string | null; toolCalls: ToolCall[] | null };
    try {
      response = callLLM(messages, TOOLS, config);
    } catch (e: any) {
      agentLog("[ERROR] LLM call failed: " + e.message);
      return "Error: " + e.message;
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
          } as any);
        }
      }
    } else {
      const finalContent = response.content || "(no response)";
      agentLog("[DONE] Task completed: " + finalContent);
      return finalContent;
    }
  }

  agentLog(
    "[ERROR] Reached max steps (" + MAX_STEPS + ") without completing task"
  );
  return "Error: reached max steps";
}

// Expose runAgent on globalThis so it can be called from Java
(globalThis as any).runAgent = runAgent;
