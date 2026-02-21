/**
 * agent-standalone.ts
 *
 * Self-contained agent loop that runs in a standalone Hermes runtime
 * (inside the AccessibilityService process). Host function metadata lives
 * in host-api.ts; prompt & tool definitions are generated in prompt.ts.
 */

import { SYSTEM_PROMPT, TOOLS } from "./prompt";

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
declare function http_post(
  url: string,
  headersJson: string,
  body: string
): string;
declare function llm_chat(body: string): string;

const MAX_STEPS = 30;
const KEEP_RECENT_TOOL_RESULTS = 3;
const MAX_GET_SCREEN_PER_EXEC = 5;
const EXEC_TIMEOUT_MS = 30_000;
const COMPACT_THRESHOLD = 100;
const COMPACT_KEEP_RECENT = 10;

interface ContentPart {
  type: string;
  text?: string;
  image_url?: { url: string };
}

interface ChatMessage {
  role: string;
  content: string | ContentPart[] | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

interface ToolCall {
  id: string;
  type: string;
  function: {
    name: string;
    arguments: string;
  };
}

interface LlmConfig {
  baseURL: string;
  apiKey: string;
  model: string;
}

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

function callLLM(
  messages: ChatMessage[],
  tools: typeof TOOLS | any[],
  config: LlmConfig
): { content: string | null; toolCalls: ToolCall[] | null } {
  const payload: any = {
    model: config.model,
    messages: messages
  };
  if (tools && tools.length > 0) {
    payload.tools = tools;
  }
  const body = JSON.stringify(payload);

  // Use llm_chat (WebSocket via DeviceConnection) when available,
  // falling back to http_post for local/direct LLM calls
  const useLlmChat =
    typeof llm_chat === "function" &&
    (!config.baseURL || config.baseURL.includes("connect-screen.com"));

  const MAX_RETRIES = 2;
  const RETRY_DELAY_MS = 2000;
  let lastError = "";

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      sleep(RETRY_DELAY_MS);
    }

    let responseStr: string;
    if (useLlmChat) {
      responseStr = llm_chat(body);
    } else {
      let apiUrl = config.baseURL;
      if (apiUrl.endsWith("/")) apiUrl = apiUrl.slice(0, -1);
      apiUrl += "/chat/completions";
      const headers = JSON.stringify({
        Authorization: "Bearer " + config.apiKey,
        "Content-Type": "application/json"
      });
      responseStr = http_post(apiUrl, headers, body);
    }

    let data: any;
    try {
      data = JSON.parse(responseStr);
    } catch (e: any) {
      lastError =
        "Failed to parse LLM response: " + responseStr.substring(0, 200);
      if (attempt < MAX_RETRIES) continue;
      throw new Error(lastError);
    }

    if (data.error) {
      lastError = "LLM API error: " + JSON.stringify(data.error);
      const status = data.error.status || data.error.code || 0;
      if ((status === 429 || status >= 500) && attempt < MAX_RETRIES) continue;
      throw new Error(lastError);
    }

    if (!data.choices || !data.choices[0]) {
      lastError =
        "LLM response missing choices: " + responseStr.substring(0, 200);
      if (attempt < MAX_RETRIES) continue;
      throw new Error(lastError);
    }

    const message = data.choices[0].message;
    return {
      content: message.content || null,
      toolCalls:
        message.tool_calls && message.tool_calls.length > 0
          ? message.tool_calls
          : null
    };
  }

  throw new Error(lastError || "LLM request failed after retries");
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

function trimMessages(messages: ChatMessage[]): void {
  let toolCount = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "tool") {
      toolCount++;
      if (toolCount > KEEP_RECENT_TOOL_RESULTS) {
        const content = msg.content;
        if (typeof content === "string" && content.length > 200) {
          msg.content =
            content.substring(0, 200) +
            "...(truncated, " +
            content.length +
            " chars total)";
        }
      }
    }
  }

  // Keep only the most recent screenshot user message; replace older ones
  let screenshotCount = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "user" && Array.isArray(msg.content)) {
      const hasImage = msg.content.some(
        (p: ContentPart) => p.type === "image_url"
      );
      if (hasImage) {
        screenshotCount++;
        if (screenshotCount > 1) {
          msg.content = "[previous screenshot removed]";
        }
      }
    }
  }
}

// Global conversation buffer — persists across runAgent calls
let conversationMessages: ChatMessage[] | null = null;

function findSafeCutPoint(messages: ChatMessage[], idealCut: number): number {
  for (let i = idealCut; i > 1; i--) {
    if (
      messages[i].role === "user" &&
      typeof messages[i].content === "string"
    ) {
      return i;
    }
  }
  return idealCut;
}

function buildDigest(messages: ChatMessage[]): string {
  const parts: string[] = [];
  for (const msg of messages) {
    const role = msg.role;
    let text: string;
    if (typeof msg.content === "string") {
      text = msg.content;
    } else if (Array.isArray(msg.content)) {
      text = msg.content
        .filter((p: ContentPart) => p.type === "text" && p.text)
        .map((p: ContentPart) => p.text!)
        .join("\n");
      if (!text) text = "[image]";
    } else {
      text = "";
    }
    if (role === "tool" && text.length > 200) {
      text = text.substring(0, 200) + "...(truncated)";
    }
    if (role === "assistant" && msg.tool_calls) {
      const calls = msg.tool_calls
        .map(
          (tc: ToolCall) =>
            tc.function.name +
            "(" +
            tc.function.arguments.substring(0, 100) +
            ")"
        )
        .join("; ");
      text = (text ? text + "\n" : "") + "[called: " + calls + "]";
    }
    if (text) {
      parts.push(role + ": " + text);
    }
  }
  return parts.join("\n\n");
}

function compactConversation(messages: ChatMessage[], config: LlmConfig): void {
  if (messages.length <= COMPACT_THRESHOLD) return;

  const idealCut = messages.length - COMPACT_KEEP_RECENT;
  const cutPoint = findSafeCutPoint(messages, idealCut);

  if (cutPoint <= 1) return;

  const toCompact = messages.slice(1, cutPoint);
  const digest = buildDigest(toCompact);

  agentLog(
    "[COMPACT] Summarizing " +
      toCompact.length +
      " messages (cut at " +
      cutPoint +
      ")..."
  );

  let summary: string;
  try {
    const result = callLLM(
      [
        {
          role: "system",
          content:
            "Summarize this conversation between a user and a mobile automation assistant.\n" +
            "Focus on: tasks requested, what was accomplished, current phone state, important context for continuing.\n" +
            "Be concise (under 500 words). Output only the summary."
        },
        { role: "user", content: digest }
      ],
      [],
      config
    );
    summary = result.content || "(empty summary)";
  } catch (e: any) {
    agentLog(
      "[COMPACT] Summarization failed: " +
        e.message +
        " — falling back to truncation"
    );
    // Fallback: just keep recent messages without summary
    const system = messages[0];
    const recent = messages.slice(cutPoint);
    messages.length = 0;
    messages.push(system, ...recent);
    return;
  }

  agentLog(
    "[COMPACT] Summary (" +
      summary.length +
      " chars): " +
      summary.substring(0, 100) +
      "..."
  );

  const summaryMsg: ChatMessage = {
    role: "user",
    content: "[Prior conversation summary]\n" + summary
  };

  // Splice: replace messages[1..cutPoint) with summaryMsg
  messages.splice(1, cutPoint - 1, summaryMsg);
}

// Main entry point — called from Java via nativeEvaluateJS
function runAgent(task: string, configJson: string): string {
  const config: LlmConfig = JSON.parse(configJson);

  agentLog("[TASK] Received task: " + task);

  if (!conversationMessages) {
    conversationMessages = [{ role: "system", content: SYSTEM_PROMPT }];
  }
  conversationMessages.push({ role: "user", content: task });
  compactConversation(conversationMessages, config);

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
          });
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
