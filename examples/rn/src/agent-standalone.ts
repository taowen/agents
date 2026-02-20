/**
 * agent-standalone.ts
 *
 * Self-contained agent loop that runs in a standalone Hermes runtime
 * (inside the AccessibilityService process). No imports — all globals
 * are provided by C++ host functions registered in standalone_hermes.cpp:
 *
 *   get_screen(), click(target), long_click(target), scroll(dir),
 *   scroll_element(text, dir), type_text(text), press_home(), press_back(),
 *   press_recents(), show_notifications(), launch_app(name), list_apps(),
 *   sleep(ms), log(msg), http_post(url, headersJson, body)
 */

// Declare globals provided by C++ host functions (for TypeScript only)
declare function get_screen(): string;
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

const MAX_STEPS = 30;
const KEEP_RECENT_TOOL_RESULTS = 3;
const MAX_GET_SCREEN_PER_EXEC = 5;
const EXEC_TIMEOUT_MS = 30_000;

const SYSTEM_PROMPT =
  "You are a mobile automation assistant controlling a phone via Android Accessibility Service.\n" +
  "You can see the screen's accessibility tree where each node shows: class type, text, content description (desc), bounds coordinates, and properties.\n\n" +
  "You operate by writing JavaScript code using the execute_js tool. All calls are synchronous. Available global functions:\n" +
  "- get_screen() → returns accessibility tree as string\n" +
  '- click(target) → click by text: click("OK"), by desc: click({desc:"加"}), or by coords: click({x:100,y:200})\n' +
  "- long_click(target) → same syntax as click, but long-press\n" +
  '- scroll(direction) → "up"/"down"/"left"/"right"\n' +
  "- type_text(text) → type into focused input\n" +
  "- press_home() / press_back() → navigation\n" +
  "- press_recents() → open recent tasks list (for switching apps)\n" +
  "- show_notifications() → pull down notification shade\n" +
  "- launch_app(name) → launch app by name or package name\n" +
  '- list_apps() → returns installed launchable apps, one per line: "AppName (package.name)"\n' +
  "- scroll_element(text, direction) → scroll a specific scrollable element found by text\n" +
  "- sleep(ms) → wait for UI to settle\n" +
  "- log(msg) → log a message\n\n" +
  "Tips:\n" +
  "- Execute a SHORT sequence of actions (5-10 operations max), then return the result\n" +
  "- Do NOT write for/while loops that call get_screen() or scroll() repeatedly\n" +
  "- get_screen() is limited to 5 calls per execute_js\n" +
  "- Use globalThis to store state between calls\n" +
  '- click("text") matches BOTH text and desc attributes. Use click({desc:"X"}) for desc-only match\n' +
  "- Bounds format: [left,top][right,bottom]. Center: x=(left+right)/2, y=(top+bottom)/2\n" +
  "- After actions, call sleep(500) then get_screen() to verify results\n" +
  "- If click by text fails, calculate coordinates from bounds and use click({x, y})\n" +
  '- To open an app, prefer launch_app("AppName") over navigating the home screen\n' +
  '- For NumberPicker/time selectors, use scroll_element("当前值", "up"/"down") to change values\n' +
  "- When the task is complete, respond with a text summary (no tool call)";

const TOOLS = [
  {
    type: "function",
    function: {
      name: "execute_js",
      description:
        "Execute JavaScript code. Execute a short linear sequence of actions. " +
        "Available globals: get_screen(), click(target), long_click(target), scroll(dir), " +
        "type_text(text), press_home(), press_back(), press_recents(), show_notifications(), " +
        "launch_app(name), list_apps(), scroll_element(text, dir), " +
        "sleep(ms), log(msg). get_screen() is limited to 5 calls per execution. " +
        "Do NOT use loops to scroll and check screen repeatedly. " +
        "The globalThis object persists across calls - use it to store context.",
      parameters: {
        type: "object",
        properties: {
          code: {
            type: "string",
            description:
              "JavaScript code to execute. All screen automation functions are available as globals."
          }
        },
        required: ["code"]
      }
    }
  }
];

interface ChatMessage {
  role: string;
  content: string | null;
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
  tools: typeof TOOLS,
  config: LlmConfig
): { content: string | null; toolCalls: ToolCall[] | null } {
  let apiUrl = config.baseURL;
  if (apiUrl.endsWith("/")) apiUrl = apiUrl.slice(0, -1);
  apiUrl += "/chat/completions";

  const body = JSON.stringify({
    model: config.model,
    messages: messages,
    tools: tools
  });

  const headers = JSON.stringify({
    Authorization: "Bearer " + config.apiKey,
    "Content-Type": "application/json"
  });

  const MAX_RETRIES = 2;
  const RETRY_DELAY_MS = 2000;
  let lastError = "";

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      sleep(RETRY_DELAY_MS);
    }

    const responseStr = http_post(apiUrl, headers, body);
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

function executeCode(code: string): string {
  const actionLog: string[] = [];
  let getScreenCount = 0;
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

  try {
    // Use indirect eval to run in global scope
    const result = (0, eval)(code);
    const resultStr = result === undefined ? "undefined" : String(result);
    if (actionLog.length > 0) {
      actionLog.push("[Script returned] " + resultStr);
      return actionLog.join("\n");
    }
    return resultStr;
  } catch (e: any) {
    const error = "[JS Error] " + (e.message || String(e));
    if (actionLog.length > 0) {
      actionLog.push(error);
      return actionLog.join("\n");
    }
    return error;
  } finally {
    // Restore original
    (globalThis as any).get_screen = origGetScreen;
  }
}

function trimMessages(messages: ChatMessage[]): void {
  let toolCount = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "tool") {
      toolCount++;
      if (toolCount > KEEP_RECENT_TOOL_RESULTS) {
        const content = msg.content || "";
        if (content.length > 200) {
          msg.content =
            content.substring(0, 200) +
            "...(truncated, " +
            content.length +
            " chars total)";
        }
      }
    }
  }
}

// Main entry point — called from Java via nativeEvaluateJS
function runAgent(task: string, configJson: string): string {
  const config: LlmConfig = JSON.parse(configJson);

  agentLog("[TASK] Received task: " + task);

  const messages: ChatMessage[] = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: task }
  ];

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
        let result: string;
        if (toolCall.function.name === "execute_js") {
          const args = JSON.parse(toolCall.function.arguments);
          result = executeCode(args.code);
        } else {
          result = "Unknown tool: " + toolCall.function.name;
        }

        const logResult =
          result.length > 200
            ? result.substring(0, 200) + "... (" + result.length + " chars)"
            : result;
        agentLog("[TOOL] " + toolCall.function.name + " -> " + logResult);

        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: result
        });
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
