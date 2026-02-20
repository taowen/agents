import AccessibilityBridge from "./NativeAccessibilityBridge";
import { executeCode } from "./agentGlobals";
import { LlmClient } from "./LlmClient";
import type {
  ChatMessage,
  LlmConfig,
  LogCallback,
  ToolCall,
  ToolDefinition
} from "./types";

const MAX_STEPS = 30;
const KEEP_RECENT_TOOL_RESULTS = 3;

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

const TOOLS: ToolDefinition[] = [
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

function formatTime(): string {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}

export class AgentLoop {
  private abortController: AbortController | null = null;
  private running = false;

  get isRunning(): boolean {
    return this.running;
  }

  abort(): void {
    this.abortController?.abort();
  }

  async execute(
    task: string,
    config: LlmConfig,
    onLog: LogCallback
  ): Promise<void> {
    if (this.running) {
      onLog(`[${formatTime()}] [ERROR] Agent is already running`);
      return;
    }

    this.running = true;
    this.abortController = new AbortController();
    const signal = this.abortController.signal;

    const log = (msg: string) => {
      onLog(`[${formatTime()}] ${msg}`);
    };

    try {
      await AccessibilityBridge.resetScreens();

      let apiUrl = config.baseURL;
      if (apiUrl.endsWith("/")) {
        apiUrl = apiUrl.slice(0, -1);
      }
      apiUrl += "/chat/completions";

      const llmClient = new LlmClient(apiUrl, config.apiKey, config.model);

      log(`[TASK] Received task: ${task}`);

      const messages: ChatMessage[] = [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: task }
      ];

      for (let step = 1; step <= MAX_STEPS; step++) {
        if (signal.aborted) {
          log("[ABORT] Task aborted by user");
          return;
        }

        log(`[STEP ${step}] Calling LLM...`);
        trimMessages(messages);

        const response = await llmClient.chat(messages, TOOLS, signal);

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
          log(`[STEP ${step}] LLM returned tool_calls: ${toolNames}`);

          for (const toolCall of response.toolCalls) {
            if (signal.aborted) {
              log("[ABORT] Task aborted by user");
              return;
            }

            const args = JSON.parse(toolCall.function.arguments);
            const result = executeTool(toolCall.function.name, args);

            const logResult =
              result.length > 200
                ? `${result.substring(0, 200)}... (${result.length} chars)`
                : result;
            log(`[TOOL] ${toolCall.function.name} -> ${logResult}`);

            messages.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: result
            });
          }
        } else {
          const finalContent = response.content ?? "(no response)";
          log(`[DONE] Task completed: ${finalContent}`);
          return;
        }
      }

      log(`[ERROR] Reached max steps (${MAX_STEPS}) without completing task`);
    } catch (e: any) {
      if (e.name === "AbortError") {
        log("[ABORT] Task aborted by user");
      } else {
        log(`[ERROR] Agent loop error: ${e.message}`);
      }
    } finally {
      this.running = false;
      this.abortController = null;
    }
  }
}

function trimMessages(messages: ChatMessage[]): void {
  let toolCount = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "tool") {
      toolCount++;
      if (toolCount > KEEP_RECENT_TOOL_RESULTS) {
        const content = msg.content ?? "";
        if (content.length > 200) {
          msg.content = `${content.substring(0, 200)}...(truncated, ${content.length} chars total)`;
        }
      }
    }
  }
}

function executeTool(name: string, args: Record<string, string>): string {
  if (name === "execute_js") {
    return executeCode(args.code);
  }
  return `Unknown tool: ${name}`;
}
