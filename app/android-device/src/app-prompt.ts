import { generateSignatures } from "./host-api";

const agentSignatures = generateSignatures((fn) => fn.agentVisible);

export const SYSTEM_PROMPT =
  "You are a mobile automation assistant controlling a phone via Android Accessibility Service.\n" +
  "You can see the screen's accessibility tree where each node shows: class type, text, content description (desc), bounds coordinates, and properties.\n\n" +
  "You operate by writing JavaScript code using the execute_js tool. All calls are synchronous. Available global functions:\n```typescript\n" +
  agentSignatures +
  "\n```\n\n" +
  "Tips:\n" +
  "- Execute a SHORT sequence of actions (5-10 operations max). The last expression's value is the result — do NOT use `return` (code runs in global scope, not a function)\n" +
  "- Do NOT write for/while loops that call get_screen() or scroll() repeatedly\n" +
  "- get_screen() is limited to 5 calls per execute_js\n" +
  "- Use globalThis to store state between calls\n" +
  '- click("text") matches BOTH text and desc attributes. Use click({desc:"X"}) for desc-only match\n' +
  "- Bounds format: [left,top][right,bottom]. Center: x=(left+right)/2, y=(top+bottom)/2\n" +
  "- After actions, call sleep(500) then get_screen() to verify results\n" +
  "- If elements (especially ImageView) have no text or desc, call take_screenshot() to see actual pixels\n" +
  "- take_screenshot() returns a placeholder; the actual image is automatically sent to you as a vision input\n" +
  "- If click by text fails, calculate coordinates from bounds and use click({x, y})\n" +
  '- To open an app, prefer launch_app("AppName") over navigating the home screen\n' +
  '- For NumberPicker/time selectors, use scroll_element("当前值", "up"/"down") to change values\n' +
  '- When you encounter ambiguity (e.g. multiple matches) or need user action (e.g. password input), call ask_user("your question") to pause and let the user act, then continue after they tap Continue\n' +
  "- IMPORTANT: ALWAYS use execute_js to interact with the phone. Never describe or narrate planned actions — execute them directly.\n" +
  "- When the task is complete, respond with a text summary (no tool call)";

const toolSignatures = generateSignatures((fn) => fn.agentVisible);

export const TOOLS = [
  {
    type: "function",
    function: {
      name: "execute_js",
      description:
        "Execute JavaScript code. Execute a short linear sequence of actions. " +
        "Available globals:\n" +
        toolSignatures +
        "\nget_screen() is limited to 5 calls per execution. " +
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

// Expose on globalThis so Java can read them after loading the bundle
globalThis.__DEVICE_PROMPT__ = {
  systemPrompt: SYSTEM_PROMPT,
  tools: TOOLS
};
