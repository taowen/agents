export interface HostFunctionDef {
  name: string;
  params: { name: string; type: string }[];
  returns: string;
  description: string;
  /** Internal functions (e.g. http_post) are not shown to the LLM */
  agentVisible: boolean;
}

export const HOST_FUNCTIONS: HostFunctionDef[] = [
  {
    name: "get_screen",
    params: [],
    returns: "string",
    description: "returns accessibility tree as string",
    agentVisible: true
  },
  {
    name: "take_screenshot",
    params: [],
    returns: "string",
    description:
      "captures screen as JPEG, returns base64 (use when ImageView has no text/desc)",
    agentVisible: true
  },
  {
    name: "click",
    params: [
      {
        name: "target",
        type: "string | { desc?: string; x?: number; y?: number }"
      }
    ],
    returns: "boolean",
    description:
      'click by text: click("OK"), by desc: click({desc:"加"}), or by coords: click({x:100,y:200})',
    agentVisible: true
  },
  {
    name: "long_click",
    params: [
      {
        name: "target",
        type: "string | { desc?: string; x?: number; y?: number }"
      }
    ],
    returns: "boolean",
    description: "same syntax as click, but long-press",
    agentVisible: true
  },
  {
    name: "scroll",
    params: [{ name: "direction", type: '"up" | "down" | "left" | "right"' }],
    returns: "boolean",
    description: "scroll the screen in a direction",
    agentVisible: true
  },
  {
    name: "scroll_element",
    params: [
      { name: "text", type: "string" },
      { name: "direction", type: '"up" | "down" | "left" | "right"' }
    ],
    returns: "string",
    description: "scroll a specific scrollable element found by text",
    agentVisible: true
  },
  {
    name: "type_text",
    params: [{ name: "text", type: "string" }],
    returns: "boolean",
    description: "type into focused input",
    agentVisible: true
  },
  {
    name: "press_home",
    params: [],
    returns: "boolean",
    description: "press the home button",
    agentVisible: true
  },
  {
    name: "press_back",
    params: [],
    returns: "boolean",
    description: "press the back button",
    agentVisible: true
  },
  {
    name: "press_recents",
    params: [],
    returns: "boolean",
    description: "open recent tasks list (for switching apps)",
    agentVisible: true
  },
  {
    name: "show_notifications",
    params: [],
    returns: "boolean",
    description: "pull down notification shade",
    agentVisible: true
  },
  {
    name: "launch_app",
    params: [{ name: "name", type: "string" }],
    returns: "string",
    description: "launch app by name or package name",
    agentVisible: true
  },
  {
    name: "list_apps",
    params: [],
    returns: "string",
    description:
      'returns installed launchable apps, one per line: "AppName (package.name)"',
    agentVisible: true
  },
  {
    name: "sleep",
    params: [{ name: "ms", type: "number" }],
    returns: "void",
    description: "wait for UI to settle",
    agentVisible: true
  },
  {
    name: "log",
    params: [{ name: "msg", type: "string" }],
    returns: "void",
    description: "log a message",
    agentVisible: true
  },
  {
    name: "ask_user",
    params: [{ name: "question", type: "string" }],
    returns: "string",
    description:
      'show a question overlay and block until user responds. Returns "continue" or "abandoned". Use when you encounter ambiguity or need user action (e.g. password input)',
    agentVisible: true
  },
  {
    name: "update_status",
    params: [{ name: "text", type: "string" }],
    returns: "void",
    description: "update overlay status text",
    agentVisible: false
  },
  {
    name: "hide_overlay",
    params: [],
    returns: "void",
    description: "hide the overlay",
    agentVisible: false
  },
  {
    name: "http_post",
    params: [
      { name: "url", type: "string" },
      { name: "headersJson", type: "string" },
      { name: "body", type: "string" }
    ],
    returns: "string",
    description: "synchronous HTTP POST",
    agentVisible: false
  }
];

export function generateSignatures(
  filter?: (fn: HostFunctionDef) => boolean
): string {
  return HOST_FUNCTIONS.filter(filter ?? (() => true))
    .map((fn) => {
      const params = fn.params.map((p) => `${p.name}: ${p.type}`).join(", ");
      return `function ${fn.name}(${params}): ${fn.returns};  // ${fn.description}`;
    })
    .join("\n");
}
