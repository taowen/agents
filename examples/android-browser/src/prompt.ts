/**
 * prompt.ts
 *
 * System prompt and tool definitions for the browser automation agent.
 */

export const SYSTEM_PROMPT =
  "You are a web browser automation assistant. You see the page's DOM tree " +
  "where interactive elements have numeric IDs like [1], [2].\n\n" +
  "Workflow: call get_page() to see what's on the current page, then interact by element ID.\n\n" +
  "Tips:\n" +
  "- get_page() returns interactive elements (links, buttons, inputs) and text content\n" +
  "- click(id) clicks the element — use for links, buttons, checkboxes\n" +
  "- type(id, text) clears the input and types new text\n" +
  "- goto_url(url) navigates to a new page\n" +
  "- After actions that change the page, call get_page() again to see the new state\n" +
  "- Use screenshot() when DOM text doesn't give enough visual context (e.g. images, layout)\n" +
  "- scroll(direction) scrolls the page — use 'up' or 'down'\n" +
  "- When done, respond with a text summary (no tool call)";

export const TOOLS = [
  {
    type: "function",
    function: {
      name: "get_page",
      description:
        "Get the current page's DOM tree with interactive elements marked by numeric IDs. " +
        "Also shows text content for context. Call this first to understand the page.",
      parameters: {
        type: "object",
        properties: {}
      }
    }
  },
  {
    type: "function",
    function: {
      name: "click",
      description:
        "Click an interactive element by its numeric ID from the DOM tree.",
      parameters: {
        type: "object",
        properties: {
          id: {
            type: "number",
            description: "The numeric element ID from the DOM tree"
          }
        },
        required: ["id"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "type",
      description:
        "Clear an input field and type new text into it, identified by numeric ID.",
      parameters: {
        type: "object",
        properties: {
          id: {
            type: "number",
            description: "The numeric element ID from the DOM tree"
          },
          text: {
            type: "string",
            description: "The text to type into the input"
          }
        },
        required: ["id", "text"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "goto_url",
      description: "Navigate the browser to a new URL.",
      parameters: {
        type: "object",
        properties: {
          url: {
            type: "string",
            description: "The URL to navigate to"
          }
        },
        required: ["url"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "scroll",
      description: "Scroll the page up or down.",
      parameters: {
        type: "object",
        properties: {
          direction: {
            type: "string",
            enum: ["up", "down"],
            description: "Scroll direction"
          }
        },
        required: ["direction"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "go_back",
      description: "Go back to the previous page in browser history.",
      parameters: {
        type: "object",
        properties: {}
      }
    }
  },
  {
    type: "function",
    function: {
      name: "screenshot",
      description:
        "Capture a screenshot of the current page. Use when DOM text alone doesn't provide enough context.",
      parameters: {
        type: "object",
        properties: {}
      }
    }
  },
  {
    type: "function",
    function: {
      name: "wait",
      description:
        "Wait for the page to settle after an action (e.g. navigation, dynamic content loading).",
      parameters: {
        type: "object",
        properties: {
          ms: {
            type: "number",
            description: "Milliseconds to wait"
          }
        },
        required: ["ms"]
      }
    }
  }
];
