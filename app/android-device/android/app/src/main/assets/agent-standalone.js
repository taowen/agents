"use strict";
(() => {
  // src/host-api.ts
  var HOST_FUNCTIONS = [
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
        'click by text: click("OK"), by desc: click({desc:"\u52A0"}), or by coords: click({x:100,y:200})',
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
  function generateSignatures(filter) {
    return HOST_FUNCTIONS.filter(filter ?? (() => true))
      .map((fn) => {
        const params = fn.params.map((p) => `${p.name}: ${p.type}`).join(", ");
        return `function ${fn.name}(${params}): ${fn.returns};  // ${fn.description}`;
      })
      .join("\n");
  }

  // src/app-prompt.ts
  var agentSignatures = generateSignatures((fn) => fn.agentVisible);
  var SYSTEM_PROMPT =
    "You are a mobile automation assistant controlling a phone via Android Accessibility Service.\nYou can see the screen's accessibility tree where each node shows: class type, text, content description (desc), bounds coordinates, and properties.\n\nYou operate by writing JavaScript code using the execute_js tool. All calls are synchronous. Available global functions:\n```typescript\n" +
    agentSignatures +
    '\n```\n\nTips:\n- Execute a SHORT sequence of actions (5-10 operations max). The last expression\'s value is the result \u2014 do NOT use `return` (code runs in global scope, not a function)\n- Do NOT write for/while loops that call get_screen() or scroll() repeatedly\n- get_screen() is limited to 5 calls per execute_js\n- Use globalThis to store state between calls\n- click("text") matches BOTH text and desc attributes. Use click({desc:"X"}) for desc-only match\n- Bounds format: [left,top][right,bottom]. Center: x=(left+right)/2, y=(top+bottom)/2\n- After actions, call sleep(500) then get_screen() to verify results\n- If elements (especially ImageView) have no text or desc, call take_screenshot() to see actual pixels\n- take_screenshot() returns a placeholder; the actual image is automatically sent to you as a vision input\n- If click by text fails, calculate coordinates from bounds and use click({x, y})\n- To open an app, prefer launch_app("AppName") over navigating the home screen\n- For NumberPicker/time selectors, use scroll_element("\u5F53\u524D\u503C", "up"/"down") to change values\n- When you encounter ambiguity (e.g. multiple matches) or need user action (e.g. password input), call ask_user("your question") to pause and let the user act, then continue after they tap Continue\n- IMPORTANT: ALWAYS use execute_js to interact with the phone. Never describe or narrate planned actions \u2014 execute them directly.\n- When the task is complete, respond with a text summary (no tool call)';
  var toolSignatures = generateSignatures((fn) => fn.agentVisible);
  var TOOLS = [
    {
      type: "function",
      function: {
        name: "execute_js",
        description:
          "Execute JavaScript code. Execute a short linear sequence of actions. Available globals:\n" +
          toolSignatures +
          "\nget_screen() is limited to 5 calls per execution. Do NOT use loops to scroll and check screen repeatedly. The globalThis object persists across calls - use it to store context.",
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
  globalThis.__DEVICE_PROMPT__ = {
    systemPrompt: SYSTEM_PROMPT,
    tools: TOOLS
  };

  // src/agent-standalone.ts
  var MAX_GET_SCREEN_PER_EXEC = 5;
  var EXEC_TIMEOUT_MS = 3e4;
  function wrapHostFunctions(opts) {
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
    globalThis.get_screen = function () {
      if (Date.now() > opts.deadline)
        throw new Error("Script execution timeout");
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
    globalThis.take_screenshot = function () {
      if (Date.now() > opts.deadline)
        throw new Error("Script execution timeout");
      update_status("take screenshot");
      const b64 = origTakeScreenshot();
      if (b64.startsWith("ERROR:")) {
        return b64;
      }
      opts.onScreenshot(b64);
      return "screenshot captured - image will be sent to you";
    };
    globalThis.click = function (target) {
      const desc =
        typeof target === "string"
          ? target
          : target.desc || "(" + target.x + "," + target.y + ")";
      update_status("click " + desc);
      return origClick(target);
    };
    globalThis.long_click = function (target) {
      const desc =
        typeof target === "string"
          ? target
          : target.desc || "(" + target.x + "," + target.y + ")";
      update_status("long click " + desc);
      return origLongClick(target);
    };
    globalThis.type_text = function (text) {
      update_status("type " + text.substring(0, 15));
      return origTypeText(text);
    };
    globalThis.launch_app = function (name) {
      update_status("launch " + name);
      return origLaunchApp(name);
    };
    globalThis.scroll = function (direction) {
      update_status("scroll " + direction);
      return origScroll(direction);
    };
    globalThis.scroll_element = function (text, direction) {
      update_status("scroll " + text + " " + direction);
      return origScrollElement(text, direction);
    };
    globalThis.press_home = function () {
      update_status("press home");
      return origPressHome();
    };
    globalThis.press_back = function () {
      update_status("press back");
      return origPressBack();
    };
    globalThis.press_recents = function () {
      update_status("press recents");
      return origPressRecents();
    };
    globalThis.show_notifications = function () {
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
  function executeCodeForServer(code) {
    const capturedScreenshots = [];
    let lastGetScreenResult = null;
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
      if (result === void 0 && lastGetScreenResult !== null) {
        result = lastGetScreenResult;
      }
      return {
        result: result === void 0 ? "undefined" : String(result),
        screenshots: capturedScreenshots
      };
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("'return' not in a function")) {
        return {
          result:
            "[JS Error] Top-level 'return' is not allowed. The code runs in global scope \u2014 use the last expression as the result instead of 'return'.",
          screenshots: capturedScreenshots
        };
      }
      return {
        result: "[JS Error] " + msg,
        screenshots: capturedScreenshots
      };
    } finally {
      wrapped.restore();
    }
  }
  globalThis.executeCodeForServer = executeCodeForServer;
})();
