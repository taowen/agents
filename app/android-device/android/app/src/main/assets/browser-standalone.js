"use strict";
(() => {
  // src/browser-prompt.ts
  var SYSTEM_PROMPT =
    "You are a web browser automation assistant. You see the page's DOM tree where interactive elements have numeric IDs like [1], [2].\n\nYou operate by writing JavaScript code using the execute_js tool. All calls are synchronous. Available global functions:\n```typescript\nfunction get_page(): string;  // returns DOM tree with interactive element IDs\nfunction click_element(id: number): boolean;  // click element by numeric ID\nfunction type_text(id: number, text: string): boolean;  // type into element by ID\nfunction goto_url(url: string): boolean;  // navigate to URL (waits for load)\nfunction scroll_page(direction: string): boolean;  // scroll 'up' or 'down'\nfunction go_back(): boolean;  // browser history back\nfunction take_screenshot(): string;  // capture page as base64 JPEG\nfunction switch_ua(mode: string): string;  // switch 'mobile' or 'pc' user-agent\nfunction set_viewport(width: number, height: number): string;  // resize viewport (0,0 to restore)\nfunction sleep(ms: number): void;  // wait for UI to settle\nfunction log(msg: string): void;  // log a message\nfunction ask_user(question: string): string;  // show overlay to ask user, blocks until response\n```\n\nTips:\n- get_page() returns interactive elements (links, buttons, inputs) and text content\n- click_element(id) clicks the element \u2014 use for links, buttons, checkboxes\n- type_text(id, text) clears the input and types new text\n- goto_url(url) navigates to a new page and waits for load\n- click_element and go_back automatically wait for page load\n- After actions that change the page, call get_page() once to see the new state\n- take_screenshot() captures the page \u2014 use when DOM text doesn't give enough context\n- scroll_page('up'/'down') scrolls the page\n- The browser defaults to mobile User-Agent. Use switch_ua('pc') for desktop sites\n- Use set_viewport(1920, 1080) for full PC-width rendering, set_viewport(0, 0) to restore\n- Execute a SHORT sequence of actions (5-10 operations max). The last expression's value is the result \u2014 do NOT use `return` (code runs in global scope, not a function)\n- get_page() is limited to 5 calls per execute_js\n- Use globalThis to store state between calls\n- When you need user help (e.g. login credentials, CAPTCHA, confirmation), call ask_user('your question') \u2014 it shows an overlay and blocks until the user taps Continue or Abandon\n- IMPORTANT: ALWAYS use execute_js to interact with the browser. Never describe planned actions \u2014 execute them directly.\n- When the task is complete, respond with a text summary (no tool call)";
  var TOOLS = [
    {
      type: "function",
      function: {
        name: "execute_js",
        description:
          "Execute JavaScript code in the browser agent. Available globals:\nget_page(), click_element(id), type_text(id, text), goto_url(url), scroll_page(direction), go_back(), take_screenshot(), switch_ua(mode), set_viewport(w, h), sleep(ms), log(msg), ask_user(question)\nget_page() is limited to 5 calls per execution. The globalThis object persists across calls.",
        parameters: {
          type: "object",
          properties: {
            code: {
              type: "string",
              description:
                "JavaScript code to execute. Browser automation functions are available as globals."
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

  // src/browser-standalone.ts
  var MAX_GET_PAGE_PER_EXEC = 5;
  var EXEC_TIMEOUT_MS = 3e4;
  function executeCodeForServer(code) {
    const capturedScreenshots = [];
    let lastGetPageResult = null;
    let getPageCount = 0;
    const deadline = Date.now() + EXEC_TIMEOUT_MS;
    const origGetPage = get_page;
    const origClickElement = click_element;
    const origTypeText = type_text;
    const origGotoUrl = goto_url;
    const origScrollPage = scroll_page;
    const origGoBack = go_back;
    const origTakeScreenshot = take_screenshot;
    const origSwitchUa = switch_ua;
    const origSetViewport = set_viewport;
    globalThis.get_page = function () {
      if (Date.now() > deadline) throw new Error("Script execution timeout");
      getPageCount++;
      if (getPageCount > MAX_GET_PAGE_PER_EXEC) {
        throw new Error(
          "get_page() called " +
            getPageCount +
            " times. Max is " +
            MAX_GET_PAGE_PER_EXEC +
            ". Return and plan next actions in a new execute_js call."
        );
      }
      update_status("read page");
      const tree = origGetPage();
      lastGetPageResult = tree;
      return tree;
    };
    globalThis.click_element = function (id) {
      if (Date.now() > deadline) throw new Error("Script execution timeout");
      update_status("click [" + id + "]");
      return origClickElement(id);
    };
    globalThis.type_text = function (id, text) {
      if (Date.now() > deadline) throw new Error("Script execution timeout");
      update_status("type [" + id + "] " + text.substring(0, 15));
      return origTypeText(id, text);
    };
    globalThis.goto_url = function (url) {
      if (Date.now() > deadline) throw new Error("Script execution timeout");
      update_status("goto " + url.substring(0, 30));
      return origGotoUrl(url);
    };
    globalThis.scroll_page = function (direction) {
      if (Date.now() > deadline) throw new Error("Script execution timeout");
      update_status("scroll " + direction);
      return origScrollPage(direction);
    };
    globalThis.go_back = function () {
      if (Date.now() > deadline) throw new Error("Script execution timeout");
      update_status("go back");
      return origGoBack();
    };
    globalThis.take_screenshot = function () {
      if (Date.now() > deadline) throw new Error("Script execution timeout");
      update_status("take screenshot");
      const b64 = origTakeScreenshot();
      if (b64.startsWith("ERROR:")) {
        return b64;
      }
      capturedScreenshots.push(b64);
      return "screenshot captured - image will be sent to you";
    };
    globalThis.switch_ua = function (mode) {
      if (Date.now() > deadline) throw new Error("Script execution timeout");
      update_status("switch UA " + mode);
      return origSwitchUa(mode);
    };
    globalThis.set_viewport = function (width, height) {
      if (Date.now() > deadline) throw new Error("Script execution timeout");
      update_status("set viewport " + width + "x" + height);
      return origSetViewport(width, height);
    };
    try {
      let result = (0, eval)(code);
      if (result === void 0 && lastGetPageResult !== null) {
        result = lastGetPageResult;
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
      globalThis.get_page = origGetPage;
      globalThis.click_element = origClickElement;
      globalThis.type_text = origTypeText;
      globalThis.goto_url = origGotoUrl;
      globalThis.scroll_page = origScrollPage;
      globalThis.go_back = origGoBack;
      globalThis.take_screenshot = origTakeScreenshot;
      globalThis.switch_ua = origSwitchUa;
      globalThis.set_viewport = origSetViewport;
    }
  }
  globalThis.executeCodeForServer = executeCodeForServer;
})();
