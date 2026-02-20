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

  // src/prompt.ts
  var agentSignatures = generateSignatures((fn) => fn.agentVisible);
  var SYSTEM_PROMPT =
    "You are a mobile automation assistant controlling a phone via Android Accessibility Service.\nYou can see the screen's accessibility tree where each node shows: class type, text, content description (desc), bounds coordinates, and properties.\n\nYou operate by writing JavaScript code using the execute_js tool. All calls are synchronous. Available global functions:\n```typescript\n" +
    agentSignatures +
    '\n```\n\nTips:\n- Execute a SHORT sequence of actions (5-10 operations max), then return the result\n- Do NOT write for/while loops that call get_screen() or scroll() repeatedly\n- get_screen() is limited to 5 calls per execute_js\n- Use globalThis to store state between calls\n- click("text") matches BOTH text and desc attributes. Use click({desc:"X"}) for desc-only match\n- Bounds format: [left,top][right,bottom]. Center: x=(left+right)/2, y=(top+bottom)/2\n- After actions, call sleep(500) then get_screen() to verify results\n- If elements (especially ImageView) have no text or desc, call take_screenshot() to see actual pixels\n- take_screenshot() returns a placeholder; the actual image is automatically sent to you as a vision input\n- If click by text fails, calculate coordinates from bounds and use click({x, y})\n- To open an app, prefer launch_app("AppName") over navigating the home screen\n- For NumberPicker/time selectors, use scroll_element("\u5F53\u524D\u503C", "up"/"down") to change values\n- When the task is complete, respond with a text summary (no tool call)';
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

  // src/agent-standalone.ts
  var MAX_STEPS = 30;
  var KEEP_RECENT_TOOL_RESULTS = 3;
  var MAX_GET_SCREEN_PER_EXEC = 5;
  var EXEC_TIMEOUT_MS = 3e4;
  function formatTime() {
    const d = /* @__PURE__ */ new Date();
    const pad = (n) => (n < 10 ? "0" : "") + n;
    return (
      pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds())
    );
  }
  function agentLog(msg) {
    const line = "[" + formatTime() + "] " + msg;
    log(line);
  }
  function callLLM(messages, tools, config) {
    let apiUrl = config.baseURL;
    if (apiUrl.endsWith("/")) apiUrl = apiUrl.slice(0, -1);
    apiUrl += "/chat/completions";
    const body = JSON.stringify({
      model: config.model,
      messages,
      tools
    });
    const headers = JSON.stringify({
      Authorization: "Bearer " + config.apiKey,
      "Content-Type": "application/json"
    });
    const MAX_RETRIES = 2;
    const RETRY_DELAY_MS = 2e3;
    let lastError = "";
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      if (attempt > 0) {
        sleep(RETRY_DELAY_MS);
      }
      const responseStr = http_post(apiUrl, headers, body);
      let data;
      try {
        data = JSON.parse(responseStr);
      } catch (e) {
        lastError =
          "Failed to parse LLM response: " + responseStr.substring(0, 200);
        if (attempt < MAX_RETRIES) continue;
        throw new Error(lastError);
      }
      if (data.error) {
        lastError = "LLM API error: " + JSON.stringify(data.error);
        const status = data.error.status || data.error.code || 0;
        if ((status === 429 || status >= 500) && attempt < MAX_RETRIES)
          continue;
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
  function executeCode(code, thinking) {
    if (thinking) {
      agentLog("[THINK] " + thinking);
    }
    agentLog("[CODE] " + code);
    const actionLog = [];
    let getScreenCount = 0;
    let capturedScreenshot;
    const deadline = Date.now() + EXEC_TIMEOUT_MS;
    const origGetScreen = get_screen;
    globalThis.get_screen = function () {
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
    globalThis.take_screenshot = function () {
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
      const result = (0, eval)(code);
      const resultStr = result === void 0 ? "undefined" : String(result);
      let text;
      if (actionLog.length > 0) {
        actionLog.push("[Script returned] " + resultStr);
        text = actionLog.join("\n");
      } else {
        text = resultStr;
      }
      return { text, screenshot: capturedScreenshot };
    } catch (e) {
      const error = "[JS Error] " + (e.message || String(e));
      let text;
      if (actionLog.length > 0) {
        actionLog.push(error);
        text = actionLog.join("\n");
      } else {
        text = error;
      }
      return { text, screenshot: capturedScreenshot };
    } finally {
      globalThis.get_screen = origGetScreen;
      globalThis.take_screenshot = origTakeScreenshot;
    }
  }
  function trimMessages(messages) {
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
    let screenshotCount = 0;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.role === "user" && Array.isArray(msg.content)) {
        const hasImage = msg.content.some((p) => p.type === "image_url");
        if (hasImage) {
          screenshotCount++;
          if (screenshotCount > 1) {
            msg.content = "[previous screenshot removed]";
          }
        }
      }
    }
  }
  function runAgent(task, configJson) {
    const config = JSON.parse(configJson);
    agentLog("[TASK] Received task: " + task);
    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: task }
    ];
    for (let step = 1; step <= MAX_STEPS; step++) {
      agentLog("[STEP " + step + "] Calling LLM...");
      trimMessages(messages);
      let response;
      try {
        response = callLLM(messages, TOOLS, config);
      } catch (e) {
        agentLog("[ERROR] LLM call failed: " + e.message);
        return "Error: " + e.message;
      }
      if (response.toolCalls) {
        const assistantMsg = {
          role: "assistant",
          content: response.content,
          tool_calls: response.toolCalls
        };
        messages.push(assistantMsg);
        const toolNames = response.toolCalls
          .map((tc) => tc.function.name)
          .join(", ");
        agentLog("[STEP " + step + "] LLM returned tool_calls: " + toolNames);
        for (const toolCall of response.toolCalls) {
          let resultText;
          let screenshot;
          if (toolCall.function.name === "execute_js") {
            const args = JSON.parse(toolCall.function.arguments);
            const execResult = executeCode(
              args.code,
              response.content ?? void 0
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
  globalThis.runAgent = runAgent;
})();
