"use strict";
(() => {
  // src/agent-standalone.ts
  var MAX_STEPS = 30;
  var KEEP_RECENT_TOOL_RESULTS = 3;
  var MAX_GET_SCREEN_PER_EXEC = 5;
  var EXEC_TIMEOUT_MS = 3e4;
  var SYSTEM_PROMPT = `You are a mobile automation assistant controlling a phone via Android Accessibility Service.
You can see the screen's accessibility tree where each node shows: class type, text, content description (desc), bounds coordinates, and properties.

You operate by writing JavaScript code using the execute_js tool. All calls are synchronous. Available global functions:
- get_screen() \u2192 returns accessibility tree as string
- click(target) \u2192 click by text: click("OK"), by desc: click({desc:"\u52A0"}), or by coords: click({x:100,y:200})
- long_click(target) \u2192 same syntax as click, but long-press
- scroll(direction) \u2192 "up"/"down"/"left"/"right"
- type_text(text) \u2192 type into focused input
- press_home() / press_back() \u2192 navigation
- press_recents() \u2192 open recent tasks list (for switching apps)
- show_notifications() \u2192 pull down notification shade
- launch_app(name) \u2192 launch app by name or package name
- list_apps() \u2192 returns installed launchable apps, one per line: "AppName (package.name)"
- scroll_element(text, direction) \u2192 scroll a specific scrollable element found by text
- sleep(ms) \u2192 wait for UI to settle
- log(msg) \u2192 log a message

Tips:
- Execute a SHORT sequence of actions (5-10 operations max), then return the result
- Do NOT write for/while loops that call get_screen() or scroll() repeatedly
- get_screen() is limited to 5 calls per execute_js
- Use globalThis to store state between calls
- click("text") matches BOTH text and desc attributes. Use click({desc:"X"}) for desc-only match
- Bounds format: [left,top][right,bottom]. Center: x=(left+right)/2, y=(top+bottom)/2
- After actions, call sleep(500) then get_screen() to verify results
- If click by text fails, calculate coordinates from bounds and use click({x, y})
- To open an app, prefer launch_app("AppName") over navigating the home screen
- For NumberPicker/time selectors, use scroll_element("\u5F53\u524D\u503C", "up"/"down") to change values
- When the task is complete, respond with a text summary (no tool call)`;
  var TOOLS = [
    {
      type: "function",
      function: {
        name: "execute_js",
        description:
          "Execute JavaScript code. Execute a short linear sequence of actions. Available globals: get_screen(), click(target), long_click(target), scroll(dir), type_text(text), press_home(), press_back(), press_recents(), show_notifications(), launch_app(name), list_apps(), scroll_element(text, dir), sleep(ms), log(msg). get_screen() is limited to 5 calls per execution. Do NOT use loops to scroll and check screen repeatedly. The globalThis object persists across calls - use it to store context.",
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
  function executeCode(code) {
    const actionLog = [];
    let getScreenCount = 0;
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
    try {
      const result = (0, eval)(code);
      const resultStr = result === void 0 ? "undefined" : String(result);
      if (actionLog.length > 0) {
        actionLog.push("[Script returned] " + resultStr);
        return actionLog.join("\n");
      }
      return resultStr;
    } catch (e) {
      const error = "[JS Error] " + (e.message || String(e));
      if (actionLog.length > 0) {
        actionLog.push(error);
        return actionLog.join("\n");
      }
      return error;
    } finally {
      globalThis.get_screen = origGetScreen;
    }
  }
  function trimMessages(messages) {
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
          let result;
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
  globalThis.runAgent = runAgent;
})();
