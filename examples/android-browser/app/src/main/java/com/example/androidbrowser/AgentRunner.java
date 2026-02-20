package com.example.androidbrowser;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Agent loop running on a plain Java background thread.
 * Port of agent-standalone.ts — no WebView needed.
 */
public class AgentRunner implements Runnable {

    private static final String TAG = "AgentRunner";
    static final int MAX_STEPS = 30;
    static final int KEEP_RECENT_TOOL_RESULTS = 3;
    private static final int MAX_RETRIES = 2;
    private static final int RETRY_DELAY_MS = 2000;

    private final String task;
    private final String baseURL;
    private final String apiKey;
    private final AgentBridge bridge;
    private final Runnable onDone;

    // ---- System prompt (from prompt.ts) ----
    private static final String SYSTEM_PROMPT =
        "You are a web browser automation assistant. You see the page's DOM tree " +
        "where interactive elements have numeric IDs like [1], [2].\n\n" +
        "Workflow: call get_page() to see what's on the current page, then interact by element ID.\n\n" +
        "Tips:\n" +
        "- get_page() returns interactive elements (links, buttons, inputs) and text content\n" +
        "- click(id) clicks the element \u2014 use for links, buttons, checkboxes\n" +
        "- type(id, text) clears the input and types new text\n" +
        "- goto_url(url) navigates to a new page\n" +
        "- click, goto_url, and go_back automatically wait for page load \u2014 no manual waiting needed\n" +
        "- After actions that change the page, call get_page() once to see the new state \u2014 never call get_page() twice in a row\n" +
        "- Be efficient: one get_page() after each action is enough\n" +
        "- screenshot() captures the page at the current viewport size \u2014 use when DOM text doesn't give enough visual context\n" +
        "- scroll(direction) scrolls the page \u2014 use 'up' or 'down'\n" +
        "- The browser defaults to mobile User-Agent. Use switch_ua('pc') for desktop sites, switch_ua('mobile') to go back\n" +
        "- Use set_viewport(1920, 1080) to resize the viewport for full PC-width rendering, then screenshot() to see it. Use set_viewport(0, 0) to restore the phone default. All tools (click, type, get_page) work at the enlarged viewport\n" +
        "- When done, respond with a text summary (no tool call)";

    // ---- Tool definitions JSON (from prompt.ts) ----
    private static final String TOOLS_JSON = "["
        + "{\"type\":\"function\",\"function\":{\"name\":\"get_page\",\"description\":\"Get the current page's DOM tree with interactive elements marked by numeric IDs. Also shows text content for context. Call this first to understand the page.\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}},"
        + "{\"type\":\"function\",\"function\":{\"name\":\"click\",\"description\":\"Click an interactive element by its numeric ID from the DOM tree.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"number\",\"description\":\"The numeric element ID from the DOM tree\"}},\"required\":[\"id\"]}}},"
        + "{\"type\":\"function\",\"function\":{\"name\":\"type\",\"description\":\"Clear an input field and type new text into it, identified by numeric ID.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"number\",\"description\":\"The numeric element ID from the DOM tree\"},\"text\":{\"type\":\"string\",\"description\":\"The text to type into the input\"}},\"required\":[\"id\",\"text\"]}}},"
        + "{\"type\":\"function\",\"function\":{\"name\":\"goto_url\",\"description\":\"Navigate the browser to a new URL.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"url\":{\"type\":\"string\",\"description\":\"The URL to navigate to\"}},\"required\":[\"url\"]}}},"
        + "{\"type\":\"function\",\"function\":{\"name\":\"scroll\",\"description\":\"Scroll the page up or down.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"direction\":{\"type\":\"string\",\"enum\":[\"up\",\"down\"],\"description\":\"Scroll direction\"}},\"required\":[\"direction\"]}}},"
        + "{\"type\":\"function\",\"function\":{\"name\":\"go_back\",\"description\":\"Go back to the previous page in browser history.\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}},"
        + "{\"type\":\"function\",\"function\":{\"name\":\"screenshot\",\"description\":\"Capture a screenshot of the current page. Use when DOM text alone doesn't provide enough context.\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}},"
        + "{\"type\":\"function\",\"function\":{\"name\":\"switch_ua\",\"description\":\"Switch User-Agent. Page reloads after switching.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"mode\":{\"type\":\"string\",\"enum\":[\"mobile\",\"pc\"],\"description\":\"User-Agent mode\"}},\"required\":[\"mode\"]}}},"
        + "{\"type\":\"function\",\"function\":{\"name\":\"set_viewport\",\"description\":\"Resize browser viewport. Use larger sizes (e.g. 1920x1080) to render PC pages fully on a small phone. Use 0x0 to restore phone default. After resizing, all tools (get_page, click, screenshot) work at the new size.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"width\":{\"type\":\"number\",\"description\":\"Viewport width in CSS pixels (0 to restore)\"},\"height\":{\"type\":\"number\",\"description\":\"Viewport height in CSS pixels (0 to restore)\"}},\"required\":[\"width\",\"height\"]}}}"
        + "]";

    public AgentRunner(String task, String baseURL, String apiKey,
                       AgentBridge bridge, Runnable onDone) {
        this.task = task;
        this.baseURL = baseURL;
        this.apiKey = apiKey;
        this.bridge = bridge;
        this.onDone = onDone;
    }

    @Override
    public void run() {
        try {
            runAgent();
        } catch (Exception e) {
            agentLog("[ERROR] Agent crashed: " + e.getMessage());
            Log.e(TAG, "Agent crashed", e);
        } finally {
            if (onDone != null) {
                onDone.run();
            }
        }
    }

    private void runAgent() {
        agentLog("[TASK] Received task: " + task);

        List<JSONObject> messages = new ArrayList<>();
        messages.add(makeMessage("system", SYSTEM_PROMPT));
        messages.add(makeMessage("user", task));

        for (int step = 1; step <= MAX_STEPS; step++) {
            agentLog("[STEP " + step + "] Calling LLM...");
            trimMessages(messages);

            JSONObject response;
            try {
                response = callLLM(messages);
            } catch (Exception e) {
                agentLog("[ERROR] LLM call failed: " + e.getMessage());
                return;
            }

            JSONArray toolCalls = response.optJSONArray("tool_calls");
            if (toolCalls != null && toolCalls.length() > 0) {
                // Push assistant message with tool_calls
                JSONObject assistantMsg = new JSONObject();
                try {
                    assistantMsg.put("role", "assistant");
                    assistantMsg.put("content", response.opt("content"));
                    assistantMsg.put("tool_calls", toolCalls);
                } catch (Exception e) {
                    Log.e(TAG, "Failed to build assistant message", e);
                }
                messages.add(assistantMsg);

                // Log tool names
                StringBuilder names = new StringBuilder();
                for (int i = 0; i < toolCalls.length(); i++) {
                    if (i > 0) names.append(", ");
                    names.append(toolCalls.optJSONObject(i)
                            .optJSONObject("function").optString("name"));
                }
                agentLog("[STEP " + step + "] LLM returned tool_calls: " + names);

                // Execute each tool
                for (int i = 0; i < toolCalls.length(); i++) {
                    JSONObject tc = toolCalls.optJSONObject(i);
                    String id = tc.optString("id");
                    JSONObject fn = tc.optJSONObject("function");
                    String name = fn.optString("name");
                    String argsStr = fn.optString("arguments", "{}");

                    JSONObject args;
                    try {
                        args = new JSONObject(argsStr);
                    } catch (Exception e) {
                        args = new JSONObject();
                    }

                    ToolResult result = executeTool(name, args);

                    String logResult = result.text.length() > 200
                            ? result.text.substring(0, 200) + "... (" + result.text.length() + " chars)"
                            : result.text;
                    agentLog("[TOOL] " + name + " -> " + logResult);

                    // Push tool result
                    JSONObject toolMsg = new JSONObject();
                    try {
                        toolMsg.put("role", "tool");
                        toolMsg.put("tool_call_id", id);
                        toolMsg.put("content", result.text);
                    } catch (Exception e) {
                        Log.e(TAG, "Failed to build tool message", e);
                    }
                    messages.add(toolMsg);

                    // Inject screenshot as user vision message
                    if (result.screenshot != null) {
                        try {
                            JSONObject imageUrl = new JSONObject();
                            imageUrl.put("url", "data:image/jpeg;base64," + result.screenshot);
                            JSONObject imagePart = new JSONObject();
                            imagePart.put("type", "image_url");
                            imagePart.put("image_url", imageUrl);
                            JSONArray parts = new JSONArray();
                            parts.put(imagePart);
                            JSONObject screenshotMsg = new JSONObject();
                            screenshotMsg.put("role", "user");
                            screenshotMsg.put("content", parts);
                            messages.add(screenshotMsg);
                        } catch (Exception e) {
                            Log.e(TAG, "Failed to build screenshot message", e);
                        }
                    }
                }
            } else {
                // No tool calls — agent is done
                String finalContent = response.optString("content", "(no response)");
                agentLog("[DONE] Task completed: " + finalContent);
                return;
            }
        }

        agentLog("[ERROR] Reached max steps (" + MAX_STEPS + ") without completing task");
    }

    // ---- LLM call with retry ----

    private JSONObject callLLM(List<JSONObject> messages) throws Exception {
        String apiUrl = baseURL;
        if (apiUrl.endsWith("/")) apiUrl = apiUrl.substring(0, apiUrl.length() - 1);
        apiUrl += "/chat/completions";

        JSONObject body = new JSONObject();
        body.put("messages", new JSONArray(messages));
        body.put("tools", new JSONArray(TOOLS_JSON));
        String bodyStr = body.toString();

        String lastError = "";

        for (int attempt = 0; attempt <= MAX_RETRIES; attempt++) {
            if (attempt > 0) {
                try { Thread.sleep(RETRY_DELAY_MS); } catch (InterruptedException ignored) {}
            }

            try {
                URL url = new URL(apiUrl);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setDoOutput(true);
                conn.setConnectTimeout(30000);
                conn.setReadTimeout(120000);
                conn.setRequestProperty("Authorization", "Bearer " + apiKey);
                conn.setRequestProperty("Content-Type", "application/json");

                OutputStream os = conn.getOutputStream();
                os.write(bodyStr.getBytes("UTF-8"));
                os.close();

                int code = conn.getResponseCode();
                InputStream is = (code >= 200 && code < 400)
                        ? conn.getInputStream()
                        : conn.getErrorStream();

                BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    sb.append(line);
                }
                reader.close();
                conn.disconnect();

                String responseStr = sb.toString();
                JSONObject data;
                try {
                    data = new JSONObject(responseStr);
                } catch (Exception e) {
                    lastError = "Failed to parse LLM response: " +
                            responseStr.substring(0, Math.min(200, responseStr.length()));
                    if (attempt < MAX_RETRIES) continue;
                    throw new Exception(lastError);
                }

                if (data.has("error")) {
                    lastError = "LLM API error: " + data.opt("error");
                    JSONObject err = data.optJSONObject("error");
                    int status = err != null ? err.optInt("status", err.optInt("code", 0)) : 0;
                    if ((status == 429 || status >= 500) && attempt < MAX_RETRIES) continue;
                    throw new Exception(lastError);
                }

                JSONArray choices = data.optJSONArray("choices");
                if (choices == null || choices.length() == 0) {
                    lastError = "LLM response missing choices: " +
                            responseStr.substring(0, Math.min(200, responseStr.length()));
                    if (attempt < MAX_RETRIES) continue;
                    throw new Exception(lastError);
                }

                return choices.getJSONObject(0).getJSONObject("message");

            } catch (Exception e) {
                if (e.getMessage() != null && e.getMessage().equals(lastError)) throw e;
                lastError = e.getMessage() != null ? e.getMessage() : e.toString();
                if (attempt >= MAX_RETRIES) throw new Exception(lastError);
            }
        }

        throw new Exception(lastError.isEmpty() ? "LLM request failed after retries" : lastError);
    }

    // ---- Tool execution ----

    private static class ToolResult {
        final String text;
        final String screenshot; // null if no screenshot

        ToolResult(String text, String screenshot) {
            this.text = text;
            this.screenshot = screenshot;
        }
    }

    private ToolResult executeTool(String name, JSONObject args) {
        switch (name) {
            case "get_page": {
                String tree = bridge.getDomTree();
                return new ToolResult(tree, null);
            }
            case "click": {
                boolean ok = bridge.clickElement(args.optInt("id"));
                bridge.waitForPageLoad(5000);
                return new ToolResult(
                        ok ? "Clicked element " + args.optInt("id")
                           : "Failed to click element " + args.optInt("id"),
                        null);
            }
            case "type": {
                boolean ok = bridge.typeText(args.optInt("id"), args.optString("text"));
                return new ToolResult(
                        ok ? "Typed into element " + args.optInt("id")
                           : "Failed to type into element " + args.optInt("id"),
                        null);
            }
            case "goto_url": {
                bridge.navigateTo(args.optString("url"));
                bridge.waitForPageLoad(10000);
                return new ToolResult("Navigated to " + args.optString("url"), null);
            }
            case "scroll": {
                boolean ok = bridge.scrollPage(args.optString("direction"));
                try { Thread.sleep(150); } catch (InterruptedException ignored) {}
                return new ToolResult(
                        ok ? "Scrolled " + args.optString("direction") : "Failed to scroll",
                        null);
            }
            case "go_back": {
                boolean ok = bridge.goBack();
                bridge.waitForPageLoad(5000);
                return new ToolResult(ok ? "Went back" : "Failed to go back", null);
            }
            case "screenshot": {
                String b64 = bridge.takeScreenshot();
                if (b64.startsWith("ERROR:")) {
                    return new ToolResult(b64, null);
                }
                return new ToolResult("Screenshot captured - image will be sent to you", b64);
            }
            case "switch_ua": {
                String mode = args.optString("mode", "mobile");
                String msg = bridge.switchUserAgent(mode);
                bridge.waitForPageLoad(10000);
                return new ToolResult(msg, null);
            }
            case "set_viewport": {
                int w = args.optInt("width", 0);
                int h = args.optInt("height", 0);
                String msg = bridge.setViewport(w, h);
                return new ToolResult(msg, null);
            }
            default:
                return new ToolResult("Unknown tool: " + name, null);
        }
    }

    // ---- Message trimming (same logic as TS) ----

    private void trimMessages(List<JSONObject> messages) {
        // Truncate old tool results
        int toolCount = 0;
        for (int i = messages.size() - 1; i >= 0; i--) {
            JSONObject msg = messages.get(i);
            if ("tool".equals(msg.optString("role"))) {
                toolCount++;
                if (toolCount > KEEP_RECENT_TOOL_RESULTS) {
                    Object content = msg.opt("content");
                    if (content instanceof String) {
                        String s = (String) content;
                        if (s.length() > 200) {
                            try {
                                msg.put("content", s.substring(0, 200) +
                                        "...(truncated, " + s.length() + " chars total)");
                            } catch (Exception ignored) {}
                        }
                    }
                }
            }
        }

        // Keep only the most recent screenshot; replace older ones
        int screenshotCount = 0;
        for (int i = messages.size() - 1; i >= 0; i--) {
            JSONObject msg = messages.get(i);
            if ("user".equals(msg.optString("role"))) {
                Object content = msg.opt("content");
                if (content instanceof JSONArray) {
                    JSONArray arr = (JSONArray) content;
                    boolean hasImage = false;
                    for (int j = 0; j < arr.length(); j++) {
                        if ("image_url".equals(arr.optJSONObject(j).optString("type"))) {
                            hasImage = true;
                            break;
                        }
                    }
                    if (hasImage) {
                        screenshotCount++;
                        if (screenshotCount > 1) {
                            try {
                                msg.put("content", "[previous screenshot removed]");
                            } catch (Exception ignored) {}
                        }
                    }
                }
            }
        }
    }

    // ---- Helpers ----

    private static JSONObject makeMessage(String role, String content) {
        JSONObject msg = new JSONObject();
        try {
            msg.put("role", role);
            msg.put("content", content);
        } catch (Exception ignored) {}
        return msg;
    }

    private void agentLog(String msg) {
        SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss", Locale.US);
        String line = "[" + sdf.format(new Date()) + "] " + msg;
        Log.d(TAG, line);
        bridge.log(line);
    }
}
