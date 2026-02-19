package ai.connct_screen.com;

import android.content.Context;
import android.util.Log;

import com.google.android.accessibility.selecttospeak.SelectToSpeakService;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class AgentLoop {

    private static final String TAG = "A11yAgent";
    private static final int MAX_STEPS = 30;
    private static final int KEEP_RECENT_TOOL_RESULTS = 3; // keep last N tool results intact

    private final LlmClient llmClient;
    private final SelectToSpeakService service;
    private final Context context;
    private final JsEngine jsEngine;
    private final List<JSONObject> messages;
    private final JSONArray tools;
    private final SimpleDateFormat timeFmt = new SimpleDateFormat("HH:mm:ss", Locale.US);

    public AgentLoop(String apiUrl, String apiKey, String model, SelectToSpeakService service, Context context) {
        this.llmClient = new LlmClient(apiUrl, apiKey, model);
        this.service = service;
        this.context = context;
        File screensDir = new File(context.getFilesDir(), "screens");
        this.jsEngine = new JsEngine(service, context, screensDir);
        this.messages = new ArrayList<>();
        this.tools = buildToolDefinitions();
    }

    private void log(String msg) {
        String line = "[" + timeFmt.format(new Date()) + "] " + msg;
        // Best-effort logcat (may be dropped under rate limit)
        Log.d(TAG, msg);
        // Append to file
        try {
            File logFile = new File(context.getFilesDir(), "agent_log.txt");
            FileOutputStream fos = new FileOutputStream(logFile, true);
            OutputStreamWriter writer = new OutputStreamWriter(fos, "UTF-8");
            writer.write(line + "\n");
            writer.flush();
            writer.close();
        } catch (Exception ignored) {
        }
    }

    private void resetLogFile() {
        try {
            File logFile = new File(context.getFilesDir(), "agent_log.txt");
            // Overwrite with empty
            new FileOutputStream(logFile, false).close();
        } catch (Exception ignored) {
        }
    }

    private void resetScreensDir() {
        try {
            File screensDir = new File(context.getFilesDir(), "screens");
            if (screensDir.exists()) {
                File[] files = screensDir.listFiles();
                if (files != null) {
                    for (File f : files) {
                        f.delete();
                    }
                }
            } else {
                screensDir.mkdirs();
            }
        } catch (Exception ignored) {
        }
        jsEngine.resetScreenCounter();
    }

    public void execute(String task) {
        resetLogFile();
        resetScreensDir();
        log("[TASK] Received task: " + task);

        try {
            // Initialize messages
            messages.add(makeSystemMessage());
            messages.add(makeUserMessage(task));

            for (int step = 1; step <= MAX_STEPS; step++) {
                log("[STEP " + step + "] Calling LLM...");
                trimMessages();

                LlmClient.LlmResponse response = llmClient.chat(messages, tools);

                if (response.hasToolCalls()) {
                    // Add assistant message with tool calls
                    JSONObject assistantMsg = new JSONObject();
                    assistantMsg.put("role", "assistant");
                    if (response.content != null) {
                        assistantMsg.put("content", response.content);
                    } else {
                        assistantMsg.put("content", JSONObject.NULL);
                    }
                    assistantMsg.put("tool_calls", response.toolCalls);
                    messages.add(assistantMsg);

                    // Log tool calls
                    StringBuilder toolNames = new StringBuilder();
                    for (int i = 0; i < response.toolCalls.length(); i++) {
                        JSONObject tc = response.toolCalls.getJSONObject(i);
                        if (i > 0) toolNames.append(", ");
                        toolNames.append(tc.getJSONObject("function").getString("name"));
                    }
                    log("[STEP " + step + "] LLM returned tool_calls: " + toolNames);

                    // Execute each tool call
                    for (int i = 0; i < response.toolCalls.length(); i++) {
                        JSONObject toolCall = response.toolCalls.getJSONObject(i);
                        String toolCallId = toolCall.getString("id");
                        String functionName = toolCall.getJSONObject("function").getString("name");
                        String argumentsStr = toolCall.getJSONObject("function").getString("arguments");
                        JSONObject arguments = new JSONObject(argumentsStr);

                        String result = executeTool(functionName, arguments);

                        // Log tool result (truncated)
                        String logResult = result.length() > 200
                                ? result.substring(0, 200) + "... (" + result.length() + " chars)"
                                : result;
                        log("[TOOL] " + functionName + " -> " + logResult);

                        // Add tool result message
                        JSONObject toolMsg = new JSONObject();
                        toolMsg.put("role", "tool");
                        toolMsg.put("tool_call_id", toolCallId);
                        toolMsg.put("content", result);
                        messages.add(toolMsg);
                    }

                } else {
                    // No tool calls - task is done
                    String finalContent = response.content != null ? response.content : "(no response)";
                    log("[DONE] Task completed: " + finalContent);
                    return;
                }
            }

            log("[ERROR] Reached max steps (" + MAX_STEPS + ") without completing task");

        } catch (Exception e) {
            log("[ERROR] Agent loop error: " + e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    private void trimMessages() {
        // Count tool-role messages from the end
        int toolCount = 0;
        for (int i = messages.size() - 1; i >= 0; i--) {
            try {
                JSONObject msg = messages.get(i);
                if ("tool".equals(msg.getString("role"))) {
                    toolCount++;
                    if (toolCount > KEEP_RECENT_TOOL_RESULTS) {
                        String content = msg.optString("content", "");
                        if (content.length() > 200) {
                            msg.put("content", content.substring(0, 200)
                                    + "...(truncated, " + content.length() + " chars total)");
                        }
                    }
                }
            } catch (JSONException ignored) {
            }
        }
        if (toolCount > KEEP_RECENT_TOOL_RESULTS) {
            log("[TRIM] Truncated " + (toolCount - KEEP_RECENT_TOOL_RESULTS)
                    + " old tool results, kept " + KEEP_RECENT_TOOL_RESULTS + " recent");
        }
    }

    private String executeTool(String name, JSONObject args) throws JSONException {
        if ("execute_js".equals(name)) {
            String code = args.getString("code");
            return jsEngine.execute(code);
        }
        return "Unknown tool: " + name;
    }

    private JSONObject makeSystemMessage() throws JSONException {
        JSONObject msg = new JSONObject();
        msg.put("role", "system");
        msg.put("content",
                "You are a mobile automation assistant controlling a phone via Android Accessibility Service.\n" +
                "You can see the screen's accessibility tree where each node shows: class type, text, content description (desc), bounds coordinates, and properties.\n\n" +
                "IMPORTANT: The JS engine only supports ES5. Do NOT use arrow functions (=>), " +
                "template literals (`...`), let/const, destructuring, or spread. Use var, function(){}, string concatenation with +.\n\n" +
                "You operate by writing JavaScript code using the execute_js tool. Available global functions:\n" +
                "- get_screen() → returns accessibility tree as string\n" +
                "- click(target) → click by text: click(\"OK\"), by desc: click({desc:\"加\"}), or by coords: click({x:100,y:200})\n" +
                "- long_click(target) → same syntax as click, but long-press\n" +
                "- scroll(direction) → \"up\"/\"down\"/\"left\"/\"right\"\n" +
                "- type_text(text) → type into focused input\n" +
                "- press_home() / press_back() → navigation\n" +
                "- press_recents() → open recent tasks list (for switching apps)\n" +
                "- show_notifications() → pull down notification shade\n" +
                "- launch_app(name) → launch app by name or package name (e.g. launch_app(\"时钟\"), launch_app(\"com.coloros.clock\"))\n" +
                "- list_apps() → returns installed launchable apps, one per line: \"AppName (package.name)\"\n" +
                "- scroll_element(text, direction) → scroll a specific scrollable element found by text. direction: \"up\"/\"down\". Use for NumberPicker, time selectors, etc.\n" +
                "- sleep(ms) → wait for UI to settle\n" +
                "- log(msg) → log a message\n\n" +
                "Tips:\n" +
                "- Execute a SHORT sequence of actions (5-10 operations max), then return the result\n" +
                "- Do NOT write for/while loops that call get_screen() or scroll() repeatedly. " +
                "Do one scroll + sleep + get_screen, return the result, decide next in a new step\n" +
                "- get_screen() is limited to 5 calls per execute_js\n" +
                "- Use globalThis to store state between calls (e.g. button coordinates, screen info)\n" +
                "- click(\"text\") matches BOTH text and desc attributes. Use click({desc:\"X\"}) for desc-only match\n" +
                "- Bounds format: [left,top][right,bottom]. Center: x=(left+right)/2, y=(top+bottom)/2\n" +
                "- After actions, call sleep(500) then get_screen() to verify results\n" +
                "- If click by text fails, calculate coordinates from bounds and use click({x, y})\n" +
                "- In Settings, menu items may differ by device (e.g. \"关于本机\" vs \"关于手机\"). " +
                "If scrolling doesn't find the target, try the search function\n" +
                "- To open an app, prefer launch_app(\"AppName\") over navigating the home screen\n" +
                "- For NumberPicker/time selectors, use scroll_element(\"当前值\", \"up\"/\"down\") to change values\n" +
                "- When the task is complete, respond with a text summary (no tool call)"
        );
        return msg;
    }

    private JSONObject makeUserMessage(String task) throws JSONException {
        JSONObject msg = new JSONObject();
        msg.put("role", "user");
        msg.put("content", task);
        return msg;
    }

    private JSONArray buildToolDefinitions() {
        try {
            JSONArray toolsArray = new JSONArray();

            // execute_js - the single tool for all actions
            JSONObject codeProps = new JSONObject();
            codeProps.put("code", new JSONObject()
                    .put("type", "string")
                    .put("description", "JavaScript code to execute. All screen automation functions are available as globals."));
            toolsArray.put(makeTool("execute_js",
                    "Execute ES5 JavaScript code. Execute a short linear sequence of actions. " +
                    "Available globals: get_screen(), click(target), long_click(target), scroll(dir), " +
                    "type_text(text), press_home(), press_back(), press_recents(), show_notifications(), " +
                    "launch_app(name), list_apps(), scroll_element(text, dir), " +
                    "sleep(ms), log(msg). get_screen() is limited to 5 calls per execution. " +
                    "Do NOT use loops to scroll and check screen repeatedly. " +
                    "The globalThis object persists across calls - use it to store context.",
                    new JSONObject().put("type", "object").put("properties", codeProps)
                            .put("required", new JSONArray().put("code"))));

            return toolsArray;
        } catch (JSONException e) {
            log("[ERROR] Failed to build tool definitions: " + e.getMessage());
            return new JSONArray();
        }
    }

    private JSONObject makeTool(String name, String description, JSONObject parameters) throws JSONException {
        JSONObject tool = new JSONObject();
        tool.put("type", "function");
        JSONObject function = new JSONObject();
        function.put("name", name);
        function.put("description", description);
        function.put("parameters", parameters);
        tool.put("function", function);
        return tool;
    }
}
