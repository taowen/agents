package ai.connct_screen.rn;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import org.json.JSONObject;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.json.JSONArray;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;

/**
 * Singleton that manages a Java-level OkHttp WebSocket connection to the cloud ChatAgent.
 *
 * Responsibilities:
 * - Maintains persistent WebSocket to the device's ChatAgent session
 *   (wss://ai.connect-screen.com/agents/chat-agent/device-{name}/device-connect)
 * - Executes JS code from server exec_js requests in a persistent Hermes runtime
 * - Receives task dispatches and emits DeviceTask events to RN JS
 * - Handles ping/pong keepalive
 * - Sends task results back to the ChatAgent
 */
public class DeviceConnection {

    private static final String TAG = "DeviceConn";
    private static final long EXEC_TIMEOUT_SECONDS = 60;
    private static final long RECONNECT_DELAY_MS = 5000;

    private static DeviceConnection instance;

    private OkHttpClient client;
    private WebSocket ws;
    private boolean connected = false;
    private String deviceName;
    private String lastUrl;
    private String lastDeviceName;
    private ReactApplicationContext reactContext;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final ExecutorService execExecutor = Executors.newSingleThreadExecutor();

    // Persistent Hermes runtime for exec_js — initialized on first exec_js, reused across calls
    private boolean hermesInitialized = false;

    private DeviceConnection() {
        client = new OkHttpClient.Builder()
                .readTimeout(0, TimeUnit.MILLISECONDS) // no read timeout for WS
                .build();
    }

    public static synchronized DeviceConnection getInstance() {
        if (instance == null) {
            instance = new DeviceConnection();
        }
        return instance;
    }

    public void setReactContext(ReactApplicationContext ctx) {
        this.reactContext = ctx;
    }

    public synchronized void connect(String url, String name) {
        this.deviceName = name;
        this.lastUrl = url;
        this.lastDeviceName = name;
        if (ws != null) {
            try { ws.close(1000, "reconnecting"); } catch (Exception ignored) {}
        }

        Request request = new Request.Builder().url(url).build();
        ws = client.newWebSocket(request, new WebSocketListener() {
            @Override
            public void onOpen(WebSocket webSocket, Response response) {
                Log.i(TAG, "WebSocket connected");
                connected = true;
                emitConnectionStatus(true);
                // Send ready message with system prompt and tools
                try {
                    JSONObject ready = new JSONObject();
                    ready.put("type", "ready");
                    ready.put("deviceName", deviceName);
                    ready.put("deviceId", deviceName);
                    // Include system prompt and tool definitions from prompt.ts
                    // so server can use them in streamText
                    String promptInfo = getDevicePromptInfo();
                    if (promptInfo != null) {
                        JSONObject promptData = new JSONObject(promptInfo);
                        if (promptData.has("systemPrompt")) {
                            ready.put("systemPrompt", promptData.getString("systemPrompt"));
                        }
                        if (promptData.has("tools")) {
                            ready.put("tools", promptData.getJSONArray("tools"));
                        }
                    }
                    webSocket.send(ready.toString());
                } catch (Exception e) {
                    Log.e(TAG, "Failed to send ready", e);
                }
            }

            @Override
            public void onMessage(WebSocket webSocket, String text) {
                try {
                    JSONObject data = new JSONObject(text);
                    String type = data.optString("type", "");

                    switch (type) {
                        case "exec_js": {
                            String execId = data.getString("execId");
                            String code = data.getString("code");
                            Log.i(TAG, "Received exec_js: " + execId + " code length=" + code.length());
                            handleExecJs(webSocket, execId, code);
                            break;
                        }
                        case "task": {
                            String taskId = data.getString("taskId");
                            String description = data.getString("description");
                            Log.i(TAG, "Received task: " + taskId + " - " + description);
                            emitTaskEvent(taskId, description);
                            break;
                        }
                        case "ping": {
                            try {
                                JSONObject pong = new JSONObject();
                                pong.put("type", "pong");
                                webSocket.send(pong.toString());
                            } catch (Exception ignored) {}
                            break;
                        }
                        default:
                            Log.d(TAG, "Unknown message type: " + type);
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Failed to parse message: " + text, e);
                }
            }

            @Override
            public void onClosing(WebSocket webSocket, int code, String reason) {
                Log.i(TAG, "WebSocket closing: " + code + " " + reason);
                webSocket.close(1000, null);
            }

            @Override
            public void onClosed(WebSocket webSocket, int code, String reason) {
                Log.i(TAG, "WebSocket closed: " + code + " " + reason);
                connected = false;
                emitConnectionStatus(false);
            }

            @Override
            public void onFailure(WebSocket webSocket, Throwable t, Response response) {
                Log.e(TAG, "WebSocket failure", t);
                connected = false;
                emitConnectionStatus(false);
                // Auto-reconnect after delay
                if (lastUrl != null) {
                    Log.i(TAG, "Scheduling auto-reconnect in " + RECONNECT_DELAY_MS + "ms");
                    mainHandler.postDelayed(() -> connect(lastUrl, lastDeviceName), RECONNECT_DELAY_MS);
                }
            }
        });
    }

    public synchronized void disconnect() {
        if (ws != null) {
            try { ws.close(1000, "disconnect"); } catch (Exception ignored) {}
            ws = null;
        }
        connected = false;
        // Clean up Hermes runtime
        if (hermesInitialized) {
            try { HermesAgentRunner.nativeDestroyRuntime(); } catch (Exception ignored) {}
            hermesInitialized = false;
            cachedPromptInfo = null;
        }
    }

    public boolean isConnected() {
        return connected;
    }

    /**
     * Send a task result back to the DeviceHub.
     */
    public void sendTaskResult(String taskId, String result, boolean success) {
        if (!connected || ws == null) {
            Log.w(TAG, "Cannot send task result - not connected");
            return;
        }
        try {
            JSONObject msg = new JSONObject();
            msg.put("type", "result");
            msg.put("taskId", taskId);
            msg.put("result", result);
            msg.put("success", success);
            ws.send(msg.toString());
        } catch (Exception e) {
            Log.e(TAG, "Failed to send task result", e);
        }
    }

    public synchronized void reconnect() {
        if (lastUrl != null) {
            Log.i(TAG, "Manual reconnect");
            connect(lastUrl, lastDeviceName);
        } else {
            Log.w(TAG, "Cannot reconnect - no previous connection params");
        }
    }

    /**
     * Execute JS code in a persistent Hermes runtime and send the result back.
     * Runs on a single-thread executor to serialize access to the Hermes runtime.
     */
    private void handleExecJs(WebSocket webSocket, String execId, String code) {
        execExecutor.execute(() -> {
            String result;
            JSONArray screenshots = new JSONArray();
            try {
                initHermesRuntime();
                // Execute the code. The executeCodeInHermes function returns JSON
                // with { result, screenshots? }
                String rawResult = HermesAgentRunner.nativeEvaluateJS(
                        "JSON.stringify(executeCodeForServer(" + escapeJsString(code) + "))",
                        "exec_js"
                );
                try {
                    JSONObject parsed = new JSONObject(rawResult);
                    result = parsed.optString("result", rawResult);
                    JSONArray ss = parsed.optJSONArray("screenshots");
                    if (ss != null) screenshots = ss;
                } catch (Exception e) {
                    // If not JSON, use raw result
                    result = rawResult;
                }
            } catch (Exception e) {
                Log.e(TAG, "exec_js failed", e);
                result = "Error: " + e.getMessage();
            }

            // Send result back
            try {
                JSONObject msg = new JSONObject();
                msg.put("type", "exec_result");
                msg.put("execId", execId);
                msg.put("result", result);
                if (screenshots.length() > 0) {
                    msg.put("screenshots", screenshots);
                }
                webSocket.send(msg.toString());
            } catch (Exception e) {
                Log.e(TAG, "Failed to send exec_result", e);
            }
        });
    }

    /**
     * Initialize the persistent Hermes runtime, load the agent bundle, and set up
     * the executeCodeForServer helper. Also reads and caches __DEVICE_PROMPT__.
     * Must only be called from the execExecutor thread (Hermes is not thread-safe).
     */
    private String cachedPromptInfo = null;

    private void initHermesRuntime() {
        if (hermesInitialized) return;

        HermesAgentRunner.nativeCreateRuntime();

        // Load the agent JS bundle which defines all host function wrappers
        com.google.android.accessibility.selecttospeak.SelectToSpeakService service =
                com.google.android.accessibility.selecttospeak.SelectToSpeakService.getInstance();
        if (service != null) {
            try {
                java.io.InputStream is = service.getAssets().open("agent-standalone.js");
                java.io.BufferedReader reader = new java.io.BufferedReader(
                        new java.io.InputStreamReader(is, "UTF-8"));
                StringBuilder sb = new StringBuilder();
                char[] buffer = new char[8192];
                int read;
                while ((read = reader.read(buffer)) != -1) {
                    sb.append(buffer, 0, read);
                }
                reader.close();
                HermesAgentRunner.nativeEvaluateJS(sb.toString(), "agent-standalone.js");
            } catch (Exception e) {
                Log.e(TAG, "Failed to load agent-standalone.js", e);
            }
        }

        // Read __DEVICE_PROMPT__ (set by prompt.ts) before defining helpers
        try {
            String result = HermesAgentRunner.nativeEvaluateJS(
                    "JSON.stringify(__DEVICE_PROMPT__)",
                    "get-prompt-info"
            );
            if (result != null && !result.equals("undefined") && !result.equals("null")) {
                cachedPromptInfo = result;
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to read __DEVICE_PROMPT__", e);
        }

        // Define the executeCodeForServer helper that wraps code execution
        // and captures screenshots, similar to executeCode in agent-standalone.ts
        String helperJs =
                "function executeCodeForServer(code) {\n" +
                "  var capturedScreenshots = [];\n" +
                "  var origGetScreen = get_screen;\n" +
                "  var origTakeScreenshot = take_screenshot;\n" +
                "  var getScreenCount = 0;\n" +
                "  var lastGetScreenResult = null;\n" +
                "  get_screen = function() {\n" +
                "    getScreenCount++;\n" +
                "    if (getScreenCount > 5) {\n" +
                "      throw new Error('get_screen() called ' + getScreenCount + ' times. Max is 5.');\n" +
                "    }\n" +
                "    var tree = origGetScreen();\n" +
                "    lastGetScreenResult = tree;\n" +
                "    return tree;\n" +
                "  };\n" +
                "  take_screenshot = function() {\n" +
                "    var b64 = origTakeScreenshot();\n" +
                "    if (b64.startsWith('ERROR:')) return b64;\n" +
                "    capturedScreenshots.push(b64);\n" +
                "    return 'screenshot captured - image will be sent to you';\n" +
                "  };\n" +
                "  try {\n" +
                "    var result = (0, eval)(code);\n" +
                "    if (result === undefined && lastGetScreenResult !== null) {\n" +
                "      result = lastGetScreenResult;\n" +
                "    }\n" +
                "    return {\n" +
                "      result: result === undefined ? 'undefined' : String(result),\n" +
                "      screenshots: capturedScreenshots\n" +
                "    };\n" +
                "  } catch (e) {\n" +
                "    return {\n" +
                "      result: '[JS Error] ' + (e.message || String(e)),\n" +
                "      screenshots: capturedScreenshots\n" +
                "    };\n" +
                "  } finally {\n" +
                "    get_screen = origGetScreen;\n" +
                "    take_screenshot = origTakeScreenshot;\n" +
                "  }\n" +
                "}\n";
        HermesAgentRunner.nativeEvaluateJS(helperJs, "exec-helper.js");

        hermesInitialized = true;
        Log.i(TAG, "Hermes runtime initialized for exec_js");
    }

    /**
     * Get the device prompt info (systemPrompt + tools).
     * Initializes the Hermes runtime on the execExecutor thread (Hermes is not thread-safe).
     */
    private String getDevicePromptInfo() {
        try {
            return execExecutor.submit(() -> {
                initHermesRuntime();
                return cachedPromptInfo;
            }).get(10, TimeUnit.SECONDS);
        } catch (Exception e) {
            Log.e(TAG, "Failed to get prompt info", e);
            return null;
        }
    }

    private static String escapeJsString(String s) {
        return "\"" + s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t") + "\"";
    }

    private void emitConnectionStatus(boolean status) {
        if (reactContext == null || !reactContext.hasActiveReactInstance()) {
            return;
        }
        try {
            WritableMap params = Arguments.createMap();
            params.putBoolean("connected", status);
            reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
                    .emit("DeviceConnectionStatus", params);
        } catch (Exception e) {
            Log.e(TAG, "Failed to emit DeviceConnectionStatus", e);
        }
    }

    private void emitTaskEvent(String taskId, String description) {
        if (reactContext == null || !reactContext.hasActiveReactInstance()) {
            Log.w(TAG, "No React context for task event");
            return;
        }
        try {
            WritableMap params = Arguments.createMap();
            params.putString("taskId", taskId);
            params.putString("description", description);
            reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
                    .emit("DeviceTask", params);
        } catch (Exception e) {
            Log.e(TAG, "Failed to emit DeviceTask event", e);
        }
    }
}
