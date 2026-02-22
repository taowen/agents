package ai.connct_screen.rn;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

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
 * - Receives task dispatches and notifies the Listener
 * - Handles ping/pong keepalive
 * - Sends task results back to the ChatAgent
 */
public class DeviceConnection {

    private static final String TAG = "DeviceConn";
    private static final long EXEC_TIMEOUT_SECONDS = 60;
    private static final long RECONNECT_BASE_MS = 5000;
    private static final long RECONNECT_MAX_MS = 60000;

    public interface Listener {
        void onConnectionStatusChanged(boolean connected);
        void onTaskDone(String result);
    }

    private static DeviceConnection instance;

    private OkHttpClient client;
    private WebSocket ws;
    private boolean connected = false;
    private String deviceName;
    private String lastUrl;
    private String lastDeviceName;
    private Listener listener;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final ExecutorService execExecutor = Executors.newSingleThreadExecutor();

    private long lastPingTime = 0;
    private long reconnectDelayMs = RECONNECT_BASE_MS;
    private Runnable pendingReconnect = null;

    // Persistent Hermes runtime for exec_js — initialized on first exec_js, reused across calls
    private boolean hermesInitialized = false;

    private DeviceConnection() {
        client = new OkHttpClient.Builder()
                .readTimeout(0, TimeUnit.MILLISECONDS) // no read timeout for WS
                .pingInterval(20, TimeUnit.SECONDS)    // WebSocket protocol-level ping frames
                .build();
    }

    public static synchronized DeviceConnection getInstance() {
        if (instance == null) {
            instance = new DeviceConnection();
        }
        return instance;
    }

    public void setListener(Listener listener) {
        this.listener = listener;
    }

    public synchronized void connect(String url, String name) {
        this.deviceName = name;
        this.lastUrl = url;
        this.lastDeviceName = name;
        // Cancel any pending reconnect to avoid duplicates
        if (pendingReconnect != null) {
            mainHandler.removeCallbacks(pendingReconnect);
            pendingReconnect = null;
        }
        if (ws != null) {
            try { ws.close(1000, "reconnecting"); } catch (Exception ignored) {}
        }

        Request request = new Request.Builder().url(url).build();
        ws = client.newWebSocket(request, new WebSocketListener() {
            @Override
            public void onOpen(WebSocket webSocket, Response response) {
                Log.i(TAG, "WebSocket connected at " + System.currentTimeMillis());
                connected = true;
                reconnectDelayMs = RECONNECT_BASE_MS;
                lastPingTime = System.currentTimeMillis();
                notifyConnectionStatus(true);
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
                        case "task_done": {
                            String result = data.optString("result", "");
                            Log.i(TAG, "Received task_done: " + result.substring(0, Math.min(100, result.length())));
                            notifyTaskDone(result);
                            break;
                        }
                        case "ping": {
                            lastPingTime = System.currentTimeMillis();
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
                long secsSincePing = lastPingTime > 0
                        ? (System.currentTimeMillis() - lastPingTime) / 1000 : -1;
                Log.i(TAG, "WebSocket closed: code=" + code + " reason=" + reason
                        + " secsSinceLastPing=" + secsSincePing);
                connected = false;
                notifyConnectionStatus(false);
                scheduleReconnect();
            }

            @Override
            public void onFailure(WebSocket webSocket, Throwable t, Response response) {
                long secsSincePing = lastPingTime > 0
                        ? (System.currentTimeMillis() - lastPingTime) / 1000 : -1;
                Log.e(TAG, "WebSocket failure: " + t.getClass().getSimpleName()
                        + ": " + t.getMessage()
                        + " secsSinceLastPing=" + secsSincePing, t);
                connected = false;
                notifyConnectionStatus(false);
                scheduleReconnect();
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

    private void scheduleReconnect() {
        if (lastUrl == null) return;
        long delay = reconnectDelayMs;
        Log.i(TAG, "Scheduling auto-reconnect in " + delay + "ms");
        pendingReconnect = () -> connect(lastUrl, lastDeviceName);
        mainHandler.postDelayed(pendingReconnect, delay);
        reconnectDelayMs = Math.min(reconnectDelayMs * 2, RECONNECT_MAX_MS);
    }

    public boolean isConnected() {
        return connected;
    }

    /**
     * Send a user-initiated task to the server for processing.
     */
    public void sendUserTask(String text) {
        if (!connected || ws == null) {
            Log.w(TAG, "Cannot send user_task - not connected");
            return;
        }
        try {
            JSONObject msg = new JSONObject();
            msg.put("type", "user_task");
            msg.put("text", text);
            ws.send(msg.toString());
        } catch (Exception e) {
            Log.e(TAG, "Failed to send user_task", e);
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
            String agentJs = HermesAgentRunner.loadAsset(service, "agent-standalone.js");
            if (agentJs != null) {
                HermesAgentRunner.nativeEvaluateJS(agentJs, "agent-standalone.js");
            } else {
                Log.e(TAG, "Failed to load agent-standalone.js");
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

        // executeCodeForServer is now defined in agent-standalone.ts
        // (loaded above), sharing the same host function wrappers with update_status.

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
        return JsStringUtils.quoteForJS(s);
    }

    private void notifyConnectionStatus(boolean status) {
        Listener l = listener;
        if (l != null) {
            l.onConnectionStatusChanged(status);
        }
    }

    private void notifyTaskDone(String result) {
        Listener l = listener;
        if (l != null) {
            l.onTaskDone(result);
        }
    }
}
