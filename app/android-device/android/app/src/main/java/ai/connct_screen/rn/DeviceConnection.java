package ai.connct_screen.rn;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;
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
 * Per-agent-type WebSocket connection to the cloud ChatAgent.
 *
 * Each agent type ("app", "browser") gets its own DeviceConnection instance,
 * with its own WebSocket, OkHttpClient, reconnect state, and Hermes runtime.
 */
public class DeviceConnection {

    private static final String TAG = "DeviceConn";
    private static final long EXEC_TIMEOUT_SECONDS = 60;
    private static final long RECONNECT_BASE_MS = 5000;
    private static final long RECONNECT_MAX_MS = 60000;

    public interface Listener {
        void onConnectionStatusChanged(String agentType, boolean connected);
        void onTaskDone(String agentType, String result);
        void onUnauthorized(String agentType);
    }

    private static final Map<String, DeviceConnection> instances = new HashMap<>();

    private final String agentType;
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

    // Persistent Hermes runtime for exec_js
    private boolean hermesInitialized = false;

    private DeviceConnection(String agentType) {
        this.agentType = agentType;
        client = new OkHttpClient.Builder()
                .readTimeout(0, TimeUnit.MILLISECONDS)
                .pingInterval(20, TimeUnit.SECONDS)
                .build();
    }

    public static synchronized DeviceConnection getInstance(String agentType) {
        return instances.computeIfAbsent(agentType, k -> new DeviceConnection(k));
    }

    public String getAgentType() {
        return agentType;
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
                Log.i(TAG, "[" + agentType + "] WebSocket connected at " + System.currentTimeMillis());
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
                    Log.e(TAG, "[" + agentType + "] Failed to send ready", e);
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
                            Log.i(TAG, "[" + agentType + "] Received exec_js: " + execId + " code length=" + code.length());
                            handleExecJs(webSocket, execId, code);
                            break;
                        }
                        case "task_done": {
                            String result = data.optString("result", "");
                            Log.i(TAG, "[" + agentType + "] Received task_done: " + result.substring(0, Math.min(100, result.length())));
                            HermesRuntime.nativeHideOverlay();
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
                            Log.d(TAG, "[" + agentType + "] Unknown message type: " + type);
                    }
                } catch (Exception e) {
                    Log.e(TAG, "[" + agentType + "] Failed to parse message: " + text, e);
                }
            }

            @Override
            public void onClosing(WebSocket webSocket, int code, String reason) {
                if (webSocket != ws) return;
                Log.i(TAG, "[" + agentType + "] WebSocket closing: " + code + " " + reason);
                webSocket.close(1000, null);
            }

            @Override
            public void onClosed(WebSocket webSocket, int code, String reason) {
                if (webSocket != ws) return;
                long secsSincePing = lastPingTime > 0
                        ? (System.currentTimeMillis() - lastPingTime) / 1000 : -1;
                Log.i(TAG, "[" + agentType + "] WebSocket closed: code=" + code + " reason=" + reason
                        + " secsSinceLastPing=" + secsSincePing);
                connected = false;
                notifyConnectionStatus(false);
                scheduleReconnect();
            }

            @Override
            public void onFailure(WebSocket webSocket, Throwable t, Response response) {
                if (webSocket != ws) return;
                long secsSincePing = lastPingTime > 0
                        ? (System.currentTimeMillis() - lastPingTime) / 1000 : -1;
                Log.e(TAG, "[" + agentType + "] WebSocket failure: " + t.getClass().getSimpleName()
                        + ": " + t.getMessage()
                        + " secsSinceLastPing=" + secsSincePing, t);
                connected = false;
                notifyConnectionStatus(false);
                if (response != null && response.code() == 401) {
                    Log.w(TAG, "[" + agentType + "] Token rejected (401), notifying unauthorized");
                    notifyUnauthorized();
                } else {
                    scheduleReconnect();
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
        if (hermesInitialized) {
            try { HermesRuntime.nativeDestroyRuntime(agentType); } catch (Exception ignored) {}
            hermesInitialized = false;
            cachedPromptInfo = null;
        }
    }

    private void scheduleReconnect() {
        if (lastUrl == null) return;
        long delay = reconnectDelayMs;
        Log.i(TAG, "[" + agentType + "] Scheduling auto-reconnect in " + delay + "ms");
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
            Log.w(TAG, "[" + agentType + "] Cannot send user_task - not connected");
            return;
        }
        try {
            JSONObject msg = new JSONObject();
            msg.put("type", "user_task");
            msg.put("text", text);
            ws.send(msg.toString());
        } catch (Exception e) {
            Log.e(TAG, "[" + agentType + "] Failed to send user_task", e);
        }
    }

    public synchronized void reconnect() {
        if (lastUrl != null) {
            Log.i(TAG, "[" + agentType + "] Manual reconnect");
            connect(lastUrl, lastDeviceName);
        } else {
            Log.w(TAG, "[" + agentType + "] Cannot reconnect - no previous connection params");
        }
    }

    /**
     * Execute JS code in a persistent Hermes runtime and send the result back.
     */
    private void handleExecJs(WebSocket webSocket, String execId, String code) {
        execExecutor.execute(() -> {
            String result;
            JSONArray screenshots = new JSONArray();
            JSONArray executionLog = null;
            try {
                initHermesRuntime();
                String rawResult = HermesRuntime.nativeEvaluateJS(
                        agentType,
                        "JSON.stringify(executeCodeForServer(" + escapeJsString(code) + "))",
                        "exec_js"
                );
                // If executeCodeForServer is gone (runtime state lost), reinit and retry once
                if (rawResult != null && rawResult.contains("executeCodeForServer")
                        && rawResult.contains("doesn't exist")) {
                    Log.w(TAG, "[" + agentType + "] executeCodeForServer missing, reinitializing runtime");
                    hermesInitialized = false;
                    initHermesRuntime();
                    rawResult = HermesRuntime.nativeEvaluateJS(
                            agentType,
                            "JSON.stringify(executeCodeForServer(" + escapeJsString(code) + "))",
                            "exec_js"
                    );
                }
                try {
                    JSONObject parsed = new JSONObject(rawResult);
                    result = parsed.optString("result", rawResult);
                    JSONArray ss = parsed.optJSONArray("screenshots");
                    if (ss != null) screenshots = ss;
                    JSONArray el = parsed.optJSONArray("executionLog");
                    if (el != null && el.length() > 0) executionLog = el;
                } catch (Exception e) {
                    result = rawResult;
                }
            } catch (Exception e) {
                Log.e(TAG, "[" + agentType + "] exec_js failed", e);
                result = "Error: " + e.getMessage();
            }

            try {
                JSONObject msg = new JSONObject();
                msg.put("type", "exec_result");
                msg.put("execId", execId);
                msg.put("result", result);
                if (screenshots.length() > 0) {
                    msg.put("screenshots", screenshots);
                }
                if (executionLog != null) {
                    msg.put("executionLog", executionLog);
                }
                webSocket.send(msg.toString());
            } catch (Exception e) {
                Log.e(TAG, "[" + agentType + "] Failed to send exec_result", e);
            }
        });
    }

    private String cachedPromptInfo = null;

    private void initHermesRuntime() {
        if (hermesInitialized) return;

        HermesRuntime.nativeCreateRuntime(agentType);

        // Load the appropriate JS bundle
        String assetName = "app".equals(agentType) ? "agent-standalone.js" : "browser-standalone.js";

        com.google.android.accessibility.selecttospeak.SelectToSpeakService service =
                com.google.android.accessibility.selecttospeak.SelectToSpeakService.getInstance();
        if (service == null) {
            Log.e(TAG, "[" + agentType + "] A11y service not available, destroying bare runtime");
            HermesRuntime.nativeDestroyRuntime(agentType);
            return;
        }
        String bundleJs = HermesRuntime.loadAsset(service, assetName);
        if (bundleJs == null) {
            Log.e(TAG, "[" + agentType + "] Failed to load " + assetName + ", destroying bare runtime");
            HermesRuntime.nativeDestroyRuntime(agentType);
            return;
        }
        HermesRuntime.nativeEvaluateJS(agentType, bundleJs, assetName);

        // Read __DEVICE_PROMPT__ (set by prompt.ts / browser-prompt.ts)
        try {
            String result = HermesRuntime.nativeEvaluateJS(
                    agentType,
                    "JSON.stringify(__DEVICE_PROMPT__)",
                    "get-prompt-info"
            );
            if (result != null && !result.equals("undefined") && !result.equals("null")) {
                cachedPromptInfo = result;
            }
        } catch (Exception e) {
            Log.w(TAG, "[" + agentType + "] Failed to read __DEVICE_PROMPT__", e);
        }

        hermesInitialized = true;
        Log.i(TAG, "[" + agentType + "] Hermes runtime initialized for exec_js");
    }

    private String getDevicePromptInfo() {
        try {
            return execExecutor.submit(() -> {
                initHermesRuntime();
                return cachedPromptInfo;
            }).get(10, TimeUnit.SECONDS);
        } catch (Exception e) {
            Log.e(TAG, "[" + agentType + "] Failed to get prompt info", e);
            return null;
        }
    }

    private static String escapeJsString(String s) {
        return JsStringUtils.quoteForJS(s);
    }

    private void notifyConnectionStatus(boolean status) {
        Listener l = listener;
        if (l != null) {
            l.onConnectionStatusChanged(agentType, status);
        }
    }

    private void notifyTaskDone(String result) {
        Listener l = listener;
        if (l != null) {
            l.onTaskDone(agentType, result);
        }
    }

    private void notifyUnauthorized() {
        Listener l = listener;
        if (l != null) {
            l.onUnauthorized(agentType);
        }
    }
}
