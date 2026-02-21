package ai.connct_screen.rn;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import org.json.JSONObject;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

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
 * - Proxies LLM requests from the Hermes agent (blocking sendLlmRequest)
 * - Receives task dispatches and emits DeviceTask events to RN JS
 * - Handles ping/pong keepalive
 * - Sends task results back to the ChatAgent
 */
public class DeviceConnection {

    private static final String TAG = "DeviceConn";
    private static final long LLM_TIMEOUT_SECONDS = 120;
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

    private final ConcurrentHashMap<String, CompletableFuture<String>> llmPending =
            new ConcurrentHashMap<>();

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
                // Send ready message
                try {
                    JSONObject ready = new JSONObject();
                    ready.put("type", "ready");
                    ready.put("deviceName", deviceName);
                    ready.put("deviceId", deviceName);
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
                        case "llm_response": {
                            String requestId = data.getString("requestId");
                            String body = data.getString("body");
                            CompletableFuture<String> future = llmPending.remove(requestId);
                            if (future != null) {
                                future.complete(body);
                            } else {
                                Log.w(TAG, "No pending LLM request for id: " + requestId);
                            }
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
                failAllPending("WebSocket closed");
            }

            @Override
            public void onFailure(WebSocket webSocket, Throwable t, Response response) {
                Log.e(TAG, "WebSocket failure", t);
                connected = false;
                emitConnectionStatus(false);
                failAllPending("WebSocket failure: " + t.getMessage());
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
        failAllPending("Disconnected");
    }

    public boolean isConnected() {
        return connected;
    }

    /**
     * Send an LLM request through the WebSocket and block until the response arrives.
     * Called from the Hermes agent thread (via JNI → nativeLlmChat).
     */
    public String sendLlmRequest(String body) {
        if (!connected || ws == null) {
            return "{\"error\":{\"message\":\"Not connected to cloud\"}}";
        }

        String requestId = java.util.UUID.randomUUID().toString();
        CompletableFuture<String> future = new CompletableFuture<>();
        llmPending.put(requestId, future);

        try {
            JSONObject msg = new JSONObject();
            msg.put("type", "llm_request");
            msg.put("requestId", requestId);
            msg.put("body", new JSONObject(body));
            ws.send(msg.toString());
        } catch (Exception e) {
            llmPending.remove(requestId);
            Log.e(TAG, "Failed to send llm_request", e);
            return "{\"error\":{\"message\":\"Failed to send: " + e.getMessage().replace("\"", "\\\"") + "\"}}";
        }

        try {
            return future.get(LLM_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (Exception e) {
            llmPending.remove(requestId);
            Log.e(TAG, "LLM request timed out or failed", e);
            return "{\"error\":{\"message\":\"LLM request failed: " + e.getMessage().replace("\"", "\\\"") + "\"}}";
        }
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

    private void failAllPending(String reason) {
        for (CompletableFuture<String> future : llmPending.values()) {
            future.completeExceptionally(new RuntimeException(reason));
        }
        llmPending.clear();
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
