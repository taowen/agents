package ai.connct_screen.rn;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.util.Log;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;

import com.google.android.accessibility.selecttospeak.SelectToSpeakService;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;

/**
 * React Native bridge module — now simplified to UI-related methods only.
 * All screen-action methods (click, scroll, etc.) have moved to standalone
 * Hermes via HermesAgentRunner + standalone_hermes.cpp.
 */
public class AccessibilityBridgeModule extends ReactContextBaseJavaModule {

    private static final String TAG = "A11yAgent";

    AccessibilityBridgeModule(ReactApplicationContext context) {
        super(context);
    }

    @NonNull
    @Override
    public String getName() {
        return "AccessibilityBridge";
    }

    // --- Device info (sync) ---

    @ReactMethod(isBlockingSynchronousMethod = true)
    public String getDeviceName() {
        return Build.MANUFACTURER + " " + Build.MODEL;
    }

    // --- Log file methods (sync, used by RN UI) ---

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean appendLogLine(String line) {
        try {
            Context appContext = getReactApplicationContext().getApplicationContext();
            File logFile = new File(appContext.getFilesDir(), "agent-log.txt");
            FileOutputStream fos = new FileOutputStream(logFile, true);
            OutputStreamWriter writer = new OutputStreamWriter(fos, "UTF-8");
            writer.write(line + "\n");
            writer.flush();
            writer.close();
        } catch (Exception e) {
            Log.e(TAG, "[appendLogLine] failed", e);
        }
        return true;
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean clearLogFile() {
        try {
            Context appContext = getReactApplicationContext().getApplicationContext();
            File logFile = new File(appContext.getFilesDir(), "agent-log.txt");
            if (logFile.exists()) logFile.delete();
        } catch (Exception e) {
            Log.e(TAG, "[clearLogFile] failed", e);
        }
        return true;
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean isCloudConnected() {
        return DeviceConnection.getInstance().isConnected();
    }

    // --- Async methods ---

    @ReactMethod
    public void isServiceRunning(Promise promise) {
        promise.resolve(SelectToSpeakService.getInstance() != null);
    }

    @ReactMethod
    public void connectCloud(String url, String deviceName, Promise promise) {
        DeviceConnection conn = DeviceConnection.getInstance();
        conn.setReactContext(getReactApplicationContext());
        conn.connect(url, deviceName);
        promise.resolve(null);
    }

    @ReactMethod
    public void disconnectCloud(Promise promise) {
        DeviceConnection.getInstance().disconnect();
        promise.resolve(null);
    }

    @ReactMethod
    public void reconnectCloud(Promise promise) {
        DeviceConnection.getInstance().reconnect();
        promise.resolve(null);
    }

    /**
     * Save LLM config to SharedPreferences so that TaskReceiver (broadcast path)
     * can also read it without going through React Native.
     */
    @ReactMethod
    public void saveConfig(String baseURL, String apiKey, String model, Promise promise) {
        try {
            Context appContext = getReactApplicationContext().getApplicationContext();
            SharedPreferences prefs = appContext.getSharedPreferences("llm_config", Context.MODE_PRIVATE);
            prefs.edit()
                    .putString("baseURL", baseURL)
                    .putString("apiKey", apiKey)
                    .putString("model", model)
                    .apply();
            promise.resolve(null);
        } catch (Exception e) {
            promise.reject("SAVE_ERROR", e.getMessage(), e);
        }
    }

    /**
     * Send a task to the server for execution via the device WebSocket.
     * Resolves immediately — the result arrives async via DeviceTaskDone event.
     */
    @ReactMethod
    public void sendUserTask(String text, Promise promise) {
        DeviceConnection.getInstance().sendUserTask(text);
        promise.resolve(null);
    }
}
