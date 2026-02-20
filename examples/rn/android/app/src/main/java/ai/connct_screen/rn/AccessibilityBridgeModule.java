package ai.connct_screen.rn;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;

import com.google.android.accessibility.selecttospeak.SelectToSpeakService;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
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

    // --- Async methods ---

    @ReactMethod
    public void isServiceRunning(Promise promise) {
        promise.resolve(SelectToSpeakService.getInstance() != null);
    }

    @ReactMethod
    public void readAssetConfig(Promise promise) {
        try {
            InputStream is = getReactApplicationContext().getAssets().open("llm-config.json");
            BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            reader.close();
            promise.resolve(sb.toString());
        } catch (Exception e) {
            promise.reject("ASSET_ERROR", "Failed to read llm-config.json: " + e.getMessage(), e);
        }
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
     * Run agent task via standalone Hermes (HermesAgentRunner).
     * Runs on a background thread; resolves when the agent completes.
     */
    @ReactMethod
    public void runAgentTask(String task, String configJson, Promise promise) {
        new Thread(() -> {
            try {
                HermesAgentRunner.runAgent(task, configJson);
                promise.resolve("done");
            } catch (Exception e) {
                Log.e(TAG, "[runAgentTask] failed", e);
                promise.reject("AGENT_ERROR", e.getMessage(), e);
            }
        }, "hermes-agent").start();
    }
}
