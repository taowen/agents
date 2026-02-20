package ai.connct_screen.rn;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.util.Log;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

public class TaskReceiver extends BroadcastReceiver {

    private static final String TAG = "A11yAgent";

    @Override
    public void onReceive(Context context, Intent intent) {
        String task = intent.getStringExtra("task");
        if (task == null || task.isEmpty()) {
            Log.d(TAG, "[ERROR] No task provided in broadcast");
            return;
        }

        Log.d(TAG, "[TASK] Received task: " + task);

        // Read LLM config — try SharedPreferences first (written by RN app),
        // then fall back to asset llm-config.json
        String configJson = readConfig(context);
        if (configJson == null) {
            Log.e(TAG, "[ERROR] No LLM config found");
            return;
        }

        // Run agent on a background thread (HermesAgentRunner.runAgent blocks)
        final String finalConfig = configJson;
        final String finalTask = task;
        new Thread(() -> {
            try {
                HermesAgentRunner.runAgent(finalTask, finalConfig);
            } catch (Exception e) {
                Log.e(TAG, "[ERROR] Agent failed", e);
            }
        }, "hermes-agent").start();
    }

    private String readConfig(Context context) {
        // 1. Try SharedPreferences (set by RN UI)
        try {
            SharedPreferences prefs = context.getSharedPreferences("llm_config", Context.MODE_PRIVATE);
            String baseURL = prefs.getString("baseURL", null);
            String apiKey = prefs.getString("apiKey", null);
            String model = prefs.getString("model", null);
            if (baseURL != null && apiKey != null && model != null) {
                return "{\"baseURL\":\"" + escapeJson(baseURL) +
                       "\",\"apiKey\":\"" + escapeJson(apiKey) +
                       "\",\"model\":\"" + escapeJson(model) + "\"}";
            }
        } catch (Exception e) {
            Log.w(TAG, "[readConfig] SharedPreferences failed", e);
        }

        // 2. Fall back to asset
        try {
            InputStream is = context.getAssets().open("llm-config.json");
            BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            reader.close();
            return sb.toString();
        } catch (Exception e) {
            Log.w(TAG, "[readConfig] asset llm-config.json not found", e);
        }

        return null;
    }

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
