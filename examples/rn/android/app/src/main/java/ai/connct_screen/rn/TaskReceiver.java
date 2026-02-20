package ai.connct_screen.rn;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

import org.json.JSONObject;

public class TaskReceiver extends BroadcastReceiver {

    private static final String TAG = "A11yAgent";

    @Override
    public void onReceive(Context context, Intent intent) {
        String task = intent.getStringExtra("task");
        if (task == null || task.isEmpty()) {
            Log.d(TAG, "[ERROR] No task provided in broadcast");
            return;
        }

        try {
            JSONObject json = new JSONObject();
            json.put("task", task);

            String apiUrl = intent.getStringExtra("api_url");
            String apiKey = intent.getStringExtra("api_key");
            String model = intent.getStringExtra("model");
            if (apiUrl != null) json.put("apiUrl", apiUrl);
            if (apiKey != null) json.put("apiKey", apiKey);
            if (model != null) json.put("model", model);

            Log.d(TAG, "[TASK] Setting pending task: " + task);
            AccessibilityBridgeModule.setPendingTask(json.toString());
        } catch (Exception e) {
            Log.e(TAG, "[ERROR] Failed to create task JSON", e);
        }
    }
}
