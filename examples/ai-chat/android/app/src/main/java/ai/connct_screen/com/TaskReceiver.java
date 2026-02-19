package ai.connct_screen.com;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

import com.google.android.accessibility.selecttospeak.SelectToSpeakService;

public class TaskReceiver extends BroadcastReceiver {

    private static final String TAG = "A11yAgent";

    @Override
    public void onReceive(Context context, Intent intent) {
        String task = intent.getStringExtra("task");
        String apiUrl = intent.getStringExtra("api_url");
        String apiKey = intent.getStringExtra("api_key");
        String model = intent.getStringExtra("model");

        if (task == null || task.isEmpty()) {
            Log.d(TAG, "[ERROR] No task provided in broadcast");
            return;
        }
        if (apiUrl == null || apiUrl.isEmpty()) {
            Log.d(TAG, "[ERROR] No api_url provided in broadcast");
            return;
        }
        if (apiKey == null || apiKey.isEmpty()) {
            Log.d(TAG, "[ERROR] No api_key provided in broadcast");
            return;
        }
        if (model == null || model.isEmpty()) {
            model = "gpt-4o";
        }

        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) {
            Log.d(TAG, "[ERROR] Accessibility service not running. Please enable it in Settings.");
            return;
        }

        Log.d(TAG, "[TASK] Received task: " + task);
        Log.d(TAG, "[TASK] API URL: " + apiUrl);
        Log.d(TAG, "[TASK] Model: " + model);

        final String finalModel = model;
        new Thread(new Runnable() {
            @Override
            public void run() {
                AgentLoop agent = new AgentLoop(apiUrl, apiKey, finalModel, service, context.getApplicationContext());
                agent.execute(task);
            }
        }).start();
    }
}
