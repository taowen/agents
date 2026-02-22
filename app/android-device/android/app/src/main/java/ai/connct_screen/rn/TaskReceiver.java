package ai.connct_screen.rn;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;


public class TaskReceiver extends BroadcastReceiver {

    private static final String TAG = "A11yAgent";

    @Override
    public void onReceive(Context context, Intent intent) {
        String task = intent.getStringExtra("task");
        if (task == null || task.isEmpty()) {
            Log.d(TAG, "[ERROR] No task provided in broadcast");
            return;
        }

        Log.d(TAG, "[TASK] Received broadcast task: " + task);

        // Route to server via WebSocket (default to "app" agent)
        String agentType = intent.getStringExtra("agent");
        if (agentType == null || agentType.isEmpty()) {
            agentType = "app";
        }
        DeviceConnection conn = DeviceConnection.getInstance(agentType);
        if (!conn.isConnected()) {
            Log.e(TAG, "[ERROR] Cannot send task - " + agentType + " agent not connected to server");
            return;
        }
        conn.sendUserTask(task);
    }
}
