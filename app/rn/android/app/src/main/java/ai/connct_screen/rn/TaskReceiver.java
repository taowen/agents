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

        // Route to server via WebSocket
        DeviceConnection conn = DeviceConnection.getInstance();
        if (!conn.isConnected()) {
            Log.e(TAG, "[ERROR] Cannot send task - not connected to server");
            return;
        }
        conn.sendUserTask(task);
    }
}
