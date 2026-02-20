package com.example.androidbrowser;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

/**
 * Broadcast receiver for triggering the agent via adb:
 *
 *   adb shell am broadcast -a com.example.androidbrowser.EXECUTE_TASK \
 *       --es task "search for weather in Beijing"
 */
public class TaskReceiver extends BroadcastReceiver {

    private static final String TAG = "TaskReceiver";

    @Override
    public void onReceive(Context context, Intent intent) {
        String task = intent.getStringExtra("task");
        if (task == null || task.isEmpty()) {
            Log.w(TAG, "No task provided in broadcast");
            return;
        }

        Log.d(TAG, "Received task: " + task);

        // Forward to MainActivity
        Intent activityIntent = new Intent(context, MainActivity.class);
        activityIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        activityIntent.putExtra("agent_task", task);
        context.startActivity(activityIntent);
    }
}
