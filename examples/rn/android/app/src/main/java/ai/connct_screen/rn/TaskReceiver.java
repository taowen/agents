package ai.connct_screen.rn;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

import com.facebook.react.ReactApplication;
import com.facebook.react.ReactHost;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactContext;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.core.DeviceEventManagerModule;

public class TaskReceiver extends BroadcastReceiver {

    private static final String TAG = "A11yAgent";

    @Override
    public void onReceive(Context context, Intent intent) {
        String task = intent.getStringExtra("task");
        if (task == null || task.isEmpty()) {
            Log.d(TAG, "[ERROR] No task provided in broadcast");
            return;
        }

        ReactApplication app = (ReactApplication) context.getApplicationContext();
        ReactHost reactHost = app.getReactHost();
        if (reactHost == null) {
            Log.d(TAG, "[ERROR] ReactHost not available");
            return;
        }

        ReactContext reactContext = reactHost.getCurrentReactContext();
        if (reactContext == null) {
            Log.d(TAG, "[ERROR] ReactContext not available (app may not be fully loaded)");
            return;
        }

        WritableMap params = Arguments.createMap();
        params.putString("task", task);

        String apiUrl = intent.getStringExtra("api_url");
        String apiKey = intent.getStringExtra("api_key");
        String model = intent.getStringExtra("model");
        if (apiUrl != null) params.putString("apiUrl", apiUrl);
        if (apiKey != null) params.putString("apiKey", apiKey);
        if (model != null) params.putString("model", model);

        Log.d(TAG, "[TASK] Emitting onTaskReceived to RN: " + task);

        reactContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
                .emit("onTaskReceived", params);
    }
}
