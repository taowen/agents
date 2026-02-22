package ai.connct_screen.rn;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.ViewFlipper;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.accessibility.selecttospeak.SelectToSpeakService;

import org.json.JSONObject;

import java.io.IOException;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity implements DeviceConnection.Listener {

    private static final String TAG = "MainActivity";
    private static final String SERVER_URL = "https://ai.connect-screen.com";
    private static final String PREFS_NAME = "llm_config";
    private static final long DEVICE_POLL_INTERVAL = 2000;

    private ViewFlipper viewFlipper;

    // Login screen views
    private Button loginBtn;
    private View deviceCodePanel;
    private TextView deviceCodeText;
    private View cancelLoginBtn;
    private View loginErrorPanel;
    private View retryLoginBtn;

    // Task screen views
    private View cloudDot;
    private TextView cloudLabel;
    private View serviceDot;
    private TextView serviceLabel;
    private EditText taskInput;
    private Button sendBtn;
    private ScrollView logScroll;
    private TextView logText;

    private final OkHttpClient httpClient = new OkHttpClient();
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final SimpleDateFormat timeFmt = new SimpleDateFormat("HH:mm:ss", Locale.US);

    private String apiKey;
    private String currentDeviceCode;
    private Runnable pollRunnable;
    private boolean isRunning;
    private boolean cloudConnected;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        viewFlipper = findViewById(R.id.viewFlipper);

        // Login screen
        loginBtn = findViewById(R.id.loginBtn);
        deviceCodePanel = findViewById(R.id.deviceCodePanel);
        deviceCodeText = findViewById(R.id.deviceCodeText);
        cancelLoginBtn = findViewById(R.id.cancelLoginBtn);
        loginErrorPanel = findViewById(R.id.loginErrorPanel);
        retryLoginBtn = findViewById(R.id.retryLoginBtn);

        // Task screen
        cloudDot = findViewById(R.id.cloudDot);
        cloudLabel = findViewById(R.id.cloudLabel);
        serviceDot = findViewById(R.id.serviceDot);
        serviceLabel = findViewById(R.id.serviceLabel);
        taskInput = findViewById(R.id.taskInput);
        sendBtn = findViewById(R.id.sendBtn);
        logScroll = findViewById(R.id.logScroll);
        logText = findViewById(R.id.logText);

        // Login button handlers
        loginBtn.setOnClickListener(v -> startDeviceLogin());
        cancelLoginBtn.setOnClickListener(v -> cancelDeviceLogin());
        retryLoginBtn.setOnClickListener(v -> startDeviceLogin());

        // Task screen handlers
        findViewById(R.id.accessibilityBtn).setOnClickListener(v -> {
            Intent intent = new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS);
            startActivity(intent);
        });
        findViewById(R.id.refreshBtn).setOnClickListener(v -> checkService());
        sendBtn.setOnClickListener(v -> handleSend());

        // Cloud status tappable for reconnect
        View cloudArea = cloudDot;
        cloudArea.setOnClickListener(v -> {
            if (!cloudConnected) {
                appendLog("[CLOUD] Reconnecting...");
                DeviceConnection.getInstance().reconnect();
            }
        });

        // Check if already logged in
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        apiKey = prefs.getString("apiKey", null);
        if (apiKey != null && !apiKey.isEmpty()) {
            showTaskScreen();
            connectCloud();
        }
        // else stay on login screen (screen 0)
    }

    @Override
    protected void onResume() {
        super.onResume();
        checkService();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopDevicePoll();
        DeviceConnection.getInstance().setListener(null);
    }

    // --- Login flow ---

    private void startDeviceLogin() {
        loginBtn.setVisibility(View.GONE);
        loginErrorPanel.setVisibility(View.GONE);
        deviceCodePanel.setVisibility(View.GONE);

        Request request = new Request.Builder()
                .url(SERVER_URL + "/auth/device/start")
                .post(okhttp3.RequestBody.create(new byte[0]))
                .build();

        httpClient.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                handler.post(() -> showLoginError());
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                try {
                    String body = response.body().string();
                    JSONObject data = new JSONObject(body);
                    String code = data.getString("code");
                    handler.post(() -> {
                        currentDeviceCode = code;
                        deviceCodeText.setText(code);
                        deviceCodePanel.setVisibility(View.VISIBLE);
                        startDevicePoll(code);
                    });
                } catch (Exception e) {
                    handler.post(() -> showLoginError());
                }
            }
        });
    }

    private void startDevicePoll(String code) {
        stopDevicePoll();
        pollRunnable = new Runnable() {
            @Override
            public void run() {
                Request request = new Request.Builder()
                        .url(SERVER_URL + "/auth/device/check?code=" + code)
                        .build();

                httpClient.newCall(request).enqueue(new Callback() {
                    @Override
                    public void onFailure(Call call, IOException e) {
                        // Keep polling
                        handler.postDelayed(pollRunnable, DEVICE_POLL_INTERVAL);
                    }

                    @Override
                    public void onResponse(Call call, Response response) throws IOException {
                        try {
                            String body = response.body().string();
                            JSONObject data = new JSONObject(body);
                            String status = data.optString("status", "");

                            if ("approved".equals(status) && data.has("token")) {
                                String token = data.getString("token");
                                String baseURL = data.optString("baseURL", "");
                                String model = data.optString("model", "");

                                handler.post(() -> {
                                    stopDevicePoll();
                                    saveConfig(baseURL, token, model);
                                    apiKey = token;
                                    showTaskScreen();
                                    connectCloud();
                                });
                            } else if ("expired".equals(status)) {
                                handler.post(() -> {
                                    stopDevicePoll();
                                    showLoginError();
                                });
                            } else {
                                handler.postDelayed(pollRunnable, DEVICE_POLL_INTERVAL);
                            }
                        } catch (Exception e) {
                            handler.postDelayed(pollRunnable, DEVICE_POLL_INTERVAL);
                        }
                    }
                });
            }
        };
        handler.postDelayed(pollRunnable, DEVICE_POLL_INTERVAL);
    }

    private void stopDevicePoll() {
        if (pollRunnable != null) {
            handler.removeCallbacks(pollRunnable);
            pollRunnable = null;
        }
    }

    private void cancelDeviceLogin() {
        stopDevicePoll();
        currentDeviceCode = null;
        deviceCodePanel.setVisibility(View.GONE);
        loginBtn.setVisibility(View.VISIBLE);
    }

    private void showLoginError() {
        deviceCodePanel.setVisibility(View.GONE);
        loginBtn.setVisibility(View.GONE);
        loginErrorPanel.setVisibility(View.VISIBLE);
    }

    private void saveConfig(String baseURL, String apiKey, String model) {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        prefs.edit()
                .putString("baseURL", baseURL)
                .putString("apiKey", apiKey)
                .putString("model", model)
                .apply();
    }

    // --- Task screen ---

    private void showTaskScreen() {
        viewFlipper.setDisplayedChild(1);
    }

    private void connectCloud() {
        if (apiKey == null || apiKey.isEmpty()) return;

        String name = Build.MANUFACTURER + " " + Build.MODEL;
        try {
            String wsUrl = "wss://ai.connect-screen.com/agents/chat-agent/device-"
                    + URLEncoder.encode(name, "UTF-8")
                    + "/device-connect?token="
                    + URLEncoder.encode(apiKey, "UTF-8");

            DeviceConnection conn = DeviceConnection.getInstance();
            conn.setListener(this);
            conn.connect(wsUrl, name);
        } catch (Exception e) {
            Log.e(TAG, "Failed to connect cloud", e);
        }
    }

    private void checkService() {
        boolean running = SelectToSpeakService.getInstance() != null;
        if (running) {
            serviceDot.setBackgroundColor(0xFF4CAF50);
            serviceLabel.setText("Service");
        } else {
            serviceDot.setBackgroundColor(0xFFF44336);
            serviceLabel.setText("No Svc");
        }
    }

    private void handleSend() {
        String text = taskInput.getText().toString().trim();
        if (text.isEmpty() || isRunning || !cloudConnected) return;

        logText.setText("");
        taskInput.setText("");
        isRunning = true;
        updateSendButton();

        appendLog("[TASK] Sending to server: " + text);
        DeviceConnection.getInstance().sendUserTask(text);
    }

    private void updateSendButton() {
        if (isRunning) {
            sendBtn.setText("Running");
            sendBtn.setEnabled(false);
            sendBtn.setBackgroundColor(0xFF666666);
            taskInput.setEnabled(false);
        } else {
            boolean canSend = cloudConnected;
            sendBtn.setText(canSend ? "Send" : "Offline");
            sendBtn.setEnabled(canSend);
            sendBtn.setBackgroundColor(canSend ? 0xFFE94560 : 0x66E94560);
            taskInput.setEnabled(true);
        }
    }

    private void appendLog(String msg) {
        String time = timeFmt.format(new Date());
        String line = "[" + time + "] " + msg;
        String current = logText.getText().toString();
        if (current.equals("Enter a task and tap Send")) {
            logText.setText(line);
        } else {
            logText.append("\n" + line);
        }
        logScroll.post(() -> logScroll.fullScroll(View.FOCUS_DOWN));
    }

    // --- DeviceConnection.Listener ---

    @Override
    public void onConnectionStatusChanged(boolean connected) {
        handler.post(() -> {
            cloudConnected = connected;
            if (connected) {
                cloudDot.setBackgroundColor(0xFF2196F3);
                cloudLabel.setText("Cloud");
            } else {
                cloudDot.setBackgroundColor(0xFF666666);
                cloudLabel.setText("Offline");
            }
            updateSendButton();
        });
    }

    @Override
    public void onTaskDone(String result) {
        handler.post(() -> {
            appendLog("[DONE] " + result);
            isRunning = false;
            updateSendButton();
        });
    }
}
