package ai.connct_screen.rn;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.view.inputmethod.EditorInfo;
import android.widget.ViewFlipper;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

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

public class MainActivity extends AppCompatActivity implements DeviceConnection.Listener,
        VoiceService.AsrListener {

    private static final String TAG = "MainActivity";
    private static final String SERVER_URL = "https://ai.connect-screen.com";
    private static final String PREFS_NAME = "llm_config";
    private static final long DEVICE_POLL_INTERVAL = 2000;
    private static final int REQUEST_RECORD_AUDIO = 100;
    private static final int REQUEST_POST_NOTIFICATIONS = 101;

    private ViewFlipper viewFlipper;

    // Login screen views
    private Button loginBtn;
    private View deviceCodePanel;
    private TextView deviceCodeText;
    private View cancelLoginBtn;
    private View loginErrorPanel;
    private View retryLoginBtn;

    // Task screen views
    private View headerRow;
    private View taskInputRow;
    private Spinner agentSpinner;
    private EditText taskInput;
    private Button sendBtn;
    private WebView browserWebView;
    private View browserBar;
    private EditText browserUrlInput;
    private Button browserBackBtn, browserFwdBtn, browserGoBtn;
    private ScrollView logScroll;
    private TextView logText;

    // Voice views
    private View voiceStatusBar;
    private View voiceStatusDot;
    private TextView voiceStatusText;
    private Button micToggleBtn;
    private TextView transcriptionText;

    // Model download views
    private TextView downloadStatusText;
    private ProgressBar downloadProgress;
    private Button downloadBtn;
    private Button skipDownloadBtn;

    // Voice / model state
    private ModelManager modelManager;
    private boolean voiceEnabled;

    private final OkHttpClient httpClient = new OkHttpClient();
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final SimpleDateFormat timeFmt = new SimpleDateFormat("HH:mm:ss", Locale.US);

    private String apiKey;
    private String currentDeviceCode;
    private Runnable pollRunnable;
    private boolean isRunning;
    private boolean serviceRunning;

    // Per-agent connection status
    private boolean appConnected;
    private boolean browserConnected;

    // Spinner items: "App - Connected", "App - Offline", "Browser - Connected", "Browser - Offline"
    private String[] spinnerItems;
    private ArrayAdapter<String> spinnerAdapter;
    private String currentAgentType = "app"; // selected agent type
    private boolean spinnerInitialized = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Light status bar: dark icons on light background
        getWindow().setStatusBarColor(0xFFF5F5F5);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);

        viewFlipper = findViewById(R.id.viewFlipper);

        // Login screen
        loginBtn = findViewById(R.id.loginBtn);
        deviceCodePanel = findViewById(R.id.deviceCodePanel);
        deviceCodeText = findViewById(R.id.deviceCodeText);
        cancelLoginBtn = findViewById(R.id.cancelLoginBtn);
        loginErrorPanel = findViewById(R.id.loginErrorPanel);
        retryLoginBtn = findViewById(R.id.retryLoginBtn);

        // Task screen
        headerRow = findViewById(R.id.headerRow);
        taskInputRow = findViewById(R.id.taskInputRow);
        agentSpinner = findViewById(R.id.agentSpinner);
        taskInput = findViewById(R.id.taskInput);
        sendBtn = findViewById(R.id.sendBtn);
        browserWebView = findViewById(R.id.browserWebView);
        browserBar = findViewById(R.id.browserBar);
        browserUrlInput = findViewById(R.id.browserUrlInput);
        browserBackBtn = findViewById(R.id.browserBackBtn);
        browserFwdBtn = findViewById(R.id.browserFwdBtn);
        browserGoBtn = findViewById(R.id.browserGoBtn);
        logScroll = findViewById(R.id.logScroll);
        logText = findViewById(R.id.logText);

        // Voice views
        voiceStatusBar = findViewById(R.id.voiceStatusBar);
        voiceStatusDot = findViewById(R.id.voiceStatusDot);
        voiceStatusText = findViewById(R.id.voiceStatusText);
        micToggleBtn = findViewById(R.id.micToggleBtn);
        transcriptionText = findViewById(R.id.transcriptionText);

        // Model download views
        downloadStatusText = findViewById(R.id.downloadStatusText);
        downloadProgress = findViewById(R.id.downloadProgress);
        downloadBtn = findViewById(R.id.downloadBtn);
        skipDownloadBtn = findViewById(R.id.skipDownloadBtn);

        modelManager = new ModelManager(this);

        // Login button handlers
        loginBtn.setOnClickListener(v -> startDeviceLogin());
        cancelLoginBtn.setOnClickListener(v -> cancelDeviceLogin());
        retryLoginBtn.setOnClickListener(v -> startDeviceLogin());

        // Send button: handles both "No A11y" → open settings, and normal send
        sendBtn.setOnClickListener(v -> {
            if (!serviceRunning) {
                startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS));
            } else {
                handleSend();
            }
        });

        // Browser bar handlers
        browserBackBtn.setOnClickListener(v -> {
            if (browserWebView.canGoBack()) browserWebView.goBack();
        });
        browserFwdBtn.setOnClickListener(v -> {
            if (browserWebView.canGoForward()) browserWebView.goForward();
        });
        browserGoBtn.setOnClickListener(v -> navigateBrowserUrl());
        browserUrlInput.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_GO || actionId == EditorInfo.IME_ACTION_DONE) {
                navigateBrowserUrl();
                return true;
            }
            return false;
        });

        // Mic toggle
        micToggleBtn.setOnClickListener(v -> {
            if (voiceEnabled) {
                stopVoiceService();
            } else {
                startVoiceService();
            }
        });

        // Model download handlers
        downloadBtn.setOnClickListener(v -> startModelDownload());
        skipDownloadBtn.setOnClickListener(v -> {
            // Skip model download, go to task screen without voice
            showTaskScreen();
            connectCloud();
        });

        // Set up agent spinner
        spinnerItems = new String[] {
            "App - Offline \u25BE",
            "Browser - Offline \u25BE"
        };
        spinnerAdapter = new ArrayAdapter<>(this,
                R.layout.spinner_item, spinnerItems);
        spinnerAdapter.setDropDownViewResource(R.layout.spinner_dropdown_item);
        agentSpinner.setAdapter(spinnerAdapter);
        agentSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (!spinnerInitialized) {
                    spinnerInitialized = true;
                    return;
                }
                if (position == 0) {
                    currentAgentType = "app";
                    browserWebView.setVisibility(View.GONE);
                    browserBar.setVisibility(View.GONE);
                    logScroll.setVisibility(View.VISIBLE);
                } else {
                    currentAgentType = "browser";
                    browserWebView.setVisibility(View.VISIBLE);
                    browserBar.setVisibility(View.VISIBLE);
                    logScroll.setVisibility(View.GONE);
                }
                updateBrowserChrome();
                updateSendButton();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        // Set up WebView
        setupWebView();

        // Check if already logged in
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        apiKey = prefs.getString("apiKey", null);
        if (apiKey != null && !apiKey.isEmpty()) {
            onLoginComplete();
        }
        // else stay on login screen (screen 0)
    }

    private void setupWebView() {
        WebSettings settings = browserWebView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setLoadWithOverviewMode(true);
        settings.setUseWideViewPort(true);

        browserWebView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                String scheme = request.getUrl().getScheme();
                // Block non-http(s) schemes (e.g. baiduboxapp://) — they can't load in WebView
                if (scheme != null && !scheme.equals("https") && !scheme.equals("http")) {
                    return true;
                }
                return false;
            }

            @Override
            public void onPageStarted(WebView view, String url, Bitmap favicon) {
                BrowserToolsHost.onPageStarted();
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                BrowserToolsHost.onPageFinished();
                browserUrlInput.setText(view.getUrl());
            }
        });

        // Initialize BrowserToolsHost with this activity and webview
        BrowserToolsHost.init(this, browserWebView);

        // Load a default page
        browserWebView.loadUrl("https://www.google.com");
    }

    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        updateBrowserChrome();
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
        DeviceConnection.getInstance("app").setListener(null);
        DeviceConnection.getInstance("browser").setListener(null);
        if (voiceEnabled) {
            stopVoiceService();
        }
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
                                    onLoginComplete();
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
            String encodedName = URLEncoder.encode(name, "UTF-8");
            String encodedToken = URLEncoder.encode(apiKey, "UTF-8");

            // App agent connection
            String appUrl = "wss://ai.connect-screen.com/agents/chat-agent/device-"
                    + encodedName + "-app"
                    + "/device-connect?token=" + encodedToken;

            // Browser agent connection
            String browserUrl = "wss://ai.connect-screen.com/agents/chat-agent/device-"
                    + encodedName + "-browser"
                    + "/device-connect?token=" + encodedToken;

            DeviceConnection appConn = DeviceConnection.getInstance("app");
            appConn.setListener(this);
            appConn.connect(appUrl, name);

            DeviceConnection browserConn = DeviceConnection.getInstance("browser");
            browserConn.setListener(this);
            browserConn.connect(browserUrl, name);
        } catch (Exception e) {
            Log.e(TAG, "Failed to connect cloud", e);
        }
    }

    /** Hide header/task rows in landscape when browser is active */
    private void updateBrowserChrome() {
        boolean landscape = getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE;
        boolean browserMode = "browser".equals(currentAgentType);
        int vis = (landscape && browserMode) ? View.GONE : View.VISIBLE;
        headerRow.setVisibility(vis);
        taskInputRow.setVisibility(vis);
    }

    private void navigateBrowserUrl() {
        String input = browserUrlInput.getText().toString().trim();
        if (input.isEmpty()) return;
        if (!input.contains("://")) {
            input = "https://" + input;
        }
        browserWebView.loadUrl(input);
    }

    private void checkService() {
        serviceRunning = SelectToSpeakService.getInstance() != null;
        updateSendButton();
    }

    private void handleSend() {
        String text = taskInput.getText().toString().trim();
        boolean currentConnected = "app".equals(currentAgentType) ? appConnected : browserConnected;
        if (text.isEmpty() || isRunning || !currentConnected) return;

        logText.setText("");
        taskInput.setText("");
        isRunning = true;
        updateSendButton();

        appendLog("[TASK] Sending to server (" + currentAgentType + "): " + text);
        DeviceConnection.getInstance(currentAgentType).sendUserTask(text);
    }

    private void updateSpinnerItems() {
        spinnerItems[0] = (appConnected ? "App - Connected" : "App - Offline") + " \u25BE";
        spinnerItems[1] = (browserConnected ? "Browser - Connected" : "Browser - Offline") + " \u25BE";
        spinnerAdapter.notifyDataSetChanged();
    }

    private void updateSendButton() {
        if (!serviceRunning) {
            sendBtn.setText("No A11y");
            sendBtn.setEnabled(true);
            sendBtn.setBackgroundColor(0xFFC62828);
            taskInput.setEnabled(false);
        } else if (isRunning) {
            sendBtn.setText("Running");
            sendBtn.setEnabled(false);
            sendBtn.setBackgroundColor(0xFF666666);
            taskInput.setEnabled(false);
        } else {
            boolean currentConnected = "app".equals(currentAgentType) ? appConnected : browserConnected;
            sendBtn.setText(currentConnected ? "Send" : "Offline");
            sendBtn.setEnabled(currentConnected);
            sendBtn.setBackgroundColor(currentConnected ? 0xFF1976D2 : 0x661976D2);
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

    // --- Voice / Model ---

    private void onLoginComplete() {
        if (modelManager.isModelReady()) {
            showTaskScreen();
            connectCloud();
            requestAudioPermissionAndStart();
        } else {
            // Show model download screen
            viewFlipper.setDisplayedChild(2);
        }
    }

    private void startModelDownload() {
        downloadBtn.setEnabled(false);
        skipDownloadBtn.setEnabled(false);
        downloadProgress.setVisibility(View.VISIBLE);
        downloadStatusText.setText("Downloading...");

        new Thread(() -> modelManager.download(new ModelManager.DownloadListener() {
            @Override
            public void onProgress(long downloaded, long total, String currentFile) {
                int pct = total > 0 ? (int)(downloaded * 1000 / total) : 0;
                handler.post(() -> {
                    downloadProgress.setProgress(pct);
                    long mb = downloaded / (1024 * 1024);
                    long totalMb = total / (1024 * 1024);
                    downloadStatusText.setText("Downloading " + currentFile
                            + " (" + mb + "/" + totalMb + " MB)");
                });
            }

            @Override
            public void onComplete(String modelDir) {
                handler.post(() -> {
                    showTaskScreen();
                    connectCloud();
                    requestAudioPermissionAndStart();
                });
            }

            @Override
            public void onError(String message) {
                handler.post(() -> {
                    downloadStatusText.setText("Error: " + message);
                    downloadBtn.setEnabled(true);
                    skipDownloadBtn.setEnabled(true);
                });
            }
        })).start();
    }

    private void requestAudioPermissionAndStart() {
        // Request POST_NOTIFICATIONS first (Android 13+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this,
                    Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.POST_NOTIFICATIONS},
                        REQUEST_POST_NOTIFICATIONS);
                return;
            }
        }
        requestAudioPermission();
    }

    private void requestAudioPermission() {
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    REQUEST_RECORD_AUDIO);
        } else {
            loadModelAndStartVoice();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_POST_NOTIFICATIONS) {
            // Continue to audio permission regardless
            requestAudioPermission();
        } else if (requestCode == REQUEST_RECORD_AUDIO) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                loadModelAndStartVoice();
            } else {
                appendLog("[VOICE] Microphone permission denied");
            }
        }
    }

    private void loadModelAndStartVoice() {
        if (!modelManager.isModelReady()) {
            appendLog("[VOICE] Model not downloaded, voice disabled");
            return;
        }

        appendLog("[VOICE] Loading ASR model...");
        new Thread(() -> {
            boolean ok = VoiceService.nativeLoadModel(modelManager.getModelDir(), 4);
            handler.post(() -> {
                if (ok) {
                    appendLog("[VOICE] Model loaded successfully");
                    startVoiceService();
                } else {
                    appendLog("[VOICE] Failed to load model");
                }
            });
        }).start();
    }

    private void startVoiceService() {
        voiceEnabled = true;
        voiceStatusBar.setVisibility(View.VISIBLE);
        voiceStatusText.setText("Listening...");
        voiceStatusDot.setBackgroundColor(0xFF4CAF50);
        micToggleBtn.setText("Mic OFF");
        micToggleBtn.setBackgroundColor(0xFFF44336);

        Intent intent = new Intent(this, VoiceService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }

        // Wait for service to start, then set listener
        handler.postDelayed(() -> {
            VoiceService svc = VoiceService.getInstance();
            if (svc != null) {
                svc.setAsrListener(this);
            }
        }, 500);
    }

    private void stopVoiceService() {
        voiceEnabled = false;
        voiceStatusBar.setVisibility(View.GONE);
        transcriptionText.setVisibility(View.GONE);
        stopService(new Intent(this, VoiceService.class));
    }

    // --- VoiceService.AsrListener ---

    @Override
    public void onTextSegment(String text) {
        appendLog("[VOICE] " + text);
        transcriptionText.setVisibility(View.GONE);
        // Send to the currently selected agent
        DeviceConnection.getInstance(currentAgentType).sendUserTask(text);
    }

    @Override
    public void onAsrToken(String token) {
        transcriptionText.setVisibility(View.VISIBLE);
        transcriptionText.append(token);
        // Keep only last portion visible
        String full = transcriptionText.getText().toString();
        if (full.length() > 200) {
            transcriptionText.setText(full.substring(full.length() - 200));
        }
    }

    // --- DeviceConnection.Listener ---

    @Override
    public void onConnectionStatusChanged(String agentType, boolean connected) {
        handler.post(() -> {
            if ("app".equals(agentType)) {
                appConnected = connected;
            } else if ("browser".equals(agentType)) {
                browserConnected = connected;
            }
            updateSpinnerItems();
            updateSendButton();
        });
    }

    @Override
    public void onTaskDone(String agentType, String result) {
        handler.post(() -> {
            appendLog("[DONE] (" + agentType + ") " + result);
            isRunning = false;
            updateSendButton();
        });
    }
}
