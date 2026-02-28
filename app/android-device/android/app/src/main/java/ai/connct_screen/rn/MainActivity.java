package ai.connct_screen.rn;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.view.inputmethod.EditorInfo;
import android.widget.ProgressBar;
import android.widget.ViewFlipper;

import android.Manifest;
import android.content.pm.PackageManager;
import android.net.ConnectivityManager;
import android.net.Network;

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
    private View headerRow;
    private TextView tabApp, tabBrowser;
    private View appPane, browserPane;
    private EditText appTaskInput, browserTaskInput;
    private Button appSendBtn, browserSendBtn;
    private View browserTaskInputRow;
    private WebView browserWebView;
    private View browserBar;
    private EditText browserUrlInput;
    private Button browserBackBtn, browserFwdBtn, browserGoBtn;
    private ScrollView logScroll;
    private TextView logText;

    private final OkHttpClient httpClient = new OkHttpClient();
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final SimpleDateFormat timeFmt = new SimpleDateFormat("HH:mm:ss", Locale.US);

    private String apiKey;
    private String currentDeviceCode;
    private Runnable pollRunnable;
    private boolean appRunning;
    private boolean browserRunning;
    private boolean serviceRunning;

    // Settings screen views
    private TextView modelStatusText;
    private Spinner sourceSpinner;
    private ProgressBar downloadProgress;
    private TextView downloadProgressText;
    private Button downloadBtn;
    private Button loadModelBtn;
    private Button testStreamBtn;
    private Button freeModelBtn;
    private TextView asrResultText;

    private ModelManager modelManager;
    private volatile boolean isDownloading;
    private volatile boolean modelLoaded;
    private volatile boolean isAsrTesting;
    private android.media.AudioRecord testRecorder;
    private Thread testRecordThread;

    // Per-agent connection status
    private boolean appConnected;
    private boolean browserConnected;

    private String currentAgentType = "app"; // selected agent type

    private BroadcastReceiver screenOnReceiver;
    private ConnectivityManager.NetworkCallback networkCallback;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        DeviceConnection.setAppContext(getApplicationContext());

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
        tabApp = findViewById(R.id.tabApp);
        tabBrowser = findViewById(R.id.tabBrowser);
        appPane = findViewById(R.id.appPane);
        browserPane = findViewById(R.id.browserPane);
        appTaskInput = findViewById(R.id.appTaskInput);
        appSendBtn = findViewById(R.id.appSendBtn);
        browserTaskInput = findViewById(R.id.browserTaskInput);
        browserSendBtn = findViewById(R.id.browserSendBtn);
        browserTaskInputRow = findViewById(R.id.browserTaskInputRow);
        browserWebView = findViewById(R.id.browserWebView);
        browserBar = findViewById(R.id.browserBar);
        browserUrlInput = findViewById(R.id.browserUrlInput);
        browserBackBtn = findViewById(R.id.browserBackBtn);
        browserFwdBtn = findViewById(R.id.browserFwdBtn);
        browserGoBtn = findViewById(R.id.browserGoBtn);
        logScroll = findViewById(R.id.logScroll);
        logText = findViewById(R.id.logText);

        // Settings screen
        modelStatusText = findViewById(R.id.modelStatusText);
        sourceSpinner = findViewById(R.id.sourceSpinner);
        downloadProgress = findViewById(R.id.downloadProgress);
        downloadProgressText = findViewById(R.id.downloadProgressText);
        downloadBtn = findViewById(R.id.downloadBtn);
        loadModelBtn = findViewById(R.id.loadModelBtn);
        testStreamBtn = findViewById(R.id.testStreamBtn);
        freeModelBtn = findViewById(R.id.freeModelBtn);
        asrResultText = findViewById(R.id.asrResultText);

        modelManager = new ModelManager(this);
        setupSettingsScreen();

        // Settings button in header
        findViewById(R.id.settingsBtn).setOnClickListener(v -> openSettings());
        findViewById(R.id.settingsBackBtn).setOnClickListener(v -> viewFlipper.setDisplayedChild(1));

        // Login button handlers
        loginBtn.setOnClickListener(v -> startDeviceLogin());
        cancelLoginBtn.setOnClickListener(v -> cancelDeviceLogin());
        retryLoginBtn.setOnClickListener(v -> startDeviceLogin());

        // App send button: handles "No A11y" → open settings, and normal send
        appSendBtn.setOnClickListener(v -> {
            if (!serviceRunning) {
                startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS));
            } else {
                handleSend();
            }
        });
        // Browser send button: no a11y check needed
        browserSendBtn.setOnClickListener(v -> handleSend());

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

        // Set up agent tabs
        tabApp.setOnClickListener(v -> selectTab("app"));
        tabBrowser.setOnClickListener(v -> selectTab("browser"));

        // Set up WebView
        setupWebView();

        // Register screen-on receiver for WebSocket reconnect
        screenOnReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                if (Intent.ACTION_SCREEN_ON.equals(intent.getAction())) {
                    DeviceConnection.reconnectAll();
                }
            }
        };
        registerReceiver(screenOnReceiver, new IntentFilter(Intent.ACTION_SCREEN_ON));

        // Register network callback for WebSocket reconnect
        ConnectivityManager cm = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        if (cm != null) {
            networkCallback = new ConnectivityManager.NetworkCallback() {
                @Override
                public void onAvailable(Network network) {
                    DeviceConnection.reconnectAll();
                }
            };
            cm.registerDefaultNetworkCallback(networkCallback);
        }

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
        DeviceConnection.reconnectAll();
        // Sync UI with actual connection state (may have reconnected while paused)
        appConnected = DeviceConnection.getInstance("app").isConnected();
        browserConnected = DeviceConnection.getInstance("browser").isConnected();
        updateTabs();
        updateSendButton();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopDevicePoll();
        if (screenOnReceiver != null) {
            unregisterReceiver(screenOnReceiver);
            screenOnReceiver = null;
        }
        if (networkCallback != null) {
            ConnectivityManager cm = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
            if (cm != null) cm.unregisterNetworkCallback(networkCallback);
            networkCallback = null;
        }
        DeviceConnection.getInstance("app").setListener(null);
        DeviceConnection.getInstance("browser").setListener(null);
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
        browserTaskInputRow.setVisibility(vis);
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
        EditText input = "app".equals(currentAgentType) ? appTaskInput : browserTaskInput;
        String text = input.getText().toString().trim();
        boolean currentConnected = "app".equals(currentAgentType) ? appConnected : browserConnected;
        boolean running = "app".equals(currentAgentType) ? appRunning : browserRunning;
        if (text.isEmpty() || running || !currentConnected) return;

        if ("app".equals(currentAgentType)) {
            logText.setText("");
            appRunning = true;
        } else {
            browserRunning = true;
        }
        input.setText("");
        updateSendButton();

        appendLog("[TASK] Sending to server (" + currentAgentType + "): " + text);
        DeviceConnection.getInstance(currentAgentType).sendUserTask(text);
    }

    private void selectTab(String agentType) {
        if (agentType.equals(currentAgentType)) return;
        currentAgentType = agentType;
        if ("app".equals(agentType)) {
            appPane.setVisibility(View.VISIBLE);
            browserPane.setVisibility(View.GONE);
        } else {
            appPane.setVisibility(View.GONE);
            browserPane.setVisibility(View.VISIBLE);
        }
        updateBrowserChrome();
        updateTabs();
        updateSendButton();
    }

    private void updateTabs() {
        boolean appActive = "app".equals(currentAgentType);
        // App tab
        tabApp.setText("App");
        tabApp.setTextColor(appActive ? 0xFF1976D2 : 0xFF666666);
        tabApp.setBackgroundColor(appActive ? 0xFFFFFFFF : 0xFFE0E0E0);
        tabApp.setTypeface(null, appActive ? Typeface.BOLD : Typeface.NORMAL);
        // Browser tab
        tabBrowser.setText("Browser");
        tabBrowser.setTextColor(!appActive ? 0xFF1976D2 : 0xFF666666);
        tabBrowser.setBackgroundColor(!appActive ? 0xFFFFFFFF : 0xFFE0E0E0);
        tabBrowser.setTypeface(null, !appActive ? Typeface.BOLD : Typeface.NORMAL);
    }

    private void updateSendButton() {
        // App send button: checks a11y service
        if (!serviceRunning) {
            appSendBtn.setText("No A11y");
            appSendBtn.setEnabled(true);
            appSendBtn.setBackgroundColor(0xFFC62828);
            appTaskInput.setEnabled(false);
        } else if (appRunning) {
            appSendBtn.setText("Running");
            appSendBtn.setEnabled(false);
            appSendBtn.setBackgroundColor(0xFF666666);
            appTaskInput.setEnabled(false);
        } else {
            appSendBtn.setText(appConnected ? "Send" : "Offline");
            appSendBtn.setEnabled(appConnected);
            appSendBtn.setBackgroundColor(appConnected ? 0xFF1976D2 : 0x661976D2);
            appTaskInput.setEnabled(true);
        }
        // Browser send button: no a11y check
        if (browserRunning) {
            browserSendBtn.setText("Running");
            browserSendBtn.setEnabled(false);
            browserSendBtn.setBackgroundColor(0xFF666666);
            browserTaskInput.setEnabled(false);
        } else {
            browserSendBtn.setText(browserConnected ? "Send" : "Offline");
            browserSendBtn.setEnabled(browserConnected);
            browserSendBtn.setBackgroundColor(browserConnected ? 0xFF1976D2 : 0x661976D2);
            browserTaskInput.setEnabled(true);
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

    // --- Settings screen ---

    private void setupSettingsScreen() {
        // Source spinner
        String[] sources = {"ModelScope", "HF Mirror"};
        ArrayAdapter<String> sourceAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, sources);
        sourceAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        sourceSpinner.setAdapter(sourceAdapter);
        // Restore saved source
        if (modelManager.getSource() == ModelManager.Source.HF_MIRROR) {
            sourceSpinner.setSelection(1);
        }
        sourceSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                modelManager.setSource(pos == 0 ? ModelManager.Source.MODELSCOPE : ModelManager.Source.HF_MIRROR);
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        downloadBtn.setOnClickListener(v -> startDownload());
        loadModelBtn.setOnClickListener(v -> doLoadModel());
        testStreamBtn.setOnClickListener(v -> doTestStream());
        freeModelBtn.setOnClickListener(v -> doFreeModel());
    }

    private void openSettings() {
        refreshModelStatus();
        viewFlipper.setDisplayedChild(2);
    }

    private void refreshModelStatus() {
        if (modelManager.isModelReady()) {
            modelStatusText.setText("Model: ready" + (modelLoaded ? " (loaded)" : ""));
            downloadBtn.setText("Re-download");
            downloadBtn.setEnabled(!isDownloading);
            loadModelBtn.setEnabled(!modelLoaded);
            testStreamBtn.setEnabled(modelLoaded);
            freeModelBtn.setEnabled(modelLoaded);
        } else {
            modelStatusText.setText("Model: not downloaded");
            downloadBtn.setText("Download Model (~1.8 GB)");
            downloadBtn.setEnabled(!isDownloading);
            loadModelBtn.setEnabled(false);
            testStreamBtn.setEnabled(false);
            freeModelBtn.setEnabled(false);
        }
    }

    private void startDownload() {
        if (isDownloading) return;
        isDownloading = true;
        downloadBtn.setEnabled(false);
        downloadProgress.setVisibility(View.VISIBLE);
        downloadProgress.setProgress(0);
        downloadProgressText.setVisibility(View.VISIBLE);
        downloadProgressText.setText("Starting...");
        asrResultText.setText("");

        new Thread(() -> modelManager.download(new ModelManager.DownloadListener() {
            @Override
            public void onProgress(long downloaded, long total, String currentFile) {
                int pct = total > 0 ? (int) (downloaded * 1000 / total) : 0;
                handler.post(() -> {
                    downloadProgress.setProgress(pct);
                    downloadProgressText.setText(String.format(Locale.US,
                            "%d%% (%d/%d MB) %s",
                            pct / 10, downloaded >> 20, total >> 20, currentFile));
                });
            }

            @Override
            public void onComplete(String modelDir) {
                handler.post(() -> {
                    isDownloading = false;
                    downloadProgress.setVisibility(View.GONE);
                    downloadProgressText.setVisibility(View.GONE);
                    refreshModelStatus();
                    asrResultText.setText("Download complete: " + modelDir);
                });
            }

            @Override
            public void onError(String message) {
                handler.post(() -> {
                    isDownloading = false;
                    downloadBtn.setEnabled(true);
                    downloadProgressText.setText("Error: " + message);
                    asrResultText.setText("Download failed. Partial files kept for resume.");
                });
            }
        })).start();
    }

    private void doLoadModel() {
        if (!modelManager.isModelReady()) return;
        loadModelBtn.setEnabled(false);
        asrResultText.setText("Loading model (first time includes quantization)...");

        String cacheDir = getCacheDir().getAbsolutePath();
        new Thread(() -> {
            VoiceService.nativeSetCacheDir(cacheDir);
            boolean ok = VoiceService.nativeLoadModel(modelManager.getModelDir(), 4);
            handler.post(() -> {
                modelLoaded = ok;
                refreshModelStatus();
                asrResultText.setText(ok ? "Model loaded successfully" : "Model load failed");
            });
        }).start();
    }

    private static final int REQUEST_MIC = 1001;

    private void doTestStream() {
        if (!modelLoaded) return;

        if (isAsrTesting) {
            stopAsrTest();
            return;
        }

        // Check mic permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_MIC);
            return;
        }

        startAsrTest();
    }

    private void startAsrTest() {
        int bufSize = android.media.AudioRecord.getMinBufferSize(
                16000, android.media.AudioFormat.CHANNEL_IN_MONO,
                android.media.AudioFormat.ENCODING_PCM_16BIT);
        try {
            testRecorder = new android.media.AudioRecord(
                    android.media.MediaRecorder.AudioSource.MIC,
                    16000, android.media.AudioFormat.CHANNEL_IN_MONO,
                    android.media.AudioFormat.ENCODING_PCM_16BIT,
                    Math.max(bufSize, 16000 * 2));
        } catch (SecurityException e) {
            asrResultText.setText("No mic permission");
            return;
        }

        if (testRecorder.getState() != android.media.AudioRecord.STATE_INITIALIZED) {
            asrResultText.setText("Failed to init AudioRecord");
            testRecorder.release();
            testRecorder = null;
            return;
        }

        // Set up token listener
        VoiceService.setTokenListener(piece -> handler.post(() -> asrResultText.append(piece)));

        // Start native ASR
        if (!VoiceService.nativeStartAsr()) {
            asrResultText.setText("nativeStartAsr failed");
            testRecorder.release();
            testRecorder = null;
            VoiceService.setTokenListener(null);
            return;
        }

        isAsrTesting = true;
        asrResultText.setText("");
        testStreamBtn.setText("Stop ASR");
        testStreamBtn.setBackgroundColor(0xFFF44336);
        loadModelBtn.setEnabled(false);
        freeModelBtn.setEnabled(false);

        testRecorder.startRecording();
        testRecordThread = new Thread(() -> {
            short[] buf = new short[1600]; // 100ms chunks
            while (isAsrTesting) {
                int read = testRecorder.read(buf, 0, buf.length);
                if (read > 0) {
                    VoiceService.nativePushAudio(buf, read);
                }
            }
        }, "AsrTestRecord");
        testRecordThread.start();
    }

    private void stopAsrTest() {
        isAsrTesting = false;

        if (testRecordThread != null) {
            try { testRecordThread.join(2000); } catch (InterruptedException ignored) {}
            testRecordThread = null;
        }

        if (testRecorder != null) {
            testRecorder.stop();
            testRecorder.release();
            testRecorder = null;
        }

        VoiceService.nativeStopAsr();
        VoiceService.setTokenListener(null);

        testStreamBtn.setText("Start ASR");
        testStreamBtn.setBackgroundColor(0xFFFF9800);
        refreshModelStatus();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_MIC && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            doTestStream();
        } else if (requestCode == REQUEST_MIC) {
            asrResultText.setText("Microphone permission denied");
        }
    }

    private void doFreeModel() {
        VoiceService.nativeFreeModel();
        modelLoaded = false;
        refreshModelStatus();
        asrResultText.setText("Model freed");
    }

    private void onLoginComplete() {
        showTaskScreen();
        connectCloud();
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
            updateTabs();
            updateSendButton();
        });
    }

    @Override
    public void onTaskDone(String agentType, String result) {
        handler.post(() -> {
            appendLog("[DONE] (" + agentType + ") " + result);
            if ("app".equals(agentType)) {
                appRunning = false;
            } else {
                browserRunning = false;
            }
            updateSendButton();
        });
    }

    @Override
    public void onUnauthorized(String agentType) {
        handler.post(() -> {
            Log.w(TAG, "Token rejected (401) from " + agentType + ", clearing credentials");
            // Clear stored token
            apiKey = null;
            getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit().clear().apply();
            // Disconnect both agents
            DeviceConnection.getInstance("app").disconnect();
            DeviceConnection.getInstance("browser").disconnect();
            // Reset connection status
            appConnected = false;
            browserConnected = false;
            updateTabs();
            // Show login screen
            viewFlipper.setDisplayedChild(0);
            loginBtn.setVisibility(View.VISIBLE);
            deviceCodePanel.setVisibility(View.GONE);
            loginErrorPanel.setVisibility(View.GONE);
        });
    }
}
