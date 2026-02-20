package com.example.androidbrowser;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Browser";
    private static final String DEFAULT_URL = "https://www.google.com";
    private static final String DESKTOP_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0";
    private static final String MOBILE_UA = "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Mobile Safari/537.36 EdgA/145.0.0.0";
    private static final String PREFS_NAME = "agent_prefs";
    private static final String SERVER_URL = "https://ai.connect-screen.com";
    private static final int POLL_INTERVAL_MS = 2000;

    private WebView webView;
    private EditText urlInput;
    private ImageButton btnBack;
    private ImageButton btnForward;
    private ImageButton btnRefresh;
    private Button btnGo;
    private Button btnToggleUA;
    private ProgressBar progressBar;
    private boolean isDesktopUA = false;

    // Agent UI
    private EditText taskInput;
    private Button btnRun;
    private ScrollView logScroll;
    private TextView agentLog;
    private AgentBridge agentBridge;
    private boolean agentRunning = false;

    // Login UI
    private LinearLayout browserContainer;
    private LinearLayout loginPanel;
    private TextView loginCode;
    private TextView loginStatus;
    private TextView loginInstructions;
    private Button btnLogin;
    private Handler pollHandler;
    private String pendingCode;
    private String pendingTask; // task received via intent before login

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        webView = findViewById(R.id.webview);
        urlInput = findViewById(R.id.url_input);
        btnBack = findViewById(R.id.btn_back);
        btnForward = findViewById(R.id.btn_forward);
        btnRefresh = findViewById(R.id.btn_refresh);
        btnGo = findViewById(R.id.btn_go);
        btnToggleUA = findViewById(R.id.btn_toggle_ua);
        progressBar = findViewById(R.id.progress_bar);

        // Agent UI
        taskInput = findViewById(R.id.task_input);
        btnRun = findViewById(R.id.btn_run);
        logScroll = findViewById(R.id.log_scroll);
        agentLog = findViewById(R.id.agent_log);

        // Login UI
        browserContainer = findViewById(R.id.browser_container);
        loginPanel = findViewById(R.id.login_panel);
        loginCode = findViewById(R.id.login_code);
        loginStatus = findViewById(R.id.login_status);
        loginInstructions = findViewById(R.id.login_instructions);
        btnLogin = findViewById(R.id.btn_login);
        pollHandler = new Handler(Looper.getMainLooper());

        setupWebView();
        setupControls();
        setupAgent();
        setupLogin();

        // Migrate old SharedPreferences keys
        migratePrefs();

        // Check login state — show login or browser
        if (checkLoginState()) {
            String intentUrl = getIntent().getStringExtra("url");
            String startUrl = (intentUrl != null && !intentUrl.isEmpty()) ? intentUrl : DEFAULT_URL;
            Log.d(TAG, "Loading start URL: " + startUrl);
            webView.loadUrl(startUrl);
            handleAgentTaskIntent(getIntent());
        } else {
            // Not logged in — check if there's a pending task from intent
            String task = getIntent() != null ? getIntent().getStringExtra("agent_task") : null;
            if (task != null && !task.isEmpty()) {
                pendingTask = task;
            }
        }
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        handleAgentTaskIntent(intent);
    }

    private void handleAgentTaskIntent(Intent intent) {
        if (intent == null) return;
        String task = intent.getStringExtra("agent_task");
        if (task != null && !task.isEmpty()) {
            // If not logged in, store task for after login
            SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
            String token = prefs.getString("device_token", "");
            if (token.isEmpty()) {
                pendingTask = task;
                return;
            }
            taskInput.setText(task);
            taskInput.postDelayed(() -> startAgent(task), 500);
        }
    }

    // ---- Login flow ----

    private void migratePrefs() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        // Remove old LLM config keys if they exist
        if (prefs.contains("llm_base_url") || prefs.contains("llm_api_key") || prefs.contains("llm_model")) {
            prefs.edit()
                .remove("llm_base_url")
                .remove("llm_api_key")
                .remove("llm_model")
                .apply();
            Log.d(TAG, "Migrated: removed old LLM config keys");
        }
    }

    /**
     * Returns true if logged in (browser should be shown), false if login needed.
     */
    private boolean checkLoginState() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String token = prefs.getString("device_token", "");
        if (!token.isEmpty()) {
            loginPanel.setVisibility(View.GONE);
            browserContainer.setVisibility(View.VISIBLE);
            return true;
        } else {
            loginPanel.setVisibility(View.VISIBLE);
            browserContainer.setVisibility(View.GONE);
            return false;
        }
    }

    private void setupLogin() {
        btnLogin.setOnClickListener(v -> startDeviceLogin());
    }

    private void startDeviceLogin() {
        btnLogin.setEnabled(false);
        btnLogin.setText("Requesting...");
        loginCode.setVisibility(View.GONE);
        loginStatus.setVisibility(View.GONE);

        new Thread(() -> {
            try {
                URL url = new URL(SERVER_URL + "/auth/device/start");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setDoOutput(true);
                conn.setConnectTimeout(10000);
                conn.setReadTimeout(10000);
                conn.setRequestProperty("Content-Type", "application/json");

                OutputStream os = conn.getOutputStream();
                os.write("{}".getBytes("UTF-8"));
                os.close();

                int code = conn.getResponseCode();
                InputStream is = (code >= 200 && code < 400)
                        ? conn.getInputStream()
                        : conn.getErrorStream();
                BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) sb.append(line);
                reader.close();
                conn.disconnect();

                JSONObject data = new JSONObject(sb.toString());
                String deviceCode = data.getString("code");

                runOnUiThread(() -> {
                    pendingCode = deviceCode;
                    loginCode.setText(deviceCode);
                    loginCode.setVisibility(View.VISIBLE);
                    loginInstructions.setText("Enter this code at ai.connect-screen.com/device");
                    loginStatus.setText("Waiting for approval...");
                    loginStatus.setVisibility(View.VISIBLE);
                    btnLogin.setText("Waiting...");

                    // Start polling
                    pollHandler.postDelayed(this::pollDeviceApproval, POLL_INTERVAL_MS);
                });

            } catch (Exception e) {
                Log.e(TAG, "Device login start failed", e);
                runOnUiThread(() -> {
                    loginStatus.setText("Error: " + e.getMessage());
                    loginStatus.setVisibility(View.VISIBLE);
                    btnLogin.setEnabled(true);
                    btnLogin.setText("Login");
                });
            }
        }, "DeviceLoginStart").start();
    }

    private void pollDeviceApproval() {
        if (pendingCode == null) return;

        new Thread(() -> {
            try {
                URL url = new URL(SERVER_URL + "/auth/device/check?code=" + pendingCode);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setConnectTimeout(10000);
                conn.setReadTimeout(10000);

                int code = conn.getResponseCode();
                InputStream is = (code >= 200 && code < 400)
                        ? conn.getInputStream()
                        : conn.getErrorStream();
                BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) sb.append(line);
                reader.close();
                conn.disconnect();

                JSONObject data = new JSONObject(sb.toString());
                String status = data.optString("status", "");

                runOnUiThread(() -> {
                    switch (status) {
                        case "approved":
                            String token = data.optString("token", "");
                            String baseURL = data.optString("baseURL", "");
                            if (!token.isEmpty() && !baseURL.isEmpty()) {
                                // Save credentials
                                getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit()
                                    .putString("device_token", token)
                                    .putString("device_base_url", baseURL)
                                    .apply();
                                Log.d(TAG, "Device login approved, token saved");

                                pendingCode = null;
                                onLoginSuccess();
                            }
                            break;

                        case "expired":
                            pendingCode = null;
                            loginStatus.setText("Code expired. Try again.");
                            loginCode.setVisibility(View.GONE);
                            btnLogin.setEnabled(true);
                            btnLogin.setText("Login");
                            break;

                        default: // "pending"
                            pollHandler.postDelayed(this::pollDeviceApproval, POLL_INTERVAL_MS);
                            break;
                    }
                });

            } catch (Exception e) {
                Log.e(TAG, "Device poll failed", e);
                runOnUiThread(() -> {
                    // Retry on network error
                    pollHandler.postDelayed(this::pollDeviceApproval, POLL_INTERVAL_MS);
                });
            }
        }, "DeviceLoginPoll").start();
    }

    private void onLoginSuccess() {
        loginPanel.setVisibility(View.GONE);
        browserContainer.setVisibility(View.VISIBLE);

        // Load default page
        webView.loadUrl(DEFAULT_URL);

        // If there was a pending task from intent, start it
        if (pendingTask != null && !pendingTask.isEmpty()) {
            String task = pendingTask;
            pendingTask = null;
            taskInput.setText(task);
            taskInput.postDelayed(() -> startAgent(task), 500);
        }
    }

    // ---- WebView setup ----

    private void setupWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setUseWideViewPort(true);
        settings.setLoadWithOverviewMode(true);
        settings.setBuiltInZoomControls(true);
        settings.setDisplayZoomControls(false);
        settings.setSupportMultipleWindows(false);

        // Default to mobile UA — phone screen renders mobile layouts better
        settings.setUserAgentString(MOBILE_UA);
        Log.d(TAG, "User-Agent: " + MOBILE_UA);

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                String url = request.getUrl().toString();
                String scheme = request.getUrl().getScheme();
                Log.d(TAG, "shouldOverrideUrlLoading: " + url
                        + " isRedirect=" + request.isRedirect()
                        + " hasGesture=" + request.hasGesture()
                        + " method=" + request.getMethod());
                // Block non-http(s) schemes (e.g. slack://) — they can't load in WebView
                if (scheme != null && !scheme.equals("https") && !scheme.equals("http")) {
                    Log.d(TAG, "Blocked non-http scheme: " + scheme);
                    return true;
                }
                return false;
            }

            @Override
            public void onPageStarted(WebView view, String url, Bitmap favicon) {
                Log.d(TAG, "onPageStarted: " + url);
                urlInput.setText(url);
                updateNavButtons();
                if (agentBridge != null) agentBridge.onPageStarted();
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                Log.d(TAG, "onPageFinished: " + url);
                updateNavButtons();
                if (agentBridge != null) agentBridge.onPageFinished();
            }

            @Override
            public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                Log.d(TAG, "onReceivedError: " + request.getUrl()
                        + " code=" + error.getErrorCode()
                        + " desc=" + error.getDescription());
            }

            @Override
            public void onReceivedHttpError(WebView view, WebResourceRequest request, WebResourceResponse response) {
                Log.d(TAG, "onReceivedHttpError: " + request.getUrl()
                        + " status=" + response.getStatusCode());
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onProgressChanged(WebView view, int newProgress) {
                if (newProgress < 100) {
                    progressBar.setVisibility(View.VISIBLE);
                    progressBar.setProgress(newProgress);
                } else {
                    progressBar.setVisibility(View.GONE);
                }
            }
        });
    }

    private void setupControls() {
        btnGo.setOnClickListener(v -> navigateToUrl());

        btnBack.setOnClickListener(v -> {
            if (webView.canGoBack()) {
                webView.goBack();
            }
        });

        btnForward.setOnClickListener(v -> {
            if (webView.canGoForward()) {
                webView.goForward();
            }
        });

        btnRefresh.setOnClickListener(v -> webView.reload());

        btnToggleUA.setOnClickListener(v -> {
            isDesktopUA = !isDesktopUA;
            String ua = isDesktopUA ? DESKTOP_UA : MOBILE_UA;
            btnToggleUA.setText(isDesktopUA ? "PC" : "Mobile");
            webView.getSettings().setUserAgentString(ua);
            Log.d(TAG, "Switched UA: " + ua);
            webView.reload();
        });

        urlInput.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_GO
                    || (event != null && event.getKeyCode() == KeyEvent.KEYCODE_ENTER
                        && event.getAction() == KeyEvent.ACTION_DOWN)) {
                navigateToUrl();
                return true;
            }
            return false;
        });

        updateNavButtons();
    }

    private void setupAgent() {
        // Create bridge as a plain Java object (no hidden WebView)
        agentBridge = new AgentBridge(this, webView);
        agentBridge.setLogCallback(message -> {
            runOnUiThread(() -> {
                agentLog.append(message + "\n");
                logScroll.post(() -> logScroll.fullScroll(View.FOCUS_DOWN));
            });
        });

        // Agent run button
        btnRun.setOnClickListener(v -> {
            String task = taskInput.getText().toString().trim();
            if (!task.isEmpty() && !agentRunning) {
                startAgent(task);
            }
        });
    }

    public void startAgent(String task) {
        if (agentRunning) {
            Log.w(TAG, "Agent already running");
            return;
        }

        // Read device credentials from SharedPreferences
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String token = prefs.getString("device_token", "");
        String baseURL = prefs.getString("device_base_url", "");

        if (token.isEmpty() || baseURL.isEmpty()) {
            // Not logged in — show login panel
            pendingTask = task;
            checkLoginState();
            return;
        }

        agentRunning = true;
        btnRun.setEnabled(false);
        btnRun.setText("Running...");
        logScroll.setVisibility(View.VISIBLE);
        agentLog.setText("");

        Runnable onDone = () -> runOnUiThread(() -> {
            agentRunning = false;
            btnRun.setEnabled(true);
            btnRun.setText("Run");
        });

        AgentRunner runner = new AgentRunner(task, baseURL, token, agentBridge, onDone);
        new Thread(runner, "AgentRunner").start();
    }

    private void navigateToUrl() {
        String url = urlInput.getText().toString().trim();
        if (url.isEmpty()) {
            return;
        }
        if (!url.contains("://")) {
            url = "https://" + url;
        }
        webView.loadUrl(url);
    }

    private void updateNavButtons() {
        boolean canGoBack = webView.canGoBack();
        boolean canGoForward = webView.canGoForward();

        btnBack.setEnabled(canGoBack);
        btnBack.setAlpha(canGoBack ? 1.0f : 0.3f);

        btnForward.setEnabled(canGoForward);
        btnForward.setAlpha(canGoForward ? 1.0f : 0.3f);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        pollHandler.removeCallbacksAndMessages(null);
    }

    @SuppressWarnings("deprecation")
    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }
}
