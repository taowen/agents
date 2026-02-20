package com.example.androidbrowser;

import android.graphics.Bitmap;
import android.net.http.SslError;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.webkit.SslErrorHandler;
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
import android.widget.ProgressBar;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Browser";
    private static final String DEFAULT_URL = "https://www.google.com";
    private static final String DESKTOP_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0";
    private static final String MOBILE_UA = "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Mobile Safari/537.36 EdgA/145.0.0.0";

    private WebView webView;
    private EditText urlInput;
    private ImageButton btnBack;
    private ImageButton btnForward;
    private ImageButton btnRefresh;
    private Button btnGo;
    private Button btnToggleUA;
    private ProgressBar progressBar;
    private boolean isDesktopUA = true;

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

        setupWebView();
        setupControls();

        String intentUrl = getIntent().getStringExtra("url");
        String startUrl = (intentUrl != null && !intentUrl.isEmpty()) ? intentUrl : DEFAULT_URL;
        Log.d(TAG, "Loading start URL: " + startUrl);
        webView.loadUrl(startUrl);
    }

    private void setupWebView() {
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setUseWideViewPort(true);
        settings.setLoadWithOverviewMode(true);
        settings.setBuiltInZoomControls(true);
        settings.setDisplayZoomControls(false);
        settings.setSupportMultipleWindows(false);

        // Default to desktop UA to avoid Google OAuth blocking and mobile app redirects
        settings.setUserAgentString(DESKTOP_UA);
        Log.d(TAG, "User-Agent: " + DESKTOP_UA);

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
                // and would prevent the page's fallback web behavior from kicking in
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
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                Log.d(TAG, "onPageFinished: " + url);
                updateNavButtons();
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
