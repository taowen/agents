package com.example.androidbrowser;

import android.graphics.Bitmap;
import android.util.Base64;
import android.util.Log;
import android.view.ViewGroup;
import android.webkit.ValueCallback;
import android.webkit.WebView;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import org.json.JSONObject;

/**
 * Plain Java utility class called directly from AgentRunner's background thread.
 * No @JavascriptInterface — no hidden WebView needed.
 */
public class AgentBridge {

    private static final String TAG = "AgentBridge";
    private static final int TIMEOUT_SECONDS = 30;

    private static final String DESKTOP_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0";
    private static final String MOBILE_UA = "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Mobile Safari/537.36 EdgA/145.0.0.0";

    private final MainActivity activity;
    private final WebView visibleWebView;
    private String domExtractorJs;  // cached after first load
    private LogCallback logCallback;
    private ViewGroup.LayoutParams originalLayoutParams;

    // Page load tracking
    private volatile CountDownLatch pageLoadLatch;
    private volatile boolean pageLoading = false;

    public interface LogCallback {
        void onLog(String message);
    }

    public AgentBridge(MainActivity activity, WebView visibleWebView) {
        this.activity = activity;
        this.visibleWebView = visibleWebView;
    }

    public void setLogCallback(LogCallback callback) {
        this.logCallback = callback;
    }

    /** Called from WebViewClient.onPageStarted on UI thread. */
    public void onPageStarted() {
        // Don't replace a pre-armed latch (e.g. from navigateTo)
        if (!pageLoading) {
            pageLoadLatch = new CountDownLatch(1);
        }
        pageLoading = true;
    }

    /** Called from WebViewClient.onPageFinished on UI thread. */
    public void onPageFinished() {
        pageLoading = false;
        CountDownLatch latch = pageLoadLatch;
        if (latch != null) {
            latch.countDown();
        }
    }

    /**
     * Block the agent thread until the current page load finishes or timeout.
     * Polls briefly for navigation to start (handles clicks that trigger navigation).
     */
    public void waitForPageLoad(int timeoutMs) {
        // Poll briefly for navigation to start (onPageStarted fires on UI thread)
        if (!pageLoading) {
            long deadline = System.currentTimeMillis() + 300;
            while (!pageLoading && System.currentTimeMillis() < deadline) {
                try { Thread.sleep(50); } catch (InterruptedException e) { return; }
            }
        }
        CountDownLatch latch = pageLoadLatch;
        if (latch == null || !pageLoading) return;
        try {
            latch.await(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Log.w(TAG, "waitForPageLoad interrupted", e);
        }
    }

    public String getDomTree() {
        // Load and cache dom-extractor.js
        if (domExtractorJs == null) {
            try {
                InputStream is = activity.getAssets().open("dom-extractor.js");
                BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    sb.append(line).append("\n");
                }
                reader.close();
                domExtractorJs = sb.toString();
            } catch (Exception e) {
                Log.e(TAG, "Failed to load dom-extractor.js", e);
                return "ERROR: failed to load dom-extractor.js: " + e.getMessage();
            }
        }

        // Inject the extractor script, then read the result
        evaluateOnVisibleWebView(domExtractorJs);
        String tree = evaluateOnVisibleWebView("window.__agentDomTree || ''");
        return tree;
    }

    public boolean clickElement(int id) {
        String js = "(function() {" +
                "var el = window.__agentElements && window.__agentElements[" + id + "];" +
                "if (!el) return 'false';" +
                "var rect = el.getBoundingClientRect();" +
                "var cx = rect.left + rect.width / 2;" +
                "var cy = rect.top + rect.height / 2;" +
                "var opts = {bubbles:true, cancelable:true, clientX:cx, clientY:cy};" +
                "el.dispatchEvent(new PointerEvent('pointerdown', opts));" +
                "el.dispatchEvent(new MouseEvent('mousedown', opts));" +
                "el.dispatchEvent(new PointerEvent('pointerup', opts));" +
                "el.dispatchEvent(new MouseEvent('mouseup', opts));" +
                "el.click();" +
                "return 'true';" +
                "})()";
        String result = evaluateOnVisibleWebView(js);
        return "true".equals(result);
    }

    public boolean typeText(int id, String text) {
        String escaped = text.replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace("\n", "\\n")
                .replace("\r", "\\r");

        String js = "(function() {" +
                "try {" +
                "var el = window.__agentElements && window.__agentElements[" + id + "];" +
                "if (!el) return 'false';" +
                "el.focus();" +
                "var proto = (el.tagName === 'TEXTAREA')" +
                "  ? window.HTMLTextAreaElement.prototype" +
                "  : window.HTMLInputElement.prototype;" +
                "var desc = Object.getOwnPropertyDescriptor(proto, 'value');" +
                "if (desc && desc.set) {" +
                "  desc.set.call(el, '');" +
                "  desc.set.call(el, '" + escaped + "');" +
                "} else {" +
                "  el.value = '';" +
                "  el.value = '" + escaped + "';" +
                "}" +
                "el.dispatchEvent(new InputEvent('input', {bubbles:true, inputType:'insertText', data:'" + escaped + "'}));" +
                "el.dispatchEvent(new Event('change', {bubbles:true}));" +
                "return 'true';" +
                "} catch(e) { return 'false'; }" +
                "})()";
        String result = evaluateOnVisibleWebView(js);
        return "true".equals(result);
    }

    public boolean scrollPage(String direction) {
        String js;
        if ("up".equals(direction)) {
            js = "window.scrollBy(0, -window.innerHeight * 0.8); 'true'";
        } else {
            js = "window.scrollBy(0, window.innerHeight * 0.8); 'true'";
        }
        String result = evaluateOnVisibleWebView(js);
        return "true".equals(result);
    }

    public boolean navigateTo(String url) {
        // Pre-arm the page load latch before loadUrl fires onPageStarted
        onPageStarted();
        final CountDownLatch latch = new CountDownLatch(1);
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                visibleWebView.loadUrl(url);
                latch.countDown();
            }
        });
        try {
            latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "navigateTo interrupted", e);
        }
        return true;
    }

    public boolean goBack() {
        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicReference<Boolean> result = new AtomicReference<>(false);
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (visibleWebView.canGoBack()) {
                    visibleWebView.goBack();
                    result.set(true);
                }
                latch.countDown();
            }
        });
        try {
            latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "goBack interrupted", e);
        }
        return result.get();
    }

    public String takeScreenshot() {
        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicReference<String> result = new AtomicReference<>("ERROR: timeout");
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                try {
                    Bitmap bitmap = Bitmap.createBitmap(
                            visibleWebView.getWidth(),
                            visibleWebView.getHeight(),
                            Bitmap.Config.ARGB_8888);
                    android.graphics.Canvas canvas = new android.graphics.Canvas(bitmap);
                    visibleWebView.draw(canvas);

                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 70, baos);
                    bitmap.recycle();
                    result.set(Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP));
                } catch (Exception e) {
                    Log.e(TAG, "takeScreenshot error", e);
                    result.set("ERROR: " + e.getMessage());
                }
                latch.countDown();
            }
        });
        try {
            latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "takeScreenshot interrupted", e);
        }
        return result.get();
    }

    public String switchUserAgent(String mode) {
        final String ua = "pc".equalsIgnoreCase(mode) ? DESKTOP_UA : MOBILE_UA;
        final String label = "pc".equalsIgnoreCase(mode) ? "PC" : "Mobile";
        final CountDownLatch latch = new CountDownLatch(1);
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                visibleWebView.getSettings().setUserAgentString(ua);
                visibleWebView.reload();
                latch.countDown();
            }
        });
        try {
            latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "switchUserAgent interrupted", e);
        }
        return "Switched to " + label + " User-Agent, page reloading";
    }

    public String setViewport(int width, int height) {
        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicReference<String> result = new AtomicReference<>("ERROR: timeout");
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                // Save original layout params on first call
                if (originalLayoutParams == null) {
                    originalLayoutParams = new ViewGroup.LayoutParams(visibleWebView.getLayoutParams());
                }
                if (width <= 0 || height <= 0) {
                    // Restore original
                    ViewGroup.LayoutParams lp = visibleWebView.getLayoutParams();
                    lp.width = originalLayoutParams.width;
                    lp.height = originalLayoutParams.height;
                    visibleWebView.setLayoutParams(lp);
                    visibleWebView.requestLayout();
                    result.set("Viewport restored to default");
                } else {
                    float density = activity.getResources().getDisplayMetrics().density;
                    int pxWidth = (int)(width * density);
                    int pxHeight = (int)(height * density);
                    ViewGroup.LayoutParams lp = visibleWebView.getLayoutParams();
                    lp.width = pxWidth;
                    lp.height = pxHeight;
                    visibleWebView.setLayoutParams(lp);
                    visibleWebView.requestLayout();
                    result.set("Viewport set to " + width + "x" + height);
                }
                latch.countDown();
            }
        });
        try {
            latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "setViewport interrupted", e);
        }
        // Wait for reflow
        try { Thread.sleep(500); } catch (InterruptedException ignored) {}
        return result.get();
    }

    public void log(String msg) {
        Log.d(TAG, msg);
        if (logCallback != null) {
            logCallback.onLog(msg);
        }
    }

    /**
     * Evaluate JavaScript on the visible WebView (UI thread) and wait for the result.
     * Safe to call from a background thread (no JS thread contention).
     */
    private String evaluateOnVisibleWebView(String js) {
        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicReference<String> result = new AtomicReference<>("");
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                visibleWebView.evaluateJavascript(js, new ValueCallback<String>() {
                    @Override
                    public void onReceiveValue(String value) {
                        // evaluateJavascript wraps string results in quotes
                        if (value != null && value.startsWith("\"") && value.endsWith("\"")) {
                            try {
                                value = new JSONObject("{\"v\":" + value + "}").getString("v");
                            } catch (Exception e) {
                                value = value.substring(1, value.length() - 1);
                            }
                        }
                        if (value == null || "null".equals(value)) {
                            value = "";
                        }
                        result.set(value);
                        latch.countDown();
                    }
                });
            }
        });
        try {
            boolean completed = latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            if (!completed) {
                Log.w(TAG, "evaluateOnVisibleWebView: TIMEOUT after " + TIMEOUT_SECONDS + "s");
            }
        } catch (InterruptedException e) {
            Log.e(TAG, "evaluateOnVisibleWebView interrupted", e);
        }
        return result.get();
    }
}
