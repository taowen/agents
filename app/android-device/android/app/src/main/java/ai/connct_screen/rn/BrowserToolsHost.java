package ai.connct_screen.rn;

import android.app.Activity;
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
 * Java-side callbacks for the "browser" agent type (WebView-based web automation).
 * Each static method is called from C++ (tools_browser.cpp) via JNI.
 *
 * Ported from examples/android-browser AgentBridge.java as a static methods class.
 */
public class BrowserToolsHost {

    private static final String TAG = "BrowserToolsHost";
    private static final int TIMEOUT_SECONDS = 30;

    private static final String DESKTOP_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0";
    private static final String MOBILE_UA = "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Mobile Safari/537.36 EdgA/145.0.0.0";

    private static Activity activity;
    private static WebView webView;
    private static String domExtractorJs;  // cached after first load
    private static ViewGroup.LayoutParams originalLayoutParams;

    // Page load tracking
    private static volatile CountDownLatch pageLoadLatch;
    private static volatile boolean pageLoading = false;

    public static void init(Activity act, WebView wv) {
        activity = act;
        webView = wv;
    }

    /** Called from WebViewClient.onPageStarted on UI thread. */
    public static void onPageStarted() {
        if (!pageLoading) {
            pageLoadLatch = new CountDownLatch(1);
        }
        pageLoading = true;
    }

    /** Called from WebViewClient.onPageFinished on UI thread. */
    public static void onPageFinished() {
        pageLoading = false;
        CountDownLatch latch = pageLoadLatch;
        if (latch != null) {
            latch.countDown();
        }
    }

    /**
     * Block the caller thread until the current page load finishes or timeout.
     */
    public static void waitForPageLoad(int timeoutMs) {
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

    // --- Native methods called from C++ via JNI ---

    public static String nativeGetPage() {
        if (webView == null) return "ERROR: WebView not initialized";

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

        evaluateOnWebView(domExtractorJs);
        return evaluateOnWebView("window.__agentDomTree || ''");
    }

    public static boolean nativeClickElement(int id) {
        if (webView == null) return false;

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
        String result = evaluateOnWebView(js);
        waitForPageLoad(5000);
        return "true".equals(result);
    }

    public static boolean nativeTypeText(int id, String text) {
        if (webView == null) return false;

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
        return "true".equals(evaluateOnWebView(js));
    }

    public static boolean nativeGotoUrl(String url) {
        if (webView == null) return false;

        // Pre-arm the page load latch
        onPageStarted();
        final CountDownLatch latch = new CountDownLatch(1);
        activity.runOnUiThread(() -> {
            webView.loadUrl(url);
            latch.countDown();
        });
        try {
            latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "gotoUrl interrupted", e);
        }
        waitForPageLoad(10000);
        return true;
    }

    public static boolean nativeScrollPage(String direction) {
        if (webView == null) return false;

        String js;
        if ("up".equals(direction)) {
            js = "window.scrollBy(0, -window.innerHeight * 0.8); 'true'";
        } else {
            js = "window.scrollBy(0, window.innerHeight * 0.8); 'true'";
        }
        return "true".equals(evaluateOnWebView(js));
    }

    public static boolean nativeGoBack() {
        if (webView == null) return false;

        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicReference<Boolean> result = new AtomicReference<>(false);
        activity.runOnUiThread(() -> {
            if (webView.canGoBack()) {
                webView.goBack();
                result.set(true);
            }
            latch.countDown();
        });
        try {
            latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "goBack interrupted", e);
        }
        if (result.get()) {
            waitForPageLoad(5000);
        }
        return result.get();
    }

    public static String nativeTakeScreenshot() {
        if (webView == null) return "ERROR: WebView not initialized";

        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicReference<String> result = new AtomicReference<>("ERROR: timeout");
        activity.runOnUiThread(() -> {
            try {
                Bitmap bitmap = Bitmap.createBitmap(
                        webView.getWidth(),
                        webView.getHeight(),
                        Bitmap.Config.ARGB_8888);
                android.graphics.Canvas canvas = new android.graphics.Canvas(bitmap);
                webView.draw(canvas);

                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                bitmap.compress(Bitmap.CompressFormat.JPEG, 70, baos);
                bitmap.recycle();
                result.set(Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP));
            } catch (Exception e) {
                Log.e(TAG, "takeScreenshot error", e);
                result.set("ERROR: " + e.getMessage());
            }
            latch.countDown();
        });
        try {
            latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "takeScreenshot interrupted", e);
        }
        return result.get();
    }

    public static String nativeSwitchUa(String mode) {
        if (webView == null) return "ERROR: WebView not initialized";

        final String ua = "pc".equalsIgnoreCase(mode) ? DESKTOP_UA : MOBILE_UA;
        final String label = "pc".equalsIgnoreCase(mode) ? "PC" : "Mobile";
        final CountDownLatch latch = new CountDownLatch(1);
        activity.runOnUiThread(() -> {
            webView.getSettings().setUserAgentString(ua);
            webView.reload();
            latch.countDown();
        });
        try {
            latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "switchUa interrupted", e);
        }
        waitForPageLoad(10000);
        return "Switched to " + label + " User-Agent, page reloading";
    }

    public static String nativeSetViewport(int width, int height) {
        if (webView == null) return "ERROR: WebView not initialized";

        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicReference<String> result = new AtomicReference<>("ERROR: timeout");
        activity.runOnUiThread(() -> {
            if (originalLayoutParams == null) {
                originalLayoutParams = new ViewGroup.LayoutParams(webView.getLayoutParams());
            }
            if (width <= 0 || height <= 0) {
                ViewGroup.LayoutParams lp = webView.getLayoutParams();
                lp.width = originalLayoutParams.width;
                lp.height = originalLayoutParams.height;
                webView.setLayoutParams(lp);
                webView.requestLayout();
                result.set("Viewport restored to default");
            } else {
                float density = activity.getResources().getDisplayMetrics().density;
                int pxWidth = (int)(width * density);
                int pxHeight = (int)(height * density);
                ViewGroup.LayoutParams lp = webView.getLayoutParams();
                lp.width = pxWidth;
                lp.height = pxHeight;
                webView.setLayoutParams(lp);
                webView.requestLayout();
                result.set("Viewport set to " + width + "x" + height);
            }
            latch.countDown();
        });
        try {
            latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Log.e(TAG, "setViewport interrupted", e);
        }
        try { Thread.sleep(500); } catch (InterruptedException ignored) {}
        return result.get();
    }

    // --- Internal helper ---

    /**
     * Evaluate JavaScript on the WebView (UI thread) and wait for the result.
     * Safe to call from a background thread.
     */
    private static String evaluateOnWebView(String js) {
        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicReference<String> result = new AtomicReference<>("");
        activity.runOnUiThread(() -> {
            webView.evaluateJavascript(js, value -> {
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
            });
        });
        try {
            boolean completed = latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            if (!completed) {
                Log.w(TAG, "evaluateOnWebView: TIMEOUT after " + TIMEOUT_SECONDS + "s");
            }
        } catch (InterruptedException e) {
            Log.e(TAG, "evaluateOnWebView interrupted", e);
        }
        return result.get();
    }
}
