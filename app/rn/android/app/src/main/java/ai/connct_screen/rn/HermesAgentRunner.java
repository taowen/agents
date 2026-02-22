package ai.connct_screen.rn;

import android.content.Context;
import android.util.Log;

import com.google.android.accessibility.selecttospeak.SelectToSpeakService;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Iterator;
import java.util.Locale;

import org.json.JSONObject;

/**
 * Wraps a standalone Hermes JS runtime that runs independently of React Native.
 * The agent loop (LLM calls + tool execution) runs entirely inside this Hermes
 * instance, which lives in the AccessibilityService process and is unaffected
 * by Android's background-app restrictions on the RN JS thread.
 */
public class HermesAgentRunner {

    private static final String TAG = "HermesAgent";
    private static int screenCounter = 0;

    // Load the standalone native library
    static {
        System.loadLibrary("hermesagent");
    }

    // --- Native methods (implemented in standalone_hermes.cpp) ---
    // Package-private so DeviceConnection can also use the runtime
    static native void nativeCreateRuntime();
    static native String nativeEvaluateJS(String code, String sourceURL);
    static native void nativeDestroyRuntime();

    // --- Static callbacks invoked from C++ via JNI ---

    public static String nativeTakeScreenshot() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return "ERROR: accessibility service not running";
        return service.takeScreenshotSync();
    }

    public static String nativeGetScreen() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return "(accessibility service not running)";
        String tree = service.getAccessibilityTree();
        screenCounter++;
        // Save screen dump to file for debugging
        try {
            Context ctx = service.getApplicationContext();
            File screensDir = new File(ctx.getFilesDir(), "screens");
            if (!screensDir.exists()) screensDir.mkdirs();
            String filename = String.format(Locale.US, "screen_%03d.txt", screenCounter);
            File file = new File(screensDir, filename);
            FileOutputStream fos = new FileOutputStream(file);
            OutputStreamWriter writer = new OutputStreamWriter(fos, "UTF-8");
            writer.write(tree);
            writer.flush();
            writer.close();
            Log.d(TAG, "[getScreen] saved " + filename + " (" + tree.length() + " chars)");
        } catch (Exception ignored) {}
        return tree;
    }

    public static boolean nativeClickByText(String text) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.clickByText(text);
    }

    public static boolean nativeClickByDesc(String desc) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.clickByDesc(desc);
    }

    public static boolean nativeClickByCoords(int x, int y) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.clickByCoordinates(x, y);
    }

    public static boolean nativeLongClickByText(String text) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.longClickByText(text);
    }

    public static boolean nativeLongClickByDesc(String desc) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.longClickByDesc(desc);
    }

    public static boolean nativeLongClickByCoords(int x, int y) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.longClickByCoordinates(x, y);
    }

    public static boolean nativeScrollScreen(String direction) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.scrollScreen(direction);
    }

    public static String nativeScrollElement(String text, String direction) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return "Accessibility service not running";
        return service.scrollElementByText(text, direction);
    }

    public static boolean nativeTypeText(String text) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.inputText(text);
    }

    public static boolean nativePressHome() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.globalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_HOME);
    }

    public static boolean nativePressBack() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.globalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK);
    }

    public static boolean nativePressRecents() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.globalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_RECENTS);
    }

    public static boolean nativeShowNotifications() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return false;
        return service.globalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_NOTIFICATIONS);
    }

    public static String nativeLaunchApp(String name) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return "Accessibility service not running";
        return service.launchApp(name);
    }

    public static String nativeListApps() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return "Accessibility service not running";
        return service.listApps();
    }

    public static void nativeSleepMs(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Synchronous HTTP POST. Called from C++ host function.
     * @param urlStr  The URL to POST to
     * @param headersJson  JSON object of headers, e.g. {"Authorization":"Bearer ...","Content-Type":"application/json"}
     * @param body  The request body string
     * @return The response body string
     */
    public static String nativeHttpPost(String urlStr, String headersJson, String body) {
        HttpURLConnection conn = null;
        try {
            URL url = new URL(urlStr);
            conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(120000);

            // Parse and set headers
            JSONObject headers = new JSONObject(headersJson);
            Iterator<String> keys = headers.keys();
            while (keys.hasNext()) {
                String key = keys.next();
                conn.setRequestProperty(key, headers.getString(key));
            }

            // Write body
            byte[] bodyBytes = body.getBytes("UTF-8");
            conn.setFixedLengthStreamingMode(bodyBytes.length);
            OutputStream os = conn.getOutputStream();
            os.write(bodyBytes);
            os.flush();
            os.close();

            // Read response
            int code = conn.getResponseCode();
            InputStream is = (code >= 200 && code < 300)
                    ? conn.getInputStream()
                    : conn.getErrorStream();
            if (is == null) {
                return "{\"error\":\"HTTP " + code + " (no body)\"}";
            }
            BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            reader.close();

            if (code < 200 || code >= 300) {
                Log.e(TAG, "[httpPost] HTTP " + code + ": " + sb.toString().substring(0, Math.min(200, sb.length())));
            }
            return sb.toString();
        } catch (Exception e) {
            Log.e(TAG, "[httpPost] failed", e);
            return "{\"error\":\"" + e.getMessage().replace("\"", "\\\"") + "\"}";
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    public static void nativeUpdateStatus(String text) {
        Log.d(TAG, "[overlay] updateStatus: " + text);
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service != null) service.updateOverlayStatus(text);
        else Log.w(TAG, "[overlay] service is null!");
    }

    public static String nativeAskUser(String question) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) return "abandoned";
        boolean continued = service.askUser(question);
        return continued ? "continue" : "abandoned";
    }

    public static void nativeHideOverlay() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service != null) service.hideOverlay();
    }

    public static void nativeAppendLog(String line) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) {
            Log.d(TAG, "[log] (no service context) " + line);
            return;
        }
        try {
            Context ctx = service.getApplicationContext();
            File logFile = new File(ctx.getFilesDir(), "agent-log.txt");
            FileOutputStream fos = new FileOutputStream(logFile, true);
            OutputStreamWriter writer = new OutputStreamWriter(fos, "UTF-8");
            writer.write(line + "\n");
            writer.flush();
            writer.close();
        } catch (Exception e) {
            Log.e(TAG, "[appendLog] failed", e);
        }
    }

    // --- Public API ---

    /**
     * Run the agent with the given task and LLM config.
     * This blocks the calling thread until the agent completes.
     * Should be called from a background thread (not the main/UI thread).
     */
    public static String runAgent(String task, String configJson) {
        Log.d(TAG, "[runAgent] Starting agent with task: " + task);
        screenCounter = 0;

        try {
            nativeCreateRuntime();

            // Load agent JS from assets
            SelectToSpeakService service = SelectToSpeakService.getInstance();
            if (service == null) {
                Log.e(TAG, "[runAgent] Accessibility service not running");
                return "Error: Accessibility service not running";
            }
            String agentJs = loadAsset(service, "agent-standalone.js");
            if (agentJs == null) {
                Log.e(TAG, "[runAgent] Failed to load agent-standalone.js from assets");
                nativeDestroyRuntime();
                return "Error: Failed to load agent JS";
            }

            // Load and execute the agent JS (defines runAgent function)
            nativeEvaluateJS(agentJs, "agent-standalone.js");

            // Escape strings for JS embedding
            String escapedTask = escapeForJS(task);
            String escapedConfig = escapeForJS(configJson);

            // Call the agent's runAgent function
            String result = nativeEvaluateJS(
                "runAgent(\"" + escapedTask + "\", \"" + escapedConfig + "\")",
                "runAgent-call"
            );
            Log.d(TAG, "[runAgent] Agent completed: " + result);
            return result;
        } catch (Exception e) {
            Log.e(TAG, "[runAgent] Error", e);
            return "Error: " + e.getMessage();
        } finally {
            try {
                nativeDestroyRuntime();
            } catch (Exception e) {
                Log.e(TAG, "[runAgent] Error destroying runtime", e);
            }
            // Show completion status and auto-hide after 3 seconds
            try {
                nativeUpdateStatus("\u4efb\u52a1\u5b8c\u6210"); // "任务完成"
                Thread.sleep(3000);
                nativeHideOverlay();
            } catch (Exception e) {
                Log.e(TAG, "[runAgent] Error hiding overlay", e);
            }
        }
    }

    private static String loadAsset(Context context, String filename) {
        try {
            InputStream is = context.getAssets().open(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            StringBuilder sb = new StringBuilder();
            char[] buffer = new char[8192];
            int read;
            while ((read = reader.read(buffer)) != -1) {
                sb.append(buffer, 0, read);
            }
            reader.close();
            return sb.toString();
        } catch (Exception e) {
            Log.e(TAG, "[loadAsset] Failed to load " + filename, e);
            return null;
        }
    }

    private static String escapeForJS(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
