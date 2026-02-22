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

import org.json.JSONObject;

/**
 * Wraps standalone Hermes JS runtimes. Each runtime is identified by an
 * agent type string ("app", "browser", etc.) and gets common host functions
 * (http_post, log, sleep, update_status, ask_user, hide_overlay) registered
 * automatically. Agent-specific tools are registered by the C++ layer based
 * on the agent type.
 */
public class HermesRuntime {

    private static final String TAG = "HermesRuntime";

    // Load the native library
    static {
        System.loadLibrary("hermesruntime");
    }

    // --- Native methods (implemented in hermes_runtime.cpp) ---
    static native void nativeCreateRuntime(String agentType);
    static native String nativeEvaluateJS(String agentType, String code, String sourceURL);
    static native void nativeDestroyRuntime(String agentType);

    // --- Shared callbacks invoked from C++ via JNI ---

    public static void nativeSleepMs(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Synchronous HTTP POST. Called from C++ host function.
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

    public static String loadAsset(Context context, String filename) {
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

}
