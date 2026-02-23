package ai.connct_screen.rn;

import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;
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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

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

    // Load native libraries
    static {
        System.loadLibrary("hermesruntime");
        System.loadLibrary("qwentts");
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

    // --- TTS native methods (implemented in qwen_tts_jni.c) ---
    static native boolean nativeTtsLoadModel(String modelDir);
    static native short[] nativeTtsGenerate(String tokenIds, String speaker, String language);
    static native void nativeTtsGenerateStream(String tokenIds, String speaker, String language,
                                                int chunkSize, TtsStreamCallback callback);
    static native boolean nativeTtsIsLoaded();
    static native void nativeTtsFree();
    static native int nativeTtsVerifyIncremental(String tokenIds, String speaker, String language);

    /** Callback interface for streaming TTS audio delivery. */
    public interface TtsStreamCallback {
        /** Called with each chunk of PCM int16 samples at 24kHz mono. */
        void onAudioChunk(short[] samples, int numSamples);
        /** Called when generation completes successfully. */
        void onComplete(int totalSamples, long elapsedMs);
        /** Called on error. */
        void onError(String message);
    }

    // Lazy-loaded tokenizer instance
    private static BpeTokenizer sBpeTokenizer = null;

    /**
     * speak(text, speaker, language) -> "true" or "false"
     * Called from C++ host function. Downloads model on first use, tokenizes text,
     * runs TTS inference, and plays audio via AudioTrack (24kHz mono 16-bit).
     * Blocks until audio playback completes.
     *
     * @param text     Text to speak
     * @param speaker  Optional speaker name (null for default)
     * @param language Optional language code e.g. "zh", "en" (null for auto)
     */
    public static String nativeSpeak(String text, String speaker, String language) {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) {
            Log.e(TAG, "[speak] Accessibility service not available");
            return "false";
        }
        Context ctx = service.getApplicationContext();

        try {
            // 1. Ensure model is downloaded
            TtsModelManager mgr = new TtsModelManager(ctx);
            if (!mgr.isModelReady()) {
                nativeUpdateStatus("Downloading TTS model...");
                Log.i(TAG, "[speak] Starting TTS model download");

                final CountDownLatch latch = new CountDownLatch(1);
                final AtomicBoolean success = new AtomicBoolean(false);

                mgr.download(new TtsModelManager.DownloadListener() {
                    @Override
                    public void onProgress(long downloaded, long total, String currentFile) {
                        int pct = total > 0 ? (int)(downloaded * 100 / total) : 0;
                        nativeUpdateStatus("TTS model: " + pct + "% (" + currentFile + ")");
                    }
                    @Override
                    public void onComplete(String modelDir) {
                        Log.i(TAG, "[speak] TTS model download complete");
                        success.set(true);
                        latch.countDown();
                    }
                    @Override
                    public void onError(String message) {
                        Log.e(TAG, "[speak] TTS download error: " + message);
                        latch.countDown();
                    }
                });

                latch.await();
                if (!success.get()) {
                    nativeUpdateStatus("TTS download failed");
                    return "false";
                }
            }

            String modelDir = mgr.getModelDir();

            // 2. Load model if not already loaded
            if (!nativeTtsIsLoaded()) {
                nativeUpdateStatus("Loading TTS model...");
                if (!nativeTtsLoadModel(modelDir)) {
                    Log.e(TAG, "[speak] Failed to load TTS model");
                    nativeUpdateStatus("TTS model load failed");
                    return "false";
                }
            }

            // 3. Tokenize text
            if (sBpeTokenizer == null) {
                sBpeTokenizer = BpeTokenizer.load(modelDir);
            }
            if (sBpeTokenizer == null) {
                Log.e(TAG, "[speak] Failed to load tokenizer");
                return "false";
            }
            String tokenIds = sBpeTokenizer.tokenizeForTts(text);
            Log.i(TAG, "[speak] Tokenized: " + tokenIds.length() + " chars");

            // 4. Stream generate + play via AudioTrack MODE_STREAM
            nativeUpdateStatus("Generating speech...");
            final int sampleRate = 24000;
            int bufSize = AudioTrack.getMinBufferSize(sampleRate,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT);
            bufSize = Math.max(bufSize, 8192);

            final AudioTrack track = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_ASSISTANT)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build())
                    .setAudioFormat(new AudioFormat.Builder()
                            .setSampleRate(sampleRate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .build())
                    .setBufferSizeInBytes(bufSize)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build();

            track.play();
            nativeUpdateStatus("Speaking...");

            final int[] totalSamples = {0};
            final CountDownLatch doneLatch = new CountDownLatch(1);
            final AtomicBoolean genSuccess = new AtomicBoolean(false);

            nativeTtsGenerateStream(tokenIds, speaker, language, 10, new TtsStreamCallback() {
                @Override
                public void onAudioChunk(short[] samples, int numSamples) {
                    track.write(samples, 0, numSamples);
                    totalSamples[0] += numSamples;
                }

                @Override
                public void onComplete(int total, long elapsedMs) {
                    Log.i(TAG, "[speak] Stream complete: " + total + " samples in " + elapsedMs + "ms");
                    genSuccess.set(true);
                    doneLatch.countDown();
                }

                @Override
                public void onError(String message) {
                    Log.e(TAG, "[speak] Stream error: " + message);
                    doneLatch.countDown();
                }
            });

            doneLatch.await();

            // Wait for AudioTrack to finish playing remaining buffer
            if (totalSamples[0] > 0) {
                long durationMs = (long)(totalSamples[0] * 1000.0 / sampleRate);
                Thread.sleep(Math.min(durationMs, 500) + 200);
            }

            track.stop();
            track.release();

            return genSuccess.get() ? "true" : "false";
        } catch (Exception e) {
            Log.e(TAG, "[speak] failed", e);
            return "false";
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
