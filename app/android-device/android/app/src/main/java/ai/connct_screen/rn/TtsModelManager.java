package ai.connct_screen.rn;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Downloads Qwen3-TTS-0.6B CustomVoice model files from R2 public storage.
 * Supports HTTP Range for resuming interrupted downloads.
 * Model files are stored in context.getFilesDir()/qwen-tts-model/.
 */
public class TtsModelManager {

    private static final String TAG = "TtsModelManager";
    private static final String MODEL_DIR_NAME = "qwen-tts-model";
    private static final String BASE_URL =
            "https://pub-f464632870e64014a498bb2860410020.r2.dev/qwen3-tts-0.6b/";

    // Files required for inference (relative paths within model dir)
    private static final String[] MODEL_FILES = {
            "config.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
    };

    // Expected sizes for progress calculation (actual measured sizes)
    private static final long[] FILE_SIZES = {
            50_000L,            // config.json ~50KB
            2_800_000L,         // vocab.json ~2.8MB
            5_400_000L,         // merges.txt ~5.4MB
            1_811_626_576L,     // model.safetensors ~1.8GB
            50_000L,            // speech_tokenizer/config.json ~50KB
            682_293_092L,       // speech_tokenizer/model.safetensors ~682MB
    };

    public interface DownloadListener {
        void onProgress(long downloaded, long total, String currentFile);
        void onComplete(String modelDir);
        void onError(String message);
    }

    private final File modelDir;

    public TtsModelManager(Context context) {
        modelDir = new File(context.getFilesDir(), MODEL_DIR_NAME);
    }

    /** Returns the model directory path (for passing to native code). */
    public String getModelDir() {
        return modelDir.getAbsolutePath();
    }

    /** Check if all required model files are present. */
    public boolean isModelReady() {
        if (!modelDir.isDirectory()) return false;
        for (String name : MODEL_FILES) {
            File f = new File(modelDir, name);
            if (!f.exists() || f.length() == 0) return false;
        }
        return true;
    }

    /** Download model files. Must be called from a background thread. */
    public void download(DownloadListener listener) {
        if (!modelDir.exists() && !modelDir.mkdirs()) {
            listener.onError("Failed to create TTS model directory");
            return;
        }

        // Create subdirectory for speech_tokenizer
        File speechDir = new File(modelDir, "speech_tokenizer");
        if (!speechDir.exists() && !speechDir.mkdirs()) {
            listener.onError("Failed to create speech_tokenizer directory");
            return;
        }

        long totalSize = 0;
        for (long s : FILE_SIZES) totalSize += s;
        long downloadedSoFar = 0;

        for (int i = 0; i < MODEL_FILES.length; i++) {
            String name = MODEL_FILES[i];
            File target = new File(modelDir, name);
            File partial = new File(modelDir, name + ".part");

            // Ensure parent directory exists (for nested paths)
            File parentDir = partial.getParentFile();
            if (parentDir != null && !parentDir.exists()) {
                parentDir.mkdirs();
            }

            // Skip already downloaded files
            if (target.exists() && target.length() > 0) {
                downloadedSoFar += target.length();
                continue;
            }

            String url = BASE_URL + name;
            long existingBytes = partial.exists() ? partial.length() : 0;

            try {
                HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
                conn.setConnectTimeout(30_000);
                conn.setReadTimeout(60_000);

                if (existingBytes > 0) {
                    conn.setRequestProperty("Range", "bytes=" + existingBytes + "-");
                }

                int code = conn.getResponseCode();
                if (code != 200 && code != 206) {
                    conn.disconnect();
                    listener.onError("HTTP " + code + " downloading " + name);
                    return;
                }

                long contentLength = conn.getContentLength();
                long fileTotal = (code == 206) ? existingBytes + contentLength : contentLength;

                boolean append = (code == 206);
                FileOutputStream fos = new FileOutputStream(partial, append);
                InputStream is = conn.getInputStream();

                byte[] buf = new byte[65536];
                long fileDownloaded = existingBytes;
                int read;

                while ((read = is.read(buf)) != -1) {
                    fos.write(buf, 0, read);
                    fileDownloaded += read;
                    long totalProgress = downloadedSoFar + fileDownloaded;
                    listener.onProgress(totalProgress, totalSize, name);
                }

                fos.close();
                is.close();
                conn.disconnect();

                // Rename .part to final name
                if (!partial.renameTo(target)) {
                    listener.onError("Failed to rename " + name + ".part");
                    return;
                }

                downloadedSoFar += fileDownloaded;
                Log.i(TAG, "Downloaded: " + name + " (" + fileDownloaded + " bytes)");

            } catch (Exception e) {
                Log.e(TAG, "Download failed: " + name, e);
                listener.onError("Download failed: " + name + " - " + e.getMessage());
                return;
            }
        }

        listener.onComplete(modelDir.getAbsolutePath());
    }
}
