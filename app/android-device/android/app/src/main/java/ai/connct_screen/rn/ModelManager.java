package ai.connct_screen.rn;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Downloads Qwen3-ASR model files from ModelScope or Hugging Face mirror.
 *
 * Downloads raw safetensors; native code auto-quantizes on first load and
 * saves a .qcache for instant subsequent loads.
 *
 * Sources:
 *   ModelScope: https://modelscope.cn/models/{repo}/resolve/master/{file}
 *   HF mirror:  https://hf-mirror.com/{repo}/resolve/main/{file}
 */
public class ModelManager {

    private static final String TAG = "ModelManager";
    private static final String PREFS_NAME = "model_manager";
    private static final String PREF_SOURCE = "download_source";

    public enum Source {
        MODELSCOPE("https://modelscope.cn/models/%s/resolve/master/%s"),
        HF_MIRROR("https://hf-mirror.com/%s/resolve/main/%s");

        final String urlPattern;
        Source(String pattern) { this.urlPattern = pattern; }

        String fileUrl(String repo, String filename) {
            return String.format(urlPattern, repo, filename);
        }
    }

    private static final String REPO = "Qwen/Qwen3-ASR-0.6B";
    private static final String MODEL_DIR_NAME = "qwen3-asr-0.6b";

    // Files required for inference (native code needs these in model_dir)
    private static final String[] MODEL_FILES = {
            "model.safetensors",
            "vocab.json",
            "merges.txt",
    };

    // Expected sizes for progress reporting
    private static final long[] FILE_SIZES = {
            1_876_091_704L, // model.safetensors
            2_776_833L,     // vocab.json
            1_671_853L,     // merges.txt
    };

    public interface DownloadListener {
        /** Called periodically during download. All values in bytes. */
        void onProgress(long downloaded, long total, String currentFile);
        /** Called when all files are downloaded. modelDir is the path to pass to native. */
        void onComplete(String modelDir);
        /** Called on any error. Partial .part files are kept for resume. */
        void onError(String message);
    }

    private final File modelDir;
    private final SharedPreferences prefs;

    public ModelManager(Context context) {
        modelDir = new File(context.getFilesDir(), MODEL_DIR_NAME);
        prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    /** Model directory path for native code. */
    public String getModelDir() {
        return modelDir.getAbsolutePath();
    }

    /** True if all required files are present and non-empty. */
    public boolean isModelReady() {
        if (!modelDir.isDirectory()) return false;
        for (String name : MODEL_FILES) {
            File f = new File(modelDir, name);
            if (!f.exists() || f.length() == 0) return false;
        }
        return true;
    }

    /** Get persisted download source preference. Default: MODELSCOPE. */
    public Source getSource() {
        String s = prefs.getString(PREF_SOURCE, Source.MODELSCOPE.name());
        try {
            return Source.valueOf(s);
        } catch (IllegalArgumentException e) {
            return Source.MODELSCOPE;
        }
    }

    /** Persist download source preference. */
    public void setSource(Source source) {
        prefs.edit().putString(PREF_SOURCE, source.name()).apply();
    }

    /**
     * Download model files. Must be called from a background thread.
     * Uses the persisted source preference.
     * Supports resume via HTTP Range on interrupted downloads.
     */
    public void download(DownloadListener listener) {
        download(getSource(), listener);
    }

    /**
     * Download model files from a specific source.
     * Must be called from a background thread.
     */
    public void download(Source source, DownloadListener listener) {
        if (!modelDir.exists() && !modelDir.mkdirs()) {
            listener.onError("Failed to create directory: " + modelDir);
            return;
        }

        long totalSize = 0;
        for (long s : FILE_SIZES) totalSize += s;
        long downloadedSoFar = 0;

        for (int i = 0; i < MODEL_FILES.length; i++) {
            String name = MODEL_FILES[i];
            File target = new File(modelDir, name);
            File partial = new File(modelDir, name + ".part");

            // Skip already completed files
            if (target.exists() && target.length() > 0) {
                downloadedSoFar += target.length();
                Log.i(TAG, "Skip (exists): " + name);
                continue;
            }

            String url = source.fileUrl(REPO, name);
            long existingBytes = partial.exists() ? partial.length() : 0;

            Log.i(TAG, "Downloading: " + url
                    + (existingBytes > 0 ? " (resume from " + existingBytes + ")" : ""));

            try {
                HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
                conn.setConnectTimeout(30_000);
                conn.setReadTimeout(60_000);
                conn.setRequestProperty("User-Agent", "QwenASR-Android/1.0");

                if (existingBytes > 0) {
                    conn.setRequestProperty("Range", "bytes=" + existingBytes + "-");
                }

                int code = conn.getResponseCode();

                // Handle redirects (ModelScope sometimes 302s)
                if (code == 301 || code == 302 || code == 307 || code == 308) {
                    String location = conn.getHeaderField("Location");
                    conn.disconnect();
                    if (location == null) {
                        listener.onError("Redirect without Location for " + name);
                        return;
                    }
                    conn = (HttpURLConnection) new URL(location).openConnection();
                    conn.setConnectTimeout(30_000);
                    conn.setReadTimeout(60_000);
                    conn.setRequestProperty("User-Agent", "QwenASR-Android/1.0");
                    if (existingBytes > 0) {
                        conn.setRequestProperty("Range", "bytes=" + existingBytes + "-");
                    }
                    code = conn.getResponseCode();
                }

                if (code != 200 && code != 206) {
                    conn.disconnect();
                    listener.onError("HTTP " + code + " for " + name + " from " + source.name());
                    return;
                }

                long contentLength = conn.getContentLengthLong();
                boolean append = (code == 206);

                try (FileOutputStream fos = new FileOutputStream(partial, append);
                     InputStream is = conn.getInputStream()) {

                    byte[] buf = new byte[65536];
                    long fileDownloaded = existingBytes;
                    int read;

                    while ((read = is.read(buf)) != -1) {
                        fos.write(buf, 0, read);
                        fileDownloaded += read;
                        listener.onProgress(downloadedSoFar + fileDownloaded, totalSize, name);
                    }

                    downloadedSoFar += fileDownloaded;
                }
                conn.disconnect();

                // Rename .part → final
                if (!partial.renameTo(target)) {
                    listener.onError("Failed to rename " + name + ".part");
                    return;
                }

                Log.i(TAG, "Done: " + name + " (" + target.length() + " bytes)");

            } catch (Exception e) {
                Log.e(TAG, "Download failed: " + name, e);
                listener.onError(name + ": " + e.getMessage());
                return;
            }
        }

        listener.onComplete(modelDir.getAbsolutePath());
    }

    /** Delete all downloaded model files (including .part and .qcache). */
    public void deleteAll() {
        if (!modelDir.exists()) return;
        File[] files = modelDir.listFiles();
        if (files != null) {
            for (File f : files) f.delete();
        }
        modelDir.delete();
        Log.i(TAG, "Deleted model directory");
    }
}
