package ai.connct_screen.rn;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;
import android.util.Log;

/**
 * Debug receiver for testing TTS via adb broadcast.
 *
 * Usage:
 *   # Check status (model downloaded? loaded? tokenizer ready?)
 *   adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd status"
 *
 *   # Download model
 *   adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd download"
 *
 *   # Load model into memory (default path or custom)
 *   adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd load"
 *   adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd load --es path /data/local/tmp/qwen-tts-model"
 *
 *   # Tokenize only (print token IDs to logcat)
 *   adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd tokenize --es text '你好世界'"
 *   adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd tokenize --es text '你好世界' --es path /data/local/tmp/qwen-tts-model"
 *
 *   # Full TTS: tokenize + generate + play
 *   adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd speak --es text 'Hello world'"
 *
 *   # Speak with speaker/language and custom model path
 *   adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd speak --es text '你好' --es speaker serena --es language chinese"
 *   adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd speak --es text 'Hello' --es path /data/local/tmp/qwen-tts-model"
 *
 *   # Free model memory
 *   adb shell "am broadcast -a ai.connct_screen.rn.TTS_DEBUG -p ai.connct_screen.rn --es cmd free"
 *
 * Logcat filter:
 *   adb logcat -s TtsDebug:V QwenTTS_JNI:V BpeTokenizer:V
 */
public class TtsDebugReceiver extends BroadcastReceiver {

    private static final String TAG = "TtsDebug";

    private static BpeTokenizer sTokenizer = null;

    @Override
    public void onReceive(Context context, Intent intent) {
        String cmd = intent.getStringExtra("cmd");
        if (cmd == null || cmd.isEmpty()) {
            Log.e(TAG, "No cmd provided. Use: --es cmd status|download|load|tokenize|speak|free");
            return;
        }

        String path = intent.getStringExtra("path");
        Log.i(TAG, "cmd=" + cmd + (path != null ? " path=" + path : ""));

        switch (cmd) {
            case "status":
                doStatus(context);
                break;
            case "download":
                doDownload(context);
                break;
            case "load":
                doLoad(context, path);
                break;
            case "tokenize":
                doTokenize(context, intent.getStringExtra("text"), path);
                break;
            case "speak":
                doSpeak(context, intent.getStringExtra("text"),
                        intent.getStringExtra("speaker"),
                        intent.getStringExtra("language"), path);
                break;
            case "free":
                doFree();
                break;
            default:
                Log.e(TAG, "Unknown cmd: " + cmd);
                break;
        }
    }

    private void doStatus(Context context) {
        TtsModelManager mgr = new TtsModelManager(context);
        boolean downloaded = mgr.isModelReady();
        boolean loaded = HermesRuntime.nativeTtsIsLoaded();
        boolean tokenizerReady = sTokenizer != null;
        Log.i(TAG, "model_downloaded=" + downloaded);
        Log.i(TAG, "model_loaded=" + loaded);
        Log.i(TAG, "tokenizer_ready=" + tokenizerReady);
        Log.i(TAG, "model_dir=" + mgr.getModelDir());
    }

    private void doDownload(Context context) {
        TtsModelManager mgr = new TtsModelManager(context);
        if (mgr.isModelReady()) {
            Log.i(TAG, "Model already downloaded at " + mgr.getModelDir());
            return;
        }
        Log.i(TAG, "Starting download...");
        long startMs = System.currentTimeMillis();

        new Thread(() -> {
            mgr.download(new TtsModelManager.DownloadListener() {
                @Override
                public void onProgress(long downloaded, long total, String currentFile) {
                    int pct = total > 0 ? (int)(downloaded * 100 / total) : 0;
                    Log.i(TAG, "download progress: " + pct + "% (" + currentFile + ") "
                            + (downloaded / 1024 / 1024) + "MB/" + (total / 1024 / 1024) + "MB");
                }

                @Override
                public void onComplete(String modelDir) {
                    long elapsed = System.currentTimeMillis() - startMs;
                    Log.i(TAG, "download complete in " + elapsed + "ms, dir=" + modelDir);
                }

                @Override
                public void onError(String message) {
                    Log.e(TAG, "download failed: " + message);
                }
            });
        }).start();
    }

    private void doLoad(Context context, String path) {
        if (HermesRuntime.nativeTtsIsLoaded()) {
            Log.i(TAG, "Model already loaded");
            return;
        }
        String modelDir = resolveModelDir(context, path);
        if (modelDir == null) return;
        Log.i(TAG, "Loading model from " + modelDir + "...");

        new Thread(() -> {
            long startMs = System.currentTimeMillis();
            boolean ok = HermesRuntime.nativeTtsLoadModel(modelDir);
            long elapsed = System.currentTimeMillis() - startMs;
            Log.i(TAG, "load_model result=" + ok + " in " + elapsed + "ms");
        }).start();
    }

    private void doTokenize(Context context, String text, String path) {
        if (text == null || text.isEmpty()) {
            Log.e(TAG, "tokenize requires --es text <text>");
            return;
        }
        String modelDir = resolveModelDir(context, path);
        if (modelDir == null) return;
        ensureTokenizer(modelDir);
        if (sTokenizer == null) {
            Log.e(TAG, "Failed to load tokenizer");
            return;
        }

        long startMs = System.currentTimeMillis();
        String tokenIds = sTokenizer.tokenizeForTts(text);
        long elapsed = System.currentTimeMillis() - startMs;

        // Count tokens
        int tokenCount = tokenIds.isEmpty() ? 0 : tokenIds.split(",").length;
        Log.i(TAG, "tokenize text=\"" + text + "\"");
        Log.i(TAG, "tokenize count=" + tokenCount + " in " + elapsed + "ms");
        Log.i(TAG, "tokenize ids=" + tokenIds);
    }

    private void doSpeak(Context context, String text, String speaker, String language, String path) {
        if (text == null || text.isEmpty()) {
            Log.e(TAG, "speak requires --es text <text>");
            return;
        }
        String modelDir = resolveModelDir(context, path);
        if (modelDir == null) return;

        new Thread(() -> {
            try {
                // Load model if needed
                if (!HermesRuntime.nativeTtsIsLoaded()) {
                    Log.i(TAG, "speak: loading model...");
                    long loadStart = System.currentTimeMillis();
                    boolean ok = HermesRuntime.nativeTtsLoadModel(modelDir);
                    Log.i(TAG, "speak: load_model=" + ok + " in "
                            + (System.currentTimeMillis() - loadStart) + "ms");
                    if (!ok) {
                        Log.e(TAG, "speak: model load failed");
                        return;
                    }
                }

                // Tokenize
                ensureTokenizer(modelDir);
                if (sTokenizer == null) {
                    Log.e(TAG, "speak: tokenizer load failed");
                    return;
                }
                long tokStart = System.currentTimeMillis();
                String tokenIds = sTokenizer.tokenizeForTts(text);
                int tokenCount = tokenIds.isEmpty() ? 0 : tokenIds.split(",").length;
                Log.i(TAG, "speak: tokenized " + tokenCount + " tokens in "
                        + (System.currentTimeMillis() - tokStart) + "ms");

                // Generate PCM
                Log.i(TAG, "speak: generating audio (speaker=" + speaker
                        + ", language=" + language + ")...");
                long genStart = System.currentTimeMillis();
                short[] pcm = HermesRuntime.nativeTtsGenerate(tokenIds, speaker, language);
                long genElapsed = System.currentTimeMillis() - genStart;
                if (pcm == null || pcm.length == 0) {
                    Log.e(TAG, "speak: generate returned no audio");
                    return;
                }
                double durationSec = pcm.length / 24000.0;
                Log.i(TAG, "speak: generated " + pcm.length + " samples ("
                        + String.format("%.2f", durationSec) + "s) in " + genElapsed + "ms");

                // Play via AudioTrack (24kHz mono 16-bit)
                Log.i(TAG, "speak: playing audio...");
                int sampleRate = 24000;
                int bufSize = AudioTrack.getMinBufferSize(sampleRate,
                        AudioFormat.CHANNEL_OUT_MONO,
                        AudioFormat.ENCODING_PCM_16BIT);
                bufSize = Math.max(bufSize, pcm.length * 2);

                AudioTrack track = new AudioTrack.Builder()
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
                        .setTransferMode(AudioTrack.MODE_STATIC)
                        .build();

                track.write(pcm, 0, pcm.length);
                track.play();

                long durationMs = (long)(durationSec * 1000) + 200;
                Thread.sleep(durationMs);

                track.stop();
                track.release();
                Log.i(TAG, "speak: playback complete");

            } catch (Exception e) {
                Log.e(TAG, "speak: failed", e);
            }
        }).start();
    }

    private void doFree() {
        Log.i(TAG, "Freeing TTS model...");
        HermesRuntime.nativeTtsFree();
        sTokenizer = null;
        Log.i(TAG, "Model freed, tokenizer cleared");
    }

    /**
     * Resolve model directory: use explicit path if provided, otherwise fall back to
     * TtsModelManager default. Returns null and logs error if model is not available.
     */
    private String resolveModelDir(Context context, String path) {
        if (path != null && !path.isEmpty()) {
            Log.i(TAG, "Using custom model path: " + path);
            return path;
        }
        TtsModelManager mgr = new TtsModelManager(context);
        if (!mgr.isModelReady()) {
            Log.e(TAG, "Model not found. Use --es path or run 'download' first.");
            return null;
        }
        return mgr.getModelDir();
    }

    private static synchronized void ensureTokenizer(String modelDir) {
        if (sTokenizer != null) return;
        long startMs = System.currentTimeMillis();
        sTokenizer = BpeTokenizer.load(modelDir);
        long elapsed = System.currentTimeMillis() - startMs;
        if (sTokenizer != null) {
            Log.i(TAG, "Tokenizer loaded in " + elapsed + "ms");
        }
    }
}
