package ai.connct_screen.rn;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.util.Log;

import java.io.File;

/**
 * Debug receiver for testing ASR via adb broadcast.
 *
 * Usage:
 *   # Check status
 *   adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd status"
 *
 *   # Load model (from default path or custom)
 *   adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd load_model"
 *   adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd load_model --es path /data/local/tmp/model"
 *
 *   # Test with WAV file (model must be loaded first)
 *   adb shell "am broadcast -a ai.connct_screen.rn.VOICE_DEBUG -p ai.connct_screen.rn --es cmd test_wav --es path /data/local/tmp/test.wav"
 */
public class VoiceDebugReceiver extends BroadcastReceiver {

    private static final String TAG = "VoiceDebug";

    @Override
    public void onReceive(Context context, Intent intent) {
        String cmd = intent.getStringExtra("cmd");
        if (cmd == null || cmd.isEmpty()) {
            Log.e(TAG, "No cmd provided. Use: --es cmd status|load_model|test_wav");
            return;
        }

        Log.i(TAG, "cmd=" + cmd);

        switch (cmd) {
            case "status":
                doStatus(context);
                break;
            case "load_model":
                doLoadModel(context, intent.getStringExtra("path"));
                break;
            case "test_wav":
                doTestWav(intent.getStringExtra("path"));
                break;
            case "test_wav_stream":
                doTestWavStream(intent.getStringExtra("path"));
                break;
            case "free_model":
                doFreeModel();
                break;
            default:
                Log.e(TAG, "Unknown cmd: " + cmd);
                break;
        }
    }

    private void doStatus(Context context) {
        ModelManager mm = new ModelManager(context);
        Log.i(TAG, "model_ready=" + mm.isModelReady());
        Log.i(TAG, "model_dir=" + mm.getModelDir());
        Log.i(TAG, "voice_service=" + (VoiceService.getInstance() != null));
    }

    private void doLoadModel(Context context, String path) {
        if (path == null || path.isEmpty()) {
            path = new ModelManager(context).getModelDir();
        }
        Log.i(TAG, "Loading model from: " + path);

        final String modelPath = path;
        final String cacheDir = context.getCacheDir().getAbsolutePath();
        new Thread(() -> {
            VoiceService.nativeSetCacheDir(cacheDir);
            boolean ok = VoiceService.nativeLoadModel(modelPath, 4);
            Log.i(TAG, "load_model result=" + ok);
        }).start();
    }

    private void doTestWav(String path) {
        if (path == null || path.isEmpty()) {
            Log.e(TAG, "test_wav requires --es path <wav_file>");
            return;
        }
        if (!new File(path).exists()) {
            Log.e(TAG, "WAV file not found: " + path);
            return;
        }
        Log.i(TAG, "Testing WAV: " + path);

        new Thread(() -> {
            try {
                VoiceService.nativeTestWav(path);
                Log.i(TAG, "test_wav completed");
            } catch (Exception e) {
                Log.e(TAG, "test_wav failed", e);
            }
        }).start();
    }

    private void doTestWavStream(String path) {
        if (path == null || path.isEmpty()) {
            Log.e(TAG, "test_wav_stream requires --es path <wav_file>");
            return;
        }
        if (!new File(path).exists()) {
            Log.e(TAG, "WAV file not found: " + path);
            return;
        }
        Log.i(TAG, "Testing WAV (stream): " + path);

        new Thread(() -> {
            try {
                VoiceService.nativeTestWavStream(path);
                Log.i(TAG, "test_wav_stream completed");
            } catch (Exception e) {
                Log.e(TAG, "test_wav_stream failed", e);
            }
        }).start();
    }

    private void doFreeModel() {
        Log.i(TAG, "Freeing model...");
        VoiceService.nativeFreeModel();
        Log.i(TAG, "free_model completed");
    }
}
