package ai.connct_screen.rn;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;

/**
 * Foreground service that continuously records audio, feeds it to the Qwen ASR
 * native engine, accumulates recognized tokens, and sends completed text segments
 * to the LLM via DeviceConnection.
 *
 * Text segmentation: tokens are accumulated; when no new token arrives for
 * SILENCE_TIMEOUT_MS, the accumulated text is dispatched and the ASR is reset.
 */
public class VoiceService extends Service {

    private static final String TAG = "VoiceService";
    private static final String CHANNEL_ID = "voice_service_channel";
    private static final int NOTIFICATION_ID = 1001;

    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static final int CHUNK_SAMPLES = 1600; // 100ms at 16kHz
    private static final long SILENCE_TIMEOUT_MS = 2000;

    // Native methods (implemented in qwen_asr_jni.cpp)
    static { System.loadLibrary("qwenasr_jni"); }
    public static native boolean nativeLoadModel(String modelDir, int nThreads);
    public static native boolean nativeStartAsr();
    public static native void nativePushAudio(short[] samples, int length);
    public static native void nativeStopAsr();
    public static native void nativeResetAsr();
    public static native void nativeFreeModel();
    public static native void nativeTestWav(String wavPath);
    public static native void nativeTestWavStream(String wavPath);

    private static VoiceService sInstance;
    private AudioRecord audioRecord;
    private Thread recordingThread;
    private volatile boolean isRecording;

    // Text accumulation (accessed on main thread only)
    private final StringBuilder textBuffer = new StringBuilder();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private Runnable silenceRunnable;
    private AsrListener listener;

    public interface AsrListener {
        void onTextSegment(String text);
        void onAsrToken(String token);
    }

    public static VoiceService getInstance() {
        return sInstance;
    }

    public void setAsrListener(AsrListener l) {
        this.listener = l;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        sInstance = this;
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Notification notification = buildNotification("Voice recognition active");
        startForeground(NOTIFICATION_ID, notification);
        startRecording();
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        stopRecording();
        sInstance = null;
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    // Called from JNI on the ASR inference thread
    public static void onNativeToken(String piece) {
        VoiceService instance = sInstance;
        if (instance == null || piece == null) return;
        instance.mainHandler.post(() -> instance.handleToken(piece));
    }

    private void handleToken(String piece) {
        Log.d(TAG, "Token: " + piece);
        textBuffer.append(piece);

        if (listener != null) {
            listener.onAsrToken(piece);
        }

        // Reset silence timer
        if (silenceRunnable != null) {
            mainHandler.removeCallbacks(silenceRunnable);
        }
        silenceRunnable = () -> {
            String text = textBuffer.toString().trim();
            if (!text.isEmpty()) {
                Log.i(TAG, "Text segment: " + text);
                textBuffer.setLength(0);
                if (listener != null) {
                    listener.onTextSegment(text);
                }
                // Reset ASR for next segment
                nativeResetAsr();
            }
        };
        mainHandler.postDelayed(silenceRunnable, SILENCE_TIMEOUT_MS);
    }

    private void startRecording() {
        if (isRecording) return;

        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        bufferSize = Math.max(bufferSize, CHUNK_SAMPLES * 2); // at least one chunk

        try {
            audioRecord = new AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize);
        } catch (SecurityException e) {
            Log.e(TAG, "No RECORD_AUDIO permission", e);
            return;
        }

        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord failed to initialize");
            audioRecord.release();
            audioRecord = null;
            return;
        }

        // Start native ASR
        if (!nativeStartAsr()) {
            Log.e(TAG, "nativeStartAsr failed");
            audioRecord.release();
            audioRecord = null;
            return;
        }

        isRecording = true;
        audioRecord.startRecording();

        recordingThread = new Thread(() -> {
            short[] buffer = new short[CHUNK_SAMPLES];
            while (isRecording) {
                int read = audioRecord.read(buffer, 0, CHUNK_SAMPLES);
                if (read > 0) {
                    nativePushAudio(buffer, read);
                }
            }
        }, "AudioRecordThread");
        recordingThread.start();

        Log.i(TAG, "Recording started");
    }

    private void stopRecording() {
        isRecording = false;

        if (recordingThread != null) {
            try {
                recordingThread.join(2000);
            } catch (InterruptedException ignored) {}
            recordingThread = null;
        }

        if (audioRecord != null) {
            audioRecord.stop();
            audioRecord.release();
            audioRecord = null;
        }

        nativeStopAsr();

        // Flush any remaining text
        if (silenceRunnable != null) {
            mainHandler.removeCallbacks(silenceRunnable);
            silenceRunnable = null;
        }
        String remaining = textBuffer.toString().trim();
        if (!remaining.isEmpty() && listener != null) {
            listener.onTextSegment(remaining);
        }
        textBuffer.setLength(0);

        Log.i(TAG, "Recording stopped");
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID, "Voice Recognition",
                    NotificationManager.IMPORTANCE_LOW);
            channel.setDescription("Always-on voice recognition service");
            NotificationManager nm = getSystemService(NotificationManager.class);
            if (nm != null) nm.createNotificationChannel(channel);
        }
    }

    private Notification buildNotification(String text) {
        Notification.Builder builder;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            builder = new Notification.Builder(this, CHANNEL_ID);
        } else {
            builder = new Notification.Builder(this);
        }
        return builder
                .setContentTitle("RN Agent")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setOngoing(true)
                .build();
    }
}
