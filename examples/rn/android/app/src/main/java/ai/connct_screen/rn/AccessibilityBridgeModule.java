package ai.connct_screen.rn;

import android.content.Context;
import android.util.Log;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;

import com.google.android.accessibility.selecttospeak.SelectToSpeakService;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Locale;

public class AccessibilityBridgeModule extends ReactContextBaseJavaModule {

    private static final String TAG = "A11yAgent";
    private static volatile String pendingTask = null;
    private int screenCounter = 0;

    AccessibilityBridgeModule(ReactApplicationContext context) {
        super(context);
    }

    @NonNull
    @Override
    public String getName() {
        return "AccessibilityBridge";
    }

    static void setPendingTask(String taskJson) {
        pendingTask = taskJson;
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public String pollPendingTask() {
        String task = pendingTask;
        pendingTask = null;
        return task != null ? task : "";
    }

    private SelectToSpeakService requireService() {
        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) {
            throw new RuntimeException("Accessibility service is not running");
        }
        return service;
    }

    // --- Sync methods (called from Hermes via eval) ---

    @ReactMethod(isBlockingSynchronousMethod = true)
    public String getScreen() {
        SelectToSpeakService service = requireService();
        String tree = service.getAccessibilityTree();
        screenCounter++;
        String filename = String.format(Locale.US, "screen_%03d.txt", screenCounter);
        try {
            Context appContext = getReactApplicationContext().getApplicationContext();
            File screensDir = new File(appContext.getFilesDir(), "screens");
            if (!screensDir.exists()) screensDir.mkdirs();
            File file = new File(screensDir, filename);
            FileOutputStream fos = new FileOutputStream(file);
            OutputStreamWriter writer = new OutputStreamWriter(fos, "UTF-8");
            writer.write(tree);
            writer.flush();
            writer.close();
        } catch (Exception ignored) {
        }
        Log.d(TAG, "[getScreen] saved " + filename + " (" + tree.length() + " chars)");
        return tree;
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean clickByText(String text) {
        return requireService().clickByText(text);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean clickByDesc(String desc) {
        return requireService().clickByDesc(desc);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean clickByCoords(double x, double y) {
        return requireService().clickByCoordinates((int) x, (int) y);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean longClickByText(String text) {
        return requireService().longClickByText(text);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean longClickByDesc(String desc) {
        return requireService().longClickByDesc(desc);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean longClickByCoords(double x, double y) {
        return requireService().longClickByCoordinates((int) x, (int) y);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean scrollScreen(String direction) {
        return requireService().scrollScreen(direction);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public String scrollElement(String text, String direction) {
        return requireService().scrollElementByText(text, direction);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean typeText(String text) {
        return requireService().inputText(text);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean pressHome() {
        return requireService().globalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_HOME);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean pressBack() {
        return requireService().globalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean pressRecents() {
        return requireService().globalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_RECENTS);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean showNotifications() {
        return requireService().globalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_NOTIFICATIONS);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public boolean sleepMs(double ms) {
        try {
            Thread.sleep((long) ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return true;
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public String launchApp(String name) {
        return requireService().launchApp(name);
    }

    @ReactMethod(isBlockingSynchronousMethod = true)
    public String listApps() {
        return requireService().listApps();
    }

    // --- Async methods ---

    @ReactMethod
    public void isServiceRunning(Promise promise) {
        promise.resolve(SelectToSpeakService.getInstance() != null);
    }

    @ReactMethod
    public void resetScreens(Promise promise) {
        try {
            Context appContext = getReactApplicationContext().getApplicationContext();
            File screensDir = new File(appContext.getFilesDir(), "screens");
            if (screensDir.exists()) {
                File[] files = screensDir.listFiles();
                if (files != null) {
                    for (File f : files) {
                        f.delete();
                    }
                }
            } else {
                screensDir.mkdirs();
            }
            screenCounter = 0;
            promise.resolve(null);
        } catch (Exception e) {
            promise.reject("RESET_ERROR", e.getMessage(), e);
        }
    }

    @ReactMethod
    public void readAssetConfig(Promise promise) {
        try {
            InputStream is = getReactApplicationContext().getAssets().open("llm-config.json");
            BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            reader.close();
            promise.resolve(sb.toString());
        } catch (Exception e) {
            promise.reject("ASSET_ERROR", "Failed to read llm-config.json: " + e.getMessage(), e);
        }
    }
}
