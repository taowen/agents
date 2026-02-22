package com.google.android.accessibility.selecttospeak;

import android.accessibilityservice.AccessibilityService;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.hardware.HardwareBuffer;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.util.Base64;
import android.util.Log;
import android.view.Display;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;

/**
 * Accessibility service entry point. Delegates domain logic to:
 * - {@link A11yTreeScanner} for tree traversal and node search
 * - {@link GestureExecutor} for click/scroll/input gestures
 * - {@link AppLauncher} for app listing and launching
 */
public class SelectToSpeakService extends AccessibilityService {

    private static final String TAG = "A11yTree";
    private static final String AGENT_TAG = "A11yAgent";
    private static final int MAX_LOG_ENTRIES = 50;

    private static SelectToSpeakService instance;
    private GestureExecutor gestureExecutor;
    private AppLauncher appLauncher;
    private AgentOverlay agentOverlay;
    private WifiManager.WifiLock wifiLock;

    private static final List<String> logEntries =
            Collections.synchronizedList(new ArrayList<String>());

    public static SelectToSpeakService getInstance() {
        return instance;
    }

    public static List<String> getLogEntries() {
        synchronized (logEntries) {
            return new ArrayList<>(logEntries);
        }
    }

    public static void clearLogEntries() {
        logEntries.clear();
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return;

        StringBuilder sb = new StringBuilder();
        sb.append("=== Accessibility Tree ===\n");
        sb.append("Event: ").append(AccessibilityEvent.eventTypeToString(event.getEventType()));
        sb.append(" pkg=").append(event.getPackageName()).append("\n");
        A11yTreeScanner.traverseNode(root, sb, 0);
        sb.append("=== End Tree ===");

        String treeText = sb.toString();

        synchronized (logEntries) {
            logEntries.add(treeText);
            while (logEntries.size() > MAX_LOG_ENTRIES) {
                logEntries.remove(0);
            }
        }

        root.recycle();
    }

    @Override
    public void onInterrupt() {
        Log.d(TAG, "Service interrupted");
    }

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        instance = this;
        gestureExecutor = new GestureExecutor(this);
        appLauncher = new AppLauncher(this);
        agentOverlay = new AgentOverlay(this);
        // Acquire WiFi lock to keep network alive during Doze mode
        WifiManager wm = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
        wifiLock = wm.createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "RNAgent:ws");
        wifiLock.acquire();
        Log.d(TAG, "SelectToSpeakService connected, WiFi lock acquired");
        Log.d(AGENT_TAG, "SelectToSpeakService connected - agent ready");
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (wifiLock != null && wifiLock.isHeld()) {
            wifiLock.release();
            wifiLock = null;
        }
        if (agentOverlay != null) {
            agentOverlay.hide();
            agentOverlay = null;
        }
        instance = null;
        gestureExecutor = null;
        appLauncher = null;
    }

    // ========== Screenshot ==========

    public String takeScreenshotSync() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            return "ERROR: takeScreenshot requires API 30+";
        }

        final CountDownLatch latch = new CountDownLatch(1);
        final String[] resultHolder = { null };

        takeScreenshot(Display.DEFAULT_DISPLAY,
                getMainExecutor(),
                new TakeScreenshotCallback() {
                    @Override
                    public void onSuccess(ScreenshotResult screenshot) {
                        try {
                            HardwareBuffer hwBuffer = screenshot.getHardwareBuffer();
                            Bitmap hwBitmap = Bitmap.wrapHardwareBuffer(hwBuffer,
                                    screenshot.getColorSpace());
                            hwBuffer.close();
                            if (hwBitmap == null) {
                                resultHolder[0] = "ERROR: wrapHardwareBuffer returned null";
                                latch.countDown();
                                return;
                            }
                            Bitmap swBitmap = hwBitmap.copy(Bitmap.Config.ARGB_8888, false);
                            hwBitmap.recycle();

                            int origW = swBitmap.getWidth();
                            int origH = swBitmap.getHeight();
                            int targetW = 720;
                            int targetH = (int) ((long) origH * targetW / origW);
                            Bitmap scaled = Bitmap.createScaledBitmap(swBitmap, targetW, targetH, true);
                            if (scaled != swBitmap) swBitmap.recycle();

                            ByteArrayOutputStream baos = new ByteArrayOutputStream();
                            scaled.compress(Bitmap.CompressFormat.JPEG, 80, baos);
                            scaled.recycle();

                            resultHolder[0] = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP);
                            Log.d(AGENT_TAG, "[SCREENSHOT] captured, base64 length=" + resultHolder[0].length());
                        } catch (Exception e) {
                            resultHolder[0] = "ERROR: " + e.getMessage();
                        }
                        latch.countDown();
                    }

                    @Override
                    public void onFailure(int errorCode) {
                        resultHolder[0] = "ERROR: takeScreenshot failed, code=" + errorCode;
                        latch.countDown();
                    }
                });

        try {
            boolean ok = latch.await(10, java.util.concurrent.TimeUnit.SECONDS);
            if (!ok) return "ERROR: takeScreenshot timed out (10s)";
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return "ERROR: takeScreenshot interrupted";
        }

        return resultHolder[0];
    }

    // ========== Delegated methods (maintain API for HermesAgentRunner) ==========

    public String getAccessibilityTree() {
        return A11yTreeScanner.buildTree(getRootInActiveWindow());
    }

    public boolean clickByText(String text) {
        return gestureExecutor.clickByText(text);
    }

    public boolean clickByDesc(String desc) {
        return gestureExecutor.clickByDesc(desc);
    }

    public boolean clickByCoordinates(int x, int y) {
        return gestureExecutor.clickByCoordinates(x, y);
    }

    public boolean longClickByText(String text) {
        return gestureExecutor.longClickByText(text);
    }

    public boolean longClickByDesc(String desc) {
        return gestureExecutor.longClickByDesc(desc);
    }

    public boolean longClickByCoordinates(int x, int y) {
        return gestureExecutor.longClickByCoordinates(x, y);
    }

    public boolean scrollScreen(String direction) {
        return gestureExecutor.scrollScreen(direction);
    }

    public String scrollElementByText(String text, String direction) {
        return gestureExecutor.scrollElementByText(text, direction);
    }

    public boolean inputText(String text) {
        return gestureExecutor.inputText(text);
    }

    public boolean globalAction(int action) {
        return gestureExecutor.globalAction(action);
    }

    public String listApps() {
        return appLauncher.listApps();
    }

    public String launchApp(String name) {
        return appLauncher.launchApp(name);
    }

    // ========== Agent Overlay ==========

    public void updateOverlayStatus(String text) {
        if (agentOverlay != null) agentOverlay.updateStatus(text);
    }

    public boolean askUser(String question) {
        if (agentOverlay != null) return agentOverlay.askUser(question);
        return false;
    }

    public void hideOverlay() {
        if (agentOverlay != null) agentOverlay.hide();
    }
}
