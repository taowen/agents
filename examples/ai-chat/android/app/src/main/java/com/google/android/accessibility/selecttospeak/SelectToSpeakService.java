package com.google.android.accessibility.selecttospeak;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.AccessibilityServiceInfo;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import android.graphics.Rect;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SelectToSpeakService extends AccessibilityService {

    private static final String TAG = "A11yTree";
    private static final int MAX_LOG_ENTRIES = 50;

    private static final List<String> logEntries =
            Collections.synchronizedList(new ArrayList<String>());

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
        if (root == null) {
            return;
        }

        StringBuilder sb = new StringBuilder();
        sb.append("=== Accessibility Tree ===\n");
        sb.append("Event: ").append(AccessibilityEvent.eventTypeToString(event.getEventType()));
        sb.append(" pkg=").append(event.getPackageName()).append("\n");
        traverseNode(root, sb, 0);
        sb.append("=== End Tree ===");

        String treeText = sb.toString();

        // Log to logcat (primary viewing method)
        // Split into chunks since logcat has line length limits
        String[] lines = treeText.split("\n");
        for (String line : lines) {
            Log.d(TAG, line);
        }

        // Store in memory for MainActivity
        synchronized (logEntries) {
            logEntries.add(treeText);
            while (logEntries.size() > MAX_LOG_ENTRIES) {
                logEntries.remove(0);
            }
        }

        root.recycle();
    }

    private void traverseNode(AccessibilityNodeInfo node, StringBuilder sb, int depth) {
        if (node == null) {
            return;
        }

        String indent = repeat("  ", depth);
        Rect bounds = new Rect();
        node.getBoundsInScreen(bounds);

        sb.append(indent);
        sb.append("[").append(node.getClassName()).append("]");

        if (node.getText() != null) {
            sb.append(" text=\"").append(node.getText()).append("\"");
        }
        if (node.getContentDescription() != null) {
            sb.append(" desc=\"").append(node.getContentDescription()).append("\"");
        }
        if (node.getViewIdResourceName() != null) {
            sb.append(" id=").append(node.getViewIdResourceName());
        }
        sb.append(" bounds=").append(bounds.toShortString());

        if (node.isClickable()) sb.append(" clickable");
        if (node.isScrollable()) sb.append(" scrollable");
        if (node.isChecked()) sb.append(" checked");
        if (node.isEnabled()) sb.append(" enabled");

        sb.append("\n");

        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                traverseNode(child, sb, depth + 1);
                child.recycle();
            }
        }
    }

    private static String repeat(String s, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(s);
        }
        return sb.toString();
    }

    @Override
    public void onInterrupt() {
        Log.d(TAG, "Service interrupted");
    }

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        Log.d(TAG, "SelectToSpeakService connected - monitoring com.tencent.mm");
    }
}
