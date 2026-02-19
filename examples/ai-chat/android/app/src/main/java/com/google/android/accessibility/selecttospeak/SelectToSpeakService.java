package com.google.android.accessibility.selecttospeak;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.graphics.Path;
import android.graphics.Rect;
import android.os.Bundle;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SelectToSpeakService extends AccessibilityService {

    private static final String TAG = "A11yTree";
    private static final String AGENT_TAG = "A11yAgent";
    private static final int MAX_LOG_ENTRIES = 50;

    private static SelectToSpeakService instance;

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
        instance = this;
        Log.d(TAG, "SelectToSpeakService connected");
        Log.d(AGENT_TAG, "SelectToSpeakService connected - agent ready");
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        instance = null;
    }

    // ========== Agent action methods ==========

    /**
     * Get current accessibility tree as a string.
     */
    public String getAccessibilityTree() {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) {
            return "(no active window)";
        }
        StringBuilder sb = new StringBuilder();
        traverseNode(root, sb, 0);
        root.recycle();
        return sb.toString();
    }

    /**
     * Click a node matching the given text.
     * Returns true if a matching clickable node was found and clicked.
     */
    public boolean clickByText(String text) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return false;
        boolean result = clickNodeByText(root, text);
        root.recycle();
        return result;
    }

    private boolean clickNodeByText(AccessibilityNodeInfo node, String text) {
        if (node == null) return false;

        // Check this node
        CharSequence nodeText = node.getText();
        CharSequence nodeDesc = node.getContentDescription();
        boolean matches = (nodeText != null && nodeText.toString().contains(text))
                || (nodeDesc != null && nodeDesc.toString().contains(text));

        if (matches) {
            // Find the nearest clickable ancestor or self
            AccessibilityNodeInfo clickable = findClickableAncestor(node);
            if (clickable != null) {
                boolean clicked = clickable.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                Log.d(AGENT_TAG, "[TOOL] clickByText: clicked node with text containing \"" + text + "\" -> " + clicked);
                return clicked;
            }
            // If no clickable ancestor, try clicking by coordinates
            Rect bounds = new Rect();
            node.getBoundsInScreen(bounds);
            int x = bounds.centerX();
            int y = bounds.centerY();
            Log.d(AGENT_TAG, "[TOOL] clickByText: no clickable ancestor, tapping coordinates (" + x + "," + y + ")");
            return clickByCoordinates(x, y);
        }

        // Recurse children
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                if (clickNodeByText(child, text)) {
                    child.recycle();
                    return true;
                }
                child.recycle();
            }
        }
        return false;
    }

    private AccessibilityNodeInfo findClickableAncestor(AccessibilityNodeInfo node) {
        AccessibilityNodeInfo current = node;
        while (current != null) {
            if (current.isClickable()) {
                return current;
            }
            AccessibilityNodeInfo parent = current.getParent();
            if (current != node) {
                // Don't recycle the original node
            }
            current = parent;
        }
        return null;
    }

    /**
     * Click a node matching the given contentDescription only (not text).
     */
    public boolean clickByDesc(String desc) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return false;
        AccessibilityNodeInfo target = findNodeByDesc(root, desc);
        if (target != null) {
            AccessibilityNodeInfo clickable = findClickableAncestor(target);
            if (clickable != null) {
                boolean result = clickable.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                Log.d(AGENT_TAG, "[TOOL] clickByDesc: \"" + desc + "\" -> " + result);
                root.recycle();
                return result;
            }
            Rect bounds = new Rect();
            target.getBoundsInScreen(bounds);
            root.recycle();
            return clickByCoordinates(bounds.centerX(), bounds.centerY());
        }
        root.recycle();
        return false;
    }

    /**
     * Long click a node matching the given contentDescription only (not text).
     */
    public boolean longClickByDesc(String desc) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return false;
        AccessibilityNodeInfo target = findNodeByDesc(root, desc);
        if (target != null) {
            AccessibilityNodeInfo clickable = findClickableAncestor(target);
            if (clickable != null) {
                boolean result = clickable.performAction(AccessibilityNodeInfo.ACTION_LONG_CLICK);
                Log.d(AGENT_TAG, "[TOOL] longClickByDesc: \"" + desc + "\" -> " + result);
                root.recycle();
                return result;
            }
            Rect bounds = new Rect();
            target.getBoundsInScreen(bounds);
            root.recycle();
            return longClickByCoordinates(bounds.centerX(), bounds.centerY());
        }
        root.recycle();
        return false;
    }

    private AccessibilityNodeInfo findNodeByDesc(AccessibilityNodeInfo node, String desc) {
        if (node == null) return null;
        CharSequence nodeDesc = node.getContentDescription();
        if (nodeDesc != null && nodeDesc.toString().contains(desc)) {
            return node;
        }
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                AccessibilityNodeInfo found = findNodeByDesc(child, desc);
                if (found != null) return found;
                child.recycle();
            }
        }
        return null;
    }

    /**
     * Click at specific screen coordinates using a gesture.
     */
    public boolean clickByCoordinates(int x, int y) {
        Path clickPath = new Path();
        clickPath.moveTo(x, y);

        GestureDescription.StrokeDescription stroke =
                new GestureDescription.StrokeDescription(clickPath, 0, 100);
        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(stroke)
                .build();

        boolean dispatched = dispatchGesture(gesture, null, null);
        Log.d(AGENT_TAG, "[TOOL] clickByCoordinates(" + x + "," + y + ") -> " + dispatched);
        return dispatched;
    }

    /**
     * Long click at specific screen coordinates using a gesture (hold 1 second).
     */
    public boolean longClickByCoordinates(int x, int y) {
        Path clickPath = new Path();
        clickPath.moveTo(x, y);

        GestureDescription.StrokeDescription stroke =
                new GestureDescription.StrokeDescription(clickPath, 0, 1000);
        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(stroke)
                .build();

        boolean dispatched = dispatchGesture(gesture, null, null);
        Log.d(AGENT_TAG, "[TOOL] longClickByCoordinates(" + x + "," + y + ") -> " + dispatched);
        return dispatched;
    }

    /**
     * Long click a node matching the given text.
     */
    public boolean longClickByText(String text) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return false;
        AccessibilityNodeInfo target = findNodeByText(root, text);
        if (target != null) {
            // Try ACTION_LONG_CLICK on clickable ancestor
            AccessibilityNodeInfo clickable = findClickableAncestor(target);
            if (clickable != null) {
                boolean result = clickable.performAction(AccessibilityNodeInfo.ACTION_LONG_CLICK);
                Log.d(AGENT_TAG, "[TOOL] longClickByText: \"" + text + "\" -> " + result);
                root.recycle();
                return result;
            }
            // Fallback to gesture
            Rect bounds = new Rect();
            target.getBoundsInScreen(bounds);
            root.recycle();
            return longClickByCoordinates(bounds.centerX(), bounds.centerY());
        }
        root.recycle();
        return false;
    }

    private AccessibilityNodeInfo findNodeByText(AccessibilityNodeInfo node, String text) {
        if (node == null) return null;
        CharSequence nodeText = node.getText();
        CharSequence nodeDesc = node.getContentDescription();
        if ((nodeText != null && nodeText.toString().contains(text))
                || (nodeDesc != null && nodeDesc.toString().contains(text))) {
            return node;
        }
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                AccessibilityNodeInfo found = findNodeByText(child, text);
                if (found != null) return found;
                child.recycle();
            }
        }
        return null;
    }

    /**
     * Scroll the screen in a direction: up, down, left, right.
     */
    public boolean scrollScreen(String direction) {
        // Get screen dimensions approximately
        int screenWidth = getResources().getDisplayMetrics().widthPixels;
        int screenHeight = getResources().getDisplayMetrics().heightPixels;
        int centerX = screenWidth / 2;
        int centerY = screenHeight / 2;

        int startX, startY, endX, endY;
        switch (direction.toLowerCase()) {
            case "up":
                startX = centerX; startY = screenHeight * 3 / 4;
                endX = centerX; endY = screenHeight / 4;
                break;
            case "down":
                startX = centerX; startY = screenHeight / 4;
                endX = centerX; endY = screenHeight * 3 / 4;
                break;
            case "left":
                startX = screenWidth * 3 / 4; startY = centerY;
                endX = screenWidth / 4; endY = centerY;
                break;
            case "right":
                startX = screenWidth / 4; startY = centerY;
                endX = screenWidth * 3 / 4; endY = centerY;
                break;
            default:
                Log.d(AGENT_TAG, "[TOOL] scrollScreen: unknown direction " + direction);
                return false;
        }

        Path swipePath = new Path();
        swipePath.moveTo(startX, startY);
        swipePath.lineTo(endX, endY);

        GestureDescription.StrokeDescription stroke =
                new GestureDescription.StrokeDescription(swipePath, 0, 500);
        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(stroke)
                .build();

        boolean dispatched = dispatchGesture(gesture, null, null);
        Log.d(AGENT_TAG, "[TOOL] scrollScreen(" + direction + ") -> " + dispatched);
        return dispatched;
    }

    /**
     * Input text into the currently focused field.
     */
    public boolean inputText(String text) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return false;

        AccessibilityNodeInfo focused = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
        if (focused != null) {
            Bundle args = new Bundle();
            args.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
            boolean result = focused.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args);
            Log.d(AGENT_TAG, "[TOOL] inputText: set text on focused node -> " + result);
            focused.recycle();
            root.recycle();
            return result;
        }

        // Fallback: find first editable node
        AccessibilityNodeInfo editable = findEditableNode(root);
        if (editable != null) {
            Bundle args = new Bundle();
            args.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
            boolean result = editable.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args);
            Log.d(AGENT_TAG, "[TOOL] inputText: set text on editable node -> " + result);
            root.recycle();
            return result;
        }

        Log.d(AGENT_TAG, "[TOOL] inputText: no focused or editable node found");
        root.recycle();
        return false;
    }

    private AccessibilityNodeInfo findEditableNode(AccessibilityNodeInfo node) {
        if (node == null) return null;
        if (node.isEditable()) return node;
        for (int i = 0; i < node.getChildCount(); i++) {
            AccessibilityNodeInfo child = node.getChild(i);
            if (child != null) {
                AccessibilityNodeInfo found = findEditableNode(child);
                if (found != null) return found;
                child.recycle();
            }
        }
        return null;
    }

    /**
     * Perform a global action (back, home, etc.)
     */
    public boolean globalAction(int action) {
        boolean result = performGlobalAction(action);
        Log.d(AGENT_TAG, "[TOOL] globalAction(" + action + ") -> " + result);
        return result;
    }

    /**
     * List all launchable apps. Returns one per line: "AppName (package.name)"
     */
    public String listApps() {
        PackageManager pm = getPackageManager();
        List<ApplicationInfo> apps = pm.getInstalledApplications(0);
        StringBuilder sb = new StringBuilder();
        for (ApplicationInfo app : apps) {
            Intent launchIntent = pm.getLaunchIntentForPackage(app.packageName);
            if (launchIntent != null) {
                String label = pm.getApplicationLabel(app).toString();
                sb.append(label).append(" (").append(app.packageName).append(")\n");
            }
        }
        return sb.toString().trim();
    }

    /**
     * Launch an app by name or package name.
     * Returns a description of what happened.
     */
    public String launchApp(String name) {
        PackageManager pm = getPackageManager();

        // Try as package name first
        Intent intent = pm.getLaunchIntentForPackage(name);
        if (intent != null) {
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
            Log.d(AGENT_TAG, "[TOOL] launchApp: launched by package " + name);
            return "Launched " + name;
        }

        // Fuzzy match by app label
        List<ApplicationInfo> apps = pm.getInstalledApplications(0);
        for (ApplicationInfo app : apps) {
            String label = pm.getApplicationLabel(app).toString();
            if (label.contains(name) || name.contains(label)) {
                Intent launchIntent = pm.getLaunchIntentForPackage(app.packageName);
                if (launchIntent != null) {
                    launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                    startActivity(launchIntent);
                    Log.d(AGENT_TAG, "[TOOL] launchApp: launched " + label + " (" + app.packageName + ")");
                    return "Launched " + label + " (" + app.packageName + ")";
                }
            }
        }

        Log.d(AGENT_TAG, "[TOOL] launchApp: no app found for \"" + name + "\"");
        return "App not found: " + name;
    }

    /**
     * Scroll a specific element found by text/desc.
     * direction: "up" = SCROLL_BACKWARD, "down" = SCROLL_FORWARD
     */
    public String scrollElementByText(String text, String direction) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return "No active window";

        AccessibilityNodeInfo target = findScrollableByText(root, text);
        if (target == null) {
            root.recycle();
            return "No scrollable element found for \"" + text + "\"";
        }

        int action;
        if ("up".equalsIgnoreCase(direction)) {
            action = AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD;
        } else {
            action = AccessibilityNodeInfo.ACTION_SCROLL_FORWARD;
        }

        boolean result = target.performAction(action);
        Log.d(AGENT_TAG, "[TOOL] scrollElementByText: \"" + text + "\" " + direction + " -> " + result);
        root.recycle();
        return result ? "Scrolled " + direction : "Scroll failed for \"" + text + "\"";
    }

    /**
     * Find a scrollable node matching text/desc, or its nearest scrollable ancestor.
     */
    private AccessibilityNodeInfo findScrollableByText(AccessibilityNodeInfo node, String text) {
        AccessibilityNodeInfo textNode = findNodeByText(node, text);
        if (textNode == null) return null;

        // Walk up to find the nearest scrollable ancestor (including self)
        AccessibilityNodeInfo current = textNode;
        while (current != null) {
            if (current.isScrollable()) {
                return current;
            }
            current = current.getParent();
        }
        return null;
    }

}
