package com.google.android.accessibility.selecttospeak;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.graphics.Path;
import android.graphics.Rect;
import android.os.Bundle;
import android.util.Log;
import android.view.accessibility.AccessibilityNodeInfo;

/**
 * Gesture-based actions (click, long-click, scroll, text input) via AccessibilityService.
 * Extracted from SelectToSpeakService.
 */
public class GestureExecutor {

    private static final String TAG = "A11yAgent";

    private final AccessibilityService service;

    public GestureExecutor(AccessibilityService service) {
        this.service = service;
    }

    public boolean clickByCoordinates(int x, int y) {
        Path clickPath = new Path();
        clickPath.moveTo(x, y);

        GestureDescription.StrokeDescription stroke =
                new GestureDescription.StrokeDescription(clickPath, 0, 100);
        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(stroke).build();

        boolean dispatched = service.dispatchGesture(gesture, null, null);
        Log.d(TAG, "[TOOL] clickByCoordinates(" + x + "," + y + ") -> " + dispatched);
        return dispatched;
    }

    public boolean longClickByCoordinates(int x, int y) {
        Path clickPath = new Path();
        clickPath.moveTo(x, y);

        GestureDescription.StrokeDescription stroke =
                new GestureDescription.StrokeDescription(clickPath, 0, 1000);
        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(stroke).build();

        boolean dispatched = service.dispatchGesture(gesture, null, null);
        Log.d(TAG, "[TOOL] longClickByCoordinates(" + x + "," + y + ") -> " + dispatched);
        return dispatched;
    }

    public boolean clickByText(String text) {
        AccessibilityNodeInfo root = service.getRootInActiveWindow();
        if (root == null) return false;
        boolean result = clickNodeByText(root, text);
        root.recycle();
        return result;
    }

    private boolean clickNodeByText(AccessibilityNodeInfo node, String text) {
        if (node == null) return false;

        CharSequence nodeText = node.getText();
        CharSequence nodeDesc = node.getContentDescription();
        boolean matches = (nodeText != null && nodeText.toString().contains(text))
                || (nodeDesc != null && nodeDesc.toString().contains(text));

        if (matches) {
            AccessibilityNodeInfo clickable = A11yTreeScanner.findClickableAncestor(node);
            if (clickable != null) {
                boolean clicked = clickable.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                Log.d(TAG, "[TOOL] clickByText: clicked node with text containing \"" + text + "\" -> " + clicked);
                return clicked;
            }
            Rect bounds = new Rect();
            node.getBoundsInScreen(bounds);
            Log.d(TAG, "[TOOL] clickByText: no clickable ancestor, tapping coordinates (" + bounds.centerX() + "," + bounds.centerY() + ")");
            return clickByCoordinates(bounds.centerX(), bounds.centerY());
        }

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

    public boolean clickByDesc(String desc) {
        AccessibilityNodeInfo root = service.getRootInActiveWindow();
        if (root == null) return false;
        AccessibilityNodeInfo target = A11yTreeScanner.findNodeByDesc(root, desc);
        if (target != null) {
            AccessibilityNodeInfo clickable = A11yTreeScanner.findClickableAncestor(target);
            if (clickable != null) {
                boolean result = clickable.performAction(AccessibilityNodeInfo.ACTION_CLICK);
                Log.d(TAG, "[TOOL] clickByDesc: \"" + desc + "\" -> " + result);
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

    public boolean longClickByText(String text) {
        AccessibilityNodeInfo root = service.getRootInActiveWindow();
        if (root == null) return false;
        AccessibilityNodeInfo target = A11yTreeScanner.findNodeByText(root, text);
        if (target != null) {
            AccessibilityNodeInfo clickable = A11yTreeScanner.findClickableAncestor(target);
            if (clickable != null) {
                boolean result = clickable.performAction(AccessibilityNodeInfo.ACTION_LONG_CLICK);
                Log.d(TAG, "[TOOL] longClickByText: \"" + text + "\" -> " + result);
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

    public boolean longClickByDesc(String desc) {
        AccessibilityNodeInfo root = service.getRootInActiveWindow();
        if (root == null) return false;
        AccessibilityNodeInfo target = A11yTreeScanner.findNodeByDesc(root, desc);
        if (target != null) {
            AccessibilityNodeInfo clickable = A11yTreeScanner.findClickableAncestor(target);
            if (clickable != null) {
                boolean result = clickable.performAction(AccessibilityNodeInfo.ACTION_LONG_CLICK);
                Log.d(TAG, "[TOOL] longClickByDesc: \"" + desc + "\" -> " + result);
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

    public boolean scrollScreen(String direction) {
        int screenWidth = service.getResources().getDisplayMetrics().widthPixels;
        int screenHeight = service.getResources().getDisplayMetrics().heightPixels;
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
                Log.d(TAG, "[TOOL] scrollScreen: unknown direction " + direction);
                return false;
        }

        Path swipePath = new Path();
        swipePath.moveTo(startX, startY);
        swipePath.lineTo(endX, endY);

        GestureDescription.StrokeDescription stroke =
                new GestureDescription.StrokeDescription(swipePath, 0, 500);
        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(stroke).build();

        boolean dispatched = service.dispatchGesture(gesture, null, null);
        Log.d(TAG, "[TOOL] scrollScreen(" + direction + ") -> " + dispatched);
        return dispatched;
    }

    public String scrollElementByText(String text, String direction) {
        AccessibilityNodeInfo root = service.getRootInActiveWindow();
        if (root == null) return "No active window";

        AccessibilityNodeInfo target = A11yTreeScanner.findScrollableByText(root, text);
        if (target == null) {
            root.recycle();
            return "No scrollable element found for \"" + text + "\"";
        }

        int action = "up".equalsIgnoreCase(direction)
                ? AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
                : AccessibilityNodeInfo.ACTION_SCROLL_FORWARD;

        boolean result = target.performAction(action);
        Log.d(TAG, "[TOOL] scrollElementByText: \"" + text + "\" " + direction + " -> " + result);
        root.recycle();
        return result ? "Scrolled " + direction : "Scroll failed for \"" + text + "\"";
    }

    public boolean inputText(String text) {
        AccessibilityNodeInfo root = service.getRootInActiveWindow();
        if (root == null) return false;

        AccessibilityNodeInfo focused = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
        if (focused != null) {
            Bundle args = new Bundle();
            args.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
            boolean result = focused.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args);
            Log.d(TAG, "[TOOL] inputText: set text on focused node -> " + result);
            focused.recycle();
            root.recycle();
            return result;
        }

        AccessibilityNodeInfo editable = A11yTreeScanner.findEditableNode(root);
        if (editable != null) {
            Bundle args = new Bundle();
            args.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
            boolean result = editable.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args);
            Log.d(TAG, "[TOOL] inputText: set text on editable node -> " + result);
            root.recycle();
            return result;
        }

        Log.d(TAG, "[TOOL] inputText: no focused or editable node found");
        root.recycle();
        return false;
    }

    public boolean globalAction(int action) {
        boolean result = service.performGlobalAction(action);
        Log.d(TAG, "[TOOL] globalAction(" + action + ") -> " + result);
        return result;
    }
}
