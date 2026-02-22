package com.google.android.accessibility.selecttospeak;

import android.graphics.Rect;
import android.view.accessibility.AccessibilityNodeInfo;

/**
 * Accessibility tree traversal and node search utilities.
 * Extracted from SelectToSpeakService to keep it focused.
 */
public class A11yTreeScanner {

    public static String buildTree(AccessibilityNodeInfo root) {
        if (root == null) return "(no active window)";
        StringBuilder sb = new StringBuilder();
        traverseNode(root, sb, 0);
        root.recycle();
        return sb.toString();
    }

    public static void traverseNode(AccessibilityNodeInfo node, StringBuilder sb, int depth) {
        if (node == null) return;

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

    public static AccessibilityNodeInfo findNodeByText(AccessibilityNodeInfo node, String text) {
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

    public static AccessibilityNodeInfo findNodeByDesc(AccessibilityNodeInfo node, String desc) {
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

    public static AccessibilityNodeInfo findClickableAncestor(AccessibilityNodeInfo node) {
        AccessibilityNodeInfo current = node;
        while (current != null) {
            if (current.isClickable()) return current;
            current = current.getParent();
        }
        return null;
    }

    public static AccessibilityNodeInfo findEditableNode(AccessibilityNodeInfo node) {
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

    public static AccessibilityNodeInfo findScrollableByText(AccessibilityNodeInfo node, String text) {
        AccessibilityNodeInfo textNode = findNodeByText(node, text);
        if (textNode == null) return null;
        AccessibilityNodeInfo current = textNode;
        while (current != null) {
            if (current.isScrollable()) return current;
            current = current.getParent();
        }
        return null;
    }

    private static String repeat(String s, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) sb.append(s);
        return sb.toString();
    }
}
