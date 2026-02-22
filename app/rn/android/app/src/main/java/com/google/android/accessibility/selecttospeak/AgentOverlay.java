package com.google.android.accessibility.selecttospeak;

import android.accessibilityservice.AccessibilityService;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.drawable.GradientDrawable;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Manages a floating overlay window for the agent.
 *
 * Two modes:
 * - Status: non-interactive text display (FLAG_NOT_TOUCHABLE), positioned at bottom
 * - Ask: shows question + "Continue"/"Abandon" buttons, draggable, blocks caller via CountDownLatch
 *
 * Uses TYPE_ACCESSIBILITY_OVERLAY which requires no extra permissions
 * beyond the AccessibilityService itself.
 */
public class AgentOverlay {

    private static final String TAG = "AgentOverlay";
    private static final long ASK_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

    private final AccessibilityService service;
    private final WindowManager windowManager;
    private final Handler mainHandler;

    private LinearLayout container;
    private TextView textView;
    private LinearLayout buttonRow;
    private Button continueButton;
    private Button abandonButton;
    private boolean isShowing = false;
    private CountDownLatch askLatch;
    private boolean abandoned = false;

    // Drag state
    private float touchStartX, touchStartY;
    private int dragStartX, dragStartY;
    private boolean isDragging;
    private static final int DRAG_THRESHOLD_DP = 10;

    public AgentOverlay(AccessibilityService service) {
        this.service = service;
        this.windowManager = (WindowManager) service.getSystemService(AccessibilityService.WINDOW_SERVICE);
        this.mainHandler = new Handler(Looper.getMainLooper());
        buildView();
    }

    private void buildView() {
        // Container: vertical LinearLayout with rounded dark background
        container = new LinearLayout(service);
        container.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(12);
        container.setPadding(pad, dp(8), pad, dp(8));

        GradientDrawable bg = new GradientDrawable();
        bg.setColor(0xDD333333);
        bg.setCornerRadius(dp(12));
        container.setBackground(bg);

        // Text
        textView = new TextView(service);
        textView.setTextColor(Color.WHITE);
        textView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        LinearLayout.LayoutParams textLp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        container.addView(textView, textLp);

        // Button row (horizontal, hidden by default)
        buttonRow = new LinearLayout(service);
        buttonRow.setOrientation(LinearLayout.HORIZONTAL);
        buttonRow.setGravity(Gravity.CENTER_HORIZONTAL);
        LinearLayout.LayoutParams rowLp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        rowLp.topMargin = dp(8);
        rowLp.gravity = Gravity.CENTER_HORIZONTAL;
        buttonRow.setVisibility(View.GONE);
        container.addView(buttonRow, rowLp);

        // Continue button
        continueButton = new Button(service);
        continueButton.setText("\u7ee7\u7eed"); // "继续"
        continueButton.setTextColor(Color.WHITE);
        continueButton.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);

        GradientDrawable continueBg = new GradientDrawable();
        continueBg.setColor(0xFF4CAF50);
        continueBg.setCornerRadius(dp(8));
        continueButton.setBackground(continueBg);
        continueButton.setMinimumHeight(0);
        continueButton.setMinHeight(0);
        continueButton.setPadding(dp(16), dp(6), dp(16), dp(6));

        LinearLayout.LayoutParams continueLp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        buttonRow.addView(continueButton, continueLp);

        continueButton.setOnClickListener(v -> {
            abandoned = false;
            if (askLatch != null) {
                askLatch.countDown();
            }
        });

        // Abandon button
        abandonButton = new Button(service);
        abandonButton.setText("\u653e\u5f03"); // "放弃"
        abandonButton.setTextColor(Color.WHITE);
        abandonButton.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);

        GradientDrawable abandonBg = new GradientDrawable();
        abandonBg.setColor(0xFFE53935);
        abandonBg.setCornerRadius(dp(8));
        abandonButton.setBackground(abandonBg);
        abandonButton.setMinimumHeight(0);
        abandonButton.setMinHeight(0);
        abandonButton.setPadding(dp(16), dp(6), dp(16), dp(6));

        LinearLayout.LayoutParams abandonLp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        abandonLp.leftMargin = dp(12);
        buttonRow.addView(abandonButton, abandonLp);

        abandonButton.setOnClickListener(v -> {
            abandoned = true;
            if (askLatch != null) {
                askLatch.countDown();
            }
        });

        // Drag support for ask mode
        final int dragThreshold = dp(DRAG_THRESHOLD_DP);
        container.setOnTouchListener((v, event) -> {
            if (buttonRow.getVisibility() != View.VISIBLE) {
                // Status mode — not touchable anyway, but just in case
                return false;
            }
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    touchStartX = event.getRawX();
                    touchStartY = event.getRawY();
                    WindowManager.LayoutParams lp = (WindowManager.LayoutParams) container.getLayoutParams();
                    dragStartX = lp.x;
                    dragStartY = lp.y;
                    isDragging = false;
                    return true; // Consume to track move
                case MotionEvent.ACTION_MOVE:
                    float dx = event.getRawX() - touchStartX;
                    float dy = event.getRawY() - touchStartY;
                    if (!isDragging && (Math.abs(dx) > dragThreshold || Math.abs(dy) > dragThreshold)) {
                        isDragging = true;
                    }
                    if (isDragging) {
                        WindowManager.LayoutParams wlp = (WindowManager.LayoutParams) container.getLayoutParams();
                        wlp.x = dragStartX + (int) dx;
                        wlp.y = dragStartY + (int) dy;
                        windowManager.updateViewLayout(container, wlp);
                    }
                    return true;
                case MotionEvent.ACTION_UP:
                    if (!isDragging) {
                        // Not a drag — let child views handle it
                        return false;
                    }
                    return true;
            }
            return false;
        });
    }

    private WindowManager.LayoutParams makeStatusLayoutParams() {
        int flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                  | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE;
        WindowManager.LayoutParams lp = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
                flags,
                PixelFormat.TRANSLUCENT);
        lp.gravity = Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL;
        lp.x = 0;
        lp.y = dp(40);
        return lp;
    }

    private WindowManager.LayoutParams makeAskLayoutParams() {
        int flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE;
        WindowManager.LayoutParams lp = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
                flags,
                PixelFormat.TRANSLUCENT);
        lp.gravity = Gravity.TOP | Gravity.CENTER_HORIZONTAL;
        lp.x = 0;
        lp.y = dp(100);
        return lp;
    }

    /**
     * Show or update the status text. Non-interactive (cannot be touched).
     * Positioned at bottom of screen to avoid blocking content.
     * Safe to call from any thread.
     */
    public void updateStatus(String text) {
        Log.d(TAG, "updateStatus: " + text);
        mainHandler.post(() -> {
            try {
                textView.setText(text);
                buttonRow.setVisibility(View.GONE);
                if (isShowing) {
                    windowManager.updateViewLayout(container, makeStatusLayoutParams());
                } else {
                    windowManager.addView(container, makeStatusLayoutParams());
                    isShowing = true;
                }
            } catch (Exception e) {
                Log.e(TAG, "updateStatus failed", e);
            }
        });
    }

    /**
     * Show a question with "Continue" and "Abandon" buttons. Blocks the calling
     * thread until the user taps a button or timeout (5 min).
     * Must NOT be called from the main thread.
     *
     * @return true if user tapped "Continue", false if "Abandon" or timeout
     */
    public boolean askUser(String question) {
        askLatch = new CountDownLatch(1);
        abandoned = false;

        mainHandler.post(() -> {
            try {
                textView.setText(question);
                buttonRow.setVisibility(View.VISIBLE);
                if (isShowing) {
                    windowManager.updateViewLayout(container, makeAskLayoutParams());
                } else {
                    windowManager.addView(container, makeAskLayoutParams());
                    isShowing = true;
                }
            } catch (Exception e) {
                Log.e(TAG, "askUser failed", e);
                askLatch.countDown();
            }
        });

        try {
            boolean completed = askLatch.await(ASK_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            if (!completed) {
                Log.w(TAG, "askUser timed out after " + ASK_TIMEOUT_MS + "ms");
                abandoned = true;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.w(TAG, "askUser interrupted");
            abandoned = true;
        }

        // After user taps, revert to status mode (non-touchable, at bottom)
        mainHandler.post(() -> {
            try {
                buttonRow.setVisibility(View.GONE);
                textView.setText(abandoned ? "\u5df2\u653e\u5f03" : "\u6267\u884c\u4e2d\u2026"); // "已放弃" or "执行中…"
                if (isShowing) {
                    windowManager.updateViewLayout(container, makeStatusLayoutParams());
                }
            } catch (Exception e) {
                Log.e(TAG, "askUser revert failed", e);
            }
        });

        return !abandoned;
    }

    /**
     * Hide and remove the overlay. Safe to call from any thread.
     */
    public void hide() {
        mainHandler.post(() -> {
            try {
                if (isShowing) {
                    windowManager.removeView(container);
                    isShowing = false;
                }
            } catch (Exception e) {
                Log.e(TAG, "hide failed", e);
            }
        });
    }

    private int dp(int dp) {
        float density = service.getResources().getDisplayMetrics().density;
        return (int) (dp * density + 0.5f);
    }
}
