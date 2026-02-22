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
 * - Status: non-interactive text display (FLAG_NOT_TOUCHABLE)
 * - Ask: shows question + "Continue" button, blocks caller via CountDownLatch
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
    private Button continueButton;
    private boolean isShowing = false;
    private CountDownLatch askLatch;

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
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        container.addView(textView, textLp);

        // Continue button (hidden by default)
        continueButton = new Button(service);
        continueButton.setText("\u7ee7\u7eed"); // "继续"
        continueButton.setTextColor(Color.WHITE);
        continueButton.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);

        GradientDrawable btnBg = new GradientDrawable();
        btnBg.setColor(0xFF4CAF50);
        btnBg.setCornerRadius(dp(8));
        continueButton.setBackground(btnBg);
        continueButton.setMinimumHeight(0);
        continueButton.setMinHeight(0);
        continueButton.setPadding(dp(16), dp(6), dp(16), dp(6));

        LinearLayout.LayoutParams btnLp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        btnLp.topMargin = dp(8);
        btnLp.gravity = Gravity.CENTER_HORIZONTAL;
        continueButton.setVisibility(View.GONE);
        container.addView(continueButton, btnLp);

        continueButton.setOnClickListener(v -> {
            if (askLatch != null) {
                askLatch.countDown();
            }
        });
    }

    private WindowManager.LayoutParams makeLayoutParams(boolean touchable) {
        int flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE;
        if (!touchable) {
            flags |= WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE;
        }
        WindowManager.LayoutParams lp = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
                flags,
                PixelFormat.TRANSLUCENT);
        lp.gravity = Gravity.TOP | Gravity.CENTER_HORIZONTAL;
        lp.x = 0;
        lp.y = dp(40); // Below status bar
        // Add horizontal margins via width and padding
        lp.width = WindowManager.LayoutParams.MATCH_PARENT;
        lp.horizontalMargin = 0.03f; // 3% margin on each side
        return lp;
    }

    /**
     * Show or update the status text. Non-interactive (cannot be touched).
     * Safe to call from any thread.
     */
    public void updateStatus(String text) {
        Log.d(TAG, "updateStatus: " + text);
        mainHandler.post(() -> {
            try {
                textView.setText(text);
                continueButton.setVisibility(View.GONE);
                if (isShowing) {
                    windowManager.updateViewLayout(container, makeLayoutParams(false));
                } else {
                    windowManager.addView(container, makeLayoutParams(false));
                    isShowing = true;
                }
            } catch (Exception e) {
                Log.e(TAG, "updateStatus failed", e);
            }
        });
    }

    /**
     * Show a question with a "Continue" button. Blocks the calling thread
     * until the user taps "Continue" or timeout (5 min).
     * Must NOT be called from the main thread.
     */
    public void askUser(String question) {
        askLatch = new CountDownLatch(1);

        mainHandler.post(() -> {
            try {
                textView.setText(question);
                continueButton.setVisibility(View.VISIBLE);
                if (isShowing) {
                    windowManager.updateViewLayout(container, makeLayoutParams(true));
                } else {
                    windowManager.addView(container, makeLayoutParams(true));
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
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.w(TAG, "askUser interrupted");
        }

        // After user taps Continue, revert to status mode (non-touchable)
        mainHandler.post(() -> {
            try {
                continueButton.setVisibility(View.GONE);
                textView.setText("\u6267\u884c\u4e2d\u2026"); // "执行中…"
                if (isShowing) {
                    windowManager.updateViewLayout(container, makeLayoutParams(false));
                }
            } catch (Exception e) {
                Log.e(TAG, "askUser revert failed", e);
            }
        });
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
