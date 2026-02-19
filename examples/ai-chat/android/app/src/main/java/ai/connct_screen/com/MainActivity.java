package ai.connct_screen.com;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.provider.Settings;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import com.google.android.accessibility.selecttospeak.SelectToSpeakService;

import java.util.List;

public class MainActivity extends Activity {

    private TextView logTextView;
    private ScrollView scrollView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(16, 16, 16, 16);

        // Title
        TextView title = new TextView(this);
        title.setText("Accessibility Tree Logger");
        title.setTextSize(20);
        title.setPadding(0, 0, 0, 16);
        layout.addView(title);

        // Button row
        LinearLayout buttonRow = new LinearLayout(this);
        buttonRow.setOrientation(LinearLayout.HORIZONTAL);
        buttonRow.setPadding(0, 0, 0, 16);

        Button openSettingsBtn = new Button(this);
        openSettingsBtn.setText("打开无障碍设置");
        openSettingsBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS);
                startActivity(intent);
            }
        });
        buttonRow.addView(openSettingsBtn);

        Button clearBtn = new Button(this);
        clearBtn.setText("清除日志");
        clearBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                SelectToSpeakService.clearLogEntries();
                logTextView.setText("(日志已清除)");
            }
        });
        buttonRow.addView(clearBtn);

        Button refreshBtn = new Button(this);
        refreshBtn.setText("刷新");
        refreshBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                loadLogs();
            }
        });
        buttonRow.addView(refreshBtn);

        layout.addView(buttonRow);

        // Hint
        TextView hint = new TextView(this);
        hint.setText("提示：主要通过 adb logcat -s A11yTree:D 查看实时日志");
        hint.setTextSize(12);
        hint.setPadding(0, 0, 0, 8);
        layout.addView(hint);

        // ScrollView with log display
        scrollView = new ScrollView(this);
        logTextView = new TextView(this);
        logTextView.setTextSize(10);
        logTextView.setTypeface(android.graphics.Typeface.MONOSPACE);
        scrollView.addView(logTextView);

        LinearLayout.LayoutParams scrollParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT,
                1.0f);
        layout.addView(scrollView, scrollParams);

        setContentView(layout);
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadLogs();
    }

    private void loadLogs() {
        List<String> entries = SelectToSpeakService.getLogEntries();
        if (entries.isEmpty()) {
            logTextView.setText("(暂无日志，请开启无障碍服务后打开微信)");
        } else {
            StringBuilder sb = new StringBuilder();
            for (String entry : entries) {
                sb.append(entry).append("\n\n");
            }
            logTextView.setText(sb.toString());
            // Scroll to bottom
            scrollView.post(new Runnable() {
                @Override
                public void run() {
                    scrollView.fullScroll(View.FOCUS_DOWN);
                }
            });
        }
    }
}
