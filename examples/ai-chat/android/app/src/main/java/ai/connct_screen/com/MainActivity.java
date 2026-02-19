package ai.connct_screen.com;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import com.google.android.accessibility.selecttospeak.SelectToSpeakService;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;

public class MainActivity extends Activity {

    private TextView logTextView;
    private ScrollView scrollView;
    private EditText taskInput;
    private Button sendBtn;

    private String apiUrl;
    private String apiKey;
    private String model;
    private boolean llmConfigLoaded = false;

    private Handler handler;
    private Runnable logPoller;
    private volatile boolean polling = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        handler = new Handler(Looper.getMainLooper());
        loadLlmConfig();

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
        buttonRow.setPadding(0, 0, 0, 8);

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

        // Task input area
        taskInput = new EditText(this);
        taskInput.setHint("输入任务...");
        taskInput.setMinLines(2);
        taskInput.setMaxLines(4);
        taskInput.setPadding(8, 8, 8, 8);
        layout.addView(taskInput);

        // Send button
        sendBtn = new Button(this);
        sendBtn.setText("发送");
        sendBtn.setEnabled(llmConfigLoaded);
        sendBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                sendTask();
            }
        });
        layout.addView(sendBtn);

        // Hint
        TextView hint = new TextView(this);
        if (!llmConfigLoaded) {
            hint.setText("警告：llm-config.json 加载失败，发送按钮已禁用");
        } else {
            hint.setText("提示：输入任务后点发送，日志区域会实时刷新");
        }
        hint.setTextSize(12);
        hint.setPadding(0, 4, 0, 8);
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

    private void loadLlmConfig() {
        try {
            InputStream is = getAssets().open("llm-config.json");
            BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            reader.close();

            JSONObject config = new JSONObject(sb.toString());
            String baseUrl = config.optString("baseURL", "");
            if (baseUrl.endsWith("/")) {
                baseUrl = baseUrl.substring(0, baseUrl.length() - 1);
            }
            apiUrl = baseUrl + "/chat/completions";
            apiKey = config.optString("apiKey", "");
            model = config.optString("model", "gpt-4o");

            if (!baseUrl.isEmpty() && !apiKey.isEmpty()) {
                llmConfigLoaded = true;
            }
        } catch (Exception e) {
            llmConfigLoaded = false;
        }
    }

    private void sendTask() {
        String task = taskInput.getText().toString().trim();
        if (task.isEmpty()) return;

        SelectToSpeakService service = SelectToSpeakService.getInstance();
        if (service == null) {
            logTextView.setText("错误：无障碍服务未运行，请先开启");
            return;
        }

        sendBtn.setEnabled(false);
        logTextView.setText("任务已发送，等待执行...\n");

        final String finalTask = task;
        new Thread(new Runnable() {
            @Override
            public void run() {
                AgentLoop agent = new AgentLoop(apiUrl, apiKey, model, service, getApplicationContext());
                agent.execute(finalTask);
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        sendBtn.setEnabled(true);
                        stopLogPolling();
                        loadAgentLog(); // final refresh
                    }
                });
            }
        }).start();

        startLogPolling();
    }

    private void startLogPolling() {
        polling = true;
        logPoller = new Runnable() {
            @Override
            public void run() {
                if (!polling) return;
                loadAgentLog();
                handler.postDelayed(this, 2000);
            }
        };
        handler.postDelayed(logPoller, 2000);
    }

    private void stopLogPolling() {
        polling = false;
        if (logPoller != null) {
            handler.removeCallbacks(logPoller);
            logPoller = null;
        }
    }

    private void loadAgentLog() {
        File logFile = new File(getFilesDir(), "agent_log.txt");
        if (!logFile.exists()) {
            logTextView.setText("(暂无 agent 日志)");
            return;
        }
        try {
            FileInputStream fis = new FileInputStream(logFile);
            BufferedReader reader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append("\n");
            }
            reader.close();

            String content = sb.toString();
            logTextView.setText(content);
            scrollView.post(new Runnable() {
                @Override
                public void run() {
                    scrollView.fullScroll(View.FOCUS_DOWN);
                }
            });

            // Stop polling if done or error
            if (content.contains("[DONE]") || content.contains("[ERROR]")) {
                stopLogPolling();
            }
        } catch (Exception e) {
            logTextView.setText("读取日志失败: " + e.getMessage());
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadLogs();
    }

    @Override
    protected void onPause() {
        super.onPause();
        stopLogPolling();
    }

    private void loadLogs() {
        // Show agent log if it exists, otherwise show accessibility logs
        File agentLog = new File(getFilesDir(), "agent_log.txt");
        if (agentLog.exists() && agentLog.length() > 0) {
            loadAgentLog();
            return;
        }
        java.util.List<String> entries = SelectToSpeakService.getLogEntries();
        if (entries.isEmpty()) {
            logTextView.setText("(暂无日志，请开启无障碍服务后打开微信)");
        } else {
            StringBuilder sb = new StringBuilder();
            for (String entry : entries) {
                sb.append(entry).append("\n\n");
            }
            logTextView.setText(sb.toString());
            scrollView.post(new Runnable() {
                @Override
                public void run() {
                    scrollView.fullScroll(View.FOCUS_DOWN);
                }
            });
        }
    }
}
