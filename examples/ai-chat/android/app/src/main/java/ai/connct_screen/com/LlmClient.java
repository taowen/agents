package ai.connct_screen.com;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class LlmClient {

    private static final String TAG = "A11yAgent";
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    private static final int MAX_RETRIES = 2;
    private static final long RETRY_DELAY_MS = 2000;

    private final OkHttpClient client;
    private final String apiUrl;
    private final String apiKey;
    private final String model;

    public LlmClient(String apiUrl, String apiKey, String model) {
        this.apiUrl = apiUrl;
        this.apiKey = apiKey;
        this.model = model;
        this.client = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .build();
    }

    public static class LlmResponse {
        public final String content;
        public final JSONArray toolCalls; // null if no tool calls

        public LlmResponse(String content, JSONArray toolCalls) {
            this.content = content;
            this.toolCalls = toolCalls;
        }

        public boolean hasToolCalls() {
            return toolCalls != null && toolCalls.length() > 0;
        }
    }

    private boolean isRetryable(int httpCode) {
        return httpCode == 429 || httpCode >= 500;
    }

    public LlmResponse chat(List<JSONObject> messages, JSONArray tools) throws IOException, JSONException {
        JSONObject requestBody = new JSONObject();
        requestBody.put("model", model);

        JSONArray messagesArray = new JSONArray();
        for (JSONObject msg : messages) {
            messagesArray.put(msg);
        }
        requestBody.put("messages", messagesArray);

        if (tools != null && tools.length() > 0) {
            requestBody.put("tools", tools);
        }

        String bodyStr = requestBody.toString();
        Log.d(TAG, "[LLM] Request body length: " + bodyStr.length());

        IOException lastException = null;

        for (int attempt = 0; attempt <= MAX_RETRIES; attempt++) {
            if (attempt > 0) {
                Log.d(TAG, "[LLM] Retry attempt " + attempt + "/" + MAX_RETRIES);
                try {
                    Thread.sleep(RETRY_DELAY_MS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("Interrupted during retry delay", lastException);
                }
            }

            RequestBody body = RequestBody.create(bodyStr, JSON);
            Request request = new Request.Builder()
                    .url(apiUrl)
                    .addHeader("Authorization", "Bearer " + apiKey)
                    .addHeader("Content-Type", "application/json")
                    .post(body)
                    .build();

            try (Response response = client.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    String errorBody = response.body() != null ? response.body().string() : "no body";
                    lastException = new IOException("LLM API error " + response.code() + ": " + errorBody);
                    if (isRetryable(response.code()) && attempt < MAX_RETRIES) {
                        Log.d(TAG, "[LLM] Retryable error " + response.code() + ", will retry");
                        continue;
                    }
                    throw lastException;
                }

                String responseStr = response.body().string();
                JSONObject responseJson = new JSONObject(responseStr);
                JSONObject choice = responseJson.getJSONArray("choices").getJSONObject(0);
                JSONObject message = choice.getJSONObject("message");

                String content = message.optString("content", null);
                JSONArray toolCalls = message.optJSONArray("tool_calls");

                return new LlmResponse(content, toolCalls);
            } catch (IOException e) {
                lastException = e;
                if (attempt < MAX_RETRIES) {
                    Log.d(TAG, "[LLM] IOException: " + e.getMessage() + ", will retry");
                    continue;
                }
                throw e;
            }
        }

        // Should not reach here, but just in case
        throw lastException != null ? lastException : new IOException("LLM request failed after retries");
    }
}
