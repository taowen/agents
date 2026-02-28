import { generateText, type UIMessage } from "ai";

const TOPIC_DETECTION_PROMPT = `Decide: does the recent conversation provide useful context for answering the new message?

- N = The new message builds on, refers to, or needs context from the recent conversation to be answered properly.
- Y = The recent conversation provides NO useful context for the new message. Answering it requires completely different knowledge.

Examples:
- History: 南昌今天天气 → New: 昨天发布了什么 → Y (weather context doesn't help answer a release question)
- History: code refactoring → New: 南昌今天天气 → Y (code context doesn't help answer a weather question)
- History: code refactoring → New: 把那个文件名也改一下 → N (refers back to the code being discussed)
- History: fixing a bug → New: 再跑一下测试 → N (testing the same bug fix)
- History: project A → New: 帮我写个完全不同的脚本 → Y (unrelated script, no context needed)

Output ONLY Y or N, nothing else.`;

function extractText(msg: UIMessage): string {
  return msg.parts
    .filter((p): p is { type: "text"; text: string } => p.type === "text")
    .map((p) => p.text)
    .join("\n");
}

export interface TopicDetectionResult {
  isNewTopic: boolean;
  usage: { inputTokens: number; outputTokens: number } | null;
}

export async function detectTopicChange(
  model: Parameters<typeof generateText>[0]["model"],
  messages: UIMessage[],
  compactBoundaryId: string | null
): Promise<TopicDetectionResult> {
  try {
    let boundaryIndex = 0;
    if (compactBoundaryId) {
      const idx = messages.findIndex((m) => m.id === compactBoundaryId);
      if (idx !== -1) boundaryIndex = idx;
    }

    const historyMessages = messages.slice(boundaryIndex, -1);
    if (historyMessages.length === 0) return { isNewTopic: false, usage: null };

    const historyEntries = historyMessages.map((m) => {
      const text = extractText(m);
      const prefix = m.role === "user" ? "User" : "Assistant";
      const truncated =
        m.role === "assistant" && text.length > 300
          ? text.slice(0, 300) + "..."
          : text;
      return `${prefix}: ${truncated}`;
    });
    const historyText = historyEntries.join("\n");

    const newMessage = messages[messages.length - 1];
    const newMessageText = extractText(newMessage);

    const result = await generateText({
      model,
      maxOutputTokens: 10,
      system: TOPIC_DETECTION_PROMPT,
      messages: [
        {
          role: "user",
          content: `<history>\n${historyText}\n</history>\n\n<new_message>\n${newMessageText}\n</new_message>`
        }
      ]
    });

    return {
      isNewTopic: result.text.trim() === "Y",
      usage: {
        inputTokens: result.usage.inputTokens ?? 0,
        outputTokens: result.usage.outputTokens ?? 0
      }
    };
  } catch {
    return { isNewTopic: false, usage: null };
  }
}
