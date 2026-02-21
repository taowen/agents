/**
 * Conversation history management: trimming, compaction (summarization),
 * and state persistence across agent calls.
 * Extracted from agent-standalone.ts.
 */

import type { ChatMessage, ToolCall, LlmConfig } from "./types";
import { callLLM } from "./llm-client";

const KEEP_RECENT_TOOL_RESULTS = 3;
const COMPACT_THRESHOLD = 100;
const COMPACT_KEEP_RECENT = 10;

interface ContentPart {
  type: string;
  text?: string;
  image_url?: { url: string };
}

/**
 * Trim older tool results and remove duplicate screenshots in-place.
 */
export function trimMessages(messages: ChatMessage[]): void {
  let toolCount = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "tool") {
      toolCount++;
      if (toolCount > KEEP_RECENT_TOOL_RESULTS) {
        const content = msg.content;
        if (typeof content === "string" && content.length > 200) {
          msg.content =
            content.substring(0, 200) +
            "...(truncated, " +
            content.length +
            " chars total)";
        }
      }
    }
  }

  // Keep only the most recent screenshot user message; replace older ones
  let screenshotCount = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "user" && Array.isArray(msg.content)) {
      const hasImage = (msg.content as ContentPart[]).some(
        (p) => p.type === "image_url"
      );
      if (hasImage) {
        screenshotCount++;
        if (screenshotCount > 1) {
          msg.content = "[previous screenshot removed]";
        }
      }
    }
  }
}

function findSafeCutPoint(messages: ChatMessage[], idealCut: number): number {
  for (let i = idealCut; i > 1; i--) {
    if (
      messages[i].role === "user" &&
      typeof messages[i].content === "string"
    ) {
      return i;
    }
  }
  return idealCut;
}

function buildDigest(messages: ChatMessage[]): string {
  const parts: string[] = [];
  for (const msg of messages) {
    const role = msg.role;
    let text: string;
    if (typeof msg.content === "string") {
      text = msg.content;
    } else if (Array.isArray(msg.content)) {
      text = (msg.content as ContentPart[])
        .filter((p) => p.type === "text" && p.text)
        .map((p) => p.text!)
        .join("\n");
      if (!text) text = "[image]";
    } else {
      text = "";
    }
    if (role === "tool" && text.length > 200) {
      text = text.substring(0, 200) + "...(truncated)";
    }
    if (role === "assistant" && (msg as any).tool_calls) {
      const calls = ((msg as any).tool_calls as ToolCall[])
        .map(
          (tc) =>
            tc.function.name +
            "(" +
            tc.function.arguments.substring(0, 100) +
            ")"
        )
        .join("; ");
      text = (text ? text + "\n" : "") + "[called: " + calls + "]";
    }
    if (text) {
      parts.push(role + ": " + text);
    }
  }
  return parts.join("\n\n");
}

/**
 * Compact conversation by summarizing older messages via LLM.
 * Mutates the messages array in place.
 */
export function compactConversation(
  messages: ChatMessage[],
  config: LlmConfig,
  agentLog: (msg: string) => void
): void {
  if (messages.length <= COMPACT_THRESHOLD) return;

  const idealCut = messages.length - COMPACT_KEEP_RECENT;
  const cutPoint = findSafeCutPoint(messages, idealCut);

  if (cutPoint <= 1) return;

  const toCompact = messages.slice(1, cutPoint);
  const digest = buildDigest(toCompact);

  agentLog(
    "[COMPACT] Summarizing " +
      toCompact.length +
      " messages (cut at " +
      cutPoint +
      ")..."
  );

  let summary: string;
  try {
    const result = callLLM(
      [
        {
          role: "system",
          content:
            "Summarize this conversation between a user and a mobile automation assistant.\n" +
            "Focus on: tasks requested, what was accomplished, current phone state, important context for continuing.\n" +
            "Be concise (under 500 words). Output only the summary."
        },
        { role: "user", content: digest }
      ],
      [],
      config
    );
    summary = result.content || "(empty summary)";
  } catch (e: any) {
    agentLog(
      "[COMPACT] Summarization failed: " +
        e.message +
        " — falling back to truncation"
    );
    const system = messages[0];
    const recent = messages.slice(cutPoint);
    messages.length = 0;
    messages.push(system, ...recent);
    return;
  }

  agentLog(
    "[COMPACT] Summary (" +
      summary.length +
      " chars): " +
      summary.substring(0, 100) +
      "..."
  );

  const summaryMsg: ChatMessage = {
    role: "user",
    content: "[Prior conversation summary]\n" + summary
  };

  // Splice: replace messages[1..cutPoint) with summaryMsg
  messages.splice(1, cutPoint - 1, summaryMsg);
}
