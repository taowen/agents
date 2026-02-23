import * as Sentry from "@sentry/cloudflare";
import { mkdirStatement, writeFileOnceStatement } from "./db";

/**
 * Minimal message shape needed for writing chat history.
 * Compatible with UIMessage from the ai SDK.
 */
interface WritableMessage {
  role: string;
  parts: Array<{
    type: string;
    text?: string;
    toolCallId?: string;
    toolName?: string;
    input?: unknown;
    output?: unknown;
  }>;
}

/**
 * Write chat history to /home/user/.chat/{sessionDir}/ as per-message files.
 * All writes use INSERT OR IGNORE — only new messages actually hit disk.
 */
export async function writeChatHistory(
  messages: WritableMessage[],
  db: D1Database,
  userId: string,
  sessionUuid: string | null,
  sessionDir: string
): Promise<void> {
  if (messages.length === 0) return;

  const enc = new TextEncoder();
  const base = `/home/user/.chat/${sessionDir}`;
  const toolsDir = `${base}/tools`;

  const stmts: D1PreparedStatement[] = [
    mkdirStatement(db, userId, "/home/user/.chat", "/home/user", ".chat"),
    mkdirStatement(db, userId, base, "/home/user/.chat", sessionDir),
    mkdirStatement(db, userId, toolsDir, base, "tools"),
    // Ensure .memory/ directory exists
    mkdirStatement(db, userId, "/home/user/.memory", "/home/user", ".memory")
  ];

  // Write .meta.md (INSERT OR IGNORE — only on first write)
  const firstUserMsg = messages.find((m) => m.role === "user");
  let title = "Untitled";
  if (firstUserMsg) {
    for (const part of firstUserMsg.parts) {
      if (part.type === "text" && part.text) {
        title = part.text.slice(0, 100).replace(/\n/g, " ");
        break;
      }
    }
  }
  const date = new Date().toISOString().slice(0, 10);
  const metaContent = `title: ${title}\ndate: ${date}\nsession: ${sessionUuid || "unknown"}\n`;
  const metaBuf = enc.encode(metaContent);
  stmts.push(
    writeFileOnceStatement(
      db,
      userId,
      `${base}/.meta.md`,
      base,
      ".meta.md",
      metaBuf
    )
  );

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const num = String(i + 1).padStart(4, "0");
    let text = "";

    for (const part of msg.parts) {
      if (part.type === "text") {
        text += part.text + "\n";
      } else if ("toolCallId" in part && part.toolCallId) {
        const p = part as {
          type: string;
          toolCallId: string;
          toolName?: string;
          input?: Record<string, unknown>;
          output?: unknown;
        };
        const toolName = p.toolName || p.type.replace("tool-", "");
        const shortId = p.toolCallId.slice(0, 8);
        const inputStr =
          toolName === "bash" && p.input?.command
            ? `\`${p.input.command}\``
            : JSON.stringify(p.input || {}).slice(0, 100);
        text += `[${toolName}(${inputStr}) → tools/${shortId}.txt]\n`;

        if (p.output != null) {
          let output: string;
          if (
            toolName === "bash" &&
            typeof p.output === "object" &&
            p.output !== null
          ) {
            const r = p.output as {
              stdout?: string;
              stderr?: string;
              exitCode?: number;
            };
            output = `$ exit ${r.exitCode ?? "?"}\n`;
            if (r.stdout) output += r.stdout;
            if (r.stderr) output += `\n--- stderr ---\n${r.stderr}`;
          } else {
            output =
              typeof p.output === "string"
                ? p.output
                : JSON.stringify(p.output, null, 2);
          }
          const buf = enc.encode(output);
          stmts.push(
            writeFileOnceStatement(
              db,
              userId,
              `${toolsDir}/${shortId}.txt`,
              toolsDir,
              `${shortId}.txt`,
              buf
            )
          );
        }
      }
    }

    const fileName = `${num}-${msg.role}.md`;
    const buf = enc.encode(text);
    stmts.push(
      writeFileOnceStatement(
        db,
        userId,
        `${base}/${fileName}`,
        base,
        fileName,
        buf
      )
    );
  }

  for (let i = 0; i < stmts.length; i += 100) {
    await db.batch(stmts.slice(i, i + 100));
  }
}

/**
 * Read user memory files (.memory/profile.md, preferences.md, entities.md)
 * and return a formatted block for the system prompt.
 */
export async function readMemoryBlock(
  db: D1Database,
  userId: string
): Promise<string> {
  const memPaths: [string, string][] = [
    ["/home/user/.memory/profile.md", "User Profile"],
    ["/home/user/.memory/preferences.md", "User Preferences"],
    ["/home/user/.memory/entities.md", "Known Entities"]
  ];
  try {
    const memResults = await db.batch(
      memPaths.map(([p]) =>
        db
          .prepare(
            "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id=? AND path=?"
          )
          .bind(userId, p)
      )
    );
    const sections: string[] = [];
    for (let i = 0; i < memPaths.length; i++) {
      const row = memResults[i].results[0] as
        | { content: string | null }
        | undefined;
      if (row?.content) {
        sections.push(`## ${memPaths[i][1]}\n${row.content}`);
      }
    }
    if (sections.length > 0) {
      return "\n\n# Memory\n" + sections.join("\n\n");
    }
  } catch (e) {
    console.error("memory read:", e);
    Sentry.captureException(e);
  }
  return "";
}
