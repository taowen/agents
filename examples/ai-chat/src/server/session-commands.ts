import { defineCommand } from "just-bash";
import type { CustomCommand } from "just-bash";

/**
 * Create a `sessions` custom command that queries D1 for session list.
 *
 * Usage:
 *   sessions                  # list recent 20 sessions
 *   sessions --last 5         # limit to N
 *   sessions --date 2025-06   # filter by date prefix
 *   sessions deploy           # search by title keyword
 */
export function createSessionsCommand(
  db: D1Database,
  userId: string,
  chatAgentNs: DurableObjectNamespace
): CustomCommand {
  return defineCommand("sessions", async (args: string[]) => {
    let limit = 20;
    let dateFilter: string | undefined;
    let keyword: string | undefined;

    for (let i = 0; i < args.length; i++) {
      if (args[i] === "--last" && i + 1 < args.length) {
        limit = parseInt(args[++i], 10) || 20;
      } else if (args[i] === "--date" && i + 1 < args.length) {
        dateFilter = args[++i];
      } else if (args[i] === "--help" || args[i] === "-h") {
        return {
          stdout:
            "Usage: sessions [--last N] [--date YYYY-MM] [keyword]\n" +
            "  --last N       Show last N sessions (default 20)\n" +
            "  --date PREFIX  Filter by date prefix (e.g. 2025-06)\n" +
            "  keyword        Search session titles\n",
          stderr: "",
          exitCode: 0
        };
      } else if (!args[i].startsWith("-")) {
        keyword = args[i];
      }
    }

    let sql = "SELECT id, title, created_at FROM sessions WHERE user_id = ?";
    const bindings: (string | number)[] = [userId];

    if (dateFilter) {
      sql += " AND created_at LIKE ?";
      bindings.push(`${dateFilter}%`);
    }
    if (keyword) {
      sql += " AND title LIKE ?";
      bindings.push(`%${keyword}%`);
    }

    sql += " ORDER BY created_at DESC LIMIT ?";
    bindings.push(limit);

    try {
      const result = await db
        .prepare(sql)
        .bind(...bindings)
        .all<{ id: string; title: string; created_at: string }>();

      if (result.results.length === 0) {
        return { stdout: "(no sessions found)\n", stderr: "", exitCode: 0 };
      }

      const lines = result.results.map((row) => {
        const date = row.created_at.slice(0, 10);
        const doId = chatAgentNs
          .idFromName(`${userId}:${row.id}`)
          .toString()
          .slice(0, 12);
        const title = row.title || "Untitled";
        return `${date}  ${doId}  ${title}`;
      });

      return { stdout: lines.join("\n") + "\n", stderr: "", exitCode: 0 };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { stdout: "", stderr: `sessions: ${msg}\n`, exitCode: 1 };
    }
  });
}
