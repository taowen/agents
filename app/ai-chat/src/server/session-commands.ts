import { defineCommand } from "just-bash";
import type { CustomCommand } from "just-bash";

/**
 * Create a `web-search` custom command that searches the web via the configured search API.
 *
 * Usage:
 *   web-search <query>   # search the web (query must be in English)
 */
export function createSearchCommand(env: Env): CustomCommand {
  return defineCommand("web-search", async (args: string[]) => {
    if (args.length === 0 || args[0] === "--help" || args[0] === "-h") {
      return {
        stdout:
          "Usage: web-search <query>\n" +
          "  Search the web for real-time information.\n" +
          "  The query must be in English.\n",
        stderr: "",
        exitCode: 0
      };
    }

    if (!env.SEARCH_API_KEY) {
      return {
        stdout: "",
        stderr: "web-search: SEARCH_API_KEY is not set\n",
        exitCode: 1
      };
    }

    const query = args.join(" ");
    try {
      const resp = await fetch(env.SEARCH_API_BASE_URL + "/v1/responses", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer " + env.SEARCH_API_KEY
        },
        body: JSON.stringify({
          model: env.SEARCH_API_MODEL,
          input: [{ role: "user", content: "Use web_search tool: " + query }],
          tools: [{ type: "web_search" }]
        })
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const data = (await resp.json()) as any;
      if (data.error) {
        return {
          stdout: "",
          stderr: "web-search: " + data.error.message + "\n",
          exitCode: 1
        };
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const outputMsg = data.output?.find((o: any) => o.type === "message");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const textContent = outputMsg?.content?.find(
        (c: any) => c.type === "output_text"
      );
      const text = textContent?.text || JSON.stringify(data.output);
      return { stdout: text + "\n", stderr: "", exitCode: 0 };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { stdout: "", stderr: "web-search: " + msg + "\n", exitCode: 1 };
    }
  });
}

/**
 * Create a `web-fetch` custom command that fetches a URL and returns its content as markdown
 * using the same search API as web-search.
 *
 * Usage:
 *   web-fetch <url>   # fetch a webpage and return markdown
 */
export function createWebFetchCommand(env: Env): CustomCommand {
  return defineCommand("web-fetch", async (args: string[]) => {
    if (args.length === 0 || args[0] === "--help" || args[0] === "-h") {
      return {
        stdout:
          "Usage: web-fetch <url>\n" +
          "  Fetch a webpage and return its content as markdown.\n",
        stderr: "",
        exitCode: 0
      };
    }

    if (!env.SEARCH_API_KEY) {
      return {
        stdout: "",
        stderr: "web-fetch: SEARCH_API_KEY is not set\n",
        exitCode: 1
      };
    }

    const url = args[0];
    try {
      const resp = await fetch(env.SEARCH_API_BASE_URL + "/v1/responses", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer " + env.SEARCH_API_KEY
        },
        body: JSON.stringify({
          model: env.SEARCH_API_MODEL,
          input: [
            {
              role: "user",
              content:
                "Use web_search tool to fetch and read the content of this URL: " +
                url +
                "\nReturn the full page content as markdown."
            }
          ],
          tools: [{ type: "web_search" }]
        })
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const data = (await resp.json()) as any;
      if (data.error) {
        return {
          stdout: "",
          stderr: "web-fetch: " + data.error.message + "\n",
          exitCode: 1
        };
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const outputMsg = data.output?.find((o: any) => o.type === "message");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const textContent = outputMsg?.content?.find(
        (c: any) => c.type === "output_text"
      );
      const text = textContent?.text || JSON.stringify(data.output);
      return { stdout: text + "\n", stderr: "", exitCode: 0 };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { stdout: "", stderr: "web-fetch: " + msg + "\n", exitCode: 1 };
    }
  });
}

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
