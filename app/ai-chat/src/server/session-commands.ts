import { defineCommand } from "just-bash";
import type { CustomCommand } from "just-bash";
import { generateText } from "ai";
import { getCachedLlmConfig, getLlmModel } from "./llm-config";

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
 * via Cloudflare Browser Rendering API. An optional prompt triggers LLM extraction.
 *
 * Usage:
 *   web-fetch <url>          # fetch a webpage and return markdown
 *   web-fetch <url> "<prompt>" # fetch then extract via LLM
 */
export function createWebFetchCommand(
  env: Env,
  db: D1Database,
  userId: string
): CustomCommand {
  return defineCommand("web-fetch", async (args: string[]) => {
    if (args.length === 0 || args[0] === "--help" || args[0] === "-h") {
      return {
        stdout:
          'Usage: web-fetch <url> ["prompt"]\n' +
          "  Fetch a webpage and return its content as markdown.\n" +
          "  If a prompt is given (in double quotes), use LLM to extract relevant content.\n",
        stderr: "",
        exitCode: 0
      };
    }

    if (!env.CF_ACCOUNT_ID || !env.CF_BROWSER_TOKEN) {
      return {
        stdout: "",
        stderr: "web-fetch: CF_ACCOUNT_ID and CF_BROWSER_TOKEN must be set\n",
        exitCode: 1
      };
    }

    const url = args[0];
    const prompt = args.length > 1 ? args.slice(1).join(" ") : null;
    try {
      const resp = await fetch(
        `https://api.cloudflare.com/client/v4/accounts/${env.CF_ACCOUNT_ID}/browser-rendering/markdown`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${env.CF_BROWSER_TOKEN}`
          },
          body: JSON.stringify({ url, gotoOptions: { timeout: 60000 } })
        }
      );
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const data = (await resp.json()) as any;
      if (!data.success) {
        return {
          stdout: "",
          stderr: "web-fetch: " + JSON.stringify(data.errors || data) + "\n",
          exitCode: 1
        };
      }
      const markdown = data.result || "(empty page)";

      if (!prompt) {
        return { stdout: markdown + "\n", stderr: "", exitCode: 0 };
      }

      // LLM extraction
      try {
        const { data: llmConfig } = await getCachedLlmConfig(db, userId, null);
        const model = getLlmModel(env, llmConfig);
        const { text: extracted } = await generateText({
          model,
          system:
            "You are a content extractor. Output ONLY the extracted content, no explanations.",
          messages: [
            { role: "user", content: `${prompt}\n\n---\n\n${markdown}` }
          ],
          maxOutputTokens: 4096
        });
        if (extracted) {
          return { stdout: extracted + "\n", stderr: "", exitCode: 0 };
        }
        return {
          stdout: markdown + "\n",
          stderr:
            "web-fetch: LLM returned no content, falling back to raw markdown\n",
          exitCode: 0
        };
      } catch (llmErr) {
        const llmMsg =
          llmErr instanceof Error ? llmErr.message : String(llmErr);
        return {
          stdout: markdown + "\n",
          stderr:
            "web-fetch: LLM extraction failed (" +
            llmMsg +
            "), falling back to raw markdown\n",
          exitCode: 0
        };
      }
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
