import * as Sentry from "@sentry/cloudflare";
import type { MountableFs } from "just-bash";

export type McpEntry = {
  name: string;
  url: string;
  headers?: Record<string, string>;
};

/**
 * Read MCP server config from /etc/mcp-servers.json.
 */
export async function readMcpConfig(
  mountableFs: MountableFs
): Promise<McpEntry[]> {
  try {
    const buf = await mountableFs.readFileBuffer("/etc/mcp-servers.json");
    const text = new TextDecoder().decode(buf);
    return JSON.parse(text);
  } catch {
    return [];
  }
}

/**
 * Write MCP server config to /etc/mcp-servers.json.
 */
export async function writeMcpConfig(
  mountableFs: MountableFs,
  entries: McpEntry[]
): Promise<void> {
  const text = JSON.stringify(entries, null, 2);
  await mountableFs.writeFile("/etc/mcp-servers.json", text);
}

/**
 * Auto-connect MCP servers from /etc/mcp-servers.json on session start.
 */
export async function ensureMcpServers(
  mountableFs: MountableFs,
  getMcpServers: () => { servers: Record<string, { name: string }> },
  addMcpServer: (
    name: string,
    url: string,
    options: {
      callbackHost: string;
      callbackPath: string;
      transport?: { headers: Record<string, string> };
    }
  ) => Promise<unknown>,
  callbackHost: string,
  callbackPath: string
): Promise<void> {
  const config = await readMcpConfig(mountableFs);
  if (config.length === 0) return;

  const existing = getMcpServers();
  const connectedNames = new Set(
    Object.values(existing.servers).map((s) => s.name)
  );

  for (const entry of config) {
    if (connectedNames.has(entry.name)) continue;
    try {
      await addMcpServer(entry.name, entry.url, {
        callbackHost,
        callbackPath,
        transport: entry.headers ? { headers: entry.headers } : undefined
      });
    } catch (e) {
      console.error(`ensureMcpServers: failed to connect "${entry.name}":`, e);
      Sentry.captureException(e);
    }
  }
}
