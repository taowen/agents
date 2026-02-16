/**
 * git-credentials — parse/write the git-credential-store format.
 *
 * Each line: <protocol>://<username>:<password>@<host>[/<path>]
 * See: https://git-scm.com/docs/git-credential-store
 */

export interface GitCredential {
  protocol: string;
  host: string;
  username: string;
  password?: string;
  path?: string;
}

/**
 * Parse a git-credentials file into an array of credential entries.
 * Blank lines and lines that don't parse as URLs are skipped.
 */
export function parseGitCredentials(content: string): GitCredential[] {
  const results: GitCredential[] = [];
  for (const raw of content.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    try {
      const url = new URL(line);
      if (!url.username) continue;
      results.push({
        protocol: url.protocol.replace(/:$/, ""),
        host: url.host,
        username: decodeURIComponent(url.username),
        password: url.password ? decodeURIComponent(url.password) : undefined,
        path:
          url.pathname && url.pathname !== "/"
            ? url.pathname.slice(1)
            : undefined
      });
    } catch {
      // skip unparseable lines
    }
  }
  return results;
}

/**
 * Find the first credential matching the given URL's protocol and host.
 */
export function findCredential(
  credentials: GitCredential[],
  url: string
): GitCredential | undefined {
  try {
    const parsed = new URL(url);
    const protocol = parsed.protocol.replace(/:$/, "");
    const host = parsed.host;
    return credentials.find((c) => c.protocol === protocol && c.host === host);
  } catch {
    return undefined;
  }
}

/**
 * Serialize a single credential entry to the git-credential-store line format.
 */
export function formatCredentialLine(cred: GitCredential): string {
  const userinfo = cred.password
    ? `${encodeURIComponent(cred.username)}:${encodeURIComponent(cred.password)}`
    : encodeURIComponent(cred.username);
  const path = cred.path ? `/${cred.path}` : "";
  return `${cred.protocol}://${userinfo}@${cred.host}${path}`;
}

/**
 * Update or append a credential in a git-credentials file.
 * Matches by protocol + host. Returns the new file content.
 */
export function upsertCredential(content: string, cred: GitCredential): string {
  const lines = content.split("\n");
  const newLine = formatCredentialLine(cred);
  let replaced = false;

  const result = lines.map((raw) => {
    const line = raw.trim();
    if (!line) return raw;
    try {
      const url = new URL(line);
      const protocol = url.protocol.replace(/:$/, "");
      if (protocol === cred.protocol && url.host === cred.host) {
        replaced = true;
        return newLine;
      }
    } catch {
      // keep unparseable lines as-is
    }
    return raw;
  });

  if (!replaced) {
    // Append, ensuring a newline before the new entry
    const trimmed = result.join("\n").trimEnd();
    return trimmed ? `${trimmed}\n${newLine}\n` : `${newLine}\n`;
  }

  return result.join("\n");
}
