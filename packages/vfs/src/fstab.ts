/**
 * fstab parser — reads /etc/fstab format and returns structured entries.
 *
 * Standard 6-column format:
 *   <device>  <mountpoint>  <type>  <options>  <dump>  <pass>
 */

export interface FstabEntry {
  device: string;
  mountPoint: string;
  type: string;
  options: Record<string, string>;
  dump: number;
  pass: number;
}

export const DEFAULT_FSTAB = `# /etc/fstab - virtual filesystem table
# <device>  <mountpoint>  <type>  <options>  <dump>  <pass>
none  /etc        d1  defaults  0  0
none  /home/user  d1  defaults  0  0
none  /data       r2  defaults  0  0
`;

/**
 * Parse a comma-separated options string into key=value pairs.
 * "defaults" → { defaults: "" }
 * "ref=main,depth=1" → { ref: "main", depth: "1" }
 */
export function parseOptions(str: string): Record<string, string> {
  const result: Record<string, string> = {};
  for (const pair of str.split(",")) {
    const trimmed = pair.trim();
    if (!trimmed) continue;
    const eq = trimmed.indexOf("=");
    if (eq === -1) {
      result[trimmed] = "";
    } else {
      result[trimmed.slice(0, eq)] = trimmed.slice(eq + 1);
    }
  }
  return result;
}

/**
 * Parse fstab content into structured entries.
 * Skips blank lines and comments (lines starting with #).
 */
export function parseFstab(content: string): FstabEntry[] {
  const entries: FstabEntry[] = [];
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const fields = trimmed.split(/\s+/);
    if (fields.length < 3) continue;
    entries.push({
      device: fields[0],
      mountPoint: fields[1],
      type: fields[2],
      options: parseOptions(fields[3] || "defaults"),
      dump: parseInt(fields[4] || "0", 10) || 0,
      pass: parseInt(fields[5] || "0", 10) || 0
    });
  }
  return entries;
}
