/**
 * Shared path helpers for filesystem adapters.
 */

export function parentPath(p: string): string {
  if (p === "/") return "/";
  const last = p.lastIndexOf("/");
  return last === 0 ? "/" : p.slice(0, last);
}

export function baseName(p: string): string {
  const last = p.lastIndexOf("/");
  return last === -1 ? p : p.slice(last + 1);
}

/** Normalize path: resolve "..", dedupe "/", strip trailing "/" */
export function normalizePath(raw: string): string {
  if (!raw || raw === "/") return "/";
  const parts = raw.split("/").filter((p) => p && p !== ".");
  const resolved: string[] = [];
  for (const part of parts) {
    if (part === "..") resolved.pop();
    else resolved.push(part);
  }
  return `/${resolved.join("/")}` || "/";
}
