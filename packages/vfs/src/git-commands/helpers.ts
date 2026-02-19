import type { GitFs } from "../git-fs";

export type GitResult = { stdout: string; stderr: string; exitCode: number };
export type GitMount = { gitFs: GitFs; mountPoint: string };

export const VERSION = "git version 2.47.0 (vfs)\n";
export const USAGE_TEXT = `usage: git <command>

Available commands:
  clone  status  commit  push  pull  log  diff  branch  remote  show  rev-parse  version
`;

export function stripLeadingSlash(path: string): string {
  return path.startsWith("/") ? path.slice(1) : path;
}

export function shortBranchName(ref: string): string {
  if (ref.startsWith("refs/heads/")) return ref.slice("refs/heads/".length);
  return ref;
}
