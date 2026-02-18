/**
 * mount — fstab-driven filesystem mounting.
 *
 * Reads /etc/fstab from AgentFS, parses entries, and mounts them
 * onto a MountableFs. Extracted from server.ts for testability.
 */

import type { AgentFS } from "agentfs-sdk/cloudflare";
import { AgentFsAdapter } from "./agentfs-adapter";
import { GitFs } from "./git-fs";
import { parseFstab, DEFAULT_FSTAB } from "./fstab";
import { parseGitCredentials, findCredential } from "./git-credentials";
import type { FstabEntry } from "./fstab";
import type { IFileSystem, MountableFs } from "just-bash";

/** Factory that creates a filesystem from an fstab entry, or null to skip. */
export type FsFactory = (entry: FstabEntry) => IFileSystem | null;

/** Registry mapping fstab type names to filesystem factories. */
export type FsTypeRegistry = Record<string, FsFactory>;

/** Recursively ensure a directory path exists in AgentFS. */
async function ensureDirRecursive(
  agentFs: AgentFS,
  dirPath: string
): Promise<void> {
  const parts = dirPath.split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current += `/${part}`;
    try {
      await agentFs.mkdir(current);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (!msg.includes("EEXIST") && !msg.includes("already exists")) throw e;
    }
  }
}

export interface MountOptions {
  gitHttp?: any; // optional: inject mock HTTP transport for testing
  /** Registry of filesystem factories keyed by fstab type name. */
  fsTypeRegistry?: FsTypeRegistry;
  /** Mount points that cannot be umounted (e.g. ["/etc"]). */
  protectedMounts?: string[];
  /** R2 bucket for GitFs overlay persistence. */
  r2Bucket?: R2Bucket;
  /** User ID for scoping R2 keys. */
  userId?: string;
}

/**
 * Mount all entries declared in /etc/fstab.
 * If /etc/fstab doesn't exist, writes the default one first.
 */
export async function mountFstabEntries(
  agentFs: AgentFS | null,
  mountableFs: MountableFs,
  options?: MountOptions
): Promise<void> {
  if (agentFs) {
    // Ensure /etc directory exists in AgentFS
    await ensureDirRecursive(agentFs, "/etc");
  }

  // Read /etc/fstab — if missing, write default
  let fstabContent: string;
  try {
    const data = await mountableFs.readFile("/etc/fstab", { encoding: "utf8" });
    fstabContent = typeof data === "string" ? data : String(data);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes("ENOENT") || msg.includes("not found")) {
      fstabContent = DEFAULT_FSTAB;
      try {
        await mountableFs.writeFile("/etc/fstab", fstabContent);
      } catch {
        // /etc may not be mounted yet during bootstrap
      }
    } else {
      throw e;
    }
  }

  // Parse and mount entries
  const entries = parseFstab(fstabContent);
  for (const entry of entries) {
    await mountEntry(entry, agentFs, mountableFs, options);
  }
}

/**
 * Mount a single fstab entry.
 */
export async function mountEntry(
  entry: FstabEntry,
  agentFs: AgentFS | null,
  mountableFs: MountableFs,
  options?: MountOptions
): Promise<void> {
  if (entry.type === "agentfs") {
    if (!agentFs) {
      console.error(`fstab: agentfs not available for ${entry.mountPoint}`);
      return;
    }
    const adapter = new AgentFsAdapter(agentFs, entry.mountPoint);
    try {
      mountableFs.mount(entry.mountPoint, adapter, "agentfs");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      // Already mounted (e.g. /etc) — skip
      if (msg.includes("already mounted")) return;
      throw e;
    }
    // Ensure the directory (and parents) exist in AgentFS
    await ensureDirRecursive(agentFs, entry.mountPoint);
  } else if (entry.type === "git") {
    const ref = entry.options.ref || "main";
    const depth = entry.options.depth ? parseInt(entry.options.depth, 10) : 1;

    let username: string | undefined = entry.options.username;
    let password: string | undefined = entry.options.password;

    // Fallback: read /etc/git-credentials
    if (!username) {
      try {
        const credContent = await mountableFs.readFile("/etc/git-credentials", {
          encoding: "utf8"
        });
        const creds = parseGitCredentials(credContent as string);
        const match = findCredential(creds, entry.device);
        if (match) {
          username = match.username;
          password = match.password;
        }
      } catch {
        // file doesn't exist — skip
      }
    }

    const onAuth = username ? () => ({ username, password }) : undefined;

    if (!options?.r2Bucket || !options?.userId) {
      console.error(
        `fstab: git mount ${entry.mountPoint} requires r2Bucket and userId`
      );
      return;
    }

    const gitFs = new GitFs({
      url: entry.device,
      ref,
      depth: isNaN(depth) || depth < 1 ? 1 : depth,
      onAuth,
      http: options?.gitHttp,
      r2Bucket: options.r2Bucket,
      userId: options.userId,
      mountPoint: entry.mountPoint
    });

    try {
      mountableFs.mount(entry.mountPoint, gitFs, "git");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("already mounted")) return;
      console.error(`fstab: failed to mount ${entry.mountPoint}: ${msg}`);
      return;
    }

    try {
      await gitFs.init();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error(
        `fstab: clone failed for ${entry.device} at ${entry.mountPoint}: ${msg}`
      );
      // Unmount broken entry so mount point isn't left in broken state
      try {
        mountableFs.unmount(entry.mountPoint);
      } catch {
        /* ignore */
      }
    }
  } else if (options?.fsTypeRegistry?.[entry.type]) {
    // Registry-based mounting for types like d1, r2, etc.
    const factory = options.fsTypeRegistry[entry.type];
    const filesystem = factory(entry);
    if (!filesystem) return; // factory declined (e.g. R2 binding not available)
    try {
      mountableFs.mount(entry.mountPoint, filesystem, entry.type);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("already mounted")) return;
      console.error(`fstab: failed to mount ${entry.mountPoint}: ${msg}`);
    }
  }
}
