/**
 * fs-init — generic filesystem initialization for just-bash + vfs.
 *
 * Provides buildFsTypeRegistry, initBash, and doFstabMount as reusable
 * building blocks so applications can bootstrap a sandboxed bash shell
 * with D1/R2/Google Drive-backed filesystems.
 */

import { Bash, InMemoryFs, MountableFs } from "just-bash";
import type { CustomCommand } from "just-bash";
import { D1FsAdapter } from "./d1-fs-adapter";
import { R2FsAdapter } from "./r2-fs-adapter";
import { GoogleDriveFsAdapter } from "./gdrive-fs-adapter";
import { createMountCommands } from "./commands";
import { createGitCommands } from "./git-commands/index";
import { bootFilesystem } from "./boot";
import type { FsTypeRegistry, MountOptions } from "./mount";
import type { FstabEntry } from "./fstab";

/** Generic bindings that fs-init needs from the host environment. */
export interface FsBindings {
  db: D1Database;
  r2?: R2Bucket;
  googleClientId?: string;
  googleClientSecret?: string;
}

export interface InitBashOptions {
  bindings: FsBindings;
  userId: string;
  customCommands?: CustomCommand[];
  cwd?: string;
}

export function buildFsTypeRegistry(
  bindings: FsBindings,
  userId: string
): FsTypeRegistry {
  return {
    d1: (entry: FstabEntry) =>
      new D1FsAdapter(bindings.db, userId, entry.mountPoint),
    r2: (entry: FstabEntry) =>
      bindings.r2
        ? new R2FsAdapter(bindings.r2, userId, entry.mountPoint)
        : null,
    // Legacy compat: treat agentfs as d1
    agentfs: (entry: FstabEntry) =>
      new D1FsAdapter(bindings.db, userId, entry.mountPoint),
    gdrive: (entry: FstabEntry) =>
      bindings.googleClientId && bindings.googleClientSecret
        ? new GoogleDriveFsAdapter(
            bindings.db,
            userId,
            entry.mountPoint,
            bindings.googleClientId,
            bindings.googleClientSecret,
            entry.options.root_folder_id || undefined
          )
        : null
  };
}

/**
 * Create a new Bash instance + MountableFs for the given user.
 * Phase 1 only (bootstrap): does NOT mount fstab entries.
 */
export function initBash(options: InitBashOptions): {
  bash: Bash;
  mountableFs: MountableFs;
} {
  const { bindings, userId, customCommands = [], cwd = "/home/user" } = options;

  const inMemoryFs = new InMemoryFs();
  inMemoryFs.mkdirSync("/mnt");
  const fs = new MountableFs({ base: inMemoryFs });

  const mountOptions: MountOptions = {
    fsTypeRegistry: buildFsTypeRegistry(bindings, userId),
    protectedMounts: ["/etc", "/home/user", "/data"],
    r2Bucket: bindings.r2,
    userId
  };

  const bash = new Bash({
    fs,
    customCommands: [
      ...createMountCommands(fs, mountOptions),
      ...createGitCommands(fs, mountOptions),
      ...customCommands
    ],
    cwd,
    network: { dangerouslyAllowFullInternetAccess: true },
    executionLimits: {
      maxCommandCount: 1000,
      maxLoopIterations: 1000,
      maxCallDepth: 50,
      maxStringLength: 1_048_576
    }
  });

  return { bash, mountableFs: fs };
}

/**
 * Read /etc/fstab, parse entries, mount all filesystems.
 * Delegates to bootFilesystem from vfs.
 */
export async function doFstabMount(
  mountableFs: MountableFs,
  bindings: FsBindings,
  userId: string
): Promise<void> {
  return bootFilesystem(mountableFs, {
    etcFs: new D1FsAdapter(bindings.db, userId, "/etc"),
    etcFsType: "d1",
    fsTypeRegistry: buildFsTypeRegistry(bindings, userId),
    protectedMounts: ["/etc", "/home/user", "/data"],
    r2Bucket: bindings.r2,
    userId
  });
}
