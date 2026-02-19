import * as Sentry from "@sentry/cloudflare";
import {
  createMountCommands,
  createGitCommands,
  bootFilesystem,
  D1FsAdapter,
  R2FsAdapter,
  GoogleDriveFsAdapter
} from "vfs";
import type { FsTypeRegistry, FstabEntry } from "vfs";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { createSessionsCommand } from "./session-commands";

export function buildFsTypeRegistry(env: Env, userId: string): FsTypeRegistry {
  return {
    d1: (entry: FstabEntry) =>
      new D1FsAdapter(env.DB, userId, entry.mountPoint),
    r2: (entry: FstabEntry) =>
      env.R2 ? new R2FsAdapter(env.R2, userId, entry.mountPoint) : null,
    // Legacy compat: treat agentfs as d1
    agentfs: (entry: FstabEntry) =>
      new D1FsAdapter(env.DB, userId, entry.mountPoint),
    gdrive: (entry: FstabEntry) =>
      env.GOOGLE_CLIENT_ID && env.GOOGLE_CLIENT_SECRET
        ? new GoogleDriveFsAdapter(
            env.DB,
            userId,
            entry.mountPoint,
            env.GOOGLE_CLIENT_ID,
            env.GOOGLE_CLIENT_SECRET,
            entry.options.root_folder_id || undefined
          )
        : null
  };
}

/**
 * Create a new Bash instance + MountableFs for the given user.
 * Phase 1 only (bootstrap): does NOT mount fstab entries.
 */
export function initBash(
  env: Env,
  userId: string
): { bash: Bash; mountableFs: MountableFs } {
  const inMemoryFs = new InMemoryFs();
  inMemoryFs.mkdirSync("/mnt");
  const fs = new MountableFs({ base: inMemoryFs });

  const mountOptions = {
    fsTypeRegistry: buildFsTypeRegistry(env, userId),
    protectedMounts: ["/etc", "/home/user", "/data"],
    r2Bucket: env.R2,
    userId
  };

  const bash = new Bash({
    fs,
    customCommands: [
      ...createMountCommands(fs, mountOptions),
      ...createGitCommands(fs, mountOptions),
      createSessionsCommand(env.DB, userId, env.ChatAgent)
    ],
    cwd: "/home/user",
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
  env: Env,
  userId: string
): Promise<void> {
  return Sentry.startSpan({ name: "doFstabMount", op: "mount" }, () =>
    bootFilesystem(mountableFs, {
      etcFs: new D1FsAdapter(env.DB, userId, "/etc"),
      etcFsType: "d1",
      fsTypeRegistry: buildFsTypeRegistry(env, userId),
      protectedMounts: ["/etc", "/home/user", "/data"],
      r2Bucket: env.R2,
      userId
    })
  );
}
