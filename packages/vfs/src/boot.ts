/**
 * boot — bootstrap a MountableFs from /etc/fstab.
 *
 * Mounts /etc first (so fstab can be read), then parses fstab and
 * mounts all remaining entries. Handles agentfs→d1 migration.
 */

import type { IFileSystem, MountableFs } from "just-bash";
import type { MountOptions } from "./mount";
import { parseFstab, DEFAULT_FSTAB } from "./fstab";
import { mountEntry } from "./mount";

export interface BootOptions extends MountOptions {
  /** Filesystem backing /etc (typically D1FsAdapter). */
  etcFs: IFileSystem;
  /** fstab type label for /etc (default: "d1"). */
  etcFsType?: string;
}

/**
 * Two-phase boot:
 *   Phase 1 — mount /etc, ensure it exists.
 *   Phase 2 — read fstab, migrate if needed, mount remaining entries.
 */
export async function bootFilesystem(
  mountableFs: MountableFs,
  options: BootOptions
): Promise<void> {
  const { etcFs, etcFsType } = options;

  // Phase 1: mount /etc, ensure it exists
  mountableFs.mount("/etc", etcFs, etcFsType || "d1");
  await etcFs.mkdir("/", { recursive: true });

  // Phase 2: read fstab (through mounted filesystem)
  let fstabContent: string;
  try {
    fstabContent = (await mountableFs.readFile("/etc/fstab", {
      encoding: "utf8"
    })) as string;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (!msg.includes("ENOENT")) throw e;
    fstabContent = DEFAULT_FSTAB;
    await mountableFs.writeFile("/etc/fstab", fstabContent);
  }

  // Phase 3: agentfs→d1 migration (preserve git lines)
  let entries = parseFstab(fstabContent);
  const hasD1OrR2 = entries.some((e) => e.type === "d1" || e.type === "r2");
  if (!hasD1OrR2) {
    const gitLines = fstabContent.split("\n").filter((line) => {
      const t = line.trim();
      if (!t || t.startsWith("#")) return false;
      return t.split(/\s+/)[2] === "git";
    });
    fstabContent = DEFAULT_FSTAB;
    if (gitLines.length > 0) fstabContent += gitLines.join("\n") + "\n";
    try {
      await mountableFs.writeFile("/etc/fstab", fstabContent);
    } catch (e) {
      console.error("fstab migration write failed:", e);
    }
    entries = parseFstab(fstabContent);
  }

  // Phase 4: mount all entries (ensureRootDir in mountEntry handles dir creation)
  for (const entry of entries) {
    try {
      await mountEntry(entry, mountableFs, options);
    } catch (e) {
      console.error(`fstab: mount failed for ${entry.mountPoint}:`, e);
    }
  }
}
