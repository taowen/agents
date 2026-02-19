import { describe, it, expect } from "vitest";
import { InMemoryFs, MountableFs } from "just-bash";
import type { IFileSystem } from "just-bash";
import { bootFilesystem } from "./boot";
import { DEFAULT_FSTAB } from "./fstab";
import type { FsTypeRegistry } from "./mount";
import type { FstabEntry } from "./fstab";

function setup() {
  const etcFs = new InMemoryFs();
  const mountableFs = new MountableFs();
  return { etcFs, mountableFs };
}

/**
 * Build a registry that returns etcFs for /etc entries (matching production
 * behaviour where multiple D1FsAdapter instances share the same D1 database)
 * and fresh InMemoryFs instances for everything else.
 */
function makeRegistry(etcFs: IFileSystem): FsTypeRegistry {
  return {
    d1: (entry: FstabEntry) =>
      entry.mountPoint === "/etc" ? etcFs : new InMemoryFs(),
    r2: () => new InMemoryFs()
  };
}

describe("bootFilesystem — Phase 1: /etc mount", () => {
  it("mounts /etc and creates root dir", async () => {
    const { etcFs, mountableFs } = setup();
    await bootFilesystem(mountableFs, { etcFs });
    // /etc should be readable (we can write a file and read it back)
    await mountableFs.writeFile("/etc/test", "ok");
    const content = await mountableFs.readFile("/etc/test", {
      encoding: "utf8"
    });
    expect(content).toBe("ok");
  });
});

describe("bootFilesystem — Phase 2: default fstab", () => {
  it("writes DEFAULT_FSTAB when /etc/fstab does not exist", async () => {
    const { etcFs, mountableFs } = setup();
    const fsTypeRegistry = makeRegistry(etcFs);
    await bootFilesystem(mountableFs, { etcFs, fsTypeRegistry });
    const fstab = (await mountableFs.readFile("/etc/fstab", {
      encoding: "utf8"
    })) as string;
    expect(fstab).toBe(DEFAULT_FSTAB);
  });

  it("reads existing fstab without overwriting", async () => {
    const { etcFs, mountableFs } = setup();
    const customFstab =
      "none  /etc  d1  defaults  0  0\nnone  /data  r2  defaults  0  0\n";
    await etcFs.writeFile("/fstab", customFstab);
    const fsTypeRegistry = makeRegistry(etcFs);
    await bootFilesystem(mountableFs, { etcFs, fsTypeRegistry });
    const fstab = (await mountableFs.readFile("/etc/fstab", {
      encoding: "utf8"
    })) as string;
    expect(fstab).toBe(customFstab);
  });
});

describe("bootFilesystem — Phase 3: agentfs→d1 migration", () => {
  it("rewrites fstab when only agentfs entries exist", async () => {
    const { etcFs, mountableFs } = setup();
    const agentfsFstab =
      "none  /etc  agentfs  defaults  0  0\nnone  /home  agentfs  defaults  0  0\n";
    await etcFs.writeFile("/fstab", agentfsFstab);
    const fsTypeRegistry = makeRegistry(etcFs);
    await bootFilesystem(mountableFs, { etcFs, fsTypeRegistry });
    const fstab = (await mountableFs.readFile("/etc/fstab", {
      encoding: "utf8"
    })) as string;
    // Should have been rewritten to DEFAULT_FSTAB (no agentfs entries)
    expect(fstab).toBe(DEFAULT_FSTAB);
    expect(fstab).not.toContain("agentfs");
  });

  it("preserves git lines during migration", async () => {
    const { etcFs, mountableFs } = setup();
    const mixedFstab =
      "none  /etc  agentfs  defaults  0  0\nhttps://github.com/user/repo  /mnt/repo  git  ref=main  0  0\n";
    await etcFs.writeFile("/fstab", mixedFstab);
    const fsTypeRegistry = makeRegistry(etcFs);
    // git entries require r2Bucket+userId; skip the mount error by not providing them
    await bootFilesystem(mountableFs, { etcFs, fsTypeRegistry });
    const fstab = (await mountableFs.readFile("/etc/fstab", {
      encoding: "utf8"
    })) as string;
    expect(fstab).toContain("git");
    expect(fstab).toContain("/mnt/repo");
    expect(fstab).not.toContain("agentfs");
  });
});

describe("bootFilesystem — Phase 4: fsTypeRegistry mounting", () => {
  it("mounts d1 and r2 entries via registry", async () => {
    const { etcFs, mountableFs } = setup();
    const d1Fs = new InMemoryFs();
    const r2Fs = new InMemoryFs();
    const fsTypeRegistry: FsTypeRegistry = {
      d1: (entry: FstabEntry) => {
        if (entry.mountPoint === "/etc") return etcFs;
        if (entry.mountPoint === "/home/user") return d1Fs;
        return new InMemoryFs();
      },
      r2: () => r2Fs
    };
    await bootFilesystem(mountableFs, { etcFs, fsTypeRegistry });

    // Write through mountableFs and verify via the backing store
    await mountableFs.writeFile("/home/user/test.txt", "hello");
    const content = await d1Fs.readFile("/test.txt", { encoding: "utf8" });
    expect(content).toBe("hello");
  });

  it("factory returning null skips mount", async () => {
    const { etcFs, mountableFs } = setup();
    const fsTypeRegistry: FsTypeRegistry = {
      d1: (entry: FstabEntry) =>
        entry.mountPoint === "/etc" ? etcFs : new InMemoryFs(),
      r2: () => null as unknown as IFileSystem // factory declines
    };
    // Should not throw even though r2 factory returns null
    await expect(
      bootFilesystem(mountableFs, { etcFs, fsTypeRegistry })
    ).resolves.not.toThrow();
  });
});
