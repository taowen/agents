import { describe, it, expect, beforeEach } from "vitest";
import { R2FsAdapter } from "./r2-fs-adapter";
import { MockR2Bucket } from "./mock-r2-bucket";

function createAdapter(rootPrefix = "") {
  const bucket = new MockR2Bucket() as unknown as R2Bucket;
  const adapter = new R2FsAdapter(bucket, "test-user", rootPrefix);
  return { adapter, bucket };
}

describe("R2FsAdapter — readFile / writeFile / appendFile", () => {
  let adapter: R2FsAdapter;

  beforeEach(() => {
    ({ adapter } = createAdapter());
  });

  it("round-trips a text file", async () => {
    await adapter.writeFile("/hello.txt", "world");
    const content = await adapter.readFile("/hello.txt", { encoding: "utf8" });
    expect(content).toBe("world");
  });

  it("round-trips a binary file", async () => {
    const data = new Uint8Array([1, 2, 3]);
    await adapter.writeFile("/bin", data);
    const buf = await adapter.readFileBuffer("/bin");
    expect([...buf]).toEqual([1, 2, 3]);
  });

  it("appendFile creates file if missing", async () => {
    await adapter.appendFile("/new.txt", "hello");
    const content = await adapter.readFile("/new.txt", { encoding: "utf8" });
    expect(content).toBe("hello");
  });

  it("appendFile appends to existing file", async () => {
    await adapter.writeFile("/log.txt", "line1\n");
    await adapter.appendFile("/log.txt", "line2\n");
    const content = await adapter.readFile("/log.txt", { encoding: "utf8" });
    expect(content).toBe("line1\nline2\n");
  });

  it("readFile throws ENOENT for missing file", async () => {
    await expect(adapter.readFile("/nope")).rejects.toThrow("ENOENT");
  });
});

describe("R2FsAdapter — exists", () => {
  let adapter: R2FsAdapter;

  beforeEach(() => {
    ({ adapter } = createAdapter());
  });

  it("returns true for existing file", async () => {
    await adapter.writeFile("/f.txt", "x");
    expect(await adapter.exists("/f.txt")).toBe(true);
  });

  it("returns true for existing directory", async () => {
    await adapter.mkdir("/dir");
    expect(await adapter.exists("/dir")).toBe(true);
  });

  it("returns false for non-existent path", async () => {
    expect(await adapter.exists("/nope")).toBe(false);
  });

  it("returns true for root", async () => {
    expect(await adapter.exists("/")).toBe(true);
  });
});

describe("R2FsAdapter — stat", () => {
  let adapter: R2FsAdapter;

  beforeEach(() => {
    ({ adapter } = createAdapter());
  });

  it("stat of file", async () => {
    await adapter.writeFile("/f.txt", "abc");
    const st = await adapter.stat("/f.txt");
    expect(st.isFile).toBe(true);
    expect(st.isDirectory).toBe(false);
    expect(st.size).toBe(3);
  });

  it("stat of directory", async () => {
    await adapter.mkdir("/dir");
    const st = await adapter.stat("/dir");
    expect(st.isFile).toBe(false);
    expect(st.isDirectory).toBe(true);
  });

  it("stat of root", async () => {
    const st = await adapter.stat("/");
    expect(st.isDirectory).toBe(true);
  });

  it("stat throws ENOENT for missing path", async () => {
    await expect(adapter.stat("/nope")).rejects.toThrow("ENOENT");
  });
});

describe("R2FsAdapter — mkdir", () => {
  let adapter: R2FsAdapter;

  beforeEach(() => {
    ({ adapter } = createAdapter());
  });

  it("creates a directory", async () => {
    await adapter.mkdir("/dir");
    const st = await adapter.stat("/dir");
    expect(st.isDirectory).toBe(true);
  });

  it("mkdir recursive creates parents", async () => {
    await adapter.mkdir("/a/b/c", { recursive: true });
    expect(await adapter.exists("/a")).toBe(true);
    expect(await adapter.exists("/a/b")).toBe(true);
    expect(await adapter.exists("/a/b/c")).toBe(true);
  });

  it("mkdir throws EEXIST if already exists (non-recursive)", async () => {
    await adapter.mkdir("/dir");
    await expect(adapter.mkdir("/dir")).rejects.toThrow("EEXIST");
  });

  it("mkdir recursive does not throw if already exists", async () => {
    await adapter.mkdir("/dir");
    await expect(
      adapter.mkdir("/dir", { recursive: true })
    ).resolves.not.toThrow();
  });
});

describe("R2FsAdapter — readdir", () => {
  let adapter: R2FsAdapter;

  beforeEach(() => {
    ({ adapter } = createAdapter());
  });

  it("lists files in directory", async () => {
    await adapter.mkdir("/dir");
    await adapter.writeFile("/dir/a.txt", "a");
    await adapter.writeFile("/dir/b.txt", "b");
    const entries = await adapter.readdir("/dir");
    expect(entries).toEqual(["a.txt", "b.txt"]);
  });

  it("lists subdirectories", async () => {
    await adapter.mkdir("/parent");
    await adapter.mkdir("/parent/child");
    await adapter.writeFile("/parent/child/f.txt", "x");
    const entries = await adapter.readdir("/parent");
    expect(entries).toContain("child");
  });

  it("readdir of empty dir returns empty array", async () => {
    await adapter.mkdir("/empty");
    const entries = await adapter.readdir("/empty");
    expect(entries).toEqual([]);
  });

  it("readdir throws ENOENT for missing dir", async () => {
    await expect(adapter.readdir("/nope")).rejects.toThrow("ENOENT");
  });
});

describe("R2FsAdapter — rm", () => {
  let adapter: R2FsAdapter;

  beforeEach(() => {
    ({ adapter } = createAdapter());
  });

  it("removes a file", async () => {
    await adapter.writeFile("/f.txt", "x");
    await adapter.rm("/f.txt");
    expect(await adapter.exists("/f.txt")).toBe(false);
  });

  it("rm recursive removes directory and contents", async () => {
    await adapter.mkdir("/dir");
    await adapter.writeFile("/dir/f.txt", "x");
    await adapter.rm("/dir", { recursive: true });
    expect(await adapter.exists("/dir")).toBe(false);
    expect(await adapter.exists("/dir/f.txt")).toBe(false);
  });

  it("rm force on missing path does not throw", async () => {
    await expect(adapter.rm("/nope", { force: true })).resolves.not.toThrow();
  });

  it("rm on missing path without force throws ENOENT", async () => {
    await expect(adapter.rm("/nope")).rejects.toThrow("ENOENT");
  });
});

describe("R2FsAdapter — rootPrefix scoping", () => {
  it("isolates data under rootPrefix", async () => {
    const bucket = new MockR2Bucket() as unknown as R2Bucket;
    const adapterA = new R2FsAdapter(bucket, "user1", "/mnt/a");
    const adapterB = new R2FsAdapter(bucket, "user1", "/mnt/b");

    await adapterA.writeFile("/file.txt", "from A");
    await adapterB.writeFile("/file.txt", "from B");

    const contentA = await adapterA.readFile("/file.txt", { encoding: "utf8" });
    const contentB = await adapterB.readFile("/file.txt", { encoding: "utf8" });
    expect(contentA).toBe("from A");
    expect(contentB).toBe("from B");
  });
});
