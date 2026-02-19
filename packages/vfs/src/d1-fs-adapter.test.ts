import { describe, it, expect, beforeEach } from "vitest";
import { D1FsAdapter } from "./d1-fs-adapter";
import { MockD1Database } from "./mock-d1-database";

const SCHEMA = `
CREATE TABLE IF NOT EXISTS files (
  user_id TEXT NOT NULL,
  path TEXT NOT NULL,
  parent_path TEXT NOT NULL,
  name TEXT NOT NULL,
  content BLOB,
  is_directory INTEGER DEFAULT 0,
  mode INTEGER DEFAULT 33188,
  size INTEGER DEFAULT 0,
  mtime REAL DEFAULT (unixepoch('now')),
  PRIMARY KEY (user_id, path)
);
CREATE INDEX IF NOT EXISTS idx_files_parent ON files(user_id, parent_path);
`;

function createAdapter(rootPrefix = "") {
  const mockDb = new MockD1Database();
  mockDb.exec(SCHEMA);
  const db = mockDb as unknown as D1Database;
  const adapter = new D1FsAdapter(db, "test-user", rootPrefix);
  return { adapter, mockDb };
}

describe("D1FsAdapter — readFile / writeFile / appendFile", () => {
  let adapter: D1FsAdapter;

  beforeEach(async () => {
    ({ adapter } = createAdapter());
    await adapter.mkdir("/", { recursive: true });
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

  it("writeFile auto-creates parent directories", async () => {
    await adapter.writeFile("/a/b/c.txt", "deep");
    expect(await adapter.exists("/a")).toBe(true);
    expect(await adapter.exists("/a/b")).toBe(true);
    const content = await adapter.readFile("/a/b/c.txt", { encoding: "utf8" });
    expect(content).toBe("deep");
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

  it("readFile throws EISDIR for directory", async () => {
    await adapter.mkdir("/dir");
    await expect(adapter.readFile("/dir")).rejects.toThrow("EISDIR");
  });
});

describe("D1FsAdapter — exists", () => {
  let adapter: D1FsAdapter;

  beforeEach(async () => {
    ({ adapter } = createAdapter());
    await adapter.mkdir("/", { recursive: true });
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
});

describe("D1FsAdapter — stat / lstat", () => {
  let adapter: D1FsAdapter;

  beforeEach(async () => {
    ({ adapter } = createAdapter());
    await adapter.mkdir("/", { recursive: true });
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

  it("stat throws ENOENT for missing path", async () => {
    await expect(adapter.stat("/nope")).rejects.toThrow("ENOENT");
  });

  it("lstat delegates to stat", async () => {
    await adapter.writeFile("/f.txt", "abc");
    const st = await adapter.lstat("/f.txt");
    expect(st.isFile).toBe(true);
  });
});

describe("D1FsAdapter — mkdir", () => {
  let adapter: D1FsAdapter;

  beforeEach(async () => {
    ({ adapter } = createAdapter());
    await adapter.mkdir("/", { recursive: true });
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

  it("mkdir throws ENOENT when parent missing (non-recursive)", async () => {
    await expect(adapter.mkdir("/a/b")).rejects.toThrow("ENOENT");
  });
});

describe("D1FsAdapter — readdir", () => {
  let adapter: D1FsAdapter;

  beforeEach(async () => {
    ({ adapter } = createAdapter());
    await adapter.mkdir("/", { recursive: true });
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

describe("D1FsAdapter — rm", () => {
  let adapter: D1FsAdapter;

  beforeEach(async () => {
    ({ adapter } = createAdapter());
    await adapter.mkdir("/", { recursive: true });
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

  it("rm non-recursive on non-empty dir throws ENOTEMPTY", async () => {
    await adapter.mkdir("/dir");
    await adapter.writeFile("/dir/f.txt", "x");
    await expect(adapter.rm("/dir")).rejects.toThrow("ENOTEMPTY");
  });

  it("rm force on missing path does not throw", async () => {
    await expect(adapter.rm("/nope", { force: true })).resolves.not.toThrow();
  });

  it("rm on missing path without force throws ENOENT", async () => {
    await expect(adapter.rm("/nope")).rejects.toThrow("ENOENT");
  });
});

describe("D1FsAdapter — cp", () => {
  let adapter: D1FsAdapter;

  beforeEach(async () => {
    ({ adapter } = createAdapter());
    await adapter.mkdir("/", { recursive: true });
  });

  it("copies a file", async () => {
    await adapter.writeFile("/src.txt", "data");
    await adapter.cp("/src.txt", "/dst.txt");
    const content = await adapter.readFile("/dst.txt", { encoding: "utf8" });
    expect(content).toBe("data");
  });

  it("cp recursive copies directory tree", async () => {
    await adapter.mkdir("/src");
    await adapter.writeFile("/src/a.txt", "a");
    await adapter.mkdir("/src/sub");
    await adapter.writeFile("/src/sub/b.txt", "b");
    await adapter.cp("/src", "/dst", { recursive: true });
    expect(await adapter.readFile("/dst/a.txt", { encoding: "utf8" })).toBe(
      "a"
    );
    expect(await adapter.readFile("/dst/sub/b.txt", { encoding: "utf8" })).toBe(
      "b"
    );
  });

  it("cp directory without recursive throws EISDIR", async () => {
    await adapter.mkdir("/src");
    await expect(adapter.cp("/src", "/dst")).rejects.toThrow("EISDIR");
  });
});

describe("D1FsAdapter — mv", () => {
  let adapter: D1FsAdapter;

  beforeEach(async () => {
    ({ adapter } = createAdapter());
    await adapter.mkdir("/", { recursive: true });
  });

  it("moves a file", async () => {
    await adapter.writeFile("/old.txt", "data");
    await adapter.mv("/old.txt", "/new.txt");
    expect(await adapter.exists("/old.txt")).toBe(false);
    const content = await adapter.readFile("/new.txt", { encoding: "utf8" });
    expect(content).toBe("data");
  });

  it("moves a file into a subdirectory", async () => {
    await adapter.writeFile("/f.txt", "x");
    await adapter.mkdir("/dir");
    await adapter.mv("/f.txt", "/dir/f.txt");
    expect(await adapter.exists("/f.txt")).toBe(false);
    expect(await adapter.exists("/dir/f.txt")).toBe(true);
  });

  it("mv throws ENOENT for missing source", async () => {
    await expect(adapter.mv("/nope", "/dst")).rejects.toThrow("ENOENT");
  });
});

describe("D1FsAdapter — rootPrefix scoping", () => {
  it("isolates data under rootPrefix", async () => {
    const mockDb = new MockD1Database();
    mockDb.exec(SCHEMA);
    const db = mockDb as unknown as D1Database;
    const adapterA = new D1FsAdapter(db, "user1", "/mnt/a");
    const adapterB = new D1FsAdapter(db, "user1", "/mnt/b");

    await adapterA.writeFile("/file.txt", "from A");
    await adapterB.writeFile("/file.txt", "from B");

    const contentA = await adapterA.readFile("/file.txt", { encoding: "utf8" });
    const contentB = await adapterB.readFile("/file.txt", { encoding: "utf8" });
    expect(contentA).toBe("from A");
    expect(contentB).toBe("from B");
  });
});
