import { describe, it, expect, beforeAll } from "vitest";
import { GitFs } from "./git-fs";
import { createMockGitServer } from "./mock-git-server";

describe("GitFs (local mock)", () => {
  let fs: GitFs;

  beforeAll(async () => {
    const { url, http } = await createMockGitServer({
      "README.md": "Test content",
      "src/index.ts": "export default 42;"
    });
    fs = new GitFs({ url, ref: "main", http });
    await fs.init();
  });

  it("readdir('/') returns file list containing README.md", async () => {
    const entries = await fs.readdir("/");
    expect(entries).toContain("README.md");
  });

  it("readdir('/') returns src directory", async () => {
    const entries = await fs.readdir("/");
    expect(entries).toContain("src");
  });

  it("stat('/') returns isDirectory: true", async () => {
    const st = await fs.stat("/");
    expect(st.isDirectory).toBe(true);
    expect(st.isFile).toBe(false);
  });

  it("stat('/README.md') returns isFile: true", async () => {
    const st = await fs.stat("/README.md");
    expect(st.isFile).toBe(true);
    expect(st.isDirectory).toBe(false);
  });

  it("readFile('/README.md') returns string content", async () => {
    const content = await fs.readFile("/README.md");
    expect(typeof content).toBe("string");
    expect(content).toBe("Test content");
  });

  it("readFile('/src/index.ts') returns nested file content", async () => {
    const content = await fs.readFile("/src/index.ts");
    expect(content).toBe("export default 42;");
  });

  it("readdir('/src') lists files in subdirectory", async () => {
    const entries = await fs.readdir("/src");
    expect(entries).toContain("index.ts");
  });

  it("exists('/README.md') returns true", async () => {
    expect(await fs.exists("/README.md")).toBe(true);
  });

  it("exists('/nonexistent') returns false", async () => {
    expect(await fs.exists("/nonexistent")).toBe(false);
  });

  it("writeFile stores in overlay and can be read back", async () => {
    await fs.writeFile("/foo", "bar");
    const content = await fs.readFile("/foo");
    expect(content).toBe("bar");
    expect(fs.isDirty()).toBe(true);
  });

  it("overlay file appears in readdir", async () => {
    await fs.writeFile("/newfile.txt", "hello");
    const entries = await fs.readdir("/");
    expect(entries).toContain("newfile.txt");
  });

  it("rm marks file as deleted", async () => {
    await fs.rm("/README.md");
    expect(await fs.exists("/README.md")).toBe(false);
    const entries = await fs.readdir("/");
    expect(entries).not.toContain("README.md");
  });

  it("appendFile appends to existing content", async () => {
    // Write to overlay first
    await fs.writeFile("/append-test.txt", "hello");
    await fs.appendFile("/append-test.txt", " world");
    const content = await fs.readFile("/append-test.txt");
    expect(content).toBe("hello world");
  });

  it("mkdir creates overlay directory", async () => {
    await fs.mkdir("/newdir");
    const st = await fs.stat("/newdir");
    expect(st.isDirectory).toBe(true);
  });
});
