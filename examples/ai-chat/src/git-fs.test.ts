import { describe, it, expect, beforeAll } from "vitest";
import { GitFs } from "./git-fs";

// Real integration test — no mocking.
// Uses octocat/Spoon-Knife: a tiny, stable public repo on GitHub.
const REPO_URL = "https://github.com/octocat/Spoon-Knife";

describe("GitFs (real clone)", () => {
  let fs: GitFs;

  beforeAll(async () => {
    fs = new GitFs({ url: REPO_URL, ref: "main" });
    await fs.init();
  }, 30_000);

  it("readdir('/') returns file list containing README.md", async () => {
    const entries = await fs.readdir("/");
    expect(entries).toContain("README.md");
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
    expect(content.length).toBeGreaterThan(0);
  });

  it("exists('/README.md') returns true", async () => {
    expect(await fs.exists("/README.md")).toBe(true);
  });

  it("exists('/nonexistent') returns false", async () => {
    expect(await fs.exists("/nonexistent")).toBe(false);
  });

  it("writeFile throws EROFS", async () => {
    await expect(fs.writeFile("/foo", "bar")).rejects.toThrow("EROFS");
  });
});
