import { describe, it, expect, beforeEach } from "vitest";
import type { Bash } from "just-bash";
import { createGitTestEnv } from "./git-test-helpers";

describe("git log", () => {
  let bash: Bash;

  beforeEach(async () => {
    ({ bash } = await createGitTestEnv({
      "README.md": "Hello"
    }));
  });

  it("shows initial commit log", async () => {
    const result = await bash.exec("git log");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("commit ");
    expect(result.stdout).toContain("Author:");
    expect(result.stdout).toContain("Date:");
    expect(result.stdout).toContain("Initial commit");
  });

  it("shows oneline format", async () => {
    const result = await bash.exec("git log --oneline");
    expect(result.exitCode).toBe(0);
    // Should be short oid + message
    const lines = result.stdout.trim().split("\n");
    expect(lines.length).toBeGreaterThanOrEqual(1);
    expect(lines[0]).toMatch(/^[a-f0-9]{7} /);
    expect(lines[0]).toContain("Initial commit");
  });

  it("limits entries with -n flag", async () => {
    await bash.exec("echo a > /mnt/repo/a.txt");
    await bash.exec('git commit -m "first"');
    await bash.exec("echo b > /mnt/repo/b.txt");
    await bash.exec('git commit -m "second"');
    const result = await bash.exec("git log --oneline -n 1");
    expect(result.exitCode).toBe(0);
    const lines = result.stdout.trim().split("\n");
    expect(lines.length).toBe(1);
    expect(lines[0]).toContain("second");
  });

  it("limits entries with -N shorthand", async () => {
    await bash.exec("echo a > /mnt/repo/a.txt");
    await bash.exec('git commit -m "first"');
    await bash.exec("echo b > /mnt/repo/b.txt");
    await bash.exec('git commit -m "second"');
    const result = await bash.exec("git log --oneline -1");
    expect(result.exitCode).toBe(0);
    const lines = result.stdout.trim().split("\n");
    expect(lines.length).toBe(1);
  });
});

describe("git log - multi commit", () => {
  it("shows all commits from a multi-commit repo", async () => {
    const { bash } = await createGitTestEnv([
      { files: { "README.md": "v1" }, message: "first commit" },
      { files: { "README.md": "v2" }, message: "second commit" },
      { files: { "README.md": "v3" }, message: "third commit" }
    ]);
    const result = await bash.exec("git log --oneline");
    expect(result.exitCode).toBe(0);
    const lines = result.stdout.trim().split("\n");
    expect(lines.length).toBe(3);
    expect(lines[0]).toContain("third commit");
    expect(lines[1]).toContain("second commit");
    expect(lines[2]).toContain("first commit");
  });
});
