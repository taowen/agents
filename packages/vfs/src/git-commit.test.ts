import { describe, it, expect, beforeEach } from "vitest";
import type { Bash } from "just-bash";
import { createGitTestEnv } from "./git-test-helpers";

describe("git commit", () => {
  let bash: Bash;

  beforeEach(async () => {
    ({ bash } = await createGitTestEnv({
      "README.md": "Hello"
    }));
  });

  it("commits with message and shows branch + oid", async () => {
    await bash.exec("echo new > /mnt/repo/file.txt");
    const result = await bash.exec('git commit -m "add file"');
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("[main ");
    expect(result.stdout).toContain("add file");
  });

  it("shows nothing to commit after clean commit", async () => {
    await bash.exec("echo new > /mnt/repo/file.txt");
    await bash.exec('git commit -m "add file"');
    const status = await bash.exec("git status");
    expect(status.stdout).toContain("nothing to commit");
  });

  it("errors without -m flag", async () => {
    await bash.exec("echo new > /mnt/repo/file.txt");
    const result = await bash.exec("git commit");
    expect(result.exitCode).toBe(1);
    expect(result.stderr).toContain("requires a value");
  });

  it("errors with no changes", async () => {
    const result = await bash.exec('git commit -m "empty"');
    expect(result.exitCode).toBe(1);
    expect(result.stderr).toContain("nothing to commit");
  });

  it("shows ahead of remote after commit before push", async () => {
    await bash.exec("echo new > /mnt/repo/file.txt");
    await bash.exec('git commit -m "add file"');
    const status = await bash.exec("git status");
    expect(status.stdout).toContain("ahead");
  });

  it("uses --author flag for commit", async () => {
    await bash.exec("echo new > /mnt/repo/file.txt");
    const result = await bash.exec(
      'git commit -m "authored commit" --author="John Doe <john@example.com>"'
    );
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("authored commit");
    // Verify author via git log
    const log = await bash.exec("git log -1");
    expect(log.stdout).toContain("John Doe");
    expect(log.stdout).toContain("john@example.com");
  });
});
