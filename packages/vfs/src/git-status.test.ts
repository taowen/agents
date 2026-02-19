import { describe, it, expect, beforeEach } from "vitest";
import type { Bash } from "just-bash";
import { createGitTestEnv } from "./git-test-helpers";

describe("git status", () => {
  let bash: Bash;

  beforeEach(async () => {
    ({ bash } = await createGitTestEnv({
      "README.md": "Hello",
      "src/index.ts": "export default 42;"
    }));
  });

  it("shows nothing to commit on clean tree", async () => {
    const result = await bash.exec("git status");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("nothing to commit");
  });

  it("shows new file after writing", async () => {
    await bash.exec("echo hello > /mnt/repo/newfile.txt");
    const result = await bash.exec("git status");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("new file:");
    expect(result.stdout).toContain("newfile.txt");
  });

  it("shows modified after editing existing file", async () => {
    await bash.exec("echo updated > /mnt/repo/README.md");
    const result = await bash.exec("git status");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("modified:");
    expect(result.stdout).toContain("README.md");
  });

  it("shows deleted after removing file", async () => {
    await bash.exec("rm /mnt/repo/README.md");
    const result = await bash.exec("git status");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("deleted:");
    expect(result.stdout).toContain("README.md");
  });

  it("shows short status with -s flag", async () => {
    await bash.exec("echo hello > /mnt/repo/newfile.txt");
    await bash.exec("echo updated > /mnt/repo/README.md");
    const result = await bash.exec("git status -s");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("A  newfile.txt");
    expect(result.stdout).toContain("M  README.md");
  });

  it("shows short status with --short flag", async () => {
    await bash.exec("rm /mnt/repo/README.md");
    const result = await bash.exec("git status --short");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("D  README.md");
  });

  it("shows empty output for short status on clean tree", async () => {
    const result = await bash.exec("git status -s");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toBe("");
  });
});
