import { describe, it, expect, beforeEach } from "vitest";
import type { Bash } from "just-bash";
import { createGitTestEnv } from "./git-test-helpers";

describe("git diff", () => {
  let bash: Bash;

  beforeEach(async () => {
    ({ bash } = await createGitTestEnv({
      "README.md": "Hello",
      "src/index.ts": "export default 42;"
    }));
  });

  it("shows empty diff on clean tree", async () => {
    const result = await bash.exec("git diff");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toBe("");
  });

  it("shows unified diff with +/- lines for modified files", async () => {
    await bash.exec("echo updated > /mnt/repo/README.md");
    const result = await bash.exec("git diff");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("diff --git a/README.md b/README.md");
    expect(result.stdout).toContain("--- a/README.md");
    expect(result.stdout).toContain("+++ b/README.md");
    expect(result.stdout).toContain("-Hello");
    expect(result.stdout).toContain("+updated");
  });

  it("shows unified diff for added files", async () => {
    await bash.exec("echo new content > /mnt/repo/new.txt");
    const result = await bash.exec("git diff");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("diff --git a/new.txt b/new.txt");
    expect(result.stdout).toContain("new file");
    expect(result.stdout).toContain("--- /dev/null");
    expect(result.stdout).toContain("+++ b/new.txt");
    expect(result.stdout).toContain("+new content");
  });

  it("shows unified diff for deleted files", async () => {
    await bash.exec("rm /mnt/repo/README.md");
    const result = await bash.exec("git diff");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("diff --git a/README.md b/README.md");
    expect(result.stdout).toContain("deleted file");
    expect(result.stdout).toContain("--- a/README.md");
    expect(result.stdout).toContain("+++ /dev/null");
    expect(result.stdout).toContain("-Hello");
  });

  it("shows --name-only output", async () => {
    await bash.exec("echo updated > /mnt/repo/README.md");
    await bash.exec("echo new > /mnt/repo/new.txt");
    const result = await bash.exec("git diff --name-only");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("new.txt");
    expect(result.stdout).toContain("README.md");
  });

  it("shows --stat output", async () => {
    await bash.exec("echo updated > /mnt/repo/README.md");
    const result = await bash.exec("git diff --stat");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("README.md");
    expect(result.stdout).toContain("1 file changed");
  });

  it("shows new file and deleted file in diff", async () => {
    await bash.exec("echo new > /mnt/repo/new.txt");
    await bash.exec("rm /mnt/repo/README.md");
    const result = await bash.exec("git diff");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("new file");
    expect(result.stdout).toContain("deleted file");
  });
});
