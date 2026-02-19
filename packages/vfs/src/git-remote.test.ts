import { describe, it, expect } from "vitest";
import { createGitTestEnv } from "./git-test-helpers";

describe("git remote", () => {
  it("shows origin", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git remote");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toBe("origin\n");
  });

  it("shows verbose remote info", async () => {
    const { bash, url } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git remote -v");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("origin");
    expect(result.stdout).toContain(url);
    expect(result.stdout).toContain("(fetch)");
    expect(result.stdout).toContain("(push)");
  });
});
