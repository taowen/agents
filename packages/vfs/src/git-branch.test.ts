import { describe, it, expect } from "vitest";
import { createGitTestEnv } from "./git-test-helpers";

describe("git branch", () => {
  it("shows current branch", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git branch");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("* main");
  });
});
