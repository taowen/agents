import { describe, it, expect } from "vitest";
import { createGitTestEnv } from "./git-test-helpers";

describe("git rev-parse", () => {
  it("shows full HEAD oid", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git rev-parse HEAD");
    expect(result.exitCode).toBe(0);
    expect(result.stdout.trim()).toMatch(/^[a-f0-9]{40}$/);
  });

  it("shows short HEAD oid with --short", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git rev-parse --short HEAD");
    expect(result.exitCode).toBe(0);
    expect(result.stdout.trim()).toMatch(/^[a-f0-9]{7}$/);
  });

  it("errors with no arguments", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git rev-parse");
    expect(result.exitCode).toBe(128);
    expect(result.stderr).toContain("fatal: bad revision");
  });
});
