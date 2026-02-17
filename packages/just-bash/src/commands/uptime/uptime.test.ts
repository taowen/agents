import { describe, expect, it } from "vitest";
import { Bash } from "../../Bash.js";

describe("uptime", () => {
  it("should output uptime info", async () => {
    const env = new Bash();
    const result = await env.exec("uptime");
    expect(result.stdout).toMatch(
      /^\s\d{2}:\d{2}:\d{2} up 0 min,  1 user,  load average: 0\.00, 0\.00, 0\.00\n$/
    );
    expect(result.stderr).toBe("");
    expect(result.exitCode).toBe(0);
  });

  it("should respect TZ env var", async () => {
    const env = new Bash({ env: { TZ: "UTC" } });
    const result = await env.exec("uptime");
    // Should contain a valid HH:MM:SS time
    expect(result.stdout).toMatch(/^\s\d{2}:\d{2}:\d{2} up/);
    expect(result.exitCode).toBe(0);

    // Extract the hour from the output
    const match = result.stdout.match(/(\d{2}):\d{2}:\d{2}/);
    expect(match).not.toBeNull();
    const hour = Number.parseInt(match![1], 10);
    const utcHour = new Date().getUTCHours();
    // Allow 1-hour tolerance for test running near hour boundary
    expect(
      Math.abs(hour - utcHour) <= 1 || Math.abs(hour - utcHour) === 23
    ).toBe(true);
  });

  it("should respect TZ set via setEnv()", async () => {
    const env = new Bash();
    env.setEnv("TZ", "UTC");
    const result = await env.exec("uptime");
    expect(result.stdout).toMatch(/^\s\d{2}:\d{2}:\d{2} up/);
    expect(result.exitCode).toBe(0);
  });
});
