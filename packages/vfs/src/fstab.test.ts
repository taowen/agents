import { describe, it, expect } from "vitest";
import { parseOptions, parseFstab, DEFAULT_FSTAB } from "./fstab";

describe("parseOptions", () => {
  it("parses 'defaults' as key with empty value", () => {
    expect(parseOptions("defaults")).toEqual({ defaults: "" });
  });

  it("parses key=value pairs", () => {
    expect(parseOptions("ref=main,depth=1")).toEqual({
      ref: "main",
      depth: "1"
    });
  });

  it("parses empty string as empty object", () => {
    expect(parseOptions("")).toEqual({});
  });

  it("handles mixed keys and key=value", () => {
    expect(parseOptions("rw,ref=dev")).toEqual({ rw: "", ref: "dev" });
  });
});

describe("parseFstab", () => {
  it("parses a standard line", () => {
    const entries = parseFstab("none  /home/user  agentfs  defaults  0  0\n");
    expect(entries).toHaveLength(1);
    expect(entries[0]).toEqual({
      device: "none",
      mountPoint: "/home/user",
      type: "agentfs",
      options: { defaults: "" },
      dump: 0,
      pass: 0
    });
  });

  it("skips comments", () => {
    const entries = parseFstab(
      "# this is a comment\nnone /data agentfs defaults 0 0\n"
    );
    expect(entries).toHaveLength(1);
    expect(entries[0].mountPoint).toBe("/data");
  });

  it("skips blank lines", () => {
    const entries = parseFstab("\n\nnone /data agentfs defaults 0 0\n\n");
    expect(entries).toHaveLength(1);
  });

  it("parses DEFAULT_FSTAB correctly", () => {
    const entries = parseFstab(DEFAULT_FSTAB);
    expect(entries.length).toBeGreaterThanOrEqual(3);
    const types = entries.map((e) => e.type);
    expect(types).toContain("d1");
    const mountPoints = entries.map((e) => e.mountPoint);
    expect(mountPoints).toContain("/etc");
    expect(mountPoints).toContain("/home/user");
    expect(mountPoints).toContain("/data");
  });
});
