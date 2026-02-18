import { describe, it, expect } from "vitest";
import { normalizePath, parentPath, baseName } from "./fs-helpers";

describe("normalizePath", () => {
  it("returns / for root", () => {
    expect(normalizePath("/")).toBe("/");
  });

  it("returns / for empty string", () => {
    expect(normalizePath("")).toBe("/");
  });

  it("resolves .. segments", () => {
    expect(normalizePath("/a/b/../c")).toBe("/a/c");
  });

  it("resolves leading ..", () => {
    expect(normalizePath("/../a")).toBe("/a");
  });

  it("deduplicates slashes", () => {
    expect(normalizePath("/a//b///c")).toBe("/a/b/c");
  });

  it("strips trailing slash", () => {
    expect(normalizePath("/a/b/")).toBe("/a/b");
  });

  it("strips . segments", () => {
    expect(normalizePath("/a/./b")).toBe("/a/b");
  });
});

describe("parentPath", () => {
  it("returns / for root", () => {
    expect(parentPath("/")).toBe("/");
  });

  it("returns / for top-level path", () => {
    expect(parentPath("/a")).toBe("/");
  });

  it("returns parent for nested path", () => {
    expect(parentPath("/a/b")).toBe("/a");
  });

  it("returns parent for deeper path", () => {
    expect(parentPath("/a/b/c")).toBe("/a/b");
  });
});

describe("baseName", () => {
  it("returns last segment of path", () => {
    expect(baseName("/a/b")).toBe("b");
  });

  it("returns name when no slash", () => {
    expect(baseName("file")).toBe("file");
  });

  it("returns last segment for deeper path", () => {
    expect(baseName("/a/b/c.txt")).toBe("c.txt");
  });
});
