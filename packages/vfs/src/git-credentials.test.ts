import { describe, it, expect } from "vitest";
import {
  parseGitCredentials,
  findCredential,
  formatCredentialLine,
  upsertCredential
} from "./git-credentials";

describe("parseGitCredentials", () => {
  it("parses a single credential line", () => {
    const creds = parseGitCredentials("https://user:pass@github.com\n");
    expect(creds).toHaveLength(1);
    expect(creds[0]).toEqual({
      protocol: "https",
      host: "github.com",
      username: "user",
      password: "pass",
      path: undefined
    });
  });

  it("parses multiple lines", () => {
    const content =
      "https://alice:secret@github.com\nhttps://bob:token@gitlab.com\n";
    const creds = parseGitCredentials(content);
    expect(creds).toHaveLength(2);
    expect(creds[0].username).toBe("alice");
    expect(creds[1].username).toBe("bob");
  });

  it("skips blank lines", () => {
    const creds = parseGitCredentials("\n\nhttps://u:p@host.com\n\n");
    expect(creds).toHaveLength(1);
  });

  it("handles URL-encoded characters", () => {
    const creds = parseGitCredentials("https://user%40org:p%40ss@github.com\n");
    expect(creds[0].username).toBe("user@org");
    expect(creds[0].password).toBe("p@ss");
  });

  it("skips lines without username", () => {
    const creds = parseGitCredentials("https://github.com\n");
    expect(creds).toHaveLength(0);
  });
});

describe("findCredential", () => {
  const creds = parseGitCredentials(
    "https://user:pass@github.com\nhttps://other:token@gitlab.com\n"
  );

  it("finds matching credential by protocol and host", () => {
    const match = findCredential(creds, "https://github.com/org/repo");
    expect(match).toBeDefined();
    expect(match!.username).toBe("user");
  });

  it("returns undefined for non-matching URL", () => {
    const match = findCredential(creds, "https://bitbucket.org/repo");
    expect(match).toBeUndefined();
  });

  it("returns undefined for invalid URL", () => {
    const match = findCredential(creds, "not-a-url");
    expect(match).toBeUndefined();
  });
});

describe("formatCredentialLine", () => {
  it("formats credential with password", () => {
    const line = formatCredentialLine({
      protocol: "https",
      host: "github.com",
      username: "user",
      password: "pass"
    });
    expect(line).toBe("https://user:pass@github.com");
  });

  it("formats credential without password", () => {
    const line = formatCredentialLine({
      protocol: "https",
      host: "github.com",
      username: "user"
    });
    expect(line).toBe("https://user@github.com");
  });

  it("URL-encodes special characters", () => {
    const line = formatCredentialLine({
      protocol: "https",
      host: "github.com",
      username: "user@org",
      password: "p@ss"
    });
    expect(line).toContain("user%40org");
    expect(line).toContain("p%40ss");
  });
});

describe("upsertCredential", () => {
  it("appends new credential", () => {
    const result = upsertCredential("", {
      protocol: "https",
      host: "github.com",
      username: "user",
      password: "pass"
    });
    expect(result).toContain("https://user:pass@github.com");
  });

  it("replaces existing credential for same host", () => {
    const content = "https://old:oldpass@github.com\n";
    const result = upsertCredential(content, {
      protocol: "https",
      host: "github.com",
      username: "new",
      password: "newpass"
    });
    expect(result).toContain("https://new:newpass@github.com");
    expect(result).not.toContain("old");
  });

  it("preserves other credentials", () => {
    const content =
      "https://user1:pass1@github.com\nhttps://user2:pass2@gitlab.com\n";
    const result = upsertCredential(content, {
      protocol: "https",
      host: "github.com",
      username: "updated",
      password: "updatedpass"
    });
    expect(result).toContain("https://updated:updatedpass@github.com");
    expect(result).toContain("https://user2:pass2@gitlab.com");
  });
});
