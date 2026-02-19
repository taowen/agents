import {
  generateOAuthState,
  verifyOAuthState,
  upsertCredential,
  D1FsAdapter
} from "vfs";

async function readGithubConfig(
  db: D1Database,
  userId: string
): Promise<{ client_id: string; client_secret: string } | null> {
  const row = await db
    .prepare(
      "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id=? AND path=?"
    )
    .bind(userId, "/etc/github.json")
    .first<{ content: string | null }>();
  if (!row?.content) return null;
  try {
    return JSON.parse(row.content);
  } catch {
    return null;
  }
}

export async function handleGitHubOAuth(
  request: Request,
  env: Env,
  userId: string
): Promise<Response | null> {
  const url = new URL(request.url);

  // GET /oauth/github — initiate OAuth flow
  if (url.pathname === "/oauth/github" && request.method === "GET") {
    const config = await readGithubConfig(env.DB, userId);
    const clientId = config?.client_id || env.GITHUB_CLIENT_ID;
    const clientSecret = config?.client_secret || env.GITHUB_CLIENT_SECRET;

    if (!clientId || !clientSecret) {
      return new Response("GitHub OAuth not configured", { status: 400 });
    }

    const state = await generateOAuthState(userId, clientSecret);
    const redirectUri = `${url.origin}/oauth/github/callback`;
    const ghUrl = new URL("https://github.com/login/oauth/authorize");
    ghUrl.searchParams.set("client_id", clientId);
    ghUrl.searchParams.set("redirect_uri", redirectUri);
    ghUrl.searchParams.set("scope", "repo");
    ghUrl.searchParams.set("state", state);
    return Response.redirect(ghUrl.toString(), 302);
  }

  // GET /oauth/github/callback — exchange code for token
  if (url.pathname === "/oauth/github/callback" && request.method === "GET") {
    const code = url.searchParams.get("code");
    const state = url.searchParams.get("state");
    if (!code || !state) {
      return new Response("Missing code or state", { status: 400 });
    }

    const config = await readGithubConfig(env.DB, userId);
    const clientId = config?.client_id || env.GITHUB_CLIENT_ID;
    const clientSecret = config?.client_secret || env.GITHUB_CLIENT_SECRET;

    if (!clientId || !clientSecret) {
      return new Response("GitHub OAuth not configured", { status: 400 });
    }

    const payload = await verifyOAuthState(state, clientSecret);
    if (!payload) {
      return new Response("Invalid or expired state", { status: 403 });
    }

    // Exchange code for token
    const tokenRes = await fetch(
      "https://github.com/login/oauth/access_token",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json"
        },
        body: JSON.stringify({
          client_id: clientId,
          client_secret: clientSecret,
          code
        })
      }
    );
    const tokenData = (await tokenRes.json()) as {
      access_token?: string;
      error?: string;
    };
    if (!tokenData.access_token) {
      return new Response(
        `GitHub token exchange failed: ${tokenData.error || "unknown"}`,
        { status: 502 }
      );
    }

    // Store token in D1 files table as /etc/git-credentials
    const d1Fs = new D1FsAdapter(env.DB, userId, "/etc");
    let existing = "";
    try {
      existing = await d1Fs.readFile("/git-credentials");
    } catch {
      /* file doesn't exist yet */
    }

    const updated = upsertCredential(existing, {
      protocol: "https",
      host: "github.com",
      username: "oauth2",
      password: tokenData.access_token
    });
    await d1Fs.writeFile("/git-credentials", updated);

    return Response.redirect(url.origin + "/", 302);
  }

  return null;
}
