import {
  generateOAuthState,
  verifyOAuthState,
  upsertCredential,
  D1FsAdapter
} from "vfs";
import { getSettings, upsertSettings as upsertSettingsDb } from "./db";

export async function handleGitHubOAuth(
  request: Request,
  env: Env,
  userId: string
): Promise<Response | null> {
  const url = new URL(request.url);

  // GET /oauth/github/config — read config (never expose secret)
  if (url.pathname === "/oauth/github/config" && request.method === "GET") {
    const settings = await getSettings(env.DB, userId);
    return Response.json({
      clientId: settings?.github_client_id || "",
      configured: !!(
        settings?.github_client_id && settings?.github_client_secret
      )
    });
  }

  // POST /oauth/github/config — save config
  if (url.pathname === "/oauth/github/config" && request.method === "POST") {
    const body = (await request.json()) as {
      clientId?: string;
      clientSecret?: string;
    };
    const partial: Record<string, string | null> = {};
    if (body.clientId !== undefined)
      partial.github_client_id = body.clientId || null;
    if (body.clientSecret !== undefined)
      partial.github_client_secret = body.clientSecret || null;
    await upsertSettingsDb(env.DB, userId, partial);
    return Response.json({ ok: true });
  }

  // GET /oauth/github — initiate OAuth flow
  if (url.pathname === "/oauth/github" && request.method === "GET") {
    const settings = await getSettings(env.DB, userId);
    const clientId = settings?.github_client_id;
    const clientSecret = settings?.github_client_secret;

    if (!clientId || !clientSecret) {
      // Fall back to env vars
      if (!env.GITHUB_CLIENT_ID || !env.GITHUB_CLIENT_SECRET) {
        return new Response("GitHub OAuth not configured", { status: 400 });
      }
    }

    const secret = clientSecret || env.GITHUB_CLIENT_SECRET!;
    const state = await generateOAuthState(userId, secret);
    const redirectUri = `${url.origin}/oauth/github/callback`;
    const ghUrl = new URL("https://github.com/login/oauth/authorize");
    ghUrl.searchParams.set("client_id", clientId || env.GITHUB_CLIENT_ID!);
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

    const settings = await getSettings(env.DB, userId);
    const clientId = settings?.github_client_id || env.GITHUB_CLIENT_ID;
    const clientSecret =
      settings?.github_client_secret || env.GITHUB_CLIENT_SECRET;

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
