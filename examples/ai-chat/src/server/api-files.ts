/**
 * File Manager API routes.
 * Thin proxy — all requests are forwarded to the ChatAgent DO
 * so the file manager uses the same MountableFs as the bash agent.
 */

export async function handleFileRoutes(
  request: Request,
  env: Env,
  userId: string
): Promise<Response | null> {
  const url = new URL(request.url);
  if (!url.pathname.startsWith("/api/files")) return null;

  // Forward to a dedicated ChatAgent DO instance for this user
  const doId = env.ChatAgent.idFromName(`_files_${userId}`);
  const stub = env.ChatAgent.get(doId);
  const doReq = new Request(request.url, {
    method: request.method,
    headers: new Headers({
      "x-user-id": userId,
      "x-partykit-room": `_files_${userId}`,
      "content-type": request.headers.get("content-type") || ""
    }),
    body: request.body
  });
  return stub.fetch(doReq);
}
