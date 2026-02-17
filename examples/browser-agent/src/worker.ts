interface Env {
  LLM_BASE_URL: string;
  LLM_MODEL: string;
  LLM_API_KEY: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // GET /api/config — return model info to the client
    if (url.pathname === "/api/config") {
      return Response.json({ model: env.LLM_MODEL });
    }

    // POST /api/v1/* — proxy to LLM provider
    if (url.pathname.startsWith("/api/v1/")) {
      const targetPath = url.pathname.replace("/api/v1", "");
      const targetUrl = env.LLM_BASE_URL + targetPath + url.search;

      const headers = new Headers(request.headers);
      headers.set("Authorization", `Bearer ${env.LLM_API_KEY}`);
      // Remove host header so it doesn't conflict with the target
      headers.delete("host");

      const response = await fetch(targetUrl, {
        method: request.method,
        headers,
        body: request.body
      });

      // Pass through the response (including streaming SSE)
      return new Response(response.body, {
        status: response.status,
        headers: response.headers
      });
    }

    return new Response("Not found", { status: 404 });
  }
} satisfies ExportedHandler<Env>;
