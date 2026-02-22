const CACHE_NAME = "ai-chat-v1";

const STATIC_EXTENSIONS =
  /\.(js|css|png|jpg|jpeg|svg|gif|webp|woff|woff2|ttf|ico)$/;

self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key))
        )
      )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Skip non-GET requests and WebSocket upgrades
  if (event.request.method !== "GET") return;
  if (event.request.headers.get("upgrade") === "websocket") return;

  // Skip API/auth routes — always network
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/auth/"))
    return;

  // Static assets: cache-first
  if (STATIC_EXTENSIONS.test(url.pathname)) {
    event.respondWith(
      caches.match(event.request).then(
        (cached) =>
          cached ||
          fetch(event.request).then((response) => {
            if (response.ok) {
              const clone = response.clone();
              caches
                .open(CACHE_NAME)
                .then((cache) => cache.put(event.request, clone));
            }
            return response;
          })
      )
    );
    return;
  }

  // HTML and everything else: network-first
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response.ok) {
          const clone = response.clone();
          caches
            .open(CACHE_NAME)
            .then((cache) => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
