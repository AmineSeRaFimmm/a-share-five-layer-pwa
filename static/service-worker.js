const CACHE_NAME = "a-share-five-layer-v2";
const ICON_VERSION = "20260619-light-logo-v2";
const CORE_ASSETS = [
  "/",
  `/app/static/manifest.webmanifest?v=${ICON_VERSION}`,
  `/app/static/icon-180.png?v=${ICON_VERSION}`,
  `/app/static/icon-192.png?v=${ICON_VERSION}`,
  `/app/static/icon-512.png?v=${ICON_VERSION}`
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(CORE_ASSETS))
      .catch(() => undefined)
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
