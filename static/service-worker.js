const CACHE_NAME = "a-share-five-layer-v3";
const ICON_VERSION = "20260619-light-logo-v2";
const CORE_ASSETS = [
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

  if (event.request.mode === "navigate") {
    event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
    return;
  }

  const url = new URL(event.request.url);
  const isVersionedStaticAsset = url.pathname.startsWith("/app/static/") && url.searchParams.has("v");
  if (isVersionedStaticAsset) {
    event.respondWith(caches.match(event.request).then((cached) => cached || fetch(event.request)));
    return;
  }

  event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
});
