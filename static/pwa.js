(() => {
  const targetWindow = window.parent || window;
  const targetDocument = targetWindow.document || document;
  const head = targetDocument.head || targetDocument.getElementsByTagName("head")[0];
  if (!head) return;

  const manifestHref = "/app/static/manifest.webmanifest";
  if (!targetDocument.querySelector(`link[rel="manifest"][href="${manifestHref}"]`)) {
    const manifest = targetDocument.createElement("link");
    manifest.rel = "manifest";
    manifest.href = manifestHref;
    head.appendChild(manifest);
  }

  if (!targetDocument.querySelector('meta[name="theme-color"]')) {
    const theme = targetDocument.createElement("meta");
    theme.name = "theme-color";
    theme.content = "#eef6fb";
    head.appendChild(theme);
  }

  if (!targetDocument.querySelector('meta[name="apple-mobile-web-app-capable"]')) {
    const appleCapable = targetDocument.createElement("meta");
    appleCapable.name = "apple-mobile-web-app-capable";
    appleCapable.content = "yes";
    head.appendChild(appleCapable);
  }

  if (!targetDocument.querySelector('meta[name="apple-mobile-web-app-title"]')) {
    const appleTitle = targetDocument.createElement("meta");
    appleTitle.name = "apple-mobile-web-app-title";
    appleTitle.content = "五层博弈";
    head.appendChild(appleTitle);
  }

  if (!targetDocument.querySelector('link[rel="apple-touch-icon"]')) {
    const icon = targetDocument.createElement("link");
    icon.rel = "apple-touch-icon";
    icon.href = "/app/static/icon-192.png";
    head.appendChild(icon);
  }

  if ("serviceWorker" in targetWindow.navigator) {
    targetWindow.navigator.serviceWorker.register("/app/static/service-worker.js").catch(() => {});
  }
})();
