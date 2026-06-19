(() => {
  const ICON_VERSION = "20260619-light-logo-v2";
  const targetWindow = window.parent || window;
  const targetDocument = targetWindow.document || document;
  const head = targetDocument.head || targetDocument.getElementsByTagName("head")[0];
  if (!head) return;

  const versioned = (path) => `${path}?v=${ICON_VERSION}`;

  const replaceLink = (selector, rel, href) => {
    targetDocument.querySelectorAll(selector).forEach((node) => node.remove());
    const link = targetDocument.createElement("link");
    link.rel = rel;
    link.href = href;
    head.appendChild(link);
  };

  replaceLink('link[rel="manifest"]', "manifest", versioned("/app/static/manifest.webmanifest"));

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

  replaceLink('link[rel="apple-touch-icon"], link[rel="apple-touch-icon-precomposed"]', "apple-touch-icon", versioned("/app/static/icon-180.png"));

  if ("serviceWorker" in targetWindow.navigator) {
    targetWindow.navigator.serviceWorker.register(versioned("/app/static/service-worker.js")).catch(() => {});
  }
})();
