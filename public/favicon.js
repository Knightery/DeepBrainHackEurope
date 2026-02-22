(() => {
  const links = [
    { rel: "icon", type: "image/x-icon", href: "/public/favicon.ico" },
    { rel: "icon", type: "image/svg+xml", href: "/public/favicon.svg" },
    { rel: "icon", type: "image/png", sizes: "32x32", href: "/public/favicon-32x32.png" },
    { rel: "icon", type: "image/png", sizes: "16x16", href: "/public/favicon-16x16.png" },
    { rel: "apple-touch-icon", sizes: "180x180", href: "/public/apple-touch-icon.png" }
  ];

  const applyFavicon = () => {
    for (const spec of links) {
      const selector = spec.sizes
        ? `link[rel="${spec.rel}"][sizes="${spec.sizes}"]`
        : `link[rel="${spec.rel}"]${spec.type ? `[type="${spec.type}"]` : ""}`;
      let node = document.head.querySelector(selector);
      if (!node) {
        node = document.createElement("link");
        document.head.appendChild(node);
      }
      node.rel = spec.rel;
      if (spec.type) node.type = spec.type;
      if (spec.sizes) node.sizes = spec.sizes;
      node.href = spec.href;
    }
  };

  applyFavicon();
  window.addEventListener("DOMContentLoaded", applyFavicon);
})();
