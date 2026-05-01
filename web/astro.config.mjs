// @ts-check
import { defineConfig } from 'astro/config';

import node from '@astrojs/node';
import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';

// API_INTERNAL is read by `src/middleware.ts` to proxy /api/* to the local
// axum backend. Default mirrors `cargo run -p melodie-api` on its default port.
const apiInternal = process.env.API_INTERNAL ?? 'http://127.0.0.1:8080';

// https://astro.build/config
export default defineConfig({
  // SSR everywhere: the /api/* proxy middleware and the cookie-based auth
  // helpers need a request object on every route. Pages can opt back into
  // static prerendering with `export const prerender = true`.
  output: 'server',

  // Astro's built-in `checkOrigin` runs before user middleware and 403s any
  // POST without a same-origin `Origin` header. We don't render any HTML
  // forms that POST to .astro routes — every POST goes through `/api/*`,
  // which is reverse-proxied to the axum backend. CSRF on those endpoints
  // is the cookie's `SameSite=Lax` defense (set by tower-sessions). So the
  // origin check just gets in the way of the proxy.
  security: { checkOrigin: false },

  adapter: node({
    mode: 'standalone',
  }),

  integrations: [react()],

  // Match the port `just live` exposes via cloudflared, and the dev port the
  // operator hits in their browser.
  server: { port: 3000, host: true },

  vite: {
    plugins: [tailwindcss()],
    // Surface API_INTERNAL to the middleware via import.meta.env without
    // pulling in a full env-typing setup.
    define: {
      'import.meta.env.API_INTERNAL': JSON.stringify(apiInternal),
    },
    server: {
      // Vite's anti-DNS-rebinding host check rejects requests whose Host
      // header isn't in this list. cloudflared quick-tunnels arrive as
      // `*.trycloudflare.com`, which Vite would otherwise 404. The wildcard
      // entry allows any subdomain.
      allowedHosts: ['.trycloudflare.com', 'localhost', '127.0.0.1'],
    },
  },
});
