import { defineMiddleware } from 'astro:middleware';

const API_BACKEND = import.meta.env.API_INTERNAL ?? 'http://127.0.0.1:8080';

/**
 * Single-tunnel deployment plan: cloudflared exposes only the Astro server,
 * which forwards /api/* to the local axum backend. Keeps the surface small
 * and avoids CORS.
 *
 * Streaming-friendly: the request body is passed through and so is the
 * response. Works for SSE on /api/songs/{id}/events because neither side
 * buffers.
 */
export const onRequest = defineMiddleware(async (ctx, next) => {
  const url = new URL(ctx.request.url);
  if (url.pathname !== '/api' && !url.pathname.startsWith('/api/')) {
    return next();
  }

  const target = API_BACKEND + url.pathname + url.search;
  const headers = new Headers(ctx.request.headers);
  // Drop hop-by-hop / origin-bound headers that would confuse the backend.
  headers.delete('host');
  headers.delete('content-length');

  const method = ctx.request.method.toUpperCase();
  const noBody = method === 'GET' || method === 'HEAD';

  const init: RequestInit & { duplex?: 'half' } = {
    method,
    headers,
    redirect: 'manual',
  };
  if (!noBody) {
    init.body = ctx.request.body;
    // Required by the WHATWG spec when streaming a request body. Node's
    // undici fetch enforces this at runtime.
    init.duplex = 'half';
  }

  return fetch(target, init);
});
