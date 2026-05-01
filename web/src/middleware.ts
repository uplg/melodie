import { defineMiddleware } from 'astro:middleware';

const API_BACKEND = import.meta.env.API_INTERNAL ?? 'http://127.0.0.1:8080';

/**
 * Single-tunnel deployment plan: cloudflared exposes only the Astro server,
 * which forwards /api/* to the local axum backend. Keeps the surface small
 * and avoids CORS.
 *
 * Streaming-friendly: we rebuild the Response with the upstream body stream
 * rather than returning the fetch() Response verbatim. Some Astro adapter +
 * undici combinations buffer when handed back the raw Response — most
 * visible on `text/event-stream` where the SSE pipeline silently stalls
 * after the initial frame. The new-Response trick forces the adapter into
 * streaming mode.
 */
export const onRequest = defineMiddleware(async (ctx, next) => {
  const url = new URL(ctx.request.url);
  if (url.pathname !== '/api' && !url.pathname.startsWith('/api/')) {
    return next();
  }

  const target = API_BACKEND + url.pathname + url.search;
  const reqHeaders = new Headers(ctx.request.headers);
  // Drop hop-by-hop / origin-bound headers that would confuse the backend.
  reqHeaders.delete('host');
  reqHeaders.delete('content-length');

  const method = ctx.request.method.toUpperCase();
  const noBody = method === 'GET' || method === 'HEAD';

  const init: RequestInit & { duplex?: 'half' } = {
    method,
    headers: reqHeaders,
    redirect: 'manual',
  };
  if (!noBody) {
    init.body = ctx.request.body;
    // Required by the WHATWG spec when streaming a request body. Node's
    // undici fetch enforces this at runtime.
    init.duplex = 'half';
  }

  const upstream = await fetch(target, init);

  // Rebuild the Response so the adapter sees a fresh streaming body. Without
  // this, SSE responses stall on some Astro/undici combos (handler sends the
  // first frame then nothing reaches the browser until the connection ends).
  const respHeaders = new Headers(upstream.headers);
  // Disable buffering hints for any intermediate (nginx, reverse proxies).
  if (respHeaders.get('content-type')?.includes('text/event-stream')) {
    respHeaders.set('x-accel-buffering', 'no');
    respHeaders.set('cache-control', 'no-cache, no-transform');
  }

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: respHeaders,
  });
});
