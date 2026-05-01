/**
 * Server-side helpers for calling the backend during SSR.
 *
 * Browser code should use plain `fetch('/api/...')` — the proxy middleware
 * forwards those to the same backend. These helpers are only for `.astro`
 * pages that need to render based on backend data.
 *
 * Domain TS types live here too so React islands can import them without
 * dragging in any server-side imports.
 */

const API_BACKEND = import.meta.env.API_INTERNAL ?? 'http://127.0.0.1:8080';

export interface User {
  id: string;
  email: string;
  display_name: string;
  role: 'admin' | 'member';
}

export type SongStatus = 'pending' | 'generating' | 'complete' | 'failed';

export interface Clip {
  id: string;
  variant_index: number;
  status: string;
  duration_s: number | null;
  image_url: string | null;
}

export interface Song {
  id: string;
  mode: 'custom' | 'describe';
  title: string | null;
  tags: string | null;
  exclude_tags: string | null;
  lyrics: string | null;
  prompt: string | null;
  model: string;
  status: SongStatus;
  error: string | null;
  created_at: string;
  updated_at: string;
  clips: Clip[];
}

export async function fetchMe(req: Request): Promise<User | null> {
  const cookie = req.headers.get('cookie') ?? '';
  if (!cookie) return null;
  const res = await fetch(`${API_BACKEND}/api/me`, {
    headers: { cookie },
  });
  if (!res.ok) return null;
  return (await res.json()) as User;
}

// --- Browser-side helpers (relative URLs, go through the Astro proxy) ---

export interface SongEvent {
  song_id: string;
  status: SongStatus;
  clips: Array<{
    id: string;
    variant_index: number;
    status: string;
    duration_s: number | null;
    image_url: string | null;
  }>;
}

export async function fetchSongs(): Promise<Song[]> {
  const res = await fetch('/api/songs');
  if (!res.ok) throw new Error(`fetchSongs failed: ${res.status}`);
  return (await res.json()) as Song[];
}

export interface AdminSong extends Song {
  owner: { id: string; display_name: string };
}

export async function fetchAdminSongs(): Promise<AdminSong[]> {
  const res = await fetch('/api/admin/songs');
  if (!res.ok) throw new Error(`fetchAdminSongs failed: ${res.status}`);
  return (await res.json()) as AdminSong[];
}

export async function deleteSong(id: string): Promise<void> {
  const res = await fetch(`/api/songs/${id}`, { method: 'DELETE' });
  if (!res.ok && res.status !== 204) {
    throw new Error(`deleteSong failed: ${res.status}`);
  }
}

export async function renameSong(id: string, title: string): Promise<Song> {
  const res = await fetch(`/api/songs/${id}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) {
    const body = await res
      .json()
      .catch(() => null as { error?: { message?: string } } | null);
    throw new Error(body?.error?.message ?? `renameSong failed: ${res.status}`);
  }
  return (await res.json()) as Song;
}

// --- Admin types + helpers ---

export interface Health {
  status: string;
  last_check: string | null;
  has_jwt: boolean;
  has_clerk_cookie: boolean;
}

export interface Invite {
  code: string;
  role: 'member' | 'admin';
  created_at: string;
  created_by: string | null;
  used_by: string | null;
}

export async function fetchHealth(): Promise<Health> {
  const res = await fetch('/api/admin/health');
  if (!res.ok) throw new Error(`fetchHealth failed: ${res.status}`);
  return (await res.json()) as Health;
}

export async function fetchInvites(): Promise<Invite[]> {
  const res = await fetch('/api/admin/invites');
  if (!res.ok) throw new Error(`fetchInvites failed: ${res.status}`);
  return (await res.json()) as Invite[];
}

export async function createInvite(role: 'member' | 'admin'): Promise<Invite> {
  const res = await fetch('/api/admin/invites', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ role }),
  });
  if (!res.ok) {
    const body = await res
      .json()
      .catch(() => null as { error?: { message?: string } } | null);
    throw new Error(body?.error?.message ?? `createInvite failed: ${res.status}`);
  }
  return (await res.json()) as Invite;
}

export async function setSunoCookie(cookie: string): Promise<void> {
  const res = await fetch('/api/admin/suno-auth', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ clerk_cookie: cookie }),
  });
  if (!res.ok) {
    const body = await res
      .json()
      .catch(() => null as { error?: { message?: string } } | null);
    throw new Error(body?.error?.message ?? `setSunoCookie failed: ${res.status}`);
  }
}

export interface QuotaRow {
  user_id: string;
  display_name: string;
  role: 'admin' | 'member';
  count_today: number;
  /** `null` for admins (no cap) */
  cap: number | null;
}

export async function fetchQuotas(): Promise<QuotaRow[]> {
  const res = await fetch('/api/admin/quotas');
  if (!res.ok) throw new Error(`fetchQuotas failed: ${res.status}`);
  return (await res.json()) as QuotaRow[];
}

export async function resetUserQuota(userId: string): Promise<void> {
  const res = await fetch(`/api/admin/quotas/${userId}`, { method: 'DELETE' });
  if (!res.ok && res.status !== 204)
    throw new Error(`resetUserQuota failed: ${res.status}`);
}

export async function resetAllQuotas(): Promise<void> {
  const res = await fetch('/api/admin/quotas', { method: 'DELETE' });
  if (!res.ok && res.status !== 204)
    throw new Error(`resetAllQuotas failed: ${res.status}`);
}

/**
 * Merge an SSE update into a known Song, preserving fields the event omits.
 *
 * Generic over T so admin views (`AdminSong`) keep their `owner` field
 * through the merge — TS otherwise widens to plain `Song`.
 */
export function applySongEvent<T extends Song>(song: T, ev: SongEvent): T {
  const updatedClips = song.clips.map((c) => {
    const u = ev.clips.find((uc) => uc.id === c.id);
    if (!u) return c;
    return {
      ...c,
      status: u.status,
      duration_s: u.duration_s ?? c.duration_s,
      image_url: u.image_url ?? c.image_url,
    };
  });
  // SSE may report clips the song doesn't yet know about (rare; defensive).
  for (const u of ev.clips) {
    if (!updatedClips.some((c) => c.id === u.id)) {
      updatedClips.push({
        id: u.id,
        variant_index: u.variant_index,
        status: u.status,
        duration_s: u.duration_s,
        image_url: u.image_url,
      });
    }
  }
  return { ...song, status: ev.status, clips: updatedClips };
}
