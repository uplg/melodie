import { useCallback, useEffect, useState } from 'react';
import {
  applySongEvent,
  deleteSong as deleteSongApi,
  fetchMyProposedClips,
  renameSong as renameSongApi,
  type Song,
  type SongEvent,
} from './api';

interface UseSongFeedOptions<T extends Song> {
  fetcher: () => Promise<T[]>;
  /** Re-fetch on this interval (ms) to pick up songs from other users — SSE
   * only covers status changes on songs we already know about. Omit to fetch
   * once on mount. */
  pollIntervalMs?: number;
}

/**
 * Shared state/handlers for a list-of-songs view (`MelodieApp`, `AdminFeed`):
 * fetch, SSE/poll-driven updates, optimistic delete with rollback, rename,
 * and the "proposed for club" set. Generic over `T` so `AdminFeed` keeps its
 * `owner` field through every update.
 */
export function useSongFeed<T extends Song>({
  fetcher,
  pollIntervalMs,
}: UseSongFeedOptions<T>) {
  const [songs, setSongs] = useState<T[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [proposedClipIds, setProposedClipIds] = useState<ReadonlySet<string>>(
    () => new Set()
  );

  const refresh = useCallback(
    async (showSpinner = false) => {
      if (showSpinner) setRefreshing(true);
      try {
        const rows = await fetcher();
        setSongs(rows);
        setError(null);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : 'Failed to load songs');
      } finally {
        setLoading(false);
        if (showSpinner) setRefreshing(false);
      }
    },
    [fetcher]
  );

  useEffect(() => {
    let cancelled = false;
    refresh();
    fetchMyProposedClips()
      .then((ids) => {
        if (!cancelled) setProposedClipIds(new Set(ids));
      })
      .catch(() => {
        // Non-critical: the worst case is the user re-proposes a clip and the
        // backend silently no-ops it (idempotent UNIQUE constraint).
      });
    if (!pollIntervalMs) {
      return () => {
        cancelled = true;
      };
    }
    const id = setInterval(() => refresh(false), pollIntervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [refresh, pollIntervalMs]);

  const handleClubProposed = useCallback((clipId: string) => {
    setProposedClipIds((prev) => {
      const next = new Set(prev);
      next.add(clipId);
      return next;
    });
  }, []);

  const handleUpdate = useCallback((ev: SongEvent) => {
    setSongs((prev) =>
      prev.map((s) => (s.id === ev.song_id ? applySongEvent(s, ev) : s))
    );
  }, []);

  const handleDelete = useCallback(async (id: string) => {
    // Optimistic: drop locally first, roll back just this item if the API
    // rejects. Rolling back to a full pre-delete snapshot instead would
    // resurrect any *other* item a concurrent delete had since (successfully)
    // removed — this only ever re-adds the one delete that actually failed.
    let removed: T | undefined;
    setSongs((prev) => {
      removed = prev.find((s) => s.id === id);
      return prev.filter((s) => s.id !== id);
    });
    try {
      await deleteSongApi(id);
    } catch {
      setSongs((prev) => {
        if (!removed || prev.some((s) => s.id === id)) return prev;
        return [...prev, removed].sort(
          (a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)
        );
      });
      alert('Failed to delete — try again.');
    }
  }, []);

  const handleRename = useCallback(async (id: string, title: string) => {
    await renameSongApi(id, title);
    setSongs((prev) => prev.map((s) => (s.id === id ? { ...s, title } : s)));
  }, []);

  return {
    songs,
    setSongs,
    loading,
    error,
    refreshing,
    refresh,
    proposedClipIds,
    handleClubProposed,
    handleUpdate,
    handleDelete,
    handleRename,
  };
}
