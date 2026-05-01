import { useCallback, useEffect, useState } from 'react';
import {
  applySongEvent,
  deleteSong as deleteSongApi,
  fetchAdminSongs,
  renameSong as renameSongApi,
  type AdminSong,
  type SongEvent,
} from '../lib/api';
import SongCard from './SongCard';

export default function AdminFeed() {
  const [songs, setSongs] = useState<AdminSong[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(
    async (showSpinner = false) => {
      if (showSpinner) setRefreshing(true);
      try {
        const rows = await fetchAdminSongs();
        setSongs(rows);
        setError(null);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : 'Failed to load feed');
      } finally {
        setLoading(false);
        if (showSpinner) setRefreshing(false);
      }
    },
    []
  );

  useEffect(() => {
    refresh();
    // Pick up new songs from other users without manual refresh. Per-card
    // SSE handles status changes for songs we already know about; only
    // *new* songs need polling.
    const id = setInterval(() => refresh(false), 30_000);
    return () => clearInterval(id);
  }, [refresh]);

  const handleUpdate = useCallback((ev: SongEvent) => {
    setSongs((prev) =>
      prev.map((s) => (s.id === ev.song_id ? applySongEvent(s, ev) : s))
    );
  }, []);

  const handleDelete = useCallback(
    async (id: string) => {
      const snapshot = songs;
      setSongs((prev) => prev.filter((s) => s.id !== id));
      try {
        await deleteSongApi(id);
      } catch {
        setSongs(snapshot);
        alert('Failed to delete — try again.');
      }
    },
    [songs]
  );

  const handleRename = useCallback(async (id: string, title: string) => {
    await renameSongApi(id, title);
    setSongs((prev) => prev.map((s) => (s.id === id ? { ...s, title } : s)));
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-neutral-500">
          {songs.length} song{songs.length === 1 ? '' : 's'} · all users · newest first
        </p>
        <button
          type="button"
          onClick={() => refresh(true)}
          disabled={refreshing}
          className="text-sm rounded-md border border-neutral-300 dark:border-neutral-700 px-2.5 py-1 hover:bg-neutral-100 dark:hover:bg-neutral-900 disabled:opacity-50"
        >
          {refreshing ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div
          role="alert"
          className="rounded-md border border-red-300 bg-red-50 dark:border-red-900 dark:bg-red-950/40 p-4 text-sm text-red-700 dark:text-red-300"
        >
          {error}
        </div>
      )}

      {loading ? (
        <div className="rounded-md border border-dashed border-neutral-300 dark:border-neutral-700 p-6 text-sm text-neutral-500">
          Loading feed…
        </div>
      ) : songs.length === 0 ? (
        <div className="rounded-md border border-dashed border-neutral-300 dark:border-neutral-700 p-6 text-sm text-neutral-500">
          No generations yet across the whole instance.
        </div>
      ) : (
        <ul className="space-y-3">
          {songs.map((song) => (
            <SongCard
              key={song.id}
              song={song}
              owner={song.owner.display_name}
              onUpdate={handleUpdate}
              onDelete={handleDelete}
              onRename={handleRename}
            />
          ))}
        </ul>
      )}
    </div>
  );
}
