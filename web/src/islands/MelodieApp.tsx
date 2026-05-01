import { useCallback, useEffect, useState } from 'react';
import {
  applySongEvent,
  deleteSong as deleteSongApi,
  fetchSongs,
  renameSong as renameSongApi,
  type Song,
  type SongEvent,
} from '../lib/api';
import CreatePanel from './CreatePanel';
import SongList from './SongList';

export default function MelodieApp() {
  const [songs, setSongs] = useState<Song[]>([]);
  const [loading, setLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchSongs()
      .then((s) => {
        if (!cancelled) setSongs(s);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setListError(e instanceof Error ? e.message : 'Failed to load songs');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleCreated = useCallback((song: Song) => {
    setSongs((prev) => [song, ...prev]);
  }, []);

  const handleUpdate = useCallback((ev: SongEvent) => {
    setSongs((prev) =>
      prev.map((s) => (s.id === ev.song_id ? applySongEvent(s, ev) : s))
    );
  }, []);

  const handleDelete = useCallback(async (id: string) => {
    // Optimistic: drop locally first, roll back if the API rejects.
    const snapshot = songs;
    setSongs((prev) => prev.filter((s) => s.id !== id));
    try {
      await deleteSongApi(id);
    } catch {
      setSongs(snapshot);
      alert('Failed to delete — try again.');
    }
  }, [songs]);

  const handleRename = useCallback(async (id: string, title: string) => {
    await renameSongApi(id, title);
    setSongs((prev) => prev.map((s) => (s.id === id ? { ...s, title } : s)));
  }, []);

  return (
    <div className="space-y-6">
      <CreatePanel onCreated={handleCreated} />
      <SongList
        songs={songs}
        loading={loading}
        error={listError}
        onUpdate={handleUpdate}
        onDelete={handleDelete}
        onRename={handleRename}
      />
    </div>
  );
}
