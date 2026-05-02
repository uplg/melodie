import { useCallback, useEffect, useState } from 'react';
import {
  applySongEvent,
  deleteSong as deleteSongApi,
  fetchMyProposedClips,
  fetchSongs,
  renameSong as renameSongApi,
  type Features,
  type Song,
  type SongEvent,
} from '../lib/api';
import CreatePanel from './CreatePanel';
import SongList from './SongList';

interface Props {
  features: Features;
}

export default function MelodieApp({ features }: Props) {
  const [songs, setSongs] = useState<Song[]>([]);
  const [loading, setLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);
  const [proposedClipIds, setProposedClipIds] = useState<ReadonlySet<string>>(
    () => new Set()
  );

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
    fetchMyProposedClips()
      .then((ids) => {
        if (!cancelled) setProposedClipIds(new Set(ids));
      })
      .catch(() => {
        // Non-critical: the worst case is the user re-proposes a clip and the
        // backend silently no-ops it (idempotent UNIQUE constraint).
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleClubProposed = useCallback((clipId: string) => {
    setProposedClipIds((prev) => {
      const next = new Set(prev);
      next.add(clipId);
      return next;
    });
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
        features={features}
        proposedClipIds={proposedClipIds}
        onClubProposed={handleClubProposed}
        onUpdate={handleUpdate}
        onDelete={handleDelete}
        onRename={handleRename}
      />
    </div>
  );
}
