import { useSongFeed } from '../lib/useSongFeed';
import { fetchAdminSongs, type Features } from '../lib/api';
import ErrorBoundary from './ErrorBoundary';
import SongCard from './SongCard';

interface Props {
  features: Features;
}

// Pick up new songs from other users without manual refresh. Per-card SSE
// handles status changes for songs we already know about; only *new* songs
// need polling.
const POLL_INTERVAL_MS = 30_000;

export default function AdminFeed({ features }: Props) {
  const {
    songs,
    loading,
    error,
    refreshing,
    refresh,
    proposedClipIds,
    handleClubProposed,
    handleUpdate,
    handleDelete,
    handleRename,
  } = useSongFeed({ fetcher: fetchAdminSongs, pollIntervalMs: POLL_INTERVAL_MS });

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
            <ErrorBoundary key={song.id}>
              <SongCard
                song={song}
                features={features}
                proposedClipIds={proposedClipIds}
                onClubProposed={handleClubProposed}
                owner={song.owner.display_name}
                onUpdate={handleUpdate}
                onDelete={handleDelete}
                onRename={handleRename}
              />
            </ErrorBoundary>
          ))}
        </ul>
      )}
    </div>
  );
}
