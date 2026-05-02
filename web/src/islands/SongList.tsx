import type { Features, Song, SongEvent } from '../lib/api';
import SongCard from './SongCard';

interface Props {
  songs: Song[];
  loading: boolean;
  error: string | null;
  features: Features;
  proposedClipIds: ReadonlySet<string>;
  onClubProposed: (clipId: string) => void;
  onUpdate: (ev: SongEvent) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => Promise<void>;
}

export default function SongList({
  songs,
  loading,
  error,
  features,
  proposedClipIds,
  onClubProposed,
  onUpdate,
  onDelete,
  onRename,
}: Props) {
  if (loading) {
    return (
      <div className="rounded-md border border-dashed border-neutral-300 dark:border-neutral-700 p-6 text-sm text-neutral-500">
        Loading your songs…
      </div>
    );
  }
  if (error) {
    return (
      <div
        role="alert"
        className="rounded-md border border-red-300 bg-red-50 dark:border-red-900 dark:bg-red-950/40 p-4 text-sm text-red-700 dark:text-red-300"
      >
        {error}
      </div>
    );
  }
  if (songs.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-neutral-300 dark:border-neutral-700 p-6 text-sm text-neutral-500">
        No songs yet — fill the form above and hit Generate.
      </div>
    );
  }
  return (
    <ul className="space-y-3">
      {songs.map((song) => (
        <SongCard
          key={song.id}
          song={song}
          features={features}
          proposedClipIds={proposedClipIds}
          onClubProposed={onClubProposed}
          onUpdate={onUpdate}
          onDelete={onDelete}
          onRename={onRename}
        />
      ))}
    </ul>
  );
}
