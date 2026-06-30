import { useSongFeed } from '../lib/useSongFeed';
import { fetchSongs, type Features, type Song } from '../lib/api';
import CreatePanel from './CreatePanel';
import SongList from './SongList';

interface Props {
  features: Features;
}

export default function MelodieApp({ features }: Props) {
  const {
    songs,
    setSongs,
    loading,
    error,
    proposedClipIds,
    handleClubProposed,
    handleUpdate,
    handleDelete,
    handleRename,
  } = useSongFeed<Song>({ fetcher: fetchSongs });

  const handleCreated = (song: Song) => {
    setSongs((prev) => [song, ...prev]);
  };

  return (
    <div className="space-y-6">
      <CreatePanel onCreated={handleCreated} />
      <SongList
        songs={songs}
        loading={loading}
        error={error}
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
