import { useEffect, useState } from 'react';
import type { Clip, Song, SongEvent } from '../lib/api';

interface Props {
  song: Song;
  onUpdate: (ev: SongEvent) => void;
  onDelete: (id: string) => void;
  /** Async rename — parent owns the API call and updates its state. */
  onRename: (id: string, title: string) => Promise<void>;
  /**
   * When set, surfaces the song's owner on the card. Used by the admin feed
   * — the regular /app list is owner-implicit so this stays undefined there.
   */
  owner?: string;
}

export default function SongCard({ song, onUpdate, onDelete, onRename, owner }: Props) {
  // Subscribe to live updates while the song is still in flight. EventSource
  // is same-origin (proxied via /api/*), so cookies ride along automatically.
  useEffect(() => {
    if (song.status === 'complete' || song.status === 'failed') return;
    const es = new EventSource(`/api/songs/${song.id}/events`);

    const handleUpdate = (e: MessageEvent) => {
      try {
        const data: SongEvent = JSON.parse(e.data);
        onUpdate(data);
        if (data.status === 'complete' || data.status === 'failed') {
          es.close();
        }
      } catch {
        // Malformed frame — ignore; the next tick will resync.
      }
    };
    es.addEventListener('update', handleUpdate as EventListener);

    return () => {
      es.removeEventListener('update', handleUpdate as EventListener);
      es.close();
    };
  }, [song.id, song.status, onUpdate]);

  const handleDelete = () => {
    if (confirm(`Delete "${song.title ?? 'untitled'}"? This trashes the clips on Suno too.`)) {
      onDelete(song.id);
    }
  };

  // --- inline title rename ---
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);

  const startEdit = () => {
    setDraft(song.title ?? '');
    setEditing(true);
  };
  const cancel = () => {
    setEditing(false);
    setDraft('');
  };
  const save = async () => {
    const t = draft.trim();
    if (!t || t === (song.title ?? '')) {
      cancel();
      return;
    }
    setSaving(true);
    try {
      await onRename(song.id, t);
      setEditing(false);
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Rename failed');
    } finally {
      setSaving(false);
    }
  };

  return (
    <li className="rounded-md border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4 space-y-3">
      <header className="flex items-baseline justify-between gap-3">
        <div className="min-w-0">
          {editing ? (
            <input
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  void save();
                } else if (e.key === 'Escape') {
                  cancel();
                }
              }}
              onBlur={() => void save()}
              disabled={saving}
              maxLength={100}
              className="w-full rounded border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-2 py-1 text-base font-semibold tracking-tight focus:outline-none focus:ring-2 focus:ring-neutral-400"
            />
          ) : (
            <h3
              role="button"
              tabIndex={0}
              onClick={startEdit}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  startEdit();
                }
              }}
              title="Click to rename"
              className="truncate text-base font-semibold tracking-tight cursor-text rounded hover:bg-neutral-100 dark:hover:bg-neutral-800 px-1 -mx-1"
            >
              {song.title ?? <span className="text-neutral-500">Untitled</span>}
            </h3>
          )}
          <p className="mt-0.5 text-xs text-neutral-500 truncate">
            {owner && (
              <>
                <span className="font-medium text-neutral-700 dark:text-neutral-300">
                  {owner}
                </span>{' '}
                ·{' '}
              </>
            )}
            {song.tags ?? '—'} · {new Date(song.created_at).toLocaleString()}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <StatusBadge status={song.status} />
          <button
            type="button"
            onClick={handleDelete}
            className="text-xs text-neutral-500 hover:text-red-600 dark:hover:text-red-400 underline"
          >
            Delete
          </button>
        </div>
      </header>

      {song.error && (
        <p
          role="alert"
          className="rounded-md border border-red-300 bg-red-50 dark:border-red-900 dark:bg-red-950/40 px-3 py-2 text-sm text-red-700 dark:text-red-300"
        >
          {song.error}
        </p>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {[0, 1].map((idx) => {
          const clip = song.clips.find((c) => c.variant_index === idx);
          return <ClipSlot key={idx} index={idx} clip={clip} title={song.title} />;
        })}
      </div>
    </li>
  );
}

function ClipSlot({
  index,
  clip,
  title,
}: {
  index: number;
  clip: Clip | undefined;
  title: string | null;
}) {
  if (!clip) {
    return (
      <div className="rounded-md border border-dashed border-neutral-300 dark:border-neutral-700 p-3 text-xs text-neutral-500">
        Variant {index + 1} pending…
      </div>
    );
  }

  const playable = clip.status === 'streaming' || clip.status === 'complete';
  const downloadName = `${slugify(title ?? 'melodie')}-${clip.id.slice(0, 8)}.mp3`;
  const audioUrl = `/api/clips/${clip.id}/audio`;

  return (
    <div className="rounded-md border border-neutral-200 dark:border-neutral-800 p-3 space-y-2">
      <div className="flex items-baseline justify-between text-xs">
        <span className="font-medium">Variant {index + 1}</span>
        <span className="text-neutral-500">
          {clip.status}
          {clip.duration_s ? ` · ${clip.duration_s.toFixed(1)}s` : ''}
        </span>
      </div>

      {playable ? (
        <>
          <audio
            controls
            preload="none"
            src={audioUrl}
            className="w-full h-10"
          />
          <a
            href={audioUrl}
            download={downloadName}
            className="inline-block text-xs text-neutral-700 dark:text-neutral-300 underline"
          >
            Download MP3
          </a>
        </>
      ) : (
        <div className="h-10 rounded bg-neutral-100 dark:bg-neutral-800 animate-pulse" />
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: Song['status'] }) {
  const styles: Record<Song['status'], string> = {
    pending:
      'bg-neutral-200 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300',
    generating:
      'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300',
    complete:
      'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300',
    failed: 'bg-red-100 text-red-800 dark:bg-red-950/60 dark:text-red-300',
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${styles[status]}`}
    >
      {status}
    </span>
  );
}

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '')
    .slice(0, 60);
}
