import * as React from 'react';
import { useState } from 'react';
import type { Song } from '../lib/api';

// Caps mirror the backend's validation (`crates/melodie-api/src/routes/songs.rs`).
// Keep these in sync so the client errors before round-tripping.
const TAGS_MAX = 1000;
const LYRICS_MAX = 5000;

// The local HeartMuLa engine is the only generator. Recorded on the song row.
const ENGINE_MODEL = 'heartmula';

// Languages the engine sings in. `value` is the lowercase tag the backend
// prepends to the style list; `english` is the default.
const LANGUAGES: Array<{ value: string; label: string }> = [
  { value: 'english', label: 'English' },
  { value: 'french', label: 'French' },
  { value: 'spanish', label: 'Spanish' },
  { value: 'german', label: 'German' },
  { value: 'italian', label: 'Italian' },
  { value: 'portuguese', label: 'Portuguese' },
  { value: 'japanese', label: 'Japanese' },
  { value: 'korean', label: 'Korean' },
];

interface FormState {
  styles: string;
  language: string;
  lyrics: string;
}

const INITIAL: FormState = {
  styles: '',
  language: 'english',
  lyrics: '',
};

interface SubmitState {
  kind: 'idle' | 'submitting' | 'error' | 'success';
  message?: string;
  song?: Song;
}

interface Props {
  onCreated?: (song: Song) => void;
}

export default function CreatePanel({ onCreated }: Props) {
  const [form, setForm] = useState<FormState>(INITIAL);
  const [submit, setSubmit] = useState<SubmitState>({ kind: 'idle' });

  const update = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    setForm((s) => ({ ...s, [key]: value }));

  function clientValidate(): string | null {
    const styles = form.styles.trim();
    if (!styles) return 'Styles are required (e.g. "pop, acoustic, warm").';
    if (styles.length > TAGS_MAX) return `Styles must be at most ${TAGS_MAX} characters.`;
    const lyrics = form.lyrics.trim();
    if (!lyrics)
      return 'Lyrics are required. Use [Verse] / [Chorus] / [Bridge] tags for structure.';
    if (lyrics.length > LYRICS_MAX) return `Lyrics must be at most ${LYRICS_MAX} characters.`;
    return null;
  }

  function buildBody(): Record<string, unknown> {
    return {
      lyrics: form.lyrics,
      styles: form.styles.trim(),
      language: form.language,
      model: ENGINE_MODEL,
    };
  }

  async function handleSubmit(e: React.SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    const err = clientValidate();
    if (err) {
      setSubmit({ kind: 'error', message: err });
      return;
    }

    setSubmit({ kind: 'submitting' });
    try {
      const res = await fetch('/api/songs', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(buildBody()),
      });

      if (res.status === 201) {
        const song = (await res.json()) as Song;
        setSubmit({ kind: 'success', song });
        onCreated?.(song);
        // Soft reset — keep styles + language so the user can iterate quickly.
        setForm((s) => ({ ...s, lyrics: '' }));
        return;
      }

      const errBody = await res
        .json()
        .catch(() => null as { error?: { message?: string } } | null);
      setSubmit({
        kind: 'error',
        message: errBody?.error?.message ?? `Generation failed (${res.status})`,
      });
    } catch {
      setSubmit({
        kind: 'error',
        message: 'Network error — check that the backend is running.',
      });
    }
  }

  const isSubmitting = submit.kind === 'submitting';

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-md border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 space-y-5"
    >
      <div>
        <h2 className="text-lg font-semibold tracking-tight">Create</h2>
        <p className="mt-1 text-sm text-neutral-500">
          The local engine renders one clip per request — one quota slot each.
        </p>
      </div>

      <Field
        label="Styles"
        hint={`${form.styles.length}/${TAGS_MAX}`}
      >
        <input
          type="text"
          value={form.styles}
          onChange={(e) => update('styles', e.target.value)}
          maxLength={TAGS_MAX}
          required
          placeholder="pop, acoustic, warm"
          className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
        />
      </Field>
      <p className="-mt-3 text-xs text-neutral-500">
        Comma-separated genres/moods, e.g. <code>pop, acoustic, warm</code>.
      </p>

      <Field label="Language">
        <select
          value={form.language}
          onChange={(e) => update('language', e.target.value)}
          className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
        >
          {LANGUAGES.map((l) => (
            <option key={l.value} value={l.value}>
              {l.label}
            </option>
          ))}
        </select>
      </Field>

      <Field label="Lyrics" hint={`${form.lyrics.length}/${LYRICS_MAX}`}>
        <textarea
          value={form.lyrics}
          onChange={(e) => update('lyrics', e.target.value)}
          maxLength={LYRICS_MAX}
          required
          rows={10}
          placeholder={'[Verse]\nLine one of the verse\nLine two\n\n[Chorus]\nHook line\n'}
          className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-neutral-400"
        />
      </Field>

      {submit.kind === 'error' && (
        <p
          role="alert"
          className="rounded-md border border-red-300 bg-red-50 dark:border-red-900 dark:bg-red-950/40 px-3 py-2 text-sm text-red-700 dark:text-red-300"
        >
          {submit.message}
        </p>
      )}

      {submit.kind === 'success' && submit.song && <SuccessBanner song={submit.song} />}

      <div className="flex items-center justify-between gap-3 pt-1">
        <p className="text-xs text-neutral-500">
          {isSubmitting
            ? 'Generating locally on the engine — this can take a few minutes.'
            : ' '}
        </p>
        <button
          type="submit"
          disabled={isSubmitting}
          className="rounded-md bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 px-4 py-2 text-sm font-medium hover:opacity-90 disabled:opacity-50"
        >
          {isSubmitting ? 'Generating…' : 'Generate'}
        </button>
      </div>
    </form>
  );
}

// --- shared bits ---

interface FieldProps {
  label: string;
  hint?: string;
  children: React.ReactNode;
}

function Field({ label, hint, children }: FieldProps) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium">{label}</span>
        {hint && <span className="text-xs text-neutral-500">{hint}</span>}
      </div>
      <div className="mt-1">{children}</div>
    </label>
  );
}

function SuccessBanner({ song }: { song: Song }) {
  return (
    <div className="rounded-md border border-emerald-300 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950/40 px-3 py-2 text-sm text-emerald-800 dark:text-emerald-300">
      <p className="font-medium">Submitted ✓</p>
      <p className="mt-1 text-xs">
        Song <span className="font-mono">{song.id.slice(0, 8)}</span> is{' '}
        <span className="font-medium">{song.status}</span> with {song.clips.length} clip
        {song.clips.length === 1 ? '' : 's'}. Watch the live status below.
      </p>
    </div>
  );
}
