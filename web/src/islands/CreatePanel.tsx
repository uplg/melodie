import * as React from 'react';
import { useState } from 'react';
import type { Song } from '../lib/api';

// Caps mirror the backend's validation (`crates/melodie-api/src/routes/songs.rs`).
// Keep these in sync so the client errors before round-tripping.
const TITLE_MAX = 100;
const TAGS_MAX = 1000;
const EXCLUDE_MAX = 1000;
const LYRICS_MAX = 5000;
const PROMPT_MAX = 500;

type Vocal = '' | 'male' | 'female';
type Variation = '' | 'high' | 'normal' | 'subtle';

type Mode = 'simple' | 'advanced';

interface SimpleState {
  prompt: string;
  instrumental: boolean;
}

interface AdvancedState {
  title: string;
  tags: string;
  exclude_tags: string;
  lyrics: string;
  vocal: Vocal;
  variation: Variation;
  weirdness: number;
  style_influence: number;
  instrumental: boolean;
}

const SIMPLE_INITIAL: SimpleState = {
  prompt: '',
  instrumental: false,
};

const ADVANCED_INITIAL: AdvancedState = {
  title: '',
  tags: '',
  exclude_tags: '',
  lyrics: '',
  vocal: '',
  variation: '',
  weirdness: 50,
  style_influence: 50,
  instrumental: false,
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
  // Default to Simple — fewer fields, lower friction. Each mode keeps its own
  // state so switching tabs doesn't wipe what the user typed.
  const [mode, setMode] = useState<Mode>('simple');
  const [simple, setSimple] = useState<SimpleState>(SIMPLE_INITIAL);
  const [advanced, setAdvanced] = useState<AdvancedState>(ADVANCED_INITIAL);
  const [submit, setSubmit] = useState<SubmitState>({ kind: 'idle' });

  const updateSimple = <K extends keyof SimpleState>(key: K, value: SimpleState[K]) =>
    setSimple((s) => ({ ...s, [key]: value }));
  const updateAdvanced = <K extends keyof AdvancedState>(key: K, value: AdvancedState[K]) =>
    setAdvanced((s) => ({ ...s, [key]: value }));

  function clientValidate(): string | null {
    if (mode === 'simple') {
      const prompt = simple.prompt.trim();
      if (!prompt) return 'Prompt is required.';
      if (prompt.length > PROMPT_MAX)
        return `Prompt must be at most ${PROMPT_MAX} characters.`;
      return null;
    }
    const title = advanced.title.trim();
    if (!title) return 'Title is required.';
    if (title.length > TITLE_MAX) return `Title must be at most ${TITLE_MAX} characters.`;
    const tags = advanced.tags.trim();
    if (!tags) return 'Tags are required (e.g. "indie rock, guitar, upbeat").';
    if (tags.length > TAGS_MAX) return `Tags must be at most ${TAGS_MAX} characters.`;
    if (advanced.exclude_tags.length > EXCLUDE_MAX)
      return `Exclude tags must be at most ${EXCLUDE_MAX} characters.`;
    const lyrics = advanced.lyrics.trim();
    if (!lyrics) return 'Lyrics are required. Use [Verse] / [Chorus] / [Bridge] tags for structure.';
    if (lyrics.length > LYRICS_MAX) return `Lyrics must be at most ${LYRICS_MAX} characters.`;
    return null;
  }

  function buildBody(): Record<string, unknown> {
    if (mode === 'simple') {
      return {
        mode: 'describe',
        prompt: simple.prompt.trim(),
        instrumental: simple.instrumental,
      };
    }
    const body: Record<string, unknown> = {
      mode: 'custom',
      title: advanced.title.trim(),
      tags: advanced.tags.trim(),
      lyrics: advanced.lyrics,
      instrumental: advanced.instrumental,
      // Sliders/selects only sent when the user moved away from the default —
      // backend treats `null` as "let Suno decide".
      weirdness: advanced.weirdness === 50 ? null : advanced.weirdness,
      style_influence:
        advanced.style_influence === 50 ? null : advanced.style_influence,
    };
    if (advanced.exclude_tags.trim()) body.exclude_tags = advanced.exclude_tags.trim();
    if (advanced.vocal) body.vocal = advanced.vocal;
    if (advanced.variation) body.variation = advanced.variation;
    return body;
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
        // Soft reset — keep most of the form so the user can iterate quickly.
        if (mode === 'simple') setSimple({ prompt: '', instrumental: false });
        else setAdvanced((s) => ({ ...s, title: '' }));
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
          Suno generates two variants per request — both consume one quota slot.
        </p>
      </div>

      <ModeTabs mode={mode} onChange={setMode} disabled={isSubmitting} />

      {mode === 'simple' ? (
        <SimpleFields state={simple} update={updateSimple} />
      ) : (
        <AdvancedFields state={advanced} update={updateAdvanced} />
      )}

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
            ? 'Solving captcha + submitting to Suno — this can take ~30–60 s.'
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

// --- mode tabs ---

interface ModeTabsProps {
  mode: Mode;
  onChange: (m: Mode) => void;
  disabled: boolean;
}

function ModeTabs({ mode, onChange, disabled }: ModeTabsProps) {
  const base =
    'flex-1 px-3 py-1.5 text-sm font-medium rounded-md transition-colors disabled:opacity-50';
  const active = 'bg-white dark:bg-neutral-950 shadow-sm';
  const inactive =
    'text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100';
  return (
    <div
      role="tablist"
      aria-label="Generation mode"
      className="inline-flex w-full max-w-xs gap-1 rounded-lg bg-neutral-100 dark:bg-neutral-800 p-1"
    >
      <button
        type="button"
        role="tab"
        aria-selected={mode === 'simple'}
        disabled={disabled}
        onClick={() => onChange('simple')}
        className={`${base} ${mode === 'simple' ? active : inactive}`}
      >
        Simple
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={mode === 'advanced'}
        disabled={disabled}
        onClick={() => onChange('advanced')}
        className={`${base} ${mode === 'advanced' ? active : inactive}`}
      >
        Advanced
      </button>
    </div>
  );
}

// --- simple fields (describe mode) ---

interface SimpleFieldsProps {
  state: SimpleState;
  update: <K extends keyof SimpleState>(key: K, value: SimpleState[K]) => void;
}

function SimpleFields({ state, update }: SimpleFieldsProps) {
  return (
    <>
      <Field label="Prompt" hint={`${state.prompt.length}/${PROMPT_MAX}`}>
        <textarea
          value={state.prompt}
          onChange={(e) => update('prompt', e.target.value)}
          maxLength={PROMPT_MAX}
          required
          rows={5}
          placeholder="a chill lo-fi track about rainy mornings, mellow piano, distant rain"
          className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
        />
      </Field>
      <p className="text-xs text-neutral-500">
        Suno picks the title, tags and lyrics from your prompt. Switch to Advanced
        for full control.
      </p>
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={state.instrumental}
          onChange={(e) => update('instrumental', e.target.checked)}
          className="h-4 w-4"
        />
        Instrumental
      </label>
    </>
  );
}

// --- advanced fields (custom mode) ---

interface AdvancedFieldsProps {
  state: AdvancedState;
  update: <K extends keyof AdvancedState>(key: K, value: AdvancedState[K]) => void;
}

function AdvancedFields({ state, update }: AdvancedFieldsProps) {
  return (
    <>
      <Field label="Title" hint={`${state.title.length}/${TITLE_MAX}`}>
        <input
          type="text"
          value={state.title}
          onChange={(e) => update('title', e.target.value)}
          maxLength={TITLE_MAX}
          required
          placeholder="Weekend Code"
          className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
        />
      </Field>

      <Field label="Tags" hint={`${state.tags.length}/${TAGS_MAX}`}>
        <input
          type="text"
          value={state.tags}
          onChange={(e) => update('tags', e.target.value)}
          maxLength={TAGS_MAX}
          required
          placeholder="indie rock, guitar, upbeat"
          className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
        />
      </Field>

      <Field label="Exclude tags" hint={`${state.exclude_tags.length}/${EXCLUDE_MAX} · optional`}>
        <input
          type="text"
          value={state.exclude_tags}
          onChange={(e) => update('exclude_tags', e.target.value)}
          maxLength={EXCLUDE_MAX}
          placeholder="metal, heavy"
          className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-400"
        />
      </Field>

      <Field label="Lyrics" hint={`${state.lyrics.length}/${LYRICS_MAX}`}>
        <textarea
          value={state.lyrics}
          onChange={(e) => update('lyrics', e.target.value)}
          maxLength={LYRICS_MAX}
          required
          rows={10}
          placeholder={'[Verse]\nLine one of the verse\nLine two\n\n[Chorus]\nHook line\n'}
          className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-neutral-400"
        />
      </Field>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Slider
          label="Weirdness"
          value={state.weirdness}
          onChange={(v) => update('weirdness', v)}
        />
        <Slider
          label="Style influence"
          value={state.style_influence}
          onChange={(v) => update('style_influence', v)}
        />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Field label="Vocal">
          <select
            value={state.vocal}
            onChange={(e) => update('vocal', e.target.value as Vocal)}
            className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm"
          >
            <option value="">Auto</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </Field>
        <Field label="Variation">
          <select
            value={state.variation}
            onChange={(e) => update('variation', e.target.value as Variation)}
            className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm"
          >
            <option value="">Auto</option>
            <option value="high">High</option>
            <option value="normal">Normal</option>
            <option value="subtle">Subtle</option>
          </select>
        </Field>
        <label className="flex items-center gap-2 self-end pb-2 text-sm">
          <input
            type="checkbox"
            checked={state.instrumental}
            onChange={(e) => update('instrumental', e.target.checked)}
            className="h-4 w-4"
          />
          Instrumental
        </label>
      </div>
    </>
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

interface SliderProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
}

function Slider({ label, value, onChange }: SliderProps) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-xs text-neutral-500 tabular-nums">{value}</span>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-2 w-full accent-neutral-900 dark:accent-neutral-100"
      />
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
