import { useCallback, useEffect, useState } from 'react';
import {
  dismissClubProposal,
  fetchAdminClubProposals,
  type AdminClubProposal,
} from '../lib/api';

export default function AdminClub() {
  const [rows, setRows] = useState<AdminClubProposal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const fresh = await fetchAdminClubProposals();
      setRows(fresh);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load proposals');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleDismiss = useCallback(
    async (id: number) => {
      const snapshot = rows;
      setRows((prev) => prev.filter((r) => r.id !== id));
      try {
        await dismissClubProposal(id);
      } catch {
        setRows(snapshot);
        alert('Failed to dismiss — try again.');
      }
    },
    [rows]
  );

  if (loading) {
    return (
      <div className="rounded-md border border-dashed border-neutral-300 dark:border-neutral-700 p-6 text-sm text-neutral-500">
        Loading proposals…
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

  if (rows.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-neutral-300 dark:border-neutral-700 p-6 text-sm text-neutral-500">
        No proposals yet — friends will appear here when they hit "Propose for club".
      </div>
    );
  }

  return (
    <ul className="space-y-3">
      {rows.map((p) => (
        <ProposalCard key={p.id} proposal={p} onDismiss={handleDismiss} />
      ))}
    </ul>
  );
}

function ProposalCard({
  proposal,
  onDismiss,
}: {
  proposal: AdminClubProposal;
  onDismiss: (id: number) => void;
}) {
  const audioUrl = `/api/clips/${proposal.clip_id}/audio`;
  return (
    <li className="rounded-md border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-4 space-y-3">
      <header className="flex items-baseline justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate text-base font-semibold tracking-tight">
            {proposal.song_title ?? <span className="text-neutral-500">Untitled</span>}
            <span className="ml-2 text-xs font-normal text-neutral-500">
              · variant {proposal.variant_index + 1}
            </span>
          </h3>
          <p className="mt-0.5 text-xs text-neutral-500">
            owner{' '}
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              {proposal.owner.display_name}
            </span>
            {' · '}proposed by{' '}
            <span className="font-medium text-neutral-700 dark:text-neutral-300">
              {proposal.proposer.display_name}
            </span>
            {' · '}
            {new Date(proposal.created_at).toLocaleString()}
            {proposal.clip_duration_s
              ? ` · ${proposal.clip_duration_s.toFixed(1)}s`
              : ''}
          </p>
        </div>
        <button
          type="button"
          onClick={() => onDismiss(proposal.id)}
          className="text-xs text-neutral-500 hover:text-red-600 dark:hover:text-red-400 underline shrink-0"
        >
          Dismiss
        </button>
      </header>

      {proposal.note && (
        <p className="rounded-md bg-neutral-100 dark:bg-neutral-800 px-3 py-2 text-sm text-neutral-700 dark:text-neutral-300">
          {proposal.note}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        <audio controls preload="none" src={audioUrl} className="h-10 flex-1 min-w-[200px]" />
        <a
          href={audioUrl}
          download={`${slugify(proposal.song_title ?? 'melodie')}-${proposal.clip_id.slice(0, 8)}.mp3`}
          className="text-neutral-700 dark:text-neutral-300 underline"
        >
          Download MP3
        </a>
      </div>
    </li>
  );
}

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '')
    .slice(0, 60);
}
