import { useCallback, useEffect, useState } from 'react';
import {
  createInvite,
  fetchHealth,
  fetchInvites,
  fetchQuotas,
  resetAllQuotas,
  resetUserQuota,
  setSunoCookie,
  type Health,
  type Invite,
  type QuotaRow,
} from '../lib/api';

export default function AdminPanel() {
  return (
    <div className="space-y-6">
      <SunoSection />
      <QuotasSection />
      <InvitesSection />
    </div>
  );
}

function SunoSection() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [cookie, setCookie] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitOk, setSubmitOk] = useState(false);

  const refresh = useCallback(() => {
    fetchHealth()
      .then((h) => {
        setHealth(h);
        setHealthError(null);
      })
      .catch((e: unknown) => {
        setHealthError(e instanceof Error ? e.message : 'Failed to load health');
      });
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, [refresh]);

  const onSubmit = async (e: React.SubmitEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!cookie.trim()) return;
    setSubmitting(true);
    setSubmitError(null);
    setSubmitOk(false);
    try {
      await setSunoCookie(cookie.trim());
      setSubmitOk(true);
      setCookie('');
      refresh();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Submission failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="rounded-md border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 space-y-4">
      <header>
        <h2 className="text-lg font-semibold tracking-tight">Suno session</h2>
        <p className="mt-1 text-sm text-neutral-500">
          One Clerk cookie is shared across every Melodie user. Re-up here when the
          health-check loop pings you on Telegram.
        </p>
      </header>

      <div className="rounded-md border border-neutral-200 dark:border-neutral-800 px-4 py-3 text-sm">
        {healthError ? (
          <span className="text-red-600 dark:text-red-400">{healthError}</span>
        ) : !health ? (
          <span className="text-neutral-500">Loading…</span>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            <Cell label="Status">
              <HealthBadge status={health.status} />
            </Cell>
            <Cell label="Last check">
              {health.last_check
                ? new Date(health.last_check).toLocaleString()
                : '—'}
            </Cell>
            <Cell label="JWT in DB">{health.has_jwt ? 'yes' : 'no'}</Cell>
            <Cell label="Clerk cookie in DB">
              {health.has_clerk_cookie ? 'yes' : 'no'}
            </Cell>
          </div>
        )}
      </div>

      <form onSubmit={onSubmit} className="space-y-3">
        <label className="block">
          <span className="text-sm font-medium">New Clerk cookie</span>
          <textarea
            value={cookie}
            onChange={(e) => setCookie(e.target.value)}
            rows={3}
            placeholder="Paste the __client cookie value from auth.suno.com"
            className="mt-1 w-full rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm font-mono"
          />
        </label>
        {submitError && (
          <p
            role="alert"
            className="rounded-md border border-red-300 bg-red-50 dark:border-red-900 dark:bg-red-950/40 px-3 py-2 text-sm text-red-700 dark:text-red-300"
          >
            {submitError}
          </p>
        )}
        {submitOk && (
          <p className="rounded-md border border-emerald-300 bg-emerald-50 dark:border-emerald-900 dark:bg-emerald-950/40 px-3 py-2 text-sm text-emerald-800 dark:text-emerald-300">
            Suno session updated.
          </p>
        )}
        <button
          type="submit"
          disabled={submitting || !cookie.trim()}
          className="rounded-md bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 px-4 py-2 text-sm font-medium hover:opacity-90 disabled:opacity-50"
        >
          {submitting ? 'Submitting…' : 'Submit cookie'}
        </button>
      </form>
    </section>
  );
}

function QuotasSection() {
  const [rows, setRows] = useState<QuotaRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    fetchQuotas()
      .then((rs) => {
        setRows(rs);
        setError(null);
      })
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : 'Failed to load quotas')
      )
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const onResetUser = async (id: string) => {
    setBusy(id);
    try {
      await resetUserQuota(id);
      refresh();
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Reset failed');
    } finally {
      setBusy(null);
    }
  };

  const onResetAll = async () => {
    if (!confirm('Reset today\'s quota for ALL users?')) return;
    setBusy('__all__');
    try {
      await resetAllQuotas();
      refresh();
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Reset failed');
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="rounded-md border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 space-y-4">
      <header className="flex items-baseline justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Quotas</h2>
          <p className="mt-1 text-sm text-neutral-500">
            Daily generation count per user (UTC). Admins have no cap.
          </p>
        </div>
        <button
          type="button"
          onClick={onResetAll}
          disabled={busy !== null || rows.every((r) => r.count_today === 0)}
          className="text-sm rounded-md border border-neutral-300 dark:border-neutral-700 px-2.5 py-1 hover:bg-neutral-100 dark:hover:bg-neutral-900 disabled:opacity-50"
        >
          Reset all
        </button>
      </header>

      {error && (
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
      )}

      {loading ? (
        <p className="text-sm text-neutral-500">Loading…</p>
      ) : rows.length === 0 ? (
        <p className="text-sm text-neutral-500">No users yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase text-neutral-500">
              <tr>
                <th className="py-2 pr-3 font-medium">User</th>
                <th className="py-2 pr-3 font-medium">Role</th>
                <th className="py-2 pr-3 font-medium">Today</th>
                <th className="py-2 pr-3 font-medium"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-200 dark:divide-neutral-800">
              {rows.map((r) => {
                const atCap = r.cap !== null && r.count_today >= r.cap;
                return (
                  <tr key={r.user_id}>
                    <td className="py-2 pr-3">{r.display_name}</td>
                    <td className="py-2 pr-3 text-neutral-500">{r.role}</td>
                    <td className="py-2 pr-3">
                      <span className={atCap ? 'text-red-600 dark:text-red-400 font-medium' : ''}>
                        {r.count_today}
                        {r.cap !== null ? ` / ${r.cap}` : ''}
                      </span>
                    </td>
                    <td className="py-2 pr-3 text-right">
                      {r.cap !== null && r.count_today > 0 ? (
                        <button
                          type="button"
                          onClick={() => onResetUser(r.user_id)}
                          disabled={busy !== null}
                          className="text-xs rounded border border-neutral-300 dark:border-neutral-700 px-2 py-0.5 hover:bg-neutral-100 dark:hover:bg-neutral-900 disabled:opacity-50"
                        >
                          {busy === r.user_id ? '…' : 'Reset'}
                        </button>
                      ) : (
                        <span className="text-xs text-neutral-400">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function InvitesSection() {
  const [invites, setInvites] = useState<Invite[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [role, setRole] = useState<'member' | 'admin'>('member');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    fetchInvites()
      .then((rows) => {
        setInvites(rows);
        setLoadError(null);
      })
      .catch((e: unknown) => {
        setLoadError(e instanceof Error ? e.message : 'Failed to load invites');
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const onCreate = async () => {
    setCreating(true);
    setCreateError(null);
    try {
      const inv = await createInvite(role);
      setInvites((prev) => [inv, ...prev]);
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : 'Create failed');
    } finally {
      setCreating(false);
    }
  };

  return (
    <section className="rounded-md border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 space-y-4">
      <header className="flex items-baseline justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Invites</h2>
          <p className="mt-1 text-sm text-neutral-500">
            Invite codes are single-use. Members get default quota; admins skip
            quota and see this panel.
          </p>
        </div>
      </header>

      <div className="flex items-center gap-2">
        <select
          value={role}
          onChange={(e) => setRole(e.target.value as 'member' | 'admin')}
          className="rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm"
        >
          <option value="member">member</option>
          <option value="admin">admin</option>
        </select>
        <button
          type="button"
          onClick={onCreate}
          disabled={creating}
          className="rounded-md bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 px-3 py-2 text-sm font-medium hover:opacity-90 disabled:opacity-50"
        >
          {creating ? 'Creating…' : 'Create invite'}
        </button>
      </div>

      {createError && (
        <p
          role="alert"
          className="rounded-md border border-red-300 bg-red-50 dark:border-red-900 dark:bg-red-950/40 px-3 py-2 text-sm text-red-700 dark:text-red-300"
        >
          {createError}
        </p>
      )}

      {loadError ? (
        <p className="text-sm text-red-600 dark:text-red-400">{loadError}</p>
      ) : loading ? (
        <p className="text-sm text-neutral-500">Loading…</p>
      ) : invites.length === 0 ? (
        <p className="text-sm text-neutral-500">No invites yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase text-neutral-500">
              <tr>
                <th className="py-2 pr-3 font-medium">Code</th>
                <th className="py-2 pr-3 font-medium">Role</th>
                <th className="py-2 pr-3 font-medium">Status</th>
                <th className="py-2 pr-3 font-medium">Created</th>
                <th className="py-2 pr-3 font-medium">By</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-200 dark:divide-neutral-800">
              {invites.map((inv) => (
                <tr key={inv.code}>
                  <td className="py-2 pr-3 font-mono text-xs">
                    <CodeCell code={inv.code} />
                  </td>
                  <td className="py-2 pr-3">{inv.role}</td>
                  <td className="py-2 pr-3">
                    {inv.used_by ? (
                      <span className="text-neutral-500">used by {inv.used_by}</span>
                    ) : (
                      <span className="text-emerald-700 dark:text-emerald-400">unused</span>
                    )}
                  </td>
                  <td className="py-2 pr-3 text-neutral-500">
                    {new Date(inv.created_at).toLocaleString()}
                  </td>
                  <td className="py-2 pr-3 text-neutral-500">
                    {inv.created_by ?? <em>system</em>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function Cell({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs uppercase text-neutral-500">{label}</div>
      <div className="mt-0.5">{children}</div>
    </div>
  );
}

function HealthBadge({ status }: { status: string }) {
  const tone =
    status === 'ok'
      ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
      : status === 'expired'
        ? 'bg-red-100 text-red-800 dark:bg-red-950/60 dark:text-red-300'
        : status === 'missing'
          ? 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300'
          : 'bg-neutral-200 text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300';
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${tone}`}>
      {status}
    </span>
  );
}

function CodeCell({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Some browsers refuse without user gesture or in non-secure contexts;
      // fall through silently — the user can select-and-copy manually.
    }
  };
  return (
    <button
      type="button"
      onClick={onCopy}
      title="Click to copy"
      className="rounded border border-transparent hover:border-neutral-300 dark:hover:border-neutral-700 px-1.5 py-0.5 hover:bg-neutral-100 dark:hover:bg-neutral-800"
    >
      <span className="select-all">{code}</span>
      {copied && <span className="ml-2 text-xs text-emerald-700 dark:text-emerald-400">copied</span>}
    </button>
  );
}
