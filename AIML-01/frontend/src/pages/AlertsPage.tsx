import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getCases, updateCaseStatus, type CaseItem, type CasesResponse } from '@/lib/api';
import { AlertTriangle, CheckCircle2, Loader2, RefreshCw, ShieldAlert, ShieldCheck } from 'lucide-react';

export function AlertsPage() {
  const [data, setData] = useState<CasesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  async function loadCases() {
    setLoading(true);
    setError(null);
    try {
      const response = await getCases();
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cases');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadCases();
  }, []);

  async function handleStatusChange(transactionId: string, newStatus: string) {
    setActionLoading(transactionId);
    try {
      await updateCaseStatus(transactionId, {
        case_status: newStatus,
        analyst_feedback: `Status changed to ${newStatus} by analyst console.`,
      });
      await loadCases();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to update case');
    } finally {
      setActionLoading(null);
    }
  }

  const cases = data?.cases ?? [];

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-slate-400">Fraud alerts</p>
          <h2 className="mt-2 text-3xl font-semibold text-white">Managed analyst queue</h2>
        </div>
        <button
          onClick={() => void loadCases()}
          className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/80 px-3 py-2 text-sm text-slate-300 hover:bg-slate-700"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh alerts
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {[
          ['Open / Active alerts', data?.open_count ?? 0, 'Requires investigation', 'text-amber-400'],
          ['Resolved cases', data?.resolved_count ?? 0, 'Reviewed & finalized', 'text-emerald-400'],
          ['Total flagged events', data?.total_count ?? 0, 'Audit trail logged', 'text-cyan-400'],
        ].map(([label, value, detail, tone]) => (
          <Card key={label as string}>
            <CardHeader>
              <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{label}</div>
            </CardHeader>
            <CardContent>
              <div className={`text-3xl font-semibold ${tone}`}>{value}</div>
              <p className="mt-1 text-xs text-slate-400">{detail}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center justify-center p-12 text-slate-400">
          <Loader2 className="mr-2 h-6 w-6 animate-spin" /> Loading real alerts from FastAPI /cases...
        </div>
      ) : error ? (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-red-300">
          {error}
        </div>
      ) : cases.length === 0 ? (
        <Card>
          <CardContent className="p-8 text-center text-slate-400">
            <CheckCircle2 className="mx-auto mb-2 text-emerald-400" size={32} />
            No active fraud alerts in the queue.
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {cases.map((c) => {
            const isHigh = c.risk_score >= 80;
            const isMedium = c.risk_score >= 40 && c.risk_score < 80;
            const badgeClass = isHigh
              ? 'bg-red-500/10 text-red-400 border border-red-500/20'
              : isMedium
              ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
              : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';

            return (
              <Card key={c.transaction_id} className="flex flex-col justify-between border border-slate-800">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs text-slate-400">{c.transaction_id}</span>
                    <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${badgeClass}`}>
                      {c.case_status}
                    </span>
                  </div>
                  <CardTitle className="mt-2 text-lg font-semibold text-white">
                    {c.transaction_details?.type ?? 'TRANSACTION'} — ${Number(c.transaction_details?.amount ?? 0).toLocaleString()}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between rounded-lg bg-slate-950/60 p-2.5 text-sm">
                    <span className="text-slate-400">Risk Score</span>
                    <span className="font-bold text-white">{c.risk_score}%</span>
                  </div>

                  <div className="text-xs text-slate-300">
                    <div>
                      <span className="text-slate-500">Origin: </span>
                      <span className="font-mono text-slate-300">{c.transaction_details?.nameOrig ?? 'N/A'}</span>
                    </div>
                    <div className="mt-1">
                      <span className="text-slate-500">Destination: </span>
                      <span className="font-mono text-slate-300">{c.transaction_details?.nameDest ?? 'N/A'}</span>
                    </div>
                  </div>

                  <p className="rounded-lg border border-slate-800/80 bg-slate-900/40 p-2 text-xs italic text-slate-300">
                    "{c.recommended_action}"
                  </p>

                  {c.analyst_feedback && (
                    <p className="text-xs text-slate-400">
                      <span className="font-semibold text-slate-300">Note:</span> {c.analyst_feedback}
                    </p>
                  )}

                  <div className="flex items-center gap-2 border-t border-slate-800 pt-3">
                    {c.case_status !== 'Resolved' ? (
                      <>
                        <button
                          disabled={actionLoading === c.transaction_id}
                          onClick={() => void handleStatusChange(c.transaction_id, 'Resolved')}
                          className="flex-1 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1.5 text-xs font-medium text-emerald-300 hover:bg-emerald-500/20"
                        >
                          Resolve
                        </button>
                        <button
                          disabled={actionLoading === c.transaction_id}
                          onClick={() => void handleStatusChange(c.transaction_id, 'Escalated')}
                          className="flex-1 rounded-lg border border-red-500/30 bg-red-500/10 px-2.5 py-1.5 text-xs font-medium text-red-300 hover:bg-red-500/20"
                        >
                          Escalate
                        </button>
                      </>
                    ) : (
                      <button
                        disabled={actionLoading === c.transaction_id}
                        onClick={() => void handleStatusChange(c.transaction_id, 'Under Review')}
                        className="flex-1 rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1.5 text-xs text-slate-300 hover:bg-slate-700"
                      >
                        Re-open case
                      </button>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
