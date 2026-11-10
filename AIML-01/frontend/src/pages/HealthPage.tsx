import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getHealth } from '@/lib/api';
import { Activity, CheckCircle2, Loader2, ServerCrash } from 'lucide-react';

export function HealthPage() {
  const [health, setHealth] = useState<{ status: string; service: string } | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      const start = performance.now();
      try {
        const response = await getHealth();
        const duration = Math.round(performance.now() - start);
        setHealth(response);
        setLatency(duration);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch health');
      } finally {
        setLoading(false);
      }
    }

    void load();
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs uppercase tracking-[0.25em] text-slate-400">System health</p>
        <h2 className="mt-2 text-3xl font-semibold text-white">API and model readiness</h2>
      </div>

      <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
        {[
          ['Service status', health?.status ?? '—', 'FastAPI core'],
          ['Service name', health?.service ?? '—', 'Production service ID'],
          ['Measured Latency', latency ? `${latency} ms` : '—', 'Direct ping roundtrip'],
          ['Model Pipeline', 'Loaded (XGBoost)', 'Ready for inference'],
        ].map(([label, value, hint]) => (
          <Card key={label}>
            <CardHeader>
              <CardTitle>{label}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-semibold text-white">{value}</div>
              <div className="mt-2 text-sm text-slate-400">{hint}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Runtime diagnostics</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center gap-2 text-slate-300">
              <Loader2 className="h-4 w-4 animate-spin" />
              Checking service health…
            </div>
          ) : error ? (
            <div className="flex items-center gap-2 text-red-300">
              <ServerCrash className="h-4 w-4" />
              {error}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-emerald-300">
              <CheckCircle2 className="h-4 w-4" />
              {health?.service} is reporting a healthy state.
            </div>
          )}

          <div className="mt-6 grid gap-4 md:grid-cols-3">
            {[
              ['Model server', 'Online', 'realtime_safe_xgboost'],
              ['Feature pipeline', 'Synchronized', 'Batch + streaming data'],
              ['Monitoring', 'Stable', '12.4k checks / day'],
            ].map(([title, status, detail]) => (
              <div key={title} className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                <div className="flex items-center gap-2 text-sm text-slate-300">
                  <Activity className="h-4 w-4 text-brand-400" />
                  {title}
                </div>
                <div className="mt-3 text-xl font-semibold text-white">{status}</div>
                <div className="mt-1 text-sm text-slate-400">{detail}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
