import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getPerformanceMetrics, type PerformanceMetricsResponse } from '@/lib/api';
import { Loader2 } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export function PerformancePage() {
  const [data, setData] = useState<PerformanceMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const perf = await getPerformanceMetrics();
        setData(perf);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch performance metrics');
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, []);

  const metrics = [
    { name: 'Precision', value: Number(data?.precision ?? 0.4553) },
    { name: 'Recall', value: Number(data?.recall ?? 1.0) },
    { name: 'Average Precision', value: Number(data?.average_precision ?? 0.9926) },
    { name: 'TPR', value: Number(data?.tpr ?? 1.0) },
  ];

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs uppercase tracking-[0.25em] text-slate-400">Model performance</p>
        <h2 className="mt-2 text-3xl font-semibold text-white">Realtime-safe model health</h2>
      </div>

      {loading ? (
        <div className="flex items-center justify-center p-12 text-slate-400">
          <Loader2 className="mr-2 h-6 w-6 animate-spin" /> Loading metrics from FastAPI /metrics/performance...
        </div>
      ) : error ? (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-red-300">
          {error}
        </div>
      ) : (
        <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <Card>
            <CardHeader>
              <CardTitle>Evaluation metrics</CardTitle>
            </CardHeader>
            <CardContent className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={metrics} margin={{ left: -8, right: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" tickLine={false} axisLine={false} stroke="#64748b" />
                  <YAxis tickLine={false} axisLine={false} stroke="#64748b" domain={[0, 1.1]} />
                  <Tooltip
                    formatter={(value) => Number(value).toFixed(4)}
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '0.75rem' }}
                  />
                  <Bar dataKey="value" fill="#5f9efa" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Model profile & confusion matrix</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {[
                ['Deployed Model', data?.model_name ?? 'realtime_safe_xgboost'],
                ['Decision Threshold', Number(data?.threshold ?? 0.00009248).toExponential(4)],
                ['True Positives (TP)', data?.confusion_matrix?.true_positives.toLocaleString() ?? '846'],
                ['False Positives (FP)', data?.confusion_matrix?.false_positives.toLocaleString() ?? '1,012'],
                ['True Negatives (TN)', data?.confusion_matrix?.true_negatives.toLocaleString() ?? '66,990'],
                ['False Negatives (FN)', data?.confusion_matrix?.false_negatives.toLocaleString() ?? '0'],
                ['Shadow Model (Upper Bound)', `${data?.shadow_model?.model_name} (PR-AUC: ${data?.shadow_model?.average_precision})`],
              ].map(([label, value]) => (
                <div key={label} className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{label}</div>
                  <div className="mt-1 text-sm font-medium text-white">{value}</div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
