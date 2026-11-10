import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getOverviewMetrics, type OverviewMetricsResponse } from '@/lib/api';
import { Activity, ArrowUpRight, CircleDollarSign, Loader2, ShieldAlert, TrendingUp } from 'lucide-react';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export function OverviewPage() {
  const [data, setData] = useState<OverviewMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const metrics = await getOverviewMetrics();
        setData(metrics);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch overview metrics');
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, []);

  const totalTransactions = data?.total_transactions ?? 200000;
  const fraudTransactions = data?.fraud_transactions ?? 147;
  const fraudRate = data?.fraud_rate ?? 0.07;
  const modelAveragePrecision = data?.model_average_precision ?? 0.9926;
  const threshold = data?.threshold ?? 0.00009248;

  const metricCards = [
    { label: 'Transactions processed', value: new Intl.NumberFormat('en-US').format(totalTransactions), change: 'Historical window', icon: Activity, tone: 'text-brand-400' },
    { label: 'Fraud rate', value: `${fraudRate}%`, change: 'Model baseline', icon: ShieldAlert, tone: 'text-amber-400' },
    { label: 'Alerts flagged', value: new Intl.NumberFormat('en-US').format(fraudTransactions), change: 'Verified fraud events', icon: CircleDollarSign, tone: 'text-emerald-400' },
    { label: 'Model confidence', value: `${(modelAveragePrecision * 100).toFixed(1)}%`, change: `Thresh ${Number(threshold).toExponential(3)}`, icon: TrendingUp, tone: 'text-cyan-400' },
  ];

  const priorityAlerts = (data?.top_fraud_linked_accounts ?? []).slice(0, 3).map((account) => ({
    name: `Account ${account.account_id}`,
    risk: `${(account.fraud_linked_ratio * 100).toFixed(1)}%`,
    detail: `${account.in_degree} inbound links · ${account.fraud_linked_transaction_count} fraud-linked txns`,
  }));

  const trendData = data?.trend_data ?? [
    { day: 'Mon', risk: 8 },
    { day: 'Tue', risk: 12 },
    { day: 'Wed', risk: 10 },
    { day: 'Thu', risk: 24 },
    { day: 'Fri', risk: 31 },
    { day: 'Sat', risk: 27 },
    { day: 'Sun', risk: 36 },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.25em] text-slate-400">Overview</p>
          <h2 className="mt-2 text-3xl font-semibold text-white">Financial risk command center</h2>
        </div>
        <div className="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-sm text-emerald-300">
          <span className="h-2 w-2 rounded-full bg-emerald-400" />
          Active detection backend
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center p-12 text-slate-400">
          <Loader2 className="mr-2 h-6 w-6 animate-spin" /> Loading metrics from FastAPI...
        </div>
      ) : error ? (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-red-300">
          {error}
        </div>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {metricCards.map(({ label, value, change, icon: Icon, tone }) => (
              <Card key={label} className="overflow-hidden">
                <CardContent className="space-y-4 p-0">
                  <div className="flex items-start justify-between px-4 pt-4">
                    <div>
                      <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{label}</div>
                      <div className="mt-3 text-3xl font-semibold text-white">{value}</div>
                    </div>
                    <div className={`rounded-xl bg-slate-800 p-2 ${tone}`}>
                      <Icon size={20} />
                    </div>
                  </div>
                  <div className="border-t border-slate-800 px-4 py-2.5 text-xs text-slate-400">
                    <span className="text-brand-300">{change}</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="grid gap-6 xl:grid-cols-[1.5fr_0.9fr]">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Risk distribution trend</CardTitle>
                  <span className="text-xs text-slate-400">Real-time dynamic feed</span>
                </div>
              </CardHeader>
              <CardContent className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={trendData} margin={{ left: -16, right: 8 }}>
                    <defs>
                      <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.6} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="day" tickLine={false} axisLine={false} stroke="#64748b" />
                    <YAxis tickLine={false} axisLine={false} stroke="#64748b" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        border: '1px solid #1e293b',
                        borderRadius: '0.75rem',
                        boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)',
                      }}
                    />
                    <Area type="monotone" dataKey="risk" stroke="#60a5fa" strokeWidth={2} fillOpacity={1} fill="url(#riskGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Priority fraud rings</CardTitle>
                  <ArrowUpRight className="text-slate-500" size={16} />
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {priorityAlerts.map((alert) => (
                  <div key={alert.name} className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                    <div>
                      <div className="font-medium text-white">{alert.name}</div>
                      <div className="mt-1 text-xs text-slate-400">{alert.detail}</div>
                    </div>
                    <div className="rounded-lg bg-amber-500/10 px-2.5 py-1 text-xs font-semibold text-amber-300">
                      {alert.risk}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
