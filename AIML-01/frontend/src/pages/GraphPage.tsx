import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getGraphSummary, type GraphSummaryResponse } from '@/lib/api';
import { Loader2 } from 'lucide-react';

export function GraphPage() {
  const [data, setData] = useState<GraphSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const summary = await getGraphSummary();
        setData(summary);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load graph summary');
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, []);

  const metrics = data?.graph_metrics;
  const topAccounts = data?.top_fraud_linked_accounts ?? [];

  // Generate visual layout for real top fraud accounts
  const radius = 32;
  const centerX = 50;
  const centerY = 50;
  const visualNodes = topAccounts.slice(0, 7).map((acc, index, arr) => {
    const angle = (index / (arr.length || 1)) * 2 * Math.PI;
    return {
      id: acc.account_id,
      inDegree: acc.in_degree,
      fraudCount: acc.fraud_linked_transaction_count,
      ratio: acc.fraud_linked_ratio,
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    };
  });

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs uppercase tracking-[0.25em] text-slate-400">Network analysis</p>
        <h2 className="mt-2 text-3xl font-semibold text-white">Fraud ring relationship graph</h2>
      </div>

      {loading ? (
        <div className="flex items-center justify-center p-12 text-slate-400">
          <Loader2 className="mr-2 h-6 w-6 animate-spin" /> Loading entity network from FastAPI /graph/summary...
        </div>
      ) : error ? (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-red-300">
          {error}
        </div>
      ) : (
        <div className="grid gap-6 xl:grid-cols-[1.4fr_0.6fr]">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Top fraud ring clusters</CardTitle>
                <span className="text-xs text-slate-400">Click node to inspect</span>
              </div>
            </CardHeader>
            <CardContent>
              <div className="relative h-[420px] overflow-hidden rounded-2xl border border-slate-800 bg-[radial-gradient(circle_at_center,_rgba(59,130,246,0.15),_transparent_65%)]">
                <svg viewBox="0 0 100 100" className="h-full w-full">
                  {/* Center hub */}
                  <circle cx={centerX} cy={centerY} r="6" fill="#1e3a8a" stroke="#60a5fa" strokeWidth="1" opacity="0.8" />
                  <text x={centerX} y={centerY + 1.5} textAnchor="middle" fill="#93c5fd" fontSize="2.8" fontWeight="bold">
                    HUB
                  </text>

                  {/* Edges from hub to account clusters */}
                  {visualNodes.map((node) => (
                    <line
                      key={`edge-${node.id}`}
                      x1={centerX}
                      y1={centerY}
                      x2={node.x}
                      y2={node.y}
                      stroke={selectedAccount === node.id ? '#f43f5e' : '#3b82f6'}
                      strokeWidth={selectedAccount === node.id ? '1.5' : '0.8'}
                      strokeDasharray={selectedAccount === node.id ? 'none' : '2,1'}
                      opacity="0.8"
                    />
                  ))}

                  {/* Inter-cluster connections */}
                  {visualNodes.map((node, i) => {
                    const next = visualNodes[(i + 1) % visualNodes.length];
                    return (
                      <line
                        key={`ring-${node.id}`}
                        x1={node.x}
                        y1={node.y}
                        x2={next.x}
                        y2={next.y}
                        stroke="#1d4ed8"
                        strokeWidth="0.5"
                        opacity="0.4"
                      />
                    );
                  })}

                  {/* Nodes */}
                  {visualNodes.map((node) => {
                    const isSelected = selectedAccount === node.id;
                    return (
                      <g
                        key={node.id}
                        transform={`translate(${node.x}, ${node.y})`}
                        className="cursor-pointer transition-transform hover:scale-110"
                        onClick={() => setSelectedAccount(node.id)}
                      >
                        <circle
                          r={isSelected ? '6.5' : '5'}
                          fill={isSelected ? '#e11d48' : '#1e40af'}
                          stroke={isSelected ? '#fda4af' : '#60a5fa'}
                          strokeWidth="1"
                        />
                        <text x="0" y="1" textAnchor="middle" fill="#ffffff" fontSize="2.2" fontWeight="bold">
                          {node.fraudCount}
                        </text>
                        <text x="0" y="8" textAnchor="middle" fill="#94a3b8" fontSize="2.2">
                          {node.id.substring(0, 7)}…
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>

              {selectedAccount && (
                <div className="mt-4 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-200">
                  <span className="font-bold">Inspecting Account:</span> {selectedAccount} — high-velocity mule destination node with multiple incoming transfers.
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Network telemetry</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                ['Total network entities', metrics?.node_count != null ? metrics.node_count.toLocaleString() : '—'],
                ['Directed edges observed', metrics?.edge_count != null ? metrics.edge_count.toLocaleString() : '—'],
                ['Fraud-linked relationships', metrics?.fraud_linked_relationship_count != null ? metrics.fraud_linked_relationship_count.toLocaleString() : '—'],
                ['Largest connected cluster', `${metrics?.largest_weakly_connected_component_size ?? 0} accounts`],
                ['Weakly connected clusters', metrics?.weakly_connected_component_count != null ? metrics.weakly_connected_component_count.toLocaleString() : '—'],
                ['Graph state', 'Directed Multigraph (Time-windowed)'],
              ].map(([label, value]) => (
                <div key={label} className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                  <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{label}</div>
                  <div className="mt-1 text-lg font-semibold text-white">{value}</div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
