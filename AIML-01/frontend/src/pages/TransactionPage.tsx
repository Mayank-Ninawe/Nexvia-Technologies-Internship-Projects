import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { predictTransaction, type TransactionRequest } from '@/lib/api';
import { AlertTriangle, CheckCircle2, Loader2 } from 'lucide-react';

const initialRequest: TransactionRequest = {
  transaction_id: 'TXN-2025-0918',
  step: 6,
  type: 'CASH_OUT',
  amount: 12500,
  nameOrig: 'C1231006815',
  oldbalanceOrg: 6500,
  newbalanceOrig: 0,
  nameDest: 'M1979787155',
  oldbalanceDest: 10100,
  newbalanceDest: 24000,
};

export function TransactionPage() {
  const [payload, setPayload] = useState<TransactionRequest>(initialRequest);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateField = (field: keyof TransactionRequest, value: string) => {
    const typedValue = field === 'step' ? Number(value) : value;
    const parsed = field === 'amount' || field === 'oldbalanceOrg' || field === 'newbalanceOrig' || field === 'oldbalanceDest' || field === 'newbalanceDest'
      ? Number(value)
      : typedValue;

    setPayload((current) => ({ ...current, [field]: parsed }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await predictTransaction(payload);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction request failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs uppercase tracking-[0.25em] text-slate-400">Transaction analysis</p>
        <h2 className="mt-2 text-3xl font-semibold text-white">Evaluate a transfer in real time</h2>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <Card>
          <CardHeader>
            <CardTitle>Input transaction</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              {[
                ['transaction_id', 'Transaction ID'],
                ['step', 'Step'],
                ['type', 'Type'],
                ['amount', 'Amount'],
                ['nameOrig', 'Origin account'],
                ['oldbalanceOrg', 'Original balance'],
                ['newbalanceOrig', 'New balance after transfer'],
                ['nameDest', 'Destination account'],
                ['oldbalanceDest', 'Destination old balance'],
                ['newbalanceDest', 'Destination new balance'],
              ].map(([field, label]) => (
                <label key={field} className="space-y-2 text-sm text-slate-300">
                  <span>{label}</span>
                  <input
                    value={String(payload[field as keyof TransactionRequest])}
                    onChange={(event) => updateField(field as keyof TransactionRequest, event.target.value)}
                    className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none ring-0 transition focus:border-brand-500"
                  />
                </label>
              ))}
            </div>

            <button
              type="button"
              onClick={handleSubmit}
              disabled={loading}
              className="mt-6 inline-flex items-center gap-2 rounded-xl bg-brand-500 px-4 py-2.5 font-medium text-white transition hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
              {loading ? 'Scoring transaction...' : 'Run risk evaluation'}
            </button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Assessment result</CardTitle>
          </CardHeader>
          <CardContent>
            {error ? (
              <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-red-300">
                <div className="flex items-center gap-2 font-medium">
                  <AlertTriangle className="h-4 w-4" />
                  Evaluation failed
                </div>
                <div className="mt-2 text-sm">{error}</div>
              </div>
            ) : result ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950/50 p-3">
                  <span className="text-sm text-slate-400">Alert status</span>
                  <span className={`rounded-full px-2 py-1 text-xs font-medium ${result.is_alert ? 'bg-red-500/10 text-red-300' : 'bg-emerald-500/10 text-emerald-300'}`}>
                    {result.is_alert ? 'High risk' : 'Normal'}
                  </span>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                    <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Fraud probability</div>
                    <div className="mt-2 text-2xl font-semibold text-white">{(result.fraud_probability * 100).toFixed(2)}%</div>
                  </div>
                  <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                    <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Risk score</div>
                    <div className="mt-2 text-2xl font-semibold text-white">{result.risk_score.toFixed(2)}</div>
                  </div>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3 text-sm text-slate-200">
                  <div className="mb-2 text-xs uppercase tracking-[0.2em] text-slate-400">Model</div>
                  <div>{result.model_name}</div>
                  <div className="mt-2 text-slate-400">Threshold: {result.threshold}</div>
                  <div className="mt-2 text-slate-400">Action: {result.recommended_action}</div>
                </div>
              </div>
            ) : (
              <div className="rounded-xl border border-dashed border-slate-700 bg-slate-950/40 p-5 text-sm text-slate-400">
                Submit a transaction to see the risk assessment.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
