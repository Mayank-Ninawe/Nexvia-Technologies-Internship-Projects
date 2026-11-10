export type HealthResponse = {
  status: string;
  service: string;
};

export type TransactionRequest = {
  transaction_id: string;
  step: number;
  type: string;
  amount: number;
  nameOrig: string;
  oldbalanceOrg: number;
  newbalanceOrig: number;
  nameDest: string;
  oldbalanceDest: number;
  newbalanceDest: number;
};

export type PredictionResponse = {
  transaction_id: string;
  fraud_probability: number;
  risk_score: number;
  is_alert: boolean;
  model_name: string;
  threshold: number;
  recommended_action: string;
  confidence?: number;
};

export type CaseItem = {
  transaction_id: string;
  fraud_probability: number;
  risk_score: number;
  is_alert: boolean;
  recommended_action: string;
  transaction_details: {
    transaction_id: string;
    step: number;
    type: string;
    amount: number;
    nameOrig: string;
    oldbalanceOrg: number;
    newbalanceOrig: number;
    nameDest: string;
    oldbalanceDest: number;
    newbalanceDest: number;
  };
  case_status: string;
  analyst_feedback?: string;
  created_at?: string;
  updated_at?: string;
};

export type CasesResponse = {
  cases: CaseItem[];
  open_count: number;
  resolved_count: number;
  total_count: number;
};

export type TopFraudAccount = {
  account_id: string;
  in_degree: number;
  out_degree: number;
  fraud_linked_transaction_count: number;
  fraud_linked_ratio: number;
};

export type GraphSummaryResponse = {
  analysis_window: {
    actual_transaction_rows: number;
    fraud_transaction_rows: number;
    max_step: number;
    min_step: number;
  };
  graph_metrics: {
    node_count: number;
    edge_count: number;
    density: number;
    is_directed: boolean;
    largest_weakly_connected_component_size: number;
    weakly_connected_component_count: number;
    fraud_linked_relationship_count: number;
  };
  top_fraud_linked_accounts: TopFraudAccount[];
};

export type OverviewMetricsResponse = {
  total_transactions: number;
  fraud_transactions: number;
  fraud_rate: number;
  model_average_precision: number;
  threshold: number;
  top_fraud_linked_accounts: TopFraudAccount[];
  trend_data: { day: string; risk: number; alerts: number }[];
};

export type PerformanceMetricsResponse = {
  model_name: string;
  precision: number;
  recall: number;
  average_precision: number;
  threshold: number;
  tpr: number;
  confusion_matrix: {
    true_positives: number;
    false_positives: number;
    false_negatives: number;
    true_negatives: number;
  };
  shadow_model: {
    model_name: string;
    average_precision: number;
    threshold: number;
  };
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api';
const API_KEY = import.meta.env.VITE_API_KEY;

async function fetchJson<T>(url: string, options?: RequestInit) {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
      ...(options?.headers ?? {}),
    },
    ...options,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || 'Request failed');
  }

  return (await response.json()) as T;
}

export async function getHealth() {
  return fetchJson<HealthResponse>('/health');
}

export async function predictTransaction(payload: TransactionRequest) {
  return fetchJson<PredictionResponse>('/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function getCases() {
  return fetchJson<CasesResponse>('/cases');
}

export async function updateCaseStatus(transaction_id: string, update: { case_status: string; analyst_feedback?: string }) {
  return fetchJson<CaseItem>(`/cases/${transaction_id}`, {
    method: 'PATCH',
    body: JSON.stringify(update),
  });
}

export async function getGraphSummary() {
  return fetchJson<GraphSummaryResponse>('/graph/summary');
}

export async function getOverviewMetrics() {
  return fetchJson<OverviewMetricsResponse>('/metrics/overview');
}

export async function getPerformanceMetrics() {
  return fetchJson<PerformanceMetricsResponse>('/metrics/performance');
}
