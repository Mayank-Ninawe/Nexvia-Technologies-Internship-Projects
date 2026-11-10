import { Navigate, Route, Routes } from 'react-router-dom';
import { Shell } from '@/components/layout/Shell';
import { OverviewPage } from '@/pages/OverviewPage';
import { TransactionPage } from '@/pages/TransactionPage';
import { AlertsPage } from '@/pages/AlertsPage';
import { GraphPage } from '@/pages/GraphPage';
import { PerformancePage } from '@/pages/PerformancePage';
import { HealthPage } from '@/pages/HealthPage';

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/transaction" element={<TransactionPage />} />
        <Route path="/alerts" element={<AlertsPage />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="/performance" element={<PerformancePage />} />
        <Route path="/health" element={<HealthPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
