import { Activity, BarChart3, BellRing, BrainCircuit, LayoutDashboard, ShieldAlert, ShieldCheck } from 'lucide-react';
import { NavLink, Outlet } from 'react-router-dom';

const navItems = [
  { path: '/', label: 'Overview', icon: LayoutDashboard },
  { path: '/transaction', label: 'Analyze Transaction', icon: ShieldCheck },
  { path: '/alerts', label: 'Fraud Alerts', icon: BellRing },
  { path: '/graph', label: 'Graph Analysis', icon: Activity },
  { path: '/performance', label: 'Model Performance', icon: BarChart3 },
  { path: '/health', label: 'System Health', icon: BrainCircuit },
];

export function Shell() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-[1600px]">
        <aside className="hidden w-72 shrink-0 border-r border-slate-800 bg-slate-950/80 p-5 lg:block">
          <div className="mb-8 flex items-center gap-3">
            <div className="rounded-xl bg-brand-500/15 p-2 text-brand-400">
              <ShieldAlert size={20} />
            </div>
            <div>
              <div className="text-sm uppercase tracking-[0.2em] text-slate-400">Sentinel</div>
              <div className="text-xl font-semibold text-white">Graph</div>
            </div>
          </div>

          <nav className="space-y-2">
            {navItems.map(({ path, label, icon: Icon }) => (
              <NavLink
                key={path}
                to={path}
                end={path === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition ${
                    isActive
                      ? 'bg-slate-800 text-white shadow-soft'
                      : 'text-slate-300 hover:bg-slate-800/50 hover:text-white'
                  }`
                }
              >
                <Icon size={16} />
                {label}
              </NavLink>
            ))}
          </nav>

          <div className="mt-10 rounded-2xl border border-emerald-500/25 bg-emerald-500/5 p-4">
            <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-emerald-300">
              <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" />
              System status
            </div>
            <div className="text-2xl font-semibold text-white">Healthy</div>
            <div className="mt-1 text-sm text-slate-300">Realtime-safe model active</div>
          </div>
        </aside>

        <main className="flex-1">
          <header className="border-b border-slate-800 bg-slate-950/70 px-4 py-4 backdrop-blur-sm sm:px-6">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-[0.25em] text-brand-300">Fraud monitoring</p>
                <h1 className="mt-1 text-xl font-semibold text-white">Enterprise Command Center</h1>
              </div>
              <div className="flex items-center gap-3">
                <div className="hidden rounded-full border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-300 md:block">
                  UTC · 09:42:18
                </div>
                <div className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-300">
                  Live monitoring
                </div>
              </div>
            </div>
          </header>

          <div className="p-4 sm:p-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
