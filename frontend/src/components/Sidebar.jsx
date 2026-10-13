import React from 'react';
import { 
  LayoutDashboard, 
  ArrowLeftRight, 
  AlertTriangle, 
  Search, 
  Network, 
  Lock, 
  Cpu,
  Layers,
  BarChart3
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab }) {
  const primaryNav = [
    { id: 'overview', name: 'Overview', icon: LayoutDashboard, stage: 'Stage 1' },
    { id: 'transactions', name: 'Transactions', icon: ArrowLeftRight, stage: 'Stage 1' },
    { id: 'alerts', name: 'Alerts', icon: AlertTriangle, stage: 'Stage 1' },
  ];

  const stage2Nav = [
    { id: 'network', name: 'Network Analysis', icon: Network, badge: 'Stage 2B', live: true },
    { id: 'investigation', name: 'Investigation', icon: Search, badge: 'Stage 2A', live: true },
    { id: 'privacy', name: 'Privacy (PSI)', icon: Lock, badge: 'Stage 2C-1', live: true },
    { id: 'federated', name: 'Federated (HE)', icon: Cpu, badge: 'Stage 2C-2', live: true },
    { id: 'metrics', name: 'Performance Metrics', icon: BarChart3, badge: 'Stage 2D', live: true },
  ];

  const roadmapNav = [];


  return (
    <aside className="w-64 bg-[#080d16] border-r border-slate-800/80 flex flex-col justify-between shrink-0 h-[calc(100vh-4rem)] sticky top-16 select-none">
      <div className="p-4 space-y-5">
        {/* Core modules (Stage 1) */}
        <div>
          <span className="px-3 text-[11px] font-semibold font-mono uppercase tracking-wider text-slate-400 block mb-2">
            Operations (Stage 1)
          </span>
          <nav className="space-y-1">
            {primaryNav.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 border border-transparent'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-blue-400' : 'text-slate-500'}`} />
                  <span>{item.name}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Intelligence Engine (Stage 2B Active) */}
        <div>
          <span className="px-3 text-[11px] font-semibold font-mono uppercase tracking-wider text-slate-400 block mb-2 flex items-center justify-between">
            <span>Detection Engine</span>
            <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20 font-bold">Live</span>
          </span>
          <nav className="space-y-1">
            {stage2Nav.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-blue-600/20 text-white border border-blue-500/40 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Icon className={`w-4 h-4 ${isActive ? 'text-blue-400' : 'text-slate-400'}`} />
                    <span className="font-semibold">{item.name}</span>
                  </div>
                  <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20 font-bold">
                    {item.badge}
                  </span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Stage 2 Future modules */}
        <div>
          <span className="px-3 text-[11px] font-semibold font-mono uppercase tracking-wider text-slate-500 block mb-2">
            Future Architecture
          </span>
          <nav className="space-y-1">
            {roadmapNav.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-slate-800/80 text-white border border-slate-700'
                      : 'text-slate-500 hover:text-slate-300 hover:bg-slate-900/40 border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Icon className="w-4 h-4 text-slate-600" />
                    <span>{item.name}</span>
                  </div>
                  <span className="text-[10px] font-mono text-slate-500 bg-slate-800/90 px-1.5 py-0.5 rounded border border-slate-700/60">
                    Roadmap
                  </span>
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Footer Info Box */}
      <div className="p-4 border-t border-slate-800/70">
        <div className="p-3 bg-slate-900/50 rounded-lg border border-slate-800 text-xs space-y-1">
          <div className="flex items-center gap-1.5 text-slate-300 font-semibold font-mono text-[11px]">
            <Layers className="w-3.5 h-3.5 text-emerald-400" />
            <span>Stage 2C-1 Active</span>
          </div>
          <p className="text-slate-400 text-[11px] leading-tight">
            Commutative DH-PSI matching, GraphSAGE GNN node classifier, NetworkX graph & forensic rule engine active.
          </p>
        </div>
      </div>
    </aside>
  );
}
