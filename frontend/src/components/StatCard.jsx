import React from 'react';

export default function StatCard({ title, value, subtitle, icon: Icon, badge, alertLevel }) {
  return (
    <div className="bg-[#0e1626] border border-slate-800/80 rounded-xl p-5 relative overflow-hidden transition-all hover:border-slate-700">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">{title}</p>
          <h3 className="text-2xl font-bold tracking-tight text-white font-mono">{value}</h3>
        </div>
        {Icon && (
          <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50 text-blue-400">
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>

      <div className="mt-4 flex items-center justify-between text-xs text-slate-400 pt-3 border-t border-slate-800/60">
        <span className="truncate">{subtitle}</span>
        {badge && (
          <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700/60">
            {badge}
          </span>
        )}
      </div>
    </div>
  );
}
