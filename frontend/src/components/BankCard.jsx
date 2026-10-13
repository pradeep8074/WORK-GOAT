import React from 'react';
import { Building2, ShieldCheck, Activity, Database } from 'lucide-react';
import StatusBadge from './StatusBadge';

export default function BankCard({ bank }) {
  return (
    <div className="bg-[#0e1626] border border-slate-800 rounded-xl p-5 flex flex-col justify-between relative transition-all hover:border-blue-900/50 hover:bg-[#101b30]">
      {/* Header */}
      <div>
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400">
              <Building2 className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-semibold text-white tracking-tight">{bank.short_name || bank.name}</h4>
              <p className="text-xs text-slate-400 font-mono">{bank.code || bank.name}</p>
            </div>
          </div>
          <StatusBadge status={bank.status} />
        </div>

        {/* Bank Simulation Note */}
        <div className="mb-4">
          <span className="inline-block text-[11px] font-mono px-2 py-0.5 rounded bg-blue-950/60 text-blue-300 border border-blue-800/40">
            {bank.environment || "Simulated bank environment"}
          </span>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-3 py-3 border-y border-slate-800/80 text-xs">
          <div>
            <span className="text-slate-400 block mb-1 flex items-center gap-1">
              <Activity className="w-3.5 h-3.5 text-slate-500" />
              Transactions
            </span>
            <span className="text-sm font-semibold text-slate-200 font-mono">
              {bank.transaction_count?.toLocaleString() || "0"}
            </span>
          </div>

          <div>
            <span className="text-slate-400 block mb-1 flex items-center gap-1">
              <Database className="w-3.5 h-3.5 text-slate-500" />
              Active Alerts
            </span>
            <span className="text-sm font-semibold text-amber-400 font-mono">
              {bank.active_alerts ?? 0}
            </span>
          </div>
        </div>
      </div>

      {/* Data Isolation Status Footer */}
      <div className="mt-4 pt-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span className="text-xs text-slate-300 font-medium">Data isolation:</span>
        </div>
        <span className="text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 rounded">
          {bank.data_isolation || "ACTIVE"}
        </span>
      </div>
    </div>
  );
}
