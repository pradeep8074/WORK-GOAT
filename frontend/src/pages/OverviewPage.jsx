import React from 'react';
import { 
  ArrowLeftRight, 
  AlertTriangle, 
  ShieldAlert, 
  Building2, 
  Info,
  Clock,
  ArrowUpRight
} from 'lucide-react';
import StatCard from '../components/StatCard';
import BankCard from '../components/BankCard';
import StatusBadge from '../components/StatusBadge';
import LoadingState from '../components/LoadingState';
import ErrorState from '../components/ErrorState';
import { formatINR } from '../utils/currency';

export default function OverviewPage({
  banks,
  transactions,
  alerts,
  loading,
  error,
  onRetry,
  onSelectTransaction,
  onNavigateToTab
}) {
  if (loading) {
    return <LoadingState message="Connecting to simulated banking nodes..." />;
  }

  // Calculate high risk accounts from mock alerts/transactions
  const highRiskCount = banks.reduce((acc, b) => acc + (b.high_risk_accounts || 0), 0);
  const totalTxnCount = banks.reduce((acc, b) => acc + (b.transaction_count || 0), 0);
  const activeAlertsCount = alerts.length;

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Simulation Banner */}
      <div className="bg-blue-950/20 border border-blue-900/40 rounded-xl p-4 flex items-start gap-3 text-xs text-blue-300">
        <Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-white">Stage 1 Architecture Preview:</span> All transaction records, account identifiers, and risk metrics displayed are strictly <strong>synthetic prototype data</strong> for evaluation. No live banking connections or machine learning algorithms are executed in this phase.
        </div>
      </div>

      {error && (
        <ErrorState message={error} onRetry={onRetry} isFallback={banks.length > 0} />
      )}

      {/* Top Stat Cards */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold font-mono uppercase tracking-wider text-slate-400">
            System Metrics Summary
          </h2>
          <span className="text-[11px] font-mono text-slate-400">Synthetic prototype data</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Total Transactions"
            value={totalTxnCount ? totalTxnCount.toLocaleString() : "35,580"}
            subtitle="Combined mock ledger"
            icon={ArrowLeftRight}
            badge="Aggregated"
          />
          <StatCard
            title="Active Alerts"
            value={activeAlertsCount.toString()}
            subtitle="Flagged synthetic events"
            icon={AlertTriangle}
            badge="Stage 1 Mock"
          />
          <StatCard
            title="High Risk Accounts"
            value={highRiskCount.toString()}
            subtitle="Under mock surveillance"
            icon={ShieldAlert}
            badge="Simulated"
          />
          <StatCard
            title="Connected Banks"
            value={`${banks.length} / 3`}
            subtitle="Apex, Beacon, Crestline"
            icon={Building2}
            badge="Nodes Online"
          />
        </div>
      </div>

      {/* Connected Banks Overview Section */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-semibold font-mono uppercase tracking-wider text-slate-400">
              Simulated Bank Environments
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Simulated bank environment • Data isolation: ACTIVE
            </p>
          </div>
          <span className="text-[11px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
            3 Isolated Nodes
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {banks.map((bank) => (
            <BankCard key={bank.id} bank={bank} />
          ))}
        </div>
      </div>

      {/* Recent Transactions Table */}
      <div className="bg-[#0e1626] border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <div className="p-5 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-white">Recent Transactions</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Synthetic cross-bank event feed (click any row for audit breakdown)
            </p>
          </div>
          <button
            onClick={() => onNavigateToTab('transactions')}
            className="flex items-center gap-1 text-xs font-mono text-blue-400 hover:text-blue-300 transition-colors"
          >
            <span>View All</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#0a0f1c] text-slate-400 border-b border-slate-800 font-mono uppercase text-[11px]">
              <tr>
                <th className="py-3.5 px-4">Transaction ID</th>
                <th className="py-3.5 px-4">Bank</th>
                <th className="py-3.5 px-4">Sender</th>
                <th className="py-3.5 px-4">Receiver</th>
                <th className="py-3.5 px-4">Amount</th>
                <th className="py-3.5 px-4">Timestamp</th>
                <th className="py-3.5 px-4">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-sans">
              {transactions.slice(0, 6).map((txn) => (
                <tr
                  key={txn.id}
                  onClick={() => onSelectTransaction(txn)}
                  className="hover:bg-slate-800/40 cursor-pointer transition-colors group"
                >
                  <td className="py-3.5 px-4 font-mono font-semibold text-blue-400 group-hover:text-blue-300">
                    {txn.id}
                  </td>
                  <td className="py-3.5 px-4 text-slate-300 font-medium">
                    {txn.bank}
                  </td>
                  <td className="py-3.5 px-4 font-mono text-slate-400 max-w-[140px] truncate" title={txn.sender}>
                    {txn.sender}
                  </td>
                  <td className="py-3.5 px-4 font-mono text-slate-400 max-w-[140px] truncate" title={txn.receiver}>
                    {txn.receiver}
                  </td>
                  <td className="py-3.5 px-4 font-mono font-semibold text-white">
                    {formatINR(txn.amount)}
                  </td>
                  <td className="py-3.5 px-4 text-slate-400 font-mono text-[11px]">
                    {txn.timestamp}
                  </td>
                  <td className="py-3.5 px-4">
                    <StatusBadge status={txn.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
