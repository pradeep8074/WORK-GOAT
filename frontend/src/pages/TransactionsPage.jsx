import React, { useState, useMemo, useEffect, useRef } from 'react';
import { 
  Search, 
  Filter, 
  ArrowUpDown, 
  Download, 
  ShieldCheck, 
  RefreshCw,
  SlidersHorizontal,
  X,
  Play,
  Pause,
  Zap,
  Activity,
  Radio,
  ArrowRight,
  TrendingUp,
  AlertTriangle
} from 'lucide-react';
import StatusBadge from '../components/StatusBadge';
import LoadingState from '../components/LoadingState';
import ErrorState from '../components/ErrorState';
import { formatINR } from '../utils/currency';
import { apiService } from '../services/api';

export default function TransactionsPage({
  transactions: initialTransactions = [],
  loading,
  error,
  onRetry,
  onSelectTransaction
}) {
  // Live Streaming State
  const [isLiveMode, setIsLiveMode] = useState(true);
  const [isPlaying, setIsPlaying] = useState(true);
  const [pollIntervalMs, setPollIntervalMs] = useState(600); // 300ms Turbo, 600ms Ultra-Fast, 1s Fast
  const [liveStream, setLiveStream] = useState([]);
  const [flashTxnIds, setFlashTxnIds] = useState(new Set());
  const [liveStats, setLiveStats] = useState({
    totalProcessed: 0,
    highRiskDetected: 0,
    totalVolumeINR: 0,
    txPerSecond: '1.2'
  });

  // Filter & Search
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedBank, setSelectedBank] = useState('ALL');
  const [selectedStatus, setSelectedStatus] = useState('ALL');

  // Initial seed from backend / initialTransactions
  useEffect(() => {
    let isSubscribed = true;

    // Fetch initial buffer from live endpoint or fall back to initialTransactions
    apiService.getLiveTransactions(5, true)
      .then((res) => {
        if (!isSubscribed) return;
        const items = res.new_transactions || [];
        const buffer = res.buffer || [];
        const merged = items.concat(buffer);
        if (merged.length > 0) {
          // Deduplicate
          const seen = new Set();
          const clean = [];
          for (const tx of merged) {
            if (!seen.has(tx.id)) {
              seen.add(tx.id);
              clean.push(tx);
            }
          }
          setLiveStream(clean);
          updateCumulativeStats(clean);
        } else if (initialTransactions.length > 0) {
          setLiveStream(initialTransactions);
          updateCumulativeStats(initialTransactions);
        }
      })
      .catch(() => {
        if (!isSubscribed) return;
        if (initialTransactions.length > 0) {
          setLiveStream(initialTransactions);
          updateCumulativeStats(initialTransactions);
        }
      });

    return () => {
      isSubscribed = false;
    };
  }, [initialTransactions]);

  const updateCumulativeStats = (list) => {
    let vol = 0;
    let high = 0;
    for (const tx of list) {
      vol += (tx.amount || 0);
      if (tx.status === 'High Risk' || (tx.risk_score || 0) >= 65) {
        high += 1;
      }
    }
    setLiveStats((prev) => ({
      ...prev,
      totalProcessed: list.length,
      highRiskDetected: high,
      totalVolumeINR: vol,
      txPerSecond: (1000 / pollIntervalMs).toFixed(1)
    }));
  };

  // Live polling loop
  useEffect(() => {
    if (!isLiveMode || !isPlaying) return;

    const timer = setInterval(async () => {
      try {
        const res = await apiService.getLiveTransactions(1);
        if (res && res.new_transactions && res.new_transactions.length > 0) {
          const fresh = res.new_transactions[0];

          setLiveStream((prev) => {
            // Keep recent 200 items in buffer for smooth performance
            const updated = [fresh, ...prev.filter(t => t.id !== fresh.id)].slice(0, 200);
            return updated;
          });

          // Trigger row highlight flash
          setFlashTxnIds((prev) => {
            const next = new Set(prev);
            next.add(fresh.id);
            return next;
          });

          // Clear highlight after 1.8s
          setTimeout(() => {
            setFlashTxnIds((prev) => {
              const next = new Set(prev);
              next.delete(fresh.id);
              return next;
            });
          }, 1800);

          // Update live metrics
          setLiveStats((prev) => ({
            totalProcessed: prev.totalProcessed + 1,
            highRiskDetected: (fresh.status === 'High Risk' || (fresh.risk_score || 0) >= 65) 
              ? prev.highRiskDetected + 1 
              : prev.highRiskDetected,
            totalVolumeINR: prev.totalVolumeINR + (fresh.amount || 0),
            txPerSecond: (1000 / pollIntervalMs).toFixed(1)
          }));
        }
      } catch (err) {
        // Silent fail on live poll to avoid noisy alerts
        console.warn('Live stream poll error:', err.message);
      }
    }, pollIntervalMs);

    return () => clearInterval(timer);
  }, [isLiveMode, isPlaying, pollIntervalMs]);

  // Active dataset depending on mode
  const currentDataset = isLiveMode ? liveStream : initialTransactions;

  // Filter logic
  const filteredTransactions = useMemo(() => {
    return currentDataset.filter((txn) => {
      if (selectedBank !== 'ALL' && txn.bank !== selectedBank) {
        return false;
      }
      if (selectedStatus !== 'ALL' && txn.status !== selectedStatus) {
        return false;
      }
      if (searchTerm.trim() !== '') {
        const term = searchTerm.toLowerCase();
        const matchesId = (txn.id || '').toLowerCase().includes(term);
        const matchesSender = (txn.sender || '').toLowerCase().includes(term);
        const matchesReceiver = (txn.receiver || '').toLowerCase().includes(term);
        const matchesBank = (txn.bank || '').toLowerCase().includes(term);
        const matchesType = (txn.type || '').toLowerCase().includes(term);
        return matchesId || matchesSender || matchesReceiver || matchesBank || matchesType;
      }
      return true;
    });
  }, [currentDataset, selectedBank, selectedStatus, searchTerm]);

  const hasActiveFilters = searchTerm !== '' || selectedBank !== 'ALL' || selectedStatus !== 'ALL';

  const resetFilters = () => {
    setSearchTerm('');
    setSelectedBank('ALL');
    setSelectedStatus('ALL');
  };

  const handleTriggerManualTick = async () => {
    try {
      const res = await apiService.getLiveTransactions(2);
      if (res && res.new_transactions) {
        setLiveStream(prev => [...res.new_transactions, ...prev].slice(0, 200));
        res.new_transactions.forEach(tx => {
          setFlashTxnIds(p => new Set(p).add(tx.id));
        });
        setTimeout(() => {
          setFlashTxnIds(new Set());
        }, 1800);
      }
    } catch (e) {
      console.error(e);
    }
  };

  if (loading && currentDataset.length === 0) {
    return <LoadingState message="Connecting to transaction pipeline..." />;
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Top Header & Mode Toggle */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-bold text-white tracking-tight">Cross-Bank Transactions</h2>
            
            {/* Live Indicator Badge */}
            {isLiveMode ? (
              <div className="flex items-center gap-2 bg-emerald-950/70 border border-emerald-500/40 text-emerald-300 px-2.5 py-1 rounded-full text-xs font-mono font-medium shadow-sm shadow-emerald-900/30">
                <span className="relative flex h-2 w-2">
                  <span className={`animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75 ${isPlaying ? '' : 'hidden'}`}></span>
                  <span className={`relative inline-flex rounded-full h-2 w-2 ${isPlaying ? 'bg-emerald-400' : 'bg-amber-400'}`}></span>
                </span>
                <span>{isPlaying ? 'LIVE STREAMING' : 'STREAM PAUSED'}</span>
              </div>
            ) : (
              <span className="text-[11px] font-mono uppercase px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
                Archived Ledger
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Real-time inter-bank settlement stream across Apex Bank, Beacon Bank, and Crestline Bank. Evaluated continuously by G.O.A.T. Shield rule engine.
          </p>
        </div>

        {/* Action Buttons: Live vs Archived & Stream Controls */}
        <div className="flex items-center flex-wrap gap-2.5">
          {/* Feed Switch */}
          <div className="bg-[#0b121e] border border-slate-800 p-0.5 rounded-lg flex items-center">
            <button
              onClick={() => setIsLiveMode(true)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5 ${
                isLiveMode 
                  ? 'bg-blue-600 text-white shadow-sm' 
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Radio className="w-3.5 h-3.5 text-emerald-300" />
              Live Feed
            </button>
            <button
              onClick={() => setIsLiveMode(false)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                !isLiveMode 
                  ? 'bg-slate-700 text-white shadow-sm' 
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Snapshot
            </button>
          </div>

          {/* Stream Controls (Only visible in Live Mode) */}
          {isLiveMode && (
            <div className="flex items-center gap-2 bg-[#0b121e] border border-slate-800 px-3 py-1.5 rounded-lg">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`p-1 rounded hover:bg-slate-800 transition-colors ${
                  isPlaying ? 'text-amber-400 hover:text-amber-300' : 'text-emerald-400 hover:text-emerald-300'
                }`}
                title={isPlaying ? 'Pause stream' : 'Resume stream'}
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>

              <div className="h-4 w-px bg-slate-800" />

              <span className="text-[11px] font-mono text-slate-400">Speed:</span>
              <select
                value={pollIntervalMs}
                onChange={(e) => setPollIntervalMs(Number(e.target.value))}
                className="bg-[#080d16] border border-slate-700 rounded px-2 py-0.5 text-xs text-slate-200 font-mono focus:outline-none"
              >
                <option value={300}>⚡ Turbo (0.3s)</option>
                <option value={600}>🚀 Ultra-Fast (0.6s)</option>
                <option value={1000}>Fast (1.0s)</option>
                <option value={2000}>Normal (2.0s)</option>
                <option value={4000}>Slow (4.0s)</option>
              </select>

              <button
                onClick={handleTriggerManualTick}
                className="ml-1 p-1 rounded text-slate-400 hover:text-cyan-400 hover:bg-slate-800 transition-colors"
                title="Emit next transaction now"
              >
                <Zap className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Live Stream Telemetry Banner (when live) */}
      {isLiveMode && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-[#0a101d] border border-slate-800/80 rounded-xl p-3.5 text-xs">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Activity className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[10px] uppercase font-mono text-slate-400">Throughput</div>
              <div className="text-sm font-bold font-mono text-white flex items-center gap-1">
                {isPlaying ? `${liveStats.txPerSecond} txn/s` : '0.0 txn/s'}
                <span className="text-[10px] font-normal text-emerald-400">● live</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <TrendingUp className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[10px] uppercase font-mono text-slate-400">Active Buffer</div>
              <div className="text-sm font-bold font-mono text-white">
                {currentDataset.length} <span className="text-xs font-normal text-slate-500">records</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20">
              <AlertTriangle className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[10px] uppercase font-mono text-slate-400">High Risk Flagged</div>
              <div className="text-sm font-bold font-mono text-rose-400">
                {liveStats.highRiskDetected}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/20">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <div>
              <div className="text-[10px] uppercase font-mono text-slate-400">Volume Scored</div>
              <div className="text-sm font-bold font-mono text-slate-200">
                {formatINR(liveStats.totalVolumeINR)}
              </div>
            </div>
          </div>
        </div>
      )}

      {error && (
        <ErrorState message={error} onRetry={onRetry} isFallback={currentDataset.length > 0} />
      )}

      {/* Filter and Search Bar */}
      <div className="bg-[#0e1626] border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Search Input */}
          <div className="relative lg:col-span-2">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search by ID, sender, receiver, or type..."
              className="w-full bg-[#080d16] border border-slate-700/80 rounded-lg pl-9 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 font-mono"
            />
          </div>

          {/* Bank Filter */}
          <div className="relative">
            <select
              value={selectedBank}
              onChange={(e) => setSelectedBank(e.target.value)}
              className="w-full bg-[#080d16] border border-slate-700/80 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-blue-500 font-medium appearance-none cursor-pointer"
            >
              <option value="ALL">All Banks (A, B, C)</option>
              <option value="Apex Bank">Apex Bank</option>
              <option value="Beacon Bank">Beacon Bank</option>
              <option value="Crestline Bank">Crestline Bank</option>
              <option value="Bank A">Bank A</option>
              <option value="Bank B">Bank B</option>
              <option value="Bank C">Bank C</option>
            </select>
          </div>

          {/* Status Filter */}
          <div className="relative">
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="w-full bg-[#080d16] border border-slate-700/80 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-blue-500 font-medium appearance-none cursor-pointer"
            >
              <option value="ALL">All Statuses</option>
              <option value="Normal">Normal</option>
              <option value="Review">Review</option>
              <option value="High Risk">High Risk</option>
            </select>
          </div>
        </div>

        {/* Filter tags & reset */}
        {hasActiveFilters && (
          <div className="flex items-center gap-2 pt-2 border-t border-slate-800/80 text-xs">
            <span className="text-slate-500">Active filters:</span>
            {selectedBank !== 'ALL' && (
              <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-[11px] font-mono">
                Bank: {selectedBank}
              </span>
            )}
            {selectedStatus !== 'ALL' && (
              <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-[11px] font-mono">
                Status: {selectedStatus}
              </span>
            )}
            {searchTerm && (
              <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-[11px] font-mono">
                Query: "{searchTerm}"
              </span>
            )}
            <button
              onClick={resetFilters}
              className="ml-auto flex items-center gap-1 text-[11px] text-rose-400 hover:text-rose-300 transition-colors font-mono"
            >
              <X className="w-3.5 h-3.5" />
              Clear filters
            </button>
          </div>
        )}
      </div>

      {/* Transaction Table */}
      <div className="bg-[#0e1626] border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#0a0f1c] text-slate-400 border-b border-slate-800 font-mono uppercase text-[11px]">
              <tr>
                <th className="py-3.5 px-4">Transaction ID</th>
                <th className="py-3.5 px-4">Bank</th>
                <th className="py-3.5 px-4">Sender</th>
                <th className="py-3.5 px-4">Receiver</th>
                <th className="py-3.5 px-4">Type</th>
                <th className="py-3.5 px-4">Amount</th>
                <th className="py-3.5 px-4">Timestamp</th>
                <th className="py-3.5 px-4 text-center">Risk Score</th>
                <th className="py-3.5 px-4">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredTransactions.length > 0 ? (
                filteredTransactions.map((txn) => {
                  const isFlashing = flashTxnIds.has(txn.id);
                  const isLiveItem = txn.id?.startsWith('TXN-LIVE');
                  const riskScore = txn.risk_score !== undefined && txn.risk_score !== null 
                    ? txn.risk_score 
                    : (txn.status === 'High Risk' ? 85 : txn.status === 'Review' ? 50 : 15);

                  return (
                    <tr
                      key={txn.id}
                      onClick={() => onSelectTransaction(txn)}
                      className={`cursor-pointer transition-all duration-300 group ${
                        isFlashing 
                          ? 'bg-blue-900/40 border-l-4 border-l-cyan-400' 
                          : 'hover:bg-slate-800/40'
                      }`}
                    >
                      <td className="py-3.5 px-4 font-mono font-semibold text-blue-400 group-hover:text-blue-300 flex items-center gap-2">
                        {isLiveItem && (
                          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shrink-0" title="Live streamed transaction"></span>
                        )}
                        <span>{txn.id}</span>
                      </td>
                      <td className="py-3.5 px-4 text-slate-300 font-medium">
                        {txn.bank}
                      </td>
                      <td className="py-3.5 px-4 font-mono text-slate-400 max-w-[130px] truncate" title={txn.sender}>
                        {txn.sender}
                      </td>
                      <td className="py-3.5 px-4 font-mono text-slate-400 max-w-[130px] truncate" title={txn.receiver}>
                        {txn.receiver}
                      </td>
                      <td className="py-3.5 px-4 text-slate-400 font-mono text-[11px]">
                        {txn.type || 'Wire'}
                      </td>
                      <td className="py-3.5 px-4 font-mono font-semibold text-white">
                        {formatINR(txn.amount)}
                      </td>
                      <td className="py-3.5 px-4 text-slate-400 font-mono text-[11px]">
                        {txn.timestamp}
                      </td>
                      <td className="py-3.5 px-4 text-center font-mono">
                        <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                          riskScore >= 65 ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' :
                          riskScore >= 30 ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' :
                          'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        }`}>
                          {riskScore}
                        </span>
                      </td>
                      <td className="py-3.5 px-4">
                        <StatusBadge status={txn.status} />
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={9} className="py-12 text-center text-slate-500">
                    <p className="text-sm font-medium text-slate-400">No transactions match your criteria</p>
                    <p className="text-xs mt-1">Try adjusting the filter options above.</p>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
