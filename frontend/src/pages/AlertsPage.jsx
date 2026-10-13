import React, { useState, useMemo, useEffect } from 'react';
import { 
  AlertTriangle, 
  ShieldAlert, 
  Info, 
  Clock, 
  Terminal, 
  Layers, 
  ArrowRight,
  Radio,
  Play,
  Pause,
  Zap,
  Activity,
  Filter,
  Search,
  X,
  ExternalLink,
  ShieldCheck,
  CheckCircle2
} from 'lucide-react';
import StatusBadge from '../components/StatusBadge';
import LoadingState from '../components/LoadingState';
import ErrorState from '../components/ErrorState';
import { formatINR } from '../utils/currency';
import { apiService } from '../services/api';

export default function AlertsPage({ 
  alerts: initialAlerts = [], 
  loading, 
  error, 
  onRetry,
  onSelectTransaction,
  onNavigateToTab 
}) {
  // Live Streaming State
  const [isLiveMode, setIsLiveMode] = useState(true);
  const [isPlaying, setIsPlaying] = useState(true);
  const [pollIntervalMs, setPollIntervalMs] = useState(600);
  const [liveAlerts, setLiveAlerts] = useState([]);
  const [flashAlertIds, setFlashAlertIds] = useState(new Set());
  const [stats, setStats] = useState({
    critical: 0,
    high: 0,
    medium: 0,
    total: 0
  });

  // Filters
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedBank, setSelectedBank] = useState('ALL');
  const [selectedRisk, setSelectedRisk] = useState('ALL');

  // Compute summary metrics
  const updateMetrics = (list) => {
    let crit = 0;
    let hi = 0;
    let med = 0;
    for (const a of list) {
      const r = (a.risk || '').toLowerCase();
      if (r === 'critical') crit += 1;
      else if (r === 'high') hi += 1;
      else med += 1;
    }
    setStats({
      critical: crit,
      high: hi,
      medium: med,
      total: list.length
    });
  };

  // Initial seed from backend
  useEffect(() => {
    let isSubscribed = true;

    apiService.getLiveAlerts(true)
      .then((res) => {
        if (!isSubscribed) return;
        const items = res.alerts || [];
        if (items.length > 0) {
          setLiveAlerts(items);
          updateMetrics(items);
        } else if (initialAlerts.length > 0) {
          setLiveAlerts(initialAlerts);
          updateMetrics(initialAlerts);
        }
      })
      .catch(() => {
        if (!isSubscribed) return;
        if (initialAlerts.length > 0) {
          setLiveAlerts(initialAlerts);
          updateMetrics(initialAlerts);
        }
      });

    return () => {
      isSubscribed = false;
    };
  }, [initialAlerts]);

  // Live polling connected to the active transaction engine
  useEffect(() => {
    if (!isLiveMode || !isPlaying) return;

    const timer = setInterval(async () => {
      try {
        const res = await apiService.getLiveAlerts(true);
        if (res && res.alerts && res.alerts.length > 0) {
          const freshAlerts = res.alerts;

          setLiveAlerts((prev) => {
            const existingIds = new Set(prev.map(a => a.id));
            const brandNew = freshAlerts.filter(a => !existingIds.has(a.id));

            if (brandNew.length > 0) {
              // Highlight incoming alerts
              setFlashAlertIds((f) => {
                const n = new Set(f);
                brandNew.forEach(b => n.add(b.id));
                return n;
              });

              setTimeout(() => {
                setFlashAlertIds((f) => {
                  const n = new Set(f);
                  brandNew.forEach(b => n.delete(b.id));
                  return n;
                });
              }, 2000);
            }

            // Deduplicate and retain recent 100 alerts
            const seen = new Set();
            const merged = [];
            for (const item of [...freshAlerts, ...prev]) {
              if (!seen.has(item.id)) {
                seen.add(item.id);
                merged.push(item);
              }
            }
            const capped = merged.slice(0, 100);
            updateMetrics(capped);
            return capped;
          });
        }
      } catch (err) {
        console.warn('Live alerts poll error:', err.message);
      }
    }, pollIntervalMs);

    return () => clearInterval(timer);
  }, [isLiveMode, isPlaying, pollIntervalMs]);

  const activeDataset = isLiveMode ? liveAlerts : initialAlerts;

  // Filter logic
  const filteredAlerts = useMemo(() => {
    return activeDataset.filter((alert) => {
      if (selectedBank !== 'ALL' && alert.bank !== selectedBank) {
        return false;
      }
      if (selectedRisk !== 'ALL' && alert.risk !== selectedRisk) {
        return false;
      }
      if (searchTerm.trim() !== '') {
        const term = searchTerm.toLowerCase();
        const matchesId = (alert.id || '').toLowerCase().includes(term);
        const matchesAccount = (alert.account || '').toLowerCase().includes(term);
        const matchesBank = (alert.bank || '').toLowerCase().includes(term);
        const matchesReason = (alert.reason || '').toLowerCase().includes(term);
        const matchesTxn = (alert.transaction_id || '').toLowerCase().includes(term);
        return matchesId || matchesAccount || matchesBank || matchesReason || matchesTxn;
      }
      return true;
    });
  }, [activeDataset, selectedBank, selectedRisk, searchTerm]);

  const hasActiveFilters = searchTerm !== '' || selectedBank !== 'ALL' || selectedRisk !== 'ALL';

  const resetFilters = () => {
    setSearchTerm('');
    setSelectedBank('ALL');
    setSelectedRisk('ALL');
  };

  const handleAlertClick = (alert) => {
    if (onSelectTransaction) {
      // Map alert into transaction representation for the drawer
      const simulatedTxn = {
        id: alert.transaction_id || alert.id,
        bank: alert.bank,
        sender: alert.account,
        receiver: alert.counterparty || 'Flagged Mule Collector',
        amount: alert.amount || 850000.0,
        currency: 'INR',
        timestamp: alert.created,
        status: alert.risk === 'Critical' ? 'High Risk' : alert.risk === 'High' ? 'High Risk' : 'Review',
        risk_score: alert.risk_score || (alert.risk === 'Critical' ? 85 : 50),
        triggered_rules: alert.triggered_rules || (alert.rule_flag ? [alert.rule_flag.replace('RULE-', '')] : ['R2']),
        explanation: alert.details || alert.reason,
        note: alert.details || alert.reason
      };
      onSelectTransaction(simulatedTxn);
    }
  };

  if (loading && activeDataset.length === 0) {
    return <LoadingState message="Connecting to live security alert feed..." />;
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page Title & Feed Controls */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-bold text-white tracking-tight">Security & Fraud Alerts</h2>
            
            {/* Live Indicator Badge */}
            {isLiveMode ? (
              <div className="flex items-center gap-2 bg-rose-950/70 border border-rose-500/40 text-rose-300 px-2.5 py-1 rounded-full text-xs font-mono font-medium shadow-sm shadow-rose-900/30">
                <span className="relative flex h-2 w-2">
                  <span className={`animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75 ${isPlaying ? '' : 'hidden'}`}></span>
                  <span className={`relative inline-flex rounded-full h-2 w-2 ${isPlaying ? 'bg-rose-400' : 'bg-amber-400'}`}></span>
                </span>
                <span>{isPlaying ? 'LIVE ALERTS ACTIVE' : 'STREAM PAUSED'}</span>
              </div>
            ) : (
              <span className="text-[11px] font-mono uppercase px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
                Archived Alerts
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Real-time security anomaly detections generated continuously by the G.O.A.T. Shield rule & GNN engine. Click any alert to inspect transaction forensic details.
          </p>
        </div>

        {/* Live Controls */}
        <div className="flex items-center flex-wrap gap-2.5">
          {/* Mode switch */}
          <div className="bg-[#0b121e] border border-slate-800 p-0.5 rounded-lg flex items-center">
            <button
              onClick={() => setIsLiveMode(true)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5 ${
                isLiveMode 
                  ? 'bg-rose-600 text-white shadow-sm' 
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Radio className="w-3.5 h-3.5 text-rose-200" />
              Live Alerts
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

          {/* Stream playback controls */}
          {isLiveMode && (
            <div className="flex items-center gap-2 bg-[#0b121e] border border-slate-800 px-3 py-1.5 rounded-lg">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`p-1 rounded hover:bg-slate-800 transition-colors ${
                  isPlaying ? 'text-amber-400 hover:text-amber-300' : 'text-emerald-400 hover:text-emerald-300'
                }`}
                title={isPlaying ? 'Pause alerts feed' : 'Resume alerts feed'}
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>

              <div className="h-4 w-px bg-slate-800" />

              <span className="text-[11px] font-mono text-slate-400">Sync:</span>
              <select
                value={pollIntervalMs}
                onChange={(e) => setPollIntervalMs(Number(e.target.value))}
                className="bg-[#080d16] border border-slate-700 rounded px-2 py-0.5 text-xs text-slate-200 font-mono focus:outline-none"
              >
                <option value={300}>⚡ Turbo (0.3s)</option>
                <option value={600}>🚀 Ultra-Fast (0.6s)</option>
                <option value={1000}>1.0s</option>
                <option value={2000}>2.0s</option>
                <option value={4000}>4.0s</option>
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Live Alert Metric Counters */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-[#0a101d] border border-slate-800/80 rounded-xl p-3.5 text-xs">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <AlertTriangle className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[10px] uppercase font-mono text-slate-400">Critical Alerts</div>
            <div className="text-sm font-bold font-mono text-rose-400 flex items-center gap-1">
              {stats.critical}
              <span className="text-[10px] font-normal text-rose-400/80">Immediate</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <ShieldAlert className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[10px] uppercase font-mono text-slate-400">High Severity</div>
            <div className="text-sm font-bold font-mono text-amber-400">
              {stats.high}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[10px] uppercase font-mono text-slate-400">Medium / Review</div>
            <div className="text-sm font-bold font-mono text-slate-200">
              {stats.medium}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[10px] uppercase font-mono text-slate-400">Total Flagged</div>
            <div className="text-sm font-bold font-mono text-white">
              {stats.total} <span className="text-xs font-normal text-slate-500">signals</span>
            </div>
          </div>
        </div>
      </div>

      {error && (
        <ErrorState message={error} onRetry={onRetry} isFallback={activeDataset.length > 0} />
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
              placeholder="Search by Alert ID, account, reason, or Txn ID..."
              className="w-full bg-[#080d16] border border-slate-700/80 rounded-lg pl-9 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-rose-500 font-mono"
            />
          </div>

          {/* Bank Filter */}
          <div className="relative">
            <select
              value={selectedBank}
              onChange={(e) => setSelectedBank(e.target.value)}
              className="w-full bg-[#080d16] border border-slate-700/80 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-rose-500 font-medium appearance-none cursor-pointer"
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

          {/* Risk Filter */}
          <div className="relative">
            <select
              value={selectedRisk}
              onChange={(e) => setSelectedRisk(e.target.value)}
              className="w-full bg-[#080d16] border border-slate-700/80 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-rose-500 font-medium appearance-none cursor-pointer"
            >
              <option value="ALL">All Risk Levels</option>
              <option value="Critical">Critical</option>
              <option value="High">High</option>
              <option value="Medium">Medium</option>
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
            {selectedRisk !== 'ALL' && (
              <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-[11px] font-mono">
                Risk: {selectedRisk}
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

      {/* Alerts Table */}
      <div className="bg-[#0e1626] border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#0a0f1c] text-slate-400 border-b border-slate-800 font-mono uppercase text-[11px]">
              <tr>
                <th className="py-3.5 px-4">Alert ID</th>
                <th className="py-3.5 px-4">Account</th>
                <th className="py-3.5 px-4">Bank</th>
                <th className="py-3.5 px-4">Risk Level</th>
                <th className="py-3.5 px-4">Reason & Triggered Rules</th>
                <th className="py-3.5 px-4">Status</th>
                <th className="py-3.5 px-4">Time</th>
                <th className="py-3.5 px-4 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredAlerts.length > 0 ? (
                filteredAlerts.map((alert) => {
                  const isFlashing = flashAlertIds.has(alert.id);
                  const isLive = alert.id?.startsWith('ALT-LIVE');

                  return (
                    <tr 
                      key={alert.id} 
                      onClick={() => handleAlertClick(alert)}
                      className={`cursor-pointer transition-all duration-300 group ${
                        isFlashing 
                          ? 'bg-rose-950/40 border-l-4 border-l-rose-500' 
                          : 'hover:bg-slate-800/40'
                      }`}
                    >
                      <td className="py-3.5 px-4 font-mono font-semibold text-rose-400 group-hover:text-rose-300 flex items-center gap-2">
                        {isLive && (
                          <span className="w-1.5 h-1.5 rounded-full bg-rose-400 shrink-0" title="Live streamed alert"></span>
                        )}
                        <span>{alert.id}</span>
                      </td>
                      <td className="py-3.5 px-4 font-mono text-slate-200">
                        {alert.account}
                      </td>
                      <td className="py-3.5 px-4 text-slate-300 font-medium">
                        {alert.bank}
                      </td>
                      <td className="py-3.5 px-4">
                        <StatusBadge status={alert.risk} />
                      </td>
                      <td className="py-3.5 px-4 text-slate-200 max-w-[340px]">
                        <div>
                          <p className="font-medium text-slate-200 leading-snug">{alert.reason}</p>
                          {alert.details && (
                            <p className="text-[11px] text-slate-400 font-sans mt-0.5 line-clamp-1">
                              {alert.details}
                            </p>
                          )}
                          {alert.triggered_rules && alert.triggered_rules.length > 0 && (
                            <div className="flex gap-1 mt-1">
                              {alert.triggered_rules.map(r => (
                                <span key={r} className="px-1.5 py-0.2 bg-rose-950 text-rose-300 border border-rose-800/60 rounded text-[9px] font-mono">
                                  {r}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="py-3.5 px-4">
                        <StatusBadge status={alert.status} />
                      </td>
                      <td className="py-3.5 px-4 text-slate-400 font-mono text-[11px]">
                        {alert.created}
                      </td>
                      <td className="py-3.5 px-4 text-center">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleAlertClick(alert);
                          }}
                          className="px-2.5 py-1 text-[11px] font-mono rounded bg-slate-800 hover:bg-slate-700 text-blue-400 hover:text-blue-300 border border-slate-700 transition-colors inline-flex items-center gap-1"
                        >
                          <span>Inspect</span>
                          <ArrowRight className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-slate-500">
                    <p className="text-sm font-medium text-slate-400">No security alerts match your criteria</p>
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
