import React, { useState, useEffect, useCallback } from 'react';
import { 
  Search, 
  Filter, 
  ShieldAlert, 
  AlertTriangle, 
  CheckCircle2, 
  Clock, 
  ArrowRight, 
  ChevronDown, 
  ChevronUp, 
  Eye, 
  RefreshCw,
  Layers,
  Sparkles,
  Info,
  Building2,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { apiService } from '../services/api';
import { FALLBACK_SIMULATION } from '../data/fallbackData';
import { formatINR } from '../utils/currency';
import StatusBadge from '../components/StatusBadge';

export default function InvestigationPage({ onSelectTransaction }) {
  const [transactions, setTransactions] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [detectionSummary, setDetectionSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters & Pagination
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [ringFilter, setRingFilter] = useState('ALL');
  const [bankFilter, setBankFilter] = useState('ALL');
  const [muleOnly, setMuleOnly] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedRows, setExpandedRows] = useState(new Set());

  const fetchInvestigationData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const params = {
        limit: pageSize,
        offset: page * pageSize
      };
      if (statusFilter !== 'ALL') params.status = statusFilter;
      if (ringFilter !== 'ALL') params.ring_id = ringFilter;
      if (bankFilter !== 'ALL') params.bank = bankFilter;
      if (muleOnly) params.mule_only = true;
      if (searchTerm.trim()) params.search = searchTerm.trim();

      const [simRes, detectRes] = await Promise.all([
        apiService.getSimulation(params),
        apiService.getDetection()
      ]);

      setTransactions(simRes.transactions || []);
      setTotalCount(simRes.total || 0);
      setDetectionSummary(detectRes);
    } catch (err) {
      console.warn('Failed to fetch live investigation data:', err);
      setError('Live investigation engine offline. Showing offline fallback snapshot.');
      setTransactions(FALLBACK_SIMULATION.transactions);
      setTotalCount(FALLBACK_SIMULATION.total);
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, statusFilter, ringFilter, bankFilter, muleOnly, searchTerm]);

  useEffect(() => {
    fetchInvestigationData();
  }, [fetchInvestigationData]);

  const toggleRow = (id) => {
    setExpandedRows(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const totalPages = Math.ceil(totalCount / pageSize);
  const isOffline = Boolean(error);
  const ruleFreqs = detectionSummary?.rule_frequencies || {};
  const ruleDefs = detectionSummary?.rule_definitions || {};

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Top Banner & Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-[#0d1527] border border-slate-800 rounded-xl p-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase px-2.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-semibold flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5" />
              Stage 2A Transparent Forensic Engine
            </span>
            <span className="text-xs text-slate-400 font-mono">500 Synthetic INR Stream</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2.5">
            <Search className="w-6 h-6 text-blue-400" />
            Deep Transaction Investigation & Rule Auditing
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Deterministic, rule-based fraud scoring with transparent rule audit trails and ground-truth validation.
          </p>
        </div>

        <button
          onClick={fetchInvestigationData}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700 transition disabled:opacity-50 self-start md:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-blue-400' : ''}`} />
          Refresh Stream
        </button>
      </div>

      {/* Offline Notice */}
      {isOffline && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-center gap-3 text-amber-300 text-xs">
          <AlertTriangle className="w-5 h-5 flex-shrink-0 text-amber-400" />
          <div>
            <span className="font-semibold">Backend Offline Notice:</span> Live investigation dataset is offline. Displaying preview fallback transactions.
          </div>
        </div>
      )}

      {/* Transparent Rules Breakdown Cards */}
      <div className="bg-[#0b1222] border border-slate-800 rounded-xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-mono uppercase text-slate-300 font-semibold tracking-wider flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-amber-400" />
            Transparent Rule-Based Detection Engine (R1 ? R6)
          </h3>
          <span className="text-xs font-mono text-slate-400">
            Total Evaluated: {detectionSummary?.total_evaluated || 500} Txns
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-1">
          {Object.entries(ruleDefs).map(([code, def]) => {
            const count = ruleFreqs[code] || 0;
            return (
              <div key={code} className="bg-slate-900/70 border border-slate-800/80 rounded-lg p-3 space-y-1.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono text-xs font-bold text-amber-400 px-1.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/20">
                      {code}
                    </span>
                    <span className="text-xs font-semibold text-slate-200">{def.name}</span>
                  </div>
                  <span className="text-[11px] font-mono text-slate-400 font-bold">
                    +{def.weight} pts
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-snug">{def.description}</p>
                <div className="flex items-center justify-between pt-1 border-t border-slate-800/60 text-[10px] font-mono">
                  <span className="text-slate-500">Trigger Count:</span>
                  <span className="text-emerald-400 font-bold">{count} matches</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Search & Filter Toolbar */}
      <div className="bg-[#0c1424] border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {/* Search Box */}
          <div className="relative lg:col-span-2">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search account, Txn ID, note..."
              value={searchTerm}
              onChange={(e) => { setSearchTerm(e.target.value); setPage(0); }}
              className="w-full bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg pl-9 pr-3 py-2 focus:outline-none focus:border-blue-500 placeholder-slate-500 font-mono"
            />
          </div>

          {/* Status Filter */}
          <div>
            <select
              value={statusFilter}
              onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }}
              className="w-full bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500 font-mono"
            >
              <option value="ALL">All Risk Statuses</option>
              <option value="High Risk">High Risk (Score ? 65)</option>
              <option value="Review">Review (40 ? Score &lt; 65)</option>
              <option value="Normal">Normal (Score &lt; 40)</option>
            </select>
          </div>

          {/* Mule Ring Filter */}
          <div>
            <select
              value={ringFilter}
              onChange={(e) => { setRingFilter(e.target.value); setPage(0); }}
              className="w-full bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500 font-mono"
            >
              <option value="ALL">All Mule Topologies</option>
              <option value="RING-001">RING-001 (Fan-In Funnel)</option>
              <option value="RING-002">RING-002 (Fan-Out Dispersion)</option>
              <option value="RING-003">RING-003 (Layering Chains)</option>
              <option value="RING-004">RING-004 (Round-Trip Loops)</option>
            </select>
          </div>

          {/* Bank Filter */}
          <div>
            <select
              value={bankFilter}
              onChange={(e) => { setBankFilter(e.target.value); setPage(0); }}
              className="w-full bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500 font-mono"
            >
              <option value="ALL">All Banks</option>
              <option value="Apex Bank">Apex Bank</option>
              <option value="Beacon Bank">Beacon Bank</option>
              <option value="Crestline Bank">Crestline Bank</option>
            </select>
          </div>
        </div>

        {/* Second Filter Row */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-800/80 text-xs">
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer select-none text-slate-300">
              <input
                type="checkbox"
                checked={muleOnly}
                onChange={(e) => { setMuleOnly(e.target.checked); setPage(0); }}
                className="rounded border-slate-700 bg-slate-900 text-red-500 focus:ring-0 w-3.5 h-3.5"
              />
              Show Ground-Truth Mule Transactions Only
            </label>
          </div>

          <div className="text-slate-400 font-mono text-[11px]">
            Showing <span className="text-white font-bold">{transactions.length}</span> of <span className="text-white font-bold">{totalCount}</span> transactions
          </div>
        </div>
      </div>

      {/* Transactions Table */}
      <div className="bg-[#0b1222] border border-slate-800 rounded-xl overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#080d1a] border-b border-slate-800 text-[11px] font-mono text-slate-400 uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4">Txn ID / Time</th>
                <th className="py-3 px-4">Origin Bank</th>
                <th className="py-3 px-4">Route (Sender ? Receiver)</th>
                <th className="py-3 px-4">Amount</th>
                <th className="py-3 px-4">Payment Rail</th>
                <th className="py-3 px-4">Risk Score</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4 text-right">Ground Truth</th>
                <th className="py-3 px-3 text-center">Audit</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {loading ? (
                <tr>
                  <td colSpan="9" className="py-12 text-center text-slate-400">
                    <RefreshCw className="w-6 h-6 animate-spin mx-auto text-blue-400 mb-2" />
                    Loading simulated transaction ledger...
                  </td>
                </tr>
              ) : transactions.length === 0 ? (
                <tr>
                  <td colSpan="9" className="py-12 text-center text-slate-500">
                    No transactions found matching the selected filter criteria.
                  </td>
                </tr>
              ) : (
                transactions.map((txn) => {
                  const isExpanded = expandedRows.has(txn.id);
                  const isMule = txn.is_mule;
                  const isCross = txn.is_cross_bank;
                  const score = txn.risk_score || 0;

                  return (
                    <React.Fragment key={txn.id}>
                      <tr 
                        className={`hover:bg-slate-800/40 transition cursor-pointer ${
                          isExpanded ? 'bg-slate-800/30' : ''
                        } ${isMule ? 'bg-red-950/10' : ''}`}
                        onClick={() => toggleRow(txn.id)}
                      >
                        <td className="py-3 px-4">
                          <div className="font-bold text-white flex items-center gap-1.5">
                            {txn.id}
                          </div>
                          <span className="text-[10px] text-slate-500 block">{txn.timestamp}</span>
                        </td>

                        <td className="py-3 px-4">
                          <span className="text-slate-300 font-sans text-xs">{txn.bank}</span>
                        </td>

                        <td className="py-3 px-4">
                          <div className="flex items-center gap-1.5 text-slate-200">
                            <span className="text-blue-400">{txn.sender}</span>
                            <ArrowRight className="w-3 h-3 text-slate-600 flex-shrink-0" />
                            <span className="text-emerald-400">{txn.receiver}</span>
                          </div>
                          {isCross && (
                            <span className="text-[10px] text-amber-400 font-mono">Cross-Bank</span>
                          )}
                        </td>

                        <td className="py-3 px-4">
                          <span className="font-bold text-white">{formatINR(txn.amount)}</span>
                        </td>

                        <td className="py-3 px-4">
                          <span className="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 border border-slate-700">
                            {txn.type}
                          </span>
                        </td>

                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <div className="w-12 bg-slate-800 rounded-full h-2 overflow-hidden">
                              <div
                                className={`h-full ${
                                  score >= 65 ? 'bg-red-500' :
                                  score >= 40 ? 'bg-amber-500' : 'bg-emerald-500'
                                }`}
                                style={{ width: `${score}%` }}
                              ></div>
                            </div>
                            <span className={`font-bold ${
                              score >= 65 ? 'text-red-400' :
                              score >= 40 ? 'text-amber-400' : 'text-emerald-400'
                            }`}>
                              {score}
                            </span>
                          </div>
                        </td>

                        <td className="py-3 px-4">
                          <StatusBadge status={txn.status} />
                        </td>

                        <td className="py-3 px-4 text-right">
                          {isMule ? (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-300 border border-red-500/30">
                              MULE ({txn.mule_ring_id})
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] bg-slate-800/80 text-slate-400">
                              Normal
                            </span>
                          )}
                        </td>

                        <td className="py-3 px-3 text-center">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleRow(txn.id);
                            }}
                            className="p-1 hover:bg-slate-700 rounded text-slate-400 hover:text-white transition"
                          >
                            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                          </button>
                        </td>
                      </tr>

                      {/* Expandable Forensic Audit Row */}
                      {isExpanded && (
                        <tr className="bg-[#070c17] border-y border-slate-800/80">
                          <td colSpan="9" className="p-4 space-y-3">
                            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-mono font-bold text-slate-200">
                                  Triggered Rules ({txn.triggered_rules?.length || 0}):
                                </span>
                                <div className="flex flex-wrap gap-1.5">
                                  {(txn.triggered_rules || []).map((r) => (
                                    <span key={r} className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/10 text-amber-300 border border-amber-500/20">
                                      {r} (+{ruleDefs[r]?.weight || 0} pts)
                                    </span>
                                  ))}
                                  {(!txn.triggered_rules || txn.triggered_rules.length === 0) && (
                                    <span className="text-[11px] text-slate-500 font-sans">
                                      None (Standard Baseline Profile)
                                    </span>
                                  )}
                                </div>
                              </div>

                              <button
                                onClick={() => onSelectTransaction?.(txn)}
                                className="flex items-center gap-1.5 px-3 py-1 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 text-xs font-semibold rounded border border-blue-500/30 transition self-start md:self-auto font-sans"
                              >
                                <Eye className="w-3.5 h-3.5" />
                                Inspect in Drawer
                              </button>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-sans">
                              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                                <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold block mb-1">
                                  Audit Trail & Deterministic Explanation
                                </span>
                                <p className="text-slate-300 leading-relaxed">{txn.explanation}</p>
                              </div>

                              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                                <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold block mb-1">
                                  Pattern Classification & Metadata
                                </span>
                                <div className="space-y-1 font-mono text-[11px]">
                                  <div className="flex justify-between">
                                    <span className="text-slate-500">Pattern Archetype:</span>
                                    <span className="text-slate-200">{txn.pattern_type}</span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-slate-500">Synthetic Note:</span>
                                    <span className="text-slate-300 font-sans">{txn.note}</span>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Controls */}
        <div className="p-4 border-t border-slate-800 bg-[#080d1a] flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Rows per page:</span>
            <select
              value={pageSize}
              onChange={(e) => { setPageSize(Number(e.target.value)); setPage(0); }}
              className="bg-slate-900 border border-slate-700 text-slate-200 rounded px-2 py-1 focus:outline-none"
            >
              <option value="25">25</option>
              <option value="50">50</option>
              <option value="100">100</option>
            </select>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-slate-400">
              Page <span className="text-white font-bold">{page + 1}</span> of <span className="text-white font-bold">{Math.max(1, totalPages)}</span>
            </span>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="p-1.5 bg-slate-900 border border-slate-700 rounded text-slate-300 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => setPage(p => p + 1)}
                disabled={page + 1 >= totalPages}
                className="p-1.5 bg-slate-900 border border-slate-700 rounded text-slate-300 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
