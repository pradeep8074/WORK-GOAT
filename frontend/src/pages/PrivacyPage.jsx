import React, { useState, useEffect, useCallback } from 'react';
import {
  Lock,
  RefreshCw,
  Search,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Database,
  Share2,
  FileKey,
  EyeOff,
  Zap,
  ShieldCheck,
  ShieldAlert,
  Network
} from 'lucide-react';
import { apiService } from '../services/api';

export default function PrivacyPage({ onNavigateToTab }) {
  const [vaultsData, setVaultsData] = useState(null);
  const [psiResults, setPsiResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [runningPsi, setRunningPsi] = useState(false);
  const [activeStep, setActiveStep] = useState(5);
  const [filterRing, setFilterRing] = useState('ALL');
  const [searchTerm, setSearchTerm] = useState('');
  const [error, setError] = useState(null);

  // Fetch initial vault metadata and cached PSI results from backend
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [vaults, results] = await Promise.all([
        apiService.getPsiVaults(),
        apiService.getPsiResults(),
      ]);
      setVaultsData(vaults);
      setPsiResults(results);
    } catch (err) {
      console.error("Error loading PSI data:", err);
      setError("PSI backend unavailable. Live detection and cryptographic PSI services are currently offline. Please ensure the backend server is running.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Execute live PSI protocol via backend POST /api/v2/privacy/psi/run
  const handleRunPsi = async () => {
    setRunningPsi(true);
    setError(null);
    try {
      // Step progression visual animation
      setActiveStep(1);
      await new Promise((r) => setTimeout(r, 300));
      setActiveStep(2);
      await new Promise((r) => setTimeout(r, 400));
      setActiveStep(3);
      await new Promise((r) => setTimeout(r, 400));
      setActiveStep(4);
      
      const res = await apiService.runPsiProtocol();
      setPsiResults(res);
      
      setActiveStep(5);
    } catch (err) {
      console.error("Error running PSI protocol:", err);
      setError("Failed to execute live PSI protocol: " + err.message + ". PSI backend unavailable.");
    } finally {
      setRunningPsi(false);
    }
  };

  // Reset ephemeral keys and session via backend POST /api/v2/privacy/psi/reset
  const handleResetSession = async () => {
    setLoading(true);
    setError(null);
    try {
      await apiService.resetPsiSession();
      setActiveStep(1);
      setPsiResults(null);
      await loadData();
    } catch (err) {
      console.error("Error resetting PSI session:", err);
      setError("Failed to reset PSI session. Backend service unreachable.");
    } finally {
      setLoading(false);
    }
  };

  // Matches list
  const matches = psiResults?.all_pairwise_matches || [];
  const filteredMatches = matches.filter((m) => {
    const matchesRing = filterRing === 'ALL' || m.associated_ring === filterRing;
    const matchesSearch =
      !searchTerm ||
      m.party_a_account.toLowerCase().includes(searchTerm.toLowerCase()) ||
      m.party_b_account.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (m.verification_token && m.verification_token.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (m.associated_ring && m.associated_ring.toLowerCase().includes(searchTerm.toLowerCase()));
    return matchesRing && matchesSearch;
  });

  const tripartiteCommonEntities = psiResults?.tripartite_results?.common_entities || [];
  const pairwiseResults = psiResults?.pairwise_results;

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* 1. Header Banner */}
      <div className="bg-[#0e1626] border border-slate-800 rounded-xl p-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl pointer-events-none" />
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 relative z-10">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-mono uppercase px-2.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Stage 2C-1 — Cryptographic PSI Prototype
              </span>
              <span className="text-xs font-mono uppercase px-2.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-semibold">
                Semi-Honest Threat Model
              </span>
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2.5">
              <Lock className="w-6 h-6 text-blue-400" />
              Privacy-Preserving Entity Matching
            </h1>
            <p className="text-xs text-slate-400 max-w-2xl leading-relaxed">
              Cryptographic PSI prototype enabling <span className="text-blue-300 font-semibold">Apex Bank</span>, <span className="text-blue-300 font-semibold">Beacon Bank</span>, and <span className="text-blue-300 font-semibold">Crestline Bank</span> to discover common cross-bank entities without disclosing non-intersecting private rosters. Discovered common entities are bridged into the fraud-analysis graph.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={handleResetSession}
              disabled={loading || runningPsi}
              className="px-3.5 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-medium border border-slate-700 transition-all flex items-center gap-2 disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              RESET PSI SESSION
            </button>
            <button
              onClick={handleRunPsi}
              disabled={loading || runningPsi}
              className="px-5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold shadow-lg shadow-blue-600/20 transition-all flex items-center gap-2 disabled:opacity-50"
            >
              <Zap className={`w-4 h-4 ${runningPsi ? 'animate-bounce text-amber-300' : ''}`} />
              {runningPsi ? 'EXECUTING PSI EXCHANGE...' : 'RUN PSI'}
            </button>
            {onNavigateToTab && (
              <button
                onClick={() => onNavigateToTab('network')}
                className="px-4 py-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-blue-400 text-xs font-semibold border border-blue-500/30 transition-all flex items-center gap-1.5 shadow-sm"
              >
                <span>Bridge to Graph</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Backend Offline / Error State Notice */}
      {error && (
        <div className="p-5 bg-rose-500/10 border border-rose-500/30 rounded-xl text-xs text-rose-300 flex items-start gap-3.5">
          <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-bold text-rose-300 uppercase font-mono tracking-wide">PSI Backend Status</span>
            <p className="leading-relaxed">{error}</p>
          </div>
        </div>
      )}

      {/* 2. Security Disclosure Banner */}
      <div className="bg-[#0b111e] border border-amber-500/30 rounded-xl p-5 space-y-3">
        <div className="flex items-center gap-2.5 text-amber-400">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <h2 className="text-xs font-bold font-mono uppercase tracking-wider text-amber-300">
            Cryptographic PSI Prototype — Security Disclosures & Privacy Boundary
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-400 pt-2 border-t border-slate-800/80">
          <div className="space-y-1">
            <span className="font-semibold text-slate-200 block font-mono text-[11px]">Threat Model</span>
            <p className="text-[11px] leading-relaxed">
              Semi-Honest / Honest-but-Curious participants. Assumes Decisional Diffie-Hellman (DDH) hardness in the prime-order subgroup of RFC 3526 MODP-2048.
            </p>
          </div>
          <div className="space-y-1">
            <span className="font-semibold text-slate-200 block font-mono text-[11px]">Privacy Boundary</span>
            <p className="text-[11px] leading-relaxed">
              Raw identifiers remain strictly local to each bank. Peer institutions only observe blinded group elements permuted to prevent correlation.
            </p>
          </div>
          <div className="space-y-1">
            <span className="font-semibold text-slate-200 block font-mono text-[11px]">Prototype Limitations</span>
            <p className="text-[11px] leading-relaxed">
              • Not malicious-secure (no ZK proofs of exponentiation).<br />
              • Low-entropy identifiers require salted slow KDFs/OPRFs in production.<br />
              • Intersection cardinality is inherently revealed.
            </p>
          </div>
        </div>
      </div>

      {/* 3. Three Bank Vault Cards */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-bold text-white font-mono uppercase tracking-wider flex items-center gap-2">
            <Database className="w-4 h-4 text-blue-400" />
            Isolated Bank Vaults (Stage 1 & 2A Silos)
          </h2>
          <span className="text-xs text-slate-400 font-mono">3 Independent Institutions</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {vaultsData?.banks ? (
            vaultsData.banks.map((bank) => (
              <div
                key={bank.bank_code}
                className="bg-[#0b111e] border border-slate-800/90 rounded-xl p-5 space-y-4 relative overflow-hidden"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400">
                      <Database className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-white">{bank.bank_name}</h3>
                      <span className="text-[11px] font-mono text-slate-400">Silo ID: {bank.bank_code}</span>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 flex items-center gap-1">
                    <EyeOff className="w-3 h-3" />
                    Private Silo
                  </span>
                </div>

                <div className="space-y-2 pt-2 border-t border-slate-800/60 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Local Entity Count:</span>
                    <span className="font-mono font-bold text-white">{bank.local_entity_count} Accounts</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Session Status:</span>
                    <span className="font-mono text-emerald-400 font-semibold uppercase text-[11px]">
                      {bank.session_status}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Key Fingerprint:</span>
                    <span className="font-mono text-blue-300 text-[11px] bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                      {bank.key_fingerprint}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Group Parameters:</span>
                    <span className="font-mono text-slate-300 text-[11px]">{bank.group_type}</span>
                  </div>
                </div>

                <div className="p-3 bg-[#070b13] rounded-lg border border-slate-800/80 text-[11px] font-mono text-slate-400 space-y-1.5">
                  <span className="text-[10px] uppercase text-slate-500 block font-semibold">Masked Sample Records (Safe)</span>
                  {bank.sample_masked_records.map((s, idx) => (
                    <div key={idx} className="flex items-center justify-between text-slate-300">
                      <span>{s.account_id}</span>
                      <span className="text-slate-500">{s.masked_pan}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))
          ) : (
            <div className="col-span-3 p-8 text-center bg-[#0b111e] border border-slate-800 rounded-xl text-xs text-slate-500">
              Loading bank vault metadata from backend...
            </div>
          )}
        </div>
      </div>

      {/* 4. Privacy Boundary Visualization */}
      <div className="bg-[#0e1626] border border-slate-800 rounded-xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <h2 className="text-sm font-bold text-white flex items-center gap-2 font-mono uppercase tracking-wide">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              Privacy Boundary & Cryptographic Exchange Flow
            </h2>
            <p className="text-xs text-slate-400">
              End-to-end separation between private institutional data and cross-bank investigative graph linkages.
            </p>
          </div>
          <span className="text-[11px] font-mono text-slate-400 bg-slate-900 px-2.5 py-1 rounded border border-slate-800">
            Zero Raw Rosters Exchanged
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 pt-2">
          <div className="p-3.5 bg-[#090d16] border border-slate-800 rounded-lg space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white font-mono">1. Private Vaults</span>
              <Database className="w-3.5 h-3.5 text-blue-400" />
            </div>
            <p className="text-[11px] text-slate-400 leading-tight">
              Raw account records & KYC attributes remain strictly inside Apex, Beacon, and Crestline silos.
            </p>
          </div>

          <div className="p-3.5 bg-[#090d16] border border-slate-800 rounded-lg space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white font-mono">2. Group Blinding</span>
              <FileKey className="w-3.5 h-3.5 text-amber-400" />
            </div>
            <p className="text-[11px] text-slate-400 leading-tight">
              Identifiers mapped to subgroup $G_q$ and exponentiated with private keys ($H(x)^k \bmod p$).
            </p>
          </div>

          <div className="p-3.5 bg-[#090d16] border border-slate-800 rounded-lg space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white font-mono">3. Re-Blinding Exchange</span>
              <RefreshCw className="w-3.5 h-3.5 text-blue-400" />
            </div>
            <p className="text-[11px] text-slate-400 leading-tight">
              Shuffled blinded group elements cross-exponentiated: (H(x)^k_A)^k_B = (H(x)^k_B)^k_A mod p.
            </p>
          </div>

          <div className="p-3.5 bg-[#090d16] border border-emerald-500/30 bg-emerald-950/10 rounded-lg space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-emerald-300 font-mono">4. Graph Identity Link</span>
              <Network className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <p className="text-[11px] text-slate-300 leading-tight">
              Only matched common entities become identity edges in the Stage 2A/2B NetworkX graph.
            </p>
          </div>
        </div>
      </div>

      {/* 5. Cryptographic Stepper */}
      <div className="bg-[#0e1626] border border-slate-800 rounded-xl p-6 space-y-5">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <h2 className="text-sm font-bold text-white flex items-center gap-2 font-mono uppercase tracking-wide">
              <FileKey className="w-4 h-4 text-blue-400" />
              Diffie-Hellman Commutative PSI Stepper (Prototype)
            </h2>
            <p className="text-xs text-slate-400">
              Raw identifiers remain inside their respective bank vaults throughout the protocol.
            </p>
          </div>
          {psiResults && (
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-slate-400 bg-slate-900 px-2 py-1 rounded border border-slate-800">
                Session: {psiResults.session_id}
              </span>
              <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded border border-emerald-500/20 font-semibold">
                Latency: {psiResults.performance.total_elapsed_ms}ms
              </span>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
          {[
            {
              step: 1,
              title: "Step 1: Hash-to-Group",
              desc: "Deterministic mapping of raw entity to subgroup element",
              formula: "x -> H(x) mod p"
            },
            {
              step: 2,
              title: "Step 2: Local Blinding",
              desc: "Each bank applies ephemeral secret exponent k_i",
              formula: "U = H(x)^k_i mod p"
            },
            {
              step: 3,
              title: "Step 3: Shuffled Exchange",
              desc: "Blinded group elements exchanged with peers",
              formula: "Permute(U)"
            },
            {
              step: 4,
              title: "Step 4: Cross Re-Blinding",
              desc: "Peer banks apply second secret exponent",
              formula: "U' = (H(x)^k_A)^k_B mod p"
            },
            {
              step: 5,
              title: "Step 5: Set Intersection",
              desc: "Exact double-blinded match evaluated",
              formula: "I = U' ∩ V'"
            }
          ].map((s) => (
            <div
              key={s.step}
              className={`p-3.5 rounded-lg border transition-all ${
                activeStep >= s.step
                  ? 'bg-blue-600/10 border-blue-500/40 text-white'
                  : 'bg-slate-900/40 border-slate-800 text-slate-500'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold font-mono">{s.title}</span>
                {activeStep >= s.step && (
                  <CheckCircle2 className="w-3.5 h-3.5 text-blue-400" />
                )}
              </div>
              <p className="text-[11px] text-slate-400 leading-tight mb-2">{s.desc}</p>
              <span className="text-[10px] font-mono text-blue-300 bg-slate-900/90 px-1.5 py-0.5 rounded border border-slate-800 block text-center">
                {s.formula}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* 6. Pairwise Intersections & 7. Tripartite Common Entity Discovery */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Pairwise Results Cards */}
        <div className="lg:col-span-1 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white font-mono uppercase tracking-wide flex items-center gap-2">
              <Share2 className="w-4 h-4 text-blue-400" />
              Pairwise Intersections
            </h2>
            <span className="text-xs text-slate-500 font-mono">3 Bank Pairs</span>
          </div>

          <div className="space-y-3">
            {[
              { key: 'apex_beacon', name: 'Apex ↔ Beacon', data: pairwiseResults?.apex_beacon },
              { key: 'apex_crestline', name: 'Apex ↔ Crestline', data: pairwiseResults?.apex_crestline },
              { key: 'beacon_crestline', name: 'Beacon ↔ Crestline', data: pairwiseResults?.beacon_crestline },
            ].map((pair, idx) => (
              <div
                key={idx}
                className="bg-[#0b111e] border border-slate-800 rounded-xl p-4 space-y-2.5"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-white font-mono">{pair.name}</span>
                  <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    Status: Match Verified
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs pt-1 border-t border-slate-800/60">
                  <span className="text-slate-400">Intersection Count:</span>
                  <span className="font-mono font-bold text-blue-400">
                    {pair.data ? `${pair.data.intersection_count} Common Entities` : 'Run PSI to evaluate'}
                  </span>
                </div>
                <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono">
                  <span>Execution Time:</span>
                  <span>{pair.data ? `${pair.data.elapsed_ms}ms` : '—'}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Tripartite Discovery Overview */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white font-mono uppercase tracking-wide flex items-center gap-2">
              <Layers className="w-4 h-4 text-amber-400" />
              Common Entity Discovery (Apex ∩ Beacon ∩ Crestline)
            </h2>
            <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
              {tripartiteCommonEntities.length} Multi-Bank Syndicates
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {tripartiteCommonEntities.map((ent, idx) => (
              <div
                key={idx}
                className="bg-[#090d16] border border-slate-800/90 rounded-xl p-4 space-y-3 hover:border-amber-500/40 transition-all"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold font-mono px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">
                      {ent.associated_ring || 'CROSS-RING'}
                    </span>
                    <span className="text-[11px] font-mono text-slate-300">{ent.masked_entity_id}</span>
                  </div>
                  <span className="text-[10px] font-mono text-slate-500">Token: {ent.verification_token.substring(0, 8)}...</span>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center text-xs">
                  <div className="p-2 bg-slate-900/80 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-500 block uppercase font-mono">Apex</span>
                    <span className="font-mono text-blue-300 font-semibold">{ent.apex_account}</span>
                  </div>
                  <div className="p-2 bg-slate-900/80 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-500 block uppercase font-mono">Beacon</span>
                    <span className="font-mono text-blue-300 font-semibold">{ent.beacon_account}</span>
                  </div>
                  <div className="p-2 bg-slate-900/80 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-500 block uppercase font-mono">Crestline</span>
                    <span className="font-mono text-blue-300 font-semibold">{ent.crestline_account}</span>
                  </div>
                </div>

                <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/60 font-mono">
                  <span>Type: {ent.entity_type}</span>
                  <span className="text-emerald-400 flex items-center gap-1 text-[10px]">
                    <CheckCircle2 className="w-3 h-3" />
                    Identity Edge Ready
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 8. Detailed Pairwise Relationship Table with Bridge to Graph */}
      <div className="bg-[#0e1626] border border-slate-800 rounded-xl p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h2 className="text-sm font-bold text-white flex items-center gap-2 font-mono uppercase tracking-wide">
              <Network className="w-4 h-4 text-blue-400" />
              Cross-Bank Identity Linkages ({filteredMatches.length} Matches)
            </h2>
            <p className="text-xs text-slate-400">
              PSI matches represent cross-bank identity relationships, which are bridged to the graph engine for fraud analysis.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Filter by Ring */}
            <select
              value={filterRing}
              onChange={(e) => setFilterRing(e.target.value)}
              className="bg-slate-900 border border-slate-800 text-xs text-slate-200 rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500 font-mono"
            >
              <option value="ALL">All Rings</option>
              <option value="RING-001">RING-001 (Funnel)</option>
              <option value="RING-002">RING-002 (Star)</option>
              <option value="RING-003">RING-003 (Layering)</option>
              <option value="RING-004">RING-004 (Cycle)</option>
            </select>

            {/* Search */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search account / token..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="bg-slate-900 border border-slate-800 text-xs text-slate-200 rounded-lg pl-8 pr-3 py-1.5 focus:outline-none focus:border-blue-500 w-52 font-mono"
              />
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto border border-slate-800/80 rounded-lg">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-900/90 text-slate-400 border-b border-slate-800 font-mono">
                <th className="py-2.5 px-4">Verification Token</th>
                <th className="py-2.5 px-4">Party A (Account)</th>
                <th className="py-2.5 px-4">Party B (Account)</th>
                <th className="py-2.5 px-4">Entity Type</th>
                <th className="py-2.5 px-4">Syndicate Tag</th>
                <th className="py-2.5 px-4">Graph Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filteredMatches.slice(0, 15).map((m, idx) => (
                <tr key={idx} className="hover:bg-slate-800/30 transition-colors text-slate-300">
                  <td className="py-2.5 px-4 text-blue-300 font-semibold">{m.verification_token}</td>
                  <td className="py-2.5 px-4">
                    <span className="text-white font-semibold">{m.party_a_account}</span>
                    <span className="text-slate-500 text-[10px] block">{m.party_a_bank}</span>
                  </td>
                  <td className="py-2.5 px-4">
                    <span className="text-white font-semibold">{m.party_b_account}</span>
                    <span className="text-slate-500 text-[10px] block">{m.party_b_bank}</span>
                  </td>
                  <td className="py-2.5 px-4 text-slate-400 text-[11px]">{m.entity_type}</td>
                  <td className="py-2.5 px-4">
                    <span className="px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20 font-bold">
                      {m.associated_ring || 'CROSS-RING'}
                    </span>
                  </td>
                  <td className="py-2.5 px-4 text-emerald-400 flex items-center gap-1.5 py-3">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Identity Edge Ready</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filteredMatches.length === 0 && (
            <div className="p-8 text-center text-xs text-slate-500">
              No matching records found. Click <strong>RUN PSI</strong> to execute live multi-bank intersection.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
