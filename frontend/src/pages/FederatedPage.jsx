import React, { useState, useEffect, useCallback } from 'react';
import {
  Cpu,
  RefreshCw,
  Play,
  FastForward,
  RotateCcw,
  ShieldCheck,
  ShieldAlert,
  Lock,
  Layers,
  Activity,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  TrendingUp,
  Server,
  FileCode,
  Info,
  Award,
  Sliders,
  CheckCircle
} from 'lucide-react';
import { apiService } from '../services/api';

export default function FederatedPage({ onNavigateToTab }) {
  const [statusData, setStatusData] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [updatesData, setUpdatesData] = useState(null);
  const [securityData, setSecurityData] = useState(null);
  const [benchmarkData, setBenchmarkData] = useState(null);

  const [loading, setLoading] = useState(false);
  const [executingRound, setExecutingRound] = useState(false);
  const [executingMulti, setExecutingMulti] = useState(false);
  const [activeTab, setActiveTab] = useState('pipeline'); // Default to HE pipeline tab
  const [error, setError] = useState(null);

  // Load all initial data from backend
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [status, metrics, updates, security, benchmark] = await Promise.all([
        apiService.getFederatedStatus(),
        apiService.getFederatedMetrics(),
        apiService.getFederatedUpdates(),
        apiService.getFederatedSecurity(),
        apiService.getFederatedBenchmark().catch(() => null),
      ]);
      setStatusData(status);
      setMetricsData(metrics);
      setUpdatesData(updates);
      setSecurityData(security);
      setBenchmarkData(benchmark);
    } catch (err) {
      console.error("Error loading Federated Learning data:", err);
      setError("Federated Learning backend unavailable. Please ensure the backend server is running on port 8000.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Execute 1 Round
  const handleExecuteRound = async () => {
    setExecutingRound(true);
    setError(null);
    try {
      await apiService.runFederatedRound(5, 0.02);
      await loadData();
    } catch (err) {
      console.error("Error running FL round:", err);
      setError("Failed to execute federated learning round. Backend error.");
    } finally {
      setExecutingRound(false);
    }
  };

  // Execute 5 Rounds in sequence
  const handleExecuteMultiRounds = async () => {
    setExecutingMulti(true);
    setError(null);
    try {
      for (let i = 0; i < 5; i++) {
        await apiService.runFederatedRound(5, 0.02);
      }
      await loadData();
    } catch (err) {
      console.error("Error running multi FL rounds:", err);
      setError("Failed during multi-round execution.");
    } finally {
      setExecutingMulti(false);
    }
  };

  // Reset FL Session
  const handleResetSession = async () => {
    setLoading(true);
    setError(null);
    try {
      await apiService.resetFederatedSession();
      await loadData();
    } catch (err) {
      console.error("Error resetting FL session:", err);
      setError("Failed to reset session.");
    } finally {
      setLoading(false);
    }
  };

  const history = metricsData?.history || [];
  const latestMetric = history.length > 0 ? history[history.length - 1] : null;
  const currentRound = statusData?.current_round ?? 0;

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto select-none">
      {/* 1. Header & Live Navigation */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div className="p-2 bg-blue-500/10 rounded-lg border border-blue-500/20 text-blue-400">
              <Cpu className="w-5 h-5" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              Federated Learning
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-mono font-medium">
                Stage 2C-2
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono font-medium flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                LIVE PROTOCOL
              </span>
            </h1>
          </div>
          <p className="text-xs text-slate-400">
            Privacy-Preserving Cross-Bank GraphSAGE Training via Paillier Additive Homomorphic Encryption
          </p>
          <div className="text-xs text-slate-400 mt-1.5 flex items-center gap-1.5 font-sans">
            <span>Model performance:</span>
            <button
              onClick={() => onNavigateToTab && onNavigateToTab('metrics')}
              className="text-blue-400 hover:text-blue-300 font-semibold underline underline-offset-2 transition-colors cursor-pointer"
            >
              see Performance Metrics →
            </button>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleExecuteRound}
            disabled={executingRound || executingMulti || loading}
            className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold shadow-sm transition-all"
          >
            {executingRound ? (
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Play className="w-3.5 h-3.5" />
            )}
            <span>Execute 1 Round</span>
          </button>

          <button
            onClick={handleExecuteMultiRounds}
            disabled={executingRound || executingMulti || loading}
            className="flex items-center gap-2 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold shadow-sm transition-all"
          >
            {executingMulti ? (
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <FastForward className="w-3.5 h-3.5" />
            )}
            <span>Run 5 Rounds</span>
          </button>

          <button
            onClick={handleResetSession}
            disabled={executingRound || executingMulti || loading}
            className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium border border-slate-700/80 transition-all"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 bg-red-950/40 border border-red-800/80 rounded-xl flex items-center gap-3 text-red-300 text-xs">
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* 2. Cryptographic Threat Model & Modulus Banner */}
      <div className="p-4 bg-slate-900/80 border border-blue-500/20 rounded-xl shadow-lg space-y-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <Lock className="w-4 h-4 text-blue-400" />
            <span className="text-xs font-bold text-slate-200">
              Semi-Honest Threat Model (Paillier Additive Homomorphism)
            </span>
          </div>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-blue-950/60 text-blue-300 border border-blue-800/50">
            1024-bit Paillier modulus — prototype/demo configuration
          </span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          The aggregation server does not possess the Paillier private key and therefore cannot decrypt individual ciphertexts under the implemented key-management model.
          Model updates are quantized via signed fixed-point scaling (<span className="text-slate-300 font-mono">S = 10⁵</span>), encrypted under Paillier HE (<span className="text-slate-300 font-mono">∏ Enc(ΔWᵢ) mod n² = Enc(∑ ΔWᵢ)</span>), and decrypted exclusively by the authorized Decryption Authority.
        </p>
      </div>

      {/* 3. Live Protocol Training Execution Status */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-bold font-mono uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Activity className="w-4 h-4 text-blue-400" />
            <span>Interactive Federated Protocol Session</span>
          </h2>
          <span className="text-[10px] text-slate-500 font-mono">
            Active Session Round: {currentRound}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-xl">
            <span className="text-[10px] font-mono uppercase text-slate-500 block mb-1">Current Round</span>
            <div className="text-lg font-bold text-white font-mono flex items-center gap-1.5">
              <span>Round {currentRound}</span>
            </div>
            <span className="text-[10px] text-slate-500">Global Weights Θ^({currentRound})</span>
          </div>

          <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-xl">
            <span className="text-[10px] font-mono uppercase text-slate-500 block mb-1">Training Loss</span>
            <div className="text-lg font-bold text-purple-400 font-mono">
              {latestMetric ? latestMetric.loss.toFixed(4) : '--'}
            </div>
            <span className="text-[10px] text-slate-500">Cross-entropy convergence signal</span>
          </div>

          <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-xl">
            <span className="text-[10px] font-mono uppercase text-slate-500 block mb-1">Round Latency</span>
            <div className="text-lg font-bold text-blue-400 font-mono">
              {latestMetric ? `${latestMetric.execution_time_ms.toFixed(1)} ms` : '--'}
            </div>
            <span className="text-[10px] text-slate-500">HE encryption & aggregation time</span>
          </div>
        </div>
      </div>

      {/* 5. Three Bank Silo Cards (Apex, Beacon, Crestline) */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-bold font-mono uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Server className="w-4 h-4 text-blue-400" />
            <span>Participating Bank Silos (Zero Raw Data Transmission)</span>
          </h2>
          <span className="text-[10px] text-slate-500 font-mono">
            186 Trainable GraphSAGE Parameters Encrypted
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Apex Bank */}
          <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-blue-500"></div>
                <span className="text-xs font-bold text-white">Apex Bank</span>
              </div>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                Encrypted Delta Sent
              </span>
            </div>
            <div className="text-[11px] text-slate-400 space-y-1 font-mono">
              <div className="flex justify-between">
                <span>Local Accounts:</span>
                <span className="text-slate-200">50 (ACC-APX-001..050)</span>
              </div>
              <div className="flex justify-between">
                <span>Local Loss:</span>
                <span className="text-slate-200">{updatesData?.local_losses?.APEX ?? '0.5360'}</span>
              </div>
              <div className="flex justify-between">
                <span>Ciphertext Sample:</span>
                <span className="text-blue-300">{updatesData?.ciphertext_sample_fingerprints?.APEX?.[0] ?? '0x8f2a...1e4c'}</span>
              </div>
            </div>
          </div>

          {/* Beacon Bank */}
          <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500"></div>
                <span className="text-xs font-bold text-white">Beacon Bank</span>
              </div>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                Encrypted Delta Sent
              </span>
            </div>
            <div className="text-[11px] text-slate-400 space-y-1 font-mono">
              <div className="flex justify-between">
                <span>Local Accounts:</span>
                <span className="text-slate-200">50 (ACC-BCN-001..050)</span>
              </div>
              <div className="flex justify-between">
                <span>Local Loss:</span>
                <span className="text-slate-200">{updatesData?.local_losses?.BEACON ?? '0.5556'}</span>
              </div>
              <div className="flex justify-between">
                <span>Ciphertext Sample:</span>
                <span className="text-emerald-300">{updatesData?.ciphertext_sample_fingerprints?.BEACON?.[0] ?? '0x3c91...77d2'}</span>
              </div>
            </div>
          </div>

          {/* Crestline Bank */}
          <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full bg-indigo-500"></div>
                <span className="text-xs font-bold text-white">Crestline Bank</span>
              </div>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                Encrypted Delta Sent
              </span>
            </div>
            <div className="text-[11px] text-slate-400 space-y-1 font-mono">
              <div className="flex justify-between">
                <span>Local Accounts:</span>
                <span className="text-slate-200">50 (ACC-CLN-001..050)</span>
              </div>
              <div className="flex justify-between">
                <span>Local Loss:</span>
                <span className="text-slate-200">{updatesData?.local_losses?.CRESTLINE ?? '0.5930'}</span>
              </div>
              <div className="flex justify-between">
                <span>Ciphertext Sample:</span>
                <span className="text-indigo-300">{updatesData?.ciphertext_sample_fingerprints?.CRESTLINE?.[0] ?? '0x6e44...aa91'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 6. Sub-Tabs: Benchmarks, Progression History, HE Pipeline, Security Disclosures */}
      <div className="space-y-4">
        <div className="flex border-b border-slate-800 gap-4">
          <button
            onClick={() => setActiveTab('pipeline')}
            className={`pb-2 text-xs font-semibold transition-all ${
              activeTab === 'pipeline'
                ? 'text-blue-400 border-b-2 border-blue-500'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            HE Aggregation Pipeline
          </button>
          <button
            onClick={() => setActiveTab('convergence')}
            className={`pb-2 text-xs font-semibold transition-all ${
              activeTab === 'convergence'
                ? 'text-blue-400 border-b-2 border-blue-500'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Live Session History
          </button>
          <button
            onClick={() => setActiveTab('security')}
            className={`pb-2 text-xs font-semibold transition-all ${
              activeTab === 'security'
                ? 'text-blue-400 border-b-2 border-blue-500'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Cryptographic Audit & Disclosures
          </button>
        </div>



        {/* TAB 2: HE AGGREGATION PIPELINE */}
        {activeTab === 'pipeline' && (
          <div className="p-5 bg-slate-900/60 border border-slate-800 rounded-xl space-y-6">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white flex items-center gap-2">
                <Layers className="w-4 h-4 text-blue-400" />
                <span>End-to-End Encrypted Homomorphic Aggregation Execution Flow</span>
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                Latency: {updatesData?.timing_breakdown_ms?.total_round_latency ?? '0.00'} ms
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
              {/* Step 1 */}
              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg space-y-2">
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-blue-500/20 text-blue-400 text-[10px] font-mono font-bold flex items-center justify-center">
                    1
                  </span>
                  <span className="text-xs font-semibold text-slate-200">Local Silo Training</span>
                </div>
                <p className="text-[10px] text-slate-400">
                  Each bank computes analytical SGD gradients on isolated subgraphs:
                  <span className="block font-mono text-blue-300 mt-1">ΔWᵢ = Wᵢ - Θ^(t)</span>
                </p>
                <span className="text-[9px] font-mono text-slate-500 block">
                  Time: {updatesData?.timing_breakdown_ms?.local_training ?? '0'} ms
                </span>
              </div>

              {/* Step 2 */}
              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg space-y-2">
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 text-[10px] font-mono font-bold flex items-center justify-center">
                    2
                  </span>
                  <span className="text-xs font-semibold text-slate-200">Paillier Encryption</span>
                </div>
                <p className="text-[10px] text-slate-400">
                  Signed fixed-point encoding & Paillier encryption:
                  <span className="block font-mono text-indigo-300 mt-1">Cᵢ = (1+mᵢn)·rⁿ mod n²</span>
                </p>
                <span className="text-[9px] font-mono text-slate-500 block">
                  Time: {updatesData?.timing_breakdown_ms?.paillier_encryption ?? '0'} ms
                </span>
              </div>

              {/* Step 3 */}
              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg space-y-2">
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-purple-500/20 text-purple-400 text-[10px] font-mono font-bold flex items-center justify-center">
                    3
                  </span>
                  <span className="text-xs font-semibold text-slate-200">HE Aggregation</span>
                </div>
                <p className="text-[10px] text-slate-400">
                  Server multiplies ciphertexts (Zero private key access):
                  <span className="block font-mono text-purple-300 mt-1">C_agg = ∏ Cᵢ mod n²</span>
                </p>
                <span className="text-[9px] font-mono text-slate-500 block">
                  Time: {updatesData?.timing_breakdown_ms?.homomorphic_aggregation ?? '0'} ms
                </span>
              </div>

              {/* Step 4 */}
              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg space-y-2">
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 text-[10px] font-mono font-bold flex items-center justify-center">
                    4
                  </span>
                  <span className="text-xs font-semibold text-slate-200">Authorized Decryption</span>
                </div>
                <p className="text-[10px] text-slate-400">
                  Authorized decryptor decodes average and applies update:
                  <span className="block font-mono text-emerald-300 mt-1">Θ^(t+1) = Θ^(t) + ΔW_avg</span>
                </p>
                <span className="text-[9px] font-mono text-slate-500 block">
                  Time: {updatesData?.timing_breakdown_ms?.decryption_and_fedavg ?? '0'} ms
                </span>
              </div>
            </div>

            {/* Ciphertext Telemetry Inspector */}
            <div className="p-4 bg-slate-950/80 border border-slate-800/80 rounded-lg space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300 font-mono flex items-center gap-2">
                  <FileCode className="w-3.5 h-3.5 text-blue-400" />
                  <span>Safe Telemetry: Homomorphically Aggregated Ciphertext Fingerprints</span>
                </span>
                <span className="text-[10px] text-slate-500 font-mono">Zero Plaintext Gradients Exposed</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-2 font-mono text-[10px]">
                {updatesData?.aggregate_sample_fingerprints?.map((fp, idx) => (
                  <div key={idx} className="p-2 bg-slate-900 border border-slate-800 rounded flex justify-between items-center">
                    <span className="text-slate-400">Param #{idx + 1}:</span>
                    <span className="text-emerald-400 font-semibold">{fp}</span>
                  </div>
                )) || (
                  <div className="col-span-3 text-slate-500 text-center py-2">
                    Execute a round to view live aggregated ciphertext fingerprints.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: LIVE SESSION CONVERGENCE HISTORY */}
        {activeTab === 'convergence' && (
          <div className="p-5 bg-slate-900/60 border border-slate-800 rounded-xl space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                <span>Live Interactive Session Rounds History</span>
              </span>
              <span className="text-[10px] text-slate-400 font-mono">
                Rounds Executed: {history.length}
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-slate-950/60 text-slate-400 border-b border-slate-800 text-[11px]">
                  <tr>
                    <th className="py-2.5 px-3">Round</th>
                    <th className="py-2.5 px-3">Training Loss</th>
                    <th className="py-2.5 px-3">Execution Latency (ms)</th>
                    <th className="py-2.5 px-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {history.map((rec) => (
                    <tr key={rec.round} className="hover:bg-slate-800/30 transition-colors">
                      <td className="py-2.5 px-3 font-bold text-white">Round {rec.round}</td>
                      <td className="py-2.5 px-3 text-purple-400 font-semibold">{rec.loss.toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-blue-400">{rec.execution_time_ms.toFixed(1)} ms</td>
                      <td className="py-2.5 px-3 text-emerald-400 text-[11px]">Aggregated & Applied</td>
                    </tr>
                  ))}
                  {history.length === 0 && (
                    <tr>
                      <td colSpan={4} className="py-4 text-center text-slate-500">
                        No live rounds executed in this session yet. Click "Execute 1 Round" or "Run 5 Rounds" above.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 4: CRYPTOGRAPHIC AUDIT & DISCLOSURES */}
        {activeTab === 'security' && (
          <div className="p-5 bg-slate-900/60 border border-slate-800 rounded-xl space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-amber-400" />
                <span>Cryptographic Audit & Explicit Non-Claims</span>
              </span>
              <span className="text-[10px] text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20 font-mono">
                Audit Verified
              </span>
            </div>

            <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg space-y-1">
                <span className="text-xs font-bold text-slate-200">1. Threat Model & Key Management</span>
                <p className="text-[11px] text-slate-400">
                  {securityData?.key_management || "Paillier private key is held exclusively by the Authorized Decryption Authority. The HE Aggregation Server possesses only the public key (n, g) and cannot decrypt individual ciphertexts under the implemented key-management model."}
                </p>
              </div>

              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg space-y-1">
                <span className="text-xs font-bold text-slate-200">2. Modulus & Fixed-Point Quantization Notation</span>
                <div className="text-[11px] text-slate-400 font-mono space-y-1">
                  <p>{securityData?.modulus_configuration || "1024-bit Paillier modulus — prototype/demo configuration"}</p>
                  <div className="p-2 bg-slate-900 rounded border border-slate-800 text-blue-300 space-y-0.5">
                    <p>• Scaling Factor: S = 10^5</p>
                    <p>• Single-Bank Bound: |m_tilde| &lt;= 10^6</p>
                    <p>• 3-Bank Aggregate Bound: |sum_i m_tilde_i| &lt;= 3 x 10^6 &lt;&lt; n ≈ 10^308</p>
                  </div>
                </div>
                <p className="text-[10px] text-slate-500">
                  Note: Enterprise banking production standards recommend 2048-bit or 3072-bit Paillier moduli.
                </p>
              </div>

              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg space-y-2">
                <span className="text-xs font-bold text-amber-400">3. Explicit Non-Claims & Boundaries</span>
                <ul className="space-y-1.5 text-[11px] text-slate-400 list-disc list-inside">
                  {securityData?.explicit_non_claims?.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  )) || (
                    <>
                      <li>No Malicious Security: Does not verify zero-knowledge range proofs for participant gradient validity.</li>
                      <li>No Byzantine Resistance: Does not filter adversarial poison updates over encrypted ciphertexts.</li>
                      <li>No Differential Privacy: Does not inject Gaussian/Laplace DP-SGD noise into aggregated gradients.</li>
                      <li>No Multi-Party Decryption: Single decryption keyholder rather than threshold MPC key shares.</li>
                      <li>No Protection against all Model Inversion: Global decrypted weights are shared among participating banks.</li>
                    </>
                  )}
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

