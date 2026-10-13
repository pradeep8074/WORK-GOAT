import React from 'react';
import { 
  Network, 
  Lock, 
  Cpu, 
  Search, 
  Layers, 
  Clock, 
  ArrowRight,
  ShieldAlert,
  GitBranch,
  Key
} from 'lucide-react';

const MODULE_CONFIG = {
  investigation: {
    title: "Deep Graph Investigation",
    tagline: "Cross-bank mule trail tracing and timeline reconstruction",
    badge: "Investigation Engine",
    icon: Search,
    stage2Note: "Deep multi-hop tracing and timeline analysis will be connected in Stage 2.",
    specs: [
      { label: "Target Architecture", value: "Multi-Hop Entity Linkage" },
      { label: "Data Source", value: "Cross-Bank Anonymized Subgraphs" },
      { label: "Analyst Workflow", value: "Case Dossier Export (SAR Format)" },
      { label: "Implementation Status", value: "Roadmap - Stage 2" }
    ],
    previewItems: [
      { title: "Entity Path Discovery", desc: "Traces fund displacement through intermediary mule nodes." },
      { title: "Timeline Correlator", desc: "Reconstructs coordinated transfers occurring within tight delta windows." },
      { title: "Regulatory Case Dossier", desc: "Automated SAR-ready documentation without leaking peer bank identity." }
    ]
  },
  network: {
    title: "Network Analysis & Graph Intelligence",
    tagline: "Cross-bank graph neural networks (GNN / GraphSAGE)",
    badge: "Graph Intelligence",
    icon: Network,
    stage2Note: "Cross-bank graph intelligence will be connected in Stage 2.",
    specs: [
      { label: "Planned Model", value: "Heterogeneous GraphSAGE / GNN" },
      { label: "Graph Topology", value: "Distributed Multi-Bank Graph" },
      { label: "Target Detection", value: "Mule Ring Topologies & Star Patterns" },
      { label: "Implementation Status", value: "Roadmap - Stage 2" }
    ],
    previewItems: [
      { title: "Multi-Institution Subgraph Matching", desc: "Matches transaction chains crossing institutional borders." },
      { title: "Fan-In / Fan-Out Cluster Scoring", desc: "Identifies funnel accounts receiving funds from dispersed sources." },
      { title: "Dynamic Edge Weighting", desc: "Weight calculations based on temporal velocity and denomination." }
    ]
  },
  privacy: {
    title: "Privacy & Cryptographic Layer",
    tagline: "Private Set Intersection (PSI) and Homomorphic Encryption",
    badge: "Cryptographic Core",
    icon: Lock,
    stage2Note: "Cryptographic privacy protocols (PSI & Paillier HE) will be connected in Stage 2.",
    specs: [
      { label: "Privacy Protocol", value: "Private Set Intersection (PSI)" },
      { label: "Encryption", value: "Paillier Homomorphic Encryption" },
      { label: "Compliance Model", value: "Zero-Knowledge Account Hashing" },
      { label: "Implementation Status", value: "Roadmap - Stage 2" }
    ],
    previewItems: [
      { title: "Private Set Intersection (PSI)", desc: "Allows banks to discover common suspicious counterparties without revealing their customer bases." },
      { title: "Homomorphic Risk Aggregation", desc: "Aggregates risk score weights over ciphertext without decrypting individual bank weights." },
      { title: "Blind Identifier Tokens", desc: "One-way cryptographic salt tokens protecting account numbers across queries." }
    ]
  },
  federated: {
    title: "Federated Learning Orchestration",
    tagline: "Collaborative fraud model training without centralizing raw data",
    badge: "Federated Training",
    icon: Cpu,
    stage2Note: "Federated Learning aggregation server and client nodes will be connected in Stage 2.",
    specs: [
      { label: "Federated Strategy", value: "Federated Averaging (FedAvg) with Secure Aggregation" },
      { label: "Client Deployment", value: "Local Bank Secure Enclaves" },
      { label: "Differential Privacy", value: "Renyi Differential Privacy (RDP)" },
      { label: "Implementation Status", value: "Roadmap - Stage 2" }
    ],
    previewItems: [
      { title: "Secure Model Aggregator", desc: "Centralized parameter coordination without raw dataset access." },
      { title: "Local Gradient Verification", desc: "Verifies model updates against poisoning attacks before merging." },
      { title: "Differential Noise Injection", desc: "Ensures model weights cannot be reversed to extract training samples." }
    ]
  }
};

export default function PlaceholderPage({ moduleId }) {
  const config = MODULE_CONFIG[moduleId] || MODULE_CONFIG.network;
  const Icon = config.icon;

  return (
    <div className="space-y-8 max-w-5xl mx-auto py-2">
      {/* Planned Notice Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 flex items-start gap-4">
        <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400">
          <Icon className="w-6 h-6" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-mono uppercase px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 font-semibold">
              Coming in Stage 2
            </span>
            <span className="text-xs text-slate-500 font-mono">Architectural Specification</span>
          </div>
          <h2 className="text-lg font-bold text-white tracking-tight">{config.title}</h2>
          <p className="text-xs text-slate-400 mt-0.5">{config.tagline}</p>
        </div>
      </div>

      {/* Main Roadmap Notice */}
      <div className="bg-[#0e1626] border border-slate-800 rounded-xl p-6 text-center space-y-3">
        <span className="text-xs font-mono uppercase text-blue-400 tracking-wider font-semibold">
          Phase Milestone
        </span>
        <h3 className="text-xl font-bold text-white max-w-lg mx-auto">
          "{config.stage2Note}"
        </h3>
        <p className="text-xs text-slate-400 max-w-xl mx-auto leading-relaxed">
          Stage 1 establishes the operational API contracts and dashboard architecture. Advanced computational graph engines and cryptographic protocols will be integrated incrementally in Stage 2.
        </p>
      </div>

      {/* Technical Specifications Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {config.specs.map((spec, idx) => (
          <div key={idx} className="bg-[#0b111e] border border-slate-800/80 rounded-xl p-4 flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">{spec.label}</span>
            <span className="text-xs font-mono text-slate-200 font-semibold bg-slate-900 px-2.5 py-1 rounded border border-slate-800">
              {spec.value}
            </span>
          </div>
        ))}
      </div>

      {/* Preview Functional Modules (Disabled / Spec) */}
      <div className="bg-[#0e1626] border border-slate-800 rounded-xl p-6">
        <h4 className="text-xs uppercase font-mono font-semibold text-slate-400 tracking-wider mb-4">
          Planned Stage 2 Capabilities
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {config.previewItems.map((item, idx) => (
            <div key={idx} className="p-4 bg-[#080d16] border border-slate-800/70 rounded-lg space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200">{item.title}</span>
                <span className="text-[10px] font-mono text-slate-500">Preview</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
