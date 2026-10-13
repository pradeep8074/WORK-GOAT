import React from 'react';
import { Shield, ShieldAlert, Wifi, WifiOff } from 'lucide-react';

export default function Navbar({ backendHealthy, onCheckHealth }) {
  return (
    <header className="h-16 border-b border-slate-800/80 bg-[#080d16]/90 backdrop-blur sticky top-0 z-30 flex items-center justify-between px-6">
      {/* Brand & Subtitle */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-white shadow-lg shadow-blue-900/20 border border-blue-500/30">
            <Shield className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold tracking-tight text-white font-mono">
                G.O.A.T. Shield
              </h1>
              <span className="text-[10px] font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                PROTOTYPE
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Privacy-Preserving Cross-Bank Fraud Intelligence
            </p>
          </div>
        </div>
      </div>

      {/* Right status & indicators */}
      <div className="flex items-center gap-3">
        {/* Stage 1 badge indicator */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-400">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
          Stage 1 Foundation
        </div>

        {/* Backend health status badge */}
        <button
          onClick={onCheckHealth}
          title="Click to re-check FastAPI backend health"
          className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium border transition-colors ${
            backendHealthy
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20'
              : 'bg-rose-500/10 text-rose-400 border-rose-500/20 hover:bg-rose-500/20'
          }`}
        >
          {backendHealthy ? (
            <>
              <Wifi className="w-3.5 h-3.5" />
              <span>API Online</span>
            </>
          ) : (
            <>
              <WifiOff className="w-3.5 h-3.5" />
              <span>API Offline</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
}
