import React, { useEffect } from 'react';
import { X, ShieldAlert, CheckCircle, ArrowRight, Copy, Clock, Layers, IndianRupee } from 'lucide-react';
import StatusBadge from './StatusBadge';
import { formatINR } from '../utils/currency';

export default function TransactionDrawer({ transaction, isOpen, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !transaction) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black/70 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      <div className="fixed inset-y-0 right-0 max-w-full flex pl-10">
        <div className="w-screen max-w-md bg-[#0b111e] border-l border-slate-800 shadow-2xl flex flex-col">
          {/* Drawer Header */}
          <div className="p-6 border-b border-slate-800 flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-[11px] font-mono uppercase px-2 py-0.5 rounded ${
                  transaction.id?.startsWith('ALT-LIVE') 
                    ? 'bg-rose-950/80 text-rose-300 border border-rose-800/50'
                    : transaction.id?.startsWith('TXN-LIVE')
                    ? 'bg-cyan-950/80 text-cyan-300 border border-cyan-800/50'
                    : 'bg-slate-800 text-slate-300'
                }`}>
                  {transaction.id?.startsWith('ALT-LIVE') ? 'Live Security Alert' : transaction.id?.startsWith('TXN-LIVE') ? 'Live Transaction Feed' : 'Synthetic Record'}
                </span>
                <StatusBadge status={transaction.status} />
              </div>
              <h3 className="text-lg font-bold text-white font-mono">{transaction.id}</h3>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              title="Close panel (Esc)"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Drawer Body */}
          <div className="p-6 flex-1 overflow-y-auto space-y-6">
            {/* Amount Callout */}
            <div className="bg-[#121c2e] border border-slate-800 rounded-xl p-5 text-center">
              <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold block mb-1">
                Transaction Value
              </span>
              <div className="text-3xl font-extrabold text-white font-mono flex items-center justify-center gap-1.5">
                <IndianRupee className="w-6 h-6 text-blue-400" />
                <span>{formatINR(transaction.amount).replace('₹', '').trim()}</span>
                <span className="text-sm font-normal text-slate-400 ml-1 font-sans">{transaction.currency || 'INR'}</span>
              </div>
              <span className="inline-block mt-2 text-[11px] font-mono text-slate-400 bg-slate-900/80 px-2.5 py-0.5 rounded border border-slate-800">
                Type: {transaction.type || 'Standard Transfer'}
              </span>
            </div>

            {/* Counterparties Flow */}
            <div className="bg-[#0e1626] border border-slate-800/80 rounded-xl p-4 space-y-4">
              <h4 className="text-xs uppercase font-semibold text-slate-400 tracking-wider">
                Simulated Flow Route
              </h4>

              {/* Sender */}
              <div className="p-3 bg-[#080d16] border border-slate-800 rounded-lg">
                <span className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">Origin / Sender</span>
                <p className="text-sm font-mono text-slate-200 font-medium break-all">{transaction.sender}</p>
                <div className="mt-1 flex items-center justify-between text-xs text-slate-400">
                  <span>Host: {transaction.bank}</span>
                  <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">Verified Format</span>
                </div>
              </div>

              <div className="flex justify-center my-1 text-slate-500">
                <ArrowRight className="w-5 h-5 rotate-90" />
              </div>

              {/* Receiver */}
              <div className="p-3 bg-[#080d16] border border-slate-800 rounded-lg">
                <span className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">Beneficiary / Receiver</span>
                <p className="text-sm font-mono text-slate-200 font-medium break-all">{transaction.receiver}</p>
                <div className="mt-1 flex items-center justify-between text-xs text-slate-400">
                  <span>Cross-Institution Transfer</span>
                  <span className="text-[10px] text-blue-400 bg-blue-500/10 px-1.5 py-0.5 rounded border border-blue-500/20">Inter-Bank Mock</span>
                </div>
              </div>
            </div>

            {/* Metadata Table */}
            <div className="space-y-3">
              <h4 className="text-xs uppercase font-semibold text-slate-400 tracking-wider">
                Audit Details
              </h4>
              <div className="bg-[#0e1626] border border-slate-800/80 rounded-xl divide-y divide-slate-800/70 text-xs">
                <div className="p-3 flex justify-between">
                  <span className="text-slate-400">Institution</span>
                  <span className="text-white font-medium">{transaction.bank}</span>
                </div>
                <div className="p-3 flex justify-between">
                  <span className="text-slate-400">Timestamp</span>
                  <span className="text-slate-300 font-mono">{transaction.timestamp}</span>
                </div>
                <div className="p-3 flex justify-between">
                  <span className="text-slate-400">Engine Risk Score</span>
                  <span className={`font-mono font-semibold ${
                    (transaction.risk_score || 0) >= 65 ? 'text-rose-400' :
                    (transaction.risk_score || 0) >= 30 ? 'text-amber-400' : 'text-emerald-400'
                  }`}>
                    {transaction.risk_score !== undefined && transaction.risk_score !== null 
                      ? `${transaction.risk_score} / 100` 
                      : (transaction.status === 'High Risk' ? '85 / 100' : transaction.status === 'Review' ? '50 / 100' : '15 / 100')}
                  </span>
                </div>
                <div className="p-3 flex justify-between">
                  <span className="text-slate-400">Status / Feed</span>
                  <span className={transaction.id?.startsWith('TXN-LIVE') ? 'text-cyan-400 font-mono text-[11px]' : 'text-emerald-400'}>
                    {transaction.id?.startsWith('TXN-LIVE') ? 'Real-Time Live Stream' : 'Ledger Snapshot'}
                  </span>
                </div>
                {transaction.triggered_rules && transaction.triggered_rules.length > 0 && (
                  <div className="p-3 flex justify-between items-center">
                    <span className="text-slate-400">Triggered Rules</span>
                    <div className="flex gap-1 flex-wrap justify-end">
                      {transaction.triggered_rules.map(r => (
                        <span key={r} className="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30 font-mono text-[10px]">
                          {r}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Simulated / Live Note & Explanation */}
            {(transaction.explanation || transaction.note) && (
              <div className="p-3.5 bg-blue-950/30 border border-blue-900/40 rounded-lg text-xs text-blue-300 space-y-1">
                <span className="font-semibold block uppercase text-[10px] tracking-wider text-blue-400">
                  {transaction.id?.startsWith('TXN-LIVE') ? 'Detection Engine Audit' : 'Annotation'}
                </span>
                <p>{transaction.explanation || transaction.note}</p>
              </div>
            )}

            <div className="p-3 bg-slate-900/50 border border-slate-800 rounded-lg text-[11px] text-slate-400">
              <p>
                <strong>Disclaimer:</strong> This transaction and its counterparties are synthetic mock data created solely for UI demonstration. No real financial accounts are represented.
              </p>
            </div>
          </div>

          {/* Drawer Footer */}
          <div className="p-4 border-t border-slate-800 bg-[#080d16] flex items-center justify-end">
            <button
              onClick={onClose}
              className="px-4 py-2 text-xs font-semibold text-slate-300 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
            >
              Close Details
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
