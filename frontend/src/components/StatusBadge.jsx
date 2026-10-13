import React from 'react';

export default function StatusBadge({ status, type = 'status' }) {
  if (!status) return null;

  const getStyle = () => {
    const s = String(status).toLowerCase();
    
    if (s === 'normal' || s === 'connected' || s === 'active' || s === 'resolved') {
      return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
    }
    if (s === 'review' || s === 'under review' || s === 'medium' || s === 'pending') {
      return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
    }
    if (s === 'high risk' || s === 'high' || s === 'critical' || s === 'escalated') {
      return 'bg-rose-500/10 text-rose-400 border-rose-500/20';
    }
    if (s === 'open') {
      return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
    }
    return 'bg-slate-700/30 text-slate-300 border-slate-700/50';
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border font-mono tracking-wide ${getStyle()}`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current opacity-80" />
      {status}
    </span>
  );
}
