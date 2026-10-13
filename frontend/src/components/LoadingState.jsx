import React from 'react';
import { Loader2 } from 'lucide-react';

export default function LoadingState({ message = "Retrieving synthetic dataset..." }) {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center">
      <div className="p-3 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 mb-3 animate-pulse">
        <Loader2 className="w-6 h-6 animate-spin text-blue-400" />
      </div>
      <p className="text-sm text-slate-300 font-medium">{message}</p>
      <span className="text-xs text-slate-500 mt-1 font-mono">Simulating isolated endpoint query</span>
    </div>
  );
}
