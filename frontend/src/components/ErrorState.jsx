import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default function ErrorState({ message, onRetry, isFallback = false }) {
  return (
    <div className="bg-rose-950/20 border border-rose-800/40 rounded-xl p-6 text-center max-w-lg mx-auto my-6">
      <div className="w-12 h-12 mx-auto rounded-full bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-400 mb-4">
        <AlertTriangle className="w-6 h-6" />
      </div>
      
      <h3 className="text-base font-semibold text-white mb-1">
        {isFallback ? "Using Local Fallback Data" : "Unable to Connect to Backend API"}
      </h3>
      
      <p className="text-xs text-slate-300 mb-4 font-mono leading-relaxed">
        {message || "Could not reach FastAPI server at http://localhost:8000."}
      </p>

      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-rose-600 hover:bg-rose-500 rounded-lg shadow-sm transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Retry Connection
        </button>
      )}
    </div>
  );
}
