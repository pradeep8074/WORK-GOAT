import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import OverviewPage from './pages/OverviewPage';
import TransactionsPage from './pages/TransactionsPage';
import AlertsPage from './pages/AlertsPage';
import NetworkPage from './pages/NetworkPage';
import InvestigationPage from './pages/InvestigationPage';
import PrivacyPage from './pages/PrivacyPage';
import FederatedPage from './pages/FederatedPage';
import PerformanceMetricsPage from './pages/PerformanceMetricsPage';
import PlaceholderPage from './pages/PlaceholderPage';
import TransactionDrawer from './components/TransactionDrawer';
import { apiService } from './services/api';
import { 
  FALLBACK_BANKS, 
  FALLBACK_TRANSACTIONS, 
  FALLBACK_ALERTS 
} from './data/fallbackData';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [banks, setBanks] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [backendHealthy, setBackendHealthy] = useState(false);
  const [selectedTransaction, setSelectedTransaction] = useState(null);

  // Check health and load initial data
  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    let isHealthy = false;
    try {
      const healthRes = await apiService.getHealth();
      if (healthRes && healthRes.status === 'healthy') {
        isHealthy = true;
      }
    } catch (err) {
      console.warn("Backend health check failed:", err.message);
      isHealthy = false;
    }
    setBackendHealthy(isHealthy);

    try {
      // Fetch all three main datasets in parallel
      const [banksData, txnsData, alertsData] = await Promise.all([
        apiService.getBanks(),
        apiService.getTransactions(),
        apiService.getAlerts(),
      ]);

      setBanks(banksData);
      setTransactions(txnsData);
      setAlerts(alertsData);
      setError(null);
    } catch (err) {
      console.warn("Error fetching data from backend, utilizing fallback simulation cache:", err.message);
      setError(`Failed to fetch live data: ${err.message}. Showing local prototype fallback data.`);
      // Use fallback prototype mock data so UI is testable even before backend is booted
      setBanks(FALLBACK_BANKS);
      setTransactions(FALLBACK_TRANSACTIONS);
      setAlerts(FALLBACK_ALERTS);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Handle transaction selection for drawer
  const handleSelectTransaction = (txn) => {
    setSelectedTransaction(txn);
  };

  const handleCloseDrawer = () => {
    setSelectedTransaction(null);
  };

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col antialiased">
      {/* Top Navbar */}
      <Navbar 
        backendHealthy={backendHealthy} 
        onCheckHealth={fetchData} 
      />

      {/* Main Layout Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <Sidebar 
          activeTab={activeTab} 
          setActiveTab={setActiveTab} 
        />

        {/* Content Container */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8 bg-[#090d16]">
          {activeTab === 'overview' && (
            <OverviewPage
              banks={banks}
              transactions={transactions}
              alerts={alerts}
              loading={loading}
              error={error}
              onRetry={fetchData}
              onSelectTransaction={handleSelectTransaction}
              onNavigateToTab={setActiveTab}
            />
          )}

          {activeTab === 'transactions' && (
            <TransactionsPage
              transactions={transactions}
              loading={loading}
              error={error}
              onRetry={fetchData}
              onSelectTransaction={handleSelectTransaction}
            />
          )}

          {activeTab === 'alerts' && (
            <AlertsPage
              alerts={alerts}
              loading={loading}
              error={error}
              onRetry={fetchData}
              onSelectTransaction={handleSelectTransaction}
              onNavigateToTab={setActiveTab}
            />
          )}

          {activeTab === 'network' && (
            <NetworkPage
              onSelectTransaction={handleSelectTransaction}
              onNavigateToTab={setActiveTab}
            />
          )}

          {activeTab === 'investigation' && (
            <InvestigationPage
              onSelectTransaction={handleSelectTransaction}
            />
          )}

          {activeTab === 'privacy' && (
            <PrivacyPage
              onNavigateToTab={setActiveTab}
            />
          )}

          {activeTab === 'federated' && (
            <FederatedPage
              onNavigateToTab={setActiveTab}
            />
          )}

          {activeTab === 'metrics' && (
            <PerformanceMetricsPage />
          )}
        </main>
      </div>

      {/* Slide-over Transaction Details Drawer */}
      <TransactionDrawer
        transaction={selectedTransaction}
        isOpen={Boolean(selectedTransaction)}
        onClose={handleCloseDrawer}
      />
    </div>
  );
}
