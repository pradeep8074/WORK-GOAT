// Stage 1 Local Fallback Data (Synthetic INR)
export const FALLBACK_BANKS = [
  {
    id: "BANK-A",
    name: "Bank A (Apex National Bank)",
    short_name: "Bank A",
    code: "APEX-IN",
    status: "Connected",
    transaction_count: 14820,
    active_alerts: 4,
    high_risk_accounts: 3,
    data_isolation: "ACTIVE",
    node_type: "Simulated Partner Node",
    environment: "Simulated bank environment",
    last_sync: "2026-09-17T10:30:00Z"
  },
  {
    id: "BANK-B",
    name: "Bank B (Beacon Mutual Trust)",
    short_name: "Bank B",
    code: "BCON-IN",
    status: "Connected",
    transaction_count: 11350,
    active_alerts: 3,
    high_risk_accounts: 2,
    data_isolation: "ACTIVE",
    node_type: "Simulated Partner Node",
    environment: "Simulated bank environment",
    last_sync: "2026-09-17T10:29:15Z"
  },
  {
    id: "BANK-C",
    name: "Bank C (Crestline Commercial)",
    short_name: "Bank C",
    code: "CRST-IN",
    status: "Connected",
    transaction_count: 9410,
    active_alerts: 5,
    high_risk_accounts: 4,
    data_isolation: "ACTIVE",
    node_type: "Simulated Partner Node",
    environment: "Simulated bank environment",
    last_sync: "2026-09-17T10:31:40Z"
  }
];

export const FALLBACK_TRANSACTIONS = [
  {
    id: "TXN-84912",
    bank: "Bank A",
    sender: "ACC-49102-A (Syn-Sender-01)",
    receiver: "ACC-88319-B (Syn-Receiver-08)",
    amount: 985000.00,
    currency: "INR",
    timestamp: "2026-09-17 10:24:12 IST",
    type: "Instant Payment (IMPS)",
    status: "High Risk",
    risk_score: 88,
    note: "Rapid succession transfer near ₹10,00,000 FIU-IND reporting threshold (Synthetic)"
  },
  {
    id: "TXN-84911",
    bank: "Bank C",
    sender: "ACC-71204-C (Syn-Sender-14)",
    receiver: "ACC-49102-A (Syn-Sender-01)",
    amount: 950000.00,
    currency: "INR",
    timestamp: "2026-09-17 10:18:45 IST",
    type: "RTGS",
    status: "High Risk",
    risk_score: 91,
    note: "Immediate pass-through funds pattern across bank boundary (Synthetic)"
  },
  {
    id: "TXN-84910",
    bank: "Bank B",
    sender: "ACC-33109-B (Syn-Sender-05)",
    receiver: "ACC-10492-C (Syn-Receiver-22)",
    amount: 185000.00,
    currency: "INR",
    timestamp: "2026-09-17 10:12:08 IST",
    type: "NEFT",
    status: "Review",
    risk_score: 64,
    note: "Dormant account sudden outflow spike (Synthetic)"
  },
  {
    id: "TXN-84909",
    bank: "Bank A",
    sender: "ACC-18492-A (Syn-Corporate-02)",
    receiver: "ACC-66190-A (Syn-Vendor-11)",
    amount: 1250000.00,
    currency: "INR",
    timestamp: "2026-09-17 10:05:31 IST",
    type: "RTGS",
    status: "Normal",
    risk_score: 12,
    note: "Standard vendor scheduled disbursement (Synthetic)"
  }
];

export const FALLBACK_ALERTS = [
  {
    id: "ALT-9041",
    account: "ACC-88319-B",
    bank: "Bank B",
    risk: "Critical",
    reason: "Unusual transaction velocity & multi-bank counterparty concentration",
    status: "Open",
    created: "2026-09-17 10:25:15 IST",
    rule_flag: "MOCK-VELOCITY-04",
    details: "Synthetic alert: Account received 4 rapid transfers totaling ₹37,45,000 across Bank A and Bank C within 90 minutes."
  },
  {
    id: "ALT-9042",
    account: "ACC-49102-A",
    bank: "Bank A",
    risk: "High",
    reason: "Rapid succession structuring near reporting thresholds",
    status: "Under Review",
    created: "2026-09-17 10:20:00 IST",
    rule_flag: "MOCK-STRUCT-01",
    details: "Synthetic alert: Transfer amount (₹9,85,000) positioned right below ₹10,00,000 regulatory reporting threshold."
  }
];

// Stage 2A Offline Fallback Structures (explicit offline notice)
export const FALLBACK_SIMULATION = {
  isOfflineFallback: true,
  total: 4,
  offset: 0,
  limit: 100,
  dataset_total: 4,
  notice: "Backend offline — Live simulation and detection engine unavailable.",
  transactions: [
    {
      id: "TXN-S2A-DEMO-01",
      bank: "Apex Bank",
      sender: "ACC-APX-003",
      receiver: "ACC-BCN-008",
      amount: 945000.00,
      currency: "INR",
      timestamp: "2026-09-17 10:24:12 IST",
      iso_timestamp: "2026-09-17T10:24:12",
      type: "IMPS",
      is_cross_bank: true,
      is_mule: true,
      mule_ring_id: "RING-001",
      pattern_type: "Fan-In Funnel",
      risk_score: 70,
      status: "High Risk",
      triggered_rules: ["R1", "R2", "R3", "R5"],
      explanation: "Offline preview sample. Live detection unavailable."
    },
    {
      id: "TXN-S2A-DEMO-02",
      bank: "Apex Bank",
      sender: "ACC-APX-015",
      receiver: "ACC-BCN-021",
      amount: 890000.00,
      currency: "INR",
      timestamp: "2026-09-17 11:15:00 IST",
      iso_timestamp: "2026-09-17T11:15:00",
      type: "RTGS",
      is_cross_bank: true,
      is_mule: true,
      mule_ring_id: "RING-002",
      pattern_type: "Fan-Out Dispersion",
      risk_score: 65,
      status: "High Risk",
      triggered_rules: ["R1", "R2", "R4", "R6"],
      explanation: "Offline preview sample. Live detection unavailable."
    }
  ]
};

export const FALLBACK_GRAPH = {
  isOfflineFallback: true,
  notice: "Backend offline — Connect backend to render live NetworkX multi-bank graph topology.",
  nodes: [],
  edges: [],
  stats: {
    total_nodes: 0,
    total_edges: 0,
    displayed_nodes: 0,
    displayed_edges: 0,
    cross_bank_edges: 0,
    mule_nodes: 0,
    mule_edges: 0,
    graph_density: 0.0,
    connected_subgraphs: 0,
    top_inbound_hubs: [],
    top_outbound_sources: []
  }
};

export const FALLBACK_METRICS = {
  isOfflineFallback: true,
  notice: "Backend offline — Live precision/recall/F1 metrics unavailable.",
  total_transactions: 0,
  ground_truth: {
    mule_transactions: 0,
    normal_transactions: 0,
    mule_percentage: 0
  },
  detection_summary: {
    high_risk: 0,
    review: 0,
    normal: 0,
    threshold_used: 65
  },
  metrics: {
    precision: 0.0,
    recall: 0.0,
    f1_score: 0.0,
    accuracy: 0.0,
    precision_percent: 0.0,
    recall_percent: 0.0,
    f1_percent: 0.0,
    accuracy_percent: 0.0
  },
  confusion_matrix: {
    true_positives: 0,
    false_positives: 0,
    true_negatives: 0,
    false_negatives: 0
  },
  per_ring_breakdown: [],
  rule_frequencies: { R1: 0, R2: 0, R3: 0, R4: 0, R5: 0, R6: 0 }
};
