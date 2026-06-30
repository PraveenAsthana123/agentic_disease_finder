'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLORS = { ok: 'success', warning: 'warning', exceeded: 'danger', critical: 'danger' };

export default function TokenCostPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [budget, setBudget] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/token-cost/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/token-cost/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/token-cost/budget`).then(r => r.json()).then(setBudget).catch(() => {});
    fetch(`${API}/api/token-cost/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = overview.summary || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'tokens', label: 'Token Usage' },
    { id: 'budget', label: 'Budget' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f4b0; Token / Cost Dashboard</h3>
      <p className="text-muted">LLM token usage, operation costs, budget tracking, and savings from clinical.db</p>

      {/* Summary cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Tokens', value: (s.total_tokens || 0).toLocaleString(), color: 'primary' },
          { label: 'Operations', value: s.total_operations || 0, color: 'info' },
          { label: 'Inferences', value: s.model_inferences || 0, color: 'success' },
          { label: 'Est. Cost', value: `$${(s.total_cost_usd || 0).toFixed(2)}`, color: 'warning' },
          { label: 'Budget Used', value: `${s.budget_utilization_pct || 0}%`, color: s.budget_utilization_pct >= 80 ? 'danger' : 'success' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header bg-primary text-white">LLM Token Summary</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <tbody>
                    <tr><td>Conversations</td><td className="text-end fw-bold">{overview.llm_usage?.conversations || 0}</td></tr>
                    <tr><td>Input Tokens</td><td className="text-end fw-bold">{(s.input_tokens || 0).toLocaleString()}</td></tr>
                    <tr><td>Output Tokens</td><td className="text-end fw-bold">{(s.output_tokens || 0).toLocaleString()}</td></tr>
                    <tr><td>Model</td><td className="text-end"><span className="badge bg-success">{overview.llm_usage?.model_label || 'local'}</span></td></tr>
                    <tr><td>Cloud Equivalent Cost</td><td className="text-end text-muted">${(overview.llm_usage?.hypothetical_cloud_cost || 0).toFixed(4)}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header bg-info text-white">Operations by Component</div>
              <div className="card-body" style={{ maxHeight: 280, overflowY: 'auto' }}>
                <table className="table table-sm mb-0">
                  <thead><tr><th>Component</th><th className="text-end">Ops</th></tr></thead>
                  <tbody>
                    {Object.entries(overview.operations?.by_component || {}).map(([k, v]) => (
                      <tr key={k}><td>{k}</td><td className="text-end">{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header bg-dark text-white">Daily Trend (14 days)</div>
              <div className="card-body">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Date</th><th className="text-end">Input Tok</th><th className="text-end">Output Tok</th><th className="text-end">Ops</th><th className="text-end">Cost</th></tr></thead>
                    <tbody>
                      {(overview.daily_trend || []).map(d => (
                        <tr key={d.date}>
                          <td>{d.date}</td>
                          <td className="text-end">{d.input_tokens.toLocaleString()}</td>
                          <td className="text-end">{d.output_tokens.toLocaleString()}</td>
                          <td className="text-end">{d.operations}</td>
                          <td className="text-end">${d.estimated_cost.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Token Usage Tab */}
      {tab === 'tokens' && breakdown && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header bg-primary text-white">Tokens by Role</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Role</th><th className="text-end">Tokens</th><th className="text-end">Messages</th></tr></thead>
                  <tbody>
                    {(breakdown.token_breakdown?.by_role || []).map(r => (
                      <tr key={r.role}>
                        <td><span className={`badge bg-${r.role === 'assistant' ? 'success' : r.role === 'operator' ? 'warning' : 'secondary'}`}>{r.role}</span></td>
                        <td className="text-end">{r.tokens.toLocaleString()}</td>
                        <td className="text-end">{r.messages}</td>
                      </tr>
                    ))}
                    <tr className="table-dark">
                      <td><strong>Total</strong></td>
                      <td className="text-end"><strong>{(breakdown.token_breakdown?.total_tokens || 0).toLocaleString()}</strong></td>
                      <td></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header bg-warning text-dark">Cost by Component</div>
              <div className="card-body" style={{ maxHeight: 350, overflowY: 'auto' }}>
                <table className="table table-sm mb-0">
                  <thead><tr><th>Component</th><th className="text-end">Ops</th><th className="text-end">Cost</th></tr></thead>
                  <tbody>
                    {(breakdown.component_breakdown || []).map(c => (
                      <tr key={c.component}>
                        <td>{c.component}</td>
                        <td className="text-end">{c.operations}</td>
                        <td className="text-end">${c.cost.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header bg-success text-white">Model Inference Breakdown</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Disease</th><th className="text-end">Inferences</th><th className="text-end">Avg Confidence</th><th className="text-end">Cost</th></tr></thead>
                  <tbody>
                    {(breakdown.model_breakdown || []).map(m => (
                      <tr key={m.disease}>
                        <td>{m.disease}</td>
                        <td className="text-end">{m.inferences}</td>
                        <td className="text-end">{(m.avg_confidence * 100).toFixed(1)}%</td>
                        <td className="text-end">${m.cost.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Budget Tab */}
      {tab === 'budget' && budget && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className={`card-header bg-${STATUS_COLORS[budget.status] || 'secondary'} text-white`}>
                Budget Status: {budget.status?.toUpperCase()}
              </div>
              <div className="card-body">
                <div className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span>Monthly Budget</span>
                    <strong>${budget.total_spent_usd?.toFixed(2)} / ${budget.total_budget_usd}</strong>
                  </div>
                  <div className="progress" style={{ height: 24 }}>
                    <div
                      className={`progress-bar bg-${budget.utilization_pct >= 80 ? 'danger' : budget.utilization_pct >= 50 ? 'warning' : 'success'}`}
                      style={{ width: `${Math.min(budget.utilization_pct || 0, 100)}%` }}
                    >
                      {budget.utilization_pct}%
                    </div>
                  </div>
                </div>
                <table className="table table-sm">
                  <thead><tr><th>Category</th><th className="text-end">Budget</th><th className="text-end">Spent</th><th className="text-end">%</th></tr></thead>
                  <tbody>
                    {(budget.categories || []).map(c => (
                      <tr key={c.category}>
                        <td>{c.category}</td>
                        <td className="text-end">${c.budget_usd}</td>
                        <td className="text-end">${c.spent_usd?.toFixed(4)}</td>
                        <td className="text-end"><span className={`badge bg-${c.pct >= 80 ? 'danger' : c.pct >= 50 ? 'warning' : 'success'}`}>{c.pct}%</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header bg-dark text-white">Alerts</div>
              <div className="card-body">
                {(budget.alerts || []).map((a, i) => (
                  <div key={i} className={`alert alert-${a.level === 'ok' ? 'success' : a.level === 'warning' ? 'warning' : 'danger'} py-2 mb-2`}>
                    {a.level === 'ok' ? '\u2705' : a.level === 'warning' ? '\u26a0\ufe0f' : '\ud83d\udea8'} {a.message}
                  </div>
                ))}
              </div>
            </div>
            <div className="card shadow-sm mt-3">
              <div className="card-header bg-success text-white">Local LLM Savings</div>
              <div className="card-body">
                <p className="mb-1"><strong>Cloud equivalent saved:</strong> ${budget.savings?.local_llm_savings?.toFixed(4)}</p>
                <p className="text-muted mb-0 small">{budget.savings?.note}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header bg-secondary text-white">Metric Definitions</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Metric</th><th>Description</th><th>Unit</th></tr></thead>
                  <tbody>
                    {(defs.metrics || []).map(m => (
                      <tr key={m.name}><td className="fw-bold">{m.name}</td><td>{m.description}</td><td><code>{m.unit}</code></td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header bg-warning text-dark">LLM Rate Cards</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Model</th><th className="text-end">Input/1K</th><th className="text-end">Output/1K</th></tr></thead>
                  <tbody>
                    {(defs.rate_cards || []).map(r => (
                      <tr key={r.model}><td>{r.model}</td><td className="text-end">${r.input_per_1k}</td><td className="text-end">${r.output_per_1k}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card shadow-sm mt-3">
              <div className="card-header bg-info text-white">Operation Rates</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Operation</th><th className="text-end">Rate (USD)</th></tr></thead>
                  <tbody>
                    {(defs.operation_rates || []).map(r => (
                      <tr key={r.operation}><td>{r.operation}</td><td className="text-end">${r.rate_usd}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-12 mb-3">
            <div className="alert alert-info">{defs.clinical_relevance}</div>
          </div>
        </div>
      )}
    </div>
  );
}
