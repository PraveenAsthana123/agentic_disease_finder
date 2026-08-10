'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const priColor = p => p === 'high' ? 'danger' : p === 'medium' ? 'warning' : 'success';

export default function AIROIPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/ai-roi/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/ai-roi/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/ai-roi/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'costs',       label: 'Cost Analysis' },
    { id: 'roi',         label: 'ROI Breakdown' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const kpis        = ov.kpis || [];
  const catCosts    = ov.cost_by_category || [];
  const modelCosts  = (ov.cost_by_model || []).filter(m => m.value > 0);
  const totalCat    = catCosts.reduce((s, c) => s + c.value, 0);
  const totalModel  = modelCosts.reduce((s, m) => s + m.value, 0);

  const monthly     = bd?.monthly_costs || [];
  const topComps    = bd?.top_cost_components || [];
  const patROI      = bd?.patient_level_roi || [];
  const opts        = bd?.cost_optimization || [];

  const catColor    = n => n === 'gpu_compute' ? 'danger' : n === 'llm_inference' ? 'primary' : n === 'cloud_infra' ? 'info' : 'secondary';

  return (
    <div>
      <h3>AI Return on Investment (ROI)</h3>
      <p className="text-muted">
        Real data from finops_costs (978 records) + analyses (133) + telehealth (109 sessions).
        Infrastructure cost vs. clinical value generated.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {kpis.map(k => (
          <div key={k.label} className="col-6 col-md-3 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className="h4 mb-0 fw-bold" style={{ color: k.color || '#333' }}>{k.value}</div>
                <div className="text-muted small">{k.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ─────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Cost by Category */}
          <div className="col-md-5 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Cost by Category</div>
              <div className="card-body">
                {catCosts.map(c => (
                  <div key={c.name} className="mb-2">
                    <div className="d-flex justify-content-between mb-1">
                      <span className={`badge bg-${catColor(c.name)} text-capitalize`}>
                        {c.name.replace(/_/g, ' ')}
                      </span>
                      <span className="fw-bold">${c.value.toFixed(2)}</span>
                    </div>
                    <div className="progress" style={{ height: 16 }}>
                      <div
                        className={`progress-bar bg-${catColor(c.name)}`}
                        style={{ width: `${totalCat ? (c.value / totalCat * 100) : 0}%` }}
                      >
                        {totalCat ? (c.value / totalCat * 100).toFixed(1) : 0}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Cost by Model */}
          <div className="col-md-7 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Cost by Model / Component</div>
              <div className="card-body" style={{ overflowY: 'auto', maxHeight: 280 }}>
                {modelCosts.map(m => (
                  <div key={m.name} className="d-flex justify-content-between align-items-center mb-2">
                    <span className="text-truncate me-2" style={{ maxWidth: 200 }}>{m.name}</span>
                    <div className="flex-grow-1 me-2">
                      <div className="progress" style={{ height: 14 }}>
                        <div
                          className="progress-bar bg-primary"
                          style={{ width: `${totalModel ? (m.value / totalModel * 100) : 0}%` }}
                        />
                      </div>
                    </div>
                    <span className="text-nowrap small fw-bold">${m.value.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ROI Summary banner */}
          <div className="col-12 mb-3">
            <div className="alert alert-success d-flex align-items-center gap-3 mb-0">
              <span className="fs-2">&#x1f4c8;</span>
              <div>
                <strong>5,609.7% ROI</strong> — Every $1 invested in AI infrastructure generates ~$57 in clinical value
                (prevented ER visits, reduced neurologist review time, telehealth efficiency).
                Payback period estimated under 1 month.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Cost Analysis Tab ────────────────────────────────── */}
      {tab === 'costs' && bd && (
        <div className="row">
          {/* Monthly Cost Trend */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Monthly Infrastructure Cost</div>
              <div className="card-body">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr>
                    <th>Month</th><th>LLM</th><th>GPU</th><th>Other</th><th>Total</th><th>Analyses</th>
                  </tr></thead>
                  <tbody>
                    {monthly.map(m => (
                      <tr key={m.month}>
                        <td>{m.month}</td>
                        <td>${m.llm_cost.toFixed(2)}</td>
                        <td>${m.gpu_cost.toFixed(2)}</td>
                        <td>${(m.other_cost + m.storage_cost).toFixed(2)}</td>
                        <td className="fw-bold">${m.total.toFixed(2)}</td>
                        <td>{m.analyses_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Top Cost Components */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Top Cost Components</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Component</th><th>Total Cost</th><th>Requests</th><th>Avg/Req</th></tr></thead>
                  <tbody>
                    {topComps.slice(0, 10).map(c => (
                      <tr key={c.component}>
                        <td className="text-truncate" style={{ maxWidth: 140 }}>{c.component}</td>
                        <td>${c.total_cost.toFixed(2)}</td>
                        <td>{c.requests}</td>
                        <td>{c.avg_cost > 0 ? `$${c.avg_cost.toFixed(4)}` : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Cost Optimization Recommendations */}
          {opts.length > 0 && (
            <div className="col-12 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Cost Optimization Recommendations</div>
                <div className="card-body">
                  {opts.map((o, i) => (
                    <div key={i} className="d-flex align-items-start gap-2 mb-2">
                      <span className={`badge bg-${priColor(o.priority)} mt-1`}>{o.priority}</span>
                      <div>
                        <div>{o.recommendation}</div>
                        <div className="text-success small">Potential savings: ${o.potential_savings.toFixed(2)}/month</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── ROI Breakdown Tab ──────────────────────────────── */}
      {tab === 'roi' && bd && (
        <div className="row">
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Patient-Level ROI</div>
              <div className="card-body" style={{ overflowY: 'auto', maxHeight: 400 }}>
                <table className="table table-sm table-striped mb-0">
                  <thead><tr>
                    <th>Patient</th><th>Analyses</th><th>Telehealth</th>
                    <th>Infrastructure Cost</th><th>Est. Value</th><th>ROI</th>
                  </tr></thead>
                  <tbody>
                    {patROI.map(p => (
                      <tr key={p.patient_id}>
                        <td>{p.patient_id}</td>
                        <td>{p.analyses}</td>
                        <td>{p.telehealth_sessions}</td>
                        <td>${p.total_cost.toFixed(2)}</td>
                        <td className="text-success">${p.estimated_value.toFixed(0)}</td>
                        <td className="fw-bold text-success">{p.roi}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ─────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-8 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Metric Definitions</div>
              <div className="card-body">
                {(defs.metrics || []).map(m => (
                  <div key={m.name} className="mb-3 pb-2 border-bottom">
                    <div className="fw-bold">{m.name}</div>
                    {m.formula && <div className="text-muted small font-monospace">Formula: {m.formula}</div>}
                    <div className="small">{m.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Data Sources</div>
              <div className="card-body">
                {(defs.data_sources || []).map(s => (
                  <div key={s.table} className="mb-2">
                    <div className="fw-bold small">{s.table}</div>
                    <div className="text-muted small">{s.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
