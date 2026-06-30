'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SEV_COLOR = { high: 'danger', medium: 'warning', low: 'info' };
const STATUS_COLOR = { open: 'danger', monitoring: 'warning', mitigated: 'success' };

export default function AIRiskPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/ai-risk/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/ai-risk/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/ai-risk/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = overview.summary || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'register', label: 'Risk Register' },
    { id: 'trends', label: 'Trends & Alerts' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x26a0;&#xfe0f; AI Risk Dashboard</h3>
      <p className="text-muted">Clinical AI risk register from real clinical.db: severity scoring, mitigation status, alert trends, guardrail blocks, and category breakdown</p>

      {/* Summary cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Risks', value: s.total_risks || 0, color: 'primary' },
          { label: 'High Severity', value: s.high_severity || 0, color: s.high_severity > 0 ? 'danger' : 'success' },
          { label: 'Medium Severity', value: s.medium_severity || 0, color: 'warning' },
          { label: 'Mitigated', value: s.mitigated || 0, color: 'success' },
          { label: 'Monitoring', value: s.monitoring || 0, color: 'info' },
          { label: 'Mitigated %', value: `${s.mitigated_pct || 0}%`, color: s.mitigated_pct >= 80 ? 'success' : 'warning' },
          { label: 'Clinical Alerts', value: s.clinical_alerts || 0, color: s.clinical_alerts > 10 ? 'danger' : 'warning' },
          { label: 'Guardrail Blocks', value: s.guardrail_blocks || 0, color: s.guardrail_blocks > 0 ? 'warning' : 'success' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 col-lg mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h4 mb-0 text-${c.color}`}>{c.value}</div>
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

      {/* Overview tab */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Risk Register Summary</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>ID</th><th>Title</th><th>Severity</th><th>Status</th></tr></thead>
                  <tbody>
                    {(overview.risks || []).map(r => (
                      <tr key={r.id}>
                        <td><code>{r.id}</code></td>
                        <td>{r.title}</td>
                        <td><span className={`badge bg-${SEV_COLOR[r.severity] || 'secondary'}`}>{r.severity}</span></td>
                        <td><span className={`badge bg-${STATUS_COLOR[r.status] || 'secondary'}`}>{r.status}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">Alerts by Instrument</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Instrument</th><th className="text-end">Alerts</th><th>Bar</th></tr></thead>
                  <tbody>
                    {Object.entries(overview.alert_by_instrument || {}).sort((a, b) => b[1] - a[1]).map(([inst, cnt]) => {
                      const maxV = Math.max(...Object.values(overview.alert_by_instrument || {}));
                      const pct = maxV > 0 ? (cnt / maxV * 100) : 0;
                      return (
                        <tr key={inst}>
                          <td><code>{inst}</code></td>
                          <td className="text-end">{cnt}</td>
                          <td style={{width:'40%'}}><div className="bg-danger" style={{height:14,width:`${pct}%`,borderRadius:3,opacity:0.7}} /></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Guardrail Blocks by Component</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Component</th><th className="text-end">Blocks</th></tr></thead>
                  <tbody>
                    {Object.entries(overview.blocks_by_component || {}).map(([comp, cnt]) => (
                      <tr key={comp}><td><code>{comp}</code></td><td className="text-end">{cnt}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Risk Register tab */}
      {tab === 'register' && breakdown && (
        <div>
          <p className="text-muted mb-2">{breakdown.total_categories} risk categories identified from clinical.db</p>
          {(breakdown.categories || []).map(cat => (
            <div key={cat.category} className="card shadow-sm mb-3">
              <div className="card-header fw-bold d-flex justify-content-between">
                <span>{cat.category}</span>
                <span>
                  {cat.high > 0 && <span className="badge bg-danger me-1">{cat.high} high</span>}
                  {cat.medium > 0 && <span className="badge bg-warning me-1">{cat.medium} medium</span>}
                  {cat.low > 0 && <span className="badge bg-info me-1">{cat.low} low</span>}
                </span>
              </div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>ID</th><th>Title</th><th>Severity</th><th>Status</th><th>Count</th><th>Mitigation</th></tr></thead>
                  <tbody>
                    {(cat.risks || []).map(r => (
                      <tr key={r.id}>
                        <td><code>{r.id}</code></td>
                        <td>{r.title}</td>
                        <td><span className={`badge bg-${SEV_COLOR[r.severity] || 'secondary'}`}>{r.severity}</span></td>
                        <td><span className={`badge bg-${STATUS_COLOR[r.status] || 'secondary'}`}>{r.status}</span></td>
                        <td className="text-end">{r.count != null ? r.count : '—'}</td>
                        <td className="small text-muted">{r.mitigation}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Trends & Alerts tab */}
      {tab === 'trends' && (
        <div className="row">
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Daily Alert & Guardrail Trend</div>
              <div className="card-body">
                {(overview.daily_trend || []).length === 0 ? (
                  <p className="text-muted">No trend data</p>
                ) : (
                  <div style={{overflowX:'auto'}}>
                    <table className="table table-sm">
                      <thead><tr><th>Date</th><th className="text-end">Alerts</th><th className="text-end">Blocks</th><th className="text-end">Combined</th><th>Sparkline</th></tr></thead>
                      <tbody>
                        {(overview.daily_trend || []).map(d => {
                          const maxC = Math.max(...(overview.daily_trend || []).map(x => x.combined));
                          const pct = maxC > 0 ? (d.combined / maxC * 100) : 0;
                          return (
                            <tr key={d.date}>
                              <td>{d.date}</td>
                              <td className="text-end">{d.alerts}</td>
                              <td className="text-end">{d.guardrail_blocks}</td>
                              <td className="text-end fw-bold">{d.combined}</td>
                              <td style={{width:'30%'}}><div className={`bg-${d.combined > 5 ? 'danger' : 'primary'}`} style={{height:14,width:`${pct}%`,borderRadius:3,opacity:0.7}} /></td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Seizure Episodes Flagged</div>
              <div className="card-body text-center">
                <div className="display-4 text-danger">{s.seizure_episodes || 0}</div>
                <div className="text-muted">total episodes in seizure diary</div>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Risk Status Distribution</div>
              <div className="card-body">
                {[
                  { label: 'Open', val: s.open || 0, color: 'danger' },
                  { label: 'Monitoring', val: s.monitoring || 0, color: 'warning' },
                  { label: 'Mitigated', val: s.mitigated || 0, color: 'success' },
                ].map(x => (
                  <div key={x.label} className="d-flex align-items-center mb-2">
                    <span className="me-2" style={{width:90}}>{x.label}</span>
                    <div className="flex-grow-1 bg-light rounded" style={{height:20}}>
                      <div className={`bg-${x.color} rounded`} style={{height:20,width:`${s.total_risks > 0 ? (x.val / s.total_risks * 100) : 0}%`,transition:'width 0.3s'}} />
                    </div>
                    <span className="ms-2 fw-bold">{x.val}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold">Term Definitions</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead><tr><th style={{width:'25%'}}>Term</th><th>Definition</th></tr></thead>
              <tbody>
                {(defs.definitions || []).map(d => (
                  <tr key={d.term}><td className="fw-bold">{d.term}</td><td>{d.definition}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
