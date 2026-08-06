'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusColor = s =>
  s === 'active' ? 'success' : s === 'onboarding' ? 'warning' : 'secondary';

const driftColor = d =>
  d >= 0.45 ? 'danger' : d >= 0.3 ? 'warning' : 'success';

export default function AIFederationPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [sel,  setSel]  = useState(null);

  useEffect(() => {
    fetch(`${API}/api/ai-federation/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/ai-federation/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/ai-federation/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const k = ov.kpis || {};
  const kpis = [
    { label: 'Total Sites',       value: k.total_sites,                                       color: 'primary' },
    { label: 'Active Sites',      value: k.active_sites,                                       color: 'success' },
    { label: 'Total Rounds',      value: k.total_rounds,                                       color: 'info'    },
    { label: 'Completed Rounds',  value: k.completed_rounds,                                   color: 'success' },
    { label: 'Global Accuracy',   value: k.global_accuracy != null ? `${k.global_accuracy.toFixed(1)}%` : '—', color: 'primary' },
    { label: 'Avg Site Accuracy', value: k.avg_site_accuracy != null ? `${k.avg_site_accuracy.toFixed(1)}%` : '—', color: 'info' },
    { label: 'Avg Data Quality',  value: k.avg_data_quality != null ? `${(k.avg_data_quality * 100).toFixed(0)}%` : '—', color: 'success' },
    { label: 'Avg Drift Score',   value: k.avg_drift_score != null ? k.avg_drift_score.toFixed(2) : '—', color: 'warning' },
  ];

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'sites',      label: `Sites${bd ? ` (${(bd.sites || []).length})` : ''}` },
    { id: 'rounds',     label: `Rounds${bd ? ` (${(bd.rounds || []).length})` : ''}` },
    { id: 'definitions', label: 'Standards' },
  ];

  const sites  = (bd || {}).sites || [];
  const rounds = (bd || {}).rounds || [];
  const selSite = sites.find(s => s.site_id === sel);
  const trend  = ov.accuracy_trend || [];

  return (
    <div>
      <h3>&#x1f310; AI Federation Dashboard</h3>
      <p className="text-muted small">
        Federated learning across multi-hospital epilepsy centers — round-by-round global accuracy,
        per-site drift scores, data quality, aggregation methods (FedAvg/FedProx/Scaffold),
        privacy-preserving training without raw data sharing. HIPAA / Differential Privacy.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {kpis.map(kp => (
          <div key={kp.label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${kp.color}`}>{kp.value ?? '—'}</div>
                <div className="text-muted" style={{ fontSize: '0.72rem' }}>{kp.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div className="row">

          {/* Accuracy Trend */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-primary text-white py-2 small fw-bold">Global Accuracy Trend (by Round)</div>
              <div className="card-body p-2">
                <div className="d-flex align-items-end" style={{ height: '100px', gap: '4px' }}>
                  {trend.map(t => {
                    const min = Math.min(...trend.map(x => x.accuracy));
                    const max = Math.max(...trend.map(x => x.accuracy), 1);
                    const pct = Math.round(((t.accuracy - min) / Math.max(max - min, 1)) * 80) + 20;
                    return (
                      <div key={t.round} className="d-flex flex-column align-items-center flex-grow-1"
                           title={`Round ${t.round}: ${t.accuracy.toFixed(1)}%`}>
                        <div style={{ width: '100%', height: `${pct}%`, background: '#0d6efd', borderRadius: '2px 2px 0 0', minHeight: '4px' }} />
                        <div style={{ fontSize: '0.55rem', color: '#666' }}>{t.round}</div>
                      </div>
                    );
                  })}
                </div>
                <div className="d-flex justify-content-between small text-muted mt-1">
                  <span>Round {trend[0]?.round || ''}</span>
                  <span>Global: {k.global_accuracy?.toFixed(1)}%</span>
                  <span>Round {trend[trend.length - 1]?.round || ''}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Sites by Status & Region */}
          <div className="col-md-3 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-success text-white py-2 small fw-bold">Sites by Status</div>
              <div className="card-body p-2">
                {(ov.sites_by_status || []).filter(s => s.count > 0).map(s => {
                  const max = Math.max(...(ov.sites_by_status || []).map(x => x.count), 1);
                  return (
                    <div key={s.status} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '90px' }}>
                        <span className={`badge bg-${statusColor(s.status)}`}>{s.status}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${statusColor(s.status)}`}
                             style={{ width: `${(s.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{s.count}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div className="mt-2 pt-2 border-top">
                  {(ov.sites_by_region || []).map(r => (
                    <div key={r.region} className="d-flex justify-content-between small">
                      <span>{r.region}</span><span className="fw-bold">{r.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Aggregation Methods */}
          <div className="col-md-3 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-info text-white py-2 small fw-bold">Aggregation Methods</div>
              <div className="card-body p-2">
                {(ov.aggregation_methods || []).map(a => {
                  const max = Math.max(...(ov.aggregation_methods || []).map(x => x.count), 1);
                  return (
                    <div key={a.method} className="d-flex align-items-center mb-2">
                      <span className="small me-2 fw-bold" style={{ minWidth: '75px' }}>{a.method}</span>
                      <div className="progress flex-grow-1" style={{ height: '18px' }}>
                        <div className="progress-bar bg-info"
                             style={{ width: `${(a.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{a.count}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Top Drifting Sites */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">Top Drifting Sites</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Site</th><th>Name</th><th>Status</th><th className="text-end">Drift Score</th><th style={{ width: '35%' }}>Bar</th></tr></thead>
                  <tbody>
                    {(ov.top_drifting_sites || []).map(s => {
                      const max = Math.max(...(ov.top_drifting_sites || []).map(x => x.drift_score), 1);
                      return (
                        <tr key={s.site_id}>
                          <td className="small fw-bold">{s.site_id}</td>
                          <td className="small">{s.site_name}</td>
                          <td><span className={`badge bg-${statusColor(s.status)}`} style={{ fontSize: '0.65rem' }}>{s.status}</span></td>
                          <td className="text-end small">
                            <span className={`text-${driftColor(s.drift_score)} fw-bold`}>{s.drift_score.toFixed(3)}</span>
                          </td>
                          <td>
                            <div className="progress" style={{ height: '14px' }}>
                              <div className={`progress-bar bg-${driftColor(s.drift_score)}`}
                                   style={{ width: `${(s.drift_score / max * 100).toFixed(0)}%` }} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

        </div>
      )}

      {/* ── Sites ── */}
      {tab === 'sites' && (
        <div>
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-header bg-dark text-white py-2 small fw-bold">
              All Federation Sites ({sites.length})
            </div>
            <div className="card-body p-2">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Site</th><th>Name</th><th>Region</th><th>Status</th>
                      <th className="text-end">Patients</th><th className="text-end">Local Acc.</th>
                      <th className="text-end">Data Quality</th><th className="text-end">Drift</th>
                      <th>Model Ver.</th><th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {sites.map(s => (
                      <tr key={s.site_id} className={sel === s.site_id ? 'table-active' : ''}>
                        <td className="small fw-bold">{s.site_id}</td>
                        <td className="small">{s.site_name}</td>
                        <td className="small">{s.region}</td>
                        <td><span className={`badge bg-${statusColor(s.status)}`} style={{ fontSize: '0.65rem' }}>{s.status}</span></td>
                        <td className="text-end small">{s.patients_contributed}</td>
                        <td className="text-end small">
                          <span className={s.local_accuracy >= 85 ? 'text-success fw-bold' : s.local_accuracy >= 75 ? 'text-warning' : 'text-danger'}>
                            {s.local_accuracy?.toFixed(1)}%
                          </span>
                        </td>
                        <td className="text-end small">{s.data_quality_score != null ? `${(s.data_quality_score * 100).toFixed(0)}%` : '—'}</td>
                        <td className="text-end small">
                          <span className={`text-${driftColor(s.drift_score)}`}>{s.drift_score?.toFixed(3)}</span>
                        </td>
                        <td className="small">{s.model_version}</td>
                        <td>
                          <button className="btn btn-outline-primary btn-sm py-0 px-1" style={{ fontSize: '0.7rem' }}
                                  onClick={() => setSel(sel === s.site_id ? null : s.site_id)}>
                            {sel === s.site_id ? 'Hide' : 'Detail'}
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Site detail */}
          {sel && selSite && (
            <div className="card border-primary shadow-sm">
              <div className="card-header bg-primary text-white py-1 small fw-bold">
                {selSite.site_id} — {selSite.site_name} · Site Detail
              </div>
              <div className="card-body p-3">
                <div className="row">
                  <div className="col-md-6">
                    <table className="table table-sm mb-0">
                      <tbody>
                        <tr><td className="fw-bold small">Status</td>
                          <td><span className={`badge bg-${statusColor(selSite.status)}`}>{selSite.status}</span></td></tr>
                        <tr><td className="fw-bold small">Region</td><td className="small">{selSite.region}</td></tr>
                        <tr><td className="fw-bold small">Patients Contributed</td><td className="small">{selSite.patients_contributed}</td></tr>
                        <tr><td className="fw-bold small">Onboarded</td><td className="small">{selSite.onboarded_date ? new Date(selSite.onboarded_date).toLocaleDateString() : '—'}</td></tr>
                        <tr><td className="fw-bold small">Last Sync</td><td className="small">{selSite.last_sync ? new Date(selSite.last_sync).toLocaleString() : '—'}</td></tr>
                      </tbody>
                    </table>
                  </div>
                  <div className="col-md-6">
                    <table className="table table-sm mb-0">
                      <tbody>
                        <tr><td className="fw-bold small">Model Version</td><td className="small">{selSite.model_version}</td></tr>
                        <tr><td className="fw-bold small">Local Accuracy</td>
                          <td className="small">{selSite.local_accuracy?.toFixed(1)}%</td></tr>
                        <tr><td className="fw-bold small">Data Quality</td>
                          <td className="small">{selSite.data_quality_score != null ? `${(selSite.data_quality_score * 100).toFixed(0)}%` : '—'}</td></tr>
                        <tr><td className="fw-bold small">Drift Score</td>
                          <td className="small">
                            <span className={`text-${driftColor(selSite.drift_score)} fw-bold`}>{selSite.drift_score?.toFixed(3)}</span>
                          </td></tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Rounds ── */}
      {tab === 'rounds' && (
        <div className="card shadow-sm border-0">
          <div className="card-header bg-dark text-white py-2 small fw-bold">
            Federation Rounds ({rounds.length})
          </div>
          <div className="card-body p-2">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>Round</th><th>Method</th><th>Status</th>
                    <th className="text-end">Sites</th><th className="text-end">Accuracy</th>
                    <th className="text-end">Loss</th><th className="text-end">Conv. Delta</th>
                    <th>Started</th>
                  </tr>
                </thead>
                <tbody>
                  {rounds.map(r => (
                    <tr key={r.round_number}>
                      <td className="small fw-bold">{r.round_number}</td>
                      <td className="small">{r.aggregation_method}</td>
                      <td>
                        <span className={`badge bg-${r.status === 'completed' ? 'success' : r.status === 'in_progress' ? 'warning' : 'secondary'}`}
                              style={{ fontSize: '0.65rem' }}>{r.status}</span>
                      </td>
                      <td className="text-end small">{r.participating_sites}</td>
                      <td className="text-end small">
                        <span className={r.global_accuracy >= 85 ? 'text-success fw-bold' : r.global_accuracy >= 75 ? 'text-warning' : ''}>
                          {r.global_accuracy?.toFixed(1)}%
                        </span>
                      </td>
                      <td className="text-end small">{r.global_loss?.toFixed(4)}</td>
                      <td className="text-end small">{r.convergence_delta?.toFixed(4)}</td>
                      <td className="small">{r.started_at ? new Date(r.started_at).toLocaleDateString() : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Standards ── */}
      {tab === 'definitions' && defs && (
        <div>
          {(defs.sections || []).map(s => (
            <div key={s.heading} className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-dark text-white py-2 small fw-bold">{s.heading}</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {(s.items || []).map(item => (
                      <tr key={item.term}>
                        <td className="small fw-bold" style={{ width: '28%', verticalAlign: 'top' }}>{item.term}</td>
                        <td className="small">{item.detail}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
