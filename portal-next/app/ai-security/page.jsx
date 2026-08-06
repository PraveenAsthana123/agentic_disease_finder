'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const riskColor = r =>
  r === 'HIGH' ? 'danger' : r === 'MEDIUM' ? 'warning' : 'success';

export default function AISecurityPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/ai-security/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/ai-security/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/ai-security/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const k = ov.kpis || {};
  const kpis = [
    { label: 'Total Events',       value: (k.total_events || 0).toLocaleString(),          color: 'primary' },
    { label: 'Security Events',    value: k.security_events,                                color: 'danger'  },
    { label: 'Distinct Actors',    value: k.distinct_actors,                                color: 'info'    },
    { label: 'Components Covered', value: k.distinct_components,                            color: 'primary' },
    { label: 'HITL Reviews',       value: k.hitl_review_count,                              color: 'success' },
    { label: 'PHI Access Events',  value: k.phi_access_count,                               color: 'warning' },
    { label: 'PHI Access Rate',    value: k.phi_access_rate_pct != null ? `${k.phi_access_rate_pct.toFixed(2)}%` : '—', color: 'warning' },
    { label: 'Human Oversight',    value: k.human_oversight_rate_pct != null ? `${k.human_oversight_rate_pct.toFixed(2)}%` : '—', color: 'info' },
  ];

  const tabs = [
    { id: 'overview',     label: 'Overview' },
    { id: 'phi',          label: 'PHI Access Log' },
    { id: 'matrix',       label: 'Actor×Component' },
    { id: 'definitions',  label: 'Standards' },
  ];

  const phiLog  = (bd || {}).phi_access_log || [];
  const matrix  = ov.actor_component_matrix || [];
  const riskDist = ov.risk_distribution || [];
  const hourly  = (ov.hourly_pattern || []).filter(h => h.hour != null);
  const phiByActor = ov.phi_access_by_actor || [];

  const defSec  = (defs && defs.definitions) || {};

  return (
    <div>
      <h3>&#x1f6e1;&#xfe0f; AI Security Dashboard</h3>
      <p className="text-muted small">
        Transaction-log security posture — actor access patterns, PHI exposure,
        risk distribution, HITL oversight, HIPAA §164.312 / FDA AI/ML / IEC 62304 / EU AI Act.
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

          {/* Risk Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-danger text-white py-2 small fw-bold">Risk Distribution</div>
              <div className="card-body p-2">
                {riskDist.map(r => {
                  const max = Math.max(...riskDist.map(x => x.count), 1);
                  return (
                    <div key={r.risk_level} className="d-flex align-items-center mb-2">
                      <span className="small me-2" style={{ minWidth: '75px' }}>
                        <span className={`badge bg-${riskColor(r.risk_level)}`}>{r.risk_level}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '18px' }}>
                        <div className={`progress-bar bg-${riskColor(r.risk_level)}`}
                             style={{ width: `${(r.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{r.count.toLocaleString()}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Actor Coverage */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-primary text-white py-2 small fw-bold">Actor Component Coverage</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Actor</th><th className="text-end">Components</th></tr></thead>
                  <tbody>
                    {(ov.actor_component_coverage || []).map(a => (
                      <tr key={a.actor}>
                        <td className="small">{a.actor}</td>
                        <td className="text-end small fw-bold">{a.components_accessed}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* PHI Access by Actor */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">PHI Access by Actor</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Actor</th><th>Component</th><th className="text-end">Events</th></tr></thead>
                  <tbody>
                    {phiByActor.map((p, i) => (
                      <tr key={i}>
                        <td className="small">{p.actor}</td>
                        <td className="small">{p.component}</td>
                        <td className="text-end small fw-bold">{p.events}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Hourly pattern */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-dark text-white py-2 small fw-bold">Hourly Event Pattern (24h)</div>
              <div className="card-body p-2">
                <div className="d-flex align-items-end" style={{ height: '80px', gap: '2px' }}>
                  {hourly.map(h => {
                    const max = Math.max(...hourly.map(x => x.events), 1);
                    const pct = Math.round((h.events / max) * 100);
                    const color = h.hour < 7 || h.hour >= 20 ? '#dc3545' : '#0d6efd';
                    return (
                      <div key={h.hour} className="d-flex flex-column align-items-center flex-grow-1"
                           title={`${h.hour}:00 — ${h.events} events`}>
                        <div style={{ width: '100%', height: `${pct}%`, background: color, borderRadius: '2px 2px 0 0', minHeight: '2px' }} />
                        <div style={{ fontSize: '0.55rem', color: '#666' }}>{h.hour}</div>
                      </div>
                    );
                  })}
                </div>
                <div className="small text-muted mt-1">
                  <span className="me-3"><span style={{ color: '#dc3545' }}>■</span> Off-hours (before 7AM / after 8PM)</span>
                  <span><span style={{ color: '#0d6efd' }}>■</span> Business hours</span>
                </div>
              </div>
            </div>
          </div>

        </div>
      )}

      {/* ── PHI Access Log ── */}
      {tab === 'phi' && (
        <div className="card shadow-sm border-0">
          <div className="card-header bg-warning text-dark py-2 small fw-bold">
            PHI Access Log — Most Recent {phiLog.length} Events
          </div>
          <div className="card-body p-2">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr><th>ID</th><th>Actor</th><th>Component</th><th>Action</th><th>Patient</th><th>Detail</th><th>Timestamp</th></tr>
                </thead>
                <tbody>
                  {phiLog.map(p => (
                    <tr key={p.id}>
                      <td className="small">{p.id}</td>
                      <td className="small fw-bold">{p.actor}</td>
                      <td className="small">{p.component}</td>
                      <td className="small">{p.action}</td>
                      <td className="small">{p.patient_id}</td>
                      <td className="small text-truncate" style={{ maxWidth: '200px' }} title={p.detail}>{p.detail}</td>
                      <td className="small">{p.timestamp ? new Date(p.timestamp).toLocaleString() : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Actor×Component Matrix ── */}
      {tab === 'matrix' && (
        <div>
          {matrix.map(a => (
            <div key={a.actor} className="card shadow-sm border-0 mb-2">
              <div className="card-header bg-secondary text-white py-1 small fw-bold">
                Actor: {a.actor} ({(a.components || []).length} components)
              </div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Component</th><th className="text-end">Events</th><th style={{ width: '40%' }}>Bar</th></tr></thead>
                  <tbody>
                    {(a.components || []).map(c => {
                      const maxC = Math.max(...(a.components || []).map(x => x.events), 1);
                      return (
                        <tr key={c.component}>
                          <td className="small">{c.component}</td>
                          <td className="text-end small fw-bold">{c.events.toLocaleString()}</td>
                          <td>
                            <div className="progress" style={{ height: '12px' }}>
                              <div className="progress-bar bg-secondary"
                                   style={{ width: `${(c.events / maxC * 100).toFixed(0)}%` }} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Standards / Definitions ── */}
      {tab === 'definitions' && defSec && (
        <div>
          {Object.entries(defSec).map(([section, items]) => (
            <div key={section} className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-dark text-white py-2 small fw-bold">
                {section.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              </div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {Object.entries(items).map(([term, detail]) => (
                      <tr key={term}>
                        <td className="small fw-bold" style={{ width: '28%', verticalAlign: 'top' }}>{term}</td>
                        <td className="small">{detail}</td>
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
