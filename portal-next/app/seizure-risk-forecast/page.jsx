'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'breakdown',   label: 'Breakdown' },
  { id: 'definitions', label: 'Definitions' },
];

const TIER_COLOR = { critical: 'danger', high: 'warning', moderate: 'info', low: 'secondary' };
const TIER_BG    = { critical: '#dc3545', high: '#ffc107', moderate: '#0dcaf0', low: '#6c757d' };

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function TierBadge({ tier }) {
  return <span className={`badge bg-${TIER_COLOR[tier] || 'secondary'}`}>{tier || '—'}</span>;
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted p-3"><div className="spinner-border spinner-border-sm text-danger me-2" />Loading...</div>;
  if (!data.available) return <div className="alert alert-warning">{data.error || 'No data available'}</div>;

  const k   = data.kpis || {};
  const pie = data.charts?.risk_tier_pie || [];
  const hor = data.charts?.horizon_bar || [];
  const rd  = data.charts?.risk_distribution || [];
  const esc = data.charts?.escalation_actions_bar || [];
  const thr = data.thresholds || [];

  const maxPie = Math.max(...pie.map(p => p.value), 1);
  const maxHor = Math.max(...hor.map(h => h.value), 1);
  const maxRd  = Math.max(...rd.map(r => r.value), 1);
  const maxEsc = Math.max(...esc.map(e => e.value), 1);

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Patients Monitored"   value={k.total_patients_monitored}                      color="primary"   sub="unique patients in forecasting" />
        <KPI label="Avg Risk Score"        value={k.avg_risk_score != null ? k.avg_risk_score.toFixed(3) : '—'}  color={k.avg_risk_score >= 0.5 ? 'danger' : 'success'} sub="0–1 scale" />
        <KPI label="Critical Patients"     value={k.patients_critical != null ? `${k.patients_critical} (${k.critical_pct}%)` : '—'} color="danger" sub="risk ≥ 0.75" />
        <KPI label="High Risk Patients"    value={k.patients_high != null ? `${k.patients_high} (${k.high_pct}%)` : '—'}   color="warning" sub="risk 0.50–0.75" />
      </div>
      <div className="row mb-3">
        <KPI label="Forecasting Gaps"      value={k.forecasting_gaps_total} color="secondary" sub="capability gaps identified" />
        <KPI label="Partial Gaps"          value={k.gaps_partial}           color="warning"   sub="partially addressed" />
      </div>

      <div className="row">
        {/* Risk Tier Distribution */}
        <div className="col-md-6 mb-3">
          <div className="card">
            <div className="card-header fw-semibold">Risk Tier Distribution</div>
            <div className="card-body">
              {pie.map(p => (
                <div key={p.name} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <TierBadge tier={p.name} />
                    <span>{p.value} patients</span>
                  </div>
                  <div className="progress" style={{ height: 14 }}>
                    <div className="progress-bar" role="progressbar"
                      style={{ width: `${(p.value / maxPie) * 100}%`, backgroundColor: TIER_BG[p.name] || '#6c757d' }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Forecast Horizon */}
        <div className="col-md-6 mb-3">
          <div className="card">
            <div className="card-header fw-semibold">Forecast Horizon Breakdown</div>
            <div className="card-body">
              {hor.map(h => (
                <div key={h.name} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{h.name}</span>
                    <span>{h.value} patients</span>
                  </div>
                  <div className="progress" style={{ height: 14 }}>
                    <div className="progress-bar bg-info" role="progressbar"
                      style={{ width: `${(h.value / maxHor) * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Risk Score Distribution Histogram */}
      {rd.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Risk Score Distribution Histogram</div>
          <div className="card-body">
            <div className="d-flex align-items-end gap-1" style={{ height: 120 }}>
              {rd.map((bin, i) => {
                const h = Math.max((bin.value / maxRd) * 100, 2);
                const color = i >= 7 ? '#dc3545' : i >= 5 ? '#ffc107' : i >= 3 ? '#0dcaf0' : '#198754';
                return (
                  <div key={bin.name} className="d-flex flex-column align-items-center flex-grow-1">
                    <div className="small text-muted mb-1">{bin.value}</div>
                    <div style={{ width: '100%', height: `${h}%`, backgroundColor: color, borderRadius: '3px 3px 0 0' }} />
                    <div className="small text-muted mt-1" style={{ fontSize: '0.6rem', writingMode: 'vertical-rl', transform: 'rotate(180deg)', height: 40 }}>{bin.name}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Escalation Actions */}
      {esc.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Escalation Actions by Tier</div>
          <div className="card-body">
            {esc.map(e => (
              <div key={e.name} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <TierBadge tier={e.name} />
                  <span>{e.value} logged</span>
                </div>
                <div className="progress" style={{ height: 14 }}>
                  <div className="progress-bar" role="progressbar"
                    style={{ width: `${(e.value / maxEsc) * 100}%`, backgroundColor: TIER_BG[e.name] || '#6c757d' }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Alert Thresholds */}
      {thr.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Alert Threshold Configuration</div>
          <div className="card-body" style={{ overflowX: 'auto' }}>
            <table className="table table-sm table-striped mb-0">
              <thead>
                <tr><th>Tier</th><th>Min Score</th><th>Max Response (min)</th><th>Escalation Action</th></tr>
              </thead>
              <tbody>
                {thr.map(t => (
                  <tr key={t.tier}>
                    <td><TierBadge tier={t.tier} /></td>
                    <td className="fw-semibold">≥ {t.min_score}</td>
                    <td>{t.max_response_min != null ? `${t.max_response_min} min` : 'Routine'}</td>
                    <td>{t.escalation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted p-3"><div className="spinner-border spinner-border-sm text-danger me-2" />Loading...</div>;
  if (!data.available) return <div className="alert alert-warning">{data.error || 'No data available'}</div>;

  const forecasts = data.per_patient_forecasts || [];
  const gaps      = data.gaps || [];

  return (
    <div>
      {/* Per-Patient Forecasts */}
      {forecasts.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Per-Patient Risk Forecasts ({forecasts.length})</div>
          <div className="card-body" style={{ overflowX: 'auto' }}>
            <table className="table table-sm table-striped mb-0">
              <thead>
                <tr>
                  <th>Patient</th>
                  <th>Risk Score</th>
                  <th>Tier</th>
                  <th>Horizon</th>
                  <th>Confidence</th>
                  <th>Escalation Actions</th>
                </tr>
              </thead>
              <tbody>
                {forecasts.map(p => (
                  <tr key={p.patient_id}>
                    <td className="font-monospace small">{p.patient_id}</td>
                    <td>
                      <div className="d-flex align-items-center gap-1">
                        <span className="fw-semibold">{p.risk_score?.toFixed(3)}</span>
                        <div className="progress flex-grow-1" style={{ height: 8, minWidth: 60 }}>
                          <div className="progress-bar" role="progressbar"
                            style={{ width: `${(p.risk_score || 0) * 100}%`, backgroundColor: TIER_BG[p.risk_tier] || '#6c757d' }} />
                        </div>
                      </div>
                    </td>
                    <td><TierBadge tier={p.risk_tier} /></td>
                    <td><span className="badge bg-secondary">{p.forecast_horizon}</span></td>
                    <td>
                      <span className={`badge bg-${p.model_confidence >= 0.85 ? 'success' : p.model_confidence >= 0.7 ? 'warning' : 'danger'}`}>
                        {p.model_confidence != null ? `${(p.model_confidence * 100).toFixed(0)}%` : '—'}
                      </span>
                    </td>
                    <td>
                      <ul className="mb-0 ps-3 small">
                        {(p.escalation_actions || []).map((a, i) => <li key={i}>{a}</li>)}
                      </ul>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Gaps */}
      {gaps.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Forecasting Capability Gaps ({gaps.length})</div>
          <div className="card-body">
            {gaps.map((g, i) => (
              <div key={i} className="d-flex align-items-start mb-2 gap-2">
                <span className={`badge bg-${g.severity === 'high' ? 'danger' : g.severity === 'medium' ? 'warning' : 'secondary'} mt-1`}>
                  {g.severity || 'info'}
                </span>
                <div>
                  <div className="fw-semibold small">{g.gap}</div>
                  {g.impact && <div className="text-muted" style={{ fontSize: '0.75rem' }}>{g.impact}</div>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted p-3"><div className="spinner-border spinner-border-sm text-danger me-2" />Loading...</div>;
  const defs = Array.isArray(data) ? data : (data.definitions || []);
  if (!defs.length) return <div className="alert alert-warning">No definitions available</div>;
  return (
    <div className="row">
      {defs.map((d, i) => (
        <div key={i} className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">{d.term || d.title}</div>
            <div className="card-body small"><p className="mb-0">{d.definition || d.description}</p></div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function SeizureRiskForecastPage() {
  const [tab,  setTab]  = useState('overview');
  const [data, setData] = useState({});

  useEffect(() => {
    if (data[tab]) return;
    fetch(`${API}/api/seizure-risk-forecast/${tab}`)
      .then(r => r.json())
      .then(d => setData(prev => ({ ...prev, [tab]: d })))
      .catch(() => {});
  }, [tab]);

  return (
    <div className="container-fluid py-4">
      <h3 className="mb-1">⚠️ Seizure Risk Forecasting</h3>
      <p className="text-muted mb-4">
        4-tier risk scoring (low / moderate / high / critical) with configurable alert thresholds,
        5 forecast horizons, per-patient risk forecasts, escalation action chains, and gap analysis.
      </p>

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel    data={data.overview} />}
      {tab === 'breakdown'   && <BreakdownPanel   data={data.breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={data.definitions} />}
    </div>
  );
}
