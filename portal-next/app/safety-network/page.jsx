'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const tierColor = t =>
  t === 'Critical' ? 'danger' :
  t === 'At Risk'  ? 'warning' :
  t === 'Adequate' ? 'primary' : 'success';

const tierBg = t =>
  t === 'Critical' ? '#dc3545' :
  t === 'At Risk'  ? '#ffc107' :
  t === 'Adequate' ? '#0d6efd' : '#198754';

export default function SafetyNetworkPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/safety-network/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/safety-network/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/safety-network/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpis = ov.kpis || {};
  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'patients',   label: 'Per Patient' },
    { id: 'dimensions', label: 'Dimension Analysis' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>🛡️ Patient Safety Network Dashboard</h3>
      <p className="text-muted small">
        Composite safety score across 5 dimensions: caregiver coverage, emergency readiness,
        medication adherence, wearable monitoring, and IoT alert health. Identifies patients
        needing proactive intervention.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Patients',      value: kpis.total_patients,              color: 'primary' },
          { label: 'Avg Safety Score',    value: `${kpis.avg_composite_score}`,    color: kpis.avg_composite_score >= 60 ? 'success' : kpis.avg_composite_score >= 40 ? 'warning' : 'danger' },
          { label: 'Critical',            value: kpis.critical_count,              color: 'danger' },
          { label: 'At Risk',             value: kpis.at_risk_count,               color: 'warning' },
          { label: 'Adequate',            value: kpis.adequate_count,              color: 'primary' },
          { label: 'Has Caregiver',       value: kpis.patients_with_caregiver,     color: 'success' },
          { label: 'Has Wearable',        value: kpis.patients_with_wearable,      color: 'info' },
          { label: 'Emergency Contact',   value: kpis.patients_with_emergency_contact, color: 'secondary' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value ?? '—'}</div>
                <div className="text-muted" style={{fontSize: '0.72rem'}}>{c.label}</div>
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

      {/* ── Overview Tab ─────────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Tier Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Safety Tier Distribution</div>
              <div className="card-body">
                {(ov.tier_distribution || []).map((t, i) => (
                  <div key={i} className="d-flex align-items-center mb-3">
                    <span className={`badge bg-${tierColor(t.tier)} me-2`} style={{minWidth: 72}}>{t.tier}</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 22}}>
                        <div className={`progress-bar bg-${tierColor(t.tier)}`}
                             style={{width: `${kpis.total_patients ? (t.count / kpis.total_patients * 100) : 0}%`}}>
                          {t.count}
                        </div>
                      </div>
                    </div>
                    <span className="small text-muted">{kpis.total_patients ? Math.round(t.count / kpis.total_patients * 100) : 0}%</span>
                  </div>
                ))}
                <div className="alert alert-warning small mt-2 mb-0 py-2">
                  <strong>{kpis.critical_count + kpis.at_risk_count}</strong> patients need intervention
                </div>
              </div>
            </div>
          </div>

          {/* Score Histogram */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Score Distribution (0–100)</div>
              <div className="card-body">
                <div className="small text-muted mb-2">Higher = stronger safety network</div>
                {(ov.score_histogram || []).map((b, i) => (
                  <div key={i} className="d-flex align-items-center mb-2">
                    <span className="small" style={{minWidth: 64}}>{b.range}</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 20}}>
                        <div className={`progress-bar ${b.range === '0-40' ? 'bg-danger' : b.range === '40-60' ? 'bg-warning' : b.range === '60-80' ? 'bg-primary' : 'bg-success'}`}
                             style={{width: `${b.count > 0 ? Math.max(8, b.count / kpis.total_patients * 100) : 0}%`}}>
                          {b.count}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Dimension Averages */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Dimension Averages</div>
              <div className="card-body">
                {(ov.dimension_averages || []).map((d, i) => (
                  <div key={i} className="mb-3">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className="fw-semibold">{d.dimension}</span>
                      <span className="text-muted">wt: {d.weight_pct}% · avg: {d.score}</span>
                    </div>
                    <div className="progress" style={{height: 16}}>
                      <div className={`progress-bar ${d.score >= 70 ? 'bg-success' : d.score >= 50 ? 'bg-warning' : 'bg-danger'}`}
                           style={{width: `${d.score}%`}}>
                        {d.score}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Critical Patients */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm border-danger">
              <div className="card-header fw-bold bg-danger text-white">
                Critical Patients — Immediate Attention Required ({(ov.critical_patients || []).length})
              </div>
              <div className="card-body p-0">
                <div style={{maxHeight: 280, overflowY: 'auto'}}>
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark sticky-top">
                      <tr>
                        <th>Patient</th><th>Score</th><th>Tier</th>
                        <th>Caregiver</th><th>Emergency</th><th>Adherence</th>
                        <th>Wearable</th><th>IoT Alert</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(ov.critical_patients || []).map((p, i) => (
                        <tr key={i} className="table-danger">
                          <td className="fw-semibold small">{p.patient_id}</td>
                          <td className="fw-bold text-danger">{p.composite_score}</td>
                          <td><span className={`badge bg-${tierColor(p.tier)}`}>{p.tier}</span></td>
                          <td>{scoreBar(p.caregiver_score)}</td>
                          <td>{scoreBar(p.emergency_score)}</td>
                          <td>{scoreBar(p.adherence_score)}</td>
                          <td>{scoreBar(p.wearable_score)}</td>
                          <td>{scoreBar(p.alert_score)}</td>
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

      {/* ── Per Patient Tab ────────────────────────────────────────── */}
      {tab === 'patients' && bd && (
        <div>
          <div className="card shadow-sm">
            <div className="card-header fw-bold">
              All Patients — Safety Scores ({(bd.per_patient || []).length} patients)
            </div>
            <div className="card-body p-0">
              <div style={{maxHeight: 520, overflowY: 'auto'}}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark sticky-top">
                    <tr>
                      <th>Patient</th>
                      <th>Composite</th>
                      <th>Tier</th>
                      <th>Caregiver (25%)</th>
                      <th>Emergency (20%)</th>
                      <th>Adherence (30%)</th>
                      <th>Wearable (15%)</th>
                      <th>IoT Alert (10%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.per_patient || []).sort((a, b) => a.composite_score - b.composite_score).map((p, i) => (
                      <tr key={i}>
                        <td className="fw-semibold small">{p.patient_id}</td>
                        <td>
                          <div className="d-flex align-items-center gap-1">
                            <div className="progress flex-grow-1" style={{height: 14, minWidth: 60}}>
                              <div className={`progress-bar bg-${tierColor(p.tier)}`}
                                   style={{width: `${p.composite_score}%`}} />
                            </div>
                            <span className="small fw-bold">{p.composite_score}</span>
                          </div>
                        </td>
                        <td><span className={`badge bg-${tierColor(p.tier)}`}>{p.tier}</span></td>
                        <td>{miniBar(p.caregiver_score)}</td>
                        <td>{miniBar(p.emergency_score)}</td>
                        <td>{miniBar(p.adherence_score)}</td>
                        <td>{miniBar(p.wearable_score)}</td>
                        <td>{miniBar(p.alert_score)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Dimension Analysis Tab ─────────────────────────────────── */}
      {tab === 'dimensions' && bd && (
        <div className="row">
          {/* Caregiver Training */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Caregiver Training Coverage (n={bd.caregiver_training?.total})</div>
              <div className="card-body">
                {bd.caregiver_training && [
                  { label: 'First Aid Certified',   value: bd.caregiver_training.first_aid },
                  { label: 'Rescue Med Trained',    value: bd.caregiver_training.rescue_med },
                  { label: 'Epilepsy Training',     value: bd.caregiver_training.epilepsy_training },
                  { label: 'Safety Plan in Place',  value: bd.caregiver_training.safety_plan },
                  { label: 'Seizure Action Plan',   value: bd.caregiver_training.seizure_action_plan },
                ].map((item, i) => {
                  const total = bd.caregiver_training.total || 1;
                  const pct = Math.round(item.value / total * 100);
                  return (
                    <div key={i} className="d-flex align-items-center mb-2">
                      <span className="small" style={{minWidth: 160}}>{item.label}</span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{height: 18}}>
                          <div className={`progress-bar ${pct >= 70 ? 'bg-success' : pct >= 50 ? 'bg-warning' : 'bg-danger'}`}
                               style={{width: `${pct}%`}}>
                            {item.value}/{total}
                          </div>
                        </div>
                      </div>
                      <span className="small text-muted">{pct}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Emergency Contact Coverage */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Emergency Contact Coverage</div>
              <div className="card-body">
                {bd.emergency_contact_coverage && [
                  { label: 'Primary Contacts Assigned', value: bd.emergency_contact_coverage.primary_contacts },
                  { label: 'Notify-Enabled Contacts',   value: bd.emergency_contact_coverage.notify_enabled },
                  { label: 'Patients Covered',          value: bd.emergency_contact_coverage.patients_covered },
                ].map((item, i) => {
                  const total = kpis.total_patients || 1;
                  const pct = Math.round(item.value / total * 100);
                  return (
                    <div key={i} className="d-flex align-items-center mb-3">
                      <span className="small" style={{minWidth: 180}}>{item.label}</span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{height: 22}}>
                          <div className={`progress-bar ${pct >= 70 ? 'bg-success' : pct >= 50 ? 'bg-warning' : 'bg-danger'}`}
                               style={{width: `${pct}%`}}>
                            {item.value}
                          </div>
                        </div>
                      </div>
                      <span className="small text-muted">{pct}%</span>
                    </div>
                  );
                })}
                <div className="alert alert-info small py-2 mb-0">
                  <strong>{kpis.patients_with_emergency_contact}</strong> of {kpis.total_patients} patients have emergency contacts registered
                </div>
              </div>
            </div>
          </div>

          {/* Wearable Status */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Wearable Device Status Distribution</div>
              <div className="card-body">
                {(bd.wearable_status_distribution || []).map((w, i) => {
                  const total = kpis.patients_with_wearable || 1;
                  const pct = Math.round(w.count / total * 100);
                  const color = w.status === 'active' ? 'success' : w.status === 'offline' ? 'danger' : w.status === 'charging' ? 'info' : 'warning';
                  return (
                    <div key={i} className="d-flex align-items-center mb-2">
                      <span className={`badge bg-${color} me-2`} style={{minWidth: 88}}>{w.status}</span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{height: 20}}>
                          <div className={`progress-bar bg-${color}`}
                               style={{width: `${w.count > 0 ? Math.max(6, pct) : 0}%`}}>
                            {w.count}
                          </div>
                        </div>
                      </div>
                      <span className="small text-muted">{pct}%</span>
                    </div>
                  );
                })}
                <div className="small text-muted mt-2">
                  Seizure detection events: <strong>{bd.wearable_seizure_detection_count ?? 0}</strong>
                </div>
              </div>
            </div>
          </div>

          {/* IoT Alert Breakdown */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">IoT Alert Health</div>
              <div className="card-body">
                {(bd.iot_alert_breakdown || []).map((a, i) => {
                  const color = a.severity === 'critical' ? 'danger' : a.severity === 'warning' ? 'warning' : 'info';
                  const resolvedPct = a.total > 0 ? Math.round((a.total - a.unresolved) / a.total * 100) : 0;
                  return (
                    <div key={i} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className={`badge bg-${color}`}>{a.severity.toUpperCase()}</span>
                        <span className="text-muted">Total: {a.total} · Unresolved: <span className="text-danger fw-bold">{a.unresolved}</span></span>
                      </div>
                      <div className="progress" style={{height: 18}}>
                        <div className={`progress-bar bg-${color}`} style={{width: `${resolvedPct}%`}}>
                          {resolvedPct}% resolved
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ───────────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          {/* Description */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm border-primary">
              <div className="card-header fw-bold bg-primary text-white">Dashboard Purpose</div>
              <div className="card-body small">{defs.description}</div>
            </div>
          </div>

          {/* Tiers */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Safety Tiers</div>
              <div className="card-body">
                {(defs.tiers || []).map((t, i) => (
                  <div key={i} className="mb-3">
                    <div className="d-flex align-items-center mb-1">
                      <span className={`badge bg-${tierColor(t.tier)} me-2`}>{t.tier}</span>
                      <span className="small text-muted">Score: {t.range}</span>
                    </div>
                    <div className="small">{t.meaning}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Dimensions */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Scoring Dimensions</div>
              <div className="card-body" style={{maxHeight: 320, overflowY: 'auto'}}>
                {(defs.dimensions || []).map((d, i) => (
                  <div key={i} className="mb-3">
                    <div className="fw-semibold small text-primary">{d.name} <span className="badge bg-secondary ms-1">{d.weight}</span></div>
                    <div className="small text-muted mt-1"><strong>Source:</strong> {d.source}</div>
                    <div className="small text-muted"><strong>Metric:</strong> {d.metric}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Data Sources + Glossary */}
          {defs.data_sources && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Data Sources</div>
                <div className="card-body">
                  {defs.data_sources.map ? (
                    defs.data_sources.map((s, i) => (
                      <div key={i} className="mb-2 small">
                        <strong>{s.table}</strong>: {s.rows} rows — {s.description}
                      </div>
                    ))
                  ) : (
                    <pre className="small">{JSON.stringify(defs.data_sources, null, 2)}</pre>
                  )}
                </div>
              </div>
            </div>
          )}

          {defs.glossary && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Glossary</div>
                <div className="card-body" style={{maxHeight: 280, overflowY: 'auto'}}>
                  {defs.glossary.map ? (
                    defs.glossary.map((g, i) => (
                      <div key={i} className="mb-2">
                        <strong className="text-primary small">{g.term}</strong>
                        <div className="small text-muted">{g.definition}</div>
                      </div>
                    ))
                  ) : Object.entries(defs.glossary).map(([k, v], i) => (
                    <div key={i} className="mb-2">
                      <strong className="text-primary small">{k}</strong>
                      <div className="small text-muted">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function scoreBar(val) {
  const v = val ?? 0;
  const color = v >= 70 ? 'bg-success' : v >= 40 ? 'bg-warning' : 'bg-danger';
  return (
    <div className="d-flex align-items-center gap-1">
      <div className="progress flex-grow-1" style={{height: 12, minWidth: 48}}>
        <div className={`progress-bar ${color}`} style={{width: `${v}%`}} />
      </div>
      <span className="small">{v}</span>
    </div>
  );
}

function miniBar(val) {
  const v = val ?? 0;
  const color = v >= 70 ? 'bg-success' : v >= 40 ? 'bg-warning' : 'bg-danger';
  return (
    <div className="d-flex align-items-center gap-1">
      <div className="progress" style={{height: 10, minWidth: 44, flexGrow: 1}}>
        <div className={`progress-bar ${color}`} style={{width: `${v}%`}} />
      </div>
      <span style={{fontSize: '0.7rem'}}>{v}</span>
    </div>
  );
}
