'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = l =>
  l === 'Severe' || l === 'severe' ? 'danger' :
  l === 'Moderately Severe' ? 'danger' :
  l === 'Moderate' || l === 'moderate' ? 'warning' :
  l === 'Mild' || l === 'mild' ? 'info' :
  l === 'Minimal' || l === 'minimal' ? 'success' : 'secondary';

const riskColor = l =>
  l === 'High Risk' || l === 'critical' ? 'danger' :
  l === 'Moderate Risk' || l === 'high' ? 'warning' :
  l === 'Low Risk' || l === 'moderate' ? 'info' :
  'success';

export default function PsychiatristDashboardPage() {
  const [ov,    setOv]    = useState(null);
  const [bd,    setBd]    = useState(null);
  const [defs,  setDefs]  = useState(null);
  const [flags, setFlags] = useState(null);
  const [tab,   setTab]   = useState('overview');
  const [sel,   setSel]   = useState(null);

  useEffect(() => {
    fetch(`${API}/api/psychiatrist/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/psychiatrist/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/psychiatrist/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
    fetch(`${API}/api/psychiatrist/threshold-flags`).then(r => r.json()).then(setFlags).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const k = ov.kpis || {};
  const kpis = [
    { label: 'Total Patients',       value: k.total_patients },
    { label: 'PHQ-9 Elevated',       value: k.phq9_elevated,       color: 'danger' },
    { label: 'GAD-7 Elevated',       value: k.gad7_elevated,       color: 'warning' },
    { label: 'C-SSRS Positive',      value: k.cssrs_positive,      color: 'danger' },
    { label: 'Avg PHQ-9',            value: k.avg_phq9?.toFixed(1) },
    { label: 'Avg GAD-7',            value: k.avg_gad7?.toFixed(1) },
    { label: 'Screen Rate',          value: `${k.comorbidity_screen_rate?.toFixed(0)}%`, color: 'success' },
    { label: 'High Risk',            value: k.high_risk_patients,   color: 'danger' },
    { label: 'Comorbidities',        value: k.total_comorbidities },
    { label: 'Referrals To Neuro',   value: k.referrals_to_neurology },
    { label: 'Referrals From Neuro', value: k.referrals_from_neurology },
  ];

  const tabs = [
    { id: 'overview',     label: 'Overview' },
    { id: 'flags',        label: `Threshold Flags${flags ? ` (${flags.total_flagged})` : ''}` },
    { id: 'breakdown',    label: 'PNES / Mood-Seizure' },
    { id: 'definitions',  label: 'Clinical Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f4ac; Psychiatrist Dashboard</h3>
      <p className="text-muted small">
        Psychiatric comorbidity screening, PHQ-9 / GAD-7 / C-SSRS severity tracking,
        AED psychiatric risk profiles, PNES screening, and threshold-based clinical alerts.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {kpis.map(kpi => (
          <div key={kpi.label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${kpi.color || 'primary'}`}>
                  {kpi.value ?? '\u2014'}
                </div>
                <div className="text-muted" style={{ fontSize: '0.72rem' }}>{kpi.label}</div>
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
          {/* PHQ-9 Severity */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-primary text-white py-2 small fw-bold">PHQ-9 Depression Severity</div>
              <div className="card-body p-2">
                {(ov.phq9_severity || []).map(s => {
                  const max = Math.max(...(ov.phq9_severity || []).map(x => x.count), 1);
                  return (
                    <div key={s.level} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '110px' }}>
                        <span className={`badge bg-${sevColor(s.level)}`}>{s.level}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${sevColor(s.level)}`}
                             style={{ width: `${(s.count / max * 100).toFixed(0)}%` }}>
                          {s.count > 0 && <span style={{ fontSize: '0.65rem' }}>{s.count}</span>}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* GAD-7 Severity */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">GAD-7 Anxiety Severity</div>
              <div className="card-body p-2">
                {(ov.gad7_severity || []).map(s => {
                  const max = Math.max(...(ov.gad7_severity || []).map(x => x.count), 1);
                  return (
                    <div key={s.level} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '110px' }}>
                        <span className={`badge bg-${sevColor(s.level)}`}>{s.level}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${sevColor(s.level)}`}
                             style={{ width: `${(s.count / max * 100).toFixed(0)}%` }}>
                          {s.count > 0 && <span style={{ fontSize: '0.65rem' }}>{s.count}</span>}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* C-SSRS Risk */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-danger text-white py-2 small fw-bold">C-SSRS Suicide Risk</div>
              <div className="card-body p-2">
                {(ov.cssrs_risk || []).map(s => {
                  const max = Math.max(...(ov.cssrs_risk || []).map(x => x.count), 1);
                  return (
                    <div key={s.level} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '110px' }}>
                        <span className={`badge bg-${riskColor(s.level)}`}>{s.level}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${riskColor(s.level)}`}
                             style={{ width: `${(s.count / max * 100).toFixed(0)}%` }}>
                          {s.count > 0 && <span style={{ fontSize: '0.65rem' }}>{s.count}</span>}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Comorbidity Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-dark text-white py-2 small fw-bold">Psychiatric Comorbidity Distribution</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Condition</th><th className="text-end">Count</th><th style={{ width: '40%' }}>Bar</th></tr></thead>
                  <tbody>
                    {(ov.comorbidity_distribution || []).map(c => {
                      const max = Math.max(...(ov.comorbidity_distribution || []).map(x => x.count), 1);
                      return (
                        <tr key={c.condition}>
                          <td className="small">{c.condition}</td>
                          <td className="text-end small fw-bold">{c.count}</td>
                          <td>
                            <div className="progress" style={{ height: '14px' }}>
                              <div className="progress-bar bg-info"
                                   style={{ width: `${(c.count / max * 100).toFixed(0)}%` }} />
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

          {/* AED Psychiatric Risk */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-secondary text-white py-2 small fw-bold">AED Psychiatric Side-Effect Risk</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Drug</th><th>Risk</th><th>Psychiatric Effects</th></tr></thead>
                  <tbody>
                    {(ov.aed_psychiatric_risk || []).map(d => (
                      <tr key={d.drug}>
                        <td className="small fw-bold">{d.drug}</td>
                        <td>
                          <span className={`badge bg-${
                            d.severity === 'moderate' ? 'warning' :
                            d.severity === 'beneficial' ? 'success' :
                            d.severity === 'low' ? 'info' : 'danger'
                          }`}>{d.severity}</span>
                        </td>
                        <td className="small">{(d.effects || []).join(', ')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Treatment Status */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-info text-white py-2 small fw-bold">Treatment Status</div>
              <div className="card-body p-2">
                {(ov.treatment_status || []).map(t => {
                  const max = Math.max(...(ov.treatment_status || []).map(x => x.count), 1);
                  const color = t.status === 'stable' ? 'success' :
                                t.status === 'under_treatment' ? 'primary' :
                                t.status === 'untreated' ? 'warning' : 'danger';
                  return (
                    <div key={t.status} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '120px' }}>
                        {t.status.replace(/_/g, ' ')}
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${color}`}
                             style={{ width: `${(t.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{t.count}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Risk Severity */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-danger text-white py-2 small fw-bold">Behavioral Risk Severity</div>
              <div className="card-body p-2">
                {(ov.risk_severity_distribution || []).map(r => {
                  const max = Math.max(...(ov.risk_severity_distribution || []).map(x => x.count), 1);
                  return (
                    <div key={r.level} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '80px' }}>
                        <span className={`badge bg-${sevColor(r.level)}`}>{r.level}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${sevColor(r.level)}`}
                             style={{ width: `${(r.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{r.count}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Referral Summary */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-primary text-white py-2 small fw-bold">Referral Summary</div>
              <div className="card-body p-3 text-center">
                <div className="row">
                  <div className="col-6">
                    <div className="h2 text-primary mb-0">{ov.referral_summary?.to_neurology ?? 0}</div>
                    <div className="small text-muted">To Neurology</div>
                  </div>
                  <div className="col-6">
                    <div className="h2 text-info mb-0">{ov.referral_summary?.from_neurology ?? 0}</div>
                    <div className="small text-muted">From Neurology</div>
                  </div>
                </div>
                <hr className="my-2" />
                <div className="small text-muted">Total cross-referrals: {ov.referral_summary?.total ?? 0}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Threshold Flags ── */}
      {tab === 'flags' && flags && (
        <div>
          {/* Instrument summary */}
          <div className="row mb-3">
            {(flags.instrument_summary || []).map(ins => (
              <div key={ins.instrument} className="col-md-3 mb-2">
                <div className="card shadow-sm border-0">
                  <div className="card-body py-2 px-3 text-center">
                    <div className={`h3 mb-0 text-${ins.flagged_count > 10 ? 'danger' : ins.flagged_count > 5 ? 'warning' : 'primary'}`}>
                      {ins.flagged_count}
                    </div>
                    <div className="small fw-bold">{ins.instrument} &ge; {ins.cutoff}</div>
                    <div className="text-muted" style={{ fontSize: '0.68rem' }}>{ins.action}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Severity distribution */}
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-dark text-white py-2 small fw-bold">Alert Priority Distribution</div>
                <div className="card-body p-2">
                  {(flags.severity_distribution || []).map(s => {
                    const max = Math.max(...(flags.severity_distribution || []).map(x => x.count), 1);
                    return (
                      <div key={s.priority} className="d-flex align-items-center mb-1">
                        <span className="small me-2" style={{ minWidth: '80px' }}>
                          <span className={`badge bg-${riskColor(s.priority)}`}>{s.priority}</span>
                        </span>
                        <div className="progress flex-grow-1" style={{ height: '16px' }}>
                          <div className={`progress-bar bg-${riskColor(s.priority)}`}
                               style={{ width: `${(s.count / max * 100).toFixed(0)}%` }}>
                            {s.count > 0 && <span style={{ fontSize: '0.65rem' }}>{s.count}</span>}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="alert alert-info small mb-0">
                <strong>Total flagged patients:</strong> {flags.total_flagged} &middot;{' '}
                <strong>Total alerts:</strong> {flags.total_alerts}
              </div>
            </div>
          </div>

          {/* Flagged patients table */}
          <div className="card shadow-sm border-0">
            <div className="card-header bg-danger text-white py-2 small fw-bold">Flagged Patients</div>
            <div className="card-body p-2">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Patient</th>
                      <th>Age</th>
                      <th>Gender</th>
                      <th>Priority</th>
                      <th className="text-end">Alerts</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {(flags.flagged_patients || []).map(p => (
                      <tr key={p.patient_id} className={sel === p.patient_id ? 'table-active' : ''}>
                        <td className="small fw-bold">{p.patient_id}</td>
                        <td className="small">{p.age ?? '\u2014'}</td>
                        <td className="small">{p.gender || '\u2014'}</td>
                        <td><span className={`badge bg-${riskColor(p.priority)}`}>{p.priority}</span></td>
                        <td className="text-end small">{p.alert_count}</td>
                        <td>
                          <button className="btn btn-outline-danger btn-sm py-0 px-1"
                                  style={{ fontSize: '0.7rem' }}
                                  onClick={() => setSel(sel === p.patient_id ? null : p.patient_id)}>
                            {sel === p.patient_id ? 'Hide' : 'Detail'}
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Patient alert detail */}
              {sel && flags.flagged_patients && (() => {
                const p = flags.flagged_patients.find(x => x.patient_id === sel);
                if (!p) return null;
                return (
                  <div className="card border-danger mt-2">
                    <div className="card-header bg-danger text-white py-1 small fw-bold">
                      {p.patient_id} \u2014 {p.name} &middot; {p.priority}
                    </div>
                    <div className="card-body p-2">
                      <table className="table table-sm mb-0">
                        <thead><tr><th>Instrument</th><th className="text-end">Score</th><th className="text-end">Threshold</th><th>Severity</th><th>Action</th></tr></thead>
                        <tbody>
                          {(p.alerts || []).map((a, i) => (
                            <tr key={i}>
                              <td className="small fw-bold">{a.instrument}</td>
                              <td className="text-end small text-danger fw-bold">{a.score}</td>
                              <td className="text-end small">{a.threshold}</td>
                              <td><span className={`badge bg-${sevColor(a.severity)}`}>{a.severity}</span></td>
                              <td className="small">{a.action}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        </div>
      )}

      {/* ── PNES / Mood-Seizure Breakdown ── */}
      {tab === 'breakdown' && bd && (
        <div className="row">
          {/* PNES Candidates */}
          {bd.pnes_candidates && bd.pnes_candidates.length > 0 && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-warning text-dark py-2 small fw-bold">
                  PNES Candidates ({bd.pnes_candidates.length})
                </div>
                <div className="card-body p-2">
                  <div className="table-responsive">
                    <table className="table table-sm table-hover mb-0">
                      <thead><tr><th>Patient</th><th>Indicators</th></tr></thead>
                      <tbody>
                        {bd.pnes_candidates.map((p, i) => (
                          <tr key={i}>
                            <td className="small fw-bold">{p.patient_id || p.name}</td>
                            <td className="small">{
                              typeof p === 'object'
                                ? Object.entries(p).filter(([k]) => k !== 'patient_id' && k !== 'name')
                                    .map(([k, v]) => `${k}: ${v}`).join(', ')
                                : String(p)
                            }</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Mood-Seizure Correlation */}
          {bd.mood_seizure_correlation && bd.mood_seizure_correlation.length > 0 && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-info text-white py-2 small fw-bold">
                  Mood-Seizure Correlation ({bd.mood_seizure_correlation.length})
                </div>
                <div className="card-body p-2">
                  <div className="table-responsive">
                    <table className="table table-sm table-hover mb-0">
                      <thead><tr><th>Patient</th><th>Details</th></tr></thead>
                      <tbody>
                        {bd.mood_seizure_correlation.map((m, i) => (
                          <tr key={i}>
                            <td className="small fw-bold">{m.patient_id || m.name}</td>
                            <td className="small">{
                              typeof m === 'object'
                                ? Object.entries(m).filter(([k]) => k !== 'patient_id' && k !== 'name')
                                    .map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(2) : v}`).join(', ')
                                : String(m)
                            }</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Patient Profiles */}
          {bd.patient_profiles && bd.patient_profiles.length > 0 && (
            <div className="col-12 mb-3">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-dark text-white py-2 small fw-bold">
                  Patient Psychiatric Profiles ({bd.patient_profiles.length})
                </div>
                <div className="card-body p-2">
                  <div className="table-responsive">
                    <table className="table table-sm table-hover mb-0">
                      <thead className="table-dark">
                        <tr>
                          <th>Patient</th>
                          <th>Age</th>
                          <th>Gender</th>
                          <th>Comorbidities</th>
                          <th>Treatment</th>
                          <th className="text-end">PHQ-9</th>
                          <th className="text-end">GAD-7</th>
                          <th>Risk</th>
                        </tr>
                      </thead>
                      <tbody>
                        {bd.patient_profiles.map((p, i) => (
                          <tr key={i}>
                            <td className="small fw-bold">{p.patient_id}</td>
                            <td className="small">{p.age ?? '\u2014'}</td>
                            <td className="small">{p.gender || '\u2014'}</td>
                            <td className="small">{
                              Array.isArray(p.comorbidities) ? p.comorbidities.join(', ') :
                              typeof p.comorbidity_count === 'number' ? `${p.comorbidity_count} conditions` :
                              p.comorbidities || '\u2014'
                            }</td>
                            <td className="small">
                              {p.treatment_status && (
                                <span className={`badge bg-${
                                  p.treatment_status === 'stable' ? 'success' :
                                  p.treatment_status === 'under_treatment' ? 'primary' :
                                  p.treatment_status === 'untreated' ? 'warning' : 'danger'
                                }`}>{p.treatment_status.replace(/_/g, ' ')}</span>
                              )}
                            </td>
                            <td className="text-end small">{p.phq9_score ?? p.latest_phq9 ?? '\u2014'}</td>
                            <td className="text-end small">{p.gad7_score ?? p.latest_gad7 ?? '\u2014'}</td>
                            <td>
                              {p.risk_level && (
                                <span className={`badge bg-${sevColor(p.risk_level)}`}>{p.risk_level}</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* PNES Features */}
          {bd.pnes_features && bd.pnes_features.length > 0 && (
            <div className="col-12 mb-3">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-secondary text-white py-2 small fw-bold">PNES Differentiating Features</div>
                <div className="card-body p-2">
                  <table className="table table-sm mb-0">
                    <tbody>
                      {bd.pnes_features.map((f, i) => (
                        <tr key={i}>
                          <td className="small fw-bold" style={{ width: '30%' }}>{f.feature || f.name || Object.keys(f)[0]}</td>
                          <td className="small">{f.description || f.value || Object.values(f)[0]}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Clinical Definitions ── */}
      {tab === 'definitions' && defs && (
        <div>
          {defs.sections ? (
            defs.sections.map(s => (
              <div key={s.heading} className="card shadow-sm border-0 mb-3">
                <div className="card-header bg-dark text-white py-2 small fw-bold">{s.heading}</div>
                <div className="card-body p-2">
                  <table className="table table-sm mb-0">
                    <tbody>
                      {(s.definitions || []).map(d => (
                        <tr key={d.term}>
                          <td className="small fw-bold" style={{ width: '30%' }}>{d.term}</td>
                          <td className="small">{d.definition}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))
          ) : defs.concepts ? (
            <div className="card shadow-sm border-0">
              <div className="card-header bg-dark text-white py-2 small fw-bold">Psychiatric Concepts in Epilepsy</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {defs.concepts.map(c => (
                      <tr key={c.name}>
                        <td className="small fw-bold" style={{ width: '30%' }}>{c.name}</td>
                        <td className="small">{c.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
