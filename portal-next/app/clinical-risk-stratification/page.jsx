'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const tierColor = t =>
  t === 'Critical' ? 'danger' :
  t === 'High'     ? 'warning' :
  t === 'Moderate' ? 'primary' : 'success';

export default function ClinicalRiskStratificationPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/clinical-risk-stratification/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/clinical-risk-stratification/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/clinical-risk-stratification/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'patients',    label: 'Per Patient' },
    { id: 'components',  label: 'Component Analysis' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f6a8; Clinical Risk Stratification Dashboard</h3>
      <p className="text-muted small">
        Composite per-patient epilepsy risk scoring from seizure burden, medication adherence,
        genetic risk, comorbidity burden, and quality-of-life deficit. Identifies Critical/High/Moderate/Low
        risk patients for targeted intervention.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Patients',   value: ov.total_patients,                   color: 'primary' },
          { label: 'Avg Risk Score',   value: ov.avg_composite_score,              color: ov.avg_composite_score >= 35 ? 'danger' : ov.avg_composite_score >= 23 ? 'warning' : 'info' },
          { label: 'Critical',         value: ov.critical_count,                   color: 'danger' },
          { label: 'High',             value: ov.high_count,                       color: 'warning' },
          { label: 'Moderate',         value: ov.moderate_count,                   color: 'primary' },
          { label: 'Low',              value: ov.low_count,                        color: 'success' },
          { label: 'High-Risk Rate',   value: `${ov.high_risk_rate_pct}%`,         color: ov.high_risk_rate_pct > 30 ? 'danger' : 'secondary' },
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

      {/* ── Overview Tab ──────────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Tier Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Risk Tier Distribution</div>
              <div className="card-body">
                {(ov.tier_distribution || []).map((t, i) => (
                  <div key={i} className="d-flex align-items-center mb-3">
                    <span className={`badge bg-${tierColor(t.tier)} me-2`} style={{minWidth: 72}}>{t.tier}</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 22}}>
                        <div className={`progress-bar bg-${tierColor(t.tier)}`}
                             style={{width: `${t.pct}%`}}>
                          {t.count}
                        </div>
                      </div>
                    </div>
                    <span className="small text-muted">{t.pct}%</span>
                  </div>
                ))}
                <div className="alert alert-danger small mt-2 mb-0 py-2">
                  <strong>{ov.critical_count + ov.high_count}</strong> patients need enhanced follow-up
                </div>
              </div>
            </div>
          </div>

          {/* Score Histogram */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Score Distribution (0–100)</div>
              <div className="card-body">
                <div className="small text-muted mb-2">Higher score = greater risk</div>
                {(ov.score_histogram || []).filter(b => b.count > 0 || parseInt(b.range) < 50).map((b, i) => {
                  const rangeStart = parseInt(b.range);
                  const barColor = rangeStart >= 35 ? 'bg-danger' : rangeStart >= 23 ? 'bg-warning' : rangeStart >= 12 ? 'bg-primary' : 'bg-success';
                  return (
                    <div key={i} className="d-flex align-items-center mb-2">
                      <span className="small" style={{minWidth: 48}}>{b.range}</span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{height: 20}}>
                          <div className={`progress-bar ${barColor}`}
                               style={{width: `${b.count > 0 ? Math.max(8, b.count / ov.total_patients * 100) : 0}%`}}>
                            {b.count}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Component Averages */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Average Component Scores</div>
              <div className="card-body">
                {(ov.avg_components || []).map((c, i) => {
                  const pct = c.max > 0 ? (c.avg / c.max * 100) : 0;
                  return (
                    <div key={i} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold">{c.component}</span>
                        <span className="text-muted">{c.avg} / {c.max}</span>
                      </div>
                      <div className="progress" style={{height: 16}}>
                        <div className={`progress-bar ${pct >= 50 ? 'bg-danger' : pct >= 30 ? 'bg-warning' : 'bg-success'}`}
                             style={{width: `${pct}%`}}>
                          {c.avg}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Gender Breakdown */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Gender Breakdown</div>
              <div className="card-body">
                {(ov.gender_breakdown || []).map((g, i) => (
                  <div key={i} className="d-flex justify-content-between align-items-center mb-2">
                    <span className="small fw-semibold">{g.gender || 'Unknown'}</span>
                    <span className="small text-muted">{g.count} patients</span>
                    <span className={`badge bg-${g.avg_score >= 23 ? 'warning' : 'info'}`}>avg: {g.avg_score}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* High-Risk Patients */}
          <div className="col-md-8 mb-3">
            <div className="card shadow-sm border-danger">
              <div className="card-header fw-bold bg-danger text-white">
                High-Risk Patients — Top {(ov.high_risk_patients || []).length}
              </div>
              <div className="card-body p-0">
                <div style={{maxHeight: 320, overflowY: 'auto'}}>
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark sticky-top">
                      <tr>
                        <th>Patient</th><th>Score</th><th>Tier</th>
                        <th>Seizure</th><th>Adherence</th><th>Genetic</th>
                        <th>Comorbidity</th><th>QoL</th><th>Age</th><th>Gender</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(ov.high_risk_patients || []).map((p, i) => (
                        <tr key={i} className={p.tier === 'Critical' ? 'table-danger' : ''}>
                          <td className="fw-semibold small">{p.patient_id}</td>
                          <td className="fw-bold text-danger">{p.composite_score}</td>
                          <td><span className={`badge bg-${tierColor(p.tier)}`}>{p.tier}</span></td>
                          <td className="small">{p.seizure_burden}</td>
                          <td className="small">{p.adherence_risk}</td>
                          <td className="small">{p.genetic_risk}</td>
                          <td className="small">{p.comorbidity_burden}</td>
                          <td className="small">{p.qol_deficit}</td>
                          <td className="small">{p.age ?? '—'}</td>
                          <td className="small">{p.gender || '—'}</td>
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

      {/* ── Per Patient Tab ───────────────────────────────────────── */}
      {tab === 'patients' && bd && (
        <div>
          <div className="card shadow-sm">
            <div className="card-header fw-bold">
              All Patients — Risk Profiles ({(bd.all_patients || []).length} patients)
            </div>
            <div className="card-body p-0">
              <div style={{maxHeight: 560, overflowY: 'auto'}}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark sticky-top">
                    <tr>
                      <th>Patient</th>
                      <th>Composite</th>
                      <th>Tier</th>
                      <th>Seizure (30)</th>
                      <th>Adherence (25)</th>
                      <th>Genetic (20)</th>
                      <th>Comorbidity (15)</th>
                      <th>QoL (10)</th>
                      <th>Age</th>
                      <th>Gender</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.all_patients || []).map((p, i) => (
                      <tr key={i} className={p.tier === 'Critical' ? 'table-danger' : p.tier === 'High' ? 'table-warning' : ''}>
                        <td className="fw-semibold small">{p.patient_id}</td>
                        <td>
                          <div className="d-flex align-items-center gap-1">
                            <div className="progress flex-grow-1" style={{height: 14, minWidth: 60}}>
                              <div className={`progress-bar bg-${tierColor(p.tier)}`}
                                   style={{width: `${Math.min(p.composite_score, 100)}%`}} />
                            </div>
                            <span className="small fw-bold">{p.composite_score}</span>
                          </div>
                        </td>
                        <td><span className={`badge bg-${tierColor(p.tier)}`}>{p.tier}</span></td>
                        <td>{compBar(p.seizure_burden, 30)}</td>
                        <td>{compBar(p.adherence_risk, 25)}</td>
                        <td>{compBar(p.genetic_risk, 20)}</td>
                        <td>{compBar(p.comorbidity_burden, 15)}</td>
                        <td>{compBar(p.qol_deficit, 10)}</td>
                        <td className="small">{p.age ?? '—'}</td>
                        <td className="small">{p.gender || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Component Analysis Tab ────────────────────────────────── */}
      {tab === 'components' && bd && (
        <div className="row">
          {/* Component waterfall - highest contributors */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Component Contribution (Avg Points)</div>
              <div className="card-body">
                {(ov.avg_components || []).map((c, i) => {
                  const pct = c.max > 0 ? (c.avg / c.max * 100) : 0;
                  return (
                    <div key={i} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold">{c.component}</span>
                        <span className="text-muted">{c.avg} / {c.max} pts ({Math.round(pct)}%)</span>
                      </div>
                      <div className="progress" style={{height: 20}}>
                        <div className={`progress-bar ${pct >= 50 ? 'bg-danger' : pct >= 30 ? 'bg-warning' : 'bg-success'}`}
                             style={{width: `${pct}%`}}>
                          {c.avg}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Critical patients component breakdown */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-danger">
              <div className="card-header fw-bold bg-danger text-white">Critical Patient Component Waterfall</div>
              <div className="card-body">
                {(ov.high_risk_patients || []).filter(p => p.tier === 'Critical').map((p, i) => (
                  <div key={i} className="mb-4">
                    <div className="fw-semibold small mb-2">{p.patient_id} — Score: {p.composite_score}</div>
                    {[
                      { label: 'Seizure Burden',    val: p.seizure_burden,    max: 30 },
                      { label: 'Adherence Risk',    val: p.adherence_risk,    max: 25 },
                      { label: 'Genetic Risk',      val: p.genetic_risk,      max: 20 },
                      { label: 'Comorbidity Burden', val: p.comorbidity_burden, max: 15 },
                      { label: 'QoL Deficit',        val: p.qol_deficit,       max: 10 },
                    ].map((c, j) => (
                      <div key={j} className="d-flex align-items-center mb-1">
                        <span className="small" style={{minWidth: 130}}>{c.label}</span>
                        <div className="flex-grow-1 mx-2">
                          <div className="progress" style={{height: 14}}>
                            <div className={`progress-bar ${c.val / c.max >= 0.5 ? 'bg-danger' : c.val / c.max >= 0.3 ? 'bg-warning' : 'bg-info'}`}
                                 style={{width: `${c.max > 0 ? c.val / c.max * 100 : 0}%`}}>
                            </div>
                          </div>
                        </div>
                        <span style={{fontSize: '0.7rem', minWidth: 48}}>{c.val}/{c.max}</span>
                      </div>
                    ))}
                  </div>
                ))}
                {(ov.high_risk_patients || []).filter(p => p.tier === 'Critical').length === 0 && (
                  <div className="text-muted small">No critical patients currently.</div>
                )}
              </div>
            </div>
          </div>

          {/* Gender comparison */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Risk by Gender</div>
              <div className="card-body">
                {(ov.gender_breakdown || []).map((g, i) => {
                  const maxScore = Math.max(...(ov.gender_breakdown || []).map(x => x.avg_score), 1);
                  return (
                    <div key={i} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold">{g.gender || 'Unknown'} ({g.count})</span>
                        <span className="text-muted">avg: {g.avg_score}</span>
                      </div>
                      <div className="progress" style={{height: 20}}>
                        <div className={`progress-bar bg-${g.avg_score >= 23 ? 'warning' : 'info'}`}
                             style={{width: `${g.avg_score / maxScore * 100}%`}}>
                          {g.avg_score}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Tier action recommendations */}
          {defs && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Tier Action Recommendations</div>
                <div className="card-body">
                  {(defs.risk_tiers || []).map((t, i) => (
                    <div key={i} className="mb-3">
                      <div className="d-flex align-items-center mb-1">
                        <span className={`badge bg-${tierColor(t.tier)} me-2`} style={{minWidth: 72}}>{t.tier}</span>
                        <span className="small text-muted">{t.range}</span>
                      </div>
                      <div className="small">{t.action}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Definitions Tab ───────────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-12 mb-3">
            <div className="card shadow-sm border-primary">
              <div className="card-header fw-bold bg-primary text-white">Dashboard Purpose</div>
              <div className="card-body small">{defs.description}</div>
            </div>
          </div>

          {/* Risk Tiers */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Risk Tiers</div>
              <div className="card-body">
                {(defs.risk_tiers || []).map((t, i) => (
                  <div key={i} className="mb-3">
                    <div className="d-flex align-items-center mb-1">
                      <span className={`badge bg-${tierColor(t.tier)} me-2`}>{t.tier}</span>
                      <span className="small text-muted">Score: {t.range}</span>
                    </div>
                    <div className="small">{t.action}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Risk Components */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Scoring Components</div>
              <div className="card-body" style={{maxHeight: 400, overflowY: 'auto'}}>
                {(defs.risk_components || []).map((c, i) => (
                  <div key={i} className="mb-3 pb-2 border-bottom">
                    <div className="fw-semibold small text-primary">
                      {c.component} <span className="badge bg-secondary ms-1">max {c.max_points} pts</span>
                    </div>
                    <div className="small text-muted mt-1">Source: {(c.sources || []).join(', ')}</div>
                    {(c.sub_factors || []).map((sf, j) => (
                      <div key={j} className="small ms-3 mt-1">
                        <span className="fw-semibold">{sf.factor}</span> (max {sf.max}) — {sf.note}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Data Sources */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Data Sources</div>
              <div className="card-body">
                {(defs.data_sources || []).map((s, i) => (
                  <div key={i} className="mb-2 small">
                    <strong>{s.table}</strong>: {s.rows} rows, {s.patients} patients
                    <div className="text-muted">Fields: {s.key_fields}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Glossary */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Glossary</div>
              <div className="card-body" style={{maxHeight: 320, overflowY: 'auto'}}>
                {(defs.glossary || []).map((g, i) => (
                  <div key={i} className="mb-2">
                    <strong className="text-primary small">{g.term}</strong>
                    <div className="small text-muted">{g.definition}</div>
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

function compBar(val, max) {
  const v = val ?? 0;
  const pct = max > 0 ? (v / max * 100) : 0;
  const color = pct >= 50 ? 'bg-danger' : pct >= 30 ? 'bg-warning' : 'bg-success';
  return (
    <div className="d-flex align-items-center gap-1">
      <div className="progress" style={{height: 10, minWidth: 44, flexGrow: 1}}>
        <div className={`progress-bar ${color}`} style={{width: `${pct}%`}} />
      </div>
      <span style={{fontSize: '0.7rem'}}>{v}</span>
    </div>
  );
}
