'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const classColor = c =>
  c === 'pnes_likely'      ? 'danger'  :
  c === 'epilepsy_likely'  ? 'primary' :
  c === 'mixed'            ? 'warning' : 'secondary';

const classLabel = c =>
  c === 'pnes_likely'      ? 'PNES Likely'     :
  c === 'epilepsy_likely'  ? 'Epilepsy Likely'  :
  c === 'mixed'            ? 'Mixed'            :
  c === 'indeterminate'    ? 'Indeterminate'    : c;

const statusColor = s =>
  s === 'confirmed_pnes'      ? 'danger'  :
  s === 'confirmed_epilepsy'  ? 'primary' :
  s === 'reviewed'            ? 'success' : 'warning';

export default function PNESScreeningPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/pnes-screening/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/pnes-screening/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/pnes-screening/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const pnesCount      = (ov.classification_distribution || []).find(d => d.classification === 'pnes_likely')?.count ?? 0;
  const epilepsyCount  = (ov.classification_distribution || []).find(d => d.classification === 'epilepsy_likely')?.count ?? 0;
  const mixedCount     = (ov.classification_distribution || []).find(d => d.classification === 'mixed')?.count ?? 0;
  const indetermCount  = (ov.classification_distribution || []).find(d => d.classification === 'indeterminate')?.count ?? 0;

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'patients',   label: 'Per Patient' },
    { id: 'semiology',  label: 'Semiology & EEG' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f9e0; PNES Screening Dashboard</h3>
      <p className="text-muted small">
        Psychogenic Non-Epileptic Seizure (PNES) differential screening — semiological scores,
        EEG findings, PNES vs epilepsy probability, psychiatric comorbidities, and referral tracking.
        {ov.total_screenings} screenings across {ov.total_patients} patients.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Screenings', value: ov.total_screenings,           color: 'primary' },
          { label: 'Patients',         value: ov.total_patients,              color: 'info' },
          { label: 'PNES Likely',      value: pnesCount,                      color: 'danger' },
          { label: 'Epilepsy Likely',  value: epilepsyCount,                  color: 'primary' },
          { label: 'Mixed/Concurrent', value: mixedCount,                     color: 'warning' },
          { label: 'Indeterminate',    value: indetermCount,                  color: 'secondary' },
          { label: 'Avg PNES Prob',    value: `${(ov.avg_pnes_probability * 100).toFixed(0)}%`, color: ov.avg_pnes_probability > 0.5 ? 'danger' : 'success' },
          { label: 'Video-EEG Rec\'d', value: `${ov.video_eeg_recommendation_rate?.toFixed(0)}%`, color: 'dark' },
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

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Classification distribution */}
          <div className="col-md-5">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Classification Distribution</div>
              <div className="card-body p-2">
                {(ov.classification_distribution || []).map(d => {
                  const pct = ((d.count / ov.total_screenings) * 100).toFixed(1);
                  return (
                    <div key={d.classification} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>
                          <span className={`badge bg-${classColor(d.classification)} me-2`}>{classLabel(d.classification)}</span>
                        </span>
                        <span className="fw-bold">{d.count} <span className="text-muted">({pct}%)</span></span>
                      </div>
                      <div className="progress" style={{height: '10px'}}>
                        <div className={`progress-bar bg-${classColor(d.classification)}`} style={{width: `${pct}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Status breakdown */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Review Status</div>
              <div className="card-body p-2">
                {(ov.status_distribution || []).map(s => {
                  const pct = ((s.count / ov.total_screenings) * 100).toFixed(1);
                  const label = s.status === 'confirmed_pnes' ? 'Confirmed PNES' :
                                s.status === 'confirmed_epilepsy' ? 'Confirmed Epilepsy' :
                                s.status === 'reviewed' ? 'Reviewed' : 'Pending';
                  return (
                    <div key={s.status} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span><span className={`badge bg-${statusColor(s.status)} me-2`}>{label}</span></span>
                        <span className="fw-bold">{s.count}</span>
                      </div>
                      <div className="progress" style={{height: '8px'}}>
                        <div className={`progress-bar bg-${statusColor(s.status)}`} style={{width: `${pct}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Referral + EEG stats */}
          <div className="col-md-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Clinical Flags</div>
              <div className="card-body p-2">
                <div className="mb-3">
                  <div className="small text-muted mb-1">Video-EEG Recommended</div>
                  <div className="progress mb-1" style={{height: '14px'}}>
                    <div className="progress-bar bg-info" style={{width: `${ov.video_eeg_recommendation_rate}%`}} />
                  </div>
                  <div className="small fw-bold">{ov.video_eeg_recommendation_rate?.toFixed(1)}% of screenings</div>
                </div>
                <div className="mb-3">
                  <div className="small text-muted mb-1">Psychiatry Referral Rate</div>
                  <div className="progress mb-1" style={{height: '14px'}}>
                    <div className="progress-bar bg-danger" style={{width: `${ov.psychiatry_referral_rate}%`}} />
                  </div>
                  <div className="small fw-bold">{ov.psychiatry_referral_rate?.toFixed(1)}%</div>
                </div>
                <div className="mb-3">
                  <div className="small text-muted mb-1">Duration &gt;2 min</div>
                  <div className="progress mb-1" style={{height: '14px'}}>
                    <div className="progress-bar bg-warning" style={{width: `${ov.duration_gt2min_rate}%`}} />
                  </div>
                  <div className="small fw-bold">{ov.duration_gt2min_rate?.toFixed(1)}%</div>
                </div>
                <div className="row text-center small">
                  <div className="col-6 border-end">
                    <div className="fw-bold text-danger">{ov.eeg_ictal_normal_count}</div>
                    <div className="text-muted" style={{fontSize:'0.68rem'}}>Ictal EEG Normal</div>
                  </div>
                  <div className="col-6">
                    <div className="fw-bold text-warning">{ov.eeg_interictal_normal_count}</div>
                    <div className="text-muted" style={{fontSize:'0.68rem'}}>Interictal EEG Normal</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Psychiatric comorbidities */}
          <div className="col-md-5">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Psychiatric Comorbidities</div>
              <div className="card-body p-2">
                {(ov.comorbidity_distribution || []).map(c => {
                  const pct = ((c.count / ov.total_patients) * 100).toFixed(0);
                  return (
                    <div key={c.comorbidity} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize">{c.comorbidity}</span>
                        <span className="fw-bold">{c.count} pts <span className="text-muted">({pct}%)</span></span>
                      </div>
                      <div className="progress" style={{height: '8px'}}>
                        <div className="progress-bar bg-danger" style={{width: `${pct}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Referral reasons */}
          <div className="col-md-4">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Referral Reasons</div>
              <div className="card-body p-2">
                {(ov.referral_reason_distribution || []).map(r => {
                  const pct = ((r.count / ov.total_screenings) * 100).toFixed(0);
                  return (
                    <div key={r.reason} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize">{r.reason}</span>
                        <span className="fw-bold">{r.count}</span>
                      </div>
                      <div className="progress" style={{height: '8px'}}>
                        <div className="progress-bar bg-primary" style={{width: `${pct}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Monthly trend */}
          <div className="col-md-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Monthly Screening Volume</div>
              <div className="card-body p-2" style={{maxHeight:'220px', overflowY:'auto'}}>
                <table className="table table-sm table-hover mb-0" style={{fontSize:'0.78rem'}}>
                  <thead><tr><th>Month</th><th>Screenings</th></tr></thead>
                  <tbody>
                    {(ov.monthly_trend || []).slice(-8).map(m => (
                      <tr key={m.month}>
                        <td>{m.month}</td>
                        <td><span className="badge bg-primary">{m.screenings}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Per Patient Tab ── */}
      {tab === 'patients' && bd && (
        <div>
          <p className="small text-muted mb-2">Per-patient summary — {(bd.per_patient || []).length} patients with ≥1 screening</p>
          <div style={{overflowX:'auto'}}>
            <table className="table table-sm table-hover table-bordered" style={{fontSize:'0.8rem'}}>
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Screenings</th>
                  <th>Latest Classification</th>
                  <th>Dominant Class</th>
                  <th>Avg PNES Prob</th>
                  <th>Avg Confidence</th>
                </tr>
              </thead>
              <tbody>
                {(bd.per_patient || []).map(p => (
                  <tr key={p.patient_id}>
                    <td><strong>{p.patient_id}</strong></td>
                    <td className="text-center">{p.screenings}</td>
                    <td><span className={`badge bg-${classColor(p.latest_classification)}`}>{classLabel(p.latest_classification)}</span></td>
                    <td><span className={`badge bg-${classColor(p.dominant_classification)} bg-opacity-75`}>{classLabel(p.dominant_classification)}</span></td>
                    <td>
                      <div className="d-flex align-items-center gap-2">
                        <div className="progress flex-grow-1" style={{height:'8px'}}>
                          <div
                            className={`progress-bar bg-${p.avg_pnes_probability > 0.65 ? 'danger' : p.avg_pnes_probability > 0.35 ? 'warning' : 'success'}`}
                            style={{width:`${(p.avg_pnes_probability * 100).toFixed(0)}%`}}
                          />
                        </div>
                        <span style={{minWidth:'32px'}}>{(p.avg_pnes_probability * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td>{(p.avg_confidence * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Semiology & EEG Tab ── */}
      {tab === 'semiology' && (
        <div className="row g-3">
          {/* Semiological score averages */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Avg Semiological Scores (0–3 scale)</div>
              <div className="card-body p-3">
                {(ov.semiological_averages || []).map(s => {
                  const pct = ((s.avg_score / 3) * 100).toFixed(0);
                  const color = s.avg_score >= 2 ? 'danger' : s.avg_score >= 1 ? 'warning' : 'success';
                  return (
                    <div key={s.feature} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold">{s.feature}</span>
                        <span className={`text-${color} fw-bold`}>{s.avg_score.toFixed(2)}</span>
                      </div>
                      <div className="progress" style={{height:'12px'}}>
                        <div className={`progress-bar bg-${color}`} style={{width:`${pct}%`}} />
                      </div>
                      <div className="text-muted" style={{fontSize:'0.7rem', marginTop:'2px'}}>
                        {s.avg_score >= 2 ? 'Prominently elevated — strong PNES indicator' :
                         s.avg_score >= 1 ? 'Mildly elevated — noteworthy' :
                         'Low — not strongly indicative of PNES'}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* PNES probability distribution */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">PNES Probability Distribution</div>
              <div className="card-body p-3">
                {(ov.probability_distribution || []).map(p => {
                  const pct = ((p.count / ov.total_screenings) * 100).toFixed(1);
                  const color = p.category.startsWith('High') ? 'danger' :
                                p.category.startsWith('Moderate') ? 'warning' : 'success';
                  return (
                    <div key={p.category} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{p.category}</span>
                        <span className="fw-bold">{p.count} <span className="text-muted">({pct}%)</span></span>
                      </div>
                      <div className="progress" style={{height:'16px'}}>
                        <div className={`progress-bar bg-${color}`} style={{width:`${pct}%`}}>
                          {pct}%
                        </div>
                      </div>
                    </div>
                  );
                })}

                <hr />
                <div className="row text-center small mt-2">
                  <div className="col-6">
                    <div className="text-muted mb-1">Avg Confidence</div>
                    <div className="h4 text-success mb-0">{(ov.avg_confidence * 100).toFixed(0)}%</div>
                  </div>
                  <div className="col-6">
                    <div className="text-muted mb-1">Avg PNES Prob</div>
                    <div className={`h4 mb-0 text-${ov.avg_pnes_probability > 0.5 ? 'danger' : 'primary'}`}>
                      {(ov.avg_pnes_probability * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                <hr />
                <div className="small">
                  <div className="fw-semibold mb-2">EEG Findings Summary</div>
                  <div className="d-flex gap-3">
                    <div className="text-center">
                      <div className="h5 text-danger mb-0">{ov.eeg_ictal_normal_count}</div>
                      <div className="text-muted" style={{fontSize:'0.7rem'}}>Ictal EEG Normal<br/>(PNES gold standard)</div>
                    </div>
                    <div className="text-center">
                      <div className="h5 text-warning mb-0">{ov.eeg_interictal_normal_count}</div>
                      <div className="text-muted" style={{fontSize:'0.7rem'}}>Interictal EEG Normal<br/>(non-specific)</div>
                    </div>
                    <div className="text-center">
                      <div className="h5 text-info mb-0">{(ov.duration_gt2min_rate).toFixed(0)}%</div>
                      <div className="text-muted" style={{fontSize:'0.7rem'}}>Events &gt;2 min<br/>(PNES-suggestive)</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          {/* Classification tiers */}
          <div className="col-md-5">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Classification Tiers & Actions</div>
              <div className="card-body p-2">
                {(defs.classification_tiers || []).map(t => (
                  <div key={t.classification} className={`alert alert-${classColor(t.classification)} py-2 px-3 mb-2`} style={{fontSize:'0.82rem'}}>
                    <div className="fw-bold mb-1">{classLabel(t.classification)}</div>
                    <div className="mb-1">{t.description}</div>
                    <div className="text-muted"><strong>Action:</strong> {t.action}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Scoring guide */}
          <div className="col-md-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Semiological Scoring Guide</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0" style={{fontSize:'0.8rem'}}>
                  <thead><tr><th>Score</th><th>Label</th><th>Meaning</th></tr></thead>
                  <tbody>
                    {(defs.scoring_guide || []).map(s => (
                      <tr key={s.score}>
                        <td className="fw-bold">{s.score}</td>
                        <td><span className={`badge bg-${s.score === 0 ? 'success' : s.score === 1 ? 'warning' : s.score === 2 ? 'danger' : 'dark'}`}>{s.label}</span></td>
                        <td>{s.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Glossary */}
          <div className="col-md-4">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Clinical Glossary</div>
              <div className="card-body p-2" style={{maxHeight:'460px', overflowY:'auto'}}>
                {(defs.glossary || []).map(g => (
                  <div key={g.term} className="mb-2 pb-2 border-bottom">
                    <div className="fw-semibold small">{g.term}</div>
                    <div className="text-muted" style={{fontSize:'0.78rem'}}>{g.definition}</div>
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
