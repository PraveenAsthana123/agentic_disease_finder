'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const scoreColor = s =>
  s === 4 ? 'success' :
  s === 3 ? 'primary' :
  s === 2 ? 'warning' :
  s === 1 ? 'danger' : 'secondary';

const scoreLabel = s =>
  ['Not candidate', 'Poor', 'Fair', 'Good', 'Excellent'][s] ?? '—';

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className={`h4 mb-0 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Flag({ ok, label }) {
  return (
    <span className={`badge me-1 bg-${ok ? 'success' : 'secondary'}`}>{label}</span>
  );
}

const TABS = [
  { id: 'overview',     label: 'Overview' },
  { id: 'candidates',   label: 'Candidacy Cards' },
  { id: 'mri',          label: 'MRI Concordance' },
  { id: 'neuropsych',   label: 'Neuropsych Risk' },
  { id: 'definitions',  label: 'Definitions' },
];

export default function PreSurgicalEvalPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [err,  setErr]  = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/presurgical-eval/overview`).then(r => r.json()),
      fetch(`${API}/api/presurgical-eval/breakdown`).then(r => r.json()),
      fetch(`${API}/api/presurgical-eval/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-4">Loading Pre-Surgical Evaluation…</div>;

  const k = ov.kpis || {};

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">🔬 Pre-Surgical Epilepsy Evaluation</h4>
      <p className="text-muted small mb-3">
        Surgical candidacy screening — focal onset · MRI lesion concordance · DRE flags · neuropsychological risk.
        Sources: <code>seizure_metadata</code> ({ov.data_sources?.seizure_metadata_records} records) ·{' '}
        <code>mri_findings</code> ({ov.data_sources?.mri_records}) ·{' '}
        <code>neuropsych</code> ({ov.data_sources?.neuropsych_records}) ·{' '}
        <code>medications</code> ({ov.data_sources?.medication_records})
      </p>

      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Patients Screened"   value={k.total_patients}                      color="primary" />
        <KPI label="Focal Onset"         value={`${k.focal_onset_pct}%`}               color="info"    sub="of screened patients" />
        <KPI label="MRI Lesional"        value={`${k.mri_lesional_pct}%`}              color="info"    sub="structural lesion found" />
        <KPI label="Lateralized"         value={`${k.lateralized_pct}%`}               color="warning" sub="clear hemisphere" />
        <KPI label="DRE (≥2 ASM fails)"  value={`${k.dre_pct}%`}                       color={k.dre_pct > 30 ? 'danger' : 'secondary'} sub="drug-resistant" />
        <KPI label="Good Candidates (≥3)" value={k.high_candidates}                    color="success" sub="score ≥ 3/4" />
        <KPI label="Avg MoCA"            value={k.avg_moca}                            color={k.avg_moca < 24 ? 'danger' : 'success'} sub="threshold ≥ 24" />
        <KPI label="Neuro-Psych Risk"    value={k.np_risk_count}                       color="warning" sub="cognitive or mood risk" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Candidacy Score Distribution */}
          <div className="col-md-5">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Surgical Candidacy Score Distribution</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Score</th><th>Label</th><th>Patients</th></tr>
                  </thead>
                  <tbody>
                    {(ov.candidacy_dist || []).map(row => (
                      <tr key={row.score}>
                        <td><span className={`badge bg-${scoreColor(row.score)}`}>{row.score}/4</span></td>
                        <td>{row.label}</td>
                        <td className="fw-semibold">{row.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Onset Zone Distribution */}
          <div className="col-md-7">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Seizure Onset Zone Distribution</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Onset Zone</th><th>Count</th><th>Bar</th></tr>
                  </thead>
                  <tbody>
                    {(ov.onset_zone_dist || []).map(row => {
                      const pct = Math.round(row.count / k.total_patients * 100);
                      const focal = row.zone && !row.zone.includes('Generalized') && !row.zone.includes('bilateral');
                      return (
                        <tr key={row.zone}>
                          <td>
                            {focal && <span className="badge bg-info me-1" style={{fontSize:'0.65rem'}}>focal</span>}
                            {row.zone}
                          </td>
                          <td className="fw-semibold">{row.count}</td>
                          <td style={{ width: '40%' }}>
                            <div className="progress" style={{ height: 10 }}>
                              <div
                                className={`progress-bar bg-${focal ? 'primary' : 'secondary'}`}
                                style={{ width: `${pct}%` }}
                              />
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

          {/* Laterality & Lesion */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Lateralization Distribution</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Side</th><th>Count</th></tr></thead>
                  <tbody>
                    {(ov.laterality_dist || []).map(row => (
                      <tr key={row.side}>
                        <td>{row.side}</td>
                        <td className="fw-semibold">{row.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">MRI Lesion Types</div>
              <div className="card-body p-2">
                {(ov.lesion_type_dist || []).length === 0
                  ? <p className="text-muted small mb-0">No lesion types extracted</p>
                  : (
                    <table className="table table-sm mb-0">
                      <thead className="table-light"><tr><th>Lesion Type</th><th>Count</th></tr></thead>
                      <tbody>
                        {(ov.lesion_type_dist || []).map(row => (
                          <tr key={row.lesion}>
                            <td><code>{row.lesion}</code></td>
                            <td className="fw-semibold">{row.count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── CANDIDACY CARDS TAB ── */}
      {tab === 'candidates' && (
        <div>
          <p className="text-muted small">
            {bd?.total_evaluated} patients scored. Sorted by candidacy score (highest first).
            Score = focal onset + MRI lesional + lateralized + DRE (≥2 ASMs). Max = 4.
          </p>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th><th>Age</th><th>Score</th><th>Criteria Met</th>
                  <th>Onset Zone</th><th>Lateralization</th><th>Lesion</th>
                  <th>MoCA</th><th>PHQ-9</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.patient_cards || []).slice(0, 50).map(c => (
                  <tr key={c.patient_id}>
                    <td><code className="small">{c.patient_id}</code></td>
                    <td>{c.age ?? '—'}</td>
                    <td>
                      <span className={`badge bg-${scoreColor(c.candidacy_score)}`}>
                        {c.candidacy_score}/4 {scoreLabel(c.candidacy_score)}
                      </span>
                    </td>
                    <td>
                      <Flag ok={c.focal_onset}  label="Focal" />
                      <Flag ok={c.mri_lesional} label="MRI" />
                      <Flag ok={c.lateralized}  label="Lat" />
                      <Flag ok={c.dre}          label="DRE" />
                    </td>
                    <td><span className="small">{c.onset_zone || '—'}</span></td>
                    <td>{c.lateralization || '—'}</td>
                    <td>{c.lesion_type ? <code className="small">{c.lesion_type}</code> : <span className="text-muted">—</span>}</td>
                    <td>
                      {c.moca != null
                        ? <span className={c.moca < 24 ? 'text-danger fw-bold' : ''}>{c.moca}</span>
                        : '—'}
                    </td>
                    <td>
                      {c.phq9 != null
                        ? <span className={c.phq9 >= 15 ? 'text-danger fw-bold' : ''}>{c.phq9}</span>
                        : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {(bd?.patient_cards || []).length > 50 && (
            <p className="text-muted small">Showing top 50 of {bd.patient_cards.length} evaluated patients.</p>
          )}
        </div>
      )}

      {/* ── MRI CONCORDANCE TAB ── */}
      {tab === 'mri' && (
        <div>
          <p className="text-muted small">
            MRI concordance = structural lesion laterality matches EEG lateralization.
            {bd?.mri_concordance?.length} patients with MRI available.
          </p>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th><th>EEG Lateralization</th><th>MRI Lateralization</th>
                  <th>Lesion</th><th>Concordant</th><th>MRI Quality</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.mri_concordance || []).map(r => (
                  <tr key={r.patient_id}>
                    <td><code className="small">{r.patient_id}</code></td>
                    <td>{r.eeg_lateralization}</td>
                    <td>{r.mri_lateralization}</td>
                    <td><span className="small">{r.lesion_label || r.lesion_type || '—'}</span></td>
                    <td>
                      {r.concordant === true
                        ? <span className="badge bg-success">✓ Yes</span>
                        : r.concordant === false
                        ? <span className="badge bg-warning text-dark">✗ No</span>
                        : <span className="badge bg-secondary">Unknown</span>}
                    </td>
                    <td>{r.mri_quality}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── NEUROPSYCH RISK TAB ── */}
      {tab === 'neuropsych' && (
        <div>
          <p className="text-muted small">
            Neuropsychological risk screening. MoCA &lt; 24 = cognitive risk.
            PHQ-9 ≥ 15 = moderate-severe depression. GAD-7 ≥ 15 = severe anxiety.
          </p>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th><th>Battery</th><th>MoCA</th><th>MMSE</th>
                  <th>PHQ-9</th><th>GAD-7</th><th>Cognitive Risk</th><th>Mood Risk</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.neuropsych_matrix || []).map(r => (
                  <tr key={r.patient_id} className={r.cognitive_risk || r.mood_risk ? 'table-warning' : ''}>
                    <td><code className="small">{r.patient_id}</code></td>
                    <td><span className="badge bg-secondary">{r.battery}</span></td>
                    <td>
                      {r.moca != null
                        ? <span className={r.cognitive_risk ? 'text-danger fw-bold' : ''}>{r.moca}</span>
                        : '—'}
                    </td>
                    <td>{r.mmse ?? '—'}</td>
                    <td>
                      {r.phq9 != null
                        ? <span className={r.mood_risk ? 'text-danger fw-bold' : ''}>{r.phq9}</span>
                        : '—'}
                    </td>
                    <td>{r.gad7 ?? '—'}</td>
                    <td>
                      {r.cognitive_risk
                        ? <span className="badge bg-danger">Yes</span>
                        : <span className="badge bg-success">No</span>}
                    </td>
                    <td>
                      {r.mood_risk
                        ? <span className="badge bg-danger">Yes</span>
                        : <span className="badge bg-success">No</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Surgical Candidacy Score</div>
              <div className="card-body p-3">
                <p className="small mb-2">{defs.surgical_candidacy_score?.description}</p>
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Score</th><th>Label</th></tr></thead>
                  <tbody>
                    {Object.entries(defs.surgical_candidacy_score?.thresholds || {}).map(([s, label]) => (
                      <tr key={s}>
                        <td><span className={`badge bg-${scoreColor(parseInt(s))}`}>{s}/4</span></td>
                        <td className="small">{label}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">DRE Definition</div>
              <div className="card-body p-3">
                <p className="small mb-1"><strong>Standard:</strong> {defs.dre_definition?.standard}</p>
                <p className="small mb-0">{defs.dre_definition?.criteria}</p>
              </div>
            </div>
            <div className="card shadow-sm mt-3">
              <div className="card-header fw-semibold">Clinical Standards</div>
              <div className="card-body p-3">
                <ul className="small mb-0">
                  {(defs.clinical_standards || []).map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Abbreviations</div>
              <div className="card-body p-3">
                <table className="table table-sm mb-0">
                  <tbody>
                    {Object.entries(defs.abbreviations || {}).map(([abbr, full]) => (
                      <tr key={abbr}>
                        <td><code>{abbr}</code></td>
                        <td className="small">{full}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Data Sources</div>
              <div className="card-body p-3">
                <table className="table table-sm mb-0">
                  <tbody>
                    {Object.entries(defs.data_sources || {}).map(([src, desc]) => (
                      <tr key={src}>
                        <td><code className="small">{src}</code></td>
                        <td className="small">{desc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
