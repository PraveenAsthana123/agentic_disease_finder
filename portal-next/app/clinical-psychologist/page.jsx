'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const impairColor = l =>
  l === 'severe' ? 'danger' :
  l === 'moderate' ? 'warning' :
  l === 'mild' ? 'info' :
  l === 'none' ? 'success' : 'secondary';

const sevColor = l =>
  l === 'severe' || l === 'moderately_severe' ? 'danger' :
  l === 'moderate' ? 'warning' :
  l === 'mild' ? 'info' :
  l === 'minimal' ? 'success' : 'secondary';

const latColor = l =>
  l === 'left' ? 'primary' :
  l === 'right' ? 'info' :
  l === 'bilateral' ? 'warning' : 'secondary';

export default function ClinicalPsychologistPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [sel,  setSel]  = useState(null);

  useEffect(() => {
    fetch(`${API}/api/clinical-psychologist/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/clinical-psychologist/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/clinical-psychologist/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const cogIdxColor = score =>
    score >= 110 ? 'success' :
    score >= 90  ? 'primary' :
    score >= 80  ? 'info' :
    score >= 70  ? 'warning' : 'danger';

  const kpis = [
    { label: 'Total Assessments',     value: ov.total_assessments },
    { label: 'Patients Assessed',     value: ov.total_patients_assessed },
    { label: 'Avg MoCA',              value: ov.avg_moca?.toFixed(1),      color: ov.avg_moca < 26 ? 'warning' : 'success' },
    { label: 'Avg MMSE',              value: ov.avg_mmse?.toFixed(1),      color: ov.avg_mmse < 26 ? 'warning' : 'success' },
    { label: 'MoCA Impaired %',       value: `${ov.moca_impaired_rate?.toFixed(0)}%`, color: 'danger' },
    { label: 'MMSE Impaired %',       value: `${ov.mmse_impaired_rate?.toFixed(0)}%`, color: 'warning' },
    { label: 'Avg PHQ-9',             value: ov.avg_phq9?.toFixed(1),      color: ov.avg_phq9 >= 10 ? 'danger' : 'secondary' },
    { label: 'Avg GAD-7',             value: ov.avg_gad7?.toFixed(1),      color: ov.avg_gad7 >= 10 ? 'danger' : 'secondary' },
  ];

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'cognitive',  label: 'Cognitive Profiles' },
    { id: 'patients',   label: `Patients (${ov.total_patients_assessed})` },
    { id: 'defs',       label: 'Clinical Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f9e0; Clinical Psychologist Dashboard</h3>
      <p className="text-muted small">
        Neuropsychological battery results — MoCA / MMSE / cognitive indices, memory lateralization,
        Trail Making, depression/anxiety comorbidity, and pre-surgical evaluation summaries.
      </p>

      {/* KPI row */}
      <div className="row mb-3">
        {kpis.map(k => (
          <div key={k.label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${k.color || 'primary'}`}>{k.value ?? '\u2014'}</div>
                <div className="text-muted" style={{ fontSize: '0.72rem' }}>{k.label}</div>
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
          {/* Battery Type */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-primary text-white py-2 small fw-bold">Battery Type Distribution</div>
              <div className="card-body p-2">
                {(ov.battery_type_distribution || []).map(b => {
                  const max = Math.max(...(ov.battery_type_distribution || []).map(x => x.count), 1);
                  const color = b.battery_type === 'Full' ? 'primary' : b.battery_type === 'Screening' ? 'info' : 'secondary';
                  return (
                    <div key={b.battery_type} className="d-flex align-items-center mb-2">
                      <span className="small me-2" style={{ minWidth: '90px' }}>
                        <span className={`badge bg-${color}`}>{b.battery_type}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '18px' }}>
                        <div className={`progress-bar bg-${color}`} style={{ width: `${(b.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.7rem' }}>{b.count}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Cognitive Impairment */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">Cognitive Impairment Classification</div>
              <div className="card-body p-2">
                {(ov.impairment_distribution || []).map(b => {
                  const max = Math.max(...(ov.impairment_distribution || []).map(x => x.count), 1);
                  return (
                    <div key={b.level} className="d-flex align-items-center mb-2">
                      <span className="small me-2" style={{ minWidth: '80px' }}>
                        <span className={`badge bg-${impairColor(b.level)}`}>{b.level}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '18px' }}>
                        <div className={`progress-bar bg-${impairColor(b.level)}`}
                             style={{ width: `${(b.count / max * 100).toFixed(0)}%` }}>
                          {b.count > 0 && <span style={{ fontSize: '0.7rem' }}>{b.count}</span>}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Lateralization */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-info text-white py-2 small fw-bold">Memory Lateralization</div>
              <div className="card-body p-2">
                {(ov.lateralization_distribution || []).map(b => {
                  const max = Math.max(...(ov.lateralization_distribution || []).map(x => x.count), 1);
                  return (
                    <div key={b.lateralization} className="d-flex align-items-center mb-2">
                      <span className="small me-2" style={{ minWidth: '110px' }}>
                        <span className={`badge bg-${latColor(b.lateralization)}`}>{b.lateralization}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '18px' }}>
                        <div className={`progress-bar bg-${latColor(b.lateralization)}`}
                             style={{ width: `${(b.count / max * 100).toFixed(0)}%` }}>
                          {b.count > 0 && <span style={{ fontSize: '0.7rem' }}>{b.count}</span>}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Depression Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-primary text-white py-2 small fw-bold">PHQ-9 Depression Severity</div>
              <div className="card-body p-2">
                {(ov.depression_distribution || []).map(b => {
                  const max = Math.max(...(ov.depression_distribution || []).map(x => x.count), 1);
                  return (
                    <div key={b.level} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '120px' }}>
                        <span className={`badge bg-${sevColor(b.level)}`}>{b.level.replace(/_/g, ' ')}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${sevColor(b.level)}`}
                             style={{ width: `${(b.count / max * 100).toFixed(0)}%` }}>
                          {b.count > 0 && <span style={{ fontSize: '0.65rem' }}>{b.count}</span>}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Anxiety Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">GAD-7 Anxiety Severity</div>
              <div className="card-body p-2">
                {(ov.anxiety_distribution || []).map(b => {
                  const max = Math.max(...(ov.anxiety_distribution || []).map(x => x.count), 1);
                  return (
                    <div key={b.level} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '80px' }}>
                        <span className={`badge bg-${sevColor(b.level)}`}>{b.level}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${sevColor(b.level)}`}
                             style={{ width: `${(b.count / max * 100).toFixed(0)}%` }}>
                          {b.count > 0 && <span style={{ fontSize: '0.65rem' }}>{b.count}</span>}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Referral Reasons */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-dark text-white py-2 small fw-bold">Referral Reasons</div>
              <div className="card-body p-2">
                {(ov.referral_reason_distribution || []).map(b => {
                  const max = Math.max(...(ov.referral_reason_distribution || []).map(x => x.count), 1);
                  const color = b.reason === 'pre-surgical' ? 'danger' : b.reason === 'cognitive_complaint' ? 'warning' :
                                b.reason === 'baseline' ? 'success' : 'info';
                  return (
                    <div key={b.reason} className="d-flex align-items-center mb-2">
                      <span className="small me-2" style={{ minWidth: '130px' }}>
                        {b.reason.replace(/_/g, ' ')}
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${color}`}
                             style={{ width: `${(b.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{b.count}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Cognitive Index Means */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-success text-white py-2 small fw-bold">Mean Cognitive Indices (Standardized Scores — Normal Range 90–109)</div>
              <div className="card-body p-2">
                <div className="row">
                  {Object.entries(ov.cognitive_index_means || {}).map(([idx, score]) => (
                    <div key={idx} className="col-md-2 col-6 mb-2 text-center">
                      <div className={`h4 mb-0 text-${cogIdxColor(score)}`}>{score?.toFixed(1)}</div>
                      <div className="small text-muted">{idx}</div>
                      <div className="progress mt-1" style={{ height: '6px' }}>
                        <div className={`progress-bar bg-${cogIdxColor(score)}`}
                             style={{ width: `${Math.min((score / 130) * 100, 100).toFixed(0)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Cognitive Profiles ── */}
      {tab === 'cognitive' && bd && (
        <div className="row">
          {/* Cognitive Profile Chart */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-success text-white py-2 small fw-bold">Cognitive Index Profile (All Patients)</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Index</th><th className="text-end">Mean Score</th><th className="text-end">N</th><th style={{ width: '40%' }}>Profile</th></tr></thead>
                  <tbody>
                    {(bd.cognitive_profile_chart || []).map(c => (
                      <tr key={c.index}>
                        <td className="small fw-bold">{c.index}</td>
                        <td className={`text-end small text-${cogIdxColor(c.mean_score)}`}>{c.mean_score?.toFixed(1)}</td>
                        <td className="text-end small text-muted">{c.n}</td>
                        <td>
                          <div className="progress" style={{ height: '14px' }}>
                            <div className={`progress-bar bg-${cogIdxColor(c.mean_score)}`}
                                 style={{ width: `${Math.min((c.mean_score / 130) * 100, 100).toFixed(0)}%` }} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="mt-2 small text-muted">
                  Normal = 90–109 &nbsp;|&nbsp; Low Average = 80–89 &nbsp;|&nbsp; Impaired &lt; 70
                </div>
              </div>
            </div>
          </div>

          {/* Trail Making */}
          <div className="col-md-3 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">Trail Making Test (TMT)</div>
              <div className="card-body p-3 text-center">
                {bd.trail_making_stats && (
                  <>
                    <div className="mb-2">
                      <div className={`h3 mb-0 text-${bd.trail_making_stats.avg_trail_a_seconds > 29 ? 'warning' : 'success'}`}>
                        {bd.trail_making_stats.avg_trail_a_seconds?.toFixed(1)}s
                      </div>
                      <div className="small text-muted">Avg Trail A</div>
                      <div style={{ fontSize: '0.68rem' }} className="text-muted">Normal &lt; 29s</div>
                    </div>
                    <hr className="my-2" />
                    <div className="mb-2">
                      <div className={`h3 mb-0 text-${bd.trail_making_stats.avg_trail_b_seconds > 75 ? 'danger' : 'success'}`}>
                        {bd.trail_making_stats.avg_trail_b_seconds?.toFixed(1)}s
                      </div>
                      <div className="small text-muted">Avg Trail B</div>
                      <div style={{ fontSize: '0.68rem' }} className="text-muted">Normal &lt; 75s</div>
                    </div>
                    <hr className="my-2" />
                    <div>
                      <div className={`h4 mb-0 text-${bd.trail_making_stats.trail_b_a_ratio > 2.5 ? 'danger' : 'primary'}`}>
                        {bd.trail_making_stats.trail_b_a_ratio?.toFixed(2)}
                      </div>
                      <div className="small text-muted">B:A Ratio</div>
                      <div style={{ fontSize: '0.68rem' }} className="text-muted">&gt;2.5 = executive dysfunction</div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Memory Lateralization by Mean Score */}
          <div className="col-md-3 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-info text-white py-2 small fw-bold">Memory Index by Lateralization</div>
              <div className="card-body p-2">
                {(bd.memory_lateralization || []).map(m => (
                  <div key={m.lateralization} className="mb-2 p-2 rounded" style={{ background: '#f8f9fa' }}>
                    <div className="d-flex justify-content-between">
                      <span className={`badge bg-${latColor(m.lateralization)}`}>{m.lateralization}</span>
                      <span className={`small fw-bold text-${cogIdxColor(m.mean_memory_index)}`}>
                        {m.mean_memory_index?.toFixed(1)}
                      </span>
                    </div>
                    <div className="progress mt-1" style={{ height: '8px' }}>
                      <div className={`progress-bar bg-${cogIdxColor(m.mean_memory_index)}`}
                           style={{ width: `${Math.min((m.mean_memory_index / 130) * 100, 100).toFixed(0)}%` }} />
                    </div>
                    <div className="text-muted" style={{ fontSize: '0.65rem' }}>n = {m.n}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Mood Comorbidity Summary */}
          {bd.mood_comorbidity && (
            <div className="col-md-12 mb-3">
              <div className="card shadow-sm border-0 border-start border-4 border-danger">
                <div className="card-body py-2 px-3">
                  <div className="row text-center">
                    <div className="col-3">
                      <div className="h4 mb-0 text-danger">{bd.mood_comorbidity.phq9_elevated_count}</div>
                      <div className="small text-muted">PHQ-9 Elevated (&ge;10)</div>
                    </div>
                    <div className="col-3">
                      <div className="h4 mb-0 text-warning">{bd.mood_comorbidity.gad7_elevated_count}</div>
                      <div className="small text-muted">GAD-7 Elevated (&ge;10)</div>
                    </div>
                    <div className="col-3">
                      <div className="h4 mb-0 text-danger">{bd.mood_comorbidity.both_elevated_count}</div>
                      <div className="small text-muted">Both Elevated</div>
                    </div>
                    <div className="col-3">
                      <div className="h4 mb-0 text-primary">{bd.mood_comorbidity.total_assessments}</div>
                      <div className="small text-muted">Total Assessments</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Patients ── */}
      {tab === 'patients' && bd && (
        <div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Age</th>
                  <th>Gender</th>
                  <th className="text-end">Assessments</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {(bd.patients || []).map(p => (
                  <>
                    <tr key={p.patient_id} className={sel === p.patient_id ? 'table-active' : ''}>
                      <td className="small fw-bold">{p.patient_id}</td>
                      <td className="small">{p.age ?? '\u2014'}</td>
                      <td className="small">{p.gender || '\u2014'}</td>
                      <td className="text-end small">{p.assessment_count}</td>
                      <td>
                        <button className="btn btn-outline-primary btn-sm py-0 px-1"
                                style={{ fontSize: '0.7rem' }}
                                onClick={() => setSel(sel === p.patient_id ? null : p.patient_id)}>
                          {sel === p.patient_id ? 'Hide' : 'Detail'}
                        </button>
                      </td>
                    </tr>
                    {sel === p.patient_id && (
                      <tr key={`${p.patient_id}-detail`}>
                        <td colSpan={5} className="bg-light p-0">
                          <div className="p-2">
                            <table className="table table-sm table-bordered mb-0">
                              <thead className="table-secondary">
                                <tr>
                                  <th>Date</th>
                                  <th>Type</th>
                                  <th>MoCA</th>
                                  <th>MMSE</th>
                                  <th>PHQ-9</th>
                                  <th>GAD-7</th>
                                  <th>Mem</th>
                                  <th>Attn</th>
                                  <th>Exec</th>
                                  <th>Lang</th>
                                  <th>PS</th>
                                  <th>Impairment</th>
                                  <th>Lateralization</th>
                                  <th>Assessor</th>
                                </tr>
                              </thead>
                              <tbody>
                                {(p.assessments || []).map(a => (
                                  <tr key={a.id}>
                                    <td className="small">{a.created_at}</td>
                                    <td className="small">{a.battery_type}</td>
                                    <td className={`small text-${a.moca < 26 ? 'warning' : 'success'} fw-bold`}>{a.moca ?? '\u2014'}</td>
                                    <td className={`small text-${a.mmse < 26 ? 'warning' : 'success'} fw-bold`}>{a.mmse ?? '\u2014'}</td>
                                    <td className={`small text-${a.phq9 >= 10 ? 'danger' : 'secondary'}`}>{a.phq9 ?? '\u2014'}</td>
                                    <td className={`small text-${a.gad7 >= 10 ? 'danger' : 'secondary'}`}>{a.gad7 ?? '\u2014'}</td>
                                    <td className="small">{a.memory_index ?? '\u2014'}</td>
                                    <td className="small">{a.attention_index ?? '\u2014'}</td>
                                    <td className="small">{a.executive_index ?? '\u2014'}</td>
                                    <td className="small">{a.language_index ?? '\u2014'}</td>
                                    <td className="small">{a.processing_speed_index ?? '\u2014'}</td>
                                    <td>
                                      {a.impairment_flag && (
                                        <span className={`badge bg-${impairColor(a.impairment_flag)}`}>{a.impairment_flag}</span>
                                      )}
                                    </td>
                                    <td>
                                      {a.lateralization_hypothesis && (
                                        <span className={`badge bg-${latColor(a.lateralization_hypothesis)}`}>{a.lateralization_hypothesis}</span>
                                      )}
                                    </td>
                                    <td className="small">{a.assessor || '\u2014'}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'defs' && defs && (
        <div className="card shadow-sm border-0">
          <div className="card-header bg-dark text-white py-2 small fw-bold">Neuropsychological Concepts in Epilepsy</div>
          <div className="card-body p-2">
            <table className="table table-sm mb-0">
              <tbody>
                {(defs.concepts || []).map(c => (
                  <tr key={c.name}>
                    <td className="small fw-bold align-top" style={{ width: '28%' }}>{c.name}</td>
                    <td className="small">{c.description}</td>
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
