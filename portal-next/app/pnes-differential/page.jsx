'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const classColor = c =>
  c === 'PNES likely'       ? 'danger'  :
  c === 'Epileptic likely'  ? 'primary' :
  c === 'Epileptic confirmed' ? 'dark'  :
  c === 'Mixed / Comorbid'  ? 'warning' : 'secondary';

const certColor = c =>
  c === 'clinically_established' ? 'success' :
  c === 'documented'             ? 'primary' :
  c === 'probable'               ? 'warning' : 'secondary';

const certLabel = c =>
  c === 'clinically_established' ? 'Clinically Established' :
  c === 'documented'             ? 'Documented' :
  c === 'probable'               ? 'Probable'   :
  c === 'possible'               ? 'Possible'   : c;

const veegColor = v =>
  v === 'urgent'  ? 'danger'  :
  v === 'high'    ? 'warning' : 'success';

const weightLabel = w => w >= 3 ? 'Strong' : w >= 2 ? 'Moderate' : 'Mild';
const weightColor = w => w >= 3 ? 'danger' : w >= 2 ? 'warning' : 'secondary';

export default function PNESDifferentialPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetch(`${API}/api/pnes-differential/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/pnes-differential/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/pnes-differential/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpis = ov.kpis || {};
  const patients = (bd?.patients || []).filter(p =>
    !search || p.patient_id.toLowerCase().includes(search.toLowerCase()) ||
    p.classification.toLowerCase().includes(search.toLowerCase())
  );

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'patients',    label: `Per Patient (${(bd?.patients || []).length})` },
    { id: 'semiology',   label: 'Semiology Signs' },
    { id: 'risk',        label: 'Risk Factors' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f9e0; PNES Differential Diagnosis</h3>
      <p className="text-muted small">
        Psychogenic Non-Epileptic Seizures (PNES) vs. epilepsy differential — semiological scoring,
        PNES / epilepsy probability, diagnostic certainty levels, vEEG priority triage, and
        psychiatric risk factors. {ov.total_patients} patients from real <code>pnes_screening</code> data.
      </p>

      {/* KPI Cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Patients',        value: kpis.pnes_likely + kpis.mixed_comorbid + kpis.epileptic_likely + (ov.classification_distribution?.find(c => c.label === 'Epileptic confirmed')?.count || 0), color: 'primary' },
          { label: 'PNES Likely',           value: kpis.pnes_likely,           color: 'danger'   },
          { label: 'Mixed / Comorbid',      value: kpis.mixed_comorbid,        color: 'warning'  },
          { label: 'Epileptic Likely',      value: kpis.epileptic_likely,      color: 'dark'     },
          { label: 'Urgent vEEG',           value: kpis.urgent_veeg_needed,    color: 'danger'   },
          { label: 'Avg PNES Probability',  value: `${(kpis.avg_pnes_probability * 100).toFixed(0)}%`, color: kpis.avg_pnes_probability > 0.5 ? 'danger' : 'success' },
          { label: 'Psychiatric Comorbid',  value: kpis.psychiatric_comorbidity, color: 'warning' },
          { label: 'Documented Certainty',  value: kpis.documented_certainty,  color: 'success'  },
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

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Classification distribution */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Classification Distribution</div>
              <div className="card-body p-3">
                {(ov.classification_distribution || []).map(d => {
                  const pct = ((d.count / ov.total_patients) * 100).toFixed(1);
                  return (
                    <div key={d.label} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span><span className={`badge bg-${classColor(d.label)} me-2`}>{d.label}</span></span>
                        <span className="fw-bold">{d.count} <span className="text-muted">({pct}%)</span></span>
                      </div>
                      <div className="progress" style={{height: '12px'}}>
                        <div className={`progress-bar bg-${classColor(d.label)}`} style={{width: `${pct}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Certainty levels */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Diagnostic Certainty</div>
              <div className="card-body p-3">
                {(ov.certainty_distribution || []).map(c => {
                  const pct = ((c.count / ov.total_patients) * 100).toFixed(1);
                  return (
                    <div key={c.label} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span><span className={`badge bg-${certColor(c.label)} me-2`}>{certLabel(c.label)}</span></span>
                        <span className="fw-bold">{c.count} <span className="text-muted">({pct}%)</span></span>
                      </div>
                      <div className="progress" style={{height: '12px'}}>
                        <div className={`progress-bar bg-${certColor(c.label)}`} style={{width: `${pct}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* vEEG Priority */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Video-EEG Priority Triage</div>
              <div className="card-body p-3">
                {(ov.veeg_priority_distribution || []).map(v => {
                  const pct = ((v.count / ov.total_patients) * 100).toFixed(1);
                  return (
                    <div key={v.label} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize fw-semibold">{v.label}</span>
                        <span className="fw-bold">{v.count} <span className="text-muted">({pct}%)</span></span>
                      </div>
                      <div className="progress" style={{height: '12px'}}>
                        <div className={`progress-bar bg-${veegColor(v.label)}`} style={{width: `${pct}%`}} />
                      </div>
                    </div>
                  );
                })}
                {kpis.urgent_veeg_needed > 0 && (
                  <div className="alert alert-danger py-2 px-3 mt-2 mb-0 small">
                    <strong>&#x26a0;&#xfe0f; {kpis.urgent_veeg_needed} patients</strong> need urgent vEEG — schedule within 48h
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Risk factor frequency */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Risk Factor Frequency</div>
              <div className="card-body p-3">
                {(ov.risk_factor_frequency || []).map(r => (
                  <div key={r.factor} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{r.factor}</span>
                      <span className="fw-bold text-danger">{r.count} <span className="text-muted">({r.pct}%)</span></span>
                    </div>
                    <div className="progress" style={{height: '8px'}}>
                      <div className="progress-bar bg-danger bg-opacity-75" style={{width: `${r.pct}%`}} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* PNES probability summary */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">PNES Probability — Cohort Summary</div>
              <div className="card-body p-3">
                <div className="row text-center mb-3">
                  <div className="col-4 border-end">
                    <div className={`h3 mb-0 text-${kpis.avg_pnes_probability > 0.5 ? 'danger' : 'primary'}`}>
                      {(kpis.avg_pnes_probability * 100).toFixed(0)}%
                    </div>
                    <div className="text-muted small">Avg PNES Prob</div>
                  </div>
                  <div className="col-4 border-end">
                    <div className="h3 mb-0 text-success">{kpis.documented_certainty}</div>
                    <div className="text-muted small">Documented Certainty</div>
                  </div>
                  <div className="col-4">
                    <div className="h3 mb-0 text-warning">{kpis.psychiatric_comorbidity}</div>
                    <div className="text-muted small">Psychiatric Comorbid</div>
                  </div>
                </div>
                <div className="alert alert-info py-2 px-3 small mb-0">
                  <strong>Clinical Insight:</strong> {kpis.mixed_comorbid} patients ({((kpis.mixed_comorbid / ov.total_patients) * 100).toFixed(0)}%)
                  have Mixed / Comorbid presentation — both PNES and epileptic seizures co-exist,
                  requiring simultaneous psychiatric <em>and</em> neurological management.
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Per Patient ── */}
      {tab === 'patients' && bd && (
        <div>
          <div className="mb-3">
            <input
              className="form-control form-control-sm w-auto"
              placeholder="Search patient ID or classification…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
          <div style={{overflowX:'auto'}}>
            <table className="table table-sm table-hover table-bordered" style={{fontSize:'0.78rem'}}>
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Age</th>
                  <th>Gender</th>
                  <th>Classification</th>
                  <th>PNES Prob</th>
                  <th>Epilepsy Prob</th>
                  <th>Certainty</th>
                  <th>vEEG</th>
                  <th>PHQ-9</th>
                  <th>GAD-7</th>
                  <th>Risk Factors</th>
                </tr>
              </thead>
              <tbody>
                {patients.map(p => (
                  <tr key={p.patient_id}>
                    <td><strong>{p.patient_id}</strong></td>
                    <td>{p.age}</td>
                    <td>{p.gender}</td>
                    <td><span className={`badge bg-${classColor(p.classification)}`}>{p.classification}</span></td>
                    <td>
                      <div className="d-flex align-items-center gap-1">
                        <div className="progress flex-grow-1" style={{height:'8px'}}>
                          <div
                            className={`progress-bar bg-${p.pnes_probability > 0.65 ? 'danger' : p.pnes_probability > 0.35 ? 'warning' : 'success'}`}
                            style={{width:`${(p.pnes_probability * 100).toFixed(0)}%`}}
                          />
                        </div>
                        <span style={{minWidth:'32px'}}>{(p.pnes_probability * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td>
                      <div className="d-flex align-items-center gap-1">
                        <div className="progress flex-grow-1" style={{height:'8px'}}>
                          <div className="progress-bar bg-primary" style={{width:`${(p.epilepsy_probability * 100).toFixed(0)}%`}} />
                        </div>
                        <span style={{minWidth:'32px'}}>{(p.epilepsy_probability * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td><span className={`badge bg-${certColor(p.diagnostic_certainty)} bg-opacity-75`}>{certLabel(p.diagnostic_certainty)}</span></td>
                    <td><span className={`badge bg-${veegColor(p.veeg_priority)}`}>{p.veeg_priority}</span></td>
                    <td className={p.phq9_score >= 10 ? 'text-danger fw-bold' : ''}>{p.phq9_score?.toFixed(0) ?? '—'}</td>
                    <td className={p.gad7_score >= 10 ? 'text-warning fw-bold' : ''}>{p.gad7_score?.toFixed(0) ?? '—'}</td>
                    <td>{p.risk_factor_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Semiology Signs ── */}
      {tab === 'semiology' && (
        <div className="row g-3">
          {/* PNES signs */}
          <div className="col-md-6">
            <div className="card shadow-sm border-danger">
              <div className="card-header fw-semibold text-danger">PNES Semiological Signs</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0" style={{fontSize:'0.8rem'}}>
                  <thead><tr><th>Sign</th><th>Weight</th><th>Specificity</th></tr></thead>
                  <tbody>
                    {(ov.pnes_signs_reference || []).map(s => (
                      <tr key={s.sign}>
                        <td>{s.sign}</td>
                        <td><span className={`badge bg-${weightColor(s.weight)}`}>{weightLabel(s.weight)}</span></td>
                        <td>
                          <div className="d-flex align-items-center gap-1">
                            <div className="progress flex-grow-1" style={{height:'6px'}}>
                              <div className="progress-bar bg-danger" style={{width:`${(s.specificity * 100).toFixed(0)}%`}} />
                            </div>
                            <span>{(s.specificity * 100).toFixed(0)}%</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Epilepsy signs */}
          <div className="col-md-6">
            <div className="card shadow-sm border-primary">
              <div className="card-header fw-semibold text-primary">Epilepsy Semiological Signs</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0" style={{fontSize:'0.8rem'}}>
                  <thead><tr><th>Sign</th><th>Weight</th><th>Specificity</th></tr></thead>
                  <tbody>
                    {(ov.epilepsy_signs_reference || []).map(s => (
                      <tr key={s.sign}>
                        <td>{s.sign}</td>
                        <td><span className={`badge bg-${weightColor(s.weight)}`}>{weightLabel(s.weight)}</span></td>
                        <td>
                          <div className="d-flex align-items-center gap-1">
                            <div className="progress flex-grow-1" style={{height:'6px'}}>
                              <div className="progress-bar bg-primary" style={{width:`${(s.specificity * 100).toFixed(0)}%`}} />
                            </div>
                            <span>{(s.specificity * 100).toFixed(0)}%</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Semiology reference from breakdown */}
          {bd?.pnes_features_reference && (
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">PNES Feature Reference (weighted scoring)</div>
                <div className="card-body p-2">
                  <div className="row">
                    {(bd.pnes_features_reference || []).map(f => (
                      <div key={f.feature} className="col-md-4 col-lg-3 mb-2">
                        <div className={`border rounded p-2 small border-${weightColor(f.weight)}`}>
                          <div className="fw-semibold">{f.feature}</div>
                          <div className="text-muted" style={{fontSize:'0.7rem'}}>
                            Weight: {f.weight} · {(f.specificity * 100).toFixed(0)}% specific
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Risk Factors ── */}
      {tab === 'risk' && (
        <div className="row g-3">
          <div className="col-md-7">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Risk Factor Prevalence in Cohort</div>
              <div className="card-body p-3">
                {(ov.risk_factor_frequency || []).map(r => (
                  <div key={r.factor} className="mb-3">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className="fw-semibold">{r.factor}</span>
                      <span className="text-danger fw-bold">{r.count} pts ({r.pct}%)</span>
                    </div>
                    <div className="progress" style={{height:'16px'}}>
                      <div className="progress-bar bg-danger" style={{width:`${r.pct}%`}}>
                        {r.pct >= 15 ? `${r.pct}%` : ''}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-5">
            {/* Definitions risk factors */}
            {defs?.risk_factors && (
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Risk Factor Definitions</div>
                <div className="card-body p-2" style={{maxHeight:'420px', overflowY:'auto'}}>
                  {(defs.risk_factors || []).map(rf => (
                    <div key={rf.factor} className="mb-3 pb-2 border-bottom">
                      <div className="fw-semibold small">{rf.factor}</div>
                      <div className="text-muted" style={{fontSize:'0.76rem'}}>{rf.description}</div>
                      {rf.prevalence && (
                        <span className="badge bg-danger bg-opacity-75 mt-1" style={{fontSize:'0.68rem'}}>
                          Prevalence: {rf.prevalence}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* PHQ-9 / GAD-7 distribution from patient data */}
          {bd?.patients && (
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">PHQ-9 &amp; GAD-7 Score Distribution</div>
                <div className="card-body p-3">
                  <div className="row">
                    {[
                      { label: 'PHQ-9 (Depression)', key: 'phq9_score', thresholds: [{t:10,label:'Moderate+',color:'danger'},{t:5,label:'Mild',color:'warning'},{t:0,label:'Minimal',color:'success'}] },
                      { label: 'GAD-7 (Anxiety)',    key: 'gad7_score', thresholds: [{t:10,label:'Moderate+',color:'danger'},{t:5,label:'Mild',color:'warning'},{t:0,label:'Minimal',color:'success'}] },
                    ].map(({ label, key, thresholds }) => {
                      const vals = bd.patients.filter(p => p[key] != null).map(p => p[key]);
                      if (!vals.length) return null;
                      const counts = thresholds.map(({ t, label: l, color }) => ({
                        label: l, color,
                        count: t === 0 ? vals.filter(v => v < 5).length : t === 5 ? vals.filter(v => v >= 5 && v < 10).length : vals.filter(v => v >= 10).length
                      }));
                      return (
                        <div key={key} className="col-md-6">
                          <div className="fw-semibold small mb-2">{label}</div>
                          {counts.map(c => (
                            <div key={c.label} className="mb-2">
                              <div className="d-flex justify-content-between small mb-1">
                                <span>{c.label}</span>
                                <span className="fw-bold">{c.count} pts ({((c.count / vals.length) * 100).toFixed(0)}%)</span>
                              </div>
                              <div className="progress" style={{height:'10px'}}>
                                <div className={`progress-bar bg-${c.color}`} style={{width:`${((c.count / vals.length) * 100).toFixed(0)}%`}} />
                              </div>
                            </div>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          {/* Core concepts */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Clinical Concepts</div>
              <div className="card-body p-2" style={{maxHeight:'480px', overflowY:'auto'}}>
                {(defs.concepts || []).map(c => (
                  <div key={c.name} className="mb-3 pb-2 border-bottom">
                    <div className="fw-semibold small">{c.name}</div>
                    <div className="text-muted" style={{fontSize:'0.77rem'}}>{c.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Diagnostic levels */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Diagnostic Certainty Levels</div>
              <div className="card-body p-2">
                {(defs.diagnostic_levels || []).map(dl => (
                  <div key={dl.level} className={`alert alert-${certColor(dl.level)} py-2 px-3 mb-2`} style={{fontSize:'0.82rem'}}>
                    <div className="fw-bold">{certLabel(dl.level)}</div>
                    <div>{dl.criteria}</div>
                    {dl.action && <div className="text-muted mt-1"><strong>Action:</strong> {dl.action}</div>}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Semiology table */}
          {defs.semiology_table && (
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Semiology Differential Table</div>
                <div className="card-body p-2">
                  <table className="table table-sm table-bordered" style={{fontSize:'0.79rem'}}>
                    <thead className="table-light">
                      <tr>
                        <th>Feature</th>
                        <th className="text-danger">PNES</th>
                        <th className="text-primary">Epilepsy</th>
                        <th>Weight</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(defs.semiology_table || []).map(row => (
                        <tr key={row.feature}>
                          <td className="fw-semibold">{row.feature}</td>
                          <td className="text-danger">{row.pnes}</td>
                          <td className="text-primary">{row.epilepsy}</td>
                          <td><span className={`badge bg-${weightColor(row.weight ?? 1)}`}>{weightLabel(row.weight ?? 1)}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Management */}
          {defs.management && (
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Management Guidelines</div>
                <div className="card-body p-2">
                  {(defs.management || []).map(m => (
                    <div key={m.phase} className="mb-2 pb-2 border-bottom">
                      <div className="fw-semibold small text-primary">{m.phase}</div>
                      <div className="text-muted" style={{fontSize:'0.77rem'}}>{m.action}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Quality metrics */}
          {defs.quality_metrics && (
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Quality Metrics</div>
                <div className="card-body p-2">
                  {(defs.quality_metrics || []).map(q => (
                    <div key={q.metric} className="mb-2 pb-2 border-bottom">
                      <div className="fw-semibold small">{q.metric}</div>
                      <div className="text-muted" style={{fontSize:'0.77rem'}}>{q.description}</div>
                      {q.target && <div className="badge bg-success bg-opacity-75 mt-1" style={{fontSize:'0.68rem'}}>Target: {q.target}</div>}
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
