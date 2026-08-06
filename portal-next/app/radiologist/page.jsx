'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const qualColor = q =>
  q === 'Diagnostic' ? 'success' :
  q === 'Adequate'   ? 'primary' :
  q === 'Suboptimal' ? 'warning' : 'danger';

const confColor = c =>
  c === 'High'     ? 'success' :
  c === 'Moderate' ? 'warning' :
  c === 'Low'      ? 'danger'  : 'secondary';

const laterColor = l =>
  l === 'Left'     ? 'primary' :
  l === 'Right'    ? 'info'    :
  l === 'Bilateral'? 'warning' : 'secondary';

export default function RadiologistDashboardPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [sel,  setSel]  = useState(null);

  useEffect(() => {
    fetch(`${API}/api/radiologist/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/radiologist/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/radiologist/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpis = [
    { label: 'Total MRI Scans',     value: ov.total_mri_scans,                        color: 'primary' },
    { label: 'Lesion Positive',     value: ov.lesion_positive_count,                   color: 'danger'  },
    { label: 'Non-Lesional',        value: ov.non_lesional_count,                      color: 'success' },
    { label: 'Lesion Rate',         value: `${(ov.lesion_positive_rate ?? 0).toFixed(1)}%`, color: 'warning' },
    { label: 'Total Patients',      value: ov.total_patients,                          color: 'info'    },
    { label: 'MRI Coverage',        value: `${(ov.mri_coverage ?? 0).toFixed(1)}%`,   color: 'success' },
    { label: 'Hippocampal Scl.',    value: ov.hippocampal_sclerosis_count,             color: 'danger'  },
    { label: 'Avg Confidence',      value: ov.avg_radiologist_confidence ?? '—',       color: 'primary' },
  ];

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'findings',   label: `MRI Findings${bd ? ` (${(bd.patients || []).length})` : ''}` },
    { id: 'definitions', label: 'MRI Concepts' },
  ];

  const patients = (bd || {}).patients || [];
  const selPat   = patients.find(p => p.patient_id === sel);

  return (
    <div>
      <h3>&#x1f9b4; Radiologist Dashboard</h3>
      <p className="text-muted small">
        Epilepsy MRI interpretation workstation — lesion detection, hippocampal sclerosis,
        focal cortical dysplasia, scan quality grading, laterality, and EEG–MRI concordance.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {kpis.map(k => (
          <div key={k.label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${k.color}`}>{k.value ?? '—'}</div>
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

          {/* Lesion Type Distribution */}
          <div className="col-md-5 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-danger text-white py-2 small fw-bold">Lesion Type Distribution</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Type</th><th>Label</th><th className="text-end">Count</th><th style={{ width: '35%' }}>Bar</th></tr></thead>
                  <tbody>
                    {(ov.lesion_type_distribution || []).map(l => {
                      const max = Math.max(...(ov.lesion_type_distribution || []).map(x => x.count), 1);
                      return (
                        <tr key={l.lesion_type}>
                          <td className="small fw-bold">{l.lesion_type}</td>
                          <td className="small">{l.label}</td>
                          <td className="text-end small">{l.count}</td>
                          <td>
                            <div className="progress" style={{ height: '14px' }}>
                              <div className="progress-bar bg-danger"
                                   style={{ width: `${(l.count / max * 100).toFixed(0)}%` }} />
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

          {/* Scan Quality Distribution */}
          <div className="col-md-3 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-primary text-white py-2 small fw-bold">Scan Quality</div>
              <div className="card-body p-2">
                {(ov.quality_distribution || []).map(q => {
                  const max = Math.max(...(ov.quality_distribution || []).map(x => x.count), 1);
                  return (
                    <div key={q.quality} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '100px' }}>
                        <span className={`badge bg-${qualColor(q.quality)}`}>{q.quality}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${qualColor(q.quality)}`}
                             style={{ width: `${(q.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{q.count}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Laterality Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-info text-white py-2 small fw-bold">Laterality</div>
              <div className="card-body p-2">
                {(ov.laterality_distribution || []).map(l => {
                  const max = Math.max(...(ov.laterality_distribution || []).map(x => x.count), 1);
                  return (
                    <div key={l.laterality} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '80px' }}>
                        <span className={`badge bg-${laterColor(l.laterality)}`}>{l.laterality}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${laterColor(l.laterality)}`}
                             style={{ width: `${(l.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{l.count}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Lesion Location Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">Lesion Location</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Location</th><th className="text-end">Count</th><th style={{ width: '40%' }}>Bar</th></tr></thead>
                  <tbody>
                    {(ov.location_distribution || []).map(l => {
                      const max = Math.max(...(ov.location_distribution || []).map(x => x.count), 1);
                      return (
                        <tr key={l.location}>
                          <td className="small">{l.location}</td>
                          <td className="text-end small">{l.count}</td>
                          <td>
                            <div className="progress" style={{ height: '14px' }}>
                              <div className="progress-bar bg-warning"
                                   style={{ width: `${(l.count / max * 100).toFixed(0)}%` }} />
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

          {/* Classification Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-dark text-white py-2 small fw-bold">MRI Classification</div>
              <div className="card-body p-2">
                {(ov.classification_distribution || []).map(c => {
                  const max = Math.max(...(ov.classification_distribution || []).map(x => x.count), 1);
                  const color = c.classification === 'LESIONAL' ? 'danger' :
                                c.classification === 'NON_LESIONAL' ? 'success' : 'secondary';
                  return (
                    <div key={c.classification} className="d-flex align-items-center mb-2">
                      <span className="small me-2" style={{ minWidth: '130px' }}>
                        <span className={`badge bg-${color}`}>{c.label || c.classification}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '20px' }}>
                        <div className={`progress-bar bg-${color}`}
                             style={{ width: `${(c.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.7rem' }}>{c.count}</span>
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

      {/* ── MRI Findings (per-patient) ── */}
      {tab === 'findings' && (
        <div>
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-header bg-dark text-white py-2 small fw-bold">
              Per-Patient MRI Findings ({patients.length} patients)
            </div>
            <div className="card-body p-2">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Patient</th>
                      <th>Age</th>
                      <th>Gender</th>
                      <th>Quality</th>
                      <th>Lesion</th>
                      <th>Location</th>
                      <th>Laterality</th>
                      <th>HS</th>
                      <th>Vol.Asym</th>
                      <th>Confidence</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {patients.map(p => {
                      const m = p.mri || {};
                      return (
                        <tr key={p.patient_id} className={sel === p.patient_id ? 'table-active' : ''}>
                          <td className="small fw-bold">{p.patient_id}</td>
                          <td className="small">{p.age ?? '—'}</td>
                          <td className="small">{p.gender || '—'}</td>
                          <td>
                            {m.quality && (
                              <span className={`badge bg-${qualColor(m.quality)}`}
                                    style={{ fontSize: '0.65rem' }}>{m.quality}</span>
                            )}
                          </td>
                          <td className="small">{m.lesion_label || m.lesion_type || '—'}</td>
                          <td className="small">{m.lesion_location || '—'}</td>
                          <td>
                            {m.laterality && (
                              <span className={`badge bg-${laterColor(m.laterality)}`}
                                    style={{ fontSize: '0.65rem' }}>{m.laterality}</span>
                            )}
                          </td>
                          <td className="small">{m.hippocampal_sclerosis === 'Yes' || m.hippocampal_sclerosis === true ? '✓' : '—'}</td>
                          <td className="small">
                            {m.hippocampal_volume_asymmetry != null
                              ? `${(m.hippocampal_volume_asymmetry * 100).toFixed(0)}%`
                              : '—'}
                          </td>
                          <td>
                            {m.radiologist_confidence && (
                              <span className={`badge bg-${confColor(m.radiologist_confidence)}`}
                                    style={{ fontSize: '0.65rem' }}>{m.radiologist_confidence}</span>
                            )}
                          </td>
                          <td>
                            <button className="btn btn-outline-primary btn-sm py-0 px-1"
                                    style={{ fontSize: '0.7rem' }}
                                    onClick={() => setSel(sel === p.patient_id ? null : p.patient_id)}>
                              {sel === p.patient_id ? 'Hide' : 'Detail'}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Patient detail panel */}
          {sel && selPat && (() => {
            const m = selPat.mri || {};
            const e = selPat.eeg_analysis || {};
            return (
              <div className="card border-primary shadow-sm">
                <div className="card-header bg-primary text-white py-1 small fw-bold">
                  {selPat.patient_id} — {selPat.name || '—'} · MRI Detail
                </div>
                <div className="card-body p-3">
                  <div className="row">
                    <div className="col-md-6">
                      <table className="table table-sm mb-0">
                        <tbody>
                          <tr><td className="fw-bold small">Protocol</td><td className="small">{m.protocol || '—'}</td></tr>
                          <tr><td className="fw-bold small">Classification</td><td className="small">{m.classification_label || m.classification || '—'}</td></tr>
                          <tr><td className="fw-bold small">Lesion</td><td className="small">{m.lesion_label || '—'}</td></tr>
                          <tr><td className="fw-bold small">Description</td><td className="small">{m.lesion_description || '—'}</td></tr>
                          <tr><td className="fw-bold small">T2/FLAIR Signal</td><td className="small">{m.t2_flair_signal || '—'}</td></tr>
                          <tr><td className="fw-bold small">Enhancing</td><td className="small">{m.enhancing ? 'Yes' : 'No'}</td></tr>
                          <tr><td className="fw-bold small">HS Vol. Asymmetry</td>
                            <td className="small">
                              {m.hippocampal_volume_asymmetry != null
                                ? `${(m.hippocampal_volume_asymmetry * 100).toFixed(0)}%` : '—'}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                    <div className="col-md-6">
                      <table className="table table-sm mb-0">
                        <tbody>
                          <tr><td className="fw-bold small">EEG Prediction</td><td className="small">{e.predicted_label || '—'}</td></tr>
                          <tr><td className="fw-bold small">EEG Confidence</td>
                            <td className="small">
                              {e.confidence != null ? `${(e.confidence * 100).toFixed(1)}%` : '—'}
                            </td>
                          </tr>
                          <tr><td className="fw-bold small">EEG Signal Quality</td><td className="small">{e.signal_quality || '—'}</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* ── MRI Concepts / Definitions ── */}
      {tab === 'definitions' && defs && (
        <div>
          {defs.concepts ? (
            <div className="card shadow-sm border-0">
              <div className="card-header bg-dark text-white py-2 small fw-bold">Epilepsy MRI — Clinical Concepts</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {defs.concepts.map(c => (
                      <tr key={c.name}>
                        <td className="small fw-bold" style={{ width: '25%', verticalAlign: 'top' }}>{c.name}</td>
                        <td className="small">{c.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : defs.sections ? (
            defs.sections.map(s => (
              <div key={s.heading} className="card shadow-sm border-0 mb-3">
                <div className="card-header bg-dark text-white py-2 small fw-bold">{s.heading}</div>
                <div className="card-body p-2">
                  <table className="table table-sm mb-0">
                    <tbody>
                      {(s.definitions || []).map(d => (
                        <tr key={d.term}>
                          <td className="small fw-bold" style={{ width: '25%' }}>{d.term}</td>
                          <td className="small">{d.definition}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))
          ) : null}
        </div>
      )}
    </div>
  );
}
