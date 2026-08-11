'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SEV_COLOR = s =>
  s === 'Normal' ? 'success' : s === 'Mild' ? 'info' : s === 'Moderate' ? 'warning' : 'danger';

const PAT_LABEL = {
  normal:      'Normal',
  neuropathic: 'Neuropathic',
  myopathic:   'Myopathic',
  mixed:       'Mixed',
  nmj:         'NMJ Disorder',
};

const PAT_COLOR = p => ({
  normal:      'success',
  neuropathic: 'warning',
  myopathic:   'info',
  mixed:       'secondary',
  nmj:         'danger',
}[p] || 'secondary');

const LIMB_ICON = l => (l || '').toLowerCase().includes('upper') ? '🖐️' : '🦶';

function Bar({ label, val, max, colorClass = 'primary' }) {
  const pct = max > 0 ? Math.round((val / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{val}</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className={`progress-bar bg-${colorClass}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function EMGPage() {
  const [ov, setOv]         = useState(null);
  const [bd, setBd]         = useState(null);
  const [defs, setDefs]     = useState(null);
  const [tab, setTab]       = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/emg/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/emg/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/emg/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpi = ov.kpis || {};
  const tabs = [
    { id: 'overview',  label: '📊 Overview' },
    { id: 'breakdown', label: '🔬 Muscle Analysis' },
    { id: 'patients',  label: '🧑‍⚕️ Per Patient' },
    { id: 'defs',      label: '📖 Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1 fw-bold">💪 Electromyography (EMG) Dashboard</h4>
      <p className="text-muted small mb-3">
        Needle EMG — MUAP duration, amplitude, recruitment &amp; spontaneous activity
        across 8 muscles (FDI · APB · Biceps · Deltoid · TA · Gastroc · VastMed · Iliopsoas), 30 studies
      </p>

      {/* Nav tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          {/* KPI cards */}
          <div className="row g-3 mb-4">
            {[
              { label: 'Total Studies',       val: kpi.total_studies,           color: 'primary' },
              { label: 'Abnormal',            val: kpi.abnormal_count,          color: 'danger' },
              { label: 'Abnormal Rate',       val: `${kpi.abnormal_rate_pct}%`, color: 'warning' },
              { label: 'Mean MUAP Duration',  val: `${kpi.mean_muap_duration_ms} ms`, color: 'info' },
              { label: 'Mean MUAP Amplitude', val: `${kpi.mean_muap_amplitude_uv} µV`, color: 'secondary' },
              { label: 'Muscles / Study',     val: kpi.muscles_tested_per_study, color: 'success' },
            ].map(c => (
              <div key={c.label} className="col-6 col-md-4 col-lg-2">
                <div className={`card border-${c.color} h-100`}>
                  <div className="card-body p-2 text-center">
                    <div className={`display-6 fw-bold text-${c.color}`}>{c.val}</div>
                    <div className="small text-muted">{c.label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="row g-3 mb-4">
            {/* Severity distribution */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header py-2 small fw-bold">Severity Distribution</div>
                <div className="card-body">
                  {(ov.severity_distribution || []).map(s => (
                    <div key={s.severity} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className={`badge bg-${SEV_COLOR(s.severity)}`}>{s.severity}</span>
                        <span className="fw-bold">{s.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${SEV_COLOR(s.severity)}`}
                          style={{ width: `${kpi.total_studies > 0 ? (s.count / kpi.total_studies) * 100 : 0}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Diagnostic patterns */}
            <div className="col-md-5">
              <div className="card h-100">
                <div className="card-header py-2 small fw-bold">Diagnostic Patterns</div>
                <div className="card-body">
                  {(ov.diagnostic_pattern_distribution || []).filter(p => p.count > 0).map(p => (
                    <div key={p.pattern} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className={`badge bg-${PAT_COLOR(p.pattern)}`}>{PAT_LABEL[p.pattern] || p.label}</span>
                        <span className="fw-bold">{p.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${PAT_COLOR(p.pattern)}`}
                          style={{ width: `${kpi.total_studies > 0 ? (p.count / kpi.total_studies) * 100 : 0}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Muscle abnormality rates */}
            <div className="col-md-3">
              <div className="card h-100">
                <div className="card-header py-2 small fw-bold">Muscle Abnormality Rates</div>
                <div className="card-body">
                  {(ov.muscle_abnormality_rates || []).map(m => (
                    <div key={m.muscle} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-truncate" style={{ maxWidth: '65%' }} title={m.muscle}>
                          {LIMB_ICON(m.limb || '')} {m.muscle.split('(')[0].trim()}
                        </span>
                        <span className={`fw-bold text-${m.rate_pct > 25 ? 'danger' : m.rate_pct > 10 ? 'warning' : 'success'}`}>
                          {m.rate_pct}%
                        </span>
                      </div>
                      <div className="progress" style={{ height: 6 }}>
                        <div
                          className={`progress-bar bg-${m.rate_pct > 25 ? 'danger' : m.rate_pct > 10 ? 'warning' : 'success'}`}
                          style={{ width: `${m.rate_pct}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* MUAP summary table */}
          <div className="card">
            <div className="card-header py-2 small fw-bold">Mean MUAP Parameters by Muscle</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Muscle</th>
                      <th>Innervation</th>
                      <th>Mean Duration (ms)</th>
                      <th>Ref Range (ms)</th>
                      <th>Mean Amplitude (µV)</th>
                      <th>Ref Range (µV)</th>
                      <th>Polyphasic %</th>
                      <th>Abnormal %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd ? bd.muap_summary : ov.muscle_abnormality_rates || []).map(m => (
                      <tr key={m.muscle}>
                        <td className="small fw-bold">{m.muscle}</td>
                        <td className="text-muted small">{m.innervation || '—'}</td>
                        <td className={
                          m.duration_ref_upper && m.mean_duration_ms > m.duration_ref_upper ? 'text-danger fw-bold' :
                          m.duration_ref_lower && m.mean_duration_ms < m.duration_ref_lower ? 'text-danger fw-bold' :
                          'text-success'
                        }>{m.mean_duration_ms}</td>
                        <td className="text-muted small">
                          {m.duration_ref_lower}–{m.duration_ref_upper}
                        </td>
                        <td className={
                          m.amplitude_ref_upper && m.mean_amplitude_uv > m.amplitude_ref_upper ? 'text-danger fw-bold' :
                          m.amplitude_ref_lower && m.mean_amplitude_uv < m.amplitude_ref_lower ? 'text-danger fw-bold' :
                          'text-success'
                        }>{m.mean_amplitude_uv}</td>
                        <td className="text-muted small">
                          {m.amplitude_ref_lower}–{m.amplitude_ref_upper}
                        </td>
                        <td>{m.polyphasic_pct != null ? `${m.polyphasic_pct}%` : '—'}</td>
                        <td>
                          <span className={`badge bg-${m.abnormal_pct > 25 ? 'danger' : m.abnormal_pct > 10 ? 'warning' : 'success'}`}>
                            {m.abnormal_pct != null ? `${m.abnormal_pct}%` : `${m.rate_pct}%`}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── MUSCLE ANALYSIS ── */}
      {tab === 'breakdown' && bd && (
        <>
          <div className="row g-3 mb-4">
            {/* Duration histogram */}
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header py-2 small fw-bold">
                  MUAP Duration Distribution (all muscles)
                  <span className="text-muted ms-1">(normal: 5–15 ms)</span>
                </div>
                <div className="card-body">
                  {(bd.duration_histogram || []).map(b => (
                    <Bar key={b.bin} label={`${b.bin} ms`} val={b.count}
                      max={Math.max(...(bd.duration_histogram || []).map(x => x.count), 1)}
                      colorClass={parseFloat(b.bin) > 15 || parseFloat(b.bin) < 5 ? 'danger' : 'success'} />
                  ))}
                </div>
              </div>
            </div>

            {/* Amplitude histogram */}
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header py-2 small fw-bold">
                  MUAP Amplitude Distribution (all muscles)
                  <span className="text-muted ms-1">(normal: 200–2000 µV)</span>
                </div>
                <div className="card-body">
                  {(bd.amplitude_histogram || []).map(b => (
                    <Bar key={b.bin} label={`${b.bin} µV`} val={b.count}
                      max={Math.max(...(bd.amplitude_histogram || []).map(x => x.count), 1)}
                      colorClass={parseFloat(b.bin) > 2000 ? 'warning' : parseFloat(b.bin) < 200 ? 'danger' : 'success'} />
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="row g-3 mb-4">
            {/* Recruitment distribution */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header py-2 small fw-bold">Recruitment Pattern Distribution</div>
                <div className="card-body">
                  {(bd.recruitment_distribution || []).map(r => (
                    <div key={r.pattern} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{r.pattern}</span>
                        <span className="fw-bold">{r.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${r.pattern === 'Full' ? 'success' : r.pattern === 'Reduced' ? 'warning' : 'danger'}`}
                          style={{ width: `${Math.max(...(bd.recruitment_distribution || []).map(x => x.count), 1) > 0 ? (r.count / Math.max(...(bd.recruitment_distribution || []).map(x => x.count), 1)) * 100 : 0}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Spontaneous activity */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header py-2 small fw-bold">Spontaneous Activity Distribution</div>
                <div className="card-body">
                  {(bd.spontaneous_activity_distribution || []).map(s => (
                    <div key={s.type} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{s.type}</span>
                        <span className="fw-bold">{s.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${s.type === 'None' ? 'success' : 'danger'}`}
                          style={{ width: `${Math.max(...(bd.spontaneous_activity_distribution || []).map(x => x.count), 1) > 0 ? (s.count / Math.max(...(bd.spontaneous_activity_distribution || []).map(x => x.count), 1)) * 100 : 0}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Limb comparison */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header py-2 small fw-bold">Upper vs Lower Limb Comparison</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Limb</th><th>Mean Dur (ms)</th><th>Mean Amp (µV)</th><th>Abn %</th></tr>
                    </thead>
                    <tbody>
                      {(bd.limb_comparison || []).map(l => (
                        <tr key={l.limb}>
                          <td>{LIMB_ICON(l.limb)} {l.limb}</td>
                          <td>{l.mean_duration_ms}</td>
                          <td>{l.mean_amplitude_uv}</td>
                          <td>
                            <span className={`badge bg-${l.abnormal_pct > 25 ? 'danger' : l.abnormal_pct > 10 ? 'warning' : 'success'}`}>
                              {l.abnormal_pct}%
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Per-muscle MUAP summary table */}
          <div className="card">
            <div className="card-header py-2 small fw-bold">Per-Muscle MUAP Detail</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Muscle</th>
                      <th>Limb</th>
                      <th>Innervation</th>
                      <th>Mean Dur (ms)</th>
                      <th>Mean Amp (µV)</th>
                      <th>Mean Phases</th>
                      <th>Polyphasic %</th>
                      <th>Abnormal %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.muap_summary || []).map(m => (
                      <tr key={m.muscle}>
                        <td className="small fw-bold">{m.muscle}</td>
                        <td>{LIMB_ICON(m.limb)} {m.limb}</td>
                        <td className="text-muted small">{m.innervation}</td>
                        <td className={m.mean_duration_ms > m.duration_ref_upper ? 'text-danger fw-bold' :
                                       m.mean_duration_ms < m.duration_ref_lower ? 'text-warning fw-bold' : 'text-success'}>
                          {m.mean_duration_ms}
                        </td>
                        <td className={m.mean_amplitude_uv > m.amplitude_ref_upper ? 'text-warning fw-bold' :
                                       m.mean_amplitude_uv < m.amplitude_ref_lower ? 'text-danger fw-bold' : 'text-success'}>
                          {m.mean_amplitude_uv}
                        </td>
                        <td>{m.mean_phases}</td>
                        <td>{m.polyphasic_pct}%</td>
                        <td>
                          <span className={`badge bg-${m.abnormal_pct > 25 ? 'danger' : m.abnormal_pct > 10 ? 'warning' : 'success'}`}>
                            {m.abnormal_pct}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && bd && (
        <div className="card">
          <div className="card-header py-2 small fw-bold">Per-Patient EMG Results ({(bd.patient_details || ov.patient_summary || []).length} studies)</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>Patient</th>
                    <th>Age</th>
                    <th>Disease</th>
                    <th>Severity</th>
                    <th>Pattern</th>
                    <th>Abnormal Muscles</th>
                    <th>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.patient_details || ov.patient_summary || []).map(pt => (
                    <>
                      <tr key={pt.patient_id}>
                        <td>
                          <span className="fw-bold small">{pt.patient_id}</span><br />
                          <span className="text-muted small">{pt.name}</span>
                        </td>
                        <td>{pt.age}</td>
                        <td className="small">{pt.disease}</td>
                        <td><span className={`badge bg-${SEV_COLOR(pt.overall_severity)}`}>{pt.overall_severity}</span></td>
                        <td><span className={`badge bg-${PAT_COLOR(pt.diagnostic_pattern)}`}>{PAT_LABEL[pt.diagnostic_pattern] || pt.diagnostic_pattern}</span></td>
                        <td className="text-muted small">{pt.abnormal_muscles} / {pt.total_muscles}</td>
                        <td>
                          <button className="btn btn-outline-secondary btn-sm"
                            onClick={() => setExpandedPt(expandedPt === pt.patient_id ? null : pt.patient_id)}>
                            {expandedPt === pt.patient_id ? '▲' : '▼'}
                          </button>
                        </td>
                      </tr>
                      {expandedPt === pt.patient_id && pt.muscles && (
                        <tr key={`${pt.patient_id}-detail`}>
                          <td colSpan={7} className="bg-light p-3">
                            <table className="table table-sm table-bordered mb-0 small">
                              <thead className="table-secondary">
                                <tr>
                                  <th>Muscle</th>
                                  <th>Limb</th>
                                  <th>MUAP Dur (ms)</th>
                                  <th>MUAP Amp (µV)</th>
                                  <th>Phases</th>
                                  <th>Polyphasic</th>
                                  <th>Recruitment</th>
                                  <th>Spontaneous</th>
                                  <th>Severity</th>
                                </tr>
                              </thead>
                              <tbody>
                                {(pt.muscles || []).map(m => (
                                  <tr key={m.muscle}>
                                    <td className="fw-bold">{m.muscle}</td>
                                    <td>{LIMB_ICON(m.limb)} {m.limb}</td>
                                    <td className={m.duration_abnormal ? 'text-danger fw-bold' : ''}>{m.muap_duration_ms}</td>
                                    <td className={m.amplitude_abnormal ? 'text-danger fw-bold' : ''}>{m.muap_amplitude_uv}</td>
                                    <td>{m.phases}</td>
                                    <td>{m.polyphasic ? <span className="text-warning">Yes</span> : 'No'}</td>
                                    <td>{m.recruitment}</td>
                                    <td>{m.spontaneous_activity}</td>
                                    <td><span className={`badge bg-${SEV_COLOR(m.severity)}`}>{m.severity}</span></td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </td>
                        </tr>
                      )}
                    </>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'defs' && defs && (
        <>
          <div className="card mb-3">
            <div className="card-header py-2 small fw-bold">Overview — {defs.title}</div>
            <div className="card-body small">{defs.protocol?.description}</div>
          </div>

          {/* Parameters table */}
          <div className="card mb-3">
            <div className="card-header py-2 small fw-bold">EMG Parameters</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0 small">
                  <thead className="table-dark">
                    <tr><th>Parameter</th><th>Unit</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {(defs.parameters || []).map((p, i) => (
                      <tr key={i}>
                        <td className="fw-bold">{p.name}</td>
                        <td><span className="badge bg-secondary">{p.unit}</span></td>
                        <td className="text-muted">{p.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Reference ranges */}
          {defs.reference_ranges && (
            <div className="card mb-3">
              <div className="card-header py-2 small fw-bold">Reference Ranges</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0 small">
                    <thead className="table-dark">
                      <tr><th>Parameter</th><th>Normal Range</th><th>Increased in</th><th>Decreased in</th></tr>
                    </thead>
                    <tbody>
                      {(defs.reference_ranges || []).map((r, i) => (
                        <tr key={i}>
                          <td className="fw-bold">{r.parameter}</td>
                          <td><span className="badge bg-success">{r.normal_range}</span></td>
                          <td className="text-muted small">{r.increased_in}</td>
                          <td className="text-muted small">{r.decreased_in}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Diagnostic patterns */}
          <div className="card mb-3">
            <div className="card-header py-2 small fw-bold">Diagnostic Patterns</div>
            <div className="card-body">
              <div className="row g-2">
                {(defs.diagnostic_patterns || []).map(p => (
                  <div key={p.pattern} className="col-md-6">
                    <div className={`card border-${PAT_COLOR(p.pattern)} h-100`}>
                      <div className={`card-header py-1 small bg-${PAT_COLOR(p.pattern)} text-white`}>
                        {p.label}
                      </div>
                      <div className="card-body small">
                        <p className="mb-1">{p.description}</p>
                        {p.muap_changes && (
                          <p className="text-info mb-1">
                            <strong>MUAP:</strong> {p.muap_changes}
                          </p>
                        )}
                        <p className="text-muted mb-0"><em>{p.clinical_note}</em></p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Clinical significance */}
          {defs.clinical_significance && (
            <div className="card mb-3">
              <div className="card-header py-2 small fw-bold">EMG in Epilepsy Practice</div>
              <div className="card-body">
                <div className="row g-2">
                  {(defs.clinical_significance || []).map((e, i) => (
                    <div key={i} className="col-md-6">
                      <div className="card border-info h-100">
                        <div className="card-header py-1 small bg-info text-white">{e.topic}</div>
                        <div className="card-body small">
                          <p className="mb-1">{e.description}</p>
                          {e.citation && <p className="text-muted mb-0 fst-italic">{e.citation}</p>}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Reference */}
          {defs.reference && (
            <div className="card">
              <div className="card-header py-2 small fw-bold">Standards &amp; References</div>
              <div className="card-body small text-muted">{defs.reference}</div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
