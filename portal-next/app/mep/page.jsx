'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SEV_COLOR = s =>
  s === 'Normal' ? 'success' : s === 'Mild' ? 'info' : s === 'Moderate' ? 'warning' : 'danger';

const PAT_LABEL = {
  normal: 'Normal',
  upper_motor_neuron: 'UMN Dysfunction',
  cortical_lesion: 'Cortical Lesion',
  corticospinal_tract: 'CST Lesion',
  postictal_motor: "Postictal (Todd's)",
  cervical_myelopathy: 'Cervical Myelopathy',
};

const PAT_COLOR = p => ({
  normal: 'success',
  upper_motor_neuron: 'warning',
  cortical_lesion: 'danger',
  corticospinal_tract: 'danger',
  postictal_motor: 'info',
  cervical_myelopathy: 'warning',
}[p] || 'secondary');

const LIMB_ICON = l => l === 'Upper' ? '🖐️' : '🦶';

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

export default function MEPPage() {
  const [ov, setOv]         = useState(null);
  const [bd, setBd]         = useState(null);
  const [defs, setDefs]     = useState(null);
  const [tab, setTab]       = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);
  const [bdTab, setBdTab]   = useState('APB');

  useEffect(() => {
    fetch(`${API}/api/mep/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/mep/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/mep/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpi = ov.kpis || {};
  const tabs = [
    { id: 'overview',  label: '📊 Overview' },
    { id: 'breakdown', label: '🔬 Muscle Analysis' },
    { id: 'patients',  label: '🧑‍⚕️ Per Patient' },
    { id: 'defs',      label: '📖 Definitions' },
  ];

  const muscles = ['APB', 'ADM', 'TA', 'AH'];

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1 fw-bold">⚡ Motor Evoked Potentials (MEP) Dashboard</h4>
      <p className="text-muted small mb-3">
        TMS-evoked corticospinal tract integrity — CMCT, latency, amplitude &amp; CSP
        across 4 muscles (APB · ADM · TA · AH), bilateral assessment, 30 studies
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
              { label: 'Total Studies',     val: kpi.total_studies,           color: 'primary' },
              { label: 'Abnormal',          val: kpi.abnormal_count,          color: 'danger' },
              { label: 'Abnormal Rate',     val: `${kpi.abnormal_rate_pct}%`, color: 'warning' },
              { label: 'Muscles Tested',    val: kpi.muscles_tested,          color: 'info' },
              { label: 'Total Recordings',  val: kpi.total_muscle_recordings, color: 'secondary' },
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
                  {(ov.pattern_distribution || []).filter(p => p.count > 0).map(p => (
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

            {/* Limb abnormality rates */}
            <div className="col-md-3">
              <div className="card h-100">
                <div className="card-header py-2 small fw-bold">Limb Abnormality Rate</div>
                <div className="card-body">
                  {(ov.limb_abnormality_rates || []).map(l => (
                    <div key={l.limb} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{LIMB_ICON(l.limb)} {l.limb} limb</span>
                        <span className="fw-bold text-warning">{l.rate_pct}%</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div className="progress-bar bg-warning" style={{ width: `${l.rate_pct}%` }} />
                      </div>
                      <div className="text-muted small mt-1">{l.abnormal} / {l.total} recordings</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Muscle summary table */}
          <div className="card">
            <div className="card-header py-2 small fw-bold">Mean Parameters by Muscle</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Muscle</th>
                      <th>Limb</th>
                      <th>Mean CMCT (ms)</th>
                      <th>Ref CMCT</th>
                      <th>Mean Latency (ms)</th>
                      <th>Ref Latency</th>
                      <th>Mean Amplitude (µV)</th>
                      <th>Ref Amplitude</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.muscle_summary || []).map(m => (
                      <tr key={m.muscle}>
                        <td><span className="badge bg-secondary">{m.muscle}</span> {m.muscle_name}</td>
                        <td>{LIMB_ICON(m.limb)} {m.limb}</td>
                        <td className={m.mean_cmct_ms > m.cmct_ref_ms ? 'text-danger fw-bold' : 'text-success'}>
                          {m.mean_cmct_ms}
                        </td>
                        <td className="text-muted">≤{m.cmct_ref_ms}</td>
                        <td className={m.mean_latency_ms > m.latency_ref_ms ? 'text-danger fw-bold' : 'text-success'}>
                          {m.mean_latency_ms}
                        </td>
                        <td className="text-muted">≤{m.latency_ref_ms}</td>
                        <td className={m.mean_amplitude_uv < m.amplitude_ref_uv ? 'text-danger fw-bold' : 'text-success'}>
                          {m.mean_amplitude_uv}
                        </td>
                        <td className="text-muted">≥{m.amplitude_ref_uv}</td>
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
          {/* Muscle selector */}
          <ul className="nav nav-pills mb-3">
            {muscles.map(m => (
              <li key={m} className="nav-item">
                <button
                  className={`nav-link btn btn-sm me-1 ${bdTab === m ? 'active' : 'btn-outline-secondary'}`}
                  onClick={() => setBdTab(m)}
                >{m}</button>
              </li>
            ))}
          </ul>

          {bd.muscle_histograms && bd.muscle_histograms[bdTab] && (() => {
            const mh = bd.muscle_histograms[bdTab];
            const sc = (bd.side_comparison || []).filter(x => x.muscle === bdTab);
            return (
              <>
                <div className="row g-3 mb-3">
                  {/* CMCT histogram */}
                  <div className="col-md-4">
                    <div className="card h-100">
                      <div className="card-header py-2 small fw-bold">
                        {mh.muscle_name} — CMCT Distribution
                        <span className="text-muted ms-1">(ref ≤{mh.cmct_ref} ms)</span>
                      </div>
                      <div className="card-body">
                        {(mh.cmct_histogram || []).map(b => (
                          <Bar key={b.bin} label={`${b.bin} ms`} val={b.count}
                            max={Math.max(...(mh.cmct_histogram || []).map(x => x.count), 1)}
                            colorClass={parseFloat(b.bin) > mh.cmct_ref ? 'danger' : 'success'} />
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Latency histogram */}
                  <div className="col-md-4">
                    <div className="card h-100">
                      <div className="card-header py-2 small fw-bold">
                        MEP Latency Distribution
                        <span className="text-muted ms-1">(ref ≤{mh.latency_ref} ms)</span>
                      </div>
                      <div className="card-body">
                        {(mh.latency_histogram || []).map(b => (
                          <Bar key={b.bin} label={`${b.bin} ms`} val={b.count}
                            max={Math.max(...(mh.latency_histogram || []).map(x => x.count), 1)}
                            colorClass={parseFloat(b.bin) > mh.latency_ref ? 'danger' : 'info'} />
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Amplitude histogram */}
                  <div className="col-md-4">
                    <div className="card h-100">
                      <div className="card-header py-2 small fw-bold">
                        MEP Amplitude Distribution
                        <span className="text-muted ms-1">(ref ≥{mh.amplitude_ref} µV)</span>
                      </div>
                      <div className="card-body">
                        {(mh.amplitude_histogram || []).map(b => (
                          <Bar key={b.bin} label={`${b.bin} µV`} val={b.count}
                            max={Math.max(...(mh.amplitude_histogram || []).map(x => x.count), 1)}
                            colorClass={parseFloat(b.bin) < mh.amplitude_ref ? 'danger' : 'success'} />
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Left vs Right comparison */}
                <div className="card">
                  <div className="card-header py-2 small fw-bold">Left vs Right Comparison — {bdTab} ({mh.muscle_name})</div>
                  <div className="card-body p-0">
                    <div className="table-responsive">
                      <table className="table table-sm table-hover mb-0">
                        <thead className="table-dark">
                          <tr>
                            <th>Side</th>
                            <th>Mean CMCT (ms)</th>
                            <th>Mean Latency (ms)</th>
                            <th>Mean Amplitude (µV)</th>
                            <th>Abnormal</th>
                            <th>Abnormal Rate</th>
                          </tr>
                        </thead>
                        <tbody>
                          {sc.map(s => (
                            <tr key={s.side}>
                              <td><span className={`badge bg-${s.side === 'Left' ? 'primary' : 'success'}`}>{s.side}</span></td>
                              <td>{s.mean_cmct_ms} ms</td>
                              <td>{s.mean_latency_ms} ms</td>
                              <td>{s.mean_amplitude_uv} µV</td>
                              <td>{s.abnormal_count} / {s.total}</td>
                              <td>
                                <span className={`badge bg-${s.abnormal_rate_pct > 20 ? 'danger' : 'success'}`}>
                                  {s.abnormal_rate_pct}%
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
            );
          })()}
        </>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && bd && (
        <div className="card">
          <div className="card-header py-2 small fw-bold">Per-Patient MEP Results ({(bd.per_patient || []).length} studies)</div>
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
                    <th>Abnormal</th>
                    <th>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.per_patient || []).map(pt => (
                    <>
                      <tr key={pt.patient_id}>
                        <td><span className="fw-bold small">{pt.patient_id}</span><br /><span className="text-muted small">{pt.name}</span></td>
                        <td>{pt.age}</td>
                        <td className="small">{pt.disease}</td>
                        <td><span className={`badge bg-${SEV_COLOR(pt.overall_severity)}`}>{pt.overall_severity}</span></td>
                        <td><span className={`badge bg-${PAT_COLOR(pt.diagnostic_pattern)}`}>{PAT_LABEL[pt.diagnostic_pattern] || pt.diagnostic_pattern}</span></td>
                        <td className="text-muted small">{pt.muscles ? pt.muscles.flatMap(m => m.sides).filter(s => s.severity !== 'Normal').length : 0} / {pt.muscles ? pt.muscles.length * 2 : 8}</td>
                        <td>
                          <button className="btn btn-outline-secondary btn-sm"
                            onClick={() => setExpandedPt(expandedPt === pt.patient_id ? null : pt.patient_id)}>
                            {expandedPt === pt.patient_id ? '▲' : '▼'}
                          </button>
                        </td>
                      </tr>
                      {expandedPt === pt.patient_id && (
                        <tr key={`${pt.patient_id}-detail`}>
                          <td colSpan={7} className="bg-light p-3">
                            <table className="table table-sm table-bordered mb-0 small">
                              <thead className="table-secondary">
                                <tr>
                                  <th>Muscle</th>
                                  <th>Limb</th>
                                  <th>Side</th>
                                  <th>Latency (ms)</th>
                                  <th>Amplitude (µV)</th>
                                  <th>CMCT (ms)</th>
                                  <th>CSP (ms)</th>
                                  <th>Severity</th>
                                </tr>
                              </thead>
                              <tbody>
                                {(pt.muscles || []).flatMap(m =>
                                  (m.sides || []).map(s => (
                                    <tr key={`${m.muscle}-${s.side}`}>
                                      <td><span className="badge bg-secondary">{m.muscle}</span></td>
                                      <td>{LIMB_ICON(m.limb)} {m.limb}</td>
                                      <td>{s.side}</td>
                                      <td className={s.latency_abnormal ? 'text-danger fw-bold' : ''}>{s.latency_ms}</td>
                                      <td className={s.amplitude_abnormal ? 'text-danger fw-bold' : ''}>{s.amplitude_uv}</td>
                                      <td className={s.cmct_abnormal ? 'text-danger fw-bold' : ''}>{s.cmct_ms}</td>
                                      <td>{s.csp_ms}</td>
                                      <td><span className={`badge bg-${SEV_COLOR(s.severity)}`}>{s.severity}</span></td>
                                    </tr>
                                  ))
                                )}
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
            <div className="card-body small">{defs.overview}</div>
          </div>

          {/* Normative values table */}
          <div className="card mb-3">
            <div className="card-header py-2 small fw-bold">Normative Reference Values</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0 small">
                  <thead className="table-dark">
                    <tr><th>Parameter</th><th>Limit</th><th>Note</th></tr>
                  </thead>
                  <tbody>
                    {(defs.normative_values || []).map((v, i) => (
                      <tr key={i}>
                        <td className="fw-bold">{v.parameter}</td>
                        <td>
                          {v.upper_limit_ms !== undefined && <span className="badge bg-warning text-dark">≤{v.upper_limit_ms} ms</span>}
                          {v.lower_limit_ms !== undefined && <span className="badge bg-info text-dark">≥{v.lower_limit_ms} ms</span>}
                          {v.lower_limit_uv !== undefined && <span className="badge bg-info text-dark">≥{v.lower_limit_uv} µV</span>}
                        </td>
                        <td className="text-muted">{v.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

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
                        <p className="text-muted mb-0"><em>{p.clinical_note}</em></p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Epilepsy relevance */}
          <div className="card mb-3">
            <div className="card-header py-2 small fw-bold">MEP in Epilepsy Practice</div>
            <div className="card-body">
              <div className="row g-2">
                {(defs.epilepsy_relevance || []).map((e, i) => (
                  <div key={i} className="col-md-6">
                    <div className="card border-info h-100">
                      <div className="card-header py-1 small bg-info text-white">{e.topic}</div>
                      <div className="card-body small">
                        <p className="mb-1">{e.description}</p>
                        <p className="text-muted mb-0 fst-italic">{e.citation}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Safety */}
          <div className="card">
            <div className="card-header py-2 small fw-bold">Safety &amp; Contraindications</div>
            <div className="card-body small">
              <ul className="mb-0">
                {(defs.safety_contraindications || []).map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
