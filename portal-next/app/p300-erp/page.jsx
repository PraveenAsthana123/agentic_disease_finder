'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s =>
  s === 'Normal' ? 'success' : s === 'Mild' ? 'info' : s === 'Moderate' ? 'warning' : 'danger';

const patLabel = p => ({
  normal: 'Normal',
  prolonged_latency: 'Prolonged Latency',
  reduced_amplitude: 'Reduced Amplitude',
  combined: 'Combined',
  absent_p300: 'Absent P300',
}[p] || p);

const patColor = p =>
  p === 'normal' ? 'success'
  : p === 'prolonged_latency' ? 'warning'
  : p === 'reduced_amplitude' ? 'info'
  : p === 'combined' ? 'danger'
  : p === 'absent_p300' ? 'dark' : 'secondary';

const abnVal = (val, isAbn) =>
  <span className={isAbn ? 'text-danger fw-bold' : 'text-success'}>{val}{isAbn ? ' !' : ''}</span>;

export default function P300ERPPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/p300-erp/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/p300-erp/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/p300-erp/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'analysis',    label: 'ERP Analysis' },
    { id: 'patients',    label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const k = ov.kpis;

  return (
    <div>
      <h3>P300 / ERP — Cognitive Event-Related Potentials</h3>
      <p className="text-muted">
        Auditory 3-tone oddball paradigm · P300/N200/P3a latency &amp; amplitude ·
        Age-adjusted references · AED cognitive impact · Epilepsy-relevant ERP assessment
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Studies',     value: k.total_studies,          color: 'primary' },
          { label: 'Abnormal',          value: k.abnormal_count,         color: 'danger' },
          { label: 'Abnormal Rate',     value: `${k.abnormal_rate_pct}%`, color: k.abnormal_rate_pct > 35 ? 'danger' : 'warning' },
          { label: 'Mean P300 (ms)',    value: k.mean_p300_latency_ms,   color: k.mean_p300_latency_ms > 380 ? 'danger' : k.mean_p300_latency_ms > 340 ? 'warning' : 'success' },
          { label: 'Mean P300 Amp (µV)', value: k.mean_p300_amplitude_uv, color: k.mean_p300_amplitude_uv < 5 ? 'danger' : 'success' },
          { label: 'Mean N200 (ms)',    value: k.mean_n200_latency_ms,   color: k.mean_n200_latency_ms > 280 ? 'danger' : 'success' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className={`card-body p-2 bg-${c.color} bg-opacity-10`}>
                <div className={`fs-4 fw-bold text-${c.color}`}>{c.value}</div>
                <div className="small text-muted">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tab nav */}
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
        <div>
          <div className="row">
            {/* Severity */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Severity Distribution</div>
                <div className="card-body">
                  {ov.severity_distribution.map(sv => (
                    <div key={sv.severity} className="mb-2">
                      <div className="d-flex justify-content-between">
                        <span className={`badge bg-${sevColor(sv.severity)}`}>{sv.severity}</span>
                        <span className="fw-bold">{sv.count}</span>
                      </div>
                      <div className="progress" style={{ height: 6 }}>
                        <div
                          className={`progress-bar bg-${sevColor(sv.severity)}`}
                          style={{ width: `${(sv.count / k.total_studies) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Diagnostic patterns */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Diagnostic Patterns</div>
                <div className="card-body">
                  {ov.pattern_distribution.map(p => (
                    <div key={p.pattern} className="d-flex justify-content-between align-items-center mb-2">
                      <span className={`badge bg-${patColor(p.pattern)} me-2`}>{p.label}</span>
                      <span className="fw-bold">{p.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* AED cognitive impact */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">AED Cognitive Impact</div>
                <div className="card-body">
                  <table className="table table-sm mb-0">
                    <thead><tr>
                      <th>AED Burden</th><th>N</th><th>Mean P300 (ms)</th><th>Abnormal%</th>
                    </tr></thead>
                    <tbody>
                      {ov.aed_cognitive_impact && (
                        <>
                          <tr>
                            <td><span className="badge bg-danger">High (&gt;2 AEDs)</span></td>
                            <td>{ov.aed_cognitive_impact.high_aed_burden.n}</td>
                            <td className="text-danger fw-bold">{ov.aed_cognitive_impact.high_aed_burden.mean_p300_ms}</td>
                            <td>{ov.aed_cognitive_impact.high_aed_burden.abnormal_pct}%</td>
                          </tr>
                          <tr>
                            <td><span className="badge bg-success">Low (≤2 AEDs)</span></td>
                            <td>{ov.aed_cognitive_impact.low_aed_burden.n}</td>
                            <td className="text-success fw-bold">{ov.aed_cognitive_impact.low_aed_burden.mean_p300_ms}</td>
                            <td>{ov.aed_cognitive_impact.low_aed_burden.abnormal_pct}%</td>
                          </tr>
                        </>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Age-band analysis */}
          {ov.age_band_analysis && ov.age_band_analysis.length > 0 && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">P300 Latency by Age Band (Polich 2007 age-adjusted reference)</div>
              <div className="card-body">
                <table className="table table-sm table-hover">
                  <thead><tr>
                    <th>Age Band</th><th>N</th>
                    <th>Mean P300 (ms)</th><th>Age-Adj Ref (ms)</th>
                    <th>Mean Amp (µV)</th><th>Abnormal%</th>
                  </tr></thead>
                  <tbody>
                    {ov.age_band_analysis.map(b => (
                      <tr key={b.age_band}>
                        <td><strong>{b.age_band}</strong></td>
                        <td>{b.n}</td>
                        <td className={b.mean_p300_ms > b.ref_upper_ms ? 'text-danger fw-bold' : 'text-success'}>
                          {b.mean_p300_ms}
                        </td>
                        <td className="text-muted">≤{b.ref_upper_ms}</td>
                        <td className={b.mean_amp_uv < 5 ? 'text-danger' : 'text-success'}>
                          {b.mean_amp_uv}
                        </td>
                        <td>
                          <span className={`badge bg-${b.abnormal_pct > 50 ? 'danger' : b.abnormal_pct > 25 ? 'warning' : 'success'}`}>
                            {b.abnormal_pct}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Per-patient summary */}
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Per-Patient Summary</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr>
                    <th>Patient</th><th>Age</th><th>Disease</th>
                    <th>Severity</th><th>Pattern</th>
                    <th>P300 (ms)</th><th>Ref (ms)</th><th>Amp (µV)</th>
                  </tr></thead>
                  <tbody>
                    {ov.patient_summary.map(p => (
                      <tr key={p.patient_id}>
                        <td className="small">{p.name}</td>
                        <td>{p.age}</td>
                        <td className="small text-muted">{(p.disease || '').slice(0, 25)}</td>
                        <td><span className={`badge bg-${sevColor(p.severity)}`}>{p.severity}</span></td>
                        <td><span className={`badge bg-${patColor(p.pattern)}`}>{patLabel(p.pattern)}</span></td>
                        <td className={p.p300_ms > p.ref_upper_ms ? 'text-danger fw-bold' : 'text-success'}>
                          {p.p300_ms}
                        </td>
                        <td className="text-muted small">≤{p.ref_upper_ms}</td>
                        <td className={p.p300_amp_uv < 5 ? 'text-danger fw-bold' : 'text-success'}>
                          {p.p300_amp_uv}
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

      {/* ── ERP ANALYSIS ── */}
      {tab === 'analysis' && bd && (
        <div>
          <div className="row">
            {/* P300 latency histogram */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">P300 Latency Distribution (ms) — ref varies by age</div>
                <div className="card-body">
                  {bd.p300_latency_histogram.map(b => (
                    <div key={b.range} className="mb-2">
                      <div className="d-flex justify-content-between">
                        <span className="small">{b.range} ms</span>
                        <span className="fw-bold">{b.count}</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div
                          className={`progress-bar ${b.range.includes('+') || b.range.includes('390') || b.range.includes('360-390') ? 'bg-danger' : b.range.includes('330') ? 'bg-warning' : 'bg-success'}`}
                          style={{ width: `${(b.count / ov.kpis.total_studies) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                  <div className="text-muted small mt-2">≥360 ms may exceed age-adjusted reference for patients &lt;60 yr</div>
                </div>
              </div>
            </div>

            {/* P300 amplitude histogram */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">P300 Amplitude Distribution (µV) — ref ≥5.0 µV</div>
                <div className="card-body">
                  {bd.p300_amplitude_histogram.map(b => (
                    <div key={b.range} className="mb-2">
                      <div className="d-flex justify-content-between">
                        <span className="small">{b.range} µV</span>
                        <span className="fw-bold">{b.count}</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div
                          className={`progress-bar ${b.range === '<3' ? 'bg-danger' : b.range === '3-5' ? 'bg-warning' : 'bg-success'}`}
                          style={{ width: `${(b.count / ov.kpis.total_studies) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                  <div className="text-muted small mt-2">&lt;5 µV = below normal reference lower bound</div>
                </div>
              </div>
            </div>
          </div>

          {/* Component summary table */}
          <div className="card shadow-sm">
            <div className="card-header fw-bold">ERP Component Summary (P300 / N200 / P3a)</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead><tr>
                  <th>Component</th><th>Mean Latency (ms)</th><th>Reference</th>
                  <th>Mean Amplitude (µV)</th><th>Amp Reference</th><th>Abnormal%</th>
                </tr></thead>
                <tbody>
                  {bd.component_summary.map(c => (
                    <tr key={c.component}>
                      <td><strong>{c.component}</strong></td>
                      <td className="fw-bold">{c.mean_latency_ms}</td>
                      <td className="text-muted small">≤{c.ref_upper_ms}</td>
                      <td>{c.mean_amplitude_uv}</td>
                      <td className="text-muted small">{c.ref_lower_uv ? `≥${c.ref_lower_uv} µV` : '—'}</td>
                      <td>
                        <span className={`badge bg-${c.abnormal_pct > 40 ? 'danger' : c.abnormal_pct > 20 ? 'warning' : 'success'}`}>
                          {c.abnormal_pct}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── PATIENT DETAIL ── */}
      {tab === 'patients' && bd && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold">Per-Patient ERP Detail</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead><tr>
                  <th>Patient</th><th>Age</th><th>Sev</th><th>Pattern</th>
                  <th>P300 (ms)</th><th>Ref</th><th>P300 Amp</th>
                  <th>N200 (ms)</th><th>N200 Amp</th><th>P3a (ms)</th>
                  <th>Meds</th><th></th>
                </tr></thead>
                <tbody>
                  {bd.patient_details.map(p => (
                    <>
                      <tr
                        key={p.patient_id}
                        style={{ cursor: 'pointer' }}
                        onClick={() => setExpandedPt(expandedPt === p.patient_id ? null : p.patient_id)}
                      >
                        <td className="small">{p.name}</td>
                        <td>{p.age}</td>
                        <td><span className={`badge bg-${sevColor(p.severity)}`}>{p.severity}</span></td>
                        <td><span className={`badge bg-${patColor(p.pattern)} small`}>{patLabel(p.pattern)}</span></td>
                        <td>{abnVal(p.p300_ms, p.p300_lat_abn)}</td>
                        <td className="text-muted small">≤{p.p300_ref_ms}</td>
                        <td>{abnVal(p.p300_amp_uv + ' µV', p.p300_amp_abn)}</td>
                        <td>{abnVal(p.n200_ms, p.n200_lat_abn)}</td>
                        <td>{abnVal(p.n200_amp_uv + ' µV', p.n200_amp_abn)}</td>
                        <td>{abnVal(p.p3a_ms, p.p3a_lat_abn)}</td>
                        <td>{p.med_count}</td>
                        <td>{expandedPt === p.patient_id ? '▲' : '▼'}</td>
                      </tr>
                      {expandedPt === p.patient_id && (
                        <tr key={p.patient_id + '_exp'}>
                          <td colSpan={12} className="bg-light p-3">
                            <div className="row">
                              <div className="col-md-6">
                                <strong>Disease:</strong> {p.disease}<br/>
                                <strong>Medications:</strong> {p.med_count}<br/>
                                <strong>MMSE:</strong> {p.mmse != null ? p.mmse : 'N/A'} &nbsp;
                                <strong>MoCA:</strong> {p.moca != null ? p.moca : 'N/A'}
                              </div>
                              <div className="col-md-6">
                                <strong>P300 Latency:</strong> {p.p300_ms} ms
                                {p.p300_lat_abn ? <span className="text-danger ms-1">(exceeds ref ≤{p.p300_ref_ms} ms)</span> : <span className="text-success ms-1">(within ref)</span>}<br/>
                                <strong>P300 Amplitude:</strong> {p.p300_amp_uv} µV
                                {p.p300_amp_abn ? <span className="text-danger ms-1">(below ref ≥5.0 µV)</span> : <span className="text-success ms-1">(within ref)</span>}<br/>
                                <strong>N200 Latency:</strong> {p.n200_ms} ms
                                {p.n200_lat_abn ? <span className="text-danger ms-1">(exceeds ref ≤280 ms)</span> : ''}<br/>
                                <strong>P3a Latency:</strong> {p.p3a_ms} ms
                                {p.p3a_lat_abn ? <span className="text-danger ms-1">(exceeds ref ≤330 ms)</span> : ''}
                              </div>
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
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">{defs.title}</div>
            <div className="card-body">
              <p>{defs.protocol.description}</p>
              <div className="row">
                <div className="col-md-6">
                  <strong>Paradigm:</strong> {defs.protocol.paradigm}<br/>
                  <strong>Epoch window:</strong> {defs.protocol.epoch_window}<br/>
                  <strong>Recording sites:</strong> {defs.protocol.recording_sites.join(', ')}
                </div>
                <div className="col-md-6">
                  <strong>Standard:</strong> <span className="small text-muted">{defs.protocol.standard}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Parameters */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">ERP Components &amp; Parameters</div>
            <div className="card-body">
              {defs.parameters.map(p => (
                <div key={p.name} className="mb-3">
                  <strong>{p.name}</strong>
                  {p.unit && <span className="text-muted ms-1">({p.unit})</span>}
                  <p className="small mt-1 mb-0">{p.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Epilepsy relevance */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Epilepsy Relevance</div>
            <div className="card-body">
              {defs.epilepsy_relevance.map(er => (
                <div key={er.context} className="mb-3">
                  <span className="badge bg-primary me-2">{er.context}</span>
                  <p className="small mt-1 mb-0">{er.detail}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Diagnostic patterns */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Diagnostic Patterns</div>
            <div className="card-body">
              {defs.diagnostic_patterns.map(p => (
                <div key={p.pattern} className="d-flex align-items-start mb-2">
                  <span className={`badge bg-${patColor(p.pattern)} me-2`}>{patLabel(p.pattern)}</span>
                  <span className="small">{p.description}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Severity */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Severity Levels</div>
            <div className="card-body">
              {defs.severity_levels.map(sv => (
                <div key={sv.level} className="mb-2">
                  <span className={`badge bg-${sevColor(sv.level)} me-2`}>{sv.level}</span>
                  <span className="small">{sv.criteria}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="alert alert-light small">
            <strong>Reference:</strong> {defs.reference}
          </div>
        </div>
      )}
    </div>
  );
}
