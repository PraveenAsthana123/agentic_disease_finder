'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s =>
  s === 'Normal' ? 'success' : s === 'Mild' ? 'info' : s === 'Moderate' ? 'warning' : 'danger';

const patLabel = p => ({
  normal:             'Normal',
  prolonged_latency:  'Prolonged Latency',
  reduced_amplitude:  'Reduced Amplitude',
  combined:           'Combined',
  absent_mmn:         'Absent MMN',
  asymmetric_mmn:     'Asymmetric MMN',
}[p] || p);

const patColor = p =>
  p === 'normal'            ? 'success'
  : p === 'prolonged_latency' ? 'warning'
  : p === 'reduced_amplitude' ? 'info'
  : p === 'combined'          ? 'danger'
  : p === 'absent_mmn'        ? 'dark'
  : p === 'asymmetric_mmn'    ? 'secondary'
  : 'secondary';

const abnBadge = (val, isAbn) =>
  <span className={isAbn ? 'text-danger fw-bold' : 'text-success'}>{val}{isAbn ? ' !' : ''}</span>;

export default function MmnPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/mmn/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/mmn/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/mmn/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',  label: 'Overview' },
    { id: 'analysis',  label: 'ERP Analysis' },
    { id: 'patients',  label: 'Patient Detail' },
    { id: 'defs',      label: 'Definitions' },
  ];

  const k = ov.kpis;

  return (
    <div>
      <h3>MMN — Mismatch Negativity Dashboard</h3>
      <p className="text-muted">
        Auditory deviant-standard paradigm · MMN latency &amp; amplitude · Hemispheric asymmetry ·
        P3a novelty response · AED impact · Epilepsy-relevant pre-attentive auditory discrimination
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Studies',       value: k.total_studies,            color: 'primary' },
          { label: 'Abnormal',            value: k.abnormal_count,           color: 'danger' },
          { label: 'Abnormal Rate',       value: `${k.abnormal_rate_pct}%`,  color: k.abnormal_rate_pct > 35 ? 'danger' : 'warning' },
          { label: 'Mean MMN Lat (ms)',   value: k.mean_mmn_latency_ms,      color: k.mean_mmn_latency_ms > 200 ? 'danger' : k.mean_mmn_latency_ms > 180 ? 'warning' : 'success' },
          { label: 'Mean MMN Amp (µV)',   value: k.mean_mmn_amplitude_uv,    color: k.mean_mmn_amplitude_uv < 1.5 ? 'danger' : 'success' },
          { label: 'Mean Asymmetry (%)',  value: `${k.mean_asymmetry_pct}%`, color: k.mean_asymmetry_pct > 30 ? 'danger' : k.mean_asymmetry_pct > 20 ? 'warning' : 'success' },
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
            {/* Severity distribution */}
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
                      <span className={`badge bg-${patColor(p.pattern)} me-2`}>{p.label || patLabel(p.pattern)}</span>
                      <span className="fw-bold">{p.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* AED impact */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">AED Impact on MMN</div>
                <div className="card-body">
                  {ov.aed_impact && (
                    <table className="table table-sm mb-0">
                      <thead><tr>
                        <th>AED Burden</th><th>N</th><th>Mean MMN (ms)</th><th>Abnormal%</th>
                      </tr></thead>
                      <tbody>
                        <tr>
                          <td><span className="badge bg-danger">High (&gt;2 AEDs)</span></td>
                          <td>{ov.aed_impact.high_aed_burden.n}</td>
                          <td className="text-danger fw-bold">{ov.aed_impact.high_aed_burden.mean_mmn_ms}</td>
                          <td>{ov.aed_impact.high_aed_burden.abnormal_pct}%</td>
                        </tr>
                        <tr>
                          <td><span className="badge bg-success">Low (≤2 AEDs)</span></td>
                          <td>{ov.aed_impact.low_aed_burden.n}</td>
                          <td className="text-success fw-bold">{ov.aed_impact.low_aed_burden.mean_mmn_ms}</td>
                          <td>{ov.aed_impact.low_aed_burden.abnormal_pct}%</td>
                        </tr>
                      </tbody>
                    </table>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Hemisphere analysis */}
          {ov.hemisphere_analysis && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">Hemispheric Dominance Analysis</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr>
                    <th>Dominance Group</th><th>N</th>
                    <th>Avg Asymmetry (%)</th><th>Avg MMN Amp (µV)</th><th>Abnormal%</th>
                  </tr></thead>
                  <tbody>
                    {[
                      { key: 'left_dominant',  label: 'Left-Dominant',  color: 'primary' },
                      { key: 'right_dominant', label: 'Right-Dominant', color: 'info' },
                      { key: 'symmetric',      label: 'Symmetric',      color: 'success' },
                    ].map(row => {
                      const g = ov.hemisphere_analysis[row.key];
                      if (!g) return null;
                      return (
                        <tr key={row.key}>
                          <td><span className={`badge bg-${row.color}`}>{row.label}</span></td>
                          <td>{g.n}</td>
                          <td className={g.avg_asymmetry_pct > 30 ? 'text-danger fw-bold' : 'text-success'}>
                            {g.avg_asymmetry_pct}%
                          </td>
                          <td className={g.avg_mmn_amp_uv < 1.5 ? 'text-danger fw-bold' : 'text-success'}>
                            {g.avg_mmn_amp_uv} µV
                          </td>
                          <td>
                            <span className={`badge bg-${g.abnormal_pct > 50 ? 'danger' : g.abnormal_pct > 25 ? 'warning' : 'success'}`}>
                              {g.abnormal_pct}%
                            </span>
                          </td>
                        </tr>
                      );
                    })}
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
                    <th>MMN Lat (ms)</th><th>MMN Amp (µV)</th><th>Asymmetry (%)</th>
                  </tr></thead>
                  <tbody>
                    {ov.patient_summary.map(p => (
                      <tr key={p.patient_id}>
                        <td className="small">{p.name}</td>
                        <td>{p.age}</td>
                        <td className="small text-muted">{(p.disease || '').slice(0, 25)}</td>
                        <td><span className={`badge bg-${sevColor(p.severity)}`}>{p.severity}</span></td>
                        <td><span className={`badge bg-${patColor(p.pattern)}`}>{patLabel(p.pattern)}</span></td>
                        <td className={p.mmn_latency_ms > 200 ? 'text-danger fw-bold' : 'text-success'}>
                          {p.mmn_latency_ms}
                        </td>
                        <td className={p.mmn_amplitude_uv < 1.5 ? 'text-danger fw-bold' : 'text-success'}>
                          {p.mmn_amplitude_uv}
                        </td>
                        <td className={p.asymmetry_index_pct > 30 ? 'text-danger fw-bold' : 'text-success'}>
                          {p.asymmetry_index_pct}%
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
            {/* MMN latency histogram */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">MMN Latency Distribution (ms) — ref ≤200 ms</div>
                <div className="card-body">
                  {bd.mmn_latency_histogram.map(b => (
                    <div key={b.range} className="mb-2">
                      <div className="d-flex justify-content-between">
                        <span className="small">{b.range} ms</span>
                        <span className="fw-bold">{b.count}</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div
                          className={`progress-bar ${b.range.includes('+') || b.range.includes('200') ? 'bg-danger' : b.range.includes('180') ? 'bg-warning' : 'bg-success'}`}
                          style={{ width: `${(b.count / k.total_studies) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                  <div className="text-muted small mt-2">&gt;200 ms exceeds normal upper bound</div>
                </div>
              </div>
            </div>

            {/* MMN amplitude histogram */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">MMN Amplitude Distribution (µV) — ref ≥1.5 µV</div>
                <div className="card-body">
                  {bd.mmn_amplitude_histogram.map(b => (
                    <div key={b.range} className="mb-2">
                      <div className="d-flex justify-content-between">
                        <span className="small">{b.range} µV</span>
                        <span className="fw-bold">{b.count}</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div
                          className={`progress-bar ${b.range === '<1' ? 'bg-danger' : b.range === '1-1.5' ? 'bg-warning' : 'bg-success'}`}
                          style={{ width: `${(b.count / k.total_studies) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                  <div className="text-muted small mt-2">&lt;1.5 µV = below normal reference</div>
                </div>
              </div>
            </div>

            {/* Asymmetry distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Asymmetry Index Distribution (%) — ref ≤30%</div>
                <div className="card-body">
                  {bd.asymmetry_distribution.map(b => (
                    <div key={b.range} className="mb-2">
                      <div className="d-flex justify-content-between">
                        <span className="small">{b.range}%</span>
                        <span className="fw-bold">{b.count}</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div
                          className={`progress-bar ${b.range.includes('+') || b.range.includes('40') ? 'bg-danger' : b.range.includes('30') ? 'bg-warning' : 'bg-success'}`}
                          style={{ width: `${(b.count / k.total_studies) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                  <div className="text-muted small mt-2">&gt;30% = significant hemispheric asymmetry</div>
                </div>
              </div>
            </div>
          </div>

          {/* Component summary table */}
          <div className="card shadow-sm">
            <div className="card-header fw-bold">MMN Component Summary (MMN Fz / P3a Fz)</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead><tr>
                  <th>Component</th><th>Mean Latency (ms)</th><th>Ref Upper (ms)</th>
                  <th>Mean Amplitude (µV)</th><th>Ref Lower (µV)</th><th>Abnormal%</th>
                </tr></thead>
                <tbody>
                  {bd.component_summary.map(c => (
                    <tr key={c.component}>
                      <td><strong>{c.component}</strong></td>
                      <td className={c.mean_latency_ms > c.ref_upper_ms ? 'text-danger fw-bold' : 'text-success fw-bold'}>
                        {c.mean_latency_ms}
                      </td>
                      <td className="text-muted small">≤{c.ref_upper_ms}</td>
                      <td className={c.mean_amplitude_uv < c.ref_lower_uv ? 'text-danger' : 'text-success'}>
                        {c.mean_amplitude_uv}
                      </td>
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
          <div className="card-header fw-bold">Per-Patient MMN Detail</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead><tr>
                  <th>Patient</th><th>Age</th><th>Sev</th><th>Pattern</th>
                  <th>MMN Lat (ms)</th><th>MMN Amp (µV)</th>
                  <th>P3a Lat (ms)</th><th>Asymmetry (%)</th>
                  <th>Hemisphere</th><th></th>
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
                        <td>{abnBadge(p.mmn_latency_ms, p.mmn_lat_abnormal)}</td>
                        <td>{abnBadge(`${p.mmn_amplitude_uv} µV`, p.mmn_amp_abnormal)}</td>
                        <td>{abnBadge(p.p3a_latency_ms, p.p3a_lat_abnormal)}</td>
                        <td>{abnBadge(`${p.asymmetry_index_pct}%`, p.asymmetry_abnormal)}</td>
                        <td><span className="badge bg-secondary">{p.hemisphere_dominant}</span></td>
                        <td>{expandedPt === p.patient_id ? '▲' : '▼'}</td>
                      </tr>
                      {expandedPt === p.patient_id && (
                        <tr key={p.patient_id + '_exp'}>
                          <td colSpan={10} className="bg-light p-3">
                            <div className="row">
                              <div className="col-md-6">
                                <strong>Disease:</strong> {p.disease}<br/>
                                <strong>Hemisphere Dominant:</strong> {p.hemisphere_dominant}<br/>
                                <strong>MMN Amp Left:</strong> {p.mmn_amp_left_uv} µV &nbsp;
                                <strong>Right:</strong> {p.mmn_amp_right_uv} µV
                              </div>
                              <div className="col-md-6">
                                <strong>MMN Latency:</strong> {p.mmn_latency_ms} ms
                                {p.mmn_lat_abnormal
                                  ? <span className="text-danger ms-1">(exceeds ref ≤200 ms)</span>
                                  : <span className="text-success ms-1">(within ref)</span>}<br/>
                                <strong>MMN Amplitude:</strong> {p.mmn_amplitude_uv} µV
                                {p.mmn_amp_abnormal
                                  ? <span className="text-danger ms-1">(below ref ≥1.5 µV)</span>
                                  : <span className="text-success ms-1">(within ref)</span>}<br/>
                                <strong>P3a Latency:</strong> {p.p3a_latency_ms} ms
                                {p.p3a_lat_abnormal ? <span className="text-danger ms-1">(exceeds ref ≤330 ms)</span> : ''}<br/>
                                <strong>P3a Amplitude:</strong> {p.p3a_amplitude_uv} µV<br/>
                                <strong>Asymmetry Index:</strong> {p.asymmetry_index_pct}%
                                {p.asymmetry_abnormal ? <span className="text-danger ms-1">(exceeds ref ≤30%)</span> : ''}
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
      {tab === 'defs' && defs && (
        <div>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">{defs.title}</div>
            <div className="card-body">
              <p>{defs.protocol.description}</p>
              <div className="row">
                <div className="col-md-6">
                  <strong>Paradigm:</strong> {defs.protocol.paradigm}<br/>
                  <strong>Epoch window:</strong> {defs.protocol.epoch_window}<br/>
                  <strong>Recording sites:</strong>{' '}
                  {Array.isArray(defs.protocol.recording_sites)
                    ? defs.protocol.recording_sites.join(', ')
                    : defs.protocol.recording_sites}
                </div>
                <div className="col-md-6">
                  <strong>Standard:</strong>{' '}
                  <span className="small text-muted">{defs.protocol.standard}</span><br/>
                  {defs.protocol.indications && (
                    <>
                      <strong>Indications:</strong>
                      <ul className="small mt-1 mb-0">
                        {defs.protocol.indications.map((ind, i) => <li key={i}>{ind}</li>)}
                      </ul>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Parameters */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">MMN Components &amp; Parameters</div>
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

          {/* Reference ranges */}
          {defs.reference_ranges && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">Reference Ranges</div>
              <div className="card-body">
                <table className="table table-sm mb-2">
                  <tbody>
                    <tr>
                      <th>MMN Latency Upper</th>
                      <td>{defs.reference_ranges.mmn_latency_upper_ms} ms</td>
                    </tr>
                    <tr>
                      <th>MMN Amplitude Ref</th>
                      <td>≥{defs.reference_ranges.mmn_amplitude_ref_uv} µV</td>
                    </tr>
                    <tr>
                      <th>P3a Latency Upper</th>
                      <td>{defs.reference_ranges.p3a_latency_upper_ms} ms</td>
                    </tr>
                    <tr>
                      <th>Asymmetry Index Upper</th>
                      <td>≤{defs.reference_ranges.asymmetry_index_upper_pct}%</td>
                    </tr>
                  </tbody>
                </table>
                {defs.reference_ranges.notes && (
                  <p className="small text-muted mb-0">{defs.reference_ranges.notes}</p>
                )}
              </div>
            </div>
          )}

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

          {/* Severity levels */}
          {defs.severity_levels && (
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
          )}

          <div className="alert alert-light small">
            <strong>Reference:</strong> {defs.reference}
          </div>
        </div>
      )}
    </div>
  );
}
