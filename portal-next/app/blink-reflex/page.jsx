'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SEV_COLOR = s =>
  s === 'Normal' ? 'success' : s === 'Mild' ? 'info' : s === 'Moderate' ? 'warning' : s === 'Severe' ? 'danger' : 'secondary';

const PAT_COLOR = p => ({
  normal:              'success',
  trigeminal_neuropathy: 'warning',
  facial_neuropathy:   'info',
  pontine_lesion:      'danger',
  medullary_lesion:    'primary',
}[p] || 'secondary');

function KPI({ label, value, sub, color = 'primary' }) {
  return (
    <div className={`card border-${color} h-100`}>
      <div className="card-body text-center p-3">
        <div className={`display-6 fw-bold text-${color}`}>{value}</div>
        <div className="small fw-semibold">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
      </div>
    </div>
  );
}

function Bar({ label, val, max, colorClass = 'primary', unit = '' }) {
  const pct = max > 0 ? Math.round((val / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{val}{unit}</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className={`progress-bar bg-${colorClass}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function LatencyBar({ label, mean, ref: refVal, unit = ' ms' }) {
  const pct = Math.min(Math.round((mean / (refVal * 1.5)) * 100), 100);
  const refPct = Math.round((refVal / (refVal * 1.5)) * 100);
  const over = mean > refVal;
  return (
    <div className="mb-3">
      <div className="d-flex justify-content-between small mb-1">
        <span className="fw-semibold">{label}</span>
        <span>
          <span className={`fw-bold text-${over ? 'warning' : 'success'}`}>{mean}{unit}</span>
          <span className="text-muted ms-2">ref &lt;{refVal}{unit}</span>
        </span>
      </div>
      <div className="progress" style={{ height: 12, position: 'relative' }}>
        <div
          className={`progress-bar bg-${over ? 'warning' : 'success'}`}
          style={{ width: `${pct}%` }}
        />
        <div
          style={{
            position: 'absolute', top: 0, bottom: 0, left: `${refPct}%`,
            width: 2, background: '#dc3545'
          }}
          title={`Reference upper limit: ${refVal}${unit}`}
        />
      </div>
      <div className="d-flex justify-content-end" style={{ fontSize: 10 }}>
        <span className="text-danger">▲ ref {refVal}{unit}</span>
      </div>
    </div>
  );
}

export default function BlinkReflexPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);
  const [sortKey, setSortKey] = useState('overall_severity');
  const [sortDir, setSortDir] = useState('asc');

  useEffect(() => {
    fetch(`${API}/api/blink-reflex/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/blink-reflex/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/blink-reflex/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  const tabs = ['overview', 'reflex-analysis', 'per-patient', 'definitions'];
  const tabLabel = { 'overview': 'Overview', 'reflex-analysis': 'Reflex Analysis', 'per-patient': 'Per Patient', 'definitions': 'Definitions' };

  const SEV_ORDER = { Normal: 0, Mild: 1, Moderate: 2, Severe: 3 };
  const sortedPatients = ov?.patient_summary ? [...ov.patient_summary].sort((a, b) => {
    let av = a[sortKey], bv = b[sortKey];
    if (sortKey === 'overall_severity') { av = SEV_ORDER[av] ?? 99; bv = SEV_ORDER[bv] ?? 99; }
    if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
    return sortDir === 'asc' ? (av ?? 0) - (bv ?? 0) : (bv ?? 0) - (av ?? 0);
  }) : [];

  const toggleSort = key => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('asc'); }
  };
  const sortIcon = key => sortKey === key ? (sortDir === 'asc' ? ' ▲' : ' ▼') : ' ⇅';

  if (!ov) return <div className="container py-5 text-center"><div className="spinner-border text-primary" /></div>;

  const kpis = ov.kpis || {};
  const maxSevCount = Math.max(...(ov.severity_distribution || []).map(x => x.count), 1);
  const maxPatCount = Math.max(...(ov.pattern_distribution || []).map(x => x.count), 1);
  const maxHistCount = Math.max(...(bd?.r1_latency_histogram || []).map(x => x.count), 1);

  return (
    <div className="container-fluid py-4">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-4">
        <span style={{ fontSize: 36 }}>👁️</span>
        <div>
          <h2 className="mb-0 fw-bold">Blink Reflex Dashboard</h2>
          <div className="text-muted small">
            Trigeminal-Facial Arc · Brainstem Pathway Assessment · AANEM Guidelines
          </div>
        </div>
        <div className="ms-auto">
          <span className="badge bg-primary fs-6">{kpis.total_studies} Studies</span>
          <span className={`badge ms-2 bg-${kpis.abnormal_rate_pct >= 40 ? 'danger' : kpis.abnormal_rate_pct >= 20 ? 'warning' : 'success'} fs-6`}>
            {kpis.abnormal_rate_pct}% Abnormal
          </span>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {tabs.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
            >
              {tabLabel[t]}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div>
          {/* KPIs */}
          <div className="row g-3 mb-4">
            <div className="col-6 col-md-3">
              <KPI label="Total Studies" value={kpis.total_studies} sub="bilateral testing" color="primary" />
            </div>
            <div className="col-6 col-md-3">
              <KPI label="Abnormal Studies" value={kpis.abnormal_count} sub={`${kpis.abnormal_rate_pct}% rate`}
                color={kpis.abnormal_rate_pct >= 40 ? 'danger' : kpis.abnormal_rate_pct >= 20 ? 'warning' : 'success'} />
            </div>
            <div className="col-6 col-md-3">
              <KPI label="Mean R1 Latency" value={`${kpis.mean_r1_latency_ms} ms`} sub="ref < 13 ms (pons)" color="info" />
            </div>
            <div className="col-6 col-md-3">
              <KPI label="Mean R2 Latency" value={`${kpis.mean_r2_latency_ms} ms`} sub="ref < 40 ms (medulla)" color="warning" />
            </div>
          </div>

          <div className="row g-4">
            {/* Severity Distribution */}
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold">Severity Distribution</div>
                <div className="card-body">
                  {(ov.severity_distribution || []).map(d => (
                    <Bar key={d.severity} label={d.severity} val={d.count} max={maxSevCount}
                      colorClass={SEV_COLOR(d.severity)} unit={` (${d.count})`} />
                  ))}
                </div>
              </div>
            </div>

            {/* Pattern Distribution */}
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold">Diagnostic Pattern Distribution</div>
                <div className="card-body">
                  {(ov.pattern_distribution || []).map(d => (
                    <Bar key={d.pattern} label={d.label} val={d.count} max={maxPatCount}
                      colorClass={PAT_COLOR(d.pattern)} unit={` (${d.count})`} />
                  ))}
                </div>
              </div>
            </div>

            {/* Side Abnormality Rates */}
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold">Side Abnormality Rates</div>
                <div className="card-body">
                  {(ov.side_abnormality_rates || []).map(s => (
                    <div key={s.side} className="mb-3">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="fw-semibold">{s.side} Side</span>
                        <span>
                          <span className={`badge bg-${s.rate_pct >= 30 ? 'warning' : 'success'}`}>{s.rate_pct}% abnormal</span>
                          <span className="text-muted ms-2 small">{s.abnormal}/{s.total}</span>
                        </span>
                      </div>
                      <div className="progress" style={{ height: 14 }}>
                        <div
                          className={`progress-bar bg-${s.rate_pct >= 30 ? 'warning' : 'success'}`}
                          style={{ width: `${s.rate_pct}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Brainstem Pathway Info */}
            <div className="col-md-6">
              <div className="card h-100 border-info">
                <div className="card-header fw-bold bg-info text-white">Brainstem Circuit</div>
                <div className="card-body">
                  <div className="d-flex align-items-start mb-3">
                    <span className="badge bg-primary me-2 mt-1">R1</span>
                    <div>
                      <div className="fw-semibold">Oligosynaptic Arc (Pons)</div>
                      <div className="text-muted small">Trigeminal V1 → Main sensory nucleus → Ipsilateral facial nucleus → Orbicularis oculi. Latency &lt;13 ms.</div>
                    </div>
                  </div>
                  <div className="d-flex align-items-start mb-3">
                    <span className="badge bg-warning text-dark me-2 mt-1">R2i</span>
                    <div>
                      <div className="fw-semibold">Polysynaptic Ipsilateral (Medulla)</div>
                      <div className="text-muted small">Spinal trigeminal nucleus → lateral reticular formation → ipsilateral facial nucleus. Latency &lt;40 ms.</div>
                    </div>
                  </div>
                  <div className="d-flex align-items-start">
                    <span className="badge bg-secondary me-2 mt-1">R2c</span>
                    <div>
                      <div className="fw-semibold">Polysynaptic Contralateral (Medulla)</div>
                      <div className="text-muted small">Crossed medullary arc → contralateral facial nucleus. Latency &lt;42 ms.</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Epilepsy Relevance */}
            <div className="col-12">
              <div className="card border-warning">
                <div className="card-header fw-bold bg-warning text-dark">Epilepsy Relevance</div>
                <div className="card-body">
                  <div className="row g-3">
                    {[
                      { icon: '⚡', title: 'AED Neurotoxicity', text: 'Carbamazepine and phenytoin can delay blink reflex latencies, reflecting trigeminal nerve dysfunction with chronic use.' },
                      { icon: '🧠', title: 'Brainstem Seizure Propagation', text: 'Abnormal R1/R2 patterns may indicate seizure spread to brainstem structures, critical for localisation.' },
                      { icon: '🔗', title: 'Ictal Brainstem Involvement', text: 'Prolonged R2 latency during postictal state may reflect temporary brainstem dysfunction after generalized seizures.' },
                      { icon: '📋', title: 'Pre-surgical Baseline', text: 'Blink reflex establishes brainstem integrity baseline before epilepsy surgery near posterior fossa structures.' },
                    ].map(r => (
                      <div key={r.title} className="col-md-3">
                        <div className="border rounded p-3 h-100">
                          <div className="fs-4 mb-1">{r.icon}</div>
                          <div className="fw-semibold small mb-1">{r.title}</div>
                          <div className="text-muted" style={{ fontSize: 12 }}>{r.text}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── REFLEX ANALYSIS TAB ── */}
      {tab === 'reflex-analysis' && bd && (
        <div>
          <div className="row g-4 mb-4">
            {/* Side Summary Cards */}
            {(bd.side_summary || []).map(s => (
              <div key={s.side} className="col-md-6">
                <div className={`card border-${s.abnormal_pct >= 20 ? 'warning' : 'success'}`}>
                  <div className={`card-header fw-bold d-flex justify-content-between bg-${s.abnormal_pct >= 20 ? 'warning' : 'success'} text-${s.abnormal_pct >= 20 ? 'dark' : 'white'}`}>
                    <span>{s.side} Side</span>
                    <span>{s.abnormal_pct}% abnormal</span>
                  </div>
                  <div className="card-body">
                    <LatencyBar label="R1 Latency (pons)" mean={s.mean_r1_latency_ms} ref={s.r1_latency_ref} />
                    <LatencyBar label="R2 Ipsilateral (medulla)" mean={s.mean_r2_ipsi_latency_ms} ref={s.r2_ipsi_latency_ref} />
                    <LatencyBar label="R2 Contralateral (crossed)" mean={s.mean_r2_contra_latency_ms} ref={s.r2_contra_latency_ref} />
                    <hr />
                    <div className="small">
                      <div className="d-flex justify-content-between">
                        <span>Mean R1 Amplitude</span>
                        <span className="fw-bold">{s.mean_r1_amplitude_mv} mV</span>
                      </div>
                      <div className="d-flex justify-content-between">
                        <span>Mean R2 Amplitude</span>
                        <span className="fw-bold">{s.mean_r2_amplitude_mv} mV</span>
                      </div>
                    </div>
                    <div className="mt-3">
                      <div className="small fw-semibold mb-1">Severity Breakdown</div>
                      <div className="d-flex gap-1 flex-wrap">
                        {Object.entries(s.severity_dist || {}).map(([sev, cnt]) => (
                          <span key={sev} className={`badge bg-${SEV_COLOR(sev)}`}>{sev}: {cnt}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* R1 Latency Histogram */}
          <div className="row g-4">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold">R1 Latency Distribution (ms)</div>
                <div className="card-body">
                  {(bd.r1_latency_histogram || []).map(h => (
                    <Bar key={h.range} label={h.range + ' ms'} val={h.count} max={maxHistCount}
                      colorClass={h.range === '13-15' || h.range === '15-18' || h.range === '18+' ? 'warning' : 'success'}
                      unit={` (${h.count})`} />
                  ))}
                  <div className="alert alert-info mt-3 mb-0 small py-2">
                    <strong>Reference:</strong> R1 latency &lt;13 ms. Values ≥13 ms suggest pontine arc dysfunction or trigeminal neuropathy.
                  </div>
                </div>
              </div>
            </div>

            {/* Ipsi vs Contra R2 */}
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold">R2 Component Comparison</div>
                <div className="card-body">
                  {(bd.ipsi_vs_contra_r2 || []).map(c => (
                    <div key={c.component} className="mb-4">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="fw-semibold">{c.component}</span>
                        <span className="badge bg-secondary">{c.abnormal_count}/{c.total} abnormal ({c.abnormal_pct}%)</span>
                      </div>
                      <LatencyBar label="Mean latency" mean={c.mean_latency_ms} ref={c.ref_upper_ms} />
                    </div>
                  ))}
                  <div className="table-responsive mt-2">
                    <table className="table table-sm small">
                      <thead><tr>
                        <th>Component</th><th>Mean (ms)</th><th>Ref (ms)</th><th>Abnormal %</th>
                      </tr></thead>
                      <tbody>
                        {(bd.ipsi_vs_contra_r2 || []).map(c => (
                          <tr key={c.component}>
                            <td>{c.component}</td>
                            <td>{c.mean_latency_ms}</td>
                            <td>&lt;{c.ref_upper_ms}</td>
                            <td><span className={`badge bg-${c.abnormal_pct >= 20 ? 'warning' : 'success'}`}>{c.abnormal_pct}%</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── PER PATIENT TAB ── */}
      {tab === 'per-patient' && (
        <div>
          <div className="card">
            <div className="card-header">
              <div className="d-flex align-items-center gap-3">
                <span className="fw-bold">Patient Blink Reflex Results</span>
                <span className="text-muted small">({sortedPatients.length} patients)</span>
              </div>
            </div>
            <div className="table-responsive">
              <table className="table table-hover mb-0 small">
                <thead className="table-dark">
                  <tr>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('patient_id')}>Patient{sortIcon('patient_id')}</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('age')}>Age{sortIcon('age')}</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('overall_severity')}>Severity{sortIcon('overall_severity')}</th>
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('diagnostic_pattern')}>Pattern{sortIcon('diagnostic_pattern')}</th>
                    <th>Abnormal Sides</th>
                    <th>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedPatients.map(pt => {
                    const expanded = expandedPt === pt.patient_id;
                    const bdPt = (bd?.patient_details || []).find(p => p.patient_id === pt.patient_id);
                    return (
                      <>
                        <tr key={pt.patient_id} className={expanded ? 'table-light' : ''}>
                          <td>
                            <div className="fw-semibold">{pt.patient_id}</div>
                            {pt.name && <div className="text-muted" style={{ fontSize: 11 }}>{pt.name}</div>}
                          </td>
                          <td>{pt.age}y</td>
                          <td><span className={`badge bg-${SEV_COLOR(pt.overall_severity)}`}>{pt.overall_severity}</span></td>
                          <td>
                            <span className={`badge bg-${PAT_COLOR(pt.diagnostic_pattern)}`}>
                              {(pt.diagnostic_pattern || '').replace(/_/g, ' ')}
                            </span>
                          </td>
                          <td>
                            <span className={`badge bg-${pt.abnormal_sides > 0 ? 'warning' : 'success'}`}>
                              {pt.abnormal_sides}/{pt.total_sides}
                            </span>
                          </td>
                          <td>
                            <button
                              className="btn btn-outline-primary btn-sm py-0 px-2"
                              onClick={() => setExpandedPt(expanded ? null : pt.patient_id)}
                            >
                              {expanded ? '▲ Hide' : '▼ Show'}
                            </button>
                          </td>
                        </tr>
                        {expanded && bdPt && (
                          <tr key={pt.patient_id + '-detail'}>
                            <td colSpan={6} className="bg-light p-3">
                              <div className="row g-3">
                                {['left', 'right'].map(side => {
                                  const sd = bdPt[side];
                                  if (!sd) return null;
                                  return (
                                    <div key={side} className="col-md-6">
                                      <div className={`border rounded p-3 border-${SEV_COLOR(sd.severity)}`}>
                                        <div className="d-flex justify-content-between mb-2">
                                          <span className="fw-bold text-capitalize">{side} Side</span>
                                          <span className={`badge bg-${SEV_COLOR(sd.severity)}`}>{sd.severity}</span>
                                        </div>
                                        <table className="table table-sm small mb-0">
                                          <tbody>
                                            <tr>
                                              <td>R1 Latency</td>
                                              <td className={`fw-bold text-${sd.r1_latency_ms > 13 ? 'warning' : 'success'}`}>{sd.r1_latency_ms} ms</td>
                                              <td className="text-muted">ref &lt;13 ms</td>
                                            </tr>
                                            <tr>
                                              <td>R2 Ipsi Latency</td>
                                              <td className={`fw-bold text-${sd.r2_ipsi_latency_ms > 40 ? 'warning' : 'success'}`}>{sd.r2_ipsi_latency_ms} ms</td>
                                              <td className="text-muted">ref &lt;40 ms</td>
                                            </tr>
                                            <tr>
                                              <td>R2 Contra Latency</td>
                                              <td className={`fw-bold text-${sd.r2_contra_latency_ms > 42 ? 'warning' : 'success'}`}>{sd.r2_contra_latency_ms} ms</td>
                                              <td className="text-muted">ref &lt;42 ms</td>
                                            </tr>
                                            <tr>
                                              <td>R1 Amplitude</td>
                                              <td className="fw-bold">{sd.r1_amplitude_mv} mV</td>
                                              <td className="text-muted">ref &gt;0.1 mV</td>
                                            </tr>
                                            <tr>
                                              <td>R2 Amplitude</td>
                                              <td className="fw-bold">{sd.r2_amplitude_mv} mV</td>
                                              <td className="text-muted">ref &gt;0.05 mV</td>
                                            </tr>
                                          </tbody>
                                        </table>
                                        {sd.interpretation && (
                                          <div className="mt-2 text-muted small fst-italic">{sd.interpretation}</div>
                                        )}
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            </td>
                          </tr>
                        )}
                      </>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-4">
          {/* Protocol */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">Test Protocol</div>
              <div className="card-body small">
                <p>{defs.protocol?.description}</p>
                <table className="table table-sm">
                  <tbody>
                    <tr><td className="fw-semibold">Stimulus Site</td><td>{defs.protocol?.stimulus_site}</td></tr>
                    <tr><td className="fw-semibold">Recording Site</td><td>{defs.protocol?.recording_site}</td></tr>
                    <tr><td className="fw-semibold">Standard</td><td>{defs.protocol?.standard}</td></tr>
                  </tbody>
                </table>
                <div className="fw-semibold mb-1">Responses per Side:</div>
                <ul className="mb-0">
                  {(defs.protocol?.responses_per_side || []).map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* Parameters */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">Parameters & Reference Values</div>
              <div className="card-body small">
                <table className="table table-sm">
                  <thead><tr><th>Parameter</th><th>Unit</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.parameters || []).map(p => (
                      <tr key={p.name}>
                        <td className="fw-semibold text-nowrap">{p.name}</td>
                        <td className="text-nowrap">{p.unit}</td>
                        <td>{p.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Reference Ranges */}
          {defs.reference_ranges && (
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold">Reference Ranges</div>
                <div className="card-body small">
                  <table className="table table-sm">
                    <thead><tr><th>Parameter</th><th>Upper Limit</th><th>Notes</th></tr></thead>
                    <tbody>
                      {Object.entries(defs.reference_ranges).map(([k, v]) => (
                        <tr key={k}>
                          <td className="fw-semibold">{k.replace(/_/g, ' ')}</td>
                          <td>{typeof v === 'object' ? JSON.stringify(v) : v}</td>
                          <td className="text-muted">AANEM</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Diagnostic Patterns */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">Diagnostic Patterns</div>
              <div className="card-body small">
                {(defs.diagnostic_patterns || []).map(p => (
                  <div key={p.pattern} className="mb-3 border-bottom pb-2">
                    <div className="d-flex align-items-center gap-2 mb-1">
                      <span className={`badge bg-${PAT_COLOR(p.pattern)}`}>{p.label}</span>
                    </div>
                    <div className="text-muted">{p.description}</div>
                    {p.r1_finding && (
                      <div className="small mt-1">
                        <span className="text-muted">R1: </span>{p.r1_finding}
                        {p.r2_finding && <> · <span className="text-muted">R2: </span>{p.r2_finding}</>}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Indications */}
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-bold">Clinical Indications</div>
              <div className="card-body">
                <div className="row">
                  <div className="col-md-6">
                    <ul className="mb-0 small">
                      {(defs.protocol?.indications || []).slice(0, 4).map((ind, i) => (
                        <li key={i}>{ind}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="col-md-6">
                    <ul className="mb-0 small">
                      {(defs.protocol?.indications || []).slice(4).map((ind, i) => (
                        <li key={i}>{ind}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
