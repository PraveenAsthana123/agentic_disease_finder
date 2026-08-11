'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SEV_COLOR = s =>
  s === 'Normal' ? 'success' : s === 'Mild' ? 'info' : s === 'Moderate' ? 'warning' : s === 'Severe' ? 'danger' : 'secondary';

const PAT_COLOR = p => ({
  normal:                  'success',
  peripheral_neuropathy:   'warning',
  small_fiber_neuropathy:  'info',
  postganglionic_lesion:   'danger',
  preganglionic_lesion:    'primary',
  generalized_dysautonomia:'dark',
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

function LatencyBar({ label, mean, refVal, unit = ' s' }) {
  const scale = refVal * 1.5;
  const pct = Math.min(Math.round((mean / scale) * 100), 100);
  const refPct = Math.round((refVal / scale) * 100);
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
        <div className={`progress-bar bg-${over ? 'warning' : 'success'}`} style={{ width: `${pct}%` }} />
        <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${refPct}%`, width: 2, background: '#dc3545' }}
          title={`Reference upper limit: ${refVal}${unit}`} />
      </div>
      <div className="d-flex justify-content-end" style={{ fontSize: 10 }}>
        <span className="text-danger">▲ ref {refVal}{unit}</span>
      </div>
    </div>
  );
}

export default function SSRPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);
  const [sortKey, setSortKey] = useState('overall_severity');
  const [sortDir, setSortDir] = useState('asc');

  useEffect(() => {
    fetch(`${API}/api/ssr/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/ssr/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/ssr/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  const tabs = ['overview', 'site-analysis', 'per-patient', 'definitions'];
  const tabLabel = { overview: 'Overview', 'site-analysis': 'Site Analysis', 'per-patient': 'Per Patient', definitions: 'Definitions' };

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
  const maxHandHist = Math.max(...(bd?.hand_latency_histogram || []).map(x => x.count), 1);
  const maxFootHist = Math.max(...(bd?.foot_latency_histogram || []).map(x => x.count), 1);

  return (
    <div className="container-fluid py-4">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-4">
        <span style={{ fontSize: 36 }}>💦</span>
        <div>
          <h2 className="mb-0 fw-bold">SSR — Sympathetic Skin Response</h2>
          <div className="text-muted small">
            Sudomotor Pathway · Autonomic C-Fiber Integrity · AANEM Guidelines
          </div>
        </div>
        <div className="ms-auto">
          <span className="badge bg-primary fs-6">{kpis.total_studies} Studies</span>
          <span className={`badge ms-2 bg-${kpis.abnormal_rate_pct >= 30 ? 'danger' : kpis.abnormal_rate_pct >= 15 ? 'warning' : 'success'} fs-6`}>
            {kpis.abnormal_rate_pct}% Abnormal
          </span>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {tabs.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
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
              <KPI label="Total Studies" value={kpis.total_studies} sub="bilateral hand + foot" color="primary" />
            </div>
            <div className="col-6 col-md-3">
              <KPI label="Abnormal Studies" value={kpis.abnormal_count}
                sub={`${kpis.abnormal_rate_pct}% rate`}
                color={kpis.abnormal_rate_pct >= 30 ? 'danger' : kpis.abnormal_rate_pct >= 15 ? 'warning' : 'success'} />
            </div>
            <div className="col-6 col-md-3">
              <KPI label="Mean Hand Latency" value={`${kpis.mean_hand_latency_s} s`} sub="ref ≤1.50 s (palmar)" color="info" />
            </div>
            <div className="col-6 col-md-3">
              <KPI label="Mean Foot Latency" value={`${kpis.mean_foot_latency_s} s`} sub="ref ≤2.20 s (plantar)" color="warning" />
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

            {/* Diagnostic Patterns */}
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

            {/* Site Abnormality Rates */}
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold">Recording Site Abnormality Rates</div>
                <div className="card-body">
                  {(ov.site_abnormality_rates || []).map(s => (
                    <div key={s.site} className="mb-3">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="fw-semibold">{s.site} (Palmar / Plantar)</span>
                        <span>
                          <span className={`badge bg-${s.rate_pct >= 20 ? 'warning' : 'success'}`}>{s.rate_pct}% abnormal</span>
                          <span className="text-muted ms-2 small">{s.abnormal}/{s.total}</span>
                          {s.absent > 0 && <span className="badge bg-danger ms-1">{s.absent} absent</span>}
                        </span>
                      </div>
                      <div className="progress" style={{ height: 14 }}>
                        <div className={`progress-bar bg-${s.rate_pct >= 20 ? 'warning' : 'success'}`}
                          style={{ width: `${s.rate_pct}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Sympathetic Pathway Info */}
            <div className="col-md-6">
              <div className="card h-100 border-info">
                <div className="card-header fw-bold bg-info text-white">Sympathetic Sudomotor Arc</div>
                <div className="card-body small">
                  {[
                    { badge: 'Afferent', color: 'primary', text: 'Myelinated sensory fibers (median nerve) → spinal cord dorsal horn.' },
                    { badge: 'Central', color: 'secondary', text: 'Brainstem reticular formation → hypothalamus (thermoregulatory center).' },
                    { badge: 'Efferent', color: 'warning', text: 'IML column (T2-L2) → paravertebral ganglia → postganglionic C-fibers.' },
                    { badge: 'Effector', color: 'success', text: 'Eccrine sweat glands → skin conductance change (SSR waveform).' },
                  ].map(r => (
                    <div key={r.badge} className="d-flex align-items-start mb-3">
                      <span className={`badge bg-${r.color} me-2 mt-1`} style={{ minWidth: 64 }}>{r.badge}</span>
                      <div className="text-muted">{r.text}</div>
                    </div>
                  ))}
                  <div className="alert alert-secondary mb-0 py-2 small">
                    <strong>Dysautonomia Score:</strong> mean {kpis.mean_dysautonomia_score}/10 across cohort
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
                      { icon: '⚡', title: 'SUDEP Risk Screening', text: 'Autonomic dysfunction (absent foot SSR) correlates with SUDEP risk. SSR complements cardiac autonomic monitoring for high-risk stratification.' },
                      { icon: '💊', title: 'AED Autonomic Effects', text: 'Carbamazepine and oxcarbazepine can impair sympathetic transmission. Phenytoin and phenobarbital may reduce SSR amplitude with chronic use.' },
                      { icon: '🧠', title: 'Ictal Autonomic Changes', text: 'Seizures cause transient sympathetic surges — SSR baseline establishes the interictal resting autonomic state for comparison.' },
                      { icon: '🩺', title: 'Comorbid Neuropathy', text: 'Epilepsy patients with diabetic or chemotherapy-induced neuropathy may show length-dependent SSR changes (foot > hand abnormality).' },
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

      {/* ── SITE ANALYSIS TAB ── */}
      {tab === 'site-analysis' && bd && (
        <div>
          <div className="row g-4 mb-4">
            {/* Hand Summary */}
            <div className="col-md-6">
              <div className={`card border-${bd.hand_summary?.abnormal_pct >= 20 ? 'warning' : 'success'}`}>
                <div className={`card-header fw-bold d-flex justify-content-between bg-${bd.hand_summary?.abnormal_pct >= 20 ? 'warning' : 'success'} text-${bd.hand_summary?.abnormal_pct >= 20 ? 'dark' : 'white'}`}>
                  <span>Hand (Palmar) — Sudomotor</span>
                  <span>{bd.hand_summary?.abnormal_pct}% abnormal</span>
                </div>
                <div className="card-body">
                  <LatencyBar label="Mean Onset Latency" mean={bd.hand_summary?.mean_latency_s} refVal={bd.hand_summary?.refs?.latency_upper_s} />
                  <div className="row g-2 mb-3">
                    <div className="col-6 text-center border-end">
                      <div className="fw-bold text-info">{bd.hand_summary?.mean_amplitude_mv} mV</div>
                      <div className="text-muted small">Mean Amplitude (ref ≥{bd.hand_summary?.refs?.amplitude_lower_mv} mV)</div>
                    </div>
                    <div className="col-6 text-center">
                      <div className="fw-bold text-secondary">{bd.hand_summary?.mean_habituation_pct}%</div>
                      <div className="text-muted small">Mean Habituation (ref &lt;{bd.hand_summary?.habituation_ref_pct}%)</div>
                    </div>
                  </div>
                  <div className="d-flex gap-1 flex-wrap">
                    {Object.entries(bd.hand_summary?.severity_dist || {}).map(([sev, cnt]) => (
                      <span key={sev} className={`badge bg-${SEV_COLOR(sev)}`}>{sev}: {cnt}</span>
                    ))}
                    {bd.hand_summary?.absent_count > 0 &&
                      <span className="badge bg-dark">Absent: {bd.hand_summary.absent_count}</span>}
                  </div>
                </div>
              </div>
            </div>

            {/* Foot Summary */}
            <div className="col-md-6">
              <div className={`card border-${bd.foot_summary?.abnormal_pct >= 15 ? 'warning' : 'success'}`}>
                <div className={`card-header fw-bold d-flex justify-content-between bg-${bd.foot_summary?.abnormal_pct >= 15 ? 'warning' : 'success'} text-${bd.foot_summary?.abnormal_pct >= 15 ? 'dark' : 'white'}`}>
                  <span>Foot (Plantar) — Sudomotor</span>
                  <span>{bd.foot_summary?.abnormal_pct}% abnormal</span>
                </div>
                <div className="card-body">
                  <LatencyBar label="Mean Onset Latency" mean={bd.foot_summary?.mean_latency_s} refVal={bd.foot_summary?.refs?.latency_upper_s} />
                  <div className="row g-2 mb-3">
                    <div className="col-6 text-center border-end">
                      <div className="fw-bold text-info">{bd.foot_summary?.mean_amplitude_mv} mV</div>
                      <div className="text-muted small">Mean Amplitude (ref ≥{bd.foot_summary?.refs?.amplitude_lower_mv} mV)</div>
                    </div>
                    <div className="col-6 text-center">
                      <div className="fw-bold text-secondary">{bd.foot_summary?.mean_habituation_pct}%</div>
                      <div className="text-muted small">Mean Habituation (ref &lt;{bd.foot_summary?.habituation_ref_pct}%)</div>
                    </div>
                  </div>
                  <div className="d-flex gap-1 flex-wrap">
                    {Object.entries(bd.foot_summary?.severity_dist || {}).map(([sev, cnt]) => (
                      <span key={sev} className={`badge bg-${SEV_COLOR(sev)}`}>{sev}: {cnt}</span>
                    ))}
                    {bd.foot_summary?.absent_count > 0 &&
                      <span className="badge bg-dark">Absent: {bd.foot_summary.absent_count}</span>}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Latency Histograms */}
          <div className="row g-4">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold">Hand Latency Distribution (s)</div>
                <div className="card-body">
                  {(bd.hand_latency_histogram || []).map(h => (
                    <Bar key={h.range} label={h.range + ' s'} val={h.count} max={maxHandHist}
                      colorClass={h.abnormal ? 'warning' : 'success'} unit={` (${h.count})`} />
                  ))}
                  <div className="alert alert-info mt-3 mb-0 small py-2">
                    <strong>Reference:</strong> Hand latency ≤1.50 s. Values &gt;1.50 s indicate sudomotor pathway delay.
                  </div>
                </div>
              </div>
            </div>

            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold">Foot Latency Distribution (s)</div>
                <div className="card-body">
                  {(bd.foot_latency_histogram || []).map(h => (
                    <Bar key={h.range} label={h.range + ' s'} val={h.count} max={maxFootHist}
                      colorClass={h.abnormal ? 'warning' : 'success'} unit={` (${h.count})`} />
                  ))}
                  <div className="alert alert-info mt-3 mb-0 small py-2">
                    <strong>Reference:</strong> Foot latency ≤2.20 s. Length-dependent neuropathies show foot-predominant abnormality.
                  </div>
                </div>
              </div>
            </div>

            {/* Hand vs Foot Comparison Table */}
            <div className="col-12">
              <div className="card">
                <div className="card-header fw-bold">Hand vs Foot Comparison</div>
                <div className="table-responsive">
                  <table className="table table-sm small mb-0">
                    <thead className="table-dark">
                      <tr>
                        <th>Parameter</th>
                        <th>Hand (Palmar)</th>
                        <th>Foot (Plantar)</th>
                        <th>Reference (Hand)</th>
                        <th>Reference (Foot)</th>
                        <th>Clinical Significance</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="fw-semibold">Onset Latency</td>
                        <td className={bd.hand_summary?.mean_latency_s > 1.5 ? 'text-warning fw-bold' : 'text-success fw-bold'}>{bd.hand_summary?.mean_latency_s} s</td>
                        <td className={bd.foot_summary?.mean_latency_s > 2.2 ? 'text-warning fw-bold' : 'text-success fw-bold'}>{bd.foot_summary?.mean_latency_s} s</td>
                        <td className="text-muted">≤1.50 s</td>
                        <td className="text-muted">≤2.20 s</td>
                        <td className="text-muted">Prolongation → conduction delay</td>
                      </tr>
                      <tr>
                        <td className="fw-semibold">Amplitude</td>
                        <td className="fw-bold">{bd.hand_summary?.mean_amplitude_mv} mV</td>
                        <td className="fw-bold">{bd.foot_summary?.mean_amplitude_mv} mV</td>
                        <td className="text-muted">≥0.50 mV</td>
                        <td className="text-muted">≥0.20 mV</td>
                        <td className="text-muted">Reduction → sweat gland loss</td>
                      </tr>
                      <tr>
                        <td className="fw-semibold">Habituation</td>
                        <td className="fw-bold">{bd.hand_summary?.mean_habituation_pct}%</td>
                        <td className="fw-bold">{bd.foot_summary?.mean_habituation_pct}%</td>
                        <td className="text-muted">&lt;50%</td>
                        <td className="text-muted">&lt;50%</td>
                        <td className="text-muted">≥50% → impaired autonomic reserve</td>
                      </tr>
                      <tr>
                        <td className="fw-semibold">Abnormal Rate</td>
                        <td className={`fw-bold text-${bd.hand_summary?.abnormal_pct >= 20 ? 'warning' : 'success'}`}>{bd.hand_summary?.abnormal_pct}%</td>
                        <td className={`fw-bold text-${bd.foot_summary?.abnormal_pct >= 15 ? 'warning' : 'success'}`}>{bd.foot_summary?.abnormal_pct}%</td>
                        <td className="text-muted">—</td>
                        <td className="text-muted">—</td>
                        <td className="text-muted">Foot predominance → length-dependent</td>
                      </tr>
                    </tbody>
                  </table>
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
                <span className="fw-bold">Patient SSR Results</span>
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
                    <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('dysautonomia_score')}>Dysautonomia{sortIcon('dysautonomia_score')}</th>
                    <th>Sites</th>
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
                            <span className="fw-bold">{pt.dysautonomia_score?.toFixed(1)}</span>
                            <span className="text-muted">/10</span>
                          </td>
                          <td>
                            <span className={`badge bg-${pt.abnormal_sites > 0 ? 'warning' : 'success'}`}>
                              {pt.abnormal_sites}/{pt.total_sites}
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
                            <td colSpan={7} className="bg-light p-3">
                              <div className="row g-3">
                                {['hand', 'foot'].map(site => {
                                  const sd = bdPt[site];
                                  if (!sd) return null;
                                  return (
                                    <div key={site} className="col-md-6">
                                      <div className={`border rounded p-3 border-${SEV_COLOR(sd.severity)}`}>
                                        <div className="d-flex justify-content-between mb-2">
                                          <span className="fw-bold text-capitalize">{sd.site} ({sd.recording})</span>
                                          <div>
                                            <span className={`badge bg-${SEV_COLOR(sd.severity)}`}>{sd.severity}</span>
                                            {sd.absent && <span className="badge bg-dark ms-1">Absent</span>}
                                          </div>
                                        </div>
                                        <table className="table table-sm small mb-0">
                                          <tbody>
                                            <tr>
                                              <td>Onset Latency</td>
                                              <td className={`fw-bold text-${sd.latency_abnormal ? 'warning' : 'success'}`}>
                                                {sd.onset_latency_s} s
                                              </td>
                                              <td className="text-muted">ref &lt;{sd.latency_ref_s} s</td>
                                            </tr>
                                            <tr>
                                              <td>Amplitude</td>
                                              <td className={`fw-bold text-${sd.amplitude_abnormal ? 'warning' : 'success'}`}>
                                                {sd.amplitude_mv} mV
                                              </td>
                                              <td className="text-muted">ref ≥{sd.amplitude_ref_mv} mV</td>
                                            </tr>
                                            <tr>
                                              <td>Habituation</td>
                                              <td className={`fw-bold text-${sd.habituation_abnormal ? 'warning' : 'success'}`}>
                                                {sd.habituation_pct?.toFixed(1)}%
                                              </td>
                                              <td className="text-muted">ref &lt;{sd.habituation_ref_pct}%</td>
                                            </tr>
                                          </tbody>
                                        </table>
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
                    {defs.protocol?.stimulus && Object.entries(defs.protocol.stimulus).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-semibold text-nowrap text-capitalize">{k.replace(/_/g, ' ')}</td>
                        <td>{String(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
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
                    {p.hand_finding && (
                      <div className="small mt-1">
                        <span className="text-muted">Hand: </span>{p.hand_finding}
                        {p.foot_finding && <> · <span className="text-muted">Foot: </span>{p.foot_finding}</>}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Indications & References */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">Clinical Indications</div>
              <div className="card-body small">
                <ul className="mb-3">
                  {(defs.protocol?.indications || []).map((ind, i) => (
                    <li key={i} className="mb-1">{ind}</li>
                  ))}
                </ul>
                {defs.references && (
                  <>
                    <div className="fw-semibold mb-1">References</div>
                    <ul className="text-muted">
                      {(Array.isArray(defs.references) ? defs.references : [defs.references]).map((ref, i) => (
                        <li key={i}>{ref}</li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
