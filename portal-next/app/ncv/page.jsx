'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SEV_COLOR = s =>
  s === 'Normal' ? 'success' : s === 'Mild' ? 'info' : s === 'Moderate' ? 'warning' : 'danger';

const TYPE_COLOR = t => ({
  normal:       'success',
  demyelinating:'warning',
  axonal:       'info',
  mixed:        'secondary',
}[t] || 'secondary');

const LIMB_ICON = l => (l || '').toLowerCase() === 'upper' ? '🖐️' : '🦶';

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

export default function NCVPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);
  const [sortKey, setSortKey] = useState('overall_severity');
  const [sortDir, setSortDir] = useState('asc');

  useEffect(() => {
    fetch(`${API}/api/ncv/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/ncv/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/ncv/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  const tabs = ['overview', 'nerve-analysis', 'per-patient', 'definitions'];
  const tabLabel = { 'overview': 'Overview', 'nerve-analysis': 'Nerve Analysis', 'per-patient': 'Per Patient', 'definitions': 'Definitions' };

  /* ── SORT helper ── */
  const SEV_ORDER = { Normal: 0, Mild: 1, Moderate: 2, Severe: 3 };
  const sortedPatients = bd?.patient_details ? [...bd.patient_details].sort((a, b) => {
    let av = a[sortKey], bv = b[sortKey];
    if (sortKey === 'overall_severity') { av = SEV_ORDER[av] ?? 99; bv = SEV_ORDER[bv] ?? 99; }
    if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
    return sortDir === 'asc' ? av - bv : bv - av;
  }) : [];

  const toggleSort = key => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('asc'); }
  };
  const sortIcon = key => sortKey === key ? (sortDir === 'asc' ? ' ▲' : ' ▼') : ' ⇅';

  if (!ov) return <div className="container py-5 text-center"><div className="spinner-border text-primary"/></div>;

  const kpis = ov.kpis || {};
  const maxSevCount = Math.max(...(ov.severity_distribution || []).map(x => x.count), 1);
  const maxNerveRate = Math.max(...(ov.nerve_abnormality_rates || []).map(x => x.rate_pct), 1);

  return (
    <div className="container-fluid py-3">
      <h2 className="mb-1">⚡ NCV — Nerve Conduction Velocity Dashboard</h2>
      <p className="text-muted mb-3 small">
        Electrodiagnostic assessment of peripheral nerve function · AANEM guidelines ·
        AED-induced neuropathy monitoring · {kpis.total_studies} studies
      </p>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>
              {tabLabel[t]}
            </button>
          </li>
        ))}
      </ul>

      {/* ══════════ OVERVIEW ══════════ */}
      {tab === 'overview' && (
        <div>
          {/* KPIs */}
          <div className="row g-3 mb-4">
            <div className="col-6 col-md-2">
              <KPI label="Total Studies" value={kpis.total_studies} color="primary"/>
            </div>
            <div className="col-6 col-md-2">
              <KPI label="Abnormal Studies" value={kpis.abnormal_count} sub={`${kpis.abnormal_rate_pct}% rate`} color="danger"/>
            </div>
            <div className="col-6 col-md-2">
              <KPI label="Mean MCV" value={`${kpis.mean_mcv} m/s`} sub="Motor conduction" color="info"/>
            </div>
            <div className="col-6 col-md-2">
              <KPI label="Mean SCV" value={`${kpis.mean_scv} m/s`} sub="Sensory conduction" color="success"/>
            </div>
            <div className="col-6 col-md-2">
              <KPI label="Nerves / Study" value={kpis.nerves_tested_per_study} sub="8 motor + sensory" color="secondary"/>
            </div>
          </div>

          <div className="row g-3">
            {/* Severity distribution */}
            <div className="col-md-5">
              <div className="card h-100">
                <div className="card-header fw-semibold">Overall Severity Distribution</div>
                <div className="card-body">
                  {(ov.severity_distribution || []).map(s => (
                    <div key={s.severity} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className={`badge bg-${SEV_COLOR(s.severity)}`}>{s.severity}</span>
                        <span className="fw-bold">{s.count} studies</span>
                      </div>
                      <div className="progress" style={{ height: 14 }}>
                        <div className={`progress-bar bg-${SEV_COLOR(s.severity)}`}
                             style={{ width: `${Math.round((s.count / maxSevCount) * 100)}%` }}>
                          {s.count > 0 && `${Math.round((s.count / kpis.total_studies) * 100)}%`}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Neuropathy types */}
            <div className="col-md-3">
              <div className="card h-100">
                <div className="card-header fw-semibold">Neuropathy Pattern</div>
                <div className="card-body">
                  {(ov.neuropathy_type_distribution || []).map(n => (
                    <div key={n.type} className="d-flex justify-content-between align-items-center mb-2">
                      <span className={`badge bg-${TYPE_COLOR(n.type)}`}>{n.label}</span>
                      <span className="fw-bold">{n.count}</span>
                    </div>
                  ))}
                  <hr className="my-2"/>
                  <div className="small text-muted">
                    Axonal: amplitude loss · Demyelinating: velocity slowing · Mixed: both
                  </div>
                </div>
              </div>
            </div>

            {/* Per-nerve abnormality rates */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Abnormality Rate by Nerve</div>
                <div className="card-body">
                  {(ov.nerve_abnormality_rates || []).map(n => (
                    <Bar key={n.nerve} label={n.nerve}
                         val={`${n.rate_pct}%`}
                         max={100}
                         colorClass={n.rate_pct >= 15 ? 'danger' : n.rate_pct >= 8 ? 'warning' : 'success'}/>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Patient severity table summary */}
          <div className="card mt-3">
            <div className="card-header fw-semibold">Patient Overview (Sorted by Severity)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Patient</th>
                      <th>Age</th>
                      <th>Severity</th>
                      <th>Neuropathy Type</th>
                      <th>Abnormal Nerves</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.patient_summary || []).map(p => (
                      <tr key={p.patient_id}>
                        <td className="fw-semibold">{p.name || p.patient_id}</td>
                        <td>{p.age}</td>
                        <td><span className={`badge bg-${SEV_COLOR(p.overall_severity)}`}>{p.overall_severity}</span></td>
                        <td><span className={`badge bg-${TYPE_COLOR(p.neuropathy_type)}`}>{p.neuropathy_type}</span></td>
                        <td>{p.abnormal_nerves}/{p.total_nerves}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ══════════ NERVE ANALYSIS ══════════ */}
      {tab === 'nerve-analysis' && bd && (
        <div>
          {/* Motor nerves */}
          <div className="card mb-3">
            <div className="card-header fw-semibold">🖐️🦶 Motor Nerve Summary (DML / CMAP / MCV / F-wave)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Nerve</th>
                      <th>Limb</th>
                      <th>Mean DML (ms)</th>
                      <th>Ref ≤</th>
                      <th>Mean CMAP (mV)</th>
                      <th>Ref ≥</th>
                      <th>Mean MCV (m/s)</th>
                      <th>Ref ≥</th>
                      <th>Mean F-wave (ms)</th>
                      <th>Ref ≤</th>
                      <th>Abnormal %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.motor_summary || []).map(n => (
                      <tr key={n.nerve}>
                        <td className="fw-semibold">{n.nerve}</td>
                        <td>{LIMB_ICON(n.limb)} {n.limb}</td>
                        <td className={n.mean_dml > n.dml_ref ? 'text-danger fw-bold' : ''}>{n.mean_dml}</td>
                        <td className="text-muted small">{n.dml_ref}</td>
                        <td className={n.mean_cmap < n.cmap_ref ? 'text-danger fw-bold' : ''}>{n.mean_cmap}</td>
                        <td className="text-muted small">{n.cmap_ref}</td>
                        <td className={n.mean_mcv < n.mcv_ref ? 'text-danger fw-bold' : ''}>{n.mean_mcv}</td>
                        <td className="text-muted small">{n.mcv_ref}</td>
                        <td className={n.mean_fwave > n.fwave_ref ? 'text-danger fw-bold' : ''}>{n.mean_fwave}</td>
                        <td className="text-muted small">{n.fwave_ref}</td>
                        <td>
                          <span className={`badge bg-${n.abnormal_pct >= 15 ? 'danger' : n.abnormal_pct >= 8 ? 'warning' : 'success'}`}>
                            {n.abnormal_pct}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Sensory nerves */}
          <div className="card mb-3">
            <div className="card-header fw-semibold">🖐️🦶 Sensory Nerve Summary (SNAP / SCV)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Nerve</th>
                      <th>Limb</th>
                      <th>Mean SNAP (µV)</th>
                      <th>Ref ≥</th>
                      <th>Mean SCV (m/s)</th>
                      <th>Ref ≥</th>
                      <th>Abnormal %</th>
                      <th>Severity Distribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.sensory_summary || []).map(n => (
                      <tr key={n.nerve}>
                        <td className="fw-semibold">{n.nerve}</td>
                        <td>{LIMB_ICON(n.limb)} {n.limb}</td>
                        <td className={n.mean_snap < n.snap_ref ? 'text-danger fw-bold' : ''}>{n.mean_snap}</td>
                        <td className="text-muted small">{n.snap_ref}</td>
                        <td className={n.mean_scv < n.scv_ref ? 'text-danger fw-bold' : ''}>{n.mean_scv}</td>
                        <td className="text-muted small">{n.scv_ref}</td>
                        <td>
                          <span className={`badge bg-${n.abnormal_pct >= 15 ? 'danger' : n.abnormal_pct >= 8 ? 'warning' : 'success'}`}>
                            {n.abnormal_pct}%
                          </span>
                        </td>
                        <td className="small">
                          {Object.entries(n.severity_dist || {}).map(([sev, cnt]) => (
                            <span key={sev} className={`badge bg-${SEV_COLOR(sev)} me-1`}>{sev}: {cnt}</span>
                          ))}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* MCV histogram + Motor vs Sensory + Limb comparison */}
          <div className="row g-3">
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">MCV Distribution (m/s)</div>
                <div className="card-body">
                  {(bd.mcv_histogram || []).map(b => (
                    <Bar key={b.range} label={b.range + ' m/s'} val={b.count}
                         max={Math.max(...bd.mcv_histogram.map(x => x.count), 1)}
                         colorClass={b.range === '<30' ? 'danger' : b.range === '30-39' ? 'warning' : b.range === '40-49' ? 'info' : 'success'}/>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Motor vs Sensory Involvement</div>
                <div className="card-body">
                  {(bd.motor_vs_sensory || []).map(m => (
                    <div key={m.type} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold">{m.type} nerves</span>
                        <span>{m.abnormal} / {m.abnormal + m.normal} abnormal</span>
                      </div>
                      <div className="progress" style={{ height: 18 }}>
                        <div className={`progress-bar bg-${m.abnormal_pct >= 15 ? 'danger' : 'warning'}`}
                             style={{ width: `${m.abnormal_pct}%` }}>
                          {m.abnormal_pct}%
                        </div>
                        <div className="progress-bar bg-success"
                             style={{ width: `${100 - m.abnormal_pct}%` }}>
                          {(100 - m.abnormal_pct).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Upper vs Lower Limb</div>
                <div className="card-body">
                  {(bd.limb_comparison || []).map(l => (
                    <div key={l.limb} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold">{LIMB_ICON(l.limb)} {l.limb} limb</span>
                        <span>{l.abnormal}/{l.abnormal + l.normal} abnormal ({l.abnormal_pct}%)</span>
                      </div>
                      <div className="progress" style={{ height: 18 }}>
                        <div className={`progress-bar bg-${l.abnormal_pct >= 15 ? 'danger' : 'info'}`}
                             style={{ width: `${l.abnormal_pct}%` }}>
                          {l.abnormal_pct}%
                        </div>
                        <div className="progress-bar bg-success"
                             style={{ width: `${100 - l.abnormal_pct}%` }}/>
                      </div>
                    </div>
                  ))}
                  <div className="small text-muted mt-2">
                    Equal upper/lower involvement suggests length-dependent neuropathy
                    or systemic AED effect.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ══════════ PER PATIENT ══════════ */}
      {tab === 'per-patient' && bd && (
        <div>
          <div className="card">
            <div className="card-header fw-semibold">
              Per-Patient NCV Results — click row to expand nerve detail
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('name')}>
                        Patient{sortIcon('name')}
                      </th>
                      <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('age')}>
                        Age{sortIcon('age')}
                      </th>
                      <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('overall_severity')}>
                        Severity{sortIcon('overall_severity')}
                      </th>
                      <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('neuropathy_type')}>
                        Neuropathy Type{sortIcon('neuropathy_type')}
                      </th>
                      <th>Abnormal Nerves</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedPatients.map(p => (
                      <>
                        <tr key={p.patient_id}
                            style={{ cursor: 'pointer' }}
                            onClick={() => setExpandedPt(expandedPt === p.patient_id ? null : p.patient_id)}>
                          <td className="fw-semibold">{p.name || p.patient_id}</td>
                          <td>{p.age}</td>
                          <td><span className={`badge bg-${SEV_COLOR(p.overall_severity)}`}>{p.overall_severity}</span></td>
                          <td><span className={`badge bg-${TYPE_COLOR(p.neuropathy_type)}`}>{p.neuropathy_type}</span></td>
                          <td>{(p.motor || []).filter(n => n.abnormal).length + (p.sensory || []).filter(n => n.abnormal).length} / {(p.motor?.length || 0) + (p.sensory?.length || 0)}</td>
                          <td className="text-muted">{expandedPt === p.patient_id ? '▲' : '▼'}</td>
                        </tr>
                        {expandedPt === p.patient_id && (
                          <tr key={`${p.patient_id}-detail`}>
                            <td colSpan={6} className="bg-light">
                              <div className="row g-2 p-2">
                                <div className="col-md-6">
                                  <div className="small fw-bold mb-1">Motor Nerves</div>
                                  <table className="table table-xs table-bordered table-sm mb-0">
                                    <thead className="table-secondary">
                                      <tr>
                                        <th>Nerve</th><th>DML</th><th>CMAP</th><th>MCV</th><th>F-wave</th><th>Status</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {(p.motor || []).map(n => (
                                        <tr key={n.nerve} className={n.abnormal ? 'table-danger' : ''}>
                                          <td className="small">{n.nerve}</td>
                                          <td className="small">{n.dml_ms}</td>
                                          <td className="small">{n.cmap_mv}</td>
                                          <td className="small">{n.mcv_ms}</td>
                                          <td className="small">{n.fwave_ms}</td>
                                          <td><span className={`badge bg-${n.abnormal ? SEV_COLOR(n.severity) : 'success'}`}>{n.severity || 'Normal'}</span></td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                                <div className="col-md-6">
                                  <div className="small fw-bold mb-1">Sensory Nerves</div>
                                  <table className="table table-xs table-bordered table-sm mb-0">
                                    <thead className="table-secondary">
                                      <tr>
                                        <th>Nerve</th><th>SNAP (µV)</th><th>SCV (m/s)</th><th>Status</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {(p.sensory || []).map(n => (
                                        <tr key={n.nerve} className={n.abnormal ? 'table-warning' : ''}>
                                          <td className="small">{n.nerve}</td>
                                          <td className="small">{n.snap_uv}</td>
                                          <td className="small">{n.scv_ms}</td>
                                          <td><span className={`badge bg-${n.abnormal ? SEV_COLOR(n.severity) : 'success'}`}>{n.severity || 'Normal'}</span></td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
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
        </div>
      )}

      {/* ══════════ DEFINITIONS ══════════ */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">Protocol</div>
              <div className="card-body">
                <p className="small">{defs.protocol?.description}</p>
                <p className="small mb-1"><strong>Standard:</strong> {defs.protocol?.standard}</p>
                <div className="mb-2">
                  <div className="small fw-semibold">Motor Nerves Tested:</div>
                  {(defs.protocol?.nerves_tested?.motor || []).map(n => (
                    <span key={n} className="badge bg-primary me-1">{n}</span>
                  ))}
                </div>
                <div className="mb-2">
                  <div className="small fw-semibold">Sensory Nerves Tested:</div>
                  {(defs.protocol?.nerves_tested?.sensory || []).map(n => (
                    <span key={n} className="badge bg-info me-1">{n}</span>
                  ))}
                </div>
                <div className="small fw-semibold mb-1">Indications:</div>
                <ul className="small mb-0">
                  {(defs.protocol?.indications || []).map(ind => <li key={ind}>{ind}</li>)}
                </ul>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">Parameters</div>
              <div className="card-body">
                {(defs.parameters || []).map(p => (
                  <div key={p.name} className="mb-2 pb-2 border-bottom">
                    <div className="fw-semibold small">{p.name} <span className="badge bg-secondary">{p.unit}</span></div>
                    <div className="small text-muted">{p.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card">
              <div className="card-header fw-semibold">Reference Ranges</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-dark"><tr><th>Nerve</th><th>DML ≤</th><th>CMAP ≥</th><th>MCV ≥</th><th>F-wave ≤</th></tr></thead>
                    <tbody>
                      {(defs.reference_ranges?.motor || []).map(r => (
                        <tr key={r.nerve}>
                          <td className="small">{r.nerve}</td>
                          <td>{r.dml_upper_ms} ms</td>
                          <td>{r.cmap_lower_mv} mV</td>
                          <td>{r.mcv_lower_ms} m/s</td>
                          <td>{r.fwave_upper_ms} ms</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="table-responsive">
                  <table className="table table-sm mb-0 border-top">
                    <thead className="table-secondary"><tr><th>Sensory Nerve</th><th>SNAP ≥</th><th>SCV ≥</th></tr></thead>
                    <tbody>
                      {(defs.reference_ranges?.sensory || []).map(r => (
                        <tr key={r.nerve}>
                          <td className="small">{r.nerve}</td>
                          <td>{r.snap_lower_uv} µV</td>
                          <td>{r.scv_lower_ms} m/s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card mb-3">
              <div className="card-header fw-semibold">Neuropathy Types</div>
              <div className="card-body">
                {(defs.neuropathy_types || []).map(n => (
                  <div key={n.type} className="mb-2">
                    <span className={`badge bg-${TYPE_COLOR(n.type)} me-2`}>{n.type}</span>
                    <span className="small">{n.description}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="card mb-3">
              <div className="card-header fw-semibold">Severity Levels</div>
              <div className="card-body">
                {(defs.severity_levels || []).map(s => (
                  <div key={s.level} className="mb-2">
                    <span className={`badge bg-${SEV_COLOR(s.level)} me-2`}>{s.level}</span>
                    <span className="small">{s.criteria}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <div className="card-header fw-semibold">Clinical Significance in Epilepsy</div>
              <div className="card-body small">{defs.clinical_significance}</div>
              <div className="card-footer small text-muted">{defs.reference}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
