'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const gradeColor = g =>
  g === 'Good'  ? 'success' :
  g === 'Fair'  ? 'warning' :
  g === 'Poor'  ? 'danger'  : 'secondary';

const impColor = v => {
  if (v === null || v === undefined) return '#e9ecef';
  if (v < 5)  return '#198754';   // Good — green
  if (v < 10) return '#ffc107';   // Fair — yellow
  return '#dc3545';               // Poor — red
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className={`h4 mb-0 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function GradeBadge({ grade }) {
  return <span className={`badge bg-${gradeColor(grade)}`}>{grade}</span>;
}

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'channels',    label: 'Channel Stats' },
  { id: 'heatmap',     label: 'Impedance Heatmap' },
  { id: 'artifacts',   label: 'Artifacts' },
  { id: 'patients',    label: 'Per Patient' },
  { id: 'definitions', label: 'Definitions' },
];

export default function EEGChannelQualityMapPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);
  const [err,  setErr]  = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/eeg-channel-quality-map/overview`).then(r => r.json()),
      fetch(`${API}/api/eeg-channel-quality-map/breakdown`).then(r => r.json()),
      fetch(`${API}/api/eeg-channel-quality-map/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-4">Loading EEG Channel Quality Map…</div>;

  const k = ov.kpis || {};

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">🧠 EEG Channel Quality Map</h4>
      <p className="text-muted small mb-3">
        10-20 electrode system — impedance, SNR, and artifact burden across 30 patients × 19 channels.
        Sources: <code>channel_quality</code> (570 records) · <code>artifact_annotations</code> (169) ·{' '}
        <code>eeg_acquisition</code> (30 recordings)
      </p>

      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Patients"            value={k.n_patients}                   color="primary" />
        <KPI label="Channel Records"     value={k.total_channel_records}         color="info" />
        <KPI label="Avg Impedance"       value={k.avg_impedance_kohm ? `${k.avg_impedance_kohm} kΩ` : '—'} color={k.avg_impedance_kohm > 10 ? 'danger' : 'warning'} sub="ACNS threshold < 5 kΩ" />
        <KPI label="Avg SNR"             value={k.avg_snr_db ? `${k.avg_snr_db} dB` : '—'} color={k.avg_snr_db >= 20 ? 'success' : 'warning'} sub="≥ 20 dB acceptable" />
        <KPI label="Poor Impedance"      value={`${k.poor_impedance_pct}%`}     color={k.poor_impedance_pct > 40 ? 'danger' : 'warning'} sub="of channel records" />
        <KPI label="Total Artifacts"     value={k.total_artifacts}              color="warning" />
        <KPI label="Artifact Types"      value={k.artifact_types}               color="secondary" />
        <KPI label="Channels (10-20)"    value={k.n_channels}                   color="primary" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div className="row">
          {/* Grade distributions */}
          <div className="col-md-4 mb-3">
            <div className="card h-100">
              <div className="card-header fw-semibold">Impedance Grade Distribution</div>
              <div className="card-body p-2">
                {(ov.impedance_grade_distribution || []).map(g => {
                  const total = (ov.impedance_grade_distribution || []).reduce((s, x) => s + x.count, 0);
                  const pct = total ? ((g.count / total) * 100).toFixed(1) : 0;
                  return (
                    <div key={g.grade} className="mb-2">
                      <div className="d-flex justify-content-between small">
                        <span><GradeBadge grade={g.grade} /> {g.grade}</span>
                        <span className="fw-bold">{g.count} ({pct}%)</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${gradeColor(g.grade)}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="col-md-4 mb-3">
            <div className="card h-100">
              <div className="card-header fw-semibold">Quality Grade Distribution</div>
              <div className="card-body p-2">
                {(ov.quality_grade_distribution || []).map(g => {
                  const total = (ov.quality_grade_distribution || []).reduce((s, x) => s + x.count, 0);
                  const pct = total ? ((g.count / total) * 100).toFixed(1) : 0;
                  return (
                    <div key={g.grade} className="mb-2">
                      <div className="d-flex justify-content-between small">
                        <span><GradeBadge grade={g.grade} /> {g.grade}</span>
                        <span className="fw-bold">{g.count} ({pct}%)</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${gradeColor(g.grade)}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="col-md-4 mb-3">
            <div className="card h-100">
              <div className="card-header fw-semibold">Artifact Types</div>
              <div className="card-body p-2">
                {(ov.artifact_by_type || []).map(a => {
                  const total = (ov.artifact_by_type || []).reduce((s, x) => s + x.count, 0);
                  const pct = total ? ((a.count / total) * 100).toFixed(1) : 0;
                  return (
                    <div key={a.artifact_type} className="mb-2">
                      <div className="d-flex justify-content-between small">
                        <span className="text-capitalize">{a.artifact_type.replace(/_/g, ' ')}</span>
                        <span className="fw-bold">{a.count} ({pct}%)</span>
                      </div>
                      <div className="progress" style={{ height: 6 }}>
                        <div className="progress-bar bg-warning" style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Region summary */}
          <div className="col-md-6 mb-3">
            <div className="card">
              <div className="card-header fw-semibold">Region Summary</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Region</th><th>Channels</th>
                      <th>Avg Impedance (kΩ)</th><th>Avg SNR (dB)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.region_summary || []).map(r => (
                      <tr key={r.region}>
                        <td><strong>{r.region}</strong></td>
                        <td>{r.n_channels}</td>
                        <td>
                          <span className={`fw-bold text-${r.avg_impedance_kohm > 10 ? 'danger' : r.avg_impedance_kohm > 5 ? 'warning' : 'success'}`}>
                            {r.avg_impedance_kohm ?? '—'}
                          </span>
                        </td>
                        <td>
                          <span className={`fw-bold text-${r.avg_snr_db >= 20 ? 'success' : 'warning'}`}>
                            {r.avg_snr_db ?? '—'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Top problematic channels */}
          <div className="col-md-6 mb-3">
            <div className="card">
              <div className="card-header fw-semibold text-danger">🚨 Top Problematic Channels (by Poor Impedance count)</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Channel</th><th>Region</th>
                      <th>Avg Ω (kΩ)</th><th>Poor Imp.</th><th>Artifacts</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.top_problematic_channels || []).map(ch => {
                      const artCh = (ov.top_artifact_channels || []).find(a => a.channel === ch.channel);
                      return (
                        <tr key={ch.channel}>
                          <td><strong>{ch.channel}</strong></td>
                          <td><span className="badge bg-secondary">{ch.region}</span></td>
                          <td>
                            <span className={`fw-bold text-${ch.avg_impedance_kohm > 10 ? 'danger' : 'warning'}`}>
                              {ch.avg_impedance_kohm ?? '—'}
                            </span>
                          </td>
                          <td><span className="badge bg-danger">{ch.poor_impedance_count}</span></td>
                          <td>{artCh ? artCh.artifact_count : 0}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Top artifact channels */}
          <div className="col-12 mb-3">
            <div className="card">
              <div className="card-header fw-semibold">Top Artifact Channels</div>
              <div className="card-body d-flex flex-wrap gap-2">
                {(ov.top_artifact_channels || []).map(a => (
                  <span key={a.channel} className="badge bg-warning text-dark fs-6">
                    {a.channel}: {a.artifact_count}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Channel Stats tab */}
      {tab === 'channels' && (
        <div className="card">
          <div className="card-header fw-semibold">All 19 Channels — Aggregate Stats (30 patients each)</div>
          <div className="card-body p-0">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Channel</th><th>Region</th>
                  <th>Avg Impedance (kΩ)</th><th>Avg SNR (dB)</th>
                  <th>Dominant Imp. Grade</th><th>Poor Imp. Count</th><th>Poor Quality</th>
                </tr>
              </thead>
              <tbody>
                {(ov.channel_summary || []).map(ch => (
                  <tr key={ch.channel}>
                    <td><strong>{ch.channel}</strong></td>
                    <td><span className="badge bg-light text-dark">{ch.region}</span></td>
                    <td>
                      <span className={`fw-bold text-${ch.avg_impedance_kohm > 10 ? 'danger' : ch.avg_impedance_kohm > 5 ? 'warning' : 'success'}`}>
                        {ch.avg_impedance_kohm ?? '—'}
                      </span>
                    </td>
                    <td>
                      <span className={`fw-bold text-${(ch.avg_snr_db || 0) >= 20 ? 'success' : 'warning'}`}>
                        {ch.avg_snr_db ?? '—'}
                      </span>
                    </td>
                    <td><GradeBadge grade={ch.impedance_grade_dominant} /></td>
                    <td>
                      {ch.poor_impedance_count > 0
                        ? <span className="badge bg-danger">{ch.poor_impedance_count}</span>
                        : <span className="text-success">0</span>}
                    </td>
                    <td>
                      {ch.poor_quality_count > 0
                        ? <span className="badge bg-warning text-dark">{ch.poor_quality_count}</span>
                        : <span className="text-success">0</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Impedance Heatmap tab */}
      {tab === 'heatmap' && bd?.heatmap && (
        <div>
          <p className="text-muted small mb-2">
            Cell color: green &lt; 5 kΩ (Good) · yellow 5–10 kΩ (Fair) · red &gt; 10 kΩ (Poor) · grey = no data
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table className="table table-bordered table-sm" style={{ minWidth: 900, fontSize: '0.72rem' }}>
              <thead className="table-light">
                <tr>
                  <th style={{ minWidth: 50 }}>Ch</th>
                  <th style={{ minWidth: 55 }}>Region</th>
                  {bd.heatmap.patients.map(pid => (
                    <th key={pid} style={{ minWidth: 55, writingMode: 'vertical-rl', transform: 'rotate(180deg)', maxWidth: 55 }}
                        title={pid}>{pid.slice(0, 7)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.heatmap.channels || []).map(row => (
                  <tr key={row.channel}>
                    <td className="fw-bold">{row.channel}</td>
                    <td className="text-muted">{row.region}</td>
                    {(row.values || []).map((val, i) => (
                      <td
                        key={i}
                        title={val !== null ? `${val} kΩ` : 'no data'}
                        style={{
                          background: impColor(val),
                          color: val > 10 || (val > 5 && val <= 10) ? '#000' : '#fff',
                          textAlign: 'center',
                          padding: '1px 2px',
                        }}
                      >
                        {val !== null ? val : '—'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="d-flex gap-3 mt-2 small">
            <span><span style={{ background: '#198754', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>■</span> Good (&lt;5 kΩ)</span>
            <span><span style={{ background: '#ffc107', color: '#000', padding: '2px 8px', borderRadius: 4 }}>■</span> Fair (5–10 kΩ)</span>
            <span><span style={{ background: '#dc3545', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>■</span> Poor (&gt;10 kΩ)</span>
            <span><span style={{ background: '#e9ecef', color: '#333', padding: '2px 8px', borderRadius: 4 }}>■</span> No data</span>
          </div>
        </div>
      )}

      {/* Artifacts tab */}
      {tab === 'artifacts' && bd && (
        <div className="row">
          <div className="col-md-4 mb-3">
            <div className="card h-100">
              <div className="card-header fw-semibold">Artifact Severity</div>
              <div className="card-body p-2">
                {(bd.artifact_severity_distribution || []).map(s => {
                  const total = (bd.artifact_severity_distribution || []).reduce((x, y) => x + y.count, 0);
                  const pct = total ? ((s.count / total) * 100).toFixed(1) : 0;
                  const col = s.severity === 'severe' ? 'danger' : s.severity === 'moderate' ? 'warning' : 'info';
                  return (
                    <div key={s.severity} className="mb-2">
                      <div className="d-flex justify-content-between small">
                        <span className="text-capitalize">{s.severity}</span>
                        <span className="fw-bold">{s.count} ({pct}%)</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className={`progress-bar bg-${col}`} style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="col-md-8 mb-3">
            <div className="card">
              <div className="card-header fw-semibold">Acquisition Parameters</div>
              <div className="card-body">
                <div className="row">
                  <div className="col-sm-6">
                    <h6 className="text-muted small">Recording Types</h6>
                    {(bd.acquisition_summary?.recording_types || []).map(rt => (
                      <div key={rt.type} className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize">{rt.type.replace(/_/g, ' ')}</span>
                        <span className="badge bg-info text-dark">{rt.count}</span>
                      </div>
                    ))}
                  </div>
                  <div className="col-sm-6">
                    <h6 className="text-muted small">Sampling Rates</h6>
                    {(bd.acquisition_summary?.sampling_rates || []).map(sr => (
                      <div key={sr.rate_hz} className="d-flex justify-content-between small mb-1">
                        <span>{sr.rate_hz} Hz</span>
                        <span className="badge bg-primary">{sr.count}</span>
                      </div>
                    ))}
                    {bd.acquisition_summary?.avg_duration_min && (
                      <div className="mt-2 text-muted small">
                        Avg duration: <strong>{bd.acquisition_summary.avg_duration_min} min</strong>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">Artifact Log (first 60)</div>
              <div className="card-body p-0" style={{ maxHeight: 380, overflowY: 'auto' }}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light sticky-top">
                    <tr>
                      <th>Patient</th><th>Channel</th><th>Artifact Type</th>
                      <th>Start (min)</th><th>Duration (s)</th><th>Severity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.artifact_table || []).map((a, i) => (
                      <tr key={i}>
                        <td className="small">{a.patient_id}</td>
                        <td><strong>{a.channel}</strong></td>
                        <td className="text-capitalize small">{(a.artifact_type || '').replace(/_/g, ' ')}</td>
                        <td className="small">{a.start_time_min ?? '—'}</td>
                        <td className="small">{a.duration_sec ?? '—'}</td>
                        <td>
                          <span className={`badge bg-${a.severity === 'severe' ? 'danger' : a.severity === 'moderate' ? 'warning text-dark' : 'info'}`}>
                            {a.severity}
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
      )}

      {/* Per Patient tab */}
      {tab === 'patients' && bd && (
        <div>
          <p className="text-muted small mb-2">Click a patient card to expand channel-level detail.</p>
          <div className="row">
            {(bd.patient_cards || []).map(pt => (
              <div key={pt.patient_id} className="col-md-4 col-lg-3 mb-3">
                <div
                  className={`card h-100 border-${gradeColor(pt.overall_grade)}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setExpandedPt(expandedPt === pt.patient_id ? null : pt.patient_id)}
                >
                  <div className={`card-header py-1 bg-${gradeColor(pt.overall_grade)} bg-opacity-10`}>
                    <strong>{pt.patient_id}</strong>{' '}
                    <GradeBadge grade={pt.overall_grade} />
                  </div>
                  <div className="card-body py-2 px-3 small">
                    <div>Avg Impedance: <strong>{pt.avg_impedance_kohm ?? '—'} kΩ</strong></div>
                    <div>Avg SNR: <strong>{pt.avg_snr_db ?? '—'} dB</strong></div>
                    <div>
                      Good: <span className="text-success fw-bold">{pt.good_channels}</span>{' '}
                      Poor Imp: <span className="text-danger fw-bold">{pt.poor_channels}</span>{' '}
                      Poor Qual: <span className="text-warning fw-bold">{pt.poor_quality_channels}</span>
                    </div>
                  </div>
                  {expandedPt === pt.patient_id && (
                    <div className="card-footer p-1">
                      <table className="table table-sm mb-0" style={{ fontSize: '0.68rem' }}>
                        <thead><tr><th>Ch</th><th>Ω(kΩ)</th><th>SNR</th><th>ImpG</th><th>QualG</th></tr></thead>
                        <tbody>
                          {(pt.channels || []).map(c => (
                            <tr key={c.channel}>
                              <td>{c.channel}</td>
                              <td className={`text-${c.impedance_kohm > 10 ? 'danger' : c.impedance_kohm > 5 ? 'warning' : 'success'}`}>
                                {c.impedance_kohm ?? '—'}
                              </td>
                              <td className={`text-${(c.snr_db || 0) >= 20 ? 'success' : 'warning'}`}>
                                {c.snr_db ?? '—'}
                              </td>
                              <td><GradeBadge grade={c.impedance_grade} /></td>
                              <td><GradeBadge grade={c.quality_grade} /></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card h-100">
              <div className="card-header fw-semibold">Glossary</div>
              <div className="card-body">
                {(defs.terms || []).map(t => (
                  <div key={t.term} className="mb-3">
                    <h6 className="fw-bold">{t.term}</h6>
                    <p className="small text-muted mb-1">{t.definition}</p>
                    {t.thresholds && (
                      <div className="d-flex gap-2 flex-wrap">
                        {Object.entries(t.thresholds).map(([g, v]) => (
                          <span key={g} className={`badge bg-${gradeColor(g)}`}>{g}: {v}</span>
                        ))}
                      </div>
                    )}
                    {t.types && (
                      <div className="d-flex gap-1 flex-wrap mt-1">
                        {t.types.map(ty => (
                          <span key={ty} className="badge bg-warning text-dark small">{ty}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card mb-3">
              <div className="card-header fw-semibold">Electrode Regions (10-20 System)</div>
              <div className="card-body p-2">
                {(defs.regions || []).map(r => (
                  <div key={r.region} className="mb-2">
                    <div className="fw-semibold">{r.region}</div>
                    <div className="d-flex flex-wrap gap-1 mb-1">
                      {r.channels.map(ch => (
                        <span key={ch} className="badge bg-primary">{ch}</span>
                      ))}
                    </div>
                    <div className="text-muted small">{r.clinical_significance}</div>
                  </div>
                ))}
              </div>
            </div>
            <div className="card">
              <div className="card-header fw-semibold">Standards</div>
              <div className="card-body">
                <ul className="list-unstyled mb-0 small">
                  {(defs.standards || []).map(s => (
                    <li key={s} className="mb-1">✓ {s}</li>
                  ))}
                </ul>
              </div>
            </div>
            <div className="card mt-3">
              <div className="card-header fw-semibold">Data Sources</div>
              <div className="card-body">
                {Object.entries(defs.data_sources || {}).map(([k, v]) => (
                  <div key={k} className="mb-1 small">
                    <code>{k}</code>: {v}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
