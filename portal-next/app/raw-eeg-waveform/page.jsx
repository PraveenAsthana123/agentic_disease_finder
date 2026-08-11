'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TIER_COLOR = { Excellent: 'success', Good: 'primary', Fair: 'warning', Poor: 'danger' };
const GRADE_COLOR = { Good: 'success', Fair: 'warning', Poor: 'danger' };
const SEV_COLOR = { mild: 'warning', moderate: 'orange', severe: 'danger' };

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-2">
      <div className="card shadow-sm border-0 h-100">
        <div className="card-body text-center py-2">
          <div className={`h4 mb-0 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: 10 }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ items, labelKey, valueKey, colorFn }) {
  if (!items || !items.length) return null;
  const mx = Math.max(...items.map(i => i[valueKey] || 0));
  return (
    <div>
      {items.map((item, idx) => (
        <div key={idx} className="d-flex align-items-center mb-1">
          <div className="text-end small me-2" style={{ width: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {item[labelKey]}
          </div>
          <div className="flex-grow-1">
            <div className="progress" style={{ height: 20 }}>
              <div
                className={`progress-bar bg-${colorFn ? colorFn(item) : 'primary'}`}
                style={{ width: `${mx ? (item[valueKey] / mx) * 100 : 0}%` }}
              >
                <span className="small">{item[valueKey]}</span>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function SNRBar({ items }) {
  if (!items || !items.length) return null;
  const mx = 40;
  return (
    <div style={{ maxHeight: 320, overflowY: 'auto' }}>
      {items.map((item, idx) => {
        const snr = item.avg_snr;
        const color = snr >= 25 ? 'success' : snr >= 20 ? 'warning' : 'danger';
        return (
          <div key={idx} className="d-flex align-items-center mb-1">
            <div className="text-end small me-2 fw-semibold" style={{ width: 32 }}>
              {item.channel}
            </div>
            <div className="flex-grow-1">
              <div className="progress" style={{ height: 18 }}>
                <div className={`progress-bar bg-${color}`} style={{ width: `${(snr / mx) * 100}%` }}>
                  <span className="small">{snr} dB</span>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function RawEEGWaveformDashboard() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/raw-eeg-waveform/overview`).then(r => r.json()),
      fetch(`${API}/api/raw-eeg-waveform/breakdown`).then(r => r.json()),
      fetch(`${API}/api/raw-eeg-waveform/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefs(df);
    }).catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;
  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const k = overview.kpis || {};
  const ap = overview.activation_procedures || {};

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'channels', label: 'Channel SNR' },
    { id: 'artifacts', label: 'Artifacts' },
    { id: 'recordings', label: 'Recordings' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <h3>🧠 Raw EEG Waveform Dashboard</h3>
      <p className="text-muted small mb-3">
        {k.total_recordings} recordings · {k.unique_patients} patients ·
        {k.total_eeg_hours}h total EEG · real channel_quality + artifact_annotations data
      </p>

      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Total Recordings" value={k.total_recordings} color="primary" />
        <KPI label="Avg Duration" value={k.avg_duration_min ? `${k.avg_duration_min} min` : '—'} color="info"
          sub={`${k.total_eeg_hours}h total`} />
        <KPI label="Avg Channel SNR" value={k.avg_channel_snr_db ? `${k.avg_channel_snr_db} dB` : '—'}
          color={k.avg_channel_snr_db >= 20 ? 'success' : 'warning'}
          sub="target ≥ 20 dB" />
        <KPI label="Avg Impedance" value={k.avg_impedance_kohm ? `${k.avg_impedance_kohm} kΩ` : '—'}
          color={k.avg_impedance_kohm <= 5 ? 'success' : k.avg_impedance_kohm <= 10 ? 'warning' : 'danger'}
          sub="target < 5 kΩ" />
        <KPI label="Total Artifacts" value={k.total_artifacts} color="warning"
          sub={`${k.avg_artifacts_per_recording}/recording avg`} />
        <KPI label="Artifact Duration" value={k.total_artifact_sec ? `${k.total_artifact_sec}s` : '—'}
          color="secondary" sub="across all recordings" />
      </div>

      {/* Activation Procedure Coverage */}
      <div className="row mb-3">
        {[
          { label: 'Hyperventilation', key: 'hyperventilation', color: 'info' },
          { label: 'Photic Stimulation', key: 'photic_stimulation', color: 'purple' },
          { label: 'Sleep Recorded', key: 'sleep_recorded', color: 'secondary' },
          { label: 'Eyes Open', key: 'eyes_open', color: 'primary' },
        ].map(proc => {
          const d = ap[proc.key] || {};
          return (
            <div key={proc.key} className="col-6 col-md-3 mb-2">
              <div className="card shadow-sm border-0">
                <div className="card-body text-center py-2">
                  <div className={`h5 fw-bold text-${proc.color}`}>{d.pct ?? 0}%</div>
                  <div className="text-muted small">{proc.label}</div>
                  <div className="small text-muted">{d.count ?? 0}/{ap.n_recordings ?? 0} recordings</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active fw-semibold' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Channel quality distribution */}
          <div className="col-md-4">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Channel Quality Distribution</div>
              <div className="card-body">
                {(overview.channel_quality_distribution || []).map((g, i) => (
                  <div key={i} className="mb-2">
                    <div className="d-flex justify-content-between mb-1">
                      <span className={`badge bg-${g.color}`}>{g.grade}</span>
                      <span className="small fw-bold">{g.count} ch ({g.pct}%)</span>
                    </div>
                    <div className="progress" style={{ height: 12 }}>
                      <div className={`progress-bar bg-${g.color}`} style={{ width: `${g.pct}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Impedance distribution */}
          <div className="col-md-4">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Impedance Grade Distribution</div>
              <div className="card-body">
                {(overview.channel_impedance_distribution || []).map((g, i) => (
                  <div key={i} className="mb-2">
                    <div className="d-flex justify-content-between mb-1">
                      <span className={`badge bg-${g.color}`}>{g.grade}</span>
                      <span className="small fw-bold">{g.count} ch ({g.pct}%)</span>
                    </div>
                    <div className="progress" style={{ height: 12 }}>
                      <div className={`progress-bar bg-${g.color}`} style={{ width: `${g.pct}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Patient state */}
          <div className="col-md-4">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Patient State During Recording</div>
              <div className="card-body">
                {(overview.patient_state_distribution || []).map((s, i) => (
                  <div key={i} className="d-flex justify-content-between mb-2">
                    <span className="badge bg-info text-dark">{s.state}</span>
                    <span className="fw-bold">{s.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Recording types */}
          <div className="col-md-4">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Recording Types</div>
              <div className="card-body">
                {(overview.recording_type_distribution || []).map((r, i) => (
                  <div key={i} className="d-flex justify-content-between mb-2">
                    <span className="small">{(r.type || '').replace(/_/g, ' ')}</span>
                    <span className="badge bg-primary">{r.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Montage distribution */}
          <div className="col-md-4">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Montage Types</div>
              <div className="card-body">
                {(overview.montage_distribution || []).map((m, i) => (
                  <div key={i} className="d-flex justify-content-between mb-2">
                    <span className="small">{m.montage}</span>
                    <span className="badge bg-secondary">{m.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sampling rate */}
          <div className="col-md-4">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Sampling Rates (Hz)</div>
              <div className="card-body">
                {(overview.sampling_rate_distribution || []).map((s, i) => (
                  <div key={i} className="d-flex justify-content-between mb-2">
                    <span className="small">{s.sampling_rate} Hz</span>
                    <span className={`badge ${Number(s.sampling_rate) >= 256 ? 'bg-success' : 'bg-warning'}`}>{s.count}</span>
                  </div>
                ))}
                <div className="small text-muted mt-2">ACNS minimum: 256 Hz</div>
              </div>
            </div>
          </div>

          {/* Top artifact types */}
          <div className="col-md-6">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Artifact Types</div>
              <div className="card-body">
                <Bar
                  items={overview.artifact_type_distribution}
                  labelKey="type"
                  valueKey="count"
                  colorFn={() => 'warning'}
                />
              </div>
            </div>
          </div>

          {/* Artifact channels */}
          <div className="col-md-6">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Most Artifact-Prone Channels (Top 15)</div>
              <div className="card-body">
                <Bar
                  items={overview.artifact_channel_distribution}
                  labelKey="channel"
                  valueKey="count"
                  colorFn={() => 'danger'}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── CHANNEL SNR ── */}
      {tab === 'channels' && (
        <div className="row g-3">
          <div className="col-md-8">
            <div className="card shadow-sm border-0">
              <div className="card-header fw-semibold">Mean SNR per Channel (dB) — all patients</div>
              <div className="card-body">
                <div className="small text-muted mb-2">
                  <span className="badge bg-success me-1">≥25 dB Excellent</span>
                  <span className="badge bg-warning me-1">20–25 dB Good</span>
                  <span className="badge bg-danger me-1">&lt;20 dB Poor</span>
                </div>
                <SNRBar items={overview.snr_by_channel} />
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header fw-semibold">Summary</div>
              <div className="card-body">
                <div className="d-flex justify-content-between mb-2">
                  <span className="small">Fleet avg SNR</span>
                  <span className={`fw-bold text-${k.avg_channel_snr_db >= 20 ? 'success' : 'warning'}`}>
                    {k.avg_channel_snr_db} dB
                  </span>
                </div>
                <div className="d-flex justify-content-between mb-2">
                  <span className="small">Fleet avg impedance</span>
                  <span className={`fw-bold text-${k.avg_impedance_kohm <= 5 ? 'success' : k.avg_impedance_kohm <= 10 ? 'warning' : 'danger'}`}>
                    {k.avg_impedance_kohm} kΩ
                  </span>
                </div>
                <hr />
                <div className="small text-muted">
                  <strong>ACNS targets:</strong><br />
                  SNR &gt; 20 dB · Impedance &lt; 5 kΩ
                </div>
              </div>
            </div>
            <div className="card shadow-sm border-0">
              <div className="card-header fw-semibold">Impedance by Grade</div>
              <div className="card-body">
                {(overview.channel_impedance_distribution || []).map((g, i) => (
                  <div key={i} className="mb-2">
                    <div className="d-flex justify-content-between">
                      <span className={`badge bg-${g.color}`}>{g.grade}</span>
                      <span className="small">{g.count} ch · {g.pct}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── ARTIFACTS ── */}
      {tab === 'artifacts' && (
        <div className="row g-3">
          {/* Severity */}
          <div className="col-md-4">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Artifact Severity</div>
              <div className="card-body">
                {(overview.artifact_severity_distribution || []).map((s, i) => (
                  <div key={i} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${SEV_COLOR[s.severity] || 'secondary'}`}>{s.severity}</span>
                    <span className="fw-bold">{s.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Artifact types */}
          <div className="col-md-8">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Artifact Count by Type</div>
              <div className="card-body">
                <Bar
                  items={overview.artifact_type_distribution}
                  labelKey="type"
                  valueKey="count"
                  colorFn={() => 'warning'}
                />
              </div>
            </div>
          </div>

          {/* Per-patient artifact timelines (from breakdown) */}
          {breakdown && (
            <div className="col-12">
              <div className="card shadow-sm border-0">
                <div className="card-header fw-semibold">Artifact Timelines — Select Patient</div>
                <div className="card-body">
                  <div className="d-flex flex-wrap gap-2 mb-3">
                    {Object.keys(breakdown.artifact_timelines || {}).sort().map(pid => (
                      <button
                        key={pid}
                        className={`btn btn-sm ${expandedPt === pid ? 'btn-primary' : 'btn-outline-secondary'}`}
                        onClick={() => setExpandedPt(expandedPt === pid ? null : pid)}
                      >
                        {pid}
                      </button>
                    ))}
                  </div>
                  {expandedPt && breakdown.artifact_timelines[expandedPt] && (
                    <div>
                      <h6>Artifact Timeline — {expandedPt}</h6>
                      <div className="table-responsive">
                        <table className="table table-sm table-hover">
                          <thead className="table-dark">
                            <tr>
                              <th>Start (min)</th>
                              <th>Duration (sec)</th>
                              <th>Type</th>
                              <th>Channel</th>
                              <th>Severity</th>
                            </tr>
                          </thead>
                          <tbody>
                            {breakdown.artifact_timelines[expandedPt].map((a, i) => (
                              <tr key={i}>
                                <td className="small">{a.start_time_min?.toFixed(1)}</td>
                                <td className="small">{a.duration_sec?.toFixed(1)}</td>
                                <td className="small">{a.artifact_type}</td>
                                <td className="small"><code>{a.channel}</code></td>
                                <td>
                                  <span className={`badge bg-${SEV_COLOR[a.severity] || 'secondary'}`}>
                                    {a.severity}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── RECORDINGS ── */}
      {tab === 'recordings' && breakdown && (
        <div>
          <h5 className="mb-3">Per-Patient Recording Quality ({(breakdown.patient_profiles || []).length} patients)</h5>
          <div className="table-responsive">
            <table className="table table-hover table-sm">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Quality</th>
                  <th>Tier</th>
                  <th>Date</th>
                  <th>Type</th>
                  <th>Duration</th>
                  <th>Hz</th>
                  <th>SNR</th>
                  <th>Imp</th>
                  <th>Artifacts</th>
                  <th>Poor Ch.</th>
                  <th>HV</th>
                  <th>PS</th>
                  <th>EEG Pattern</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.patient_profiles || []).map(p => (
                  <>
                    <tr key={p.patient_id}>
                      <td><code>{p.patient_id}</code></td>
                      <td>
                        <div className="progress" style={{ height: 16, minWidth: 50 }}>
                          <div
                            className={`progress-bar bg-${TIER_COLOR[p.quality_tier] || 'secondary'}`}
                            style={{ width: `${p.quality_score}%` }}
                          >
                            <span style={{ fontSize: 10 }}>{p.quality_score}</span>
                          </div>
                        </div>
                      </td>
                      <td>
                        <span className={`badge bg-${TIER_COLOR[p.quality_tier] || 'secondary'}`}>
                          {p.quality_tier}
                        </span>
                      </td>
                      <td className="small">{p.study_date || '—'}</td>
                      <td className="small">{(p.recording_type || '').replace(/_/g, ' ')}</td>
                      <td className="small">{p.duration_min != null ? `${p.duration_min}m` : '—'}</td>
                      <td className="small">{p.sampling_rate ?? '—'}</td>
                      <td className="small">
                        <span className={`text-${p.mean_snr_db >= 20 ? 'success' : 'danger'}`}>
                          {p.mean_snr_db != null ? `${p.mean_snr_db}dB` : '—'}
                        </span>
                      </td>
                      <td className="small">
                        <span className={`text-${p.mean_impedance_kohm <= 5 ? 'success' : p.mean_impedance_kohm <= 10 ? 'warning' : 'danger'}`}>
                          {p.mean_impedance_kohm != null ? `${p.mean_impedance_kohm}kΩ` : '—'}
                        </span>
                      </td>
                      <td className="small">
                        <span className="badge bg-warning text-dark">{p.artifact_count}</span>
                        {p.severe_artifacts > 0 &&
                          <span className="badge bg-danger ms-1">{p.severe_artifacts} sev</span>}
                      </td>
                      <td className="small">{p.poor_channels ?? 0}</td>
                      <td>{p.hyperventilation
                        ? <span className="badge bg-success">✓</span>
                        : <span className="badge bg-light text-muted">—</span>}
                      </td>
                      <td>{p.photic_stimulation
                        ? <span className="badge bg-success">✓</span>
                        : <span className="badge bg-light text-muted">—</span>}
                      </td>
                      <td className="small">{p.eeg_pattern ? p.eeg_pattern.slice(0, 25) : '—'}</td>
                      <td>
                        <button
                          className="btn btn-sm btn-outline-secondary"
                          onClick={() => setExpandedPt(expandedPt === p.patient_id ? null : p.patient_id)}
                        >
                          {expandedPt === p.patient_id ? 'Hide' : 'Detail'}
                        </button>
                      </td>
                    </tr>
                    {expandedPt === p.patient_id && (
                      <tr>
                        <td colSpan={15} className="bg-light">
                          <div className="p-2">
                            <div className="row">
                              <div className="col-md-6">
                                <strong>Channel Quality — {p.patient_id}</strong>
                                <div className="table-responsive mt-2" style={{ maxHeight: 260, overflowY: 'auto' }}>
                                  <table className="table table-sm mb-0">
                                    <thead>
                                      <tr><th>Channel</th><th>Imp (kΩ)</th><th>Imp Grade</th><th>SNR (dB)</th><th>Quality</th></tr>
                                    </thead>
                                    <tbody>
                                      {(breakdown.channel_details[p.patient_id] || []).map((ch, i) => (
                                        <tr key={i}>
                                          <td className="small fw-semibold"><code>{ch.channel}</code></td>
                                          <td className="small">{ch.impedance_kohm?.toFixed(1)}</td>
                                          <td>
                                            <span className={`badge bg-${GRADE_COLOR[ch.impedance_grade] || 'secondary'}`}>
                                              {ch.impedance_grade}
                                            </span>
                                          </td>
                                          <td className="small">{ch.snr_db?.toFixed(1)}</td>
                                          <td>
                                            <span className={`badge bg-${GRADE_COLOR[ch.quality_grade] || 'secondary'}`}>
                                              {ch.quality_grade}
                                            </span>
                                          </td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                              <div className="col-md-6">
                                <strong>Recording Details</strong>
                                <table className="table table-sm mt-2">
                                  <tbody>
                                    <tr><td className="small text-muted">Montage</td><td className="small">{p.montage}</td></tr>
                                    <tr><td className="small text-muted">Electrode System</td><td className="small">{p.electrode_system}</td></tr>
                                    <tr><td className="small text-muted">Patient State</td><td className="small">{p.patient_state}</td></tr>
                                    <tr><td className="small text-muted">Sleep Recorded</td><td className="small">{p.sleep_recorded ? 'Yes' : 'No'}</td></tr>
                                    <tr><td className="small text-muted">Artifact Total</td><td className="small">{p.artifact_sec}s</td></tr>
                                    <tr><td className="small text-muted">Artifact Types</td><td className="small">{(p.artifact_types || []).join(', ') || '—'}</td></tr>
                                    <tr><td className="small text-muted">EEG Pattern</td><td className="small">{p.eeg_pattern || '—'}</td></tr>
                                    <tr><td className="small text-muted">Onset Zone</td><td className="small">{p.onset_zone || '—'}</td></tr>
                                    {p.technician_notes && (
                                      <tr>
                                        <td className="small text-muted">Tech Notes</td>
                                        <td className="small text-muted fst-italic">{p.technician_notes}</td>
                                      </tr>
                                    )}
                                  </tbody>
                                </table>
                                {(p.poor_channel_names || []).length > 0 && (
                                  <div>
                                    <strong className="small text-danger">Poor-Quality Channels:</strong>{' '}
                                    {p.poor_channel_names.map(ch => (
                                      <span key={ch} className="badge bg-danger me-1">{ch}</span>
                                    ))}
                                  </div>
                                )}
                              </div>
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
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div>
          <p className="text-muted mb-3">{defs.description}</p>

          <h5>Recording Quality Tiers</h5>
          <div className="row mb-4">
            {(defs.quality_tiers || []).map((t, i) => (
              <div key={i} className="col-md-3 mb-2">
                <div className={`card border-${TIER_COLOR[t.tier] || 'secondary'} shadow-sm h-100`}>
                  <div className="card-header">
                    <span className={`badge bg-${TIER_COLOR[t.tier] || 'secondary'} me-2`}>{t.tier}</span>
                    Score {t.score_range}
                  </div>
                  <div className="card-body small text-muted">{t.description}</div>
                </div>
              </div>
            ))}
          </div>

          <h5>Quality Score Components</h5>
          <table className="table table-sm table-bordered mb-4">
            <thead className="table-dark">
              <tr><th>Component</th><th>Weight</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(defs.quality_score_components || []).map((c, i) => (
                <tr key={i}>
                  <td className="small fw-semibold">{c.component}</td>
                  <td><span className="badge bg-primary">{c.weight} pts</span></td>
                  <td className="small text-muted">{c.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="row mb-4">
            <div className="col-md-6">
              <h5>Artifact Types</h5>
              {Object.entries(defs.artifact_types || {}).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <span className="badge bg-warning text-dark me-2">{k}</span>
                  <span className="small text-muted">{v}</span>
                </div>
              ))}
            </div>
            <div className="col-md-6">
              <h5>Channel Quality Grades</h5>
              {Object.entries(defs.channel_grades || {}).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <span className={`badge bg-${GRADE_COLOR[k] || 'secondary'} me-2`}>{k}</span>
                  <span className="small text-muted">{v}</span>
                </div>
              ))}
              <h5 className="mt-3">Impedance Grades</h5>
              {Object.entries(defs.impedance_grades || {}).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <span className={`badge bg-${GRADE_COLOR[k] || 'secondary'} me-2`}>{k}</span>
                  <span className="small text-muted">{v}</span>
                </div>
              ))}
            </div>
          </div>

          <h5>Activation Procedures</h5>
          <div className="row mb-4">
            {Object.entries(defs.activation_procedures || {}).map(([k, v]) => (
              <div key={k} className="col-md-6 mb-2">
                <div className="card border-0 shadow-sm">
                  <div className="card-body small">
                    <span className="badge bg-info text-dark me-2">{k.replace(/_/g, ' ')}</span>
                    <span className="text-muted">{v}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <h5>Data Sources</h5>
          <ul className="list-group list-group-flush mb-3">
            {(defs.data_sources || []).map((s, i) => (
              <li key={i} className="list-group-item small">{s}</li>
            ))}
          </ul>

          <h5>Standards</h5>
          <ul className="list-group list-group-flush">
            {(defs.standards || []).map((s, i) => (
              <li key={i} className="list-group-item small">{s}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
