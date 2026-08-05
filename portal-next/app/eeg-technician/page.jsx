'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',     label: 'Overview' },
  { id: 'quality',      label: 'Signal Quality' },
  { id: 'patients',     label: 'Patient List' },
  { id: 'protocol',     label: 'Protocol Completion' },
  { id: 'definitions',  label: 'ACNS Standards' },
];

const QUALITY_COLOR = { Excellent: 'success', Good: 'info', Acceptable: 'warning', Poor: 'danger' };
const TYPE_COLOR = { routine: 'primary', ambulatory: 'secondary', video_eeg: 'info', LTM: 'warning' };
const ARTIFACT_COLOR = {
  eye_blink: '#6366f1', muscle: '#ef4444', movement: '#f59e0b',
  electrode_pop: '#10b981', sweat: '#06b6d4', ECG: '#8b5cf6',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ pct, color = 'primary', height = 16 }) {
  const c = typeof color === 'string' && color.startsWith('#') ? null : color;
  return (
    <div className="progress" style={{ height, borderRadius: 6 }}>
      <div
        className={c ? `progress-bar bg-${c}` : 'progress-bar'}
        style={{ width: `${Math.min(Math.max(pct, 0), 100)}%`, borderRadius: 6, ...(c ? {} : { backgroundColor: color }) }}
      />
    </div>
  );
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;
  const k = ov.kpis || {};
  const sev = ov.severity_distribution || [];
  const rec = ov.recording_type_distribution || [];
  const art = ov.artifact_type_distribution || [];

  return (
    <div>
      <div className="row mb-4">
        <KPI label="Total Recordings"      value={k.total_recordings}                              color="primary"   sub="Across all patients" />
        <KPI label="Quality Pass Rate"     value={`${k.signal_quality_pass_rate?.toFixed(1)}%`}   color="success"   sub="Acceptable or better" />
        <KPI label="Artifact Rate"         value={`${k.artifact_rate?.toFixed(1)}%`}              color="warning"   sub="Segments with artifacts" />
        <KPI label="Artifacts Logged"      value={k.total_artifacts_logged}                        color="danger"    sub="Total annotations" />
      </div>
      <div className="row mb-4">
        <KPI label="Mean Duration"         value={`${(k.mean_recording_duration_min / 60).toFixed(1)} h`} color="info"  sub="Per recording" />
        <KPI label="Avg Channels"          value={k.avg_channels_per_study}                        color="secondary" sub="10-20 system" />
        <KPI label="Photic Stim Rate"      value={`${k.photic_stim_rate?.toFixed(0)}%`}           color="primary"   sub="Protocol adherence" />
        <KPI label="Sleep Capture Rate"    value={`${k.sleep_capture_rate?.toFixed(1)}%`}         color="success"   sub="Sleep EEG obtained" />
      </div>

      <div className="row mb-4">
        {/* Quality distribution */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white"><strong>Quality Grade Distribution</strong></div>
            <div className="card-body">
              {sev.map(s => (
                <div key={s.grade} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span><span className={`badge bg-${QUALITY_COLOR[s.grade] || 'secondary'} me-1`}>{s.grade}</span></span>
                    <span className="fw-bold">{s.count}</span>
                  </div>
                  <Bar pct={(s.count / (k.total_recordings || 1)) * 100} color={QUALITY_COLOR[s.grade] || 'secondary'} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Recording type distribution */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white"><strong>Recording Types</strong></div>
            <div className="card-body">
              {rec.map(r => (
                <div key={r.type} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span><span className={`badge bg-${TYPE_COLOR[r.type] || 'secondary'} me-1`}>{r.type.replace('_', '-')}</span></span>
                    <span className="fw-bold">{r.count}</span>
                  </div>
                  <Bar pct={(r.count / (k.total_recordings || 1)) * 100} color={TYPE_COLOR[r.type] || 'secondary'} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Artifact type distribution */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white"><strong>Artifact Types</strong></div>
            <div className="card-body">
              {art.map(a => {
                const max = Math.max(...art.map(x => x.count));
                return (
                  <div key={a.type} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{a.type.replace('_', ' ')}</span>
                      <span className="fw-bold">{a.count}</span>
                    </div>
                    <Bar pct={(a.count / (max || 1)) * 100} color={ARTIFACT_COLOR[a.type] || '#6b7280'} />
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function QualityPanel({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;
  const qh = ov.quality_histogram || [];
  const ih = ov.impedance_histogram || [];
  const qMax = Math.max(...qh.map(x => x.count), 1);
  const iMax = Math.max(...ih.map(x => x.count), 1);

  return (
    <div className="row">
      <div className="col-md-6 mb-3">
        <div className="card shadow-sm">
          <div className="card-header py-2 bg-dark text-white">
            <strong>Channel Quality Score Distribution</strong>
            <span className="text-muted ms-2" style={{ fontSize: '0.75rem' }}>% good channels per study</span>
          </div>
          <div className="card-body">
            {qh.map(q => (
              <div key={q.range} className="mb-3">
                <div className="d-flex justify-content-between small mb-1">
                  <span className="fw-bold">{q.range}</span>
                  <span className="text-muted">{q.count} studies</span>
                </div>
                <Bar pct={(q.count / qMax) * 100} color={q.count > 10 ? 'warning' : 'info'} height={20} />
              </div>
            ))}
            <div className="alert alert-warning small py-1 px-2 mt-2 mb-0">
              Most studies fall in 40–80% good-channel range — targets ≥90% per ACNS guidelines.
            </div>
          </div>
        </div>
      </div>

      <div className="col-md-6 mb-3">
        <div className="card shadow-sm">
          <div className="card-header py-2 bg-dark text-white">
            <strong>Impedance Distribution</strong>
            <span className="text-muted ms-2" style={{ fontSize: '0.75rem' }}>Electrode-scalp impedance (kΩ)</span>
          </div>
          <div className="card-body">
            {ih.map(b => {
              const good = b.range.includes('0-2') || b.range.includes('2-5');
              return (
                <div key={b.range} className="mb-3">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="fw-bold">{b.range}</span>
                    <span className={good ? 'text-success fw-bold' : 'text-danger'}>{b.count} electrodes</span>
                  </div>
                  <Bar pct={(b.count / iMax) * 100} color={good ? 'success' : 'danger'} height={20} />
                </div>
              );
            })}
            <div className="alert alert-danger small py-1 px-2 mt-2 mb-0">
              High impedance (&gt;5 kΩ) detected — re-prep gel application recommended before recording.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function PatientsPanel({ bd }) {
  const [expanded, setExpanded] = useState(null);
  if (!bd || !bd.patients) return <div className="text-muted p-3">Loading…</div>;
  const pts = bd.patients;

  return (
    <div>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-dark">
            <tr>
              <th>Patient</th><th>Type</th><th>Duration</th>
              <th>Channels</th><th>Impedance</th><th>SNR (dB)</th>
              <th>Artifacts</th><th>Grade</th><th></th>
            </tr>
          </thead>
          <tbody>
            {pts.map(p => (
              <>
                <tr key={p.patient_id}>
                  <td>
                    <div className="fw-bold small">{p.patient_id}</div>
                    <div className="text-muted" style={{ fontSize: '0.7rem' }}>{p.gender}, {p.age}y</div>
                  </td>
                  <td><span className={`badge bg-${TYPE_COLOR[p.recording_type] || 'secondary'}`}>{p.recording_type?.replace('_', '-')}</span></td>
                  <td className="small">{p.duration_min >= 60 ? `${(p.duration_min / 60).toFixed(1)}h` : `${p.duration_min}m`}</td>
                  <td className="small">{p.good_channels}G / {p.fair_channels}F / {p.poor_channels}P</td>
                  <td className="small">
                    <span className={p.avg_impedance_kohm > 10 ? 'text-danger fw-bold' : 'text-success'}>
                      {p.avg_impedance_kohm?.toFixed(1)} kΩ
                    </span>
                  </td>
                  <td className="small">
                    <span className={p.avg_snr_db > 20 ? 'text-success' : p.avg_snr_db > 10 ? 'text-warning' : 'text-danger'}>
                      {p.avg_snr_db?.toFixed(1)}
                    </span>
                  </td>
                  <td className="small">{p.artifact_count} <span className="text-muted">({p.dominant_artifact})</span></td>
                  <td><span className={`badge bg-${QUALITY_COLOR[p.overall_quality_grade] || 'secondary'}`}>{p.overall_quality_grade}</span></td>
                  <td>
                    <button className="btn btn-xs btn-outline-secondary" style={{ fontSize: '0.7rem', padding: '1px 6px' }}
                      onClick={() => setExpanded(expanded === p.patient_id ? null : p.patient_id)}>
                      {expanded === p.patient_id ? '▲' : '▼'}
                    </button>
                  </td>
                </tr>
                {expanded === p.patient_id && p.channel_detail && (
                  <tr key={`${p.patient_id}-detail`}>
                    <td colSpan={9} className="bg-light">
                      <div className="p-2">
                        <div className="small fw-bold mb-1">Channel Detail — {p.patient_id}</div>
                        <div className="row g-1">
                          {p.channel_detail.map(ch => (
                            <div key={ch.channel} className="col-6 col-md-3 col-lg-2">
                              <div className="card border-0 shadow-none bg-white p-1 text-center" style={{ fontSize: '0.65rem' }}>
                                <div className="fw-bold">{ch.channel}</div>
                                <div className={`badge bg-${ch.impedance_grade === 'Good' ? 'success' : ch.impedance_grade === 'Fair' ? 'warning' : 'danger'}`} style={{ fontSize: '0.6rem' }}>
                                  {ch.impedance_kohm?.toFixed(1)} kΩ
                                </div>
                                <div className="text-muted">{ch.quality_grade}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                        {p.technician_notes && (
                          <div className="alert alert-info small py-1 px-2 mt-2 mb-0">
                            <strong>Notes:</strong> {p.technician_notes}
                          </div>
                        )}
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
  );
}

function ProtocolPanel({ bd }) {
  if (!bd || !bd.patients) return <div className="text-muted p-3">Loading…</div>;
  const pts = bd.patients;
  const total = pts.length;

  const hvDone    = pts.filter(p => p.hyperventilation_done).length;
  const photicDone = pts.filter(p => p.photic_done).length;
  const sleepDone = pts.filter(p => p.sleep_recorded).length;
  const eyesOpen  = pts.filter(p => p.eyes_open).length;
  const impPass   = pts.filter(p => p.impedance_pass).length;

  const protocols = [
    { label: 'Hyperventilation (HV)', done: hvDone, color: 'success' },
    { label: 'Photic Stimulation',    done: photicDone, color: 'info' },
    { label: 'Sleep EEG Captured',    done: sleepDone,  color: 'primary' },
    { label: 'Eyes Open/Closed',      done: eyesOpen,   color: 'warning' },
    { label: 'Impedance Pass (<5 kΩ)', done: impPass,   color: 'danger' },
  ];

  // Cooperation breakdown
  const coop = {};
  pts.forEach(p => { coop[p.cooperation] = (coop[p.cooperation] || 0) + 1; });

  // Patient state breakdown
  const state = {};
  pts.forEach(p => { state[p.patient_state] = (state[p.patient_state] || 0) + 1; });

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white"><strong>Protocol Completion Rates</strong></div>
        <div className="card-body">
          {protocols.map(pr => (
            <div key={pr.label} className="mb-3">
              <div className="d-flex justify-content-between small mb-1">
                <span className="fw-bold">{pr.label}</span>
                <span>{pr.done}/{total} ({((pr.done / total) * 100).toFixed(0)}%)</span>
              </div>
              <Bar pct={(pr.done / total) * 100} color={pr.color} height={20} />
            </div>
          ))}
        </div>
      </div>

      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header py-2 bg-dark text-white"><strong>Patient Cooperation</strong></div>
            <div className="card-body">
              {Object.entries(coop).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="text-capitalize">{k}</span>
                    <span className="fw-bold">{v} ({((v / total) * 100).toFixed(0)}%)</span>
                  </div>
                  <Bar pct={(v / total) * 100} color={k === 'good' ? 'success' : k === 'fair' ? 'warning' : 'danger'} />
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header py-2 bg-dark text-white"><strong>Patient State During Recording</strong></div>
            <div className="card-body">
              {Object.entries(state).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="text-capitalize">{k}</span>
                    <span className="fw-bold">{v} ({((v / total) * 100).toFixed(0)}%)</span>
                  </div>
                  <Bar pct={(v / total) * 100} color="info" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ defs }) {
  if (!defs) return <div className="text-muted p-3">Loading…</div>;
  const [open, setOpen] = useState(null);

  return (
    <div>
      <div className="mb-2 text-muted small">{defs.title}</div>
      {(defs.sections || []).map((sec, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div
            className="card-header py-2 bg-dark text-white d-flex justify-content-between align-items-center"
            style={{ cursor: 'pointer' }}
            onClick={() => setOpen(open === i ? null : i)}
          >
            <strong>{sec.heading}</strong>
            <span>{open === i ? '▲' : '▼'}</span>
          </div>
          {open === i && (
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <tbody>
                  {(sec.items || []).map((item, j) => (
                    <tr key={j}>
                      <td className="fw-bold small text-nowrap" style={{ width: 220, verticalAlign: 'top', paddingLeft: 12 }}>{item.term}</td>
                      <td className="small text-muted">{item.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function EEGTechnicianDashboard() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/eeg-technician/overview`).then(r => r.json()),
      fetch(`${API}/api/eeg-technician/breakdown`).then(r => r.json()),
      fetch(`${API}/api/eeg-technician/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err)  return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-3">Loading EEG Technician data…</div>;

  const k = ov.kpis || {};

  return (
    <div className="p-3">
      <h3>EEG Technician Dashboard</h3>
      <p className="text-muted">
        Acquisition quality · {k.total_recordings} recordings · {k.total_artifacts_logged} artifacts logged · {k.avg_channels_per_study}-channel 10-20 system
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel ov={ov} />}
      {tab === 'quality'     && <QualityPanel ov={ov} />}
      {tab === 'patients'    && <PatientsPanel bd={bd} />}
      {tab === 'protocol'    && <ProtocolPanel bd={bd} />}
      {tab === 'definitions' && <DefinitionsPanel defs={defs} />}
    </div>
  );
}
