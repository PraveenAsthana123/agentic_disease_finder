'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const pct = (n, total) => total ? ((n / total) * 100).toFixed(1) : '0.0';
const gradeColor = g => ({ Good: 'success', Fair: 'warning', Poor: 'danger' }[g] || 'secondary');
const severityColor = s => ({ mild: 'success', moderate: 'warning', severe: 'danger' }[s] || 'secondary');

const Bar = ({ val, max, color }) => (
  <div className="progress" style={{ height: 8 }}>
    <div className={`progress-bar bg-${color}`} style={{ width: `${Math.min(100, (val / max) * 100)}%` }} />
  </div>
);

export default function SignalQualityDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [ptSort, setPtSort] = useState('patient_id');
  const [ptDir, setPtDir] = useState(1);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/signal-quality/overview`).then(r => r.json()),
      fetch(`${API}/api/signal-quality/breakdown`).then(r => r.json()),
      fetch(`${API}/api/signal-quality/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading signal quality data…</div>;

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'channels', label: 'Channel Quality' },
    { id: 'artifacts', label: 'Artifact Analysis' },
    { id: 'patients', label: 'Per Recording' },
    { id: 'defs', label: 'Definitions' },
  ];

  const sortedPt = bd ? [...(bd.patient_scorecards || [])].sort((a, b) => {
    const av = a[ptSort] ?? '';
    const bv = b[ptSort] ?? '';
    if (av < bv) return -ptDir;
    if (av > bv) return ptDir;
    return 0;
  }) : [];

  const sortBy = col => {
    if (ptSort === col) setPtDir(d => -d);
    else { setPtSort(col); setPtDir(1); }
  };
  const sortIcon = col => ptSort === col ? (ptDir === 1 ? ' ▲' : ' ▼') : '';

  const totalCh = ov.total_channels || 1;

  return (
    <div className="container-fluid py-3">
      <h3 className="mb-1">📡 Signal Quality Dashboard</h3>
      <p className="text-muted small mb-3">
        {ov.total_patients} patients · {ov.total_channels} channel assessments · {ov.total_artifacts} artifact events · {ov.total_recordings} recordings
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <div className="row g-3 mb-4">
            {[
              { label: 'Patients', val: ov.total_patients, sub: `${ov.total_recordings} recordings`, color: 'primary' },
              { label: 'Channels Assessed', val: ov.total_channels, sub: `${ov.channels_per_patient} per patient`, color: 'info' },
              { label: 'Poor Impedance Channels', val: ov.poor_impedance_channels, sub: `avg ${ov.avg_impedance_kohm} kΩ`, color: 'danger' },
              { label: 'Poor SNR Channels', val: ov.poor_snr_channels, sub: `avg ${ov.avg_snr_db} dB`, color: 'warning' },
              { label: 'Artifact Events', val: ov.total_artifacts, sub: `${ov.severe_artifacts} severe`, color: 'secondary' },
              { label: 'Avg Duration', val: ov.avg_recording_duration_min, sub: 'min per recording', color: 'success' },
            ].map(k => (
              <div key={k.label} className="col-6 col-md-4 col-xl-2">
                <div className={`card border-${k.color} h-100`}>
                  <div className="card-body text-center p-2">
                    <div className={`display-6 fw-bold text-${k.color}`}>{k.val}</div>
                    <div className="small fw-semibold">{k.label}</div>
                    <div className="text-muted" style={{ fontSize: '0.72rem' }}>{k.sub}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="row g-3 mb-3">
            {/* Channel quality grade */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Channel Quality Grade</div>
                <div className="card-body">
                  {(ov.quality_grade_distribution || []).map(item => (
                    <div key={item.grade} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold">{item.grade}</span>
                        <span>
                          <span className={`badge bg-${gradeColor(item.grade)} me-1`}>{item.count}</span>
                          <span className="text-muted">{pct(item.count, totalCh)}%</span>
                        </span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div className={`progress-bar bg-${gradeColor(item.grade)}`}
                          style={{ width: `${pct(item.count, totalCh)}%` }} />
                      </div>
                    </div>
                  ))}
                  <p className="text-muted small mb-0">
                    Good = impedance ≤5 kΩ AND SNR ≥10 dB. Poor = impedance &gt;10 kΩ and/or SNR &lt;10 dB.
                  </p>
                </div>
              </div>
            </div>

            {/* Impedance distribution */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Impedance Grade Distribution</div>
                <div className="card-body">
                  {(ov.impedance_distribution || []).map(item => (
                    <div key={item.grade} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.grade}</span>
                        <span className="badge bg-secondary">{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-primary"
                          style={{ width: `${pct(item.count, totalCh)}%` }} />
                      </div>
                    </div>
                  ))}
                  <p className="text-muted small mt-2 mb-0">Avg {ov.avg_impedance_kohm} kΩ · ACNS threshold: &lt;10 kΩ</p>
                </div>
              </div>
            </div>

            {/* SNR distribution */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">SNR Grade Distribution</div>
                <div className="card-body">
                  {(ov.snr_distribution || []).map(item => (
                    <div key={item.grade} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.grade}</span>
                        <span className="badge bg-secondary">{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-info"
                          style={{ width: `${pct(item.count, totalCh)}%` }} />
                      </div>
                    </div>
                  ))}
                  <p className="text-muted small mt-2 mb-0">Avg {ov.avg_snr_db} dB · HFO threshold: ≥20 dB</p>
                </div>
              </div>
            </div>
          </div>

          <div className="row g-3">
            {/* Recording parameters */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Sampling Rate Distribution</div>
                <div className="card-body">
                  {(ov.sampling_rate_distribution || []).map(item => (
                    <div key={item.rate_hz} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.rate_hz} Hz</span>
                        <span className="badge bg-info text-dark">{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-info"
                          style={{ width: `${pct(item.count, ov.total_recordings)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Montage Distribution</div>
                <div className="card-body">
                  {(ov.montage_distribution || []).map(item => (
                    <div key={item.montage} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize">{item.montage}</span>
                        <span className="badge bg-secondary">{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-secondary"
                          style={{ width: `${pct(item.count, ov.total_recordings)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Activation procedures */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Activation Procedures</div>
                <div className="card-body">
                  {(ov.activation_procedures || []).map(item => (
                    <div key={item.procedure} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.procedure}</span>
                        <span>
                          <span className="badge bg-primary me-1">{item.count}</span>
                          <span className="text-muted">{pct(item.count, item.total)}%</span>
                        </span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-primary"
                          style={{ width: `${pct(item.count, item.total)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── CHANNEL QUALITY ── */}
      {tab === 'channels' && (
        <>
          <div className="row g-3 mb-3">
            {/* SNR histogram */}
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">SNR Histogram (dB buckets)</div>
                <div className="card-body">
                  {(bd?.snr_histogram || []).map(item => (
                    <div key={item.bucket} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.bucket}</span>
                        <span className={`badge bg-${item.count > 100 ? 'success' : item.count > 30 ? 'warning' : 'danger'}`}>
                          {item.count}
                        </span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div className={`progress-bar bg-${item.count > 100 ? 'success' : item.count > 30 ? 'warning' : 'danger'}`}
                          style={{ width: `${pct(item.count, totalCh)}%` }} />
                      </div>
                    </div>
                  ))}
                  <p className="text-muted small mb-0">Poor channel threshold: &lt;10 dB. HFO analysis requires ≥20 dB.</p>
                </div>
              </div>
            </div>

            {/* Impedance histogram */}
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">Impedance Histogram (kΩ buckets)</div>
                <div className="card-body">
                  {(bd?.impedance_histogram || []).map(item => (
                    <div key={item.bucket} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.bucket}</span>
                        <span className={`badge bg-${item.bucket.includes('>') ? 'danger' : item.bucket.includes('10') ? 'warning' : 'success'}`}>
                          {item.count}
                        </span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div className={`progress-bar bg-${item.bucket.includes('>') ? 'danger' : item.bucket.includes('10') ? 'warning' : 'success'}`}
                          style={{ width: `${pct(item.count, totalCh)}%` }} />
                      </div>
                    </div>
                  ))}
                  <p className="text-muted small mb-0">ACNS: &lt;5 kΩ = Good; 5-10 = Fair; &gt;10 = Poor (flag for re-prep).</p>
                </div>
              </div>
            </div>
          </div>

          {/* Per-channel poor rate */}
          <div className="card">
            <div className="card-header fw-semibold">Poor Quality Rate by Channel (across all patients)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered table-hover mb-0" style={{ fontSize: '0.8rem' }}>
                  <thead className="table-dark">
                    <tr>
                      <th>Channel</th>
                      <th>Poor Grade</th>
                      <th>Total Recordings</th>
                      <th>Poor Rate</th>
                      <th style={{ width: '40%' }}>Rate Bar</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.channel_poor_rate || []).map(row => (
                      <tr key={row.channel}>
                        <td className="fw-semibold">{row.channel}</td>
                        <td>
                          <span className={`badge bg-${row.poor_count > 5 ? 'danger' : row.poor_count > 2 ? 'warning' : 'success'}`}>
                            {row.poor_count}
                          </span>
                        </td>
                        <td>{row.total}</td>
                        <td className={row.poor_pct > 20 ? 'text-danger fw-bold' : ''}>{row.poor_pct}%</td>
                        <td>
                          <div className="progress" style={{ height: 8 }}>
                            <div className={`progress-bar bg-${row.poor_pct > 20 ? 'danger' : row.poor_pct > 10 ? 'warning' : 'success'}`}
                              style={{ width: `${row.poor_pct}%` }} />
                          </div>
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

      {/* ── ARTIFACT ANALYSIS ── */}
      {tab === 'artifacts' && (
        <>
          <div className="row g-3 mb-3">
            {/* Artifact type */}
            <div className="col-md-5">
              <div className="card h-100">
                <div className="card-header fw-semibold">Artifact Type Distribution ({ov.total_artifacts} events)</div>
                <div className="card-body">
                  {(ov.artifact_type_distribution || []).map(item => (
                    <div key={item.type} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize">{item.type.replace(/_/g, ' ')}</span>
                        <span className="badge bg-secondary">{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-primary"
                          style={{ width: `${pct(item.count, ov.total_artifacts)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Severity */}
            <div className="col-md-3">
              <div className="card h-100">
                <div className="card-header fw-semibold">Severity Distribution</div>
                <div className="card-body">
                  {(ov.artifact_severity_distribution || []).map(item => (
                    <div key={item.severity} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize">{item.severity}</span>
                        <span className={`badge bg-${severityColor(item.severity)}`}>{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div className={`progress-bar bg-${severityColor(item.severity)}`}
                          style={{ width: `${pct(item.count, ov.total_artifacts)}%` }} />
                      </div>
                    </div>
                  ))}
                  {ov.severe_artifacts > 0 && (
                    <div className="alert alert-danger p-2 small mb-0">
                      <strong>{ov.severe_artifacts}</strong> severe artifacts — may require re-recording per IFCN standards.
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Top channels */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Top 10 Most Artifacted Channels</div>
                <div className="card-body">
                  {(bd?.top_artifact_channels || []).map((item, idx) => (
                    <div key={item.channel} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span><span className="text-muted me-1">#{idx + 1}</span>{item.channel}</span>
                        <span className="badge bg-warning text-dark">{item.artifact_count}</span>
                      </div>
                      <div className="progress" style={{ height: 6 }}>
                        <div className="progress-bar bg-warning"
                          style={{ width: `${pct(item.artifact_count, bd.top_artifact_channels[0].artifact_count)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── PER RECORDING ── */}
      {tab === 'patients' && (
        <div className="card">
          <div className="card-header fw-semibold">
            Per-Recording Quality Scorecard ({sortedPt.length} recordings)
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-bordered table-hover mb-0" style={{ fontSize: '0.78rem' }}>
                <thead className="table-dark">
                  <tr>
                    {[
                      ['patient_id', 'Patient'],
                      ['good_channels', 'Good Ch'],
                      ['poor_channels', 'Poor Ch'],
                      ['good_pct', 'Good %'],
                      ['avg_impedance_kohm', 'Avg Imp (kΩ)'],
                      ['avg_snr_db', 'Avg SNR (dB)'],
                      ['artifact_count', 'Artifacts'],
                      ['artifact_duration_sec', 'Art Dur (s)'],
                      ['severe_artifact_count', 'Severe Art'],
                      ['recording_type', 'Type'],
                      ['duration_min', 'Duration (min)'],
                      ['sampling_rate', 'Fs (Hz)'],
                      ['montage', 'Montage'],
                      ['study_date', 'Study Date'],
                    ].map(([col, label]) => (
                      <th key={col} onClick={() => sortBy(col)}
                        style={{ cursor: 'pointer', whiteSpace: 'nowrap' }}>
                        {label}{sortIcon(col)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedPt.map((p, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{p.patient_id}</td>
                      <td className="text-success fw-bold">{p.good_channels}</td>
                      <td className={p.poor_channels > 2 ? 'text-danger fw-bold' : ''}>{p.poor_channels}</td>
                      <td className={p.good_pct < 60 ? 'text-danger' : p.good_pct < 80 ? 'text-warning' : 'text-success'}>
                        {p.good_pct}%
                      </td>
                      <td className={p.avg_impedance_kohm > 10 ? 'text-danger fw-bold' : ''}>{p.avg_impedance_kohm}</td>
                      <td className={p.avg_snr_db < 10 ? 'text-danger fw-bold' : p.avg_snr_db < 20 ? 'text-warning' : 'text-success'}>
                        {p.avg_snr_db}
                      </td>
                      <td>{p.artifact_count}</td>
                      <td>{p.artifact_duration_sec}</td>
                      <td className={p.severe_artifact_count > 0 ? 'text-danger fw-bold' : ''}>
                        {p.severe_artifact_count}
                      </td>
                      <td>{p.recording_type}</td>
                      <td>{p.duration_min ?? '—'}</td>
                      <td>{p.sampling_rate ?? '—'}</td>
                      <td className="text-capitalize">{p.montage}</td>
                      <td className="text-muted">{p.study_date}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'defs' && defs && (
        <div className="row g-3">
          <div className="col-lg-8">
            <div className="card">
              <div className="card-header fw-semibold">Clinical Definitions</div>
              <div className="card-body p-0">
                <table className="table table-sm table-bordered mb-0">
                  <thead className="table-light">
                    <tr><th style={{ width: '28%' }}>Term</th><th>Definition</th></tr>
                  </thead>
                  <tbody>
                    {(defs.terms || []).map(t => (
                      <tr key={t.term}>
                        <td className="fw-semibold">{t.term}</td>
                        <td className="small">{t.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-lg-4">
            <div className="card mb-3">
              <div className="card-header fw-semibold">Abbreviations</div>
              <div className="card-body p-0">
                <table className="table table-sm table-bordered mb-0">
                  <tbody>
                    {Object.entries(defs.abbreviations || {}).map(([abbr, full]) => (
                      <tr key={abbr}>
                        <td className="fw-bold">{abbr}</td>
                        <td className="small">{full}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card">
              <div className="card-header fw-semibold">Data Sources</div>
              <div className="card-body small text-muted">
                {(defs.data_sources || []).map((s, i) => <p key={i} className="mb-1">{s}</p>)}
                <p className="mb-0 mt-1"><strong>Role:</strong> {defs.role}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
