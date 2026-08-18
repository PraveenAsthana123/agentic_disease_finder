'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'montage',    label: 'Montage Comparison' },
  { id: 'localize',   label: 'Localization' },
  { id: 'artifacts',  label: 'Artifacts' },
  { id: 'falseAlarm', label: 'False Alarms' },
  { id: 'propagation',label: 'Propagation' },
  { id: 'sleep',      label: 'Sleep Architecture' },
  { id: 'ictal',      label: 'Ictal vs Interictal' },
  { id: 'badCh',      label: 'Bad Channels' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card text-center shadow-sm border-0">
        <div className="card-body py-2">
          <div className={`h4 mb-0 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted" style={{ fontSize: '0.75rem' }}>{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Badge({ label, variant }) {
  return <span className={`badge bg-${variant || 'secondary'} me-1`}>{label}</span>;
}

/* ── MONTAGE COMPARISON ─────────────────────────────────────── */
function MontageTab() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    fetch(`${API}/api/eeg-viz/montage-comparison`)
      .then(r => r.json()).then(setD).catch(() => setErr('Load error'));
  }, []);
  if (err) return <p className="text-danger">{err}</p>;
  if (!d) return <div className="text-center py-4"><span className="spinner-border text-primary" /></div>;
  const montages = Object.entries(d.montages || {});
  return (
    <div>
      <p className="text-muted small mb-3">File: <code>{d.file}</code> · {d.sfreq} Hz · {d.seconds}s window</p>
      <div className="row">
        {montages.map(([key, m]) => (
          <div key={key} className="col-md-4 mb-3">
            <div className="card h-100 shadow-sm">
              <div className="card-header py-2 bg-primary text-white">
                <strong className="text-capitalize">{key.replace(/_/g,' ')}</strong>
              </div>
              <div className="card-body small">
                <p className="text-muted mb-2">{m.description}</p>
                <table className="table table-sm mb-2">
                  <tbody>
                    <tr><td>Channels</td><td className="fw-bold">{m.n_channels}</td></tr>
                    <tr><td>Mean Amp (µV)</td><td className="fw-bold">{m.mean_amplitude_uv?.toFixed(2)}</td></tr>
                  </tbody>
                </table>
                <div className="small text-muted mb-1 fw-semibold">Band Power</div>
                {Object.entries(m.band_power || {}).map(([band, val]) => (
                  <div key={band} className="mb-1">
                    <div className="d-flex justify-content-between" style={{ fontSize: '0.72rem' }}>
                      <span className="text-capitalize">{band}</span>
                      <span>{(val * 100).toFixed(1)}%</span>
                    </div>
                    <div className="progress" style={{ height: 6 }}>
                      <div className="progress-bar" style={{ width: `${val * 100}%`, backgroundColor: '#1565c0' }} />
                    </div>
                  </div>
                ))}
                {m.example_derivations && (
                  <div className="mt-2">
                    <div className="small text-muted fw-semibold">Example derivations:</div>
                    {m.example_derivations.map((e, i) => (
                      <div key={i} className="font-monospace" style={{ fontSize: '0.68rem' }}>{e}</div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
      <div className="alert alert-info small mt-2">
        Montage re-referencing changes apparent amplitude and localisation — bipolar montages sharpen focal abnormalities; CAR improves spectral analysis.
      </div>
    </div>
  );
}

/* ── SEIZURE LOCALIZATION ───────────────────────────────────── */
function LocalizeTab() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    fetch(`${API}/api/eeg-viz/localization`)
      .then(r => r.json()).then(setD).catch(() => setErr('Load error'));
  }, []);
  if (err) return <p className="text-danger">{err}</p>;
  if (!d) return <div className="text-center py-4"><span className="spinner-border text-primary" /></div>;
  const focus = d.top_focus_channels || [];
  const regions = d.region_summary || {};
  return (
    <div>
      <p className="text-muted small mb-3">
        File: <code>{d.file}</code> · Seizure: {d.seizure_window?.start_s}s – {d.seizure_window?.end_s}s
      </p>
      <div className="row g-3 mb-4">
        <div className="col-md-4">
          <div className="card text-center shadow-sm border-primary">
            <div className="card-body py-3">
              <div className="h4 fw-bold text-primary">{d.localization_verdict || 'Focal'}</div>
              <div className="small text-muted">Localization Verdict</div>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card text-center shadow-sm border-warning">
            <div className="card-body py-3">
              <div className="h4 fw-bold text-warning">{focus.length}</div>
              <div className="small text-muted">Focus Channels</div>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card text-center shadow-sm border-success">
            <div className="card-body py-3">
              <div className="h4 fw-bold text-success">{focus[0]?.ictal_increase_x?.toFixed(1)}×</div>
              <div className="small text-muted">Peak Ictal Increase</div>
            </div>
          </div>
        </div>
      </div>
      <div className="card shadow-sm mb-3">
        <div className="card-header py-2"><strong>Top Focus Channels</strong></div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light"><tr><th>Channel</th><th>Ictal Increase</th><th>Region</th><th>Hemisphere</th></tr></thead>
            <tbody>
              {focus.slice(0,12).map((c, i) => (
                <tr key={i}>
                  <td className="font-monospace small">{c.channel}</td>
                  <td><span className="badge bg-danger">{c.ictal_increase_x?.toFixed(1)}×</span></td>
                  <td className="small text-capitalize">{c.region}</td>
                  <td><Badge label={c.hemisphere} variant={c.hemisphere?.includes('right') ? 'warning' : 'primary'} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {Object.keys(regions).length > 0 && (
        <div className="card shadow-sm">
          <div className="card-header py-2"><strong>Regional Summary</strong></div>
          <div className="card-body">
            {Object.entries(regions).map(([region, vals]) => (
              <div key={region} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span className="text-capitalize fw-semibold">{region}</span>
                  <span>{vals.channels} channels · {vals.mean_increase?.toFixed(1)}× mean</span>
                </div>
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar bg-primary" style={{ width: `${Math.min(vals.mean_increase * 10, 100)}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="alert alert-warning small mt-3">
        Ictal power increase ratio (ictal/interictal). Not a substitute for clinical video-EEG interpretation by a trained epileptologist.
      </div>
    </div>
  );
}

/* ── ARTIFACT REVIEW ────────────────────────────────────────── */
function ArtifactsTab() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    fetch(`${API}/api/eeg-viz/artifacts`)
      .then(r => r.json()).then(setD).catch(() => setErr('Load error'));
  }, []);
  if (err) return <p className="text-danger">{err}</p>;
  if (!d) return <div className="text-center py-4"><span className="spinner-border text-primary" /></div>;
  const counts = d.artifact_type_counts || {};
  const wins = (d.windows || []).slice(0, 30);
  const COLORS = { eye_blink: '#6366f1', muscle: '#ef4444', line_noise: '#f59e0b', movement: '#10b981' };
  return (
    <div>
      <p className="text-muted small mb-3">
        File: <code>{d.file}</code> · {d.n_channels} ch · {d.n_windows} windows × {d.window_s}s
      </p>
      <div className="row g-3 mb-4">
        <KPI label="Clean windows" value={`${d.clean_pct?.toFixed(1)}%`} color="success" />
        <KPI label="Eye blink" value={counts.eye_blink || 0} color="primary" />
        <KPI label="Muscle" value={counts.muscle || 0} color="danger" />
        <KPI label="Movement" value={counts.movement || 0} color="warning" />
      </div>
      <div className="card shadow-sm mb-3">
        <div className="card-header py-2"><strong>Artifact Timeline</strong></div>
        <div className="card-body p-2">
          <div className="d-flex flex-wrap gap-1">
            {wins.map((w, i) => (
              <div key={i} title={`${w.start_s}s: ${w.artifacts.join(', ') || 'clean'}`}
                style={{
                  width: 20, height: 20, borderRadius: 3,
                  backgroundColor: w.clean ? '#22c55e' : (w.artifacts[0] ? (COLORS[w.artifacts[0]] || '#94a3b8') : '#94a3b8'),
                  cursor: 'default'
                }} />
            ))}
          </div>
          <div className="mt-2 d-flex flex-wrap gap-3">
            {Object.entries(COLORS).map(([type, color]) => (
              <span key={type} className="small d-flex align-items-center gap-1">
                <span style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: color, display: 'inline-block' }} />
                {type.replace(/_/g,' ')}
              </span>
            ))}
            <span className="small d-flex align-items-center gap-1">
              <span style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: '#22c55e', display: 'inline-block' }} />
              clean
            </span>
          </div>
        </div>
      </div>
      <div className="alert alert-info small">
        Artifact detection uses heuristic thresholds (power, frequency content) — validate with visual EEG review. Eye blink: high delta amplitude Fp1/Fp2; Muscle: high gamma; Line noise: 50/60 Hz peak.
      </div>
    </div>
  );
}

/* ── FALSE ALARM REVIEW ─────────────────────────────────────── */
function FalseAlarmTab() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    fetch(`${API}/api/eeg-viz/false-alarm`)
      .then(r => r.json()).then(setD).catch(() => setErr('Load error'));
  }, []);
  if (err) return <p className="text-danger">{err}</p>;
  if (!d) return <div className="text-center py-4"><span className="spinner-border text-primary" /></div>;
  const faWindows = d.false_alarm_windows || [];
  const verdictColor = { acceptable: 'success', high: 'warning', critical: 'danger' };
  return (
    <div>
      <p className="text-muted small mb-3">
        File: <code>{d.file}</code> · Recording: {d.recording_hours?.toFixed(2)} h
      </p>
      <div className="row g-3 mb-4">
        <KPI label="Sensitivity" value={`${((d.sensitivity || 0) * 100).toFixed(0)}%`} color="success" sub="seizure detection" />
        <KPI label="True Positives" value={d.true_positive_windows} color="primary" sub="windows" />
        <KPI label="False Alarms" value={d.false_alarms} color="danger" />
        <KPI label="FA / hour" value={d.false_alarms_per_hour?.toFixed(1)} color="warning" />
      </div>
      <div className="card shadow-sm mb-3">
        <div className="card-header py-2 d-flex align-items-center gap-2">
          <strong>Detector Performance</strong>
          <span className={`badge bg-${verdictColor[d.verdict] || 'secondary'} ms-auto`}>{d.verdict}</span>
        </div>
        <div className="card-body">
          <div className="mb-2 small text-muted">Method: {d.detector?.method}</div>
          <div className="mb-2 small">Threshold: <code>{d.detector?.threshold_k}× MAD</code></div>
          <div className="mb-2 small">Annotated seizures: <strong>{d.n_seizures_annotated}</strong> · Detected: <strong>{d.seizures_detected}</strong></div>
        </div>
      </div>
      {faWindows.length > 0 && (
        <div className="card shadow-sm mb-3">
          <div className="card-header py-2"><strong>False Alarm Windows</strong></div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-light"><tr><th>Window</th><th>Time (s)</th></tr></thead>
              <tbody>
                {faWindows.map((w, i) => (
                  <tr key={i}>
                    <td>{w.window}</td>
                    <td className="font-monospace">{w.time_s?.toFixed(1)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <div className="alert alert-info small">{d.note}</div>
    </div>
  );
}

/* ── SEIZURE PROPAGATION ────────────────────────────────────── */
function PropagationTab() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    fetch(`${API}/api/eeg-viz/propagation`)
      .then(r => r.json()).then(setD).catch(() => setErr('Load error'));
  }, []);
  if (err) return <p className="text-danger">{err}</p>;
  if (!d) return <div className="text-center py-4"><span className="spinner-border text-primary" /></div>;
  const onset = d.onset_leaders || [];
  const prop = d.propagation_order || [];
  const hemiColor = (h) => h?.includes('right') ? '#f59e0b' : h?.includes('left') ? '#3b82f6' : '#6b7280';
  return (
    <div>
      <p className="text-muted small mb-3">
        File: <code>{d.file}</code> · Seizure: {d.seizure_window?.start_s}s – {d.seizure_window?.end_s}s · {d.sfreq} Hz
      </p>
      <div className="row g-3 mb-4">
        <KPI label="Onset Leaders" value={onset.length} color="danger" />
        <KPI label="Propagation Steps" value={prop.length} color="warning" />
        <KPI label="Dominant Region" value={onset[0]?.region || '—'} color="primary" />
        <KPI label="Dominant Hemisphere" value={onset[0]?.hemisphere || '—'} color="info" />
      </div>
      <div className="row">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 bg-danger text-white"><strong>Onset Leaders</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Channel</th><th>Onset (s)</th><th>Region</th></tr></thead>
                <tbody>
                  {onset.slice(0,8).map((c, i) => (
                    <tr key={i}>
                      <td className="font-monospace small">{c.channel}</td>
                      <td>{c.onset_s?.toFixed(1)}s</td>
                      <td>
                        <span style={{ color: hemiColor(c.hemisphere) }} className="small">{c.region}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 bg-warning text-dark"><strong>Propagation Order</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>#</th><th>Channel</th><th>Time (s)</th></tr></thead>
                <tbody>
                  {prop.slice(0,8).map((c, i) => (
                    <tr key={i}>
                      <td className="text-muted">{i+1}</td>
                      <td className="font-monospace small">{c.channel}</td>
                      <td>{c.onset_s?.toFixed(1)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      {d.propagation_note && <div className="alert alert-warning small">{d.propagation_note}</div>}
      <div className="alert alert-info small">
        Propagation estimated from first-significant-power epoch in each channel during the ictal window. Onset time is relative to seizure window start. Clinical localisation requires SEEG/iEEG.
      </div>
    </div>
  );
}

/* ── SLEEP ARCHITECTURE ─────────────────────────────────────── */
function SleepTab() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    fetch(`${API}/api/eeg-viz/sleep-architecture`)
      .then(r => r.json()).then(setD).catch(() => setErr('Load error'));
  }, []);
  if (err) return <p className="text-danger">{err}</p>;
  if (!d) return <div className="text-center py-4"><span className="spinner-border text-primary" /></div>;
  const STAGE_COLORS = { W: '#f59e0b', N1: '#93c5fd', N2: '#3b82f6', N3: '#1d4ed8', REM: '#8b5cf6' };
  const stages = d.stages || {};
  return (
    <div>
      <p className="text-muted small mb-3">
        Hypnogram: <code>{d.hypnogram}</code> · Dataset: {d.dataset} · {d.n_recordings} recordings
      </p>
      <div className="row g-3 mb-4">
        <KPI label="Sleep Efficiency" value={`${d.sleep_efficiency_pct?.toFixed(1)}%`} color={d.sleep_efficiency_pct >= 85 ? 'success' : 'warning'} />
        <KPI label="Total Sleep Time" value={`${d.total_sleep_time_min?.toFixed(0)} min`} color="primary" />
        <KPI label="Deep Sleep (N3)" value={`${d.deep_sleep_pct?.toFixed(1)}%`} color="info" />
        <KPI label="REM" value={`${d.rem_pct?.toFixed(1)}%`} color="secondary" />
      </div>
      <div className="card shadow-sm mb-3">
        <div className="card-header py-2 d-flex align-items-center gap-2">
          <strong>Stage Breakdown</strong>
          <span className={`badge ms-auto bg-${d.quality === 'PASS' ? 'success' : 'warning'}`}>{d.quality}</span>
        </div>
        <div className="card-body">
          {Object.entries(stages).map(([stage, vals]) => (
            <div key={stage} className="mb-3">
              <div className="d-flex justify-content-between mb-1">
                <span className="fw-semibold small">{stage}</span>
                <span className="small text-muted">{vals.minutes?.toFixed(0)} min · {vals.pct_of_sleep?.toFixed(1)}%</span>
              </div>
              <div className="progress" style={{ height: 16 }}>
                <div className="progress-bar" style={{
                  width: `${vals.pct_of_sleep}%`,
                  backgroundColor: STAGE_COLORS[stage] || '#94a3b8'
                }}>{vals.pct_of_sleep?.toFixed(1)}%</div>
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="card shadow-sm mb-3">
        <div className="card-body">
          <div className="row text-center">
            <div className="col-6">
              <div className="small text-muted">Stage Transitions</div>
              <div className="fw-bold">{d.stage_transitions}</div>
            </div>
            <div className="col-6">
              <div className="small text-muted">Time in Bed</div>
              <div className="fw-bold">{d.time_in_bed_min?.toFixed(0)} min</div>
            </div>
          </div>
        </div>
      </div>
      {d.flags && d.flags.length > 0 && (
        <div className="alert alert-info small">
          {d.flags.map((f, i) => <div key={i}>• {f}</div>)}
        </div>
      )}
      <div className="alert alert-light small border">{d.note}</div>
    </div>
  );
}

/* ── ICTAL VS INTERICTAL ────────────────────────────────────── */
function IctalTab() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    fetch(`${API}/api/eeg-viz/ictal-interictal`)
      .then(r => r.json()).then(setD).catch(() => setErr('Load error'));
  }, []);
  if (err) return <p className="text-danger">{err}</p>;
  if (!d) return <div className="text-center py-4"><span className="spinner-border text-primary" /></div>;
  const bp = d.band_power || {};
  const ictal = bp.ictal || {};
  const interictal = bp.interictal || {};
  const sig = d.ictal_signature || {};
  const BAND_COLORS = { delta: '#6366f1', theta: '#8b5cf6', alpha: '#06b6d4', beta: '#10b981', gamma: '#f59e0b' };
  const BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma'];
  return (
    <div>
      <p className="text-muted small mb-3">
        File: <code>{d.file}</code> · {d.sfreq} Hz · Seizure: {d.seizure_window?.start_s}s – {d.seizure_window?.end_s}s ({d.seizure_window?.duration_s}s)
      </p>
      <div className="row g-3 mb-4">
        <KPI label="Annotated Files" value={d.annotated_files} color="primary" />
        <KPI label="Delta Shift" value={`${sig.delta_shift >= 0 ? '+' : ''}${(sig.delta_shift * 100)?.toFixed(1)}%`}
          color={sig.delta_shift > 0 ? 'danger' : 'success'} sub="ictal vs interictal" />
        <KPI label="Alpha Shift" value={`${sig.alpha_shift >= 0 ? '+' : ''}${(sig.alpha_shift * 100)?.toFixed(1)}%`}
          color={sig.alpha_shift < 0 ? 'warning' : 'info'} />
        <KPI label="Recording" value={`${(d.recording_s / 60)?.toFixed(0)} min`} color="secondary" />
      </div>
      <div className="card shadow-sm mb-3">
        <div className="card-header py-2"><strong>Ictal Signature</strong></div>
        <div className="card-body">
          <div className="alert alert-danger mb-3">{sig.verdict}</div>
          <div className="row">
            {BANDS.map(band => {
              const ict = ictal[band] || 0;
              const inter = interictal[band] || 0;
              return (
                <div key={band} className="col-md mb-3">
                  <div className="fw-semibold small text-capitalize mb-1" style={{ color: BAND_COLORS[band] }}>{band}</div>
                  <div className="d-flex gap-1 align-items-end" style={{ height: 60 }}>
                    <div style={{
                      width: 18, backgroundColor: '#94a3b8',
                      height: `${inter * 100}px`, alignSelf: 'flex-end'
                    }} title={`Interictal: ${(inter*100).toFixed(1)}%`} />
                    <div style={{
                      width: 18, backgroundColor: BAND_COLORS[band],
                      height: `${ict * 100}px`, alignSelf: 'flex-end'
                    }} title={`Ictal: ${(ict*100).toFixed(1)}%`} />
                  </div>
                  <div style={{ fontSize: '0.6rem', color: '#6b7280' }}>
                    <div>Inter: {(inter*100).toFixed(1)}%</div>
                    <div>Ictal: {(ict*100).toFixed(1)}%</div>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="small text-muted mt-2">Grey = interictal · Coloured = ictal · Height = relative band power</div>
        </div>
      </div>
      {(d.available_files || []).length > 0 && (
        <div className="card shadow-sm">
          <div className="card-header py-2"><strong>Annotated Files ({d.annotated_files})</strong></div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-light"><tr><th>File</th><th>Seizures</th></tr></thead>
              <tbody>
                {[...new Map((d.available_files||[]).map(f=>[f.file,f])).values()].slice(0,10).map((f,i) => (
                  <tr key={i}>
                    <td className="font-monospace small">{f.file}</td>
                    <td><Badge label={f.n_seizures} variant="danger" /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── BAD CHANNELS ───────────────────────────────────────────── */
function BadChannelsTab() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    fetch(`${API}/api/eeg-viz/bad-channels`)
      .then(r => r.json()).then(setD).catch(() => setErr('Load error'));
  }, []);
  if (err) return <p className="text-danger">{err}</p>;
  if (!d) return <div className="text-center py-4"><span className="spinner-border text-primary" /></div>;
  const channels = d.channels || [];
  const bad = channels.filter(c => c.verdict !== 'good');
  const criteria = d.criteria || {};
  return (
    <div>
      <p className="text-muted small mb-3">
        File: <code>{d.file}</code> · {d.n_channels} channels · {d.seconds_analyzed?.toFixed(0)}s analysed · {d.sfreq} Hz
      </p>
      <div className="row g-3 mb-4">
        <KPI label="Bad Channels" value={bad.length}
          color={bad.length === 0 ? 'success' : 'danger'} sub={`of ${channels.length}`} />
        <KPI label="Good Channels" value={channels.length - bad.length} color="success" />
        <KPI label="Methods" value={d.methods?.length || 3} color="secondary" sub="detection algorithms" />
      </div>
      {Object.keys(criteria).length > 0 && (
        <div className="card shadow-sm mb-3">
          <div className="card-header py-2"><strong>Detection Criteria</strong></div>
          <div className="card-body small">
            {Object.entries(criteria).map(([k, v]) => (
              <div key={k} className="d-flex gap-2 mb-1">
                <span className="text-muted text-capitalize">{k.replace(/_/g,' ')}:</span>
                <span className="fw-semibold font-monospace">{typeof v === 'number' ? v.toFixed(3) : String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="card shadow-sm">
        <div className="card-header py-2"><strong>Channel QC Table</strong></div>
        <div className="card-body p-0">
          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light sticky-top">
                <tr>
                  <th>Channel</th>
                  <th>Std (µV)</th>
                  <th>P2P (µV)</th>
                  <th>Flat ratio</th>
                  <th>Line noise</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {channels.map((c, i) => (
                  <tr key={i} className={c.verdict !== 'good' ? 'table-danger' : ''}>
                    <td className="font-monospace small">{c.channel}</td>
                    <td>{c.std_uv?.toFixed(1)}</td>
                    <td>{c.p2p_uv?.toFixed(1)}</td>
                    <td>{c.flat_ratio?.toFixed(3)}</td>
                    <td>{c.line_noise_rel?.toFixed(3)}</td>
                    <td>
                      <span className={`badge bg-${c.verdict === 'good' ? 'success' : 'danger'}`}>
                        {c.verdict}
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
  );
}

const TAB_COMPONENTS = {
  montage:    MontageTab,
  localize:   LocalizeTab,
  artifacts:  ArtifactsTab,
  falseAlarm: FalseAlarmTab,
  propagation:PropagationTab,
  sleep:      SleepTab,
  ictal:      IctalTab,
  badCh:      BadChannelsTab,
};

/* ── MAIN PAGE ──────────────────────────────────────────────── */
export default function EegVizPage() {
  const [tab, setTab] = useState('montage');
  const ActiveTab = TAB_COMPONENTS[tab] || MontageTab;

  return (
    <div>
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: '#0d47a1' }}>
            EEG Visualization Platform
          </h4>
          <div className="small text-muted">
            Montage · Localization · Artifacts · False Alarms · Propagation · Sleep · Ictal/Interictal · Channel QC
          </div>
        </div>
        <div className="ms-auto">
          <a href="/eeg-viewer" className="btn btn-outline-primary btn-sm me-2">Waveform Viewer →</a>
          <span className="badge" style={{ backgroundColor: '#0d47a1' }}>8 sub-dashboards</span>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3 flex-wrap">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active fw-semibold' : ''}`}
              onClick={() => setTab(t.id)}
              style={tab === t.id ? { color: '#0d47a1', borderBottomColor: '#0d47a1' } : {}}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      <ActiveTab />
    </div>
  );
}
