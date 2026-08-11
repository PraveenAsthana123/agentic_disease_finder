'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const BAND_COLOR = {
  delta: 'danger',
  theta: 'warning',
  alpha: 'success',
  beta: 'primary',
  gamma: 'info',
};

const ART_COLOR = {
  eye_blink: 'warning',
  muscle: 'danger',
  movement: 'secondary',
  ECG: 'info',
  electrode_pop: 'dark',
  sweat: 'primary',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-2 mb-2">
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

function HBar({ items, labelKey, valueKey, colorKey, maxVal }) {
  if (!items || !items.length) return null;
  const mx = maxVal || Math.max(...items.map(i => i[valueKey] || 0));
  return (
    <div>
      {items.map((item, idx) => (
        <div key={idx} className="d-flex align-items-center mb-1">
          <div className="text-end small me-2" style={{ width: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {item[labelKey]}
          </div>
          <div className="flex-grow-1">
            <div className="progress" style={{ height: 20 }}>
              <div
                className={`progress-bar bg-${colorKey ? (ART_COLOR[item[labelKey]] || 'primary') : 'primary'}`}
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

// ── PSD Chart: stacked horizontal band power per channel ──
function PSDChart({ channels }) {
  if (!channels || !channels.length) return <p className="text-muted">No PSD data.</p>;
  const keys = ['delta', 'theta', 'alpha', 'beta', 'gamma'];
  return (
    <div className="table-responsive">
      <table className="table table-sm table-bordered mb-0" style={{ minWidth: 600 }}>
        <thead className="table-dark">
          <tr>
            <th>Channel</th>
            <th>SNR (dB)</th>
            {['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'].map(b => (
              <th key={b} className="text-center">{b}</th>
            ))}
            <th>Band Stacks</th>
          </tr>
        </thead>
        <tbody>
          {channels.map((ch, i) => (
            <tr key={i}>
              <td className="fw-semibold">{ch.channel}</td>
              <td>{ch.avg_snr_db}</td>
              {keys.map(k => (
                <td key={k} className="text-center">
                  <span className={`badge bg-${BAND_COLOR[k]}`}>{(ch[k] * 100).toFixed(1)}%</span>
                </td>
              ))}
              <td>
                <div className="d-flex" style={{ height: 18 }}>
                  {keys.map(k => (
                    <div
                      key={k}
                      title={`${k}: ${(ch[k] * 100).toFixed(1)}%`}
                      className={`bg-${BAND_COLOR[k]}`}
                      style={{ width: `${ch[k] * 100}%`, minWidth: 1 }}
                    />
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Spectrogram: time-bin × band energy grid ──
function SpectrogramGrid({ matrix }) {
  if (!matrix || !matrix.length) return <p className="text-muted">No spectrogram data.</p>;
  const keys = ['delta', 'theta', 'alpha', 'beta', 'gamma'];
  const maxVal = Math.max(...matrix.flatMap(row => keys.map(k => row[k] || 0)));
  function cellColor(val) {
    const pct = maxVal ? val / maxVal : 0;
    const r = Math.round(200 * pct);
    const b = Math.round(255 * (1 - pct));
    return `rgb(${r},50,${b})`;
  }
  return (
    <div>
      <p className="text-muted small mb-2">Colour = relative band energy (blue=low, red=high). Artifact bursts inflate delta/theta.</p>
      <div className="table-responsive">
        <table className="table table-sm table-bordered mb-0" style={{ minWidth: 500 }}>
          <thead className="table-dark">
            <tr>
              <th>Time Bin</th>
              <th>Artifacts</th>
              {['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'].map(b => (
                <th key={b} className="text-center">{b}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td className="fw-semibold">{row.time_bin}</td>
                <td>{row.n_artifacts}</td>
                {keys.map(k => (
                  <td key={k} style={{ background: cellColor(row[k]), color: '#fff', textAlign: 'center', fontWeight: 600 }}>
                    {row[k]?.toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Event Timeline ──
function EventTimeline({ events }) {
  if (!events || !events.length) return <p className="text-muted">No events.</p>;
  const maxSz = Math.max(...events.map(e => e.seizures || 0));
  const maxAr = Math.max(...events.map(e => e.artifacts || 0));
  return (
    <div className="table-responsive">
      <table className="table table-sm table-hover">
        <thead className="table-dark">
          <tr><th>Date</th><th>Seizure Events</th><th>Artifact Events</th></tr>
        </thead>
        <tbody>
          {events.map((ev, i) => (
            <tr key={i}>
              <td className="fw-semibold">{ev.date}</td>
              <td>
                {ev.seizures > 0 ? (
                  <div className="d-flex align-items-center gap-2">
                    <div className="progress flex-grow-1" style={{ height: 16 }}>
                      <div className="progress-bar bg-danger" style={{ width: `${maxSz ? (ev.seizures / maxSz) * 100 : 0}%` }}>
                        {ev.seizures}
                      </div>
                    </div>
                  </div>
                ) : <span className="text-muted">—</span>}
              </td>
              <td>
                {ev.artifacts > 0 ? (
                  <div className="progress" style={{ height: 16 }}>
                    <div className="progress-bar bg-warning" style={{ width: `${maxAr ? (ev.artifacts / maxAr) * 100 : 0}%` }}>
                      {ev.artifacts}
                    </div>
                  </div>
                ) : <span className="text-muted">—</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Spike Overlay ──
function SpikeOverlay({ patterns, lateralization }) {
  return (
    <div className="row">
      <div className="col-md-7">
        <h6 className="fw-semibold mb-2">Spike / Sharp-Wave Pattern Distribution</h6>
        <p className="text-muted small">From EEG interpretation reports (seizure_metadata). Patterns with spike/wave/sharp-wave/hypsarrhythmia morphology.</p>
        {patterns && patterns.map((p, i) => (
          <div key={i} className="d-flex align-items-center mb-1">
            <div className="me-2 text-end small" style={{ width: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {p.pattern}
            </div>
            <div className="flex-grow-1">
              <div className="progress" style={{ height: 18 }}>
                <div className="progress-bar bg-danger"
                  style={{ width: `${patterns[0]?.count ? (p.count / patterns[0].count) * 100 : 0}%` }}>
                  {p.count}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      <div className="col-md-5">
        <h6 className="fw-semibold mb-2">Lateralization</h6>
        {lateralization && lateralization.map((l, i) => (
          <div key={i} className="d-flex align-items-center mb-1">
            <div className="me-2 text-end small" style={{ width: 180 }}>{l.lateralization}</div>
            <div className="flex-grow-1">
              <div className="progress" style={{ height: 18 }}>
                <div className="progress-bar bg-warning"
                  style={{ width: `${lateralization[0]?.count ? (l.count / lateralization[0].count) * 100 : 0}%` }}>
                  {l.count}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Artifact Overlay (channel × type heatmap) ──
function ArtifactOverlay({ overlay, artTypes }) {
  if (!overlay || !overlay.length) return <p className="text-muted">No artifact data.</p>;
  const maxVal = Math.max(...overlay.flatMap(row => (artTypes || []).map(t => row[t] || 0)));
  function cellStyle(val) {
    const pct = maxVal ? val / maxVal : 0;
    if (pct === 0) return { background: '#f8f9fa', color: '#adb5bd' };
    const alpha = 0.2 + 0.8 * pct;
    return { background: `rgba(220,53,69,${alpha})`, color: pct > 0.5 ? '#fff' : '#212529', fontWeight: 600 };
  }
  return (
    <div>
      <p className="text-muted small mb-2">Heatmap: artifact count per channel × type. Darker red = more artifacts.</p>
      <div className="table-responsive">
        <table className="table table-sm table-bordered mb-0" style={{ minWidth: 500 }}>
          <thead className="table-dark">
            <tr>
              <th>Channel</th>
              <th>Total</th>
              {(artTypes || []).map(t => <th key={t} className="text-center">{t}</th>)}
            </tr>
          </thead>
          <tbody>
            {overlay.map((row, i) => (
              <tr key={i}>
                <td className="fw-semibold">{row.channel}</td>
                <td><span className="badge bg-secondary">{row.total}</span></td>
                {(artTypes || []).map(t => (
                  <td key={t} style={{ ...cellStyle(row[t] || 0), textAlign: 'center' }}>
                    {row[t] || 0}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function EEGClinicalPanel() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/eeg-clinical-panel/overview`).then(r => r.json()),
      fetch(`${API}/api/eeg-clinical-panel/breakdown`).then(r => r.json()),
      fetch(`${API}/api/eeg-clinical-panel/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const TABS = [
    { id: 'overview',   label: '📊 Overview' },
    { id: 'psd',        label: '📈 PSD Graph' },
    { id: 'spectrogram',label: '🌈 Spectrogram' },
    { id: 'timeline',   label: '📅 Event Timeline' },
    { id: 'spikes',     label: '⚡ Spike Overlay' },
    { id: 'artifacts',  label: '🔍 Artifact Overlay' },
    { id: 'definitions',label: '📖 Definitions' },
  ];

  if (loading) return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center gap-2">
        <div className="spinner-border spinner-border-sm text-primary" />
        <span>Loading EEG Clinical Signal Panel…</span>
      </div>
    </div>
  );
  if (error) return (
    <div className="container-fluid py-4">
      <div className="alert alert-danger">Failed to load: {error}</div>
    </div>
  );

  const ov = overview || {};
  const bk = breakdown || {};
  const df = definitions || {};

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: 28 }}>🧠</span>
        <div>
          <h4 className="mb-0 fw-bold">EEG Clinical Signal Panel</h4>
          <p className="text-muted mb-0 small">
            P0 EEG visualisation suite — PSD Graph · Spectrogram · Event Timeline · Spike/Sharp-Wave Overlay · Artifact Overlay
          </p>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        {(ov.kpis || []).map((k, i) => (
          <KPI key={i} label={k.label} value={k.value} color={k.color} />
        ))}
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

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold bg-primary text-white">Average Band Power</div>
              <div className="card-body">
                {(ov.avg_band_power || []).map((b, i) => (
                  <div key={i} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{b.band}</span>
                      <span className={`badge bg-${BAND_COLOR[b.key]}`}>{(b.power * 100).toFixed(1)}%</span>
                    </div>
                    <div className="progress" style={{ height: 12 }}>
                      <div className={`progress-bar bg-${BAND_COLOR[b.key]}`} style={{ width: `${b.power * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold bg-warning text-dark">Artifact Type Distribution</div>
              <div className="card-body">
                <HBar items={ov.artifact_type_distribution || []} labelKey="type" valueKey="count" colorKey="type" />
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold bg-danger text-white">Spike Pattern Distribution</div>
              <div className="card-body">
                {(ov.top_spike_patterns || []).map((p, i) => (
                  <div key={i} className="mb-1">
                    <div className="d-flex justify-content-between small">
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 200 }}>{p.pattern}</span>
                      <span className="badge bg-danger">{p.count}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Artifact Severity</div>
              <div className="card-body">
                {(ov.artifact_severity_distribution || []).map((s, i) => (
                  <div key={i} className="d-flex justify-content-between small mb-1">
                    <span className="text-capitalize">{s.severity}</span>
                    <span className={`badge bg-${s.severity === 'severe' ? 'danger' : s.severity === 'moderate' ? 'warning' : 'success'}`}>{s.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Seizure Event Severity</div>
              <div className="card-body">
                {(ov.seizure_severity_distribution || []).map((s, i) => (
                  <div key={i} className="d-flex justify-content-between small mb-1">
                    <span>{s.severity}</span>
                    <span className="badge bg-secondary">{s.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Sampling Rate Distribution</div>
              <div className="card-body">
                {(ov.sampling_rate_distribution || []).map((s, i) => (
                  <div key={i} className="d-flex justify-content-between small mb-1">
                    <span>{s.rate}</span>
                    <span className="badge bg-info text-dark">{s.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── PSD Graph ── */}
      {tab === 'psd' && (
        <div>
          <div className="alert alert-info small mb-3">
            <strong>Power Spectral Density (PSD)</strong> — relative band power per scalp electrode,
            derived from channel SNR (dB). Stacked bars show the proportion of total power in each
            frequency band. High delta/theta proportion indicates pathological slow-wave activity.
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold bg-primary text-white">Per-Channel Band Power (19 channels)</div>
            <div className="card-body p-2">
              <PSDChart channels={bk.psd_channels || []} />
            </div>
          </div>
          <div className="mt-3 d-flex flex-wrap gap-2">
            {['delta', 'theta', 'alpha', 'beta', 'gamma'].map((k, i) => (
              <span key={k} className={`badge bg-${BAND_COLOR[k]} fs-6`}>
                {['Δ Delta', 'θ Theta', 'α Alpha', 'β Beta', 'γ Gamma'][i]}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ── Spectrogram ── */}
      {tab === 'spectrogram' && (
        <div>
          <div className="alert alert-secondary small mb-3">
            <strong>Time-Frequency Spectrogram</strong> — relative EEG band energy across six 10-minute
            recording windows. Artifact count per bin drives delta/theta energy (noise floor).
            Colour: blue = low energy, red = high energy.
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Spectrogram: Time Bin × Band Energy</div>
            <div className="card-body p-2">
              <SpectrogramGrid matrix={bk.spectrogram_matrix || []} />
            </div>
          </div>
        </div>
      )}

      {/* ── Event Timeline ── */}
      {tab === 'timeline' && (
        <div>
          <div className="alert alert-warning small mb-3">
            <strong>Seizure & Artifact Event Timeline</strong> — clinical seizure events (patient diary)
            overlaid with EEG artifact events (technician annotations) on a shared date axis.
            Enables correlation between clinical episodes and recording-quality degradation.
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Event Timeline ({(bk.event_timeline || []).length} active days)</div>
            <div className="card-body p-2">
              <EventTimeline events={bk.event_timeline || []} />
            </div>
          </div>
        </div>
      )}

      {/* ── Spike Overlay ── */}
      {tab === 'spikes' && (
        <div>
          <div className="alert alert-danger small mb-3">
            <strong>Interictal Epileptiform Discharge (IED) Overlay</strong> — distribution of spike,
            sharp-wave, spike-and-wave, hypsarrhythmia, and frontal spike patterns from EEG interpretation
            reports. IED morphology guides seizure focus localisation and surgical candidacy assessment.
          </div>
          <div className="card shadow-sm">
            <div className="card-body">
              <SpikeOverlay
                patterns={bk.spike_pattern_counts || []}
                lateralization={bk.spike_lateralization || []}
              />
            </div>
          </div>
        </div>
      )}

      {/* ── Artifact Overlay ── */}
      {tab === 'artifacts' && (
        <div>
          <div className="alert alert-warning small mb-3">
            <strong>Artifact Channel × Type Heatmap</strong> — which scalp channels are most affected
            by each artifact type. Guides electrode re-application and signal rejection before AI inference.
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Artifact Overlay ({(bk.artifact_overlay || []).length} channels affected)</div>
            <div className="card-body p-2">
              <ArtifactOverlay
                overlay={bk.artifact_overlay || []}
                artTypes={bk.artifact_types || []}
              />
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && (
        <div>
          {(df.panels || []).map((p, i) => (
            <div key={i} className="card shadow-sm mb-3">
              <div className="card-header fw-semibold bg-dark text-white">
                {p.panel} — {p.full_name}
              </div>
              <div className="card-body">
                <p className="mb-2">{p.description}</p>
                {p.standard && <p className="text-muted small mb-2"><strong>Standard:</strong> {p.standard}</p>}
                {p.bands && (
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-dark"><tr><th>Band</th><th>Clinical Significance</th></tr></thead>
                    <tbody>
                      {p.bands.map((b, j) => (
                        <tr key={j}><td className="fw-semibold">{b.band}</td><td>{b.significance}</td></tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {p.discharge_types && (
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-dark"><tr><th>Discharge Type</th><th>Duration</th><th>Significance</th></tr></thead>
                    <tbody>
                      {p.discharge_types.map((d, j) => (
                        <tr key={j}><td className="fw-semibold">{d.type}</td><td>{d.duration}</td><td>{d.significance}</td></tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {p.artifact_types && (
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-dark"><tr><th>Artifact Type</th><th>Affected Channels</th><th>Cause</th></tr></thead>
                    <tbody>
                      {p.artifact_types.map((a, j) => (
                        <tr key={j}><td className="fw-semibold">{a.type}</td><td>{a.channels}</td><td>{a.cause}</td></tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {p.interpretation && (
                  <ul className="mb-0">
                    {p.interpretation.map((line, j) => <li key={j} className="small">{line}</li>)}
                  </ul>
                )}
              </div>
            </div>
          ))}
          {(df.data_sources || []).length > 0 && (
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Data Sources</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-dark"><tr><th>Table</th><th>Rows</th><th>Used For</th></tr></thead>
                  <tbody>
                    {df.data_sources.map((s, i) => (
                      <tr key={i}>
                        <td><code>{s.source}</code></td>
                        <td>{s.rows}</td>
                        <td className="small">{s.use}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
