'use client';
import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const BAND_COLORS = {
  delta: '#6366f1',
  theta: '#22c55e',
  alpha: '#f59e0b',
  beta:  '#ef4444',
  gamma: '#a855f7',
};

function KPI({ label, value, color, icon, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className={`card border-${color || 'primary'} border-2 h-100`}>
        <div className="card-body text-center py-2 px-2">
          <div style={{ fontSize: '1.4rem' }}>{icon}</div>
          <div className={`fw-bold fs-5 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function TraceViewer({ traces, timeArr, sfreq }) {
  const [visibleN, setVisibleN] = useState(8);
  if (!traces || !traces.length) return null;

  const shown = traces.slice(0, visibleN);
  const canvasH = 60;
  const W = 900;
  const totalH = shown.length * canvasH;
  const gap = 4;

  // normalise each channel's amplitude to ±(canvasH/2 - gap)
  const svgPaths = shown.map((ch, idx) => {
    const uv = ch.uv || [];
    if (!uv.length) return null;
    const min = Math.min(...uv);
    const max = Math.max(...uv);
    const range = max - min || 1;
    const yScale = (canvasH - gap * 2) / range;
    const yBase = idx * canvasH + canvasH / 2;

    const pts = uv.map((v, i) => {
      const x = (i / (uv.length - 1)) * W;
      const y = yBase - (v - min - range / 2) * yScale;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    return {
      label: ch.channel,
      path: `M ${pts.join(' L ')}`,
      yBase,
      yMin: yBase + canvasH / 2 - gap,
      yMax: yBase - canvasH / 2 + gap,
      minUv: min.toFixed(1),
      maxUv: max.toFixed(1),
    };
  });

  const duration = timeArr && timeArr.length > 1 ? timeArr[timeArr.length - 1] : 10;

  return (
    <div className="card shadow-sm">
      <div className="card-header py-2 d-flex align-items-center gap-2">
        <strong>&#x1f4c9; Multi-Channel EEG Traces</strong>
        <span className="badge bg-secondary ms-2">{shown.length} / {traces.length} channels</span>
        <div className="ms-auto d-flex gap-2 align-items-center">
          <label className="form-label mb-0 small">Show:</label>
          <select
            className="form-select form-select-sm"
            style={{ width: 80 }}
            value={visibleN}
            onChange={e => setVisibleN(Number(e.target.value))}
          >
            {[4, 8, 12, 16, 24].map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>
      </div>
      <div className="card-body p-2" style={{ overflowX: 'auto' }}>
        <div style={{ position: 'relative', minWidth: 700 }}>
          <svg
            width="100%"
            viewBox={`0 0 ${W} ${totalH}`}
            style={{ display: 'block', background: '#0f172a', borderRadius: 6 }}
          >
            {/* time grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map(f => (
              <line
                key={f}
                x1={f * W} y1={0} x2={f * W} y2={totalH}
                stroke="#1e293b" strokeWidth={1}
              />
            ))}
            {/* channel separator lines */}
            {shown.map((_, i) => (
              <line
                key={i}
                x1={0} y1={i * canvasH} x2={W} y2={i * canvasH}
                stroke="#1e293b" strokeWidth={0.5}
              />
            ))}
            {/* traces */}
            {svgPaths.map((p, i) => p && (
              <g key={i}>
                <path
                  d={p.path}
                  fill="none"
                  stroke="#22d3ee"
                  strokeWidth={0.8}
                  opacity={0.9}
                />
                {/* channel label */}
                <text
                  x={4}
                  y={p.yBase - 2}
                  fill="#94a3b8"
                  fontSize={9}
                  fontFamily="monospace"
                >
                  {p.label}
                </text>
              </g>
            ))}
          </svg>
          {/* time axis */}
          <div className="d-flex justify-content-between" style={{ paddingLeft: 4, paddingRight: 4 }}>
            {[0, 0.25, 0.5, 0.75, 1].map(f => (
              <span key={f} className="text-muted" style={{ fontSize: '0.65rem' }}>
                {(f * duration).toFixed(1)}s
              </span>
            ))}
          </div>
        </div>
        <div className="text-muted mt-1" style={{ fontSize: '0.65rem' }}>
          Vertical scale: auto-normalised per channel. Signal in µV. Sampling rate: {sfreq} Hz.
        </div>
      </div>
    </div>
  );
}

function LateralizationPanel({ lat }) {
  if (!lat || !lat.available) return null;
  const focus = lat.focus || '';
  const idx   = lat.overall_index?.toFixed(3);
  const bands = lat.by_band || [];

  return (
    <div className="card shadow-sm h-100">
      <div className="card-header py-2 fw-semibold">&#x1f9e0; Band Lateralization</div>
      <div className="card-body pb-2">
        <div className="mb-2 d-flex align-items-center gap-2 flex-wrap">
          <span className={`badge ${focus.includes('Left') ? 'bg-info' : focus.includes('Right') ? 'bg-warning text-dark' : 'bg-secondary'} fs-6`}>
            {focus}
          </span>
          <span className="text-muted small">Asymmetry index: <strong>{idx}</strong></span>
        </div>
        <table className="table table-sm table-bordered mb-1">
          <thead className="table-dark">
            <tr><th>Band</th><th>AI</th><th>Direction</th></tr>
          </thead>
          <tbody>
            {bands.map(b => (
              <tr key={b.band}>
                <td>
                  <span
                    style={{
                      display: 'inline-block',
                      width: 10,
                      height: 10,
                      borderRadius: 2,
                      backgroundColor: BAND_COLORS[b.band] || '#94a3b8',
                      marginRight: 4,
                    }}
                  />
                  {b.band}
                </td>
                <td className="font-monospace">{b.asymmetry_index?.toFixed(3)}</td>
                <td>
                  <span className={`badge ${b.lateralization === 'Left' ? 'bg-info' : b.lateralization === 'Right' ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                    {b.lateralization}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-muted" style={{ fontSize: '0.65rem' }}>
          {lat.basis}
        </div>
      </div>
    </div>
  );
}

function SpikePanel({ spikes }) {
  if (!spikes || !spikes.available) return null;
  const top = spikes.top_channels || [];
  return (
    <div className="card shadow-sm h-100">
      <div className="card-header py-2 fw-semibold">&#x26a1; Spike Detection</div>
      <div className="card-body pb-2">
        <div className="d-flex gap-3 mb-2 flex-wrap">
          <div className="text-center">
            <div className="fw-bold fs-5 text-danger">{spikes.total_spikes}</div>
            <div className="text-muted small">Total spikes</div>
          </div>
          <div className="text-center">
            <div className="fw-bold fs-5 text-warning">{spikes.rate_per_min?.toFixed(0)}</div>
            <div className="text-muted small">Rate / min</div>
          </div>
        </div>
        <table className="table table-sm table-hover mb-1">
          <thead className="table-dark">
            <tr><th>Channel</th><th>Spikes</th><th>Rate/min</th></tr>
          </thead>
          <tbody>
            {top.slice(0, 8).map((c, i) => (
              <tr key={i} className={i === 0 ? 'table-danger' : ''}>
                <td className="font-monospace">{c.channel}</td>
                <td className="fw-semibold">{c.spikes}</td>
                <td className="font-monospace">{c.rate_per_min?.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-muted" style={{ fontSize: '0.65rem' }}>
          Spike detection: amplitude + slope threshold on 1s windows.
        </div>
      </div>
    </div>
  );
}

export default function EegVizRawPage() {
  const [overview, setOverview]   = useState(null);
  const [traces,   setTraces]     = useState(null);
  const [recs,     setRecs]       = useState(null);
  const [err,      setErr]        = useState(null);
  const [loading,  setLoading]    = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/eeg-viz`).then(r => r.json()),
      fetch(`${API}/api/eeg-viz/traces`).then(r => r.json()),
      fetch(`${API}/api/eeg-viz/recordings`).then(r => r.json()),
    ])
      .then(([ov, tr, rc]) => {
        setOverview(ov);
        setTraces(tr);
        setRecs(rc);
        setLoading(false);
      })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  useEffect(() => { load(); }, [load]);

  if (err) return (
    <div className="container-fluid py-3">
      <div className="alert alert-danger">Error loading EEG data: {err}</div>
    </div>
  );

  if (loading || !overview) return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" />
      <div className="mt-2 text-muted small">Loading EEG viewer…</div>
    </div>
  );

  if (!overview.available) return (
    <div className="container-fluid py-3">
      <div className="alert alert-warning">
        <strong>No EEG recordings found.</strong> {overview.error || 'Upload an EDF file to use this dashboard.'}
      </div>
    </div>
  );

  const lat   = overview.lateralization || {};
  const spks  = overview.spikes || {};
  const traceList = traces?.traces || [];
  const timeArr   = traces?.time_s || [];
  const nRecs     = recs?.n_total || 0;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-start gap-2 mb-3 flex-wrap">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: '#0ea5e9' }}>
            &#x1f4c9; Raw EEG Viewer
          </h4>
          <div className="small text-muted">
            Multi-channel waveform viewer — CHB-MIT EDF · Real signal · {overview.n_channels} channels · {overview.sfreq} Hz
          </div>
        </div>
        <div className="ms-auto d-flex gap-2 align-items-center flex-wrap">
          <button className="btn btn-outline-primary btn-sm" onClick={load} disabled={loading}>
            &#x21bb; Refresh
          </button>
          <Link href="/eeg-viz" className="btn btn-outline-secondary btn-sm">
            &#x2190; EEG Viz Platform
          </Link>
        </div>
      </div>

      {/* File info bar */}
      <div className="alert alert-light border mb-3 py-2 px-3">
        <span className="small">
          <strong>File:</strong> <code>{overview.file}</code>&ensp;·&ensp;
          <strong>Channels:</strong> {overview.n_channels}&ensp;·&ensp;
          <strong>Sampling rate:</strong> {overview.sfreq} Hz&ensp;·&ensp;
          <strong>Recordings available:</strong> {nRecs}
          {recs && (
            <span>&ensp;({recs.by_dataset?.eeg_datasets ?? 0} epilepsy + {recs.by_dataset?.real_eeg ?? 0} real EEG)</span>
          )}
        </span>
      </div>

      {/* KPI row */}
      <div className="row g-3 mb-4">
        <KPI
          label="Channels"
          value={overview.n_channels}
          color="primary"
          icon="&#x1f4e1;"
          sub={`${overview.sfreq} Hz`}
        />
        <KPI
          label="Total Spikes"
          value={spks.total_spikes ?? '—'}
          color={spks.total_spikes > 0 ? 'danger' : 'success'}
          icon="&#x26a1;"
          sub={spks.rate_per_min ? `${spks.rate_per_min.toFixed(0)}/min` : 'no spikes'}
        />
        <KPI
          label="Lateralization"
          value={lat.focus?.replace('-hemisphere','') || '—'}
          color={lat.focus?.includes('Left') ? 'info' : lat.focus?.includes('Right') ? 'warning' : 'secondary'}
          icon="&#x1f9e0;"
          sub={lat.overall_index ? `AI = ${lat.overall_index.toFixed(2)}` : ''}
        />
        <KPI
          label="Recordings"
          value={nRecs}
          color="success"
          icon="&#x1f4c2;"
          sub="EDF files available"
        />
      </div>

      {/* Trace viewer */}
      <div className="mb-4">
        <TraceViewer
          traces={traceList}
          timeArr={timeArr}
          sfreq={overview.sfreq}
        />
      </div>

      {/* Lateralization + Spikes side by side */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <LateralizationPanel lat={lat} />
        </div>
        <div className="col-md-6">
          <SpikePanel spikes={spks} />
        </div>
      </div>

      {/* Recording list */}
      {recs && recs.recordings && (
        <div className="card shadow-sm mb-3">
          <div className="card-header py-2 fw-semibold">
            &#x1f4c2; Available Recordings ({nRecs})
          </div>
          <div className="card-body p-0">
            <div className="table-responsive" style={{ maxHeight: 300 }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark sticky-top">
                  <tr>
                    <th>#</th>
                    <th>File</th>
                    <th>Group</th>
                    <th>Size (MB)</th>
                  </tr>
                </thead>
                <tbody>
                  {recs.recordings.slice(0, 50).map((r, i) => (
                    <tr key={i}>
                      <td className="text-muted">{i + 1}</td>
                      <td className="font-monospace small">{r.file.split('/').pop()}</td>
                      <td>
                        <span className={`badge ${r.group === 'eeg_datasets' ? 'bg-danger' : 'bg-info'}`}>
                          {r.group === 'eeg_datasets' ? 'Epilepsy' : 'Real EEG'}
                        </span>
                      </td>
                      <td className="font-monospace">
                        {r.bytes ? (r.bytes / 1048576).toFixed(1) : '—'}
                      </td>
                    </tr>
                  ))}
                  {recs.recordings.length > 50 && (
                    <tr>
                      <td colSpan={4} className="text-center text-muted py-2 small">
                        … and {recs.recordings.length - 50} more recordings
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Clinical note */}
      <div className="alert alert-info small mb-0">
        <strong>Data source:</strong> CHB-MIT Scalp EEG Database (Physionet) · real annotated recordings.
        Traces are 10s windows at {overview.sfreq} Hz. Lateralization index = (L−R)/(L+R) band power;
        screening grade only — not seizure localization.&ensp;
        <Link href="/eeg-viz-artifacts" className="alert-link">Artifact Review</Link> &middot;&ensp;
        <Link href="/eeg-viz-bad-channels" className="alert-link">Bad Channel QC</Link>
      </div>
    </div>
  );
}
