'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const ARTIFACT_COLORS = {
  eye_blink:  '#6366f1',
  muscle:     '#ef4444',
  line_noise: '#f59e0b',
  movement:   '#10b981',
};
const ARTIFACT_ICONS = {
  eye_blink:  '👁️',
  muscle:     '💪',
  line_noise: '⚡',
  movement:   '🏃',
};
const QUALITY_BADGE = {
  PASS:   'success',
  REVIEW: 'warning',
  FAIL:   'danger',
};

function KPI({ label, value, color, icon, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className={`card border-${color || 'primary'} border-2 h-100`}>
        <div className="card-body text-center py-2 px-2">
          <div style={{ fontSize: '1.3rem' }}>{icon}</div>
          <div className={`fw-bold fs-5 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

export default function EegVizArtifactsPage() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [selectedType, setSelectedType] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/eeg-viz/artifacts`)
      .then(r => r.json())
      .then(setD)
      .catch(e => setErr(e.message));
  }, []);

  if (err) return (
    <div className="container-fluid py-3">
      <div className="alert alert-danger">Error loading artifact data: {err}</div>
    </div>
  );
  if (!d) return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" />
      <div className="mt-2 text-muted small">Loading EEG artifact data…</div>
    </div>
  );

  const counts = d.artifact_type_counts || {};
  const wins = d.windows || [];
  const totalArtifactEvents = Object.values(counts).reduce((s, v) => s + v, 0);
  const dirtyWindows = wins.filter(w => !w.clean).length;
  const qualityColor = QUALITY_BADGE[d.quality] || 'secondary';

  const filteredWins = selectedType
    ? wins.filter(w => w.artifacts.includes(selectedType))
    : wins;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-start gap-2 mb-3 flex-wrap">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: '#7c3aed' }}>
            🔬 EEG Artifact Review
          </h4>
          <div className="small text-muted">
            Screening-grade artifact detection — eye blink · muscle · line noise · movement
          </div>
        </div>
        <div className="ms-auto d-flex gap-2 align-items-center flex-wrap">
          <span className={`badge bg-${qualityColor} fs-6`}>{d.quality}</span>
          <Link href="/eeg-viz" className="btn btn-outline-primary btn-sm">
            ← EEG Viz Platform
          </Link>
        </div>
      </div>

      {/* File info */}
      <div className="alert alert-light border mb-3 py-2 px-3">
        <span className="small">
          <strong>File:</strong> <code>{d.file}</code>&ensp;·&ensp;
          <strong>Channels:</strong> {d.n_channels}&ensp;·&ensp;
          <strong>Sampling rate:</strong> {d.sfreq} Hz&ensp;·&ensp;
          <strong>Window size:</strong> {d.window_s}s&ensp;·&ensp;
          <strong>Windows analyzed:</strong> {d.n_windows}
        </span>
      </div>

      {/* KPI row */}
      <div className="row g-3 mb-4">
        <KPI
          label="Clean Windows"
          value={`${d.clean_pct?.toFixed(1)}%`}
          color="success"
          icon="✅"
          sub={`${d.clean_windows} / ${d.n_windows} windows`}
        />
        <KPI
          label="Eye Blink Events"
          value={counts.eye_blink ?? 0}
          color="primary"
          icon="👁️"
          sub="frontal delta bursts"
        />
        <KPI
          label="Muscle Artifacts"
          value={counts.muscle ?? 0}
          color="danger"
          icon="💪"
          sub="high-freq EMG"
        />
        <KPI
          label="Movement"
          value={counts.movement ?? 0}
          color="warning"
          icon="🏃"
          sub="slow baseline drift"
        />
        <KPI
          label="Line Noise"
          value={counts.line_noise ?? 0}
          color={counts.line_noise ? 'warning' : 'success'}
          icon="⚡"
          sub="50/60 Hz"
        />
        <KPI
          label="Dirty Windows"
          value={dirtyWindows}
          color="secondary"
          icon="🚩"
          sub={`${((dirtyWindows / d.n_windows) * 100).toFixed(0)}% of total`}
        />
        <KPI
          label="Total Events"
          value={totalArtifactEvents}
          color="info"
          icon="📊"
          sub="across all types"
        />
        <KPI
          label="Frontal Channels"
          value={d.frontal_channels?.length ?? 0}
          color="secondary"
          icon="📡"
          sub="blink-monitored"
        />
      </div>

      <div className="row g-3">
        {/* Artifact Type Breakdown */}
        <div className="col-md-4">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2">Artifact Type Distribution</div>
            <div className="card-body">
              {Object.entries(counts).map(([type, count]) => {
                const pct = d.n_windows > 0 ? ((count / d.n_windows) * 100).toFixed(1) : 0;
                const color = ARTIFACT_COLORS[type] || '#94a3b8';
                const isActive = selectedType === type;
                return (
                  <div
                    key={type}
                    className={`mb-3 p-2 rounded border ${isActive ? 'border-2' : 'border-0'}`}
                    style={{ borderColor: isActive ? color : undefined, cursor: 'pointer', background: isActive ? `${color}15` : undefined }}
                    onClick={() => setSelectedType(isActive ? null : type)}
                    title={`Click to filter timeline to ${type}`}
                  >
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span className="small fw-semibold">
                        {ARTIFACT_ICONS[type] || '📊'} {type.replace(/_/g, ' ')}
                      </span>
                      <span className="badge" style={{ backgroundColor: color }}>{count}</span>
                    </div>
                    <div className="progress" style={{ height: 8 }}>
                      <div
                        className="progress-bar"
                        style={{ width: `${pct}%`, backgroundColor: color }}
                        title={`${pct}% of windows`}
                      />
                    </div>
                    <div className="text-muted" style={{ fontSize: '0.7rem' }}>
                      {pct}% of windows
                    </div>
                  </div>
                );
              })}
              {selectedType && (
                <button
                  className="btn btn-outline-secondary btn-sm w-100 mt-2"
                  onClick={() => setSelectedType(null)}
                >
                  Clear filter
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Artifact Timeline */}
        <div className="col-md-8">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 d-flex align-items-center gap-2">
              <strong>Artifact Timeline</strong>
              {selectedType && (
                <span className="badge ms-1" style={{ backgroundColor: ARTIFACT_COLORS[selectedType] || '#94a3b8' }}>
                  filtered: {selectedType.replace(/_/g, ' ')}
                </span>
              )}
              <span className="ms-auto badge bg-secondary">{filteredWins.length} windows</span>
            </div>
            <div className="card-body p-3">
              {/* Color legend */}
              <div className="d-flex flex-wrap gap-3 mb-3">
                {Object.entries(ARTIFACT_COLORS).map(([type, color]) => (
                  <span
                    key={type}
                    className="small d-flex align-items-center gap-1"
                    style={{ cursor: 'pointer' }}
                    onClick={() => setSelectedType(selectedType === type ? null : type)}
                  >
                    <span style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: color, display: 'inline-block' }} />
                    {ARTIFACT_ICONS[type]} {type.replace(/_/g, ' ')}
                  </span>
                ))}
                <span className="small d-flex align-items-center gap-1">
                  <span style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: '#22c55e', display: 'inline-block' }} />
                  ✅ clean
                </span>
              </div>

              {/* Heatmap grid */}
              <div className="d-flex flex-wrap gap-1 mb-3">
                {wins.map((w) => {
                  const isHighlighted = !selectedType || w.artifacts.includes(selectedType);
                  const bgColor = w.clean
                    ? '#22c55e'
                    : (w.artifacts[0] ? (ARTIFACT_COLORS[w.artifacts[0]] || '#94a3b8') : '#94a3b8');
                  return (
                    <div
                      key={w.window}
                      title={`Window ${w.window} @ ${w.start_s}s: ${w.artifacts.join(', ') || 'clean'}`}
                      style={{
                        width: 22,
                        height: 22,
                        borderRadius: 3,
                        backgroundColor: bgColor,
                        opacity: isHighlighted ? 1 : 0.15,
                        cursor: 'default',
                        transition: 'opacity 0.15s',
                      }}
                    />
                  );
                })}
              </div>

              {/* Filtered window table */}
              {selectedType && (
                <div className="mt-2">
                  <div className="small fw-semibold mb-1">Windows containing '{selectedType.replace(/_/g, ' ')}':</div>
                  <div className="table-responsive" style={{ maxHeight: 200 }}>
                    <table className="table table-sm table-bordered mb-0">
                      <thead className="table-light">
                        <tr>
                          <th>#</th>
                          <th>Start (s)</th>
                          <th>Artifacts</th>
                          <th>Clean?</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredWins.map(w => (
                          <tr key={w.window}>
                            <td>{w.window}</td>
                            <td className="font-monospace">{w.start_s?.toFixed(1)}</td>
                            <td>
                              {w.artifacts.map(a => (
                                <span key={a} className="badge me-1" style={{ backgroundColor: ARTIFACT_COLORS[a] || '#94a3b8' }}>
                                  {ARTIFACT_ICONS[a]} {a.replace(/_/g, ' ')}
                                </span>
                              ))}
                            </td>
                            <td>
                              <span className={`badge bg-${w.clean ? 'success' : 'danger'}`}>
                                {w.clean ? 'Yes' : 'No'}
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
      </div>

      {/* Frontal channels section */}
      {d.frontal_channels?.length > 0 && (
        <div className="card shadow-sm mt-3">
          <div className="card-header fw-semibold py-2">
            👁️ Frontal Channels Monitored for Eye Blink ({d.frontal_channels.length})
          </div>
          <div className="card-body py-2">
            <div className="d-flex flex-wrap gap-2">
              {d.frontal_channels.map(ch => (
                <span key={ch} className="badge bg-primary bg-opacity-75 font-monospace">
                  {ch}
                </span>
              ))}
            </div>
            <div className="text-muted small mt-2">
              Eye blink detection uses high delta-band amplitude (≥1–4 Hz) peaks at frontal electrodes.
              Fp1–F7, Fp1–F3, Fp2–F4, Fp2–F8 are most sensitive to blink artifact.
            </div>
          </div>
        </div>
      )}

      {/* Clinical note */}
      <div className="alert alert-info small mt-3 mb-0">
        <strong>Clinical note:</strong> {d.note || 'Screening-grade artifact detection. Validate with visual EEG review.'}&ensp;
        Source: CHB-MIT annotated EDF via MNE/SciPy.
      </div>
    </div>
  );
}
