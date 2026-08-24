'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const VERDICT_META = {
  good:       { color: '#22c55e', bg: 'success', icon: '✅', label: 'Good' },
  flat:       { color: '#ef4444', bg: 'danger',  icon: '📉', label: 'Flat / Disconnected' },
  noisy:      { color: '#f59e0b', bg: 'warning', icon: '⚡', label: 'Noisy (high-amplitude)' },
  line_noise: { color: '#a855f7', bg: 'primary', icon: '〰️', label: 'Line Noise (50/60 Hz)' },
};

const QUALITY_BADGE = {
  PASS:   { bg: 'success',   label: 'PASS' },
  REVIEW: { bg: 'warning',   label: 'REVIEW' },
  FAIL:   { bg: 'danger',    label: 'FAIL' },
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

function VerdictBar({ verdict, count, total }) {
  const meta = VERDICT_META[verdict] || { color: '#94a3b8', icon: '?', label: verdict };
  const pct = total > 0 ? ((count / total) * 100).toFixed(1) : 0;
  return (
    <div className="mb-3">
      <div className="d-flex justify-content-between align-items-center mb-1">
        <span className="small fw-semibold">{meta.icon} {meta.label}</span>
        <span className="badge" style={{ backgroundColor: meta.color }}>{count}</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div
          className="progress-bar"
          style={{ width: `${pct}%`, backgroundColor: meta.color }}
          title={`${pct}% of channels`}
        />
      </div>
      <div className="text-muted" style={{ fontSize: '0.7rem' }}>{pct}% of channels</div>
    </div>
  );
}

export default function EegVizBadChannelsPage() {
  const [d, setD]   = useState(null);
  const [err, setErr] = useState(null);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    fetch(`${API}/api/eeg-viz/bad-channels`)
      .then(r => r.json())
      .then(setD)
      .catch(e => setErr(e.message));
  }, []);

  if (err) return (
    <div className="container-fluid py-3">
      <div className="alert alert-danger">Error loading channel QC data: {err}</div>
    </div>
  );
  if (!d) return (
    <div className="text-center py-5">
      <div className="spinner-border text-danger" />
      <div className="mt-2 text-muted small">Loading channel quality data…</div>
    </div>
  );

  if (!d.available) return (
    <div className="container-fluid py-3">
      <div className="alert alert-warning">
        <strong>No EEG recordings found.</strong> {d.error || 'Upload an EDF file to use this dashboard.'}
      </div>
    </div>
  );

  const channels   = d.channels || [];
  const verdictDist = d.verdict_distribution || {};
  const totalCh    = channels.length;
  const badChs     = d.bad_channels || [];
  const qMeta      = QUALITY_BADGE[d.quality] || { bg: 'secondary', label: d.quality };

  const allVerdicts = ['good', 'flat', 'noisy', 'line_noise'];
  const filtered = filter === 'all' ? channels : channels.filter(c => c.verdict === filter);

  // Compute colour for heatmap cell
  const cellColor = (ch) => (VERDICT_META[ch.verdict] || { color: '#94a3b8' }).color;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-start gap-2 mb-3 flex-wrap">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: '#b91c1c' }}>
            📉 Bad Channel Dashboard
          </h4>
          <div className="small text-muted">
            Per-channel signal QC — flat / disconnected · noisy · line-noise (50/60 Hz)
          </div>
        </div>
        <div className="ms-auto d-flex gap-2 align-items-center flex-wrap">
          <span className={`badge bg-${qMeta.bg} fs-6`}>{qMeta.label}</span>
          <Link href="/eeg-viz" className="btn btn-outline-primary btn-sm">
            ← EEG Viz Platform
          </Link>
        </div>
      </div>

      {/* File info bar */}
      <div className="alert alert-light border mb-3 py-2 px-3">
        <span className="small">
          <strong>File:</strong> <code>{d.file}</code>&ensp;·&ensp;
          <strong>Channels:</strong> {d.n_channels}&ensp;·&ensp;
          <strong>Sampling rate:</strong> {d.sfreq} Hz&ensp;·&ensp;
          <strong>Window analyzed:</strong> {d.seconds_analyzed}s
        </span>
      </div>

      {/* KPI row */}
      <div className="row g-3 mb-4">
        <KPI
          label="Total Channels"
          value={totalCh}
          color="primary"
          icon="📡"
          sub="all electrodes"
        />
        <KPI
          label="Bad Channels"
          value={badChs.length}
          color={badChs.length ? 'danger' : 'success'}
          icon={badChs.length ? '🚩' : '✅'}
          sub={badChs.length ? badChs.slice(0, 3).join(', ') + (badChs.length > 3 ? '…' : '') : 'none detected'}
        />
        <KPI
          label="Good Channels"
          value={verdictDist.good ?? (totalCh - badChs.length)}
          color="success"
          icon="✅"
          sub={`${totalCh > 0 ? (((verdictDist.good ?? (totalCh - badChs.length)) / totalCh) * 100).toFixed(0) : 0}% usable`}
        />
        <KPI
          label="QC Quality"
          value={d.quality}
          color={qMeta.bg === 'success' ? 'success' : qMeta.bg === 'warning' ? 'warning' : 'danger'}
          icon={d.quality === 'PASS' ? '🟢' : d.quality === 'REVIEW' ? '🟡' : '🔴'}
          sub="overall verdict"
        />
      </div>

      <div className="row g-3">
        {/* Verdict breakdown */}
        <div className="col-md-4">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2">Verdict Distribution</div>
            <div className="card-body">
              {allVerdicts.map(v => (
                <VerdictBar
                  key={v}
                  verdict={v}
                  count={verdictDist[v] ?? 0}
                  total={totalCh}
                />
              ))}
              <hr className="my-2" />
              <div className="small text-muted">
                <strong>Thresholds:</strong><br />
                Flat: std &lt; {d.thresholds?.flat_std_uv} µV<br />
                Noisy: std &gt; {d.thresholds?.noisy_std_uv} µV<br />
                Line noise: relative power &gt; {d.thresholds?.line_noise_rel}
              </div>
            </div>
          </div>
        </div>

        {/* Channel heatmap + table */}
        <div className="col-md-8">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 d-flex align-items-center gap-2 flex-wrap">
              <strong>Channel Quality Map</strong>
              <div className="ms-auto d-flex gap-1 flex-wrap">
                <button
                  className={`btn btn-sm ${filter === 'all' ? 'btn-secondary' : 'btn-outline-secondary'}`}
                  onClick={() => setFilter('all')}
                >All</button>
                {allVerdicts.filter(v => v !== 'good').map(v => {
                  const meta = VERDICT_META[v];
                  return (
                    <button
                      key={v}
                      className={`btn btn-sm btn-outline-${meta.bg}`}
                      style={filter === v ? { opacity: 1 } : { opacity: 0.6 }}
                      onClick={() => setFilter(filter === v ? 'all' : v)}
                    >
                      {meta.icon} {meta.label.split(' ')[0]}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="card-body p-3">
              {/* Electrode grid */}
              <div className="mb-3">
                <div className="small fw-semibold mb-1">Electrode Map</div>
                <div className="d-flex flex-wrap gap-1">
                  {channels.map(ch => {
                    const meta = VERDICT_META[ch.verdict] || { color: '#94a3b8', icon: '?' };
                    const isFiltered = filter !== 'all' && ch.verdict !== filter;
                    return (
                      <div
                        key={ch.channel}
                        title={`${ch.channel}: ${ch.verdict} | std=${ch.std_uv}µV p2p=${ch.p2p_uv}µV flat=${ch.flat_ratio} line=${ch.line_noise_rel}`}
                        style={{
                          width: 44,
                          height: 36,
                          borderRadius: 4,
                          backgroundColor: cellColor(ch),
                          opacity: isFiltered ? 0.12 : 1,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '0.6rem',
                          color: '#fff',
                          fontWeight: 600,
                          cursor: 'default',
                          transition: 'opacity 0.15s',
                          textAlign: 'center',
                          padding: '0 2px',
                          wordBreak: 'break-all',
                        }}
                      >
                        {ch.channel.length > 6 ? ch.channel.slice(0, 5) + '…' : ch.channel}
                      </div>
                    );
                  })}
                </div>
                {/* Legend */}
                <div className="d-flex flex-wrap gap-3 mt-2">
                  {allVerdicts.map(v => {
                    const meta = VERDICT_META[v];
                    return (
                      <span key={v} className="small d-flex align-items-center gap-1">
                        <span style={{ width: 12, height: 12, borderRadius: 2, backgroundColor: meta.color, display: 'inline-block' }} />
                        {meta.icon} {meta.label}
                      </span>
                    );
                  })}
                </div>
              </div>

              {/* Channel table */}
              <div className="table-responsive" style={{ maxHeight: 320 }}>
                <table className="table table-sm table-hover table-bordered mb-0">
                  <thead className="table-dark sticky-top">
                    <tr>
                      <th>Channel</th>
                      <th>Verdict</th>
                      <th>Std (µV)</th>
                      <th>P2P (µV)</th>
                      <th>Flat ratio</th>
                      <th>Line noise</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map(ch => {
                      const meta = VERDICT_META[ch.verdict] || { bg: 'secondary', icon: '?', label: ch.verdict };
                      return (
                        <tr key={ch.channel} className={ch.verdict !== 'good' ? 'table-warning' : ''}>
                          <td className="font-monospace fw-semibold">{ch.channel}</td>
                          <td>
                            <span className={`badge bg-${meta.bg}`}>
                              {meta.icon} {meta.label}
                            </span>
                          </td>
                          <td className="font-monospace">{ch.std_uv?.toFixed(1)}</td>
                          <td className="font-monospace">{ch.p2p_uv?.toFixed(1)}</td>
                          <td className="font-monospace">{ch.flat_ratio?.toFixed(3)}</td>
                          <td className="font-monospace">{ch.line_noise_rel?.toFixed(3)}</td>
                        </tr>
                      );
                    })}
                    {filtered.length === 0 && (
                      <tr>
                        <td colSpan={6} className="text-center text-muted py-3">
                          No channels match the current filter.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bad channels summary */}
      {badChs.length > 0 && (
        <div className="card shadow-sm mt-3 border-danger border-2">
          <div className="card-header fw-semibold py-2 text-danger">
            🚩 Bad Channels Identified ({badChs.length})
          </div>
          <div className="card-body py-2">
            <div className="d-flex flex-wrap gap-2">
              {badChs.map(ch => (
                <span key={ch} className="badge bg-danger font-monospace fs-6">{ch}</span>
              ))}
            </div>
            <div className="text-muted small mt-2">
              These channels should be excluded or interpolated before downstream analysis.
            </div>
          </div>
        </div>
      )}

      {/* Clinical note */}
      <div className="alert alert-info small mt-3 mb-0">
        <strong>Clinical note:</strong> {d.note || 'Screening-grade channel QC. Validate with visual review before interpolation.'}&ensp;
        Source: CHB-MIT EDF via MNE/SciPy.
      </div>
    </div>
  );
}
