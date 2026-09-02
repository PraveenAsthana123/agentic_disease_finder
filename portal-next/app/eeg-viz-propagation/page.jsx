'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const REGION_COLOR = {
  frontal: '#3b82f6', temporal: '#ef4444', central: '#22c55e',
  parietal: '#f59e0b', occipital: '#8b5cf6', 'fronto-temporal': '#ec4899',
};
const HEMI_BADGE = {
  right: { bg: '#fef2f2', color: '#ef4444' },
  left:  { bg: '#eff6ff', color: '#3b82f6' },
  midline:   { bg: '#f0fdf4', color: '#22c55e' },
  bilateral: { bg: '#fdf4ff', color: '#a855f7' },
};

function KPI({ label, value, sub, color }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div style={{ fontSize: 26, fontWeight: 700, color: color || '#3b82f6' }}>{value ?? '—'}</div>
          <div className="text-muted small fw-semibold">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.68rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

export default function EegVizPropagationPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/eeg-viz/propagation`)
      .then(r => r.json())
      .then(setData)
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!data) return <div className="text-muted p-4 text-center"><div className="spinner-border text-primary mb-2" /><div>Loading propagation data…</div></div>;

  const propOrder = data.propagation_order || [];
  const leaders   = data.onset_leaders || [];
  const displayed = showAll ? propOrder : propOrder.slice(0, 10);
  const maxOnset  = Math.max(...propOrder.map(c => c.onset_s)) || 1;

  // Count by region
  const regionCounts = propOrder.reduce((acc, c) => {
    acc[c.region] = (acc[c.region] || 0) + 1; return acc;
  }, {});

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: 28 }}>🗺️</span>
        <div>
          <h4 className="mb-0 fw-bold">Seizure Propagation Map</h4>
          <div className="text-muted small">
            Time-ordered channel recruitment · {data.file} · {data.sfreq} Hz
          </div>
          <div className="text-muted" style={{ fontSize: '0.68rem' }}>{data.note}</div>
        </div>
        <span className="ms-auto badge bg-primary">
          {data.lead_hemisphere} {data.lead_region} onset
        </span>
      </div>

      {/* KPIs */}
      <div className="row mb-2">
        <KPI label="Onset Region" value={data.lead_region} sub={`${data.lead_hemisphere} hemisphere`} color="#ef4444" />
        <KPI label="Channels Activated" value={data.n_activated} sub={`${data.n_silent} silent`} color="#3b82f6" />
        <KPI label="Spread Span" value={`${data.spread_span_s}s`} sub="onset to last recruited" color="#f59e0b" />
        <KPI label="Seizure Window" value={`${data.seizure_window?.start_s}–${data.seizure_window?.end_s}s`}
          sub="from recording start" color="#8b5cf6" />
      </div>

      <div className="row">
        {/* Onset leaders */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Onset Leaders (earliest channels)</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Channel</th><th>Region</th><th>Hemisphere</th><th>Onset (s)</th></tr>
                </thead>
                <tbody>
                  {leaders.map((c, i) => {
                    const hm = HEMI_BADGE[c.hemisphere] || {};
                    return (
                      <tr key={i}>
                        <td className="fw-semibold small">{c.channel}</td>
                        <td>
                          <span className="badge" style={{ background: REGION_COLOR[c.region] || '#94a3b8', fontSize: 10 }}>
                            {c.region}
                          </span>
                        </td>
                        <td>
                          <span className="small px-1 rounded" style={{ background: hm.bg, color: hm.color, fontSize: 10 }}>
                            {c.hemisphere}
                          </span>
                        </td>
                        <td className="small fw-bold text-danger">{c.onset_s}s</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Propagation timeline */}
        <div className="col-md-8 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold d-flex justify-content-between align-items-center">
              <span>Propagation Timeline ({propOrder.length} channels)</span>
              <button className="btn btn-sm btn-outline-secondary" onClick={() => setShowAll(!showAll)}>
                {showAll ? 'Show Top 10' : `Show All ${propOrder.length}`}
              </button>
            </div>
            <div className="card-body p-2">
              {displayed.map((c, i) => {
                const pct = maxOnset > 0 ? (c.onset_s / maxOnset) * 100 : 0;
                const col = REGION_COLOR[c.region] || '#94a3b8';
                return (
                  <div key={i} className="d-flex align-items-center gap-2 mb-1">
                    <span className="text-muted" style={{ width: 18, fontSize: 10, textAlign: 'right' }}>{i + 1}</span>
                    <span className="small fw-semibold" style={{ width: 90, fontSize: 11 }}>{c.channel}</span>
                    <div className="flex-grow-1">
                      <div className="d-flex align-items-center gap-1">
                        <div style={{ width: `${pct}%`, height: 14, background: col, borderRadius: 3, minWidth: 4 }} />
                        <span className="small text-muted" style={{ fontSize: 10 }}>{c.onset_s}s</span>
                      </div>
                    </div>
                    <span className="badge" style={{ background: col, fontSize: 9, width: 80, textAlign: 'center' }}>
                      {c.region}
                    </span>
                  </div>
                );
              })}
              {!showAll && propOrder.length > 10 && (
                <div className="text-muted small text-center mt-2">
                  + {propOrder.length - 10} more channels — click "Show All"
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Region breakdown */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-semibold">Spread by Brain Region</div>
        <div className="card-body">
          <div className="row">
            {Object.entries(regionCounts).sort((a, b) => b[1] - a[1]).map(([region, count]) => {
              const col = REGION_COLOR[region] || '#94a3b8';
              const pct = (count / propOrder.length * 100).toFixed(0);
              return (
                <div key={region} className="col-md-4 mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="fw-semibold text-capitalize">{region}</span>
                    <span>{count} ch ({pct}%)</span>
                  </div>
                  <div className="progress" style={{ height: 8 }}>
                    <div className="progress-bar" style={{ width: `${pct}%`, background: col }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Method */}
      <div className="card shadow-sm">
        <div className="card-header fw-semibold">Analysis Method</div>
        <div className="card-body small text-muted">
          {data.method}
        </div>
      </div>
    </div>
  );
}
