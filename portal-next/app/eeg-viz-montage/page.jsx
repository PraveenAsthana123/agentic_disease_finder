'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const MONTAGE_META = {
  referential_original:  { label: 'Referential',    icon: '📡', color: '#6366f1', short: 'REF' },
  common_average:        { label: 'Common Average',  icon: '⚖️',  color: '#10b981', short: 'CAR' },
  bipolar_longitudinal:  { label: 'Bipolar',         icon: '↕️',  color: '#f59e0b', short: 'BIP' },
};

const BAND_COLORS = {
  delta: '#6366f1',
  theta: '#10b981',
  alpha: '#f59e0b',
  beta:  '#ef4444',
};

const BAND_HZ = {
  delta: '0.5–4 Hz',
  theta: '4–8 Hz',
  alpha: '8–13 Hz',
  beta:  '13–30 Hz',
};

function KPI({ label, value, color, icon, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className={`card border-2 h-100`} style={{ borderColor: color }}>
        <div className="card-body text-center py-2 px-2">
          <div style={{ fontSize: '1.3rem' }}>{icon}</div>
          <div className="fw-bold fs-5" style={{ color }}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function BandBar({ band, value, maxVal }) {
  const pct = maxVal > 0 ? ((value / maxVal) * 100).toFixed(1) : 0;
  const color = BAND_COLORS[band] || '#94a3b8';
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between align-items-center mb-1">
        <span className="small fw-semibold text-capitalize">
          <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, backgroundColor: color, marginRight: 4 }} />
          {band} <span className="text-muted fw-normal">({BAND_HZ[band]})</span>
        </span>
        <span className="small font-monospace">{(value * 100).toFixed(2)}%</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function DeltaBadge({ delta }) {
  if (delta == null) return <span className="text-muted">—</span>;
  const abs = Math.abs(delta * 100);
  const sign = delta >= 0 ? '+' : '−';
  const color = Math.abs(delta) < 0.005 ? 'secondary' : delta > 0 ? 'success' : 'danger';
  return (
    <span className={`badge bg-${color} font-monospace`}>
      {sign}{abs.toFixed(2)}pp
    </span>
  );
}

export default function EegVizMontagePage() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [activeMontage, setActiveMontage] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/eeg-viz/montage-comparison`)
      .then(r => r.json())
      .then(data => { setD(data); setActiveMontage('referential_original'); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return (
    <div className="container-fluid py-3">
      <div className="alert alert-danger">Error loading montage data: {err}</div>
    </div>
  );
  if (!d) return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" />
      <div className="mt-2 text-muted small">Loading montage comparison…</div>
    </div>
  );

  const montages = d.montages || {};
  const deltaVsRef = d.band_power_delta_vs_referential || {};
  const montageKeys = Object.keys(montages);

  // Compute max band power across all montages for normalised bars
  const allBandValues = montageKeys.flatMap(k =>
    Object.values(montages[k]?.band_power || {})
  );
  const maxBand = Math.max(...allBandValues, 0.01);

  const activeMontageData = activeMontage ? montages[activeMontage] : null;
  const activeMeta = activeMontage ? (MONTAGE_META[activeMontage] || {}) : {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-start gap-2 mb-3 flex-wrap">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: '#7c3aed' }}>
            📡 Montage Comparison
          </h4>
          <div className="small text-muted">
            Same EEG recording under Referential · Common Average (CAR) · Bipolar Longitudinal
          </div>
        </div>
        <div className="ms-auto">
          <Link href="/eeg-viz" className="btn btn-outline-primary btn-sm">
            ← EEG Viz Platform
          </Link>
        </div>
      </div>

      {/* File info */}
      <div className="alert alert-light border mb-3 py-2 px-3">
        <span className="small">
          <strong>File:</strong> <code>{d.file}</code>&ensp;·&ensp;
          <strong>Sampling rate:</strong> {d.sfreq} Hz&ensp;·&ensp;
          <strong>Duration:</strong> {d.seconds}s&ensp;·&ensp;
          <strong>Source:</strong> {d.source}
        </span>
      </div>

      {/* KPI row */}
      <div className="row g-3 mb-4">
        {montageKeys.map(key => {
          const m = montages[key];
          const meta = MONTAGE_META[key] || { label: key, icon: '📊', color: '#6366f1', short: key };
          return (
            <KPI
              key={key}
              label={meta.label}
              value={`${m.n_channels} ch`}
              color={meta.color}
              icon={meta.icon}
              sub={`${m.mean_amplitude_uv?.toFixed(2)} µV mean`}
            />
          );
        })}
        <KPI label="Bands Compared" value="4" color="#7c3aed" icon="🎚️" sub="δ θ α β" />
      </div>

      <div className="row g-3 mb-4">
        {/* Montage selector + detail */}
        <div className="col-md-5">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2">Select Montage</div>
            <div className="card-body">
              <div className="d-flex flex-column gap-2 mb-3">
                {montageKeys.map(key => {
                  const meta = MONTAGE_META[key] || { label: key, icon: '📊', color: '#94a3b8', short: key };
                  const isActive = activeMontage === key;
                  return (
                    <button
                      key={key}
                      className="btn btn-sm text-start d-flex align-items-center gap-2"
                      style={{
                        background: isActive ? `${meta.color}18` : 'transparent',
                        border: `2px solid ${isActive ? meta.color : '#e5e7eb'}`,
                        borderRadius: 8,
                        color: isActive ? meta.color : '#374151',
                        fontWeight: isActive ? 600 : 400,
                      }}
                      onClick={() => setActiveMontage(key)}
                    >
                      <span style={{ fontSize: '1.1rem' }}>{meta.icon}</span>
                      <div>
                        <div>{meta.label}</div>
                        <div className="text-muted fw-normal" style={{ fontSize: '0.7rem' }}>
                          {montages[key]?.description}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>

              {activeMontageData && (
                <div className="border rounded p-2" style={{ background: `${activeMeta.color}08` }}>
                  <div className="fw-semibold small mb-1" style={{ color: activeMeta.color }}>
                    {activeMeta.icon} {activeMeta.label} — Band Power
                  </div>
                  {Object.entries(activeMontageData.band_power || {}).map(([band, val]) => (
                    <BandBar key={band} band={band} value={val} maxVal={maxBand} />
                  ))}
                  {activeMontageData.example_derivations?.length > 0 && (
                    <div className="mt-2">
                      <div className="small fw-semibold text-muted mb-1">Example derivations:</div>
                      <div className="d-flex flex-wrap gap-1">
                        {activeMontageData.example_derivations.map((d, i) => (
                          <code key={i} className="small bg-light px-1 rounded">{d}</code>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Band power comparison table */}
        <div className="col-md-7">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2">Band Power Across Montages</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-bordered table-sm mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Band</th>
                      {montageKeys.map(k => {
                        const meta = MONTAGE_META[k] || { short: k, color: '#94a3b8' };
                        return (
                          <th key={k} className="text-center">
                            <span style={{ color: meta.color }}>{meta.short}</span>
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {['delta', 'theta', 'alpha', 'beta'].map(band => {
                      const vals = montageKeys.map(k => montages[k]?.band_power?.[band] ?? null);
                      const maxV = Math.max(...vals.filter(v => v != null));
                      return (
                        <tr key={band}>
                          <td>
                            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, backgroundColor: BAND_COLORS[band], marginRight: 4 }} />
                            <strong className="text-capitalize">{band}</strong>
                            <div className="text-muted" style={{ fontSize: '0.65rem' }}>{BAND_HZ[band]}</div>
                          </td>
                          {vals.map((v, i) => (
                            <td key={i} className="text-center font-monospace small">
                              <span className={v === maxV ? 'fw-bold' : ''} style={{ color: v === maxV ? BAND_COLORS[band] : undefined }}>
                                {v != null ? (v * 100).toFixed(2) + '%' : '—'}
                              </span>
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Delta vs Referential */}
            <div className="card-header fw-semibold py-2 border-top">
              Δ vs Referential (CAR & Bipolar)
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-bordered table-sm mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Band</th>
                      <th className="text-center" style={{ color: MONTAGE_META.common_average.color }}>
                        CAR Δ
                      </th>
                      <th className="text-center" style={{ color: MONTAGE_META.bipolar_longitudinal.color }}>
                        Bipolar Δ
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {['delta', 'theta', 'alpha', 'beta'].map(band => (
                      <tr key={band}>
                        <td>
                          <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, backgroundColor: BAND_COLORS[band], marginRight: 4 }} />
                          <strong className="text-capitalize">{band}</strong>
                        </td>
                        <td className="text-center">
                          <DeltaBadge delta={deltaVsRef.common_average?.[band]} />
                        </td>
                        <td className="text-center">
                          <DeltaBadge delta={deltaVsRef.bipolar_longitudinal?.[band]} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="p-2 text-muted" style={{ fontSize: '0.7rem' }}>
                pp = percentage-point difference from referential. Green = higher than REF, red = lower.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Clinical note */}
      <div className="alert alert-info small mb-0">
        <strong>Clinical note:</strong> {d.note}&ensp;
        <span className="text-muted">Source: {d.source}</span>
      </div>
    </div>
  );
}
