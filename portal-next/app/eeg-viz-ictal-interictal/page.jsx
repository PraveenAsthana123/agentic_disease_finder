'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const BAND_COLORS = {
  delta: '#6366f1', theta: '#8b5cf6', alpha: '#3b82f6',
  beta: '#22c55e', gamma: '#f59e0b',
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

function BandBar({ band, ictalPct, interictalPct }) {
  const col = BAND_COLORS[band] || '#94a3b8';
  return (
    <div className="mb-3">
      <div className="fw-semibold small mb-1 text-capitalize">{band}</div>
      <div className="d-flex align-items-center gap-2 mb-1">
        <span className="text-muted" style={{ width: 70, fontSize: 11 }}>Ictal</span>
        <div className="progress flex-grow-1" style={{ height: 10 }}>
          <div className="progress-bar" style={{ width: `${(ictalPct * 100).toFixed(1)}%`, background: col }} />
        </div>
        <span className="small" style={{ width: 42, textAlign: 'right' }}>{(ictalPct * 100).toFixed(1)}%</span>
      </div>
      <div className="d-flex align-items-center gap-2">
        <span className="text-muted" style={{ width: 70, fontSize: 11 }}>Interictal</span>
        <div className="progress flex-grow-1" style={{ height: 10 }}>
          <div className="progress-bar" style={{ width: `${(interictalPct * 100).toFixed(1)}%`, background: col, opacity: 0.4 }} />
        </div>
        <span className="small" style={{ width: 42, textAlign: 'right' }}>{(interictalPct * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}

export default function EegVizIctalInterictalPage() {
  const [data, setData] = useState(null);
  const [annot, setAnnot] = useState(null);
  const [tab, setTab] = useState('compare');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/eeg-viz/ictal-interictal`).then(r => r.json()),
      fetch(`${API}/api/eeg-viz/seizure-annotations`).then(r => r.json()),
    ]).then(([d, a]) => { setData(d); setAnnot(a); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!data) return <div className="text-muted p-4 text-center"><div className="spinner-border text-primary mb-2" /><div>Loading ictal vs interictal data…</div></div>;

  const TABS = [
    { id: 'compare',   label: 'Band Power Compare' },
    { id: 'signature', label: 'Ictal Signature' },
    { id: 'files',     label: 'Annotated Files' },
  ];

  const sig = data.ictal_signature || {};
  const bp  = data.band_power || {};
  const ict = bp.ictal || {};
  const int = bp.interictal || {};
  const bands = Object.keys(ict);

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: 28 }}>⚡</span>
        <div>
          <h4 className="mb-0 fw-bold">Ictal vs Interictal EEG</h4>
          <div className="text-muted small">
            Band-power comparison · file: {data.file} · {data.sfreq} Hz · {data.recording_s}s recording
          </div>
        </div>
        <span className={`badge ms-auto ${sig.verdict?.includes('consistent') ? 'bg-danger' : 'bg-secondary'}`}
          style={{ fontSize: 11 }}>
          {sig.verdict ? 'Seizure Confirmed' : '—'}
        </span>
      </div>

      {/* KPIs */}
      <div className="row mb-2">
        <KPI label="Seizure Window" value={`${data.seizure_window?.start_s}–${data.seizure_window?.end_s}s`}
          sub={`${data.seizure_window?.duration_s}s duration`} color="#ef4444" />
        <KPI label="Interictal Window" value={`${data.interictal_window?.start_s}–${data.interictal_window?.end_s?.toFixed(0)}s`}
          sub="baseline comparison" color="#3b82f6" />
        <KPI label="Annotated Files" value={data.annotated_files} sub="CHB-MIT dataset" color="#8b5cf6" />
        <KPI label="Delta Shift" value={sig.delta_shift > 0 ? `+${sig.delta_shift?.toFixed(3)}` : sig.delta_shift?.toFixed(3)}
          sub="ictal − interictal" color={sig.delta_shift > 0 ? '#ef4444' : '#22c55e'} />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* BAND POWER COMPARE */}
      {tab === 'compare' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Band Power — Ictal vs Interictal (normalised)</div>
              <div className="card-body">
                {bands.map(b => (
                  <BandBar key={b} band={b} ictalPct={ict[b] || 0} interictalPct={int[b] || 0} />
                ))}
                <div className="d-flex gap-3 small text-muted mt-2">
                  <span><span style={{ display: 'inline-block', width: 12, height: 8, background: '#3b82f6', borderRadius: 2 }} className="me-1" />Ictal (solid)</span>
                  <span><span style={{ display: 'inline-block', width: 12, height: 8, background: '#3b82f6', borderRadius: 2, opacity: 0.4 }} className="me-1" />Interictal (faded)</span>
                </div>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Band Power Table</div>
              <div className="table-responsive">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Band</th><th>Ictal</th><th>Interictal</th><th>Δ (Ictal−Interictal)</th></tr>
                  </thead>
                  <tbody>
                    {bands.map(b => {
                      const delta = (ict[b] - int[b]);
                      return (
                        <tr key={b}>
                          <td className="fw-semibold small text-capitalize">
                            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: BAND_COLORS[b], marginRight: 6 }} />
                            {b}
                          </td>
                          <td className="small">{(ict[b] * 100).toFixed(2)}%</td>
                          <td className="small">{(int[b] * 100).toFixed(2)}%</td>
                          <td className="small fw-bold"
                            style={{ color: delta > 0.01 ? '#ef4444' : delta < -0.01 ? '#22c55e' : undefined }}>
                            {delta > 0 ? '+' : ''}{(delta * 100).toFixed(2)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ICTAL SIGNATURE */}
      {tab === 'signature' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Ictal Signature Analysis</div>
              <div className="card-body">
                <div className="p-3 rounded mb-3" style={{ background: '#fef2f2', borderLeft: '4px solid #ef4444' }}>
                  <div className="fw-semibold small mb-1">Verdict</div>
                  <div className="small">{sig.verdict || '—'}</div>
                </div>
                <div className="row g-2">
                  <div className="col-6">
                    <div className="card border-danger border-2">
                      <div className="card-body text-center py-2">
                        <div className="fw-bold text-danger" style={{ fontSize: 22 }}>
                          {sig.delta_shift > 0 ? `+${sig.delta_shift?.toFixed(4)}` : sig.delta_shift?.toFixed(4)}
                        </div>
                        <div className="text-muted small">Delta shift (↑ ictal)</div>
                      </div>
                    </div>
                  </div>
                  <div className="col-6">
                    <div className="card border-success border-2">
                      <div className="card-body text-center py-2">
                        <div className="fw-bold text-success" style={{ fontSize: 22 }}>
                          {sig.alpha_shift?.toFixed(4)}
                        </div>
                        <div className="text-muted small">Alpha shift (↓ ictal)</div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="mt-3 small text-muted">
                  <strong>Interpretation:</strong> During seizure, delta-band power increases (slow-wave dominance)
                  and alpha decreases — a consistent electrographic signature of ictal activity on scalp EEG.
                </div>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Recording Metadata</div>
              <div className="card-body small">
                <table className="table table-sm mb-0">
                  <tbody>
                    <tr><td className="text-muted">File</td><td className="fw-semibold">{data.file}</td></tr>
                    <tr><td className="text-muted">Sample rate</td><td>{data.sfreq} Hz</td></tr>
                    <tr><td className="text-muted">Recording duration</td><td>{data.recording_s}s ({(data.recording_s/60).toFixed(1)} min)</td></tr>
                    <tr><td className="text-muted">Ictal window</td><td>{data.seizure_window?.start_s}s – {data.seizure_window?.end_s}s ({data.seizure_window?.duration_s}s)</td></tr>
                    <tr><td className="text-muted">Interictal window</td><td>{data.interictal_window?.start_s?.toFixed(0)}s – {data.interictal_window?.end_s?.toFixed(0)}s</td></tr>
                    <tr><td className="text-muted">Source</td><td>CHB-MIT (PhysioNet)</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ANNOTATED FILES */}
      {tab === 'files' && annot && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">
            CHB-MIT Annotated Files ({annot.files?.length || 0} total)
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>File</th><th>Seizures</th><th>Start (s)</th><th>End (s)</th><th>Duration (s)</th><th>On Disk</th></tr>
              </thead>
              <tbody>
                {(annot.files || []).map((f, i) => (
                  f.seizures?.map((sz, j) => (
                    <tr key={`${i}-${j}`}>
                      <td className="fw-semibold small">{j === 0 ? f.file : ''}</td>
                      <td className="small">{j === 0 ? f.n_seizures : ''}</td>
                      <td className="small">{sz.start_s}</td>
                      <td className="small">{sz.end_s}</td>
                      <td className="small">{sz.end_s - sz.start_s}s</td>
                      <td>{j === 0 ? (f.edf_on_disk
                        ? <span className="badge bg-success" style={{ fontSize: 10 }}>Yes</span>
                        : <span className="badge bg-secondary" style={{ fontSize: 10 }}>No</span>) : ''}</td>
                    </tr>
                  ))
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
