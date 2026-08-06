'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const battColor  = b => b >= 60 ? 'success' : b >= 30 ? 'warning' : 'danger';
const uploadBadge= s => ({ uploaded: 'success', recording: 'primary', error: 'danger' }[s] || 'secondary');
const qColor     = v => v >= 90 ? 'success' : v >= 70 ? 'warning' : 'danger';

export default function ECGPatchDashboard() {
  const [ov, setOv]     = useState(null);
  const [ss, setSs]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [selected, setSelected] = useState(null);
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ecg-patch/overview`).then(r => r.json()),
      fetch(`${API}/api/ecg-patch/sessions`).then(r => r.json()),
      fetch(`${API}/api/ecg-patch/definitions`).then(r => r.json()),
    ]).then(([o, s, d]) => { setOv(o); setSs(s); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-3">Loading ECG Patch fleet data…</div>;

  const TABS = [
    { id: 'overview',  label: '📊 Overview' },
    { id: 'fleet',     label: '🩹 Fleet' },
    { id: 'sessions',  label: '📋 Sessions' },
    { id: 'defs',      label: '📖 Reference' },
  ];

  const kpis    = ov.kpis || {};
  const fleet   = ov.fleet || [];
  const sessions= ss?.sessions || [];
  const device  = defs?.device || {};
  const metrics = defs?.metrics || [];
  const epilepsy= defs?.epilepsy_context || [];

  const selSession = selected !== null ? sessions[selected] : null;

  return (
    <div className="container-fluid py-3">
      <h2 className="fw-bold mb-1">🩹 ECG Patch Dashboard</h2>
      <p className="text-muted mb-3">
        Wearable cardiac monitors — continuous HR/HRV · arrhythmia detection · epilepsy co-deployment · {kpis.total_patches} patches
      </p>

      {/* KPI row */}
      <div className="row g-2 mb-3">
        {[
          { label: 'Total Patches',    val: kpis.total_patches,             cls: 'primary' },
          { label: 'Recording',        val: kpis.active_recording,          cls: 'success' },
          { label: 'Uploaded',         val: kpis.uploaded,                  cls: 'info' },
          { label: 'Upload Errors',    val: kpis.upload_errors,             cls: kpis.upload_errors > 0 ? 'danger' : 'success' },
          { label: 'Avg Battery',      val: `${kpis.avg_battery_pct}%`,     cls: 'warning' },
          { label: 'Total Beats',      val: (kpis.total_beats_analyzed || 0).toLocaleString(), cls: 'secondary' },
        ].map(k => (
          <div key={k.label} className="col-6 col-md-2">
            <div className={`card border-${k.cls} text-center py-2`}>
              <div className={`fs-5 fw-bold text-${k.cls}`}>{k.val}</div>
              <div className="small text-muted">{k.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div className="row g-3">
          <div className="col-md-8">
            <div className="card p-3">
              <h6 className="fw-bold mb-2">🩹 Fleet Status</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover table-bordered mb-0">
                  <thead className="table-dark">
                    <tr><th>Patch</th><th>Patient</th><th>Duration (d)</th><th>Battery</th><th>Upload</th><th>Lead</th><th>Total Beats</th></tr>
                  </thead>
                  <tbody>
                    {fleet.map((f, i) => (
                      <tr key={i}>
                        <td><code>{f.patch_id}</code></td>
                        <td>{f.patient}</td>
                        <td>{f.duration_days}</td>
                        <td><span className={`badge bg-${battColor(f.battery_pct)}`}>{f.battery_pct}%</span></td>
                        <td><span className={`badge bg-${uploadBadge(f.upload_status)}`}>{f.upload_status}</span></td>
                        <td className="small">{f.lead}</td>
                        <td>{(f.total_beats || 0).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="card p-3">
              <h6 className="fw-bold mb-2">📋 Fleet Summary</h6>
              <dl className="row mb-0 small">
                <dt className="col-7">Recording mode</dt>
                <dd className="col-5">{device.recording_mode || 'Offline'}</dd>
                <dt className="col-7">Sampling rate</dt>
                <dd className="col-5">{device.sampling_rate_hz} Hz</dd>
                <dt className="col-7">Storage capacity</dt>
                <dd className="col-5">{device.storage_capacity_days} days</dd>
                <dt className="col-7">Battery life</dt>
                <dd className="col-5">{device.battery_life_days} days</dd>
                <dt className="col-7">Waterproof</dt>
                <dd className="col-5">{device.waterproof}</dd>
                <dt className="col-7">FDA cleared</dt>
                <dd className="col-5">{device.fda_cleared}</dd>
                <dt className="col-7">Leads</dt>
                <dd className="col-5">{(device.leads || []).join(', ')}</dd>
              </dl>
            </div>
          </div>
        </div>
      )}

      {/* Fleet tab */}
      {tab === 'fleet' && (
        <div className="row g-3">
          {fleet.map((f, i) => (
            <div key={i} className="col-md-3">
              <div className={`card border-${uploadBadge(f.upload_status)} p-3`}>
                <div className="fw-bold">{f.patch_id}</div>
                <div className="text-muted small">{f.patient}</div>
                <div className="mt-2">
                  <span className={`badge bg-${uploadBadge(f.upload_status)} me-1`}>{f.upload_status}</span>
                  <span className={`badge bg-${battColor(f.battery_pct)}`}>{f.battery_pct}%</span>
                </div>
                <div className="small mt-1">Duration: {f.duration_days}d</div>
                <div className="small">Lead: {f.lead}</div>
                <div className="small">Beats: {(f.total_beats || 0).toLocaleString()}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Sessions tab */}
      {tab === 'sessions' && (
        <div className="row g-3">
          <div className="col-md-4">
            <div className="card p-3">
              <h6 className="fw-bold mb-2">📋 Sessions ({sessions.length})</h6>
              <div style={{ maxHeight: 500, overflowY: 'auto' }}>
                {sessions.map((s, i) => (
                  <div key={i}
                    className={`border-bottom py-2 px-1 small ${selected === i ? 'bg-light' : ''}`}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setSelected(selected === i ? null : i)}>
                    <div className="fw-bold">{s.session_id} — {s.patient_id}</div>
                    <div className="text-muted">{s.recording_start} → {s.recording_end} ({s.duration_days}d)</div>
                    <span className={`badge bg-${qColor(s.signal_quality_pct)}`}>{s.signal_quality_pct}% quality</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          {selSession && (
            <div className="col-md-8">
              <div className="card p-3">
                <h6 className="fw-bold mb-2">📊 {selSession.session_id} — {selSession.patient_id}</h6>
                <div className="row g-2 mb-3">
                  {[
                    { label: 'Duration', val: `${selSession.duration_days} days` },
                    { label: 'Mean HR', val: `${selSession.mean_hr_bpm} bpm` },
                    { label: 'HR Range', val: `${selSession.min_hr_bpm}–${selSession.max_hr_bpm} bpm` },
                    { label: 'SDNN', val: `${selSession.sdnn_ms} ms` },
                    { label: 'RMSSD', val: `${selSession.rmssd_ms} ms` },
                    { label: 'pNN50', val: `${selSession.pnn50_pct}%` },
                    { label: 'Arrhythmia Events', val: selSession.arrhythmia_events },
                    { label: 'QTc', val: `${selSession.qtc_ms} ms` },
                    { label: 'ST Changes', val: selSession.st_changes },
                    { label: 'Motion Artefact', val: `${selSession.motion_artefact_pct}%` },
                    { label: 'Signal Quality', val: `${selSession.signal_quality_pct}%` },
                    { label: 'Total Beats', val: (selSession.total_beats || 0).toLocaleString() },
                  ].map(k => (
                    <div key={k.label} className="col-6 col-md-3">
                      <div className="card bg-light text-center py-1">
                        <div className="fw-bold small">{k.val}</div>
                        <div className="text-muted" style={{ fontSize: '0.7rem' }}>{k.label}</div>
                      </div>
                    </div>
                  ))}
                </div>
                {selSession.events && selSession.events.length > 0 && (
                  <div>
                    <h6 className="fw-bold small mb-2">Event Distribution</h6>
                    {selSession.events.map((ev, j) => (
                      <div key={j} className="d-flex align-items-center mb-1">
                        <div className="small me-2" style={{ minWidth: 220 }}>{ev.type}</div>
                        <div className="progress flex-grow-1 me-2" style={{ height: 12 }}>
                          <div className="progress-bar"
                               style={{ width: `${Math.round(ev.count / selSession.total_beats * 100)}%`,
                                        backgroundColor: ev.color || '#6c757d' }} />
                        </div>
                        <div className="small text-muted">{ev.count.toLocaleString()}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
          {!selSession && (
            <div className="col-md-8 d-flex align-items-center justify-content-center">
              <div className="text-muted">← Select a session to view details</div>
            </div>
          )}
        </div>
      )}

      {/* Reference tab */}
      {tab === 'defs' && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card p-3">
              <h6 className="fw-bold mb-2">📖 ECG Metric Glossary</h6>
              <table className="table table-sm table-bordered mb-0">
                <thead className="table-light"><tr><th>Term</th><th>Unit</th><th>Normal Range</th><th>Clinical Note</th></tr></thead>
                <tbody>
                  {metrics.map((m, i) => (
                    <tr key={i}>
                      <td className="fw-bold small">{m.term}</td>
                      <td className="small">{m.unit}</td>
                      <td className="small">{m.normal_range}</td>
                      <td className="text-muted small">{m.clinical_note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card p-3 mb-3">
              <h6 className="fw-bold mb-2">🧠 Epilepsy Co-Deployment Context</h6>
              {epilepsy.map((e, i) => (
                <div key={i} className="mb-2">
                  <div className="fw-bold small">{e.finding}</div>
                  <div className="text-muted small">{e.detail}</div>
                </div>
              ))}
            </div>
            <div className="card p-3">
              <h6 className="fw-bold mb-2">📦 Device Specs</h6>
              <dl className="row mb-0 small">
                <dt className="col-6">Type</dt><dd className="col-6">{device.type}</dd>
                <dt className="col-6">Sampling Rate</dt><dd className="col-6">{device.sampling_rate_hz} Hz</dd>
                <dt className="col-6">Resolution</dt><dd className="col-6">{device.resolution_bits}-bit</dd>
                <dt className="col-6">FDA Status</dt><dd className="col-6">{device.fda_cleared}</dd>
                <dt className="col-6">Waterproof</dt><dd className="col-6">{device.waterproof}</dd>
                <dt className="col-6">Battery Life</dt><dd className="col-6">{device.battery_life_days} days</dd>
              </dl>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
