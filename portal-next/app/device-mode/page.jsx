'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const modeColor  = m => ({ online: 'success', offline: 'secondary', batch: 'warning' }[m] || 'secondary');
const qColor     = q => ({ good: 'success', fair: 'warning', poor: 'danger' }[q] || 'secondary');
const battColor  = b => b >= 60 ? 'success' : b >= 30 ? 'warning' : 'danger';
const batchColor = s => ({ queued: 'warning', processing: 'primary', done: 'success', failed: 'danger' }[s] || 'secondary');

export default function DeviceModeDashboard() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/device-mode/overview`).then(r => r.json()),
      fetch(`${API}/api/device-mode/breakdown`).then(r => r.json()),
      fetch(`${API}/api/device-mode/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-3">Loading Device Mode Manager…</div>;

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'devices',  label: '📡 Devices' },
    { id: 'batch',    label: '📦 Batch Queue' },
    { id: 'defs',     label: '📖 Reference' },
  ];

  const kpis      = ov.kpis || {};
  const byMode    = ov.by_mode || [];
  const byQuality = ov.by_quality || [];
  const byType    = ov.by_type || [];
  const batchSumm = ov.batch_queue_summary || {};
  const devices   = bd?.devices || [];
  const batchQ    = bd?.batch_queue || [];
  const modes     = defs?.modes || [];
  const sigQual   = defs?.signal_quality || {};
  const dtypes    = defs?.device_types || [];

  return (
    <div className="container-fluid py-3">
      <h2 className="fw-bold mb-1">📡 Device Mode Manager</h2>
      <p className="text-muted mb-3">
        Online · Offline · Batch — stream mode control, signal quality, batch upload queue · {kpis.total_devices} devices
      </p>

      {/* KPI row */}
      <div className="row g-2 mb-3">
        {[
          { label: 'Total Devices',   val: kpis.total_devices,            cls: 'primary' },
          { label: 'Online',          val: kpis.online,                   cls: 'success' },
          { label: 'Offline',         val: kpis.offline,                  cls: 'secondary' },
          { label: 'Batch',           val: kpis.batch,                    cls: 'warning' },
          { label: 'Avg Battery',     val: `${kpis.avg_battery_pct}%`,    cls: 'info' },
          { label: 'Streaming Now',   val: kpis.streaming_sessions,       cls: 'success' },
          { label: 'Avg Stream Hz',   val: `${kpis.avg_stream_hz} Hz`,    cls: 'info' },
          { label: 'Batch Queued',    val: batchSumm.queued,              cls: 'warning' },
          { label: 'Batch Processing',val: batchSumm.processing,          cls: 'primary' },
          { label: 'Batch Done',      val: batchSumm.done,                cls: 'success' },
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
          <div className="col-md-4">
            <div className="card p-3">
              <h6 className="fw-bold mb-2">📡 Devices by Mode</h6>
              {byMode.map((m, i) => (
                <div key={i} className="d-flex align-items-center mb-2">
                  <span className={`badge bg-${modeColor(m.mode)} me-2`} style={{ minWidth: 70 }}>{m.mode}</span>
                  <div className="progress flex-grow-1 me-2" style={{ height: 14 }}>
                    <div className={`progress-bar bg-${modeColor(m.mode)}`}
                         style={{ width: `${Math.round(m.count / kpis.total_devices * 100)}%` }} />
                  </div>
                  <span className="fw-bold small">{m.count}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="col-md-4">
            <div className="card p-3">
              <h6 className="fw-bold mb-2">📶 Devices by Signal Quality</h6>
              {byQuality.map((q, i) => (
                <div key={i} className="d-flex align-items-center mb-2">
                  <span className={`badge bg-${qColor(q.quality)} me-2`} style={{ minWidth: 70 }}>{q.quality}</span>
                  <div className="progress flex-grow-1 me-2" style={{ height: 14 }}>
                    <div className={`progress-bar bg-${qColor(q.quality)}`}
                         style={{ width: `${Math.round(q.count / kpis.total_devices * 100)}%` }} />
                  </div>
                  <span className="fw-bold small">{q.count}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="col-md-4">
            <div className="card p-3">
              <h6 className="fw-bold mb-2">🖥️ Devices by Type</h6>
              {byType.map((t2, i) => (
                <div key={i} className="d-flex align-items-center mb-2">
                  <span className="badge bg-secondary me-2" style={{ minWidth: 120 }}>{t2.type}</span>
                  <div className="progress flex-grow-1 me-2" style={{ height: 14 }}>
                    <div className="progress-bar bg-secondary"
                         style={{ width: `${Math.round(t2.count / kpis.total_devices * 100)}%` }} />
                  </div>
                  <span className="fw-bold small">{t2.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Devices tab */}
      {tab === 'devices' && (
        <div className="table-responsive">
          <table className="table table-sm table-hover table-bordered">
            <thead className="table-dark">
              <tr>
                <th>Device ID</th><th>Label</th><th>Type</th><th>Mode</th>
                <th>Stream Hz</th><th>Battery</th><th>Signal Quality</th><th>Patient</th><th>Last Sync</th>
              </tr>
            </thead>
            <tbody>
              {devices.map((d, i) => (
                <tr key={i}>
                  <td><code>{d.device_id}</code></td>
                  <td>{d.label}</td>
                  <td className="small">{d.type}</td>
                  <td><span className={`badge bg-${modeColor(d.mode)}`}>{d.mode}</span></td>
                  <td>{d.stream_hz ? `${d.stream_hz} Hz` : <span className="text-muted">—</span>}</td>
                  <td><span className={`badge bg-${battColor(d.battery_pct)}`}>{d.battery_pct}%</span></td>
                  <td><span className={`badge bg-${qColor(d.signal_quality)}`}>{d.signal_quality}</span></td>
                  <td>{d.patient}</td>
                  <td className="small text-muted">{(d.last_sync || '').replace('T', ' ').substring(0, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Batch Queue tab */}
      {tab === 'batch' && (
        <div>
          <div className="row g-2 mb-3">
            {[
              { label: 'Queued',     val: batchSumm.queued,     cls: 'warning' },
              { label: 'Processing', val: batchSumm.processing, cls: 'primary' },
              { label: 'Done',       val: batchSumm.done,       cls: 'success' },
            ].map(k => (
              <div key={k.label} className="col-4 col-md-2">
                <div className={`card border-${k.cls} text-center py-2`}>
                  <div className={`fs-4 fw-bold text-${k.cls}`}>{k.val}</div>
                  <div className="small text-muted">{k.label}</div>
                </div>
              </div>
            ))}
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-bordered">
              <thead className="table-dark">
                <tr><th>Job ID</th><th>Device</th><th>Patient</th><th>Duration (h)</th><th>Size (MB)</th><th>Status</th><th>ETA (min)</th></tr>
              </thead>
              <tbody>
                {batchQ.map((j, i) => (
                  <tr key={i}>
                    <td><code>{j.job_id}</code></td>
                    <td><code>{j.device_id}</code></td>
                    <td>{j.patient}</td>
                    <td>{j.duration_h}</td>
                    <td>{j.size_mb}</td>
                    <td><span className={`badge bg-${batchColor(j.status)}`}>{j.status}</span></td>
                    <td>{j.eta_min > 0 ? `${j.eta_min} min` : <span className="text-muted">—</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Reference tab */}
      {tab === 'defs' && (
        <div className="row g-3">
          <div className="col-md-5">
            <div className="card p-3">
              <h6 className="fw-bold mb-2">📡 Device Modes</h6>
              {modes.map((m, i) => (
                <div key={i} className="mb-3">
                  <span className={`badge bg-${modeColor(m.name)} me-2`}>{m.name}</span>
                  <span className="small text-muted">{m.description}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="col-md-4">
            <div className="card p-3">
              <h6 className="fw-bold mb-2">📶 Signal Quality Grades</h6>
              {Object.entries(sigQual).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <span className={`badge bg-${qColor(k)} me-2`}>{k}</span>
                  <span className="small text-muted">{v}</span>
                </div>
              ))}
            </div>
            <div className="card p-3 mt-3">
              <h6 className="fw-bold mb-2">🔄 Sync Policy</h6>
              <p className="small text-muted mb-0">{defs?.sync_policy}</p>
            </div>
          </div>
          <div className="col-md-3">
            <div className="card p-3">
              <h6 className="fw-bold mb-2">🖥️ Device Types</h6>
              {dtypes.map((t2, i) => (
                <span key={i} className="badge bg-secondary me-1 mb-1">{t2}</span>
              ))}
              <h6 className="fw-bold mt-3 mb-2">📦 Batch Statuses</h6>
              {(defs?.batch_statuses || []).map((s, i) => (
                <span key={i} className={`badge bg-${batchColor(s)} me-1 mb-1`}>{s}</span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
