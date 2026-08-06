'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const battColor = b => b >= 60 ? 'success' : b >= 30 ? 'warning' : 'danger';
const statusBadge = s => ({ active: 'success', charging: 'info', offline: 'secondary', error: 'danger' }[s] || 'secondary');
const qColor = q => ({ good: 'success', fair: 'warning', poor: 'danger' }[q] || 'secondary');

export default function EmotivWearableDashboard() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [search, setSearch] = useState('');
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/emotiv-wearable/overview`).then(r => r.json()),
      fetch(`${API}/api/emotiv-wearable/breakdown`).then(r => r.json()),
      fetch(`${API}/api/emotiv-wearable/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-3">Loading Emotiv Wearable fleet data…</div>;

  const TABS = [
    { id: 'overview',   label: '📊 Overview' },
    { id: 'fleet',      label: '🩺 Fleet' },
    { id: 'channels',   label: '🧠 EPOC+ Channels' },
    { id: 'patients',   label: '👤 Patient Confidence' },
    { id: 'defs',       label: '📖 Reference' },
  ];

  const kpis    = ov.kpis || {};
  const chQual  = ov.channel_quality || [];
  const devices = bd?.device_table || [];
  const patConf = bd?.patient_confidence || [];
  const readings= bd?.reading_log || [];
  const battDist= bd?.battery_distribution || [];
  const models  = defs?.device_models || [];
  const epocCh  = defs?.epoc_channels || [];
  const glossary= defs?.glossary || {};

  const filtered = devices.filter(d =>
    !search ||
    (d.patient_id || '').toLowerCase().includes(search.toLowerCase()) ||
    (d.device_id || '').toLowerCase().includes(search.toLowerCase()) ||
    (d.brand || '').toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="container-fluid py-3">
      <h2 className="fw-bold mb-1">🩺 Emotiv Wearable Dashboard</h2>
      <p className="text-muted mb-3">
        EEG-class wearables — Empatica Embrace2 · Byteflies Sensor Dot · BioStampRC · EPOC+ · {kpis.total_devices} devices fleet
      </p>

      {/* KPI row */}
      <div className="row g-2 mb-3">
        {[
          { label: 'Total Devices',    val: kpis.total_devices,              cls: 'primary' },
          { label: 'Active',           val: kpis.active_devices,             cls: 'success' },
          { label: 'Offline',          val: kpis.offline_devices,            cls: kpis.offline_devices > 0 ? 'warning' : 'success' },
          { label: 'Low Battery',      val: kpis.low_battery_devices,        cls: kpis.low_battery_devices > 0 ? 'danger' : 'success' },
          { label: 'Avg Battery',      val: `${kpis.avg_battery_pct}%`,      cls: 'info' },
          { label: 'Total Sessions',   val: kpis.total_sessions,             cls: 'primary' },
          { label: 'Seizures Detected',val: kpis.seizures_detected,          cls: kpis.seizures_detected > 0 ? 'danger' : 'success' },
          { label: 'Detection Rate',   val: `${kpis.seizure_detection_rate_pct}%`, cls: 'warning' },
          { label: 'Avg Confidence',   val: kpis.avg_detection_confidence,   cls: 'info' },
          { label: 'Avg Stress Score', val: kpis.avg_stress_score,           cls: 'warning' },
          { label: 'Avg Health Score', val: kpis.avg_health_score,           cls: 'success' },
          { label: 'Old Firmware',     val: kpis.outdated_firmware_count,    cls: kpis.outdated_firmware_count > 0 ? 'danger' : 'success' },
        ].map(k => (
          <div key={k.label} className="col-6 col-md-3 col-lg-2">
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
        <div>
          <div className="row g-3">
            {/* Channel quality chart */}
            <div className="col-md-6">
              <div className="card p-3">
                <h6 className="fw-bold mb-2">🧠 EPOC+ 14-Channel Contact Quality</h6>
                <div className="table-responsive" style={{ maxHeight: 320, overflowY: 'auto' }}>
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-light">
                      <tr><th>Channel</th><th>Quality %</th><th>Status</th><th>Bar</th></tr>
                    </thead>
                    <tbody>
                      {chQual.map(ch => (
                        <tr key={ch.channel}>
                          <td className="fw-bold">{ch.channel}</td>
                          <td>{ch.quality_pct}%</td>
                          <td><span className={`badge bg-${qColor(ch.status)}`}>{ch.status}</span></td>
                          <td>
                            <div className="progress" style={{ height: 10 }}>
                              <div className={`progress-bar bg-${qColor(ch.status)}`}
                                   style={{ width: `${ch.quality_pct}%` }} />
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Battery distribution */}
            <div className="col-md-3">
              <div className="card p-3">
                <h6 className="fw-bold mb-2">🔋 Battery Distribution</h6>
                {battDist.length > 0 ? (
                  <table className="table table-sm mb-0">
                    <thead className="table-light"><tr><th>Range</th><th>Devices</th></tr></thead>
                    <tbody>
                      {battDist.map((b, i) => (
                        <tr key={i}>
                          <td>{b.range || b.label}</td>
                          <td><span className="badge bg-secondary">{b.count}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="text-muted small">No battery data</div>
                )}
              </div>
            </div>

            {/* Recent readings */}
            <div className="col-md-3">
              <div className="card p-3">
                <h6 className="fw-bold mb-2">📡 Recent Readings</h6>
                <div style={{ maxHeight: 300, overflowY: 'auto' }}>
                  {readings.slice(0, 8).map((r, i) => (
                    <div key={i} className="border-bottom py-1 small">
                      <div className="fw-bold">{r.patient_id || r.device_id}</div>
                      <div className="text-muted">{r.timestamp || r.time || '—'}</div>
                      {r.seizure_detected !== undefined && (
                        <span className={`badge bg-${r.seizure_detected ? 'danger' : 'success'} me-1`}>
                          {r.seizure_detected ? '⚠ Seizure' : 'Normal'}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Fleet tab */}
      {tab === 'fleet' && (
        <div>
          <div className="mb-2">
            <input className="form-control form-control-sm w-auto d-inline-block"
              placeholder="Search patient / device / brand…"
              value={search} onChange={e => setSearch(e.target.value)} />
            <span className="ms-2 text-muted small">{filtered.length} of {devices.length} devices</span>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-bordered">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th><th>Device</th><th>Brand</th><th>Status</th>
                  <th>Battery</th><th>Connectivity</th><th>Firmware</th>
                  <th>Sessions</th><th>Seizures</th><th>Confidence</th><th>Last Sync</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((d, i) => (
                  <tr key={i}>
                    <td className="fw-bold">{d.patient_id}</td>
                    <td><code>{d.device_id}</code></td>
                    <td>{d.brand}</td>
                    <td><span className={`badge bg-${statusBadge(d.status)}`}>{d.status}</span></td>
                    <td>
                      <span className={`badge bg-${battColor(d.battery_level)}`}>{d.battery_level}%</span>
                    </td>
                    <td>{d.connectivity}</td>
                    <td><code>{d.firmware_version}</code></td>
                    <td>{d.total_sessions}</td>
                    <td>{d.seizures_detected > 0
                      ? <span className="badge bg-danger">{d.seizures_detected}</span>
                      : <span className="badge bg-success">0</span>}</td>
                    <td>{d.avg_confidence}</td>
                    <td className="text-muted small">{(d.last_sync || '').replace('T', ' ').substring(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Channels tab */}
      {tab === 'channels' && (
        <div className="row g-3">
          <div className="col-md-8">
            <div className="card p-3">
              <h6 className="fw-bold mb-2">🧠 EPOC+ 14-Channel Electrode Contact Quality</h6>
              <div className="row g-2">
                {chQual.map(ch => (
                  <div key={ch.channel} className="col-6 col-md-3">
                    <div className={`card border-${qColor(ch.status)} text-center p-2`}>
                      <div className="fw-bold">{ch.channel}</div>
                      <div className={`text-${qColor(ch.status)} fw-bold`}>{ch.quality_pct}%</div>
                      <span className={`badge bg-${qColor(ch.status)}`}>{ch.status}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="card p-3">
              <h6 className="fw-bold mb-2">📍 Channel Map (10-20 Layout)</h6>
              <div className="text-muted small mb-2">Emotiv EPOC+ electrode positions</div>
              {epocCh.map((ch, i) => (
                <span key={i} className="badge bg-secondary me-1 mb-1">{ch}</span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Patient confidence tab */}
      {tab === 'patients' && (
        <div className="card p-3">
          <h6 className="fw-bold mb-2">👤 Per-Patient Seizure Detection Confidence</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-bordered">
              <thead className="table-dark">
                <tr><th>Patient</th><th>Device</th><th>Seizures</th><th>Avg Confidence</th><th>Confidence Bar</th></tr>
              </thead>
              <tbody>
                {patConf.map((p, i) => (
                  <tr key={i}>
                    <td className="fw-bold">{p.patient_id}</td>
                    <td><code>{p.device_id}</code></td>
                    <td>{p.seizures_detected}</td>
                    <td>{p.avg_confidence}</td>
                    <td>
                      <div className="progress" style={{ height: 12 }}>
                        <div className="progress-bar bg-warning"
                             style={{ width: `${Math.round(p.avg_confidence * 100)}%` }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Reference tab */}
      {tab === 'defs' && (
        <div>
          <div className="card p-3 mb-3">
            <h6 className="fw-bold mb-2">ℹ️ Overview</h6>
            <p className="text-muted small mb-0">{defs?.overview}</p>
          </div>
          <div className="row g-3">
            <div className="col-md-8">
              <div className="card p-3">
                <h6 className="fw-bold mb-2">📦 Device Models</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-light">
                      <tr><th>Brand</th><th>Type</th><th>Ch</th><th>Hz</th><th>Connectivity</th><th>FDA</th><th>Battery (h)</th></tr>
                    </thead>
                    <tbody>
                      {models.map((m, i) => (
                        <tr key={i}>
                          <td className="fw-bold">{m.brand}</td>
                          <td>{m.type}</td>
                          <td>{m.channels}</td>
                          <td>{m.sample_rate_hz}</td>
                          <td>{m.connectivity}</td>
                          <td>{m.fda_cleared
                            ? <span className="badge bg-success">FDA ✓</span>
                            : <span className="badge bg-secondary">No</span>}</td>
                          <td>{m.battery_hours}h</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card p-3">
                <h6 className="fw-bold mb-2">📖 Glossary</h6>
                {Object.entries(glossary).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <div className="fw-bold small">{k}</div>
                    <div className="text-muted small">{v}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
