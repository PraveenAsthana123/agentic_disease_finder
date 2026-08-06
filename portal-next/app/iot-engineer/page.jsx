'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusColor = s =>
  s === 'online' ? 'success' : s === 'offline' ? 'danger' : s === 'maintenance' ? 'warning' : 'secondary';

const severityColor = s =>
  s === 'critical' ? 'danger' : s === 'warning' ? 'warning' : 'info';

const inferColor = s =>
  s === 'running' ? 'success' : s === 'error' ? 'danger' : 'secondary';

const gwColor = s =>
  s === 'online' ? 'success' : s === 'degraded' ? 'warning' : 'danger';

export default function IoTEngineerPage() {
  const [ov,    setOv]    = useState(null);
  const [bd,    setBd]    = useState(null);
  const [defs,  setDefs]  = useState(null);
  const [ei,    setEi]    = useState(null);
  const [ss,    setSs]    = useState(null);
  const [gw,    setGw]    = useState(null);
  const [sos,   setSos]   = useState(null);
  const [tab,   setTab]   = useState('overview');
  const [sel,   setSel]   = useState(null);

  useEffect(() => {
    fetch(`${API}/api/iot-engineer/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/iot-engineer/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/iot-engineer/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
    fetch(`${API}/api/iot-engineer/edge-inference`).then(r => r.json()).then(setEi).catch(() => {});
    fetch(`${API}/api/iot-engineer/stream-sim`).then(r => r.json()).then(setSs).catch(() => {});
    fetch(`${API}/api/iot-engineer/gateway-health`).then(r => r.json()).then(setGw).catch(() => {});
    fetch(`${API}/api/iot-engineer/sos-escalation`).then(r => r.json()).then(setSos).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const k = ov.kpis || {};
  const kpis = [
    { label: 'Total Devices',        value: k.total_devices,                               color: 'primary' },
    { label: 'Devices Online',        value: k.devices_online,                              color: 'success' },
    { label: 'Online %',             value: k.devices_online_pct != null ? `${k.devices_online_pct.toFixed(1)}%` : '—', color: 'info' },
    { label: 'Gateway Uptime',       value: k.avg_gateway_uptime_pct != null ? `${k.avg_gateway_uptime_pct.toFixed(1)}%` : '—', color: 'success' },
    { label: 'Avg Latency (ms)',     value: k.avg_stream_latency_ms != null ? k.avg_stream_latency_ms.toFixed(1) : '—', color: 'warning' },
    { label: 'SOS Alerts',           value: k.sos_alerts_fired,                            color: 'danger'  },
    { label: 'Battery/Signal Low',   value: k.battery_signal_low,                          color: 'warning' },
    { label: 'Alert Resolution',     value: k.alert_resolution_rate != null ? `${k.alert_resolution_rate.toFixed(0)}%` : '—', color: 'success' },
  ];

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'devices',    label: `Device Fleet${bd ? ` (${(bd.devices || []).length})` : ''}` },
    { id: 'alerts',     label: `Alerts${bd ? ` (${(bd.devices || []).reduce((a, d) => a + (d.alert_count || 0), 0)})` : ''}` },
    { id: 'edge-ai',   label: `Edge AI${ei ? ` (${ei.kpis?.devices_running ?? 0} running)` : ''}` },
    { id: 'stream-sos', label: `Stream & SOS${sos ? ` (${sos.kpis?.sos_events_total ?? 0} events)` : ''}` },
    { id: 'definitions', label: 'Standards' },
  ];

  const devices = (bd || {}).devices || [];
  const selDev  = devices.find(d => d.device_id === sel);
  const allAlerts = devices.flatMap(d => (d.alerts || []).map(a => ({ ...a, device_type: d.device_type, patient_name: d.patient_name })));

  return (
    <div>
      <h3>&#x1f4e1; IoT Engineer Dashboard</h3>
      <p className="text-muted small">
        Wearable EEG &amp; implantable device fleet — online status, gateway uptime, stream latency,
        SOS alerts, battery/signal telemetry, firmware versions. IEC 62304 / IEC 80001 / HIPAA.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {kpis.map(kp => (
          <div key={kp.label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${kp.color}`}>{kp.value ?? '—'}</div>
                <div className="text-muted" style={{ fontSize: '0.72rem' }}>{kp.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div className="row">

          {/* Device Status */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-success text-white py-2 small fw-bold">Device Status</div>
              <div className="card-body p-2">
                {(ov.device_status_distribution || []).map(s => {
                  const max = Math.max(...(ov.device_status_distribution || []).map(x => x.count), 1);
                  return (
                    <div key={s.status} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '90px' }}>
                        <span className={`badge bg-${statusColor(s.status)}`}>{s.status}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${statusColor(s.status)}`}
                             style={{ width: `${(s.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{s.count}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Device Type */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-primary text-white py-2 small fw-bold">Device Types</div>
              <div className="card-body p-2">
                {(ov.device_type_distribution || []).map(t => {
                  const max = Math.max(...(ov.device_type_distribution || []).map(x => x.count), 1);
                  return (
                    <div key={t.type} className="d-flex align-items-center mb-1">
                      <span className="small me-2 text-truncate" style={{ minWidth: '120px', fontSize: '0.7rem' }}>
                        {t.type.replace(/_/g, ' ')}
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className="progress-bar bg-primary"
                             style={{ width: `${(t.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{t.count}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Alert Severity */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-danger text-white py-2 small fw-bold">Alert Severity</div>
              <div className="card-body p-2">
                {(ov.alert_severity_distribution || []).map(a => {
                  const max = Math.max(...(ov.alert_severity_distribution || []).map(x => x.count), 1);
                  return (
                    <div key={a.severity} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '80px' }}>
                        <span className={`badge bg-${severityColor(a.severity)}`}>{a.severity}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${severityColor(a.severity)}`}
                             style={{ width: `${(a.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{a.count}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Gateway Uptime */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-info text-white py-2 small fw-bold">Gateway Uptime</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>GW</th><th>Location</th><th className="text-end">Uptime</th><th style={{ width: '35%' }}>Bar</th></tr></thead>
                  <tbody>
                    {(ov.gateway_uptime_bars || []).map(g => (
                      <tr key={g.gateway_id}>
                        <td className="small fw-bold">{g.gateway_id}</td>
                        <td className="small">{g.location}</td>
                        <td className="text-end small">{g.uptime_pct != null ? `${g.uptime_pct.toFixed(1)}%` : '—'}</td>
                        <td>
                          <div className="progress" style={{ height: '14px' }}>
                            <div className={`progress-bar ${g.uptime_pct >= 95 ? 'bg-success' : g.uptime_pct >= 90 ? 'bg-warning' : 'bg-danger'}`}
                                 style={{ width: `${(g.uptime_pct || 0).toFixed(0)}%` }} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Alert Types */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">Alert Types</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Type</th><th className="text-end">Count</th><th style={{ width: '40%' }}>Bar</th></tr></thead>
                  <tbody>
                    {(ov.alert_type_distribution || []).map(a => {
                      const max = Math.max(...(ov.alert_type_distribution || []).map(x => x.count), 1);
                      return (
                        <tr key={a.type}>
                          <td className="small">{a.type.replace(/_/g, ' ')}</td>
                          <td className="text-end small">{a.count}</td>
                          <td>
                            <div className="progress" style={{ height: '14px' }}>
                              <div className="progress-bar bg-warning" style={{ width: `${(a.count / max * 100).toFixed(0)}%` }} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Battery Histogram */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-dark text-white py-2 small fw-bold">Battery Level Distribution</div>
              <div className="card-body p-2">
                {(ov.battery_histogram || []).map(b => {
                  const max = Math.max(...(ov.battery_histogram || []).map(x => x.count), 1);
                  const color = b.range === '0-20%' ? 'danger' : b.range === '20-40%' ? 'warning' : 'success';
                  return (
                    <div key={b.range} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '70px' }}>{b.range}</span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className={`progress-bar bg-${color}`}
                             style={{ width: `${(b.count / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{b.count}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Firmware Versions */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-secondary text-white py-2 small fw-bold">Firmware Versions</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Version</th><th className="text-end">Devices</th><th style={{ width: '40%' }}>Bar</th></tr></thead>
                  <tbody>
                    {(ov.firmware_distribution || []).filter(f => f.count > 0).map(f => {
                      const max = Math.max(...(ov.firmware_distribution || []).map(x => x.count), 1);
                      return (
                        <tr key={f.version}>
                          <td className="small fw-bold">{f.version}</td>
                          <td className="text-end small">{f.count}</td>
                          <td>
                            <div className="progress" style={{ height: '14px' }}>
                              <div className="progress-bar bg-secondary" style={{ width: `${(f.count / max * 100).toFixed(0)}%` }} />
                            </div>
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

      {/* ── Device Fleet ── */}
      {tab === 'devices' && (
        <div>
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-header bg-dark text-white py-2 small fw-bold">
              Device Fleet ({devices.length} devices)
            </div>
            <div className="card-body p-2">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Device ID</th><th>Type</th><th>Status</th><th>Patient</th>
                      <th>Gateway</th><th className="text-end">Battery</th>
                      <th className="text-end">Latency</th><th className="text-end">Alerts</th><th>FW</th><th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {devices.map(d => (
                      <tr key={d.device_id} className={sel === d.device_id ? 'table-active' : ''}>
                        <td className="small fw-bold">{d.device_id}</td>
                        <td className="small">{(d.device_type || '').replace(/_/g, ' ')}</td>
                        <td><span className={`badge bg-${statusColor(d.status)}`} style={{ fontSize: '0.65rem' }}>{d.status}</span></td>
                        <td className="small">{d.patient_id || '—'}</td>
                        <td className="small">{d.gateway_id || '—'}</td>
                        <td className="text-end small">
                          {d.battery_pct != null
                            ? <span className={d.battery_pct < 20 ? 'text-danger fw-bold' : d.battery_pct < 40 ? 'text-warning' : 'text-success'}>
                                {d.battery_pct.toFixed(0)}%
                              </span>
                            : '—'}
                        </td>
                        <td className="text-end small">{d.latency_ms != null ? `${d.latency_ms.toFixed(0)} ms` : '—'}</td>
                        <td className="text-end small">{d.alert_count ?? 0}</td>
                        <td className="small">{d.firmware_version || '—'}</td>
                        <td>
                          <button className="btn btn-outline-primary btn-sm py-0 px-1" style={{ fontSize: '0.7rem' }}
                                  onClick={() => setSel(sel === d.device_id ? null : d.device_id)}>
                            {sel === d.device_id ? 'Hide' : 'Detail'}
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Device detail */}
          {sel && selDev && (
            <div className="card border-primary shadow-sm">
              <div className="card-header bg-primary text-white py-1 small fw-bold">
                {selDev.device_id} — {selDev.patient_name || selDev.patient_id || '—'} · Device Detail
              </div>
              <div className="card-body p-3">
                <div className="row">
                  <div className="col-md-6">
                    <table className="table table-sm mb-0">
                      <tbody>
                        <tr><td className="fw-bold small">Type</td><td className="small">{(selDev.device_type || '').replace(/_/g, ' ')}</td></tr>
                        <tr><td className="fw-bold small">Status</td>
                          <td><span className={`badge bg-${statusColor(selDev.status)}`}>{selDev.status}</span></td></tr>
                        <tr><td className="fw-bold small">Firmware</td><td className="small">{selDev.firmware_version}</td></tr>
                        <tr><td className="fw-bold small">Location</td><td className="small">{selDev.location || '—'}</td></tr>
                        <tr><td className="fw-bold small">Last Seen</td><td className="small">{selDev.last_seen ? new Date(selDev.last_seen).toLocaleString() : '—'}</td></tr>
                      </tbody>
                    </table>
                  </div>
                  <div className="col-md-6">
                    <table className="table table-sm mb-0">
                      <tbody>
                        <tr><td className="fw-bold small">Battery</td>
                          <td className="small">{selDev.battery_pct != null ? `${selDev.battery_pct.toFixed(1)}%` : '—'}</td></tr>
                        <tr><td className="fw-bold small">Signal (dBm)</td>
                          <td className="small">{selDev.signal_strength_dbm != null ? selDev.signal_strength_dbm.toFixed(1) : '—'}</td></tr>
                        <tr><td className="fw-bold small">Latency</td>
                          <td className="small">{selDev.latency_ms != null ? `${selDev.latency_ms.toFixed(1)} ms` : '—'}</td></tr>
                        <tr><td className="fw-bold small">Gateway</td>
                          <td className="small">{selDev.gateway_id || '—'} ({selDev.gateway_status || '—'})</td></tr>
                        <tr><td className="fw-bold small">Gateway Uptime</td>
                          <td className="small">{selDev.gateway_uptime_pct != null ? `${selDev.gateway_uptime_pct}%` : '—'}</td></tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                {(selDev.alerts || []).length > 0 && (
                  <div className="mt-2">
                    <div className="small fw-bold mb-1">Alerts ({selDev.alerts.length})</div>
                    <table className="table table-sm mb-0">
                      <thead><tr><th>Type</th><th>Severity</th><th>Ack</th><th>Resolved</th><th>Time</th></tr></thead>
                      <tbody>
                        {selDev.alerts.map((a, i) => (
                          <tr key={i}>
                            <td className="small">{(a.alert_type || '').replace(/_/g, ' ')}</td>
                            <td><span className={`badge bg-${severityColor(a.severity)}`} style={{ fontSize: '0.65rem' }}>{a.severity}</span></td>
                            <td className="small">{a.acknowledged ? '✓' : '—'}</td>
                            <td className="small">{a.resolved ? '✓' : '—'}</td>
                            <td className="small">{a.timestamp ? new Date(a.timestamp).toLocaleDateString() : '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Alerts ── */}
      {tab === 'alerts' && (
        <div className="card shadow-sm border-0">
          <div className="card-header bg-danger text-white py-2 small fw-bold">
            All Alerts ({allAlerts.length})
          </div>
          <div className="card-body p-2">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr><th>Device</th><th>Type</th><th>Severity</th><th>Patient</th><th>Ack</th><th>Resolved</th><th>Time</th></tr>
                </thead>
                <tbody>
                  {allAlerts.map((a, i) => (
                    <tr key={i}>
                      <td className="small fw-bold">{a.device_id}</td>
                      <td className="small">{(a.alert_type || '').replace(/_/g, ' ')}</td>
                      <td><span className={`badge bg-${severityColor(a.severity)}`} style={{ fontSize: '0.65rem' }}>{a.severity}</span></td>
                      <td className="small">{a.patient_id || '—'}</td>
                      <td className="small">{a.acknowledged ? '✓' : <span className="text-danger">✗</span>}</td>
                      <td className="small">{a.resolved ? '✓' : <span className="text-danger">✗</span>}</td>
                      <td className="small">{a.timestamp ? new Date(a.timestamp).toLocaleDateString() : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Edge AI Inference ── */}
      {tab === 'edge-ai' && (
        <div>
          {!ei && <div className="spinner-border text-primary" />}
          {ei && (() => {
            const k = ei.kpis || {};
            const acc = ei.accuracy || {};
            const mi  = ei.model_info || {};
            const devs = ei.per_device || [];
            const eiKpis = [
              { label: 'Running',          value: k.devices_running,      color: 'success' },
              { label: 'Idle',             value: k.devices_idle,         color: 'secondary' },
              { label: 'Error',            value: k.devices_error,        color: 'danger' },
              { label: 'Windows (24h)',    value: (k.total_windows_24h || 0).toLocaleString(), color: 'primary' },
              { label: 'Seizure Windows',  value: (k.total_seizure_windows || 0).toLocaleString(), color: 'warning' },
              { label: 'Fleet Sz Rate',    value: `${(k.fleet_seizure_rate_pct || 0).toFixed(3)}%`, color: 'warning' },
              { label: 'SOS Triggered',    value: k.sos_triggered,        color: 'danger' },
              { label: 'Avg Infer (ms)',   value: k.avg_infer_latency_ms != null ? `${k.avg_infer_latency_ms}` : '—', color: 'info' },
            ];
            return (
              <>
                {/* KPI row */}
                <div className="row mb-3">
                  {eiKpis.map(kp => (
                    <div key={kp.label} className="col-6 col-md-3 col-lg-2 mb-2">
                      <div className="card text-center shadow-sm border-0">
                        <div className="card-body py-2 px-1">
                          <div className={`h4 mb-0 text-${kp.color}`}>{kp.value ?? '—'}</div>
                          <div className="text-muted" style={{ fontSize: '0.70rem' }}>{kp.label}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="row mb-3">
                  {/* Model Info */}
                  <div className="col-md-5 mb-3">
                    <div className="card shadow-sm border-0 h-100">
                      <div className="card-header bg-dark text-white py-2 small fw-bold">Edge Model Info</div>
                      <div className="card-body p-3">
                        <table className="table table-sm mb-0">
                          <tbody>
                            <tr><td className="small fw-bold">Version</td><td className="small">{mi.version || '—'}</td></tr>
                            <tr><td className="small fw-bold">Runtime</td><td className="small">{mi.runtime || '—'}</td></tr>
                            <tr><td className="small fw-bold">Confidence Gate</td>
                              <td className="small fw-bold text-warning">{mi.confidence_gate != null ? `≥ ${mi.confidence_gate}` : '—'}</td></tr>
                          </tbody>
                        </table>
                        <div className="mt-2">
                          <div className="small fw-bold mb-1">Models by Device Type</div>
                          {(mi.window_types || []).map(wt => (
                            <div key={wt.device_type} className="small border-bottom py-1">
                              <span className="fw-bold">{wt.device_type.replace(/_/g,' ')}</span>
                              {' — '}{wt.model} · {wt.window_s}s window · {wt.sr_hz} Hz · {wt.channels}ch
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Fleet Accuracy */}
                  <div className="col-md-3 mb-3">
                    <div className="card shadow-sm border-0 h-100">
                      <div className="card-header bg-primary text-white py-2 small fw-bold">Fleet Accuracy</div>
                      <div className="card-body p-3">
                        <table className="table table-sm mb-2">
                          <tbody>
                            <tr><td className="small fw-bold">Precision</td>
                              <td className="small">{acc.precision != null ? `${(acc.precision * 100).toFixed(1)}%` : '—'}</td></tr>
                            <tr><td className="small fw-bold">Recall</td>
                              <td className="small">{acc.recall != null ? `${(acc.recall * 100).toFixed(1)}%` : '—'}</td></tr>
                            <tr><td className="small fw-bold">F1</td>
                              <td className="small">{acc.f1 != null ? acc.f1.toFixed(3) : '—'}</td></tr>
                          </tbody>
                        </table>
                        <div className="text-muted" style={{ fontSize: '0.68rem' }}>{acc.note}</div>
                      </div>
                    </div>
                  </div>

                  {/* Inference status pie */}
                  <div className="col-md-4 mb-3">
                    <div className="card shadow-sm border-0 h-100">
                      <div className="card-header bg-info text-white py-2 small fw-bold">Inference Status</div>
                      <div className="card-body p-3">
                        {[
                          { s: 'running', c: k.devices_running, color: 'success' },
                          { s: 'idle',    c: k.devices_idle,    color: 'secondary' },
                          { s: 'error',   c: k.devices_error,   color: 'danger' },
                        ].map(row => {
                          const total = devs.length || 1;
                          return (
                            <div key={row.s} className="d-flex align-items-center mb-2">
                              <span className={`badge bg-${row.color} me-2`} style={{ minWidth: 70 }}>{row.s}</span>
                              <div className="progress flex-grow-1" style={{ height: 18 }}>
                                <div className={`progress-bar bg-${row.color}`}
                                     style={{ width: `${((row.c || 0) / total * 100).toFixed(0)}%` }}>
                                  <span style={{ fontSize: '0.65rem' }}>{row.c}</span>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Per-device table */}
                <div className="card shadow-sm border-0">
                  <div className="card-header bg-success text-white py-2 small fw-bold">
                    Per-Device Edge Inference ({devs.length} devices)
                  </div>
                  <div className="card-body p-2">
                    <div className="table-responsive">
                      <table className="table table-sm table-hover mb-0">
                        <thead className="table-dark">
                          <tr>
                            <th>Device</th><th>Type</th><th>Patient</th><th>Status</th>
                            <th>Model</th><th className="text-end">Windows (24h)</th>
                            <th className="text-end">Sz Windows</th>
                            <th className="text-end">Latest Prob</th>
                            <th className="text-end">Latency (ms)</th>
                            <th>SOS</th>
                          </tr>
                        </thead>
                        <tbody>
                          {devs.map(d => (
                            <tr key={d.device_id}>
                              <td className="small fw-bold">{d.device_id}</td>
                              <td className="small" style={{ fontSize: '0.68rem' }}>{(d.device_type || '').replace(/_/g, ' ')}</td>
                              <td className="small">{d.patient_id || '—'}</td>
                              <td><span className={`badge bg-${inferColor(d.infer_status)}`} style={{ fontSize: '0.65rem' }}>{d.infer_status}</span></td>
                              <td className="small" style={{ fontSize: '0.68rem' }}>{d.model || '—'}</td>
                              <td className="text-end small">{d.windows_24h > 0 ? d.windows_24h.toLocaleString() : '—'}</td>
                              <td className="text-end small">{d.seizure_windows > 0 ? d.seizure_windows.toLocaleString() : '—'}</td>
                              <td className="text-end small">
                                {d.infer_status === 'running'
                                  ? <span className={d.latest_sz_prob >= 0.7 ? 'text-danger fw-bold' : d.latest_sz_prob >= 0.4 ? 'text-warning' : ''}>
                                      {(d.latest_sz_prob * 100).toFixed(1)}%
                                    </span>
                                  : '—'}
                              </td>
                              <td className="text-end small">
                                {d.infer_latency_ms > 0 ? `${d.infer_latency_ms.toFixed(1)}` : '—'}
                              </td>
                              <td className="small text-center">
                                {d.sos_triggered ? <span className="badge bg-danger">🚨 YES</span> : <span className="text-muted">—</span>}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </>
            );
          })()}
        </div>
      )}

      {/* ── Standards / Definitions ── */}
      {tab === 'definitions' && defs && (
        <div>
          {(defs.sections || []).map(s => (
            <div key={s.heading} className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-dark text-white py-2 small fw-bold">{s.heading}</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {(s.items || []).map(item => (
                      <tr key={item.term}>
                        <td className="small fw-bold" style={{ width: '28%', verticalAlign: 'top' }}>{item.term}</td>
                        <td className="small">{item.detail}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Stream & SOS ── */}
      {tab === 'stream-sos' && (
        <div>
          {/* Stream KPIs */}
          {ss && (
            <div className="mb-4">
              <h6 className="fw-bold mb-2">EEG Stream — Packet Buffering</h6>
              <div className="row mb-2">
                {[
                  { label: 'Devices Streaming', value: ss.kpis?.total_devices, color: 'primary' },
                  { label: 'Active',             value: ss.kpis?.devices_active, color: 'success' },
                  { label: 'Buffering',          value: ss.kpis?.devices_buffering, color: 'warning' },
                  { label: 'Reconnecting',       value: ss.kpis?.devices_reconnecting, color: 'info' },
                  { label: 'Dropped',            value: ss.kpis?.devices_dropped, color: 'danger' },
                  { label: 'Avg Pkts/s',         value: ss.kpis?.avg_packets_per_sec?.toFixed(1), color: 'primary' },
                  { label: 'Avg Drop %',         value: ss.kpis?.avg_drop_rate_pct?.toFixed(2) + '%', color: 'warning' },
                  { label: 'Disconnects 24h',    value: ss.kpis?.total_disconnects_24h, color: 'danger' },
                ].map(kp => (
                  <div key={kp.label} className="col-6 col-md-3 col-lg-2 mb-2">
                    <div className="card text-center shadow-sm border-0">
                      <div className="card-body py-2 px-1">
                        <div className={`h4 mb-0 text-${kp.color}`}>{kp.value ?? '—'}</div>
                        <div className="text-muted" style={{ fontSize: '0.7rem' }}>{kp.label}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="card shadow-sm border-0 mb-3">
                <div className="card-header bg-primary text-white py-2 small fw-bold">Per-Device Stream Status</div>
                <div className="card-body p-2">
                  <div className="table-responsive">
                    <table className="table table-sm mb-0">
                      <thead><tr>
                        <th>Device</th><th>Patient</th><th>Status</th>
                        <th>Pkts/s</th><th>Buf %</th><th>Drop %</th>
                        <th>Disconn.</th><th>Seizure P</th>
                      </tr></thead>
                      <tbody>
                        {(ss.per_device || []).map(d => (
                          <tr key={d.device_id}>
                            <td className="small">{d.device_id}</td>
                            <td className="small">{d.patient_name}</td>
                            <td><span className={`badge bg-${d.stream_status === 'active' ? 'success' : d.stream_status === 'buffering' ? 'warning' : d.stream_status === 'reconnecting' ? 'info' : 'danger'}`}>{d.stream_status}</span></td>
                            <td className="small">{d.packets_per_sec?.toFixed(0)}</td>
                            <td className="small">{d.buffer_fill_pct?.toFixed(0)}%</td>
                            <td className="small">{d.drop_rate_pct?.toFixed(1)}%</td>
                            <td className="small">{d.disconnect_events_24h}</td>
                            <td className="small">{d.seizure_flag ? <span className="text-danger fw-bold">{d.seizure_prob?.toFixed(3)}</span> : d.seizure_prob?.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Gateway Health */}
          {gw && (
            <div className="mb-4">
              <h6 className="fw-bold mb-2">Gateway Heartbeat & Reconnect</h6>
              <div className="row mb-2">
                {[
                  { label: 'Gateways',         value: gw.kpis?.gateways_total, color: 'primary' },
                  { label: 'Online',            value: gw.kpis?.gateways_online, color: 'success' },
                  { label: 'Degraded',          value: gw.kpis?.gateways_degraded, color: 'warning' },
                  { label: 'Offline',           value: gw.kpis?.gateways_offline, color: 'danger' },
                  { label: 'Avg Uptime %',      value: gw.kpis?.avg_uptime_pct?.toFixed(1) + '%', color: 'success' },
                  { label: 'Reconnects 24h',    value: gw.kpis?.total_reconnects_24h, color: 'warning' },
                  { label: 'HB Interval (s)',   value: gw.kpis?.avg_hb_interval_s?.toFixed(1), color: 'info' },
                ].map(kp => (
                  <div key={kp.label} className="col-6 col-md-3 col-lg-2 mb-2">
                    <div className="card text-center shadow-sm border-0">
                      <div className="card-body py-2 px-1">
                        <div className={`h4 mb-0 text-${kp.color}`}>{kp.value ?? '—'}</div>
                        <div className="text-muted" style={{ fontSize: '0.7rem' }}>{kp.label}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="row mb-3">
                <div className="col-md-7">
                  <div className="card shadow-sm border-0">
                    <div className="card-header bg-success text-white py-2 small fw-bold">Gateway Status</div>
                    <div className="card-body p-2">
                      <table className="table table-sm mb-0">
                        <thead><tr><th>ID</th><th>Location</th><th>Status</th><th>Uptime %</th><th>HB (s)</th><th>Jitter (s)</th><th>Reconn.</th><th>Devices</th></tr></thead>
                        <tbody>
                          {(gw.per_gateway || []).map(g => (
                            <tr key={g.gateway_id}>
                              <td className="small">{g.gateway_id}</td>
                              <td className="small">{g.location}</td>
                              <td><span className={`badge bg-${gwColor(g.status)}`}>{g.status}</span></td>
                              <td className="small">{g.uptime_pct?.toFixed(1)}</td>
                              <td className="small">{g.heartbeat_interval_s?.toFixed(1)}</td>
                              <td className="small">{g.heartbeat_jitter_s?.toFixed(2)}</td>
                              <td className="small">{g.reconnects_24h}</td>
                              <td className="small">{g.connected_devices}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
                <div className="col-md-5">
                  <div className="card shadow-sm border-0">
                    <div className="card-header bg-warning text-dark py-2 small fw-bold">Reconnect Log (24h)</div>
                    <div className="card-body p-2" style={{ maxHeight: 220, overflowY: 'auto' }}>
                      <table className="table table-sm mb-0">
                        <thead><tr><th>Time</th><th>GW</th><th>Reason</th><th>Down(s)</th></tr></thead>
                        <tbody>
                          {(gw.reconnect_log || []).map((ev, i) => (
                            <tr key={i}>
                              <td className="small" style={{ fontSize: '0.7rem' }}>{ev.ts?.slice(11, 16)}</td>
                              <td className="small">{ev.gateway_id}</td>
                              <td className="small">{ev.reason}</td>
                              <td className="small">{ev.downtime_s}s</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* SOS Escalation */}
          {sos && (
            <div className="mb-4">
              <h6 className="fw-bold mb-2">SOS Escalation Chain</h6>
              <div className="row mb-2">
                {[
                  { label: 'SOS Events',       value: sos.kpis?.sos_events_total, color: 'danger' },
                  { label: 'EMS Dispatched',   value: sos.kpis?.ems_dispatched, color: 'danger' },
                  { label: 'Avg Elapsed (s)',   value: sos.kpis?.avg_total_elapsed_s?.toFixed(0), color: 'warning' },
                  { label: 'Under 2 min %',    value: sos.kpis?.under_2min_pct?.toFixed(1) + '%', color: 'success' },
                  { label: 'Reach Rate %',     value: sos.kpis?.caregiver_reach_rate_pct?.toFixed(1) + '%', color: 'success' },
                  { label: 'False Alarm %',    value: sos.kpis?.false_alarm_rate_pct?.toFixed(1) + '%', color: 'secondary' },
                ].map(kp => (
                  <div key={kp.label} className="col-6 col-md-2 mb-2">
                    <div className="card text-center shadow-sm border-0">
                      <div className="card-body py-2 px-1">
                        <div className={`h4 mb-0 text-${kp.color}`}>{kp.value ?? '—'}</div>
                        <div className="text-muted" style={{ fontSize: '0.7rem' }}>{kp.label}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              {/* Escalation chain steps */}
              <div className="card shadow-sm border-0 mb-3">
                <div className="card-header bg-danger text-white py-2 small fw-bold">Escalation Chain: Wearable → Gateway → Cloud → Caregiver/EMS</div>
                <div className="card-body p-2">
                  <div className="d-flex flex-wrap gap-2">
                    {(sos.escalation_chain || []).map(step => (
                      <div key={step.step} className="card border-danger" style={{ minWidth: 180, flex: '1 1 180px' }}>
                        <div className="card-body py-2 px-3">
                          <div className="fw-bold small text-danger">Step {step.step}: {step.node}</div>
                          <div className="text-muted" style={{ fontSize: '0.72rem' }}>{step.action}</div>
                          {step.latency_target_s > 0 && <div className="text-info small">Target: ≤{step.latency_target_s}s</div>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              {/* Events table */}
              <div className="card shadow-sm border-0">
                <div className="card-header bg-secondary text-white py-2 small fw-bold">SOS Event Log (recent 28)</div>
                <div className="card-body p-2">
                  <div className="table-responsive">
                    <table className="table table-sm mb-0">
                      <thead><tr>
                        <th>Event</th><th>Patient</th><th>Trigger</th><th>Outcome</th>
                        <th>Channel</th><th>Detect(s)</th><th>Notify(s)</th><th>Total(s)</th><th>&lt;2min</th>
                      </tr></thead>
                      <tbody>
                        {(sos.events || []).map(ev => (
                          <tr key={ev.event_id}>
                            <td className="small">{ev.event_id}</td>
                            <td className="small">{ev.patient_name}</td>
                            <td className="small">{ev.trigger}</td>
                            <td><span className={`badge bg-${ev.outcome === 'ems_dispatched' ? 'danger' : ev.outcome === 'false_alarm' ? 'secondary' : ev.outcome === 'resolved_home' ? 'success' : 'warning'}`} style={{ fontSize: '0.65rem' }}>{ev.outcome}</span></td>
                            <td className="small">{ev.notification_channel}</td>
                            <td className="small">{ev.latency_detect_s?.toFixed(1)}</td>
                            <td className="small">{ev.latency_notify_s?.toFixed(1)}</td>
                            <td className="small">{ev.total_elapsed_s?.toFixed(0)}</td>
                            <td>{ev.under_2min ? <span className="text-success">✓</span> : <span className="text-danger">✗</span>}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
