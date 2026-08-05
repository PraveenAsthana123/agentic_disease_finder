'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusColor = s => ({ active: 'success', offline: 'danger', charging: 'warning', maintenance: 'secondary' }[s] || 'info');
const battColor = b => b >= 60 ? 'success' : b >= 30 ? 'warning' : 'danger';
const connBadge = c => ({ BLE: 'info', 'BLE+WiFi': 'primary', Cellular: 'success', WiFi: 'warning' }[c] || 'secondary');

export default function WearableDevicesDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('patient_id');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/wearable-devices/overview`).then(r => r.json()),
      fetch(`${API}/api/wearable-devices/breakdown`).then(r => r.json()),
      fetch(`${API}/api/wearable-devices/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading wearable device fleet data...</div>;

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'fleet', label: '📋 Device Fleet' },
    { id: 'features', label: '⚙️ Features' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const devices = bd?.all_devices || [];
  const filtered = devices.filter(d =>
    !search || d.patient_id.toLowerCase().includes(search.toLowerCase()) ||
    d.device_type.toLowerCase().includes(search.toLowerCase()) ||
    d.brand.toLowerCase().includes(search.toLowerCase())
  ).sort((a, b) => {
    if (sortBy === 'battery_level') return a.battery_level - b.battery_level;
    if (sortBy === 'status') return a.status.localeCompare(b.status);
    if (sortBy === 'device_type') return a.device_type.localeCompare(b.device_type);
    return a.patient_id.localeCompare(b.patient_id);
  });

  return (
    <div className="p-3">
      <h3>🩺 Wearable Devices Dashboard</h3>
      <p className="text-muted">
        Patient wearable fleet — {ov.total_devices} devices · {ov.total_patients} patients ·
        {' '}{ov.active_count} active · {ov.offline_count} offline · avg battery {ov.avg_battery_level}%
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          {/* KPI row */}
          <div className="row mb-3">
            {[
              ['Total Devices', ov.total_devices, 'primary'],
              ['Active', ov.active_count, 'success'],
              ['Offline', ov.offline_count, 'danger'],
              ['Charging', ov.charging_count, 'warning'],
              ['Maintenance', ov.maintenance_count, 'secondary'],
              ['Low Battery', ov.low_battery_count, 'danger'],
            ].map(([label, val, c]) => (
              <div key={label} className="col-6 col-md-2 mb-2">
                <div className="card shadow-sm h-100">
                  <div className="card-body text-center py-2">
                    <div className={`h5 mb-0 text-${c}`}>{val}</div>
                    <div className="text-muted small">{label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="row mb-3">
            {/* Status distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Device Status</h6>
                  {ov.status_distribution.map(({ name, value }) => {
                    const pct = ((value / ov.total_devices) * 100).toFixed(0);
                    return (
                      <div key={name} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span className="text-capitalize">{name}</span>
                          <span>{value} ({pct}%)</span>
                        </div>
                        <div className="progress" style={{ height: '10px' }}>
                          <div className={`progress-bar bg-${statusColor(name)}`} style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Device types */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>By Device Type</h6>
                  {ov.device_type_distribution.map(({ name, value }) => (
                    <div key={name} className="d-flex justify-content-between border-bottom py-1">
                      <span className="small">{name}</span>
                      <span className="badge bg-primary">{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Connectivity */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Connectivity Modes</h6>
                  {ov.connectivity_distribution.map(({ name, value }) => (
                    <div key={name} className="d-flex justify-content-between border-bottom py-1">
                      <span className="small">{name}</span>
                      <span className={`badge bg-${connBadge(name)}`}>{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Brand distribution */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6>Brand Distribution</h6>
              <div className="row">
                {ov.brand_distribution.map(({ name, value }) => {
                  const pct = ((value / ov.total_devices) * 100).toFixed(0);
                  return (
                    <div key={name} className="col-md-6 mb-2">
                      <div className="d-flex align-items-center">
                        <span className="small me-2" style={{ minWidth: '160px' }}>{name}</span>
                        <div className="flex-grow-1 me-2">
                          <div className="progress" style={{ height: '16px' }}>
                            <div className="progress-bar bg-info" style={{ width: `${pct}%` }}>
                              {value}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Avg battery */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6>Fleet Battery Level (average {ov.avg_battery_level}%)</h6>
              <div className="progress" style={{ height: '24px' }}>
                <div
                  className={`progress-bar bg-${battColor(ov.avg_battery_level)}`}
                  style={{ width: `${ov.avg_battery_level}%` }}
                >
                  {ov.avg_battery_level}%
                </div>
              </div>
              <div className="mt-2 text-muted small">
                {ov.low_battery_count} device(s) below 30% — recommend charging
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── DEVICE FLEET ── */}
      {tab === 'fleet' && (
        <div>
          <div className="d-flex gap-2 mb-3">
            <input
              className="form-control"
              style={{ maxWidth: '280px' }}
              placeholder="Search patient / type / brand…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            <select className="form-select" style={{ maxWidth: '180px' }} value={sortBy} onChange={e => setSortBy(e.target.value)}>
              <option value="patient_id">Sort: Patient ID</option>
              <option value="battery_level">Sort: Battery ↑</option>
              <option value="status">Sort: Status</option>
              <option value="device_type">Sort: Type</option>
            </select>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-hover table-bordered">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Device ID</th>
                  <th>Type</th>
                  <th>Brand</th>
                  <th>Status</th>
                  <th>Battery</th>
                  <th>Connectivity</th>
                  <th>Last Sync</th>
                  <th>Seizure Det.</th>
                  <th>Fall Det.</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(d => (
                  <tr key={d.device_id}>
                    <td><span className="badge bg-secondary">{d.patient_id}</span></td>
                    <td className="small font-monospace">{d.device_id}</td>
                    <td className="small">{d.device_type}</td>
                    <td className="small">{d.brand}</td>
                    <td>
                      <span className={`badge bg-${statusColor(d.status)} text-capitalize`}>
                        {d.status}
                      </span>
                    </td>
                    <td>
                      <div className="d-flex align-items-center gap-1">
                        <div className="progress flex-grow-1" style={{ height: '10px', minWidth: '50px' }}>
                          <div
                            className={`progress-bar bg-${battColor(d.battery_level)}`}
                            style={{ width: `${d.battery_level}%` }}
                          />
                        </div>
                        <span className="small">{d.battery_level}%</span>
                      </div>
                    </td>
                    <td>
                      <span className={`badge bg-${connBadge(d.connectivity)} small`}>{d.connectivity}</span>
                    </td>
                    <td className="small text-muted">{d.last_sync ? d.last_sync.slice(0, 10) : '—'}</td>
                    <td className="text-center">
                      {d.seizure_detection ? <span className="text-success">✓</span> : <span className="text-muted">—</span>}
                    </td>
                    <td className="text-center">
                      {d.fall_detection ? <span className="text-success">✓</span> : <span className="text-muted">—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-muted small">{filtered.length} of {devices.length} devices shown</div>
          </div>
        </div>
      )}

      {/* ── FEATURES ── */}
      {tab === 'features' && (
        <div>
          <h6>Feature Coverage Across Fleet</h6>
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              {ov.feature_coverage.map(({ feature, enabled, total, pct }) => (
                <div key={feature} className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="fw-semibold">{feature}</span>
                    <span className="text-muted small">{enabled}/{total} devices ({pct}%)</span>
                  </div>
                  <div className="progress" style={{ height: '18px' }}>
                    <div
                      className={`progress-bar ${pct >= 80 ? 'bg-success' : pct >= 50 ? 'bg-warning' : 'bg-danger'}`}
                      style={{ width: `${pct}%` }}
                    >
                      {pct}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="row">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-body">
                  <h6>Devices by Status</h6>
                  {ov.status_distribution.map(({ name, value }) => (
                    <div key={name} className="d-flex align-items-center justify-content-between py-2 border-bottom">
                      <div className="d-flex align-items-center gap-2">
                        <span className={`badge bg-${statusColor(name)}`}>{name}</span>
                      </div>
                      <span className="fw-bold">{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-body">
                  <h6>Fleet Health Summary</h6>
                  <table className="table table-sm mb-0">
                    <tbody>
                      <tr><td>Total Patients Monitored</td><td className="fw-bold">{ov.total_patients}</td></tr>
                      <tr><td>Active Monitoring Rate</td><td className="fw-bold text-success">{((ov.active_count / ov.total_devices) * 100).toFixed(1)}%</td></tr>
                      <tr><td>Average Battery</td><td className={`fw-bold text-${battColor(ov.avg_battery_level)}`}>{ov.avg_battery_level}%</td></tr>
                      <tr><td>Low Battery Alerts</td><td className="fw-bold text-danger">{ov.low_battery_count}</td></tr>
                      <tr><td>Seizure Detection Coverage</td><td className="fw-bold">{ov.seizure_detection_rate}%</td></tr>
                      <tr><td>Fall Detection Coverage</td><td className="fw-bold">{ov.fall_detection_rate}%</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="row mb-3">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-body">
                  <h6>Device Types</h6>
                  {defs.device_types?.map(({ type, description }) => (
                    <div key={type} className="mb-2 pb-2 border-bottom">
                      <div className="fw-semibold small">{type}</div>
                      <div className="text-muted small">{description}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-body">
                  <h6>Status Descriptions</h6>
                  {defs.status_descriptions?.map(({ status, description }) => (
                    <div key={status} className="mb-2 pb-2 border-bottom">
                      <div className="d-flex align-items-center gap-2 mb-1">
                        <span className={`badge bg-${statusColor(status)}`}>{status}</span>
                      </div>
                      <div className="text-muted small">{description}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-body">
              <h6>Connectivity Modes</h6>
              <div className="row">
                {defs.connectivity_descriptions?.map(({ type, description }) => (
                  <div key={type} className="col-md-6 mb-2 pb-2 border-bottom">
                    <div className="d-flex align-items-center gap-2 mb-1">
                      <span className={`badge bg-${connBadge(type)}`}>{type}</span>
                    </div>
                    <div className="text-muted small">{description}</div>
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
