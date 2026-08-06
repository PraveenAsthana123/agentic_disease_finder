'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KPICard({ label, value, color = 'primary', sub }) {
  return (
    <div className="col-6 col-md-2 mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className={`h5 mb-0 text-${color}`}>{value ?? '—'}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function DistBar({ items, nameKey = 'name', countKey = 'count', total, colorFn }) {
  const allCounts = items.map(i => i[countKey] ?? 0);
  const max = total || Math.max(...allCounts, 1);
  return (items || []).map((item, idx) => {
    const name = item[nameKey] ?? '?';
    const count = item[countKey] ?? 0;
    const pct = ((count / max) * 100).toFixed(0);
    const bg = colorFn ? colorFn(name) : 'primary';
    return (
      <div key={idx} className="d-flex align-items-center mb-2">
        <span className="me-2 small" style={{ minWidth: '140px' }}>{name}</span>
        <div className="flex-grow-1 me-2">
          <div className="progress" style={{ height: '20px' }}>
            <div className={`progress-bar bg-${bg}`} style={{ width: `${pct}%` }}>
              {count} ({pct}%)
            </div>
          </div>
        </div>
      </div>
    );
  });
}

function StatusBadge({ status }) {
  const map = { active: 'success', offline: 'danger', charging: 'warning', maintenance: 'secondary' };
  return <span className={`badge bg-${map[status] || 'light text-dark'}`}>{status}</span>;
}

function RiskBadge({ tier }) {
  const map = { Critical: 'danger', High: 'warning', Moderate: 'info', Low: 'success' };
  return <span className={`badge bg-${map[tier] || 'secondary'}`}>{tier}</span>;
}

function BatteryBar({ level }) {
  const color = level >= 50 ? 'success' : level >= 20 ? 'warning' : 'danger';
  return (
    <div className="d-flex align-items-center gap-1">
      <div className="progress flex-grow-1" style={{ height: '14px' }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${level}%` }}>{level}%</div>
      </div>
    </div>
  );
}

export default function SeizureBandDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [ptFilter, setPtFilter] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/seizure-band/overview`).then(r => r.json()),
      fetch(`${API}/api/seizure-band/breakdown`).then(r => r.json()),
      fetch(`${API}/api/seizure-band/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => {
      setOv(o); setBd(b); setDefs(d);
    }).catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">{err}</div>;
  if (!ov) return <div className="text-center p-5"><div className="spinner-border" /></div>;

  const kpis = ov.kpis || {};
  const filteredPt = (ov.per_patient_summary || []).filter(p =>
    !ptFilter || p.patient_id.toLowerCase().includes(ptFilter.toLowerCase())
  );
  const filteredDevices = (bd?.device_roster || []).filter(d =>
    !ptFilter || d.patient_id?.toLowerCase().includes(ptFilter.toLowerCase()) ||
    d.device_id?.toLowerCase().includes(ptFilter.toLowerCase())
  );

  const riskColor = (tier) => {
    const map = { 'Critical (75+)': 'danger', 'High (50-75)': 'warning', 'Moderate (25-50)': 'info', 'Low (0-25)': 'success' };
    return map[tier] || 'secondary';
  };

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1">Seizure Band Dashboard</h4>
      <p className="text-muted small mb-3">
        Wearable seizure-detection fleet — Empatica Embrace2, Wrist EEG Bands, Ankle Sensors.
        Real data: wearable_devices {kpis.total_devices} devices · wearable_readings {kpis.total_readings?.toLocaleString()} readings
      </p>

      {/* KPI Row */}
      <div className="row mb-3">
        <KPICard label="Total Devices" value={kpis.total_devices} color="primary" />
        <KPICard label="Patients Monitored" value={kpis.total_patients} color="info" />
        <KPICard label="Seizure Events" value={kpis.seizure_events_detected} color="danger" sub="detected readings" />
        <KPICard label="Fall Events" value={kpis.fall_events_detected} color="warning" />
        <KPICard label="Avg Risk Score" value={kpis.avg_risk_score} color={kpis.avg_risk_score >= 50 ? 'danger' : 'warning'} sub="/100" />
        <KPICard label="Avg Health Score" value={kpis.avg_health_score} color="success" sub="/100" />
      </div>
      <div className="row mb-3">
        <KPICard label="Active Devices" value={kpis.active_devices} color="success" />
        <KPICard label="Offline Devices" value={kpis.offline_devices} color="danger" />
        <KPICard label="Avg Battery" value={`${kpis.avg_battery_pct}%`} color={kpis.avg_battery_pct >= 50 ? 'success' : 'warning'} />
        <KPICard label="Low Battery" value={kpis.low_battery_devices} color="danger" sub="<20%" />
        <KPICard label="Pts w/ Seizure" value={kpis.patients_with_seizure_event} color="danger" sub="patients" />
        <KPICard label="Avg Confidence" value={kpis.avg_seizure_confidence} color="secondary" sub="[0–1]" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {['overview', 'devices', 'per-patient', 'events', 'definitions'].map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>
              {t === 'overview' ? 'Overview' :
               t === 'devices' ? 'Device Roster' :
               t === 'per-patient' ? 'Per Patient' :
               t === 'events' ? 'Seizure Events' : 'Definitions'}
            </button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header py-2 small fw-bold">Device Status</div>
              <div className="card-body">
                <DistBar
                  items={ov.status_distribution}
                  colorFn={n => ({ active: 'success', offline: 'danger', charging: 'warning', maintenance: 'secondary' }[n] || 'primary')}
                />
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header py-2 small fw-bold">Device Types</div>
              <div className="card-body">
                <DistBar items={ov.type_distribution} total={kpis.total_devices} />
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header py-2 small fw-bold">Brands</div>
              <div className="card-body">
                <DistBar items={ov.brand_distribution} total={kpis.total_devices} />
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header py-2 small fw-bold">Seizure Risk Tier Distribution</div>
              <div className="card-body">
                <DistBar
                  items={ov.risk_distribution}
                  nameKey="tier"
                  countKey="count"
                  total={(ov.risk_distribution || []).reduce((s, r) => s + r.count, 0)}
                  colorFn={riskColor}
                />
              </div>
            </div>
          </div>
          {bd && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 small fw-bold">Connectivity Types</div>
                <div className="card-body">
                  <DistBar items={bd.connectivity_distribution} />
                </div>
              </div>
            </div>
          )}
          {bd && (
            <div className="col-12 mb-3">
              <div className="card shadow-sm">
                <div className="card-header py-2 small fw-bold">Monthly Seizure Event Trend</div>
                <div className="card-body">
                  <div className="d-flex align-items-end gap-2 flex-wrap">
                    {(bd.monthly_seizure_trend || []).map(m => {
                      const maxVal = Math.max(...bd.monthly_seizure_trend.map(x => x.seizure_events), 1);
                      const h = Math.round((m.seizure_events / maxVal) * 80);
                      return (
                        <div key={m.month} className="text-center" style={{ minWidth: '50px' }}>
                          <div className="text-danger fw-bold small">{m.seizure_events}</div>
                          <div className="bg-danger mx-auto" style={{ width: '30px', height: `${h}px` }} />
                          <div className="text-muted" style={{ fontSize: '0.6rem' }}>{m.month.slice(5)}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* DEVICE ROSTER TAB */}
      {tab === 'devices' && bd && (
        <div>
          <div className="mb-2">
            <input
              className="form-control form-control-sm w-auto"
              placeholder="Filter by patient / device ID…"
              value={ptFilter}
              onChange={e => setPtFilter(e.target.value)}
            />
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-bordered table-hover small">
              <thead className="table-dark">
                <tr>
                  <th>Device ID</th>
                  <th>Patient</th>
                  <th>Type</th>
                  <th>Brand</th>
                  <th>Status</th>
                  <th>Battery</th>
                  <th>Connectivity</th>
                  <th>Firmware</th>
                  <th>Seizure Det.</th>
                  <th>Fall Det.</th>
                  <th>Readings</th>
                  <th>Seizure Events</th>
                  <th>Last Sync</th>
                </tr>
              </thead>
              <tbody>
                {filteredDevices.map(d => (
                  <tr key={d.device_id}>
                    <td><code>{d.device_id}</code></td>
                    <td>{d.patient_id}</td>
                    <td>{d.device_type}</td>
                    <td>{d.brand}</td>
                    <td><StatusBadge status={d.status} /></td>
                    <td style={{ minWidth: '100px' }}><BatteryBar level={d.battery_level ?? 0} /></td>
                    <td><span className="badge bg-secondary">{d.connectivity}</span></td>
                    <td><code>{d.firmware_version}</code></td>
                    <td>{d.seizure_detection_enabled ? <span className="badge bg-success">ON</span> : <span className="badge bg-secondary">OFF</span>}</td>
                    <td>{d.fall_detection_enabled ? <span className="badge bg-success">ON</span> : <span className="badge bg-secondary">OFF</span>}</td>
                    <td>{d.total_readings}</td>
                    <td>{d.seizure_events > 0 ? <span className="text-danger fw-bold">{d.seizure_events}</span> : <span className="text-muted">0</span>}</td>
                    <td><small className="text-muted">{d.last_sync?.slice(0, 10)}</small></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* PER PATIENT TAB */}
      {tab === 'per-patient' && (
        <div>
          <div className="mb-2">
            <input
              className="form-control form-control-sm w-auto"
              placeholder="Filter by patient ID…"
              value={ptFilter}
              onChange={e => setPtFilter(e.target.value)}
            />
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-bordered table-hover small">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Readings</th>
                  <th>Seizure Events</th>
                  <th>Fall Events</th>
                  <th>Avg Risk Score</th>
                  <th>Avg Health Score</th>
                  <th>Risk Tier</th>
                </tr>
              </thead>
              <tbody>
                {filteredPt.map(p => (
                  <tr key={p.patient_id}>
                    <td><strong>{p.patient_id}</strong></td>
                    <td>{p.readings}</td>
                    <td>{p.seizure_events > 0 ? <span className="text-danger fw-bold">{p.seizure_events}</span> : <span className="text-muted">0</span>}</td>
                    <td>{p.fall_events > 0 ? <span className="text-warning fw-bold">{p.fall_events}</span> : <span className="text-muted">0</span>}</td>
                    <td>
                      <div className="progress" style={{ height: '16px' }}>
                        <div
                          className={`progress-bar bg-${p.avg_risk_score >= 75 ? 'danger' : p.avg_risk_score >= 50 ? 'warning' : p.avg_risk_score >= 25 ? 'info' : 'success'}`}
                          style={{ width: `${p.avg_risk_score}%` }}
                        >{p.avg_risk_score}</div>
                      </div>
                    </td>
                    <td>
                      <div className="progress" style={{ height: '16px' }}>
                        <div className="progress-bar bg-success" style={{ width: `${p.avg_health_score}%` }}>{p.avg_health_score}</div>
                      </div>
                    </td>
                    <td><RiskBadge tier={p.risk_tier} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* SEIZURE EVENTS TAB */}
      {tab === 'events' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header py-2 small fw-bold">Detection Confidence Distribution</div>
              <div className="card-body">
                {bd && <DistBar
                  items={bd.confidence_distribution}
                  nameKey="bucket"
                  countKey="count"
                  total={(bd.confidence_distribution || []).reduce((s, i) => s + i.count, 0)}
                  colorFn={n => ({ '<0.1': 'success', '0.1-0.3': 'info', '0.3-0.5': 'warning', '0.5-0.7': 'danger', '0.7+': 'danger' }[n] || 'secondary')}
                />}
              </div>
            </div>
          </div>
          {bd && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 small fw-bold">Physiological Summary</div>
                <div className="card-body">
                  <table className="table table-sm small mb-0">
                    <tbody>
                      <tr><td>Avg Stress Score (EDA proxy)</td><td className="fw-bold">{bd.physiological_summary?.avg_stress_score}</td></tr>
                      <tr><td>Avg SpO2</td><td className="fw-bold">{bd.physiological_summary?.avg_spo2}%</td></tr>
                      <tr><td>Low SpO2 Readings (&lt;95%)</td><td className="fw-bold text-danger">{bd.physiological_summary?.low_spo2_readings}</td></tr>
                      <tr><td>Avg Heart Rate</td><td className="fw-bold">{bd.physiological_summary?.avg_heart_rate} bpm</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header py-2 small fw-bold">Recent Seizure Events (last 10)</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered small mb-0">
                    <thead className="table-dark">
                      <tr>
                        <th>Patient</th>
                        <th>Device</th>
                        <th>Date</th>
                        <th>Confidence</th>
                        <th>Risk Score</th>
                        <th>HR (bpm)</th>
                        <th>Health Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(ov.recent_seizure_events || []).map((e, i) => (
                        <tr key={i} className="table-danger">
                          <td><strong>{e.patient_id}</strong></td>
                          <td><code>{e.device_id}</code></td>
                          <td>{e.date}</td>
                          <td><span className="badge bg-danger">{(e.confidence * 100).toFixed(0)}%</span></td>
                          <td>
                            <div className="progress" style={{ height: '14px' }}>
                              <div className="progress-bar bg-danger" style={{ width: `${e.risk_score}%` }}>{e.risk_score?.toFixed(1)}</div>
                            </div>
                          </td>
                          <td>{e.hr}</td>
                          <td>
                            <div className="progress" style={{ height: '14px' }}>
                              <div className="progress-bar bg-success" style={{ width: `${e.health_score}%` }}>{e.health_score?.toFixed(1)}</div>
                            </div>
                          </td>
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

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <div className="row">
          {Object.values(defs.sections || {}).map((sec, idx) => (
            <div key={idx} className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 small fw-bold">{sec.label}</div>
                <div className="card-body">
                  {(sec.items || []).map((item, i) => (
                    <div key={i} className="mb-3">
                      <strong className="text-primary">{item.term}</strong>
                      <p className="small text-muted mb-0 mt-1">{item.definition}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
