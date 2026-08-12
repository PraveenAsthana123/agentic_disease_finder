'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TRAJ_COLOR = t => ({ improving: 'success', stable: 'primary', declining: 'danger' }[t] || 'secondary');
const TRAJ_ICON = t => ({ improving: '↑', stable: '→', declining: '↓' }[t] || '?');

const STATUS_COLOR = s => ({
  active: 'success', offline: 'danger', maintenance: 'warning', charging: 'info'
}[s] || 'secondary');

const RISK_TIER = v => {
  if (v < 25) return { label: 'Low', color: 'success' };
  if (v < 50) return { label: 'Moderate', color: 'warning' };
  if (v < 75) return { label: 'High', color: 'danger' };
  return { label: 'Critical', color: 'dark' };
};

const HEALTH_TIER = v => {
  if (v >= 80) return { label: 'Optimal', color: 'success' };
  if (v >= 60) return { label: 'Good', color: 'primary' };
  if (v >= 40) return { label: 'Fair', color: 'warning' };
  return { label: 'Poor', color: 'danger' };
};

function KPI({ label, value, sub, color = 'primary', warn }) {
  return (
    <div className={`card border-${warn ? 'warning' : color} h-100`}>
      <div className="card-body text-center p-3">
        <div className={`display-6 fw-bold text-${warn ? 'warning' : color}`}>{value ?? '—'}</div>
        <div className="small fw-semibold">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
      </div>
    </div>
  );
}

function BarChart({ data, labelKey, countKey, colorFn }) {
  if (!data || !data.length) return <div className="text-muted small">No data</div>;
  const max = Math.max(...data.map(d => d[countKey] || 0), 1);
  return (
    <div>
      {data.map((d, i) => {
        const pct = ((d[countKey] || 0) / max) * 100;
        const color = colorFn ? colorFn(d[labelKey]) : 'primary';
        return (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span className="fw-semibold">{d[labelKey]}</span>
              <span className="text-muted">{d[countKey]}</span>
            </div>
            <div className="progress" style={{ height: 12 }}>
              <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function TrajectoryBadge({ traj }) {
  return (
    <span className={`badge bg-${TRAJ_COLOR(traj)}`}>
      {TRAJ_ICON(traj)} {traj || 'unknown'}
    </span>
  );
}

function DigitalTwinCard({ twin }) {
  if (!twin) return null;
  const { physiological_baseline: pb, sleep_profile: sp, activity_profile: ap, risk_profile: rp,
    health_trajectory, health_score_avg, risk_score_avg,
    longitudinal_1yr_projection, longitudinal_5yr_projection } = twin;
  const riskTier = RISK_TIER(risk_score_avg || 0);
  const healthTier = HEALTH_TIER(health_score_avg || 0);
  return (
    <div className="card border-secondary mt-3">
      <div className="card-header py-2 d-flex justify-content-between align-items-center">
        <span className="fw-semibold small">Digital Twin Profile</span>
        <TrajectoryBadge traj={health_trajectory} />
      </div>
      <div className="card-body p-3">
        <div className="row g-2 mb-3">
          <div className="col-6">
            <div className={`card bg-${healthTier.color} bg-opacity-10 border-0 text-center p-2`}>
              <div className={`fw-bold text-${healthTier.color}`}>{health_score_avg?.toFixed(1)}</div>
              <div className="text-muted" style={{ fontSize: 10 }}>Health Score</div>
              <span className={`badge bg-${healthTier.color} mt-1`} style={{ fontSize: 9 }}>{healthTier.label}</span>
            </div>
          </div>
          <div className="col-6">
            <div className={`card bg-${riskTier.color} bg-opacity-10 border-0 text-center p-2`}>
              <div className={`fw-bold text-${riskTier.color}`}>{risk_score_avg?.toFixed(1)}</div>
              <div className="text-muted" style={{ fontSize: 10 }}>Seizure Risk</div>
              <span className={`badge bg-${riskTier.color} mt-1`} style={{ fontSize: 9 }}>{riskTier.label}</span>
            </div>
          </div>
        </div>
        {pb && (
          <div className="mb-2">
            <div className="small fw-semibold text-muted mb-1">Physiological Baseline</div>
            <div className="row g-1">
              {[
                ['HR', pb.avg_heart_rate?.toFixed(0), 'bpm'],
                ['HRV', pb.avg_hrv?.toFixed(0), 'ms'],
                ['SpO₂', pb.avg_spo2?.toFixed(1), '%'],
                ['Temp', pb.avg_skin_temp?.toFixed(1), '°C'],
              ].map(([k, v, u]) => (
                <div key={k} className="col-3 text-center">
                  <div className="fw-bold" style={{ fontSize: 12 }}>{v}{u}</div>
                  <div className="text-muted" style={{ fontSize: 9 }}>{k}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        {sp && (
          <div className="mb-2">
            <div className="small fw-semibold text-muted mb-1">Sleep Profile</div>
            <div className="row g-1">
              {[
                ['Total', sp.avg_duration_hours?.toFixed(1) + 'h', ''],
                ['Deep', sp.avg_deep_sleep_pct?.toFixed(0) + '%', ''],
                ['REM', sp.avg_rem_sleep_pct?.toFixed(0) + '%', ''],
                ['Wakes', sp.avg_awakenings?.toFixed(1), '/night'],
              ].map(([k, v, u]) => (
                <div key={k} className="col-3 text-center">
                  <div className="fw-bold" style={{ fontSize: 12 }}>{v}{u}</div>
                  <div className="text-muted" style={{ fontSize: 9 }}>{k}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        {longitudinal_1yr_projection && (
          <div className="alert alert-secondary py-1 px-2 mt-2 mb-1" style={{ fontSize: 11 }}>
            <strong>1yr:</strong> {longitudinal_1yr_projection}
          </div>
        )}
        {longitudinal_5yr_projection && (
          <div className="alert alert-secondary py-1 px-2 mb-0" style={{ fontSize: 11 }}>
            <strong>5yr:</strong> {longitudinal_5yr_projection}
          </div>
        )}
      </div>
    </div>
  );
}

export default function WearablesDigitalTwinDashboard() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);
  const [expandedPt, setExpandedPt] = useState(null);
  const [searchPt, setSearchPt] = useState('');
  const [sortKey, setSortKey] = useState('avg_health_score');
  const [sortDir, setSortDir] = useState('desc');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/wearables-digital-twin/overview`).then(r => r.json()),
      fetch(`${API}/api/wearables-digital-twin/breakdown`).then(r => r.json()),
      fetch(`${API}/api/wearables-digital-twin/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center text-muted">Loading Wearables & Digital Twin Dashboard…</div>;
  if (err) return <div className="p-4 alert alert-danger">Error: {err}</div>;

  const patients = bk?.patients || [];
  const filteredPts = patients
    .filter(p => !searchPt || p.patient_id?.toLowerCase().includes(searchPt.toLowerCase()) ||
      p.device_type?.toLowerCase().includes(searchPt.toLowerCase()) ||
      p.brand?.toLowerCase().includes(searchPt.toLowerCase()))
    .sort((a, b) => {
      const av = a[sortKey] ?? 0, bv = b[sortKey] ?? 0;
      return sortDir === 'asc' ? av - bv : bv - av;
    });

  const trajCounts = (ov?.digital_twin_trajectories || []).reduce((acc, t) => {
    acc[t.health_trajectory] = (acc[t.health_trajectory] || 0) + 1;
    return acc;
  }, {});

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'devices', label: 'Device Fleet' },
    { id: 'patients', label: 'Per Patient' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div>
          <h4 className="mb-0 fw-bold">Wearables &amp; Digital Twin Dashboard</h4>
          <div className="text-muted small">
            {ov?.total_devices} devices · {ov?.total_patients} patients · {ov?.total_readings?.toLocaleString()} readings · 30-day monitoring window
          </div>
        </div>
        <div className="ms-auto">
          <span className="badge bg-success fs-6">
            {ov?.active_devices}/{ov?.total_devices} Active
          </span>
        </div>
      </div>

      {/* KPI Row */}
      <div className="row g-3 mb-4">
        <div className="col-6 col-md-3 col-xl-2">
          <KPI label="Avg Health Score" value={ov?.avg_health_score?.toFixed(1)} sub="0–100 composite" color="success" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KPI label="Avg Seizure Risk" value={ov?.avg_seizure_risk_score?.toFixed(1)} sub="0–100 risk index" color="danger" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KPI label="Seizure Detection Rate" value={ov?.seizure_detection_rate?.toFixed(1) + '%'} sub="days with seizure detected" color="warning" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KPI label="Avg Heart Rate" value={ov?.avg_heart_rate?.toFixed(0)} sub="bpm population avg" color="primary" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KPI label="Avg HRV" value={ov?.avg_hrv?.toFixed(1)} sub="ms SDNN — autonomic index" color="info" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KPI label="Avg Daily Steps" value={ov?.avg_steps?.toFixed(0)} sub="target: 10,000/day" color="secondary" warn={ov?.avg_steps < 10000} />
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div className="row g-4">
          {/* Device Type Distribution */}
          <div className="col-md-6 col-xl-4">
            <div className="card h-100">
              <div className="card-header py-2 fw-semibold">Device Types</div>
              <div className="card-body">
                <BarChart
                  data={ov?.device_type_distribution || []}
                  labelKey="device_type"
                  countKey="count"
                  colorFn={t => ({ 'Wrist EEG Band': 'primary', 'Ring Sensor': 'info', 'Ankle Sensor': 'warning', 'Patch Monitor': 'success', 'Smartwatch': 'danger', 'Chest Strap': 'secondary' }[t] || 'secondary')}
                />
              </div>
            </div>
          </div>

          {/* Device Status */}
          <div className="col-md-6 col-xl-4">
            <div className="card h-100">
              <div className="card-header py-2 fw-semibold">Device Status</div>
              <div className="card-body">
                {(ov?.device_status_distribution || []).map((d, i) => (
                  <div key={i} className="d-flex align-items-center justify-content-between mb-3">
                    <span className={`badge bg-${STATUS_COLOR(d.status)} me-2`} style={{ minWidth: 90 }}>
                      {d.status}
                    </span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{ height: 14 }}>
                        <div
                          className={`progress-bar bg-${STATUS_COLOR(d.status)}`}
                          style={{ width: `${(d.count / (ov?.total_devices || 1)) * 100}%` }}
                        />
                      </div>
                    </div>
                    <span className="fw-bold small">{d.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Digital Twin Trajectories */}
          <div className="col-md-6 col-xl-4">
            <div className="card h-100">
              <div className="card-header py-2 fw-semibold">Health Trajectories</div>
              <div className="card-body">
                {['improving', 'stable', 'declining'].map(traj => (
                  <div key={traj} className="d-flex align-items-center justify-content-between mb-3">
                    <TrajectoryBadge traj={traj} />
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{ height: 14 }}>
                        <div
                          className={`progress-bar bg-${TRAJ_COLOR(traj)}`}
                          style={{ width: `${((trajCounts[traj] || 0) / (ov?.total_patients || 1)) * 100}%` }}
                        />
                      </div>
                    </div>
                    <span className="fw-bold small">{trajCounts[traj] || 0} patients</span>
                  </div>
                ))}
                <hr className="my-2" />
                <div className="small text-muted">
                  Trajectory: comparing first-half vs second-half 30-day health score. &gt;3pt improvement = ↑ improving.
                </div>
              </div>
            </div>
          </div>

          {/* HRV Distribution */}
          <div className="col-md-6 col-xl-4">
            <div className="card h-100">
              <div className="card-header py-2 fw-semibold">HRV Distribution (SDNN, ms)</div>
              <div className="card-body">
                <BarChart
                  data={ov?.hrv_distribution || []}
                  labelKey="bucket"
                  countKey="count"
                  colorFn={b => {
                    if (b === '<20') return 'danger';
                    if (b === '20–39') return 'warning';
                    if (b === '40–59') return 'info';
                    return 'success';
                  }}
                />
                <div className="small text-muted mt-2">
                  SDNN target: ≥50ms (normal autonomic function). &lt;20ms = severe autonomic impairment.
                </div>
              </div>
            </div>
          </div>

          {/* Sleep Quality Distribution */}
          <div className="col-md-6 col-xl-4">
            <div className="card h-100">
              <div className="card-header py-2 fw-semibold">Sleep Quality Distribution</div>
              <div className="card-body">
                <BarChart
                  data={ov?.sleep_quality_distribution || []}
                  labelKey="bucket"
                  countKey="count"
                  colorFn={b => {
                    if (b?.includes('Poor')) return 'danger';
                    if (b?.includes('Fair')) return 'warning';
                    if (b?.includes('Good')) return 'success';
                    return 'primary';
                  }}
                />
                <div className="small text-muted mt-2">
                  Avg quality: {ov?.avg_sleep_quality?.toFixed(1)}/100 — avg duration: {ov?.avg_sleep_duration?.toFixed(1)}h/night
                </div>
              </div>
            </div>
          </div>

          {/* Brand Distribution */}
          <div className="col-md-6 col-xl-4">
            <div className="card h-100">
              <div className="card-header py-2 fw-semibold">Device Brands</div>
              <div className="card-body">
                <BarChart
                  data={(ov?.brand_distribution || []).slice(0, 8)}
                  labelKey="brand"
                  countKey="count"
                />
              </div>
            </div>
          </div>

          {/* Key Clinical References */}
          <div className="col-12">
            <div className="card border-info">
              <div className="card-header py-2 fw-semibold bg-info bg-opacity-10">Key Clinical References</div>
              <div className="card-body">
                <div className="row g-2">
                  {[
                    ['Empatica Embrace2', 'FDA Breakthrough Device — GTCS detection >98% sensitivity, <1 false alarm/day (CE marked)'],
                    ['ILAE Wearables WG', 'Multimodal wearables detect pre-ictal physiological changes up to 30 min before clinical seizure onset'],
                    ['HRV & Epilepsy', 'HRV suppression occurs in 83% of focal seizures; chronically low SDNN (<50ms) associated with SUDEP risk'],
                    ['Digital Twin Framework', 'ILAE Digital Health Task Force (2023) — precision epilepsy care priority; c-statistic 0.78–0.84 for 12-month seizure prediction'],
                    ['EDA / PPG', 'EDA surges in 70–80% of seizures, up to 40s before clinical onset; PPG ictal tachycardia >120% baseline in 80–90% of GTCS'],
                  ].map(([ref, desc]) => (
                    <div key={ref} className="col-md-6">
                      <div className="alert alert-info py-2 px-3 mb-0">
                        <strong>{ref}</strong> — <span className="small">{desc}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── DEVICE FLEET TAB ── */}
      {tab === 'devices' && (
        <div>
          <div className="row g-3 mb-3">
            {(ov?.device_status_distribution || []).map((d, i) => (
              <div key={i} className="col-6 col-md-3">
                <div className={`card border-${STATUS_COLOR(d.status)} text-center`}>
                  <div className="card-body p-3">
                    <div className={`display-6 fw-bold text-${STATUS_COLOR(d.status)}`}>{d.count}</div>
                    <div className="small fw-semibold text-capitalize">{d.status}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="card">
            <div className="card-header py-2 fw-semibold">Device Fleet — All Patients</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Patient</th>
                      <th>Device ID</th>
                      <th>Type</th>
                      <th>Brand</th>
                      <th>Status</th>
                      <th>Battery</th>
                      <th>Seizure Detect</th>
                      <th>Sleep Track</th>
                      <th>Last Sync</th>
                    </tr>
                  </thead>
                  <tbody>
                    {patients.map((p, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{p.patient_id}</td>
                        <td><code>{p.device_id}</code></td>
                        <td>{p.device_type}</td>
                        <td>{p.brand}</td>
                        <td><span className={`badge bg-${STATUS_COLOR(p.device_status)}`}>{p.device_status}</span></td>
                        <td>
                          <div className="d-flex align-items-center gap-1">
                            <div className="progress flex-grow-1" style={{ height: 8, minWidth: 50 }}>
                              <div
                                className={`progress-bar ${p.battery_level < 20 ? 'bg-danger' : p.battery_level < 50 ? 'bg-warning' : 'bg-success'}`}
                                style={{ width: `${p.battery_level}%` }}
                              />
                            </div>
                            <small>{p.battery_level}%</small>
                          </div>
                        </td>
                        <td>
                          {p.seizure_detection_enabled
                            ? <span className="badge bg-success">Yes</span>
                            : <span className="badge bg-secondary">No</span>}
                        </td>
                        <td>
                          {p.sleep_tracking
                            ? <span className="badge bg-primary">Yes</span>
                            : <span className="badge bg-secondary">No</span>}
                        </td>
                        <td className="text-muted small">{p.last_sync?.slice(0, 10)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── PER PATIENT TAB ── */}
      {tab === 'patients' && (
        <div>
          <div className="row g-2 mb-3 align-items-center">
            <div className="col-md-4">
              <input
                className="form-control form-control-sm"
                placeholder="Search patient / device type / brand…"
                value={searchPt}
                onChange={e => setSearchPt(e.target.value)}
              />
            </div>
            <div className="col-auto">
              <select
                className="form-select form-select-sm"
                value={sortKey}
                onChange={e => setSortKey(e.target.value)}
              >
                <option value="avg_health_score">Sort: Health Score ↓</option>
                <option value="avg_seizure_risk_score">Sort: Seizure Risk ↑</option>
                <option value="avg_hrv">Sort: HRV ↓</option>
                <option value="seizure_days_detected">Sort: Seizure Days ↓</option>
                <option value="avg_sleep_quality">Sort: Sleep Quality ↓</option>
              </select>
            </div>
            <div className="col-auto">
              <button
                className="btn btn-outline-secondary btn-sm"
                onClick={() => setSortDir(d => d === 'asc' ? 'desc' : 'asc')}
              >
                {sortDir === 'desc' ? '↓ Desc' : '↑ Asc'}
              </button>
            </div>
            <div className="col-auto text-muted small">{filteredPts.length} patients</div>
          </div>

          <div className="table-responsive mb-3">
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Device</th>
                  <th>Status</th>
                  <th>Health Score</th>
                  <th>Seizure Risk</th>
                  <th>HR (bpm)</th>
                  <th>HRV (ms)</th>
                  <th>Steps/day</th>
                  <th>Sleep (h)</th>
                  <th>Seizure Days</th>
                  <th>Trajectory</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {filteredPts.map((p, i) => {
                  const riskTier = RISK_TIER(p.avg_seizure_risk_score || 0);
                  const healthTier = HEALTH_TIER(p.avg_health_score || 0);
                  const isExp = expandedPt === p.patient_id;
                  return (
                    <>
                      <tr key={i}>
                        <td className="fw-semibold">{p.patient_id}</td>
                        <td className="small text-muted">{p.device_type}</td>
                        <td><span className={`badge bg-${STATUS_COLOR(p.device_status)}`}>{p.device_status}</span></td>
                        <td>
                          <span className={`badge bg-${healthTier.color}`}>
                            {p.avg_health_score?.toFixed(1)} ({healthTier.label})
                          </span>
                        </td>
                        <td>
                          <span className={`badge bg-${riskTier.color}`}>
                            {p.avg_seizure_risk_score?.toFixed(1)} ({riskTier.label})
                          </span>
                        </td>
                        <td>{p.avg_heart_rate?.toFixed(0)}</td>
                        <td>{p.avg_hrv?.toFixed(1)}</td>
                        <td>{p.avg_steps?.toFixed(0)}</td>
                        <td>{p.avg_sleep_duration?.toFixed(1)}</td>
                        <td>{p.seizure_days_detected}/{p.total_reading_days}</td>
                        <td><TrajectoryBadge traj={p.digital_twin?.health_trajectory} /></td>
                        <td>
                          <button
                            className="btn btn-outline-primary btn-sm py-0 px-1"
                            onClick={() => setExpandedPt(isExp ? null : p.patient_id)}
                          >
                            {isExp ? '▲' : '▼'}
                          </button>
                        </td>
                      </tr>
                      {isExp && (
                        <tr key={`${i}-detail`}>
                          <td colSpan={12} className="bg-light">
                            <div className="row g-3 p-2">
                              <div className="col-md-5">
                                <DigitalTwinCard twin={p.digital_twin} />
                              </div>
                              <div className="col-md-7">
                                <div className="card">
                                  <div className="card-header py-2 fw-semibold small">Biomarker Details</div>
                                  <div className="card-body p-3">
                                    <div className="row g-2">
                                      {[
                                        ['Heart Rate', p.avg_heart_rate?.toFixed(1), 'bpm', '65–75'],
                                        ['HRV (SDNN)', p.avg_hrv?.toFixed(1), 'ms', '≥50'],
                                        ['SpO₂', p.avg_spo2?.toFixed(1), '%', '≥95'],
                                        ['Steps', p.avg_steps?.toFixed(0), '/day', '10,000'],
                                        ['Sleep Duration', p.avg_sleep_duration?.toFixed(1), 'h', '7–9'],
                                        ['Sleep Quality', p.avg_sleep_quality?.toFixed(1), '/100', '≥60'],
                                        ['Stress Score', p.avg_stress_score?.toFixed(1), '/100', '<40'],
                                        ['Seizure Days', `${p.seizure_days_detected}/${p.total_reading_days}`, '', ''],
                                        ['Fall Events', p.fall_events, '', '0'],
                                      ].map(([k, v, u, ref]) => (
                                        <div key={k} className="col-sm-6">
                                          <div className="d-flex justify-content-between border-bottom pb-1">
                                            <span className="text-muted small">{k}</span>
                                            <span className="fw-semibold small">{v}{u} {ref && <span className="text-muted">(ref: {ref})</span>}</span>
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                    <div className="d-flex gap-2 mt-3 flex-wrap">
                                      {p.seizure_detection_enabled && <span className="badge bg-success">Seizure Detection On</span>}
                                      {p.sleep_tracking && <span className="badge bg-primary">Sleep Tracking On</span>}
                                      {p.stress_tracking && <span className="badge bg-warning text-dark">Stress Tracking On</span>}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && (
        <div className="row g-3">
          {(df?.concepts || []).map((c, i) => (
            <div key={i} className="col-md-6">
              <div className="card h-100 border-start border-4 border-primary">
                <div className="card-header py-2 fw-semibold">{c.name}</div>
                <div className="card-body small text-muted">{c.description}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
