'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const riskColor = r => r === 'high' ? 'danger' : r === 'medium' ? 'warning' : 'success';
const adsColor  = ads => ads >= 60 ? 'danger' : ads >= 35 ? 'warning' : ads >= 20 ? 'info' : 'success';
const hrvColor  = hrv => hrv < 20 ? 'danger' : hrv < 40 ? 'warning' : hrv <= 60 ? 'success' : 'primary';
const trendArrow = slope => slope > 0.05 ? '↑' : slope < -0.05 ? '↓' : '→';

export default function AutonomicAnalysisDashboard() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('ads');
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/autonomic-analysis/overview`).then(r => r.json()),
      fetch(`${API}/api/autonomic-analysis/breakdown`).then(r => r.json()),
      fetch(`${API}/api/autonomic-analysis/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err)  return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-3">Loading autonomic analysis data…</div>;

  const TABS = [
    { id: 'overview',     label: '📊 Overview' },
    { id: 'patients',     label: '👤 Patient Profiles' },
    { id: 'correlation',  label: '⚡ Seizure Correlation' },
    { id: 'definitions',  label: '📖 Definitions' },
  ];

  const patients = bd?.patient_profiles || [];
  const kpis = ov.kpis || {};
  const corr = ov.seizure_autonomic_correlation || {};

  const filtered = patients
    .filter(p => !search ||
      p.patient_id?.toLowerCase().includes(search.toLowerCase()) ||
      p.name?.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === 'ads')  return (b.ads || 0) - (a.ads || 0);
      if (sortBy === 'hrv')  return (a.metrics?.avg_hrv || 0) - (b.metrics?.avg_hrv || 0);
      if (sortBy === 'risk') return (b.avg_seizure_risk || 0) - (a.avg_seizure_risk || 0);
      return (a.patient_id || '').localeCompare(b.patient_id || '');
    });

  return (
    <div className="p-3">
      <h3>🫀 Autonomic Analysis Dashboard</h3>
      <p className="text-muted">
        Wearable biometric analysis — {kpis.total_readings} readings · {kpis.total_patients} patients ·{' '}
        avg ADS {kpis.avg_ads} · median HRV {kpis.median_hrv} ms ·{' '}
        {kpis.seizure_detections} seizure detections · {kpis.high_risk_patients} high-risk
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
          <div className="row mb-4 g-2">
            {[
              { label: 'Patients',         val: kpis.total_patients,   icon: '👤', color: 'primary' },
              { label: 'Readings',         val: kpis.total_readings,   icon: '📈', color: 'info' },
              { label: 'Avg ADS',          val: kpis.avg_ads,          icon: '🫀', color: adsColor(kpis.avg_ads) },
              { label: 'Median HRV (ms)',  val: kpis.median_hrv,       icon: '💓', color: hrvColor(kpis.median_hrv) },
              { label: 'Seizure Events',   val: kpis.seizure_detections,icon: '⚡', color: 'warning' },
              { label: 'High Risk',        val: kpis.high_risk_patients,icon: '🚨', color: kpis.high_risk_patients > 0 ? 'danger' : 'success' },
            ].map(k => (
              <div key={k.label} className="col-6 col-md-4 col-lg-2">
                <div className={`card border-${k.color} h-100`}>
                  <div className="card-body p-2 text-center">
                    <div className="fs-4">{k.icon}</div>
                    <div className={`fs-5 fw-bold text-${k.color}`}>{k.val}</div>
                    <div className="text-muted small">{k.label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Distributions row */}
          <div className="row mb-4 g-3">
            {/* Risk Distribution */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-bold">Risk Distribution</div>
                <div className="card-body">
                  {(ov.risk_distribution || []).map(r => (
                    <div key={r.risk} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className={`badge bg-${riskColor(r.risk)} text-uppercase`}>{r.risk}</span>
                        <span>{r.count} patients</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${riskColor(r.risk)}`}
                          style={{ width: `${kpis.total_patients ? (r.count / kpis.total_patients * 100) : 0}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* HRV Distribution */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-bold">HRV Distribution</div>
                <div className="card-body">
                  {(ov.hrv_distribution || []).map(h => {
                    const total = (ov.hrv_distribution || []).reduce((s, x) => s + x.count, 0);
                    const pct = total ? Math.round(h.count / total * 100) : 0;
                    return (
                      <div key={h.range} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>{h.range}</span>
                          <span>{h.count} ({pct}%)</span>
                        </div>
                        <div className="progress" style={{ height: 8 }}>
                          <div className="progress-bar bg-info" style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* ADS Distribution */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-bold">ADS Distribution</div>
                <div className="card-body">
                  {(ov.ads_distribution || []).map(a => {
                    const total = (ov.ads_distribution || []).reduce((s, x) => s + x.count, 0);
                    const pct = total ? Math.round(a.count / total * 100) : 0;
                    return (
                      <div key={a.range} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>{a.range}</span>
                          <span>{a.count} ({pct}%)</span>
                        </div>
                        <div className="progress" style={{ height: 8 }}>
                          <div className="progress-bar bg-warning" style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* HRV Trend + Device Breakdown */}
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-bold">HRV Monthly Trend</div>
                <div className="card-body p-2">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Month</th><th>Avg HRV (ms)</th><th>Readings</th></tr></thead>
                    <tbody>
                      {(ov.hrv_trend || []).map(t => (
                        <tr key={t.month}>
                          <td>{t.month}</td>
                          <td><span className={`badge bg-${hrvColor(t.avg_hrv)}`}>{t.avg_hrv}</span></td>
                          <td>{t.count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-bold">Device Type Breakdown</div>
                <div className="card-body p-2">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Device Type</th><th>Count</th></tr></thead>
                    <tbody>
                      {(ov.device_breakdown || []).map(d => (
                        <tr key={d.type}>
                          <td>{d.type}</td>
                          <td><span className="badge bg-secondary">{d.count}</span></td>
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

      {/* ── PATIENT PROFILES ── */}
      {tab === 'patients' && (
        <div>
          <div className="d-flex gap-2 mb-3 flex-wrap">
            <input
              className="form-control form-control-sm"
              style={{ maxWidth: 220 }}
              placeholder="Search patient…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            <select className="form-select form-select-sm" style={{ maxWidth: 180 }} value={sortBy} onChange={e => setSortBy(e.target.value)}>
              <option value="ads">Sort: ADS ↓</option>
              <option value="hrv">Sort: HRV ↑ (worst first)</option>
              <option value="risk">Sort: Seizure Risk ↓</option>
              <option value="id">Sort: Patient ID</option>
            </select>
            <span className="text-muted small align-self-center">{filtered.length} / {patients.length} patients</span>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Age/Sex</th>
                  <th>Device</th>
                  <th>ADS</th>
                  <th>Risk</th>
                  <th>Avg HRV</th>
                  <th>Avg HR</th>
                  <th>SpO₂</th>
                  <th>Seizures</th>
                  <th>HRV Trend</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(p => (
                  <tr key={p.patient_id}>
                    <td>
                      <div className="fw-semibold">{p.patient_id}</div>
                      <div className="text-muted small">{p.name}</div>
                    </td>
                    <td className="small">{p.age} / {p.gender?.[0]}</td>
                    <td className="small">{p.device_type}<br /><span className="text-muted">{p.device_brand}</span></td>
                    <td>
                      <span className={`badge bg-${adsColor(p.ads)}`}>{p.ads?.toFixed(1)}</span>
                    </td>
                    <td>
                      <span className={`badge bg-${riskColor(p.risk)}`}>{p.risk}</span>
                    </td>
                    <td>
                      <span className={`badge bg-${hrvColor(p.metrics?.avg_hrv)}`}>{p.metrics?.avg_hrv?.toFixed(1)}</span>
                    </td>
                    <td>{p.metrics?.avg_hr?.toFixed(0)} bpm</td>
                    <td>{p.metrics?.avg_spo2?.toFixed(1)}%</td>
                    <td>{p.metrics?.seizure_events}</td>
                    <td>
                      <span title={`slope ${p.hrv_trend_slope?.toFixed(3)}`}>
                        {trendArrow(p.hrv_trend_slope)} {p.hrv_trend_direction}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── SEIZURE CORRELATION ── */}
      {tab === 'correlation' && (
        <div>
          <h5 className="mb-3">Seizure-Autonomic Correlation</h5>
          <div className="row g-3 mb-4">
            <div className="col-md-6">
              <div className="card border-warning">
                <div className="card-header bg-warning text-dark fw-bold">⚡ During Seizure Events ({corr.seizure_events})</div>
                <div className="card-body">
                  <table className="table table-sm mb-0">
                    <tbody>
                      <tr><td>Avg HRV</td><td className="fw-bold">{corr.avg_hrv_during_seizure?.toFixed(1)} ms</td></tr>
                      <tr><td>Avg Heart Rate</td><td className="fw-bold text-danger">{corr.avg_hr_during_seizure?.toFixed(1)} bpm</td></tr>
                      <tr><td>Avg Stress</td><td className="fw-bold">{corr.avg_stress_during_seizure?.toFixed(1)}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card border-success">
                <div className="card-header bg-success text-white fw-bold">✅ Non-Seizure Intervals ({corr.non_seizure_events})</div>
                <div className="card-body">
                  <table className="table table-sm mb-0">
                    <tbody>
                      <tr><td>Avg HRV</td><td className="fw-bold">{corr.avg_hrv_no_seizure?.toFixed(1)} ms</td></tr>
                      <tr><td>Avg Heart Rate</td><td className="fw-bold text-success">{corr.avg_hr_no_seizure?.toFixed(1)} bpm</td></tr>
                      <tr><td>Avg Stress</td><td className="fw-bold">{corr.avg_stress_no_seizure?.toFixed(1)}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div className="card mb-4">
            <div className="card-header fw-bold">Key Findings</div>
            <div className="card-body">
              <ul className="mb-0">
                <li>
                  <strong>Heart Rate ↑ during seizures:</strong>{' '}
                  {corr.avg_hr_during_seizure?.toFixed(1)} bpm vs {corr.avg_hr_no_seizure?.toFixed(1)} bpm baseline
                  {' '}(+{((corr.avg_hr_during_seizure || 0) - (corr.avg_hr_no_seizure || 0)).toFixed(1)} bpm ictal tachycardia)
                </li>
                <li className="mt-1">
                  <strong>HRV change minimal:</strong>{' '}
                  {corr.avg_hrv_during_seizure?.toFixed(1)} ms seizure vs {corr.avg_hrv_no_seizure?.toFixed(1)} ms baseline
                </li>
                <li className="mt-1">
                  <strong>Seizure detection rate:</strong>{' '}
                  {kpis.seizure_detections} events / {kpis.total_readings} total readings
                  {' '}({kpis.total_readings ? ((kpis.seizure_detections / kpis.total_readings) * 100).toFixed(1) : 0}%)
                </li>
              </ul>
            </div>
          </div>

          <h6 className="mb-2">Per-Patient Seizure Profile</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Seizure Events</th>
                  <th>Avg Seizure Risk</th>
                  <th>Fall Events</th>
                  <th>Avg Stress</th>
                  <th>Sleep (hrs)</th>
                  <th>Steps/day</th>
                </tr>
              </thead>
              <tbody>
                {patients.slice().sort((a, b) => (b.metrics?.seizure_events || 0) - (a.metrics?.seizure_events || 0)).map(p => (
                  <tr key={p.patient_id}>
                    <td><span className="fw-semibold">{p.patient_id}</span><br /><span className="text-muted small">{p.name}</span></td>
                    <td><span className={`badge bg-${(p.metrics?.seizure_events || 0) > 5 ? 'danger' : 'warning'}`}>{p.metrics?.seizure_events}</span></td>
                    <td>{p.avg_seizure_risk?.toFixed(1)}%</td>
                    <td>{p.metrics?.fall_events}</td>
                    <td>{p.metrics?.avg_stress?.toFixed(1)}</td>
                    <td>{p.metrics?.avg_sleep?.toFixed(1)}</td>
                    <td>{p.metrics?.avg_steps?.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div>
          <h5>{defs.title}</h5>
          <p className="text-muted">{defs.description}</p>

          <h6 className="mt-3">Data Sources</h6>
          <table className="table table-sm mb-4">
            <thead><tr><th>Table</th><th>Rows</th><th>Fields</th></tr></thead>
            <tbody>
              {(defs.data_sources || []).map(ds => (
                <tr key={ds.table}>
                  <td className="fw-semibold">{ds.table}</td>
                  <td>{ds.rows}</td>
                  <td className="small text-muted">{ds.fields}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {defs.metrics?.ads && (
            <div className="card mb-3">
              <div className="card-header fw-bold">ADS — {defs.metrics.ads.name}</div>
              <div className="card-body">
                <p className="mb-1"><strong>Range:</strong> {defs.metrics.ads.range}</p>
                <p className="mb-1"><strong>Interpretation:</strong> {defs.metrics.ads.interpretation}</p>
                <ul className="mb-0 small">
                  {(defs.metrics.ads.components || []).map((c, i) => <li key={i}>{c}</li>)}
                </ul>
              </div>
            </div>
          )}

          {defs.metrics?.hrv && (
            <div className="card mb-3">
              <div className="card-header fw-bold">HRV — {defs.metrics.hrv.name}</div>
              <div className="card-body">
                <p className="mb-1"><strong>Unit:</strong> {defs.metrics.hrv.unit}</p>
                <ul className="mb-0 small">
                  {Object.entries(defs.metrics.hrv.ranges || {}).map(([k, v]) => (
                    <li key={k}><strong>{k}:</strong> {v}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {defs.metrics?.risk_levels && (
            <div className="card mb-3">
              <div className="card-header fw-bold">Risk Level Thresholds</div>
              <div className="card-body">
                <ul className="mb-0 small">
                  {Object.entries(defs.metrics.risk_levels).map(([k, v]) => (
                    <li key={k}><span className={`badge bg-${riskColor(k)} me-1`}>{k}</span>{v}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          <h6 className="mt-3">Glossary</h6>
          <table className="table table-sm">
            <thead><tr><th>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {(defs.glossary || []).map(g => (
                <tr key={g.term}>
                  <td className="fw-semibold">{g.term}</td>
                  <td>{g.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
