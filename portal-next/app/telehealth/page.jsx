'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const qualColor = q => ({ excellent: 'success', good: 'info', fair: 'warning', poor: 'danger' }[q] || 'secondary');
const typeIcon = t => ({
  'video-visit': '📹', 'phone-consult': '📞', 'async-message': '💬', 'remote-monitoring-review': '📡'
}[t] || '🏥');
const starColor = r => r >= 4 ? 'success' : r >= 3 ? 'warning' : 'danger';

export default function TelehealthDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [sort, setSort] = useState('sessions');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/telehealth/overview`).then(r => r.json()),
      fetch(`${API}/api/telehealth/breakdown`).then(r => r.json()),
      fetch(`${API}/api/telehealth/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov || !bd || !defs) return <div className="text-center p-5"><div className="spinner-border text-primary" /></div>;

  const k = ov.kpis;
  const maxTypeCt = Math.max(...ov.session_type_distribution.map(t => t.count), 1);
  const maxPlatCt = Math.max(...ov.platform_distribution.map(p => p.count), 1);
  const maxQualCt = Math.max(...ov.quality_distribution.map(q => q.count), 1);
  const maxSatCt = Math.max(...ov.satisfaction_histogram.map(s => s.count), 1);
  const maxProvSess = Math.max(...bd.provider_stats.map(p => p.sessions), 1);

  const sortedProviders = [...bd.provider_stats].sort((a, b) => b[sort] - a[sort]);
  const sortedPatients = [...(bd.patient_sessions || [])].sort((a, b) => b.total_sessions - a.total_sessions);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <h4 className="mb-0">📹 Telehealth Sessions</h4>
        <span className="badge bg-secondary">{k.total_sessions} sessions · {k.unique_patients} patients · {k.unique_providers} providers</span>
      </div>

      {/* KPI row */}
      <div className="row g-3 mb-3">
        {[
          { label: 'Total Sessions', val: k.total_sessions, color: 'primary', icon: '📋' },
          { label: 'Unique Patients', val: k.unique_patients, color: 'info', icon: '👤' },
          { label: 'Providers', val: k.unique_providers, color: 'secondary', icon: '👩‍⚕️' },
          { label: 'Avg Duration', val: `${k.avg_duration_min} min`, color: 'warning', icon: '⏱️' },
          { label: 'Avg Satisfaction', val: `${k.avg_satisfaction}/5`, color: 'success', icon: '⭐' },
          { label: 'Technical Issue Rate', val: `${(k.technical_issue_rate * 100).toFixed(1)}%`, color: 'danger', icon: '⚠️' },
        ].map(({ label, val, color, icon }) => (
          <div className="col-6 col-md-4 col-lg-2" key={label}>
            <div className={`card border-${color} border-2 h-100`}>
              <div className="card-body text-center p-2">
                <div style={{ fontSize: '1.4rem' }}>{icon}</div>
                <div className={`fw-bold fs-5 text-${color}`}>{val}</div>
                <div className="text-muted small">{label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {['overview', 'providers', 'patients', 'trend', 'definitions'].map(t => (
          <li className="nav-item" key={t}>
            <button className={`nav-link ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
              {{ overview: '📊 Overview', providers: '👩‍⚕️ Providers', patients: '👤 Per Patient', trend: '📈 Trend', definitions: '📚 Definitions' }[t]}
            </button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Session Type Distribution</div>
              <div className="card-body">
                {ov.session_type_distribution.map(({ name, count, pct }) => (
                  <div key={name} className="mb-2">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span>{typeIcon(name)} <strong>{name}</strong></span>
                      <span className="badge bg-primary">{count} <small>({(pct * 100).toFixed(1)}%)</small></span>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div className="progress-bar bg-primary" style={{ width: `${(count / maxTypeCt) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Platform Distribution</div>
              <div className="card-body">
                {ov.platform_distribution.map(({ name, count, pct }) => (
                  <div key={name} className="mb-2">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span>🖥️ <strong>{name}</strong></span>
                      <span className="badge bg-info text-dark">{count} <small>({(pct * 100).toFixed(1)}%)</small></span>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div className="progress-bar bg-info" style={{ width: `${(count / maxPlatCt) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Connection Quality</div>
              <div className="card-body">
                {ov.quality_distribution.map(({ name, count, pct }) => (
                  <div key={name} className="mb-2">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span className={`text-${qualColor(name)} fw-semibold`}>
                        {{ excellent: '🟢', good: '🔵', fair: '🟡', poor: '🔴' }[name] || '⚪'} {name}
                      </span>
                      <span className={`badge bg-${qualColor(name)}`}>{count} <small>({(pct * 100).toFixed(1)}%)</small></span>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div className={`progress-bar bg-${qualColor(name)}`} style={{ width: `${(count / maxQualCt) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Patient Satisfaction (1–5 Stars)</div>
              <div className="card-body">
                {ov.satisfaction_histogram.map(({ rating, count }) => (
                  <div key={rating} className="mb-2">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span className={`text-${starColor(rating)} fw-semibold`}>{'★'.repeat(rating)}{'☆'.repeat(5 - rating)} ({rating})</span>
                      <span className={`badge bg-${starColor(rating)}`}>{count}</span>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div className={`progress-bar bg-${starColor(rating)}`} style={{ width: `${(count / maxSatCt) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* PROVIDERS TAB */}
      {tab === 'providers' && (
        <div className="card shadow-sm">
          <div className="card-header d-flex justify-content-between align-items-center">
            <span className="fw-semibold">Provider Performance</span>
            <div>
              Sort:&nbsp;
              {['sessions', 'avg_satisfaction', 'avg_duration'].map(s => (
                <button key={s} className={`btn btn-sm me-1 ${sort === s ? 'btn-primary' : 'btn-outline-secondary'}`}
                  onClick={() => setSort(s)}>
                  {{ sessions: 'Sessions', avg_satisfaction: 'Satisfaction', avg_duration: 'Duration' }[s]}
                </button>
              ))}
            </div>
          </div>
          <div className="card-body p-0">
            <table className="table table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Provider</th>
                  <th>Sessions</th>
                  <th>Patients</th>
                  <th>Avg Duration</th>
                  <th>Avg Satisfaction</th>
                  <th>Issue Rate</th>
                  <th>Load</th>
                </tr>
              </thead>
              <tbody>
                {sortedProviders.map(({ provider, sessions, patients, avg_duration, avg_satisfaction, issue_rate }) => (
                  <tr key={provider}>
                    <td className="fw-semibold">👩‍⚕️ {provider}</td>
                    <td>{sessions}</td>
                    <td>{patients}</td>
                    <td>{avg_duration} min</td>
                    <td>
                      <span className={`badge bg-${starColor(Math.round(avg_satisfaction))}`}>
                        {avg_satisfaction.toFixed(2)} / 5
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${issue_rate > 0.2 ? 'danger' : issue_rate > 0.15 ? 'warning' : 'success'}`}>
                        {(issue_rate * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td style={{ width: 120 }}>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-primary" style={{ width: `${(sessions / maxProvSess) * 100}%` }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* PATIENTS TAB */}
      {tab === 'patients' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Per-Patient Telehealth Summary</div>
          <div className="card-body p-0">
            <table className="table table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Patient ID</th>
                  <th>Sessions</th>
                  <th>Avg Satisfaction</th>
                  <th>Primary Type</th>
                  <th>Last Session</th>
                </tr>
              </thead>
              <tbody>
                {sortedPatients.map(({ patient_id, total_sessions, avg_satisfaction, last_session, primary_type }) => (
                  <tr key={patient_id}>
                    <td className="fw-semibold font-monospace">{patient_id}</td>
                    <td>{total_sessions}</td>
                    <td>
                      <span className={`badge bg-${starColor(Math.round(avg_satisfaction))}`}>
                        {avg_satisfaction.toFixed(1)} / 5
                      </span>
                    </td>
                    <td>{typeIcon(primary_type)} {primary_type}</td>
                    <td className="text-muted small">{last_session}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TREND TAB */}
      {tab === 'trend' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Monthly Telehealth Trend</div>
          <div className="card-body p-0">
            <table className="table table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Month</th>
                  <th>Sessions</th>
                  <th>Avg Duration (min)</th>
                  <th>Avg Satisfaction</th>
                  <th>Issues</th>
                  <th>Volume</th>
                </tr>
              </thead>
              <tbody>
                {ov.monthly_trend.map(({ month, sessions, avg_duration, avg_satisfaction, issue_count }) => {
                  const maxSess = Math.max(...ov.monthly_trend.map(r => r.sessions), 1);
                  return (
                    <tr key={month}>
                      <td className="fw-semibold">{month}</td>
                      <td>{sessions}</td>
                      <td>{avg_duration}</td>
                      <td>
                        <span className={`badge bg-${starColor(Math.round(avg_satisfaction))}`}>
                          {avg_satisfaction.toFixed(2)}
                        </span>
                      </td>
                      <td>
                        <span className={`badge bg-${issue_count > 3 ? 'danger' : issue_count > 1 ? 'warning' : 'success'}`}>
                          {issue_count}
                        </span>
                      </td>
                      <td style={{ width: 160 }}>
                        <div className="progress" style={{ height: 8 }}>
                          <div className="progress-bar bg-info" style={{ width: `${(sessions / maxSess) * 100}%` }} />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">📋 Session Types</div>
              <div className="card-body">
                <dl className="mb-0">
                  {Object.entries(defs.session_types || {}).map(([k, v]) => (
                    <div key={k} className="mb-2">
                      <dt>{typeIcon(k)} {k}</dt>
                      <dd className="text-muted small mb-0">{v}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">📶 Connection Quality</div>
              <div className="card-body">
                <dl className="mb-0">
                  {Object.entries(defs.connection_quality || {}).map(([k, v]) => (
                    <div key={k} className="mb-2">
                      <dt className={`text-${qualColor(k)}`}>{k}</dt>
                      <dd className="text-muted small mb-0">{v}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">⭐ Satisfaction Scale</div>
              <div className="card-body">
                <dl className="mb-0">
                  {Object.entries(defs.satisfaction_scale || {}).map(([k, v]) => (
                    <div key={k} className="mb-2">
                      <dt className={`text-${starColor(parseInt(k))}`}>{'★'.repeat(parseInt(k))} ({k})</dt>
                      <dd className="text-muted small mb-0">{v}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">📏 KPI Definitions</div>
              <div className="card-body">
                {(defs.kpi_definitions || []).map(({ kpi, desc }) => (
                  <div key={kpi} className="mb-2">
                    <strong>{kpi}</strong>
                    <p className="text-muted small mb-0">{desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
          {defs.clinical_references && (
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">📚 Clinical References</div>
                <div className="card-body">
                  <ul className="mb-0">
                    {defs.clinical_references.map(r => <li key={r} className="text-muted small">{r}</li>)}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
