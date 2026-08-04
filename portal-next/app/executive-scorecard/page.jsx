'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function ExecutiveScorecardPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/executive-scorecard/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/executive-scorecard/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/executive-scorecard/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = overview.summary || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'departments', label: 'Departments' },
    { id: 'instruments', label: 'Instruments' },
    { id: 'operations', label: 'Operations' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f4ca; Executive Scorecard</h3>
      <p className="text-muted">Enterprise clinical & AI operations summary &mdash; {s.total_patients} patients, {s.total_assessments} assessments, {s.total_ai_operations} AI ops</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Patients', value: s.total_patients, color: 'primary' },
          { label: 'Assessments', value: s.total_assessments, color: 'info' },
          { label: 'Seizure Events', value: s.total_seizure_events, color: 'danger' },
          { label: 'Medications', value: s.total_medications, color: 'success' },
          { label: 'AI Operations', value: s.total_ai_operations, color: 'warning' },
          { label: 'Instruments', value: s.instruments_used, color: 'secondary' },
          { label: 'Expert Reviews', value: s.total_expert_reviews, color: 'dark' },
          { label: 'HITL Reviews', value: s.total_hitl_reviews, color: 'primary' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value ?? '\u2014'}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Severity distribution */}
      {overview.severity_distribution && (
        <div className="card mb-3 shadow-sm border-0">
          <div className="card-body">
            <h6 className="card-title">Severity Distribution</h6>
            <div className="progress" style={{height: '28px'}}>
              {Object.entries(overview.severity_distribution).map(([sev, count]) => {
                const total = Object.values(overview.severity_distribution).reduce((a, b) => a + b, 0);
                const pct = total > 0 ? ((count / total) * 100).toFixed(1) : 0;
                const color = sev === 'Severe' ? 'danger' : sev === 'Moderate' ? 'warning' : 'success';
                return (
                  <div key={sev} className={`progress-bar bg-${color}`}
                    style={{width: `${pct}%`}} title={`${sev}: ${count} (${pct}%)`}>
                    {sev} {count}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview tab - department census + top components */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-6">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6 className="card-title">Department Census</h6>
                <table className="table table-sm">
                  <thead><tr><th>Department</th><th className="text-end">Patients</th></tr></thead>
                  <tbody>
                    {(overview.department_census || []).map(d => (
                      <tr key={d.department}><td>{d.department}</td><td className="text-end">{d.count}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6 className="card-title">Top AI Components</h6>
                <table className="table table-sm">
                  <thead><tr><th>Component</th><th className="text-end">Operations</th></tr></thead>
                  <tbody>
                    {(overview.top_components || []).map(c => (
                      <tr key={c.component}><td>{c.component}</td><td className="text-end">{c.operations}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Departments tab */}
      {tab === 'departments' && breakdown && (
        <div className="card shadow-sm border-0 mb-3">
          <div className="card-body">
            <h6 className="card-title">Department Detail</h6>
            <table className="table table-sm table-striped">
              <thead><tr><th>Department</th><th className="text-end">Patients</th><th className="text-end">Assessments</th><th className="text-end">Ratio</th></tr></thead>
              <tbody>
                {(breakdown.department_detail || []).map(d => (
                  <tr key={d.dept}>
                    <td>{d.dept}</td>
                    <td className="text-end">{d.patients}</td>
                    <td className="text-end">{d.assessments}</td>
                    <td className="text-end">{d.patients > 0 ? (d.assessments / d.patients).toFixed(1) : '\u2014'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Instruments tab */}
      {tab === 'instruments' && (
        <div className="row">
          <div className="col-md-6">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6 className="card-title">Instrument Usage (Overview)</h6>
                {(overview.instrument_usage || []).map(inst => {
                  const maxCount = Math.max(...(overview.instrument_usage || []).map(x => x.count));
                  const pct = maxCount > 0 ? ((inst.count / maxCount) * 100).toFixed(0) : 0;
                  return (
                    <div key={inst.instrument} className="mb-1">
                      <div className="d-flex justify-content-between small">
                        <span>{inst.instrument}</span><span>{inst.count}</span>
                      </div>
                      <div className="progress" style={{height: '8px'}}>
                        <div className="progress-bar bg-info" style={{width: `${pct}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          {breakdown && (
            <div className="col-md-6">
              <div className="card shadow-sm border-0 mb-3">
                <div className="card-body">
                  <h6 className="card-title">Instrument Stats (Breakdown)</h6>
                  <table className="table table-sm table-striped">
                    <thead><tr><th>Instrument</th><th className="text-end">Total</th><th className="text-end">Avg Score</th><th className="text-end">Avg Max</th><th className="text-end">Alerts</th></tr></thead>
                    <tbody>
                      {(breakdown.instrument_stats || []).map(ist => (
                        <tr key={ist.instrument}>
                          <td>{ist.instrument}</td>
                          <td className="text-end">{ist.total}</td>
                          <td className="text-end">{ist.avg_score}</td>
                          <td className="text-end">{ist.avg_max}</td>
                          <td className="text-end">{ist.alerts > 0 ? <span className="badge bg-danger">{ist.alerts}</span> : <span className="text-muted">0</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Operations tab - daily ops + monthly seizures */}
      {tab === 'operations' && (
        <div className="row">
          <div className="col-md-6">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6 className="card-title">Daily AI Operations (last 7d)</h6>
                {(overview.daily_operations || []).map(d => {
                  const maxOps = Math.max(...(overview.daily_operations || []).map(x => x.operations));
                  const pct = maxOps > 0 ? ((d.operations / maxOps) * 100).toFixed(0) : 0;
                  return (
                    <div key={d.date} className="mb-1">
                      <div className="d-flex justify-content-between small">
                        <span>{d.date}</span><span>{d.operations}</span>
                      </div>
                      <div className="progress" style={{height: '8px'}}>
                        <div className="progress-bar bg-warning" style={{width: `${pct}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          {breakdown && breakdown.monthly_seizures && (
            <div className="col-md-6">
              <div className="card shadow-sm border-0 mb-3">
                <div className="card-body">
                  <h6 className="card-title">Monthly Seizure Events</h6>
                  <table className="table table-sm">
                    <thead><tr><th>Month</th><th className="text-end">Total Events</th><th className="text-end">Severe</th></tr></thead>
                    <tbody>
                      {(breakdown.monthly_seizures || []).map(m => (
                        <tr key={m.month}>
                          <td>{m.month}</td>
                          <td className="text-end">{m.events}</td>
                          <td className="text-end">{m.severe > 0 ? <span className="badge bg-danger">{m.severe}</span> : <span className="text-muted">0</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div className="card shadow-sm border-0 mb-3">
          <div className="card-body">
            <h6 className="card-title">Metric Definitions</h6>
            <table className="table table-sm table-striped">
              <thead><tr><th>Metric</th><th>Description</th><th>Source</th></tr></thead>
              <tbody>
                {(defs.definitions || []).map(d => (
                  <tr key={d.metric}><td className="fw-bold">{d.metric}</td><td>{d.description}</td><td><code>{d.source}</code></td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="text-muted small mt-3">
        Avg assessments/patient: {s.avg_assessments_per_patient} &middot; Alert rate: {s.alert_rate_pct}% &middot; Source: clinical.db
      </div>
    </div>
  );
}
