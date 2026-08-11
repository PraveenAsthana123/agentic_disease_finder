'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = (sev) => {
  const s = (sev || '').toLowerCase().replace(/\s+/g, '_');
  if (s.includes('normal'))   return 'success';
  if (s.includes('mild'))     return 'warning';
  if (s.includes('moderate')) return 'orange';
  if (s.includes('severe'))   return 'danger';
  return 'secondary';
};

const sevBadge = (sev) => {
  const s = (sev || '').toLowerCase().replace(/\s+/g, '_');
  if (s.includes('normal'))   return 'success';
  if (s.includes('mild'))     return 'warning';
  if (s.includes('moderate')) return 'warning';
  if (s.includes('severe'))   return 'danger';
  return 'secondary';
};

function KpiCard({ label, value, unit = '', sub = '', color = 'primary' }) {
  return (
    <div className="col-6 col-md-3 mb-2">
      <div className="card text-center shadow-sm border-0">
        <div className="card-body py-2">
          <div className={`h3 mb-0 text-${color}`}>{value}{unit && <small className="fs-6 ms-1">{unit}</small>}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.72rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

export default function MoCADashboardPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/moca-dashboard/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/moca-dashboard/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/moca-dashboard/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = overview;
  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'domains',     label: 'Cognitive Domains' },
    { id: 'patients',    label: 'Per Patient' },
    { id: 'transitions', label: 'Score Trends' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const totalSev = Object.values(s.severity_distribution || {}).reduce((a, b) => a + b, 0);

  return (
    <div>
      <h3>&#x1f9e9; MoCA — Montreal Cognitive Assessment</h3>
      <p className="text-muted">
        Cognitive screening (0–30 pts) &mdash; {s.total_assessments} assessments across {s.unique_patients} patients &middot;
        avg score {(s.avg_score || 0).toFixed(1)}/30 &middot; cutoff &lt;26
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        <KpiCard label="Assessments"    value={s.total_assessments}                      color="primary" />
        <KpiCard label="Unique Patients" value={s.unique_patients}                        color="info" />
        <KpiCard label="Avg Score"       value={(s.avg_score || 0).toFixed(1)} unit="/30" color="warning" />
        <KpiCard label="% Impaired (<26)" value={`${(s.pct_impaired || 0).toFixed(1)}%`} color="danger" />
      </div>

      {/* Severity distribution bar */}
      {s.severity_distribution && totalSev > 0 && (
        <div className="card mb-3 shadow-sm border-0">
          <div className="card-body">
            <h6 className="card-title">Severity Distribution</h6>
            <div className="progress" style={{ height: '28px' }}>
              {[
                { key: 'normal',   label: 'Normal',   cls: 'bg-success' },
                { key: 'mild_mci', label: 'Mild MCI', cls: 'bg-warning' },
                { key: 'moderate', label: 'Moderate', cls: 'bg-danger bg-opacity-75' },
                { key: 'severe',   label: 'Severe',   cls: 'bg-danger' },
              ].map(({ key, label, cls }) => {
                const count = s.severity_distribution[key] || 0;
                const pct = totalSev > 0 ? ((count / totalSev) * 100).toFixed(1) : 0;
                if (!count) return null;
                return (
                  <div key={key} className={`progress-bar ${cls}`}
                    style={{ width: `${pct}%` }} title={`${label}: ${count} (${pct}%)`}>
                    {label} {count}
                  </div>
                );
              })}
            </div>
            <div className="d-flex flex-wrap gap-2 mt-2">
              {Object.entries(s.severity_distribution).map(([k, v]) => (
                <span key={k} className={`badge bg-${sevBadge(k)}`}>{k.replace(/_/g,' ')}: {v}</span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Active alerts */}
      {s.active_alerts && s.active_alerts.length > 0 && (
        <div className="alert alert-danger shadow-sm mb-3">
          <strong>&#x26a0;&#xfe0f; {s.active_alerts.length} patient{s.active_alerts.length > 1 ? 's' : ''} with moderate/severe impairment</strong>
          <div className="d-flex flex-wrap gap-2 mt-1">
            {s.active_alerts.map(a => (
              <span key={a.patient_id} className="badge bg-danger">
                {a.patient_id}: {a.score}/30 — {a.alert.split('—')[1]?.trim()}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && s.patient_summary && (
        <div className="card shadow-sm border-0">
          <div className="card-body">
            <h6>Patient Summary — Latest MoCA Scores</h6>
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead>
                  <tr>
                    <th>Patient</th><th>Score</th><th>Max</th><th>Severity</th><th>Interpretation</th><th>Assessed</th>
                  </tr>
                </thead>
                <tbody>
                  {s.patient_summary
                    .sort((a, b) => (a.latest_score || 0) - (b.latest_score || 0))
                    .map(p => (
                    <tr key={p.patient_id}>
                      <td><code>{p.patient_id}</code></td>
                      <td><strong className={`text-${sevBadge(p.severity)}`}>{p.latest_score}</strong></td>
                      <td>{p.max_score}</td>
                      <td><span className={`badge bg-${sevBadge(p.severity)}`}>{p.severity}</span></td>
                      <td className="small">{p.interpretation}</td>
                      <td className="small text-muted">{(p.assessed_at || '').slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Domains tab */}
      {tab === 'domains' && breakdown?.domain_scores && (
        <div className="card shadow-sm border-0">
          <div className="card-body">
            <h6>Cognitive Domain Scores — Cohort Averages</h6>
            <p className="text-muted small mb-3">
              Bar width = % of domain maximum. Lower scores indicate domain-specific impairment.
            </p>
            {breakdown.domain_scores.map(d => (
              <div key={d.id} className="mb-3">
                <div className="d-flex justify-content-between mb-1">
                  <span className="small fw-bold">{d.label}</span>
                  <span className="small text-muted">{d.avg_score.toFixed(2)} / {d.max} ({d.pct_of_max}%)</span>
                </div>
                <div className="progress" style={{ height: '18px' }}>
                  <div
                    className={`progress-bar ${d.pct_of_max >= 80 ? 'bg-success' : d.pct_of_max >= 60 ? 'bg-warning' : 'bg-danger'}`}
                    style={{ width: `${d.pct_of_max}%` }}
                  />
                </div>
              </div>
            ))}
            <div className="mt-3">
              <h6 className="small fw-bold text-muted">Domain maxima reference</h6>
              <div className="d-flex flex-wrap gap-2">
                {breakdown.domain_scores.map(d => (
                  <span key={d.id} className="badge bg-secondary">
                    {d.label}: /{d.max}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Per Patient tab */}
      {tab === 'patients' && breakdown?.patient_history && (
        <div className="card shadow-sm border-0">
          <div className="card-body">
            <h6>Per-Patient Assessment History</h6>
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead>
                  <tr><th>Patient</th><th>Assessments</th><th>First Score</th><th>Latest Score</th><th>Change</th><th>Latest Severity</th></tr>
                </thead>
                <tbody>
                  {Object.entries(breakdown.patient_history)
                    .sort(([, a], [, b]) => (a[a.length-1]?.score || 0) - (b[b.length-1]?.score || 0))
                    .map(([pid, recs]) => {
                    const first = recs[0];
                    const last = recs[recs.length - 1];
                    const change = last.score - first.score;
                    return (
                      <tr key={pid}>
                        <td><code>{pid}</code></td>
                        <td>{recs.length}</td>
                        <td>{first.score}/30</td>
                        <td><strong className={`text-${sevBadge(last.severity)}`}>{last.score}/30</strong></td>
                        <td>
                          <span className={`badge ${change > 0 ? 'bg-success' : change < 0 ? 'bg-danger' : 'bg-secondary'}`}>
                            {change > 0 ? '+' : ''}{change}
                          </span>
                        </td>
                        <td><span className={`badge bg-${sevBadge(last.severity)}`}>{last.severity}</span></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Transitions tab */}
      {tab === 'transitions' && (
        <div>
          {breakdown?.trend && breakdown.trend.length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6>Monthly Trend</h6>
                <div className="table-responsive">
                  <table className="table table-sm">
                    <thead><tr><th>Month</th><th>Assessments</th><th>Avg Score</th><th>% Impaired</th></tr></thead>
                    <tbody>
                      {breakdown.trend.map(t => (
                        <tr key={t.month}>
                          <td>{t.month}</td>
                          <td>{t.count}</td>
                          <td><strong>{t.avg_score}/30</strong></td>
                          <td>
                            <span className={`badge ${t.pct_impaired > 50 ? 'bg-danger' : t.pct_impaired > 25 ? 'bg-warning' : 'bg-success'}`}>
                              {t.pct_impaired}%
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
          {breakdown?.severity_transitions && breakdown.severity_transitions.length > 0 && (
            <div className="card shadow-sm border-0">
              <div className="card-body">
                <h6>Severity Transitions (patients with 2+ assessments)</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-hover">
                    <thead>
                      <tr><th>Patient</th><th>First</th><th>First Severity</th><th>Latest</th><th>Latest Severity</th><th>Change</th><th>N Assessments</th></tr>
                    </thead>
                    <tbody>
                      {breakdown.severity_transitions.map(t => (
                        <tr key={t.patient_id}>
                          <td><code>{t.patient_id}</code></td>
                          <td>{t.first_score}/30</td>
                          <td><span className={`badge bg-${sevBadge(t.first_severity)}`}>{t.first_severity}</span></td>
                          <td>{t.latest_score}/30</td>
                          <td><span className={`badge bg-${sevBadge(t.latest_severity)}`}>{t.latest_severity}</span></td>
                          <td>
                            <span className={`badge ${t.change > 0 ? 'bg-success' : t.change < 0 ? 'bg-danger' : 'bg-secondary'}`}>
                              {t.change > 0 ? '+' : ''}{t.change}
                            </span>
                          </td>
                          <td>{t.assessments}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
          {(!breakdown?.severity_transitions || breakdown.severity_transitions.length === 0) &&
           (!breakdown?.trend || breakdown.trend.length === 0) && (
            <div className="text-muted">No trend data available (requires multiple assessments).</div>
          )}
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6>{defs.title}</h6>
              <p className="small text-muted mb-1"><em>{defs.reference}</em></p>
            </div>
          </div>
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6>Severity Bands</h6>
              <table className="table table-sm">
                <thead><tr><th>Range</th><th>Label</th><th>Clinical Action</th></tr></thead>
                <tbody>
                  {defs.severity_tiers?.map(t => (
                    <tr key={t.label}>
                      <td><strong>{t.range[0]}–{t.range[1]}</strong></td>
                      <td><span className="badge" style={{ background: t.color }}>{t.label}</span></td>
                      <td className="small">{t.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6>Cognitive Domains</h6>
              <table className="table table-sm">
                <thead><tr><th>Domain</th><th>Max</th></tr></thead>
                <tbody>
                  {defs.domains?.map(d => (
                    <tr key={d.id}><td>{d.label}</td><td>{d.max}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="card shadow-sm border-0">
            <div className="card-body">
              <h6>Clinical Notes</h6>
              <dl className="mb-0">
                {defs.clinical_notes?.map(n => (
                  <div key={n.term} className="mb-2">
                    <dt className="small fw-bold">{n.term}</dt>
                    <dd className="small text-muted mb-0">{n.definition}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
