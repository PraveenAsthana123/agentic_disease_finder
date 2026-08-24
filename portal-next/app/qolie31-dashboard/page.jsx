'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const qolColor = (sev) => {
  const s = (sev || '').toLowerCase();
  if (s === 'excellent') return 'success';
  if (s === 'good')      return 'primary';
  if (s === 'fair')      return 'warning';
  if (s === 'poor')      return 'danger';
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

export default function QOLIE31DashboardPage() {
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [tab, setTab]             = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/qolie31-dashboard/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/qolie31-dashboard/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/qolie31-dashboard/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = overview;
  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'domains',    label: '7 QoL Domains' },
    { id: 'patients',   label: 'Per Patient' },
    { id: 'trends',     label: 'Trends' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const totalSev = Object.values(s.severity_distribution || {}).reduce((a, b) => a + b, 0);

  return (
    <div>
      <h3>&#x2764;&#xfe0f; QOLIE-31 — Quality of Life in Epilepsy</h3>
      <p className="text-muted">
        31-item validated PRO instrument (0–100, higher = better QoL) &mdash;{' '}
        {s.total_assessments} assessments across {s.unique_patients} patients &middot;
        cohort avg {(s.avg_score || 0).toFixed(1)}/100 &middot; {(s.pct_poor_fair || 0).toFixed(1)}% poor/fair
      </p>

      {/* KPI row */}
      <div className="row mb-3">
        <KpiCard label="Assessments"     value={s.total_assessments}                        color="primary" />
        <KpiCard label="Unique Patients"  value={s.unique_patients}                           color="info" />
        <KpiCard label="Avg QoL Score"    value={(s.avg_score || 0).toFixed(1)} unit="/100"   color="warning" />
        <KpiCard label="Poor / Fair"      value={`${(s.pct_poor_fair || 0).toFixed(1)}%`}     color="danger"
          sub="score ≤50" />
      </div>

      {/* Severity distribution bar */}
      {s.severity_distribution && totalSev > 0 && (
        <div className="card mb-3 shadow-sm border-0">
          <div className="card-body">
            <h6 className="card-title">QoL Severity Distribution</h6>
            <div className="progress mb-2" style={{ height: '28px' }}>
              {[
                { key: 'poor',      label: 'Poor',      cls: 'bg-danger' },
                { key: 'fair',      label: 'Fair',      cls: 'bg-warning' },
                { key: 'good',      label: 'Good',      cls: 'bg-primary' },
                { key: 'excellent', label: 'Excellent', cls: 'bg-success' },
              ].map(({ key, label, cls }) => {
                const count = (s.severity_distribution || {})[key] || 0;
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
            <div className="d-flex flex-wrap gap-2">
              {Object.entries(s.severity_distribution).map(([k, v]) => (
                <span key={k} className={`badge bg-${qolColor(k)}`}>{k}: {v}</span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Active alerts */}
      {s.active_alerts && s.active_alerts.length > 0 && (
        <div className="alert alert-warning shadow-sm mb-3">
          <strong>&#x26a0;&#xfe0f; {s.active_alerts.length} patient{s.active_alerts.length > 1 ? 's' : ''} with poor/fair QoL requiring intervention</strong>
          <div className="d-flex flex-wrap gap-2 mt-1">
            {s.active_alerts.map(a => (
              <span key={a.patient_id} className="badge bg-warning text-dark">
                {a.patient_id}: {a.score}/100 — {a.severity}
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

      {/* Overview tab — patient summary table */}
      {tab === 'overview' && s.patient_summary && (
        <div className="card shadow-sm border-0">
          <div className="card-body">
            <h6>Patient Summary — Latest QOLIE-31 Scores</h6>
            <p className="small text-muted">Higher score = better quality of life. MCID = 5-point change.</p>
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead>
                  <tr>
                    <th>Patient</th><th>Score</th><th>Max</th><th>QoL Level</th><th>Interpretation</th><th>Assessed</th>
                  </tr>
                </thead>
                <tbody>
                  {s.patient_summary
                    .slice()
                    .sort((a, b) => (a.latest_score || 0) - (b.latest_score || 0))
                    .map(p => (
                    <tr key={p.patient_id}>
                      <td><code>{p.patient_id}</code></td>
                      <td><strong className={`text-${qolColor(p.severity)}`}>{(p.latest_score || 0).toFixed(0)}</strong></td>
                      <td className="text-muted">{p.max_score}</td>
                      <td><span className={`badge bg-${qolColor(p.severity)}`}>{p.severity}</span></td>
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

      {/* Domains tab — 7 QOLIE-31 subscales */}
      {tab === 'domains' && breakdown?.domain_averages && (
        <div className="card shadow-sm border-0">
          <div className="card-body">
            <h6>7 QOLIE-31 Subscale Domain Averages (0–100, higher = better)</h6>
            <p className="small text-muted mb-3">
              Each subscale is scored 0–100. Bar width = % of maximum. Domains below 50 indicate clinically significant impairment.
            </p>
            {breakdown.domain_averages.map(d => {
              const pct = Math.min(100, (d.avg_score / 100) * 100);
              return (
                <div key={d.id} className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-bold">{d.label}</span>
                    <span className="small text-muted">
                      avg {d.avg_score.toFixed(1)} &nbsp;·&nbsp; range {d.min_score}–{d.max_score} (n={d.n})
                    </span>
                  </div>
                  <div className="progress" style={{ height: '18px' }}>
                    <div
                      className={`progress-bar ${pct >= 75 ? 'bg-success' : pct >= 51 ? 'bg-primary' : pct >= 26 ? 'bg-warning' : 'bg-danger'}`}
                      style={{ width: `${pct}%` }}
                      title={`${d.label}: ${d.avg_score.toFixed(1)}/100`}
                    />
                  </div>
                </div>
              );
            })}
            <div className="alert alert-info mt-3 mb-0 small">
              <strong>Clinical note:</strong> Seizure Worry often has the lowest score — correlates with seizure frequency.
              Cognitive Functioning and Energy/Fatigue are most sensitive to AED changes.
            </div>
          </div>
        </div>
      )}

      {/* Per Patient tab */}
      {tab === 'patients' && breakdown?.patient_history && (
        <div className="card shadow-sm border-0">
          <div className="card-body">
            <h6>Per-Patient QOLIE-31 History</h6>
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead>
                  <tr><th>Patient</th><th>Assessments</th><th>First Score</th><th>Latest Score</th><th>Change</th><th>Latest Level</th></tr>
                </thead>
                <tbody>
                  {Object.entries(breakdown.patient_history)
                    .sort(([, a], [, b]) => {
                      const la = a[a.length-1]?.score ?? 0;
                      const lb = b[b.length-1]?.score ?? 0;
                      return la - lb;
                    })
                    .map(([pid, recs]) => {
                      const first = recs[0];
                      const last  = recs[recs.length - 1];
                      const change = (last.score || 0) - (first.score || 0);
                      return (
                        <tr key={pid}>
                          <td><code>{pid}</code></td>
                          <td>{recs.length}</td>
                          <td>{first.score}/100</td>
                          <td><strong className={`text-${qolColor(last.severity)}`}>{last.score}/100</strong></td>
                          <td>
                            <span className={`badge ${change > 0 ? 'bg-success' : change < 0 ? 'bg-danger' : 'bg-secondary'}`}>
                              {change > 0 ? '+' : ''}{change}
                            </span>
                          </td>
                          <td><span className={`badge bg-${qolColor(last.severity)}`}>{last.severity}</span></td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Trends tab */}
      {tab === 'trends' && (
        <div>
          {breakdown?.trend && breakdown.trend.length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6>Monthly QoL Trend</h6>
                <div className="table-responsive">
                  <table className="table table-sm">
                    <thead><tr><th>Month</th><th>Assessments</th><th>Avg Score</th><th>% Poor / Fair</th></tr></thead>
                    <tbody>
                      {breakdown.trend.map(t => (
                        <tr key={t.month}>
                          <td>{t.month}</td>
                          <td>{t.count}</td>
                          <td><strong className={`text-${t.avg_score >= 75 ? 'success' : t.avg_score >= 51 ? 'primary' : t.avg_score >= 26 ? 'warning' : 'danger'}`}>
                            {(t.avg_score || 0).toFixed(1)}
                          </strong></td>
                          <td>
                            <span className={`badge ${t.pct_poor_fair > 50 ? 'bg-danger' : t.pct_poor_fair > 25 ? 'bg-warning' : 'bg-success'}`}>
                              {(t.pct_poor_fair || 0).toFixed(1)}%
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
                <h6>QoL Level Transitions (patients with 2+ assessments)</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-hover">
                    <thead>
                      <tr><th>Patient</th><th>First</th><th>First Level</th><th>Latest</th><th>Latest Level</th><th>Change</th></tr>
                    </thead>
                    <tbody>
                      {breakdown.severity_transitions.map(t => (
                        <tr key={t.patient_id}>
                          <td><code>{t.patient_id}</code></td>
                          <td>{t.first_score}/100</td>
                          <td><span className={`badge bg-${qolColor(t.first_severity)}`}>{t.first_severity}</span></td>
                          <td>{t.latest_score}/100</td>
                          <td><span className={`badge bg-${qolColor(t.latest_severity)}`}>{t.latest_severity}</span></td>
                          <td>
                            <span className={`badge ${t.change > 0 ? 'bg-success' : t.change < 0 ? 'bg-danger' : 'bg-secondary'}`}>
                              {t.change > 0 ? '+' : ''}{t.change}
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
          {(!breakdown?.trend || breakdown.trend.length === 0) &&
           (!breakdown?.severity_transitions || breakdown.severity_transitions.length === 0) && (
            <div className="text-muted">No trend data yet — requires multiple assessments over time.</div>
          )}
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6>{defs.title}</h6>
              <p className="small text-muted mb-0"><em>{defs.reference}</em></p>
            </div>
          </div>

          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6>Severity Tiers</h6>
              <table className="table table-sm">
                <thead><tr><th>Range</th><th>Level</th><th>Clinical Action</th></tr></thead>
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
              <h6>7 Subscale Domains</h6>
              <table className="table table-sm">
                <thead><tr><th>Domain</th><th>Description</th><th>Scoring</th></tr></thead>
                <tbody>
                  {defs.domains?.map(d => (
                    <tr key={d.id}>
                      <td><strong>{d.label}</strong></td>
                      <td className="small">{d.description}</td>
                      <td className="small text-muted">{d.scoring}</td>
                    </tr>
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
