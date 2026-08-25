'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = (lvl) => {
  const l = (lvl || '').toLowerCase();
  if (l.includes('normal'))   return 'success';
  if (l.includes('mild'))     return 'warning';
  if (l.includes('moderate')) return 'warning';
  if (l.includes('severe'))   return 'danger';
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

export default function EpworthDashboardPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/epworth-dashboard/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/epworth-dashboard/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/epworth-dashboard/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = overview;
  const tabs = [
    { id: 'overview',  label: 'Overview' },
    { id: 'items',     label: 'Situation Items' },
    { id: 'patients',  label: 'Per Patient' },
    { id: 'trend',     label: 'Trends' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const totalSev = Object.values(s.severity_distribution || {}).reduce((a, b) => a + b, 0);

  const SEV_KEYS = [
    { key: 'normal',   label: 'Normal (0–9)',    cls: 'bg-success' },
    { key: 'mild',     label: 'Mild (10–12)',     cls: 'bg-warning' },
    { key: 'moderate', label: 'Moderate (13–15)', cls: 'bg-warning' },
    { key: 'severe',   label: 'Severe (16–24)',   cls: 'bg-danger' },
  ];

  return (
    <div>
      <h3>&#x1f634; ESS — Epworth Sleepiness Scale</h3>
      <p className="text-muted">
        Daytime sleepiness screening (0–24 pts) &mdash; {s.total_assessments} assessments across {s.unique_patients} patients &middot;
        avg score {(s.avg_score || 0).toFixed(1)}/24 &middot; excessive sleepiness threshold &gt;10
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        <KpiCard label="Assessments"       value={s.total_assessments}                                    color="primary" />
        <KpiCard label="Unique Patients"   value={s.unique_patients}                                      color="info" />
        <KpiCard label="Avg Score"         value={(s.avg_score || 0).toFixed(1)} unit="/24"               color="warning" />
        <KpiCard label="Excessive (&gt;10)" value={`${(s.excessive_sleepiness_rate_pct || 0).toFixed(1)}%`} color="danger" />
      </div>

      {/* Severity distribution bar */}
      {totalSev > 0 && (
        <div className="card mb-3 shadow-sm border-0">
          <div className="card-body">
            <h6 className="card-title">Severity Distribution</h6>
            <div className="progress" style={{ height: '28px' }}>
              {SEV_KEYS.map(({ key, label, cls }) => {
                const count = (s.severity_distribution || {})[key] || 0;
                const pct = totalSev > 0 ? ((count / totalSev) * 100).toFixed(1) : 0;
                if (!count) return null;
                return (
                  <div key={key} className={`progress-bar ${cls}`}
                    style={{ width: `${pct}%` }} title={`${label}: ${count} (${pct}%)`}>
                    {label.split(' ')[0]} {count}
                  </div>
                );
              })}
            </div>
            <div className="d-flex flex-wrap gap-2 mt-2">
              {SEV_KEYS.map(({ key, label, cls }) => {
                const count = (s.severity_distribution || {})[key] || 0;
                if (!count) return null;
                return (
                  <span key={key} className={`badge ${cls}`}>{label.split(' ')[0]}: {count}</span>
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
            <h6>Patient Summary — Latest ESS Scores</h6>
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead>
                  <tr>
                    <th>Patient</th><th>Score</th><th>Max</th><th>Severity</th><th>Interpretation</th><th>Assessed</th>
                  </tr>
                </thead>
                <tbody>
                  {[...s.patient_summary]
                    .sort((a, b) => (b.latest_score || 0) - (a.latest_score || 0))
                    .map(p => (
                    <tr key={p.patient_id}>
                      <td><code>{p.patient_id}</code></td>
                      <td><strong className={`text-${sevColor(p.level)}`}>{p.latest_score}</strong></td>
                      <td>{p.max_score}</td>
                      <td><span className={`badge bg-${sevColor(p.level)}`}>{p.level}</span></td>
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

      {/* Items tab — per-situation means */}
      {tab === 'items' && breakdown?.item_means && (
        <div>
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6>Mean Dozing Chance — by Situation (0–3 scale)</h6>
              <p className="text-muted small mb-3">
                0 = would never doze · 1 = slight · 2 = moderate · 3 = high chance of dozing
              </p>
              {breakdown.item_means.map(item => (
                <div key={item.id} className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-bold">{item.label}</span>
                    <span className="small text-muted">{item.mean_score.toFixed(2)} / {item.max} (n={item.n_rated})</span>
                  </div>
                  <div className="progress" style={{ height: '18px' }}>
                    <div
                      className={`progress-bar ${item.mean_score >= 2 ? 'bg-danger' : item.mean_score >= 1 ? 'bg-warning' : 'bg-success'}`}
                      style={{ width: `${(item.mean_score / item.max) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
          {breakdown.item_means_ranked && (
            <div className="card shadow-sm border-0">
              <div className="card-body">
                <h6>Situations Ranked by Sleepiness (highest first)</h6>
                <ol className="mb-0">
                  {breakdown.item_means_ranked.map(item => (
                    <li key={item.id} className="mb-1 small">
                      <strong>{item.label}</strong> &mdash;{' '}
                      <span className={`badge bg-${item.mean_score >= 2 ? 'danger' : item.mean_score >= 1 ? 'warning' : 'success'}`}>
                        {item.mean_score.toFixed(2)}/3
                      </span>
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          )}
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
                  <tr><th>Patient</th><th>Score</th><th>Severity</th><th>Interpretation</th><th>Date</th></tr>
                </thead>
                <tbody>
                  {Object.entries(breakdown.patient_history)
                    .flatMap(([pid, recs]) => recs.map(r => ({ ...r, patient_id: pid })))
                    .sort((a, b) => (b.score || 0) - (a.score || 0))
                    .map((r, i) => (
                    <tr key={`${r.patient_id}-${i}`}>
                      <td><code>{r.patient_id}</code></td>
                      <td><strong className={`text-${sevColor(r.level)}`}>{r.score}/24</strong></td>
                      <td><span className={`badge bg-${sevColor(r.level)}`}>{r.level}</span></td>
                      <td className="small">{r.interpretation}</td>
                      <td className="small text-muted">{(r.date || '').slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Trend tab */}
      {tab === 'trend' && (
        <div>
          {breakdown?.trend && breakdown.trend.length > 0 ? (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6>Monthly Trend</h6>
                <div className="table-responsive">
                  <table className="table table-sm">
                    <thead>
                      <tr><th>Month</th><th>Assessments</th><th>Avg Score</th><th>Excessive Sleepiness %</th></tr>
                    </thead>
                    <tbody>
                      {breakdown.trend.map(t => (
                        <tr key={t.month}>
                          <td>{t.month}</td>
                          <td>{t.count}</td>
                          <td><strong>{(t.avg_score || 0).toFixed(1)}/24</strong></td>
                          <td>
                            <span className={`badge ${t.excessive_pct > 50 ? 'bg-danger' : t.excessive_pct > 25 ? 'bg-warning' : 'bg-success'}`}>
                              {(t.excessive_pct || 0).toFixed(1)}%
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-muted">No monthly trend data available.</div>
          )}
          {breakdown?.severity_transitions && breakdown.severity_transitions.length > 0 && (
            <div className="card shadow-sm border-0">
              <div className="card-body">
                <h6>Severity Transitions (patients with 2+ assessments)</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-hover">
                    <thead>
                      <tr><th>Patient</th><th>First Score</th><th>First Sev.</th><th>Latest Score</th><th>Latest Sev.</th><th>Change</th></tr>
                    </thead>
                    <tbody>
                      {breakdown.severity_transitions.map(t => (
                        <tr key={t.patient_id}>
                          <td><code>{t.patient_id}</code></td>
                          <td>{t.first_score}/24</td>
                          <td><span className={`badge bg-${sevColor(t.first_severity)}`}>{t.first_severity}</span></td>
                          <td>{t.latest_score}/24</td>
                          <td><span className={`badge bg-${sevColor(t.latest_severity)}`}>{t.latest_severity}</span></td>
                          <td>
                            <span className={`badge ${t.change < 0 ? 'bg-success' : t.change > 0 ? 'bg-danger' : 'bg-secondary'}`}>
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
                <thead><tr><th>Score</th><th>Level</th></tr></thead>
                <tbody>
                  <tr><td>0–9</td><td><span className="badge bg-success">Normal</span></td></tr>
                  <tr><td>10–12</td><td><span className="badge bg-warning text-dark">Mild sleepiness</span></td></tr>
                  <tr><td>13–15</td><td><span className="badge bg-warning text-dark">Moderate sleepiness</span></td></tr>
                  <tr><td>16–24</td><td><span className="badge bg-danger">Severe sleepiness</span></td></tr>
                </tbody>
              </table>
            </div>
          </div>
          <div className="card shadow-sm border-0">
            <div className="card-body">
              <dl className="mb-0">
                {(defs.definitions || []).map(d => (
                  <div key={d.term} className="mb-2">
                    <dt className="small fw-bold">{d.term}</dt>
                    <dd className="small text-muted mb-0">{d.definition}</dd>
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
