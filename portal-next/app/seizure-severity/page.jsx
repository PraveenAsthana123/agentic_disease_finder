'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const LVL_COLORS = { mild: 'success', moderate: 'warning', severe: 'danger', critical: 'dark' };

export default function SeizureSeverityPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/seizure-severity-dashboard/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/seizure-severity-dashboard/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/seizure-severity-dashboard/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'domains', label: 'Domains & Items' },
    { id: 'trends', label: 'Trends & History' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const sevDist = overview.severity_distribution || {};
  const totalAssess = overview.total_assessments || 0;

  return (
    <div>
      <h3>&#x26a1; Seizure Severity Dashboard</h3>
      <p className="text-muted">Liverpool Seizure Severity Scale (LSSS) — real assessment data from clinical.db</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Assessments', value: overview.total_assessments, color: 'primary' },
          { label: 'Unique Patients', value: overview.unique_patients, color: 'info' },
          { label: 'Avg Score', value: `${overview.avg_score}/80`, color: 'warning' },
          { label: 'Min Score', value: overview.min_score, color: 'success' },
          { label: 'Max Score', value: overview.max_score_observed, color: 'danger' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value ?? '\u2014'}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Severity distribution bar */}
      <div className="card mb-3 shadow-sm border-0">
        <div className="card-body">
          <h6 className="card-title">Severity Distribution</h6>
          <div className="progress" style={{height: '28px'}}>
            {['mild','moderate','severe','critical'].map(lvl => {
              const cnt = sevDist[lvl] || 0;
              const pct = totalAssess ? Math.round(cnt / totalAssess * 100) : 0;
              return pct > 0 ? (
                <div key={lvl} className={`progress-bar bg-${LVL_COLORS[lvl]}`}
                  style={{width: `${pct}%`}} title={`${lvl}: ${cnt}`}>
                  {lvl} {cnt} ({pct}%)
                </div>
              ) : null;
            })}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview tab — per-patient latest */}
      {tab === 'overview' && (
        <div>
          <h5>Per-Patient Latest LSSS</h5>
          <div className="table-responsive">
            <table className="table table-hover table-sm">
              <thead className="table-dark">
                <tr><th>Patient</th><th>Score</th><th>Severity</th><th>Interpretation</th><th>Assessed</th></tr>
              </thead>
              <tbody>
                {(overview.patient_summary || []).map(p => (
                  <tr key={p.patient_id}>
                    <td><code>{p.patient_id}</code></td>
                    <td><strong>{p.latest_score}</strong>/80</td>
                    <td><span className={`badge bg-${LVL_COLORS[p.level] || 'secondary'}`}>{p.level}</span></td>
                    <td>{p.interpretation}</td>
                    <td className="text-muted small">{(p.assessed_at || '').slice(0,10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Domains & Items tab */}
      {tab === 'domains' && breakdown && (
        <div>
          <h5>Domain Analysis</h5>
          <div className="row mb-3">
            {(breakdown.domain_summary || []).map(d => {
              const pct = d.max_possible ? Math.round(d.avg_score / d.max_possible * 100) : 0;
              const color = pct > 70 ? 'danger' : pct > 50 ? 'warning' : 'success';
              return (
                <div key={d.domain} className="col-md-4 mb-2">
                  <div className="card shadow-sm border-0">
                    <div className="card-body">
                      <h6>{d.label}</h6>
                      <div className="h4 mb-1">{d.avg_score}<span className="text-muted small">/{d.max_possible}</span></div>
                      <div className="progress" style={{height: '8px'}}>
                        <div className={`progress-bar bg-${color}`} style={{width: `${pct}%`}} />
                      </div>
                      <div className="text-muted small mt-1">{d.n} assessments</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <h5>Per-Item Severity Heatmap <span className="text-muted small">(sorted by severity)</span></h5>
          <div className="table-responsive">
            <table className="table table-sm">
              <thead className="table-dark">
                <tr><th>Item</th><th>Avg Score</th><th>Severity</th></tr>
              </thead>
              <tbody>
                {(breakdown.item_heatmap || []).map(it => {
                  const pct = Math.round(it.avg / it.max * 100);
                  const color = pct > 70 ? 'danger' : pct > 50 ? 'warning' : pct > 30 ? 'info' : 'success';
                  return (
                    <tr key={it.item}>
                      <td>{it.label}</td>
                      <td>{it.avg}/4</td>
                      <td style={{width:'40%'}}>
                        <div className="progress" style={{height:'16px'}}>
                          <div className={`progress-bar bg-${color}`} style={{width: `${pct}%`}}>
                            {pct}%
                          </div>
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

      {/* Trends & History tab */}
      {tab === 'trends' && breakdown && (
        <div>
          <h5>Monthly Severity Trend</h5>
          {(breakdown.trend || []).length > 0 ? (
            <div className="table-responsive mb-4">
              <table className="table table-sm table-bordered">
                <thead className="table-dark">
                  <tr><th>Month</th><th>Avg LSSS Score</th><th>Assessments</th><th>Trend</th></tr>
                </thead>
                <tbody>
                  {breakdown.trend.map((t, i) => {
                    const prev = i > 0 ? breakdown.trend[i-1].avg_score : t.avg_score;
                    const diff = t.avg_score - prev;
                    const arrow = diff > 0 ? '\u2191' : diff < 0 ? '\u2193' : '\u2192';
                    const arrowColor = diff > 0 ? 'text-danger' : diff < 0 ? 'text-success' : 'text-muted';
                    return (
                      <tr key={t.month}>
                        <td>{t.month}</td>
                        <td><strong>{t.avg_score}</strong>/80</td>
                        <td>{t.count}</td>
                        <td><span className={arrowColor}>{arrow} {diff !== 0 ? Math.abs(diff).toFixed(1) : ''}</span></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : <p className="text-muted">No trend data yet</p>}

          <h5>Per-Patient Score History</h5>
          <div className="row">
            {Object.entries(breakdown.patient_history || {}).sort().map(([pid, hist]) => (
              <div key={pid} className="col-md-6 col-lg-4 mb-3">
                <div className="card shadow-sm border-0">
                  <div className="card-header bg-light d-flex justify-content-between align-items-center"
                    style={{cursor:'pointer'}} onClick={() => setExpandedPt(expandedPt === pid ? null : pid)}>
                    <code>{pid}</code>
                    <span className="badge bg-secondary">{hist.length} assessments</span>
                  </div>
                  {(expandedPt === pid || hist.length <= 3) && (
                    <ul className="list-group list-group-flush">
                      {hist.map((h, i) => (
                        <li key={i} className="list-group-item d-flex justify-content-between small">
                          <span>{(h.date || '').slice(0,10)}</span>
                          <span><strong>{h.score}</strong>/80</span>
                          <span className={`badge bg-${LVL_COLORS[h.level] || 'secondary'}`}>{h.level}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          <h5>{defs.title}</h5>
          <div className="table-responsive">
            <table className="table table-striped table-sm">
              <thead className="table-dark"><tr><th>Term</th><th>Definition</th></tr></thead>
              <tbody>
                {(defs.definitions || []).map(d => (
                  <tr key={d.term}><td><strong>{d.term}</strong></td><td>{d.definition}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
