'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function PHQ9DashboardPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/phq9-dashboard/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/phq9-dashboard/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/phq9-dashboard/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = overview;
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'items', label: 'Item Endorsement' },
    { id: 'patients', label: 'Patient Scores' },
    { id: 'transitions', label: 'Severity Transitions' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const sevColor = (sev) => {
    const m = { severe: 'danger', moderate: 'warning', mild: 'info', minimal: 'success' };
    return m[(sev || '').toLowerCase()] || 'secondary';
  };

  return (
    <div>
      <h3>&#x1f4cb; PHQ-9 Depression Screening</h3>
      <p className="text-muted">Patient Health Questionnaire-9 &mdash; {s.total_assessments} assessments across {s.unique_patients} patients, avg score {(s.avg_score || 0).toFixed(1)}/27</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Assessments', value: s.total_assessments, color: 'primary' },
          { label: 'Unique Patients', value: s.unique_patients, color: 'info' },
          { label: 'Avg Score', value: (s.avg_score || 0).toFixed(1), color: 'warning' },
          { label: 'Item 9 Flag Rate', value: `${(s.item9_flag_rate_pct || 0).toFixed(1)}%`, color: 'danger' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 mb-2">
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
      {s.severity_distribution && (
        <div className="card mb-3 shadow-sm border-0">
          <div className="card-body">
            <h6 className="card-title">Severity Distribution</h6>
            <div className="progress" style={{ height: '28px' }}>
              {Object.entries(s.severity_distribution).map(([sev, count]) => {
                const total = Object.values(s.severity_distribution).reduce((a, b) => a + b, 0);
                const pct = total > 0 ? ((count / total) * 100).toFixed(1) : 0;
                return (
                  <div key={sev} className={`progress-bar bg-${sevColor(sev)}`}
                    style={{ width: `${pct}%` }} title={`${sev}: ${count} (${pct}%)`}>
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

      {/* Overview tab */}
      {tab === 'overview' && s.patient_summary && (
        <div className="card shadow-sm border-0">
          <div className="card-body">
            <h6>Patient Summary (top scores)</h6>
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead><tr><th>Patient</th><th>Latest Score</th><th>Max</th><th>Severity</th><th>Assessed</th></tr></thead>
                <tbody>
                  {s.patient_summary
                    .sort((a, b) => (b.latest_score || 0) - (a.latest_score || 0))
                    .slice(0, 20)
                    .map(p => (
                    <tr key={p.patient_id}>
                      <td>{p.patient_id}</td>
                      <td className="fw-bold">{p.latest_score}/{p.max_score}</td>
                      <td>{p.max_score}</td>
                      <td><span className={`badge bg-${sevColor(p.severity)}`}>{p.interpretation || p.severity}</span></td>
                      <td className="text-muted small">{p.assessed_at ? new Date(p.assessed_at).toLocaleDateString() : '\u2014'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Item Endorsement tab */}
      {tab === 'items' && breakdown?.item_endorsement && (
        <div className="card shadow-sm border-0">
          <div className="card-body">
            <h6>PHQ-9 Item Endorsement Rates</h6>
            <p className="text-muted small">Percentage of assessments endorsing each item (any = score &ge; 1, frequent = score &ge; 2)</p>
            {breakdown.item_endorsement.map(item => (
              <div key={item.id} className="mb-3">
                <div className="d-flex justify-content-between mb-1">
                  <span className="fw-bold">{item.label}</span>
                  <span className="text-muted small">Any: {item.any_pct}% | Frequent: {item.frequent_pct}%</span>
                </div>
                <div className="progress" style={{ height: '20px' }}>
                  <div className={`progress-bar ${item.id === 'item9' ? 'bg-danger' : 'bg-primary'}`}
                    style={{ width: `${item.any_pct}%` }}
                    title={`Any endorsement: ${item.any_pct}%`}>
                    {item.any_pct}%
                  </div>
                </div>
                <div className="progress mt-1" style={{ height: '12px' }}>
                  <div className={`progress-bar ${item.id === 'item9' ? 'bg-danger' : 'bg-info'} bg-opacity-75`}
                    style={{ width: `${item.frequent_pct}%` }}
                    title={`Frequent: ${item.frequent_pct}%`}>
                  </div>
                </div>
              </div>
            ))}
            {breakdown.item_endorsement.some(i => i.id === 'item9' && i.any_pct > 0) && (
              <div className="alert alert-danger mt-3">
                <strong>Item 9 Alert:</strong> {breakdown.item_endorsement.find(i => i.id === 'item9')?.any_pct}% of assessments flagged suicidal ideation. Each requires immediate clinical follow-up.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Patient Scores tab */}
      {tab === 'patients' && s.patient_summary && (
        <div className="card shadow-sm border-0">
          <div className="card-body">
            <h6>All Patient PHQ-9 Scores</h6>
            <div className="table-responsive">
              <table className="table table-sm table-striped">
                <thead><tr><th>Patient</th><th>Score</th><th>Severity</th><th>Date</th></tr></thead>
                <tbody>
                  {s.patient_summary.map(p => (
                    <tr key={p.patient_id}>
                      <td>{p.patient_id}</td>
                      <td>
                        <div className="d-flex align-items-center gap-2">
                          <div className="progress flex-grow-1" style={{ height: '16px' }}>
                            <div className={`progress-bar bg-${sevColor(p.severity)}`}
                              style={{ width: `${((p.latest_score || 0) / 27) * 100}%` }}>
                              {p.latest_score}
                            </div>
                          </div>
                          <span className="text-muted small">/27</span>
                        </div>
                      </td>
                      <td><span className={`badge bg-${sevColor(p.severity)}`}>{p.interpretation || p.severity}</span></td>
                      <td className="text-muted small">{p.assessed_at ? new Date(p.assessed_at).toLocaleDateString() : '\u2014'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Severity Transitions tab */}
      {tab === 'transitions' && breakdown?.severity_transitions && (
        <div className="card shadow-sm border-0">
          <div className="card-body">
            <h6>Severity Transitions (Change Over Time)</h6>
            <p className="text-muted small">Score changes between consecutive assessments</p>
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead><tr><th>Patient</th><th>From</th><th>To</th><th>Change</th></tr></thead>
                <tbody>
                  {breakdown.severity_transitions.slice(0, 30).map((t, i) => {
                    const delta = (t.to_score || 0) - (t.from_score || 0);
                    return (
                      <tr key={i}>
                        <td>{t.patient_id}</td>
                        <td><span className={`badge bg-${sevColor(t.from_severity)}`}>{t.from_severity} ({t.from_score})</span></td>
                        <td><span className={`badge bg-${sevColor(t.to_severity)}`}>{t.to_severity} ({t.to_score})</span></td>
                        <td className={delta < 0 ? 'text-success fw-bold' : delta > 0 ? 'text-danger fw-bold' : 'text-muted'}>
                          {delta > 0 ? `+${delta}` : delta === 0 ? '0' : delta}
                          {delta <= -5 && ' \u2705'}
                          {delta >= 5 && ' \u26a0\ufe0f'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div className="card shadow-sm border-0">
          <div className="card-body">
            <h6>{defs.title}</h6>
            <p className="text-muted small">{defs.reference}</p>

            <h6 className="mt-3">Items</h6>
            <div className="table-responsive">
              <table className="table table-sm">
                <thead><tr><th>#</th><th>Label</th><th>Description</th><th>Scoring</th></tr></thead>
                <tbody>
                  {(defs.items || []).map(item => (
                    <tr key={item.id} className={item.id === 'item9' ? 'table-danger' : ''}>
                      <td>{item.id}</td>
                      <td className="fw-bold">{item.label}</td>
                      <td>{item.description}</td>
                      <td className="text-muted small">{item.scoring}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {defs.severity_tiers && (
              <>
                <h6 className="mt-3">Severity Tiers</h6>
                <div className="table-responsive">
                  <table className="table table-sm">
                    <thead><tr><th>Range</th><th>Label</th><th>Action</th></tr></thead>
                    <tbody>
                      {defs.severity_tiers.map(tier => (
                        <tr key={tier.label}>
                          <td><span className="badge" style={{ backgroundColor: tier.color }}>{tier.range[0]}&ndash;{tier.range[1]}</span></td>
                          <td className="fw-bold">{tier.label}</td>
                          <td>{tier.action}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}

            {defs.clinical_notes && (
              <>
                <h6 className="mt-3">Clinical Notes</h6>
                <dl>
                  {defs.clinical_notes.map(n => (
                    <div key={n.term}>
                      <dt>{n.term}</dt>
                      <dd className="text-muted">{n.definition}</dd>
                    </div>
                  ))}
                </dl>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
