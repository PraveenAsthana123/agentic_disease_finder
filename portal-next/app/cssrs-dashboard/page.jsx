'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const RISK_COLOR = r =>
  r === 'high'     ? 'danger'  :
  r === 'moderate' ? 'warning' :
  r === 'low'      ? 'info'    : 'success';

export default function CSSRSDashboardPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetch(`${API}/api/cssrs-dashboard/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/cssrs-dashboard/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/cssrs-dashboard/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-danger" /></div>;

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'screening',  label: 'Screening Items' },
    { id: 'patients',   label: 'Per Patient' },
    { id: 'trend',      label: 'Monthly Trend' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const filteredPts = (ov.patient_summary || []).filter(p =>
    !search || p.patient_id.toLowerCase().includes(search.toLowerCase())
  );

  const rd = ov.risk_distribution || {};
  const rdTotal = Object.values(rd).reduce((a, b) => a + b, 0);

  return (
    <div>
      <h3>&#x26a0;&#xfe0f; C-SSRS — Columbia Suicide Severity Rating Scale</h3>
      <p className="text-muted small">
        {ov.total_assessments} assessments · {ov.unique_patients} patients · avg score {(ov.avg_score || 0).toFixed(1)}/31 · ideation rate {(ov.ideation_rate_pct || 0).toFixed(1)}%
        &nbsp;&mdash;&nbsp;Epilepsy patients carry 3-5× higher suicide risk (Christensen 2007)
      </p>

      {/* Active alerts banner */}
      {(ov.active_alerts || []).length > 0 && (
        <div className="alert alert-danger py-2 mb-3">
          <strong>&#x1f6a8; {(ov.active_alerts || []).length} Active Safety Alert{(ov.active_alerts||[]).length > 1 ? 's' : ''}</strong>
          <div className="mt-1">
            {(ov.active_alerts || []).map((a, i) => (
              <span key={i} className={`badge bg-${a.level === 'high' ? 'danger' : 'warning'} me-2`}>
                {a.patient_id}: {a.alert} (score {a.score})
              </span>
            ))}
          </div>
        </div>
      )}

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Assessments',    value: ov.total_assessments,              color: 'primary' },
          { label: 'Patients',       value: ov.unique_patients,                 color: 'info'    },
          { label: 'Avg Score /31',  value: (ov.avg_score || 0).toFixed(1),    color: 'warning' },
          { label: 'Ideation Rate',  value: `${(ov.ideation_rate_pct||0).toFixed(1)}%`, color: 'danger' },
          { label: 'High Risk',      value: rd.high || 0,                       color: 'danger'  },
          { label: 'Moderate Risk',  value: rd.moderate || 0,                   color: 'warning' },
          { label: 'Low Risk',       value: rd.low || 0,                        color: 'info'    },
          { label: 'None',           value: rd.none || 0,                       color: 'success' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 col-lg-3 mb-2">
            <div className="card text-center shadow-sm border-0 h-100">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value ?? '—'}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Risk distribution bar */}
      <div className="card mb-3 shadow-sm border-0">
        <div className="card-body">
          <h6 className="card-title mb-2">Risk Distribution</h6>
          <div className="progress" style={{ height: '28px' }}>
            {[
              { key: 'high',     label: 'High',     color: 'danger'  },
              { key: 'moderate', label: 'Moderate', color: 'warning' },
              { key: 'low',      label: 'Low',      color: 'info'    },
              { key: 'none',     label: 'None',     color: 'success' },
            ].map(({ key, label, color }) => {
              const n = rd[key] || 0;
              const pct = rdTotal > 0 ? ((n / rdTotal) * 100).toFixed(1) : 0;
              return (
                <div key={key}
                  className={`progress-bar bg-${color}`}
                  style={{ width: `${pct}%` }}
                  title={`${label}: ${n} (${pct}%)`}>
                  {n > 0 ? `${label} ${n}` : ''}
                </div>
              );
            })}
          </div>
          <div className="d-flex gap-3 mt-1 flex-wrap">
            {[
              { key: 'high', label: 'High (17-31)', color: 'danger' },
              { key: 'moderate', label: 'Moderate (8-16)', color: 'warning' },
              { key: 'low', label: 'Low (1-7)', color: 'info' },
              { key: 'none', label: 'None (0)', color: 'success' },
            ].map(({ key, label, color }) => (
              <span key={key} className="small">
                <span className={`badge bg-${color} me-1`}>{rd[key] || 0}</span>{label}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div>
          <h5>Active Safety Alerts</h5>
          {(ov.active_alerts || []).length === 0
            ? <p className="text-muted">No active alerts.</p>
            : (
              <div className="table-responsive mb-4">
                <table className="table table-sm table-bordered">
                  <thead className="table-dark">
                    <tr><th>Patient</th><th>Alert</th><th>Score</th><th>Level</th><th>Date</th></tr>
                  </thead>
                  <tbody>
                    {(ov.active_alerts || []).map((a, i) => (
                      <tr key={i}>
                        <td><strong>{a.patient_id}</strong></td>
                        <td>{a.alert}</td>
                        <td>{a.score}</td>
                        <td><span className={`badge bg-${RISK_COLOR(a.level)}`}>{a.level}</span></td>
                        <td>{a.date ? a.date.split('T')[0] : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          }

          {/* Risk transitions */}
          {bd?.risk_transitions && bd.risk_transitions.length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6 className="card-title">Risk Level Transitions</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-secondary">
                      <tr><th>Patient</th><th>From</th><th>To</th><th>Change</th><th>Date</th></tr>
                    </thead>
                    <tbody>
                      {bd.risk_transitions.slice(0, 10).map((t, i) => (
                        <tr key={i}>
                          <td>{t.patient_id}</td>
                          <td><span className={`badge bg-${RISK_COLOR(t.from_level)}`}>{t.from_level}</span></td>
                          <td><span className={`badge bg-${RISK_COLOR(t.to_level)}`}>{t.to_level}</span></td>
                          <td className={t.score_change > 0 ? 'text-danger' : 'text-success'}>
                            {t.score_change > 0 ? '+' : ''}{(t.score_change || 0).toFixed(1)}
                          </td>
                          <td>{t.assessed_at ? t.assessed_at.split('T')[0] : '—'}</td>
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

      {/* Screening Items tab */}
      {tab === 'screening' && bd && (
        <div>
          <h5>Screening Item Endorsement Rates</h5>
          <p className="text-muted small">6 binary screening items (yes/no). Higher items = greater severity.</p>
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              {(bd.screening_rates || []).map((item, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-semibold">{i + 1}. {item.label}</span>
                    <span className="small text-muted">{item.endorsed_count}/{item.total} ({(item.rate_pct || 0).toFixed(1)}%)</span>
                  </div>
                  <div className="progress" style={{ height: '18px' }}>
                    <div
                      className={`progress-bar ${i >= 2 ? 'bg-danger' : i === 1 ? 'bg-warning' : 'bg-info'}`}
                      style={{ width: `${item.rate_pct || 0}%` }}
                      title={`${item.rate_pct?.toFixed(1)}%`}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <h5 className="mt-4">Intensity Profile (Ideators Only)</h5>
          <p className="text-muted small">Rated 1-5 for patients endorsing any ideation (n={bd.intensity_summary?.[0]?.n_rated || 0})</p>
          <div className="card shadow-sm border-0">
            <div className="card-body">
              {(bd.intensity_summary || []).map((item, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-semibold">{item.label}</span>
                    <span className="small text-muted">avg {(item.avg || 0).toFixed(1)} / max {item.max}</span>
                  </div>
                  <div className="progress" style={{ height: '18px' }}>
                    <div
                      className="progress-bar bg-warning"
                      style={{ width: `${((item.avg || 0) / 5) * 100}%` }}
                      title={`avg ${item.avg?.toFixed(1)}`}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Per Patient tab */}
      {tab === 'patients' && (
        <div>
          <div className="mb-2">
            <input className="form-control form-control-sm w-25"
              placeholder="Search patient ID…"
              value={search} onChange={e => setSearch(e.target.value)} />
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-bordered table-hover">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Latest Score</th>
                  <th>Max Score</th>
                  <th>Risk Level</th>
                  <th>Interpretation</th>
                  <th>Last Assessed</th>
                  <th>Alert</th>
                </tr>
              </thead>
              <tbody>
                {filteredPts.map((p, i) => (
                  <tr key={i}>
                    <td><strong>{p.patient_id}</strong></td>
                    <td>{(p.latest_score || 0).toFixed(0)}</td>
                    <td className="text-muted">{(p.max_score || 0).toFixed(0)}</td>
                    <td><span className={`badge bg-${RISK_COLOR(p.level)}`}>{p.level}</span></td>
                    <td className="small">{p.interpretation}</td>
                    <td className="small text-muted">{p.assessed_at ? p.assessed_at.split('T')[0] : '—'}</td>
                    <td className="small">{p.alert
                      ? <span className="badge bg-danger">{p.alert}</span>
                      : <span className="text-muted">—</span>
                    }</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-muted small">{filteredPts.length} of {(ov.patient_summary||[]).length} patients shown</p>
        </div>
      )}

      {/* Monthly Trend tab */}
      {tab === 'trend' && bd && (
        <div>
          <h5>Monthly Trend</h5>
          <div className="table-responsive">
            <table className="table table-sm table-bordered">
              <thead className="table-secondary">
                <tr><th>Month</th><th>Assessments</th><th>Avg Score</th><th>Ideation Rate</th></tr>
              </thead>
              <tbody>
                {(bd.trend || []).map((t, i) => (
                  <tr key={i}>
                    <td>{t.month}</td>
                    <td>{t.count}</td>
                    <td>{(t.avg_score || 0).toFixed(1)}</td>
                    <td>
                      <div className="d-flex align-items-center gap-2">
                        <div className="progress flex-grow-1" style={{ height: '12px' }}>
                          <div className="progress-bar bg-danger"
                            style={{ width: `${t.ideation_pct || 0}%` }} />
                        </div>
                        <span className="small text-muted">{(t.ideation_pct || 0).toFixed(0)}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {bd.patient_history && bd.patient_history.length > 0 && (
            <div className="mt-4">
              <h5>Patient History</h5>
              <div className="table-responsive">
                <table className="table table-sm table-bordered">
                  <thead className="table-secondary">
                    <tr><th>Patient</th><th>Date</th><th>Score</th><th>Ideation</th><th>Behavior</th><th>Risk</th></tr>
                  </thead>
                  <tbody>
                    {bd.patient_history.slice(0, 20).map((h, i) => (
                      <tr key={i}>
                        <td>{h.patient_id}</td>
                        <td className="small">{h.assessed_at ? h.assessed_at.split('T')[0] : '—'}</td>
                        <td>{(h.total_score || 0).toFixed(0)}</td>
                        <td>{h.has_ideation
                          ? <span className="badge bg-warning">Yes</span>
                          : <span className="text-muted">No</span>
                        }</td>
                        <td>{h.has_behavior
                          ? <span className="badge bg-danger">Yes</span>
                          : <span className="text-muted">No</span>
                        }</td>
                        <td><span className={`badge bg-${RISK_COLOR(h.risk_level)}`}>{h.risk_level}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {bd.patient_history.length > 20 && (
                  <p className="text-muted small">Showing 20 of {bd.patient_history.length} records</p>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          <h5>{defs.title}</h5>
          <div className="row">
            {(defs.definitions || []).map((d, i) => (
              <div key={i} className="col-md-6 mb-3">
                <div className="card shadow-sm border-0 h-100">
                  <div className="card-body py-2">
                    <h6 className="card-title text-primary mb-1">{d.term}</h6>
                    <p className="card-text small mb-0">{d.definition}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="alert alert-warning mt-3 py-2">
            <strong>&#x26a0;&#xfe0f; Clinical Note:</strong> Epilepsy patients have 3-5× higher suicide risk than the general population
            (Christensen et al., 2007). Standardized C-SSRS screening is recommended at each clinical encounter.
            High-risk patients (score &ge;17) require immediate psychiatric referral and means restriction counseling.
          </div>
          <div className="mt-3">
            <h6>References</h6>
            <ul className="small text-muted">
              <li>Posner K et al. (2011). <em>Columbia Suicide Severity Rating Scale (C-SSRS)</em>. Archives of General Psychiatry.</li>
              <li>Christensen J et al. (2007). <em>Suicide in patients with epilepsy</em>. BMJ. doi:10.1136/bmj.39174.560800.BE</li>
              <li>Hesdorffer DC et al. (2012). <em>ILAE suicidality report</em>. Epilepsia.</li>
              <li>FDA (2008). <em>Antiepileptic drug suicidality warning</em>. FDA Safety Communication.</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
