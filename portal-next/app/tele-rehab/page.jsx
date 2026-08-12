'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Per Patient' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function BarChart({ title, data, keyField, valueField, color }) {
  if (!data || data.length === 0) return null;
  const max = Math.max(...data.map(d => d[valueField] || 0), 1);
  return (
    <div className="mb-4">
      <h6 className="text-muted mb-2">{title}</h6>
      {data.map((d, i) => (
        <div key={i} className="mb-1">
          <div className="d-flex justify-content-between small mb-0">
            <span>{d[keyField]}</span>
            <span className="fw-bold">{d[valueField]}</span>
          </div>
          <div className="progress" style={{ height: 10 }}>
            <div
              className={`progress-bar bg-${color || 'primary'}`}
              style={{ width: `${Math.round((d[valueField] / max) * 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function QualityBadge({ quality }) {
  const map = { excellent: 'success', good: 'info', fair: 'warning', poor: 'danger' };
  return <span className={`badge bg-${map[quality] || 'secondary'}`}>{quality}</span>;
}

function SatisfactionStars({ score }) {
  if (!score) return <span className="text-muted">—</span>;
  const stars = Math.round(score);
  return (
    <span title={`${score}/5`}>
      {'★'.repeat(stars)}{'☆'.repeat(5 - stars)}
      <span className="text-muted ms-1 small">({score})</span>
    </span>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const k = data.kpis || {};
  return (
    <>
      <div className="alert alert-info py-2 mb-3">
        <strong>OT Tele-Rehab Dashboard</strong> — {k.total_telehealth_sessions} telehealth sessions ·{' '}
        {k.unique_patients_telehealth} patients · {k.total_rehab_plans} rehab plans ·{' '}
        avg satisfaction {k.avg_patient_satisfaction}/5
      </div>

      {/* Telehealth KPIs */}
      <h5 className="mb-3">📹 Telehealth Sessions</h5>
      <div className="row mb-3">
        <KPI label="Total Sessions" value={k.total_telehealth_sessions} color="primary" />
        <KPI label="Unique Patients" value={k.unique_patients_telehealth} color="info" />
        <KPI label="Avg Duration" value={`${k.avg_session_duration_min} min`} color="success" />
        <KPI label="Avg Satisfaction" value={`${k.avg_patient_satisfaction}/5`} color="warning" sub="1–5 Likert" />
      </div>
      <div className="row mb-4">
        <KPI label="Tech Issue Rate" value={`${k.tech_issue_rate_pct}%`} color={k.tech_issue_rate_pct > 20 ? 'danger' : 'success'} />
      </div>

      {/* Rehab Plans KPIs */}
      <h5 className="mb-3">🦴 Rehab Plans</h5>
      <div className="row mb-4">
        <KPI label="Total Plans" value={k.total_rehab_plans} color="primary" />
        <KPI label="Unique Patients" value={k.unique_patients_rehab} color="info" />
        <KPI label="Avg Progress" value={`${k.avg_rehab_progress_pct}%`} color="success" />
        <KPI label="Session Completion" value={`${k.session_completion_rate_pct}%`} color="warning" />
      </div>

      <div className="row">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <BarChart title="Session Types" data={data.session_type_distribution} keyField="session_type" valueField="count" color="primary" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <BarChart title="Platform Distribution" data={data.platform_distribution} keyField="platform" valueField="count" color="info" />
            </div>
          </div>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <BarChart title="Connection Quality" data={data.connection_quality_distribution} keyField="quality" valueField="count" color="success" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <BarChart title="Rehab Plan Status" data={data.rehab_status_distribution} keyField="status" valueField="count" color="warning" />
            </div>
          </div>
        </div>
      </div>

      {/* Provider Summary */}
      <h5 className="mb-3">Provider Workload</h5>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr>
              <th>Provider</th>
              <th>Sessions</th>
              <th>Avg Satisfaction</th>
            </tr>
          </thead>
          <tbody>
            {(data.provider_summary || []).map((p, i) => (
              <tr key={i}>
                <td>{p.provider}</td>
                <td>{p.sessions}</td>
                <td><SatisfactionStars score={p.avg_satisfaction} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Goal Category Distribution */}
      <h5 className="mb-3">Rehab Goal Categories</h5>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr>
              <th>Category</th>
              <th>Plans</th>
              <th>Avg Progress</th>
              <th>Progress Bar</th>
            </tr>
          </thead>
          <tbody>
            {(data.goal_category_distribution || []).map((g, i) => (
              <tr key={i}>
                <td><span className="badge bg-secondary">{g.category.replace(/_/g, ' ')}</span></td>
                <td>{g.count}</td>
                <td>{g.avg_progress}%</td>
                <td style={{ width: 120 }}>
                  <div className="progress" style={{ height: 8 }}>
                    <div className="progress-bar bg-success" style={{ width: `${g.avg_progress}%` }} />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Monthly Trend */}
      <h5 className="mb-3">Monthly Session Trend</h5>
      <div className="table-responsive">
        <table className="table table-sm table-bordered">
          <thead className="table-light">
            <tr>
              <th>Month</th>
              <th>Sessions</th>
              <th>Avg Satisfaction</th>
            </tr>
          </thead>
          <tbody>
            {(data.monthly_trend || []).map((m, i) => (
              <tr key={i}>
                <td>{m.month}</td>
                <td>{m.sessions}</td>
                <td><SatisfactionStars score={m.avg_satisfaction} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const patients = data.per_patient || [];
  return (
    <>
      <div className="alert alert-secondary py-2 mb-3">
        {data.total_patients} patients with telehealth or rehab data
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-hover table-bordered">
          <thead className="table-light">
            <tr>
              <th>Patient ID</th>
              <th>Tele Sessions</th>
              <th>Avg Duration</th>
              <th>Avg Sat.</th>
              <th>Tech Issues</th>
              <th>Last Session</th>
              <th>Rehab Plans</th>
              <th>Avg Progress</th>
              <th>Sessions Done/Planned</th>
              <th>Goals</th>
            </tr>
          </thead>
          <tbody>
            {patients.map((p, i) => (
              <tr key={i}>
                <td><code>{p.patient_id}</code></td>
                <td>{p.tele_sessions || '—'}</td>
                <td>{p.avg_session_duration ? `${p.avg_session_duration} min` : '—'}</td>
                <td><SatisfactionStars score={p.avg_satisfaction} /></td>
                <td>
                  {p.tech_issues > 0
                    ? <span className="badge bg-warning text-dark">{p.tech_issues} issues</span>
                    : <span className="badge bg-success">None</span>}
                </td>
                <td>{p.last_session || '—'}</td>
                <td>{p.rehab_plans || '—'}</td>
                <td>
                  {p.avg_rehab_progress != null ? (
                    <div className="d-flex align-items-center gap-1">
                      <div className="progress flex-grow-1" style={{ height: 8, minWidth: 60 }}>
                        <div
                          className={`progress-bar ${p.avg_rehab_progress >= 75 ? 'bg-success' : p.avg_rehab_progress >= 40 ? 'bg-warning' : 'bg-danger'}`}
                          style={{ width: `${p.avg_rehab_progress}%` }}
                        />
                      </div>
                      <small>{p.avg_rehab_progress}%</small>
                    </div>
                  ) : '—'}
                </td>
                <td>{p.sessions_done}/{p.sessions_planned}</td>
                <td>
                  {(p.goal_categories || '').split(',').filter(Boolean).map((g, gi) => (
                    <span key={gi} className="badge bg-light text-dark border me-1" style={{ fontSize: '0.65rem' }}>
                      {g.trim().replace(/_/g, ' ')}
                    </span>
                  ))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <>
      <div className="alert alert-info py-2 mb-3">
        <strong>{data.dashboard}</strong> — {data.purpose}
      </div>

      <div className="row">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-header bg-primary text-white">Session Types</div>
            <div className="card-body">
              {Object.entries(data.session_types || {}).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <strong>{k}</strong>: {v}
                </div>
              ))}
            </div>
          </div>
          <div className="card shadow-sm mb-3">
            <div className="card-header bg-info text-white">Platforms</div>
            <div className="card-body">
              {Object.entries(data.platforms || {}).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <strong>{k}</strong>: {v}
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-header bg-success text-white">Data Sources</div>
            <div className="card-body">
              {(data.data_sources || []).map((s, i) => <div key={i}>• {s}</div>)}
            </div>
          </div>
          <div className="card shadow-sm mb-3">
            <div className="card-header bg-warning text-dark">Seizure Safety Note</div>
            <div className="card-body small">{data.seizure_safety_note}</div>
          </div>
          <div className="card shadow-sm mb-3">
            <div className="card-header bg-secondary text-white">Rehab Goal Categories</div>
            <div className="card-body">
              {(data.rehab_goal_categories || []).map((g, i) => (
                <div key={i}><span className="badge bg-secondary me-1">{g.split(' ')[0]}</span> {g}</div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header">Metric Definitions</div>
        <div className="card-body">
          <p><strong>Satisfaction Scale:</strong> {data.satisfaction_scale}</p>
          <p><strong>Connection Quality:</strong> {data.connection_quality}</p>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header">References</div>
        <div className="card-body">
          {(data.references || []).map((r, i) => <div key={i}>• {r}</div>)}
        </div>
      </div>
    </>
  );
}

export default function TeleRehabPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [tabData, setTabData] = useState({});
  const [loading, setLoading] = useState({});

  useEffect(() => {
    fetchTab(activeTab);
  }, [activeTab]);

  const fetchTab = async (tab) => {
    if (tabData[tab]) return;
    setLoading(l => ({ ...l, [tab]: true }));
    try {
      const ep = tab === 'overview' ? 'overview' : tab === 'breakdown' ? 'breakdown' : 'definitions';
      const res = await fetch(`${API}/api/tele-rehab/${ep}`);
      const json = await res.json();
      setTabData(d => ({ ...d, [tab]: json }));
    } catch (e) {
      setTabData(d => ({ ...d, [tab]: { error: e.message } }));
    } finally {
      setLoading(l => ({ ...l, [tab]: false }));
    }
  };

  return (
    <div className="container-fluid py-3">
      <h2 className="mb-1">📹 OT Tele-Rehab Dashboard</h2>
      <p className="text-muted mb-3">
        Telehealth sessions × Rehab plans for epilepsy patients under Occupational Therapist care —
        109 sessions · 311 rehab plans · 30 patients · AOTA Telehealth 2018
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button
              className={`nav-link${activeTab === t.id ? ' active' : ''}`}
              onClick={() => setActiveTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      <div>
        {loading[activeTab] && (
          <div className="text-center py-4">
            <div className="spinner-border text-primary" />
          </div>
        )}
        {!loading[activeTab] && activeTab === 'overview' && <OverviewPanel data={tabData['overview']} />}
        {!loading[activeTab] && activeTab === 'breakdown' && <BreakdownPanel data={tabData['breakdown']} />}
        {!loading[activeTab] && activeTab === 'definitions' && <DefinitionsPanel data={tabData['definitions']} />}
      </div>
    </div>
  );
}
