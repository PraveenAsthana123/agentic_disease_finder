'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'categories',  label: 'Categories' },
  { id: 'patients',    label: 'Per Patient' },
  { id: 'definitions', label: 'Definitions' },
];

const CAT_COLOR = {
  adl_restoration:   '#22c55e',
  cognitive_rehab:   '#8b5cf6',
  fine_motor:        '#3b82f6',
  mobility_training: '#f59e0b',
  social_skills:     '#06b6d4',
  vocational_rehab:  '#ef4444',
};

const STATUS_COLOR = {
  active:       'success',
  completed:    'secondary',
  on_hold:      'warning',
  discontinued: 'danger',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-3">
          <div className={`h4 fw-bold mb-1 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function MiniBar({ pct, color }) {
  const w = Math.min(100, Math.max(0, pct || 0));
  return (
    <div className="progress" style={{ height: 8, minWidth: 60 }}>
      <div className="progress-bar" style={{ width: `${w}%`, backgroundColor: color || '#3b82f6' }} />
    </div>
  );
}

function StatusBadge({ status }) {
  const cls = STATUS_COLOR[status] || 'secondary';
  return <span className={`badge bg-${cls}`}>{status?.replace('_', ' ')}</span>;
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;

  const statusDist  = ov.status_dist     || [];
  const catDist     = ov.category_dist   || [];
  const progressDist= ov.progress_dist   || [];
  const catProgress = ov.category_progress|| [];
  const trend       = ov.monthly_trend   || [];

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Total Plans"       value={ov.total_plans}                       color="primary"   sub="across all patients" />
        <KPI label="Total Patients"    value={ov.total_patients}                    color="info"      sub="enrolled in rehab" />
        <KPI label="Avg Progress"      value={`${ov.avg_progress?.toFixed(1)}%`}    color="success"   sub="across active plans" />
        <KPI label="Completion Rate"   value={`${ov.completion_rate?.toFixed(1)}%`} color="warning"   sub="plans fully completed" />
      </div>
      <div className="row mb-4">
        <KPI label="Sessions Planned"    value={ov.total_sessions_planned?.toLocaleString()}   color="secondary" sub="target across all plans" />
        <KPI label="Sessions Completed"  value={ov.total_sessions_completed?.toLocaleString()} color="success"   sub="delivered so far" />
        <KPI label="Avg Session Rate"    value={`${ov.avg_session_rate?.toFixed(1)}%`}         color="info"      sub="sessions attended" />
        <KPI label="Active Plans"        value={statusDist.find(s=>s.name==='active')?.value}  color="primary"   sub="currently running" />
      </div>

      <div className="row mb-4">
        {/* Status distribution */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white"><strong>Plan Status</strong></div>
            <div className="card-body">
              {statusDist.map(s => (
                <div key={s.name} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small text-capitalize">{s.name.replace('_',' ')}</span>
                    <span className="fw-bold small">{s.value}</span>
                  </div>
                  <MiniBar pct={(s.value / ov.total_plans) * 100}
                    color={s.name==='active'?'#22c55e':s.name==='completed'?'#6b7280':s.name==='on_hold'?'#f59e0b':'#ef4444'} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Category distribution */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white"><strong>Category Mix</strong></div>
            <div className="card-body">
              {catDist.map(c => (
                <div key={c.name} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small" style={{ color: CAT_COLOR[c.name] || '#6b7280' }}>
                      {c.name.replace(/_/g,' ')}
                    </span>
                    <span className="fw-bold small">{c.value}</span>
                  </div>
                  <MiniBar pct={(c.value / ov.total_plans) * 100} color={CAT_COLOR[c.name]} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Progress distribution */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white"><strong>Progress Distribution</strong></div>
            <div className="card-body">
              {progressDist.map(p => (
                <div key={p.name} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small">{p.name}</span>
                    <span className="fw-bold small">{p.value} plans</span>
                  </div>
                  <MiniBar pct={(p.value / ov.total_plans) * 100} color="#3b82f6" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Category avg progress */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white"><strong>Avg Progress by Category</strong></div>
        <div className="card-body">
          <div className="row">
            {catProgress.map(c => (
              <div key={c.name} className="col-md-4 mb-3">
                <div className="d-flex justify-content-between mb-1">
                  <span className="small fw-semibold" style={{ color: CAT_COLOR[c.name] }}>
                    {c.name.replace(/_/g,' ')}
                  </span>
                  <span className="small text-muted">{c.count} plans · {c.avg_progress?.toFixed(1)}%</span>
                </div>
                <div className="progress" style={{ height: 14 }}>
                  <div className="progress-bar"
                    style={{ width: `${c.avg_progress}%`, backgroundColor: CAT_COLOR[c.name] || '#3b82f6' }}>
                    {c.avg_progress?.toFixed(0)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Monthly trend */}
      {trend.length > 0 && (
        <div className="card shadow-sm">
          <div className="card-header py-2 bg-dark text-white"><strong>Monthly Trend</strong></div>
          <div className="card-body p-0">
            <table className="table table-sm table-striped mb-0">
              <thead className="table-dark"><tr>
                <th>Month</th><th>New Plans</th><th>Completed</th><th>Avg Progress</th>
              </tr></thead>
              <tbody>
                {trend.map(t => (
                  <tr key={t.month}>
                    <td>{t.month}</td>
                    <td>{t.new_plans}</td>
                    <td>{t.completed}</td>
                    <td>
                      <div className="d-flex align-items-center gap-2">
                        <MiniBar pct={t.avg_progress} color="#22c55e" />
                        <span className="small">{t.avg_progress?.toFixed(1)}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function CategoriesPanel({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  const catStatus   = ov.category_status   || [];
  const catProgress = ov.category_progress || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Category × Status Breakdown</strong>
          <span className="text-muted small ms-2">6 rehab categories, 4 statuses</span>
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-dark"><tr>
              <th>Category</th>
              <th className="text-success">Active</th>
              <th className="text-secondary">Completed</th>
              <th className="text-warning">On Hold</th>
              <th className="text-danger">Discontinued</th>
              <th>Avg Progress</th>
            </tr></thead>
            <tbody>
              {catStatus.map(c => {
                const prog = catProgress.find(p=>p.name===c.category);
                const total = (c.active||0)+(c.completed||0)+(c.on_hold||0)+(c.discontinued||0);
                return (
                  <tr key={c.category}>
                    <td>
                      <span style={{
                        display:'inline-block', width:10, height:10,
                        borderRadius:'50%', backgroundColor: CAT_COLOR[c.category]||'#6b7280',
                        marginRight:6
                      }}/>
                      {c.category.replace(/_/g,' ')}
                    </td>
                    <td className="text-success fw-semibold">{c.active}</td>
                    <td className="text-secondary">{c.completed}</td>
                    <td className="text-warning">{c.on_hold}</td>
                    <td className="text-danger">{c.discontinued}</td>
                    <td>
                      {prog && (
                        <div className="d-flex align-items-center gap-2">
                          <MiniBar pct={prog.avg_progress} color={CAT_COLOR[c.category]} />
                          <span className="small">{prog.avg_progress?.toFixed(1)}%</span>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="row">
        {catStatus.map(c => {
          const total = (c.active||0)+(c.completed||0)+(c.on_hold||0)+(c.discontinued||0);
          const prog = catProgress.find(p=>p.name===c.category);
          return (
            <div key={c.category} className="col-md-4 mb-3">
              <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${CAT_COLOR[c.category]||'#6b7280'}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: CAT_COLOR[c.category] }}>
                    {c.category.replace(/_/g,' ')}
                  </h6>
                  <div className="d-flex justify-content-between small text-muted mb-2">
                    <span>{total} total plans</span>
                    {prog && <span>{prog.avg_progress?.toFixed(1)}% avg progress</span>}
                  </div>
                  <div className="d-flex gap-1 flex-wrap">
                    <span className="badge bg-success">{c.active} active</span>
                    <span className="badge bg-secondary">{c.completed} done</span>
                    <span className="badge bg-warning text-dark">{c.on_hold} hold</span>
                    <span className="badge bg-danger">{c.discontinued} disc.</span>
                  </div>
                  {prog && (
                    <div className="mt-2">
                      <MiniBar pct={prog.avg_progress} color={CAT_COLOR[c.category]} />
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PatientsPanel({ bk }) {
  const [sort, setSort] = useState('total');
  if (!bk) return <div className="text-muted">Loading…</div>;
  const pts = [...(bk.per_patient || [])].sort((a, b) => b[sort] - a[sort]);

  return (
    <div>
      <div className="d-flex align-items-center gap-3 mb-3">
        <span className="text-muted small">Sort by:</span>
        {['total','avg_progress','session_rate','completed','active'].map(f => (
          <button key={f}
            className={`btn btn-sm ${sort===f?'btn-dark':'btn-outline-secondary'}`}
            onClick={()=>setSort(f)}>
            {f.replace('_',' ')}
          </button>
        ))}
      </div>

      <div className="card shadow-sm">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Per-Patient Rehab Plan Summary</strong>
          <span className="text-muted small ms-2">{pts.length} patients</span>
        </div>
        <div className="card-body p-0" style={{ maxHeight: 520, overflowY: 'auto' }}>
          <table className="table table-sm table-hover mb-0">
            <thead className="table-dark" style={{ position:'sticky', top:0 }}><tr>
              <th>Patient</th>
              <th>Total</th>
              <th className="text-success">Active</th>
              <th className="text-secondary">Done</th>
              <th className="text-warning">Hold</th>
              <th className="text-danger">Disc.</th>
              <th>Avg Progress</th>
              <th>Session Rate</th>
            </tr></thead>
            <tbody>
              {pts.map(p => (
                <tr key={p.patient_id}>
                  <td className="fw-semibold">{p.patient_id}</td>
                  <td>{p.total}</td>
                  <td className="text-success">{p.active}</td>
                  <td className="text-secondary">{p.completed}</td>
                  <td className="text-warning">{p.on_hold}</td>
                  <td className="text-danger">{p.discontinued}</td>
                  <td>
                    <div className="d-flex align-items-center gap-2">
                      <MiniBar pct={p.avg_progress}
                        color={p.avg_progress>=75?'#22c55e':p.avg_progress>=50?'#f59e0b':'#ef4444'} />
                      <span className="small">{p.avg_progress?.toFixed(1)}%</span>
                    </div>
                  </td>
                  <td>
                    <div className="d-flex align-items-center gap-2">
                      <MiniBar pct={p.session_rate} color="#3b82f6" />
                      <span className="small">{p.session_rate?.toFixed(1)}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ def }) {
  if (!def) return <div className="text-muted">Loading…</div>;

  const goalCats  = def.goal_categories     || {};
  const statuses  = def.statuses            || {};
  const milestones= def.progress_milestones || {};

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white"><strong>Rehabilitation Goal Categories</strong></div>
        <div className="card-body">
          {Object.entries(goalCats).map(([k, v]) => (
            <div key={k} className="mb-3 pb-3 border-bottom">
              <div className="fw-semibold" style={{ color: CAT_COLOR[k] || '#6b7280' }}>
                {k.replace(/_/g,' ')}
              </div>
              <div className="text-muted small mt-1">{v}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white"><strong>Plan Statuses</strong></div>
            <div className="card-body">
              {Object.entries(statuses).map(([k, v]) => (
                <div key={k} className="mb-2 pb-2 border-bottom">
                  <StatusBadge status={k} />
                  <div className="text-muted small mt-1">{v}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white"><strong>Progress Milestones</strong></div>
            <div className="card-body">
              {Object.entries(milestones).map(([k, v]) => (
                <div key={k} className="mb-2 pb-2 border-bottom">
                  <span className="fw-semibold text-primary">{k}</span>
                  <div className="text-muted small mt-1">{v}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function RehabPlansDashboard() {
  const [tab, setTab]     = useState('overview');
  const [ov,  setOv]      = useState(null);
  const [bk,  setBk]      = useState(null);
  const [def, setDef]     = useState(null);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/rehab-plans/overview`).then(r => r.json()),
      fetch(`${API}/api/rehab-plans/breakdown`).then(r => r.json()),
      fetch(`${API}/api/rehab-plans/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDef(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-4">Loading Rehab Plans…</div>;

  return (
    <div>
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-4 flex-wrap">
        <div>
          <h4 className="mb-0 fw-bold">&#x1f9b4; Rehabilitation Plans</h4>
          <div className="text-muted small">
            {ov.total_plans} plans · {ov.total_patients} patients · 6 rehab categories · real rehab_plans table
          </div>
        </div>
        <div className="ms-auto d-flex gap-2 flex-wrap">
          <span className="badge bg-success fs-6">{ov.total_plans} Plans</span>
          <span className="badge bg-info fs-6">{ov.avg_progress?.toFixed(1)}% Avg Progress</span>
          <span className="badge bg-warning text-dark fs-6">{ov.completion_rate?.toFixed(1)}% Complete</span>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel    ov={ov} />}
      {tab === 'categories'  && <CategoriesPanel  ov={ov} />}
      {tab === 'patients'    && <PatientsPanel     bk={bk} />}
      {tab === 'definitions' && <DefinitionsPanel  def={def} />}
    </div>
  );
}
