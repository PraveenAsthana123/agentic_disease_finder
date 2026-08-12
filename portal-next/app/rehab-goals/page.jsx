'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLOR = {
  active: 'primary',
  completed: 'success',
  on_hold: 'warning',
  discontinued: 'danger',
};
const STATUS_LABEL = {
  active: 'Active',
  completed: 'Completed',
  on_hold: 'On Hold',
  discontinued: 'Discontinued',
};
const CAT_COLOR = {
  adl_restoration: 'primary',
  vocational_rehab: 'success',
  cognitive_rehab: 'info',
  social_skills: 'warning',
  fine_motor: 'secondary',
  mobility_training: 'danger',
};

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'categories', label: 'Goal Categories' },
  { id: 'patients', label: 'Per Patient' },
  { id: 'recent', label: 'Recent Plans' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, sub, color = 'primary' }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className={`card border-${color} h-100`}>
        <div className="card-body text-center p-3">
          <div className={`display-6 fw-bold text-${color}`}>{value ?? '—'}</div>
          <div className="small fw-semibold">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function ProgressBar({ value, color = 'primary', height = 10 }) {
  return (
    <div className="progress" style={{ height }}>
      <div
        className={`progress-bar bg-${color}`}
        style={{ width: `${Math.min(100, value || 0)}%` }}
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={100}
      />
    </div>
  );
}

function StatusBadge({ status }) {
  return (
    <span className={`badge bg-${STATUS_COLOR[status] || 'secondary'}`}>
      {STATUS_LABEL[status] || status}
    </span>
  );
}

function CatBadge({ cat, label }) {
  return (
    <span className={`badge bg-${CAT_COLOR[cat] || 'secondary'} bg-opacity-75`}>
      {label || cat}
    </span>
  );
}

function OverviewPanel({ ov }) {
  if (!ov?.kpis) return <div className="text-muted p-3">Loading…</div>;
  const k = ov.kpis;
  const cats = ov.category_distribution || [];
  const statuses = ov.status_distribution || [];
  const topPatients = ov.top_patients_by_goals || [];

  return (
    <div>
      {/* KPI row 1 */}
      <div className="row mb-2">
        <KPI label="Total Plans" value={k.total_plans} sub="rehab_plans records" color="primary" />
        <KPI label="Patients" value={k.patients} sub="with ≥1 goal" color="info" />
        <KPI label="Completed" value={k.completed} sub={`${k.completion_rate_pct}% completion`} color="success" />
        <KPI label="Active" value={k.active} sub={`${k.avg_active_progress_pct}% avg progress`} color="primary" />
      </div>
      {/* KPI row 2 */}
      <div className="row mb-4">
        <KPI label="On Hold" value={k.on_hold} sub="temporarily paused" color="warning" />
        <KPI label="Discontinued" value={k.discontinued} sub="re-evaluated / replaced" color="danger" />
        <KPI label="Sessions Planned" value={k.total_sessions_planned} sub="across all plans" color="secondary" />
        <KPI label="Sessions Done" value={k.total_sessions_completed} sub={`${k.session_completion_rate_pct}% rate`} color="success" />
      </div>

      <div className="row">
        {/* Category distribution */}
        <div className="col-md-7 mb-4">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Goal Category Distribution</div>
            <div className="card-body">
              {cats.map((c, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between align-items-center mb-1">
                    <span className="fw-semibold small">
                      <CatBadge cat={c.category} label={c.label} />
                    </span>
                    <span className="small text-muted">
                      {c.count} plans · {c.avg_progress}% avg · {c.completion_rate}% complete
                    </span>
                  </div>
                  <ProgressBar value={c.avg_progress} color={CAT_COLOR[c.category] || 'primary'} height={12} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Status distribution */}
        <div className="col-md-5 mb-4">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Status Breakdown</div>
            <div className="card-body">
              {statuses.map((s, i) => (
                <div key={i} className="d-flex justify-content-between align-items-center mb-3">
                  <StatusBadge status={s.status} />
                  <div className="flex-grow-1 mx-3">
                    <ProgressBar value={(s.count / k.total_plans) * 100} color={STATUS_COLOR[s.status] || 'secondary'} height={8} />
                  </div>
                  <span className="small fw-bold" style={{ minWidth: 28 }}>{s.count}</span>
                </div>
              ))}

              <hr />
              <div className="small text-muted fw-semibold mb-1">Top Patients by Goals</div>
              <table className="table table-sm mb-0" style={{ fontSize: 12 }}>
                <thead><tr><th>Patient</th><th>Goals</th><th>Done</th><th>Progress</th></tr></thead>
                <tbody>
                  {topPatients.map((p, i) => (
                    <tr key={i}>
                      <td>{p.patient_id}</td>
                      <td>{p.total_goals}</td>
                      <td><span className="badge bg-success">{p.completed_goals}</span></td>
                      <td>
                        <ProgressBar value={p.avg_progress} color="primary" height={6} />
                        <span style={{ fontSize: 10 }}>{p.avg_progress}%</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function CategoriesPanel({ bd }) {
  const cats = bd?.category_detail || [];
  const [expanded, setExpanded] = useState(null);

  return (
    <div>
      <p className="text-muted small mb-3">
        6 OT goal categories across {cats.reduce((a, c) => a + c.total, 0)} plans.
        Click a category to expand status breakdown.
      </p>
      {cats.map((c, i) => {
        const isOpen = expanded === c.category;
        const completedSt = c.statuses.find(s => s.status === 'completed');
        const activeSt = c.statuses.find(s => s.status === 'active');
        const totalSess = c.statuses.reduce((a, s) => a + (s.sessions_planned || 0), 0);
        const doneSess = c.statuses.reduce((a, s) => a + (s.sessions_completed || 0), 0);
        return (
          <div className="card shadow-sm mb-3" key={i}>
            <div
              className="card-header d-flex justify-content-between align-items-center"
              style={{ cursor: 'pointer' }}
              onClick={() => setExpanded(isOpen ? null : c.category)}
            >
              <span>
                <CatBadge cat={c.category} label={c.label} />
                <span className="ms-2 text-muted small">{c.description}</span>
              </span>
              <span>
                <span className="badge bg-light text-dark me-1">{c.total} plans</span>
                {completedSt && <span className="badge bg-success me-1">{completedSt.count} done</span>}
                {activeSt && <span className="badge bg-primary me-1">{activeSt.count} active</span>}
                <span className="ms-1">{isOpen ? '▲' : '▼'}</span>
              </span>
            </div>
            {isOpen && (
              <div className="card-body">
                <div className="row mb-3">
                  <div className="col-md-6">
                    <div className="small text-muted mb-1">Status breakdown</div>
                    {c.statuses.map((s, j) => (
                      <div key={j} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <StatusBadge status={s.status} />
                          <span className="text-muted">{s.count} plans · {s.avg_progress}% avg progress</span>
                        </div>
                        <ProgressBar value={s.avg_progress} color={STATUS_COLOR[s.status] || 'secondary'} height={8} />
                      </div>
                    ))}
                  </div>
                  <div className="col-md-6">
                    <div className="small text-muted mb-2">Session completion</div>
                    <div className="d-flex justify-content-between small mb-1">
                      <span>Sessions planned</span>
                      <strong>{totalSess}</strong>
                    </div>
                    <div className="d-flex justify-content-between small mb-1">
                      <span>Sessions completed</span>
                      <strong>{doneSess}</strong>
                    </div>
                    <ProgressBar value={totalSess ? (doneSess / totalSess) * 100 : 0} color="success" height={12} />
                    <div className="small text-muted mt-1">{totalSess ? Math.round((doneSess / totalSess) * 100) : 0}% session rate</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function PatientsPanel({ bd }) {
  const patients = bd?.per_patient || [];
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState('patient_id');
  const [sortAsc, setSortAsc] = useState(true);
  const [selectedPat, setSelectedPat] = useState(null);

  const filtered = patients
    .filter(p => p.patient_id.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      const av = a[sortKey] ?? 0;
      const bv = b[sortKey] ?? 0;
      return sortAsc
        ? (typeof av === 'string' ? av.localeCompare(bv) : av - bv)
        : (typeof bv === 'string' ? bv.localeCompare(av) : bv - av);
    });

  const thClick = key => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(true); }
  };

  const Th = ({ k, label }) => (
    <th style={{ cursor: 'pointer' }} onClick={() => thClick(k)}>
      {label} {sortKey === k ? (sortAsc ? '▲' : '▼') : ''}
    </th>
  );

  return (
    <div>
      <input
        className="form-control mb-3"
        placeholder="Search patient ID…"
        value={search}
        onChange={e => setSearch(e.target.value)}
        style={{ maxWidth: 280 }}
      />
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle" style={{ fontSize: 13 }}>
          <thead className="table-light">
            <tr>
              <Th k="patient_id" label="Patient" />
              <Th k="total_goals" label="Goals" />
              <Th k="completed" label="Done" />
              <Th k="active" label="Active" />
              <Th k="on_hold" label="On Hold" />
              <Th k="discontinued" label="Disc." />
              <Th k="avg_progress" label="Avg Progress" />
              <Th k="sessions_completed" label="Sessions" />
              <Th k="completion_rate" label="Compl. Rate" />
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((p, i) => (
              <>
                <tr key={i} style={{ cursor: 'pointer' }} onClick={() => setSelectedPat(selectedPat === p.patient_id ? null : p.patient_id)}>
                  <td className="fw-semibold">{p.patient_id}</td>
                  <td>{p.total_goals}</td>
                  <td><span className="badge bg-success">{p.completed}</span></td>
                  <td><span className="badge bg-primary">{p.active}</span></td>
                  <td><span className="badge bg-warning text-dark">{p.on_hold}</span></td>
                  <td><span className="badge bg-danger">{p.discontinued}</span></td>
                  <td>
                    <ProgressBar value={p.avg_progress} color="primary" height={8} />
                    <span style={{ fontSize: 11 }}>{p.avg_progress}%</span>
                  </td>
                  <td>{p.sessions_completed}<span className="text-muted">/{p.sessions_planned}</span></td>
                  <td>
                    <span className={`badge bg-${p.completion_rate >= 50 ? 'success' : p.completion_rate >= 25 ? 'warning' : 'secondary'}`}>
                      {p.completion_rate}%
                    </span>
                  </td>
                  <td><span className="text-muted" style={{ fontSize: 11 }}>{selectedPat === p.patient_id ? '▲ hide' : '▼ show'}</span></td>
                </tr>
                {selectedPat === p.patient_id && (
                  <tr key={`${i}-detail`}>
                    <td colSpan={10}>
                      <PatientDetail bd={bd} patientId={p.patient_id} p={p} />
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PatientDetail({ bd, patientId, p }) {
  const plans = (bd?.recent_plans || []).filter(r => r.patient_id === patientId);
  return (
    <div className="bg-light p-3 rounded">
      <div className="fw-semibold mb-2">{patientId} — Goal Detail</div>
      <div className="row mb-2">
        <div className="col-md-3 small"><strong>Total goals:</strong> {p.total_goals}</div>
        <div className="col-md-3 small"><strong>Completed:</strong> {p.completed} ({p.completion_rate}%)</div>
        <div className="col-md-3 small"><strong>Sessions:</strong> {p.sessions_completed}/{p.sessions_planned}</div>
        <div className="col-md-3 small"><strong>Avg progress:</strong> {p.avg_progress}%</div>
      </div>
      {plans.length > 0 ? (
        <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
          <thead className="table-secondary">
            <tr>
              <th>Category</th>
              <th>Goal</th>
              <th>Status</th>
              <th>Progress</th>
              <th>Sessions</th>
              <th>Target Date</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {plans.map((r, j) => (
              <tr key={j}>
                <td><CatBadge cat={r.goal_category} label={r.category_label} /></td>
                <td style={{ maxWidth: 220, whiteSpace: 'normal' }}>{r.goal_description}</td>
                <td><StatusBadge status={r.status} /></td>
                <td>
                  <ProgressBar value={r.progress_pct} color={STATUS_COLOR[r.status] || 'primary'} height={8} />
                  <span style={{ fontSize: 10 }}>{r.progress_pct}%</span>
                </td>
                <td>{r.sessions_completed}/{r.sessions_planned}</td>
                <td style={{ fontSize: 11 }}>{r.target_date}</td>
                <td style={{ fontSize: 11, maxWidth: 180, whiteSpace: 'normal' }}>{r.therapist_notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="text-muted small">No recent plan records shown (only last 20 across all patients shown in Recent Plans tab).</div>
      )}
    </div>
  );
}

function RecentPlansPanel({ bd }) {
  const plans = bd?.recent_plans || [];
  return (
    <div>
      <p className="text-muted small mb-3">Most recently updated 20 rehab goal plans across all patients.</p>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle" style={{ fontSize: 12 }}>
          <thead className="table-light">
            <tr>
              <th>Patient</th>
              <th>Category</th>
              <th>Goal Description</th>
              <th>Status</th>
              <th>Progress</th>
              <th>Sessions</th>
              <th>Target Date</th>
              <th>Last Updated</th>
              <th>Therapist Notes</th>
            </tr>
          </thead>
          <tbody>
            {plans.map((r, i) => (
              <tr key={i}>
                <td className="fw-semibold">{r.patient_id}</td>
                <td><CatBadge cat={r.goal_category} label={r.category_label} /></td>
                <td style={{ maxWidth: 200, whiteSpace: 'normal' }}>{r.goal_description}</td>
                <td><StatusBadge status={r.status} /></td>
                <td>
                  <ProgressBar value={r.progress_pct} color={STATUS_COLOR[r.status] || 'primary'} height={8} />
                  <span style={{ fontSize: 10 }}>{r.progress_pct}%</span>
                </td>
                <td>{r.sessions_completed}/{r.sessions_planned} <span className="text-muted">({r.session_rate}%)</span></td>
                <td>{r.target_date}</td>
                <td style={{ fontSize: 11 }}>{r.last_updated?.slice(0, 10)}</td>
                <td style={{ fontSize: 11, maxWidth: 180, whiteSpace: 'normal' }}>{r.therapist_notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsPanel({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  const cats = defs.goal_categories || {};
  const statuses = defs.statuses || {};
  const metrics = defs.metrics || {};
  const epilepsy = defs.epilepsy_ot_context || {};
  const refs = defs.references || [];

  return (
    <div>
      <div className="alert alert-info mb-4">
        <strong>{defs.dashboard}</strong> — {defs.scope}
        <br /><small className="text-muted">{defs.data_source}</small>
      </div>

      <div className="row">
        <div className="col-md-6 mb-4">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Goal Categories</div>
            <div className="card-body">
              {Object.entries(cats).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <CatBadge cat={k} label={v.label} />
                  <span className="ms-2 small text-muted">{v.description}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-4">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Plan Statuses</div>
            <div className="card-body">
              {Object.entries(statuses).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <StatusBadge status={k} />
                  <span className="ms-2 small text-muted">{v}</span>
                </div>
              ))}
              <hr />
              <div className="small fw-semibold mb-1">Key Metrics</div>
              {Object.entries(metrics).map(([k, v]) => (
                <div key={k} className="mb-1 small">
                  <strong>{k}:</strong> <span className="text-muted">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Epilepsy OT Context</div>
        <div className="card-body">
          {Object.entries(epilepsy).map(([k, v]) => (
            <div key={k} className="mb-2">
              <strong className="small">{k.replace(/_/g, ' ').toUpperCase()}:</strong>
              <span className="ms-2 small text-muted">{v}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold">References</div>
        <ul className="list-group list-group-flush">
          {refs.map((r, i) => (
            <li key={i} className="list-group-item small">{r}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default function RehabGoalsPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/rehab-goals/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/rehab-goals/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/rehab-goals/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-0">Rehab Goal Tracking</h4>
        <p className="text-muted small mb-0">
          Occupational Therapy goal management — 311 plans · 30 patients · 6 categories · 4 statuses
        </p>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && <OverviewPanel ov={overview} />}
      {tab === 'categories' && <CategoriesPanel bd={breakdown} />}
      {tab === 'patients' && <PatientsPanel bd={breakdown} />}
      {tab === 'recent' && <RecentPlansPanel bd={breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel defs={definitions} />}
    </div>
  );
}
