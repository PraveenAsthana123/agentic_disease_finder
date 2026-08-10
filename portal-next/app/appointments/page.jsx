'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',   label: 'Overview' },
  { id: 'trends',     label: 'Trends & Patterns' },
  { id: 'providers',  label: 'Providers' },
  { id: 'breakdown',  label: 'Per Patient' },
  { id: 'definitions',label: 'Definitions' },
];

const STATUS_COLOR = {
  completed: 'success',
  booked: 'primary',
  confirmed: 'primary',
  'no-show': 'danger',
  cancelled: 'warning',
  rescheduled: 'secondary',
};
function statusColor(s) { return STATUS_COLOR[s] || 'secondary'; }

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

function BarChart({ data, labelKey, valueKey, color = 'primary', title, colorMap, suffix = '' }) {
  if (!data || !data.length) return null;
  const max = Math.max(...data.map(d => d[valueKey]));
  return (
    <div className="mb-4">
      {title && <div className="fw-semibold mb-2 small text-muted">{title}</div>}
      {data.map((d, i) => {
        const barColor = colorMap ? (colorMap[d[labelKey]] || color) : color;
        return (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span className="text-capitalize">{d[labelKey]}</span>
              <span className="fw-semibold">{d[valueKey]}{suffix}</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div
                className={`progress-bar bg-${barColor}`}
                style={{ width: `${max > 0 ? (d[valueKey] / max) * 100 : 0}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  if (ov.error) return <div className="alert alert-warning">{ov.error}</div>;

  const k = ov.kpi || {};
  const crPct = k.completion_rate_pct ?? 0;
  const nsPct = k.no_show_rate_pct ?? 0;

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Appointments" value={k.total_appointments} color="primary" sub="all statuses" />
        <KPI label="Completed" value={k.completed} color="success" sub="attended" />
        <KPI label="Upcoming" value={k.booked_upcoming} color="info" sub="booked / confirmed" />
        <KPI label="No-Shows" value={k.no_shows} color="danger" sub="missed" />
      </div>
      <div className="row mb-3">
        <KPI
          label="Completion Rate"
          value={`${crPct}%`}
          color={crPct >= 85 ? 'success' : crPct >= 70 ? 'warning' : 'danger'}
          sub="target ≥85%"
        />
        <KPI
          label="No-Show Rate"
          value={`${nsPct}%`}
          color={nsPct <= 10 ? 'success' : nsPct <= 15 ? 'warning' : 'danger'}
          sub="target <10%"
        />
        <KPI label="Avg Duration" value={`${k.avg_duration_min ?? '—'} min`} color="secondary" sub="completed appts" />
        <KPI label="Unique Patients" value={k.unique_patients} color="primary" sub="with appointments" />
      </div>

      <div className="row">
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold small">Status Distribution</div>
            <div className="card-body">
              <BarChart
                data={ov.status_distribution || []}
                labelKey="status"
                valueKey="count"
                colorMap={STATUS_COLOR}
              />
            </div>
          </div>
        </div>
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold small">By Department</div>
            <div className="card-body">
              <BarChart
                data={ov.department_distribution || []}
                labelKey="department"
                valueKey="count"
                color="info"
              />
            </div>
          </div>
        </div>
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold small">By Appointment Type</div>
            <div className="card-body">
              <BarChart
                data={ov.appointment_types || []}
                labelKey="type"
                valueKey="count"
                color="primary"
              />
            </div>
          </div>
        </div>
      </div>

      {(ov.provider_workload || []).length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold small">Provider Workload</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Provider</th>
                    <th>Total</th>
                    <th>Completed</th>
                    <th>No-Show</th>
                    <th>Completion %</th>
                  </tr>
                </thead>
                <tbody>
                  {ov.provider_workload.map((p, i) => {
                    const pct = p.total > 0 ? Math.round((p.completed / p.total) * 100) : 0;
                    return (
                      <tr key={i}>
                        <td className="fw-semibold">{p.provider}</td>
                        <td>{p.total}</td>
                        <td>{p.completed}</td>
                        <td>{p.no_show ?? '—'}</td>
                        <td>
                          <span className={`badge bg-${pct >= 85 ? 'success' : pct >= 70 ? 'warning' : 'danger'}`}>
                            {pct}%
                          </span>
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
    </div>
  );
}

function TrendsPanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  if (bd.error) return <div className="alert alert-warning">{bd.error}</div>;

  const daily = bd.daily_trend || [];
  const hourly = bd.hourly_pattern || [];
  const nsByType = bd.noshow_by_type || [];

  return (
    <div>
      {daily.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold small">Daily Trend — Total vs Completed vs No-Show</div>
          <div className="card-body p-0">
            <div className="table-responsive" style={{ maxHeight: 320 }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Date</th>
                    <th>Total</th>
                    <th>Completed</th>
                    <th>No-Show</th>
                  </tr>
                </thead>
                <tbody>
                  {daily.map((d, i) => (
                    <tr key={i}>
                      <td className="small">{d.date}</td>
                      <td>{d.total}</td>
                      <td><span className="badge bg-success">{d.completed}</span></td>
                      <td>
                        {d.no_show > 0
                          ? <span className="badge bg-danger">{d.no_show}</span>
                          : <span className="text-muted small">0</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold small">Hourly Pattern (appointments by hour)</div>
            <div className="card-body">
              {hourly.map((h, i) => {
                const max = Math.max(...hourly.map(x => x.count));
                const pct = max > 0 ? (h.count / max) * 100 : 0;
                return (
                  <div key={i} className="d-flex align-items-center mb-1">
                    <div className="text-end me-2 small text-muted" style={{ width: 45 }}>{h.hour}:00</div>
                    <div className="flex-grow-1">
                      <div className="progress" style={{ height: 14 }}>
                        <div className="progress-bar bg-info" style={{ width: `${pct}%` }}>
                          <span className="small px-1">{h.count}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold small">No-Show Rate by Appointment Type</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Type</th>
                    <th>Total</th>
                    <th>No-Show</th>
                    <th>Rate %</th>
                  </tr>
                </thead>
                <tbody>
                  {nsByType.map((t, i) => (
                    <tr key={i}>
                      <td className="small">{t.type}</td>
                      <td>{t.total}</td>
                      <td>{t.no_show}</td>
                      <td>
                        <span className={`badge bg-${t.rate_pct >= 15 ? 'danger' : t.rate_pct >= 10 ? 'warning' : 'success'}`}>
                          {t.rate_pct}%
                        </span>
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

function ProvidersPanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  if (bd.error) return <div className="alert alert-warning">{bd.error}</div>;

  const matrix = bd.provider_department_matrix || [];
  const recent = bd.recent_appointments || [];

  return (
    <div>
      {matrix.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold small">Provider × Department Matrix</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Provider</th>
                    <th>Department</th>
                    <th>Total</th>
                    <th>Completed</th>
                    <th>Completion %</th>
                  </tr>
                </thead>
                <tbody>
                  {matrix.map((m, i) => {
                    const pct = m.total > 0 ? Math.round((m.completed / m.total) * 100) : 0;
                    return (
                      <tr key={i}>
                        <td className="fw-semibold">{m.provider}</td>
                        <td>{m.department}</td>
                        <td>{m.total}</td>
                        <td>{m.completed}</td>
                        <td>
                          <span className={`badge bg-${pct >= 85 ? 'success' : pct >= 70 ? 'warning' : 'danger'}`}>
                            {pct}%
                          </span>
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

      {recent.length > 0 && (
        <div className="card">
          <div className="card-header fw-semibold small">Recent Appointments</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th>Provider</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Scheduled</th>
                    <th>Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.map((r, i) => (
                    <tr key={i}>
                      <td><span className="badge bg-secondary me-1">{r.patient_id}</span>{r.name}</td>
                      <td>{r.provider}</td>
                      <td className="small">{r.type}</td>
                      <td>
                        <span className={`badge bg-${statusColor(r.status)}`}>
                          {r.status}
                        </span>
                      </td>
                      <td className="small">{r.scheduled_for}</td>
                      <td className="small">{r.duration_min} min</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function BreakdownPanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  if (bd.error) return <div className="alert alert-warning">{bd.error}</div>;

  const patients = bd.patient_appointments || [];

  return (
    <div>
      <div className="card">
        <div className="card-header fw-semibold small">Per-Patient Appointment Summary ({patients.length} patients)</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Patient ID</th>
                  <th>Name</th>
                  <th>Total</th>
                  <th>Completed</th>
                  <th>No-Show</th>
                  <th>Completion %</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => {
                  const pct = p.total > 0 ? Math.round((p.completed / p.total) * 100) : 0;
                  return (
                    <tr key={i}>
                      <td><span className="badge bg-secondary">{p.patient_id}</span></td>
                      <td>{p.name}</td>
                      <td>{p.total}</td>
                      <td>{p.completed}</td>
                      <td>
                        {p.no_show > 0
                          ? <span className="badge bg-danger">{p.no_show}</span>
                          : <span className="text-muted small">0</span>}
                      </td>
                      <td>
                        <span className={`badge bg-${pct >= 85 ? 'success' : pct >= 70 ? 'warning' : 'danger'}`}>
                          {pct}%
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  if (defs.error) return <div className="alert alert-warning">{defs.error}</div>;

  const entries = Object.entries(defs.definitions || {});

  return (
    <div className="card">
      <div className="card-header fw-semibold small">Metric Definitions</div>
      <div className="card-body p-0">
        <table className="table table-sm mb-0">
          <thead className="table-light">
            <tr><th>Metric</th><th>Definition</th></tr>
          </thead>
          <tbody>
            {entries.map(([k, v], i) => (
              <tr key={i}>
                <td className="fw-semibold text-nowrap small">{k.replace(/_/g, ' ')}</td>
                <td className="small">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function AppointmentsPage() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv]   = useState(null);
  const [bd, setBd]   = useState(null);
  const [defs, setDefs] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/appointments/overview`).then(r => r.json()).then(setOv).catch(() => setOv({ error: 'Failed to load overview' }));
    fetch(`${API}/api/appointments/breakdown`).then(r => r.json()).then(setBd).catch(() => setBd({ error: 'Failed to load breakdown' }));
    fetch(`${API}/api/appointments/definitions`).then(r => r.json()).then(setDefs).catch(() => setDefs({ error: 'Failed to load definitions' }));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: '1.6rem' }}>&#x1f4cb;</span>
        <div>
          <h4 className="mb-0 fw-bold">Clinic Appointments Dashboard</h4>
          <div className="text-muted small">
            Provider workload · department distribution · no-show analysis · daily trend — 120 appointments, 34 patients
          </div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel ov={ov} />}
      {tab === 'trends'      && <TrendsPanel bd={bd} />}
      {tab === 'providers'   && <ProvidersPanel bd={bd} />}
      {tab === 'breakdown'   && <BreakdownPanel bd={bd} />}
      {tab === 'definitions' && <DefinitionsPanel defs={defs} />}
    </div>
  );
}
