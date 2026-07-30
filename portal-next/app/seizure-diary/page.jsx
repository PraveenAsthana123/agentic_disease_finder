'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Per-Patient' },
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

function DistBar({ items, colorFn }) {
  const total = Object.values(items || {}).reduce((a, b) => a + b, 0);
  return (
    <table className="table table-sm mb-0">
      <tbody>
        {Object.entries(items || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => {
          const pct = total > 0 ? ((v / total) * 100).toFixed(1) : 0;
          const color = colorFn ? colorFn(k) : 'primary';
          return (
            <tr key={k}>
              <td className="text-nowrap small fw-semibold" style={{ width: '40%' }}>{k.replace(/_/g, ' ')}</td>
              <td style={{ width: '45%' }}>
                <div className="progress" style={{ height: 10 }}>
                  <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
                </div>
              </td>
              <td className="small text-end">{v} <span className="text-muted">({pct}%)</span></td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function SeverityBadge({ severity }) {
  const map = { Critical: 'danger', Severe: 'danger', Moderate: 'warning', Mild: 'success' };
  return <span className={`badge bg-${map[severity] || 'secondary'}`}>{severity}</span>;
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const sevColor = s => ({ Severe: 'danger', Moderate: 'warning', Mild: 'success', Critical: 'dark' }[s] || 'secondary');
  const trigColor = () => 'info';

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Seizure Events" value={data.total_events} color="primary" />
        <KPI label="Unique Patients" value={data.unique_patients} color="info" sub="with diary entries" />
        <KPI label="Avg Duration" value={data.avg_duration_sec != null ? `${data.avg_duration_sec}s` : '—'} color="warning" sub="seconds per event" />
        <KPI label="ER Visits" value={data.er_visits} color="danger" sub={data.er_rate_pct != null ? `${data.er_rate_pct}% of events` : ''} />
      </div>
      <div className="row mb-4">
        <KPI label="Injury Count" value={data.injury_count} color={data.injury_count > 0 ? 'danger' : 'success'} sub="events with injury" />
        <KPI label="ER Rate" value={data.er_rate_pct != null ? `${data.er_rate_pct}%` : '—'} color={data.er_rate_pct > 20 ? 'danger' : 'success'} sub="% events → ER" />
      </div>

      <div className="row mb-3">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Severity Distribution</div>
            <div className="card-body p-2">
              <DistBar items={data.severity_distribution} colorFn={sevColor} />
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Top Triggers</div>
            <div className="card-body p-2">
              <DistBar items={data.top_triggers} colorFn={trigColor} />
            </div>
          </div>
        </div>
      </div>

      {(data.monthly_trend || []).length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Monthly Trend</div>
          <div className="card-body p-2">
            <table className="table table-sm table-striped mb-0">
              <thead><tr><th>Month</th><th className="text-end">Events</th><th className="text-end">ER Visits</th></tr></thead>
              <tbody>
                {data.monthly_trend.map(m => (
                  <tr key={m.month}>
                    <td>{m.month}</td>
                    <td className="text-end fw-bold">{m.events}</td>
                    <td className="text-end"><span className={m.er_visits > 0 ? 'text-danger fw-bold' : ''}>{m.er_visits}</span></td>
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

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const patients = data.patient_summary || [];
  const events = data.event_log || [];

  return (
    <div>
      <div className="card mb-3">
        <div className="card-header fw-semibold">Patient Summary ({patients.length})</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped mb-0">
              <thead>
                <tr>
                  <th>Patient</th>
                  <th className="text-end">Events</th>
                  <th className="text-end">Avg Duration</th>
                  <th className="text-end">ER Visits</th>
                  <th className="text-end">Injuries</th>
                  <th>Worst</th>
                  <th>Last Event</th>
                </tr>
              </thead>
              <tbody>
                {patients.map(p => (
                  <tr key={p.patient_id}>
                    <td className="fw-semibold">{p.patient_id}</td>
                    <td className="text-end">{p.total_events}</td>
                    <td className="text-end">{p.avg_duration_sec != null ? `${p.avg_duration_sec}s` : '—'}</td>
                    <td className="text-end">{p.er_visits > 0 ? <span className="text-danger fw-bold">{p.er_visits}</span> : 0}</td>
                    <td className="text-end">{p.injuries > 0 ? <span className="text-danger fw-bold">{p.injuries}</span> : 0}</td>
                    <td><SeverityBadge severity={p.worst_severity} /></td>
                    <td className="small text-muted">{p.last_event}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {events.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Event Log ({events.length})</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0">
                <thead>
                  <tr>
                    <th>Patient</th>
                    <th>Date</th>
                    <th>Type</th>
                    <th>Duration</th>
                    <th>Severity</th>
                    <th>Trigger</th>
                    <th>ER</th>
                    <th>Injury</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((e, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{e.patient_id}</td>
                      <td className="small">{e.date || e.event_date}</td>
                      <td className="small">{e.seizure_type || e.type || '—'}</td>
                      <td className="text-end">{e.duration_sec != null ? `${e.duration_sec}s` : '—'}</td>
                      <td><SeverityBadge severity={e.severity} /></td>
                      <td className="small">{e.trigger || '—'}</td>
                      <td>{e.er_visit ? <span className="badge bg-danger">Yes</span> : <span className="badge bg-light text-dark">No</span>}</td>
                      <td>{e.injury ? <span className="badge bg-danger">Yes</span> : <span className="badge bg-light text-dark">No</span>}</td>
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

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div className="card">
      <div className="card-header fw-semibold">{data.title || 'Seizure Diary Dashboard'} — Metric Definitions</div>
      <div className="card-body p-0">
        <table className="table table-sm table-striped mb-0">
          <thead><tr><th>Metric</th><th>Description</th></tr></thead>
          <tbody>
            {(data.metrics || []).map((m, i) => (
              <tr key={i}>
                <td className="fw-semibold text-nowrap">{m.name}</td>
                <td className="small">{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.description && <div className="card-footer text-muted small">{data.description}</div>}
    </div>
  );
}

export default function SeizureDiaryDashboard() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/seizure-diary-dashboard/overview`).then(r => r.json()).then(setOverview).catch(() => setOverview({ error: 'Failed to load overview' }));
    fetch(`${API}/api/seizure-diary-dashboard/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setBreakdown({ error: 'Failed to load breakdown' }));
    fetch(`${API}/api/seizure-diary-dashboard/definitions`).then(r => r.json()).then(setDefinitions).catch(() => setDefinitions({ error: 'Failed to load definitions' }));
  }, []);

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-3">📓 Seizure Diary Dashboard</h4>
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>
      {tab === 'overview' && <OverviewPanel data={overview} />}
      {tab === 'breakdown' && <BreakdownPanel data={breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={definitions} />}
    </div>
  );
}
