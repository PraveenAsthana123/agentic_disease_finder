'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const burnoutColor = tier => ({ Low: 'success', Moderate: 'info', High: 'warning', Critical: 'danger' }[tier] || 'secondary');
const stressColor = tier => ({ Low: 'success', Moderate: 'warning', High: 'danger' }[tier] || 'secondary');
const outcomeColor = o => ({
  'caregiver-responded': 'success',
  'resolved-home': 'info',
  'false-alarm': 'secondary',
  'ems-dispatched': 'warning',
  'er-visit': 'danger',
}[o] || 'secondary');
const bool2badge = v => v
  ? <span className="badge bg-success">Yes</span>
  : <span className="badge bg-danger">No</span>;

const KPI = ({ label, value, sub, color = 'primary' }) => (
  <div className="col-6 col-md-3 mb-3">
    <div className={`card border-${color} h-100`}>
      <div className="card-body p-3 text-center">
        <div className={`fs-3 fw-bold text-${color}`}>{value}</div>
        <div className="small text-muted">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: '0.72rem' }}>{sub}</div>}
      </div>
    </div>
  </div>
);

export default function CaregiverAppDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('burnout_score');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/caregiver-app/overview`).then(r => r.json()),
      fetch(`${API}/api/caregiver-app/breakdown`).then(r => r.json()),
      fetch(`${API}/api/caregiver-app/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading Caregiver App data…</div>;

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'caregivers', label: '🤝 Caregivers' },
    { id: 'sos', label: '🚨 SOS Alerts' },
    { id: 'contacts', label: '📞 Emergency Contacts' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  // filtered/sorted caregiver table
  const cgRows = (bd?.caregiver_table || [])
    .filter(c => !search || Object.values(c).join(' ').toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === 'burnout_score') return (b.burnout_score ?? 0) - (a.burnout_score ?? 0);
      if (sortBy === 'stress') return (b.stress ?? 0) - (a.stress ?? 0);
      if (sortBy === 'name') return (a.name || '').localeCompare(b.name || '');
      return 0;
    });

  return (
    <div className="container-fluid py-3">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <div>
          <h3 className="mb-0">📱 Caregiver App Dashboard</h3>
          <small className="text-muted">
            {ov.total_caregivers} caregivers · {ov.total_sos_events} SOS events ·{' '}
            {ov.total_emergency_contacts} emergency contacts
          </small>
        </div>
        <span className="badge bg-primary fs-6">Companion App</span>
      </div>

      {/* Tabs */}
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

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <>
          {/* KPIs */}
          <div className="row g-3 mb-3">
            <KPI label="Caregivers Enrolled" value={ov.total_caregivers} color="primary" />
            <KPI label="Avg Burnout Score" value={ov.avg_burnout_score} sub="/100" color="warning" />
            <KPI label="High-Burnout Caregivers" value={ov.high_burnout_count}
              sub={`${ov.high_burnout_pct}% of team`} color="danger" />
            <KPI label="Training Completion" value={`${ov.training_completion_pct}%`}
              sub="Epilepsy training done" color="success" />
            <KPI label="First Aid Certified" value={`${ov.first_aid_certified_pct}%`} color="info" />
            <KPI label="Rescue Med Trained" value={`${ov.rescue_med_trained_pct}%`} color="success" />
            <KPI label="SOS Events" value={ov.total_sos_events} color="danger" />
            <KPI label="Avg SOS Response" value={`${Math.round(ov.sos_avg_response_sec)}s`}
              sub={`${ov.sos_fast_response_pct}% under 2 min`} color="warning" />
          </div>

          <div className="row g-3">
            {/* Burnout Tier Distribution */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header">Burnout Tier Distribution</div>
                <div className="card-body">
                  {Object.entries(ov.burnout_tier_distribution || {}).map(([tier, cnt]) => (
                    <div key={tier} className="mb-2">
                      <div className="d-flex justify-content-between mb-1">
                        <span className={`badge bg-${burnoutColor(tier)}`}>{tier}</span>
                        <span>{cnt} ({Math.round(cnt / ov.total_caregivers * 100)}%)</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${burnoutColor(tier)}`}
                          style={{ width: `${cnt / ov.total_caregivers * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Role Distribution */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header">Caregiver Roles</div>
                <div className="card-body">
                  {Object.entries(ov.role_distribution || {}).sort((a, b) => b[1] - a[1]).map(([role, cnt]) => (
                    <div key={role} className="mb-2">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="text-capitalize">{role}</span>
                        <span>{cnt}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className="progress-bar bg-primary"
                          style={{ width: `${cnt / ov.total_caregivers * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Availability */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header">Availability</div>
                <div className="card-body">
                  {Object.entries(ov.availability_distribution || {}).sort((a, b) => b[1] - a[1]).map(([av, cnt]) => (
                    <div key={av} className="mb-2">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="text-capitalize">{av.replace('-', ' ')}</span>
                        <span>{cnt}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className="progress-bar bg-info"
                          style={{ width: `${cnt / ov.total_caregivers * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* SOS Outcome Distribution */}
            <div className="col-md-6">
              <div className="card">
                <div className="card-header">SOS Outcomes</div>
                <div className="card-body">
                  {Object.entries(ov.sos_outcome_distribution || {}).sort((a, b) => b[1] - a[1]).map(([out, cnt]) => (
                    <div key={out} className="d-flex align-items-center mb-2 gap-2">
                      <span className={`badge bg-${outcomeColor(out)}`} style={{ minWidth: 130 }}>
                        {out.replace(/-/g, ' ')}
                      </span>
                      <div className="progress flex-grow-1" style={{ height: 14 }}>
                        <div
                          className={`progress-bar bg-${outcomeColor(out)}`}
                          style={{ width: `${cnt / ov.total_sos_events * 100}%` }}
                        />
                      </div>
                      <span className="small">{cnt}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Monthly SOS Trend */}
            <div className="col-md-6">
              <div className="card">
                <div className="card-header">Monthly SOS Trend</div>
                <div className="card-body">
                  {(ov.sos_monthly_trend || []).map(m => (
                    <div key={m.month} className="d-flex align-items-center mb-2 gap-2">
                      <span className="small text-muted" style={{ minWidth: 60 }}>{m.month}</span>
                      <div className="progress flex-grow-1" style={{ height: 16 }}>
                        <div
                          className="progress-bar bg-danger"
                          style={{ width: `${m.count / Math.max(...(ov.sos_monthly_trend || [{ count: 1 }]).map(x => x.count)) * 100}%` }}
                        />
                      </div>
                      <span className="small fw-bold">{m.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── Caregivers Table ── */}
      {tab === 'caregivers' && (
        <>
          <div className="d-flex gap-2 mb-3">
            <input
              className="form-control"
              style={{ maxWidth: 260 }}
              placeholder="Search caregivers…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            <select className="form-select" style={{ maxWidth: 180 }} value={sortBy} onChange={e => setSortBy(e.target.value)}>
              <option value="burnout_score">Sort: Burnout ↓</option>
              <option value="stress">Sort: Stress ↓</option>
              <option value="name">Sort: Name</option>
            </select>
            <span className="align-self-center text-muted small">{cgRows.length} caregivers</span>
          </div>

          {/* High-burnout alert */}
          {(bd?.high_burnout_caregivers?.length > 0) && (
            <div className="alert alert-danger py-2 mb-3">
              <strong>⚠ {bd.high_burnout_caregivers.length} caregivers</strong> have Critical burnout (≥75) — urgent support recommended.
            </div>
          )}

          {/* Training gaps */}
          {bd?.training_gaps && (
            <div className="row g-2 mb-3">
              {Object.entries(bd.training_gaps).map(([gap, cnt]) => cnt > 0 && (
                <div key={gap} className="col-auto">
                  <span className="badge bg-warning text-dark">
                    {cnt} {gap.replace(/_/g, ' ')}
                  </span>
                </div>
              ))}
            </div>
          )}

          <div className="table-responsive">
            <table className="table table-sm table-hover table-striped">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th><th>Name</th><th>Role</th><th>Availability</th>
                  <th>Exp (yrs)</th><th>Training</th><th>First Aid</th><th>Rescue Med</th>
                  <th>Confidence</th><th>Stress</th><th>Burnout</th><th>Burnout Tier</th>
                  <th>Safety Plan</th><th>Action Plan</th>
                </tr>
              </thead>
              <tbody>
                {cgRows.map((c, i) => (
                  <tr key={i}>
                    <td><code>{c.patient_id}</code></td>
                    <td>{c.name}</td>
                    <td className="text-capitalize">{c.role}</td>
                    <td className="text-capitalize">{(c.availability || '').replace('-', ' ')}</td>
                    <td>{c.experience_years}</td>
                    <td>{bool2badge(c.epilepsy_training)}</td>
                    <td>{bool2badge(c.first_aid_certified)}</td>
                    <td>{bool2badge(c.rescue_med_trained)}</td>
                    <td>{c.confidence}/10</td>
                    <td>
                      <span className={`badge bg-${stressColor(c.stress_tier)}`}>
                        {c.stress} ({c.stress_tier})
                      </span>
                    </td>
                    <td>{c.burnout_score}</td>
                    <td>
                      <span className={`badge bg-${burnoutColor(c.burnout_tier)}`}>
                        {c.burnout_tier}
                      </span>
                    </td>
                    <td>{bool2badge(c.safety_plan)}</td>
                    <td>{bool2badge(c.action_plan)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── SOS Alerts ── */}
      {tab === 'sos' && (
        <>
          <div className="row g-3 mb-3">
            <KPI label="Total SOS Events" value={ov.total_sos_events} color="danger" />
            <KPI label="Avg Response Time" value={`${Math.round(ov.sos_avg_response_sec)}s`} color="warning" />
            <KPI label="Fast Response (<2 min)" value={`${ov.sos_fast_response_pct}%`} color="success" />
            <KPI label="Notified Rate" value={`${ov.sos_notified_pct}%`} color="info" />
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-striped">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th><th>Date</th><th>Event Type</th><th>Trigger</th>
                  <th>Response (s)</th><th>Notified</th><th>Location Shared</th><th>Outcome</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.sos_alert_log || []).map((s, i) => (
                  <tr key={i}>
                    <td><code>{s.patient_id}</code></td>
                    <td>{s.date}</td>
                    <td>{s.event_type?.replace(/-/g, ' ')}</td>
                    <td className="small text-muted">{s.trigger_method?.replace(/-/g, ' ')}</td>
                    <td className={s.response_time_sec <= 120 ? 'text-success fw-bold' : ''}>
                      {s.response_time_sec ?? '—'}
                    </td>
                    <td>{bool2badge(s.notified)}</td>
                    <td>{bool2badge(s.location_shared)}</td>
                    <td>
                      <span className={`badge bg-${outcomeColor(s.outcome)}`}>
                        {(s.outcome || '').replace(/-/g, ' ')}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── Emergency Contacts ── */}
      {tab === 'contacts' && (
        <>
          <div className="row g-3 mb-3">
            <KPI label="Emergency Contacts" value={ov.total_emergency_contacts} color="primary" />
            <KPI label="Notify on Seizure" value={`${ov.contact_notify_on_seizure_pct}%`} color="warning" />
          </div>

          {/* Relationship distribution */}
          <div className="card mb-3">
            <div className="card-header">Contact Relationship Distribution</div>
            <div className="card-body">
              <div className="row">
                {Object.entries(ov.contact_relationship_distribution || {}).sort((a, b) => b[1] - a[1]).map(([rel, cnt]) => (
                  <div key={rel} className="col-auto mb-2">
                    <span className="badge bg-secondary me-1">{cnt}</span>
                    <span className="text-capitalize">{rel}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-hover table-striped">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th><th>Contact Name</th><th>Relationship</th>
                  <th>Primary</th><th>Notify on Seizure</th><th>Last Verified</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.emergency_contacts || []).map((c, i) => (
                  <tr key={i}>
                    <td><code>{c.patient_id}</code></td>
                    <td>{c.name}</td>
                    <td className="text-capitalize">{c.relationship}</td>
                    <td>{c.is_primary ? <span className="badge bg-primary">Primary</span> : '—'}</td>
                    <td>{bool2badge(c.notify_on_seizure)}</td>
                    <td className="text-muted small">{c.last_verified || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && defs && (
        <>
          <div className="card mb-3">
            <div className="card-header fw-bold">About the Caregiver App</div>
            <div className="card-body">{defs.description}</div>
          </div>

          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header">App Features</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Feature</th><th>Status</th></tr>
                    </thead>
                    <tbody>
                      {(defs.app_features || []).map((f, i) => (
                        <tr key={i}>
                          <td>
                            <strong>{f.feature}</strong>
                            <div className="text-muted small">{f.description}</div>
                          </td>
                          <td>
                            <span className={`badge bg-${f.status === 'active' ? 'success' : 'secondary'}`}>
                              {f.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="col-md-6">
              <div className="card mb-3">
                <div className="card-header">Alert Types</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light"><tr><th>Type</th><th>Description</th></tr></thead>
                    <tbody>
                      {(defs.alert_types || []).map((a, i) => (
                        <tr key={i}><td><code>{a.type}</code></td><td className="small">{a.description}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="card">
                <div className="card-header">SOS Outcomes</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light"><tr><th>Outcome</th><th>Description</th></tr></thead>
                    <tbody>
                      {(defs.sos_outcomes || []).map((o, i) => (
                        <tr key={i}>
                          <td><span className={`badge bg-${outcomeColor(o.outcome)}`}>{o.outcome.replace(/-/g, ' ')}</span></td>
                          <td className="small">{o.description}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Burnout scale */}
          <div className="card mb-3">
            <div className="card-header">{defs.burnout_scale?.name}</div>
            <div className="card-body">
              <div className="row g-2">
                {(defs.burnout_scale?.thresholds || []).map((t, i) => (
                  <div key={i} className="col-md-3">
                    <div className={`card border-${burnoutColor(t.tier)}`}>
                      <div className="card-body p-2 text-center">
                        <span className={`badge bg-${burnoutColor(t.tier)} mb-1`}>{t.tier}</span>
                        <div className="small fw-bold">{t.range}</div>
                        <div className="text-muted" style={{ fontSize: '0.72rem' }}>{t.action}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* References */}
          <div className="card">
            <div className="card-header">Clinical References</div>
            <div className="card-body">
              <ul className="mb-0">
                {(defs.references || []).map((r, i) => <li key={i} className="small">{r}</li>)}
              </ul>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
