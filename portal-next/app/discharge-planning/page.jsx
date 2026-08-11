'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TIER_COLOR = { Ready: 'success', Conditional: 'warning', 'Not Ready': 'danger' };
const TYPE_COLOR = { emergency: 'danger', planned: 'primary', observation: 'info' };
const DISP_COLOR = { home: 'success', rehabilitation: 'info', transferred: 'warning', ama: 'danger' };

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-2">
      <div className="card shadow-sm border-0 h-100">
        <div className="card-body text-center py-2">
          <div className={`h4 mb-0 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: 10 }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ items, labelKey, valueKey, colorKey }) {
  if (!items || !items.length) return null;
  const mx = Math.max(...items.map(i => i[valueKey] || 0));
  return (
    <div>
      {items.map((item, idx) => (
        <div key={idx} className="d-flex align-items-center mb-1">
          <div className="text-end small me-2" style={{ width: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {item[labelKey]}
          </div>
          <div className="flex-grow-1">
            <div className="progress" style={{ height: 20 }}>
              <div
                className={`progress-bar bg-${item.color || colorKey || 'primary'}`}
                style={{ width: `${mx ? ((item[valueKey] / mx) * 100) : 0}%` }}
              >
                <span className="small">{item[valueKey]}</span>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function CheckBadge({ met }) {
  return met
    ? <span className="badge bg-success">✓ Met</span>
    : <span className="badge bg-danger">✗ Not Met</span>;
}

export default function DischargePlanningDashboard() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/discharge-planning/overview`).then(r => r.json()),
      fetch(`${API}/api/discharge-planning/breakdown`).then(r => r.json()),
      fetch(`${API}/api/discharge-planning/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefs(df);
    }).catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;
  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const k = overview.kpis || {};
  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'readiness', label: 'Readiness' },
    { id: 'admissions', label: 'Admissions Log' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <h3>🏥 Discharge Planning Dashboard</h3>
      <p className="text-muted small mb-3">
        115 admissions · 30 patients · real hospitalization + adherence + appointments data
      </p>

      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Total Admissions" value={k.total_admissions} color="primary" />
        <KPI label="Unique Patients" value={k.unique_patients} color="info" />
        <KPI label="Avg LOS" value={k.avg_length_of_stay_days ? `${k.avg_length_of_stay_days}d` : '—'} color="secondary" />
        <KPI label="30-Day Readmissions" value={k.readmissions_30d}
          color={k.readmission_rate_pct > 10 ? 'danger' : 'success'}
          sub={`${k.readmission_rate_pct}% rate (target <10%)`} />
        <KPI label="Seizure-Free at Dc." value={`${k.seizure_free_at_discharge_pct}%`} color="success" />
        <KPI label="Complication Rate" value={`${k.complication_pct}%`}
          color={k.complication_pct > 50 ? 'warning' : 'success'} />
        <KPI label="Avg Cost (USD)" value={k.avg_cost_usd ? `$${k.avg_cost_usd.toLocaleString()}` : '—'} color="secondary" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active fw-semibold' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Admission reasons */}
          <div className="col-md-6">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Top Admission Reasons</div>
              <div className="card-body">
                <Bar items={overview.admission_reason_distribution} labelKey="reason" valueKey="count" colorKey="primary" />
              </div>
            </div>
          </div>

          {/* Disposition */}
          <div className="col-md-6">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Discharge Disposition</div>
              <div className="card-body">
                {(overview.discharge_disposition_distribution || []).map((d, i) => (
                  <div key={i} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${DISP_COLOR[d.disposition] || 'secondary'}`}>{d.disposition}</span>
                    <span className="fw-bold">{d.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Admission types */}
          <div className="col-md-4">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Admission Types</div>
              <div className="card-body">
                {(overview.admission_type_distribution || []).map((a, i) => (
                  <div key={i} className="d-flex justify-content-between mb-2">
                    <span className={`badge bg-${TYPE_COLOR[a.type] || 'secondary'}`}>{a.type}</span>
                    <span className="fw-bold">{a.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Ward */}
          <div className="col-md-4">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Ward Distribution</div>
              <div className="card-body">
                {(overview.ward_distribution || []).map((w, i) => (
                  <div key={i} className="d-flex justify-content-between mb-1 small">
                    <span>{w.ward}</span>
                    <span className="badge bg-info text-dark">{w.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Physician workload */}
          <div className="col-md-4">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Physician Workload</div>
              <div className="card-body">
                {(overview.physician_workload || []).map((p, i) => (
                  <div key={i} className="d-flex justify-content-between mb-1 small">
                    <span>{p.physician}</span>
                    <span className="badge bg-primary">{p.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Readiness distribution */}
          <div className="col-md-6">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Discharge Readiness (Last Admission per Patient)</div>
              <div className="card-body">
                {(overview.readiness_distribution || []).map((r, i) => (
                  <div key={i} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${TIER_COLOR[r.tier] || 'secondary'} fs-6`}>{r.tier}</span>
                    <span className="fw-bold">{r.count} patients</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Monthly trend */}
          <div className="col-md-6">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header fw-semibold">Monthly Admissions Trend</div>
              <div className="card-body">
                <Bar items={(overview.monthly_trend || []).slice(-12)} labelKey="month" valueKey="admissions" colorKey="primary" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── READINESS ── */}
      {tab === 'readiness' && breakdown && (
        <div>
          <h5 className="mb-3">Per-Patient Discharge Readiness Profiles ({(breakdown.patient_discharge_profiles || []).length} patients)</h5>
          <div className="table-responsive">
            <table className="table table-hover table-sm">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Score</th>
                  <th>Tier</th>
                  <th>Last Discharge</th>
                  <th>Reason</th>
                  <th>Disposition</th>
                  <th>Seizure-Free</th>
                  <th>Adherence</th>
                  <th>Next Appt</th>
                  <th>Readmit</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.patient_discharge_profiles || []).map(p => (
                  <>
                    <tr key={p.patient_id}>
                      <td><code>{p.patient_id}</code></td>
                      <td>
                        <div className="progress" style={{ height: 18, minWidth: 60 }}>
                          <div
                            className={`progress-bar bg-${TIER_COLOR[p.readiness_tier] || 'secondary'}`}
                            style={{ width: `${p.readiness_score}%` }}
                          >
                            <span className="small">{p.readiness_score}</span>
                          </div>
                        </div>
                      </td>
                      <td><span className={`badge bg-${TIER_COLOR[p.readiness_tier] || 'secondary'}`}>{p.readiness_tier}</span></td>
                      <td className="small">{p.last_discharge_date || '—'}</td>
                      <td className="small">{(p.admission_reason || '').replace(/_/g, ' ')}</td>
                      <td><span className={`badge bg-${DISP_COLOR[p.disposition] || 'secondary'}`}>{p.disposition || '—'}</span></td>
                      <td>{p.seizure_free
                        ? <span className="badge bg-success">Yes</span>
                        : <span className="badge bg-danger">No</span>}
                      </td>
                      <td className="small">{p.avg_adherence_pct != null ? `${p.avg_adherence_pct}%` : '—'}</td>
                      <td className="small">{p.next_appointment || <span className="text-danger">None</span>}</td>
                      <td>{p.readmission_within_30d
                        ? <span className="badge bg-danger">Yes</span>
                        : <span className="badge bg-success">No</span>}
                      </td>
                      <td>
                        <button
                          className="btn btn-sm btn-outline-secondary"
                          onClick={() => setExpandedPt(expandedPt === p.patient_id ? null : p.patient_id)}
                        >
                          {expandedPt === p.patient_id ? 'Hide' : 'Criteria'}
                        </button>
                      </td>
                    </tr>
                    {expandedPt === p.patient_id && (
                      <tr>
                        <td colSpan={11} className="bg-light">
                          <div className="p-2">
                            <strong>Readiness Checklist — {p.patient_id}</strong>
                            <table className="table table-sm mt-2 mb-0">
                              <thead><tr><th>Criterion</th><th>Status</th><th>Note</th><th>Weight</th></tr></thead>
                              <tbody>
                                {(p.readiness_checks || []).map((c, i) => (
                                  <tr key={i}>
                                    <td className="small">{c.criterion}</td>
                                    <td><CheckBadge met={c.met} /></td>
                                    <td className="small text-muted">{c.note}</td>
                                    <td className="small">{c.weight} pts</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>

          {/* Readmissions detail */}
          {(breakdown.readmission_detail || []).length > 0 && (
            <div className="mt-4">
              <h5 className="text-danger">30-Day Readmissions ({breakdown.readmission_detail.length})</h5>
              <div className="table-responsive">
                <table className="table table-sm table-danger">
                  <thead className="table-dark">
                    <tr><th>Patient</th><th>Admission</th><th>Discharge</th><th>Reason</th><th>Ward</th><th>Physician</th><th>Complications</th></tr>
                  </thead>
                  <tbody>
                    {breakdown.readmission_detail.map((r, i) => (
                      <tr key={i}>
                        <td><code>{r.patient_id}</code></td>
                        <td className="small">{r.admission_date}</td>
                        <td className="small">{r.discharge_date || '—'}</td>
                        <td className="small">{(r.reason || '').replace(/_/g, ' ')}</td>
                        <td className="small">{r.ward}</td>
                        <td className="small">{r.physician}</td>
                        <td className="small text-warning">{r.complications || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── ADMISSIONS LOG ── */}
      {tab === 'admissions' && breakdown && (
        <div>
          <h5>Recent Admissions (last 50)</h5>
          <div className="table-responsive">
            <table className="table table-hover table-sm">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th><th>Admitted</th><th>Discharged</th><th>Type</th>
                  <th>Reason</th><th>Ward</th><th>LOS</th><th>Disposition</th>
                  <th>Sz-Free</th><th>Cost</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.admissions_log || []).map((a, i) => (
                  <tr key={i}>
                    <td><code>{a.patient_id}</code></td>
                    <td className="small">{a.admission_date}</td>
                    <td className="small">{a.discharge_date || '—'}</td>
                    <td><span className={`badge bg-${TYPE_COLOR[a.type] || 'secondary'}`}>{a.type}</span></td>
                    <td className="small">{(a.reason || '').replace(/_/g, ' ')}</td>
                    <td className="small">{a.ward}</td>
                    <td className="small">{a.los_days != null ? `${a.los_days}d` : '—'}</td>
                    <td><span className={`badge bg-${DISP_COLOR[a.disposition] || 'secondary'}`}>{a.disposition || '—'}</span></td>
                    <td>{a.seizure_free
                      ? <span className="badge bg-success">Yes</span>
                      : <span className="badge bg-danger">No</span>}
                    </td>
                    <td className="small">{a.cost_usd ? `$${a.cost_usd.toLocaleString()}` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div>
          <p className="text-muted mb-3">{defs.description}</p>

          <h5>Readiness Tiers</h5>
          <div className="row mb-4">
            {(defs.readiness_tiers || []).map((t, i) => (
              <div key={i} className="col-md-4 mb-2">
                <div className={`card border-${TIER_COLOR[t.tier] || 'secondary'} shadow-sm`}>
                  <div className="card-header">
                    <span className={`badge bg-${TIER_COLOR[t.tier] || 'secondary'} me-2`}>{t.tier}</span>
                    Score {t.score_range}
                  </div>
                  <div className="card-body small text-muted">{t.description}</div>
                </div>
              </div>
            ))}
          </div>

          <h5>Readiness Criteria (total 100 pts)</h5>
          <table className="table table-sm table-bordered mb-4">
            <thead className="table-dark">
              <tr><th>Criterion</th><th>Weight</th><th>Clinical Rationale</th></tr>
            </thead>
            <tbody>
              {(defs.readiness_criteria || []).map((c, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{c.criterion}</td>
                  <td className="small"><span className="badge bg-primary">{c.weight} pts</span></td>
                  <td className="small text-muted">{c.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="row">
            <div className="col-md-6">
              <h5>Admission Types</h5>
              {Object.entries(defs.admission_types || {}).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <span className={`badge bg-${TYPE_COLOR[k] || 'secondary'} me-2`}>{k}</span>
                  <span className="small text-muted">{v}</span>
                </div>
              ))}
            </div>
            <div className="col-md-6">
              <h5>Discharge Dispositions</h5>
              {Object.entries(defs.discharge_dispositions || {}).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <span className={`badge bg-${DISP_COLOR[k] || 'secondary'} me-2`}>{k}</span>
                  <span className="small text-muted">{v}</span>
                </div>
              ))}
            </div>
          </div>

          <h5 className="mt-3">Data Sources</h5>
          <ul className="list-group list-group-flush mb-3">
            {(defs.data_sources || []).map((s, i) => (
              <li key={i} className="list-group-item small">{s}</li>
            ))}
          </ul>

          <h5>Standards</h5>
          <ul className="list-group list-group-flush">
            {(defs.standards || []).map((s, i) => (
              <li key={i} className="list-group-item small">{s}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
