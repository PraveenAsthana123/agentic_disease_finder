'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLOR = { completed: 'success', scheduled: 'primary', 'no-show': 'danger', cancelled: 'secondary', rescheduled: 'warning' };
const pct = (n, d) => (d ? Math.round((n / d) * 100) : 0);

export default function WorkflowEfficiencyPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/workflow-efficiency/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/workflow-efficiency/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/workflow-efficiency/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpi = overview.kpis || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'appointments', label: 'Appointments' },
    { id: 'providers', label: 'Providers' },
    { id: 'eeg-inpatient', label: 'EEG & Inpatient' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const maxType = overview.appointment_types?.length
    ? Math.max(...overview.appointment_types.map(t => t.total))
    : 1;
  const maxProv = overview.provider_workload?.length
    ? Math.max(...overview.provider_workload.map(p => p.total_appointments))
    : 1;

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">&#x2699;&#xfe0f; Clinical Workflow Efficiency Dashboard</h4>
      <p className="text-muted small mb-3">
        Operational analytics — {kpi.total_appointments} appointments · {kpi.patients_served} patients ·{' '}
        {kpi.providers_active} providers · {kpi.eeg_reads_total} EEG reads · {kpi.hospital_admissions} admissions
      </p>

      {/* KPI row */}
      <div className="row g-2 mb-3">
        {[
          { label: 'Total Appointments', value: kpi.total_appointments, sub: '191 across 8 types', color: 'primary' },
          { label: 'Completion Rate', value: `${kpi.completion_rate_pct}%`, sub: `${kpi.completed} completed`, color: 'success' },
          { label: 'No-Show Rate', value: `${kpi.no_show_rate_pct}%`, sub: 'target <10%', color: kpi.no_show_rate_pct > 10 ? 'danger' : 'warning' },
          { label: 'Avg Visit Duration', value: `${kpi.avg_visit_duration_min} min`, sub: 'completed appointments', color: 'info' },
          { label: 'Upcoming Scheduled', value: kpi.upcoming_scheduled, sub: 'future appointments', color: 'secondary' },
          { label: 'EEG AI Reads', value: kpi.eeg_reads_total, sub: `${kpi.avg_eeg_confidence_pct}% avg confidence`, color: 'primary' },
          { label: 'Hospital Admissions', value: kpi.hospital_admissions, sub: `avg LOS ${kpi.avg_los_days}d`, color: 'danger' },
          { label: 'Active Providers', value: kpi.providers_active, sub: `${kpi.patients_served} patients served`, color: 'dark' },
        ].map(k => (
          <div key={k.label} className="col-6 col-md-3">
            <div className={`card border-${k.color} h-100`}>
              <div className="card-body p-2 text-center">
                <div className={`fs-5 fw-bold text-${k.color}`}>{k.value}</div>
                <div className="small fw-semibold">{k.label}</div>
                <div className="text-muted" style={{ fontSize: '0.7rem' }}>{k.sub}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div>
          <div className="row g-3">
            {/* Appointment types */}
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">Appointment Type Throughput</div>
                <div className="card-body p-2">
                  {(overview.appointment_types || []).map(t => (
                    <div key={t.type} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{t.type}</span>
                        <span className="text-muted">{t.total} · {t.completion_rate_pct}% comp · {t.avg_duration_min}min</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div
                          className="progress-bar bg-primary"
                          style={{ width: `${pct(t.total, maxType * 1.05)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Location utilisation */}
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">Location Utilisation</div>
                <div className="card-body p-2">
                  {(overview.location_utilisation || []).map(l => (
                    <div key={l.location} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{l.location}</span>
                        <span className="text-muted">{l.total} total · {l.completion_rate_pct}% complete</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div
                          className="progress-bar bg-info"
                          style={{ width: `${l.completion_rate_pct}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Status summary cards */}
          <div className="row g-2 mt-2">
            {[
              { label: 'Completed', value: kpi.completed, pct: kpi.completion_rate_pct, color: 'success' },
              { label: 'Upcoming', value: kpi.upcoming_scheduled, pct: pct(kpi.upcoming_scheduled, kpi.total_appointments), color: 'primary' },
              { label: 'No-Show', value: Math.round(kpi.total_appointments * kpi.no_show_rate_pct / 100), pct: kpi.no_show_rate_pct, color: 'danger' },
              { label: 'Cancelled', value: Math.round(kpi.total_appointments * kpi.cancelled_pct / 100), pct: kpi.cancelled_pct, color: 'secondary' },
              { label: 'Rescheduled', value: Math.round(kpi.total_appointments * kpi.rescheduled_pct / 100), pct: kpi.rescheduled_pct, color: 'warning' },
            ].map(s => (
              <div key={s.label} className="col">
                <div className={`card border-${s.color} text-center`}>
                  <div className="card-body p-2">
                    <div className={`fs-4 fw-bold text-${s.color}`}>{s.value}</div>
                    <div className="small">{s.label}</div>
                    <div className="text-muted" style={{ fontSize: '0.7rem' }}>{s.pct}%</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* APPOINTMENTS TAB */}
      {tab === 'appointments' && breakdown && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">Department Throughput</div>
                <div className="card-body p-0">
                  <table className="table table-sm table-hover mb-0">
                    <thead><tr><th>Department</th><th>Total</th><th>Completed</th><th>Rate</th><th>Avg Duration</th></tr></thead>
                    <tbody>
                      {(breakdown.departments || []).map(d => (
                        <tr key={d.department}>
                          <td className="small">{d.department}</td>
                          <td>{d.total}</td>
                          <td>{d.completed}</td>
                          <td><span className={`badge bg-${d.completion_rate_pct >= 70 ? 'success' : 'warning'}`}>{d.completion_rate_pct}%</span></td>
                          <td className="small text-muted">{d.avg_duration_min}min</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">Monthly Appointment Trend</div>
                <div className="card-body p-2">
                  {(breakdown.monthly_trend || []).map(m => (
                    <div key={m.month} className="mb-1">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{m.month}</span>
                        <span className="text-muted">{m.total} appts · {m.comp} completed</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-success" style={{ width: `${pct(m.comp, m.total)}%` }} />
                        <div className="progress-bar bg-light" style={{ width: `${100 - pct(m.comp, m.total)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-12">
              <div className="card">
                <div className="card-header fw-semibold">Per-Patient Appointment Summary (Top 15)</div>
                <div className="card-body p-0">
                  <div className="table-responsive">
                    <table className="table table-sm table-hover mb-0">
                      <thead><tr><th>Patient ID</th><th>Visits</th><th>Completed</th><th>No-Shows</th><th>Completion Rate</th><th>Avg Duration</th></tr></thead>
                      <tbody>
                        {(breakdown.per_patient || []).map(p => (
                          <tr key={p.patient_id}>
                            <td className="fw-semibold small">{p.patient_id}</td>
                            <td>{p.total_visits}</td>
                            <td>{p.completed}</td>
                            <td>{p.no_shows > 0 ? <span className="text-danger fw-bold">{p.no_shows}</span> : 0}</td>
                            <td>
                              <div className="progress" style={{ height: 8, width: 80 }}>
                                <div className={`progress-bar bg-${p.completion_rate_pct >= 70 ? 'success' : 'warning'}`} style={{ width: `${p.completion_rate_pct}%` }} />
                              </div>
                              <span className="small text-muted">{p.completion_rate_pct}%</span>
                            </td>
                            <td className="small text-muted">{p.avg_duration_min}min</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* PROVIDERS TAB */}
      {tab === 'providers' && (
        <div>
          <div className="row g-3">
            {(overview.provider_workload || []).map(p => (
              <div key={p.provider} className="col-md-4">
                <div className="card h-100">
                  <div className="card-header fw-semibold">{p.provider}</div>
                  <div className="card-body p-2">
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small text-muted">Total Appointments</span>
                      <span className="fw-bold">{p.total_appointments}</span>
                    </div>
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small text-muted">Completed</span>
                      <span className="text-success fw-bold">{p.completed}</span>
                    </div>
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small text-muted">No-Shows</span>
                      <span className={p.no_shows > 0 ? 'text-danger' : 'text-muted'}>{p.no_shows}</span>
                    </div>
                    <div className="mt-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>Completion Rate</span><span className="fw-bold">{p.completion_rate_pct}%</span>
                      </div>
                      <div className="progress" style={{ height: 12 }}>
                        <div
                          className={`progress-bar bg-${p.completion_rate_pct >= 70 ? 'success' : p.completion_rate_pct >= 50 ? 'warning' : 'danger'}`}
                          style={{ width: `${p.completion_rate_pct}%` }}
                        />
                      </div>
                    </div>
                    <div className="d-flex justify-content-between mt-2">
                      <span className="small text-muted">Workload share</span>
                      <span className="small">{pct(p.total_appointments, maxProv * 1.05)}%</span>
                    </div>
                    <div className="progress mt-1" style={{ height: 6 }}>
                      <div className="progress-bar bg-primary" style={{ width: `${pct(p.total_appointments, maxProv * 1.05)}%` }} />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* EEG & INPATIENT TAB */}
      {tab === 'eeg-inpatient' && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card">
              <div className="card-header fw-semibold">EEG AI Reads by Disease & Signal Quality</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Disease</th><th>Signal Quality</th><th>Reads</th><th>Avg Confidence</th></tr></thead>
                  <tbody>
                    {(breakdown.eeg_reads || []).slice(0, 15).map((r, i) => (
                      <tr key={i}>
                        <td className="small">{r.disease}</td>
                        <td><span className={`badge bg-${r.signal_quality === 'Good' ? 'success' : r.signal_quality === 'Moderate' ? 'warning' : 'danger'}`}>{r.signal_quality}</span></td>
                        <td>{r.cnt}</td>
                        <td>{r.avg_conf_pct}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card">
              <div className="card-header fw-semibold">Hospital Ward × Disposition</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Ward</th><th>Disposition</th><th>Admissions</th><th>Avg LOS</th></tr></thead>
                  <tbody>
                    {(breakdown.hospitalization_breakdown || []).map((r, i) => (
                      <tr key={i}>
                        <td className="small">{r.ward}</td>
                        <td className="small">{r.discharge_disposition}</td>
                        <td>{r.cnt}</td>
                        <td>{r.avg_los}d</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card">
              <div className="card-header fw-semibold">Key Efficiency Metrics</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Metric</th><th>Formula</th><th>Benchmark</th></tr></thead>
                  <tbody>
                    {(defs.metrics || []).map(m => (
                      <tr key={m.metric}>
                        <td className="fw-semibold small">{m.metric}</td>
                        <td className="small text-muted">{m.formula}</td>
                        <td className="small text-success">{m.benchmark}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card">
              <div className="card-header fw-semibold">Appointment Types</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Type</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.appointment_types || []).map(t => (
                      <tr key={t.type}>
                        <td className="fw-semibold small">{t.type}</td>
                        <td className="small text-muted">{t.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card mt-3">
              <div className="card-header fw-semibold">Abbreviations</div>
              <div className="card-body p-2">
                {defs.abbreviations && Object.entries(defs.abbreviations).map(([k, v]) => (
                  <div key={k} className="d-flex gap-2 small mb-1">
                    <span className="badge bg-secondary">{k}</span>
                    <span className="text-muted">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">Data Sources</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Table</th><th>Rows</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.data_sources || []).map(ds => (
                      <tr key={ds.table}>
                        <td className="fw-semibold small font-monospace">{ds.table}</td>
                        <td>{ds.rows}</td>
                        <td className="small text-muted">{ds.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="text-muted small mt-3">
        Data: patient_appointments {kpi.total_appointments} records · analyses {kpi.eeg_reads_total} reads ·
        hospitalization {kpi.hospital_admissions} admissions · {defs?.references?.[0]}
      </div>
    </div>
  );
}
