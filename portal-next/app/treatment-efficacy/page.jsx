'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8010';

const responseColor = cat => ({
  Excellent: 'success',
  Good: 'info',
  Partial: 'warning',
  Poor: 'danger',
}[cat] || 'secondary');

const severityBar = (val, max = 10) => {
  const pct = Math.round((val / max) * 100);
  const color = pct < 35 ? 'success' : pct < 60 ? 'warning' : 'danger';
  return (
    <div className="progress" style={{ height: 8 }}>
      <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
    </div>
  );
};

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

export default function TreatmentEfficacyDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('adherence_pct');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/treatment-efficacy/overview`).then(r => r.json()),
      fetch(`${API}/api/treatment-efficacy/breakdown`).then(r => r.json()),
      fetch(`${API}/api/treatment-efficacy/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading Treatment Efficacy data…</div>;

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'drugs', label: '💊 By Drug' },
    { id: 'patients', label: '🧑‍⚕️ Per Patient' },
    { id: 'sideeffects', label: '⚠️ Side Effects' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const patRows = (bd?.per_patient || [])
    .filter(p => !search || Object.values(p).join(' ').toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === 'adherence_pct') return (b.adherence_pct ?? 0) - (a.adherence_pct ?? 0);
      if (sortBy === 'seizure_count') return (b.seizure_count ?? 0) - (a.seizure_count ?? 0);
      if (sortBy === 'missed_doses') return (b.missed_doses ?? 0) - (a.missed_doses ?? 0);
      if (sortBy === 'patient_id') return (a.patient_id || '').localeCompare(b.patient_id || '');
      return 0;
    });

  return (
    <div className="container-fluid py-3">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <div>
          <h3 className="mb-0">💊 Treatment Efficacy Dashboard</h3>
          <small className="text-muted">
            {ov.total_patients} patients · {ov.unique_drugs} AEDs ·{' '}
            {ov.overall_adherence_pct}% overall adherence
          </small>
        </div>
        <span className="badge bg-primary fs-6">Clinical</span>
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

      {/* OVERVIEW */}
      {tab === 'overview' && (
        <>
          <div className="row">
            <KPI label="Total Patients" value={ov.total_patients} color="primary" />
            <KPI label="Overall Adherence" value={`${ov.overall_adherence_pct}%`} color="success" />
            <KPI label="On-Time Doses" value={`${ov.on_time_pct}%`} color="info" />
            <KPI label="Missed Doses" value={`${ov.missed_pct}%`} color="danger" sub="of all scheduled" />
          </div>
          <div className="row">
            <KPI label="Total Seizures" value={ov.total_seizures} color="warning" />
            <KPI label="Avg Seizure/Patient" value={ov.avg_seizure_frequency} color="warning" />
            <KPI label="Unique AEDs" value={ov.unique_drugs} color="secondary" />
            <KPI label="Most Common Response" value="Good" color="success" />
          </div>

          {/* Treatment Response */}
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold">Treatment Response Categories</div>
                <div className="card-body">
                  {(ov.treatment_response_categories || []).map(r => (
                    <div key={r.category} className="d-flex justify-content-between align-items-center mb-2">
                      <span className={`badge bg-${responseColor(r.category)} me-2`}>{r.category}</span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{ height: 14 }}>
                          <div
                            className={`progress-bar bg-${responseColor(r.category)}`}
                            style={{ width: `${Math.round((r.count / ov.total_patients) * 100)}%` }}
                          />
                        </div>
                      </div>
                      <span className="text-muted small">{r.count} pts</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold">Monthly Adherence Trend</div>
                <div className="card-body">
                  <table className="table table-sm">
                    <thead>
                      <tr><th>Month</th><th>Adherence %</th><th>Seizures</th></tr>
                    </thead>
                    <tbody>
                      {(ov.monthly_adherence_trend || []).map(m => (
                        <tr key={m.month}>
                          <td>{m.month}</td>
                          <td>
                            <span className="badge bg-success">{m.adherence_pct}%</span>
                          </td>
                          <td>{m.seizure_count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Adherence by time of day */}
          {bd?.adherence_by_time_of_day && (
            <div className="card mb-3">
              <div className="card-header fw-bold">Adherence by Time of Day</div>
              <div className="card-body">
                <div className="row text-center">
                  {Object.entries(bd.adherence_by_time_of_day).map(([slot, pct]) => (
                    <div key={slot} className="col-4">
                      <div className="fs-4 fw-bold text-info">{pct}%</div>
                      <div className="text-muted text-capitalize">{slot}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Drug List summary */}
          <div className="card">
            <div className="card-header fw-bold">AED Overview</div>
            <div className="card-body p-0">
              <table className="table table-striped table-sm mb-0">
                <thead><tr><th>Drug</th><th>Patients</th><th>Adherence %</th></tr></thead>
                <tbody>
                  {(ov.drug_list || []).map(d => (
                    <tr key={d.drug_name}>
                      <td>{d.drug_name}</td>
                      <td>{d.patient_count}</td>
                      <td><span className="badge bg-success">{d.adherence_pct}%</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* BY DRUG */}
      {tab === 'drugs' && (
        <div className="row">
          {(bd?.per_drug || []).map(d => (
            <div key={d.drug_name} className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header fw-bold d-flex justify-content-between">
                  <span>💊 {d.drug_name}</span>
                  <span className="badge bg-secondary">{d.patient_count} pts</span>
                </div>
                <div className="card-body">
                  <div className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>Avg Adherence</span>
                      <strong>{d.avg_adherence_pct}%</strong>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div className="progress-bar bg-success" style={{ width: `${d.avg_adherence_pct}%` }} />
                    </div>
                  </div>
                  <div className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>Avg Side Effect Severity</span>
                      <strong>{d.avg_side_effect_severity}/10</strong>
                    </div>
                    {severityBar(d.avg_side_effect_severity)}
                  </div>
                  <div className="small text-muted">
                    <strong>Common side effects:</strong>{' '}
                    {(d.most_common_side_effects || []).join(', ') || 'None recorded'}
                  </div>
                  {d.seizure_reduction_corr !== undefined && (
                    <div className="small text-muted mt-1">
                      <strong>Seizure reduction correlation:</strong>{' '}
                      {d.seizure_reduction_corr}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* PER PATIENT */}
      {tab === 'patients' && (
        <div className="card">
          <div className="card-header d-flex justify-content-between align-items-center flex-wrap gap-2">
            <span className="fw-bold">Per-Patient Treatment Data ({patRows.length})</span>
            <div className="d-flex gap-2">
              <select
                className="form-select form-select-sm"
                style={{ width: 180 }}
                value={sortBy}
                onChange={e => setSortBy(e.target.value)}
              >
                <option value="adherence_pct">Sort: Adherence ↓</option>
                <option value="seizure_count">Sort: Seizures ↓</option>
                <option value="missed_doses">Sort: Missed Doses ↓</option>
                <option value="patient_id">Sort: Patient ID</option>
              </select>
              <input
                className="form-control form-control-sm"
                style={{ width: 180 }}
                placeholder="Search…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-striped table-hover table-sm mb-0">
                <thead>
                  <tr>
                    <th>Patient</th>
                    <th>Drugs</th>
                    <th>Adherence %</th>
                    <th>On-Time %</th>
                    <th>Missed</th>
                    <th>Seizures</th>
                    <th>Response</th>
                    <th>Mood</th>
                    <th>Top Side Effect</th>
                  </tr>
                </thead>
                <tbody>
                  {patRows.map(p => (
                    <tr key={p.patient_id}>
                      <td><strong>{p.patient_id}</strong></td>
                      <td><small>{Array.isArray(p.drug_names) ? p.drug_names.join(', ') : p.drug_names}</small></td>
                      <td><span className="badge bg-success">{p.adherence_pct}%</span></td>
                      <td>{p.on_time_pct}%</td>
                      <td>{p.missed_doses}</td>
                      <td>{p.seizure_count}</td>
                      <td>
                        <span className={`badge bg-${responseColor(p.response_category)}`}>
                          {p.response_category}
                        </span>
                      </td>
                      <td>{p.avg_mood}/10</td>
                      <td><small className="text-muted">{p.most_common_side_effect || '—'}</small></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* SIDE EFFECTS */}
      {tab === 'sideeffects' && (
        <div className="row">
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">Side Effect Profile (All Patients)</div>
              <div className="card-body p-0">
                <table className="table table-striped table-sm mb-0">
                  <thead>
                    <tr><th>#</th><th>Side Effect</th><th>Occurrences</th><th>Avg Severity</th><th>Severity</th></tr>
                  </thead>
                  <tbody>
                    {(ov.side_effect_profile || []).map((s, i) => (
                      <tr key={s.side_effect}>
                        <td className="text-muted">{i + 1}</td>
                        <td>{s.side_effect}</td>
                        <td>{s.count}</td>
                        <td>{s.avg_severity}/10</td>
                        <td style={{ minWidth: 80 }}>{severityBar(s.avg_severity)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">Side Effects by Drug</div>
              <div className="card-body p-0">
                <table className="table table-striped table-sm mb-0">
                  <thead>
                    <tr><th>Drug</th><th>Side Effects</th><th>Avg Severity</th></tr>
                  </thead>
                  <tbody>
                    {(bd?.side_effects_by_drug || []).map(d => (
                      <tr key={d.drug_name}>
                        <td><strong>{d.drug_name}</strong></td>
                        <td><small>{(d.side_effects || []).join(', ') || '—'}</small></td>
                        <td>{d.avg_severity ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DEFINITIONS */}
      {tab === 'definitions' && (
        <div className="card">
          <div className="card-header fw-bold">Glossary &amp; Definitions</div>
          <div className="card-body">
            {Array.isArray(defs?.concepts) && defs.concepts.map(c => (
              <div key={c.name} className="mb-3">
                <h6 className="fw-bold">{c.name}</h6>
                <p className="text-muted small mb-0">{c.description}</p>
              </div>
            ))}
            {!defs?.concepts && (
              <p className="text-muted">No definitions available.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
