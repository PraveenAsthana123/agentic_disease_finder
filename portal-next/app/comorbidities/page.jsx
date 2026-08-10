'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'conditions',  label: 'Conditions' },
  { id: 'patients',    label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
];

const SEV_COLOR = {
  minimal: 'success', mild: 'info', moderate: 'warning', severe: 'danger',
};
const IMPACT_COLOR = {
  none: 'success', mild: 'info', moderate: 'warning', severe: 'danger',
};
const TX_COLOR = {
  none: 'secondary', untreated: 'danger', stable: 'success',
  under_treatment: 'primary', partial_response: 'warning', treatment_resistant: 'dark',
};

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

function Bar({ label, value, max, color, pct }) {
  const p = pct !== undefined ? pct : Math.min(100, Math.round((value / Math.max(max, 1)) * 100));
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <span className="small">{label}</span>
        <span className="small fw-bold">{value}</span>
      </div>
      <div className="progress" style={{ height: '8px' }}>
        <div className={`progress-bar bg-${color || 'primary'}`} style={{ width: `${p}%` }} />
      </div>
    </div>
  );
}

function Badge({ label, color }) {
  return (
    <span className={`badge bg-${color || 'secondary'} me-1 mb-1`} style={{ fontSize: '0.72rem' }}>
      {label}
    </span>
  );
}

export default function ComorbiditiesDashboard() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('behavioral_risk_score');

  useEffect(() => {
    fetch(`${API}/api/comorbidities/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/comorbidities/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/comorbidities/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return (
    <div className="p-5 text-center">
      <div className="spinner-border text-primary" />
      <div className="mt-2 text-muted small">Loading Comorbidity Data…</div>
    </div>
  );

  const patients = (bd?.patients || []).filter(p =>
    !search || p.patient_id?.toLowerCase().includes(search.toLowerCase()) ||
    (p.comorbidities || '').toLowerCase().includes(search.toLowerCase())
  ).sort((a, b) => {
    if (sortBy === 'behavioral_risk_score') return (b.behavioral_risk_score || 0) - (a.behavioral_risk_score || 0);
    if (sortBy === 'comorbidity_count') return (b.comorbidity_count || 0) - (a.comorbidity_count || 0);
    return (a.patient_id || '').localeCompare(b.patient_id || '');
  });

  const conditions = ov.condition_distribution || [];
  const maxCond = Math.max(1, ...conditions.map(c => c.count));

  return (
    <div>
      {/* Header */}
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: '1.8rem' }}>🧠</span>
        <div>
          <h4 className="mb-0 fw-bold">Comorbidity Analytics Dashboard</h4>
          <p className="text-muted mb-0 small">
            Psychiatric &amp; neurological comorbidities in epilepsy — {ov.total_patients} patients screened
          </p>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          <div className="row g-3 mb-4">
            <KPI label="Total Patients" value={ov.total_patients} color="primary" />
            <KPI label="With Comorbidities" value={ov.total_with_comorbidities}
              color="warning" sub={`${ov.comorbidity_rate?.toFixed(1)}% prevalence`} />
            <KPI label="Avg Comorbidity Count" value={ov.avg_comorbidity_count?.toFixed(1)}
              color="info" sub={`Max: ${ov.max_comorbidity_count}`} />
            <KPI label="Avg Behavioral Risk" value={ov.avg_behavioral_risk_score?.toFixed(0)}
              color="danger" sub="Score 0–100" />
          </div>

          <div className="row g-3">
            {/* Severity distribution */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Risk Severity Distribution</div>
                <div className="card-body">
                  {(ov.severity_distribution || []).map(s => (
                    <div key={s.severity} className="d-flex justify-content-between align-items-center mb-2">
                      <Badge label={s.severity} color={SEV_COLOR[s.severity]} />
                      <div className="d-flex align-items-center gap-2">
                        <div className="progress flex-grow-1" style={{ width: '80px', height: '8px' }}>
                          <div
                            className={`progress-bar bg-${SEV_COLOR[s.severity]}`}
                            style={{ width: `${Math.round((s.count / ov.total_patients) * 100)}%` }}
                          />
                        </div>
                        <span className="small fw-bold">{s.count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Treatment status */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Treatment Status</div>
                <div className="card-body">
                  {(ov.treatment_distribution || []).map(t => (
                    <div key={t.status} className="d-flex justify-content-between align-items-center mb-2">
                      <Badge label={t.status.replace('_', ' ')} color={TX_COLOR[t.status]} />
                      <div className="d-flex align-items-center gap-2">
                        <div className="progress flex-grow-1" style={{ width: '80px', height: '8px' }}>
                          <div
                            className={`progress-bar bg-${TX_COLOR[t.status]}`}
                            style={{ width: `${Math.round((t.count / ov.total_patients) * 100)}%` }}
                          />
                        </div>
                        <span className="small fw-bold">{t.count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Functional impact */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Functional Impact</div>
                <div className="card-body">
                  {(ov.impact_distribution || []).map(i => (
                    <div key={i.impact} className="d-flex justify-content-between align-items-center mb-2">
                      <Badge label={i.impact} color={IMPACT_COLOR[i.impact]} />
                      <div className="d-flex align-items-center gap-2">
                        <div className="progress flex-grow-1" style={{ width: '80px', height: '8px' }}>
                          <div
                            className={`progress-bar bg-${IMPACT_COLOR[i.impact]}`}
                            style={{ width: `${Math.round((i.count / ov.total_patients) * 100)}%` }}
                          />
                        </div>
                        <span className="small fw-bold">{i.count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Comorbidity count distribution */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-semibold">Comorbidity Count Distribution (per patient)</div>
            <div className="card-body">
              <div className="d-flex align-items-end gap-3 flex-wrap">
                {(ov.count_distribution || []).map(c => {
                  const h = Math.max(20, Math.round((c.count / ov.total_patients) * 120));
                  return (
                    <div key={c.bucket} className="text-center" style={{ minWidth: '52px' }}>
                      <div className="text-muted small mb-1 fw-bold">{c.count}</div>
                      <div className="rounded-top"
                        style={{ height: `${h}px`, background: '#6366f1', width: '40px', margin: '0 auto' }} />
                      <div className="small mt-1">{c.bucket}</div>
                    </div>
                  );
                })}
              </div>
              <div className="text-muted small mt-2">Number of comorbid conditions per patient</div>
            </div>
          </div>

          {/* Risk by severity */}
          {ov.risk_by_severity?.length > 0 && (
            <div className="card shadow-sm mt-3">
              <div className="card-header fw-semibold">Average Behavioral Risk Score by Severity</div>
              <div className="card-body">
                <div className="row g-3">
                  {(ov.risk_by_severity || []).map(r => (
                    <div key={r.severity} className="col-6 col-md-3">
                      <div className="card text-center border-0 bg-light">
                        <div className="card-body py-2">
                          <div className={`h5 fw-bold text-${SEV_COLOR[r.severity]}`}>
                            {r.avg_score?.toFixed(1)}
                          </div>
                          <div className="small text-muted">{r.severity}</div>
                          <div className="text-muted" style={{ fontSize: '0.7rem' }}>n={r.count}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── CONDITIONS ── */}
      {tab === 'conditions' && (
        <div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">
              Condition Frequency (all {ov.total_patients} patients)
            </div>
            <div className="card-body">
              {conditions.length === 0 ? (
                <div className="text-muted">No condition data available.</div>
              ) : conditions.map(c => (
                <Bar key={c.condition} label={c.condition} value={c.count} max={maxCond} color="danger" />
              ))}
            </div>
          </div>

          {/* Screening instruments */}
          {ov.instrument_distribution?.length > 0 && (
            <div className="card shadow-sm mt-3">
              <div className="card-header fw-semibold">Screening Instruments Used</div>
              <div className="card-body">
                <div className="row g-3">
                  {(ov.instrument_distribution || []).map(i => (
                    <div key={i.instrument} className="col-6 col-md-3">
                      <div className="card border-0 bg-light text-center">
                        <div className="card-body py-2">
                          <div className="h5 fw-bold text-primary">{i.count}</div>
                          <div className="small fw-semibold">{i.instrument}</div>
                          <div className="text-muted" style={{ fontSize: '0.7rem' }}>uses</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── PATIENTS ── */}
      {tab === 'patients' && (
        <div>
          <div className="d-flex gap-2 mb-3 flex-wrap">
            <input
              className="form-control form-control-sm"
              style={{ maxWidth: '220px' }}
              placeholder="Search patient / condition…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            <select
              className="form-select form-select-sm"
              style={{ maxWidth: '200px' }}
              value={sortBy}
              onChange={e => setSortBy(e.target.value)}
            >
              <option value="behavioral_risk_score">Sort: Risk Score ↓</option>
              <option value="comorbidity_count">Sort: Comorbidity Count ↓</option>
              <option value="patient_id">Sort: Patient ID</option>
            </select>
            <span className="small text-muted align-self-center">{patients.length} patient(s)</span>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-hover small">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Count</th>
                  <th>Conditions</th>
                  <th>Risk Score</th>
                  <th>Severity</th>
                  <th>Impact</th>
                  <th>Treatment</th>
                  <th>Screened</th>
                </tr>
              </thead>
              <tbody>
                {patients.map(p => (
                  <tr key={p.patient_id}>
                    <td className="fw-semibold">{p.patient_id}</td>
                    <td>
                      <span className={`badge bg-${p.comorbidity_count === 0 ? 'success' : p.comorbidity_count >= 4 ? 'danger' : 'warning'}`}>
                        {p.comorbidity_count}
                      </span>
                    </td>
                    <td style={{ maxWidth: '220px', whiteSpace: 'normal' }}>
                      {p.comorbidities === 'None' || !p.comorbidities
                        ? <span className="text-muted">None</span>
                        : p.comorbidities.split(', ').map((c, i) => (
                            <Badge key={i} label={c} color="secondary" />
                          ))
                      }
                    </td>
                    <td>
                      <span className={`fw-bold text-${
                        p.behavioral_risk_score >= 65 ? 'danger' : p.behavioral_risk_score >= 35 ? 'warning' : 'success'
                      }`}>{p.behavioral_risk_score?.toFixed(1)}</span>
                    </td>
                    <td><Badge label={p.risk_severity} color={SEV_COLOR[p.risk_severity]} /></td>
                    <td><Badge label={p.functional_impact} color={IMPACT_COLOR[p.functional_impact]} /></td>
                    <td><Badge label={(p.treatment_status || '').replace('_', ' ')} color={TX_COLOR[p.treatment_status]} /></td>
                    <td className="text-muted">{p.screening_date}</td>
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
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Field Definitions</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Field</th><th>Description</th></tr></thead>
                <tbody>
                  {(defs.fields || []).map(f => (
                    <tr key={f.field}>
                      <td className="fw-semibold text-nowrap">{f.field}</td>
                      <td className="small">{f.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Comorbid Conditions Glossary</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Condition</th><th>Description</th></tr></thead>
                <tbody>
                  {(defs.conditions || []).map(c => (
                    <tr key={c.name}>
                      <td className="fw-semibold text-nowrap">{c.name}</td>
                      <td className="small">{c.description}</td>
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
