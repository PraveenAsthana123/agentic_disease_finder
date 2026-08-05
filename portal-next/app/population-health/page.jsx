'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'demographics', label: 'Demographics' },
  { id: 'risk', label: 'Risk Stratification' },
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

function DistBar({ items, labelKey, valueKey = 'count', colorFn }) {
  const total = items.reduce((a, b) => a + (b[valueKey] || 0), 0);
  return (
    <table className="table table-sm mb-0">
      <tbody>
        {items.map((item, i) => {
          const label = item[labelKey] || item.condition || item.drug || item.gender || item.group || String(i);
          const val = item[valueKey] || 0;
          const pct = item.pct != null ? item.pct : (total > 0 ? ((val / total) * 100).toFixed(1) : 0);
          const color = colorFn ? colorFn(label, item) : 'primary';
          return (
            <tr key={i}>
              <td className="small fw-semibold" style={{ width: '45%' }}>{label}</td>
              <td style={{ width: '40%' }}>
                <div className="progress" style={{ height: 10 }}>
                  <div className={`progress-bar bg-${color}`} style={{ width: `${Math.min(parseFloat(pct), 100)}%` }} />
                </div>
              </td>
              <td className="small text-end">
                {val} <span className="text-muted">({pct}%)</span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function OverviewPanel({ overview, breakdown }) {
  if (!overview) return <div className="text-muted">Loading…</div>;
  if (overview.error) return <div className="alert alert-warning">{overview.error}</div>;

  const genderColor = g => ({ Female: 'info', Male: 'primary', Unknown: 'secondary' }[g] || 'secondary');
  const ageColor = () => 'success';
  const comorbColor = () => 'warning';
  const drugColor = () => 'info';

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Total Patients" value={overview.total_patients} color="primary" sub="in registry" />
        <KPI label="Mean Age" value={overview.age_stats?.mean != null ? `${overview.age_stats.mean}y` : '—'} color="info"
          sub={`range ${overview.age_stats?.min}–${overview.age_stats?.max}y`} />
        <KPI label="Seizure Events" value={overview.seizure_burden?.total_events} color="danger"
          sub={`${overview.seizure_burden?.patients_with_events} patients affected`} />
        <KPI label="Comorbidities" value={(overview.comorbidity_prevalence || []).length} color="warning" sub="distinct conditions tracked" />
      </div>
      <div className="row mb-4">
        <KPI label="AED Prescriptions" value={overview.medication_coverage?.total_prescriptions} color="success"
          sub={`${overview.medication_coverage?.patients_with_meds} patients on AEDs`} />
        <KPI label="Median Age" value={overview.age_stats?.median != null ? `${overview.age_stats.median}y` : '—'} color="secondary" sub="median" />
        <KPI label="Severe Seizures" value={overview.seizure_burden?.severity_distribution?.find(s => s.severity === 'Severe')?.count ?? '—'}
          color="danger" sub="severe events" />
        <KPI label="EEG Records" value={overview.data_coverage?.eeg_acquisition} color="info" sub="EEG acquisitions" />
      </div>

      {/* Gender + Age distributions */}
      <div className="row mb-3">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Gender Distribution</div>
            <div className="card-body p-2">
              <DistBar items={overview.gender_distribution || []} labelKey="gender" colorFn={genderColor} />
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Age Group Distribution</div>
            <div className="card-body p-2">
              <DistBar items={overview.age_groups || []} labelKey="group" colorFn={ageColor} />
            </div>
          </div>
        </div>
      </div>

      {/* Comorbidities + Medications */}
      <div className="row mb-3">
        <div className="col-md-7 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Comorbidity Prevalence</div>
            <div className="card-body p-2" style={{ maxHeight: 280, overflowY: 'auto' }}>
              <DistBar items={overview.comorbidity_prevalence || []} labelKey="condition" colorFn={comorbColor} />
            </div>
          </div>
        </div>
        <div className="col-md-5 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">AED Drug Distribution</div>
            <div className="card-body p-2">
              <DistBar items={overview.medication_coverage?.drug_distribution || []} labelKey="drug" colorFn={drugColor} />
            </div>
          </div>
        </div>
      </div>

      {/* Seizure severity */}
      {(overview.seizure_burden?.severity_distribution || []).length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Seizure Severity Distribution</div>
          <div className="card-body p-2">
            <DistBar items={overview.seizure_burden.severity_distribution} labelKey="severity"
              colorFn={s => s === 'Severe' ? 'danger' : 'warning'} />
          </div>
        </div>
      )}

      {/* Enrollment trend */}
      {(overview.enrollment_trend || []).length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Enrollment Trend</div>
          <div className="card-body p-0">
            <table className="table table-sm table-striped mb-0">
              <thead><tr><th>Month</th><th className="text-end">New Enrollments</th></tr></thead>
              <tbody>
                {overview.enrollment_trend.map((e, i) => (
                  <tr key={i}>
                    <td>{e.month}</td>
                    <td className="text-end fw-bold">{e.count}</td>
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

function DemographicsPanel({ breakdown }) {
  if (!breakdown) return <div className="text-muted">Loading…</div>;
  if (breakdown.error) return <div className="alert alert-warning">{breakdown.error}</div>;

  const pyramid = breakdown.age_sex_pyramid || [];
  const registry = breakdown.patient_registry || [];
  const [search, setSearch] = useState('');
  const filtered = search
    ? registry.filter(p => p.patient_id?.toLowerCase().includes(search.toLowerCase()) || p.name?.toLowerCase().includes(search.toLowerCase()))
    : registry;

  return (
    <div>
      {/* Age-Sex Pyramid */}
      {pyramid.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Age–Sex Pyramid</div>
          <div className="card-body p-2">
            <table className="table table-sm mb-0 text-center">
              <thead>
                <tr>
                  <th className="text-end" style={{ width: '35%' }}>Male</th>
                  <th style={{ width: '30%' }}>Age Group</th>
                  <th className="text-start" style={{ width: '35%' }}>Female</th>
                </tr>
              </thead>
              <tbody>
                {pyramid.map((row, i) => (
                  <tr key={i}>
                    <td className="text-end">
                      <span className="badge bg-primary me-1">{row.male}</span>
                      <span className="text-muted small">{'█'.repeat(Math.min(row.male * 2, 20))}</span>
                    </td>
                    <td className="fw-semibold">{row.age_group}</td>
                    <td className="text-start">
                      <span className="text-muted small">{'█'.repeat(Math.min(row.female * 2, 20))}</span>
                      <span className="badge bg-info ms-1">{row.female}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Seizure characteristics */}
      {breakdown.seizure_characteristics && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Seizure Characteristics</div>
          <div className="card-body">
            <div className="row">
              <div className="col-md-6">
                <table className="table table-sm mb-0">
                  <tbody>
                    <tr><td>Aura Rate</td><td className="fw-bold">{breakdown.seizure_characteristics.aura_rate}%</td></tr>
                    <tr><td>Injury Rate</td><td className="fw-bold">{breakdown.seizure_characteristics.injury_rate}%</td></tr>
                    <tr><td>ER Visit Rate</td><td className="fw-bold">{breakdown.seizure_characteristics.er_visit_rate}%</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="col-md-6">
                {(breakdown.seizure_characteristics.trigger_distribution || []).length > 0 && (
                  <div>
                    <div className="small fw-semibold mb-1 text-muted">Top Triggers</div>
                    {breakdown.seizure_characteristics.trigger_distribution.slice(0, 5).map((t, i) => (
                      <span key={i} className="badge bg-warning text-dark me-1 mb-1">{t.trigger} ({t.pct}%)</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Patient Registry */}
      <div className="card">
        <div className="card-header fw-semibold d-flex align-items-center gap-2">
          Patient Registry
          <input className="form-control form-control-sm ms-auto" style={{ maxWidth: 200 }}
            placeholder="Search patient…" value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped mb-0" style={{ fontSize: '0.78rem' }}>
              <thead>
                <tr>
                  <th>Patient ID</th>
                  <th>Name</th>
                  <th className="text-center">Age</th>
                  <th>Gender</th>
                  <th className="text-center">Seizures</th>
                  <th className="text-center">Comorbidities</th>
                  <th>Medication</th>
                  <th>Last Assessment</th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 60).map((p, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{p.patient_id}</td>
                    <td>{p.name}</td>
                    <td className="text-center">{p.age ?? '—'}</td>
                    <td>
                      <span className={`badge bg-${p.gender === 'Female' ? 'info' : p.gender === 'Male' ? 'primary' : 'secondary'}`}>
                        {p.gender}
                      </span>
                    </td>
                    <td className="text-center">
                      {p.seizure_count > 0
                        ? <span className="badge bg-danger">{p.seizure_count}</span>
                        : <span className="text-muted">0</span>}
                    </td>
                    <td className="text-center">
                      {p.comorbidity_count > 0
                        ? <span className={`badge bg-${p.comorbidity_count >= 4 ? 'warning text-dark' : 'light text-dark'}`}>{p.comorbidity_count}</span>
                        : <span className="text-muted">0</span>}
                    </td>
                    <td className="small">{p.medication || '—'}</td>
                    <td className="small text-muted">{p.last_assessment ? p.last_assessment.substring(0, 10) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {filtered.length > 60 && <div className="card-footer text-muted small">Showing 60 of {filtered.length}</div>}
        </div>
      </div>
    </div>
  );
}

function RiskPanel({ breakdown }) {
  if (!breakdown) return <div className="text-muted">Loading…</div>;
  if (breakdown.error) return <div className="alert alert-warning">{breakdown.error}</div>;

  const risk = breakdown.risk_stratification || [];
  const counts = { High: 0, Moderate: 0, Low: 0 };
  risk.forEach(r => { if (counts[r.risk_level] != null) counts[r.risk_level]++; });

  const riskColor = l => ({ High: 'danger', Moderate: 'warning', Low: 'success' }[l] || 'secondary');
  const [filter, setFilter] = useState('all');
  const filtered = filter === 'all' ? risk : risk.filter(r => r.risk_level === filter);

  return (
    <div>
      <div className="row mb-3">
        {Object.entries(counts).map(([level, cnt]) => (
          <div key={level} className="col-4 mb-2">
            <div className="card text-center shadow-sm">
              <div className="card-body py-2">
                <div className={`h4 fw-bold text-${riskColor(level)}`}>{cnt}</div>
                <div className="small text-muted">{level} Risk</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="d-flex gap-2 mb-3">
        {['all', 'High', 'Moderate', 'Low'].map(f => (
          <button key={f} className={`btn btn-sm ${filter === f ? `btn-${f === 'High' ? 'danger' : f === 'Moderate' ? 'warning' : f === 'Low' ? 'success' : 'primary'}` : 'btn-outline-secondary'}`}
            onClick={() => setFilter(f)}>
            {f === 'all' ? `All (${risk.length})` : `${f} (${counts[f]})`}
          </button>
        ))}
      </div>

      <div className="card">
        <div className="card-header fw-semibold">Risk Stratification Table ({filtered.length})</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped mb-0" style={{ fontSize: '0.8rem' }}>
              <thead>
                <tr>
                  <th>Patient</th>
                  <th>Name</th>
                  <th className="text-center">Risk Level</th>
                  <th>Risk Factors</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{r.patient_id}</td>
                    <td>{r.name}</td>
                    <td className="text-center">
                      <span className={`badge bg-${riskColor(r.risk_level)}`}>{r.risk_level}</span>
                    </td>
                    <td className="small">
                      {(r.factors || []).map((f, j) => (
                        <span key={j} className="badge bg-light text-dark me-1 mb-1" style={{ fontSize: '0.7rem' }}>{f}</span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ definitions }) {
  if (!definitions) return <div className="text-muted">Loading…</div>;
  if (definitions.error) return <div className="alert alert-warning">{definitions.error}</div>;

  return (
    <div>
      <div className="card mb-3">
        <div className="card-header fw-semibold">Clinical Terminology</div>
        <div className="card-body p-0">
          <table className="table table-sm table-striped mb-0">
            <thead><tr><th style={{ width: '25%' }}>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {(definitions.terms || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold text-nowrap">{t.term}</td>
                  <td className="small">{t.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card mb-3">
        <div className="card-header fw-semibold">Data Sources</div>
        <div className="card-body p-0">
          <table className="table table-sm table-striped mb-0">
            <thead><tr><th>Source Table</th><th className="text-center">Rows</th><th>Description</th></tr></thead>
            <tbody>
              {(definitions.data_sources || []).map((s, i) => (
                <tr key={i}>
                  <td className="fw-semibold text-nowrap"><code>{s.source}</code></td>
                  <td className="text-center"><span className="badge bg-primary">{s.rows}</span></td>
                  <td className="small">{s.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {definitions.methodology && (
        <div className="card">
          <div className="card-header fw-semibold">Methodology</div>
          <div className="card-body small">{definitions.methodology}</div>
        </div>
      )}
    </div>
  );
}

export default function PopulationHealthDashboard() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/population-health/overview`)
      .then(r => r.json()).then(setOverview).catch(() => setOverview({ error: 'Failed to load overview' }));
    fetch(`${API}/api/population-health/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => setBreakdown({ error: 'Failed to load breakdown' }));
    fetch(`${API}/api/population-health/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => setDefinitions({ error: 'Failed to load definitions' }));
  }, []);

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1">&#x1f30d; Population Health Dashboard</h4>
      <p className="text-muted small mb-3">
        Cohort-level epidemiology for {overview?.total_patients ?? '—'} patients — demographics, seizure burden,
        comorbidity prevalence, medication coverage, enrollment trends, and risk stratification.
        All metrics computed live from clinical.db.
      </p>
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>
      {tab === 'overview' && <OverviewPanel overview={overview} breakdown={breakdown} />}
      {tab === 'demographics' && <DemographicsPanel breakdown={breakdown} />}
      {tab === 'risk' && <RiskPanel breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel definitions={definitions} />}
    </div>
  );
}
