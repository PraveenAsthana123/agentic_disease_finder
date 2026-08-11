'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'outcomes',   label: 'Engel / ILAE' },
  { id: 'patients',   label: 'Per Patient' },
  { id: 'definitions',label: 'Definitions' },
];

const MED_LABELS = {
  drug_responsive:    'Drug-Responsive',
  partially_responsive: 'Partially Responsive',
  drug_resistant:     'Drug-Resistant',
};
const MED_COLORS = {
  drug_responsive:    'success',
  partially_responsive: 'warning',
  drug_resistant:     'danger',
};
const SUDEP_COLORS = { low: 'success', moderate: 'warning', high: 'danger' };
const MORT_ICONS = { alive: '✅', sudep: '💀', other_cause: '⚠️', lost_to_followup: '❓' };

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

function MedBadge({ response }) {
  const color = MED_COLORS[response] || 'secondary';
  return <span className={`badge bg-${color}`}>{MED_LABELS[response] || response}</span>;
}

function EngelBadge({ engel }) {
  const color = engel === 1 ? 'success' : engel === 2 ? 'info' : engel === 3 ? 'warning' : 'danger';
  return <span className={`badge bg-${color}`}>Engel {engel}</span>;
}

function SudepBadge({ risk }) {
  const color = SUDEP_COLORS[risk] || 'secondary';
  return <span className={`badge bg-${color}`}>{risk?.toUpperCase()}</span>;
}

export default function ClinicalOutcomesPage() {
  const [tab, setTab]         = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const [patSearch, setPatSearch] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/clinical-outcomes/overview`).then(r => r.json()),
      fetch(`${API}/api/clinical-outcomes/breakdown`).then(r => r.json()),
      fetch(`${API}/api/clinical-outcomes/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border text-primary" /></div>;
  if (error)   return <div className="container py-4"><div className="alert alert-danger">Error: {error}</div></div>;

  const ov = overview;
  const bd = breakdown;
  const df = defs;

  const filteredPats = bd?.per_patient?.filter(p =>
    !patSearch || p.patient_id.toLowerCase().includes(patSearch.toLowerCase())
  ) || [];

  return (
    <div className="container-fluid py-4">
      <h2 className="mb-1 fw-bold">📊 Clinical Outcomes</h2>
      <p className="text-muted small mb-3">
        Engel / ILAE classification · Drug-Resistant Epilepsy (DRE) · SUDEP risk · Medication response ·
        30-day readmissions · {ov.total_patients} patients ·
        Engel 1987 + Kwan et al. 2010 + Nashef 1997
      </p>

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          <div className="row">
            <KPI label="Total Patients"           value={ov.total_patients}              color="primary" />
            <KPI label="Drug-Resistant (DRE)"     value={`${ov.dre_count} (${ov.dre_rate_pct}%)`}  color="danger" sub="≥2 AED trial failures" />
            <KPI label="Seizure-Free @ 12 m"      value={`${ov.seizure_free_12m_count} (${ov.seizure_free_12m_pct}%)`} color="success" />
            <KPI label="Avg AED Trials"            value={ov.avg_aed_trials}             color="info"    sub="per patient" />
          </div>
          <div className="row">
            <KPI label="Avg Seizure Reduction"    value={`${ov.avg_seizure_reduction_pct}%`} color="success" />
            <KPI label="30-Day Readmissions"      value={`${ov.readmit_30d_count} (${ov.readmit_30d_pct}%)`} color="warning" />
            <KPI label="SUDEP High Risk"          value={ov.sudep_risk_distribution.high}  color="danger" />
            <KPI label="Lost to Follow-Up"        value={ov.mortality_distribution.lost_to_followup} color="secondary" />
          </div>

          <div className="row mt-3">
            {/* Medication Response */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Medication Response</div>
                <div className="card-body">
                  {Object.entries(ov.medication_response).map(([k, v]) => (
                    <div key={k} className="mb-2">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="small">{MED_LABELS[k] || k}</span>
                        <span className="small fw-bold">{v}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${MED_COLORS[k]}`}
                          style={{ width: `${v / ov.total_patients * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Engel Distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Engel Class Distribution</div>
                <div className="card-body">
                  {[1, 2, 3, 4].map(cls => (
                    <div key={cls} className="mb-2">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="small">Class {cls}</span>
                        <span className="small fw-bold">{ov.engel_distribution[cls] ?? 0}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar ${cls === 1 ? 'bg-success' : cls === 2 ? 'bg-info' : cls === 3 ? 'bg-warning' : 'bg-danger'}`}
                          style={{ width: `${(ov.engel_distribution[cls] ?? 0) / ov.total_patients * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* SUDEP Risk */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">SUDEP Risk Distribution</div>
                <div className="card-body">
                  {['low', 'moderate', 'high'].map(level => (
                    <div key={level} className="mb-2">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="small text-capitalize">{level}</span>
                        <span className="small fw-bold">{ov.sudep_risk_distribution[level] ?? 0}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${SUDEP_COLORS[level]}`}
                          style={{ width: `${(ov.sudep_risk_distribution[level] ?? 0) / ov.total_patients * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                  <p className="text-muted small mt-2 mb-0">
                    Risk factors: GTCS, nocturnal Sz, DRE, young male, poor adherence
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Seizure-Free Trend */}
          <div className="card shadow-sm mt-2">
            <div className="card-header fw-semibold">Seizure-Free Rate by Follow-Up Milestone</div>
            <div className="card-body">
              <div className="row text-center">
                {ov.seizure_free_trend.map(item => (
                  <div key={item.month} className="col-3">
                    <div className="display-6 fw-bold text-success">{item.seizure_free_pct}%</div>
                    <div className="text-muted small">{item.month} months</div>
                    <div className="text-muted" style={{ fontSize: '0.7rem' }}>{item.seizure_free_count}/{ov.total_patients} patients</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* DRE by Epilepsy Type */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-semibold">DRE Rate by Epilepsy Type</div>
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0">
                <thead><tr><th>Type</th><th>Total</th><th>DRE</th><th>DRE Rate</th></tr></thead>
                <tbody>
                  {ov.type_dre_breakdown.map(row => (
                    <tr key={row.epilepsy_type}>
                      <td>{row.epilepsy_type}</td>
                      <td>{row.total}</td>
                      <td className="text-danger fw-bold">{row.dre_count}</td>
                      <td>
                        <div className="progress" style={{ height: 8, minWidth: 80 }}>
                          <div className="progress-bar bg-danger" style={{ width: `${row.dre_rate_pct}%` }} />
                        </div>
                        <small>{row.dre_rate_pct}%</small>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Mortality */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-semibold">Mortality / Vital Status</div>
            <div className="card-body">
              <div className="row text-center">
                {Object.entries(ov.mortality_distribution).map(([k, v]) => (
                  <div key={k} className="col-3">
                    <div className="h4">{MORT_ICONS[k] || '?'}</div>
                    <div className="fw-bold">{v}</div>
                    <div className="text-muted small text-capitalize">{k.replace(/_/g, ' ')}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <p className="text-muted small mt-3">
            Standards: {ov.engel_standard} · {ov.ilae_standard}
          </p>
        </>
      )}

      {/* ── ENGEL / ILAE ── */}
      {tab === 'outcomes' && (
        <>
          <div className="row">
            {/* Engel Detail */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Engel Scale Distribution</div>
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead><tr><th>Class</th><th>Label</th><th>Count</th><th>%</th></tr></thead>
                    <tbody>
                      {bd?.engel_distribution?.map(row => (
                        <tr key={row.engel_class}>
                          <td><EngelBadge engel={row.engel_class} /></td>
                          <td className="small">{row.label}</td>
                          <td className="fw-bold">{row.count}</td>
                          <td className="small">{Math.round(row.count / ov.total_patients * 100)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* ILAE Detail */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">ILAE 2010 Class Distribution</div>
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead><tr><th>Class</th><th>Count</th><th>%</th></tr></thead>
                    <tbody>
                      {[1, 2, 3, 4, 5, 6].map(cls => (
                        <tr key={cls}>
                          <td><span className="badge bg-secondary">Class {cls}</span></td>
                          <td className="fw-bold">{ov.ilae_distribution[cls] ?? 0}</td>
                          <td className="small">{Math.round((ov.ilae_distribution[cls] ?? 0) / ov.total_patients * 100)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Top AEDs */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Top Antiseizure Drugs (AEDs) in Use</div>
            <div className="card-body">
              <div className="row">
                {bd?.top_aeds?.map(item => (
                  <div key={item.aed} className="col-6 col-md-4 mb-2">
                    <div className="d-flex align-items-center gap-2">
                      <div className="fw-semibold small" style={{ minWidth: 130 }}>{item.aed}</div>
                      <div className="progress flex-grow-1" style={{ height: 10 }}>
                        <div
                          className="progress-bar bg-primary"
                          style={{ width: `${item.count / ov.total_patients * 100}%` }}
                        />
                      </div>
                      <div className="small fw-bold">{item.count}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* SUDEP High Risk Register */}
          {bd?.sudep_high_risk?.length > 0 && (
            <div className="card shadow-sm border-danger">
              <div className="card-header fw-semibold text-danger">SUDEP High-Risk Register ({bd.sudep_high_risk.length} patients)</div>
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Patient</th><th>Age</th><th>Sex</th><th>Epilepsy Type</th><th>Engel</th><th>Med Response</th></tr></thead>
                  <tbody>
                    {bd.sudep_high_risk.map(p => (
                      <tr key={p.patient_id}>
                        <td className="fw-bold">{p.patient_id}</td>
                        <td>{p.age}</td>
                        <td>{p.sex}</td>
                        <td>{p.epilepsy_type}</td>
                        <td><EngelBadge engel={p.engel_class} /></td>
                        <td><MedBadge response={p.medication_response} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && (
        <>
          <div className="mb-3">
            <input
              className="form-control w-auto"
              placeholder="Search patient ID..."
              value={patSearch}
              onChange={e => setPatSearch(e.target.value)}
            />
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-striped table-hover">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th><th>Age</th><th>Sex</th><th>Epilepsy Type</th>
                  <th>AED Trials</th><th>Med Response</th><th>Engel</th><th>ILAE</th>
                  <th>Sz Reduction</th><th>SUDEP Risk</th><th>Readmit 30d</th>
                  <th>SF 12m</th><th>Mortality</th>
                </tr>
              </thead>
              <tbody>
                {filteredPats.map(p => (
                  <tr key={p.patient_id}>
                    <td className="fw-bold">{p.patient_id}</td>
                    <td>{p.age}</td>
                    <td>{p.sex}</td>
                    <td><span className="badge bg-secondary">{p.epilepsy_type}</span></td>
                    <td>{p.aed_trials}</td>
                    <td><MedBadge response={p.medication_response} /></td>
                    <td><EngelBadge engel={p.engel_class} /></td>
                    <td>{p.ilae_class}</td>
                    <td>{p.seizure_reduction_pct}%</td>
                    <td><SudepBadge risk={p.sudep_risk} /></td>
                    <td>{p.readmit_30d ? <span className="badge bg-danger">Yes</span> : <span className="badge bg-success">No</span>}</td>
                    <td>{p.seizure_free_12m ? '✅' : '❌'}</td>
                    <td><span className="small">{MORT_ICONS[p.mortality_status] || '?'} {p.mortality_status?.replace(/_/g, ' ')}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && df && (
        <div className="row">
          <div className="col-md-6">
            {/* Engel Scale */}
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">{df.outcome_scales.engel_scale.full_name}</div>
              <div className="card-body">
                <p className="small text-muted">{df.outcome_scales.engel_scale.purpose}</p>
                <table className="table table-sm table-striped">
                  <thead><tr><th>Class</th><th>Definition</th></tr></thead>
                  <tbody>
                    {df.outcome_scales.engel_scale.classes.map(c => (
                      <tr key={c.class}>
                        <td><EngelBadge engel={c.class} /></td>
                        <td className="small">{c.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-muted" style={{ fontSize: '0.7rem' }}>{df.outcome_scales.engel_scale.reference}</p>
              </div>
            </div>

            {/* ILAE */}
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">{df.outcome_scales.ilae_outcome.full_name}</div>
              <div className="card-body">
                <p className="small text-muted">{df.outcome_scales.ilae_outcome.purpose}</p>
                <table className="table table-sm table-striped">
                  <thead><tr><th>Class</th><th>Definition</th></tr></thead>
                  <tbody>
                    {df.outcome_scales.ilae_outcome.classes.map(c => (
                      <tr key={c.class}>
                        <td><span className="badge bg-secondary">{c.class}</span></td>
                        <td className="small">{c.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-muted" style={{ fontSize: '0.7rem' }}>{df.outcome_scales.ilae_outcome.reference}</p>
              </div>
            </div>

            {/* Medication Response */}
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">Medication Response Categories</div>
              <div className="card-body">
                {df.medication_response.categories.map(c => (
                  <div key={c.code} className="mb-2">
                    <MedBadge response={c.code} />
                    <span className="ms-2 small">{c.definition}</span>
                  </div>
                ))}
                <p className="text-muted" style={{ fontSize: '0.7rem' }}>{df.medication_response.reference}</p>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            {/* DRE */}
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">{df.dre_definition.full_name}</div>
              <div className="card-body">
                <p className="small">{df.dre_definition.definition}</p>
                <p className="small text-muted">{df.dre_definition.clinical_note}</p>
                <p className="text-muted" style={{ fontSize: '0.7rem' }}>{df.dre_definition.reference}</p>
              </div>
            </div>

            {/* SUDEP */}
            <div className="card shadow-sm mb-3 border-danger">
              <div className="card-header fw-semibold text-danger">SUDEP — {df.sudep.full_name}</div>
              <div className="card-body">
                <p className="small">{df.sudep.definition}</p>
                <p className="small text-muted mb-1"><strong>Incidence:</strong> {df.sudep.incidence}</p>
                <p className="small fw-semibold mb-1">Risk Factors:</p>
                <ul className="small mb-2">
                  {df.sudep.risk_factors.map(f => <li key={f}>{f}</li>)}
                </ul>
                <table className="table table-sm table-bordered mb-2">
                  <thead><tr><th>Level</th><th>Criteria</th></tr></thead>
                  <tbody>
                    {Object.entries(df.sudep.risk_levels).map(([level, desc]) => (
                      <tr key={level}>
                        <td><SudepBadge risk={level} /></td>
                        <td className="small">{desc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-muted" style={{ fontSize: '0.7rem' }}>{df.sudep.reference}</p>
              </div>
            </div>

            {/* Readmission */}
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">30-Day Readmission</div>
              <div className="card-body">
                <p className="small"><strong>Definition:</strong> {df.readmission.definition}</p>
                <p className="small"><strong>Epilepsy context:</strong> {df.readmission.epilepsy_context}</p>
                <p className="small text-muted">{df.readmission.target}</p>
              </div>
            </div>

            {/* References */}
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">References</div>
              <div className="card-body">
                <ol className="small mb-0">
                  {df.references.map((ref, i) => <li key={i} className="mb-1">{ref}</li>)}
                </ol>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
