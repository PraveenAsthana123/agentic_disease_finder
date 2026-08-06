'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const riskColor = l =>
  l === 'High' ? 'danger' :
  l === 'Medium' ? 'warning' :
  l === 'Low' ? 'success' : 'secondary';

const sevColor = l =>
  l === 'high' ? 'danger' :
  l === 'moderate' ? 'warning' :
  l === 'low' ? 'info' :
  l === 'none' ? 'success' : 'secondary';

const TABS = [
  { id: 'overview',     label: 'Overview' },
  { id: 'ketogenic',    label: 'Ketogenic Diet' },
  { id: 'malnutrition', label: 'Malnutrition Risk' },
  { id: 'nutrient',     label: 'Nutrient Depletion' },
  { id: 'med-nutrition',label: 'Med-Nutrition' },
];

export default function DietitianPage() {
  const [ov,     setOv]     = useState(null);
  const [keto,   setKeto]   = useState(null);
  const [maln,   setMaln]   = useState(null);
  const [nutr,   setNutr]   = useState(null);
  const [medNut, setMedNut] = useState(null);
  const [tab,    setTab]    = useState('overview');
  const [sel,    setSel]    = useState(null);
  const [err,    setErr]    = useState(null);

  useEffect(() => {
    fetch(`${API}/api/dietitian`).then(r => r.json()).then(setOv).catch(e => setErr(e.message));
    fetch(`${API}/api/dietitian/ketogenic`).then(r => r.json()).then(setKeto).catch(() => {});
    fetch(`${API}/api/dietitian/malnutrition`).then(r => r.json()).then(setMaln).catch(() => {});
    fetch(`${API}/api/dietitian/nutrient`).then(r => r.json()).then(setNutr).catch(() => {});
    fetch(`${API}/api/dietitian/medication-nutrition`).then(r => r.json()).then(setMedNut).catch(() => {});
  }, []);

  if (err) return <div className="p-4 alert alert-danger">Error: {err}</div>;
  if (!ov) return <div className="p-4"><div className="spinner-border text-success" /></div>;

  const ketoMod  = ov.modules?.ketogenic_diet_eligibility;
  const malnMod  = ov.modules?.malnutrition_risk;
  const nutrMod  = ov.modules?.nutrient_depletion;
  const medNMod  = ov.modules?.medication_nutrition;

  const kpis = [
    { label: 'Patients Assessed',     value: ketoMod?.unique_patients ?? maln?.unique_patients ?? nutr?.unique_patients ?? '—' },
    { label: 'Keto Eligible',         value: ketoMod?.cohort_summary?.category_distribution?.find(c => c.category?.startsWith('Good'))?.count ?? 0, color: 'success' },
    { label: 'High Maln Risk',        value: malnMod?.cohort_summary?.patients_needing_referral ?? maln?.cohort_summary?.patients_needing_referral ?? '—', color: 'danger' },
    { label: 'Nutrient Depletions',   value: nutrMod?.cohort_summary?.patients_with_depletions ?? nutr?.cohort_summary?.patients_with_depletions ?? '—', color: 'warning' },
    { label: 'Med-Nutrition Interactions', value: medNMod?.cohort_summary?.patients_with_interactions ?? medNut?.cohort_summary?.patients_with_interactions ?? '—', color: 'warning' },
    { label: 'Avg Keto Score',        value: ketoMod?.cohort_summary?.mean_eligibility_score?.toFixed(1) ?? '—', color: 'primary' },
  ];

  return (
    <div>
      <h3>&#x1f957; Clinical Dietitian / Nutritionist Dashboard</h3>
      <p className="text-muted small">
        Nutritional assessment for epilepsy patients — ketogenic diet eligibility, malnutrition risk,
        AED-specific nutrient depletion, and medication-nutrition interaction counseling.
      </p>

      {/* KPI row */}
      <div className="row mb-3">
        {kpis.map(k => (
          <div key={k.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${k.color || 'primary'}`}>{k.value}</div>
                <div className="text-muted" style={{ fontSize: '0.72rem' }}>{k.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div>
          <div className="alert alert-success border-0 shadow-sm mb-3">
            <strong>Role:</strong> {ov.role} &nbsp;|&nbsp; {ov.description}
          </div>

          <div className="row">
            {/* Ketogenic eligibility summary */}
            {ketoMod && (
              <div className="col-md-4 mb-3">
                <div className="card shadow-sm border-0">
                  <div className="card-header bg-success text-white py-2 small fw-bold">&#x1f957; Ketogenic Diet Eligibility</div>
                  <div className="card-body p-2">
                    <div className="text-center mb-2">
                      <div className="h3 mb-0 text-success">{ketoMod.unique_patients}</div>
                      <div className="small text-muted">Patients Screened</div>
                    </div>
                    {(ketoMod.cohort_summary?.category_distribution || []).map(c => {
                      const max = Math.max(...(ketoMod.cohort_summary.category_distribution.map(x => x.count)), 1);
                      const color = c.pct > 50 ? 'warning' : 'success';
                      return (
                        <div key={c.category} className="mb-2">
                          <div className="d-flex justify-content-between small mb-1">
                            <span className="text-truncate" style={{ maxWidth: '75%' }} title={c.category}>{c.category}</span>
                            <span className="fw-bold">{c.pct}%</span>
                          </div>
                          <div className="progress" style={{ height: '14px' }}>
                            <div className={`progress-bar bg-${color}`} style={{ width: `${(c.count / max) * 100}%` }}>
                              <span style={{ fontSize: '0.65rem' }}>{c.count}</span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                    <div className="text-muted small mt-2">Mean eligibility score: <strong>{ketoMod.cohort_summary?.mean_eligibility_score}</strong></div>
                  </div>
                </div>
              </div>
            )}

            {/* Malnutrition risk summary from ov.modules if present */}
            {malnMod && (
              <div className="col-md-4 mb-3">
                <div className="card shadow-sm border-0">
                  <div className="card-header bg-warning text-dark py-2 small fw-bold">&#x26a0;&#xfe0f; Malnutrition Risk Summary</div>
                  <div className="card-body p-2">
                    <div className="text-center mb-2">
                      <div className="h3 mb-0 text-danger">{malnMod.cohort_summary?.patients_needing_referral}</div>
                      <div className="small text-muted">Patients Needing Referral</div>
                    </div>
                    {(malnMod.cohort_summary?.risk_distribution || []).map(r => {
                      const max = Math.max(...(malnMod.cohort_summary.risk_distribution.map(x => x.count)), 1);
                      return (
                        <div key={r.level} className="d-flex align-items-center mb-1">
                          <span className={`badge bg-${riskColor(r.level)} me-2`} style={{ minWidth: '55px' }}>{r.level}</span>
                          <div className="progress flex-grow-1" style={{ height: '16px' }}>
                            <div className={`progress-bar bg-${riskColor(r.level)}`} style={{ width: `${(r.count / max) * 100}%` }}>
                              <span style={{ fontSize: '0.65rem' }}>{r.count}</span>
                            </div>
                          </div>
                          <span className="small ms-2 text-muted">{r.pct}%</span>
                        </div>
                      );
                    })}
                    <div className="text-muted small mt-2">Mean risk score: <strong>{malnMod.cohort_summary?.mean_risk_score?.toFixed(1)}</strong></div>
                  </div>
                </div>
              </div>
            )}

            {/* Nutrient depletion summary */}
            {nutrMod && (
              <div className="col-md-4 mb-3">
                <div className="card shadow-sm border-0">
                  <div className="card-header bg-info text-white py-2 small fw-bold">&#x1f48a; Nutrient Depletion by AED</div>
                  <div className="card-body p-2">
                    <div className="text-center mb-2">
                      <div className="h3 mb-0 text-warning">{nutrMod.cohort_summary?.patients_with_depletions}</div>
                      <div className="small text-muted">Patients with Depletions</div>
                    </div>
                    {(nutrMod.cohort_summary?.nutrient_gap_frequency || []).map(n => (
                      <div key={n.nutrient} className="mb-1 small">
                        <div className="d-flex justify-content-between">
                          <span className="text-truncate" style={{ maxWidth: '75%' }}>{n.nutrient}</span>
                          <span className="text-muted">{n.patients_affected} pts ({n.pct_of_cohort}%)</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Diet types reference */}
            {ketoMod?.diet_types_reference && (
              <div className="col-12 mb-3">
                <div className="card shadow-sm border-0">
                  <div className="card-header bg-success text-white py-2 small fw-bold">Ketogenic Diet Types — Reference</div>
                  <div className="card-body p-2">
                    <div className="table-responsive">
                      <table className="table table-sm mb-0">
                        <thead className="table-dark">
                          <tr>
                            <th>Diet Type</th>
                            <th>Ratio / Carb</th>
                            <th>Indication</th>
                            <th>Suitability</th>
                          </tr>
                        </thead>
                        <tbody>
                          {ketoMod.diet_types_reference.map(dt => (
                            <tr key={dt.name}>
                              <td className="small fw-bold">{dt.name}</td>
                              <td className="small">{dt.ratio || dt.carb_limit || dt.carb_pct || '—'}</td>
                              <td className="small">{dt.indication}</td>
                              <td className="small">{dt.suitability}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Ketogenic Diet ── */}
      {tab === 'ketogenic' && keto && (
        <div>
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-header bg-success text-white py-2 small fw-bold">&#x1f957; Ketogenic Diet Eligibility — Patient Profiles</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Patient</th><th>Eligibility Score</th><th>Category</th><th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {(keto.patient_profiles || []).map(p => {
                      const catColor = (p.category || '').startsWith('Good') ? 'success' :
                                       (p.category || '').startsWith('Weak') ? 'warning' :
                                       (p.category || '').startsWith('Not') ? 'secondary' : 'info';
                      return (
                        <>
                          <tr key={p.patient_id} className={sel === p.patient_id ? 'table-active' : ''}>
                            <td className="small fw-bold">{p.patient_id}</td>
                            <td className="small text-center">
                              <span className={`badge bg-${catColor}`}>{p.eligibility_score}</span>
                            </td>
                            <td className="small">{p.category}</td>
                            <td>
                              <button className="btn btn-outline-success btn-sm py-0 px-1"
                                      style={{ fontSize: '0.7rem' }}
                                      onClick={() => setSel(sel === p.patient_id ? null : p.patient_id)}>
                                {sel === p.patient_id ? 'Hide' : 'Detail'}
                              </button>
                            </td>
                          </tr>
                          {sel === p.patient_id && (
                            <tr key={`${p.patient_id}-d`}>
                              <td colSpan={4} className="bg-light">
                                <div className="p-2">
                                  <strong className="small">Criteria Breakdown</strong>
                                  <table className="table table-sm table-bordered mt-1 mb-0">
                                    <thead className="table-secondary"><tr><th>Criterion</th><th>Points</th><th>Detail</th></tr></thead>
                                    <tbody>
                                      {(p.criteria_breakdown || []).map((c, i) => (
                                        <tr key={i}>
                                          <td className="small">{c.criterion}</td>
                                          <td className="small text-center">{c.points}</td>
                                          <td className="small">{c.detail}</td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                  {p.dietitian_recommendation && (
                                    <div className="mt-2 alert alert-info py-1 px-2 mb-0 small">
                                      <strong>Recommendation:</strong> {p.dietitian_recommendation}
                                    </div>
                                  )}
                                </div>
                              </td>
                            </tr>
                          )}
                        </>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="card shadow-sm border-0">
            <div className="card-header bg-dark text-white py-2 small fw-bold">Diet Types Reference</div>
            <div className="card-body p-2">
              <div className="table-responsive">
                <table className="table table-sm mb-0">
                  <thead className="table-dark"><tr><th>Diet Type</th><th>Ratio/Carb</th><th>Indication</th><th>Suitability</th></tr></thead>
                  <tbody>
                    {(keto.diet_types_reference || []).map(dt => (
                      <tr key={dt.name}>
                        <td className="small fw-bold">{dt.name}</td>
                        <td className="small">{dt.ratio || dt.carb_limit || dt.carb_pct || '—'}</td>
                        <td className="small">{dt.indication}</td>
                        <td className="small">{dt.suitability}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Malnutrition Risk ── */}
      {tab === 'malnutrition' && maln && (
        <div>
          <div className="row mb-3">
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-warning text-dark py-2 small fw-bold">Risk Distribution</div>
                <div className="card-body p-3">
                  {(maln.cohort_summary?.risk_distribution || []).map(r => {
                    const max = Math.max(...(maln.cohort_summary.risk_distribution.map(x => x.count)), 1);
                    return (
                      <div key={r.level} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span className={`badge bg-${riskColor(r.level)}`}>{r.level}</span>
                          <span>{r.count} ({r.pct}%)</span>
                        </div>
                        <div className="progress" style={{ height: '16px' }}>
                          <div className={`progress-bar bg-${riskColor(r.level)}`}
                               style={{ width: `${(r.count / max) * 100}%` }} />
                        </div>
                      </div>
                    );
                  })}
                  <hr />
                  <div className="text-center small">
                    <div className="h4 text-warning">{maln.cohort_summary?.mean_risk_score?.toFixed(1)}</div>
                    <div className="text-muted">Mean Risk Score</div>
                    <div className="mt-1 text-danger fw-bold">{maln.cohort_summary?.patients_needing_referral} patients need referral</div>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-8 mb-3">
              <div className="small text-muted mb-2">Scoring: MNA/MUST-adapted — {maln.scoring}</div>
            </div>
          </div>

          <div className="card shadow-sm border-0">
            <div className="card-header bg-warning text-dark py-2 small fw-bold">Patient Malnutrition Profiles</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr><th>Patient</th><th>Age</th><th>Risk Score</th><th>Risk Level</th><th>Action</th><th></th></tr>
                  </thead>
                  <tbody>
                    {(maln.patient_profiles || []).map(p => (
                      <>
                        <tr key={p.patient_id} className={sel === `m-${p.patient_id}` ? 'table-active' : ''}>
                          <td className="small fw-bold">{p.patient_id}</td>
                          <td className="small">{p.age ?? '—'}</td>
                          <td className="small text-center"><span className={`badge bg-${riskColor(p.risk_level)}`}>{p.risk_score}</span></td>
                          <td><span className={`badge bg-${riskColor(p.risk_level)}`}>{p.risk_level}</span></td>
                          <td className="small">{p.recommended_action}</td>
                          <td>
                            {(p.risk_factors || []).length > 0 && (
                              <button className="btn btn-outline-warning btn-sm py-0 px-1" style={{ fontSize: '0.7rem' }}
                                      onClick={() => setSel(sel === `m-${p.patient_id}` ? null : `m-${p.patient_id}`)}>
                                {sel === `m-${p.patient_id}` ? 'Hide' : 'Factors'}
                              </button>
                            )}
                          </td>
                        </tr>
                        {sel === `m-${p.patient_id}` && (
                          <tr key={`m-${p.patient_id}-d`}>
                            <td colSpan={6} className="bg-light">
                              <div className="p-2">
                                <table className="table table-sm table-bordered mb-0">
                                  <thead className="table-secondary"><tr><th>Risk Factor</th><th>Points</th><th>Detail</th></tr></thead>
                                  <tbody>
                                    {(p.risk_factors || []).map((rf, i) => (
                                      <tr key={i}>
                                        <td className="small">{rf.factor}</td>
                                        <td className="small text-center">{rf.points}</td>
                                        <td className="small">{rf.detail}</td>
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
            </div>
          </div>
        </div>
      )}

      {/* ── Nutrient Depletion ── */}
      {tab === 'nutrient' && nutr && (
        <div>
          <div className="alert alert-info small mb-3">
            <strong>Evidence basis:</strong> {nutr.evidence_basis}
          </div>
          <div className="row mb-3">
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-info text-white py-2 small fw-bold">Cohort Summary</div>
                <div className="card-body p-3 text-center">
                  <div className="h3 mb-0 text-warning">{nutr.cohort_summary?.patients_with_depletions}</div>
                  <div className="small text-muted mb-2">Patients with Depletions</div>
                  <div className="h4 mb-0 text-danger">{nutr.cohort_summary?.patients_with_high_severity}</div>
                  <div className="small text-muted">High Severity</div>
                </div>
              </div>
            </div>
            <div className="col-md-8 mb-3">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-secondary text-white py-2 small fw-bold">Nutrient Gap Frequency</div>
                <div className="card-body p-2">
                  {(nutr.cohort_summary?.nutrient_gap_frequency || []).map(n => (
                    <div key={n.nutrient} className="d-flex align-items-center mb-2">
                      <span className="small me-2" style={{ minWidth: '180px' }}>{n.nutrient}</span>
                      <div className="progress flex-grow-1" style={{ height: '16px' }}>
                        <div className="progress-bar bg-warning" style={{ width: `${n.pct_of_cohort * 5}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{n.patients_affected}</span>
                        </div>
                      </div>
                      <span className="small ms-2 text-muted">{n.pct_of_cohort}%</span>
                    </div>
                  ))}
                  {!(nutr.cohort_summary?.nutrient_gap_frequency?.length) && (
                    <div className="text-muted small">No significant depletions detected.</div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm border-0">
            <div className="card-header bg-info text-white py-2 small fw-bold">Patient Depletion Profiles</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr><th>Patient</th><th>Medications</th><th>Depletions</th><th></th></tr>
                  </thead>
                  <tbody>
                    {(nutr.patient_profiles || []).filter(p => p.depletions_identified > 0).map(p => (
                      <>
                        <tr key={p.patient_id} className={sel === `n-${p.patient_id}` ? 'table-active' : ''}>
                          <td className="small fw-bold">{p.patient_id}</td>
                          <td className="small">{(p.medications || []).map(m => m.drug_name).join(', ')}</td>
                          <td><span className="badge bg-warning text-dark">{p.depletions_identified}</span></td>
                          <td>
                            <button className="btn btn-outline-info btn-sm py-0 px-1" style={{ fontSize: '0.7rem' }}
                                    onClick={() => setSel(sel === `n-${p.patient_id}` ? null : `n-${p.patient_id}`)}>
                              {sel === `n-${p.patient_id}` ? 'Hide' : 'Detail'}
                            </button>
                          </td>
                        </tr>
                        {sel === `n-${p.patient_id}` && (
                          <tr key={`n-${p.patient_id}-d`}>
                            <td colSpan={4} className="bg-light">
                              <div className="p-2">
                                <table className="table table-sm table-bordered mb-0">
                                  <thead className="table-secondary"><tr><th>Drug</th><th>Nutrient Depleted</th><th>Mechanism</th><th>Severity</th></tr></thead>
                                  <tbody>
                                    {(p.depletion_details || []).map((d, i) => (
                                      <tr key={i}>
                                        <td className="small">{d.medication}</td>
                                        <td className="small fw-bold">{d.nutrient_depleted}</td>
                                        <td className="small">{d.mechanism}</td>
                                        <td><span className={`badge bg-${sevColor(d.severity)}`}>{d.severity}</span></td>
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
            </div>
          </div>
        </div>
      )}

      {/* ── Medication-Nutrition ── */}
      {tab === 'med-nutrition' && medNut && (
        <div>
          <div className="alert alert-secondary small mb-3">
            <strong>Evidence basis:</strong> {medNut.evidence_basis}
          </div>
          <div className="row mb-3">
            <div className="col-md-3">
              <div className="card shadow-sm border-0 text-center">
                <div className="card-body py-3">
                  <div className="h3 mb-0 text-warning">{medNut.cohort_summary?.patients_with_interactions}</div>
                  <div className="small text-muted">Patients with Interactions</div>
                  <div className="h4 mt-2 mb-0 text-danger">{medNut.cohort_summary?.patients_with_high_severity}</div>
                  <div className="small text-muted">High Severity</div>
                </div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm border-0">
            <div className="card-header bg-secondary text-white py-2 small fw-bold">Patient Medication-Nutrition Profiles</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr><th>Patient</th><th>Medications</th><th>Interactions Found</th><th></th></tr>
                  </thead>
                  <tbody>
                    {(medNut.patient_profiles || []).filter(p => p.interactions_found > 0 || (p.dietary_counseling_points || []).length > 0).map(p => (
                      <>
                        <tr key={p.patient_id} className={sel === `mn-${p.patient_id}` ? 'table-active' : ''}>
                          <td className="small fw-bold">{p.patient_id}</td>
                          <td className="small">{(p.medications || []).map(m => m.drug_name).join(', ') || '—'}</td>
                          <td><span className={`badge bg-${p.high_severity_count > 0 ? 'danger' : p.interactions_found > 0 ? 'warning' : 'success'}`}>{p.interactions_found}</span></td>
                          <td>
                            {((p.interaction_details || []).length > 0 || (p.dietary_counseling_points || []).length > 0) && (
                              <button className="btn btn-outline-secondary btn-sm py-0 px-1" style={{ fontSize: '0.7rem' }}
                                      onClick={() => setSel(sel === `mn-${p.patient_id}` ? null : `mn-${p.patient_id}`)}>
                                {sel === `mn-${p.patient_id}` ? 'Hide' : 'Detail'}
                              </button>
                            )}
                          </td>
                        </tr>
                        {sel === `mn-${p.patient_id}` && (
                          <tr key={`mn-${p.patient_id}-d`}>
                            <td colSpan={4} className="bg-light">
                              <div className="p-2">
                                {(p.interaction_details || []).length > 0 && (
                                  <>
                                    <strong className="small">Interactions:</strong>
                                    <table className="table table-sm table-bordered mt-1 mb-2">
                                      <thead className="table-secondary"><tr><th>Drug</th><th>Interaction</th><th>Severity</th></tr></thead>
                                      <tbody>
                                        {p.interaction_details.map((d, i) => (
                                          <tr key={i}>
                                            <td className="small">{d.medication}</td>
                                            <td className="small">{d.interaction}</td>
                                            <td><span className={`badge bg-${sevColor(d.severity)}`}>{d.severity}</span></td>
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                  </>
                                )}
                                {(p.dietary_counseling_points || []).length > 0 && (
                                  <>
                                    <strong className="small">Dietary Counseling:</strong>
                                    <ul className="small mt-1 mb-0">
                                      {p.dietary_counseling_points.map((pt, i) => <li key={i}>{pt}</li>)}
                                    </ul>
                                  </>
                                )}
                              </div>
                            </td>
                          </tr>
                        )}
                      </>
                    ))}
                    {(medNut.patient_profiles || []).filter(p => p.interactions_found > 0 || (p.dietary_counseling_points || []).length > 0).length === 0 && (
                      <tr><td colSpan={4} className="text-muted small text-center py-3">No significant medication-nutrition interactions detected.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
