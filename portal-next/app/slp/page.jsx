'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const RISK_COLOR = r =>
  r === 'High' ? 'danger' : r === 'Moderate' ? 'warning' : r === 'Low' ? 'success' : 'secondary';

const RISK_BADGE = r =>
  r === 'High' ? 'bg-danger' : r === 'Moderate' ? 'bg-warning text-dark' : r === 'Low' ? 'bg-success' : 'bg-secondary';

function KPI({ label, value, sub, color = 'primary' }) {
  return (
    <div className={`card border-${color} h-100`}>
      <div className="card-body text-center p-3">
        <div className={`display-6 fw-bold text-${color}`}>{value}</div>
        <div className="small fw-semibold">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
      </div>
    </div>
  );
}

function RiskBar({ label, score, max = 10, riskLevel }) {
  const pct = max > 0 ? Math.round((score / max) * 100) : 0;
  const cls = riskLevel === 'High' ? 'danger' : riskLevel === 'Moderate' ? 'warning' : 'success';
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className={`badge ${RISK_BADGE(riskLevel)}`}>{riskLevel} ({score}/{max})</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className={`progress-bar bg-${cls}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// ── Overview Tab ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading...</div>;
  const s = data.summary || {};
  const la = data.language_assessment || {};
  const sw = data.swallowing || {};
  const aed = data.aed_speech_effects || {};
  const th = data.therapy_goals || {};

  return (
    <div>
      <div className="alert alert-info py-2 mb-3">
        <strong>SLP — Speech-Language Pathology</strong>&nbsp;
        <span className="text-muted small">
          Language lateralization · Dysphagia screening · AED speech effects · Therapy tracking —
          Helmstaedter &amp; Witt 2017 · Bell et al. 2011 · ASHA Practice Portal
        </span>
      </div>

      {/* KPI row */}
      <div className="row g-3 mb-4">
        <div className="col-6 col-md-3">
          <KPI label="Total Patients" value={s.total_patients ?? '—'} color="primary" />
        </div>
        <div className="col-6 col-md-3">
          <KPI label="Language Assessed" value={s.patients_with_language_assessments ?? '—'}
               sub={`of ${s.total_patients ?? '?'} patients`} color="info" />
        </div>
        <div className="col-6 col-md-3">
          <KPI label="Dysphagia Risk" value={(s.high_dysphagia_risk ?? 0) + (s.moderate_dysphagia_risk ?? 0)}
               sub={`High: ${s.high_dysphagia_risk ?? 0} · Mod: ${s.moderate_dysphagia_risk ?? 0}`} color="warning" />
        </div>
        <div className="col-6 col-md-3">
          <KPI label="Active Goals" value={s.active_therapy_goals ?? '—'}
               sub={`of ${s.total_therapy_goals ?? '?'} total goals`} color="success" />
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-6 col-md-3">
          <KPI label="Language Impaired" value={s.patients_with_language_impairment ?? 0}
               sub="BNT/fluency below cutoff" color={s.patients_with_language_impairment > 0 ? 'danger' : 'secondary'} />
        </div>
        <div className="col-6 col-md-3">
          <KPI label="High AED Risk" value={s.high_aed_speech_risk ?? 0}
               sub="speech/language side effects" color={s.high_aed_speech_risk > 0 ? 'danger' : 'secondary'} />
        </div>
        <div className="col-6 col-md-3">
          <KPI label="High Cog-Comm Risk" value={s.high_cognitive_comm_risk ?? 0}
               sub="word-finding/discourse" color={s.high_cognitive_comm_risk > 0 ? 'danger' : 'secondary'} />
        </div>
        <div className="col-6 col-md-3">
          <KPI label="On Topiramate" value={s.patients_on_topiramate ?? 0}
               sub="high word-finding AED risk" color={s.patients_on_topiramate > 0 ? 'warning' : 'secondary'} />
        </div>
      </div>

      {/* Four domain summary cards */}
      <div className="row g-3">
        <div className="col-md-6">
          <div className="card h-100">
            <div className="card-header bg-info text-white fw-semibold">Language Assessment</div>
            <div className="card-body">
              <p className="small mb-1">{la.title}</p>
              <p className="text-muted small">{la.subtitle}</p>
              <ul className="list-unstyled small mb-0">
                <li>Patients assessed: <strong>{la.patients_with_assessments ?? '—'}</strong></li>
                <li>With impairment: <strong className={la.patients_with_impairment > 0 ? 'text-danger' : 'text-success'}>{la.patients_with_impairment ?? 0}</strong></li>
                <li>Instruments: BNT · Token Test · Phonemic Fluency · Semantic Fluency</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card h-100">
            <div className="card-header bg-warning text-dark fw-semibold">Swallowing / Dysphagia</div>
            <div className="card-body">
              <p className="small mb-1">{sw.title}</p>
              <p className="text-muted small">{sw.subtitle}</p>
              <ul className="list-unstyled small mb-0">
                <li>High risk: <strong className="text-danger">{sw.high_risk_count ?? 0}</strong></li>
                <li>Moderate risk: <strong className="text-warning">{sw.moderate_risk_count ?? 0}</strong></li>
                <li>Low risk: <strong className="text-success">{sw.low_risk_count ?? 0}</strong></li>
              </ul>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card h-100">
            <div className="card-header bg-danger text-white fw-semibold">AED Speech Effects</div>
            <div className="card-body">
              <p className="small mb-1">{aed.title}</p>
              <p className="text-muted small">{aed.subtitle}</p>
              <ul className="list-unstyled small mb-0">
                <li>High risk: <strong className="text-danger">{aed.high_risk_count ?? 0}</strong></li>
                <li>Moderate risk: <strong className="text-warning">{aed.moderate_risk_count ?? 0}</strong></li>
                <li>On topiramate: <strong>{aed.patients_on_topiramate ?? 0}</strong> (word-finding)</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card h-100">
            <div className="card-header bg-success text-white fw-semibold">Therapy Goals</div>
            <div className="card-body">
              <p className="small mb-1">{th.title}</p>
              <p className="text-muted small">{th.subtitle}</p>
              <ul className="list-unstyled small mb-0">
                <li>Total goals: <strong>{th.total_goals ?? 0}</strong></li>
                <li>Active: <strong className="text-success">{th.total_active_goals ?? 0}</strong></li>
                <li>Domains: Swallowing · Language · Cognitive-Communication</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Epilepsy relevance */}
      <div className="card mt-4">
        <div className="card-header fw-semibold">Epilepsy Relevance — SLP in Neurological Care</div>
        <div className="card-body">
          <div className="row g-2">
            {[
              ['Temporal Lobe Epilepsy', 'TLE surgery risks language lateralization (Broca/Wernicke); pre-surgical Wada test / fMRI language mapping. Post-op naming decline in 30–40% (Bell et al. 2011).'],
              ['Post-Ictal Aphasia', 'Transient aphasia/anomia after seizures (especially left temporal); SLP distinguishes ictal vs structural language impairment.'],
              ['AED Word-Finding', 'Topiramate → word-finding difficulty in 30–40% of patients (Perucca & Gilliam 2012). SLP monitors and provides compensatory strategies.'],
              ['Post-Ictal Aspiration', 'Impaired swallowing and aspiration risk in post-ictal period; SLP provides bedside evaluation and caregiver education.'],
              ['Cognitive-Communication', 'Epilepsy-related cognitive effects (attention, memory, processing speed) impact communication; SLP addresses functional communication goals.'],
            ].map(([title, text]) => (
              <div key={title} className="col-md-6">
                <div className="p-2 border rounded bg-light h-100">
                  <div className="small fw-semibold text-primary mb-1">{title}</div>
                  <div className="text-muted" style={{ fontSize: 11 }}>{text}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Language Assessment Tab ───────────────────────────────────────────────────
function LanguageTab({ data }) {
  const [expanded, setExpanded] = useState(null);
  if (!data) return <div className="text-center py-5 text-muted">Loading...</div>;
  const patients = data.patients || [];
  const refs = data.normative_references || {};

  return (
    <div>
      <div className="alert alert-info py-2 mb-3">
        <strong>Language Assessment</strong> — BNT · Token Test · Phonemic &amp; Semantic Fluency
        <span className="text-muted small ms-2">Bell et al. 2011 · Hamberger &amp; Cole 2011 · Helmstaedter &amp; Witt 2017</span>
      </div>

      {/* Normative reference table */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">Normative Reference Ranges</div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Instrument</th><th>Normal</th><th>Impaired</th><th>Sensitivity</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(refs).map(([key, r]) => (
                <tr key={key}>
                  <td className="fw-semibold small">{r.full_name}</td>
                  <td className="text-success small">
                    {r.normal_cutoff ? `≥ ${r.normal_cutoff}` : r.normal_range ? `${r.normal_range[0]}–${r.normal_range[1]}` : '—'}
                  </td>
                  <td className="text-danger small">
                    {r.impaired_cutoff ? `< ${r.impaired_cutoff}` : '—'}
                  </td>
                  <td className="text-muted small">{r.description?.slice(0, 60)}...</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Patient table */}
      <div className="card">
        <div className="card-header fw-semibold">
          Per-Patient Language Profiles ({patients.length} patients)
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Patient</th><th>Age</th><th>Lateralization</th>
                <th>BNT (/60)</th><th>Token Test (/36)</th>
                <th>Phon. Fluency</th><th>Sem. Fluency</th><th>Impaired</th><th></th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <>
                  <tr key={p.patient_id} className={p.has_language_assessments ? '' : 'text-muted'}>
                    <td className="fw-semibold small">{p.patient_id}</td>
                    <td className="small">{p.age}</td>
                    <td className="small">
                      <span className="badge bg-secondary">{p.lateralization}</span>
                    </td>
                    <td className="small">
                      {p.boston_naming_test !== null
                        ? <span className={p.boston_naming_test < 38 ? 'text-danger fw-bold' : p.boston_naming_test < 48 ? 'text-warning fw-bold' : 'text-success'}>{p.boston_naming_test}</span>
                        : <span className="text-muted">—</span>}
                    </td>
                    <td className="small">
                      {p.token_test !== null
                        ? <span className={p.token_test < 22 ? 'text-danger fw-bold' : p.token_test < 29 ? 'text-warning fw-bold' : 'text-success'}>{p.token_test}</span>
                        : <span className="text-muted">—</span>}
                    </td>
                    <td className="small">
                      {p.phonemic_fluency !== null
                        ? <span className={p.phonemic_fluency < 20 ? 'text-danger fw-bold' : p.phonemic_fluency < 30 ? 'text-warning fw-bold' : 'text-success'}>{p.phonemic_fluency}</span>
                        : <span className="text-muted">—</span>}
                    </td>
                    <td className="small">
                      {p.semantic_fluency !== null
                        ? <span className={p.semantic_fluency < 10 ? 'text-danger fw-bold' : p.semantic_fluency < 15 ? 'text-warning fw-bold' : 'text-success'}>{p.semantic_fluency}</span>
                        : <span className="text-muted">—</span>}
                    </td>
                    <td>
                      <span className={`badge ${p.impaired_domain_count > 0 ? 'bg-danger' : 'bg-success'}`}>
                        {p.impaired_domain_count > 0 ? `${p.impaired_domain_count} domain(s)` : 'None'}
                      </span>
                    </td>
                    <td>
                      {p.has_language_assessments && (
                        <button className="btn btn-sm btn-outline-primary py-0 px-1"
                          onClick={() => setExpanded(expanded === p.patient_id ? null : p.patient_id)}>
                          {expanded === p.patient_id ? '▲' : '▼'}
                        </button>
                      )}
                    </td>
                  </tr>
                  {expanded === p.patient_id && (
                    <tr key={`${p.patient_id}-exp`}>
                      <td colSpan={9} className="bg-light p-3">
                        <div className="small fw-semibold mb-1">Recommendations:</div>
                        <ul className="mb-1 small">
                          {(p.recommendations || []).map((r, i) => <li key={i}>{r}</li>)}
                        </ul>
                        {p.lateralization_note && (
                          <div className="text-muted small mt-1"><em>{p.lateralization_note}</em></div>
                        )}
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
  );
}

// ── Swallowing Tab ────────────────────────────────────────────────────────────
function SwallowingTab({ data }) {
  const [expanded, setExpanded] = useState(null);
  if (!data) return <div className="text-center py-5 text-muted">Loading...</div>;
  const patients = data.patients || [];

  return (
    <div>
      <div className="alert alert-warning py-2 mb-3">
        <strong>Dysphagia Screening</strong> — Risk score 0–10: post-ictal aspiration · AED sedation · VNS ·
        elderly polytherapy · seizure injury · intellectual disability
      </div>

      {/* Summary KPIs */}
      <div className="row g-3 mb-4">
        <div className="col-4">
          <KPI label="High Risk" value={data.high_risk_count ?? 0} color="danger" sub="score ≥ 6" />
        </div>
        <div className="col-4">
          <KPI label="Moderate Risk" value={data.moderate_risk_count ?? 0} color="warning" sub="score 3–5" />
        </div>
        <div className="col-4">
          <KPI label="Low Risk" value={data.low_risk_count ?? 0} color="success" sub="score 0–2" />
        </div>
      </div>

      {/* Risk factor legend */}
      {data.risk_factors_legend && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">Dysphagia Risk Factor Reference</div>
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr><th>Factor</th><th>Score</th><th>Clinical Basis</th></tr>
              </thead>
              <tbody>
                {Object.entries(data.risk_factors_legend).map(([key, f]) => (
                  <tr key={key}>
                    <td className="small fw-semibold">{f.label}</td>
                    <td><span className="badge bg-warning text-dark">{f.score}</span></td>
                    <td className="small text-muted">{f.rationale}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Patient table */}
      <div className="card">
        <div className="card-header fw-semibold">Per-Patient Dysphagia Risk ({patients.length} patients)</div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Patient</th><th>Age</th><th>AEDs</th><th>Seizures</th>
                <th>Risk Score</th><th>Risk Level</th><th>Action</th><th></th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <>
                  <tr key={p.patient_id}>
                    <td className="fw-semibold small">{p.patient_id}</td>
                    <td className="small">{p.age}</td>
                    <td className="small">{p.aed_count}</td>
                    <td className="small">{p.seizure_count}</td>
                    <td className="small">
                      <strong>{p.dysphagia_risk_score}</strong>/{p.dysphagia_risk_max}
                    </td>
                    <td>
                      <span className={`badge ${RISK_BADGE(p.risk_level)}`}>{p.risk_level}</span>
                    </td>
                    <td className="small text-muted" style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {p.recommended_action}
                    </td>
                    <td>
                      <button className="btn btn-sm btn-outline-secondary py-0 px-1"
                        onClick={() => setExpanded(expanded === p.patient_id ? null : p.patient_id)}>
                        {expanded === p.patient_id ? '▲' : '▼'}
                      </button>
                    </td>
                  </tr>
                  {expanded === p.patient_id && (
                    <tr key={`${p.patient_id}-exp`}>
                      <td colSpan={8} className="bg-light p-3">
                        <div className="small fw-semibold mb-1">Risk Factors:</div>
                        <div className="row g-2">
                          {Object.entries(p.factors || {}).map(([key, f]) => (
                            <div key={key} className="col-md-4">
                              <div className={`p-2 border rounded ${f.present ? 'border-warning bg-warning bg-opacity-10' : ''}`}>
                                <div className="small fw-semibold text-capitalize">{key.replace(/_/g, ' ')}</div>
                                <div className="text-muted" style={{ fontSize: 11 }}>{f.value}</div>
                                {f.present && <span className="badge bg-warning text-dark">+{f.score}</span>}
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="mt-2 small text-muted"><em>{p.recommended_action}</em></div>
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
  );
}

// ── AED Speech Effects Tab ────────────────────────────────────────────────────
function AEDSpeechTab({ data }) {
  const [expanded, setExpanded] = useState(null);
  if (!data) return <div className="text-center py-5 text-muted">Loading...</div>;
  const patients = data.patients || [];
  const catalog = data.aed_effects_catalog || {};

  return (
    <div>
      <div className="alert alert-danger py-2 mb-3">
        <strong>AED Speech &amp; Language Side Effects</strong> — Topiramate · Valproate · Phenytoin ·
        Carbamazepine · Levetiracetam monitoring
        <span className="text-muted small ms-2">Perucca &amp; Gilliam 2012 · Mula &amp; Trimble 2009</span>
      </div>

      {/* Summary KPIs */}
      <div className="row g-3 mb-4">
        <div className="col-4">
          <KPI label="High Risk" value={data.high_risk_count ?? 0} color="danger" sub="multiple high-risk AEDs" />
        </div>
        <div className="col-4">
          <KPI label="Moderate Risk" value={data.moderate_risk_count ?? 0} color="warning" />
        </div>
        <div className="col-4">
          <KPI label="On Topiramate" value={data.patients_on_topiramate ?? 0} color="warning" sub="word-finding risk" />
        </div>
      </div>

      {/* AED effects catalog */}
      {Object.keys(catalog).length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">AED Speech/Language Side Effect Catalog</div>
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr><th>Drug</th><th>Risk</th><th>Primary Effect</th><th>Effects</th><th>Monitoring</th></tr>
              </thead>
              <tbody>
                {Object.entries(catalog).map(([drug, info]) => (
                  <tr key={drug}>
                    <td className="fw-semibold small text-capitalize">{drug}</td>
                    <td><span className={`badge ${RISK_BADGE(info.risk_level)}`}>{info.risk_level}</span></td>
                    <td className="small">{info.primary_effect}</td>
                    <td className="small">{(info.effects || []).join(', ')}</td>
                    <td className="small text-muted">{info.monitoring}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Patient table */}
      <div className="card">
        <div className="card-header fw-semibold">Per-Patient AED Speech Risk ({patients.length} patients)</div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Patient</th><th>Age</th><th>AEDs</th><th>High-Risk Drugs</th>
                <th>Overall Risk</th><th>Additive</th><th></th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <>
                  <tr key={p.patient_id}>
                    <td className="fw-semibold small">{p.patient_id}</td>
                    <td className="small">{p.age}</td>
                    <td className="small">{p.aed_names?.join(', ') || '—'}</td>
                    <td className="small">{p.high_risk_drugs?.join(', ') || '—'}</td>
                    <td><span className={`badge ${RISK_BADGE(p.overall_speech_risk)}`}>{p.overall_speech_risk}</span></td>
                    <td className="small">{p.additive_risk ? <span className="badge bg-warning text-dark">Yes</span> : '—'}</td>
                    <td>
                      <button className="btn btn-sm btn-outline-secondary py-0 px-1"
                        onClick={() => setExpanded(expanded === p.patient_id ? null : p.patient_id)}>
                        {expanded === p.patient_id ? '▲' : '▼'}
                      </button>
                    </td>
                  </tr>
                  {expanded === p.patient_id && (
                    <tr key={`${p.patient_id}-exp`}>
                      <td colSpan={7} className="bg-light p-3">
                        <div className="small fw-semibold mb-1">Drug-Specific Effects:</div>
                        {(p.drug_speech_effects || []).map((e, i) => (
                          <div key={i} className="mb-1 p-2 border rounded">
                            <span className="fw-semibold text-capitalize">{e.drug}</span>
                            <span className={`badge ms-2 ${RISK_BADGE(e.risk_level)}`}>{e.risk_level}</span>
                            <span className="small text-muted ms-2">{e.primary_effect}</span>
                          </div>
                        ))}
                        <div className="mt-2 small">
                          <strong>Recommendations:</strong>
                          <ul className="mb-0">
                            {(p.recommendations || []).map((r, i) => <li key={i} className="text-muted">{r}</li>)}
                          </ul>
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
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading...</div>;
  const mods = data.modules || {};
  const refs = data.clinical_references || [];

  return (
    <div>
      {Object.entries(mods).map(([modKey, mod]) => (
        <div key={modKey} className="card mb-3">
          <div className="card-header fw-semibold text-capitalize">{modKey.replace(/_/g, ' ')}</div>
          <div className="card-body">
            {mod.metrics && (
              <div className="table-responsive mb-3">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Metric</th><th>Normal</th><th>Impaired</th><th>Description</th><th>Reference</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(mod.metrics).map(([mKey, m]) => (
                      <tr key={mKey}>
                        <td className="small fw-semibold">{m.full_name || mKey}</td>
                        <td className="small text-success">
                          {m.normal_cutoff ? `≥ ${m.normal_cutoff}` : m.normal_range ? `${m.normal_range[0]}–${m.normal_range[1]}` : '—'}
                        </td>
                        <td className="small text-danger">
                          {m.impaired_cutoff ? `< ${m.impaired_cutoff}` : '—'}
                        </td>
                        <td className="small text-muted">{m.description}</td>
                        <td className="small text-muted">{m.reference}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {mod.risk_factors && (
              <div className="table-responsive mb-3">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Risk Factor</th><th>Score</th><th>Rationale</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(mod.risk_factors).map(([key, f]) => (
                      <tr key={key}>
                        <td className="small fw-semibold">{f.label}</td>
                        <td><span className="badge bg-warning text-dark">{f.score}</span></td>
                        <td className="small text-muted">{f.rationale}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {mod.aed_effects && (
              <div className="table-responsive mb-3">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Drug</th><th>Risk</th><th>Primary Effect</th><th>Monitoring</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(mod.aed_effects).map(([drug, e]) => (
                      <tr key={drug}>
                        <td className="small fw-semibold text-capitalize">{drug}</td>
                        <td><span className={`badge ${RISK_BADGE(e.risk_level)}`}>{e.risk_level}</span></td>
                        <td className="small">{e.primary_effect}</td>
                        <td className="small text-muted">{e.monitoring}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {mod.goal_domains && (
              <div className="row g-2">
                {Object.entries(mod.goal_domains).map(([key, d]) => (
                  <div key={key} className="col-md-4">
                    <div className="p-2 border rounded bg-light h-100">
                      <div className="small fw-semibold">{d.label}</div>
                      <div className="text-muted" style={{ fontSize: 11 }}>{d.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      ))}

      {/* Clinical references */}
      {refs.length > 0 && (
        <div className="card mt-3">
          <div className="card-header fw-semibold">Clinical References</div>
          <ul className="list-group list-group-flush">
            {refs.map((r, i) => (
              <li key={i} className="list-group-item small">
                <strong>{r.authors} ({r.year}).</strong> {r.title}. <em>{r.journal}</em>. {r.relevance}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function SLPDashboard() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [language, setLanguage] = useState(null);
  const [swallowing, setSwallowing] = useState(null);
  const [aedSpeech, setAedSpeech] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/slp-expert/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 'language' && !language) {
      fetch(`${API}/api/slp-expert/language-assessment`)
        .then(r => r.json()).then(setLanguage).catch(e => setErr(String(e)));
    }
    if (tab === 'swallowing' && !swallowing) {
      fetch(`${API}/api/slp-expert/swallowing`)
        .then(r => r.json()).then(setSwallowing).catch(e => setErr(String(e)));
    }
    if (tab === 'aed' && !aedSpeech) {
      fetch(`${API}/api/slp-expert/aed-speech-effects`)
        .then(r => r.json()).then(setAedSpeech).catch(e => setErr(String(e)));
    }
    if (tab === 'definitions' && !definitions) {
      fetch(`${API}/api/slp-expert/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setErr(String(e)));
    }
  }, [tab]);

  const TABS = [
    { key: 'overview',    label: 'Overview' },
    { key: 'language',    label: 'Language Assessment' },
    { key: 'swallowing',  label: 'Swallowing / Dysphagia' },
    { key: 'aed',         label: 'AED Speech Effects' },
    { key: 'definitions', label: 'Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">Speech-Language Pathology (SLP) Dashboard</h4>
      <p className="text-muted small mb-3">
        Language lateralization · BNT · Token Test · Dysphagia screening · AED speech effects · Therapy goals —
        Epilepsy neurolinguistic assessment
      </p>

      {err && <div className="alert alert-danger">{err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.key} className="nav-item">
            <button className={`nav-link ${tab === t.key ? 'active' : ''}`} onClick={() => setTab(t.key)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewTab data={overview} />}
      {tab === 'language'    && <LanguageTab data={language} />}
      {tab === 'swallowing'  && <SwallowingTab data={swallowing} />}
      {tab === 'aed'         && <AEDSpeechTab data={aedSpeech} />}
      {tab === 'definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
