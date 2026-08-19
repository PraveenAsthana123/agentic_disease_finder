'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#2e3a00';   // dark-olive-green — CLN12 / Kufor-Rakeb / ATP13A2 Lysosomal Polyamine ATPase
const ACCENT2 = '#b71c1c';   // dark-red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep-orange — urgent warnings
const ACCENT4 = '#1b5e20';   // deep-green — safe treatments / monitoring
const ACCENT5 = '#0277bd';   // steel-blue — unique distinction (VGB NOT absolute CI, like CLN13)
const ACCENT6 = '#6a1b9a';   // deep-purple — antipsychotic CI (unique CLN12 danger)

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function PctBar({ label, pct, color = ACCENT }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header fw-bold" style={{ backgroundColor: '#f1f8e9', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

export default function CLN12Page() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/cln12/overview`).then(r => r.json()).then(setOv).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 && !bk) fetch(`${API}/api/cln12/breakdown`).then(r => r.json()).then(setBk).catch(e => setErr(String(e)));
    if (tab === 4 && !df) fetch(`${API}/api/cln12/definitions`).then(r => r.json()).then(setDf).catch(e => setErr(String(e)));
    if ((tab === 2 || tab === 3) && !bk) fetch(`${API}/api/cln12/breakdown`).then(r => r.json()).then(setBk).catch(e => setErr(String(e)));
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div className="rounded p-3 mb-3 text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #556b2f 100%)` }}>
        <h4 className="mb-1 fw-bold">🧬 CLN12 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 12</h4>
        <div style={{ fontSize: 13 }}>
          <strong>ATP13A2 (1p36.13)</strong> — Kufor-Rakeb Syndrome (KRS) · PARK9 · P5B-ATPase Lysosomal Polyamine Exporter · AR Biallelic LOF
          <span className="ms-3 badge bg-info text-dark">★ NO RETINAL NCL — VGB NOT Absolute CI (like CLN13)</span>
          <span className="ms-2 badge bg-warning text-dark">Juvenile Onset 6–25y</span>
          <span className="ms-2 badge" style={{ backgroundColor: ACCENT6 }}>⚠ Typical Antipsychotics ABSOLUTE CI</span>
        </div>
      </div>

      {/* DUAL UNIQUE DISTINCTION ALERTS */}
      <div className="alert py-2 mb-2" style={{ backgroundColor: '#e3f2fd', borderLeft: `5px solid ${ACCENT5}`, fontSize: 13 }}>
        <strong>★ CLN12 + CLN13 ARE THE ONLY NCLs WHERE VGB IS NOT AN ABSOLUTE CI:</strong> CLN12 does <strong>NOT</strong> cause
        retinal NCL degeneration (&lt;5%). VGB retinopathy does not compound retinal NCL blindness.
        Contrast with ALL other NCLs (CLN1/CLN2/CLN3/CLN5/CLN6/CLN7/CLN8/CLN10/CLN11) where VGB = ABSOLUTE CI.
      </div>
      <div className="alert py-2 mb-3" style={{ backgroundColor: '#f3e5f5', borderLeft: `5px solid ${ACCENT6}`, fontSize: 13 }}>
        <strong>⚠ UNIQUE CLN12 DANGER — TYPICAL ANTIPSYCHOTICS ABSOLUTE CI (NOT SHARED BY ANY OTHER NCL):</strong>{' '}
        CLN12 causes dopaminergic nigrostriatal degeneration. Haloperidol, chlorpromazine, metoclopramide, prochlorperazine
        → acute parkinsonism crisis. Psychosis in CLN12: <strong>clozapine or quetiapine ONLY</strong>.
        Alert ALL prescribers, A&amp;E, and GP at diagnosis.
      </div>

      {err && <div className="alert alert-danger py-2">{err}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && ov && (
        <div>
          <div className="row mb-3">
            <KPI label="Cohort" value={`${ov.cohort_size} pts`} color={ACCENT} />
            <KPI label="Mean Onset" value={`${ov.mean_onset_years}y`} color={ACCENT} />
            <KPI label="Parkinsonism" value={`${ov.parkinsonism_pct}%`} color={ACCENT6} />
            <KPI label="Seizures Present" value={`${ov.seizures_present_pct}%`} color={ACCENT2} />
            <KPI label="Retinal NCL" value={`${ov.retinal_degeneration_pct}%`} color={ACCENT5} />
            <KPI label="Diag. Delay" value={`${ov.mean_diagnosis_delay_years}y`} color={ACCENT2} />
          </div>
          <div className="row mb-3">
            <KPI label="L-DOPA Response" value={`${ov.ldopa_initial_response_pct}%`} color={ACCENT4} />
            <KPI label="L-DOPA Duration" value={`${ov.mean_ldopa_benefit_duration_years}y`} color={ACCENT3} />
            <KPI label="Gaze Palsy" value={`${ov.supranuclear_gaze_palsy_pct}%`} color={ACCENT} />
            <KPI label="Facial Myoclonus" value={`${ov.facial_faucial_finger_myoclonus_pct}%`} color={ACCENT} />
            <KPI label="Dense Deposits EM" value={`${ov.dense_deposits_fp_em_pct}%`} color={ACCENT3} />
            <KPI label="Cognitive Impair." value={`${ov.cognitive_impairment_pct}%`} color={ACCENT2} />
          </div>

          <SectionCard title="Gene, Protein & Disease" borderColor={ACCENT}>
            <p><strong>Gene:</strong> {ov.gene}</p>
            <p><strong>Protein:</strong> {ov.protein}</p>
            <p><strong>Inheritance:</strong> {ov.inheritance}</p>
            <p><strong>OMIM:</strong> {ov.omim}</p>
            <p><strong>Disease:</strong> {ov.disease}</p>
          </SectionCard>

          <SectionCard title="★ No Retinal NCL — VGB NOT Absolute CI (Shared with CLN13 Only)" borderColor={ACCENT5}>
            <Alert variant="info" text={ov.no_retinal_ncl} />
          </SectionCard>

          <SectionCard title="⚠ Juvenile Parkinsonism — Primary Entry Point; Supranuclear Gaze Palsy Pathognomonic" borderColor={ACCENT6}>
            <Alert variant="info" text={ov.juvenile_parkinsonism} />
          </SectionCard>

          <SectionCard title="⚠ Unique CLN12 Danger — Typical Antipsychotics Absolute CI (No Other NCL Has This)" borderColor={ACCENT6}>
            <Alert variant="warning" text={ov.unique_antipsychotic_ci} />
          </SectionCard>

          <SectionCard title="No ATP13A2 Enzyme Assay — WES Required" borderColor={ACCENT3}>
            <Alert variant="warning" text={ov.no_atp13a2_enzyme_assay} />
          </SectionCard>

          <SectionCard title="Pathomechanism" borderColor={ACCENT}>
            <p>{ov.mechanism}</p>
          </SectionCard>

          <SectionCard title="α-Synuclein Aggregation — CLN12 at NCL/Parkinson Intersection" borderColor={ACCENT6}>
            <p style={{ fontSize: 13 }}>{ov.alpha_synuclein_link}</p>
          </SectionCard>

          <SectionCard title="Polyamine Biology — Unique CLN12 NCL Mechanism" borderColor={ACCENT}>
            <p style={{ fontSize: 13 }}>{ov.alpha_synuclein_link}</p>
          </SectionCard>

          <SectionCard title="No Disease-Modifying Therapy / Gene Therapy Research" borderColor={ACCENT3}>
            <Alert variant="warning" text={ov.no_disease_modifying_therapy} />
          </SectionCard>

          <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
            {ov.key_pharmacological_distinctions && Object.entries(ov.key_pharmacological_distinctions).map(([k, v]) => (
              <div key={k} className="mb-3 p-2 rounded" style={{ backgroundColor: '#fafafa', border: '1px solid #eee' }}>
                <div className="fw-bold small mb-1" style={{ color: k.includes('ANTIPSYCHOTIC') ? ACCENT6 : ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ fontSize: 13 }}>{v}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & ETIOLOGY ── */}
      {tab === 1 && bk && (
        <div>
          <SectionCard title="Genotype Distribution — 40-Patient Cohort" borderColor={ACCENT}>
            <div className="row">
              <div className="col-md-5">
                {bk.etiologies?.map((e, i) => (
                  <PctBar key={i} label={e.class} pct={e.pct}
                    color={i === 0 ? ACCENT : i === 1 ? '#558b2f' : i === 2 ? ACCENT3 : i === 3 ? ACCENT2 : '#546e7a'} />
                ))}
              </div>
              <div className="col-md-7">
                {bk.etiologies?.map((e, i) => (
                  <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT}22`, backgroundColor: '#f1f8e911' }}>
                    <div className="fw-bold small" style={{ color: ACCENT }}>{e.class} — {e.pct}% (n={e.count})</div>
                    <div style={{ fontSize: 12 }}>{e.description}</div>
                    <div className="mt-1 small text-muted"><strong>Mechanism:</strong> {e.gene_mechanism}</div>
                    {e.key_variants && (
                      <div className="mt-1 d-flex flex-wrap gap-1">
                        {e.key_variants.map((v, j) => <span key={j} className="badge" style={{ backgroundColor: ACCENT, fontSize: 10 }}>{v}</span>)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </SectionCard>
        </div>
      )}

      {/* ── TAB 2: SEIZURES & TRIGGERS ── */}
      {tab === 2 && bk && (
        <div>
          <SectionCard title="Seizure Types — CLN12 (35% have seizures)" borderColor={ACCENT}>
            {bk.seizures?.map((s, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT}33`, backgroundColor: '#f1f8e908' }}>
                <div className="d-flex justify-content-between align-items-start">
                  <div className="fw-bold" style={{ color: ACCENT }}>{s.type}</div>
                  <span className="badge ms-2" style={{ backgroundColor: ACCENT, minWidth: 45 }}>{s.pct}%</span>
                </div>
                <div className="small mt-1"><strong>EEG:</strong> {s.eeg_signature}</div>
                <div className="small mt-1"><strong>Semiology:</strong> {s.semiology}</div>
                <div className="small mt-1 p-1 rounded" style={{ backgroundColor: '#fff9c4' }}>
                  <strong>Clinical Tip:</strong> {s.clinical_tip}
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Seizure Triggers — CLN12" borderColor={ACCENT3}>
            {bk.triggers?.map((t, i) => (
              <div key={i} className="d-flex align-items-start mb-2">
                <div className="me-3 text-center" style={{ minWidth: 45 }}>
                  <div className="fw-bold" style={{ color: t.pct === 100 ? ACCENT6 : ACCENT3, fontSize: 16 }}>{t.pct}%</div>
                </div>
                <div>
                  <div className="fw-bold small">{t.trigger}</div>
                  <div className="text-muted small">{t.note}</div>
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
      )}

      {/* ── TAB 3: TREATMENTS ── */}
      {tab === 3 && bk && (
        <div>
          <Alert variant="info"
            text="★ CLN12 DUAL UNIQUE RULES: (1) VGB is NOT an absolute CI (no retinal NCL — shared with CLN13 only). (2) TYPICAL ANTIPSYCHOTICS ARE ABSOLUTE CI (unique to CLN12 — parkinsonism). CBZ/OXC/PHT: ABSOLUTE CI (myoclonus + parkinsonism double trap). POLG1 + MERRF exclusion mandatory before VPA. L-DOPA for parkinsonism; VPA + LEV for seizures." />

          <SectionCard title="Recommended Treatments — Evidence Levels" borderColor={ACCENT4}>
            {bk.treatments?.map((t, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT4}33`, backgroundColor: '#f1f8e908' }}>
                <div className="d-flex justify-content-between align-items-start">
                  <div className="fw-bold" style={{ color: ACCENT4 }}>{t.drug}</div>
                  <span className="badge ms-2" style={{ backgroundColor: ACCENT4 }}>{t.level}</span>
                </div>
                <div className="small mt-1"><strong>Role:</strong> {t.role}</div>
                <div className="small mt-1"><strong>Dose:</strong> {t.dose}</div>
                <div className="small mt-1"><strong>MOA:</strong> {t.moa}</div>
                <div className="small mt-1"><strong>Efficacy:</strong> {t.efficacy}</div>
                <div className="small mt-1"><strong>Monitoring:</strong> {t.monitoring}</div>
                <div className="small mt-1 p-1 rounded" style={{ backgroundColor: '#e8f5e9' }}>
                  <strong>CLN12 Note:</strong> {t.cln12_note}
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Contraindications — CLN12" borderColor={ACCENT2}>
            {bk.contraindications?.map((c, i) => {
              const isAntipsychotic = c.severity.includes('Parkinsonism');
              const isCaution = c.severity.includes('CAUTION');
              return (
                <div key={i} className="mb-3 p-2 rounded"
                  style={{
                    border: `2px solid ${isAntipsychotic ? ACCENT6 : isCaution ? ACCENT5 : ACCENT2}`,
                    backgroundColor: isAntipsychotic ? '#f3e5f5' : isCaution ? '#e3f2fd' : '#ffebee'
                  }}>
                  <div className="d-flex justify-content-between align-items-start">
                    <div className="fw-bold" style={{ color: isAntipsychotic ? ACCENT6 : isCaution ? ACCENT5 : ACCENT2 }}>{c.drug}</div>
                    <span className="badge ms-2" style={{ backgroundColor: isAntipsychotic ? ACCENT6 : isCaution ? ACCENT5 : ACCENT2, fontSize: 10, whiteSpace: 'normal' }}>{c.severity}</span>
                  </div>
                  <div className="small mt-1"><strong>Reason:</strong> {c.reason}</div>
                  <div className="small mt-1 text-muted">{c.note}</div>
                </div>
              );
            })}
          </SectionCard>

          <SectionCard title="Monitoring — CLN12" borderColor={ACCENT}>
            {bk.monitoring?.map((m, i) => (
              <div key={i} className="d-flex align-items-start mb-2">
                <div className="me-3">
                  <span className="badge" style={{ backgroundColor: ACCENT, fontSize: 10, whiteSpace: 'normal', maxWidth: 80 }}>
                    {m.frequency}
                  </span>
                </div>
                <div>
                  <div className="fw-bold small">{m.item}</div>
                  <div className="text-muted small">{m.note}</div>
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
      )}

      {/* ── TAB 4: DEFINITIONS ── */}
      {tab === 4 && df && (
        <div>
          <SectionCard title="Disease Identity" borderColor={ACCENT}>
            <dl className="row mb-0" style={{ fontSize: 13 }}>
              <dt className="col-sm-3">Disease</dt><dd className="col-sm-9">{df.disease_name}</dd>
              <dt className="col-sm-3">Gene</dt><dd className="col-sm-9">{df.gene_full}</dd>
              <dt className="col-sm-3">OMIM Gene</dt><dd className="col-sm-9">{df.omim_gene}</dd>
              <dt className="col-sm-3">OMIM Disease</dt><dd className="col-sm-9">{df.omim_disease}</dd>
              <dt className="col-sm-3">Protein</dt><dd className="col-sm-9">{df.protein_full}</dd>
              <dt className="col-sm-3">Inheritance</dt><dd className="col-sm-9">{df.inheritance_mode}</dd>
              <dt className="col-sm-3">Onset</dt><dd className="col-sm-9">{df.onset_age}</dd>
              <dt className="col-sm-3">EM Pattern</dt><dd className="col-sm-9">{df.em_pattern}</dd>
              <dt className="col-sm-3">Retinal NCL</dt>
              <dd className="col-sm-9">
                <span className="badge bg-info text-dark">{df.no_retinal_ncl}</span>
              </dd>
            </dl>
          </SectionCard>

          <SectionCard title="Key Concepts (15)" borderColor={ACCENT}>
            {df.key_concepts?.map((c, i) => (
              <div key={i} className="mb-3 p-2 rounded"
                style={{
                  border: `1px solid ${ACCENT}33`,
                  backgroundColor: i === 2 ? '#f3e5f5' : i === 1 ? '#e3f2fd' : '#f1f8e908'
                }}>
                <div className="fw-bold small mb-1"
                  style={{ color: i === 2 ? ACCENT6 : i === 1 ? ACCENT5 : ACCENT }}>
                  {i + 1}. {c.name}
                </div>
                <div style={{ fontSize: 12 }}>{c.definition}</div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Clinical Thresholds (12)" borderColor={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead style={{ backgroundColor: ACCENT3, color: 'white' }}>
                  <tr><th>Parameter</th><th>Value</th><th>Action</th></tr>
                </thead>
                <tbody>
                  {df.thresholds?.map((t, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{t.parameter}</td>
                      <td><span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.value}</span></td>
                      <td>{t.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          <SectionCard title="Clinical Standards (12)" borderColor={ACCENT4}>
            <ol style={{ fontSize: 13 }}>
              {df.standards?.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
            </ol>
          </SectionCard>

          <SectionCard title="References (6)" borderColor={ACCENT}>
            <ol style={{ fontSize: 13 }}>
              {df.references?.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
            </ol>
          </SectionCard>

          <SectionCard title="Lifecycle Stages (6)" borderColor={ACCENT}>
            {df.lifecycle_stages?.map((s, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT}33` }}>
                <div className="fw-bold small" style={{ color: ACCENT }}>{i + 1}. {s.stage}</div>
                <div className="small text-muted mb-1">{s.age_range}</div>
                <div style={{ fontSize: 12 }}>{s.description}</div>
                {s.priorities && (
                  <ul className="mb-0 mt-1" style={{ fontSize: 11 }}>
                    {s.priorities.map((p, j) => <li key={j}>{p}</li>)}
                  </ul>
                )}
              </div>
            ))}
          </SectionCard>
        </div>
      )}
    </div>
  );
}
