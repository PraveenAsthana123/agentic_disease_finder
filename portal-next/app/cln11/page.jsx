'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#311b92';   // deep-indigo-plum — CLN11 / Adult NCL / GRN Progranulin
const ACCENT2 = '#b71c1c';   // dark-red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep-orange — urgent alerts / warnings
const ACCENT4 = '#1b5e20';   // deep-green — safe treatments / monitoring

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#ede7f6', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

export default function CLN11Page() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/cln11/overview`).then(r => r.json()).then(setOv).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 && !bk) fetch(`${API}/api/cln11/breakdown`).then(r => r.json()).then(setBk).catch(e => setErr(String(e)));
    if (tab === 4 && !df) fetch(`${API}/api/cln11/definitions`).then(r => r.json()).then(setDf).catch(e => setErr(String(e)));
    if ((tab === 2 || tab === 3) && !bk) fetch(`${API}/api/cln11/breakdown`).then(r => r.json()).then(setBk).catch(e => setErr(String(e)));
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div className="rounded p-3 mb-3 text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #4527a0 100%)` }}>
        <h4 className="mb-1 fw-bold">🔬 CLN11 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 11</h4>
        <div style={{ fontSize: 13 }}>
          <strong>GRN (17q21.31)</strong> · Progranulin Deficiency · Adult NCL · AR Biallelic LOF
          · Onset 15-35y · FP ± RP on EM · Plasma PGRN undetectable
        </div>
        <div className="mt-1 d-flex flex-wrap gap-2" style={{ fontSize: 11 }}>
          <span className="badge bg-danger">VGB ABSOLUTE CI — Retinal NCL 88%</span>
          <span className="badge bg-danger">CBZ/OXC/PHT ABSOLUTE CI — Myoclonus</span>
          <span className="badge bg-success">VPA SAFE — Lysosomal Regulator NOT Mitochondrial</span>
          <span className="badge bg-warning text-dark">No GRN Enzyme Assay → Plasma PGRN + WES</span>
          <span className="badge bg-info text-dark">Parents = GRN Heterozygotes → FTLD-TDP Risk</span>
        </div>
      </div>

      {/* Critical Alerts */}
      <Alert variant="danger" text="⛔ VGB ABSOLUTE CI: CLN11 retinal NCL (88%) + VGB retinopathy = catastrophic combined visual loss. Adult neurologists may prescribe VGB for focal seizures — ALWAYS check NCL status first." />
      <Alert variant="danger" text="⛔ CBZ / OXC / PHT ABSOLUTE CI: Young adult GTCS in CLN11 misidentified as idiopathic epilepsy → sodium channel blockers → acute myoclonic worsening. Mean diagnostic delay 4.2 years = years of CBZ risk." />
      <Alert variant="warning" text="⚠️ PARENT FTLD-TDP RISK — UNIQUE TO CLN11: Both parents of CLN11 proband are obligate GRN heterozygotes → FTLD (frontotemporal dementia) risk. Mandatory parental genetic counselling. Siblings: 25% CLN11 risk + 50% GRN carrier/FTLD risk." />
      <Alert variant="info" text="ℹ️ NO GRN ENZYME ASSAY: Progranulin is a lysosomal regulatory protein (not an enzyme). Rapid diagnostic: plasma PGRN level (<10 ng/mL = biallelic LOF). Then GRN WES + EM skin biopsy (FP ± RP confirms adult NCL)." />

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {err && <Alert variant="danger" text={`API error: ${err}`} />}

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <>
          {!ov && <div className="text-center py-4"><div className="spinner-border" style={{ color: ACCENT }} /></div>}
          {ov && (
            <>
              {/* KPIs */}
              <div className="row g-2 mb-3">
                <KPI label="Cohort" value={`${ov.cohort_size} pts`} color={ACCENT} />
                <KPI label="Mean Onset" value={`${ov.mean_onset_seizure_years}y`} color={ACCENT} />
                <KPI label="Dx Delay" value={`${ov.mean_diagnosis_delay_years}y`} color={ACCENT3} />
                <KPI label="Drug Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
                <KPI label="Retinal NCL" value={`${ov.retinal_degeneration_pct}%`} color={ACCENT2} />
                <KPI label="FP on EM" value={`${ov.fp_em_pct}%`} color={ACCENT} />
                <KPI label="Cognitive Imp." value={`${ov.cognitive_impairment_pct}%`} color={ACCENT3} />
                <KPI label="Parkinsonism" value={`${ov.parkinsonism_pct}%`} color={ACCENT3} />
                <KPI label="Ataxia" value={`${ov.ataxia_pct}%`} color={ACCENT3} />
                <KPI label="Photosensitivity" value={`${ov.photosensitivity_pct}%`} color={ACCENT} />
                <KPI label="On VPA" value={`${ov.on_vpa_pct}%`} color={ACCENT4} />
                <KPI label="Plasma PGRN↓" value={`${ov.plasma_pgrn_undetectable_pct}%`} color={ACCENT2} />
              </div>

              <div className="row">
                <div className="col-md-6">
                  <SectionCard title="Gene & Protein" borderColor={ACCENT}>
                    <p className="small mb-1"><strong>Gene:</strong> {ov.gene}</p>
                    <p className="small mb-1"><strong>Protein:</strong> {ov.protein}</p>
                    <p className="small mb-1"><strong>Inheritance:</strong> {ov.inheritance}</p>
                    <p className="small mb-0"><strong>OMIM:</strong> {ov.omim}</p>
                  </SectionCard>

                  <SectionCard title="Disease Overview" borderColor={ACCENT}>
                    <p className="small mb-1">{ov.disease}</p>
                    <p className="small mb-0"><strong>Mechanism:</strong> {ov.mechanism}</p>
                  </SectionCard>

                  <SectionCard title="No Disease-Modifying Therapy" borderColor={ACCENT3}>
                    <p className="small mb-0">{ov.no_disease_modifying_therapy}</p>
                  </SectionCard>
                </div>

                <div className="col-md-6">
                  <SectionCard title="⚠️ GRN/FTLD Carrier Risk (Parents)" borderColor={ACCENT2}>
                    <p className="small mb-0">{ov.grn_ftld_carrier_alert}</p>
                  </SectionCard>

                  <SectionCard title="Plasma PGRN — Rapid Diagnostic Biomarker" borderColor={ACCENT4}>
                    <p className="small mb-0">{ov.pgrn_plasma_diagnostic}</p>
                  </SectionCard>

                  <SectionCard title="Cohort Phenotype Distribution" borderColor={ACCENT}>
                    <PctBar label="Compound-Het Missense/Truncating" pct={ov.compound_het_missense_truncating_pct} color={ACCENT} />
                    <PctBar label="Homozygous Missense (Consanguineous)" pct={ov.homozygous_missense_pct} color={ACCENT} />
                    <PctBar label="Compound-Het Missense/Missense" pct={ov.compound_het_missense_missense_pct} color={'#5c6bc0'} />
                    <PctBar label="Homozygous Truncating" pct={ov.homozygous_truncating_pct} color={ACCENT2} />
                    <PctBar label="Promoter/Regulatory Variant" pct={ov.promoter_regulatory_pct} color={ACCENT3} />
                    <PctBar label="Phenocopy CLN11-Negative" pct={ov.phenocopy_negative_pct} color={'#9e9e9e'} />
                  </SectionCard>
                </div>
              </div>

              {/* Key Pharmacological Distinctions */}
              <SectionCard title="🔑 Key Pharmacological Distinctions (CLN11-Specific)" borderColor={ACCENT2}>
                <div className="row">
                  {Object.entries(ov.key_pharmacological_distinctions || {}).map(([k, v]) => (
                    <div key={k} className="col-md-6 mb-3">
                      <div className="border rounded p-2 h-100" style={{ borderColor: ACCENT + '40', backgroundColor: '#f3e5f5' }}>
                        <div className="fw-bold small mb-1" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
                        <div className="small">{v}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </SectionCard>
            </>
          )}
        </>
      )}

      {/* ── TAB 1: PATIENTS & ETIOLOGY ── */}
      {tab === 1 && (
        <>
          {!bk && <div className="text-center py-4"><div className="spinner-border" style={{ color: ACCENT }} /></div>}
          {bk && (
            <>
              <h5 className="fw-bold mb-3" style={{ color: ACCENT }}>CLN11 Etiological Classes (40 patients, 6 classes)</h5>
              {(bk.etiologies || []).map((et, i) => (
                <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT}` }}>
                  <div className="card-header d-flex justify-content-between align-items-center py-2" style={{ backgroundColor: '#ede7f6' }}>
                    <span className="fw-bold small" style={{ color: ACCENT }}>{et.class}</span>
                    <span className="badge text-white" style={{ backgroundColor: ACCENT }}>{et.pct}% · n={et.count}</span>
                  </div>
                  <div className="card-body py-2">
                    <PctBar label={et.class} pct={et.pct} color={ACCENT} />
                    <p className="small mb-1">{et.description}</p>
                    <p className="small mb-1"><strong>Mechanism:</strong> {et.gene_mechanism}</p>
                    <div className="d-flex flex-wrap gap-1 mt-1">
                      {(et.key_variants || []).map((v, j) => (
                        <span key={j} className="badge" style={{ backgroundColor: ACCENT + '20', color: ACCENT, fontSize: 10 }}>{v}</span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </>
          )}
        </>
      )}

      {/* ── TAB 2: SEIZURES & TRIGGERS ── */}
      {tab === 2 && (
        <>
          {!bk && <div className="text-center py-4"><div className="spinner-border" style={{ color: ACCENT }} /></div>}
          {bk && (
            <div className="row">
              <div className="col-md-7">
                <h5 className="fw-bold mb-3" style={{ color: ACCENT }}>Seizure Types (5 types)</h5>
                {(bk.seizures || []).map((s, i) => (
                  <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT}` }}>
                    <div className="card-header d-flex justify-content-between align-items-center py-2" style={{ backgroundColor: '#ede7f6' }}>
                      <span className="fw-bold small" style={{ color: ACCENT }}>{s.type}</span>
                      <span className="badge text-white" style={{ backgroundColor: ACCENT }}>{s.pct}%</span>
                    </div>
                    <div className="card-body py-2">
                      <PctBar label="Prevalence in CLN11" pct={s.pct} color={ACCENT} />
                      <p className="small mb-1"><strong>EEG:</strong> {s.eeg_signature}</p>
                      <p className="small mb-1"><strong>Semiology:</strong> {s.semiology}</p>
                      <div className="alert alert-info py-1 mb-0" style={{ fontSize: 12 }}>💡 {s.clinical_tip}</div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="col-md-5">
                <h5 className="fw-bold mb-3" style={{ color: ACCENT3 }}>Seizure Triggers (8 triggers)</h5>
                {(bk.triggers || []).map((t, i) => (
                  <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `3px solid ${ACCENT3}` }}>
                    <div className="card-body py-2">
                      <div className="d-flex justify-content-between align-items-center mb-1">
                        <span className="fw-bold small" style={{ color: ACCENT3 }}>{t.trigger}</span>
                        <span className="badge text-white" style={{ backgroundColor: ACCENT3 }}>{t.pct}%</span>
                      </div>
                      <PctBar label="" pct={t.pct} color={ACCENT3} />
                      <p className="small mb-0">{t.note}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* ── TAB 3: TREATMENTS ── */}
      {tab === 3 && (
        <>
          {!bk && <div className="text-center py-4"><div className="spinner-border" style={{ color: ACCENT }} /></div>}
          {bk && (
            <>
              <h5 className="fw-bold mb-3" style={{ color: ACCENT4 }}>Treatments (8 entries)</h5>
              <div className="row">
                {(bk.treatments || []).map((tx, i) => (
                  <div key={i} className="col-md-6 mb-3">
                    <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
                      <div className="card-header d-flex justify-content-between py-2" style={{ backgroundColor: '#e8f5e9' }}>
                        <span className="fw-bold small" style={{ color: ACCENT4 }}>{tx.drug}</span>
                        <span className="badge text-white" style={{ backgroundColor: ACCENT4 }}>{tx.level}</span>
                      </div>
                      <div className="card-body py-2">
                        <p className="small mb-1"><strong>Role:</strong> {tx.role}</p>
                        <p className="small mb-1"><strong>Dose:</strong> {tx.dose}</p>
                        <p className="small mb-1"><strong>MOA:</strong> {tx.moa}</p>
                        <p className="small mb-1"><strong>Efficacy:</strong> {tx.efficacy}</p>
                        <p className="small mb-1"><strong>Monitoring:</strong> {tx.monitoring}</p>
                        <div className="alert alert-success py-1 mb-0" style={{ fontSize: 11 }}>🧬 CLN11: {tx.cln11_note}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <h5 className="fw-bold mb-3 mt-2" style={{ color: ACCENT2 }}>Contraindications (7 entries)</h5>
              {(bk.contraindications || []).map((ci, i) => (
                <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ci.severity === 'ABSOLUTE CI' ? ACCENT2 : ACCENT3}` }}>
                  <div className="card-header d-flex justify-content-between py-2" style={{ backgroundColor: ci.severity === 'ABSOLUTE CI' ? '#ffebee' : '#fff3e0' }}>
                    <span className="fw-bold small" style={{ color: ci.severity === 'ABSOLUTE CI' ? ACCENT2 : ACCENT3 }}>{ci.drug}</span>
                    <span className={`badge ${ci.severity === 'ABSOLUTE CI' ? 'bg-danger' : 'bg-warning text-dark'}`}>{ci.severity}</span>
                  </div>
                  <div className="card-body py-2">
                    <p className="small mb-1"><strong>Reason:</strong> {ci.reason}</p>
                    <p className="small mb-0">{ci.note}</p>
                  </div>
                </div>
              ))}

              <h5 className="fw-bold mb-3 mt-3" style={{ color: ACCENT }}>Monitoring (14 items)</h5>
              <div className="row">
                {(bk.monitoring || []).map((m, i) => (
                  <div key={i} className="col-md-6 mb-2">
                    <div className="card shadow-sm h-100" style={{ borderLeft: `3px solid ${ACCENT}` }}>
                      <div className="card-body py-2">
                        <div className="d-flex justify-content-between mb-1">
                          <span className="fw-bold small" style={{ color: ACCENT }}>{m.item}</span>
                          <span className="badge text-white" style={{ backgroundColor: ACCENT, fontSize: 9 }}>{m.frequency}</span>
                        </div>
                        <p className="small mb-0">{m.note}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}

      {/* ── TAB 4: DEFINITIONS ── */}
      {tab === 4 && (
        <>
          {!df && <div className="text-center py-4"><div className="spinner-border" style={{ color: ACCENT }} /></div>}
          {df && (
            <>
              <div className="row mb-3">
                <div className="col-md-6">
                  <SectionCard title="Disease Identity" borderColor={ACCENT}>
                    <p className="small mb-1"><strong>Disease:</strong> {df.disease_name}</p>
                    <p className="small mb-1"><strong>Gene:</strong> {df.gene_full}</p>
                    <p className="small mb-1"><strong>OMIM Gene:</strong> {df.omim_gene}</p>
                    <p className="small mb-1"><strong>OMIM Disease:</strong> {df.omim_disease}</p>
                    <p className="small mb-1"><strong>Protein:</strong> {df.protein_full}</p>
                    <p className="small mb-1"><strong>Inheritance:</strong> {df.inheritance_mode}</p>
                    <p className="small mb-1"><strong>Onset:</strong> {df.onset_age}</p>
                    <p className="small mb-1"><strong>EM Pattern:</strong> {df.em_pattern}</p>
                    <p className="small mb-0"><strong>Plasma PGRN:</strong> {df.plasma_pgrn}</p>
                  </SectionCard>
                </div>
                <div className="col-md-6">
                  <SectionCard title="Lifecycle Stages (5 stages)" borderColor={ACCENT}>
                    {(df.lifecycle_stages || []).map((ls, i) => (
                      <div key={i} className="mb-2 border-bottom pb-2">
                        <div className="fw-bold small" style={{ color: ACCENT }}>{ls.stage}</div>
                        <div className="text-muted small">{ls.age_range}</div>
                        <div className="small">{ls.description}</div>
                      </div>
                    ))}
                  </SectionCard>
                </div>
              </div>

              <SectionCard title="Key Concepts (15 concepts)" borderColor={ACCENT}>
                <div className="row">
                  {(df.key_concepts || []).map((kc, i) => (
                    <div key={i} className="col-md-6 mb-3">
                      <div className="border rounded p-2 h-100" style={{ borderColor: ACCENT + '40', backgroundColor: '#f3e5f5' }}>
                        <div className="fw-bold small mb-1" style={{ color: ACCENT }}>{kc.name.replace(/-/g, ' ')}</div>
                        <div className="small">{kc.definition}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </SectionCard>

              <div className="row">
                <div className="col-md-6">
                  <SectionCard title="Thresholds (12)" borderColor={ACCENT3}>
                    <table className="table table-sm table-bordered mb-0" style={{ fontSize: 12 }}>
                      <thead style={{ backgroundColor: '#fff3e0' }}>
                        <tr><th>Parameter</th><th>Value</th><th>Action</th></tr>
                      </thead>
                      <tbody>
                        {(df.thresholds || []).map((th, i) => (
                          <tr key={i}>
                            <td className="fw-bold">{th.parameter}</td>
                            <td style={{ color: ACCENT2 }}>{th.value}</td>
                            <td>{th.action}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </SectionCard>
                </div>
                <div className="col-md-6">
                  <SectionCard title="Standards (12)" borderColor={ACCENT4}>
                    <ul className="small mb-0 ps-3">
                      {(df.standards || []).map((s, i) => <li key={i}>{s}</li>)}
                    </ul>
                  </SectionCard>
                  <SectionCard title="References (6)" borderColor={ACCENT}>
                    <ul className="small mb-0 ps-3">
                      {(df.references || []).map((r, i) => <li key={i}>{r}</li>)}
                    </ul>
                  </SectionCard>
                </div>
              </div>
            </>
          )}
        </>
      )}

      {/* Footer */}
      <div className="mt-3 text-muted small text-center">
        CLN11 (GRN/17q21.31) · Adult Neuronal Ceroid Lipofuscinosis · Progranulin Deficiency · AR Biallelic LOF
        · OMIM *138945 / #614706 · expert_dashboards.json count 276→277
      </div>
    </div>
  );
}
