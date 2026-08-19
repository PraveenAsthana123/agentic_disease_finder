'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a0e2a';   // dark-burgundy-maroon — CLN13 / Kufs Type B / CTSF Adult NCL
const ACCENT2 = '#b71c1c';   // dark-red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep-orange — urgent warnings
const ACCENT4 = '#1b5e20';   // deep-green — safe treatments / monitoring
const ACCENT5 = '#0277bd';   // steel-blue — unique distinction (VGB NOT absolute CI)

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#fce4ec', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

export default function CLN13Page() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/cln13/overview`).then(r => r.json()).then(setOv).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 && !bk) fetch(`${API}/api/cln13/breakdown`).then(r => r.json()).then(setBk).catch(e => setErr(String(e)));
    if (tab === 4 && !df) fetch(`${API}/api/cln13/definitions`).then(r => r.json()).then(setDf).catch(e => setErr(String(e)));
    if ((tab === 2 || tab === 3) && !bk) fetch(`${API}/api/cln13/breakdown`).then(r => r.json()).then(setBk).catch(e => setErr(String(e)));
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div className="rounded p-3 mb-3 text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #7b1f4a 100%)` }}>
        <h4 className="mb-1 fw-bold">🔬 CLN13 Epilepsy — Neuronal Ceroid Lipofuscinosis Type 13</h4>
        <div style={{ fontSize: 13 }}>
          <strong>CTSF (11q13.2)</strong> — Cathepsin F Deficiency · Kufs Disease Type B · AR Biallelic LOF
          <span className="ms-3 badge bg-info text-dark">★ NO RETINAL NCL — VGB NOT Absolute CI</span>
          <span className="ms-2 badge bg-warning text-dark">Adult Onset 20–50y</span>
        </div>
      </div>

      {/* UNIQUE DISTINCTION ALERT */}
      <div className="alert py-2 mb-3" style={{ backgroundColor: '#e3f2fd', borderLeft: `5px solid ${ACCENT5}`, fontSize: 13 }}>
        <strong>★ CLN13 IS THE ONLY NCL WHERE VGB IS NOT AN ABSOLUTE CI:</strong> CLN13 does <strong>NOT</strong> cause
        progressive retinal degeneration (&lt;5%). VGB retinopathy does not compound retinal NCL blindness.
        Contrast with ALL other NCLs (CLN1/CLN2/CLN3/CLN5/CLN6/CLN7/CLN8/CLN10/CLN11) where VGB = ABSOLUTE CI.
        In CLN13, VGB may be considered (last resort, focal seizures) with mandatory ERG baseline — but CBZ/OXC/PHT remain ABSOLUTE CI.
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
            <KPI label="Mean Onset" value={`${ov.mean_onset_seizure_years}y`} color={ACCENT} />
            <KPI label="Drug Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
            <KPI label="Retinal NCL" value={`${ov.retinal_degeneration_pct}%`} color={ACCENT5} />
            <KPI label="Myoclonus" value={`${ov.myoclonus_pct}%`} color={ACCENT3} />
            <KPI label="Dementia-First" value={`${ov.dementia_first_pct}%`} color={ACCENT} />
          </div>
          <div className="row mb-3">
            <KPI label="FP on EM" value={`${ov.fp_em_pct}%`} color={ACCENT} />
            <KPI label="GRODs on EM" value={`${ov.grods_em_pct}%`} color={ACCENT3} />
            <KPI label="Cognitive Impairment" value={`${ov.cognitive_impairment_pct}%`} color={ACCENT2} />
            <KPI label="Cerebellar Ataxia" value={`${ov.cerebellar_ataxia_pct}%`} color={ACCENT} />
            <KPI label="Diag. Delay" value={`${ov.mean_diagnosis_delay_years}y`} color={ACCENT2} />
            <KPI label="Seizures Present" value={`${ov.seizures_present_pct}%`} color={ACCENT} />
          </div>

          <SectionCard title="Gene, Protein & Disease" borderColor={ACCENT}>
            <p><strong>Gene:</strong> {ov.gene}</p>
            <p><strong>Protein:</strong> {ov.protein}</p>
            <p><strong>Inheritance:</strong> {ov.inheritance}</p>
            <p><strong>OMIM:</strong> {ov.omim}</p>
            <p><strong>Disease:</strong> {ov.disease}</p>
          </SectionCard>

          <SectionCard title="★ No Retinal NCL — Defining CLN13 Feature (VGB Not Absolute CI)" borderColor={ACCENT5}>
            <Alert variant="info" text={ov.no_retinal_ncl} />
          </SectionCard>

          <SectionCard title="Kufs Type B (CLN13/CTSF/AR) vs Kufs Type A (CLN6/AD) — Critical Differential" borderColor={ACCENT2}>
            <Alert variant="warning" text={ov.kufs_type_b_vs_type_a} />
          </SectionCard>

          <SectionCard title="No CTSF Enzyme Assay — WES Required" borderColor={ACCENT3}>
            <Alert variant="warning" text={ov.no_ctsf_enzyme_assay} />
          </SectionCard>

          <SectionCard title="Pathomechanism" borderColor={ACCENT}>
            <p>{ov.mechanism}</p>
            <p><strong>SCMAS Substrate Overlap:</strong> {ov.ctsf_substrate_overlap}</p>
          </SectionCard>

          <SectionCard title="No Disease-Modifying Therapy / CTSF ERT Investigational" borderColor={ACCENT3}>
            <Alert variant="warning" text={ov.no_disease_modifying_therapy} />
          </SectionCard>

          <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
            {ov.key_pharmacological_distinctions && Object.entries(ov.key_pharmacological_distinctions).map(([k, v]) => (
              <div key={k} className="mb-3 p-2 rounded" style={{ backgroundColor: '#fafafa', border: '1px solid #eee' }}>
                <div className="fw-bold small mb-1" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
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
                    color={i === 0 ? ACCENT : i === 1 ? '#7b1fa2' : i === 2 ? ACCENT3 : i === 3 ? ACCENT2 : '#546e7a'} />
                ))}
              </div>
              <div className="col-md-7">
                {bk.etiologies?.map((e, i) => (
                  <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT}22`, backgroundColor: '#fce4ec11' }}>
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
          <SectionCard title="Seizure Types — CLN13 Adult PME" borderColor={ACCENT}>
            {bk.seizures?.map((s, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ border: `1px solid ${ACCENT}33`, backgroundColor: '#fce4ec08' }}>
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

          <SectionCard title="Seizure Triggers — CLN13" borderColor={ACCENT3}>
            {bk.triggers?.map((t, i) => (
              <div key={i} className="d-flex align-items-start mb-2">
                <div className="me-3 text-center" style={{ minWidth: 45 }}>
                  <div className="fw-bold" style={{ color: ACCENT3, fontSize: 16 }}>{t.pct}%</div>
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
            text="★ CLN13 ONLY: VGB is NOT an absolute CI (no retinal NCL). However CBZ/OXC/PHT remain ABSOLUTE CI (myoclonus worsening). POLG1 + MERRF exclusion mandatory before VPA. Piracetam Level B for action myoclonus." />

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
                  <strong>CLN13 Note:</strong> {t.cln13_note}
                </div>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Contraindications — CLN13" borderColor={ACCENT2}>
            {bk.contraindications?.map((c, i) => {
              const isUnique = c.severity.includes('CAUTION');
              return (
                <div key={i} className="mb-3 p-2 rounded"
                  style={{ border: `2px solid ${isUnique ? ACCENT5 : ACCENT2}`, backgroundColor: isUnique ? '#e3f2fd' : '#ffebee' }}>
                  <div className="d-flex justify-content-between align-items-start">
                    <div className="fw-bold" style={{ color: isUnique ? ACCENT5 : ACCENT2 }}>{c.drug}</div>
                    <span className="badge ms-2" style={{ backgroundColor: isUnique ? ACCENT5 : ACCENT2 }}>{c.severity}</span>
                  </div>
                  <div className="small mt-1"><strong>Reason:</strong> {c.reason}</div>
                  <div className="small mt-1 text-muted">{c.note}</div>
                </div>
              );
            })}
          </SectionCard>

          <SectionCard title="Monitoring — CLN13" borderColor={ACCENT}>
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
                style={{ border: `1px solid ${ACCENT}33`, backgroundColor: i === 1 ? '#e3f2fd' : '#fce4ec08' }}>
                <div className="fw-bold small mb-1" style={{ color: i === 1 ? ACCENT5 : ACCENT }}>
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
