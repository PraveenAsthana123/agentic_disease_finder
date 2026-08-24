'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

// ALDH18A1 color scheme — P5CS Deficiency / Proline SYNTHESIS failure / Proline CRITICALLY LOW
const ACCENT  = '#4a148c';   // deep purple — proline synthesis failure (rare, deep biochemistry)
const ACCENT2 = '#b71c1c';   // deep red — proline critically low (opposite of PRODH elevation)
const ACCENT3 = '#1b5e20';   // deep green — ornithine low / polyamine depletion
const ACCENT4 = '#e65100';   // burnt orange — seizures / cutis laxa
const ACCENT5 = '#37474f';   // slate — key negatives
const ACCENT6 = '#006064';   // teal — normal biomarkers (PLP, alpha-AASA)
const ACCENT7 = '#0277bd';   // blue — treatments / supplements
const ACCENT8 = '#880e4f';   // dark pink — drug risks

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
        <span>{label}</span>
        <span className="fw-bold">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function InfoBox({ title, children, color = ACCENT }) {
  return (
    <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2">
        <div className="fw-bold small mb-1" style={{ color }}>{title}</div>
        <div className="small text-muted">{children}</div>
      </div>
    </div>
  );
}

export default function ALDH18A1Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/aldh18a1/overview`).then(r => r.json()),
      fetch(`${API}/api/aldh18a1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/aldh18a1/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading ALDH18A1 dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-3 gap-2 flex-wrap">
        <span style={{ fontSize: 32 }}>🧬</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>ALDH18A1 Epilepsy</h4>
          <div className="text-muted small">
            P5CS Deficiency · Δ1-Pyrroline-5-Carboxylate Synthase · De Barsy / Cutis Laxa IIIA/B ·
            Proline CRITICALLY LOW · 10q24.1 · AR/AD · OMIM *138250 / #219150 / #616603
          </div>
        </div>
      </div>

      {/* Pathway inversion alert */}
      <div className="alert alert-warning py-2 small mb-3" style={{ borderLeft: `5px solid ${ACCENT2}` }}>
        <strong>⚠ METABOLIC INVERSION vs PRODH/ALDH4A1:</strong> ALDH18A1 LOF causes proline{' '}
        <strong>SYNTHESIS FAILURE</strong> → Proline <strong>CRITICALLY LOW</strong> (&lt;60 µmol/L).
        PRODH/ALDH4A1 LOF cause catabolism failure → proline <strong>ELEVATED</strong> (350–2200+).
        Management is the <strong>OPPOSITE</strong>: supplement proline; do NOT restrict protein.
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Seizure Prevalence (AR)"    value={`${kpi.seizure_prevalence_pct}%`}    color={ACCENT4} />
            <KPI label="Drug-Resistant Epilepsy"    value={`${kpi.drug_resistant_pct}%`}        color={ACCENT2} />
            <KPI label="Cutis Laxa"                 value={`${kpi.cutis_laxa_pct}%`}            color={ACCENT} />
            <KPI label="Cataracts"                  value={`${kpi.cataracts_pct}%`}             color={ACCENT3} />
            <KPI label="IDD"                        value={`${kpi.idd_pct}%`}                   color={ACCENT4} />
            <KPI label="Proline Supplement"         value={`${kpi.proline_supplement_pct}%`}    color={ACCENT7} />
          </div>

          <div className="row g-2 mb-3">
            <KPI label="Avg Proline (µmol/L)"       value={kpi.avg_proline_umol}                color={ACCENT2} />
            <KPI label="Avg Ornithine (µmol/L)"     value={kpi.avg_ornithine_umol}              color={ACCENT3} />
          </div>

          {/* Pathway comparison table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Proline Pathway — Metabolic Comparison (ALDH18A1 vs PRODH vs ALDH4A1)
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0 small">
                  <thead className="table-dark">
                    <tr>
                      <th>Gene</th><th>Direction</th><th>Proline</th><th>P5C</th><th>PLP</th><th>B6 Response</th><th>Unique Feature</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ov?.metabolic_comparison?.map(r => (
                      <tr key={r.gene} style={{ background: r.gene.startsWith('ALDH18A1') ? '#f3e5f5' : undefined }}>
                        <td className="fw-bold" style={{ color: r.gene.startsWith('ALDH18A1') ? ACCENT : undefined }}>{r.gene}</td>
                        <td>{r.direction}</td>
                        <td className="fw-bold" style={{ color: r.proline.includes('LOW') ? ACCENT2 : r.proline.includes('ELEVATED') ? '#e65100' : undefined }}>{r.proline}</td>
                        <td>{r.p5c}</td>
                        <td>{r.plp}</td>
                        <td>{r.b6_response}</td>
                        <td>{r.unique}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Biomarker highlights */}
          <div className="row g-2 mb-3">
            {ov?.biomarker_highlights?.map(b => (
              <div className="col-md-6" key={b.marker}>
                <InfoBox title={b.marker} color={b.marker.toLowerCase().includes('proline') ? ACCENT2 : ACCENT}>
                  <strong>{b.finding}</strong><br />{b.significance}
                </InfoBox>
              </div>
            ))}
          </div>

          {/* Phenotype distribution */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Phenotypic Classes
            </div>
            <div className="card-body">
              {ov?.phenotypes?.map(p => (
                <PctBar key={p.label} label={p.label} pct={p.pct} color={p.color} />
              ))}
            </div>
          </div>

          {/* Gene card */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Gene / Protein / Pathway
            </div>
            <div className="card-body small">
              <div className="row">
                <div className="col-md-6">
                  <p><strong>Gene:</strong> {ov?.gene}</p>
                  <p><strong>OMIM Gene:</strong> {ov?.omim_gene}</p>
                  <p><strong>OMIM Disease (AR):</strong> {ov?.omim_disease_ar}</p>
                  <p><strong>OMIM Disease (AD):</strong> {ov?.omim_disease_ad}</p>
                </div>
                <div className="col-md-6">
                  <p><strong>Chromosome:</strong> {ov?.chromosome}</p>
                  <p><strong>Inheritance:</strong> {ov?.inheritance}</p>
                  <p><strong>Cases Worldwide:</strong> {ov?.cases_worldwide}</p>
                  <p><strong>Cohort (this portal):</strong> {ov?.cohort_n} patients</p>
                </div>
              </div>
              <div className="mt-2 p-2 rounded" style={{ background: '#f3e5f5' }}>
                <strong>Pathway Role:</strong> {ov?.pathway_role}
              </div>
              <div className="mt-2 p-2 rounded" style={{ background: '#fce4ec' }}>
                <strong>Protein:</strong> {ov?.protein}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 1: Patients & Biomarkers ── */}
      {tab === 1 && (
        <div>
          {/* Biomarker table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Key Biomarkers — ALDH18A1 Deficiency
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0 small">
                  <thead className="table-dark">
                    <tr><th>Biomarker</th><th>Normal</th><th>Expected in ALDH18A1</th><th>Key?</th></tr>
                  </thead>
                  <tbody>
                    {bd?.biomarkers?.map(b => (
                      <tr key={b.name} style={{ background: b.key ? '#fce4ec' : undefined }}>
                        <td className="fw-bold">{b.name}</td>
                        <td>{b.normal} {b.unit}</td>
                        <td style={{
                          color: b.direction === 'low' ? ACCENT2 :
                                 b.direction === 'normal' ? ACCENT6 :
                                 b.direction === 'borderline' ? ACCENT4 : undefined,
                          fontWeight: b.key ? 'bold' : undefined
                        }}>{b.expected}</td>
                        <td>{b.key ? <span className="badge" style={{ background: ACCENT }}>KEY</span> : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Variant breakdown */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Variant Distribution (cohort)
            </div>
            <div className="card-body">
              {bd?.variants?.map(v => (
                <div className="mb-2" key={v.v}>
                  <div className="d-flex justify-content-between small mb-1">
                    <span><strong>{v.v}</strong> — {v.domain}</span>
                    <span className="badge" style={{ background: ACCENT }}>{v.pct}% · {v.sev}</span>
                  </div>
                  <div className="progress" style={{ height: 8 }}>
                    <div className="progress-bar" style={{ width: `${v.pct * 4}%`, backgroundColor: ACCENT }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Seizure by phenotype */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Seizure Rate by Phenotype
            </div>
            <div className="card-body">
              {bd?.seizure_by_phenotype?.map(s => (
                <PctBar key={s.phenotype} label={s.phenotype} pct={s.seizure_pct} color={ACCENT4} />
              ))}
            </div>
          </div>

          {/* Patient table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Patient Cohort (n={bd?.patients?.length})
            </div>
            <div className="card-body p-0">
              <div className="table-responsive" style={{ maxHeight: 400 }}>
                <table className="table table-sm table-bordered mb-0 small">
                  <thead className="table-dark sticky-top">
                    <tr>
                      <th>ID</th><th>Phenotype</th><th>Variant 1</th><th>Proline (µmol/L)</th>
                      <th>Ornithine</th><th>PLP</th><th>Seizures</th><th>DRE</th><th>Cutis Laxa</th><th>Cataracts</th><th>IDD</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd?.patients?.map(p => (
                      <tr key={p.id}>
                        <td className="fw-bold">{p.id}</td>
                        <td style={{ whiteSpace: 'nowrap', color: p.phenotype.includes('Severe') ? ACCENT2 : p.phenotype.includes('Mild') ? ACCENT7 : ACCENT4 }}>
                          {p.phenotype.replace(' AR ', ' ').replace(' AD ', ' ').substring(0, 22)}
                        </td>
                        <td className="font-monospace small">{p.variant_1}</td>
                        <td className="fw-bold" style={{ color: p.proline_umol < 60 ? ACCENT2 : ACCENT3 }}>{p.proline_umol}</td>
                        <td style={{ color: p.ornithine_umol < 30 ? ACCENT2 : undefined }}>{p.ornithine_umol}</td>
                        <td style={{ color: ACCENT6 }}>{p.plp_nmol}</td>
                        <td>{p.has_seizures ? <span className="badge" style={{ background: ACCENT4 }}>Yes</span> : '—'}</td>
                        <td>{p.drug_resistant ? <span className="badge" style={{ background: ACCENT2 }}>DRE</span> : '—'}</td>
                        <td>{p.cutis_laxa ? <span className="badge" style={{ background: ACCENT }}>Yes</span> : '—'}</td>
                        <td>{p.cataracts ? <span className="badge bg-secondary">Yes</span> : '—'}</td>
                        <td>{p.idd ? <span className="badge bg-warning text-dark">Yes</span> : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: Seizures & Triggers ── */}
      {tab === 2 && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
                  Seizure Types
                </div>
                <div className="card-body">
                  {bd?.seizure_types?.map(s => (
                    <PctBar key={s.type} label={s.type} pct={s.pct} color={s.color} />
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
                  Metabolic Triggers
                </div>
                <div className="card-body">
                  {bd?.triggers?.map(t => (
                    <PctBar key={t.trigger} label={t.trigger} pct={t.pct} color={ACCENT4} />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Epileptogenic mechanisms */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Epileptogenic Mechanisms in ALDH18A1 Deficiency
            </div>
            <div className="card-body">
              {[
                { title: '1. Proline Depletion → Glutamate/GABA Imbalance', color: ACCENT2,
                  body: 'Proline is a major precursor for glutamate in neurons (via reverse proline catabolism). Low proline → reduced glutamate substrate → secondary GABA depletion via GAD65/GAD67 substrate limitation.' },
                { title: '2. Ornithine Deficiency → Polyamine Depletion', color: ACCENT3,
                  body: 'Ornithine feeds polyamine synthesis (ODC: ornithine → putrescine → spermidine → spermine). Polyamines modulate NMDA receptor gating. Low polyamines → altered NMDA function → lowered seizure threshold.' },
                { title: '3. Collagen Failure → Vascular/Cortical Structural Abnormalities', color: ACCENT4,
                  body: 'Proline (and hydroxyproline) are essential for collagen triple-helix stability. Severe proline deficiency → defective collagen → brain vascular fragility → micro-bleeds and cortical dysgenesis → focal seizures.' },
                { title: '4. Mitochondrial NADPH Redox Disruption', color: ACCENT,
                  body: 'PYCR1/2 recycle NADP+ → NADPH in mitochondria (using P5C as substrate). Without P5C from ALDH18A1, NADP+/NADPH balance is disrupted → oxidative stress → neuronal mitochondrial dysfunction → epileptogenesis.' },
              ].map(m => (
                <InfoBox key={m.title} title={m.title} color={m.color}>{m.body}</InfoBox>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: Treatments ── */}
      {tab === 3 && (
        <div>
          {/* Treatment table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT7, color: '#fff' }}>
              Treatments — ALDH18A1 Deficiency
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0 small">
                  <thead className="table-dark">
                    <tr><th>Treatment</th><th>Level</th><th>Rationale</th></tr>
                  </thead>
                  <tbody>
                    {bd?.treatments?.map(t => (
                      <tr key={t.treatment}>
                        <td className="fw-bold">{t.treatment}</td>
                        <td><span className="badge" style={{ background: t.level.includes('A') ? '#1565c0' : t.level.includes('B') ? '#2e7d32' : '#ef6c00' }}>{t.level}</span></td>
                        <td>{t.rationale}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Drug risks */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT8, color: '#fff' }}>
              Drug Risks / Contraindications
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm mb-0 small">
                  <thead className="table-dark">
                    <tr><th>Drug</th><th>Risk</th><th>Reason</th></tr>
                  </thead>
                  <tbody>
                    {bd?.drug_risks?.map(d => (
                      <tr key={d.drug} style={{ background: d.risk.includes('ABSOLUTE') ? '#ffebee' : d.risk.includes('HIGH') ? '#fff3e0' : undefined }}>
                        <td className="fw-bold">{d.drug}</td>
                        <td>
                          <span className="badge" style={{
                            background: d.risk.includes('ABSOLUTE') ? '#b71c1c' :
                                        d.risk.includes('HIGH') ? '#e65100' :
                                        d.risk.includes('MODERATE') ? '#f57f17' :
                                        d.risk.includes('BENEFICIAL') ? '#2e7d32' :
                                        d.risk.includes('SAFE') || d.risk.includes('Level A') || d.risk.includes('Level B') ? '#1565c0' : '#757575'
                          }}>{d.risk}</span>
                        </td>
                        <td>{d.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Differential diagnoses */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT5, color: '#fff' }}>
              Differential Diagnoses
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm mb-0 small">
                  <thead className="table-dark">
                    <tr><th>Disease</th><th>Shared Features</th><th>How to Distinguish</th></tr>
                  </thead>
                  <tbody>
                    {bd?.differential_diagnoses?.map(d => (
                      <tr key={d.disease}>
                        <td className="fw-bold" style={{ color: ACCENT }}>{d.disease}</td>
                        <td>{d.shared}</td>
                        <td className="small">{d.distinguish}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 4: Definitions ── */}
      {tab === 4 && (
        <div>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Disease Identity
            </div>
            <div className="card-body small">
              <div className="row">
                <div className="col-md-6">
                  <p><strong>Disease:</strong> {def?.disease}</p>
                  <p><strong>Gene:</strong> {def?.gene_full}</p>
                  <p><strong>OMIM Gene:</strong> {def?.omim_gene}</p>
                  <p><strong>OMIM Disease (AR):</strong> {def?.omim_disease_ar}</p>
                  <p><strong>OMIM Disease (AD):</strong> {def?.omim_disease_ad}</p>
                </div>
                <div className="col-md-6">
                  <p><strong>Chromosome:</strong> {def?.chromosome}</p>
                  <p><strong>Inheritance:</strong> {def?.inheritance}</p>
                  <p><strong>Protein:</strong> {def?.protein}</p>
                </div>
              </div>
              <div className="mt-2 p-2 rounded" style={{ background: '#f3e5f5' }}>
                <strong>Pathway:</strong> {def?.pathway}
              </div>
            </div>
          </div>

          {/* Key concepts */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Key Clinical Concepts
            </div>
            <div className="card-body">
              <ul className="list-unstyled mb-0 small">
                {def?.key_concepts?.map((c, i) => (
                  <li key={i} className="mb-2 d-flex gap-2">
                    <span style={{ color: ACCENT, fontWeight: 'bold', minWidth: 18 }}>{i + 1}.</span>
                    <span>{c}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Biomarker glossary */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Biomarker Glossary
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0 small">
                  <thead className="table-dark">
                    <tr><th>Biomarker</th><th>Definition</th></tr>
                  </thead>
                  <tbody>
                    {def?.biomarker_glossary && Object.entries(def.biomarker_glossary).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold text-nowrap">{k}</td>
                        <td>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Variant glossary */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Variant Glossary
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0 small">
                  <thead className="table-dark">
                    <tr><th>Variant</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {def?.variants_glossary && Object.entries(def.variants_glossary).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold font-monospace text-nowrap">{k}</td>
                        <td>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Normal ranges */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Normal Ranges / Expected Values in ALDH18A1
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm mb-0 small">
                  <thead className="table-dark">
                    <tr><th>Biomarker</th><th>Normal Range + ALDH18A1 Expected</th></tr>
                  </thead>
                  <tbody>
                    {def?.normal_ranges && Object.entries(def.normal_ranges).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold">{k}</td>
                        <td style={{ color: v.includes('LOW') || v.includes('CRITICALLY') ? ACCENT2 : v.includes('NORMAL') ? ACCENT6 : undefined }}>{v}</td>
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
  );
}
