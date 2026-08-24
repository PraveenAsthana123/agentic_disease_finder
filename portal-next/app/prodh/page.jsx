'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

// PRODH color scheme — Hyperprolinemia Type I / proline accumulation / NO PLP inactivation
const ACCENT  = '#006064';   // teal — proline buildup (primary disease color)
const ACCENT2 = '#1565c0';   // blue — P5C NORMAL, distinguishing feature vs Type II
const ACCENT3 = '#2e7d32';   // green — PLP NORMAL, no secondary B6 deficiency
const ACCENT4 = '#ef6c00';   // amber — seizures mild, partial-risk features
const ACCENT5 = '#37474f';   // slate — key negatives
const ACCENT6 = '#4a148c';   // purple — psychiatric features, schizophrenia-like
const ACCENT7 = '#880e4f';   // dark pink — moderate-risk drugs
const ACCENT8 = '#0277bd';   // light blue — treatments

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

export default function PRODHPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/prodh/overview`).then(r => r.json()),
      fetch(`${API}/api/prodh/breakdown`).then(r => r.json()),
      fetch(`${API}/api/prodh/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (err)     return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov)     return <div className="alert alert-warning m-4">No data available.</div>;

  const k = ov.kpis || ov.key_statistics || {};
  const phDist = ov.phenotype_distribution || {};
  const varDist = bd?.variant_dist || {};
  const szDist  = bd?.seizure_type_dist || {};
  const rates   = bd?.clinical_rates || {};
  const bRanges = bd?.biomarker_ranges || {};
  const patients = (bd?.patients || []).slice(0, 20);

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="row mb-3">
        <div className="col-12">
          <div className="card shadow" style={{ borderTop: `5px solid ${ACCENT}` }}>
            <div className="card-body py-2">
              <div className="d-flex justify-content-between align-items-start flex-wrap gap-2">
                <div>
                  <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
                    &#x1f9ec; PRODH Epilepsy Dashboard
                  </h4>
                  <div className="text-muted small">
                    Hyperprolinemia Type I — Proline Dehydrogenase Deficiency / Proline Accumulation / P5C NORMAL / PLP NORMAL
                  </div>
                  <div className="mt-1">
                    <span className="badge me-1" style={{ backgroundColor: ACCENT }}>PRODH · 19q13.2 · AR</span>
                    <span className="badge me-1" style={{ backgroundColor: ACCENT4 }}>Proline 350–1000 µmol/L</span>
                    <span className="badge me-1" style={{ backgroundColor: ACCENT2 }}>P5C NORMAL — ALDH4A1 INTACT</span>
                    <span className="badge me-1" style={{ backgroundColor: ACCENT3 }}>PLP NORMAL — No B6 Deficiency</span>
                    <span className="badge" style={{ backgroundColor: ACCENT6 }}>Psychiatric Features (Schizophrenia-Like)</span>
                  </div>
                </div>
                <div className="text-end small text-muted">
                  <div>{ov.omim_gene} · {ov.omim_disease}</div>
                  <div>{ov.protein_size}</div>
                  <div>Cohort N = {ov.cohort_n}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ───────────────────── TAB 0: Overview ───────────────────── */}
      {tab === 0 && (
        <>
          {/* KPI row */}
          <div className="row mb-3">
            <KPI label="Cohort N" value={ov.cohort_n || 40} color={ACCENT} />
            <KPI label="Avg Proline (µmol/L)" value={k.avg_proline_umol_l} color={ACCENT4} />
            <KPI label="P5C: NORMAL" value="100% Normal" color={ACCENT2} />
            <KPI label="PLP: NORMAL" value="100% Normal" color={ACCENT3} />
            <KPI label="Seizures %" value={`${k.pct_seizures}%`} color={ACCENT4} />
            <KPI label="Drug-Resistant %" value={`${k.pct_drug_resistant}%`} color={ACCENT7} />
            <KPI label="B6 Response" value="No Response" color={ACCENT5} />
            <KPI label="Psychiatric %" value={`${k.pct_psychiatric}%`} color={ACCENT6} />
            <KPI label="IDD %" value={`${k.pct_idd}%`} color={ACCENT4} />
          </div>

          {/* Critical distinguishing alerts */}
          <div className="row mb-3">
            <div className="col-md-4 mb-2">
              <div className="alert mb-0 py-2" style={{ backgroundColor: '#e3f2fd', borderLeft: `4px solid ${ACCENT2}` }}>
                <div className="fw-bold small" style={{ color: ACCENT2 }}>P5C IS NORMAL — ALDH4A1 INTACT</div>
                <div className="small text-muted">This is the single fastest test distinguishing Type I from Type II. In ALDH4A1 deficiency (Type II), P5C is MARKEDLY ELEVATED and pathognomonic. Here it is normal because ALDH4A1 enzyme is functioning.</div>
              </div>
            </div>
            <div className="col-md-4 mb-2">
              <div className="alert mb-0 py-2" style={{ backgroundColor: '#e8f5e9', borderLeft: `4px solid ${ACCENT3}` }}>
                <div className="fw-bold small" style={{ color: ACCENT3 }}>PLP IS NORMAL — NO Secondary B6 Deficiency</div>
                <div className="small text-muted">Unlike Type II (ALDH4A1), there is no P5C-PLP Schiff base formation. PLP pools are intact. B6/Pyridoxine has NO indication — administering it provides no seizure benefit.</div>
              </div>
            </div>
            <div className="col-md-4 mb-2">
              <div className="alert mb-0 py-2" style={{ backgroundColor: '#f3e5f5', borderLeft: `4px solid ${ACCENT6}` }}>
                <div className="fw-bold small" style={{ color: ACCENT6 }}>UNIQUE: Schizophrenia-Like Psychiatric Features</div>
                <div className="small text-muted">Prodromal and frank schizophrenia-spectrum disorders in 15–25% of patients. NOT seen in Type II (ALDH4A1). Proline acts as partial NMDA agonist — excess proline dysregulates glutamate signaling.</div>
              </div>
            </div>
          </div>

          <div className="row">
            {/* Phenotype distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT, color: '#fff' }}>
                  Phenotypic Distribution (N={ov.cohort_n})
                </div>
                <div className="card-body">
                  {Object.entries(phDist).map(([ph, n]) => (
                    <PctBar key={ph} label={ph} pct={Math.round(100 * n / ov.cohort_n)} color={ACCENT} />
                  ))}
                </div>
              </div>
            </div>

            {/* Gene / enzyme card */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT2, color: '#fff' }}>
                  Gene &amp; Enzyme
                </div>
                <div className="card-body small">
                  <table className="table table-sm table-borderless mb-0">
                    <tbody>
                      <tr><td className="text-muted">Gene</td><td className="fw-bold">{ov.gene}</td></tr>
                      <tr><td className="text-muted">Disease</td><td>{ov.disease_name}</td></tr>
                      <tr><td className="text-muted">Locus</td><td>{ov.chromosome}</td></tr>
                      <tr><td className="text-muted">Inheritance</td><td>{ov.inheritance}</td></tr>
                      <tr><td className="text-muted">Protein</td><td>{ov.protein_size}</td></tr>
                      <tr><td className="text-muted">OMIM Gene</td><td>{ov.omim_gene}</td></tr>
                      <tr><td className="text-muted">OMIM Disease</td><td>{ov.omim_disease}</td></tr>
                      <tr><td className="text-muted">Prevalence</td><td>{ov.prevalence}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Function / mechanism */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT4, color: '#fff' }}>
                  Function &amp; Mechanism
                </div>
                <div className="card-body small">
                  <p className="text-muted">{ov.function}</p>
                  <p className="text-muted mb-0">{ov.mechanism}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Type I vs Type II comparison table — PROMINENT */}
          <div className="row mb-3">
            <div className="col-12">
              <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT}` }}>
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT, color: '#fff' }}>
                  &#x1f4ca; Type I (PRODH) vs Type II (ALDH4A1) — Critical Comparison
                </div>
                <div className="card-body p-0">
                  <div className="table-responsive">
                    <table className="table table-sm table-bordered mb-0">
                      <thead className="table-light">
                        <tr>
                          <th>Feature</th>
                          <th style={{ color: ACCENT }}>Type I (PRODH) — This Disease</th>
                          <th style={{ color: '#4a0072' }}>Type II (ALDH4A1)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(ov.type1_vs_type2 || [
                          { feature: 'Gene', type1: 'PRODH (19q13.2)', type2: 'ALDH4A1 (1p36.13)' },
                          { feature: 'Proline (µmol/L)', type1: '350–1000 µmol/L', type2: '>1000–2200+ µmol/L' },
                          { feature: 'P5C', type1: 'NORMAL ✅', type2: 'MARKEDLY ELEVATED ⚠️' },
                          { feature: 'PLP', type1: 'NORMAL ✅', type2: 'LOW — Secondary B6 Deficiency ⚠️' },
                          { feature: 'B6/Pyridoxine Response', type1: 'NONE (no mechanism)', type2: 'PARTIAL 30–50%' },
                          { feature: 'Seizure Rate', type1: '25–35%', type2: '60–80%' },
                          { feature: 'DRE Rate', type1: '<15%', type2: '25–40%' },
                          { feature: 'Psychiatric Features', type1: 'YES — Schizophrenia-like 15–25%', type2: 'Rare' },
                          { feature: 'IDD', type1: '30–40% mild-moderate', type2: '50–70% moderate' },
                        ]).map((row, i) => (
                          <tr key={i}>
                            <td className="fw-bold small">{row.feature}</td>
                            <td className="small" style={{
                              color: row.type1 && (row.type1.includes('NORMAL') || row.type1.includes('✅')) ? ACCENT3 :
                                     row.type1 && row.type1.includes('NONE') ? ACCENT5 : undefined
                            }}>{row.type1}</td>
                            <td className="small" style={{
                              color: row.type2 && (row.type2.includes('ELEVATED') || row.type2.includes('LOW') || row.type2.includes('⚠️')) ? '#b71c1c' : undefined
                            }}>{row.type2}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Epileptogenic mechanism — PRODH-specific */}
          {ov.epileptogenic_mechanism && (
            <div className="row mb-3">
              <div className="col-12">
                <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
                  <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT4, color: '#fff' }}>
                    &#x26a1; Epileptogenic Mechanism (NOT PLP-mediated — Distinct from Type II)
                  </div>
                  <div className="card-body">
                    <div className="row">
                      {Object.entries(ov.epileptogenic_mechanism).map(([step, text], i) => (
                        <div key={step} className="col-md-4 mb-2">
                          <div className="d-flex align-items-start gap-2">
                            <span className="badge rounded-pill" style={{ backgroundColor: ACCENT4, minWidth: 28 }}>{i + 1}</span>
                            <span className="small text-muted">{text}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Mechanism summary if no structured data */}
          {!ov.epileptogenic_mechanism && (
            <div className="row mb-3">
              <div className="col-12">
                <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
                  <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT4, color: '#fff' }}>
                    &#x26a1; Epileptogenic Mechanism (NOT PLP-mediated — Distinct from Type II)
                  </div>
                  <div className="card-body">
                    <div className="row">
                      {[
                        'PRODH LOF → Proline CANNOT be catabolised to P5C at normal rate → Proline ACCUMULATES (350–1000 µmol/L)',
                        'Excess proline acts as partial NMDA receptor agonist → glutamatergic hyperexcitability → seizure threshold ↓',
                        'Proline inhibits GABA transport (GAT-1/GAT-3) → synaptic GABA re-uptake impaired → paradoxically GABA dysregulation',
                        'P5C does NOT accumulate (ALDH4A1 intact) → NO Schiff base with PLP → PLP pools remain NORMAL',
                        'GAD65/GAD67 remain functional (PLP normal) → GABA synthesis capacity preserved, but GABA transport impaired',
                        'Net: Glutamate excess + GABA transport impairment → excitatory-inhibitory imbalance → seizures (milder than Type II)',
                      ].map((text, i) => (
                        <div key={i} className="col-md-6 mb-2">
                          <div className="d-flex align-items-start gap-2">
                            <span className="badge rounded-pill" style={{ backgroundColor: ACCENT4, minWidth: 28 }}>{i + 1}</span>
                            <span className="small text-muted">{text}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Key positives / negatives */}
          <div className="row mb-3">
            <div className="col-md-6 mb-3">
              <InfoBox title="KEY POSITIVES (pathognomonic for PRODH / Type I)" color={ACCENT4}>
                {ov.key_positive_features || 'Proline ELEVATED 350–1000 µmol/L (plasma). Urine proline ELEVATED. P5C NORMAL (distinguishes from Type II). PLP NORMAL (distinguishes from Type II). Psychiatric/schizophrenia-like features (unique to Type I).'}
              </InfoBox>
            </div>
            <div className="col-md-6 mb-3">
              <InfoBox title="KEY NEGATIVES (rules out other IMDs)" color={ACCENT5}>
                {ov.key_negative_features || 'P5C NORMAL — rules out ALDH4A1 (Type II). PLP NORMAL — no secondary B6 deficiency. alpha-AASA NORMAL — rules out ALDH7A1/PDE. Pipecolic acid NORMAL. MMA NORMAL. tHcy NORMAL. Methionine NORMAL. Ammonia NORMAL. Lactate NORMAL.'}
              </InfoBox>
            </div>
          </div>

          {/* PRODH & Schizophrenia */}
          {ov.prodh_and_schizophrenia && (
            <div className="row mb-3">
              <div className="col-12">
                <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT6}` }}>
                  <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT6, color: '#fff' }}>
                    &#x1f9e0; PRODH &amp; Schizophrenia — Unique Association (Not Seen in Type II)
                  </div>
                  <div className="card-body small">
                    <div className="row">
                      {Object.entries(ov.prodh_and_schizophrenia).map(([k, v]) => (
                        <div key={k} className="col-md-6 mb-2">
                          <span className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}: </span>
                          <span className="text-muted">{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Static schizophrenia card if no structured data */}
          {!ov.prodh_and_schizophrenia && (
            <div className="row mb-3">
              <div className="col-12">
                <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT6}` }}>
                  <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT6, color: '#fff' }}>
                    &#x1f9e0; PRODH &amp; Schizophrenia — Unique Psychiatric Association
                  </div>
                  <div className="card-body small">
                    <div className="row">
                      {[
                        { label: 'Prevalence in PRODH deficiency', value: '15–25% of patients develop schizophrenia-spectrum disorders' },
                        { label: 'Population risk contribution', value: '22q11.2 deletion syndrome (DiGeorge) carries PRODH; PRODH variants in ~0.5–1% of schizophrenia patients' },
                        { label: 'Mechanism', value: 'Excess proline as partial NMDA agonist → glutamate dysregulation → dopamine-NMDA axis disruption — same pathway implicated in schizophrenia pathophysiology' },
                        { label: 'NOT seen in Type II', value: 'ALDH4A1 deficiency does NOT cause schizophrenia-like features — this is pathognomonic for PRODH (Type I)' },
                        { label: 'Clinical implication', value: 'All PRODH patients need psychiatric monitoring; prodromal signs warrant early neuroleptic evaluation' },
                        { label: 'Timing', value: 'Psychiatric features often emerge in adolescence/early adulthood, independent of seizure control' },
                      ].map(({ label, value }, i) => (
                        <div key={i} className="col-md-6 mb-2">
                          <span className="fw-bold">{label}: </span>
                          <span className="text-muted">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* PRODH vs ALDH7A1 */}
          {ov.vs_aldh7a1_pde && (
            <div className="row mb-3">
              <div className="col-md-6 mb-3">
                <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
                  <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT2, color: '#fff' }}>
                    PRODH vs ALDH7A1/PDE (Antiquitin)
                  </div>
                  <div className="card-body small">
                    {Object.entries(ov.vs_aldh7a1_pde).map(([k, v]) => (
                      <div key={k} className="mb-2">
                        <span className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}: </span>
                        <span className="text-muted">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="col-md-6 mb-3">
                <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${ACCENT}` }}>
                  <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT, color: '#fff' }}>
                    PRODH vs ALDH4A1 (Type I vs Type II)
                  </div>
                  <div className="card-body small">
                    {Object.entries(ov.vs_aldh4a1_type2 || {}).map(([k, v]) => (
                      <div key={k} className="mb-2">
                        <span className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}: </span>
                        <span className="text-muted">{v}</span>
                      </div>
                    ))}
                    {!ov.vs_aldh4a1_type2 && [
                      ['Proline level', 'PRODH: 350–1000 µmol/L (moderate). ALDH4A1: >1000–2200+ µmol/L (severe)'],
                      ['P5C', 'PRODH: NORMAL (ALDH4A1 intact). ALDH4A1: MARKEDLY ELEVATED — pathognomonic'],
                      ['PLP', 'PRODH: NORMAL. ALDH4A1: LOW — secondary B6 deficiency via P5C-PLP Schiff base'],
                      ['B6 response', 'PRODH: NONE (no mechanism, PLP intact). ALDH4A1: PARTIAL 30–50%'],
                      ['Seizure severity', 'PRODH: Milder 25–35%. ALDH4A1: Severe 60–80%'],
                      ['Psychiatric', 'PRODH: YES schizophrenia-like. ALDH4A1: Rare — key distinguishing feature'],
                    ].map(([k, v]) => (
                      <div key={k} className="mb-2">
                        <span className="fw-bold">{k}: </span>
                        <span className="text-muted">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* NBS */}
          <div className="row mb-3">
            <div className="col-md-6">
              <InfoBox title="NBS Primary Screen" color={ACCENT8}>{ov.nbs_primary || 'Proline elevated on plasma amino acid chromatography (NBS). Not on standard MS/MS newborn screen — may be missed. Urine amino acids: proline spills into urine (prolinuria).'}</InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="NBS Secondary / Confirmatory" color={ACCENT8}>{ov.nbs_secondary || 'Plasma P5C NORMAL (confirms Type I, not Type II). Plasma PLP NORMAL (confirms not B6-depleted). PRODH gene sequencing. Enzyme activity in fibroblasts/lymphocytes.'}</InfoBox>
            </div>
          </div>
        </>
      )}

      {/* ───────────────────── TAB 1: Patients & Biomarkers ───────────────────── */}
      {tab === 1 && (
        <>
          <div className="row mb-3">
            {/* Variant distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT5, color: '#fff' }}>
                  Variant Distribution
                </div>
                <div className="card-body">
                  {Object.entries(varDist).sort((a, b) => b[1] - a[1]).map(([v, n]) => (
                    <PctBar key={v} label={v} pct={Math.round(100 * n / ov.cohort_n)} color={ACCENT5} />
                  ))}
                </div>
              </div>
            </div>

            {/* Biomarker ranges by phenotype */}
            <div className="col-md-8 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT4, color: '#fff' }}>
                  Biomarker Ranges by Phenotype
                </div>
                <div className="card-body">
                  {['proline_umol_l', 'p5c_umol_l', 'plp_nmol_l'].map(bk => (
                    <div key={bk} className="mb-3">
                      <div className="fw-bold small mb-1" style={{
                        color: bk === 'proline_umol_l' ? ACCENT4 :
                               bk === 'p5c_umol_l'     ? ACCENT2 :
                                                         ACCENT3
                      }}>
                        {bk === 'proline_umol_l' ? 'Proline (µmol/L) — ELEVATED (350–1000, less severe than Type II)' :
                         bk === 'p5c_umol_l'     ? 'P5C (µmol/L) — NORMAL ✅ (ALDH4A1 intact — distinguishes from Type II)' :
                                                   'PLP (nmol/L) — NORMAL ✅ (no secondary B6 deficiency — distinguishes from Type II)'}
                      </div>
                      <div className="table-responsive">
                        <table className="table table-sm table-bordered mb-0">
                          <thead className="table-light"><tr><th>Phenotype</th><th>Min</th><th>Mean</th><th>Max</th><th>N</th></tr></thead>
                          <tbody>
                            {Object.entries(bRanges[bk] || {}).map(([ph, r]) => (
                              <tr key={ph}>
                                <td className="small">{ph}</td>
                                <td>{r.min}</td>
                                <td className="fw-bold">{r.mean}</td>
                                <td>{r.max}</td>
                                <td>{r.n}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Biomarker reference panel */}
          <div className="row mb-3">
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT, color: '#fff' }}>
                  Complete Biomarker Panel — PRODH Hyperprolinemia Type I
                </div>
                <div className="card-body p-0">
                  <div className="table-responsive">
                    <table className="table table-sm table-bordered mb-0">
                      <thead className="table-light">
                        <tr><th>Biomarker</th><th>Status in PRODH</th><th>Expected Range</th><th>Diagnostic Significance</th></tr>
                      </thead>
                      <tbody>
                        {(ov.biomarkers || [
                          { name: 'Proline (plasma)', status: 'ELEVATED', range: '350–1000 µmol/L (Normal <260)', significance: 'Primary biomarker — must be elevated for diagnosis; less severe than Type II' },
                          { name: 'P5C (plasma)', status: 'NORMAL', range: '~2–3 µmol/L (Normal <5)', significance: 'PATHOGNOMONIC NORMAL — ALDH4A1 intact; P5C elevation rules IN Type II, rules OUT Type I' },
                          { name: 'PLP (plasma)', status: 'NORMAL', range: '35–110 nmol/L (Normal 20–120)', significance: 'NORMAL — no B6 deficiency; B6 trial NOT indicated in Type I (no mechanism)' },
                          { name: 'Urine proline', status: 'ELEVATED', range: 'Prolinuria present', significance: 'Proline spills into urine when plasma >800 µmol/L; supports diagnosis' },
                          { name: 'alpha-AASA (plasma)', status: 'NORMAL', range: 'Normal', significance: 'KEY NEGATIVE — elevated in ALDH7A1/PDE (antiquitin); NORMAL here rules out PDE' },
                          { name: 'Pipecolic acid', status: 'NORMAL', range: 'Normal', significance: 'KEY NEGATIVE — elevated in ALDH7A1/PDE; NORMAL here confirms not PDE' },
                          { name: 'tHcy', status: 'NORMAL', range: '<15 µmol/L', significance: 'KEY NEGATIVE — rules out CBS, MTHFR, cobalamin disorders' },
                          { name: 'MMA', status: 'NORMAL', range: 'Normal', significance: 'KEY NEGATIVE — rules out methylmalonyl coA mutase, cobalamin disorders' },
                          { name: 'Methionine', status: 'NORMAL', range: 'Normal', significance: 'KEY NEGATIVE — rules out GNMT, MAT1A, AHCY, CBS disorders' },
                          { name: 'Ammonia', status: 'NORMAL', range: 'Normal', significance: 'KEY NEGATIVE — rules out urea cycle disorders' },
                          { name: 'Lactate', status: 'NORMAL', range: 'Normal', significance: 'KEY NEGATIVE — rules out mitochondrial disease' },
                        ]).map((bm, i) => (
                          <tr key={i}>
                            <td className="fw-bold small">{bm.name}</td>
                            <td>
                              <span className="badge" style={{
                                backgroundColor: bm.status === 'NORMAL' ? ACCENT3 :
                                                 bm.status === 'ELEVATED' ? ACCENT4 : ACCENT5
                              }}>{bm.status}</span>
                            </td>
                            <td className="small">{bm.range}</td>
                            <td className="small text-muted">{bm.significance}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Clinical rates */}
          <div className="row mb-3">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT, color: '#fff' }}>
                  Clinical Feature Rates
                </div>
                <div className="card-body">
                  <PctBar label="Seizures" pct={rates.pct_seizures} color={ACCENT4} />
                  <PctBar label="Drug-Resistant Epilepsy" pct={rates.pct_drug_resistant} color={ACCENT7} />
                  <PctBar label="IDD" pct={rates.pct_idd} color={ACCENT4} />
                  <PctBar label="Psychiatric Features (Schizophrenia-like)" pct={rates.pct_psychiatric} color={ACCENT6} />
                  <PctBar label="Behavioral Issues" pct={rates.pct_behavioral} color={ACCENT5} />
                  <PctBar label="NBS Detected" pct={rates.pct_nbs_detected} color={ACCENT8} />
                  <PctBar label="Protein-Restricted Diet" pct={rates.pct_protein_restricted} color={ACCENT} />
                  <div className="alert alert-info py-2 mt-2 small mb-0">
                    <strong>B6/Pyridoxine Response: 0%</strong> — No patients responded. PLP is normal, no mechanism exists.
                    Do NOT trial B6 in Type I — it provides no seizure benefit and wastes diagnostic time.
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT2, color: '#fff' }}>
                  Key Negatives — Always NORMAL in PRODH (Type I)
                </div>
                <div className="card-body">
                  <PctBar label="P5C Normal (100% — ALDH4A1 intact)" pct={rates.pct_p5c_normal || 100} color={ACCENT2} />
                  <PctBar label="PLP Normal (100% — no B6 deficiency)" pct={rates.pct_plp_normal || 100} color={ACCENT3} />
                  <PctBar label="alpha-AASA Normal — KEY vs ALDH7A1/PDE" pct={rates.pct_alpha_aasa_normal || 100} color={ACCENT5} />
                  <PctBar label="Pipecolic Normal — KEY vs ALDH7A1/PDE" pct={rates.pct_pipecolic_normal || 100} color={ACCENT5} />
                  <PctBar label="MMA Normal" pct={rates.pct_mma_normal || 100} color={ACCENT5} />
                  <div className="alert alert-success py-2 mt-2 small mb-0">
                    <strong>Fastest discriminator from Type II (ALDH4A1):</strong> P5C is NORMAL in Type I.
                    In ALDH4A1 deficiency, P5C is MARKEDLY ELEVATED and pathognomonic — a single P5C measurement
                    differentiates Type I from Type II instantly.
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Patient table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT5, color: '#fff' }}>
              Patient Cohort Sample (first 20 of {ov.cohort_n})
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>ID</th><th>Phenotype</th><th>Variant</th>
                      <th>Pro (µmol/L)</th>
                      <th style={{ color: ACCENT2 }}>P5C (µmol/L)</th>
                      <th style={{ color: ACCENT3 }}>PLP (nmol/L)</th>
                      <th>Seizures</th><th>DRE</th><th>IDD</th><th>Psych</th>
                    </tr>
                  </thead>
                  <tbody>
                    {patients.map(p => (
                      <tr key={p.id}>
                        <td className="fw-bold small">{p.id}</td>
                        <td className="small">{p.phenotype ? p.phenotype.split('(')[0].trim() : '—'}</td>
                        <td><code className="small">{p.variant}</code></td>
                        <td className="fw-bold" style={{ color: ACCENT4 }}>{p.proline_umol_l}</td>
                        <td className="fw-bold" style={{ color: ACCENT2 }}>{p.p5c_umol_l}</td>
                        <td className="fw-bold" style={{ color: ACCENT3 }}>{p.plp_nmol_l}</td>
                        <td>{p.seizures ? <span className="badge bg-danger">Yes</span> : <span className="badge bg-secondary">No</span>}</td>
                        <td>{p.dre ? <span className="badge bg-danger">DRE</span> : <span className="badge bg-success">No</span>}</td>
                        <td>{p.idd ? <span className="badge bg-warning text-dark">Yes</span> : <span className="badge bg-success">No</span>}</td>
                        <td>{p.psychiatric ? <span className="badge" style={{ backgroundColor: ACCENT6 }}>Yes</span> : <span className="badge bg-secondary">No</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ───────────────────── TAB 2: Seizures & Triggers ───────────────────── */}
      {tab === 2 && (
        <>
          {/* Seizure type distribution */}
          <div className="row mb-3">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT4, color: '#fff' }}>
                  Seizure Type Distribution
                </div>
                <div className="card-body">
                  {Object.keys(szDist).length === 0 ? (
                    <div className="text-muted small">No seizure type data available.</div>
                  ) : (
                    Object.entries(szDist).sort((a, b) => b[1] - a[1]).map(([t, n]) => {
                      const total = Object.values(szDist).reduce((a, b) => a + b, 0);
                      return <PctBar key={t} label={t} pct={Math.round(100 * n / total)} color={ACCENT4} />;
                    })
                  )}
                </div>
              </div>
            </div>

            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT, color: '#fff' }}>
                  Seizure Severity &amp; Mechanism (NOT PLP-mediated)
                </div>
                <div className="card-body small text-muted">
                  <p><strong style={{ color: ACCENT4 }}>Epileptogenic mechanism:</strong> Proline accumulates (350–1000 µmol/L) → partial NMDA agonism + GABA transporter inhibition → excitatory-inhibitory imbalance → seizures. P5C and PLP are NORMAL — this is NOT a B6-responsive epilepsy.</p>
                  <p><strong>Seizure rates by phenotype:</strong></p>
                  <ul>
                    <li>Classic-Symptomatic: ~55% seizures; ~15% drug-resistant</li>
                    <li>Mild-Neurodevelopmental: ~20% seizures; ~3% drug-resistant</li>
                    <li>Asymptomatic-Incidental: ~5% seizures (febrile only)</li>
                  </ul>
                  <p className="mb-0"><strong style={{ color: ACCENT7 }}>B6 trial: NOT INDICATED.</strong> PLP is normal. No mechanism for B6 response in Type I. Administering B6 delays correct diagnosis without benefit.</p>
                </div>
              </div>
            </div>
          </div>

          {/* Metabolic triggers */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT4, color: '#fff' }}>
              Metabolic Triggers &amp; High-Risk Situations
            </div>
            <div className="card-body">
              <div className="row">
                {[
                  { trigger: 'High-proline diet (collagen, gelatin, casein-rich foods)', risk: 'HIGH RISK', reason: 'Increases proline flux → more proline accumulates → worsens NMDA agonism and GABA transport inhibition → seizure threshold ↓', color: ACCENT4 },
                  { trigger: 'VPA (Valproate)', risk: 'MODERATE RISK', reason: 'Inhibits PRODH (already deficient) → worsens proline accumulation. Less severe than in Type II since no P5C-PLP inactivation mechanism.', color: ACCENT7 },
                  { trigger: 'B6 antagonist drugs (INH, D-penicillamine, cycloserine)', risk: 'MODERATE RISK', reason: 'Depletes PLP, which is currently normal. Unlike Type II where PLP is already depleted, risk is lower but not zero — avoid if alternatives exist.', color: ACCENT7 },
                  { trigger: 'B6/Pyridoxine supplementation', risk: 'NOT INDICATED', reason: 'PLP is intact. No mechanism for benefit. Administering B6 is futile and delays correct AED therapy — do not use routinely.', color: ACCENT5 },
                  { trigger: 'Fasting / catabolic stress', risk: 'MODERATE', reason: 'Protein catabolism releases proline from muscle → higher plasma proline during illness/fasting episodes', color: ACCENT4 },
                  { trigger: 'Proline-hydroxyproline-rich protein supplements (gelatin)', risk: 'HIGH RISK', reason: 'Direct substrate loading → acute proline spike → potential seizure provocation', color: ACCENT4 },
                ].map(({ trigger, risk, reason, color }) => (
                  <div key={trigger} className="col-md-6 mb-2">
                    <div className="card h-100" style={{ borderLeft: `4px solid ${color}` }}>
                      <div className="card-body py-2 small">
                        <div className="fw-bold">{trigger}</div>
                        <span className="badge mb-1" style={{ backgroundColor: color }}>{risk}</span>
                        <div className="text-muted">{reason}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Differentials */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT5, color: '#fff' }}>
              Key Differentials
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0">
                  <thead className="table-light">
                    <tr><th>Disease</th><th>Shared</th><th>How to Distinguish</th></tr>
                  </thead>
                  <tbody>
                    {(bd?.key_differentials || [
                      { disease: 'ALDH4A1 (Hyperprolinemia Type II)', shared: 'Proline elevated, seizures', distinguish: 'Type II: P5C MARKEDLY ELEVATED, PLP LOW, B6 partial response 30–50%. Type I: P5C NORMAL, PLP NORMAL, B6 NO response.' },
                      { disease: 'ALDH7A1/PDE (Antiquitin)', shared: 'Drug-resistant epilepsy, B6-responsive epilepsy', distinguish: 'PDE: alpha-AASA and pipecolic acid MARKEDLY ELEVATED, B6 >85% response. PRODH: Both NORMAL, B6 no response.' },
                      { disease: 'GNMT deficiency (Sarcosinemia-like)', shared: 'Amino acid metabolism disorder', distinguish: 'GNMT: Methionine/SAM high, sarcosine absent. PRODH: Methionine/SAM normal, proline elevated.' },
                      { disease: '22q11.2 deletion (DiGeorge)', shared: 'Psychiatric features, PRODH region deleted', distinguish: '22q11.2: Cardiac defects, immune deficiency, PRODH is one of many deleted genes. Isolated PRODH: only proline elevated.' },
                      { disease: 'Schizophrenia (idiopathic)', shared: 'Psychosis, psychiatric features', distinguish: 'PRODH: proline ELEVATED on plasma amino acids. Idiopathic schizophrenia: proline NORMAL.' },
                    ]).map((d, i) => (
                      <tr key={i}>
                        <td className="fw-bold small">{d.disease}</td>
                        <td className="small text-muted">{d.shared}</td>
                        <td className="small">{d.distinguish}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ───────────────────── TAB 3: Treatments ───────────────────── */}
      {tab === 3 && (
        <>
          {/* B6 NOT INDICATED banner */}
          <div className="alert" style={{ borderLeft: `4px solid ${ACCENT5}`, backgroundColor: '#eceff1' }}>
            <strong style={{ color: ACCENT5 }}>B6/Pyridoxine: NOT INDICATED in PRODH (Type I).</strong>
            <span className="small ms-2">
              Unlike Type II (ALDH4A1), PLP is normal in Type I. There is no P5C-PLP Schiff base formation, no secondary B6 deficiency,
              and no mechanism by which B6 supplementation would reduce seizures. Do not trial pyridoxine routinely — it delays correct AED therapy.
              This is a critical distinction from ALL B6-responsive epilepsies.
            </span>
          </div>

          {/* Treatments */}
          <div className="row mb-3">
            <div className="col-12 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT8, color: '#fff' }}>
                  Treatments (Level of Evidence)
                </div>
                <div className="card-body p-0">
                  <div className="table-responsive">
                    <table className="table table-sm table-bordered mb-0">
                      <thead className="table-light">
                        <tr><th>Treatment</th><th>Level</th><th>Rationale</th></tr>
                      </thead>
                      <tbody>
                        {(bd?.treatments || def?.treatments || [
                          { treatment: 'Low-Proline Diet (Level B)', level: 'Level B', rationale: 'Reduce dietary proline substrate — restricts collagen-rich foods (gelatin, casein). Lowers plasma proline toward target <500 µmol/L. First-line metabolic intervention.' },
                          { treatment: 'LEV (Levetiracetam) — First-Line AED', level: 'Level B', rationale: 'No metabolic interaction with proline pathway. Well tolerated. First-line AED recommendation for PRODH seizures.' },
                          { treatment: 'Psychiatric Monitoring — Mandatory', level: 'Level A', rationale: 'All PRODH patients need psychiatric surveillance for schizophrenia-spectrum disorder emergence (15–25% lifetime risk). Start in early adolescence.' },
                          { treatment: 'Antipsychotics (if schizophrenia develops)', level: 'Level B', rationale: 'Standard neuroleptic therapy for prodromal/frank schizophrenia. Coordinate with psychiatry. Risk: some antipsychotics lower seizure threshold.' },
                          { treatment: 'LZP/MDZ for acute seizures', level: 'Level A', rationale: 'Benzodiazepines for acute seizure control — standard first-line emergency management regardless of metabolic cause.' },
                          { treatment: 'Protein restriction (low collagen)', level: 'Level B', rationale: 'Avoidance of high-hydroxyproline/proline protein sources (gelatin, bone broth, collagen supplements, casein-heavy formulas).' },
                          { treatment: 'Genetic counseling (AR)', level: 'Level A', rationale: 'Autosomal recessive — 25% recurrence risk per pregnancy. Prenatal/preimplantation diagnosis available for known family variants.' },
                          { treatment: 'B6/Pyridoxine', level: 'NOT INDICATED', rationale: 'PLP is normal. No mechanism. Do not administer — delays correct therapy without benefit.' },
                        ]).map((t, i) => (
                          <tr key={i}>
                            <td className="fw-bold small">{t.treatment}</td>
                            <td>
                              <span className="badge" style={{
                                backgroundColor: t.level === 'Level A' ? ACCENT8 :
                                                 t.level === 'Level B' ? ACCENT :
                                                 t.level === 'NOT INDICATED' ? ACCENT5 : ACCENT5
                              }}>{t.level}</span>
                            </td>
                            <td className="small text-muted">{t.rationale}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Drug risks */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT7, color: '#fff' }}>
              &#x26a0;&#xfe0f; Drug Risk Table — PRODH (Type I)
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0">
                  <thead className="table-light">
                    <tr><th>Drug / Intervention</th><th>Risk Level</th><th>Mechanism / Notes</th></tr>
                  </thead>
                  <tbody>
                    {(bd?.drug_risks || def?.drug_risks || [
                      { drug: 'Proline-heavy diet (collagen, gelatin)', risk: 'HIGH RISK', reason: 'Direct substrate loading → plasma proline spike → worsens NMDA agonism → seizure threshold ↓' },
                      { drug: 'VPA (Valproate)', risk: 'MODERATE RISK', reason: 'Inhibits PRODH enzyme (already deficient) → worsens proline accumulation. Less severe than in Type II (no PLP depletion additive risk).' },
                      { drug: 'B6 antagonists (INH, D-penicillamine)', risk: 'MODERATE RISK', reason: 'Deplete PLP which is currently normal. Risk lower than Type II (no pre-existing PLP deficit) but avoid if alternatives exist.' },
                      { drug: 'B6 / Pyridoxine', risk: 'NOT INDICATED', reason: 'PLP is intact. No mechanism for benefit. Administering delays correct treatment — actively unhelpful in Type I.' },
                      { drug: 'LEV (Levetiracetam)', risk: 'SAFE — Preferred', reason: 'No interaction with proline pathway. First-line AED of choice.' },
                      { drug: 'LZP / MDZ / Benzodiazepines', risk: 'SAFE — Acute use', reason: 'Standard acute seizure management. No metabolic interaction.' },
                      { drug: 'SAM supplements', risk: 'LOW RISK', reason: 'Not directly linked to proline pathway (unlike SARDH/GNMT). Not indicated but not contraindicated.' },
                      { drug: 'High-protein diet (proline-rich)', risk: 'HIGH RISK', reason: 'Increases proline substrate from protein catabolism and dietary sources — worsens accumulation.' },
                    ]).map((d, i) => (
                      <tr key={i}>
                        <td className="fw-bold small">{d.drug}</td>
                        <td>
                          <span className="badge" style={{
                            backgroundColor: d.risk.includes('HIGH') ? '#e53935' :
                                             d.risk.includes('MODERATE') ? ACCENT7 :
                                             d.risk.includes('NOT INDICATED') ? ACCENT5 :
                                             d.risk.includes('SAFE') ? ACCENT3 : ACCENT5
                          }}>{d.risk}</span>
                        </td>
                        <td className="small text-muted">{d.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Psychiatric treatment note */}
          <div className="alert" style={{ borderLeft: `4px solid ${ACCENT6}`, backgroundColor: '#f3e5f5' }}>
            <strong style={{ color: ACCENT6 }}>Psychiatric Management Protocol:</strong>
            <span className="small ms-2">
              All PRODH patients require psychiatric monitoring from age 10 onward. 15–25% lifetime schizophrenia-spectrum risk.
              Prodromal signs (social withdrawal, attenuated psychosis) warrant early referral. Low-dose antipsychotics are appropriate
              when schizophrenia emerges — note that some agents (clozapine, chlorpromazine) may lower seizure threshold and require
              dose adjustments. This psychiatric risk is unique to Type I and is NOT seen in ALDH4A1 (Type II).
            </span>
          </div>
        </>
      )}

      {/* ───────────────────── TAB 4: Definitions ───────────────────── */}
      {tab === 4 && def && (
        <>
          <div className="row mb-3">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT, color: '#fff' }}>
                  Disease Identity
                </div>
                <div className="card-body small">
                  <table className="table table-sm table-borderless mb-0">
                    <tbody>
                      {[
                        ['Disease', def.disease],
                        ['Gene (full)', def.gene_full],
                        ['OMIM Gene', def.omim_gene],
                        ['OMIM Disease', def.omim_disease],
                        ['Chromosome', def.chromosome],
                        ['Protein', def.protein],
                        ['Inheritance', def.inheritance],
                        ['Pathway', def.pathway],
                      ].map(([k, v]) => (
                        <tr key={k}><td className="text-muted">{k}</td><td>{v}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT4, color: '#fff' }}>
                  Normal Ranges (key biomarkers)
                </div>
                <div className="card-body small">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-light"><tr><th>Biomarker</th><th>Normal / PRODH-Expected</th></tr></thead>
                    <tbody>
                      {Object.entries(def.normal_ranges || {}).map(([k, v]) => (
                        <tr key={k}><td>{k}</td><td>{v}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Biomarker glossary */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT2, color: '#fff' }}>
              Biomarker Glossary
            </div>
            <div className="card-body">
              <div className="row">
                {Object.entries(def.biomarker_glossary || {}).map(([term, desc]) => (
                  <div key={term} className="col-md-6 mb-2">
                    <div className="fw-bold small" style={{ color: ACCENT2 }}>{term}</div>
                    <div className="small text-muted">{desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Variant glossary */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT5, color: '#fff' }}>
              Pathogenic Variants
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0">
                  <thead className="table-light"><tr><th>Variant</th><th>Description</th></tr></thead>
                  <tbody>
                    {Object.entries(def.variants_glossary || {}).map(([v, d]) => (
                      <tr key={v}><td><code>{v}</code></td><td className="small text-muted">{d}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Key concepts */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT8, color: '#fff' }}>
              Key Concepts
            </div>
            <div className="card-body">
              <ul className="mb-0">
                {(def.key_concepts || []).map((c, i) => (
                  <li key={i} className="small text-muted mb-1">{c}</li>
                ))}
              </ul>
            </div>
          </div>

          {/* Type I vs Type II summary in definitions */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT6, color: '#fff' }}>
              Clinical Pearls — PRODH Type I (Critical Teaching Points)
            </div>
            <div className="card-body">
              <ul className="mb-0">
                {[
                  'P5C is NORMAL in Type I — this single test immediately distinguishes Type I from Type II. If P5C is elevated, it is Type II (ALDH4A1), not Type I (PRODH).',
                  'PLP is NORMAL — B6/Pyridoxine has NO indication in Type I. Do not administer routinely.',
                  'Seizures are MILDER than Type II (25–35% vs 60–80%) because the PLP-inactivation mechanism is absent.',
                  'DRE rate is LOW (<15%) compared to Type II (25–40%) — most seizures are controllable with standard AEDs.',
                  'Psychiatric schizophrenia-like features (15–25%) are UNIQUE to Type I. ALDH4A1 (Type II) does not cause schizophrenia.',
                  'Mechanism: Excess proline acts as partial NMDA agonist + inhibits GABA transporters (GAT-1/3) — NOT via PLP depletion.',
                  'Proline range 350–1000 µmol/L in Type I vs >1000–2200+ in Type II — both elevated but severity different.',
                  'All patients require lifetime psychiatric surveillance starting from early adolescence.',
                  'PRODH is in the 22q11.2 deletion (DiGeorge) region — isolated PRODH variants cause only prolinemia without cardiac/immune features.',
                  'Low-proline diet (restrict collagen, gelatin, high-casein foods) is the primary metabolic management.',
                ].map((c, i) => (
                  <li key={i} className="small text-muted mb-1">{c}</li>
                ))}
              </ul>
            </div>
          </div>
        </>
      )}

      {/* Footer nav */}
      <div className="mt-3 pt-2 border-top text-center small text-muted">
        <Link href="/aldh4a1" className="me-3">← ALDH4A1 (Hyperprolinemia Type II)</Link>
        <Link href="/">Home</Link>
        <Link href="/gamt" className="ms-3">GAMT (Creatine Deficiency) →</Link>
      </div>
    </div>
  );
}
