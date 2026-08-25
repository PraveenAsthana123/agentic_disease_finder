'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// HMGCL / HMG-CoA lyase — step 5 leucine catabolism + ketogenesis colour scheme
const ACCENT  = '#1a237e';   // deep indigo — HMGCL enzyme / terminal step
const ACCENT2 = '#b71c1c';   // deep crimson — hypoketotic hypoglycaemia / metabolic crisis
const ACCENT3 = '#e65100';   // deep orange — fasting hazard / emergency / VPA absolute CI
const ACCENT4 = '#0d47a1';   // deep blue — 3-HMG pathognomonic / C6-DC NBS
const ACCENT5 = '#4a148c';   // deep purple — ketone absence / hypoketotic hallmark
const ACCENT6 = '#880e4f';   // deep rose — absolute CI (VPA + KD)
const ACCENT7 = '#c62828';   // red — AVOID / absolute CI
const ACCENT8 = '#37474f';   // dark slate — hepatomegaly / systemic

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

export default function HMGCLPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/hmgcl/overview`).then(r => r.json()),
      fetch(`${API}/api/hmgcl/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hmgcl/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading HMGCL / HMG-CoA Lyase Deficiency dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="card shadow mb-3" style={{ borderTop: `5px solid ${ACCENT}` }}>
        <div className="card-body pb-2">
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            &#x1f9ec; HMGCL Epilepsy — 3-Hydroxy-3-methylglutaryl-CoA Lyase Deficiency
          </h4>
          <div className="text-muted small">
            HMGCL · Leucine Catabolism STEP 5 (TERMINAL) + Sole Hepatic Ketogenesis Enzyme ·{' '}
            <strong>3-HMG PATHOGNOMONIC · HYPOKETOTIC HYPOGLYCAEMIA · KD + VPA = ABSOLUTE CI</strong> ·
            AR · 1p36.11 · OMIM Gene *600234 · Disease #246450
          </div>
          <div className="mt-2 d-flex gap-2 flex-wrap">
            <span className="badge" style={{ background: ACCENT }}>Leucine Catabolism Step 5 (Terminal)</span>
            <span className="badge" style={{ background: ACCENT4 }}>3-HMG PATHOGNOMONIC</span>
            <span className="badge" style={{ background: ACCENT5 }}>HYPOKETOTIC HYPOGLYCAEMIA</span>
            <span className="badge" style={{ background: ACCENT6 }}>Ketogenic Diet — ABSOLUTE CI</span>
            <span className="badge" style={{ background: ACCENT7 }}>VPA — ABSOLUTE CI</span>
            <span className="badge" style={{ background: ACCENT8 }}>Saudi Arabia Founder p.Arg41Gln</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row mb-3">
            {Object.values(kpi).map((k, i) => (
              <KPI key={i} label={k.label} value={k.value} color={k.color} />
            ))}
          </div>

          {/* Hypoketotic hallmark callout */}
          <div className="alert mb-3" style={{ background: '#e8eaf6', border: `2px solid ${ACCENT5}` }}>
            <strong style={{ color: ACCENT5 }}>&#x26a1; HYPOKETOTIC HYPOGLYCAEMIA — PATHOGNOMONIC HALLMARK:</strong>{' '}
            {ov.hypoketotic_hallmark}
          </div>

          {/* VPA absolute CI warning */}
          <div className="alert mb-3" style={{ background: '#fce4ec', border: `2px solid ${ACCENT7}` }}>
            <strong style={{ color: ACCENT7 }}>&#x1f6ab; VPA — ABSOLUTE CONTRAINDICATION:</strong>{' '}
            {ov.vpa_warning}
          </div>

          <div className="row">
            {/* Enzyme mechanism */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Enzyme Mechanism — HMGCL Block (Step 5 Terminal)</h6>
                  {bd?.enzyme_mechanism && (
                    <div className="small">
                      <div className="mb-2 p-2 rounded" style={{ background: '#e8eaf6' }}>
                        <strong>Function:</strong> {bd.enzyme_mechanism.function}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#e3f2fd' }}>
                        <strong>Reaction:</strong> {bd.enzyme_mechanism.reaction}
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#fce4ec' }}>
                        <strong style={{ color: ACCENT2 }}>Block (DUAL FAILURE):</strong>{' '}
                        <span style={{ whiteSpace: 'pre-wrap' }}>{bd.enzyme_mechanism.block}</span>
                      </div>
                      <div className="mb-2 p-2 rounded" style={{ background: '#f3e5f5' }}>
                        <strong style={{ color: ACCENT6 }}>Why NO ketogenic diet:</strong>{' '}
                        {bd.enzyme_mechanism.no_keto_rationale}
                      </div>
                      <div className="p-2 rounded" style={{ background: '#fff8e1' }}>
                        <strong style={{ color: ACCENT3 }}>Leucine pathway:</strong>{' '}
                        <span style={{ fontFamily: 'monospace', fontSize: '0.7rem' }}>
                          {bd.enzyme_mechanism.leucine_path}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Phenotype distribution */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT4}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT4 }}>Phenotype Distribution (n={ov.n_patients})</h6>
                  {ov.phenotype_dist?.map((pc, i) => (
                    <div key={i} className="mb-2">
                      <PctBar
                        label={`${pc.class} (n=${pc.n})`}
                        pct={pc.pct}
                        color={[ACCENT, ACCENT2, ACCENT5][i % 3]}
                      />
                    </div>
                  ))}

                  <hr className="my-2" />
                  <h6 className="fw-bold small mt-2" style={{ color: ACCENT }}>Leucine Catabolism — Step 5 Context (Terminal)</h6>
                  <div className="small text-muted" style={{ fontFamily: 'monospace', fontSize: '0.7rem' }}>
                    <div>L-Leucine → BCAT → KIC → BCKDH → Isovaleryl-CoA</div>
                    <div style={{ color: '#2e7d32' }}>  → IVD (Step 2) → 3-Methylcrotonyl-CoA</div>
                    <div style={{ color: '#0d47a1' }}>  → MCC (Step 3) → 3-Methylglutaconyl-CoA</div>
                    <div style={{ color: '#6a1b9a' }}>  → AUH (Step 4) → HMG-CoA</div>
                    <div style={{ color: ACCENT7 }}>     → [HMGCL BLOCKED ⚠ — STEP 5 TERMINAL]</div>
                    <div style={{ color: ACCENT4 }}>        ↳ 3-HMG ↑↑↑ (PATHOGNOMONIC)</div>
                    <div style={{ color: ACCENT5 }}>        ↳ Ketones ABSENT (HYPOKETOTIC)</div>
                    <div style={{ color: ACCENT2 }}>        ↳ Glucose LOW (hypoglycaemia)</div>
                    <div style={{ color: ACCENT }}>  Normal: → Acetoacetate + Acetyl-CoA</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* NBS C5-OH + C6-DC differential */}
          <div className="row mb-3">
            <div className="col-12">
              <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT4}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT4 }}>NBS C5-OH + C6-DC Differential (Critical Distinction)</h6>
                  <div className="row mt-2">
                    <div className="col-md-3">
                      <div className="p-2 rounded" style={{ background: '#e8eaf6' }}>
                        <strong style={{ color: ACCENT }}>HMGCL deficiency</strong><br />
                        <span className="text-muted small">C5-OH ↑ + <strong>C6-DC ↑</strong> + NO 3-MCG + hypoglycaemia + NO ketones. 3-HMG on urine OA = PATHOGNOMONIC.</span>
                      </div>
                    </div>
                    <div className="col-md-3">
                      <div className="p-2 rounded" style={{ background: '#e8f5e9' }}>
                        <strong style={{ color: '#2e7d32' }}>3-MCC (MCCC1/2)</strong><br />
                        <span className="text-muted small">C5-OH ↑ + <strong>3-MCG ↑↑ (PATHOGNOMONIC)</strong> + 3-HIV ↑; C3 NORMAL; NOT biotin-responsive. Most common C5-OH cause.</span>
                      </div>
                    </div>
                    <div className="col-md-3">
                      <div className="p-2 rounded" style={{ background: '#fff8e1' }}>
                        <strong style={{ color: ACCENT3 }}>HLCS / BTD</strong><br />
                        <span className="text-muted small">BIOTIN-RESPONSIVE. HLCS: C5-OH ↑ + C3 ↑ (all carboxylases). BTD: rash + SNHL + low biotinidase activity.</span>
                      </div>
                    </div>
                    <div className="col-md-3">
                      <div className="p-2 rounded" style={{ background: '#f3e5f5' }}>
                        <strong style={{ color: ACCENT6 }}>Key rule</strong><br />
                        <span className="text-muted small"><strong>C6-DC ↑ = THINK HMGCL FIRST.</strong> 3-MCG absent = NOT 3-MCC. Hypoketotic hypoglycaemia = HMGCL until proven otherwise.</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* No KD callout */}
          <div className="alert mb-3" style={{ background: '#f3e5f5', border: `2px solid ${ACCENT6}` }}>
            <strong style={{ color: ACCENT6 }}>&#x1f6ab; KETOGENIC DIET — ABSOLUTELY CONTRAINDICATED:</strong>{' '}
            {ov.no_ketogenic_diet_note}
          </div>

          {/* Systemic features */}
          <div className="card shadow-sm mb-3" style={{ borderTop: `3px solid ${ACCENT5}` }}>
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT5 }}>Systemic Features (n={ov.n_patients})</h6>
              <div className="row">
                {bd?.systemic_features?.map((f, i) => (
                  <div key={i} className="col-md-6 mb-2">
                    <PctBar label={f.feature} pct={f.pct} color={[ACCENT2, ACCENT5, ACCENT3, ACCENT4, ACCENT8, ACCENT6, ACCENT, ACCENT7][i % 8]} />
                    <div className="small text-muted ps-1">{f.note}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Hallmark alert */}
          <div className="alert mb-3" style={{ background: '#e8eaf6', border: `2px solid ${ACCENT}` }}>
            <strong style={{ color: ACCENT }}>&#x26a0; KEY BIOMARKERS:</strong>{' '}
            {ov.hallmark_biomarker}
          </div>
        </div>
      )}

      {/* ── TAB 1: Patients & Biomarkers ── */}
      {tab === 1 && (
        <div>
          <div className="row">
            {/* Biomarker table */}
            <div className="col-md-7 mb-3">
              <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Biomarker Profile — HMGCL Deficiency</h6>
                  <div className="table-responsive">
                    <table className="table table-sm table-hover small mb-0">
                      <thead>
                        <tr><th>Biomarker</th><th>Normal</th><th>Status in HMGCL</th><th>Direction</th></tr>
                      </thead>
                      <tbody>
                        {bd?.biomarkers && Object.values(bd.biomarkers).map((bm, i) => (
                          <tr key={i}>
                            <td className="fw-bold">{bm.label}</td>
                            <td className="text-muted">{bm.normal}</td>
                            <td style={{ color: bm.color === 'danger' ? '#b71c1c' : bm.color === 'warning' ? '#e65100' : bm.color === 'success' ? '#2e7d32' : '#0d47a1' }}>
                              {bm.status}
                            </td>
                            <td><span className={`badge bg-${bm.color}`}>{bm.direction}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            {/* Variants */}
            <div className="col-md-5 mb-3">
              <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT5}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT5 }}>Top Pathogenic Variants — HMGCL</h6>
                  {bd?.variants?.map((v, i) => (
                    <div key={i} className="mb-2 p-2 rounded" style={{ background: i === 0 ? '#e8eaf6' : i <= 2 ? '#fce4ec' : '#f5f5f5' }}>
                      <div className="d-flex justify-content-between">
                        <span className="fw-bold small" style={{ color: ACCENT }}>
                          {v.variant}
                        </span>
                        <span className="badge" style={{ background: ACCENT5 }}>{v.freq}%</span>
                      </div>
                      <div className="small text-muted">{v.domain} · {v.phenotype}</div>
                      <div className="small text-muted fst-italic">{v.note}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Cohort preview */}
          <div className="card shadow-sm" style={{ borderTop: `3px solid ${ACCENT4}` }}>
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: ACCENT4 }}>Cohort Preview (first 10 of {ov.n_patients})</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover small mb-0">
                  <thead>
                    <tr>
                      <th>ID</th><th>Phenotype</th><th>Variant</th><th>Onset (mo)</th>
                      <th>3-HMG (mmol/mol Cr)</th><th>C6-DC (µmol/L)</th><th>C5-OH</th>
                      <th>Glucose (mmol/L)</th><th>Carnitine</th><th>Ketones</th><th>Seizures</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd?.cohort_preview?.map((p, i) => (
                      <tr key={i}>
                        <td>{p.id}</td>
                        <td>
                          <span className="badge" style={{
                            background: p.phenotype.includes('Classic') ? ACCENT : p.phenotype.includes('Neonatal') ? ACCENT2 : ACCENT5,
                            fontSize: '0.6rem'
                          }}>{p.phenotype.split(' ')[0]}</span>
                        </td>
                        <td className="small" style={{ color: ACCENT4, fontSize: '0.7rem' }}>{p.variant}</td>
                        <td>{p.age_onset_months}</td>
                        <td className="fw-bold" style={{ color: p.hmg_urine_mmol_mol_cr > 500 ? ACCENT2 : ACCENT4 }}>
                          {p.hmg_urine_mmol_mol_cr}
                        </td>
                        <td style={{ color: ACCENT }}>{p.c6dc_umol_l}</td>
                        <td style={{ color: ACCENT3 }}>{p.c5oh_umol_l}</td>
                        <td className="fw-bold" style={{ color: p.glucose_crisis_mmol_l < 2.0 ? ACCENT7 : ACCENT3 }}>
                          {p.glucose_crisis_mmol_l}
                        </td>
                        <td style={{ color: p.free_carnitine_umol_l < 20 ? ACCENT7 : ACCENT }}>{p.free_carnitine_umol_l}</td>
                        <td>
                          {p.ketones_absent
                            ? <span className="badge bg-danger" style={{ fontSize: '0.6rem' }}>ABSENT &#x26a0;</span>
                            : <span className="badge bg-success" style={{ fontSize: '0.6rem' }}>Present</span>}
                        </td>
                        <td>{p.seizures ? '✓' : '–'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: Seizures & Treatments ── */}
      {tab === 2 && (
        <div>
          <div className="row">
            {/* Seizure types */}
            <div className="col-md-5 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT2}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT2 }}>Seizures / Neurological (n={ov.n_patients})</h6>
                  {bd?.seizure_types?.map((s, i) => (
                    <div key={i} className="mb-2">
                      <PctBar label={s.type} pct={s.pct} color={[ACCENT2, ACCENT5, ACCENT, ACCENT3, ACCENT7, ACCENT8][i % 6]} />
                      <div className="small text-muted ps-1">{s.note}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Treatments */}
            <div className="col-md-7 mb-3">
              <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${ACCENT}` }}>
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Treatment Table (Evidence-Graded)</h6>
                  <div className="table-responsive">
                    <table className="table table-sm table-hover small mb-0">
                      <thead>
                        <tr><th>Therapy</th><th>Level</th><th>Dose</th><th>Rationale</th></tr>
                      </thead>
                      <tbody>
                        {bd?.treatments?.map((t, i) => (
                          <tr key={i} style={{
                            background: t.level === 'ABSOLUTE CI' ? '#fce4ec' : t.level === 'NOT EFFECTIVE' ? '#efebe9' : 'inherit'
                          }}>
                            <td className="fw-bold">{t.therapy}</td>
                            <td>
                              <span className={`badge ${
                                t.level === 'A' ? 'bg-success' :
                                t.level === 'B' ? 'bg-warning text-dark' :
                                t.level === 'C' ? 'bg-secondary' :
                                t.level === 'ABSOLUTE CI' ? 'bg-danger' :
                                t.level === 'NOT EFFECTIVE' ? 'bg-secondary' :
                                'bg-secondary'
                              }`}>
                                {t.level}
                              </span>
                            </td>
                            <td className="text-muted">{t.dose}</td>
                            <td className="text-muted">{t.rationale}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="row">
            <div className="col-md-6 mb-3">
              <div className="alert mb-2" style={{ background: '#fce4ec', border: `2px solid ${ACCENT7}` }}>
                <strong style={{ color: ACCENT7 }}>&#x1f6ab; VPA — ABSOLUTE CI (DIRECT ENZYME INHIBITOR):</strong><br />
                Valproyl-CoA directly inhibits HMGCL at the Cys266 catalytic nucleophile.
                This worsens HMG-CoA accumulation acutely. Combined with carnitine depletion,
                hepatotoxicity, and β-oxidation inhibition — VPA is uniquely dangerous in HMGCL deficiency.
                Use levetiracetam (LEV) as first-line AED.
              </div>
              <div className="alert mb-2" style={{ background: '#f3e5f5', border: `2px solid ${ACCENT6}` }}>
                <strong style={{ color: ACCENT6 }}>&#x1f6ab; KETOGENIC DIET — ABSOLUTE CI (UNIQUE TO HMGCL):</strong><br />
                KD requires HMGCL to produce ketones. In HMGCL deficiency:
                (1) ketones CANNOT be produced — KD cannot achieve its therapeutic goal;
                (2) high fat floods the system with HMG-CoA precursors → acute metabolic crisis.
                KD is the one epilepsy treatment that is directly harmful in this specific IEM.
              </div>
              <div className="alert alert-warning mb-2">
                <strong>&#x26a0; FASTING — EXTREME HAZARD:</strong><br />
                Fasting triggers leucine release + FA mobilisation → HMG-CoA surge + hypoglycaemia
                + absent ketones → neurological crisis. Max fasting: 4 h (neonate), 6 h (infant),
                8–10 h (child). Glucose polymer drinks during illness. Written emergency plan mandatory.
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <InfoBox title="IV Glucose — Level A (FIRST-LINE EMERGENCY)" color={ACCENT2}>
                10–15 mg/kg/min glucose infusion rate (GIR) corrects hypoglycaemia, suppresses
                proteolysis and FA mobilisation, reduces leucine catabolism flux. Bolus 200 mg/kg
                dextrose if symptomatic. Maintain euglycaemia 4–8 mmol/L throughout crisis.
                Do NOT rely on ketones as backup — they CANNOT be produced.
              </InfoBox>
              <InfoBox title="Leucine Restriction — Level A" color={ACCENT}>
                Natural protein 0.8–1.5 g/kg/day + HMGCL-free amino acid formula.
                Target plasma leucine 80–180 µmol/L + urine 3-HMG trending down.
                Primary long-term strategy: reduces HMG-CoA production from leucine catabolism.
                Titrate carefully — leucine is essential; avoid over-restriction.
              </InfoBox>
              <InfoBox title="L-Carnitine — Level A (secondary depletion)" color={ACCENT4}>
                100–200 mg/kg/day oral; 50–100 mg/kg IV during crisis. Secondary carnitine
                depletion from C6-DC and C5-OH conjugation. Replenishes free carnitine pool,
                protects cardiac and skeletal muscle function.
              </InfoBox>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && (
        <div>
          {def && Object.entries(def).map(([key, val], i) => (
            <div key={i} className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${[ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5, ACCENT6, ACCENT7, ACCENT8][i % 8]}` }}>
              <div className="card-body py-2">
                <div className="fw-bold small mb-1" style={{ color: [ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5, ACCENT6, ACCENT7, ACCENT8][i % 8] }}>
                  {key.replace(/_/g, ' ').toUpperCase()}
                </div>
                <div className="small text-muted" style={{ whiteSpace: 'pre-wrap' }}>{val}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Footer nav */}
      <div className="mt-4 d-flex gap-2 flex-wrap">
        <Link href="/mcc" className="btn btn-sm btn-outline-secondary">← MCC (3-Methylcrotonyl-CoA Carboxylase — Step 3)</Link>
        <Link href="/ivd" className="btn btn-sm btn-outline-secondary">IVD (Isovaleric Acidemia — Step 2)</Link>
        <Link href="/agat" className="btn btn-sm btn-outline-secondary">AGAT (Creatine biosynthesis)</Link>
        <Link href="/gcdh" className="btn btn-sm btn-outline-secondary">GCDH (Glutaric Aciduria Type 1)</Link>
        <Link href="/mmut" className="btn btn-sm btn-outline-secondary">MMUT (MMA mut-type)</Link>
      </div>
    </div>
  );
}
