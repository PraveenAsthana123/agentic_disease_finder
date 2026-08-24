'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

// ALDH4A1 color scheme — proline catabolism / P5C-PLP inactivation / secondary B6 deficiency
const ACCENT  = '#4a0072';   // deep violet — P5C-PLP Schiff base / PLP inactivation mechanism
const ACCENT2 = '#b71c1c';   // deep red — P5C accumulation / PATHOGNOMONIC elevated
const ACCENT3 = '#e65100';   // deep orange — proline markedly elevated / Type II hallmark
const ACCENT4 = '#01579b';   // dark blue — PLP low / secondary B6 deficiency / treatment
const ACCENT5 = '#1b5e20';   // dark green — key negatives / alpha-AASA NORMAL / pipecolic NORMAL
const ACCENT6 = '#880e4f';   // dark pink — VPA absolute CI / high-risk drugs
const ACCENT7 = '#37474f';   // slate — variant data / normal ranges
const ACCENT8 = '#006064';   // teal — B6 responsive / partial treatment benefit

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

export default function ALDH4A1Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/aldh4a1/overview`).then(r => r.json()),
      fetch(`${API}/api/aldh4a1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/aldh4a1/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (err)     return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov)     return <div className="alert alert-warning m-4">No data available.</div>;

  const k = ov.kpis || {};
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
                    &#x1f9ec; ALDH4A1 Epilepsy Dashboard
                  </h4>
                  <div className="text-muted small">
                    Hyperprolinemia Type II — P5C Dehydrogenase Deficiency / P5C-PLP Inactivation / Secondary B6 Deficiency
                  </div>
                  <div className="mt-1">
                    <span className="badge me-1" style={{ backgroundColor: ACCENT }}>ALDH4A1 · 1p36.13 · AR</span>
                    <span className="badge me-1" style={{ backgroundColor: ACCENT2 }}>Proline MARKEDLY HIGH &gt;1000 µmol/L</span>
                    <span className="badge me-1" style={{ backgroundColor: ACCENT3 }}>P5C ELEVATED — PATHOGNOMONIC</span>
                    <span className="badge" style={{ backgroundColor: ACCENT4 }}>PLP LOW — Secondary B6 Deficiency</span>
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
            <KPI label="Avg Proline (µmol/L)" value={k.avg_proline_umol_l} color={ACCENT3} />
            <KPI label="Avg P5C (µmol/L)" value={k.avg_p5c_umol_l} color={ACCENT2} />
            <KPI label="Avg PLP (nmol/L)" value={k.avg_plp_nmol_l} color={ACCENT4} />
            <KPI label="Seizures %" value={`${k.pct_seizures}%`} color={ACCENT} />
            <KPI label="Drug-Resistant %" value={`${k.pct_drug_resistant}%`} color={ACCENT2} />
            <KPI label="B6 Responsive %" value={`${k.pct_b6_responsive}%`} color={ACCENT8} />
            <KPI label="IDD %" value={`${k.pct_idd}%`} color={ACCENT} />
            <KPI label="NBS Detected %" value={`${k.pct_nbs_detected}%`} color={ACCENT7} />
            <KPI label="Protein Restricted %" value={`${k.pct_protein_restricted}%`} color={ACCENT3} />
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
                    <PctBar key={ph} label={ph} pct={Math.round(100*n/ov.cohort_n)} color={ACCENT} />
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
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT3, color: '#fff' }}>
                  Function &amp; Mechanism
                </div>
                <div className="card-body small">
                  <p className="text-muted">{ov.function}</p>
                  <p className="text-muted mb-0">{ov.mechanism}</p>
                </div>
              </div>
            </div>
          </div>

          {/* P5C-PLP Inactivation Pathway */}
          {ov.plp_inactivation_mechanism && (
            <div className="row mb-3">
              <div className="col-12">
                <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
                  <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT2, color: '#fff' }}>
                    &#x1f9ea; P5C→PLP Inactivation Cascade (Epileptogenic Mechanism)
                  </div>
                  <div className="card-body">
                    <div className="row">
                      {Object.entries(ov.plp_inactivation_mechanism).map(([step, text], i) => (
                        <div key={step} className="col-md-4 mb-2">
                          <div className="d-flex align-items-start gap-2">
                            <span className="badge rounded-pill" style={{ backgroundColor: ACCENT2, minWidth: 28 }}>{i+1}</span>
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
              <InfoBox title="KEY POSITIVES (pathognomonic for ALDH4A1)" color={ACCENT2}>
                {ov.key_positive_features}
              </InfoBox>
            </div>
            <div className="col-md-6 mb-3">
              <InfoBox title="KEY NEGATIVES (rule out ALDH7A1/PDE and others)" color={ACCENT5}>
                {ov.key_negative_features}
              </InfoBox>
            </div>
          </div>

          {/* ALDH4A1 vs ALDH7A1 / vs PRODH */}
          {ov.vs_aldh7a1_pde && (
            <div className="row mb-3">
              <div className="col-md-6 mb-3">
                <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${ACCENT}` }}>
                  <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT, color: '#fff' }}>
                    ALDH4A1 vs ALDH7A1/PDE (Antiquitin)
                  </div>
                  <div className="card-body small">
                    {Object.entries(ov.vs_aldh7a1_pde).map(([k, v]) => (
                      <div key={k} className="mb-2">
                        <span className="fw-bold text-capitalize">{k}: </span>
                        <span className="text-muted">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="col-md-6 mb-3">
                <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${ACCENT3}` }}>
                  <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT3, color: '#fff' }}>
                    ALDH4A1 vs PRODH (Hyperprolinemia Type I)
                  </div>
                  <div className="card-body small">
                    {Object.entries(ov.vs_prodh_type1 || {}).map(([k, v]) => (
                      <div key={k} className="mb-2">
                        <span className="fw-bold text-capitalize">{k}: </span>
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
              <InfoBox title="NBS Primary Screen" color={ACCENT4}>{ov.nbs_primary}</InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="NBS Secondary / Confirmatory" color={ACCENT4}>{ov.nbs_secondary}</InfoBox>
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
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT7, color: '#fff' }}>
                  Variant Distribution
                </div>
                <div className="card-body">
                  {Object.entries(varDist).sort((a,b)=>b[1]-a[1]).map(([v, n]) => (
                    <PctBar key={v} label={v} pct={Math.round(100*n/ov.cohort_n)} color={ACCENT7} />
                  ))}
                </div>
              </div>
            </div>

            {/* Biomarker ranges by phenotype */}
            <div className="col-md-8 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT3, color: '#fff' }}>
                  Biomarker Ranges by Phenotype
                </div>
                <div className="card-body">
                  {['proline_umol_l', 'p5c_umol_l', 'plp_nmol_l'].map(bk => (
                    <div key={bk} className="mb-3">
                      <div className="fw-bold small mb-1" style={{ color: bk === 'proline_umol_l' ? ACCENT3 : bk === 'p5c_umol_l' ? ACCENT2 : ACCENT4 }}>
                        {bk === 'proline_umol_l' ? 'Proline (µmol/L) — MARKEDLY ELEVATED' :
                         bk === 'p5c_umol_l'     ? 'P5C (µmol/L) — PATHOGNOMONIC' :
                                                   'PLP (nmol/L) — Secondary B6 deficiency'}
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

          {/* Clinical rates */}
          <div className="row mb-3">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT, color: '#fff' }}>
                  Clinical Feature Rates
                </div>
                <div className="card-body">
                  <PctBar label="Seizures" pct={rates.pct_seizures} color={ACCENT} />
                  <PctBar label="Drug-Resistant Epilepsy" pct={rates.pct_drug_resistant} color={ACCENT2} />
                  <PctBar label="B6-Responsive" pct={rates.pct_b6_responsive} color={ACCENT8} />
                  <PctBar label="IDD" pct={rates.pct_idd} color={ACCENT} />
                  <PctBar label="Behavioral Issues" pct={rates.pct_behavioral} color={ACCENT3} />
                  <PctBar label="Psychiatric Features" pct={rates.pct_psychiatric} color={ACCENT7} />
                  <PctBar label="NBS Detected" pct={rates.pct_nbs_detected} color={ACCENT4} />
                  <PctBar label="Protein-Restricted Diet" pct={rates.pct_protein_restricted} color={ACCENT3} />
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT5, color: '#fff' }}>
                  Key Negatives (Always NORMAL in ALDH4A1)
                </div>
                <div className="card-body">
                  <PctBar label="MMA Normal (100%)" pct={rates.pct_mma_normal} color={ACCENT5} />
                  <PctBar label="Pipecolic Normal — KEY vs ALDH7A1" pct={rates.pct_pipecolic_normal} color={ACCENT5} />
                  <PctBar label="alpha-AASA Normal — KEY vs ALDH7A1" pct={rates.pct_alpha_aasa_normal} color={ACCENT5} />
                  <div className="alert alert-success py-2 mt-2 small mb-0">
                    <strong>Fastest differentiator from ALDH7A1/PDE:</strong> alpha-AASA and pipecolic acid are
                    NORMAL in ALDH4A1. In ALDH7A1, alpha-AASA is MARKEDLY ELEVATED (pathognomonic).
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Patient table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT7, color: '#fff' }}>
              Patient Cohort Sample (first 20 of {ov.cohort_n})
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>ID</th><th>Phenotype</th><th>Variant</th>
                      <th>Pro (µmol/L)</th><th>P5C (µmol/L)</th><th>PLP (nmol/L)</th>
                      <th>Seizures</th><th>DRE</th><th>B6 Resp</th><th>IDD</th>
                    </tr>
                  </thead>
                  <tbody>
                    {patients.map(p => (
                      <tr key={p.id}>
                        <td className="fw-bold small">{p.id}</td>
                        <td className="small">{p.phenotype.split('(')[0].trim()}</td>
                        <td><code className="small">{p.variant}</code></td>
                        <td className="fw-bold" style={{ color: ACCENT3 }}>{p.proline_umol_l}</td>
                        <td className="fw-bold" style={{ color: ACCENT2 }}>{p.p5c_umol_l}</td>
                        <td className="fw-bold" style={{ color: ACCENT4 }}>{p.plp_nmol_l}</td>
                        <td>{p.seizures ? <span className="badge bg-danger">Yes</span> : <span className="badge bg-secondary">No</span>}</td>
                        <td>{p.drug_resistant ? <span className="badge bg-danger">DRE</span> : <span className="badge bg-success">No</span>}</td>
                        <td>{p.b6_responsive ? <span className="badge" style={{ backgroundColor: ACCENT8 }}>Yes</span> : <span className="badge bg-secondary">No</span>}</td>
                        <td>{p.idd ? <span className="badge bg-warning text-dark">Yes</span> : <span className="badge bg-success">No</span>}</td>
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
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT, color: '#fff' }}>
                  Seizure Type Distribution
                </div>
                <div className="card-body">
                  {Object.keys(szDist).length === 0 ? (
                    <div className="text-muted small">No seizure type data available.</div>
                  ) : (
                    Object.entries(szDist).sort((a,b)=>b[1]-a[1]).map(([t, n]) => {
                      const total = Object.values(szDist).reduce((a,b)=>a+b,0);
                      return <PctBar key={t} label={t} pct={Math.round(100*n/total)} color={ACCENT} />;
                    })
                  )}
                </div>
              </div>
            </div>

            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT2, color: '#fff' }}>
                  Seizure Severity &amp; Mechanism
                </div>
                <div className="card-body small text-muted">
                  <p><strong style={{ color: ACCENT2 }}>Epileptogenic mechanism:</strong> P5C accumulates → P5C-PLP Schiff base → PLP inactivated → GAD65/67 impaired → GABA synthesis ↓ → excitation/inhibition imbalance → seizures</p>
                  <p><strong>Seizure rates by phenotype:</strong></p>
                  <ul>
                    <li>Classic-Severe: ~90% seizures; ~65% drug-resistant</li>
                    <li>Moderate: ~75% seizures; ~25% drug-resistant</li>
                    <li>Mild-Attenuated: ~45% seizures; &lt;5% drug-resistant</li>
                  </ul>
                  <p><strong style={{ color: ACCENT8 }}>Pyridoxine/PLP trial:</strong> mandatory in ALL patients at seizure onset — BEFORE diagnosing drug-resistance (30–50% partial response)</p>
                </div>
              </div>
            </div>
          </div>

          {/* Metabolic triggers */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT3, color: '#fff' }}>
              Metabolic Triggers &amp; High-Risk Situations
            </div>
            <div className="card-body">
              <div className="row">
                {[
                  { trigger: 'High-proline diet (collagen, gelatin, casein-rich)', risk: 'HIGH', reason: 'Increases proline flux → PRODH → more P5C → worsens PLP inactivation', color: ACCENT6 },
                  { trigger: 'B6 antagonist drugs (INH, D-penicillamine, cycloserine)', risk: 'ABSOLUTE CI', reason: 'Further deplete already-compromised PLP pool — catastrophic', color: ACCENT6 },
                  { trigger: 'VPA (Valproate)', risk: 'HIGH RISK', reason: 'Triple: mitochondrial inhibition + PRODH inhibition → more P5C + direct PLP depletion', color: ACCENT6 },
                  { trigger: 'Fasting / catabolic stress', risk: 'MODERATE', reason: 'Proline catabolism accelerates during fasting → more P5C generation', color: ACCENT3 },
                  { trigger: 'Phenytoin', risk: 'MODERATE', reason: 'Competes at PLP-dependent enzymes; worsens secondary B6 deficit', color: ACCENT3 },
                  { trigger: 'Folate excess (theoretical)', risk: 'LOW', reason: 'No direct interaction; folate-PLP axis not primary pathology here (unlike ALDH7A1)', color: ACCENT7 },
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
                    {(bd?.key_differentials || []).map((d, i) => (
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
          {/* Treatments */}
          <div className="row mb-3">
            <div className="col-12 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT4, color: '#fff' }}>
                  Treatments (Level of Evidence)
                </div>
                <div className="card-body p-0">
                  <div className="table-responsive">
                    <table className="table table-sm table-bordered mb-0">
                      <thead className="table-light">
                        <tr><th>Treatment</th><th>Level</th><th>Rationale</th></tr>
                      </thead>
                      <tbody>
                        {(bd?.treatments || []).map((t, i) => (
                          <tr key={i}>
                            <td className="fw-bold small">{t.treatment}</td>
                            <td>
                              <span className="badge" style={{
                                backgroundColor: t.level === 'Level A' ? ACCENT4 :
                                                 t.level === 'Level B' ? ACCENT8 : ACCENT7
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
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT6, color: '#fff' }}>
              &#x26a0;&#xfe0f; Drug Risk Table
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0">
                  <thead className="table-light">
                    <tr><th>Drug</th><th>Risk Level</th><th>Mechanism</th></tr>
                  </thead>
                  <tbody>
                    {(bd?.drug_risks || []).map((d, i) => (
                      <tr key={i}>
                        <td className="fw-bold small">{d.drug}</td>
                        <td>
                          <span className="badge" style={{
                            backgroundColor: d.risk.includes('ABSOLUTE') ? '#b71c1c' :
                                             d.risk === 'HIGH RISK' ? '#e53935' :
                                             d.risk.includes('MODERATE') ? '#f57c00' :
                                             d.risk.includes('SAFE') ? '#2e7d32' : ACCENT7
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

          {/* B6 response note */}
          <div className="alert" style={{ borderLeft: `4px solid ${ACCENT8}`, backgroundColor: '#e0f7fa' }}>
            <strong style={{ color: ACCENT8 }}>B6/PLP Trial Protocol:</strong>
            <span className="small ms-2">
              IV pyridoxine 100 mg IV slow push during active seizures — mandatory trial before diagnosing B6-non-responsive.
              Oral PLP 30–60 mg/kg/day for chronic maintenance. Expect 30–50% partial response in ALDH4A1
              (vs &gt;85% in ALDH7A1/PDE). Inadequate response does NOT exclude ALDH4A1 — gene sequencing is required.
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
                <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT3, color: '#fff' }}>
                  Normal Ranges (key biomarkers)
                </div>
                <div className="card-body small">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-light"><tr><th>Biomarker</th><th>Normal / ALDH4A1</th></tr></thead>
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
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT7, color: '#fff' }}>
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
            <div className="card-header fw-bold small" style={{ backgroundColor: ACCENT4, color: '#fff' }}>
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
        </>
      )}

      {/* Footer nav */}
      <div className="mt-3 pt-2 border-top text-center small text-muted">
        <Link href="/sardh" className="me-3">← SARDH (Sarcosinemia)</Link>
        <Link href="/">Home</Link>
        <Link href="/aldh7a1" className="ms-3">ALDH7A1/PDE (Antiquitin) →</Link>
      </div>
    </div>
  );
}
