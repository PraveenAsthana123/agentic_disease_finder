'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// GAD1 color scheme — GABA synthesis block / GABA critically low / hyperekplexia / neonatal epilepsy
const ACCENT  = '#1b5e20';   // deep forest green — GABA synthesis (anabolic pathway, deficient)
const ACCENT2 = '#b71c1c';   // deep red — GABA critically LOW / absolute CI (isoniazid)
const ACCENT3 = '#0d47a1';   // deep blue — PLP normal (distinguishes from PNPO)
const ACCENT4 = '#e65100';   // burnt orange — hyperekplexia / neonatal encephalopathy
const ACCENT5 = '#6a1b9a';   // dark purple — synthesis block / pathway position
const ACCENT6 = '#004d40';   // teal-green — key negatives / differentials
const ACCENT7 = '#f57f17';   // amber — vigabatrin POTENTIALLY BENEFICIAL (opposite of ABAT!)
const ACCENT8 = '#880e4f';   // dark pink — DRE / refractory

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

export default function GAD1Page() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/gad1/overview`).then(r => r.json()),
      fetch(`${API}/api/gad1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/gad1/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading GAD1 dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            🧬 GAD1 Epilepsy Dashboard
          </h4>
          <div className="text-muted small">
            {ov?.subtitle}
          </div>
          <div className="mt-1">
            <span className="badge me-1" style={{ background: ACCENT }}>2q31.1</span>
            <span className="badge me-1" style={{ background: ACCENT }}>AR LOF</span>
            <span className="badge me-1" style={{ background: ACCENT2 }}>GABA↓↓↓ CSF Critical</span>
            <span className="badge me-1" style={{ background: ACCENT4 }}>Hyperekplexia 80%</span>
            <span className="badge me-1" style={{ background: ACCENT7 }}>VGB Potentially Beneficial</span>
            <span className="badge me-1" style={{ background: ACCENT6 }}>OMIM #617118</span>
            <Link href="/abat" className="badge text-decoration-none ms-1" style={{ background: ACCENT5 }}>← ABAT (GABA↑↑↑ catabolic)</Link>
            <Link href="/aldh5a1" className="badge text-decoration-none ms-1" style={{ background: ACCENT8 }}>→ ALDH5A1 (GHB↑↑↑)</Link>
          </div>
        </div>
      </div>

      {/* CRITICAL METABOLIC INVERSION BANNER */}
      <div className="alert mb-3 py-2" style={{ background: '#fff8e1', border: `2px solid ${ACCENT7}` }}>
        <div className="fw-bold small" style={{ color: ACCENT7 }}>
          ⚡ CRITICAL INVERSION vs ABAT: Same GABA shunt — OPPOSITE biochemical result
        </div>
        <div className="small mt-1">
          <strong>GAD1 LOF (synthesis block):</strong> CSF GABA <strong style={{ color: ACCENT2 }}>CRITICALLY LOW</strong> (&lt;10 nmol/mL) · Glutamate HIGH · GHB LOW ·
          Vigabatrin <strong style={{ color: ACCENT7 }}>POTENTIALLY BENEFICIAL</strong>
          &nbsp;&nbsp;|&nbsp;&nbsp;
          <strong>ABAT LOF (catabolism block):</strong> CSF GABA <strong style={{ color: '#b71c1c' }}>DRAMATICALLY HIGH</strong> (&gt;800 nmol/mL) ·
          Vigabatrin <strong style={{ color: '#b71c1c' }}>ABSOLUTE CI</strong>
        </div>
      </div>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ══════════════ TAB 0 — OVERVIEW ══════════════ */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row g-2 mb-3">
            <KPI label="CSF GABA (avg nmol/mL)" value={kpi.avg_csf_gaba_nmol_ml} color={ACCENT2} />
            <KPI label="CSF Glutamate (avg µmol/mL)" value={kpi.avg_csf_glutamate_umol_ml} color={ACCENT4} />
            <KPI label="Plasma GABA (avg µmol/L)" value={kpi.avg_plasma_gaba_umol_l} color={ACCENT2} />
            <KPI label="PLP plasma (avg nmol/L)" value={kpi.avg_plp_nmol_l} color={ACCENT3} />
            <KPI label="Seizures %" value={`${kpi.pct_seizures}%`} color={ACCENT5} />
            <KPI label="Hyperekplexia %" value={`${kpi.pct_hyperekplexia}%`} color={ACCENT4} />
            <KPI label="DRE %" value={`${kpi.pct_dre}%`} color={ACCENT8} />
            <KPI label="IDD %" value={`${kpi.pct_idd}%`} color={ACCENT5} />
            <KPI label="Infantile Spasms %" value={`${kpi.pct_infantile_spasms}%`} color={ACCENT4} />
            <KPI label="Hypotonia %" value={`${kpi.pct_hypotonia}%`} color={ACCENT6} />
            <KPI label="Myoclonic sz %" value={`${kpi.pct_myoclonic_seizures}%`} color={ACCENT5} />
            <KPI label="Cohort N" value={ov?.cohort_n} color={ACCENT} />
          </div>

          <div className="row g-3">
            {/* Left: gene/disease overview */}
            <div className="col-md-6">
              <div className="card shadow-sm mb-3">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
                  🧬 Gene & Disease Summary
                </div>
                <div className="card-body small">
                  <table className="table table-sm table-borderless mb-0">
                    <tbody>
                      <tr><th>Gene</th><td>GAD1 (Glutamic Acid Decarboxylase 1 / GAD67)</td></tr>
                      <tr><th>Disease</th><td>GAD1-Related Epileptic Encephalopathy (EIEE59)</td></tr>
                      <tr><th>OMIM Gene</th><td>{ov?.omim_gene}</td></tr>
                      <tr><th>OMIM Disease</th><td>{ov?.omim_disease}</td></tr>
                      <tr><th>Chromosome</th><td>{ov?.chromosome}</td></tr>
                      <tr><th>Protein</th><td>{ov?.protein_size}</td></tr>
                      <tr><th>Inheritance</th><td>{ov?.inheritance}</td></tr>
                      <tr><th>Prevalence</th><td>{ov?.prevalence}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <InfoBox title="Enzyme Function" color={ACCENT}>
                {ov?.function}
              </InfoBox>

              <InfoBox title="Pathomechanism" color={ACCENT5}>
                {ov?.mechanism}
              </InfoBox>
            </div>

            {/* Right: biochemistry + phenotype */}
            <div className="col-md-6">
              <InfoBox title="Key Positive Biomarkers (GABA critically low + glutamate elevated)" color={ACCENT2}>
                {ov?.key_positive_features}
              </InfoBox>

              <InfoBox title="Key Negative Biomarkers (rule-out checklist)" color={ACCENT6}>
                {ov?.key_negative_features}
              </InfoBox>

              {/* Phenotype distribution */}
              <div className="card shadow-sm mb-3">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT5, color: '#fff' }}>
                  Phenotype Distribution (N={ov?.cohort_n})
                </div>
                <div className="card-body">
                  {Object.entries(ov?.phenotype_distribution || {}).map(([k, v]) => (
                    <PctBar key={k} label={`${k} (n=${v.n})`} pct={v.pct}
                      color={k === 'Severe-Neonatal' ? ACCENT2 : k === 'Classic-Infantile' ? ACCENT4 : ACCENT3} />
                  ))}
                </div>
              </div>

              {/* Pathway position */}
              <div className="card shadow-sm mb-3">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
                  Pathway Position — GABA Shunt
                </div>
                <div className="card-body small">
                  <div className="text-center fw-bold mb-2" style={{ fontSize: '0.8rem', color: ACCENT }}>
                    Glutamate → <span style={{ color: ACCENT2, fontWeight: 900 }}>[GAD1 ✖ BLOCKED]</span> →
                    GABA → [ABAT] → SSA → [ALDH5A1] → Succinate → TCA
                  </div>
                  <div className="text-muted" style={{ fontSize: '0.75rem' }}>{ov?.pathway_position?.position_summary}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Differential comparisons */}
          <div className="row g-3 mt-1">
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT5, color: '#fff' }}>
                  ⚡ GAD1 vs ABAT — METABOLIC INVERSION
                </div>
                <div className="card-body small">
                  <div className="mb-2"><strong>Shared:</strong> <span className="text-muted">{ov?.vs_abat?.shared}</span></div>
                  <div className="mb-2">
                    <span className="badge me-1" style={{ background: ACCENT }}>GAD1</span>
                    <span className="text-muted">{ov?.vs_abat?.GAD1}</span>
                  </div>
                  <div className="mb-2">
                    <span className="badge me-1" style={{ background: '#1a237e' }}>ABAT</span>
                    <span className="text-muted">{ov?.vs_abat?.ABAT}</span>
                  </div>
                  <div className="text-muted"><strong>Epilepsy:</strong> {ov?.vs_abat?.epilepsy}</div>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT8, color: '#fff' }}>
                  GAD1 vs SSADH (ALDH5A1) — GHB direction
                </div>
                <div className="card-body small">
                  <div className="mb-2"><strong>Shared:</strong> <span className="text-muted">{ov?.vs_ssadh?.shared}</span></div>
                  <div className="mb-2">
                    <span className="badge me-1" style={{ background: ACCENT }}>GAD1</span>
                    <span className="text-muted">{ov?.vs_ssadh?.GAD1}</span>
                  </div>
                  <div className="mb-2">
                    <span className="badge me-1" style={{ background: ACCENT8 }}>SSADH</span>
                    <span className="text-muted">{ov?.vs_ssadh?.SSADH}</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT3, color: '#fff' }}>
                  GAD1 vs PNPO — PLP is the key lab
                </div>
                <div className="card-body small">
                  <div className="mb-2"><strong>Shared:</strong> <span className="text-muted">{ov?.vs_pnpo?.shared}</span></div>
                  <div className="mb-2">
                    <span className="badge me-1" style={{ background: ACCENT }}>GAD1</span>
                    <span className="text-muted">{ov?.vs_pnpo?.GAD1}</span>
                  </div>
                  <div className="mb-2">
                    <span className="badge me-1" style={{ background: ACCENT3 }}>PNPO</span>
                    <span className="text-muted">{ov?.vs_pnpo?.PNPO}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════ TAB 1 — PATIENTS & BIOMARKERS ══════════════ */}
      {tab === 1 && (
        <div>
          {/* Biomarker table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Biomarker Profile — GAD1 Deficiency (GABA synthesis blocked)
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr>
                      <th>Biomarker</th><th>Mean / Status</th><th>Unit</th><th>Normal Range</th><th>Significance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd?.biomarkers || []).map((b, i) => (
                      <tr key={i}>
                        <td className="fw-bold">{b.name}</td>
                        <td style={{ color: b.mean === null ? ACCENT6 : b.mean < 10 && b.name.includes('GABA') ? ACCENT2 : ACCENT }}>
                          {b.mean === null ? b.unit : b.mean}
                        </td>
                        <td>{b.mean === null ? '' : b.unit}</td>
                        <td className="text-muted">{b.normal_range}</td>
                        <td className="text-muted small">{b.significance}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Variants */}
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-bold small" style={{ background: ACCENT5, color: '#fff' }}>
              Known GAD1 Pathogenic Variants
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr><th>Variant</th><th>Domain</th><th>Frequency %</th><th>Severity</th><th>Note</th></tr>
                  </thead>
                  <tbody>
                    {(bd?.variants || []).map((v, i) => (
                      <tr key={i}>
                        <td className="fw-bold font-monospace">{v.variant}</td>
                        <td>{v.domain}</td>
                        <td>~{v.freq_pct}%</td>
                        <td><span className="badge" style={{
                          background: v.severity.includes('Severe') ? ACCENT2 : v.severity.includes('Mild') ? ACCENT3 : ACCENT4
                        }}>{v.severity}</span></td>
                        <td className="text-muted small">{v.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Patient table (first 15) */}
          <div className="card shadow-sm">
            <div className="card-header py-2 fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Patient Cohort — First 15 of {bd?.n} (seed {def?.seed})
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr>
                      <th>ID</th><th>Phenotype</th>
                      <th>CSF GABA (nmol/mL)</th><th>CSF Glu (µmol/mL)</th>
                      <th>PLP (nmol/L)</th><th>Onset (wks)</th>
                      <th>DRE</th><th>Hyperekplexia</th><th>Seizure Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd?.patients || []).slice(0, 15).map((p, i) => (
                      <tr key={i}>
                        <td className="font-monospace small">{p.id}</td>
                        <td><span className="badge" style={{
                          background: p.phenotype === 'Severe-Neonatal' ? ACCENT2 : p.phenotype === 'Classic-Infantile' ? ACCENT4 : ACCENT3,
                          fontSize: '0.7rem'
                        }}>{p.phenotype}</span></td>
                        <td style={{ color: ACCENT2, fontWeight: 'bold' }}>{p.csf_gaba_nmol_ml}</td>
                        <td style={{ color: ACCENT4 }}>{p.csf_glutamate_umol_ml}</td>
                        <td style={{ color: ACCENT3 }}>{p.plp_nmol_l}</td>
                        <td>{p.age_onset_weeks}</td>
                        <td>{p.dre ? <span style={{ color: ACCENT2 }}>✓</span> : <span className="text-muted">–</span>}</td>
                        <td>{p.hyperekplexia ? <span style={{ color: ACCENT4 }}>✓</span> : <span className="text-muted">–</span>}</td>
                        <td className="small text-muted">{p.seizure_type}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════ TAB 2 — SEIZURES & TREATMENTS ══════════════ */}
      {tab === 2 && (
        <div>
          <div className="row g-3">
            {/* Clinical features */}
            <div className="col-md-6">
              <div className="card shadow-sm mb-3">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT5, color: '#fff' }}>
                  Clinical Features
                </div>
                <div className="card-body">
                  {(bd?.clinical_features || []).map((f, i) => (
                    <PctBar key={i} label={`${f.feature} (${f.note})`} pct={f.pct}
                      color={f.pct >= 95 ? ACCENT2 : f.pct >= 70 ? ACCENT4 : f.pct >= 50 ? ACCENT : ACCENT6} />
                  ))}
                </div>
              </div>

              {/* Seizure types */}
              <div className="card shadow-sm">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT4, color: '#fff' }}>
                  Seizure Types
                </div>
                <div className="card-body">
                  {(bd?.seizure_types || []).map((s, i) => (
                    <PctBar key={i} label={`${s.type}: ${s.note}`} pct={s.pct}
                      color={i === 0 ? ACCENT2 : i === 1 ? ACCENT4 : ACCENT5} />
                  ))}
                </div>
              </div>
            </div>

            {/* Right: treatments + drug risks */}
            <div className="col-md-6">
              {/* VIGABATRIN SPECIAL ALERT */}
              <div className="alert py-2 mb-3" style={{ background: '#fff8e1', border: `2px solid ${ACCENT7}` }}>
                <div className="fw-bold small" style={{ color: ACCENT7 }}>⚡ VIGABATRIN IN GAD1: POTENTIALLY BENEFICIAL (Level B)</div>
                <div className="small text-muted mt-1">
                  VGB blocks ABAT (GABA transaminase) → slows GABA catabolism → raises residual GABA synthesised by GAD2.
                  This is the <strong>OPPOSITE</strong> of its role in ABAT deficiency (where VGB is ABSOLUTE CI).
                  Requires retinal toxicity monitoring (ERG, visual fields).
                </div>
              </div>

              <div className="card shadow-sm mb-3">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
                  Treatment Protocol (Evidence Levels)
                </div>
                <div className="card-body p-0">
                  <div className="table-responsive">
                    <table className="table table-sm table-striped mb-0">
                      <thead><tr><th>Treatment</th><th>Level</th><th>Mechanism</th></tr></thead>
                      <tbody>
                        {(bd?.treatments || []).map((t, i) => (
                          <tr key={i}>
                            <td className="fw-bold small">{t.tx}</td>
                            <td>
                              <span className="badge" style={{
                                background: t.level === 'A' ? ACCENT : t.level === 'B' ? ACCENT3 : ACCENT6
                              }}>Level {t.level}</span>
                            </td>
                            <td className="text-muted small">{t.mechanism}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              <div className="card shadow-sm">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT2, color: '#fff' }}>
                  Drug Contraindications & Risks
                </div>
                <div className="card-body p-0">
                  <div className="table-responsive">
                    <table className="table table-sm table-striped mb-0">
                      <thead><tr><th>Drug</th><th>Risk Level</th><th>Reason</th></tr></thead>
                      <tbody>
                        {(bd?.drug_risks || []).map((r, i) => (
                          <tr key={i}>
                            <td className="fw-bold small">{r.drug}</td>
                            <td>
                              <span className="badge" style={{
                                background: r.risk === 'ABSOLUTE CI' ? ACCENT2 : r.risk === 'HIGH RISK' ? ACCENT4 : '#757575'
                              }}>{r.risk}</span>
                            </td>
                            <td className="text-muted small">{r.reason}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Differentials */}
          <div className="card shadow-sm mt-3">
            <div className="card-header py-2 fw-bold small" style={{ background: ACCENT6, color: '#fff' }}>
              Differential Diagnoses — Diseases with Neonatal Epilepsy + GABA/PLP pathway
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Disease</th><th>Shared Features</th><th>Key Distinguishing Finding</th></tr></thead>
                  <tbody>
                    {(bd?.differentials || []).map((d, i) => (
                      <tr key={i}>
                        <td className="fw-bold small">{d.disease}</td>
                        <td className="text-muted small">{d.shared}</td>
                        <td className="small" style={{ color: ACCENT2 }}>{d.distinguishing}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════ TAB 3 — DEFINITIONS ══════════════ */}
      {tab === 3 && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card shadow-sm mb-3">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
                  Gene & Disease Definitions
                </div>
                <div className="card-body small">
                  <table className="table table-sm table-borderless mb-0">
                    <tbody>
                      <tr><th>Gene</th><td>{def?.gene} ({def?.full_name})</td></tr>
                      <tr><th>Disease</th><td>{def?.disease_name}</td></tr>
                      <tr><th>OMIM Gene</th><td>{def?.omim_gene}</td></tr>
                      <tr><th>OMIM Disease</th><td>{def?.omim_disease}</td></tr>
                      <tr><th>Chromosome</th><td>{def?.chromosome}</td></tr>
                      <tr><th>Inheritance</th><td>{def?.inheritance}</td></tr>
                      <tr><th>Protein</th><td>{def?.protein}</td></tr>
                      <tr><th>Function</th><td>{def?.enzyme_function}</td></tr>
                      <tr><th>Pathway</th><td><code style={{ fontSize: '0.75rem' }}>{def?.pathway}</code></td></tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Metabolic inversions */}
              <div className="card shadow-sm mb-3">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT5, color: '#fff' }}>
                  Key Metabolic Inversions — GAD1 vs ABAT/SSADH/PNPO
                </div>
                <div className="card-body p-0">
                  <div className="table-responsive">
                    <table className="table table-sm mb-0">
                      <thead><tr><th>Comparison Pair</th><th>Inversion</th></tr></thead>
                      <tbody>
                        {(def?.key_metabolic_inversions || []).map((inv, i) => (
                          <tr key={i}>
                            <td className="fw-bold small">{inv.pair}</td>
                            <td className="text-muted small">{inv.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>

            <div className="col-md-6">
              <div className="card shadow-sm mb-3">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT3, color: '#fff' }}>
                  Key Terms & Definitions
                </div>
                <div className="card-body p-0">
                  <div className="table-responsive">
                    <table className="table table-sm table-striped mb-0">
                      <thead><tr><th>Term</th><th>Definition</th></tr></thead>
                      <tbody>
                        {(def?.key_terms || []).map((t, i) => (
                          <tr key={i}>
                            <td className="fw-bold small">{t.term}</td>
                            <td className="text-muted small">{t.definition}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              <InfoBox title="Pathway Summary" color={ACCENT}>
                {def?.pathway_summary}
              </InfoBox>

              <div className="card shadow-sm">
                <div className="card-header py-2 fw-bold small" style={{ background: ACCENT6, color: '#fff' }}>
                  NBS & Diagnosis
                </div>
                <div className="card-body small text-muted">
                  <strong>Primary:</strong> {ov?.nbs_primary}<br />
                  <strong>Secondary:</strong> {ov?.nbs_secondary}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
