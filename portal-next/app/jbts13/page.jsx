'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'TCTN2 Tectonic Pearls', 'Definitions'];

// JBTS13 colour scheme — TCTN2 / Tectonic complex / MKS8 lethal tier / lipid gate
const ACCENT  = '#1a237e';   // deep indigo — tectonic complex / MTS
const ACCENT2 = '#1565c0';   // strong blue — neurological / cerebellar
const ACCENT3 = '#880e4f';   // dark pink — MKS8 lethal tier warning
const ACCENT4 = '#1b5e20';   // deep green — curative endpoint / transplant
const ACCENT5 = '#e65100';   // burnt orange — polydactyly (12%) / MKS8 encephalocele
const ACCENT6 = '#37474f';   // dark slate — domain matrix / tectonic complex
const ACCENT7 = '#b71c1c';   // deep red — retinal rod-cone (45%, higher than TCTN1)
const ACCENT8 = '#33691e';   // dark olive — hepatic CHF (22%, higher than TCTN1)
const ACCENT9 = '#00695c';   // dark teal — renal NPHP-like (32%)
const ACCENT10= '#4527a0';   // deep purple — TCTN1 vs TCTN2 distinction

const SEED = 433;
const N_COHORT = 40;

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

function Alert({ color, children }) {
  return (
    <div className="alert mb-3" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
      {children}
    </div>
  );
}

export default function JBTS13Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts13/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts13/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts13/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="container py-5"><div className="alert alert-danger">API error: {error}</div></div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT3} 100%)`, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">&#x1f9ec; TCTN2 — Joubert Syndrome Type 13 (JBTS13)</h4>
        <div className="small opacity-90">
          Tectonic-2 · 12q24.31 · 1424 aa · Tectonic Complex Lipid-Gate · MKS8 Lethal Tier · AR · OMIM Gene #613846 · Disease #614245 · MKS8 #613885
        </div>
        <div className="small opacity-80 mt-1">
          Cohort: {N_COHORT} live-born JBTS13 patients (seed {SEED}) · null/null → MKS8 perinatal lethal (excluded from cohort)
        </div>
      </div>

      {/* MKS8 lethal tier warning */}
      <Alert color={ACCENT3}>
        <strong>⚠ MKS8 Lethal Tier — TCTN2-Specific Rule:</strong> Biallelic TCTN2 null → MKS8 (#613885) — perinatal lethal (occipital encephalocele + polydactyly + cystic kidneys). JBTS13 live births carry at least one partial-function allele. Null/null families → 25% MKS8 recurrence — PGT-M/prenatal Dx mandatory.
      </Alert>

      {/* TCTN1 vs TCTN2 distinction warning */}
      <Alert color={ACCENT10}>
        <strong>⚠ TCTN1 (JBTS11, 12q24.11) vs TCTN2 (JBTS13/MKS8, 12q24.31) — Same Chromosome Arm, Critical Distinction:</strong> TCTN1 biallelic null → JBTS11 (live birth, NO MKS tier). TCTN2 biallelic null → MKS8 (perinatal lethal). WES panels MUST distinguish these two genes — different disease tier, different recurrence-risk counselling.
      </Alert>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 0 && overview && (
        <div>
          <div className="row g-2 mb-3">
            {(overview.kpis || []).map((k, i) => <KPI key={i} {...k} />)}
          </div>

          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>TCTN2 Tectonic Complex Function</div>
                <div className="card-body small">{overview.tctn2_function_pearl}</div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>MKS8 Lethal Tier — Allele Class Rule</div>
                <div className="card-body small">{overview.mks8_pearl}</div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT10, color: '#fff' }}>TCTN1 vs TCTN2 — Critical Clinical Distinction</div>
            <div className="card-body small">{overview.tctn1_vs_tctn2_pearl}</div>
          </div>

          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Gene Summary</div>
                <div className="card-body small">{overview.gene_summary}</div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT2, color: '#fff' }}>Allele Class Distribution (N={N_COHORT})</div>
                <div className="card-body">
                  <table className="table table-sm table-bordered mb-0">
                    <thead><tr><th>Allele Class</th><th>n</th><th>%</th></tr></thead>
                    <tbody>
                      {(overview.allele_class_distribution || []).map((r, i) => (
                        <tr key={i}><td className="small">{r.allele_class}</td><td>{r.count}</td><td>{r.pct}%</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Phenotype Summary — JBTS13 Cohort (N={N_COHORT} live births)</div>
            <div className="card-body">
              <div className="row g-2">
                {Object.entries(overview.phenotype_summary || {}).map(([k, v]) => (
                  <div key={k} className="col-6 col-md-3">
                    <div className="border rounded p-2 text-center">
                      <div className="fw-bold">{v}%</div>
                      <div className="text-muted small">{k.replace(/_pct$/, '').replace(/_/g, ' ')}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Clinical Reference</div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>{overview.gene}</td><td className="fw-bold">Disease</td><td className="small">{overview.disease}</td></tr>
                  <tr><td className="fw-bold">OMIM Gene</td><td>#{overview.omim_gene}</td><td className="fw-bold">OMIM JBTS13</td><td>#{overview.omim_disease_jbts13}</td></tr>
                  <tr><td className="fw-bold">OMIM MKS8</td><td>#{overview.omim_disease_mks8}</td><td className="fw-bold">Chr</td><td>{overview.chromosome}</td></tr>
                  <tr><td className="fw-bold">Protein</td><td colSpan={3} className="small">{overview.protein}</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td colSpan={3} className="small">{overview.inheritance}</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td colSpan={3}>{overview.prevalence}</td></tr>
                  <tr><td className="fw-bold">First desc.</td><td colSpan={3} className="small">{overview.first_description}</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Breakdown tab */}
      {tab === 1 && breakdown && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>Allele Class Distribution</div>
                <div className="card-body">
                  <table className="table table-sm table-bordered mb-0">
                    <thead><tr><th>Allele Class</th><th>n</th><th>%</th></tr></thead>
                    <tbody>
                      {(breakdown.allele_distribution || []).map((r, i) => (
                        <tr key={i}><td className="small">{r.allele_class}</td><td>{r.count}</td><td>{r.pct}%</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT2, color: '#fff' }}>Ethnicity Distribution</div>
                <div className="card-body">
                  <table className="table table-sm table-bordered mb-0">
                    <thead><tr><th>Ethnicity</th><th>n</th><th>%</th></tr></thead>
                    <tbody>
                      {(breakdown.ethnicity_distribution || []).map((r, i) => (
                        <tr key={i}><td className="small">{r.ethnicity}</td><td>{r.count}</td><td>{r.pct}%</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Allele tier table */}
          {breakdown.allele_tiers && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>Allele Class → Disease Tier (TCTN2 Tier Rule)</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-dark"><tr><th>Allele Class</th><th>Clinical Tier</th><th>Outcome</th><th>Example</th><th>Counselling</th></tr></thead>
                    <tbody>
                      {breakdown.allele_tiers.map((r, i) => (
                        <tr key={i}>
                          <td className="small fw-bold" style={{ color: i === 0 ? ACCENT3 : 'inherit' }}>{r.allele_class}</td>
                          <td className="small" style={{ color: i === 0 ? ACCENT3 : 'inherit' }}>{r.clinical_tier}</td>
                          <td className="small">{r.outcome}</td>
                          <td className="small font-monospace">{r.example}</td>
                          <td className="small">{r.counselling}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Key variants */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Key TCTN2 Pathogenic Variants</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0">
                  <thead className="table-dark"><tr><th>Variant</th><th>Domain</th><th>Effect</th><th>Population</th><th>Severity</th><th>Retinal Risk</th><th>Renal Risk</th></tr></thead>
                  <tbody>
                    {(breakdown.key_variants || []).map((v, i) => (
                      <tr key={i}>
                        <td className="small font-monospace fw-bold">{v.variant}</td>
                        <td className="small">{v.domain}</td>
                        <td className="small">{v.effect}</td>
                        <td className="small">{v.population}</td>
                        <td className="small">{v.severity}</td>
                        <td className="small">{v.retinal_risk}</td>
                        <td className="small">{v.renal_risk}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Domain matrix */}
          {breakdown.domain_phenotype_matrix && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Domain–Phenotype Matrix</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-dark"><tr><th>Domain</th><th>Key Variants</th><th>Function Lost</th><th>Severity</th><th>Retinal Risk</th><th>Hepatic Risk</th></tr></thead>
                    <tbody>
                      {breakdown.domain_phenotype_matrix.map((r, i) => (
                        <tr key={i}>
                          <td className="small fw-bold">{r.domain}</td>
                          <td className="small font-monospace">{r.key_variants}</td>
                          <td className="small">{r.function_lost}</td>
                          <td className="small">{r.severity}</td>
                          <td className="small">{r.retinal_risk}</td>
                          <td className="small">{r.hepatic_risk}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Patient table */}
          {breakdown.patient_table && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold" style={{ background: ACCENT2, color: '#fff' }}>Patient Cohort Snapshot (first 20 of {N_COHORT})</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered table-hover mb-0">
                    <thead className="table-dark">
                      <tr><th>ID</th><th>Sex</th><th>Ethnicity</th><th>Allele</th><th>Age Dx</th><th>MTS</th><th>Ataxia</th><th>OMA</th><th>Retinal</th><th>Renal</th><th>Hepatic</th><th>Poly</th><th>ID</th><th>Breathing</th></tr>
                    </thead>
                    <tbody>
                      {breakdown.patient_table.map((p, i) => (
                        <tr key={i}>
                          <td className="small font-monospace">{p.id}</td>
                          <td className="small">{p.sex}</td>
                          <td className="small">{p.ethnicity}</td>
                          <td className="small">{p.allele}</td>
                          <td className="small">{p.age_dx_yr}y</td>
                          <td className="small text-success fw-bold">{p.mts}</td>
                          <td className="small">{p.ataxia}</td>
                          <td className="small">{p.oma}</td>
                          <td className="small" style={{ color: p.retinal !== 'No' ? ACCENT7 : 'inherit' }}>{p.retinal}</td>
                          <td className="small" style={{ color: p.renal !== 'No' ? ACCENT9 : 'inherit' }}>{p.renal}</td>
                          <td className="small" style={{ color: p.hepatic !== 'No' ? ACCENT8 : 'inherit' }}>{p.hepatic}</td>
                          <td className="small">{p.poly}</td>
                          <td className="small">{p.id_}</td>
                          <td className="small">{p.breathing}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Pathway steps */}
          {breakdown.pathway_steps && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>TCTN2 Tectonic Complex Pathway — Loss-of-Function Cascade</div>
              <div className="card-body p-0">
                <table className="table table-sm table-bordered mb-0">
                  <thead className="table-dark"><tr><th>Step</th><th>Normal Event</th><th>Effect When TCTN2 Lost</th></tr></thead>
                  <tbody>
                    {breakdown.pathway_steps.map((s, i) => (
                      <tr key={i}>
                        <td className="small fw-bold text-center">{s.step}</td>
                        <td className="small">{s.event}</td>
                        <td className="small text-danger">{s.effect_when_lost}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Management */}
          {breakdown.management && (
            <div className="card shadow-sm">
              <div className="card-header fw-bold" style={{ background: ACCENT4, color: '#fff' }}>Management & Surveillance Protocol</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-dark"><tr><th>Intervention</th><th>Timing</th><th>Rationale</th><th>Evidence Level</th></tr></thead>
                    <tbody>
                      {breakdown.management.map((m, i) => (
                        <tr key={i}>
                          <td className="small fw-bold">{m.intervention}</td>
                          <td className="small">{m.timing}</td>
                          <td className="small">{m.rationale}</td>
                          <td className="small">{m.level}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TCTN2 Tectonic Pearls tab */}
      {tab === 2 && overview && breakdown && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>TCTN2 Tectonic Complex Biology</div>
                <div className="card-body small">{overview.tctn2_function_pearl}</div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>MKS8 Lethal Tier — Clinical Pearl</div>
                <div className="card-body small">{overview.mks8_pearl}</div>
              </div>
            </div>
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-bold" style={{ background: ACCENT10, color: '#fff' }}>TCTN1 (JBTS11) vs TCTN2 (JBTS13/MKS8) — Same Chromosome Arm, Critical Difference</div>
                <div className="card-body small">{overview.tctn1_vs_tctn2_pearl}</div>
              </div>
            </div>
            {breakdown.tctn_distinction && (
              <div className="col-12">
                <div className="card shadow-sm">
                  <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>TCTN1 vs TCTN2 — Side-by-Side Comparison</div>
                  <div className="card-body">
                    <table className="table table-sm table-bordered">
                      <thead className="table-dark"><tr><th>Feature</th><th>TCTN1 (JBTS11)</th><th>TCTN2 (JBTS13/MKS8)</th></tr></thead>
                      <tbody>
                        {Object.entries(breakdown.tctn_distinction?.tctn1 || {}).map(([k]) => (
                          <tr key={k}>
                            <td className="small fw-bold">{k.replace(/_/g, ' ')}</td>
                            <td className="small">{breakdown.tctn_distinction.tctn1[k]}</td>
                            <td className="small" style={{ color: k === 'null_tier' ? ACCENT3 : 'inherit' }}>{breakdown.tctn_distinction.tctn2[k]}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 3 && definitions && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>Gene & Disease Reference</div>
                <div className="card-body">
                  <table className="table table-sm mb-0">
                    <tbody>
                      <tr><td className="fw-bold">Full Name</td><td>{definitions.gene_full_name}</td></tr>
                      <tr><td className="fw-bold">OMIM Gene</td><td>#{definitions.omim_gene}</td></tr>
                      <tr><td className="fw-bold">OMIM JBTS13</td><td>#{definitions.omim_jbts13}</td></tr>
                      <tr><td className="fw-bold">OMIM MKS8</td><td>#{definitions.omim_mks8}</td></tr>
                      <tr><td className="fw-bold">Chromosome</td><td>{definitions.chromosome}</td></tr>
                      <tr><td className="fw-bold">Protein</td><td>{definitions.protein_size}</td></tr>
                      <tr><td className="fw-bold">Inheritance</td><td>{definitions.inheritance}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>Allele Class → Disease Tier Rule</div>
                <div className="card-body">
                  <table className="table table-sm mb-0">
                    <tbody>
                      {Object.entries(definitions.allele_class_rule || {}).map(([k, v]) => (
                        <tr key={k}>
                          <td className="small fw-bold" style={{ color: k === 'null_null' ? ACCENT3 : 'inherit' }}>{k.replace(/_/g, '/')}</td>
                          <td className="small" style={{ color: k === 'null_null' ? ACCENT3 : 'inherit' }}>{v}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Phenotype Frequencies — JBTS13 (live births)</div>
            <div className="card-body">
              <div className="row g-2">
                {Object.entries(definitions.phenotype_frequencies || {}).map(([k, v]) => (
                  <div key={k} className="col-6 col-md-4 col-lg-3">
                    <div className="border rounded p-2">
                      <div className="small fw-bold" style={{ color: k.includes('retinal') ? ACCENT7 : k.includes('hepatic') ? ACCENT8 : k.includes('mks8') ? ACCENT3 : ACCENT }}>{v}</div>
                      <div className="text-muted" style={{ fontSize: '0.72rem' }}>{k.replace(/_/g, ' ')}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT2, color: '#fff' }}>Clinical Distinctions — Key DDx Points</div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <tbody>
                  {Object.entries(definitions.key_clinical_distinctions || {}).map(([k, v]) => (
                    <tr key={k}><td className="small fw-bold">{k.replace(/_/g, ' ')}</td><td className="small">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT4, color: '#fff' }}>Management Highlights</div>
            <div className="card-body">
              <ul className="mb-0 small">
                {(definitions.management_highlights || []).map((h, i) => <li key={i}>{h}</li>)}
              </ul>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Key Literature</div>
            <div className="card-body">
              <ul className="mb-0 small">
                {(definitions.literature_highlights || []).map((l, i) => <li key={i}>{l}</li>)}
              </ul>
            </div>
          </div>
        </div>
      )}

      <div className="mt-3 text-center">
        <Link href="/" className="btn btn-outline-secondary btn-sm me-2">&#8592; Home</Link>
        <Link href="/joubert" className="btn btn-outline-primary btn-sm me-2">Joubert Overview</Link>
        <Link href="/jbts12" className="btn btn-outline-secondary btn-sm me-2">&#8592; JBTS12 (KIF7)</Link>
        <Link href="/jbts11" className="btn btn-outline-secondary btn-sm">&#8592; JBTS11 (TCTN1)</Link>
      </div>
    </div>
  );
}
