'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Profiles & Cohort', 'DEND & Comparisons', 'Definitions'];

// PNDM colour scheme — deep blue/navy (permanent; K-ATP channel; permanent vs transient)
const ACCENT  = '#0d47a1';   // deep blue — permanent; K-ATP channel; sulfo success
const ACCENT2 = '#1565c0';   // blue — KCNJ11 / Kir6.2
const ACCENT3 = '#b71c1c';   // deep red — CRITICAL: test now; DEND; never miss
const ACCENT4 = '#1b5e20';   // deep green — sulfo SUCCESS; 90% off insulin
const ACCENT5 = '#4a148c';   // purple — DEND syndrome; neurology
const ACCENT6 = '#e65100';   // deep orange — EIF2AK3/WRS; warning
const ACCENT7 = '#37474f';   // dark slate — epidemiology
const ACCENT8 = '#006064';   // dark teal — INS/GCK (no sulfo)

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
    <div className="alert mb-2" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
      {children}
    </div>
  );
}

function Section({ title, color, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

function Badge({ text, color }) {
  return <span className="badge me-1" style={{ background: color, fontSize: '0.72em' }}>{text}</span>;
}

export default function PNDMPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/pndm/overview`).then(r => r.json()),
      fetch(`${API}/api/pndm/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pndm/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  const kpi = overview?.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3" style={{ borderLeft: `6px solid ${ACCENT}`, paddingLeft: 14 }}>
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>🧬 Permanent Neonatal Diabetes Mellitus (PNDM)</h4>
        <div className="text-muted small">
          KCNJ11 / ABCC8 / INS / GCK / EIF2AK3 · K-ATP Channel Disease · OMIM #606176 · #226980 (WRS)
          <span className="ms-3 badge" style={{ background: ACCENT2 }}>K-ATP Channel</span>
          <span className="ms-1 badge" style={{ background: ACCENT5 }}>DEND Syndrome</span>
          <span className="ms-1 badge" style={{ background: ACCENT4 }}>Sulfo &gt;90%</span>
        </div>
      </div>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)} style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          <Section title="Clinical Alerts" color={ACCENT3}>
            {(overview?.clinical_alerts || []).map((a, i) => (
              <Alert key={i} color={a.color}>
                <strong style={{ color: a.color }}>{a.title}</strong>
                <div className="small mt-1">{a.body}</div>
              </Alert>
            ))}
          </Section>

          <Section title="Key Performance Indicators — 40-Patient Cohort (Seed 301)" color={ACCENT}>
            <div className="row">
              <KPI label="Cohort (n)" value={kpi.cohort_size} color={ACCENT7} />
              <KPI label="K-ATP Cases" value={kpi.katp_channel_cases} color={ACCENT2} />
              <KPI label="Sulfo Success" value={kpi.sulfo_success_rate} color={ACCENT4} />
              <KPI label="DEND Prev (KCNJ11)" value={kpi.dend_prevalence} color={ACCENT5} />
              <KPI label="Mean Onset" value={kpi.mean_onset_week} color={ACCENT7} />
              <KPI label="Mean HbA1c" value={kpi.mean_hba1c} color={ACCENT3} />
              <KPI label="Ab-Negative" value={kpi.antibody_negative} color={ACCENT} />
              <KPI label="Insulin Lifelong" value={kpi.insulin_lifelong} color={ACCENT8} />
            </div>
          </Section>

          <div className="row">
            <div className="col-md-6">
              <Section title="Genetic Subtype Distribution" color={ACCENT2}>
                <table className="table table-sm table-hover">
                  <thead><tr><th>Gene/Subtype</th><th>n</th><th>%</th></tr></thead>
                  <tbody>
                    {(overview?.subtype_chart || []).map((s, i) => (
                      <tr key={i}>
                        <td><strong>{s.label}</strong></td>
                        <td>{s.n}</td>
                        <td>
                          <div className="d-flex align-items-center gap-2">
                            <div style={{ width: `${s.pct * 1.5}px`, height: 8, background: ACCENT, borderRadius: 3 }} />
                            <span>{s.pct}%</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Therapy Allocation" color={ACCENT4}>
                {overview?.therapy_chart && (() => {
                  const tc = overview.therapy_chart;
                  return (
                    <div>
                      <div className="mb-3 p-2" style={{ background: ACCENT4 + '15', borderRadius: 6 }}>
                        <div className="fw-bold" style={{ color: ACCENT4 }}>K-ATP (KCNJ11 + ABCC8) — Sulfonylurea eligible</div>
                        <div className="small">Total K-ATP: {tc.sulfo_eligible} | Success: {tc.sulfo_success} | Partial failure: {tc.sulfo_partial_failure}</div>
                      </div>
                      <div className="mb-2 p-2" style={{ background: ACCENT8 + '15', borderRadius: 6 }}>
                        <div className="fw-bold" style={{ color: ACCENT8 }}>Non-K-ATP (INS / GCK / EIF2AK3 / other) — Insulin lifelong</div>
                        <div className="small">Total: {tc.insulin_only} patients</div>
                      </div>
                    </div>
                  );
                })()}
              </Section>

              <Section title="Onset Timing Distribution (weeks of life)" color={ACCENT7}>
                <table className="table table-sm">
                  <tbody>
                    {(overview?.onset_histogram || []).map((b, i) => (
                      <tr key={i}>
                        <td className="text-muted small">{b.range}</td>
                        <td>
                          <div style={{ width: `${b.n * 14}px`, height: 10, background: ACCENT2, borderRadius: 3 }} />
                        </td>
                        <td className="small">{b.n} pts</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>
          </div>

          {/* K-ATP Channel Mechanism */}
          <Section title="K-ATP Channel — Why Sulfonylurea Works" color={ACCENT}>
            <div className="row g-3">
              {[
                { label: "Normal", steps: "Glucose ↑ → ATP/ADP ↑ → Kir6.2 closes → Depolarize → Ca²⁺ → Insulin ✓", color: ACCENT4 },
                { label: "KCNJ11/ABCC8 GOF", steps: "K-ATP stuck OPEN despite ATP → NO depolarize → NO Ca²⁺ → NO Insulin → Hyperglycemia", color: ACCENT3 },
                { label: "Sulfonylurea Rescue", steps: "Glibenclamide binds SUR1 NBD2 → closes K-ATP (ATP-independent) → Depolarize → Ca²⁺ → Insulin ✓", color: ACCENT4 },
                { label: "INS/GCK/EIF2AK3", steps: "K-ATP not involved → sulfo has no target → Insulin required lifelong", color: ACCENT8 },
              ].map((m, i) => (
                <div key={i} className="col-md-3">
                  <div className="card h-100 shadow-sm">
                    <div className="card-body p-2">
                      <div className="fw-bold small mb-1" style={{ color: m.color }}>{m.label}</div>
                      <div className="small text-muted">{m.steps}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 1: GENE PROFILES & COHORT ── */}
      {tab === 1 && (
        <div>
          <Section title="Per-Gene Clinical Profiles" color={ACCENT2}>
            {(breakdown?.gene_profiles || []).map((g, i) => (
              <div key={i} className="card mb-3 shadow-sm">
                <div className="card-header py-2" style={{ background: ACCENT + '18' }}>
                  <span className="fw-bold" style={{ color: ACCENT }}>{g.gene}</span>
                  <span className="ms-2 text-muted small">({g.protein})</span>
                  <Badge text={`OMIM ${g.omim_gene}`} color={ACCENT7} />
                  <Badge text={g.locus} color={ACCENT2} />
                  <Badge text={g.freq_pndm} color={ACCENT} />
                </div>
                <div className="card-body p-2">
                  <div className="row g-2">
                    <div className="col-md-4">
                      <div className="small"><strong>Mechanism:</strong> {g.mechanism}</div>
                      <div className="small mt-1"><strong>Example mutations:</strong> {g.key_mutation_examples}</div>
                    </div>
                    <div className="col-md-4">
                      <div className="small">
                        <strong style={{ color: g.sulfo_response.startsWith('NO') ? ACCENT8 : ACCENT4 }}>Sulfo response:</strong> {g.sulfo_response}
                      </div>
                      <div className="small mt-1"><strong>DEND risk:</strong> {g.dend_risk}</div>
                    </div>
                    <div className="col-md-4">
                      <div className="small"><strong>Inheritance:</strong> {g.inheritance}</div>
                      <div className="small mt-1"><strong>Surveillance:</strong> {g.surveillance}</div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </Section>

          <Section title="Sulfonylurea Transition Timeline (K-ATP Patients)" color={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm">
                <thead>
                  <tr>
                    <th>Week of Sulfo</th>
                    <th>Insulin-Free (n)</th>
                    <th>Total K-ATP</th>
                    <th>Progress</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.sulfo_timeline || []).map((t, i) => (
                    <tr key={i}>
                      <td>Week {t.week}</td>
                      <td className="fw-bold" style={{ color: ACCENT4 }}>{t.insulin_free}</td>
                      <td>{t.total_katp}</td>
                      <td>
                        <div style={{ width: `${Math.round(t.insulin_free / t.total_katp * 150)}px`, height: 8, background: ACCENT4, borderRadius: 3 }} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Patient Cohort Sample (first 20)" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-striped">
                <thead>
                  <tr>
                    <th>ID</th><th>Sex</th><th>Subtype</th><th>Onset (wk)</th>
                    <th>BWT SDS</th><th>Therapy</th><th>Sulfo ✓</th><th>DEND</th><th>HbA1c</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.cohort_table || []).map((p, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{p.id}</td>
                      <td>{p.sex}</td>
                      <td><span className="badge" style={{ background: p.subtype.includes('KCNJ11') ? ACCENT2 : p.subtype.includes('ABCC8') ? ACCENT : ACCENT8, fontSize: '0.7em' }}>{p.subtype}</span></td>
                      <td>{p.onset_wk}</td>
                      <td className={p.bwt_sds < -2 ? 'text-danger fw-bold' : ''}>{p.bwt_sds}</td>
                      <td><Badge text={p.therapy} color={p.therapy === 'sulfo' ? ACCENT4 : ACCENT8} /></td>
                      <td style={{ color: p.sulfo_success === '✓' ? ACCENT4 : p.sulfo_success === '✗' ? ACCENT3 : ACCENT7 }}>{p.sulfo_success}</td>
                      <td style={{ color: p.dend !== '—' ? ACCENT5 : undefined }}>{p.dend}</td>
                      <td>{p.hba1c}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 2: DEND & COMPARISONS ── */}
      {tab === 2 && (
        <div>
          <Section title="DEND Syndrome — Developmental delay + Epilepsy + Neonatal Diabetes (KCNJ11)" color={ACCENT5}>
            <Alert color={ACCENT5}>
              <strong>DEND = Developmental delay + Epilepsy + Neonatal Diabetes.</strong> Caused by severe KCNJ11 gain-of-function mutations where K-ATP channel dysfunction extends to neurons. High-dose sulfonylurea (glibenclamide) closes neuronal K-ATP → partial reversal of neurological features.
            </Alert>
            <table className="table table-sm table-hover mt-2">
              <thead>
                <tr><th>Phenotype</th><th>Example Mutations</th><th>Severity</th><th>Sulfo Neurological Benefit</th></tr>
              </thead>
              <tbody>
                {(breakdown?.dend_spectrum || []).map((d, i) => (
                  <tr key={i}>
                    <td><strong style={{ color: i === 2 ? ACCENT5 : i === 1 ? ACCENT : ACCENT7 }}>{d.phenotype}</strong></td>
                    <td className="font-monospace small">{d.examples}</td>
                    <td><Badge text={d.severity} color={i === 2 ? ACCENT5 : i === 1 ? ACCENT : ACCENT7} /></td>
                    <td className="small">{d.sulfo_neuro}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="row g-3 mt-1">
              {[
                { t: "D — Developmental Delay", d: "Moderate-to-severe intellectual disability; motor delay. Caused by K-ATP GOF in cortical and cerebellar neurons → hyperpolarized neurons → impaired synaptic transmission.", c: ACCENT5 },
                { t: "E — Epilepsy", d: "Focal or multifocal seizures; often poorly controlled on standard AEDs. Neuronal K-ATP channels in cortex and hippocampus → GOF → altered resting potential → hyperexcitability.", c: ACCENT3 },
                { t: "N — Neonatal Diabetes", d: "Classic PNDM component. Pancreatic Kir6.2 K-ATP GOF → no insulin secretion. Responds to sulfonylurea as does the neurological component.", c: ACCENT },
                { t: "Sulfo Neuro Response", d: "High-dose glibenclamide (0.8-1.0 mg/kg/day) closes neuronal K-ATP → motor improvement, cognitive gains, seizure ↓. Benefit begins weeks-months. Start as early as possible.", c: ACCENT4 },
              ].map((item, i) => (
                <div key={i} className="col-md-3">
                  <div className="card h-100 shadow-sm">
                    <div className="card-body p-2">
                      <div className="fw-bold small mb-1" style={{ color: item.c }}>{item.t}</div>
                      <div className="small text-muted">{item.d}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Neonatal Diabetes Mellitus — Comparison Table" color={ACCENT}>
            {breakdown?.ndm_comparison && (() => {
              const cmp = breakdown.ndm_comparison;
              return (
                <div className="table-responsive">
                  <table className="table table-sm table-bordered">
                    <thead style={{ background: ACCENT, color: 'white' }}>
                      <tr>{cmp.headers.map((h, i) => <th key={i} className="text-white">{h}</th>)}</tr>
                    </thead>
                    <tbody>
                      {cmp.rows.map((row, ri) => (
                        <tr key={ri}>
                          <td className="fw-bold">{row[0]}</td>
                          {row.slice(1).map((cell, ci) => (
                            <td key={ci} style={{
                              background: cell.includes('NEVER') ? ACCENT3 + '18' :
                                          cell.includes('YES') ? ACCENT4 + '18' :
                                          cell.includes('POSITIVE') ? ACCENT5 + '18' : undefined,
                              color: cell.includes('NEVER') ? ACCENT3 :
                                     cell.includes('YES') ? ACCENT4 :
                                     cell.includes('POSITIVE') ? ACCENT5 : undefined,
                              fontWeight: cell.includes('NEVER') || cell.includes('YES') || cell.includes('IMMEDIATE') ? 'bold' : undefined,
                            }}>{cell}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              );
            })()}
          </Section>

          <Section title="Wolcott-Rallison Syndrome (EIF2AK3 / PERK)" color={ACCENT6}>
            <Alert color={ACCENT6}>
              <strong>Wolcott-Rallison triad:</strong> PNDM + Epiphyseal dysplasia + Hepatic dysfunction. AR (bi-allelic EIF2AK3 LOF). PERK = ER stress sensor; loss → unresolved ER stress → beta-cell, chondrocyte, hepatocyte apoptosis. Most common PNDM cause in consanguineous families (Middle East, North Africa).
            </Alert>
            <div className="row g-2 mt-1">
              {[
                { t: "PNDM", d: "Onset <6 months; insulin required lifelong. No sulfo response (not K-ATP)." },
                { t: "Epiphyseal dysplasia", d: "Short stature, joint abnormalities, fracture risk. Radiographs from 18 months. No specific treatment." },
                { t: "Hepatic dysfunction", d: "Hepatitis, cirrhosis, hepatic failure (life-limiting in ~40%). LFTs + ultrasound surveillance." },
                { t: "Exocrine insufficiency", d: "Pancreatic enzyme replacement. Steatorrhea, malnutrition risk." },
              ].map((item, i) => (
                <div key={i} className="col-md-3">
                  <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT6}` }}>
                    <div className="card-body p-2">
                      <div className="fw-bold small mb-1" style={{ color: ACCENT6 }}>{item.t}</div>
                      <div className="small text-muted">{item.d}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {tab === 3 && (
        <div>
          {definitions && (
            <>
              <Section title="Disease Overview" color={ACCENT}>
                <table className="table table-sm">
                  <tbody>
                    <tr><td className="fw-bold">Full name</td><td>{definitions.disease_overview?.full_name}</td></tr>
                    <tr><td className="fw-bold">Definition</td><td>{definitions.disease_overview?.definition}</td></tr>
                    <tr><td className="fw-bold">OMIM</td><td>{definitions.disease_overview?.omim_disease}</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>{definitions.disease_overview?.prevalence}</td></tr>
                    <tr><td className="fw-bold">Key contrast (TNDM1)</td><td style={{ color: ACCENT3 }}>{definitions.disease_overview?.key_contrast_tndm}</td></tr>
                    {Object.entries(definitions.disease_overview?.mim_genes || {}).map(([g, omim]) => (
                      <tr key={g}><td className="fw-bold font-monospace">{g}</td><td>{omim}</td></tr>
                    ))}
                  </tbody>
                </table>
              </Section>

              <Section title="Gene Definitions" color={ACCENT2}>
                {Object.entries(definitions.genes || {}).map(([gene, info]) => (
                  <div key={gene} className="mb-3">
                    <h6 className="fw-bold" style={{ color: ACCENT2 }}>{gene} — {info.full_name}</h6>
                    <table className="table table-sm">
                      <tbody>
                        <tr><td>Size</td><td>{info.size}</td></tr>
                        <tr><td>Function</td><td>{info.function}</td></tr>
                        <tr><td>Locus</td><td>{info.locus}</td></tr>
                        <tr><td>OMIM</td><td>{info.omim}</td></tr>
                        <tr><td>Key fact</td><td style={{ color: ACCENT3 }}>{info.key_fact}</td></tr>
                      </tbody>
                    </table>
                  </div>
                ))}
              </Section>

              <Section title="K-ATP Channel Mechanism" color={ACCENT}>
                <table className="table table-sm">
                  <tbody>
                    {Object.entries(definitions.katp_channel || {}).map(([k, v]) => (
                      <tr key={k}><td className="fw-bold text-capitalize" style={{ minWidth: 160 }}>{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </Section>

              <Section title="DEND Syndrome — Full Definition" color={ACCENT5}>
                <table className="table table-sm">
                  <tbody>
                    {Object.entries(definitions.dend_syndrome || {}).map(([k, v]) => (
                      <tr key={k}><td className="fw-bold text-capitalize" style={{ minWidth: 140 }}>{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </Section>

              <Section title="Treatment Summary" color={ACCENT4}>
                {Object.entries(definitions.treatment_summary || {}).map(([subtype, tx]) => (
                  <div key={subtype} className="mb-3">
                    <div className="fw-bold mb-1" style={{ color: subtype.includes('KCNJ') ? ACCENT4 : subtype === 'what_NOT_to_do' ? ACCENT3 : ACCENT8 }}>
                      {subtype.replace(/_/g, ' ')}
                    </div>
                    {Array.isArray(tx) ? (
                      <ul className="small mb-0">
                        {tx.map((item, i) => <li key={i} style={{ color: ACCENT3 }}>{item}</li>)}
                      </ul>
                    ) : (
                      <table className="table table-sm">
                        <tbody>
                          {Object.entries(tx).map(([k, v]) => (
                            <tr key={k}><td className="fw-bold" style={{ minWidth: 140 }}>{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                ))}
              </Section>

              <Section title="Diagnostics & Contrasts" color={ACCENT7}>
                <div className="row">
                  <div className="col-md-6">
                    <h6 style={{ color: ACCENT7 }}>Diagnostic Sequence</h6>
                    <table className="table table-sm">
                      <tbody>
                        {Object.entries(definitions.diagnostics || {}).map(([k, v]) => (
                          <tr key={k}><td className="fw-bold" style={{ minWidth: 130 }}>{k.replace(/_/g, ' ')}</td><td className="small">{v}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="col-md-6">
                    <h6 style={{ color: ACCENT7 }}>Key Contrasts</h6>
                    {Object.entries(definitions.contrasts || {}).map(([k, v]) => (
                      <Alert key={k} color={ACCENT7}>
                        <strong>{k.replace(/_/g, ' → ')}</strong>
                        <div className="small mt-1">{v}</div>
                      </Alert>
                    ))}
                  </div>
                </div>
              </Section>

              <Section title="Surveillance — Lifelong" color={ACCENT}>
                <table className="table table-sm">
                  <tbody>
                    {Object.entries(definitions.surveillance_lifelong || {}).map(([k, v]) => (
                      <tr key={k}><td className="fw-bold" style={{ minWidth: 200 }}>{k.replace(/_/g, ' ')}</td><td className="small">{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </>
          )}
        </div>
      )}
    </div>
  );
}
