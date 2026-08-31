'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Tip Scaffold Pearls', 'Definitions'];

// JBTS25 colour scheme — CEP104 / Ciliary Tip TOG Scaffold / TTBK2 Co-Scaffold / CLUAP1-IFT-B1
// Ciliary-tip teal-cyan tones; TTBK2 amber signal; IFT-B1 accumulation orange; MENA steel founder
const ACCENT   = '#00695c';   // deep teal — ciliary tip scaffold / TOG domain
const ACCENT2  = '#00796b';   // medium teal — CEP104 coiled-coil / homo-oligomerisation
const ACCENT3  = '#1b5e20';   // forest green — cerebellar / neurological
const ACCENT4  = '#0277bd';   // sky blue — renal NPHP-like shortened tubular cilia
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#e65100';   // deep orange — IFT-B1 tip accumulation (diagnostic)
const ACCENT7  = '#f57f17';   // amber — TTBK2 co-scaffold / CP110 cap regulation
const ACCENT8  = '#455a64';   // steel blue — MENA founder Thr544Met
const ACCENT9  = '#558b2f';   // olive — hepatic CHF
const ACCENT10 = '#6a1b9a';   // violet — retinal rod-cone shortened connecting cilia

const SEED = 463;
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

function Section({ title, color, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

function Bar({ label, value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{value}</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function JBTS25Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts25/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts25/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts25/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading CEP104 / JBTS25 dashboard…</p></div>;
  if (error)   return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  const kpi = overview?.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT2} 60%, ${ACCENT7} 100%)`, color: '#fff' }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div className="fs-1">🧬</div>
          <div>
            <h4 className="mb-0 fw-bold">JBTS25 — CEP104 Joubert Syndrome Type 25</h4>
            <div className="small opacity-90">
              CEP104 · Centrosomal Protein 104kDa · Ciliary Tip TOG Scaffold · TTBK2 Co-Scaffold · CLUAP1/IFT-B1 Coupler
            </div>
            <div className="small opacity-80 mt-1">
              1p36.32 · ~1338 aa · OMIM Gene *616078 · Disease #616778 · No MKS Tier · Cilia SHORT (not absent) · MENA Founder Thr544Met
            </div>
          </div>
          <div className="ms-auto text-end small opacity-80">
            <div>Seed {SEED} · N={N_COHORT}</div>
            <div>AR · Biallelic LOF</div>
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

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && overview && (
        <div>
          {/* KPI row */}
          <div className="row g-2 mb-3">
            <KPI label="Cohort (N)" value={kpi.total_patients} color={ACCENT} />
            <KPI label="MTS %" value={`${kpi.mts_pct}%`} color={ACCENT} />
            <KPI label="Ataxia" value={`${kpi.ataxia_pct}%`} color={ACCENT3} />
            <KPI label="Hypotonia" value={`${kpi.hypotonia_pct}%`} color={ACCENT3} />
            <KPI label="OMA" value={`${kpi.oma_pct}%`} color={ACCENT3} />
            <KPI label="Breathing" value={`${kpi.breathing_pct}%`} color={ACCENT3} />
            <KPI label="Retinal" value={`${kpi.retinal_pct}%`} color={ACCENT10} />
            <KPI label="Renal" value={`${kpi.renal_pct}%`} color={ACCENT4} />
            <KPI label="Hepatic" value={`${kpi.hepatic_pct}%`} color={ACCENT9} />
            <KPI label="Polydactyly" value={`${kpi.poly_pct}%`} color={ACCENT5} />
            <KPI label="ID" value={`${kpi.id_pct}%`} color={ACCENT2} />
            <KPI label="ESRD" value={`${kpi.esrd_pct}%`} color={ACCENT6} />
          </div>

          {/* Alerts */}
          {overview.alerts && (
            <div className="mb-3">
              <Alert color={ACCENT}>
                <strong>🔬 Ciliary Tip TOG Scaffold Mechanism:</strong>{' '}
                {overview.alerts.ciliary_tip_mechanism}
              </Alert>
              <Alert color={ACCENT7}>
                <strong>⚡ TTBK2 — CEP104 Axis (NOT Allelic):</strong>{' '}
                {overview.alerts.ttbk2_distinction}
              </Alert>
              <Alert color={ACCENT6}>
                <strong>📡 CLUAP1 / IFT-B1 Coupling:</strong>{' '}
                {overview.alerts.cluap1_ift_b1_axis}
              </Alert>
              <Alert color={ACCENT8}>
                <strong>🌍 MENA Founder — Thr544Met:</strong>{' '}
                {overview.alerts.mena_founder}
              </Alert>
            </div>
          )}

          {/* Key facts */}
          <Section title="CEP104 / JBTS25 — Key Clinical Facts" color={ACCENT}>
            <ul className="small mb-0">
              {(overview.key_facts || []).map((f, i) => (
                <li key={i} className="mb-1">{f}</li>
              ))}
            </ul>
          </Section>

          {/* Patient table */}
          <Section title={`Patient Cohort (N=${N_COHORT}, seed ${SEED})`} color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead>
                  <tr>
                    <th>ID</th><th>Age</th><th>Sex</th><th>Ethnicity</th>
                    <th>Allele Class</th><th>Variant</th>
                    <th>MTS</th><th>Ataxia</th><th>Hypotonia</th><th>OMA</th>
                    <th>Breathing</th><th>Retinal</th><th>Renal</th><th>Hepatic</th><th>Poly</th><th>ID</th><th>ESRD</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.patients || []).map(p => (
                    <tr key={p.id}>
                      <td className="fw-bold">{p.id}</td>
                      <td>{p.age}</td>
                      <td>{p.sex}</td>
                      <td>{p.ethnicity}</td>
                      <td>{p.allele_class}</td>
                      <td><code className="small">{p.variant}</code></td>
                      <td>{p.mts ? '✅' : '—'}</td>
                      <td>{p.ataxia ? '✅' : '—'}</td>
                      <td>{p.hypotonia ? '✅' : '—'}</td>
                      <td>{p.oma ? '✅' : '—'}</td>
                      <td>{p.breathing ? '✅' : '—'}</td>
                      <td>{p.retinal ? '✅' : '—'}</td>
                      <td>{p.renal ? '✅' : '—'}</td>
                      <td>{p.hepatic ? '✅' : '—'}</td>
                      <td>{p.poly ? '✅' : '—'}</td>
                      <td>{p.id_flag ? '✅' : '—'}</td>
                      <td>{p.esrd ? '✅' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 1: Diagnostic Breakdown ── */}
      {tab === 1 && breakdown && (
        <div className="row g-3">
          <div className="col-md-4">
            <Section title="Ethnicity Distribution" color={ACCENT}>
              {(breakdown.ethnicity_distribution || []).map((e, i) => (
                <Bar key={i} label={e.ethnicity} value={e.count} max={N_COHORT} color={ACCENT} />
              ))}
            </Section>
          </div>

          <div className="col-md-4">
            <Section title="Allele Class Distribution" color={ACCENT2}>
              {(breakdown.allele_class_distribution || []).map((a, i) => (
                <Bar key={i} label={a.allele_class} value={a.count} max={N_COHORT} color={ACCENT2} />
              ))}
            </Section>
          </div>

          <div className="col-md-4">
            <Section title="Phenotype Summary" color={ACCENT3}>
              {Object.entries(breakdown.phenotype_summary || {}).map(([k, v]) => (
                <Bar key={k} label={k.toUpperCase()} value={v.n} max={N_COHORT} color={v.pct > 70 ? ACCENT3 : v.pct > 30 ? ACCENT7 : ACCENT4} />
              ))}
            </Section>
          </div>

          {/* Notable variants */}
          <div className="col-12">
            <Section title="Notable CEP104 Pathogenic Variants" color={ACCENT5}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead className="table-dark">
                    <tr>
                      <th>Variant</th><th>cDNA</th><th>Domain</th>
                      <th>Population</th><th>Severity</th><th>Mechanism</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.notable_variants || []).map((v, i) => (
                      <tr key={i}>
                        <td className="fw-bold"><code>{v.name}</code></td>
                        <td><code className="small">{v.cdna}</code></td>
                        <td className="small">{v.domain}</td>
                        <td className="small">{v.population}</td>
                        <td>
                          <span className="badge" style={{
                            background: v.severity.includes('Severe') ? ACCENT6 :
                                        v.severity.includes('Moderate') ? ACCENT7 : ACCENT4
                          }}>
                            {v.severity}
                          </span>
                        </td>
                        <td className="small">{v.mechanism}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 2: Tip Scaffold Pearls ── */}
      {tab === 2 && (
        <div>
          <Section title="CEP104 Protein Domain Architecture (~1338 aa)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT, color: '#fff' }}>
                  <tr>
                    <th>Domain</th><th>Residues</th><th>Function</th><th>LOF Consequence</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="fw-bold" style={{ color: ACCENT }}>TOG Domain</td>
                    <td>aa 1–280</td>
                    <td>HEAT-repeat β-propeller; binds free αβ-tubulin dimers; delivers tubulin to axonemal plus-ends; CLUAP1 co-binding</td>
                    <td>Defective tubulin delivery to ciliary tip → axonemal elongation impaired → cilia shortened</td>
                  </tr>
                  <tr>
                    <td className="fw-bold" style={{ color: ACCENT7 }}>Central Linker / TTBK2-Scaffold</td>
                    <td>aa 281–600</td>
                    <td>TTBK2 co-scaffold interaction motifs (aa 310–380, 490–555); TTBK2 kinase activity sustained at ciliary tip; MPP9 / KIF2A phosphorylation checkpoint</td>
                    <td>TTBK2 de-stabilised at tip → MPP9 hypophosphorylation → CP110 cap re-engagement → cilia retraction</td>
                  </tr>
                  <tr>
                    <td className="fw-bold" style={{ color: ACCENT2 }}>Coiled-Coil Domain</td>
                    <td>aa 601–950</td>
                    <td>Homo-oligomerisation; anchors CEP104 dimers to central pair microtubule distal ends; FOP/FGFR1OP surface for centriolar satellite tethering</td>
                    <td>Monomeric CEP104 loses ciliary tip avidity; centriolar satellite → ciliary tip transfer failed</td>
                  </tr>
                  <tr>
                    <td className="fw-bold" style={{ color: ACCENT6 }}>C-Terminal Extension / CLUAP1 Module</td>
                    <td>aa 951–1338</td>
                    <td>CLUAP1 (IFT38 / IFT-B1) interaction; IFT-B1 retrograde tip coupling; distal axonemal cap stabilisation; microtubule polymerisation checkpoint</td>
                    <td>IFT-B1 tip coupling failure → IFT particle accumulation at ciliary tip (EM diagnostic) → tubulin pool depletion</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Section>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="CEP104 LOF Pathway (JBTS25)" color={ACCENT7}>
                <div className="small">
                  <div className="d-flex align-items-start mb-2">
                    <span className="badge me-2" style={{ background: ACCENT, minWidth: 24 }}>1</span>
                    <span><strong>TOG failure:</strong> CEP104 biallelic LOF → defective αβ-tubulin dimer delivery to axonemal plus-ends → reduced polymerisation rate at ciliary tip</span>
                  </div>
                  <div className="d-flex align-items-start mb-2">
                    <span className="badge me-2" style={{ background: ACCENT7, minWidth: 24 }}>2</span>
                    <span><strong>TTBK2 de-stabilisation:</strong> CEP104 linker LOF → TTBK2 co-scaffold absent at ciliary tip → MPP9 (CP110 lock) hypophosphorylated → CP110 cap re-engages → tip retraction</span>
                  </div>
                  <div className="d-flex align-items-start mb-2">
                    <span className="badge me-2" style={{ background: ACCENT6, minWidth: 24 }}>3</span>
                    <span><strong>IFT-B1 tip accumulation:</strong> CLUAP1 coupling failure → IFT-B1 retrograde dissociation impaired → IFT particles accumulate at ciliary tip (EM finding)</span>
                  </div>
                  <div className="d-flex align-items-start mb-2">
                    <span className="badge me-2" style={{ background: ACCENT3, minWidth: 24 }}>4</span>
                    <span><strong>Cilia SHORT (not absent):</strong> ~30–50% WT length in fibroblasts; beat frequency reduced (tip instability); ultrastructure NORMAL (DA, TZ gate intact)</span>
                  </div>
                  <div className="d-flex align-items-start mb-2">
                    <span className="badge me-2" style={{ background: ACCENT3, minWidth: 24 }}>5</span>
                    <span><strong>Hedgehog partial failure:</strong> Shorter cilia → reduced SMO trafficking to ciliary membrane → Hedgehog transduction partially impaired → cerebellar vermis hypoplasia → MTS</span>
                  </div>
                </div>
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="Diagnostic DDx vs Other JBTS Subtypes" color={ACCENT5}>
                <div className="table-responsive">
                  <table className="table table-sm small">
                    <thead>
                      <tr>
                        <th>Feature</th><th>JBTS25 CEP104</th><th>JBTS22 CEP83</th><th>JBTS24 ZNF423</th><th>JBTS19 B9D1</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="fw-bold">Cilia</td>
                        <td style={{ color: ACCENT7 }}>SHORT (30–50% WT)</td>
                        <td style={{ color: ACCENT6 }}>ABSENT</td>
                        <td style={{ color: ACCENT3 }}>NORMAL</td>
                        <td style={{ color: ACCENT6 }}>ABSENT</td>
                      </tr>
                      <tr>
                        <td className="fw-bold">PCD Beat Freq</td>
                        <td>Reduced</td>
                        <td>None</td>
                        <td>Normal</td>
                        <td>None</td>
                      </tr>
                      <tr>
                        <td className="fw-bold">EM Tip Sign</td>
                        <td style={{ color: ACCENT6 }}>Enlarged tips (IFT-B1 accum.)</td>
                        <td>No cilia</td>
                        <td>Normal</td>
                        <td>No cilia</td>
                      </tr>
                      <tr>
                        <td className="fw-bold">MKS Tier</td>
                        <td>No</td>
                        <td>No</td>
                        <td>No</td>
                        <td>Yes (B9D1 biallelic null)</td>
                      </tr>
                      <tr>
                        <td className="fw-bold">Mechanism</td>
                        <td>Tip scaffold</td>
                        <td>DA foundation</td>
                        <td>TF / BMP-SMAD</td>
                        <td>TZ gate (B9)</td>
                      </tr>
                      <tr>
                        <td className="fw-bold">Renal %</td>
                        <td>~18%</td>
                        <td>~68% (highest)</td>
                        <td>~35%</td>
                        <td>~35%</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Section>
            </div>
          </div>

          <Section title="TTBK2 — CEP104 Functional Axis (NOT Allelic)" color={ACCENT7}>
            <div className="row g-2">
              <div className="col-md-6">
                <div className="p-3 rounded" style={{ background: ACCENT7 + '18', border: `1px solid ${ACCENT7}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT7 }}>JBTS25 — CEP104 Biallelic LOF</div>
                  <ul className="small mb-0">
                    <li>Mode: Autosomal <strong>Recessive</strong></li>
                    <li>Mechanism: Ciliary tip TTBK2 co-scaffold failure</li>
                    <li>Phenotype: JBTS25 — MTS, cerebellar ataxia, ciliopathy</li>
                    <li>MTS: Yes (100%)</li>
                    <li>Age onset: Neonatal hypotonia, early childhood ataxia</li>
                    <li>Ciliary biology: Primary cilia SHORT, IFT-B1 tip accumulation</li>
                  </ul>
                </div>
              </div>
              <div className="col-md-6">
                <div className="p-3 rounded" style={{ background: ACCENT6 + '18', border: `1px solid ${ACCENT6}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT6 }}>SCA11 — TTBK2 Dominant LOF</div>
                  <ul className="small mb-0">
                    <li>Mode: Autosomal <strong>Dominant</strong> (gain-of-function or haploinsufficiency)</li>
                    <li>Mechanism: Non-ciliary TTBK2 kinase activity (microtubule dynamics, tau phosphorylation)</li>
                    <li>Phenotype: Spinocerebellar Ataxia Type 11 (#604432) — neurodegeneration</li>
                    <li>MTS: <strong>No</strong> (brain MRI: cerebellar atrophy, NOT vermis hypoplasia)</li>
                    <li>Age onset: Adult-onset progressive ataxia</li>
                    <li>Ciliary biology: Primary cilia NORMAL (non-ciliary TTBK2 function affected)</li>
                  </ul>
                </div>
              </div>
            </div>
          </Section>

          <Section title="Surveillance Protocol (CEP104 / JBTS25)" color={ACCENT4}>
            <div className="row g-2 small">
              {[
                { organ: '🫘 Renal', detail: 'Annual US + creatinine/eGFR from diagnosis; ESRD median ~22–28yr when renal affected (lower penetrance than JBTS22); renal transplant curative (cell-autonomous)' },
                { organ: '👁️ Retinal', detail: 'ERG + fundus photography from age 3; annual review; rod-cone dystrophy progressive; shortened connecting cilia; no established treatment (gene therapy recruiting)' },
                { organ: '🫀 Hepatic', detail: 'LFTs + hepatic US at diagnosis; repeat 2-yearly if abnormal; CHF rate ~8% (lower than TMEM67/JBTS6 COACH 30%)' },
                { organ: '🧠 Neurological', detail: 'Brain MRI at diagnosis (MTS confirmation; MTS present but may be subtle); annual physiotherapy; cochlear screen; cognitive assessment' },
                { organ: '🔬 Cilia EM', detail: 'Nasal brushing videomicroscopy + EM at diagnosis: cilia present (distinguish from JBTS22 absent), shortened, beat frequency reduced, enlarged tips (IFT-B1 accumulation); IFT-B1 tip co-immunoprecipitation for VUS' },
              ].map((s, i) => (
                <div key={i} className="col-md-6">
                  <div className="p-2 rounded" style={{ background: ACCENT4 + '12', border: `1px solid ${ACCENT4}40` }}>
                    <div className="fw-bold small" style={{ color: ACCENT4 }}>{s.organ}</div>
                    <div className="small text-muted">{s.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && definitions && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <Section title="Gene / Disease Identity" color={ACCENT}>
                <table className="table table-sm small">
                  <tbody>
                    <tr><td className="fw-bold text-muted">Gene</td><td>CEP104 (KIAA0562; FAP256 orthologue)</td></tr>
                    <tr><td className="fw-bold text-muted">Full name</td><td>{definitions.gene_full_name}</td></tr>
                    <tr><td className="fw-bold text-muted">OMIM Gene</td><td>*{definitions.omim_gene}</td></tr>
                    <tr><td className="fw-bold text-muted">OMIM Disease</td><td>#{definitions.omim_jbts25}</td></tr>
                    <tr><td className="fw-bold text-muted">Chromosome</td><td>{definitions.chromosome}</td></tr>
                    <tr><td className="fw-bold text-muted">Protein</td><td>{definitions.protein_size}</td></tr>
                    <tr><td className="fw-bold text-muted">Inheritance</td><td>{definitions.inheritance}</td></tr>
                    <tr><td className="fw-bold text-muted">MKS Tier</td><td>{definitions.mks_tier ? 'Yes' : 'No — ciliary tip scaffold; TZ B9-complex gate INTACT'}</td></tr>
                    <tr><td className="fw-bold text-muted">Mechanism class</td><td>{definitions.mechanism_class}</td></tr>
                    <tr><td className="fw-bold text-muted">Cilia phenotype</td><td style={{ color: ACCENT7 }}>{definitions.cilia_phenotype}</td></tr>
                    <tr><td className="fw-bold text-muted">Hedgehog impact</td><td>{definitions.hedgehog_impact}</td></tr>
                    <tr><td className="fw-bold text-muted">MTS mechanism</td><td>{definitions.mts_mechanism}</td></tr>
                    <tr><td className="fw-bold text-muted">Frequency</td><td>{definitions.frequency}</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="Mechanism Detail" color={ACCENT7}>
                <p className="small">{definitions.mechanism_detail}</p>
              </Section>

              <Section title="Allelic Diseases" color={ACCENT2}>
                {(definitions.allelic_diseases || []).length === 0 ? (
                  <p className="small text-muted">No known allelic syndrome with distinct phenotype from allele-class threshold (unlike ZNF423-JBTS24/NPHP10 or CEP83-JBTS22/NPHP18).</p>
                ) : (
                  <ul className="small">{(definitions.allelic_diseases || []).map((a, i) => <li key={i}>{a}</li>)}</ul>
                )}
              </Section>

              <Section title="Founder Variants" color={ACCENT8}>
                <div className="table-responsive">
                  <table className="table table-sm small">
                    <thead>
                      <tr><th>Variant</th><th>Population</th><th>Domain</th><th>Severity</th></tr>
                    </thead>
                    <tbody>
                      {(definitions.founder_variants || []).map((fv, i) => (
                        <tr key={i}>
                          <td><code>{fv.variant}</code></td>
                          <td>{fv.population}</td>
                          <td className="small">{fv.domain}</td>
                          <td><span className="badge" style={{ background: fv.severity.includes('Severe') ? ACCENT6 : fv.severity.includes('Moderate') ? ACCENT7 : ACCENT4 }}>{fv.severity}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Section>
            </div>
          </div>

          <Section title="Key DDx (Differential Diagnosis)" color={ACCENT5}>
            <ul className="small">
              {(definitions.key_ddx || []).map((d, i) => <li key={i} className="mb-1">{d}</li>)}
            </ul>
          </Section>

          <Section title="Surveillance & Treatment" color={ACCENT4}>
            <div className="row g-2 small">
              <div className="col-md-6">
                <div className="fw-bold mb-1" style={{ color: ACCENT4 }}>Surveillance</div>
                {Object.entries(definitions.surveillance_protocol || {}).map(([k, v]) => (
                  <div key={k} className="mb-1"><span className="fw-bold text-capitalize">{k}:</span> {v}</div>
                ))}
              </div>
              <div className="col-md-6">
                <div className="fw-bold mb-1" style={{ color: ACCENT3 }}>Treatment</div>
                {Object.entries(definitions.treatment || {}).map(([k, v]) => (
                  <div key={k} className="mb-1"><span className="fw-bold text-capitalize">{k.replace('_note', ' (note)')}:</span> {v}</div>
                ))}
              </div>
            </div>
          </Section>
        </div>
      )}

      {/* Footer nav */}
      <div className="mt-4 pt-3 border-top small text-muted d-flex gap-3 flex-wrap">
        <Link href="/jbts24">← JBTS24 ZNF423</Link>
        <Link href="/jbts23">← JBTS23 KIAA0586</Link>
        <Link href="/">Home</Link>
      </div>
    </div>
  );
}
