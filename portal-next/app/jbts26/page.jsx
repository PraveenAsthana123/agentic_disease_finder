'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'IFT-B Base Adapter Pearls', 'Definitions'];

// JBTS26 colour scheme — KIAA0556 / IFT-B Basal Body Adapter / CPLANE Coupler
// Indigo-blue IFT-B base tones; CPLANE violet; MENA steel founder; pBB scaffold slate
const ACCENT   = '#283593';   // deep indigo — IFT-B base assembly / KIAA0556 scaffold
const ACCENT2  = '#1565c0';   // medium blue — IFT-B1 core coupling zone 1
const ACCENT3  = '#1b5e20';   // forest green — cerebellar / neurological
const ACCENT4  = '#0277bd';   // sky blue — renal NPHP-like shortened tubular cilia
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#e65100';   // deep orange — IFT-B base accumulation EM (diagnostic)
const ACCENT7  = '#6a1b9a';   // violet — CPLANE coupling / INTU-FUZ-WDPCP interface
const ACCENT8  = '#455a64';   // steel blue — MENA founder Thr680Met
const ACCENT9  = '#558b2f';   // olive — hepatic CHF
const ACCENT10 = '#7b1fa2';   // purple — retinal rod-cone shortened connecting cilia

const SEED = 465;
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

export default function JBTS26Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts26/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts26/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts26/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading KIAA0556 / JBTS26 dashboard…</p></div>;
  if (error)   return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  const kpi = overview?.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT2} 60%, ${ACCENT7} 100%)`, color: '#fff' }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div className="fs-1">🧬</div>
          <div>
            <h4 className="mb-0 fw-bold">JBTS26 — KIAA0556 Joubert Syndrome Type 26</h4>
            <div className="small opacity-90">
              KIAA0556 · C14orf179 · IFT-B Basal Body Assembly Platform Adapter · CPLANE Coupler · DA Outer Scaffold TZ Tether
            </div>
            <div className="small opacity-80 mt-1">
              14q24.2 · ~1311 aa · OMIM Gene *616650 · Disease #616532 · No MKS Tier · Cilia SHORT (not absent) · MENA Founder Thr680Met
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
                <strong>🔬 IFT-B Basal Body Adapter Mechanism:</strong>{' '}
                {overview.alerts.ift_b_base_adapter}
              </Alert>
              <Alert color={ACCENT7}>
                <strong>🔗 KIAA0556 — CPLANE Coupling (INTU/FUZ/WDPCP):</strong>{' '}
                {overview.alerts.cplane_coupling}
              </Alert>
              <Alert color={ACCENT8}>
                <strong>🌍 MENA Founder — Thr680Met:</strong>{' '}
                {overview.alerts.mena_founder}
              </Alert>
              <Alert color={ACCENT2}>
                <strong>🇪🇺 European Cluster — Arg341Cys:</strong>{' '}
                {overview.alerts.european_cluster}
              </Alert>
            </div>
          )}

          {/* Key facts */}
          <Section title="KIAA0556 / JBTS26 — Key Clinical Facts" color={ACCENT}>
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
            <Section title="Notable KIAA0556 Pathogenic Variants" color={ACCENT5}>
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

      {/* ── TAB 2: IFT-B Base Adapter Pearls ── */}
      {tab === 2 && (
        <div>
          <Section title="KIAA0556 Protein Domain Architecture (~1311 aa)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT, color: '#fff' }}>
                  <tr>
                    <th>Domain</th><th>Residues</th><th>Function</th><th>LOF Consequence</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="fw-bold" style={{ color: ACCENT }}>MT-Binding / Centrosomal Targeting</td>
                    <td>aa 1–300</td>
                    <td>HEAT-1 and HEAT-2 repeats bind polymerised microtubule protofilaments; CTS (aa 270–300) directs KIAA0556 to pericentriolar material and basal body proximal end; IFT88/IFT52 contact patches (outer surface aa 131–260)</td>
                    <td>Reduced centrosomal targeting → IFT-B scaffold misplaced; MT-binding variants (Gly205Asp) → partial localisation preserved</td>
                  </tr>
                  <tr>
                    <td className="fw-bold" style={{ color: ACCENT7 }}>Central IFT-B Adapter / CPLANE Coupling</td>
                    <td>aa 301–800</td>
                    <td>Zone 1 (aa 301–500): IFT-B core scaffold; IFT88 docking; Zone 2 (aa 501–700): CPLANE coupling; INTU binding surface; Zone 3 (aa 701–800): FUZ/WDPCP binding; CC homo-dimer platform for IFT assembly</td>
                    <td>IFT-B1 assembly platform collapse at DA outer scaffold → anterograde IFT trains impaired; CPLANE uncoupling → retrograde IFT-A partially impaired</td>
                  </tr>
                  <tr>
                    <td className="fw-bold" style={{ color: ACCENT2 }}>C-Terminal Basal Body Scaffold / TZ Tether</td>
                    <td>aa 801–1311</td>
                    <td>pBB scaffold (aa 801–1000): anchors KIAA0556 to proximal basal body; NPHP4 interaction (aa 900–970); CEP290 co-binding (aa 1050–1150); TZ outer-plate tethering (NOT B9 inner gate)</td>
                    <td>pBB scaffold failure (Leu820Pro misfolding) → IFT-B adapter loses basal body anchor; TZ tether truncation (Glu1060Ter) → IFT-B base dispersion; B9-gate INTACT</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Section>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="KIAA0556 LOF Pathway (JBTS26)" color={ACCENT7}>
                <div className="small">
                  <div className="d-flex align-items-start mb-2">
                    <span className="badge me-2" style={{ background: ACCENT, minWidth: 24 }}>1</span>
                    <span><strong>IFT-B adapter collapse:</strong> KIAA0556 biallelic LOF → IFT-B1 assembly platform at DA outer scaffold disrupted → IFT88/IFT52 docking reduced → anterograde IFT train assembly impaired at ciliary BASE</span>
                  </div>
                  <div className="d-flex align-items-start mb-2">
                    <span className="badge me-2" style={{ background: ACCENT7, minWidth: 24 }}>2</span>
                    <span><strong>CPLANE uncoupling:</strong> KIAA0556 central adapter LOF → INTU/FUZ binding reduced → retrograde IFT-A assembly partially impaired → compound anterograde + retrograde IFT deficit</span>
                  </div>
                  <div className="d-flex align-items-start mb-2">
                    <span className="badge me-2" style={{ background: ACCENT6, minWidth: 24 }}>3</span>
                    <span><strong>IFT-B BASE accumulation:</strong> IFT-B subunits fail to enter IFT trains → accumulate at ciliary BASE on EM (diagnostic — contrast: IFT-B tip accumulation in CEP104/JBTS25 is tip scaffold failure)</span>
                  </div>
                  <div className="d-flex align-items-start mb-2">
                    <span className="badge me-2" style={{ background: ACCENT3, minWidth: 24 }}>4</span>
                    <span><strong>Cilia SHORT (not absent):</strong> ~25–55% WT length; DA and TZ B9-gate intact; cilia form but axonemal elongation impaired due to IFT cargo deficit</span>
                  </div>
                  <div className="d-flex align-items-start mb-2">
                    <span className="badge me-2" style={{ background: ACCENT3, minWidth: 24 }}>5</span>
                    <span><strong>Hedgehog partial failure:</strong> Shorter cilia → reduced SMO trafficking → Hedgehog transduction partially impaired → cerebellar vermis hypoplasia → MTS</span>
                  </div>
                </div>
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="Diagnostic DDx — IFT-B Base vs Tip vs Absent" color={ACCENT5}>
                <div className="table-responsive">
                  <table className="table table-sm small">
                    <thead>
                      <tr>
                        <th>Feature</th><th>JBTS26 KIAA0556</th><th>JBTS25 CEP104</th><th>JBTS17 CPLANE1</th><th>JBTS22 CEP83</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="fw-bold">Cilia</td>
                        <td style={{ color: ACCENT7 }}>SHORT (25–55% WT)</td>
                        <td style={{ color: '#e65100' }}>SHORT (30–50% WT)</td>
                        <td style={{ color: ACCENT7 }}>SHORT (Retrograde)</td>
                        <td style={{ color: ACCENT6 }}>ABSENT</td>
                      </tr>
                      <tr>
                        <td className="fw-bold">IFT Accumulation</td>
                        <td style={{ color: ACCENT6 }}>IFT-B at BASE</td>
                        <td style={{ color: '#e65100' }}>IFT-B at TIP</td>
                        <td>IFT-A at tip</td>
                        <td>No cilia</td>
                      </tr>
                      <tr>
                        <td className="fw-bold">Primary Failure</td>
                        <td>IFT-B base assembly</td>
                        <td>Ciliary tip scaffold</td>
                        <td>Retrograde IFT-A loading</td>
                        <td>DA foundation absent</td>
                      </tr>
                      <tr>
                        <td className="fw-bold">CPLANE coupling</td>
                        <td>Partial loss</td>
                        <td>Intact</td>
                        <td>Primary failure</td>
                        <td>No cilia</td>
                      </tr>
                      <tr>
                        <td className="fw-bold">MKS Tier</td>
                        <td>No</td>
                        <td>No</td>
                        <td>No</td>
                        <td>No</td>
                      </tr>
                      <tr>
                        <td className="fw-bold">Renal %</td>
                        <td>~15%</td>
                        <td>~18%</td>
                        <td>~18%</td>
                        <td>~68% (highest)</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Section>
            </div>
          </div>

          <Section title="IFT-B Base vs Tip Accumulation — EM Diagnostic Key" color={ACCENT6}>
            <div className="row g-2">
              <div className="col-md-6">
                <div className="p-3 rounded" style={{ background: ACCENT + '18', border: `1px solid ${ACCENT}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT }}>JBTS26 KIAA0556 — IFT-B BASE Accumulation</div>
                  <ul className="small mb-0">
                    <li>Mechanism: IFT-B1 assembly platform failure at DA outer scaffold (base)</li>
                    <li>IFT-B subunits: accumulate at <strong>ciliary BASE</strong> (periciliary zone)</li>
                    <li>EM sign: enlarged periciliary IFT-B electron-dense particles at transition zone entry</li>
                    <li>Cilia: present, shortened, beat frequency reduced</li>
                    <li>CPLANE: partially uncoupled (INTU/FUZ reduced interaction)</li>
                    <li>Inheritance: Autosomal recessive</li>
                  </ul>
                </div>
              </div>
              <div className="col-md-6">
                <div className="p-3 rounded" style={{ background: ACCENT6 + '18', border: `1px solid ${ACCENT6}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT6 }}>JBTS25 CEP104 — IFT-B TIP Accumulation</div>
                  <ul className="small mb-0">
                    <li>Mechanism: CLUAP1/IFT-B1 retrograde coupling failure at ciliary TIP</li>
                    <li>IFT-B subunits: accumulate at <strong>ciliary TIP</strong> (distal axonemal plus-end)</li>
                    <li>EM sign: enlarged ciliary tips with IFT-B electron-dense particles</li>
                    <li>Cilia: present, shortened (30–50% WT), reduced beat frequency</li>
                    <li>Primary failure: TOG tubulin delivery + TTBK2 co-scaffold at tip</li>
                    <li>Inheritance: Autosomal recessive</li>
                  </ul>
                </div>
              </div>
            </div>
          </Section>

          <Section title="Surveillance Protocol (KIAA0556 / JBTS26)" color={ACCENT4}>
            <div className="row g-2 small">
              {[
                { organ: '🫘 Renal', detail: 'Annual renal US + creatinine/eGFR from diagnosis; ESRD median ~24–30yr when renal affected; renal transplant curative (cell-autonomous defect — transplanted kidney has WT KIAA0556)' },
                { organ: '👁️ Retinal', detail: 'ERG + fundus photography from age 3; annual review; rod-cone dystrophy progressive; shortened connecting cilia; no established treatment (gene therapy pre-clinical)' },
                { organ: '🫀 Hepatic', detail: 'LFTs + hepatic US at diagnosis; repeat 2-yearly if abnormal; CHF rate ~6% (lower than TMEM67/JBTS6 COACH 30%)' },
                { organ: '🧠 Neurological', detail: 'Brain MRI at diagnosis (MTS confirmation; may be subtle); annual physiotherapy; cochlear screen; cognitive and speech assessment' },
                { organ: '🔬 Cilia EM', detail: 'Nasal brushing + EM at diagnosis: cilia present (distinguish from JBTS22 absent), shortened; IFT-B BASE accumulation (distinguish from JBTS25 tip); KIAA0556/IFT88 co-immunoprecipitation for VUS' },
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
                    <tr><td className="fw-bold text-muted">Gene</td><td>KIAA0556 (C14orf179)</td></tr>
                    <tr><td className="fw-bold text-muted">Full name</td><td>{definitions.gene_full_name}</td></tr>
                    <tr><td className="fw-bold text-muted">OMIM Gene</td><td>*{definitions.omim_gene}</td></tr>
                    <tr><td className="fw-bold text-muted">OMIM Disease</td><td>#{definitions.omim_jbts26}</td></tr>
                    <tr><td className="fw-bold text-muted">Chromosome</td><td>{definitions.chromosome}</td></tr>
                    <tr><td className="fw-bold text-muted">Protein</td><td>{definitions.protein_size}</td></tr>
                    <tr><td className="fw-bold text-muted">Inheritance</td><td>{definitions.inheritance}</td></tr>
                    <tr><td className="fw-bold text-muted">MKS Tier</td><td>{definitions.mks_tier ? 'Yes' : 'No — IFT-B base adapter; TZ B9-complex gate INTACT'}</td></tr>
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
                  <div key={k} className="mb-1"><span className="fw-bold text-capitalize">{k.replace('_em', ' (EM)')}:</span> {v}</div>
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
        <Link href="/jbts25">← JBTS25 CEP104</Link>
        <Link href="/jbts24">← JBTS24 ZNF423</Link>
        <Link href="/">Home</Link>
      </div>
    </div>
  );
}
