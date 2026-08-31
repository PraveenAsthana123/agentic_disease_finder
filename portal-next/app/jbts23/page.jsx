'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'CC3 Domain Pearls', 'Definitions'];

// JBTS23 colour scheme — KIAA0586/TALPID3 / CPLANE scaffold / SRTD16 allelic / CC3 hypomorphic
// Forest green / teal tones — distinct from JBTS22 (amber), JBTS21 (indigo), JBTS20 (teal-blue)
const ACCENT   = '#1b5e20';   // deep forest green — CPLANE scaffold
const ACCENT2  = '#2e7d32';   // medium green — SRTD16 allelic alert
const ACCENT3  = '#4a148c';   // deep purple — neurological
const ACCENT4  = '#0d47a1';   // dark blue — renal
const ACCENT5  = '#37474f';   // slate — domain matrix
const ACCENT6  = '#b71c1c';   // red — retinal
const ACCENT7  = '#e65100';   // orange — hepatic / SRTD16
const ACCENT8  = '#f57f17';   // amber — polydactyly highlight
const ACCENT9  = '#006064';   // dark cyan — CC3 splice alleles
const ACCENT10 = '#880e4f';   // dark pink — South Asian founder

const SEED = 453;
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

export default function JBTS23Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts23/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts23/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts23/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="container py-5"><div className="alert alert-danger">API error: {error}</div></div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT2} 100%)`, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">&#x1f9ec; KIAA0586 / TALPID3 — Joubert Syndrome Type 23 (JBTS23)</h4>
        <div className="small opacity-90">
          KIAA0586 (TALPID3) · 14q23.1 · ~1,624 aa · CPLANE Centriolar Scaffold · CC3 C-Terminal IFT Platform · SRTD16-Allelic (ONLY JBTS-SRTD Allelic Pair) · No MKS Tier · AR · OMIM Gene *610178 · Disease #616490
        </div>
        <div className="small opacity-80 mt-1">
          Cohort: {N_COHORT} JBTS23 patients (seed {SEED}) · CC3 hypomorphic alleles only · Cilia SHORTENED (not absent) · Renal ~18% · Polydactyly ~22% · South Asian founder Arg1116His
        </div>
      </div>

      {/* SRTD16 allelic alert */}
      <Alert color={ACCENT7}>
        <strong>&#x1f9b4; SRTD16-JBTS23 ALLELIC PAIR (ONLY JBTS-SRTD ALLELIC GENE):</strong> KIAA0586 is the ONLY ciliopathy gene allelic with BOTH a skeletal dysplasia (SRTD16) AND Joubert syndrome (JBTS23).
        CC1/CC2 alleles → <strong>SRTD16</strong> (skeletal); CC3 C-terminal hypomorphic → <strong>JBTS23</strong> (Joubert, no thoracic disease). Domain classification is clinically mandatory.
      </Alert>

      {/* Cilia shortened alert */}
      <Alert color={ACCENT2}>
        <strong>&#x1f4cf; CILIA SHORTENED NOT ABSENT:</strong> CC3 hypomorphic alleles → cilia <strong>SHORTENED</strong> (25–70% normal length, depending on allele).
        Unlike CEP83/JBTS22 (cilia absent — DA foundation block). Lower renal (~18%) and retinal (~22%) penetrance than CEP83/JBTS22 explains shortened-vs-absent cilia difference.
        Nasal brushing: shortened cilia (not absent).
      </Alert>

      {/* No MKS / CC3 allele rule alert */}
      <Alert color={ACCENT}>
        <strong>&#x2705; NO MKS TIER + CC3 ALLELE RULE:</strong> JBTS23 CC3 hypomorphic → <strong>live birth</strong>, not Meckel-Gruber Syndrome.
        CC1/CC2 null → SRTD16 perinatal lethal (different disease). <strong>Always classify allele domain</strong> (CC1/CC2 vs CC3) before MDT allocation — neurology leads JBTS23; skeletal dysplasia leads SRTD16.
      </Alert>

      {/* Tab Nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && overview && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Cohort (N)"        value={overview.kpis?.total_patients}      color={ACCENT} />
            <KPI label="MTS (%)"           value={`${overview.kpis?.mts_pct}%`}       color={ACCENT3} />
            <KPI label="Ataxia (%)"        value={`${overview.kpis?.ataxia_pct}%`}    color={ACCENT3} />
            <KPI label="Renal (%)"         value={`${overview.kpis?.renal_pct}%`}     color={ACCENT4} />
            <KPI label="Poly (%)"          value={`${overview.kpis?.poly_pct}%`}      color={ACCENT8} />
            <KPI label="SRTD16 Allelic"    value="CC1/CC2 →"                          color={ACCENT7} />
          </div>

          <div className="row g-3">
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ color: ACCENT }}>Gene &amp; Disease Summary</div>
                <div className="card-body small">
                  <table className="table table-sm table-borderless mb-0">
                    <tbody>
                      <tr><td className="fw-bold text-muted">Gene</td><td>KIAA0586 (TALPID3)</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM Gene</td><td>*610178</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM JBTS23</td><td>#616490</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM SRTD16</td><td>#617098 (allelic via CC1/CC2)</td></tr>
                      <tr><td className="fw-bold text-muted">Chromosome</td><td>14q23.1</td></tr>
                      <tr><td className="fw-bold text-muted">Protein</td><td>~1,624 aa; CC1/anchoring (1–400); CC2+CPLANE (401–1,100); CC3/IFT-platform/JBTS23-zone (1,101–1,624)</td></tr>
                      <tr><td className="fw-bold text-muted">Inheritance</td><td>Autosomal recessive — biallelic CC3 hypomorphic LOF</td></tr>
                      <tr><td className="fw-bold text-muted">MKS tier</td><td><span className="badge" style={{ background: '#2e7d32' }}>No MKS tier — CC3 hypomorphic → live birth</span></td></tr>
                      <tr><td className="fw-bold text-muted">Complex</td><td>CPLANE (KIAA0586 + INTU + FUZ + WDPCP)</td></tr>
                      <tr><td className="fw-bold text-muted">S. Asian founder</td><td>Arg1116His (c.3347G>A) — CC3 entry domain</td></tr>
                      <tr><td className="fw-bold text-muted">Allelic pair</td><td>SRTD16 (#617098) via CC1/CC2 LOF — ONLY JBTS-SRTD allelic gene</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ color: ACCENT3 }}>Organ Penetrance Summary</div>
                <div className="card-body">
                  {[
                    { label: 'Cerebellar Ataxia (~88%)',         v: overview.kpis?.ataxia_pct,        color: ACCENT3 },
                    { label: 'Neonatal Hypotonia (~80%)',        v: overview.kpis?.hypotonia_pct,     color: ACCENT3 },
                    { label: 'Oculomotor Apraxia (~50%)',        v: overview.kpis?.oma_pct,           color: ACCENT3 },
                    { label: 'Breathing Dysreg. (~52%)',         v: overview.kpis?.breathing_pct,     color: ACCENT3 },
                    { label: 'Renal NPHP-like (~18%)',           v: overview.kpis?.renal_pct,         color: ACCENT4 },
                    { label: 'Retinal Rod-Cone (~22%)',          v: overview.kpis?.retinal_pct,       color: ACCENT6 },
                    { label: 'ESRD at study (~8%)',              v: overview.kpis?.esrd_pct,          color: ACCENT7 },
                    { label: 'Hepatic CHF (~8%)',                v: overview.kpis?.hepatic_pct,       color: ACCENT7 },
                    { label: 'Polydactyly (~22%; CPLANE-IFT)',   v: overview.kpis?.poly_pct,          color: ACCENT8 },
                    { label: 'ID (~70%)',                        v: overview.kpis?.id_pct,            color: ACCENT3 },
                    { label: 'Minor Skeletal (~8%; NOT SRTD16)', v: overview.kpis?.skeletal_minor_pct,color: ACCENT7 },
                  ].map(({ label, v, color }) => (
                    <Bar key={label} label={label} value={`${v ?? '—'}%`} max={100} color={color} />
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ color: ACCENT5 }}>Key Clinical Facts</div>
                <div className="card-body">
                  <ul className="list-unstyled small mb-0">
                    {(overview.key_facts || []).map((f, i) => (
                      <li key={i} className="mb-1">&#x2022; {f}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Patient table */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Patient Cohort (N={N_COHORT}, seed {SEED}) — JBTS23 CC3 hypomorphic / MTS-confirmed</div>
            <div className="card-body p-0">
              <div style={{ overflowX: 'auto', maxHeight: 340 }}>
                <table className="table table-sm table-striped table-hover mb-0" style={{ fontSize: '0.77em' }}>
                  <thead className="sticky-top table-light">
                    <tr>
                      <th>ID</th><th>Age</th><th>Sex</th><th>Ethnicity</th><th>Allele Class</th><th>Variant</th>
                      <th>MTS</th><th>Ataxia</th><th>Retinal</th><th>Renal</th><th>ESRD</th><th>Poly</th><th>Skeletal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(overview.patients || []).map(p => (
                      <tr key={p.id}>
                        <td>{p.id}</td>
                        <td>{p.age}</td>
                        <td>{p.sex}</td>
                        <td>{p.ethnicity}</td>
                        <td>
                          <span className="badge" style={{
                            background: p.allele_class?.includes('Splice') ? ACCENT9
                              : p.allele_class?.includes('Truncating') ? ACCENT7
                              : ACCENT3,
                            fontSize: '0.7em'
                          }}>{p.allele_class}</span>
                        </td>
                        <td style={{ fontFamily: 'monospace', fontSize: '0.85em' }}>{p.variant}</td>
                        <td>{p.mts ? '✓' : '–'}</td>
                        <td>{p.ataxia ? '✓' : '–'}</td>
                        <td style={{ color: p.retinal ? ACCENT6 : undefined }}>{p.retinal ? '✓' : '–'}</td>
                        <td style={{ color: p.renal ? ACCENT4 : undefined }}>{p.renal ? '✓' : '–'}</td>
                        <td style={{ color: p.esrd ? ACCENT7 : undefined }}>{p.esrd ? '✓' : '–'}</td>
                        <td style={{ color: p.poly ? ACCENT8 : undefined }}>{p.poly ? '✓' : '–'}</td>
                        <td style={{ color: p.skeletal ? ACCENT7 : undefined }}>{p.skeletal ? '✓' : '–'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 1: Diagnostic Breakdown ── */}
      {tab === 1 && breakdown && (
        <div>
          <div className="row g-3">
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ color: ACCENT }}>Ethnic Distribution</div>
                <div className="card-body">
                  {(breakdown.ethnicity_distribution || []).map(e => (
                    <Bar key={e.ethnicity} label={`${e.ethnicity} (${e.pct}%)`} value={e.count} max={N_COHORT} color={ACCENT} />
                  ))}
                  <div className="mt-2 p-2 rounded small" style={{ background: ACCENT10 + '12', borderLeft: `3px solid ${ACCENT10}` }}>
                    <strong>South Asian founder:</strong> Arg1116His (c.3347G>A) elevated in South Asian consanguineous cohort. Screening mandatory in all South Asian JBTS probands.
                  </div>
                </div>
              </div>
            </div>

            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ color: ACCENT2 }}>Allele Class Distribution (CC3 only)</div>
                <div className="card-body">
                  {(breakdown.allele_class_distribution || []).map(a => (
                    <Bar
                      key={a.allele_class}
                      label={`${a.allele_class} (${a.pct}%)`}
                      value={a.count}
                      max={N_COHORT}
                      color={a.allele_class?.includes('Splice') ? ACCENT9 : a.allele_class?.includes('Truncating') ? ACCENT7 : ACCENT3}
                    />
                  ))}
                  <div className="mt-2 p-2 rounded small" style={{ background: ACCENT7 + '15', borderLeft: `3px solid ${ACCENT7}` }}>
                    <strong>All alleles CC3:</strong> CC1/CC2 alleles → SRTD16 (different disease). Domain classification mandatory before reporting JBTS23.
                  </div>
                </div>
              </div>
            </div>

            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ color: ACCENT3 }}>Phenotype Counts</div>
                <div className="card-body">
                  {Object.entries(breakdown.phenotype_summary || {}).map(([k, v]) => (
                    <Bar
                      key={k}
                      label={`${k.replace(/_/g,' ')} (${v.pct}%)`}
                      value={v.n}
                      max={N_COHORT}
                      color={k === 'retinal' ? ACCENT6 : k === 'renal' ? ACCENT4 : k === 'poly' ? ACCENT8 : k === 'skeletal' ? ACCENT7 : k === 'hepatic' ? ACCENT7 : k === 'esrd' ? ACCENT7 : ACCENT3}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Variant table */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-bold" style={{ color: ACCENT5 }}>Notable KIAA0586 CC3 Variants (JBTS23-Specific)</div>
            <div className="card-body p-0">
              <div style={{ overflowX: 'auto' }}>
                <table className="table table-sm table-striped mb-0" style={{ fontSize: '0.8em' }}>
                  <thead className="table-light">
                    <tr><th>Variant</th><th>cDNA</th><th>Domain</th><th>Population</th><th>Severity</th><th>Mechanism</th></tr>
                  </thead>
                  <tbody>
                    {(breakdown.notable_variants || []).map(v => (
                      <tr key={v.name}>
                        <td className="fw-bold">{v.name}</td>
                        <td style={{ fontFamily: 'monospace' }}>{v.cdna}</td>
                        <td>{v.domain}</td>
                        <td>{v.population}</td>
                        <td>
                          <span className="badge" style={{
                            background: v.severity?.includes('Null') ? ACCENT9
                              : v.severity?.includes('Mild') ? ACCENT2
                              : v.severity?.includes('Severe') ? ACCENT7
                              : ACCENT3,
                            fontSize: '0.75em'
                          }}>{v.severity}</span>
                        </td>
                        <td style={{ maxWidth: 280, whiteSpace: 'normal' }}>{v.mechanism}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: CC3 Domain Pearls ── */}
      {tab === 2 && definitions && (
        <div>
          <Alert color={ACCENT7}>
            <strong>&#x1f9b4; SRTD16-JBTS23 Allelic Spectrum Rule:</strong> {definitions.srtd16_allelic_rule}
          </Alert>

          {/* Domain matrix */}
          <Section title="KIAA0586 Protein Domain Matrix (CC1 / CC2 / CC3)" color={ACCENT5}>
            <div className="row g-2">
              {(definitions.domain_matrix || []).map(d => (
                <div key={d.domain} className="col-md-4">
                  <div className="card h-100 shadow-sm">
                    <div className="card-header small fw-bold" style={{
                      color: d.domain.includes('CC3') ? ACCENT : ACCENT7,
                      background: (d.domain.includes('CC3') ? ACCENT : ACCENT7) + '10'
                    }}>{d.domain}</div>
                    <div className="card-body small">
                      <div className="text-muted mb-1">{d.location}</div>
                      <div className="mb-2">{d.function}</div>
                      <div style={{ borderTop: `1px solid ${ACCENT5}22`, paddingTop: 6 }}>
                        <span className="fw-bold text-muted">Variants: </span>
                        <span style={{ fontFamily: 'monospace', fontSize: '0.85em' }}>{d.variant_examples}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Clinical pearls */}
          <Section title="Clinical &amp; Counselling Pearls" color={ACCENT}>
            <div className="row g-2">
              {(definitions.clinical_pearls || []).map(p => (
                <div key={p.title} className="col-12">
                  <div className="card shadow-sm">
                    <div className="card-header small fw-bold" style={{ color: ACCENT, background: ACCENT + '0d' }}>
                      &#x1f4cc; {p.title}
                    </div>
                    <div className="card-body small">{p.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Literature */}
          <Section title="Key Literature" color={ACCENT4}>
            <ul className="list-unstyled small">
              {(definitions.literature_highlights || []).map((l, i) => (
                <li key={i} className="mb-1">&#x1f4da; {l}</li>
              ))}
            </ul>
          </Section>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && definitions && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ color: ACCENT }}>Gene &amp; Disease Information</div>
                <div className="card-body small">
                  <table className="table table-sm table-borderless mb-0">
                    <tbody>
                      <tr><td className="fw-bold text-muted w-40">Gene (full)</td><td>{definitions.gene_full_name}</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM Gene</td><td>*{definitions.omim_gene}</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM JBTS23</td><td>#{definitions.omim_jbts23}</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM SRTD16</td><td>#{definitions.omim_srtd16} (allelic via CC1/CC2)</td></tr>
                      <tr><td className="fw-bold text-muted">Chromosome</td><td>{definitions.chromosome}</td></tr>
                      <tr><td className="fw-bold text-muted">Protein</td><td>{definitions.protein_size}</td></tr>
                      <tr><td className="fw-bold text-muted">Inheritance</td><td>{definitions.inheritance}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ color: ACCENT2 }}>Phenotype Frequencies (Cohort)</div>
                <div className="card-body small">
                  <table className="table table-sm mb-0">
                    <tbody>
                      {Object.entries(definitions.phenotype_frequencies || {}).map(([k, v]) => (
                        <tr key={k}>
                          <td className="text-muted">{k.replace(/_/g, ' ')}</td>
                          <td className="fw-bold">{v}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Glossary */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-bold" style={{ color: ACCENT5 }}>Glossary</div>
            <div className="card-body">
              <div className="row g-2">
                {(definitions.glossary || []).map(g => (
                  <div key={g.term} className="col-md-6">
                    <div className="p-2 rounded" style={{ background: ACCENT5 + '08', border: `1px solid ${ACCENT5}22` }}>
                      <div className="fw-bold small" style={{ color: ACCENT }}>{g.term}</div>
                      <div className="small text-muted mt-1">{g.definition}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Navigation */}
          <div className="mt-3 d-flex gap-2 flex-wrap">
            <Link href="/joubert" className="btn btn-sm btn-outline-secondary">&#x2190; Joubert Syndrome Overview</Link>
            <Link href="/jbts22" className="btn btn-sm btn-outline-secondary">&#x2190; JBTS22 CEP83</Link>
            <Link href="/srtd16" className="btn btn-sm btn-outline-secondary">&#x2194; SRTD16 KIAA0586 (allelic)</Link>
          </div>
        </div>
      )}
    </div>
  );
}
