'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'B9-Tectonic Bridge Pearls', 'Definitions'];

// JBTS20 colour scheme — TMEM231 / B9-Tectonic Bridge / TZ Membrane / No MKS Tier
// Teal-cyan tones — distinct from JBTS19 (red-amber/B9D1), JBTS18 (teal/TCTN3), JBTS16 (purple/TMEM138)
const ACCENT   = '#00695c';   // deep teal — TMEM231 B9-Tectonic bridge
const ACCENT2  = '#00838f';   // cyan teal — TZ membrane
const ACCENT3  = '#1565c0';   // dark blue — neurological
const ACCENT4  = '#37474f';   // slate — renal
const ACCENT5  = '#4a148c';   // deep purple — domain matrix
const ACCENT6  = '#b71c1c';   // red — retinal
const ACCENT7  = '#1b5e20';   // dark green — hepatic
const ACCENT8  = '#004d40';   // very dark teal — NPHP
const ACCENT9  = '#006064';   // dark cyan — B9-Tectonic interface
const ACCENT10 = '#e65100';   // orange — polydactyly

const SEED = 447;
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

function Badge({ text, color }) {
  return <span className="badge me-1" style={{ background: color, fontSize: '0.72em' }}>{text}</span>;
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

export default function JBTS20Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts20/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts20/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts20/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="container py-5"><div className="alert alert-danger">API error: {error}</div></div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT9} 100%)`, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">&#x1f9ec; TMEM231 — Joubert Syndrome Type 20 (JBTS20)</h4>
        <div className="small opacity-90">
          Transmembrane Protein 231 · 16q23.1 · ~669 aa · B9-Tectonic TZ Bridge (B9D1/TMEM138/TMEM67 contact) · No MKS Tier (Biallelic Null → Live Birth) · AR · OMIM Gene *614949 · Disease #614990
        </div>
        <div className="small opacity-80 mt-1">
          Cohort: {N_COHORT} JBTS20 patients (seed {SEED}) · biallelic null → JBTS20 live birth (no Meckel-Gruber risk) · TMEM231 bridges B9 complex inner leaflet ↔ Tectonic lipid gate
        </div>
      </div>

      {/* No MKS tier alert */}
      <Alert color={ACCENT}>
        <strong>&#x2705; NO MKS TIER:</strong> TMEM231 biallelic null → <strong>JBTS20 live birth</strong>, NOT Meckel-Gruber Syndrome.
        Unlike B9D1/JBTS19 (MKS9), B9D2/JBTS34 (MKS10), and MKS1/JBTS28 — which all carry null/null perinatal-lethal MKS risk —
        TMEM231 LOF disrupts only the B9-Tectonic bridge while the B9D1-B9D2-MKS1 inner-leaflet anchor is retained.
        <strong> No MKS counselling needed for JBTS20 families.</strong>
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
            <KPI label="Cohort (N)"        value={overview.kpis?.total_patients}  color={ACCENT} />
            <KPI label="MTS (%)"           value={`${overview.kpis?.mts_pct}%`}   color={ACCENT3} />
            <KPI label="Ataxia (%)"        value={`${overview.kpis?.ataxia_pct}%`} color={ACCENT3} />
            <KPI label="Retinal (%)"       value={`${overview.kpis?.retinal_pct}%`} color={ACCENT6} />
            <KPI label="Renal (%)"         value={`${overview.kpis?.renal_pct}%`} color={ACCENT4} />
            <KPI label="No MKS Tier"       value="Confirmed" color={ACCENT} />
          </div>

          <div className="row g-3">
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ color: ACCENT }}>Gene &amp; Disease Summary</div>
                <div className="card-body small">
                  <table className="table table-sm table-borderless mb-0">
                    <tbody>
                      <tr><td className="fw-bold text-muted">Gene</td><td>TMEM231</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM Gene</td><td>*614949</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM JBTS20</td><td>#614990</td></tr>
                      <tr><td className="fw-bold text-muted">Chromosome</td><td>16q23.1</td></tr>
                      <tr><td className="fw-bold text-muted">Protein</td><td>~669 aa; N-tail/TMEM138/B9D1 (1–75); TM1-4/TZ membrane/B9D1-B9D2 (76–380); EL1-3/TMEM67/MKS1 (381–550); C-tail/NPHP4/IFT-A (551–669)</td></tr>
                      <tr><td className="fw-bold text-muted">Inheritance</td><td>Autosomal recessive — biallelic LOF</td></tr>
                      <tr><td className="fw-bold text-muted">MKS tier</td><td><span className="badge" style={{ background: '#2e7d32' }}>No MKS tier — null/null → live birth</span></td></tr>
                      <tr><td className="fw-bold text-muted">Function</td><td>B9-Tectonic TZ bridge</td></tr>
                      <tr><td className="fw-bold text-muted">Frequency</td><td>~1% all JBTS; ~1/4–8 million worldwide</td></tr>
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
                    { label: 'Cerebellar Ataxia (85%)', v: overview.kpis?.ataxia_pct, color: ACCENT3 },
                    { label: 'Neonatal Hypotonia (80%)', v: overview.kpis?.hypotonia_pct, color: ACCENT3 },
                    { label: 'Oculomotor Apraxia (50%)', v: overview.kpis?.oma_pct, color: ACCENT3 },
                    { label: 'Breathing Dysreg. (52%)', v: overview.kpis?.breathing_pct, color: ACCENT3 },
                    { label: 'Retinal Rod-Cone (22%)', v: overview.kpis?.retinal_pct, color: ACCENT6 },
                    { label: 'Renal NPHP-like (25%)', v: overview.kpis?.renal_pct, color: ACCENT4 },
                    { label: 'Hepatic CHF (10%)', v: overview.kpis?.hepatic_pct, color: ACCENT7 },
                    { label: 'Polydactyly Post-Axial (12%)', v: overview.kpis?.poly_pct, color: ACCENT10 },
                    { label: 'ID (68%)', v: overview.kpis?.id_pct, color: ACCENT3 },
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
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Patient Cohort (N={N_COHORT}, seed {SEED})</div>
            <div className="card-body p-0">
              <div style={{ overflowX: 'auto', maxHeight: 340 }}>
                <table className="table table-sm table-striped table-hover mb-0" style={{ fontSize: '0.77em' }}>
                  <thead className="sticky-top table-light">
                    <tr>
                      <th>ID</th><th>Age</th><th>Sex</th><th>Ethnicity</th><th>Allele Class</th><th>Variant</th>
                      <th>MTS</th><th>Ataxia</th><th>Retinal</th><th>Renal</th><th>Hepatic</th><th>Poly</th><th>CC</th>
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
                            background: p.allele_class?.includes('Null') ? ACCENT9
                              : p.allele_class?.includes('Splice') ? ACCENT
                              : ACCENT3,
                            fontSize: '0.7em'
                          }}>{p.allele_class}</span>
                        </td>
                        <td style={{ fontFamily: 'monospace', fontSize: '0.85em' }}>{p.variant}</td>
                        <td>{p.mts ? '✓' : '–'}</td>
                        <td>{p.ataxia ? '✓' : '–'}</td>
                        <td>{p.retinal ? '✓' : '–'}</td>
                        <td>{p.renal ? '✓' : '–'}</td>
                        <td>{p.hepatic ? '✓' : '–'}</td>
                        <td>{p.poly ? '✓' : '–'}</td>
                        <td>{p.cc ? '✓' : '–'}</td>
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
                </div>
              </div>
            </div>

            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ color: ACCENT2 }}>Allele Class Distribution</div>
                <div className="card-body">
                  {(breakdown.allele_class_distribution || []).map(a => (
                    <Bar
                      key={a.allele_class}
                      label={`${a.allele_class} (${a.pct}%)`}
                      value={a.count}
                      max={N_COHORT}
                      color={a.allele_class?.includes('Null') ? ACCENT9 : a.allele_class?.includes('Splice') ? ACCENT : ACCENT3}
                    />
                  ))}
                  <div className="mt-2 p-2 rounded small" style={{ background: ACCENT + '12', borderLeft: `3px solid ${ACCENT}` }}>
                    <strong>No MKS Tier:</strong> All allele classes → JBTS20 live birth. No perinatal-lethal risk regardless of allele class.
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
                      color={k === 'retinal' ? ACCENT6 : k === 'renal' ? ACCENT4 : k === 'hepatic' ? ACCENT7 : ACCENT3}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Variant table */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-bold" style={{ color: ACCENT5 }}>Notable TMEM231 Variants</div>
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
                              : v.severity?.includes('Mild') ? ACCENT7
                              : v.severity?.includes('Severe') ? ACCENT
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

      {/* ── TAB 2: B9-Tectonic Bridge Pearls ── */}
      {tab === 2 && definitions && (
        <div>
          <Alert color={ACCENT}>
            <strong>&#x2705; No MKS Tier Rule (TMEM231-Specific):</strong> {definitions.no_mks_tier_rule}
          </Alert>

          {/* Domain matrix */}
          <Section title="TMEM231 Protein Domain Matrix" color={ACCENT5}>
            <div className="row g-2">
              {(definitions.domain_matrix || []).map(d => (
                <div key={d.domain} className="col-md-6">
                  <div className="card h-100 shadow-sm">
                    <div className="card-header small fw-bold" style={{ color: ACCENT5, background: ACCENT5 + '10' }}>{d.domain}</div>
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
                      <tr><td className="fw-bold text-muted">OMIM JBTS20</td><td>#{definitions.omim_jbts20}</td></tr>
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
                      <div className="fw-bold small" style={{ color: ACCENT5 }}>{g.term}</div>
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
            <Link href="/jbts19" className="btn btn-sm btn-outline-secondary">&#x2190; JBTS19 B9D1</Link>
          </div>
        </div>
      )}
    </div>
  );
}
