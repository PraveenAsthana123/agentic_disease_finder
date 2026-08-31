'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'DA Foundation Pearls', 'Definitions'];

// JBTS22 colour scheme — CEP83 / Distal Appendage Foundation / Ciliogenesis Initiation / No MKS Tier
// Deep amber/gold tones — distinct from JBTS21 (indigo), JBTS20 (teal), JBTS19 (red-amber), JBTS18 (emerald)
const ACCENT   = '#e65100';   // deep amber/orange — DA foundation scaffold
const ACCENT2  = '#bf360c';   // dark burnt orange — DA hierarchy
const ACCENT3  = '#4a148c';   // deep purple — neurological
const ACCENT4  = '#1a237e';   // dark indigo — renal (high penetrance)
const ACCENT5  = '#37474f';   // slate — domain matrix
const ACCENT6  = '#b71c1c';   // red — retinal
const ACCENT7  = '#1b5e20';   // dark green — hepatic
const ACCENT8  = '#f9a825';   // amber — ESRD highlight
const ACCENT9  = '#006064';   // dark cyan — null allele
const ACCENT10 = '#880e4f';   // dark pink — MENA founder

const SEED = 451;
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

export default function JBTS22Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts22/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts22/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts22/definitions`).then(r => r.json()),
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
        <h4 className="mb-1 fw-bold">&#x1f9ec; CEP83 — Joubert Syndrome Type 22 (JBTS22)</h4>
        <div className="small opacity-90">
          Centrosomal Protein 83 kDa (CCDC41) · 12q22 · ~826 aa · Distal Appendage Foundation Scaffold · DA Hierarchy Nucleator (CEP83→CEP89→SCLT1→FBF1→LRRC45→CEP164) · No MKS Tier · AR · OMIM Gene *617233 · Disease #617265
        </div>
        <div className="small opacity-80 mt-1">
          Cohort: {N_COHORT} JBTS22 patients (seed {SEED}) · Cilia ABSENT (complete DA block) · Renal ~68% (highest non-NPHP1 JBTS; ESRD median ~14–18 yr) · MENA founder Arg252Cys
        </div>
      </div>

      {/* DA Foundation alert */}
      <Alert color={ACCENT}>
        <strong>&#x1f3db;&#xfe0f; DISTAL APPENDAGE FOUNDATION:</strong> CEP83 is the MOST PROXIMAL DA protein — its loss removes <strong>ALL downstream DA proteins</strong> (CEP89, SCLT1, FBF1, LRRC45, CEP164/NPHP15) simultaneously.
        Cilia are <strong>ABSENT</strong> (complete ciliogenesis initiation block — not shortened as in CSPP1/JBTS21).
      </Alert>

      {/* High renal alert */}
      <Alert color={ACCENT4}>
        <strong>&#x1f6a8; VERY HIGH RENAL PENETRANCE (~68%):</strong> Highest non-NPHP1 JBTS renal risk.
        ESRD median <strong>~14–18 yr</strong> (juvenile onset — earlier than TMEM231/25yr, CSPP1/28yr).
        Annual renal surveillance MANDATORY from diagnosis. <strong>Polyuria/polydipsia</strong> precedes proteinuria by years.
      </Alert>

      {/* No MKS alert */}
      <Alert color={ACCENT2}>
        <strong>&#x2705; NO MKS TIER:</strong> CEP83 biallelic null → <strong>JBTS22 live birth</strong>, NOT Meckel-Gruber Syndrome.
        Unlike B9D1/JBTS19 (MKS9) or B9D2/JBTS34 (MKS10), CEP83 LOF does not collapse the TZ gate B9-complex anchor.
        <strong> Brain MRI mandatory</strong> to distinguish JBTS22 from pure renal NPHP18 (same gene, ~30% pure renal).
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
            <KPI label="Cohort (N)"        value={overview.kpis?.total_patients}    color={ACCENT} />
            <KPI label="MTS (%)"           value={`${overview.kpis?.mts_pct}%`}     color={ACCENT3} />
            <KPI label="Ataxia (%)"        value={`${overview.kpis?.ataxia_pct}%`}  color={ACCENT3} />
            <KPI label="Renal (%)"         value={`${overview.kpis?.renal_pct}%`}   color={ACCENT4} />
            <KPI label="Retinal (%)"       value={`${overview.kpis?.retinal_pct}%`} color={ACCENT6} />
            <KPI label="No MKS Tier"       value="Confirmed"                         color={ACCENT2} />
          </div>

          <div className="row g-3">
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ color: ACCENT }}>Gene &amp; Disease Summary</div>
                <div className="card-body small">
                  <table className="table table-sm table-borderless mb-0">
                    <tbody>
                      <tr><td className="fw-bold text-muted">Gene</td><td>CEP83 (CCDC41)</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM Gene</td><td>*617233</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM JBTS22</td><td>#617265</td></tr>
                      <tr><td className="fw-bold text-muted">Chromosome</td><td>12q22</td></tr>
                      <tr><td className="fw-bold text-muted">Protein</td><td>~826 aa; CC1/anchoring (1–120); CC2/CEP89 (140–380); Scaffold/SCLT1 (380–600); C-term/IFT-B (600–826)</td></tr>
                      <tr><td className="fw-bold text-muted">Inheritance</td><td>Autosomal recessive — biallelic LOF</td></tr>
                      <tr><td className="fw-bold text-muted">MKS tier</td><td><span className="badge" style={{ background: '#2e7d32' }}>No MKS tier — null/null → live birth</span></td></tr>
                      <tr><td className="fw-bold text-muted">Function</td><td>Distal appendage FOUNDATION — nucleates entire DA hierarchy</td></tr>
                      <tr><td className="fw-bold text-muted">DA hierarchy</td><td>CEP83 → CEP89 → SCLT1 → FBF1 → LRRC45 → CEP164</td></tr>
                      <tr><td className="fw-bold text-muted">MENA founder</td><td>Arg252Cys (c.754C>T) — CC2 CEP89 module</td></tr>
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
                    { label: 'Cerebellar Ataxia (~70%)',         v: overview.kpis?.ataxia_pct,    color: ACCENT3 },
                    { label: 'Neonatal Hypotonia (~68%)',        v: overview.kpis?.hypotonia_pct,  color: ACCENT3 },
                    { label: 'Oculomotor Apraxia (~42%)',        v: overview.kpis?.oma_pct,        color: ACCENT3 },
                    { label: 'Breathing Dysreg. (~45%)',         v: overview.kpis?.breathing_pct,  color: ACCENT3 },
                    { label: 'Renal NPHP-like (~68%)',           v: overview.kpis?.renal_pct,      color: ACCENT4 },
                    { label: 'Retinal Rod-Cone (~35%)',          v: overview.kpis?.retinal_pct,    color: ACCENT6 },
                    { label: 'ESRD at study (~28%)',             v: overview.kpis?.esrd_pct,       color: ACCENT8 },
                    { label: 'Hepatic CHF (~8%)',                v: overview.kpis?.hepatic_pct,    color: ACCENT7 },
                    { label: 'Polydactyly (<5%; very rare)',     v: overview.kpis?.poly_pct,       color: ACCENT10 },
                    { label: 'ID (~60%)',                        v: overview.kpis?.id_pct,         color: ACCENT3 },
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
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Patient Cohort (N={N_COHORT}, seed {SEED}) — JBTS22 MTS-confirmed</div>
            <div className="card-body p-0">
              <div style={{ overflowX: 'auto', maxHeight: 340 }}>
                <table className="table table-sm table-striped table-hover mb-0" style={{ fontSize: '0.77em' }}>
                  <thead className="sticky-top table-light">
                    <tr>
                      <th>ID</th><th>Age</th><th>Sex</th><th>Ethnicity</th><th>Allele Class</th><th>Variant</th>
                      <th>MTS</th><th>Ataxia</th><th>Retinal</th><th>Renal</th><th>ESRD</th><th>Hepatic</th><th>Poly</th>
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
                              : p.allele_class?.includes('Splice') ? ACCENT2
                              : ACCENT3,
                            fontSize: '0.7em'
                          }}>{p.allele_class}</span>
                        </td>
                        <td style={{ fontFamily: 'monospace', fontSize: '0.85em' }}>{p.variant}</td>
                        <td>{p.mts ? '✓' : '–'}</td>
                        <td>{p.ataxia ? '✓' : '–'}</td>
                        <td>{p.retinal ? '✓' : '–'}</td>
                        <td style={{ color: p.renal ? ACCENT4 : undefined }}>{p.renal ? '✓' : '–'}</td>
                        <td style={{ color: p.esrd ? ACCENT8 : undefined }}>{p.esrd ? '✓' : '–'}</td>
                        <td>{p.hepatic ? '✓' : '–'}</td>
                        <td>{p.poly ? '✓' : '–'}</td>
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
                    <strong>MENA founder:</strong> Arg252Cys elevated in Middle Eastern/Arab cohort. Screening mandatory in all MENA JBTS probands.
                  </div>
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
                      color={a.allele_class?.includes('Null') ? ACCENT9 : a.allele_class?.includes('Splice') ? ACCENT2 : ACCENT3}
                    />
                  ))}
                  <div className="mt-2 p-2 rounded small" style={{ background: ACCENT4 + '15', borderLeft: `3px solid ${ACCENT4}` }}>
                    <strong>Renal risk:</strong> Null/null and null/missense genotypes → highest renal penetrance. All genotypes: annual surveillance mandatory.
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
                      color={k === 'retinal' ? ACCENT6 : k === 'renal' ? ACCENT4 : k === 'hepatic' ? ACCENT7 : k === 'esrd' ? ACCENT8 : ACCENT3}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Variant table */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-bold" style={{ color: ACCENT5 }}>Notable CEP83 Variants</div>
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

      {/* ── TAB 2: DA Foundation Pearls ── */}
      {tab === 2 && definitions && (
        <div>
          <Alert color={ACCENT}>
            <strong>&#x1f3db;&#xfe0f; DA Foundation Rule (CEP83-Specific):</strong> {definitions.da_foundation_rule}
          </Alert>

          {/* Domain matrix */}
          <Section title="CEP83 Protein Domain Matrix" color={ACCENT5}>
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
                      <tr><td className="fw-bold text-muted">OMIM JBTS22</td><td>#{definitions.omim_jbts22}</td></tr>
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
            <Link href="/jbts21" className="btn btn-sm btn-outline-secondary">&#x2190; JBTS21 CSPP1</Link>
          </div>
        </div>
      )}
    </div>
  );
}
