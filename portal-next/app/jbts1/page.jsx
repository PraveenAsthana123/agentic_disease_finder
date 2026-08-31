'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'PIP2 Axis Pearls', 'Definitions'];

// JBTS1 colour scheme — INPP5E / Ciliary PIP2 Phosphatase / MORM-Allelic / No MKS Tier
// Deep navy / royal blue tones — distinct and foundational (JBTS1 = founding type)
const ACCENT   = '#0d2b6e';   // deep navy — phosphoinositide phosphatase; foundational JBTS1
const ACCENT2  = '#1565c0';   // royal blue — Arl13b axis / ciliary PIP2 control
const ACCENT3  = '#4a148c';   // deep purple — neurological / cerebellar
const ACCENT4  = '#0d47a1';   // dark blue — renal
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#b71c1c';   // red — retinal rod-cone
const ACCENT7  = '#e65100';   // orange — MORM allelic alert
const ACCENT8  = '#f57f17';   // amber — polydactyly / CAAX alert
const ACCENT9  = '#006064';   // dark cyan — CAAX/farnesylation
const ACCENT10 = '#880e4f';   // dark pink — founder allele

const SEED = 455;
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

export default function JBTS1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts1/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts1/definitions`).then(r => r.json()),
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
        <h4 className="mb-1 fw-bold">&#x1f9ec; INPP5E — Joubert Syndrome Type 1 (JBTS1)</h4>
        <div className="small opacity-90">
          INPP5E · 9q34.3 · ~644 aa · Ciliary PI(4,5)P2 Phosphatase · Arl13b Mutual-Dependency PIP Axis · MORM-Allelic · Cilia FORM Normally · No MKS Tier · AR · OMIM Gene *613037 · Disease #213300
        </div>
        <div className="small opacity-80 mt-1">
          Cohort: {N_COHORT} JBTS1 patients (seed {SEED}) · Missense-dominant (truncating → MORM) · Retinal ~30% · Renal ~12% (INPP5B compensation) · European founder Arg435Gln · Arl13b axis DDx JBTS8
        </div>
      </div>

      {/* MORM allelic alert */}
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; MORM-JBTS1 ALLELIC SPECTRUM:</strong> INPP5E biallelic <strong>truncating null</strong> → MORM (#610156: obesity, micropenis, renal microcysts, ID — <strong>NO MTS</strong>).
        INPP5E biallelic <strong>damaging missense</strong> → JBTS1 (MTS, cerebellar — no obesity/micropenis). Brain MRI is the diagnostic gate. Same gene, two different syndromes.
      </Alert>

      {/* Ciliary PIP2 mechanism alert */}
      <Alert color={ACCENT2}>
        <strong>&#x1f9ea; CILIA FORM NORMALLY IN JBTS1:</strong> INPP5E is a lipid phosphatase — NOT a TZ scaffold or IFT component.
        Cilia <strong>form normally</strong> with normal beat frequency (nasal brushing: NORMAL). MTS arises from <strong>Hedgehog/SMO signalling failure</strong> due to PI(4,5)P2 accumulation, not structural cilia defect.
        Distinguish from CEP83/JBTS22 (cilia absent) and CSPP1/JBTS21 (cilia shortened).
      </Alert>

      {/* ARL13B axis / no MKS alert */}
      <Alert color={ACCENT}>
        <strong>&#x2705; ARL13B AXIS + NO MKS TIER:</strong> JBTS1 (INPP5E, 9q34.3) and JBTS8 (Arl13b, 3q11.1) share the same PIP2-control module — phenotypically indistinguishable.
        Panel must include <strong>BOTH</strong> genes. No MKS risk — PIP2 regulation does not collapse TZ B9-complex structural gate.
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
            <KPI label="Retinal (%)"       value={`${overview.kpis?.retinal_pct}%`}   color={ACCENT6} />
            <KPI label="Renal (%)"         value={`${overview.kpis?.renal_pct}%`}     color={ACCENT4} />
            <KPI label="MORM Allelic"      value="Null →"                             color={ACCENT7} />
          </div>

          <div className="row g-3">
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ color: ACCENT }}>Gene &amp; Disease Summary</div>
                <div className="card-body small">
                  <table className="table table-sm table-borderless mb-0">
                    <tbody>
                      <tr><td className="fw-bold text-muted">Gene</td><td>INPP5E</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM Gene</td><td>*613037</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM JBTS1</td><td>#213300</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM MORM</td><td>#610156 (allelic via biallelic null)</td></tr>
                      <tr><td className="fw-bold text-muted">Chromosome</td><td>9q34.3</td></tr>
                      <tr><td className="fw-bold text-muted">Protein</td><td>~644 aa; N-term proline-rich/PDE6D (1–160); INPP5 phosphatase domain (161–530); CC+CAAX farnesylation (531–644)</td></tr>
                      <tr><td className="fw-bold text-muted">Inheritance</td><td>Autosomal recessive — biallelic damaging missense LOF</td></tr>
                      <tr><td className="fw-bold text-muted">MKS tier</td><td><span className="badge" style={{ background: '#1565c0' }}>No MKS tier — PIP2 regulation; TZ intact</span></td></tr>
                      <tr><td className="fw-bold text-muted">Mechanism</td><td>Ciliary PI(4,5)P2 phosphatase (lipid); PIP2 accumulation → Arl13b loss → Hedgehog failure</td></tr>
                      <tr><td className="fw-bold text-muted">Cilia structure</td><td><span className="text-success fw-bold">NORMAL</span> (nasal brushing: normal beat frequency)</td></tr>
                      <tr><td className="fw-bold text-muted">Eur. founder</td><td>Arg435Gln (c.1304G>A) — INPP5 core; commonest JBTS1 allele</td></tr>
                      <tr><td className="fw-bold text-muted">MORM rule</td><td>Biallelic truncating → MORM (no MTS); missense → JBTS1 (MTS)</td></tr>
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
                    { label: 'Cerebellar Ataxia (~90%)',          v: overview.kpis?.ataxia_pct,        color: ACCENT3 },
                    { label: 'Neonatal Hypotonia (~82%)',         v: overview.kpis?.hypotonia_pct,     color: ACCENT3 },
                    { label: 'Oculomotor Apraxia (~55%)',         v: overview.kpis?.oma_pct,           color: ACCENT3 },
                    { label: 'Breathing Dysreg. (~52%)',          v: overview.kpis?.breathing_pct,     color: ACCENT3 },
                    { label: 'Retinal Rod-Cone (~30%)',           v: overview.kpis?.retinal_pct,       color: ACCENT6 },
                    { label: 'Renal NPHP-like (~12%)',            v: overview.kpis?.renal_pct,         color: ACCENT4 },
                    { label: 'Polydactyly (~8%; cilia form OK)',  v: overview.kpis?.poly_pct,          color: ACCENT8 },
                    { label: 'ID (~72%)',                         v: overview.kpis?.id_pct,            color: ACCENT3 },
                    { label: 'Hepatic CHF (~5%)',                 v: overview.kpis?.hepatic_pct,       color: ACCENT7 },
                    { label: 'ESRD at study (~5%)',               v: overview.kpis?.esrd_pct,          color: ACCENT7 },
                    { label: 'Obesity/MORM spectrum (~5%)',       v: overview.kpis?.obesity_morm_pct,  color: ACCENT7 },
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
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Patient Cohort (N={N_COHORT}, seed {SEED}) — JBTS1 INPP5E / MTS-confirmed / missense-dominant</div>
            <div className="card-body p-0">
              <div style={{ overflowX: 'auto', maxHeight: 340 }}>
                <table className="table table-sm table-striped table-hover mb-0" style={{ fontSize: '0.77em' }}>
                  <thead className="sticky-top table-light">
                    <tr>
                      <th>ID</th><th>Age</th><th>Sex</th><th>Ethnicity</th><th>Allele Class</th><th>Variant</th>
                      <th>MTS</th><th>Ataxia</th><th>Retinal</th><th>Renal</th><th>ESRD</th><th>Poly</th><th>Obesity</th>
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
                            background: p.allele_class?.includes('CAAX') ? ACCENT9
                              : p.allele_class?.includes('Null') ? ACCENT7
                              : p.allele_class?.includes('Splice') ? ACCENT2
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
                        <td style={{ color: p.obesity ? ACCENT7 : undefined }}>{p.obesity ? '✓' : '–'}</td>
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
                    <strong>European founder:</strong> Arg435Gln (c.1304G>A) elevated in non-consanguineous European cohort. Screening mandatory in all JBTS probands regardless of ethnicity.
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
                      color={a.allele_class?.includes('CAAX') ? ACCENT9 : a.allele_class?.includes('Null') ? ACCENT7 : a.allele_class?.includes('Splice') ? ACCENT2 : ACCENT3}
                    />
                  ))}
                  <div className="mt-2 p-2 rounded small" style={{ background: ACCENT7 + '15', borderLeft: `3px solid ${ACCENT7}` }}>
                    <strong>Truncating null alleles → MORM, not JBTS1.</strong> All JBTS1 patients carry ≥1 damaging missense allele. Biallelic null = MORM (#610156) — different syndrome, no MTS.
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
                      color={k === 'retinal' ? ACCENT6 : k === 'renal' ? ACCENT4 : k === 'poly' ? ACCENT8 : k === 'obesity_morm' ? ACCENT7 : k === 'hepatic' ? ACCENT7 : k === 'esrd' ? ACCENT7 : ACCENT3}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Variant table */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-bold" style={{ color: ACCENT5 }}>Notable INPP5E Variants (JBTS1-Specific)</div>
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
                            background: v.severity?.includes('MORM') ? ACCENT7
                              : v.severity?.includes('CAAX') ? ACCENT9
                              : v.severity?.includes('Severe') ? ACCENT7
                              : v.severity?.includes('Mild') ? ACCENT2
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

      {/* ── TAB 2: PIP2 Axis Pearls ── */}
      {tab === 2 && definitions && (
        <div>
          <Alert color={ACCENT7}>
            <strong>&#x26a0;&#xfe0f; MORM vs JBTS1 Allelic Rule:</strong> {definitions.morm_allelic_rule}
          </Alert>

          {/* Domain matrix */}
          <Section title="INPP5E Protein Domain Matrix (N-terminal / Phosphatase / CC+CAAX)" color={ACCENT5}>
            <div className="row g-2">
              {(definitions.domain_matrix || []).map(d => (
                <div key={d.domain} className="col-md-4">
                  <div className="card h-100 shadow-sm">
                    <div className="card-header small fw-bold" style={{
                      color: d.domain.includes('CAAX') ? ACCENT9 : d.domain.includes('phosphatase') ? ACCENT : ACCENT2,
                      background: (d.domain.includes('CAAX') ? ACCENT9 : d.domain.includes('phosphatase') ? ACCENT : ACCENT2) + '10'
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
                      <tr><td className="fw-bold text-muted">OMIM JBTS1</td><td>#{definitions.omim_jbts1}</td></tr>
                      <tr><td className="fw-bold text-muted">OMIM MORM</td><td>#{definitions.omim_morm} (allelic via biallelic truncating null)</td></tr>
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
            <Link href="/jbts8" className="btn btn-sm btn-outline-secondary">&#x2194; JBTS8 Arl13b (PIP axis DDx)</Link>
            <Link href="/jbts3" className="btn btn-sm btn-outline-secondary">&#x2192; JBTS3 AHI1</Link>
          </div>
        </div>
      )}
    </div>
  );
}
