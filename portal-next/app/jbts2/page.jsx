'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'TZ Gate Pearls', 'Definitions'];

// JBTS2 colour scheme — TMEM216 / 4-Pass TZ Membrane Scaffold / MKS2-Allelic / MKS Tier
// Deep teal / green tones — TZ membrane scaffold; MKS danger-red alert; Ashkenazi gold
const ACCENT   = '#004d40';   // deep teal — TZ membrane scaffold; TMEM216 structural gate
const ACCENT2  = '#00695c';   // medium teal — Y-link TZ gate assembly
const ACCENT3  = '#1b5e20';   // forest green — cerebellar/neurological
const ACCENT4  = '#0277bd';   // blue — renal NPHP-like
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#c62828';   // red — MKS2 lethal tier alert
const ACCENT7  = '#e65100';   // orange — MKS2 allelic warning
const ACCENT8  = '#f9a825';   // amber/gold — Ashkenazi founder allele
const ACCENT9  = '#558b2f';   // olive — hepatic CHF/DPM (MKS-module overlap)
const ACCENT10 = '#6a1b9a';   // purple — retinal rod-cone

const SEED = 457;
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

export default function JBTS2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts2/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts2/definitions`).then(r => r.json()),
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
        <h4 className="mb-1 fw-bold">&#x1f9ec; TMEM216 — Joubert Syndrome Type 2 (JBTS2)</h4>
        <div className="small opacity-90">
          TMEM216 · 11q13.2 · ~148 aa · 4-Pass TZ Membrane Scaffold · Y-Link TZ Gate · MKS2-Allelic ·
          MKS TIER · Ashkenazi Founder Arg73Leu (~1:92) · AR · OMIM Gene *613277 · Disease #608091
        </div>
        <div className="small opacity-80 mt-1">
          Cohort: {N_COHORT} JBTS2 patients (seed {SEED}) · Hypomorphic missense dominant (null → MKS2) ·
          Retinal ~35% · Renal ~28% · Hepatic ~20% (MKS-module overlap) · Ashkenazi Jewish founder Arg73Leu
        </div>
      </div>

      {/* MKS2 Tier Alert */}
      <Alert color={ACCENT6}>
        <strong style={{ color: ACCENT6 }}>&#x26A0;&#xFE0F; MKS TIER — Biallelic Null → MKS2 (Meckel Syndrome Type 2, Perinatal Lethal):</strong>{' '}
        Biallelic truncating TMEM216 → Meckel syndrome type 2 (#603194): occipital encephalocele, polycystic kidneys, polydactyly — universally lethal.
        Hypomorphic missense → JBTS2 (#608091): live birth, MTS. Allele class + brain MRI + prenatal USS = mandatory triple gate before counselling.
      </Alert>

      {/* Ashkenazi Founder Alert */}
      <Alert color={ACCENT8}>
        <strong style={{ color: ACCENT8 }}>&#x2B50; Ashkenazi Jewish Founder Arg73Leu (c.218G>T):</strong>{' '}
        Carrier frequency ~1:92–100 in Ashkenazi Jewish populations — one of the highest ciliopathy founder allele frequencies worldwide.
        Homozygous Arg73Leu → moderate JBTS2 (live birth, MTS confirmed). Mandatory on Ashkenazi Jewish reproductive carrier panels.
      </Alert>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ─────────────────────────────────────────────── */}
      {tab === 0 && overview && (
        <div>
          <div className="row g-2 mb-4">
            <KPI label="Cohort (N)" value={overview.kpis.n_patients} color={ACCENT} />
            <KPI label="MTS Confirmed" value={overview.kpis.mts_confirmed} color={ACCENT} />
            <KPI label="Cerebellar Ataxia" value={`${overview.kpis.pct_cerebellar_ataxia}%`} color={ACCENT3} />
            <KPI label="OMA" value={`${overview.kpis.pct_oma}%`} color={ACCENT2} />
            <KPI label="Retinal" value={`${overview.kpis.pct_retinal}%`} color={ACCENT10} />
            <KPI label="Renal" value={`${overview.kpis.pct_renal}%`} color={ACCENT4} />
            <KPI label="Hepatic CHF" value={`${overview.kpis.pct_hepatic}%`} color={ACCENT9} />
            <KPI label="Polydactyly" value={`${overview.kpis.pct_poly}%`} color={ACCENT5} />
            <KPI label="Age Mean" value={overview.kpis.age_mean} color={ACCENT5} />
            <KPI label="Age Range" value={overview.kpis.age_range} color={ACCENT5} />
            <KPI label="Female" value={overview.kpis.sex_f} color={ACCENT3} />
            <KPI label="Male" value={overview.kpis.sex_m} color={ACCENT2} />
          </div>

          <div className="row">
            <div className="col-md-6">
              <Section title="Gene & Disease Identity" color={ACCENT}>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>TMEM216 — Transmembrane Protein 216</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*613277</td></tr>
                    <tr><td className="fw-bold">Disease OMIM</td><td>#608091 (JBTS2)</td></tr>
                    <tr><td className="fw-bold">Allelic</td><td>#603194 (MKS2 — biallelic null)</td></tr>
                    <tr><td className="fw-bold">Chromosome</td><td>11q13.2</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>148 aa — 4-pass TZ membrane scaffold</td></tr>
                    <tr><td className="fw-bold">MKS Tier</td><td><span className="badge" style={{ background: ACCENT6 }}>YES — Biallelic null → MKS2 (lethal)</span></td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>AR — hypomorphic missense → JBTS2; null → MKS2</td></tr>
                    <tr><td className="fw-bold">Frequency</td><td>{overview.frequency}</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="TZ Gate Mechanism" color={ACCENT2}>
                <p className="small">{overview.tz_mechanism}</p>
              </Section>
              <Section title="Ashkenazi Founder Arg73Leu" color={ACCENT8}>
                <p className="small">{overview.ashkenazi_founder}</p>
              </Section>
            </div>
          </div>

          <Section title="MKS2-JBTS2 Allelic Rule" color={ACCENT6}>
            <p className="small">{overview.mks2_allelic_rule}</p>
          </Section>
        </div>
      )}

      {/* ── Tab 1: Diagnostic Breakdown ─────────────────────────────────── */}
      {tab === 1 && breakdown && (
        <div className="row">
          <div className="col-md-6">
            <Section title="Phenotype Frequencies" color={ACCENT}>
              {breakdown.phenotype_bars.map((b, i) => (
                <Bar key={i} label={b.feature} value={`${b.pct}% (n=${b.n})`}
                  max={100} color={i === 0 ? ACCENT3 : i <= 3 ? ACCENT2 : i === 4 ? ACCENT10 : i === 5 ? ACCENT4 : i === 6 ? ACCENT9 : ACCENT5} />
              ))}
            </Section>

            <Section title="Ethnicity Distribution" color={ACCENT2}>
              {breakdown.ethnicity_distribution.map((e, i) => (
                <Bar key={i} label={e.ethnicity} value={`${e.pct}% (n=${e.n})`}
                  max={100} color={i === 0 ? ACCENT8 : ACCENT2} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Allele Class Distribution" color={ACCENT3}>
              {breakdown.allele_class_distribution.map((a, i) => (
                <Bar key={i} label={a.class} value={`${a.pct}% (n=${a.n})`}
                  max={100} color={i === 0 ? ACCENT3 : ACCENT2} />
              ))}
            </Section>

            <Section title="Top Variants" color={ACCENT8}>
              {breakdown.top_variants.map((v, i) => (
                <Bar key={i} label={v.variant} value={`n=${v.n}`}
                  max={N_COHORT} color={i === 0 ? ACCENT8 : ACCENT2} />
              ))}
            </Section>

            <Section title="Renal & Hepatic Note" color={ACCENT4}>
              <p className="small">{breakdown.renal_hepatic_note}</p>
            </Section>
          </div>

          <div className="col-12 mt-2">
            <Section title="Clinical Pearls (Breakdown)" color={ACCENT6}>
              {breakdown.clinical_pearls.map((p, i) => (
                <Alert key={i} color={i === 0 ? ACCENT6 : i === 1 ? ACCENT8 : i === 2 ? ACCENT9 : i === 3 ? ACCENT2 : ACCENT4}>
                  <strong>{p.title}</strong><br /><span className="small">{p.detail}</span>
                </Alert>
              ))}
            </Section>
          </div>
        </div>
      )}

      {/* ── Tab 2: TZ Gate Pearls ───────────────────────────────────────── */}
      {tab === 2 && definitions && (
        <div>
          <Section title="Clinical Pearls — TMEM216 / JBTS2 TZ Gate" color={ACCENT6}>
            {definitions.clinical_pearls.map((p, i) => (
              <Alert key={i} color={i === 0 ? ACCENT6 : i === 1 ? ACCENT2 : i === 2 ? ACCENT9 : i === 3 ? ACCENT4 : ACCENT5}>
                <strong>{p.title}</strong><br /><span className="small">{p.detail}</span>
              </Alert>
            ))}
          </Section>

          <Section title="Domain Matrix — TMEM216 (148 aa, 4-Pass TZ Membrane Scaffold)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT, color: '#fff' }}>
                  <tr>
                    <th>Domain</th><th>Location</th><th>Function</th><th>Variant Examples</th>
                  </tr>
                </thead>
                <tbody>
                  {definitions.domain_matrix.map((d, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{d.domain}</td>
                      <td>{d.location}</td>
                      <td>{d.function}</td>
                      <td className="text-muted">{d.variant_examples}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Literature Highlights" color={ACCENT5}>
            <ul className="small">
              {definitions.literature_highlights.map((l, i) => <li key={i}>{l}</li>)}
            </ul>
          </Section>
        </div>
      )}

      {/* ── Tab 3: Definitions ──────────────────────────────────────────── */}
      {tab === 3 && definitions && (
        <div>
          <div className="row mb-3">
            <div className="col-md-6">
              <Section title="Gene Identity" color={ACCENT}>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Full Name</td><td>{definitions.gene_full_name}</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*{definitions.omim_gene}</td></tr>
                    <tr><td className="fw-bold">OMIM JBTS2</td><td>#{definitions.omim_jbts2}</td></tr>
                    <tr><td className="fw-bold">OMIM MKS2</td><td>#{definitions.omim_mks2}</td></tr>
                    <tr><td className="fw-bold">Chromosome</td><td>{definitions.chromosome}</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>{definitions.protein_size}</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{definitions.inheritance}</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Phenotype Frequencies (Educational Cohort)" color={ACCENT2}>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    {Object.entries(definitions.phenotype_frequencies).map(([k, v], i) => (
                      <tr key={i}>
                        <td className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}</td>
                        <td>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>
          </div>

          <Section title="MKS2-JBTS2 Allelic Rule" color={ACCENT6}>
            <p className="small">{definitions.mks2_allelic_rule}</p>
          </Section>

          <Section title="Glossary" color={ACCENT3}>
            {definitions.glossary.map((g, i) => (
              <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f8f9fa', border: `1px solid ${ACCENT}22` }}>
                <div className="fw-bold small" style={{ color: ACCENT }}>{g.term}</div>
                <div className="small mt-1">{g.definition}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      <div className="mt-4 text-muted small">
        <Link href="/" className="me-3">&#x2190; Back to Dashboard</Link>
        JBTS2 · TMEM216 · 11q13.2 · 4-Pass TZ Membrane Scaffold · MKS2-Allelic · MKS Tier ·
        Ashkenazi Founder Arg73Leu · {N_COHORT}-patient cohort (seed {SEED}) · OMIM #608091
      </div>
    </div>
  );
}
