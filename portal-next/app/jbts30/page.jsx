'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'IFT-A Cargo Adaptor Pearls', 'Definitions'];

// JBTS30 colour scheme — TULP3 / IFT-A Cargo Adaptor / PI(4,5)P2 Phosphoinositide / INPP5E-ARL13B Axis
// Deep teal for IFT-A adaptor; amber-gold for PI(4,5)P2 membrane; indigo for cargo (INPP5E/ARL13B)
const ACCENT   = '#00695c';   // dark teal — IFT-A adaptor / TULP3 core identity
const ACCENT2  = '#e65100';   // deep orange-amber — PI(4,5)P2 accumulation / phosphoinositide
const ACCENT3  = '#283593';   // deep indigo — INPP5E + ARL13B cargo axis
const ACCENT4  = '#1565c0';   // royal blue — ARL13B ciliary import
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#2e7d32';   // forest green — no MKS tier / all liveborn
const ACCENT7  = '#6a1b9a';   // purple — GPR161 import / Hedgehog
const ACCENT8  = '#c62828';   // crimson — renal NPHP-like
const ACCENT9  = '#795548';   // brown — retinal rod-cone
const ACCENT10 = '#00838f';   // cyan-teal — cilia full length present

const SEED = 475;
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

function Loading() {
  return <div className="text-center py-5 text-muted">Loading JBTS30 data…</div>;
}

export default function JBTS30Page() {
  const [tab, setTab]           = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]         = useState(null);
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts30/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts30/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts30/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefs(df);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-2 gap-2">
        <Link href="/" className="btn btn-sm btn-outline-secondary">← Home</Link>
        <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
          🧬 JBTS30 — TULP3 Joubert Syndrome Type 30
        </h4>
        <span className="badge ms-2" style={{ background: ACCENT6 }}>No MKS Tier</span>
        <span className="badge ms-1" style={{ background: ACCENT }}>IFT-A Cargo Adaptor</span>
        <span className="badge ms-1" style={{ background: ACCENT2 }}>PI(4,5)P₂ Axis</span>
      </div>

      <Alert color={ACCENT3}>
        <strong>JBTS30 Diagnostic Hallmark:</strong> INPP5E <strong>+</strong> ARL13B <strong>BOTH absent</strong> from
        cilia on immunofluorescence (patient fibroblasts: INPP5E &lt;30% WT, ARL13B &lt;35% WT) — cilia
        PRESENT and FULL LENGTH, GT335 NORMAL. ARL13B ciliary exclusion distinguishes JBTS30 from
        JBTS1/INPP5E (where ARL13B is present in cilia). IFT-A complex (WDR19/IFT144) structurally intact.
      </Alert>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab===i?' active':''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {loading && <Loading />}

      {/* ── TAB 0: OVERVIEW ── */}
      {!loading && tab === 0 && overview && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Cohort (N)" value={N_COHORT} color={ACCENT} />
            <KPI label="Seed" value={SEED} color={ACCENT5} />
            <KPI label="MTS Confirmed" value={`${overview.key_phenotypes?.mts_confirmed?.pct ?? 95}%`} color={ACCENT} />
            <KPI label="Cerebellar Ataxia" value={`${overview.key_phenotypes?.cerebellar_ataxia?.pct ?? 83}%`} color={ACCENT} />
            <KPI label="Neonatal Hypotonia" value={`${overview.key_phenotypes?.neonatal_hypotonia?.pct ?? 77}%`} color={ACCENT3} />
            <KPI label="Renal NPHP-like" value={`${overview.key_phenotypes?.renal_nphp_like?.pct ?? 25}%`} color={ACCENT8} />
            <KPI label="Retinal Rod-Cone" value={`${overview.key_phenotypes?.retinal_rod_cone?.pct ?? 30}%`} color={ACCENT9} />
            <KPI label="Hepatic CHF" value={`${overview.key_phenotypes?.hepatic_chf?.pct ?? 6}%`} color={ACCENT5} />
            <KPI label="Polydactyly" value={`${overview.key_phenotypes?.postaxial_polydactyly?.pct ?? 20}%`} color={ACCENT7} />
            <KPI label="INPP5E IF Low" value={`${overview.key_phenotypes?.inpp5e_if_low?.pct ?? 97}%`} color={ACCENT2} />
            <KPI label="ARL13B IF Low" value={`${overview.key_phenotypes?.arl13b_if_low?.pct ?? 96}%`} color={ACCENT4} />
            <KPI label="MKS Tier" value="NONE" color={ACCENT6} />
          </div>

          <Section title="Gene & Mechanism" color={ACCENT}>
            <div className="row g-2">
              <div className="col-md-6">
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>TULP3 (Tubby-Like Protein 3)</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*602280</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#617622 (JBTS30)</td></tr>
                    <tr><td className="fw-bold">Chromosome</td><td>12p13.31</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>~487 aa; IFT-A cargo adaptor</td></tr>
                    <tr><td className="fw-bold">MKS Tier</td><td style={{ color: ACCENT6 }}>ABSENT — all JBTS30 liveborn</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive (biallelic)</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="col-md-6">
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Cilia Status</td><td style={{ color: ACCENT10 }}>PRESENT + FULL LENGTH (structural intact)</td></tr>
                    <tr><td className="fw-bold">GT335 IF</td><td style={{ color: ACCENT6 }}>NORMAL (polyglutamylation intact)</td></tr>
                    <tr><td className="fw-bold">INPP5E IF</td><td style={{ color: ACCENT2 }}>&lt;30% WT in cilia (cargo import lost)</td></tr>
                    <tr><td className="fw-bold">ARL13B IF</td><td style={{ color: ACCENT4 }}>&lt;35% WT in cilia (cargo import lost)</td></tr>
                    <tr><td className="fw-bold">Key DDx test</td><td>ARL13B ciliary IF (present→JBTS1; absent→JBTS30)</td></tr>
                    <tr><td className="fw-bold">Frequency</td><td>~1–2% of all JBTS (TULP3-specific)</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </Section>

          <Section title="TULP3 Protein Domain Architecture (487 aa)" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT2 + '22' }}>
                  <tr>
                    <th>Domain</th><th>Residues</th><th>Function</th><th>Key Variants</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong style={{ color: ACCENT }}>HC1</strong></td>
                    <td>aa 15–45</td>
                    <td>IFT140 primary contact; IFT144 secondary electrostatic surface</td>
                    <td>Rare; IFT140-contact loss → moderate TULP3 reduction</td>
                  </tr>
                  <tr>
                    <td><strong style={{ color: ACCENT }}>HC2</strong></td>
                    <td>aa 80–110</td>
                    <td>IFT144/WDR19 TPR6–TPR8 primary binding (Arg99, Leu103, Asp115)</td>
                    <td><strong>Arg99Gln</strong> — MENA founder; IFT144 binding −55%</td>
                  </tr>
                  <tr>
                    <td><strong style={{ color: ACCENT }}>HC3</strong></td>
                    <td>aa 145–175</td>
                    <td>IFT144 secondary contact; Pro148 helix junction</td>
                    <td><strong>Pro148Arg</strong> — North African founder; moderate</td>
                  </tr>
                  <tr>
                    <td><strong style={{ color: ACCENT3 }}>Linker/Dimer</strong></td>
                    <td>aa 175–220</td>
                    <td>TULP3 homodimerisation; IFT-A avidity enhancement</td>
                    <td><strong>Leu208Pro</strong> — South Asian; self-association −45%</td>
                  </tr>
                  <tr>
                    <td><strong style={{ color: ACCENT2 }}>Tubby entrance</strong></td>
                    <td>aa 221–260</td>
                    <td>ARL13B cargo recruitment (β7–β8 outer loop); Glu254 charge gate</td>
                    <td><strong>Glu254Lys</strong> — European; ARL13B selective deficiency</td>
                  </tr>
                  <tr>
                    <td><strong style={{ color: ACCENT2 }}>Tubby lumen</strong></td>
                    <td>aa 261–380</td>
                    <td>PI(4,5)P₂ coordination (Tyr309, Lys350); membrane anchoring core</td>
                    <td><strong>Tyr309Cys</strong>; <strong>c.665+1G&gt;A</strong> splice</td>
                  </tr>
                  <tr>
                    <td><strong style={{ color: ACCENT2 }}>Tubby deep</strong></td>
                    <td>aa 381–487</td>
                    <td>PI(4,5)P₂ direct contacts (Arg429, Lys432); all cargo anchoring</td>
                    <td><strong>Arg429Ter</strong> — pan-ethnic null truncation; most severe</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="IFT-A Cargo Adaptor Mechanism" color={ACCENT3}>
            <Alert color={ACCENT3}>
              <strong>TULP3 LOF pathway:</strong> Biallelic TULP3 null → IFT-A complex transported normally
              but carries no TULP3 → INPP5E, ARL13B, GPR161 all excluded from cilia → PI(4,5)P₂ accumulates
              (INPP5E absent) + GPR161 excluded (cAMP/PKA axis impaired) → Hedgehog severely impaired despite
              cilia being <strong>PRESENT and FULL LENGTH</strong> → cerebellar vermis hypoplasia → Molar Tooth Sign.
            </Alert>
            <div className="row g-2">
              <div className="col-md-4">
                <div className="card border-0 shadow-sm h-100">
                  <div className="card-body small">
                    <div className="fw-bold mb-1" style={{ color: ACCENT }}>Step 1 — IFT-A transport intact</div>
                    IFT144/WDR19, IFT140, IFT122 form IFT-A retrograde train normally.
                    TULP3 is absent → IFT-A rides normally but carries no TULP3 payload.
                  </div>
                </div>
              </div>
              <div className="col-md-4">
                <div className="card border-0 shadow-sm h-100">
                  <div className="card-body small">
                    <div className="fw-bold mb-1" style={{ color: ACCENT2 }}>Step 2 — Cargo excluded</div>
                    INPP5E, ARL13B, GPR161 all require TULP3 Tubby domain for ciliary entry.
                    Without TULP3 → all 3 cargoes stay in cytoplasm. Cilia empty of all TULP3-dependent proteins.
                  </div>
                </div>
              </div>
              <div className="col-md-4">
                <div className="card border-0 shadow-sm h-100">
                  <div className="card-body small">
                    <div className="fw-bold mb-1" style={{ color: ACCENT7 }}>Step 3 — Hedgehog failure</div>
                    PI(4,5)P₂ accumulates (no INPP5E). GPR161 absent (no cAMP suppression).
                    Dual Hh suppression (PI(4,5)P₂ + GPR161) → GLI3R excess → Hh fully blocked → MTS.
                  </div>
                </div>
              </div>
            </div>
          </Section>

          <Section title="Clinical Phenotype Profile" color={ACCENT8}>
            <div className="row g-2">
              {[
                { label: "MTS Confirmed",          pct: overview.key_phenotypes?.mts_confirmed?.pct ?? 95,          color: ACCENT  },
                { label: "Cerebellar Ataxia",       pct: overview.key_phenotypes?.cerebellar_ataxia?.pct ?? 83,      color: ACCENT  },
                { label: "Neonatal Hypotonia",      pct: overview.key_phenotypes?.neonatal_hypotonia?.pct ?? 77,     color: ACCENT3 },
                { label: "Oculomotor Apraxia",      pct: overview.key_phenotypes?.oculomotor_apraxia?.pct ?? 45,     color: ACCENT  },
                { label: "Breathing Dysregulation", pct: overview.key_phenotypes?.breathing_dysreg?.pct ?? 42,       color: ACCENT5 },
                { label: "Intellectual Disability", pct: overview.key_phenotypes?.intellectual_disab?.pct ?? 65,     color: ACCENT3 },
                { label: "Renal NPHP-like",         pct: overview.key_phenotypes?.renal_nphp_like?.pct ?? 25,        color: ACCENT8 },
                { label: "Retinal Rod-Cone",        pct: overview.key_phenotypes?.retinal_rod_cone?.pct ?? 30,       color: ACCENT9 },
                { label: "Hepatic CHF",             pct: overview.key_phenotypes?.hepatic_chf?.pct ?? 6,             color: ACCENT5 },
                { label: "Postaxial Polydactyly",   pct: overview.key_phenotypes?.postaxial_polydactyly?.pct ?? 20,  color: ACCENT7 },
                { label: "INPP5E IF Low",           pct: overview.key_phenotypes?.inpp5e_if_low?.pct ?? 97,          color: ACCENT2 },
                { label: "ARL13B IF Low",           pct: overview.key_phenotypes?.arl13b_if_low?.pct ?? 96,          color: ACCENT4 },
              ].map(({ label, pct, color }) => (
                <div key={label} className="col-md-6">
                  <div className="d-flex align-items-center gap-2 mb-1">
                    <span className="small" style={{ width: 180, flexShrink: 0 }}>{label}</span>
                    <div className="flex-grow-1 bg-light rounded" style={{ height: 14 }}>
                      <div className="rounded" style={{ width: `${pct}%`, height: 14, background: color }} />
                    </div>
                    <span className="small fw-bold" style={{ color, width: 38, textAlign: 'right' }}>{pct}%</span>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Key DDx vs JBTS1 INPP5E" color={ACCENT}>
            <Alert color={ACCENT4}>
              <strong>ARL13B ciliary IF — the single most important distinguishing test:</strong><br />
              <strong>JBTS1 (INPP5E LOF):</strong> ARL13B <span style={{ color: ACCENT6 }}>PRESENT</span> in cilia
              (ARL13B import requires TULP3, NOT INPP5E) — INPP5E absent.<br />
              <strong>JBTS30 (TULP3 LOF):</strong> ARL13B <span style={{ color: ACCENT8 }}>ABSENT</span> from cilia
              (ARL13B needs TULP3 for import) — INPP5E also absent.<br />
              Both: cilia full length, GT335 normal, identical Hedgehog failure and MTS. Gene sequencing + ARL13B IF resolve all ambiguous cases.
            </Alert>
          </Section>
        </div>
      )}

      {/* ── TAB 1: DIAGNOSTIC BREAKDOWN ── */}
      {!loading && tab === 1 && breakdown && (
        <div>
          <Section title="Variant Allele Frequency (N=40 cohort, seed 475)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '22' }}>
                  <tr><th>Allele</th><th>Count</th><th>Allele Freq</th></tr>
                </thead>
                <tbody>
                  {(breakdown.variant_counts || []).map((v, i) => (
                    <tr key={i}>
                      <td><strong>{v.allele}</strong></td>
                      <td>{v.count}</td>
                      <td>{v.allele_freq}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Notable TULP3 Variants" color={ACCENT2}>
            {(breakdown.notable_variants || []).map((v, i) => (
              <div key={i} className="card mb-2 border-0 shadow-sm">
                <div className="card-body py-2 px-3">
                  <div className="d-flex align-items-center gap-2 mb-1">
                    <strong style={{ color: ACCENT2 }}>{v.variant}</strong>
                    <span className="badge" style={{ background: ACCENT3 }}>{v.domain?.split(' ')[0]}</span>
                    <span className="badge bg-secondary">{v.population}</span>
                    <span className="badge" style={{ background: ACCENT2 }}>{v.severity}</span>
                  </div>
                  <div className="small text-muted">{v.mechanism}</div>
                  {v.omim_note && <div className="small mt-1" style={{ color: ACCENT5 }}><em>{v.omim_note}</em></div>}
                </div>
              </div>
            ))}
          </Section>

          <Section title="Domain Vulnerability Matrix" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT3 + '22' }}>
                  <tr><th>Domain</th><th>Function</th><th>Clinical Impact</th></tr>
                </thead>
                <tbody>
                  {(breakdown.domain_matrix || []).map((d, i) => (
                    <tr key={i}>
                      <td><strong style={{ color: i < 4 ? ACCENT : ACCENT2 }}>{d.domain}</strong></td>
                      <td className="small">{d.function}</td>
                      <td className="small">{d.impact}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <div className="row g-3">
            <div className="col-md-4">
              <Section title="Sex Distribution" color={ACCENT5}>
                <table className="table table-sm small">
                  <tbody>
                    <tr><td>Male</td><td><strong>{breakdown.sex_distribution?.M}</strong></td></tr>
                    <tr><td>Female</td><td><strong>{breakdown.sex_distribution?.F}</strong></td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-4">
              <Section title="Age Range" color={ACCENT5}>
                <table className="table table-sm small">
                  <tbody>
                    <tr><td>Min</td><td>{breakdown.age_range?.min} yr</td></tr>
                    <tr><td>Median</td><td>{breakdown.age_range?.median} yr</td></tr>
                    <tr><td>Max</td><td>{breakdown.age_range?.max} yr</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-4">
              <Section title="Population" color={ACCENT5}>
                <table className="table table-sm small">
                  <tbody>
                    {Object.entries(breakdown.population_breakdown || {}).map(([pop, n]) => (
                      <tr key={pop}><td>{pop}</td><td><strong>{n}</strong></td></tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: IFT-A CARGO ADAPTOR PEARLS ── */}
      {!loading && tab === 2 && (
        <div>
          <Alert color={ACCENT}>
            <strong>TULP3/JBTS30 — IFT-A Cargo Adaptor Deficiency Class:</strong> TULP3 is the prototypical IFT-A
            cargo adaptor. Unlike structural IFT-A subunit LOF (WDR19/NPHP13, IFT140, IFT122 — cause both
            cilia elongation defects AND cargo loss), TULP3 LOF causes <em>pure cargo import deficiency</em> with
            structurally normal, full-length cilia. This makes JBTS30 mechanistically unique among JBTS subtypes.
          </Alert>

          <Section title="Pearl 1 — ARL13B IF is the Definitive DDx Test (JBTS30 vs JBTS1)" color={ACCENT4}>
            <div className="row g-2">
              <div className="col-md-6">
                <div className="card h-100 border-0 shadow-sm">
                  <div className="card-header small fw-bold" style={{ background: ACCENT + '22', color: ACCENT }}>
                    JBTS1 — INPP5E LOF
                  </div>
                  <div className="card-body small">
                    <ul className="mb-0">
                      <li>INPP5E: <strong style={{ color: ACCENT8 }}>ABSENT</strong> from cilia (&lt;20% WT)</li>
                      <li>ARL13B: <strong style={{ color: ACCENT6 }}>PRESENT</strong> in cilia (TULP3 intact → ARL13B imported normally)</li>
                      <li>GT335: Normal</li>
                      <li>Cilia: Full length, present</li>
                      <li>PI(4,5)P₂: Accumulates (INPP5E absent)</li>
                      <li>GPR161: Normal ciliary import (TULP3 present)</li>
                    </ul>
                  </div>
                </div>
              </div>
              <div className="col-md-6">
                <div className="card h-100 border-0 shadow-sm">
                  <div className="card-header small fw-bold" style={{ background: ACCENT2 + '22', color: ACCENT2 }}>
                    JBTS30 — TULP3 LOF
                  </div>
                  <div className="card-body small">
                    <ul className="mb-0">
                      <li>INPP5E: <strong style={{ color: ACCENT8 }}>ABSENT</strong> from cilia (&lt;30% WT)</li>
                      <li>ARL13B: <strong style={{ color: ACCENT8 }}>ABSENT</strong> from cilia (&lt;35% WT) ← KEY DDx</li>
                      <li>GT335: Normal</li>
                      <li>Cilia: Full length, present</li>
                      <li>PI(4,5)P₂: Accumulates (INPP5E absent)</li>
                      <li>GPR161: <strong style={{ color: ACCENT8 }}>ABSENT</strong> (TULP3 absent → also excluded)</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </Section>

          <Section title="Pearl 2 — GPR161 Exclusion: Dual Hedgehog Suppression" color={ACCENT7}>
            <p className="small mb-2">
              GPR161 suppresses Hedgehog by activating cAMP/PKA → GLI3 repressor form when localised in cilia.
              GPR161 requires TULP3 + IFT-A for ciliary import. TULP3 LOF → <strong>dual Hh suppression</strong>:
            </p>
            <div className="row g-2">
              <div className="col-md-6">
                <div className="card border-0 bg-light small p-2">
                  <strong style={{ color: ACCENT2 }}>Arm 1 — INPP5E absent</strong><br />
                  PI(4,5)P₂ accumulates → wrong phosphoinositide identity → Smo mis-trafficking → Hh impaired
                </div>
              </div>
              <div className="col-md-6">
                <div className="card border-0 bg-light small p-2">
                  <strong style={{ color: ACCENT7 }}>Arm 2 — GPR161 absent</strong><br />
                  No cAMP/PKA activation in cilia → GLI3R insufficient → Hh repression weakened but also
                  dysregulated → net effect: Smo cannot activate GLI-A → Hh pathway collapse
                </div>
              </div>
            </div>
            <div className="small mt-2 text-muted">
              This dual suppression explains why JBTS30 polydactyly (~20%) is slightly higher than JBTS1 (~12%):
              GPR161 exclusion adds an independent Hh suppression arm not present in pure INPP5E LOF.
            </div>
          </Section>

          <Section title="Pearl 3 — Cilia PRESENT and FULL LENGTH (Distinguish from Initiation / Elongation Defects)" color={ACCENT10}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT10 + '22' }}>
                  <tr><th>JBTS Type</th><th>Gene</th><th>Cilia Status</th><th>GT335</th><th>INPP5E IF</th><th>ARL13B IF</th></tr>
                </thead>
                <tbody>
                  <tr style={{ background: '#e8f5e9' }}>
                    <td><strong>JBTS30</strong></td><td>TULP3</td>
                    <td style={{ color: ACCENT6 }}>PRESENT, FULL LENGTH</td>
                    <td style={{ color: ACCENT6 }}>Normal</td>
                    <td style={{ color: ACCENT8 }}>&lt;30% WT</td>
                    <td style={{ color: ACCENT8 }}>&lt;35% WT ← key DDx</td>
                  </tr>
                  <tr>
                    <td>JBTS1</td><td>INPP5E</td>
                    <td style={{ color: ACCENT6 }}>PRESENT, FULL LENGTH</td>
                    <td style={{ color: ACCENT6 }}>Normal</td>
                    <td style={{ color: ACCENT8 }}>&lt;20% WT</td>
                    <td style={{ color: ACCENT6 }}>PRESENT (~85% WT)</td>
                  </tr>
                  <tr>
                    <td>JBTS27</td><td>ARMC9</td>
                    <td style={{ color: ACCENT8 }}>ABSENT (null) / SHORT (hypo)</td>
                    <td style={{ color: ACCENT6 }}>Normal</td>
                    <td>Reduced (absent cilia)</td>
                    <td>Absent (no cilia)</td>
                  </tr>
                  <tr>
                    <td>JBTS29</td><td>TOGARAM1</td>
                    <td style={{ color: '#e65100' }}>SHORT (30–60% WT)</td>
                    <td style={{ color: ACCENT8 }}>&lt;35% WT (GT335 reduced)</td>
                    <td>Reduced</td>
                    <td>Reduced</td>
                  </tr>
                  <tr>
                    <td>JBTS5</td><td>CEP290</td>
                    <td style={{ color: ACCENT8 }}>ABSENT / SHORTENED</td>
                    <td>Variable</td>
                    <td style={{ color: ACCENT8 }}>Reduced (gate defect)</td>
                    <td style={{ color: ACCENT8 }}>Reduced (gate defect)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Pearl 4 — Surveillance Protocol" color={ACCENT3}>
            <div className="row g-2">
              {[
                { label: "Fibroblast IF (diagnosis)", detail: "INPP5E + ARL13B ciliary IF from skin fibroblasts — confirms JBTS30; INPP5E <30% WT + ARL13B <35% WT; ARL13B absence distinguishes from JBTS1; baseline values guide prognosis", color: ACCENT2 },
                { label: "Brain MRI", detail: "At diagnosis: MTS confirmation + cerebellar vermis hypoplasia grading; annual physiotherapy, speech therapy, cognitive assessment; OMA tracking", color: ACCENT },
                { label: "Renal (annual)", detail: "Renal US + creatinine/eGFR from diagnosis; NPHP-like ~25%; ESRD ~8%; transplant curative — no recurrence (cell-autonomous TULP3 defect)", color: ACCENT8 },
                { label: "Retinal (from age 3)", detail: "ERG + fundus annually; rod-cone dystrophy ~30%; connecting cilia cargo defect → progressive; low-vision support early; ARL13B absent from photoreceptor connecting cilia", color: ACCENT9 },
                { label: "Gene panel", detail: "Full JBTS WES including INPP5E (JBTS1), ARL13B (JBTS8), CEP290 (JBTS5), IFT144/WDR19 (NPHP13) alongside TULP3 — full IFT-A adaptor axis required", color: ACCENT3 },
              ].map(({ label, detail, color }) => (
                <div key={label} className="col-md-6">
                  <div className="card border-0 shadow-sm h-100">
                    <div className="card-body small">
                      <div className="fw-bold mb-1" style={{ color }}>{label}</div>
                      {detail}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {!loading && tab === 3 && defs && (
        <div>
          <div className="row g-2 mb-3">
            <div className="col-md-4">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-body small">
                  <div className="fw-bold mb-1" style={{ color: ACCENT }}>Gene</div>
                  <div>TULP3 — OMIM *602280</div>
                  <div className="fw-bold mt-2 mb-1" style={{ color: ACCENT }}>Disease</div>
                  <div>JBTS30 — OMIM #617622</div>
                  <div className="fw-bold mt-2 mb-1" style={{ color: ACCENT }}>Chromosome</div>
                  <div>12p13.31</div>
                  <div className="fw-bold mt-2 mb-1" style={{ color: ACCENT6 }}>MKS Tier</div>
                  <div>ABSENT — all JBTS30 liveborn</div>
                </div>
              </div>
            </div>
            <div className="col-md-8">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-body small">
                  <div className="fw-bold mb-1" style={{ color: ACCENT2 }}>Key DDx Panel</div>
                  <ul className="mb-0">
                    {(defs.key_ddx || []).map((d, i) => (
                      <li key={i} className="mb-1">{d}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <Section title="Glossary" color={ACCENT}>
            {(defs.definitions || []).map((d, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: i % 2 === 0 ? ACCENT + '08' : '#f8f9fa' }}>
                <strong style={{ color: ACCENT }}>{d.term}</strong>
                <span className="text-muted mx-1">—</span>
                <span className="small">{d.definition}</span>
              </div>
            ))}
          </Section>

          <Section title="Founder Variants" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT2 + '22' }}>
                  <tr><th>Variant</th><th>Population</th><th>Domain</th><th>Severity</th></tr>
                </thead>
                <tbody>
                  {(defs.founder_variants || []).map((v, i) => (
                    <tr key={i}>
                      <td><strong>{v.variant}</strong></td>
                      <td>{v.population}</td>
                      <td className="small">{v.domain}</td>
                      <td>{v.severity}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Surveillance & Treatment" color={ACCENT3}>
            {defs.surveillance_protocol && (
              <div className="row g-2 mb-3">
                {Object.entries(defs.surveillance_protocol).map(([k, v]) => (
                  <div key={k} className="col-md-6">
                    <div className="card border-0 bg-light h-100">
                      <div className="card-body small">
                        <div className="fw-bold mb-1 text-capitalize">{k.replace(/_/g,' ')}</div>
                        {v}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {defs.treatment && (
              <div className="row g-2">
                {Object.entries(defs.treatment).map(([k, v]) => (
                  <div key={k} className="col-md-6">
                    <div className="card border-0 shadow-sm h-100">
                      <div className="card-body small">
                        <div className="fw-bold mb-1 text-capitalize" style={{ color: ACCENT3 }}>{k.replace(/_/g,' ')}</div>
                        {v}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Section>
        </div>
      )}
    </div>
  );
}
