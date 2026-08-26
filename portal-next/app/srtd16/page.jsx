'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'KIAA0586 Centriolar Scaffold & CPLANE Complex', 'Definitions'];

// SRTD16 colour scheme — KIAA0586/TALPID3 / Centriolar/CPLANE scaffold / Joubert JBTS23 / Sixth distinct class
const ACCENT  = '#1a237e';   // deep indigo — sixth distinct class; rare centriolar scaffold
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax; severe; SRPS-like spectrum
const ACCENT3 = '#01579b';   // deep blue — renal TIN; ESRD; NPHP-like
const ACCENT4 = '#e65100';   // burnt orange — polydactyly 55%; second-highest after NEK1
const ACCENT5 = '#4a148c';   // deep purple — Joubert JBTS23 alleles; MTS; unique feature
const ACCENT6 = '#004d40';   // dark teal — absent cilia EM; centriolar/CPLANE class
const ACCENT7 = '#bf360c';   // dark red-orange — perinatal lethality; SRPS-like; hydrops
const ACCENT8 = '#006064';   // dark cyan — CPLANE complex; NEK1 coordination

const SEED = 413;

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
  return (
    <span className="badge me-1 mb-1" style={{ background: color, fontSize: '0.75rem' }}>{text}</span>
  );
}

function BarRow({ label, n, total, color }) {
  const pct = total ? Math.round(n / total * 100) : 0;
  return (
    <div className="mb-1">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{n} ({pct}%)</span>
      </div>
      <div style={{ background: '#e9ecef', borderRadius: 4, height: 8 }}>
        <div style={{ width: `${pct}%`, background: color, borderRadius: 4, height: 8, transition: 'width 0.5s' }} />
      </div>
    </div>
  );
}

export default function SRTD16Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd16/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd16/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd16/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOverview(o); setBreakdown(b); setDefs(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  const kpis = overview?.kpis || {};
  const N    = overview?.cohort_n || 40;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded-3" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT5}11)`, border: `2px solid ${ACCENT}` }}>
        <div className="d-flex flex-wrap align-items-center gap-2 mb-1">
          <span style={{ fontSize: '2rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
              KIAA0586 Short-Rib Thoracic Dysplasia 16
            </h4>
            <div className="text-muted small">
              SRTD16 / ATD16 / Jeune Syndrome 16 &nbsp;·&nbsp;
              <strong>Gene:</strong> KIAA0586 / TALPID3 / CPLANE1 (*610178) &nbsp;·&nbsp;
              <strong>OMIM Disease:</strong> #617098 &nbsp;·&nbsp;
              <strong>Chr:</strong> 14q23.1 &nbsp;·&nbsp;
              <strong>Inheritance:</strong> AR biallelic LOF &nbsp;·&nbsp;
              <strong>Cohort:</strong> N={N} (seed {SEED})
            </div>
          </div>
        </div>
        <div className="d-flex flex-wrap gap-1 mt-2">
          <Badge text="Centriolar/CPLANE Scaffold — SIXTH Distinct SRTD Molecular Class" color={ACCENT} />
          <Badge text="ABSENT / RUDIMENTARY Cilia EM — Same Class as NEK1/SRTD6" color={ACCENT6} />
          <Badge text="Joubert JBTS23 Alleles — ONLY SRTD Gene with Joubert Overlap" color={ACCENT5} />
          <Badge text="Polydactyly 55% — 2nd Highest After NEK1/SRTD6 (65–75%)" color={ACCENT4} />
          <Badge text="<20 Families (2026) · Ultra-Rare" color={ACCENT2} />
          <Badge text="Chr 14q23.1 · 1,624 aa · Multi-Domain Coiled-Coil" color={ACCENT3} />
          <Badge text="CPLANE Complex · NEK1 Coordination · IFT Recruitment" color={ACCENT8} />
          <Badge text="AR Biallelic LOF" color={ACCENT4} />
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottom: `3px solid ${ACCENT}` } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ─────────────────────────────────────────── */}
      {tab === 0 && (
        <div>
          <Alert color={ACCENT}>
            <strong>KIAA0586 (SRTD16)</strong> is the defining member of the{' '}
            <strong>SIXTH distinct SRTD molecular class: Centriolar/CPLANE scaffold</strong>.
            The large centriolar scaffold (~1,624 aa) is required for basal body maturation,
            CPLANE complex assembly, and IFT protein recruitment to the ciliary base. Loss →
            ciliogenesis aborts at step 0 → <strong>ABSENT/RUDIMENTARY cilia</strong> (same
            EM as SRTD6/NEK1; gene panel mandatory). Hypomorphic C-terminal CC3 alleles →
            <strong> Joubert Syndrome 23 (JBTS23)</strong> — MTS, cerebellar vermis hypoplasia,
            ataxia — the <strong>ONLY SRTD with confirmed Joubert alleles</strong>.
          </Alert>

          {/* KPIs */}
          <div className="row g-2 mb-3">
            <KPI label="Cohort N"          value={N}                                                              color={ACCENT} />
            <KPI label="Thorax Severe"     value={`${kpis.thorax_severe_n} (${kpis.thorax_severe_pct}%)`}        color={ACCENT2} />
            <KPI label="Polydactyly"       value={`${kpis.polydactyly_n} (${kpis.polydactyly_pct}%)`}           color={ACCENT4} />
            <KPI label="Renal Involved"    value={`${kpis.renal_any_n} (${kpis.renal_any_pct}%)`}               color={ACCENT3} />
            <KPI label="Joubert JBTS23"    value={`${kpis.joubert_n} (${kpis.joubert_pct}%)`}                   color={ACCENT5} />
            <KPI label="Hydrops"           value={`${kpis.hydrops_n} (${kpis.hydrops_pct}%)`}                   color={ACCENT7} />
            <KPI label="Retinal Involved"  value={`${kpis.retinal_any_n} (${kpis.retinal_any_pct}%)`}           color={ACCENT5} />
            <KPI label="CHF/Hepatic"       value={`${kpis.hepatic_chf_n} (${kpis.hepatic_chf_pct}%)`}           color={ACCENT7} />
            <KPI label="VEPTR/MAGEC"       value={`${kpis.veptr_any_n} (${kpis.veptr_any_pct}%)`}               color={ACCENT} />
            <KPI label="Perinatal Death"   value={`${kpis.perinatal_death_n} (${kpis.perinatal_death_pct}%)`}   color={ACCENT7} />
            <KPI label="Misdiagnosed"      value={`${kpis.misdiagnosis_n} (${kpis.misdiagnosis_pct}%)`}         color="#607d8b" />
            <KPI label="Renal Tx Done"     value={kpis.transplant_done_n}                                        color={ACCENT3} />
          </div>

          {/* Mechanism */}
          <Section title="Molecular Mechanism — KIAA0586 Centriolar/CPLANE Scaffold" color={ACCENT}>
            <p className="small">{overview.mechanism}</p>
          </Section>

          {/* EM Distribution */}
          <Section title="Ciliary EM Distribution — Absent / Rudimentary (Centriolar/CPLANE Class)" color={ACCENT6}>
            {(overview.em_distribution || []).map(r => (
              <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT6} />
            ))}
            <div className="small text-muted mt-1">
              Absent/rudimentary cilia EM is <strong>identical between SRTD16 (KIAA0586) and SRTD6 (NEK1)</strong> —
              gene panel is the ONLY differentiator. Joubert CC3-hypomorphic alleles may show partial axoneme.
            </div>
          </Section>

          {/* Key distinction */}
          <Section title="Key Clinical Distinction — Why SRTD16 Is Unique Among All SRTDs" color={ACCENT5}>
            <p className="small">{overview.key_distinction}</p>
          </Section>

          {/* SRTD molecular class table */}
          <Section title="SRTD Molecular Class Table — All Six Classes (EM-Based Classification)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '22' }}>
                  <tr><th>Class</th><th>EM Finding</th><th>SRTD Genes</th><th>Mechanism</th></tr>
                </thead>
                <tbody>
                  {(overview.srtd_molecular_class_table || []).map((r, i) => (
                    <tr key={i} style={r.genes?.includes('THIS') ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                      <td>{r.class}</td><td>{r.em}</td><td>{r.genes}</td><td>{r.why}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* CPLANE complex table */}
          <Section title="CPLANE Complex Members — Ciliogenesis and Planar Polarity Effectors" color={ACCENT8}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT8 + '22' }}>
                  <tr>
                    <th>Component</th><th>Gene / SRTD</th><th>CPLANE Role</th><th>OMIM Gene</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.cplane_complex_table || []).map((r, i) => (
                    <tr key={i} style={r.gene_srtd?.includes('THIS') ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                      <td>{r.component}</td>
                      <td>{r.gene_srtd}</td>
                      <td>{r.role}</td>
                      <td>{r.omim}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="small text-muted mt-1">
              KIAA0586 (SRTD16) is the centriolar scaffold subunit of the CPLANE complex — the scaffold on
              which IFT machinery is assembled at the ciliary base. NEK1/SRTD6 is the kinase that coordinates
              CP110 removal; KIAA0586 provides the physical IFT recruitment platform.
            </div>
          </Section>

          {/* Age distribution */}
          <Section title="Age at Diagnosis" color={ACCENT4}>
            {overview.age_distribution && Object.entries({
              '0–1 yr (neonatal/infant)': overview.age_distribution.dx_0_1yr,
              '2–5 yr (early childhood)': overview.age_distribution.dx_2_5yr,
              '6–11 yr (school age)':     overview.age_distribution.dx_6_10yr,
              '12+ yr (Joubert-only CC3 alleles)': overview.age_distribution.dx_11_plus,
            }).map(([k, v]) => <BarRow key={k} label={k} n={v} total={N} color={ACCENT4} />)}
          </Section>
        </div>
      )}

      {/* ── TAB 1: DIAGNOSTIC BREAKDOWN ─────────────────────────────── */}
      {tab === 1 && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Thorax Severity Distribution" color={ACCENT2}>
              {breakdown.thorax_distribution?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT2} />)}
            </Section>
            <Section title="Polydactyly Distribution" color={ACCENT4}>
              {breakdown.polydactyly_distribution?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT4} />)}
            </Section>
            <Section title="Ciliary EM Pattern" color={ACCENT6}>
              {breakdown.em_distribution?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT6} />)}
            </Section>
            <Section title="Joubert (JBTS23) Features — CC3 Hypomorphic Alleles" color={ACCENT5}>
              {breakdown.joubert_distribution?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT5} />)}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Renal Involvement" color={ACCENT3}>
              {breakdown.renal_distribution?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT3} />)}
            </Section>
            <Section title="Allele Class Summary" color={ACCENT}>
              {breakdown.allele_class_summary?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT} />)}
            </Section>
            <Section title="Ethnicity Distribution" color={ACCENT8}>
              {breakdown.ethnicity_distribution?.map(r => <BarRow key={r.ethnicity} label={r.ethnicity} n={r.n} total={N} color={ACCENT8} />)}
            </Section>
            <Section title="Misdiagnosis Distribution" color="#607d8b">
              {breakdown.misdiagnosis_distribution?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color="#607d8b" />)}
            </Section>
          </div>

          {/* Top Variants */}
          <div className="col-12">
            <Section title="Top Pathogenic Variants (KIAA0586 / TALPID3)" color={ACCENT}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead style={{ background: ACCENT + '22' }}>
                    <tr><th>Variant</th><th>N in cohort</th></tr>
                  </thead>
                  <tbody>
                    {breakdown.top_variants?.map((v, i) => (
                      <tr key={i}><td>{v.variant}</td><td className="fw-bold">{v.n}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>

            <Section title="VEPTR / MAGEC Surgical Distribution" color={ACCENT}>
              {breakdown.veptr_distribution?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT} />)}
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 2: KIAA0586 STRUCTURE & CPLANE ──────────────────────── */}
      {tab === 2 && defs && (
        <div>
          <Alert color={ACCENT}>
            <strong>KIAA0586 molecular architecture:</strong> ~1,624 aa; large multi-domain coiled-coil
            centriolar scaffold. CC1 N-terminal (aa 1–400): distal appendage anchoring.
            CC2 central + CPLANE interface (aa 401–1,100): CPLANE complex assembly; INTU/FUZ contact; IFT recruitment.
            CC3 C-terminal (aa 1,100–1,624): NEK1 coordination; IFT platform; Joubert hypomorphic domain (JBTS23).
          </Alert>

          <Section title="KIAA0586 Domain Map — Multi-Domain Centriolar/CPLANE Scaffold" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '22' }}>
                  <tr><th>Domain</th><th>Region (aa)</th><th>Partner / Function</th><th>Variant Class Consequence</th></tr>
                </thead>
                <tbody>
                  {[
                    ['CC1 N-terminal coiled-coil', 'aa 1–400', 'Distal appendage anchoring; centriole-to-basal-body transition; ciliary membrane docking initiation', 'Missense → basal body maturation delayed; IFT recruitment reduced → moderate SRTD16'],
                    ['CC2 central coiled-coil + CPLANE interface', 'aa 401–1,100', 'CPLANE complex assembly; INTU/FUZ/WDPCP interaction; IFT-A/B recruitment to basal body platform', 'Missense → CPLANE partially uncoupled → IFT recruitment severely reduced → absent cilia; moderate-severe SRTD16'],
                    ['CC3 C-terminal coiled-coil', 'aa 1,100–1,624', 'IFT protein recruitment; NEK1 functional coordination for CP110 removal; IFT platform assembly; Joubert hypomorphic domain', 'Hypomorphic missense → partial loss; ciliary platform reduced → JBTS23 (Joubert MTS + ataxia) without thoracic SRTD'],
                  ].map(([d, aa, fn, cons], i) => (
                    <tr key={i}>
                      <td className="fw-bold" style={{ color: ACCENT }}>{d}</td>
                      <td>{aa}</td><td>{fn}</td><td>{cons}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="SRTD16 vs SRTD6 — Absent Cilia: Two Distinct Mechanisms" color={ACCENT6}>
            <Alert color={ACCENT6}>
              Both SRTD16 (KIAA0586) and SRTD6 (NEK1) produce <strong>absent/rudimentary cilia</strong> (EM indistinguishable).
              However, the mechanisms differ fundamentally:
              NEK1 is the <strong>kinase</strong> (phosphorylates TTBK2 → CP110 removal → ciliogenesis initiation);
              KIAA0586 is the <strong>centriolar scaffold</strong> (IFT recruitment platform after CP110 removal).
              Both are required at or before IFT assembly; gene panel is the ONLY differentiator.
            </Alert>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT6 + '22' }}>
                  <tr><th>Feature</th><th>SRTD6 (NEK1)</th><th>SRTD16 (KIAA0586 — THIS GENE)</th></tr>
                </thead>
                <tbody>
                  {[
                    ['Molecular class', 'Basal Body Kinase', 'Centriolar/CPLANE Scaffold'],
                    ['EM finding', 'Absent / rudimentary cilia', 'Absent / rudimentary cilia (same)'],
                    ['Mechanism', 'NEK1 fails to phosphorylate TTBK2 → CP110 not removed → step 0 fails', 'KIAA0586 absent → IFT machinery cannot be recruited to basal body after CP110 removal'],
                    ['Polydactyly', '65–75% — highest of all SRTDs', '~55% — second-highest'],
                    ['Hydrops fetalis', '~20% — UNIQUE SRTD6 feature', '~10%'],
                    ['Medial nasal hypoplasia', '~30% — UNIQUE SRTD6 feature', 'Absent'],
                    ['Joubert alleles', 'None (NEK1 has no Joubert designation)', 'JBTS23 (CC3 hypomorphic) — UNIQUE SRTD16'],
                    ['CPLANE complex', 'NEK1 coordinates with KIAA0586 but is not a CPLANE member', 'CPLANE scaffold (INTU/FUZ/WDPCP partner)'],
                    ['Gene panel', 'MANDATORY — EM identical', 'MANDATORY — EM identical to SRTD6'],
                  ].map(([feat, nek1, kiaa], i) => (
                    <tr key={i}>
                      <td className="fw-bold">{feat}</td>
                      <td>{nek1}</td>
                      <td style={{ background: ACCENT + '0a', fontWeight: kiaa.includes('UNIQUE SRTD16') ? 'bold' : 'normal' }}>{kiaa}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="SRTD16 vs JBTS23 — Allele-Phenotype Spectrum (KIAA0586)" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT5 + '22' }}>
                  <tr><th>Allele Class</th><th>Domain</th><th>Phenotype</th><th>Designation</th></tr>
                </thead>
                <tbody>
                  {[
                    ['Biallelic null (truncating)', 'Any → complete LOF', 'SRPS-like — perinatal lethal; absent cilia; hydrops possible', 'SRTD16 (severe/SRPS-like)'],
                    ['Biallelic CC1–CC2 missense', 'Distal appendage / CPLANE interface', 'Full SRTD16 — narrow thorax, polydactyly (55%), absent cilia', 'SRTD16'],
                    ['Hypomorphic CC3 missense', 'C-terminal IFT platform / Joubert domain (aa 1,100–1,624)', 'JBTS23 — cerebellar vermis hypoplasia, MTS, ataxia; no/mild thorax', 'JBTS23 (Joubert)'],
                  ].map(([a, d, ph, des], i) => (
                    <tr key={i} style={ph.includes('JBTS23') ? { background: ACCENT5 + '18' } : ph.includes('SRPS') ? { background: ACCENT7 + '18' } : {}}>
                      <td className="fw-bold">{a}</td><td>{d}</td><td>{ph}</td>
                      <td className="fw-bold" style={{ color: ph.includes('JBTS23') ? ACCENT5 : ph.includes('SRPS') ? ACCENT7 : ACCENT }}>{des}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Mechanism Glossary" color={ACCENT8}>
            {defs.mechanism_glossary?.map((g, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: ACCENT8 + '0a', border: `1px solid ${ACCENT8}33` }}>
                <div className="fw-bold small" style={{ color: ACCENT8 }}>{g.term}</div>
                <div className="small text-muted">{g.definition}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ──────────────────────────────────────── */}
      {tab === 3 && defs && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <Section title="Gene Card — KIAA0586 / TALPID3 / CPLANE1" color={ACCENT}>
                {defs.gene_card && Object.entries(defs.gene_card).map(([k, v]) => (
                  <div key={k} className="d-flex gap-2 mb-1 small">
                    <span className="fw-bold text-nowrap" style={{ color: ACCENT, minWidth: 140 }}>{k.replace(/_/g, ' ')}:</span>
                    <span>{v}</span>
                  </div>
                ))}
              </Section>
              <Section title="Disease Card — SRTD16 / JBTS23" color={ACCENT2}>
                {defs.disease_card && Object.entries(defs.disease_card).map(([k, v]) => (
                  <div key={k} className="d-flex gap-2 mb-1 small">
                    <span className="fw-bold text-nowrap" style={{ color: ACCENT2, minWidth: 140 }}>{k.replace(/_/g, ' ')}:</span>
                    <span>{v}</span>
                  </div>
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Key Pathogenic Variants" color={ACCENT}>
                {defs.key_variants?.map((v, i) => (
                  <div key={i} className="mb-2 p-2 rounded" style={{ background: ACCENT + '0a', border: `1px solid ${ACCENT}33` }}>
                    <div className="fw-bold small" style={{ color: ACCENT }}>{v.variant} — {v.domain}</div>
                    <div className="small text-muted">{v.consequence}</div>
                    <div className="small"><span className="badge" style={{ background: ACCENT5, fontSize: '0.7rem' }}>{v.ethnicity}</span></div>
                  </div>
                ))}
              </Section>
              <Section title="Differential Diagnosis Table" color={ACCENT2}>
                {defs.ddx_table?.map((d, i) => (
                  <div key={i} className="mb-2 p-2 rounded" style={{ background: '#fff3e0', border: `1px solid ${ACCENT2}44` }}>
                    <div className="fw-bold small" style={{ color: ACCENT2 }}>{d.disease}</div>
                    <div className="small text-muted">{d.key_difference}</div>
                  </div>
                ))}
              </Section>
            </div>

            <div className="col-12">
              <Section title="Diagnostic Workup — SRTD16 / JBTS23" color={ACCENT3}>
                <ol className="small ps-3">
                  {defs.diagnostic_workup?.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
                </ol>
              </Section>
              <Section title="Treatment Summary" color={ACCENT8}>
                <ol className="small ps-3">
                  {defs.treatment_summary?.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
                </ol>
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* Back link */}
      <div className="mt-4">
        <Link href="/" className="btn btn-sm btn-outline-secondary">← Back to Portal</Link>
      </div>
    </div>
  );
}
