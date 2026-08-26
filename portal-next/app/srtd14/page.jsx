'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'IFT43 Alpha-Helical Structure & IFT-A C-Cap Stabilization', 'Definitions'];

// SRTD14 colour scheme — IFT43/C14orf179 / IFT-A peripheral C-cap stabilizer / smallest IFT-A subunit / CED3-like ectodermal
const ACCENT  = '#7b1fa2';   // deep purple — IFT-A peripheral stabilizer; smallest subunit
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax; severe; SRPS spectrum
const ACCENT3 = '#01579b';   // deep blue — renal TIN; ESRD
const ACCENT4 = '#e65100';   // burnt orange — polydactyly; postaxial dominant
const ACCENT5 = '#2e7d32';   // deep green — IFT-A C-cap; short stubby cilia
const ACCENT6 = '#006064';   // dark cyan — IFT-A class EM identical to SRTD2/4/5/7/9
const ACCENT7 = '#bf360c';   // dark red-orange — perinatal lethality; SRPS
const ACCENT8 = '#4a148c';   // very deep purple — CED3-like ectodermal dual phenotype

const SEED = 411;

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

export default function SRTD14Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd14/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd14/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd14/definitions`).then(r => r.json()),
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
      <div className="mb-3 p-3 rounded-3" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT8}11)`, border: `2px solid ${ACCENT}` }}>
        <div className="d-flex flex-wrap align-items-center gap-2 mb-1">
          <span style={{ fontSize: '2rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
              IFT43 Short-Rib Thoracic Dysplasia 14
            </h4>
            <div className="text-muted small">
              SRTD14 / ATD14 / Jeune Syndrome 14 &nbsp;·&nbsp;
              <strong>Gene:</strong> IFT43 / C14orf179 (*614068) &nbsp;·&nbsp;
              <strong>OMIM Disease:</strong> #616546 &nbsp;·&nbsp;
              <strong>Chr:</strong> 14q24.3 &nbsp;·&nbsp;
              <strong>Inheritance:</strong> AR biallelic LOF &nbsp;·&nbsp;
              <strong>Cohort:</strong> N={N} (seed {SEED})
            </div>
          </div>
        </div>
        <div className="d-flex flex-wrap gap-1 mt-2">
          <Badge text="IFT-A Peripheral C-Cap Stabilizer — Smallest IFT-A Subunit (362 aa)" color={ACCENT} />
          <Badge text="SHORT STUBBY Cilia — IFT-A Class (Same SRTD2/4/5/7/9)" color={ACCENT6} />
          <Badge text="CED3-like Dual Phenotype (α-Helix-5 C-Terminal Hypomorphic)" color={ACCENT8} />
          <Badge text="<10–15 Families (2026) · Ultra-Rare" color={ACCENT2} />
          <Badge text="Chr 14q24.3 · 362 aa · Entirely Alpha-Helical" color={ACCENT3} />
          <Badge text="6th & Final IFT-A SRTD — Completes IFT-A Subunit Table" color={ACCENT5} />
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
            <strong>IFT43 (SRTD14)</strong> is the <strong>SIXTH and SMALLEST IFT-A subunit</strong> (~362 aa;
            entirely alpha-helical) — the <strong>peripheral C-cap stabilizer</strong> that docks onto
            IFT122 (SRTD2) at its WD10–12 C-terminal propeller face. Loss → partial IFT-A peripheral
            destabilisation → IFT-B import failure →
            <strong> SHORT STUBBY cilia (IFT-A class)</strong>.
            Hypomorphic C-terminal α-helix-5 alleles → <strong>CED3-like ectodermal phenotype</strong>
            (sparse hair, hypodontia) without thoracic disease.
            SRTD14 <strong>completes the IFT-A SRTD subunit table</strong>.
          </Alert>

          {/* KPIs */}
          <div className="row g-2 mb-3">
            <KPI label="Cohort N"          value={N}                                                              color={ACCENT} />
            <KPI label="Thorax Severe"     value={`${kpis.thorax_severe_n} (${kpis.thorax_severe_pct}%)`}        color={ACCENT2} />
            <KPI label="Polydactyly"       value={`${kpis.polydactyly_n} (${kpis.polydactyly_pct}%)`}           color={ACCENT4} />
            <KPI label="Renal Involved"    value={`${kpis.renal_any_n} (${kpis.renal_any_pct}%)`}               color={ACCENT3} />
            <KPI label="CED3-like"         value={`${kpis.ced3_n} (${kpis.ced3_pct}%)`}                         color={ACCENT8} />
            <KPI label="CED3-only"         value={`${kpis.ced3_only_n} (${kpis.ced3_only_pct}%)`}               color={ACCENT8} />
            <KPI label="Retinal Involved"  value={`${kpis.retinal_any_n} (${kpis.retinal_any_pct}%)`}           color={ACCENT5} />
            <KPI label="CHF/Hepatic"       value={`${kpis.hepatic_chf_n} (${kpis.hepatic_chf_pct}%)`}           color={ACCENT7} />
            <KPI label="VEPTR/MAGEC"       value={`${kpis.veptr_any_n} (${kpis.veptr_any_pct}%)`}               color={ACCENT} />
            <KPI label="Perinatal Death"   value={`${kpis.perinatal_death_n} (${kpis.perinatal_death_pct}%)`}   color={ACCENT7} />
            <KPI label="Misdiagnosed"      value={`${kpis.misdiagnosis_n} (${kpis.misdiagnosis_pct}%)`}         color="#607d8b" />
            <KPI label="Renal Tx Done"     value={kpis.transplant_done_n}                                        color={ACCENT3} />
          </div>

          {/* Mechanism */}
          <Section title="Molecular Mechanism — IFT43 IFT-A Peripheral C-Cap Stabilizer" color={ACCENT}>
            <p className="small">{overview.mechanism}</p>
          </Section>

          {/* EM Distribution */}
          <Section title="Ciliary EM Distribution (IFT-A Class — Short Stubby)" color={ACCENT6}>
            {(overview.em_distribution || []).map(r => (
              <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT6} />
            ))}
            <div className="small text-muted mt-1">
              Short stubby EM is <strong>identical across SRTD2/4/5/7/9/14</strong> — gene panel is the ONLY differentiator.
              Hypomorphic C-terminal alleles may show near-normal cilia.
            </div>
          </Section>

          {/* Key distinction */}
          <Section title="Key Clinical Distinction — Why SRTD14 vs Other IFT-A SRTDs" color={ACCENT8}>
            <p className="small">{overview.key_distinction}</p>
          </Section>

          {/* IFT-A subunit table */}
          <Section title="Complete IFT-A Subunit Table (All 6 SRTD-Relevant Subunits)" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT5 + '22' }}>
                  <tr>
                    <th>Subunit</th><th>Gene / SRTD</th><th>IFT-A Role</th><th>OMIM Gene</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.ift_a_subunit_table || []).map((r, i) => (
                    <tr key={i} style={r.gene_srtd?.includes('THIS') ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                      <td>{r.subunit}</td>
                      <td>{r.gene_srtd}</td>
                      <td>{r.role}</td>
                      <td>{r.omim}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="small text-muted mt-1">
              SRTD14 (IFT43) is the <strong>6th and final IFT-A subunit</strong> to receive a SRTD designation —
              completing the molecular IFT-A SRTD table. IFT43 was absent from all pre-2016 SRTD panels.
            </div>
          </Section>

          {/* SRTD molecular class table */}
          <Section title="SRTD Molecular Class Table (EM-Based Classification)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '22' }}>
                  <tr><th>Class</th><th>EM Finding</th><th>SRTD Genes</th><th>Mechanism</th></tr>
                </thead>
                <tbody>
                  {(overview.srtd_molecular_class_table || []).map((r, i) => (
                    <tr key={i} style={r.class?.includes('IFT-A') ? { background: ACCENT6 + '18', fontWeight: 'bold' } : {}}>
                      <td>{r.class}</td><td>{r.em}</td><td>{r.genes}</td><td>{r.why}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Age distribution */}
          <Section title="Age at Diagnosis" color={ACCENT4}>
            {overview.age_distribution && Object.entries({
              '0–1 yr (neonatal/infant)': overview.age_distribution.dx_0_1yr,
              '2–5 yr (early childhood)': overview.age_distribution.dx_2_5yr,
              '6–11 yr (school age)':     overview.age_distribution.dx_6_10yr,
              '12+ yr (adolescent/adult)': overview.age_distribution.dx_11_plus,
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
            <Section title="CED3-like Ectodermal Features (α-Helix-5 Hypomorphic Alleles)" color={ACCENT8}>
              {breakdown.ced3_distribution?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT8} />)}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Renal Involvement" color={ACCENT3}>
              {breakdown.renal_distribution?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT3} />)}
            </Section>
            <Section title="Allele Class Summary" color={ACCENT}>
              {breakdown.allele_class_summary?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT} />)}
            </Section>
            <Section title="Ethnicity Distribution" color={ACCENT5}>
              {breakdown.ethnicity_distribution?.map(r => <BarRow key={r.ethnicity} label={r.ethnicity} n={r.n} total={N} color={ACCENT5} />)}
            </Section>
            <Section title="Misdiagnosis Distribution" color="#607d8b">
              {breakdown.misdiagnosis_distribution?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color="#607d8b" />)}
            </Section>
          </div>

          {/* Top Variants */}
          <div className="col-12">
            <Section title="Top Pathogenic Variants (IFT43 / C14orf179)" color={ACCENT}>
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

      {/* ── TAB 2: IFT43 STRUCTURE & IFT-A C-CAP ──────────────────── */}
      {tab === 2 && defs && (
        <div>
          <Alert color={ACCENT}>
            <strong>IFT43 molecular architecture:</strong> ~362 aa; entirely alpha-helical (no WD40, no TPR); SMALLEST IFT-A subunit.
            α-helix 1–2 (aa 1–80): IFT122-WD10 upper-surface dock.
            α-helix 3–4 (aa 81–200): IFT122-WD11/WD12 C-cap cohesion.
            α-helix 5 (aa 201–362): ectodermal-cilia-specific domain (CED3-like hypomorphic alleles here).
          </Alert>

          <Section title="IFT43 Domain Map — Alpha-Helical IFT-A C-Cap Stabilizer" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '22' }}>
                  <tr><th>Domain</th><th>Region (aa)</th><th>Partner / Function</th><th>Variant Class Consequence</th></tr>
                </thead>
                <tbody>
                  {[
                    ['α-helix 1–2 N-terminal stabilization', 'aa 1–80', 'IFT122-WD10 upper-surface; electrostatic clamp for IFT-A C-cap; primary anchor', 'Missense → WD10 face destabilised → IFT-A C-cap weakened → moderate SRTD14'],
                    ['α-helix 3–4 central docking module',   'aa 81–200', 'IFT122-WD11/WD12 contact; stabilises entire C-terminal propeller cap; IFT-A peripheral cohesion', 'Missense → severe IFT-A C-cap loss → moderate-severe SRTD14'],
                    ['α-helix 5 C-terminal ectodermal domain', 'aa 201–362', 'Ectodermal-cilia-specific; hair follicle / ameloblast / eccrine gland cilia; distinct protein interactions', 'Hypomorphic missense → ectodermal phenotype (CED3-like) without thoracic disease'],
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

          <Section title="IFT-A C-Cap Module: IFT122-WD10–12 + IFT43 Stabilizer" color={ACCENT5}>
            <Alert color={ACCENT5}>
              IFT43 helices 1–4 dock onto IFT122 WD10–12 C-terminal propeller face, forming the
              <strong> IFT-A peripheral C-cap</strong>. This module is required for full IFT-A peripheral cohesion.
              IFT43 helix-5 is uniquely required for <strong>ectodermal cilia</strong> — the same paradigm as
              SRTD2/CED2 (IFT122 WD10-12 hypomorphic) and SRTD5/CED1 (WDR19 hypomorphic).
            </Alert>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT5 + '22' }}>
                  <tr><th>Component</th><th>Gene / SRTD</th><th>Interaction</th><th>Consequence if Lost</th></tr>
                </thead>
                <tbody>
                  {[
                    ['IFT122 WD10–12 face', 'IFT122 / SRTD2', 'C-terminal propeller face; primary IFT43 docking site', 'WD10-12 hypomorphic → CED2 (ectodermal); full LOF → SRTD2 (thoracic + skeletal)'],
                    ['IFT43 α-helix 1–4', 'IFT43 / SRTD14 (THIS GENE)', 'Docks IFT122-WD10-12; forms C-cap peripheral stabilizer module', 'LOF → IFT-A C-cap weakened → short stubby cilia → SRTD14'],
                    ['IFT43 α-helix 5', 'IFT43 / SRTD14 (THIS GENE)', 'Ectodermal-cilia-specific domain; hair follicle / ameloblast interactions', 'Hypomorphic → CED3-like (sparse hair, hypodontia) without thoracic disease'],
                  ].map(([n, s, ifc, c], i) => (
                    <tr key={i} style={s?.includes('THIS') ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                      <td>{n}</td><td>{s}</td><td>{ifc}</td><td>{c}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="CED3-like vs SRTD14 — Allele-Phenotype Spectrum (IFT43)" color={ACCENT8}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT8 + '22' }}>
                  <tr><th>Allele Class</th><th>Domain</th><th>Phenotype</th><th>Comparison</th></tr>
                </thead>
                <tbody>
                  {[
                    ['Biallelic null (truncating)', 'Any → complete LOF', 'SRPS spectrum — perinatal lethal; absent or minimal cilia', 'Same as SRTD2/5 null alleles — SRPS spectrum'],
                    ['Biallelic helix 1–4 missense', 'IFT122-WD10/11/12 docking', 'Full SRTD14 — narrow thorax, polydactyly (28%), short stubby cilia', 'Milder polydactyly than SRTD2/5 (peripheral vs core LOF)'],
                    ['Hypomorphic helix-5 missense', 'Ectodermal-cilia-specific domain (aa 201–362)', 'CED3-like — sparse hair, hypodontia, brachydactyly; absent/mild thorax', 'Same CED paradigm as SRTD2/CED2 (IFT122 WD10-12) and SRTD5/CED1 (WDR19)'],
                  ].map(([a, d, ph, comp], i) => (
                    <tr key={i} style={ph.includes('CED3') ? { background: ACCENT8 + '18' } : ph.includes('SRPS') ? { background: ACCENT7 + '18' } : {}}>
                      <td className="fw-bold">{a}</td><td>{d}</td><td>{ph}</td><td>{comp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="CED Allele Comparison — CED1 / CED2 / CED3-like" color={ACCENT8}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT8 + '22' }}>
                  <tr><th>CED Type</th><th>Gene / SRTD</th><th>Hypomorphic Domain</th><th>Ectodermal Features</th></tr>
                </thead>
                <tbody>
                  {[
                    ['CED1', 'WDR19 / SRTD5',    'WDR19 C-terminal hypomorphic alleles',     'Sparse hair · hypodontia · brachydactyly · NPHP13 alleles possible'],
                    ['CED2', 'IFT122 / SRTD2',    'IFT122 WD10–12 hypomorphic (IFT43-face)', 'Sparse hair · hypodontia · narrow forehead · brachydactyly · nail dysplasia'],
                    ['CED3-like', 'IFT43 / SRTD14 (THIS)', 'IFT43 α-helix-5 C-terminal hypomorphic', 'Sparse/fine hair · hypodontia · nail dysplasia · brachydactyly (mild)'],
                  ].map(([ced, gene, dom, feat], i) => (
                    <tr key={i} style={gene?.includes('THIS') ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                      <td className="fw-bold" style={{ color: ACCENT8 }}>{ced}</td>
                      <td>{gene}</td><td>{dom}</td><td>{feat}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Mechanism Glossary" color={ACCENT6}>
            {defs.mechanism_glossary?.map((g, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: ACCENT6 + '0a', border: `1px solid ${ACCENT6}33` }}>
                <div className="fw-bold small" style={{ color: ACCENT6 }}>{g.term}</div>
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
              <Section title="Gene Card — IFT43 / C14orf179" color={ACCENT}>
                {defs.gene_card && Object.entries(defs.gene_card).map(([k, v]) => (
                  <div key={k} className="d-flex gap-2 mb-1 small">
                    <span className="fw-bold text-nowrap" style={{ color: ACCENT, minWidth: 140 }}>{k.replace(/_/g, ' ')}:</span>
                    <span>{v}</span>
                  </div>
                ))}
              </Section>
              <Section title="Disease Card — SRTD14 / CED3-like" color={ACCENT2}>
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
                    <div className="small"><span className="badge" style={{ background: ACCENT8, fontSize: '0.7rem' }}>{v.ethnicity}</span></div>
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
              <Section title="Diagnostic Workup — SRTD14 / CED3-like" color={ACCENT3}>
                <ol className="small ps-3">
                  {defs.diagnostic_workup?.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
                </ol>
              </Section>
              <Section title="Treatment Summary" color={ACCENT5}>
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
