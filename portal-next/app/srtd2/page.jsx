'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'IFT122 WD40 Propeller & IFT-A Assembly', 'Definitions'];

// SRTD2 colour scheme — IFT122/WDR10 / IFT-A ARM hub / WD40 propeller / CED2 dual phenotype
const ACCENT  = '#1b5e20';   // deep green — IFT-A class; WD40 propeller; structural scaffold
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax; severe; SRPS spectrum
const ACCENT3 = '#01579b';   // deep blue — renal TIN; ESRD
const ACCENT4 = '#e65100';   // burnt orange — polydactyly; postaxial dominant
const ACCENT5 = '#4a148c';   // deep purple — CED2 dual phenotype; ectodermal features
const ACCENT6 = '#006064';   // dark cyan — IFT-A complex; short stubby cilia
const ACCENT7 = '#bf360c';   // dark red-orange — perinatal lethality; SRPS
const ACCENT8 = '#33691e';   // lighter green — tripartite hub; WDR19-TTC21B-IFT122

const SEED = 409;

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

export default function SRTD2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd2/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd2/definitions`).then(r => r.json()),
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
      <div className="mb-3 p-3 rounded-3" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT6}11)`, border: `2px solid ${ACCENT}` }}>
        <div className="d-flex flex-wrap align-items-center gap-2 mb-1">
          <span style={{ fontSize: '2rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
              IFT122 Short-Rib Thoracic Dysplasia 2
            </h4>
            <div className="text-muted small">
              SRTD2 / ATD2 / Jeune Syndrome 2 &nbsp;·&nbsp;
              <strong>Gene:</strong> IFT122 / WDR10 (*606045) &nbsp;·&nbsp;
              <strong>OMIM Disease:</strong> #611263 (SRTD2) · #613610 (CED2) &nbsp;·&nbsp;
              <strong>Chr:</strong> 3q21.3-q22.1 &nbsp;·&nbsp;
              <strong>Inheritance:</strong> AR biallelic LOF &nbsp;·&nbsp;
              <strong>Cohort:</strong> N={N} (seed {SEED})
            </div>
          </div>
        </div>
        <div className="d-flex flex-wrap gap-1 mt-2">
          <Badge text="IFT-A ARM Hub — WDR19 + TTC21B Tripartite Receptor" color={ACCENT} />
          <Badge text="SHORT STUBBY Cilia — IFT-A Class (Same SRTD4/5/7/9)" color={ACCENT6} />
          <Badge text="CED2 Dual Phenotype (WD10-12 Hypomorphic)" color={ACCENT5} />
          <Badge text="~15-30 Families (2026) · Rare" color={ACCENT2} />
          <Badge text="Chr 3q21.3 · 1,190aa · 12 WD40 Repeats" color={ACCENT3} />
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
            <strong>IFT122 (SRTD2)</strong> is the IFT-A core <strong>ARM HUB</strong> — its WD40 propeller
            simultaneously receives <strong>WDR19/SRTD5</strong> (N-face, WD1–4) and
            <strong> TTC21B/SRTD4</strong> (central face, WD5–7), creating the
            <strong> tripartite WDR19–TTC21B–IFT122 node</strong>. LOF → IFT-A destabilisation →
            IFT-B import failure → <strong>SHORT STUBBY cilia (IFT-A class)</strong>.
            Hypomorphic WD10–12 alleles → <strong>CED2</strong> (cranioectodermal dysplasia without thoracic disease).
          </Alert>

          {/* KPIs */}
          <div className="row g-2 mb-3">
            <KPI label="Cohort N" value={N} color={ACCENT} />
            <KPI label="Thorax Severe" value={`${kpis.thorax_severe_n} (${kpis.thorax_severe_pct}%)`} color={ACCENT2} />
            <KPI label="Polydactyly" value={`${kpis.polydactyly_n} (${kpis.polydactyly_pct}%)`} color={ACCENT4} />
            <KPI label="Renal Involved" value={`${kpis.renal_any_n} (${kpis.renal_any_pct}%)`} color={ACCENT3} />
            <KPI label="CED2 Phenotype" value={`${kpis.ced2_n} (${kpis.ced2_pct}%)`} color={ACCENT5} />
            <KPI label="CED2-Only" value={`${kpis.ced2_only_n} (${kpis.ced2_only_pct}%)`} color={ACCENT5} />
            <KPI label="Retinal Involved" value={`${kpis.retinal_any_n} (${kpis.retinal_any_pct}%)`} color={ACCENT8} />
            <KPI label="CHF/Hepatic" value={`${kpis.hepatic_chf_n} (${kpis.hepatic_chf_pct}%)`} color={ACCENT7} />
            <KPI label="VEPTR/MAGEC" value={`${kpis.veptr_any_n} (${kpis.veptr_any_pct}%)`} color={ACCENT} />
            <KPI label="Perinatal Death" value={`${kpis.perinatal_death_n} (${kpis.perinatal_death_pct}%)`} color={ACCENT7} />
            <KPI label="Misdiagnosed" value={`${kpis.misdiagnosis_n} (${kpis.misdiagnosis_pct}%)`} color="#607d8b" />
            <KPI label="Renal Tx Done" value={kpis.transplant_done_n} color={ACCENT3} />
          </div>

          {/* Mechanism */}
          <Section title="Molecular Mechanism — IFT122 IFT-A ARM Hub" color={ACCENT}>
            <p className="small">{overview.mechanism}</p>
          </Section>

          {/* EM Distribution */}
          <Section title="Ciliary EM Distribution (IFT-A Class — Short Stubby)" color={ACCENT6}>
            {(overview.em_distribution || []).map(r => (
              <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT6} />
            ))}
            <div className="small text-muted mt-1">
              Short stubby EM is <strong>identical across SRTD2/4/5/7/9</strong> — gene panel is the ONLY differentiator.
            </div>
          </Section>

          {/* Key distinction */}
          <Section title="Key Clinical Distinction — Why SRTD2 vs Other IFT-A SRTDs" color={ACCENT5}>
            <p className="small">{overview.key_distinction}</p>
          </Section>

          {/* IFT-A subunit table */}
          <Section title="Complete IFT-A Subunit Table (All SRTD-Relevant)" color={ACCENT8}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT8 + '22' }}>
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
              '0–1 yr (neonatal)': overview.age_distribution.dx_0_1yr,
              '2–5 yr (infant)': overview.age_distribution.dx_2_5yr,
              '6–10 yr (child)': overview.age_distribution.dx_6_10yr,
              '11+ yr (teen-adult)': overview.age_distribution.dx_11_plus,
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
            <Section title="CED2 Ectodermal Features (WD10-12 Hypomorphic)" color={ACCENT5}>
              {breakdown.ced2_distribution?.map(r => <BarRow key={r.label} label={r.label} n={r.n} total={N} color={ACCENT5} />)}
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
            <Section title="Top Pathogenic Variants (IFT122 / WDR10)" color={ACCENT}>
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

      {/* ── TAB 2: IFT122 WD40 PROPELLER & IFT-A ASSEMBLY ───────────── */}
      {tab === 2 && defs && (
        <div>
          <Alert color={ACCENT}>
            <strong>IFT122 molecular architecture:</strong> 12-blade WD40 beta-propeller (~1,190 aa).
            WD1–4: WDR19/IFT144 (SRTD5) N-face contact.
            WD5–7: TTC21B/IFT139 (SRTD4) hub receptor.
            WD8–9: IFT-B cargo loading platform.
            WD10–12: IFT43 C-terminal binding (CED2 alleles here).
          </Alert>

          <Section title="IFT122 Domain Map — WD40 ARM in IFT-A Core" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '22' }}>
                  <tr><th>Domain</th><th>Region (aa)</th><th>Partner / Function</th><th>Variant Class Consequence</th></tr>
                </thead>
                <tbody>
                  {[
                    ['WD1–4 N-terminal', 'aa 1–380', 'WDR19 (IFT144/SRTD5) N-face binding surface — IFT-A platform anchor', 'Missense → WDR19 dock lost → IFT-A core destabilised → moderate-severe SRTD2'],
                    ['WD5–7 central', 'aa 381–580', 'TTC21B (IFT139/SRTD4) linker hub receptor — tripartite node', 'Missense → TTC21B dock fails → WDR19–TTC21B–IFT122 assembly collapses → severe SRTD2'],
                    ['WD8–9 bridge', 'aa 581–750', 'IFT-B cargo loading lateral surface — anterograde import bridge', 'Missense → IFT-B import blunted → short stubby cilia → moderate SRTD2'],
                    ['WD10–12 C-terminal', 'aa 751–1,190', 'IFT43 small subunit binding; ectodermal cilium function', 'Hypomorphic missense → IFT43 weakened → selective ectodermal cilia → CED2 without thoracic disease'],
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

          <Section title="IFT-A Tripartite Hub: WDR19–TTC21B–IFT122" color={ACCENT8}>
            <Alert color={ACCENT8}>
              TTC21B (SRTD4) bridges WDR19 (SRTD5) C-tail → IFT122 WD5–7 face.
              IFT122 is the <strong>HUB RECEPTOR</strong> that anchors both SRTD4 and SRTD5 protein products.
              SRTD2 (IFT122 LOF) simultaneously disrupts both interfaces — uniquely broad IFT-A failure.
            </Alert>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT8 + '22' }}>
                  <tr><th>Node</th><th>Gene / SRTD</th><th>Interface</th><th>Consequence if Lost</th></tr>
                </thead>
                <tbody>
                  {[
                    ['WDR19 (IFT144)', 'SRTD5', 'Binds IFT122 WD1–4 N-face directly', 'IFT-A platform unanchored → SRTD5 phenotype'],
                    ['TTC21B (IFT139)', 'SRTD4', 'Bridges WDR19-C-tail → IFT122 WD5–7', 'Tripartite node collapse → SRTD4 phenotype'],
                    ['IFT122 (WDR10)', 'SRTD2 (THIS GENE)', 'Hub receptor receiving both WDR19 and TTC21B', 'Both SRTD4+SRTD5 interfaces disrupted simultaneously'],
                    ['IFT43', 'Not yet SRTD', 'Stabilises IFT122 WD10–12 face', 'CED2 when IFT122 WD10–12 hypomorphic (IFT43 weakened)'],
                  ].map(([n, s, ifc, c], i) => (
                    <tr key={i} style={s?.includes('THIS') ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                      <td>{n}</td><td>{s}</td><td>{ifc}</td><td>{c}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="CED2 vs SRTD2 — Allele-Phenotype Spectrum" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT5 + '22' }}>
                  <tr><th>Allele Class</th><th>WD40 Domain</th><th>Phenotype</th><th>OMIM</th></tr>
                </thead>
                <tbody>
                  {[
                    ['Biallelic null (truncating)', 'Any → complete LOF', 'SRPS spectrum — perinatal lethal; absent cilia', '#611263 severe'],
                    ['Biallelic WD4/5–WD7/8 missense', 'WDR19 + TTC21B interfaces', 'Full SRTD2 — narrow thorax, polydactyly, short stubby cilia', '#611263 moderate'],
                    ['Hypomorphic WD10–12 missense', 'IFT43-face; ectodermal cilia only', 'CED2 — sparse hair, hypodontia, brachydactyly; mild/no thorax', '#613610'],
                  ].map(([a, d, ph, om], i) => (
                    <tr key={i} style={ph.includes('CED2') ? { background: ACCENT5 + '18' } : ph.includes('SRPS') ? { background: ACCENT7 + '18' } : {}}>
                      <td className="fw-bold">{a}</td><td>{d}</td><td>{ph}</td><td>{om}</td>
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
              <Section title="Gene Card — IFT122 / WDR10" color={ACCENT}>
                {defs.gene_card && Object.entries(defs.gene_card).map(([k, v]) => (
                  <div key={k} className="d-flex gap-2 mb-1 small">
                    <span className="fw-bold text-nowrap" style={{ color: ACCENT, minWidth: 140 }}>{k.replace(/_/g, ' ')}:</span>
                    <span>{v}</span>
                  </div>
                ))}
              </Section>
              <Section title="Disease Card — SRTD2 / CED2" color={ACCENT2}>
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
              <Section title="Diagnostic Workup — SRTD2 / CED2" color={ACCENT3}>
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
