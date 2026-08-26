'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'C21orf2 LRR Scaffold & Ciliogenesis', 'Definitions'];

// SRTD12 colour scheme — C21orf2 / NEK1-IFT-A bridge / LRR / variable EM / dual phenotype
const ACCENT  = '#4527a0';   // deep violet — LRR scaffold; bridging molecular class; unique
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax; severe; perinatal lethality
const ACCENT3 = '#01579b';   // deep blue — renal TIN; ESRD
const ACCENT4 = '#1b5e20';   // deep green — retinal dystrophy; LCA/CORD; dual phenotype
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic
const ACCENT6 = '#006064';   // dark cyan — variable EM; IFT-A/basal body hybrid
const ACCENT7 = '#4a148c';   // dark purple — polydactyly
const ACCENT8 = '#880e4f';   // dark pink — dual phenotype; retinal-only alleles

const SEED = 407;

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

export default function SRTD12Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd12/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd12/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd12/definitions`).then(r => r.json()),
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
              C21orf2 Short-Rib Thoracic Dysplasia 12
            </h4>
            <div className="text-muted small">
              SRTD12 / ATD12 / Jeune Syndrome 12 &nbsp;·&nbsp;
              <strong>Gene:</strong> C21orf2 (*603503) &nbsp;·&nbsp;
              <strong>OMIM Disease:</strong> #616012 &nbsp;·&nbsp;
              <strong>Chr:</strong> 21q22.3 &nbsp;·&nbsp;
              <strong>Inheritance:</strong> AR biallelic LOF &nbsp;·&nbsp;
              <strong>Cohort:</strong> N={N} (seed {SEED})
            </div>
          </div>
        </div>
        <div className="d-flex flex-wrap gap-1 mt-2">
          <Badge text="NEK1-IFT-A Bridge — FIFTH DISTINCT SRTD CLASS" color={ACCENT} />
          <Badge text="VARIABLE EM: Short Stubby / Partially Absent" color={ACCENT6} />
          <Badge text="Dual Phenotype: SRTD12 + Retinal-only (LRR-9)" color={ACCENT8} />
          <Badge text="Ultra-Rare: <20 Families (2026)" color={ACCENT2} />
          <Badge text="Chr 21q22.3 — LRR Domain 322aa" color={ACCENT3} />
          <Badge text="AR Biallelic LOF" color={ACCENT4} />
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT}>
        <strong>🧬 FIFTH DISTINCT SRTD MOLECULAR CLASS:</strong> C21orf2 is the ONLY SRTD gene that bridges
        NEK1 (SRTD6 — basal body kinase) via LRR-1/3 AND IFT-A complex (WDR19/IFT140) via LRR-4/6.
        Dual defect → variable EM (short stubby OR partially absent) — <strong>EM alone cannot classify SRTD12; comprehensive gene panel mandatory.</strong>
      </Alert>
      <Alert color={ACCENT8}>
        <strong>🔬 DUAL PHENOTYPE (SRTD12 + Retinal-Only):</strong> Hypomorphic LRR-9/C-cap alleles (e.g. p.Ala290Val)
        → retinal dystrophy dominant (LCA/CORD-like) without significant thoracic disease.
        LCA panels often miss C21orf2 — if LCA panel negative and ciliopathy family history, add C21orf2.
      </Alert>
      <Alert color={ACCENT2}>
        <strong>⚠️ UNDER-ASCERTAINED:</strong> C21orf2 absent from pre-2016 SRTD panels; variable EM confuses
        IFT-A vs basal body classification; retinal-only cases misattributed to other LCA genes.
        <strong> Chromosome 21q22.3 ≠ trisomy 21 — Down syndrome patients have 3 copies (not causative).</strong>
      </Alert>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Severe Thorax" value={`${kpis.thorax_severe_n} (${kpis.thorax_severe_pct}%)`} color={ACCENT2} />
            <KPI label="Polydactyly" value={`${kpis.polydactyly_n} (${kpis.polydactyly_pct}%)`} color={ACCENT7} />
            <KPI label="Retinal Any" value={`${kpis.retinal_any_n} (${kpis.retinal_any_pct}%)`} color={ACCENT4} />
            <KPI label="Retinal-Only" value={`${kpis.retinal_only_n} (${kpis.retinal_only_pct}%)`} color={ACCENT8} />
            <KPI label="Renal Any" value={`${kpis.renal_any_n} (${kpis.renal_any_pct}%)`} color={ACCENT3} />
            <KPI label="CHF/Hepatic" value={`${kpis.hepatic_chf_n} (${kpis.hepatic_chf_pct}%)`} color={ACCENT5} />
            <KPI label="VEPTR/MAGEC" value={`${kpis.veptr_any_n} (${kpis.veptr_any_pct}%)`} color={ACCENT} />
            <KPI label="Perinatal Death" value={`${kpis.perinatal_death_n} (${kpis.perinatal_death_pct}%)`} color={ACCENT2} />
            <KPI label="Misdiagnosed" value={`${kpis.misdiagnosis_n} (${kpis.misdiagnosis_pct}%)`} color={ACCENT6} />
            <KPI label="Transplant" value={kpis.transplant_done_n} color={ACCENT3} />
          </div>

          <Section title="Molecular Mechanism — C21orf2 LRR Bridge" color={ACCENT}>
            <p className="small">{overview?.mechanism}</p>
          </Section>

          <Section title="Key Distinguishing Features — SRTD12 vs All Other SRTD Types" color={ACCENT6}>
            <p className="small">{overview?.key_distinction}</p>
          </Section>

          <Section title="Ciliary EM Distribution (variable — unique SRTD12 signature)" color={ACCENT6}>
            {(overview?.em_distribution || []).map((r, i) => (
              <BarRow key={i} label={r.label} n={r.n} total={N} color={ACCENT6} />
            ))}
            <div className="alert mt-2" style={{ background: ACCENT6 + '15', borderLeft: `3px solid ${ACCENT6}`, fontSize: '0.82rem' }}>
              <strong>Diagnostic note:</strong> SRTD12 is the ONLY SRTD type with variable EM (short stubby + partially absent in same cohort).
              Short stubby = IFT-A mechanism dominant (LRR-4/6 alleles); partially absent = NEK1 mechanism dominant (LRR-1/3 alleles); absent = null alleles.
              Gene panel is the ONLY definitive test — EM classification alone is insufficient for SRTD12.
            </div>
          </Section>

          <Section title="SRTD Molecular Class Comparison Table" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '22' }}>
                  <tr>
                    <th>Molecular Class</th><th>Ciliary EM</th><th>SRTD Genes</th><th>Mechanism</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview?.srtd_molecular_class_table || []).map((r, i) => (
                    <tr key={i} style={r.class.includes('SRTD12') ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                      <td>{r.class}</td><td>{r.em}</td><td>{r.genes}</td><td>{r.why}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Age at Diagnosis Distribution" color={ACCENT3}>
            {Object.entries(overview?.age_distribution || {}).map(([k, v], i) => (
              <BarRow key={i} label={k.replace(/_/g, ' ')} n={v} total={N} color={ACCENT3} />
            ))}
          </Section>

          <div className="card border-0 shadow-sm mt-2" style={{ background: ACCENT + '08' }}>
            <div className="card-body small">
              <strong style={{ color: ACCENT }}>Cohort:</strong> N={N} · Seed {SEED} ·
              Sex {overview?.sex_split?.M}M/{overview?.sex_split?.F}F ·
              <Link href="/srtd6" className="ms-1" style={{ color: ACCENT8 }}>SRTD6 (NEK1 basal body kinase) →</Link>
              <Link href="/srtd5" className="ms-1" style={{ color: ACCENT4 }}>SRTD5 (WDR19 IFT-A) →</Link>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 1: Diagnostic Breakdown ── */}
      {tab === 1 && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Thorax Severity Distribution" color={ACCENT2}>
              {breakdown.thorax_distribution?.map((r, i) => (
                <BarRow key={i} label={r.label} n={r.n} total={N} color={ACCENT2} />
              ))}
            </Section>
            <Section title="Ciliary EM Pattern (variable — SRTD12 signature)" color={ACCENT6}>
              {breakdown.em_distribution?.map((r, i) => (
                <BarRow key={i} label={r.label} n={r.n} total={N} color={ACCENT6} />
              ))}
            </Section>
            <Section title="Polydactyly Distribution" color={ACCENT7}>
              {breakdown.polydactyly_distribution?.map((r, i) => (
                <BarRow key={i} label={r.label} n={r.n} total={N} color={ACCENT7} />
              ))}
            </Section>
            <Section title="Retinal Distribution (including dual phenotype)" color={ACCENT4}>
              {breakdown.retinal_distribution?.map((r, i) => (
                <BarRow key={i} label={r.label} n={r.n} total={N} color={ACCENT4} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Renal Distribution" color={ACCENT3}>
              {breakdown.renal_distribution?.map((r, i) => (
                <BarRow key={i} label={r.label} n={r.n} total={N} color={ACCENT3} />
              ))}
            </Section>
            <Section title="Allele Class Distribution" color={ACCENT}>
              {breakdown.allele_class_summary?.map((r, i) => (
                <BarRow key={i} label={r.label} n={r.n} total={N} color={ACCENT} />
              ))}
            </Section>
            <Section title="Ethnicity Distribution" color={ACCENT6}>
              {breakdown.ethnicity_distribution?.map((r, i) => (
                <BarRow key={i} label={r.ethnicity} n={r.n} total={N} color={ACCENT6} />
              ))}
            </Section>
            <Section title="Misdiagnosis Distribution" color={ACCENT8}>
              {breakdown.misdiagnosis_distribution?.map((r, i) => (
                <BarRow key={i} label={r.label} n={r.n} total={N} color={ACCENT8} />
              ))}
            </Section>
            <Section title="Surgical / Thoracic Intervention" color={ACCENT}>
              {breakdown.veptr_distribution?.map((r, i) => (
                <BarRow key={i} label={r.label} n={r.n} total={N} color={ACCENT} />
              ))}
            </Section>
          </div>
          <div className="col-12">
            <Section title="Representative Pathogenic Variants" color={ACCENT}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead style={{ background: ACCENT + '22' }}>
                    <tr><th>Variant</th><th>n</th></tr>
                  </thead>
                  <tbody>
                    {breakdown.top_variants?.map((v, i) => (
                      <tr key={i}><td>{v.variant}</td><td>{v.n}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 2: C21orf2 LRR Scaffold & Ciliogenesis ── */}
      {tab === 2 && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="C21orf2 Gene Card" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(defs.gene_card || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-capitalize" style={{ color: ACCENT, minWidth: 140 }}>
                        {k.replace(/_/g, ' ')}
                      </td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="LRR Domain Architecture — NEK1 + IFT-A Bridge" color={ACCENT6}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead style={{ background: ACCENT6 + '22' }}>
                    <tr><th>Domain / aa</th><th>Function</th></tr>
                  </thead>
                  <tbody>
                    <tr style={{ background: ACCENT + '10' }}><td>N-cap (aa 1–36)</td><td>LRR folding scaffold; NLC motif; variants here → null-equivalent</td></tr>
                    <tr style={{ background: ACCENT + '18' }}><td>LRR 1–3 (aa 37–120)</td><td>NEK1-binding surface (SRTD6 link); missense → TTBK2 under-phosphorylated → CP110 persists</td></tr>
                    <tr style={{ background: ACCENT6 + '18' }}><td>LRR 4–6 (aa 121–210)</td><td>IFT-A docking interface (WDR19/IFT140); missense → IFT-B import failure → short stubby cilia</td></tr>
                    <tr><td>LRR 7–8 (aa 211–270)</td><td>Lateral structural scaffold; combined NEK1 + IFT-A disruption → severe SRTD12</td></tr>
                    <tr style={{ background: ACCENT8 + '18' }}><td>LRR-9 / C-cap (aa 271–322)</td><td>Photoreceptor connecting cilium function; hypomorphic here → <strong>retinal dominant (LCA/CORD)</strong> — dual phenotype</td></tr>
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Disease Card — SRTD12" color={ACCENT2}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(defs.disease_card || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-capitalize" style={{ color: ACCENT2, minWidth: 140 }}>
                        {k.replace(/_/g, ' ')}
                      </td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="Key Pathogenic Variants" color={ACCENT}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead style={{ background: ACCENT + '22' }}>
                    <tr><th>Variant</th><th>Domain</th><th>Consequence</th><th>Ethnicity</th></tr>
                  </thead>
                  <tbody>
                    {(defs.key_variants || []).map((v, i) => (
                      <tr key={i}>
                        <td className="fw-bold">{v.variant}</td>
                        <td>{v.domain}</td>
                        <td>{v.consequence}</td>
                        <td>{v.ethnicity}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Mechanism Glossary" color={ACCENT}>
              {(defs.mechanism_glossary || []).map((g, i) => (
                <div key={i} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', border: `1px solid ${ACCENT}30` }}>
                  <strong style={{ color: ACCENT }}>{g.term}</strong>
                  <p className="mb-0 small mt-1">{g.definition}</p>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Diagnostic Workup" color={ACCENT2}>
              <ol className="small ps-3">
                {(defs.diagnostic_workup || []).map((step, i) => (
                  <li key={i} className="mb-1">{step}</li>
                ))}
              </ol>
            </Section>
            <Section title="Treatment Summary" color={ACCENT3}>
              <ol className="small ps-3">
                {(defs.treatment_summary || []).map((step, i) => (
                  <li key={i} className="mb-1">{step}</li>
                ))}
              </ol>
            </Section>
            <Section title="Differential Diagnosis Table" color={ACCENT6}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead style={{ background: ACCENT6 + '22' }}>
                    <tr><th>Condition</th><th>Key Difference from SRTD12</th></tr>
                  </thead>
                  <tbody>
                    {(defs.ddx_table || []).map((r, i) => (
                      <tr key={i}>
                        <td className="fw-bold" style={{ color: ACCENT6 }}>{r.disease}</td>
                        <td>{r.key_difference}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}
    </div>
  );
}
