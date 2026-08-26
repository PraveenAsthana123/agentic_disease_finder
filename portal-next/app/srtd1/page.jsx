'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'IFT80 IFT-B2 & Anterograde IFT', 'Definitions'];

// SRTD1 colour scheme — IFT80 / IFT-B2 / anterograde IFT / founding gene / Jeune ATD1
const ACCENT  = '#1b5e20';   // deep forest green — IFT-B2 anterograde; kinesin-2; ciliary growth
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory failure; severity
const ACCENT3 = '#01579b';   // deep blue — renal TIN; secondary renal disease; ESRD
const ACCENT4 = '#4a148c';   // deep purple — retinal dystrophy; rod-cone; secondary
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic; ductal plate malformation
const ACCENT6 = '#33691e';   // olive green — IFT-B2 subcomplex architecture; WD40 structure
const ACCENT7 = '#f9a825';   // amber/gold — HISTORIC: first SRTD gene 2007; founding status
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial; EVC differential

const SEED = 399;

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

function TableSimple({ rows, cols }) {
  return (
    <div className="table-responsive">
      <table className="table table-sm table-bordered small mb-0">
        <thead className="table-light">
          <tr>{cols.map(c => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{cols.map(c => <td key={c}>{r[c] ?? '—'}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function SRTD1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd1/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd1/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  const kpis = overview?.kpis || {};
  const N    = overview?.cohort_n || 40;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3">
        <div className="d-flex align-items-center gap-2 flex-wrap mb-1">
          <span style={{ fontSize: '1.6rem' }}>🧬</span>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            IFT80 Short-Rib Thoracic Dysplasia 1
          </h4>
          <Badge text="SRTD1 / ATD1" color={ACCENT} />
          <Badge text="Jeune Syndrome" color={ACCENT2} />
          <Badge text="IFT-B2 Complex" color={ACCENT6} />
          <Badge text="⭐ FOUNDING SRTD GENE (2007)" color={ACCENT7} />
        </div>
        <div className="text-muted small">
          <strong style={{ color: ACCENT }}>IFT80</strong> (*611229) · 3q25.33 · 801 aa ·
          IFT-B2 WD40 β-Propeller · OMIM Disease #208500 (ATD1) ·
          First SRTD gene identified (Beales et al., 2007, Nature Genetics) ·
          40-patient educational cohort (seed {SEED})
        </div>
      </div>

      {/* Historic Banner */}
      <Alert color={ACCENT7}>
        <span className="fw-bold">⭐ FOUNDING GENE — HISTORIC SIGNIFICANCE:</span> IFT80 (SRTD1) was the
        <strong> first gene ever identified</strong> for Jeune Asphyxiating Thoracic Dystrophy (ATD/SRTD).
        Beales et al. (2007, <em>Nature Genetics</em>) established that IFT-B2 cilia dysfunction causes
        this syndrome — launching the molecular era of SRTD classification and enabling gene-panel diagnosis
        of all subsequent SRTD subtypes (SRTD3–SRTD17 and beyond).
      </Alert>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          {/* KPIs */}
          <div className="row g-2 mb-3">
            <KPI label="Severe Thorax" value={`${kpis.thorax_severe_n} (${kpis.thorax_severe_pct}%)`} color={ACCENT2} />
            <KPI label="Polydactyly" value={`${kpis.polydactyly_n} (${kpis.polydactyly_pct}%)`} color={ACCENT8} />
            <KPI label="Renal (any)" value={`${kpis.renal_any_n} (${kpis.renal_any_pct}%)`} color={ACCENT3} />
            <KPI label="Retinal Dyst." value={`${kpis.retinal_any_n} (${kpis.retinal_any_pct}%)`} color={ACCENT4} />
            <KPI label="Hepatic CHF" value={`${kpis.hepatic_chf_n} (${kpis.hepatic_chf_pct}%)`} color={ACCENT5} />
            <KPI label="VEPTR / MAGEC" value={`${kpis.veptr_any_n} (${kpis.veptr_any_pct}%)`} color={ACCENT} />
          </div>
          <div className="row g-2 mb-4">
            <KPI label="Renal Transplant" value={kpis.transplant_done_n} color={ACCENT3} />
            <KPI label="Misdiagnosed" value={`${kpis.misdiagnosis_n} (${kpis.misdiagnosis_pct}%)`} color={ACCENT7} />
            <KPI label="Cohort N" value={N} color="#555" />
            <KPI label="Sex M / F" value={`${overview?.sex_split?.M} / ${overview?.sex_split?.F}`} color="#666" />
          </div>

          {/* Mechanism */}
          <Section title="🔬 Molecular Mechanism — IFT-B2 Anterograde IFT Failure" color={ACCENT}>
            <div className="p-3 rounded small" style={{ background: ACCENT + '10', border: `1px solid ${ACCENT}40` }}>
              {overview?.mechanism}
            </div>
          </Section>

          {/* Key Distinction */}
          <Section title="⚡ Key Clinical Distinction — SRTD1 vs Other SRTDs" color={ACCENT2}>
            <div className="p-3 rounded small" style={{ background: ACCENT2 + '10', border: `1px solid ${ACCENT2}40` }}>
              {overview?.key_distinction}
            </div>
          </Section>

          {/* Ciliary EM Comparison */}
          <Section title="🔭 Ciliary EM Finding — IFT Class Signature" color={ACCENT6}>
            <div className="row g-2">
              {[
                { label: 'SRTD1 / IFT-B2', em: 'SHORTENED cilia', gene: 'IFT80, IFT172, CLUAP1', color: ACCENT, why: 'Anterograde delivery fails → cilia cannot extend to normal length' },
                { label: 'SRTD4/5/7/9 / IFT-A', em: 'SHORT STUBBY cilia', gene: 'TTC21B, WDR19, WDR35, IFT140', color: '#7b1fa2', why: 'IFT-B cannot enter cilia base → uniform shortening + stubby appearance' },
                { label: 'SRTD3/8/11/15/17 / Dynein-2', em: 'CLUB / BULGING TIP cilia', gene: 'DYNC2H1, WDR60, WDR34, DYNC2LI1, TCTEX1D2', color: '#bf360c', why: 'Retrograde fails → IFT-B stranded at tip → club/bulge at ciliary tip' },
              ].map(r => (
                <div key={r.label} className="col-12 col-md-4">
                  <div className="card h-100 p-2 small" style={{ borderLeft: `4px solid ${r.color}` }}>
                    <div className="fw-bold" style={{ color: r.color }}>{r.label}</div>
                    <div className="fw-bold mt-1">EM: <span style={{ color: r.color }}>{r.em}</span></div>
                    <div className="text-muted">{r.gene}</div>
                    <div className="mt-1" style={{ color: '#555' }}>{r.why}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* IFT-B2 Subunit Table */}
          <Section title="🧱 IFT-B2 Subcomplex — Complete Subunit Map (SRTD genes highlighted)" color={ACCENT6}>
            <TableSimple
              cols={['subunit', 'role', 'srtd', 'omim_gene', 'chr', 'freq']}
              rows={(overview?.ift_b2_subunit_table || []).map(r => ({
                subunit: r.subunit,
                role: r.role,
                srtd: r.srtd,
                omim_gene: r.omim_gene,
                chr: r.chr,
                freq: r.freq,
              }))}
            />
            <div className="text-muted small mt-1">
              IFT-B2 = distal (tip-proximal) half of IFT-B anterograde complex. Gene panel must include all IFT-B2 SRTD genes.
            </div>
          </Section>

          {/* Age at Diagnosis */}
          <Section title="📊 Age at Diagnosis Distribution" color={ACCENT3}>
            <div className="row g-2">
              {[
                { label: '0–1 yr (neonatal)', n: overview?.age_distribution?.dx_0_1yr },
                { label: '2–5 yr (infant)',   n: overview?.age_distribution?.dx_2_5yr },
                { label: '6–10 yr (child)',   n: overview?.age_distribution?.dx_6_10yr },
                { label: '11–16 yr (teen)',   n: overview?.age_distribution?.dx_11_16yr },
              ].map(d => (
                <div key={d.label} className="col-6 col-md-3">
                  <div className="card text-center p-2 h-100">
                    <div className="fw-bold fs-4" style={{ color: ACCENT3 }}>{d.n}</div>
                    <div className="text-muted small">{d.label}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 1: DIAGNOSTIC BREAKDOWN ── */}
      {tab === 1 && breakdown && (
        <div>
          <div className="row g-3">
            {/* Thorax severity */}
            <div className="col-12 col-md-6">
              <div className="card p-3 h-100">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT2 }}>Thorax Severity</h6>
                {(breakdown.thorax_distribution || []).map(r => (
                  <Bar key={r.label} label={r.label} value={r.n} max={N} color={ACCENT2} />
                ))}
              </div>
            </div>
            {/* Polydactyly */}
            <div className="col-12 col-md-6">
              <div className="card p-3 h-100">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT8 }}>Polydactyly Distribution</h6>
                {(breakdown.polydactyly_distribution || []).map(r => (
                  <Bar key={r.label} label={r.label} value={r.n} max={N} color={ACCENT8} />
                ))}
              </div>
            </div>
            {/* Renal */}
            <div className="col-12 col-md-6">
              <div className="card p-3 h-100">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT3 }}>Renal Disease</h6>
                {(breakdown.renal_distribution || []).map(r => (
                  <Bar key={r.label} label={r.label} value={r.n} max={N} color={ACCENT3} />
                ))}
                <div className="text-muted small mt-2">No NPHP allele variant for IFT80 (unlike TTC21B/SRTD4).</div>
              </div>
            </div>
            {/* Allele class */}
            <div className="col-12 col-md-6">
              <div className="card p-3 h-100">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Allele Class Distribution</h6>
                {(breakdown.allele_class_summary || []).map(r => (
                  <Bar key={r.label} label={r.label} value={r.n} max={N} color={ACCENT} />
                ))}
              </div>
            </div>
            {/* Ethnicity */}
            <div className="col-12 col-md-6">
              <div className="card p-3 h-100">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT6 }}>Ethnicity Distribution</h6>
                {(breakdown.ethnicity_distribution || []).map(r => (
                  <Bar key={r.ethnicity} label={r.ethnicity} value={r.n} max={N} color={ACCENT6} />
                ))}
              </div>
            </div>
            {/* Presentation */}
            <div className="col-12 col-md-6">
              <div className="card p-3 h-100">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT5 }}>Presentation Mode</h6>
                {(breakdown.presentation_distribution || []).map(r => (
                  <Bar key={r.label} label={r.label} value={r.n} max={N} color={ACCENT5} />
                ))}
              </div>
            </div>
            {/* Misdiagnosis */}
            <div className="col-12 col-md-6">
              <div className="card p-3 h-100">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT7 }}>Misdiagnosis Before Gene Panel</h6>
                {(breakdown.misdiagnosis_distribution || []).map(r => (
                  <Bar key={r.label} label={r.label} value={r.n} max={N} color={ACCENT7} />
                ))}
                <div className="text-muted small mt-2">SRTD3/DYNC2H1 is the most frequent misdiagnosis — most common SRTD; gene panel resolves.</div>
              </div>
            </div>
            {/* VEPTR */}
            <div className="col-12 col-md-6">
              <div className="card p-3 h-100">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Thoracic Surgical Management</h6>
                {(breakdown.veptr_distribution || []).map(r => (
                  <Bar key={r.label} label={r.label} value={r.n} max={N} color={ACCENT} />
                ))}
              </div>
            </div>
            {/* Top variants */}
            <div className="col-12">
              <div className="card p-3">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Top Pathogenic Variants (IFT80 — IFT-B2 WD40 β-propeller)</h6>
                <ul className="list-group list-group-flush small">
                  {(breakdown.top_variants || []).map((v, i) => (
                    <li key={i} className="list-group-item d-flex justify-content-between align-items-start px-0">
                      <span>{v.variant}</span>
                      <span className="badge rounded-pill ms-2" style={{ background: ACCENT }}>{v.n} pts</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: IFT-B2 MECHANISM ── */}
      {tab === 2 && overview && (
        <div>
          <Section title="🔬 IFT80 Protein Architecture — WD40 β-Propeller Domains" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-light">
                  <tr>
                    <th>Domain</th><th>Residues</th><th>Structure</th><th>Key Contacts</th><th>Pathogenic Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { d: 'N-terminal WD40 β-propeller', r: 'aa 1–400', s: '7-blade WD40 propeller', c: 'IFT38/CLUAP1 (SRTD13); IFT88 (partial)', p: 'Most pathogenic — blades 3–6 hotspot; disrupts IFT-B2 assembly' },
                    { d: 'Central linker / hinge', r: 'aa 401–550', s: 'Flexible interdomain hinge', c: 'IFT-B2 internal; partial IFT88 bridge', p: 'Moderate — missense → partial IFT-B2 disruption; viable phenotype' },
                    { d: 'C-terminal extension', r: 'aa 551–801', s: 'Extended C-terminal domain', c: 'IFT88 (C-terminal stabilisation); IFT-B2 anchoring', p: 'Hypomorphic — missense → mild SRTD1; partial IFT-B2 maintained' },
                  ].map(r => (
                    <tr key={r.d}>
                      <td className="fw-bold" style={{ color: ACCENT }}>{r.d}</td>
                      <td>{r.r}</td><td>{r.s}</td><td>{r.c}</td><td>{r.p}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="🔄 IFT-B Anterograde Train Assembly — IFT80's Role" color={ACCENT6}>
            <div className="row g-2">
              {[
                { step: '1', label: 'Kinesin-2 (KIF3A/B/KAP) loads at transition zone', color: ACCENT6 },
                { step: '2', label: 'IFT-B1 (core) assembles: IFT52, IFT46, IFT88, IFT70, IFT22', color: ACCENT6 },
                { step: '3', label: 'IFT-B2 assembles: IFT80 + IFT88 + IFT38/CLUAP1 + IFT57 + IFT172 + IFT25 + IFT27', color: ACCENT },
                { step: '4', label: 'Full IFT-B train (B1 + B2) departs cilia base toward tip', color: ACCENT6 },
                { step: '5', label: 'IFT-B2 carries Hedgehog components (SMO, Gli2/3, Patched) to ciliary tip', color: ACCENT },
                { step: '6', label: 'Gli2/Gli3 processed at tip → GliA (activator) returns to nucleus', color: ACCENT6 },
                { step: '7', label: '❌ IFT80 loss: IFT-B2 fails → truncated IFT-B train → cilia shortened → Hedgehog fails', color: ACCENT2 },
              ].map(s => (
                <div key={s.step} className="col-12">
                  <div className="d-flex align-items-center gap-2 p-2 rounded small" style={{ background: s.color + '12', border: `1px solid ${s.color}30` }}>
                    <span className="badge rounded-circle" style={{ background: s.color, minWidth: 24, height: 24 }}>{s.step}</span>
                    <span style={{ color: s.step === '7' ? ACCENT2 : '#333' }}>{s.label}</span>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="📊 IFT-B2 vs IFT-A vs Dynein-2 — Mechanism Comparison" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-light">
                  <tr><th>Complex</th><th>Direction</th><th>Motor</th><th>Primary Failure</th><th>EM Finding</th><th>SRTD Genes</th></tr>
                </thead>
                <tbody>
                  {[
                    { c: 'IFT-B2 (anterograde distal)', dir: 'Base → Tip (anterograde)', motor: 'Kinesin-2 (KIF3A/B)', fail: 'Cilia cannot extend; IFT-B2 cargo undelivered to tip', em: 'SHORTENED cilia', genes: 'SRTD1 (IFT80), SRTD10 (IFT172), SRTD13 (CLUAP1)' },
                    { c: 'IFT-A (retrograde adaptor)', dir: 'Tip → Base (retrograde adaptor)', motor: 'Dynein-2 (retrograde)', fail: 'IFT-B cannot enter cilia at base; cilia cannot grow', em: 'SHORT STUBBY cilia', genes: 'SRTD4 (TTC21B), SRTD5 (WDR19), SRTD7 (WDR35), SRTD9 (IFT140)' },
                    { c: 'Dynein-2 (retrograde motor)', dir: 'Tip → Base (retrograde)', motor: 'Dynein-2 (DYNC2H1/etc)', fail: 'IFT-B stranded at tip; retrograde return blocked', em: 'CLUB / BULGING TIP', genes: 'SRTD3 (DYNC2H1), SRTD8 (WDR60), SRTD11 (WDR34), SRTD15 (DYNC2LI1), SRTD17 (TCTEX1D2)' },
                  ].map(r => (
                    <tr key={r.c}>
                      <td className="fw-bold">{r.c}</td>
                      <td>{r.dir}</td><td>{r.motor}</td><td>{r.fail}</td>
                      <td className="fw-bold" style={{ color: r.c.includes('IFT-B2') ? ACCENT : r.c.includes('IFT-A') ? '#7b1fa2' : '#bf360c' }}>{r.em}</td>
                      <td className="small">{r.genes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="📅 Historic Timeline — IFT80 and the SRTD Molecular Era" color={ACCENT7}>
            <div className="row g-2">
              {[
                { yr: '1955', event: 'Jeune ATD first described clinically — phenotypic entity established (Jeune et al., Arch Fr Pédiatr)' },
                { yr: '2007', event: '⭐ IFT80 identified as FIRST SRTD gene — Beales et al., Nature Genetics; IFT-B2 ciliopathy basis established', highlight: true },
                { yr: '2009', event: 'DYNC2H1 (SRTD3) identified — most common SRTD gene (~50%); dynein-2 class established' },
                { yr: '2012', event: 'TTC21B (SRTD4), WDR19 (SRTD5) identified — IFT-A class SRTDs recognised' },
                { yr: '2013', event: 'WDR60 (SRTD8), IFT140 (SRTD9) — dynein-2 intermediate chain and IFT-A scaffold SRTDs' },
                { yr: '2015', event: 'WDR34 (SRTD11), IFT172 (SRTD10) — dynein-2 and IFT-B2 SRTDs; IFT-B2 second gene after IFT80' },
                { yr: '2016', event: 'DYNC2LI1 (SRTD15), TCTEX1D2 (SRTD17) — rare dynein-2 subunit SRTDs; CLUAP1 (SRTD13)' },
                { yr: '2026', event: 'Gene panel diagnosis standard of care; ≥17 SRTD genes; IFT80/SRTD1 is the rarest confirmed subtype' },
              ].map(r => (
                <div key={r.yr} className="col-12">
                  <div className="d-flex gap-3 p-2 rounded small" style={{ background: r.highlight ? ACCENT7 + '20' : '#f8f8f8', border: r.highlight ? `2px solid ${ACCENT7}` : '1px solid #eee' }}>
                    <span className="fw-bold" style={{ color: ACCENT7, minWidth: 40 }}>{r.yr}</span>
                    <span style={{ color: r.highlight ? '#555' : '#666' }}>{r.event}</span>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="⚕️ Treatment Pathway — IFT80 / SRTD1" color={ACCENT}>
            <div className="row g-2">
              {[
                { title: 'Narrow Thorax (Primary)', text: 'VEPTR or MAGEC growing rods — first-line surgical; serial expansion ×2/yr; same as all SRTDs', color: ACCENT2 },
                { title: 'Respiratory', text: 'Neonatal mechanical ventilation (null alleles); CPAP/NIV (moderate); wean as thorax expands', color: ACCENT3 },
                { title: 'Renal', text: 'Annual GFR/creatinine/USS; ACEi/ARB for proteinuria; renal transplant CURATIVE (cell-autonomous); NO NPHP variant counselling needed', color: ACCENT3 },
                { title: 'Retinal', text: 'Annual ERG from age 6; ophthalmology; low vision; no disease-modifying therapy 2026', color: ACCENT4 },
                { title: 'Hepatic', text: 'Annual APRI + USS if CHF suspected; hepatology; avoid hepatotoxic drugs', color: ACCENT5 },
                { title: 'Genetics', text: 'Cascade testing; 25% recurrence AR; prenatal/PGT; NO Joubert/NPHP allele counselling (unlike SRTD4)', color: ACCENT },
              ].map(r => (
                <div key={r.title} className="col-12 col-md-6">
                  <div className="card p-2 h-100 small" style={{ borderLeft: `3px solid ${r.color}` }}>
                    <div className="fw-bold" style={{ color: r.color }}>{r.title}</div>
                    <div>{r.text}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {tab === 3 && definitions && (
        <div>
          {/* Gene Card */}
          <Section title="🧬 Gene Card — IFT80 (*611229)" color={ACCENT}>
            <div className="row g-2 mb-2">
              {Object.entries(definitions.gene_card || {}).map(([k, v]) => (
                <div key={k} className="col-12 col-md-6">
                  <div className="card p-2 h-100 small">
                    <span className="text-muted text-uppercase" style={{ fontSize: '0.65em', letterSpacing: 1 }}>
                      {k.replace(/_/g, ' ')}
                    </span>
                    <div className="fw-semibold">{v}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Disease Card */}
          <Section title="🏥 Disease Card — SRTD1 / ATD1 (#208500)" color={ACCENT2}>
            <div className="row g-2">
              {Object.entries(definitions.disease_card || {}).map(([k, v]) => (
                <div key={k} className="col-12 col-md-6">
                  <div className="card p-2 h-100 small">
                    <span className="text-muted text-uppercase" style={{ fontSize: '0.65em', letterSpacing: 1 }}>
                      {k.replace(/_/g, ' ')}
                    </span>
                    <div className="fw-semibold">{v}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Diagnostic Workup */}
          <Section title="🩺 Diagnostic Workup — Step-by-Step" color={ACCENT6}>
            <ol className="list-group list-group-numbered small">
              {(definitions.diagnostic_workup || []).map((step, i) => (
                <li key={i} className="list-group-item">{step}</li>
              ))}
            </ol>
          </Section>

          {/* Mechanism Glossary */}
          <Section title="📖 Mechanism Glossary" color={ACCENT3}>
            <div className="accordion" id="glossaryAcc">
              {(definitions.mechanism_glossary || []).map((g, i) => (
                <div key={i} className="accordion-item small">
                  <h2 className="accordion-header">
                    <button className="accordion-button collapsed py-2" type="button" data-bs-toggle="collapse" data-bs-target={`#gloss-${i}`}>
                      <strong style={{ color: ACCENT3 }}>{g.term}</strong>
                    </button>
                  </h2>
                  <div id={`gloss-${i}`} className="accordion-collapse collapse" data-bs-parent="#glossaryAcc">
                    <div className="accordion-body py-2">{g.definition}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Key Variants */}
          <Section title="🧪 Key Pathogenic Variants (IFT80)" color={ACCENT}>
            <TableSimple
              cols={['variant', 'domain', 'consequence', 'ethnicity']}
              rows={definitions.key_variants || []}
            />
          </Section>

          {/* Treatment Summary */}
          <Section title="💊 Treatment Summary" color={ACCENT2}>
            <ol className="list-group list-group-numbered small">
              {(definitions.treatment_summary || []).map((t, i) => (
                <li key={i} className="list-group-item">{t}</li>
              ))}
            </ol>
          </Section>

          {/* DDx Table */}
          <Section title="🔀 Differential Diagnosis Table" color={ACCENT7}>
            <TableSimple
              cols={['disease', 'key_difference']}
              rows={definitions.ddx_table || []}
            />
          </Section>
        </div>
      )}

      {/* Footer */}
      <div className="mt-4 pt-3 border-top text-muted small d-flex justify-content-between flex-wrap gap-1">
        <span>
          SRTD1 / ATD1 — IFT80 (*611229) — 3q25.33 — 801 aa — IFT-B2 WD40 β-Propeller · OMIM #208500 ·
          Founding SRTD gene (Beales et al., 2007) · Educational 40-patient cohort (seed {SEED})
        </span>
        <Link href="/" className="text-decoration-none" style={{ color: ACCENT }}>← Portal Home</Link>
      </div>
    </div>
  );
}
