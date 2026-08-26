'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'CLUAP1 IFT-B2 Hub & Linchpin', 'Definitions'];

// SRTD13 colour scheme — CLUAP1/IFT38 / IFT-B2 hub / IFT-B1/B2 linchpin / RAREST IFT-B2 SRTD / Jeune ATD13
const ACCENT  = '#006064';   // deep cyan-teal — IFT-B2 hub; linchpin; rarest IFT-B2 SRTD
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory failure; severity
const ACCENT3 = '#01579b';   // deep blue — renal TIN; secondary renal disease; ESRD
const ACCENT4 = '#4a148c';   // deep purple — retinal dystrophy; rod-cone; secondary
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic; ductal plate malformation
const ACCENT6 = '#00695c';   // deep teal — IFT-B2 subcomplex architecture; linchpin role
const ACCENT7 = '#e65100';   // burnt orange — under-ascertainment note; rare disease alert
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial; EVC differential

const SEED = 403;

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

export default function SRTD13Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd13/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd13/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd13/definitions`).then(r => r.json()),
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
            CLUAP1 Short-Rib Thoracic Dysplasia 13
          </h4>
          <Badge text="SRTD13 / ATD13" color={ACCENT} />
          <Badge text="Jeune Syndrome" color={ACCENT2} />
          <Badge text="IFT-B2 Complex" color={ACCENT6} />
          <Badge text="IFT-B2 Hub / Linchpin" color={ACCENT7} />
          <Badge text="RAREST IFT-B2 SRTD" color={ACCENT2} />
        </div>
        <div className="text-muted small">
          <strong style={{ color: ACCENT }}>CLUAP1/IFT38</strong> (*616470) · 16q23.1 · 435 aa ·
          IFT-B2 Hub / IFT-B1/B2 Linchpin · OMIM Disease #616300 (ATD13) ·
          Roosing et al. 2016 (J Clin Invest) · &lt;10 families worldwide ·
          40-patient educational cohort (seed {SEED})
        </div>
      </div>

      {/* Rarity Banner */}
      <Alert color={ACCENT7}>
        <span className="fw-bold">⚠️ RAREST IFT-B2 SRTD — Under-Ascertainment Alert:</span> CLUAP1 (SRTD13) is the
        rarest of the three IFT-B2 SRTD genes (&lt;10 families worldwide, 2016–2026). It was{' '}
        <strong>absent from pre-2016 SRTD gene panels</strong>, meaning SRTD13 may be
        under-ascertained. Ensure current panels include CLUAP1/IFT38. Re-testing gene-panel-negative
        SRTD index cases from the pre-2016 era is recommended.
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
          <Section title="🔬 Molecular Mechanism — IFT-B2 Hub & IFT-B1/B2 Linchpin Failure (CLUAP1)" color={ACCENT}>
            <div className="p-3 rounded small" style={{ background: ACCENT + '10', border: `1px solid ${ACCENT}40` }}>
              {overview?.mechanism}
            </div>
          </Section>

          {/* Key Distinction */}
          <Section title="⚡ Key Clinical Distinction — SRTD13 vs Other SRTDs" color={ACCENT2}>
            <div className="p-3 rounded small" style={{ background: ACCENT2 + '10', border: `1px solid ${ACCENT2}40` }}>
              {overview?.key_distinction}
            </div>
          </Section>

          {/* Ciliary EM Comparison */}
          <Section title="🔭 Ciliary EM Finding — IFT Class Signature" color={ACCENT6}>
            <div className="row g-2">
              {[
                { label: 'SRTD13 / IFT-B2', em: 'SHORTENED cilia', gene: 'CLUAP1 (SRTD13), IFT80 (SRTD1), IFT172 (SRTD10)', color: ACCENT, why: 'Anterograde delivery fails → cilia cannot extend; IFT-B1/B2 linchpin lost (CLUAP1); IFT-B2 hub disrupted' },
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
              IFT-B2 = distal (tip-proximal) half of IFT-B anterograde complex. CLUAP1/IFT38 is the IFT-B2 hub and IFT-B1/B2 linchpin. Gene panel must include all IFT-B2 SRTD genes.
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
                <div className="text-muted small mt-2">No NPHP allele variant for CLUAP1. All renal disease is secondary SRTD13 skeletal phenotype.</div>
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
                <div className="text-muted small mt-2">MENA consanguinity high — homozygous variants common. Under-ascertainment in non-consanguineous populations.</div>
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
                <div className="text-muted small mt-2">SRTD1/IFT80 is the most common misdiagnosis (direct CLUAP1 binding partner; same IFT-B2 class; shortened cilia — gene panel resolves).</div>
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
                <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Top Pathogenic Variants (CLUAP1/IFT38 — IFT-B2 hub)</h6>
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

      {/* ── TAB 2: CLUAP1 IFT-B2 HUB & LINCHPIN MECHANISM ── */}
      {tab === 2 && overview && (
        <div>
          <Section title="🔬 CLUAP1/IFT38 Protein Architecture — IFT-B2 Hub / IFT-B1/B2 Linchpin" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-light">
                  <tr>
                    <th>Domain</th><th>Residues</th><th>Structure</th><th>Key Contacts</th><th>Pathogenic Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { d: 'N-terminal IFT80-contact surface', r: 'aa 1–150', s: 'IFT80 N-terminal WD40 binding region', c: 'IFT80 (SRTD1) N-terminal WD40 domain — direct and critical', p: 'Pathogenic hotspot: missense → IFT80-CLUAP1 interface lost; IFT-B2 hub destabilised; moderate SRTD13' },
                    { d: 'Central linchpin region', r: 'aa 151–280', s: 'IFT-B1/IFT-B2 bridge via IFT52', c: 'IFT52 (IFT-B1 adaptor) — bridges IFT-B1 to IFT-B2', p: 'Severe: linchpin missense → IFT-B1 + IFT-B2 uncoupling; entire anterograde IFT train disrupted; severe-moderate SRTD13' },
                    { d: 'C-terminal extension', r: 'aa 281–435', s: 'IFT57 and IFT88 contact surface', c: 'IFT57 (coiled-coil; IFT-B2) + IFT88 (structural scaffold; IFT-B2)', p: 'Hypomorphic — mild SRTD13; partial IFT57/IFT88 contacts maintained; partial IFT-B2 assembly preserved' },
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

          <Section title="🔄 CLUAP1 IFT-B1/B2 Linchpin Role in Anterograde IFT" color={ACCENT6}>
            <div className="row g-2">
              {[
                { step: '1', label: 'Kinesin-2 (KIF3A/KIF3B/KAP) docks onto IFT-B1 at the transition zone (ciliary base)', color: ACCENT6 },
                { step: '2', label: 'CLUAP1 N-terminal contacts IFT80 (SRTD1) WD40 within IFT-B2 → IFT-B2 hub assembles', color: ACCENT6 },
                { step: '3', label: 'CLUAP1 central linchpin region bridges IFT-B1 (via IFT52) to IFT-B2 scaffold → full IFT-B train formed', color: ACCENT },
                { step: '4', label: 'CLUAP1 C-terminal contacts IFT57 and IFT88 → IFT-B2 structural integrity complete', color: ACCENT6 },
                { step: '5', label: 'Full IFT-B train (B1 + B2, linked by CLUAP1) departs ciliary base carrying Hedgehog components to tip', color: ACCENT6 },
                { step: '6', label: 'Gli2/Gli3 processed at tip → GliA (activator) returns to nucleus via dynein-2 retrograde → Hedgehog ON', color: ACCENT6 },
                { step: '7', label: '❌ CLUAP1 loss: IFT-B2 hub gone → IFT-B1/B2 uncoupled → anterograde IFT truncated → cilia shortened → Hedgehog fails', color: ACCENT2 },
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

          <Section title="📊 IFT-B2 Hub Comparison — CLUAP1 vs IFT80 vs IFT172 (all three IFT-B2 SRTDs)" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-light">
                  <tr><th>Gene (SRTD)</th><th>IFT-B2 Role</th><th>Key Contacts</th><th>Loss Effect</th><th>Unique Feature</th><th>Frequency</th></tr>
                </thead>
                <tbody>
                  {[
                    { g: 'IFT80 (SRTD1)', role: 'WD40 β-propeller scaffold', contacts: 'CLUAP1 (direct); IFT88; IFT57', loss: 'IFT-B2 scaffold destabilised; shortened cilia', unique: 'Founding SRTD gene (2007); NO dual phenotype; NO NPHP allele', freq: '~0.5–1% SRTD' },
                    { g: 'IFT172 (SRTD10)', role: 'LARGEST subunit; WD40+TPR; kinesin-2 tip-anchor', contacts: 'IFT80; IFT88; IFT57; kinesin-2 (KIF3A)', loss: 'IFT-B2 tip-anchor lost; IFT train cannot complete tip delivery; shortened cilia', unique: 'DUAL PHENOTYPE: SRTD10 + BBS-like retinal (hypomorphic alleles)', freq: '~1–2% SRTD' },
                    { g: 'CLUAP1 (SRTD13)', role: 'IFT-B2 HUB; IFT-B1/B2 LINCHPIN', contacts: 'IFT80 (direct N-terminal); IFT52 (linchpin); IFT57; IFT88', loss: 'IFT-B2 hub disrupted; IFT-B1/B2 UNCOUPLED; shortened cilia', unique: 'RAREST IFT-B2 SRTD; IFT-B1/B2 linchpin (unique function); under-ascertained (absent older panels)', freq: '<10 families (2026)' },
                  ].map(r => (
                    <tr key={r.g}>
                      <td className="fw-bold" style={{ color: ACCENT }}>{r.g}</td>
                      <td>{r.role}</td><td>{r.contacts}</td><td>{r.loss}</td>
                      <td className="small" style={{ color: '#555' }}>{r.unique}</td>
                      <td>{r.freq}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
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
                    { c: 'IFT-B2 (anterograde distal)', dir: 'Base → Tip (anterograde)', motor: 'Kinesin-2 (KIF3A/B)', fail: 'Cilia cannot extend; IFT-B2 cargo undelivered to tip', em: 'SHORTENED cilia', genes: 'SRTD13 (CLUAP1), SRTD1 (IFT80), SRTD10 (IFT172)' },
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

          <Section title="⚕️ Treatment Pathway — CLUAP1 / SRTD13" color={ACCENT}>
            <div className="row g-2">
              {[
                { title: 'Narrow Thorax (Primary)', text: 'VEPTR or MAGEC growing rods — first-line surgical; serial expansion ×2/yr; same as all SRTDs', color: ACCENT2 },
                { title: 'Respiratory', text: 'Neonatal mechanical ventilation (null alleles); CPAP/NIV (moderate); wean as thorax expands', color: ACCENT3 },
                { title: 'Renal', text: 'Annual GFR/creatinine/USS; ACEi/ARB for proteinuria; renal transplant CURATIVE (cell-autonomous); NO NPHP variant counselling', color: ACCENT3 },
                { title: 'Retinal', text: 'Annual ERG from age 6; ophthalmology; low vision support (~5–10%); standard SRTD retinal surveillance — no BBS-like allele series for CLUAP1', color: ACCENT4 },
                { title: 'Hepatic', text: 'Annual APRI + USS if CHF suspected; hepatology; avoid hepatotoxic drugs', color: ACCENT5 },
                { title: 'Panel + Genetics', text: 'Ensure updated SRTD panel includes CLUAP1/IFT38. Cascade testing; 25% AR recurrence; prenatal/PGT; re-test pre-2016 gene-panel-negative SRTD index cases', color: ACCENT },
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
          <Section title="🧬 Gene Card — CLUAP1/IFT38 (*616470)" color={ACCENT}>
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
          <Section title="🏥 Disease Card — SRTD13 / ATD13 (#616300)" color={ACCENT2}>
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
          <Section title="🧪 Key Pathogenic Variants (CLUAP1/IFT38)" color={ACCENT}>
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
          SRTD13 / ATD13 — CLUAP1/IFT38 (*616470) — 16q23.1 — 435 aa — IFT-B2 Hub / IFT-B1/B2 Linchpin · OMIM #616300 ·
          Roosing et al. 2016 · Educational 40-patient cohort (seed {SEED})
        </span>
        <Link href="/" className="text-decoration-none" style={{ color: ACCENT }}>← Portal Home</Link>
      </div>
    </div>
  );
}
