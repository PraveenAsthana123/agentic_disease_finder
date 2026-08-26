'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'IFT172 IFT-B2 & Tip-Anchoring', 'Definitions'];

// SRTD10 colour scheme — IFT172 / IFT-B2 / LARGEST subunit / kinesin-2 tip-anchor / Jeune ATD10
const ACCENT  = '#1a237e';   // deep indigo-navy — IFT-B2 anterograde; largest subunit; tip-anchor
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory failure; severity
const ACCENT3 = '#01579b';   // deep blue — renal TIN; secondary renal disease; ESRD
const ACCENT4 = '#4a148c';   // deep purple — retinal dystrophy; rod-cone; BBS-like phenotype
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic; ductal plate malformation
const ACCENT6 = '#004d40';   // deep teal — IFT-B2 subcomplex architecture; WD40+TPR structure
const ACCENT7 = '#f9a825';   // amber/gold — dual phenotype note; BBS-like retinal; discovery 2015
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial; EVC differential

const SEED = 401;

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

export default function SRTD10Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd10/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd10/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd10/definitions`).then(r => r.json()),
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
            IFT172 Short-Rib Thoracic Dysplasia 10
          </h4>
          <Badge text="SRTD10 / ATD10" color={ACCENT} />
          <Badge text="Jeune Syndrome" color={ACCENT2} />
          <Badge text="IFT-B2 Complex" color={ACCENT6} />
          <Badge text="LARGEST IFT-B2 Subunit" color={ACCENT7} />
        </div>
        <div className="text-muted small">
          <strong style={{ color: ACCENT }}>IFT172</strong> (*607386) · 2p23.3 · 1749 aa ·
          IFT-B2 WD40+TPR · OMIM Disease #615490 (ATD10) ·
          Identified 2015 (Halbritter et al. / Bujakowska et al.) ·
          40-patient educational cohort (seed {SEED})
        </div>
      </div>

      {/* Dual Phenotype Banner */}
      <Alert color={ACCENT7}>
        <span className="fw-bold">⚠️ DUAL PHENOTYPE — SRTD10 + BBS-like Retinal:</span> IFT172 alleles can cause
        either <strong>full SRTD10</strong> (biallelic LOF — narrow thorax, skeletal + renal + retinal) OR
        a <strong>retinal-predominant BBS-like phenotype</strong> (hypomorphic C-terminal TPR alleles — rod-cone
        dystrophy without full thoracic skeleton). IFT172 is now included on BBS gene panels. Gene panel
        is mandatory to distinguish from SRTD1 (IFT80) and other IFT-B2 SRTDs.
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
          <Section title="🔬 Molecular Mechanism — IFT-B2 Tip-Anchoring Failure (IFT172)" color={ACCENT}>
            <div className="p-3 rounded small" style={{ background: ACCENT + '10', border: `1px solid ${ACCENT}40` }}>
              {overview?.mechanism}
            </div>
          </Section>

          {/* Key Distinction */}
          <Section title="⚡ Key Clinical Distinction — SRTD10 vs Other SRTDs" color={ACCENT2}>
            <div className="p-3 rounded small" style={{ background: ACCENT2 + '10', border: `1px solid ${ACCENT2}40` }}>
              {overview?.key_distinction}
            </div>
          </Section>

          {/* Ciliary EM Comparison */}
          <Section title="🔭 Ciliary EM Finding — IFT Class Signature" color={ACCENT6}>
            <div className="row g-2">
              {[
                { label: 'SRTD10 / IFT-B2', em: 'SHORTENED cilia', gene: 'IFT172, IFT80 (SRTD1), CLUAP1 (SRTD13)', color: ACCENT, why: 'Anterograde delivery fails → cilia cannot extend; tip-anchoring lost (IFT172 TPR domain)' },
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
              IFT-B2 = distal (tip-proximal) half of IFT-B anterograde complex. IFT172 is the LARGEST subunit. Gene panel must include all IFT-B2 SRTD genes.
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
                <div className="text-muted small mt-2">No NPHP allele variant for IFT172. All renal disease is secondary SRTD10 skeletal phenotype.</div>
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
                <div className="text-muted small mt-2">SRTD1/IFT80 is a notable misdiagnosis (same IFT-B2 class; shortened cilia — gene panel resolves).</div>
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
                <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Top Pathogenic Variants (IFT172 — IFT-B2 WD40+TPR)</h6>
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

      {/* ── TAB 2: IFT-B2 TIP-ANCHORING MECHANISM ── */}
      {tab === 2 && overview && (
        <div>
          <Section title="🔬 IFT172 Protein Architecture — WD40 + TPR Dual-Domain Scaffold" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-light">
                  <tr>
                    <th>Domain</th><th>Residues</th><th>Structure</th><th>Key Contacts</th><th>Pathogenic Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { d: 'N-terminal WD40 multi-repeat', r: 'aa 1–980', s: 'Large multi-blade WD40 propeller', c: 'IFT80 (SRTD1); IFT88; IFT57 within IFT-B2', p: 'Most pathogenic hotspot: blades 6–12 (aa 400–800); IFT-B2 assembly disrupted; moderate SRTD10' },
                    { d: 'WD40-TPR hinge / linker', r: 'aa 981–1100', s: 'Flexible interdomain hinge', c: 'Bridges WD40 scaffold to TPR; partial kinesin-2 contact', p: 'Severe-moderate — hinge missense disrupts kinesin-2 tip-anchoring; viable but significant phenotype' },
                    { d: 'C-terminal TPR domain', r: 'aa 1101–1749', s: 'Multiple TPR repeats', c: 'Kinesin-2/KIF3A (tip-anchoring at anterograde end of IFT train)', p: 'Hypomorphic — mild SRTD10; BBS-like retinal phenotype prominent; partial kinesin-2 tip-docking maintained' },
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

          <Section title="🔄 IFT172 Tip-Anchoring Role in Anterograde IFT" color={ACCENT6}>
            <div className="row g-2">
              {[
                { step: '1', label: 'Kinesin-2 (KIF3A/KIF3B/KAP) loads IFT-B1 at the transition zone (ciliary base)', color: ACCENT6 },
                { step: '2', label: 'IFT-B2 assembles: IFT172 WD40 contacts IFT80 (SRTD1), IFT88, IFT57 → IFT-B2 scaffold forms', color: ACCENT6 },
                { step: '3', label: 'IFT172 C-terminal TPR domain docks kinesin-2 (KIF3A) at the distal TIP of the IFT train', color: ACCENT },
                { step: '4', label: 'Full IFT-B train (B1 + B2) with IFT172 tip-anchor departs ciliary base toward tip', color: ACCENT6 },
                { step: '5', label: 'IFT-B2 (with IFT172 tip-anchor) carries Hedgehog components (SMO, Gli2/3, Patched) to tip', color: ACCENT },
                { step: '6', label: 'Gli2/Gli3 processed at tip → GliA (activator) returns to nucleus via dynein-2 retrograde', color: ACCENT6 },
                { step: '7', label: '❌ IFT172 loss: IFT-B2 tip-anchor gone → anterograde IFT truncated → cilia shortened → Hedgehog fails', color: ACCENT2 },
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
                    { c: 'IFT-B2 (anterograde distal)', dir: 'Base → Tip (anterograde)', motor: 'Kinesin-2 (KIF3A/B)', fail: 'Cilia cannot extend; IFT-B2 cargo undelivered to tip', em: 'SHORTENED cilia', genes: 'SRTD10 (IFT172), SRTD1 (IFT80), SRTD13 (CLUAP1)' },
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

          <Section title="👁️ IFT172 Dual Phenotype — SRTD10 vs BBS-like Retinal" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-light">
                  <tr><th>Phenotype</th><th>Allele Type</th><th>Thorax</th><th>Retinal</th><th>Renal</th><th>Key Point</th></tr>
                </thead>
                <tbody>
                  {[
                    { ph: 'Full SRTD10 (ATD10)', al: 'Biallelic LOF (null+null or null+severe missense)', th: 'Narrow thorax — PRESENT (primary)', ret: 'Rod-cone ~10–15%', ren: 'TIN/ESRD ~20–30%', key: 'Classic SRTD presentation; ≥1 severe/null allele required' },
                    { ph: 'Moderate SRTD10', al: 'Compound het missense + truncating, or homozygous WD40 missense', th: 'Narrow thorax present', ret: 'Rod-cone ~12–18%', ren: 'TIN ~15–25%', key: 'Most common presentation; WD40 blade 6–12 hotspot' },
                    { ph: 'Mild SRTD10 (hypomorphic)', al: 'Hypomorphic C-terminal TPR missense (biallelic)', th: 'Mild thoracic narrowing', ret: 'Rod-cone PROMINENT (~25–35%)', ren: 'TIN ~10–15%', key: 'TPR domain partial kinesin-2 contact; retinal more sensitive than skeleton' },
                    { ph: 'BBS-like retinal only', al: 'Hypomorphic C-terminal TPR (specific alleles)', th: 'ABSENT or minimal', ret: 'Severe rod-cone dystrophy', ren: 'Mild/absent', key: 'Photoreceptor connecting cilium most sensitive to IFT172 partial loss; include IFT172 on BBS panels' },
                  ].map(r => (
                    <tr key={r.ph}>
                      <td className="fw-bold" style={{ color: ACCENT7 }}>{r.ph}</td>
                      <td>{r.al}</td><td>{r.th}</td><td>{r.ret}</td><td>{r.ren}</td>
                      <td className="small" style={{ color: '#555' }}>{r.key}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="⚕️ Treatment Pathway — IFT172 / SRTD10" color={ACCENT}>
            <div className="row g-2">
              {[
                { title: 'Narrow Thorax (Primary)', text: 'VEPTR or MAGEC growing rods — first-line surgical; serial expansion ×2/yr; same as all SRTDs', color: ACCENT2 },
                { title: 'Respiratory', text: 'Neonatal mechanical ventilation (null alleles); CPAP/NIV (moderate); wean as thorax expands', color: ACCENT3 },
                { title: 'Renal', text: 'Annual GFR/creatinine/USS; ACEi/ARB for proteinuria; renal transplant CURATIVE (cell-autonomous); NO NPHP variant counselling', color: ACCENT3 },
                { title: 'Retinal (prominent)', text: 'Annual ERG from age 6; ophthalmology mandatory — higher retinal risk than SRTD1; hypomorphic alleles → BBS-like retinal; no disease-modifying therapy 2026', color: ACCENT4 },
                { title: 'Hepatic', text: 'Annual APRI + USS if CHF suspected; hepatology; avoid hepatotoxic drugs', color: ACCENT5 },
                { title: 'Genetics', text: 'Cascade testing; 25% recurrence AR; prenatal/PGT; COUNSEL re: retinal-allele severity spectrum — C-terminal TPR alleles → isolated retinal phenotype possible', color: ACCENT },
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
          <Section title="🧬 Gene Card — IFT172 (*607386)" color={ACCENT}>
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
          <Section title="🏥 Disease Card — SRTD10 / ATD10 (#615490)" color={ACCENT2}>
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
          <Section title="🧪 Key Pathogenic Variants (IFT172)" color={ACCENT}>
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
          SRTD10 / ATD10 — IFT172 (*607386) — 2p23.3 — 1749 aa — IFT-B2 LARGEST Subunit (WD40+TPR) · OMIM #615490 ·
          Halbritter et al. / Bujakowska et al. 2015 · Educational 40-patient cohort (seed {SEED})
        </span>
        <Link href="/" className="text-decoration-none" style={{ color: ACCENT }}>← Portal Home</Link>
      </div>
    </div>
  );
}
