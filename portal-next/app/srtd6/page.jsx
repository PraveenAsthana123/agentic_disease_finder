'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'NEK1 Kinase & Ciliogenesis', 'Definitions'];

// SRTD6 colour scheme — NEK1 / basal body kinase / Majewski / absent cilia / SRTD6
const ACCENT  = '#4a148c';   // deep purple — kinase; basal body; NEK1 unique molecular class
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax; severe; perinatal lethality; Majewski SRPS II
const ACCENT3 = '#01579b';   // deep blue — renal cysts; TIN; renal disease
const ACCENT4 = '#1b5e20';   // deep green — retinal dystrophy; rod-cone
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic; laryngeal stenosis
const ACCENT6 = '#00695c';   // deep teal — hydrops fetalis; medianasal hypoplasia — UNIQUE SRTD6 features
const ACCENT7 = '#4a148c';   // deep purple — polydactyly; highest rate; postaxial + preaxial
const ACCENT8 = '#bf360c';   // deep orange-red — ciliogenesis absent; EM absent cilia class

const SEED = 405;

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

export default function SRTD6Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd6/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd6/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd6/definitions`).then(r => r.json()),
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
            NEK1 Short-Rib Thoracic Dysplasia 6
          </h4>
          <Badge text="SRTD6 / ATD6" color={ACCENT} />
          <Badge text="Majewski / SRPS II" color={ACCENT2} />
          <Badge text="Basal Body Kinase" color={ACCENT8} />
          <Badge text="Absent Cilia" color={ACCENT8} />
          <Badge text="Hydrops Unique" color={ACCENT6} />
        </div>
        <div className="text-muted small">
          <strong style={{ color: ACCENT }}>NEK1</strong> (*604588) · 4q33 · 1258 aa ·
          Basal body kinase — TTBK2 phosphorylation / CP110 removal / axoneme nucleation ·
          OMIM Disease #263520 (SRTD6/ATD6/Majewski) ·
          ~30–60 families worldwide · 40-patient educational cohort (seed {SEED})
        </div>
      </div>

      {/* Unique feature banner */}
      <Alert color={ACCENT6}>
        <span className="fw-bold">⚠️ UNIQUE SRTD6 FEATURES — Absent Cilia + Hydrops + Medianasal Hypoplasia:</span>{' '}
        NEK1 (SRTD6) is the ONLY basal-body kinase SRTD gene — a{' '}
        <strong>fourth molecular class</strong> distinct from IFT-B2 (shortened cilia), IFT-A (short stubby), and
        Dynein-2 (club cilia). Loss causes <strong>ABSENT or RUDIMENTARY cilia</strong>. Three UNIQUE features
        not seen in any other SRTD type: (1) <strong>Hydrops fetalis ~20%</strong> (lymphatic cilia NEK1-dependent);
        (2) <strong>Medianasal hypoplasia ~30%</strong> (Majewski sign); (3) <strong>Laryngeal stenosis ~15%</strong>.
        Highest polydactyly rate of all SRTDs (65–75%, postaxial + preaxial).
        NEK1 was absent from early IFT-focussed SRTD gene panels — ensure panels include NEK1.
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
            <KPI label="Polydactyly" value={`${kpis.polydactyly_n} (${kpis.polydactyly_pct}%)`} color={ACCENT7} />
            <KPI label="Hydrops Fetalis" value={`${kpis.hydrops_n} (${kpis.hydrops_pct}%)`} color={ACCENT6} />
            <KPI label="Medianasal Hypo." value={`${kpis.med_nasal_n} (${kpis.med_nasal_pct}%)`} color={ACCENT6} />
            <KPI label="Renal (any)" value={`${kpis.renal_any_n} (${kpis.renal_any_pct}%)`} color={ACCENT3} />
            <KPI label="Retinal Dyst." value={`${kpis.retinal_any_n} (${kpis.retinal_any_pct}%)`} color={ACCENT4} />
          </div>
          <div className="row g-2 mb-4">
            <KPI label="Hepatic CHF" value={`${kpis.hepatic_chf_n} (${kpis.hepatic_chf_pct}%)`} color={ACCENT5} />
            <KPI label="Laryngeal Sten." value={`${kpis.laryngeal_n} (${kpis.laryngeal_pct}%)`} color={ACCENT5} />
            <KPI label="Perinatal Death" value={`${kpis.perinatal_death_n} (${kpis.perinatal_death_pct}%)`} color={ACCENT2} />
            <KPI label="Renal Transplant" value={kpis.transplant_done_n} color={ACCENT3} />
            <KPI label="Misdiagnosed" value={`${kpis.misdiagnosis_n} (${kpis.misdiagnosis_pct}%)`} color={ACCENT7} />
            <KPI label="Cohort N" value={N} color="#555" />
          </div>

          {/* Mechanism */}
          <Section title="🔬 Molecular Mechanism — NEK1 Kinase / TTBK2 / CP110 Removal / Ciliogenesis Initiation Failure" color={ACCENT}>
            <div className="p-3 rounded small" style={{ background: ACCENT + '10', border: `1px solid ${ACCENT}40` }}>
              {overview?.mechanism}
            </div>
          </Section>

          {/* Key Distinction */}
          <Section title="⚡ Key Clinical Distinction — SRTD6 (NEK1) vs All Other SRTD Types" color={ACCENT2}>
            <div className="p-3 rounded small" style={{ background: ACCENT2 + '10', border: `1px solid ${ACCENT2}40` }}>
              {overview?.key_distinction}
            </div>
          </Section>

          {/* Ciliary EM Class Comparison */}
          <Section title="🔭 SRTD Molecular Classification — Four Ciliary EM Classes" color={ACCENT8}>
            <div className="row g-2">
              {(overview?.srtd_molecular_class_table || []).map(r => (
                <div key={r.class} className="col-12 col-md-6">
                  <div className="card h-100 p-2 small" style={{
                    borderLeft: `4px solid ${r.class.includes('Basal') ? ACCENT : r.class.includes('IFT-B2') ? '#006064' : r.class.includes('IFT-A') ? '#7b1fa2' : '#bf360c'}`
                  }}>
                    <div className="fw-bold" style={{ color: r.class.includes('Basal') ? ACCENT : r.class.includes('IFT-B2') ? '#006064' : r.class.includes('IFT-A') ? '#7b1fa2' : '#bf360c' }}>
                      {r.class}
                    </div>
                    <div className="fw-bold mt-1">EM: <span style={{ color: r.class.includes('Basal') ? ACCENT8 : undefined }}>{r.em}</span></div>
                    <div className="text-muted">{r.genes}</div>
                    <div className="mt-1" style={{ color: '#555' }}>{r.why}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Age at Diagnosis */}
          <Section title="📊 Age at Diagnosis Distribution" color={ACCENT3}>
            <div className="row g-2">
              {[
                { label: '0–1 yr (neonatal/prenatal)', n: overview?.age_distribution?.dx_0_1yr },
                { label: '2–5 yr (infant)',             n: overview?.age_distribution?.dx_2_5yr },
                { label: '6–10 yr (child)',             n: overview?.age_distribution?.dx_6_10yr },
                { label: '11–16 yr (teen)',             n: overview?.age_distribution?.dx_11_16yr },
              ].map(d => (
                <div key={d.label} className="col-6 col-md-3">
                  <div className="card text-center p-2 h-100">
                    <div className="fw-bold fs-4" style={{ color: ACCENT3 }}>{d.n}</div>
                    <div className="text-muted small">{d.label}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="text-muted small mt-2">70% neonatal/prenatal: hydrops + polydactyly + narrow thorax detected prenatally. Remaining 30%: moderate alleles with later thoracic presentation.</div>
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
                <h6 className="fw-bold mb-2" style={{ color: ACCENT7 }}>Polydactyly — Highest of All SRTD Types (65–75%)</h6>
                {(breakdown.polydactyly_distribution || []).map(r => (
                  <Bar key={r.label} label={r.label} value={r.n} max={N} color={ACCENT7} />
                ))}
                <div className="text-muted small mt-2">NEK1/SRTD6 has the highest polydactyly rate of all SRTD genes, and uniquely includes preaxial + postaxial distribution.</div>
              </div>
            </div>
            {/* SRTD6-unique features */}
            <div className="col-12 col-md-6">
              <div className="card p-3 h-100">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT6 }}>SRTD6-Unique Features (absent in all other SRTD types)</h6>
                {(breakdown.hydrops_medianasal || []).map(r => (
                  <Bar key={r.label} label={r.label} value={r.n} max={N} color={ACCENT6} />
                ))}
                <div className="text-muted small mt-2">Hydrops fetalis, medianasal hypoplasia, and laryngeal stenosis are pathognomonic for SRTD6 among all SRTD types.</div>
              </div>
            </div>
            {/* Renal */}
            <div className="col-12 col-md-6">
              <div className="card p-3 h-100">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT3 }}>Renal Disease</h6>
                {(breakdown.renal_distribution || []).map(r => (
                  <Bar key={r.label} label={r.label} value={r.n} max={N} color={ACCENT3} />
                ))}
                <div className="text-muted small mt-2">No NPHP allele series for NEK1. All renal disease is secondary. Renal transplant is CURATIVE (cell-autonomous; no recurrence).</div>
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
                <h6 className="fw-bold mb-2" style={{ color: ACCENT8 }}>Ethnicity Distribution</h6>
                {(breakdown.ethnicity_distribution || []).map(r => (
                  <Bar key={r.ethnicity} label={r.ethnicity} value={r.n} max={N} color={ACCENT8} />
                ))}
                <div className="text-muted small mt-2">MENA consanguinity prominent — homozygous DFG-loop missense most common. Under-ascertainment in non-consanguineous populations.</div>
              </div>
            </div>
            {/* Misdiagnosis */}
            <div className="col-12 col-md-6">
              <div className="card p-3 h-100">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT2 }}>Misdiagnosis Before Gene Panel</h6>
                {(breakdown.misdiagnosis_distribution || []).map(r => (
                  <Bar key={r.label} label={r.label} value={r.n} max={N} color={ACCENT2} />
                ))}
                <div className="text-muted small mt-2">SRTD3/DYNC2H1 most common misdiagnosis (most common SRTD overall); Hydrolethalus syndrome confused when hydrops + absent cilia; gene panel resolves all.</div>
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
                <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Top Pathogenic Variants (NEK1 — basal body kinase)</h6>
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

      {/* ── TAB 2: NEK1 KINASE & CILIOGENESIS ── */}
      {tab === 2 && overview && (
        <div>
          <Section title="🔬 NEK1 Protein Architecture — Basal Body Kinase / TTBK2 Activator / Ciliogenesis Master Switch" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-light">
                  <tr>
                    <th>Domain</th><th>Residues</th><th>Structure</th><th>Key Contacts / Substrates</th><th>Pathogenic Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { d: 'Kinase N-lobe', r: 'aa 1–130', s: 'ATP-binding pocket; glycine-rich P-loop; Lys42 catalytic', c: 'ATP (substrate); P-loop forms phospho-transfer pocket', p: 'Hotspot: P-loop missense → reduced ATP binding → partial kinase activity → moderate SRTD6' },
                    { d: 'Kinase C-lobe / DFG activation loop', r: 'aa 131–270', s: 'DFG motif (Asp179-Phe180-Gly181); substrate-binding cleft; Asp179 catalytic', c: 'TTBK2 substrate; CEP164 substrate; ATRIP (DNA damage)', p: 'DFG missense → near-complete kinase abolition → severe SRTD6; biallelic truncating → SRPS II' },
                    { d: 'Kinase-coiled-coil linker', r: 'aa 271–600', s: 'Bridge between kinase output and dimerization domain', c: 'Transmits phospho-signal; required for full TTBK2 activation efficiency', p: 'Hypomorphic: partial TTBK2 phosphorylation preserved → mild SRTD6; adult survivors' },
                    { d: 'N-terminal coiled-coil (dimerization)', r: 'aa 601–900', s: 'NEK1 homodimerisation; basal body localisation signal', c: 'NEK1-NEK1 dimer; DYRK1A interaction', p: 'Dimerisation loss → kinase partially active; moderate-severe; basal body mislocalisation' },
                    { d: 'C-terminal coiled-coil (regulatory)', r: 'aa 900–1258', s: 'TTBK2 binding; CEP164 docking; regulatory partner interactions', c: 'TTBK2 (ciliogenesis switch); CEP164 (distal appendage scaffold)', p: 'C-terminal missense: retained localisation but reduced TTBK2 phosphorylation → mild to moderate' },
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

          <Section title="🔄 NEK1 → TTBK2 → CP110 Removal → Axoneme Nucleation Pathway" color={ACCENT8}>
            <div className="row g-2">
              {[
                { step: '1', label: 'NEK1 kinase localises to the basal body (mother centriole) — constitutively expressed at distal appendages', color: ACCENT },
                { step: '2', label: 'NEK1 phosphorylates TTBK2 (Tau Tubulin Kinase 2) at the distal appendage — activating TTBK2 kinase activity', color: ACCENT },
                { step: '3', label: 'Activated TTBK2 phosphorylates CEP164 (distal appendage scaffold) — enabling vesicle fusion and cilia membrane assembly', color: ACCENT },
                { step: '4', label: 'TTBK2 also phosphorylates CP110-binding partners (MPP9) — causing CP110 to dissociate from the centriole tip', color: ACCENT },
                { step: '5', label: 'CP110 removed → γ-tubulin exposed at centriole tip → axoneme nucleation begins → cilia elongate', color: ACCENT },
                { step: '6', label: 'IFT trains (IFT-A + IFT-B1/B2) begin moving cargo along growing axoneme → Hedgehog components delivered to tip', color: ACCENT },
                { step: '7', label: 'Hedgehog (Ihh/Shh) signalling active at ciliary tip → GLI2/3 activator forms → skeletal development normal', color: ACCENT },
                { step: '8', label: '❌ NEK1 loss: TTBK2 not activated → CP110 persists → axoneme cannot nucleate → NO CILIA → Hedgehog completely absent → GLI3R maximal → NARROW THORAX', color: ACCENT2 },
              ].map(s => (
                <div key={s.step} className="col-12">
                  <div className="d-flex align-items-center gap-2 p-2 rounded small" style={{ background: s.color + '12', border: `1px solid ${s.color}30` }}>
                    <span className="badge rounded-circle" style={{ background: s.color, minWidth: 24, height: 24 }}>{s.step}</span>
                    <span style={{ color: s.step === '8' ? ACCENT2 : '#333' }}>{s.label}</span>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="📊 SRTD6 (NEK1) vs All SRTD Molecular Classes — Complete Comparison" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead className="table-light">
                  <tr><th>Complex / Class</th><th>Direction</th><th>Key Protein</th><th>Failure Mode</th><th>EM Finding</th><th>Unique Features</th><th>SRTD Genes</th></tr>
                </thead>
                <tbody>
                  {[
                    { c: 'Basal Body Kinase ← SRTD6', dir: 'Ciliogenesis initiation', prot: 'NEK1 kinase → TTBK2 → CP110 removal', fail: 'Axoneme cannot nucleate; NO cilia at all', em: 'ABSENT / RUDIMENTARY cilia', uniq: 'Hydrops; medianasal hypoplasia; laryngeal stenosis; highest polydactyly (65–75%)', genes: 'SRTD6 (NEK1)' },
                    { c: 'IFT-B2 (anterograde distal)', dir: 'Base → Tip (anterograde)', prot: 'IFT-B2 subcomplex (IFT80, IFT172, CLUAP1)', fail: 'Cilia cannot extend; IFT-B2 cargo undelivered', em: 'SHORTENED cilia', uniq: 'IFT172/SRTD10: BBS-like retinal alleles', genes: 'SRTD1 (IFT80), SRTD10 (IFT172), SRTD13 (CLUAP1)' },
                    { c: 'IFT-A (retrograde adaptor)', dir: 'Tip → Base (adaptor)', prot: 'IFT-A complex (WDR19, IFT140, WDR35, TTC21B)', fail: 'IFT-B cannot enter cilia base; uniform shortening', em: 'SHORT STUBBY cilia', uniq: 'WDR19/SRTD5: ectodermal features (CED-like); TTC21B/SRTD4: Joubert JBTS12 alleles', genes: 'SRTD4 (TTC21B), SRTD5 (WDR19), SRTD7 (WDR35), SRTD9 (IFT140)' },
                    { c: 'Dynein-2 (retrograde motor)', dir: 'Tip → Base (retrograde)', prot: 'Dynein-2 motor (DYNC2H1 + WDR34/WDR60/etc.)', fail: 'IFT-B stranded at tip; retrograde blocked', em: 'CLUB / BULGING TIP', uniq: 'Most common SRTD class (DYNC2H1 50%); highest renal rate', genes: 'SRTD3 (DYNC2H1), SRTD8 (WDR60), SRTD11 (WDR34), SRTD15 (DYNC2LI1), SRTD17 (TCTEX1D2)' },
                  ].map(r => (
                    <tr key={r.c} style={{ background: r.c.includes('SRTD6') ? ACCENT + '08' : undefined }}>
                      <td className="fw-bold" style={{ color: r.c.includes('SRTD6') ? ACCENT : undefined }}>{r.c}</td>
                      <td>{r.dir}</td><td>{r.prot}</td><td>{r.fail}</td>
                      <td className="fw-bold" style={{ color: r.c.includes('SRTD6') ? ACCENT8 : r.c.includes('IFT-B2') ? '#006064' : r.c.includes('IFT-A') ? '#7b1fa2' : '#bf360c' }}>{r.em}</td>
                      <td className="small" style={{ color: '#555' }}>{r.uniq}</td>
                      <td className="small">{r.genes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="⚕️ Treatment Pathway — NEK1 / SRTD6" color={ACCENT}>
            <div className="row g-2">
              {[
                { title: 'Narrow Thorax (Primary)', text: 'VEPTR or MAGEC growing rods — first-line surgical; serial expansion ×2/yr; same as all SRTDs', color: ACCENT2 },
                { title: 'Neonatal Respiratory', text: 'Mechanical ventilation (null alleles; SRPS II); CPAP/BiPAP (moderate); wean as thorax expands post-VEPTR', color: ACCENT3 },
                { title: 'Laryngeal Stenosis', text: 'Endoscopic laryngoscopy + dilation if severe; tracheostomy rarely required; ENT mandatory consult', color: ACCENT5 },
                { title: 'Hydrops / Fetal', text: 'Antenatal ECHO + Doppler; diuretics postnatal; no fetal intervention approved 2026; delivery planning for hydrops', color: ACCENT6 },
                { title: 'Renal', text: 'Annual GFR/USS/creatinine/urinalysis; ACEi/ARB proteinuria; transplant CURATIVE (cell-autonomous; no recurrence); NO NPHP allele counselling', color: ACCENT3 },
                { title: 'Retinal', text: 'Annual ERG from age 4 (15–25%); ophthalmology; low vision support; standard SRTD retinal surveillance', color: ACCENT4 },
                { title: 'Hepatic', text: 'APRI + USS annually; hepatology; avoid hepatotoxics; no portal hypertension management needed unless CHF confirmed', color: ACCENT5 },
                { title: 'Gene Panel + Genetics', text: 'Ensure NEK1 on panel (not IFT-only panels); cascade testing; 25% AR recurrence; prenatal/PGT; enrol in NEK1 registry', color: ACCENT },
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
          <Section title="🧬 Gene Card — NEK1 (*604588)" color={ACCENT}>
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
          <Section title="🏥 Disease Card — SRTD6 / ATD6 / Majewski (#263520)" color={ACCENT2}>
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
          <Section title="🩺 Diagnostic Workup — Step-by-Step" color={ACCENT8}>
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
          <Section title="🧪 Key Pathogenic Variants (NEK1)" color={ACCENT}>
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
          SRTD6 / ATD6 — NEK1 (*604588) — 4q33 — 1258 aa — Basal Body Kinase / TTBK2 Activator / Absent Cilia ·
          OMIM #263520 · Majewski syndrome / SRPS type II · Educational 40-patient cohort (seed {SEED})
        </span>
        <Link href="/" className="text-decoration-none" style={{ color: ACCENT }}>← Portal Home</Link>
      </div>
    </div>
  );
}
