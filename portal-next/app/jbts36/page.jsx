'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'CCRK IFT-Tip Kinase Pearls', 'Definitions'];

// JBTS36 colour scheme — CCRK / ICK-activating kinase / IFT-B tip turnaround / cilia hyperelongated / 14q23.2
// Indigo for kinase domain; green for elongated cilia; orange for IFT-B tip accumulation; teal for ICK axis
const ACCENT   = '#283593';   // deep indigo — CCRK kinase domain / CMGC family
const ACCENT2  = '#1b5e20';   // dark green — cilia hyperelongated (positive ARL13B)
const ACCENT3  = '#e65100';   // deep orange — IFT-B tip accumulation
const ACCENT4  = '#00695c';   // teal — CCRK → ICK axis
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#6a1b9a';   // deep purple — pT157-ICK (absent in JBTS36)
const ACCENT7  = '#827717';   // olive — hepatic (mild ~8%)
const ACCENT8  = '#0277bd';   // sky blue — renal NPHP-like (tubular cilia elongated)
const ACCENT9  = '#880e4f';   // deep magenta — GLI3-activator shift / polydactyly
const ACCENT10 = '#4e342e';   // brown — photoreceptor connecting cilia / retinal

const SEED     = 489;
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

export default function JBTS36Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts36/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts36/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts36/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov); setBreakdown(bk); setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center text-muted">Loading JBTS36 CCRK dashboard…</div>;
  if (error)   return <div className="p-4 text-danger">Error: {error}</div>;

  const kpis = overview?.kpis || {};
  const pheno = breakdown?.phenotype_frequencies || {};
  const patients = breakdown?.patients || [];
  const mol = breakdown?.molecular_features || {};
  const alleles = breakdown?.allele_specific_notes || [];
  const defs = definitions || {};

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22 0%, ${ACCENT2}18 100%)`, borderLeft: `5px solid ${ACCENT}` }}>
        <div className="d-flex flex-wrap align-items-start gap-2 justify-content-between">
          <div>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>JBTS36 — CCRK (Cell Cycle-Related Kinase)</h4>
            <div className="text-muted small">
              ICK-Activating Kinase · IFT-B Tip Turnaround Regulator · Cilia Hyperelongated (200–400% WT) ·
              No MKS Tier · 14q23.2 · {N_COHORT}-patient cohort (seed {SEED})
            </div>
            <div className="small mt-1">
              <span className="badge me-1" style={{ background: ACCENT }}>OMIM *609478</span>
              <span className="badge me-1" style={{ background: ACCENT4 }}>JBTS36 #618317</span>
              <span className="badge me-1" style={{ background: ACCENT2 }}>14q23.2</span>
              <span className="badge me-1 bg-secondary">~456 aa</span>
              <span className="badge me-1 bg-secondary">AR Biallelic Hypomorphic</span>
              <span className="badge" style={{ background: ACCENT9 }}>Polydactyly ~14% (GLI3-activator)</span>
            </div>
          </div>
          <Link href="/jbts35" className="btn btn-sm btn-outline-secondary">← JBTS35</Link>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ───────────────────────────────────────────── */}
      {tab === 0 && (
        <div>
          {/* KPIs */}
          <div className="row g-2 mb-3">
            <KPI label="Cohort (N)" value={kpis.total_patients ?? N_COHORT} color={ACCENT} />
            <KPI label="MTS 100%" value="100%" color={ACCENT} />
            <KPI label="Ataxia" value={`${kpis.ataxia_pct ?? '—'}%`} color={ACCENT9} />
            <KPI label="Hypotonia" value={`${kpis.hypotonia_pct ?? '—'}%`} color={ACCENT9} />
            <KPI label="OMA" value={`${kpis.oma_pct ?? '—'}%`} color={ACCENT} />
            <KPI label="Retinal" value={`${kpis.retinal_pct ?? '—'}%`} color={ACCENT10} />
            <KPI label="Renal" value={`${kpis.renal_pct ?? '—'}%`} color={ACCENT8} />
            <KPI label="Polydactyly" value={`${kpis.poly_pct ?? '—'}%`} color={ACCENT9} />
            <KPI label="ID" value={`${kpis.id_pct ?? '—'}%`} color={ACCENT5} />
            <KPI label="ESRD" value={`${kpis.esrd_pct ?? '—'}%`} color={ACCENT8} />
            <KPI label="MKS Tier" value="NO" color="#388e3c" />
            <KPI label="Cilia" value="ELONGATED" color={ACCENT2} />
          </div>

          {/* Alerts */}
          <Alert color={ACCENT2}>
            <strong style={{ color: ACCENT2 }}>CILIA HYPERELONGATED — KEY DDx BIOMARKER:</strong>{' '}
            JBTS36/CCRK fibroblasts show ARL13B+ cilia <strong>&gt;10 µm</strong> (WT 3–8 µm; patient ~12–20 µm; up to 400% WT).
            IFT88/IFT172 IF shows distal tip accumulation. Mechanistically opposite to JBTS35/KIAA0753 (cilia <em>absent</em>),
            JBTS33/CPLANE1 (cilia <em>short</em>), JBTS5/CEP290 (cilia <em>present, normal length</em>).
            Elongated ARL13B+ cilia → prompt CCRK + ICK co-sequencing.
          </Alert>

          <Alert color={ACCENT6}>
            <strong style={{ color: ACCENT6 }}>pT157-ICK ABSENT — DIAGNOSTIC IF MARKER:</strong>{' '}
            CCRK LOF → ICK not T-loop phosphorylated (Thr157). Anti-pT157-ICK IF is negative; anti-ICK total IF is positive
            (ICK protein present). This pattern distinguishes JBTS36 (CCRK LOF: ICK protein present, pT157 absent)
            from direct ICK LOF (ICK protein absent or non-functional). Co-sequencing CCRK + ICK (MAK) is mandatory.
          </Alert>

          <Alert color={ACCENT3}>
            <strong style={{ color: ACCENT3 }}>IFT-B TIP ACCUMULATION:</strong>{' '}
            Anterograde IFT-B (IFT88, IFT172) accumulates at cilia distal tips without retrograde switch.
            CCRK → ICK → phospho-KIF3A cascade broken → kinesin-II motor does not release from tip → IFT-B stalls.
            Diagnostic: IFT88 IF puncta at distal tips (not in axoneme body or basal body zone).
          </Alert>

          <Alert color={ACCENT9}>
            <strong style={{ color: ACCENT9 }}>GLI3-ACTIVATOR SHIFT — POLYDACTYLY ~14%:</strong>{' '}
            Hyperelongated cilia distort (not abolish) Hh gradient. PTCH1 cycling extended → GLI3FL/GLI3R ratio elevated
            → activator excess → postaxial polydactyly ~14% (above JBTS35 ~8%, below JBTS33 ~24%).
            Mechanistically distinct from PCP/BB-docking polydactyly (CPLANE1) — IFT-tip kinase etiology.
          </Alert>

          <Alert color="#1565c0">
            <strong style={{ color: '#1565c0' }}>NO MKS TIER — ALL JBTS36 LIVEBORN:</strong>{' '}
            CCRK is a kinase, not a TZ structural protein. No biallelic null CCRK → MKS perinatal lethal has been reported.
            Recurrence counselling: 25% JBTS36 (survivable), NOT MKS. Distinguishes from JBTS5, JBTS28, JBTS34.
          </Alert>

          {/* Mechanism */}
          <Section title="Molecular Mechanism (CCRK → ICK → KIF3A → IFT-B Tip Turnaround)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '22' }}>
                  <tr>
                    <th>Step</th><th>Component</th><th>Normal Function</th><th>JBTS36 LOF Consequence</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td>1</td><td>CCRK (LOF)</td><td>Phosphorylates ICK Thr157 (T-loop activation)</td><td>ICK not T-loop–phosphorylated → ICK under-activated</td></tr>
                  <tr><td>2</td><td>ICK (inactive)</td><td>Phosphorylates KIF3A at cilia tip → retrograde switch</td><td>KIF3A not phosphorylated → anterograde continues</td></tr>
                  <tr><td>3</td><td>IFT-B (KIF3A)</td><td>Anterograde particle released at tip → IFT-A/dynein retrograde</td><td>IFT-B accumulates at distal tip → cilia hyperelongated</td></tr>
                  <tr><td>4</td><td>Cilia length</td><td>3–8 µm (controlled by tip turnaround kinetics)</td><td>12–20 µm (~200–400% WT; ARL13B+ elongated)</td></tr>
                  <tr><td>5</td><td>PTCH1 cycling</td><td>Normal SHH gradient → GLI3R dominant → Hh off in non-stimulated cells</td><td>Extended PTCH1 cycling → GLI3FL elevated → Hh distorted</td></tr>
                  <tr><td>6</td><td>Cerebellum</td><td>SHH → EGL granule cell proliferation/migration → normal vermis</td><td>Distorted SHH → EGL impaired → vermis hypoplasia → MTS</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* Cilia IF Comparison */}
          <Section title="Cilia IF Signature — JBTS36 vs Other JBTS Subtypes" color={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT4 + '22' }}>
                  <tr><th>Marker</th><th>JBTS36 CCRK</th><th>JBTS35 KIAA0753</th><th>JBTS33 CPLANE1</th><th>JBTS29 TOGARAM1</th><th>JBTS5 CEP290</th></tr>
                </thead>
                <tbody>
                  <tr><td>ARL13B IF</td><td style={{ color: ACCENT2 }}><strong>Elongated &gt;10 µm</strong></td><td className="text-danger"><strong>ABSENT</strong></td><td>Short 50–70% WT</td><td>Short 60–80% WT</td><td>Normal length</td></tr>
                  <tr><td>IFT88 IF</td><td style={{ color: ACCENT3 }}><strong>Tip accumulation</strong></td><td>N/A (no cilia)</td><td>Normal axoneme</td><td>Reduced tip</td><td>Normal</td></tr>
                  <tr><td>pT157-ICK IF</td><td className="text-danger"><strong>ABSENT</strong></td><td>Normal (ICK intact)</td><td>Normal</td><td>Normal</td><td>Normal</td></tr>
                  <tr><td>GT335 (polyGlu-tubulin)</td><td style={{ color: ACCENT2 }}>Normal/elevated</td><td>N/A (no cilia)</td><td>Normal</td><td className="text-danger"><strong>Reduced (DDx!)</strong></td><td>Normal</td></tr>
                  <tr><td>CP110 at BB tip</td><td>Absent (removed normally)</td><td className="text-danger"><strong>RETAINED</strong></td><td>Absent (normal)</td><td>Absent (normal)</td><td>Absent (normal)</td></tr>
                  <tr><td>Polydactyly rate</td><td>~14%</td><td>~8%</td><td>~24%</td><td>~15%</td><td>~10%</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* Domain Architecture */}
          <Section title="CCRK Protein Domain Architecture (~456 aa)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT + '22' }}>
                  <tr><th>Domain</th><th>Residues</th><th>Function</th><th>JBTS36 Variant(s)</th></tr>
                </thead>
                <tbody>
                  <tr><td>N-lobe (β-sheet, Gly-loop)</td><td>aa 1–100</td><td>ATP binding (Gly14, Gly17); N-lobe hydrophobic core (Leu78, Ile82)</td><td>Leu78Pro (N-lobe β3 disruption; ATP −70%)</td></tr>
                  <tr><td>HRD motif / Catalytic loop</td><td>aa 130–145</td><td>Arg141 catalytic arginine; DFG Asp141 Mg²⁺ coordination; phospho-transfer</td><td>Arg141Gln (most recurrent MENA; −92% activity)</td></tr>
                  <tr><td>Activation segment / T-loop</td><td>aa 161–188</td><td>Thr161 CCRK autophosphorylation; ICK Thr157 substrate docking geometry</td><td>Asp178His (S. Asian; ICK docking −50%)</td></tr>
                  <tr><td>Activation segment border (DFG+2)</td><td>aa 214–220</td><td>Glu216-Arg253 salt bridge; substrate cavity entrance</td><td>Glu216Lys (pan-ethnic; −40%)</td></tr>
                  <tr><td>C-lobe / extension linker</td><td>aa 325–360</td><td>Phe338 aromatic packing; CTS hinge conformation; basal-body localisation</td><td>Phe338Val (European; mislocalisation ~40%)</td></tr>
                  <tr><td>Ciliary Targeting Signal (CTS)</td><td>aa 380–410</td><td>RVSP motif (aa 393–396); IFT-A import; basal body periciliary localisation</td><td>Thr394Ile (CTS RVSP disruption)</td></tr>
                  <tr><td>Splice — intron 7</td><td>aa 249–281 (exon 7)</td><td>Arg253 (HRD+1); substrate-binding cavity; activation segment N-boundary</td><td>c.842+1G>A (European; 14% residual mRNA; −80% ICK phos.)</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Key Facts — JBTS36 / CCRK" color={ACCENT}>
            <ul className="small">
              {(overview?.key_facts || []).map((f, i) => <li key={i}>{f}</li>)}
            </ul>
          </Section>
        </div>
      )}

      {/* ── TAB 1: Diagnostic Breakdown ──────────────────────────────── */}
      {tab === 1 && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <Section title="Phenotype Frequencies (N=40)" color={ACCENT}>
                {Object.entries(pheno).map(([k, v]) => (
                  <Bar key={k} label={`${k.replace(/_/g,' ')} (${v.pct}%)`} value={v.n} max={N_COHORT} color={ACCENT} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Ethnicity Distribution" color={ACCENT4}>
                {Object.entries(overview?.ethnicity_distribution || {}).map(([eth, n]) => (
                  <Bar key={eth} label={eth} value={n} max={N_COHORT} color={ACCENT4} />
                ))}
              </Section>
              <Section title="Allele Class Distribution" color={ACCENT6}>
                {Object.entries(overview?.allele_class_distribution || {}).map(([ac, n]) => (
                  <Bar key={ac} label={ac} value={n} max={N_COHORT} color={ACCENT6} />
                ))}
              </Section>
            </div>
          </div>

          {/* Molecular Features */}
          <Section title="Molecular / Cellular Features (CCRK LOF)" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(mol).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold text-nowrap" style={{ width: '30%' }}>{k.replace(/_/g,' ')}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient Table */}
          <Section title="Cohort Detail (40 patients, seed 489)" color={ACCENT5}>
            <div className="table-responsive" style={{ maxHeight: 400, overflowY: 'auto' }}>
              <table className="table table-sm table-striped small">
                <thead className="sticky-top" style={{ background: ACCENT5 + '22' }}>
                  <tr>
                    <th>ID</th><th>Age</th><th>Sex</th><th>Ethnicity</th><th>Allele Class</th>
                    <th>MTS</th><th>Ataxia</th><th>Retinal</th><th>Renal</th><th>Poly</th><th>Cilia</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr key={p.id}>
                      <td className="fw-bold" style={{ color: ACCENT }}>{p.id}</td>
                      <td>{p.age}</td><td>{p.sex}</td><td>{p.ethnicity}</td>
                      <td><span className="badge" style={{ background: ACCENT6, fontSize: 10 }}>{p.allele_class}</span></td>
                      <td>{p.mts ? '✓' : '—'}</td>
                      <td style={{ color: p.ataxia ? ACCENT9 : '#aaa' }}>{p.ataxia ? '✓' : '—'}</td>
                      <td style={{ color: p.retinal ? ACCENT10 : '#aaa' }}>{p.retinal ? '✓' : '—'}</td>
                      <td style={{ color: p.renal ? ACCENT8 : '#aaa' }}>{p.renal ? '✓' : '—'}</td>
                      <td style={{ color: p.poly ? ACCENT9 : '#aaa' }}>{p.poly ? '✓' : '—'}</td>
                      <td style={{ color: ACCENT2, fontWeight: 'bold' }}>LONG</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 2: CCRK IFT-Tip Kinase Pearls ───────────────────────── */}
      {tab === 2 && (
        <div>
          <Section title="Allele-Specific Molecular Mechanisms" color={ACCENT}>
            {alleles.map((a, i) => (
              <div key={i} className="mb-3 p-3 rounded" style={{ background: ACCENT + '0d', border: `1px solid ${ACCENT}33` }}>
                <div className="fw-bold mb-1" style={{ color: ACCENT }}>{a.name} — {a.cdna}</div>
                <div className="small text-muted mb-1"><strong>Domain:</strong> {a.domain} | <strong>Population:</strong> {a.population} | <strong>Severity:</strong> {a.severity}</div>
                <div className="small">{a.mechanism}</div>
              </div>
            ))}
          </Section>

          <Section title="DDx Pearls — JBTS36 vs Closest Mimics" color={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT4 + '22' }}>
                  <tr><th>Feature</th><th>JBTS36 CCRK</th><th>JBTS35 KIAA0753</th><th>JBTS29 TOGARAM1</th><th>JBTS33 CPLANE1</th></tr>
                </thead>
                <tbody>
                  <tr><td>Cilia morphology</td><td style={{ color: ACCENT2 }}><strong>LONG (200–400% WT)</strong></td><td className="text-danger">ABSENT</td><td>Short (60–80%)</td><td>Short (50–70%)</td></tr>
                  <tr><td>ARL13B IF</td><td style={{ color: ACCENT2 }}>Present; elongated</td><td className="text-danger">ABSENT</td><td>Present; short</td><td>Present; short</td></tr>
                  <tr><td>pT157-ICK IF</td><td className="text-danger">ABSENT</td><td>Normal</td><td>Normal</td><td>Normal</td></tr>
                  <tr><td>IFT88 IF</td><td style={{ color: ACCENT3 }}>Tip accumulation</td><td>N/A (no cilia)</td><td>Reduced at tip</td><td>Normal axoneme</td></tr>
                  <tr><td>GT335 IF</td><td>Normal</td><td>N/A</td><td className="text-danger"><strong>REDUCED (key DDx)</strong></td><td>Normal</td></tr>
                  <tr><td>CP110 at BB</td><td>Removed (normal)</td><td className="text-danger"><strong>RETAINED</strong></td><td>Removed</td><td>Removed</td></tr>
                  <tr><td>Polydactyly</td><td>~14% GLI3-activator</td><td>~8% very low</td><td>~15% moderate</td><td>~24% PCP-enriched</td></tr>
                  <tr><td>Retinal</td><td>~28%</td><td>~18%</td><td>~20%</td><td>~22%</td></tr>
                  <tr><td>Renal</td><td>~22%</td><td>~15%</td><td>~18%</td><td>~18%</td></tr>
                  <tr><td>MKS tier</td><td>NO</td><td>NO</td><td>NO</td><td>NO</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Recommended Diagnostic Panel (JBTS36 / Elongated-Cilia Ciliopathy)" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT6 + '22' }}>
                  <tr><th>Test</th><th>Expected in JBTS36</th><th>Distinguishes From</th></tr>
                </thead>
                <tbody>
                  <tr><td>ARL13B IF (cilia length)</td><td style={{ color: ACCENT2 }}>Elongated &gt;10 µm (pathognomonic)</td><td>All other JBTS (absent, short, or normal length)</td></tr>
                  <tr><td>pT157-ICK IF</td><td className="text-danger">ABSENT (ICK not activated)</td><td>ICK direct LOF (ICK protein also absent)</td></tr>
                  <tr><td>ICK total IF</td><td>Present (ICK protein intact)</td><td>CCRK LOF (ICK present) vs ICK LOF (ICK absent)</td></tr>
                  <tr><td>IFT88 IF</td><td style={{ color: ACCENT3 }}>Tip accumulation</td><td>TOGARAM1 (reduced tip), CEP290 (normal)</td></tr>
                  <tr><td>GT335 IF</td><td>Normal/elevated</td><td>TOGARAM1/JBTS29 (GT335 reduced — key negative DDx)</td></tr>
                  <tr><td>CP110 IF</td><td>Absent from BB tip (normal removal)</td><td>KIAA0753/JBTS35 (CP110 retained)</td></tr>
                  <tr><td>WES / ciliopathy panel</td><td>CCRK biallelic variants</td><td>Co-sequence ICK (MAK) simultaneously</td></tr>
                  <tr><td>MRI brain</td><td>MTS (100%); cerebellar vermis hypoplasia</td><td>Confirms Joubert Syndrome tier</td></tr>
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 3: Definitions ───────────────────────────────────────── */}
      {tab === 3 && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card">
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT }}>Disease Overview</h6>
                  <table className="table table-sm small mb-0">
                    <tbody>
                      <tr><td className="fw-bold">Disease</td><td>{defs.disease_name}</td></tr>
                      <tr><td className="fw-bold">Gene</td><td>{defs.gene}</td></tr>
                      <tr><td className="fw-bold">OMIM Gene</td><td>{defs.omim_gene}</td></tr>
                      <tr><td className="fw-bold">OMIM Disease</td><td>{defs.omim_disease}</td></tr>
                      <tr><td className="fw-bold">Chromosome</td><td>{defs.chromosome}</td></tr>
                      <tr><td className="fw-bold">Protein size</td><td>{defs.protein_size}</td></tr>
                      <tr><td className="fw-bold">Inheritance</td><td>{defs.inheritance}</td></tr>
                      <tr><td className="fw-bold">MKS tier</td><td style={{ color: '#388e3c' }}>NO — all liveborn</td></tr>
                      <tr><td className="fw-bold">SRTD allelic</td><td>No</td></tr>
                      <tr><td className="fw-bold">Polydactyly enriched</td><td style={{ color: ACCENT9 }}>YES (~14%; GLI3-activator shift)</td></tr>
                      <tr><td className="fw-bold">Microcephaly</td><td>No</td></tr>
                      <tr><td className="fw-bold">Frequency (JBTS)</td><td>{defs.frequency_jbts}</td></tr>
                      <tr><td className="fw-bold">Worldwide prevalence</td><td>{defs.worldwide_prevalence}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card">
                <div className="card-body">
                  <h6 className="fw-bold" style={{ color: ACCENT4 }}>Key Biomarkers</h6>
                  <table className="table table-sm small mb-0">
                    <tbody>
                      {Object.entries(defs.key_biomarkers || {}).map(([k, v]) => (
                        <tr key={k}><td className="fw-bold text-nowrap">{k.replace(/_/g,' ')}</td><td>{v}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <Section title="Mechanism" color={ACCENT}>
            <p className="small">{defs.mechanism}</p>
          </Section>

          <Section title="DDx Pearls" color={ACCENT6}>
            <ul className="small">
              {(defs.ddx_pearls || []).map((p, i) => <li key={i}>{p}</li>)}
            </ul>
          </Section>

          <Section title="Glossary" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT5 + '22' }}>
                  <tr><th>Term</th><th>Definition</th></tr>
                </thead>
                <tbody>
                  {Object.entries(defs.glossary || {}).map(([term, def]) => (
                    <tr key={term}><td className="fw-bold text-nowrap">{term}</td><td>{def}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}
    </div>
  );
}
