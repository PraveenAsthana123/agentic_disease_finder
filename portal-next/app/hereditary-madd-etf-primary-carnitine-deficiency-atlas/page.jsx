'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-madd-etf-primary-carnitine-deficiency-atlas';
const TABS = ['Overview', 'Gene Table', 'Metabolic Groups', 'Definitions'];

const GENE_COLORS = {
  'ETFA':     '#1565c0',  // deep blue       — AR; MADD type I; ETF alpha; neonatal severe
  'ETFB':     '#6a1b9a',  // deep purple     — AR; MADD type II; ETF beta; neonatal severe
  'ETFDH':    '#1b5e20',  // dark green      — AR; MADD type III; riboflavin-responsive; adult myopathic
  'SLC22A5':  '#b71c1c',  // deep red        — AR; PCD; OCTN2; DCM curative L-carnitine
  'ACADS':    '#e65100',  // deep orange     — AR; SCAD; NBS benign; C4 butyrylcarnitine
  'ACADL':    '#880e4f',  // dark magenta    — AR; LCAD; very rare; C12/C14
  'HSD17B10': '#4a148c',  // dark indigo     — XLR; HSD10 disease; ONLY X-linked FAO; neurodegeneration
  'DECR1':    '#00695c',  // dark teal       — AR; DECR1 deficiency; unsaturated FA; C10:2
};

const GENE_INFO = {
  'ETFA':     { full: 'ETFA / Electron Transfer Flavoprotein Alpha / MADD type I / 333aa', locus: '15q23',    inh: 'AR LOF',
                note: '333aa/35kDa; ETF alpha subunit; accepts electrons from all acyl-CoA dehydrogenases; MADD neonatal severe form; polycystic kidneys + dysmorphic; multiple acylcarnitines C4-C18 on NBS; urine glutaric+ethylmalonic+isovalerylglycine; riboflavin trial MANDATORY (usually not responsive); neonatal lethal severe form' },
  'ETFB':     { full: 'ETFB / Electron Transfer Flavoprotein Beta / MADD type II / 255aa', locus: '19q13.41', inh: 'AR LOF',
                note: '255aa/27kDa; ETF beta subunit; forms heterodimer with ETFA; MADD type II = identical biochemistry to ETFA; severe neonatal form; polycystic kidneys+facial dysmorphism; riboflavin trial MANDATORY; same multiple acylcarnitine NBS profile as ETFA' },
  'ETFDH':    { full: 'ETFDH / ETF:Ubiquinone Oxidoreductase / MADD type III / 617aa',    locus: '4q32.1',   inh: 'AR LOF',
                note: '617aa/68kDa; ETF-QO inner IMM; transfers electrons ETF→CoQ; MOST riboflavin-responsive MADD (>80%); adult myopathic dominant phenotype; lipid storage myopathy on biopsy; rhabdomyolysis; Ala84Thr common in Chinese/East Asian; secondary CoQ10 deficiency; riboflavin 100-300mg/day DRAMATIC; CoQ10 200-300mg/day synergistic' },
  'SLC22A5':  { full: 'SLC22A5 / OCTN2 / Primary Carnitine Deficiency / 557aa',           locus: '5q31.1',   inh: 'AR LOF',
                note: '557aa/63kDa; high-affinity sodium-dependent carnitine transporter; LOF→carnitine wasting urine+depleted intracellular; DILATED CARDIOMYOPATHY cardinal presentation; C0 FREE CARNITINE VERY LOW (<5µmol/L) = PATHOGNOMONIC NBS; L-carnitine 50-100mg/kg/day CURATIVE — dramatic echo reversal; Pro467Arg Japanese founder; never stop L-carnitine; test mother too (maternal PCD)' },
  'ACADS':    { full: 'ACADS / Short-Chain Acyl-CoA Dehydrogenase / SCAD / 412aa',        locus: '12q24.31', inh: 'AR LOF',
                note: '412aa/44kDa; C4 short-chain acyl-CoA dehydrogenase; MOST COMMON FAO enzyme on NBS (1/35,000-50,000); C4 butyrylcarnitine elevated+urine ethylmalonic acid; USUALLY BENIGN — clinical significance DEBATED; current consensus: incidental NBS finding in most; Arg171Trp variant common; do NOT restrict diet without clinical decompensation' },
  'ACADL':    { full: 'ACADL / Long-Chain Acyl-CoA Dehydrogenase / LCAD / 429aa',         locus: '2q34',     inh: 'AR LOF',
                note: '429aa/46kDa; C12-C14 long-chain acyl-CoA dehydrogenase; VERY RARE (<100 cases); C12+C14 elevated NBS (C14:1 LOWER than VLCAD — key DDx); C12 most discriminating from VLCAD; hypoketotic hypoglycemia+hepatomegaly; MCT oil; similar to VLCAD management; exercise trigger' },
  'HSD17B10': { full: 'HSD17B10 / HSD10 / MHBD Deficiency / 261aa',                      locus: 'Xp11.22',  inh: 'XLR LOF',
                note: '261aa/27kDa; ONLY X-LINKED FAO disorder; 2-methyl-3-hydroxybutyryl-CoA DH; isoleucine catabolism+short-chain FA; C5-OH (2-methyl-3-hydroxybutyrylcarnitine) NBS; SEVERE PROGRESSIVE NEURODEGENERATION males (regression+choreoathetosis+seizures+optic atrophy); cardiomyopathy; Arg130Cys most common; females usually carriers; DDx C5-OH: ACAT1 (beta-ketothiolase) + HMGCL' },
  'DECR1':    { full: 'DECR1 / 2,4-Dienoyl-CoA Reductase 1 / DECR1 Deficiency / 340aa',  locus: '8q21.3',   inh: 'AR LOF',
                note: '340aa/36kDa; required for oxidation of unsaturated FA (2-trans-4-cis dienoyl-CoAs); C10:2 decadienoylcarnitine PATHOGNOMONIC NBS marker; VERY RARE (<20 reported cases); neonatal hypotonia+hypoketotic hypoglycemia+acidosis; low-fat diet+avoid fasting+L-carnitine; avoid unsaturated fat supplements' },
};

const PATHOGNOMONIC = {
  'ETFA':     'ETFA/ETFB MADD type I/II — NEONATAL SEVERE FORM: multiple C4-C18 acylcarnitines simultaneously elevated on NBS (pathognomonic for ETF defect). Polycystic kidneys + brain dysgenesis + dysmorphic = worst alleles. Urine OA: GLUTARIC ACID + ETHYLMALONIC ACID + ISOVALERYLGLYCINE + hexanoylglycine (MADD triad). Riboflavin trial MANDATORY even if not predicted responsive. ETFA and ETFB phenotypes identical — distinguish by gene sequencing only.',
  'ETFB':     'ETFB MADD type II — same NBS and urine OA profile as ETFA. ETFB p. alleles associated with severe neonatal form + polycystic kidneys + facial dysmorphism. INTERMEDIATE form: neonatal hypoglycemia+cardiomyopathy+liver failure, may survive to childhood. MILD form: exercise-induced myopathy (rare in ETFB). RIBOFLAVIN TRIAL MANDATORY. Distinguish ETFA vs ETFB: gene panel required (both loci must be tested).',
  'ETFDH':    'ETFDH MADD type III — MOST RIBOFLAVIN-RESPONSIVE FORM (>80%). ADULT MYOPATHIC = dominant presentation: proximal myopathy, exercise intolerance, rhabdomyolysis in 2nd-4th decade. LIPID STORAGE MYOPATHY on biopsy (lipid droplet accumulation). SECONDARY CoQ10 DEFICIENCY — test CoQ10 in muscle/serum. Ala84Thr: common Chinese/East Asian allele, highly riboflavin-responsive. Riboflavin 100-300mg/day → DRAMATIC improvement in days-weeks. CoQ10 200-300mg/day synergistic. Adult ETFDH = treatable myopathy — always try riboflavin FIRST.',
  'SLC22A5':  'PRIMARY CARNITINE DEFICIENCY — C0 FREE CARNITINE VERY LOW (<5 µmol/L; normal 20-50 µmol/L) = PATHOGNOMONIC NBS. C0 most reliable NBS marker. DILATED CARDIOMYOPATHY = cardinal presentation (may present as cardiac failure or sudden cardiac death in young child). L-CARNITINE 50-100mg/kg/day PO = CURATIVE — echo normalizes within months. NEVER stop L-carnitine. MATERNAL PCD: heterozygous mothers have low carnitine → may have cardiac symptoms → test mother. Japanese founder Pro467Arg accounts for ~45% Japanese PCD alleles. ANY unexplained DCM in child → test plasma free carnitine IMMEDIATELY.',
  'ACADS':    'SCAD DEFICIENCY — NBS DETECTION: C4 (butyrylcarnitine) elevated + urine ethylmalonic acid. MOST COMMON FAO enzyme on NBS (1/35,000-50,000). CLINICAL SIGNIFICANCE DEBATED — current ACMG expert consensus: SCAD deficiency is a BENIGN BIOCHEMICAL VARIANT in most NBS-detected cases. Population controls carry ACADS variants with elevated C4. DO NOT diagnose "SCAD disease" on NBS alone. No dietary restriction required for asymptomatic NBS-identified infants. Only treat with fasting avoidance if clear metabolic decompensation.',
  'ACADL':    'LCAD DEFICIENCY — VERY RARE (<100 cases worldwide). C12+C14 elevated NBS. KEY DDx FROM VLCAD: C14:1 in LCAD is LOWER than in VLCAD (VLCAD has C14:1 as primary marker). C12 acylcarnitine is most discriminating for LCAD. Clinical: hypoketotic hypoglycemia + hepatomegaly (no cardiac, no rhabdomyolysis in classical form). Exercise trigger in some. Managed similar to VLCAD: MCT oil, avoid fasting. No approved biomarker — biochemical DDx from VLCAD requires ratios.',
  'HSD17B10': 'HSD10 DISEASE (MHBD Deficiency) — ONLY X-LINKED FATTY ACID OXIDATION DISORDER. C5-OH (2-methyl-3-hydroxybutyrylcarnitine) elevated on NBS. DDx C5-OH: ALSO elevated in ACAT1 (beta-ketothiolase deficiency) and HMGCL deficiency — distinguish by urine OA (HSD10: 2-methyl-3-hydroxybutyrate; ACAT1: 2-methylacetoacetate). MALES: SEVERE PROGRESSIVE NEURODEGENERATION — normal development to 6-18mo then regression, choreoathetosis, loss of speech, seizures, optic atrophy, cardiomyopathy. FEMALES: heterozygous carriers, usually asymptomatic (skewed X-inactivation may cause learning difficulties). NO curative treatment — low-isoleucine diet may reduce substrate load.',
  'DECR1':    'DECR1 DEFICIENCY — C10:2 (DECADIENOYLCARNITINE) ON NBS = PATHOGNOMONIC for DECR1 deficiency. VERY RARE (fewer than 20 cases reported globally). Enzyme catalyzes reduction of 2-trans-4-cis dienoyl-CoA intermediates (required for even-chain unsaturated FA oxidation). Neonatal/infantile: hypotonia + hypoketotic hypoglycemia + metabolic acidosis. Treatment: low-fat diet + avoid fasting + L-carnitine + AVOID unsaturated fat supplements (worsen substrate accumulation).',
};

export default function HereditaryMADDETFAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedDef, setExpandedDef] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#0d1b2a', minHeight: '100vh', fontSize: 18 }}>Loading Hereditary-MADD-ETF-Primary-Carnitine-Deficiency-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#0d1b2a', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#e6edf3', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1565c0 0%,#6a1b9a 25%,#1b5e20 50%,#b71c1c 75%,#4a148c 100%)', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>Hereditary MADD-ETF · Primary Carnitine Deficiency · Mitochondrial Beta-Oxidation · Newborn Screening</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
          🧬 Hereditary-MADD-ETF-Primary-Carnitine-Deficiency-Atlas
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
          Complete 8-Gene Reference · ETFA · ETFB · ETFDH · SLC22A5 · ACADS · ACADL · HSD17B10 · DECR1 ·
          320 Patients (8×40) · Seeds 4126–4133
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Total Patients', val: overview?.total_patients },
            { label: 'Genes', val: overview?.genes?.length },
            { label: 'Only X-Linked FAO', val: 'HSD17B10' },
            { label: 'PCD Curative', val: 'L-Carnitine' },
            { label: 'ETFDH Riboflavin', val: '>80% Response' },
          ].map(({ label, val }) => (
            <div key={label} style={{ background: 'rgba(0,0,0,0.3)', borderRadius: 8, padding: '6px 14px', fontSize: 13 }}>
              <span style={{ color: 'rgba(255,255,255,0.6)' }}>{label}: </span>
              <span style={{ color: '#fff', fontWeight: 600 }}>{val ?? '…'}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Tab Bar */}
      <div style={{ display: 'flex', gap: 0, background: '#161b22', borderBottom: '1px solid #30363d', padding: '0 24px' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            style={{ padding: '12px 20px', background: 'none', border: 'none', cursor: 'pointer', fontSize: 14,
                     fontWeight: activeTab === t ? 700 : 400,
                     color: activeTab === t ? '#58a6ff' : '#8b949e',
                     borderBottom: activeTab === t ? '2px solid #58a6ff' : '2px solid transparent' }}>
            {t}
          </button>
        ))}
      </div>

      <div style={{ padding: '24px 32px', maxWidth: 1400 }}>

        {/* ── OVERVIEW TAB ── */}
        {activeTab === 'Overview' && overview && (
          <div>
            <h2 style={{ color: '#58a6ff', marginBottom: 16 }}>Atlas Overview — MADD-ETF Complex · Primary Carnitine Deficiency · 8-Gene Reference</h2>

            {/* Key Clinical Alerts */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 14, marginBottom: 24 }}>
              {[
                { color: '#1b5e20', title: '⚡ ETFDH — Riboflavin CURATIVE',
                  body: 'ETFDH (MADD type III): adult myopathic, riboflavin 100-300mg/day → >80% dramatic response. Ala84Thr = East Asian founder. CoQ10 synergistic. Never miss riboflavin trial.' },
                { color: '#b71c1c', title: '❤️ SLC22A5 — L-Carnitine CURATIVE',
                  body: 'PCD: DCM in infant/child. C0 very low (<5µmol/L) NBS. L-carnitine 50-100mg/kg/day → echo normalises within months. Lifelong. Test mother (maternal PCD).' },
                { color: '#4a148c', title: '⚠️ HSD17B10 — Only X-Linked FAO',
                  body: 'Only X-linked FAO disorder. Males: progressive neurodegeneration + cardiomyopathy. C5-OH NBS. DDx: ACAT1 (2-methylacetoacetate) and HMGCL. No curative Rx.' },
                { color: '#1565c0', title: '🔬 ETFA/ETFB — Neonatal MADD',
                  body: 'Multiple C4-C18 acylcarnitines on NBS = MADD signature. Urine OA: glutaric+ethylmalonic+isovalerylglycine. Riboflavin trial MANDATORY ALL MADD (even ETFA/ETFB).' },
                { color: '#e65100', title: '🟡 ACADS/SCAD — NBS Usually Benign',
                  body: 'C4+ethylmalonic acid. Most common FAO enzyme on NBS. Current consensus: BENIGN biochemical variant in most NBS-detected. Do not restrict diet without clinical decompensation.' },
                { color: '#00695c', title: '🧪 DECR1 — C10:2 Pathognomonic',
                  body: 'C10:2 decadienoylcarnitine on NBS = PATHOGNOMONIC for DECR1 deficiency. Very rare (<20 cases). Avoid unsaturated fat supplements. Low-fat diet.' },
              ].map(({ color, title, body }) => (
                <div key={title} style={{ background: '#161b22', border: `1px solid ${color}40`, borderLeft: `4px solid ${color}`, borderRadius: 8, padding: 16 }}>
                  <div style={{ fontWeight: 700, color: color, marginBottom: 6 }}>{title}</div>
                  <div style={{ fontSize: 13, color: '#8b949e', lineHeight: 1.5 }}>{body}</div>
                </div>
              ))}
            </div>

            {/* Gene summary cards */}
            <h3 style={{ color: '#e6edf3', marginBottom: 12 }}>Per-Gene Cohort Summary (40 patients each)</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12 }}>
              {overview.genes?.map(g => {
                const s = overview.cohort_summary?.[g];
                if (!s) return null;
                const col = GENE_COLORS[g] || '#58a6ff';
                return (
                  <div key={g} style={{ background: '#161b22', border: `1px solid ${col}50`, borderRadius: 10, padding: 16 }}>
                    <div style={{ fontWeight: 700, color: col, fontSize: 15, marginBottom: 4 }}>{g}</div>
                    <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 8 }}>{s.locus} · {s.inheritance} · {s.size_aa}aa</div>
                    <div style={{ fontSize: 11, color: '#c9d1d9', marginBottom: 8, lineHeight: 1.4 }}>
                      <strong>NBS:</strong> {s.nbs_marker}
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 12 }}>
                      {[
                        ['NBS Detected', s.nbs_detected_pct + '%'],
                        ['Cardiac', s.cardiac_pct + '%'],
                        ['Rhabdo', s.rhabdo_pct + '%'],
                        ['Riboflavin Resp', s.riboflavin_response_pct + '%'],
                        ['Cardiac Reversal', s.cardiac_reversal_pct + '%'],
                        ['Hypoglycemia', s.hypoglycemia_pct + '%'],
                      ].map(([k, v]) => (
                        <div key={k} style={{ background: '#0d1117', borderRadius: 4, padding: '3px 8px' }}>
                          <span style={{ color: '#8b949e' }}>{k}: </span>
                          <span style={{ color: '#e6edf3', fontWeight: 600 }}>{v}</span>
                        </div>
                      ))}
                    </div>
                    {(s.riboflavin_responsive || s.carnitine_curative || s.nbs_benign || s.neonatal_lethal || s.x_linked) && (
                      <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        {s.riboflavin_responsive && <span style={{ background: '#1b5e20', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>Riboflavin-Responsive</span>}
                        {s.carnitine_curative && <span style={{ background: '#b71c1c', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>L-Carnitine Curative</span>}
                        {s.nbs_benign && <span style={{ background: '#e65100', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>NBS Usually Benign</span>}
                        {s.neonatal_lethal && <span style={{ background: '#880e4f', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>Neonatal Severe</span>}
                        {s.x_linked && <span style={{ background: '#4a148c', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>X-Linked</span>}
                        {s.adult_myopathic && <span style={{ background: '#0d47a1', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>Adult Myopathic</span>}
                      </div>
                    )}
                    <div style={{ marginTop: 8, fontSize: 11, color: '#8b949e', borderTop: '1px solid #30363d', paddingTop: 6 }}>
                      <strong>Tx:</strong> {s.treatment?.substring(0, 100)}{s.treatment?.length > 100 ? '…' : ''}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── GENE TABLE TAB ── */}
        {activeTab === 'Gene Table' && overview && (
          <div>
            <h2 style={{ color: '#58a6ff', marginBottom: 16 }}>Gene Reference Table — MADD-ETF-PCD Atlas</h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#161b22', borderBottom: '2px solid #30363d' }}>
                    {['Gene', 'Full Name', 'Locus', 'Inh', 'NBS Marker', 'Key Notes', 'Treatment'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', textAlign: 'left', color: '#8b949e', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(GENE_INFO).map(([g, info], i) => (
                    <tr key={g} style={{ background: i % 2 === 0 ? '#0d1117' : '#161b22', borderBottom: '1px solid #21262d' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 700, color: GENE_COLORS[g] }}>{g}</td>
                      <td style={{ padding: '10px 12px', color: '#c9d1d9', maxWidth: 180 }}>{info.full}</td>
                      <td style={{ padding: '10px 12px', color: '#8b949e', whiteSpace: 'nowrap' }}>{info.locus}</td>
                      <td style={{ padding: '10px 12px', color: '#8b949e', whiteSpace: 'nowrap' }}>{info.inh}</td>
                      <td style={{ padding: '10px 12px', color: '#c9d1d9', maxWidth: 150 }}>
                        {overview.cohort_summary?.[g]?.nbs_marker || '—'}
                      </td>
                      <td style={{ padding: '10px 12px', color: '#8b949e', maxWidth: 250, fontSize: 12 }}>{info.note}</td>
                      <td style={{ padding: '10px 12px', color: '#c9d1d9', maxWidth: 180, fontSize: 12 }}>
                        {overview.cohort_summary?.[g]?.treatment || '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── METABOLIC GROUPS TAB ── */}
        {activeTab === 'Metabolic Groups' && breakdown && (
          <div>
            <h2 style={{ color: '#58a6ff', marginBottom: 16 }}>Metabolic Groups Breakdown</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
              {Object.entries(breakdown.groups || {}).map(([grpKey, grp]) => (
                <div key={grpKey} style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 10, padding: 18 }}>
                  <div style={{ fontWeight: 700, color: '#58a6ff', marginBottom: 6, textTransform: 'capitalize' }}>
                    {grpKey.replace(/_/g, ' ')}
                  </div>
                  <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 10 }}>{grp.mechanism}</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
                    {grp.genes?.map(g => (
                      <span key={g} style={{ background: GENE_COLORS[g] || '#30363d', color: '#fff', padding: '2px 10px', borderRadius: 10, fontSize: 12, fontWeight: 600 }}>{g}</span>
                    ))}
                  </div>
                  <div style={{ fontSize: 13, color: '#c9d1d9' }}>
                    <div>Patients: <strong>{grp.n_patients}</strong></div>
                    {grp.riboflavin_responsive_pct != null && <div>Riboflavin Response: <strong style={{ color: '#1b5e20' }}>{grp.riboflavin_responsive_pct}%</strong></div>}
                    {grp.etfdh_riboflavin_response_pct != null && <div>ETFDH Riboflavin: <strong style={{ color: '#2e7d32' }}>{grp.etfdh_riboflavin_response_pct}%</strong></div>}
                    {grp.cardiac_pct != null && <div>Cardiac: <strong>{grp.cardiac_pct}%</strong></div>}
                    {grp.cardiac_reversal_pct != null && <div>Cardiac Reversal: <strong style={{ color: '#b71c1c' }}>{grp.cardiac_reversal_pct}%</strong></div>}
                    {grp.acads_benign != null && <div>SCAD Benign: <strong style={{ color: '#e65100' }}>{grp.acads_benign ? 'Yes (most cases)' : 'No'}</strong></div>}
                    {grp.hsd10_neurodegeneration_pct != null && <div>HSD10 Neuropathy: <strong style={{ color: '#4a148c' }}>{grp.hsd10_neurodegeneration_pct}%</strong></div>}
                  </div>
                </div>
              ))}
            </div>

            {/* Per-gene breakdown table */}
            <h3 style={{ color: '#e6edf3', marginTop: 24, marginBottom: 12 }}>Per-Gene Outcome Breakdown</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#161b22', borderBottom: '2px solid #30363d' }}>
                    {['Gene', 'N', 'NBS%', 'Cardiac%', 'Rhabdo%', 'Retinopathy%', 'Neuropathy%', 'Hypoglycemia%', 'Riboflavin%', 'Cardiac Reversal%', 'Avg Onset (mo)', 'Avg Dx Delay (mo)'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', textAlign: 'left', color: '#8b949e', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.genes?.map((g, i) => {
                    const row = breakdown.gene_breakdown?.[g];
                    if (!row) return null;
                    return (
                      <tr key={g} style={{ background: i % 2 === 0 ? '#0d1117' : '#161b22', borderBottom: '1px solid #21262d' }}>
                        <td style={{ padding: '10px 12px', fontWeight: 700, color: GENE_COLORS[g] }}>{g}</td>
                        <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>{row.n}</td>
                        <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>{row.nbs_detected_pct}%</td>
                        <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>{row.cardiac_pct}%</td>
                        <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>{row.rhabdo_pct}%</td>
                        <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>{row.retinopathy_pct}%</td>
                        <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>{row.neuropathy_pct}%</td>
                        <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>{row.hypoglycemia_pct}%</td>
                        <td style={{ padding: '10px 12px', color: row.riboflavin_response_pct > 50 ? '#2e7d32' : '#c9d1d9', fontWeight: row.riboflavin_response_pct > 50 ? 700 : 400 }}>{row.riboflavin_response_pct}%</td>
                        <td style={{ padding: '10px 12px', color: row.cardiac_reversal_pct > 30 ? '#b71c1c' : '#c9d1d9', fontWeight: row.cardiac_reversal_pct > 30 ? 700 : 400 }}>{row.cardiac_reversal_pct}%</td>
                        <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>{row.mean_age_onset_m}</td>
                        <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>{row.mean_dx_delay_m}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {activeTab === 'Definitions' && definitions && (
          <div>
            <h2 style={{ color: '#58a6ff', marginBottom: 16 }}>Gene Definitions — Clinical Pearls & Diagnostic Algorithm</h2>

            {/* Cascade diagnostic algorithm */}
            {definitions.key_facts?.cascade_diagnostic_algorithm && (
              <div style={{ background: '#161b22', border: '1px solid #2ea04340', borderRadius: 10, padding: 18, marginBottom: 24 }}>
                <div style={{ fontWeight: 700, color: '#2ea043', marginBottom: 8 }}>🔬 Cascade NBS Diagnostic Algorithm</div>
                <div style={{ fontSize: 13, color: '#c9d1d9', lineHeight: 1.7, whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                  {definitions.key_facts.cascade_diagnostic_algorithm}
                </div>
              </div>
            )}

            {/* Key facts */}
            <h3 style={{ color: '#e6edf3', marginBottom: 12 }}>Key Clinical Facts</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: 12, marginBottom: 24 }}>
              {Object.entries(definitions.key_facts || {}).filter(([k]) => k !== 'cascade_diagnostic_algorithm').map(([k, v]) => (
                <div key={k} style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 600, color: '#58a6ff', marginBottom: 6, fontSize: 13 }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                  <div style={{ fontSize: 12, color: '#8b949e', lineHeight: 1.6 }}>{String(v).substring(0, 400)}{String(v).length > 400 ? '…' : ''}</div>
                </div>
              ))}
            </div>

            {/* Pathognomonic pearls */}
            <h3 style={{ color: '#e6edf3', marginBottom: 12 }}>Pathognomonic Pearls by Gene</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: 12, marginBottom: 24 }}>
              {Object.entries(PATHOGNOMONIC).map(([g, text]) => (
                <div key={g} style={{ background: '#161b22', border: `1px solid ${GENE_COLORS[g]}40`, borderLeft: `4px solid ${GENE_COLORS[g]}`, borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[g], marginBottom: 6 }}>{g}</div>
                  <div style={{ fontSize: 12, color: '#8b949e', lineHeight: 1.6 }}>{text}</div>
                </div>
              ))}
            </div>

            {/* Gene definitions */}
            <h3 style={{ color: '#e6edf3', marginBottom: 12 }}>Detailed Gene Definitions</h3>
            {definitions.genes?.map(g => {
              const def = definitions.definitions?.[g];
              if (!def) return null;
              const isOpen = expandedDef === g;
              return (
                <div key={g} style={{ background: '#161b22', border: `1px solid ${GENE_COLORS[g]}40`, borderRadius: 8, marginBottom: 8 }}>
                  <button onClick={() => setExpandedDef(isOpen ? null : g)}
                    style={{ width: '100%', background: 'none', border: 'none', cursor: 'pointer', padding: '14px 18px', textAlign: 'left', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 700, color: GENE_COLORS[g], fontSize: 15 }}>{g} — {def.full_name}</span>
                    <span style={{ color: '#8b949e', fontSize: 13 }}>{def.protein_size} · {def.locus} · {def.inheritance} {isOpen ? '▲' : '▼'}</span>
                  </button>
                  {isOpen && (
                    <div style={{ padding: '0 18px 18px', fontSize: 13, color: '#8b949e', lineHeight: 1.7 }}>
                      <div style={{ marginBottom: 10 }}><strong style={{ color: '#c9d1d9' }}>Mechanism:</strong> {def.mechanism_detail}</div>
                      {def.phenotype_spectrum && (
                        <div style={{ marginBottom: 10 }}>
                          <strong style={{ color: '#c9d1d9' }}>Phenotype Spectrum:</strong>
                          <ul style={{ margin: '6px 0 0 16px', padding: 0 }}>
                            {Object.entries(def.phenotype_spectrum).map(([ph, desc]) => (
                              <li key={ph} style={{ marginBottom: 4 }}><strong>{ph.replace(/_/g, ' ')}:</strong> {desc}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      <div style={{ marginBottom: 6 }}><strong style={{ color: '#c9d1d9' }}>Diagnostic Pearl:</strong> {def.diagnostic_pearl}</div>
                      <div><strong style={{ color: '#c9d1d9' }}>Treatment Pearl:</strong> {def.treatment_pearl}</div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
