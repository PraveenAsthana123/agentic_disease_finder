'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-galactosemia-fructosemia-sugar-metabolism-atlas';
const TABS = ['Overview', 'Gene Table', 'Sugar Metabolism Groups', 'Definitions'];

const GENE_COLORS = {
  'GALT':   '#b71c1c',  // deep red        — AR; Classic galactosemia type I; E.coli sepsis; POI; Q188R
  'GALK1':  '#1565c0',  // deep blue        — AR; type II; cataracts ONLY; galactitol; benign otherwise
  'GALE':   '#6a1b9a',  // deep purple      — AR; type III; epimerase; peripheral benign vs generalized severe
  'GALM':   '#0d47a1',  // dark navy        — AR; type 0; mutarotase; benign NBS finding
  'ALDOB':  '#e65100',  // deep orange      — AR; HFI; fructose-1-P toxic; A149P; fructose-loading CI
  'KHK':    '#2e7d32',  // dark green       — AR; essential fructosuria; benign; no treatment
  'FBP1':   '#880e4f',  // dark magenta     — AR; FBPase; fasting crisis; fructose NOT toxic
  'SLC2A2': '#00695c',  // dark teal        — AR; GLUT2; Fanconi-Bickel; glucosuria+hypoglycemia paradox
};

const GENE_INFO = {
  'GALT':   { full: 'GALT / Galactose-1-phosphate Uridylyltransferase / Classic Galactosemia type I / 379aa', locus: '9p13.3',  inh: 'AR LOF',
               note: '379aa/43kDa; Leloir pathway step 2; Gal-1-P accumulates → hepatotoxicity+cataracts+E.coli sepsis+POI; Q188R 70% Caucasian alleles; N314D=Duarte D2 BENIGN; E.coli neonatal sepsis in breastfed infant = PATHOGNOMONIC pre-NBS; 50-70% females develop POI; stop breastmilk STAT on positive NBS; soy or Nutramigen' },
  'GALK1':  { full: 'GALK1 / Galactokinase 1 / Galactosemia type II / 379aa',                                  locus: '17q25.1', inh: 'AR LOF',
               note: '379aa/42kDa; Leloir step 1; galactitol in lens → bilateral oil-droplet cataracts; CATARACTS ONLY — no intellectual disability, no liver disease, no POI; Gal-1-P NORMAL (upstream block); GALT enzyme NORMAL on NBS; P28T Roma/Traveller founder; galactose-free → cataracts reverse early' },
  'GALE':   { full: 'GALE / UDP-Galactose-4-Epimerase / Galactosemia type III / 348aa',                       locus: '1p36.11', inh: 'AR LOF',
               note: '348aa/38kDa; Leloir step 3 + glycoconjugate synthesis; Type I peripheral = RBCs/WBCs only = BENIGN (no Rx); Type II generalized = severe resembles classic galactosemia; Val94Met = generalized severe; Asn34Ser = peripheral benign; generalized needs galactose restriction but NOT complete (requires some for glycoconjugates)' },
  'GALM':   { full: 'GALM / Galactose Mutarotase / Galactosemia type 0 / 342aa',                              locus: '2p22.3',  inh: 'AR LOF',
               note: '342aa/38kDa; beta→alpha-D-galactose interconversion; mildest galactosemia; NBS elevated galactose only; GALT+GALK normal; CLINICALLY BENIGN in most; <30 cases worldwide; usually no treatment; OMIM 618881' },
  'ALDOB':  { full: 'ALDOB / Aldolase B / Hereditary Fructose Intolerance (HFI) / 364aa',                     locus: '9q22.33', inh: 'AR LOF',
               note: '364aa/40kDa; cleaves fructose-1-P→DHAP+glyceraldehyde; F-1-P ACCUMULATION → phosphate trapping + hepatotoxicity + hypoglycemia; A149P ~80% European alleles; innate FRUCTOSE AVERSION self-protective; FRUCTOSE LOADING TEST ABSOLUTELY CI (lethal); fructose-free diet CURATIVE; galactose NOT toxic (contrast with GALT)' },
  'KHK':    { full: 'KHK / Ketohexokinase / Essential Fructosuria / 298aa',                                   locus: '2p23.3',  inh: 'AR LOF',
               note: '298aa/33kDa; phosphorylates fructose→F-1-P; LOF→fructose not phosphorylated→fructosuria; BENIGN; no hypoglycemia, no liver disease; urine reducing substance +, glucose-oxidase −; 1/130,000; NO TREATMENT' },
  'FBP1':   { full: 'FBP1 / Fructose-1,6-Bisphosphatase 1 / FBPase Deficiency / 338aa',                      locus: '9q22.32', inh: 'AR LOF',
               note: '338aa/37kDa; irreversible gluconeogenesis step: F-1,6-BP→F-6-P; LOF→gluconeogenesis BLOCKED; fasting/illness→hypoglycemia+HIGH LACTATE+HIGH KETONES simultaneously (hyperketotic); glucagon NO response; FRUCTOSE NOT TOXIC (critical DDx from HFI); avoid fasting; raw cornstarch; IV glucose in crisis' },
  'SLC2A2': { full: 'SLC2A2 / GLUT2 / Fanconi-Bickel Syndrome / 524aa',                                      locus: '3q26.2',  inh: 'AR LOF',
               note: '524aa/57kDa; bidirectional low-affinity glucose/galactose/fructose transporter; hepatocytes+renal tubules+beta cells+intestine; LOF→hepatic glycogen accumulation+Renal Fanconi+hypoglycemia despite hepatomegaly+rickets; PATHOGNOMONIC: glucosuria WITH hypoglycemia (glucosuria usually=hyperglycemia); ~100 cases; galactose-free+cornstarch+phosphate' },
};

const PATHOGNOMONIC = {
  'GALT':   'CLASSIC GALACTOSEMIA — E. COLI NEONATAL SEPSIS in breastfed infant before NBS result = PATHOGNOMONIC clinical clue. NBS: elevated Gal-1-P + GALT enzyme <5%. Q188R = most common Caucasian allele. STOP breastmilk IMMEDIATELY on positive NBS. Long-term: ~50-70% females develop PRIMARY OVARIAN INSUFFICIENCY (POI) despite optimal diet — annual gonadotropin surveillance from age 10. DUARTE (N314D): partial enzyme (25-50%), benign — do NOT treat.',
  'GALK1':  'GALACTOKINASE DEFICIENCY — BILATERAL OIL-DROPLET CATARACTS = sole presentation. KEY DDx from GALT: Gal-1-P is NORMAL (GALK1 block is upstream; galactose cannot even reach GALT). GALT enzyme NORMAL on NBS. No intellectual disability, no hepatic disease, no E.coli sepsis, no POI. P28T = Roma/Traveller founder allele (high frequency). Early galactose-free diet → cataracts resorb.',
  'GALE':   'EPIMERASE DEFICIENCY SPECTRUM — Type I PERIPHERAL (>90%): only in RBCs/WBCs; BENIGN; NO treatment. Type II GENERALIZED (rare): severe neonatal galactosemia resembling GALT. Val94Met = generalized severe; Asn34Ser = peripheral benign. UNIQUE: generalized form requires PARTIAL galactose (body needs galactose for glycoconjugate synthesis) — complete restriction causes new deficiency.',
  'GALM':   'GALACTOSE MUTAROTASE (type 0) — MILDEST galactosemia; NBS: mild galactose elevation; GALT and GALK1 enzymes normal. Usually BENIGN and clinically silent. <30 cases worldwide. Most require no dietary treatment. OMIM 618881.',
  'ALDOB':  'HFI (Hereditary Fructose Intolerance) — PATHOGNOMONIC HISTORY: vomiting + hypoglycemia at WEANING (first fructose/sucrose exposure) + innate FRUCTOSE AVERSION (refusal of sweets, excellent dental health). A149P (c.447C>A) ~80% European alleles + A174D ~10%. FRUCTOSE LOADING TEST = ABSOLUTELY CONTRAINDICATED (lethal acute liver failure). ALDOB gene sequencing is the diagnostic standard. Fructose-free diet = CURATIVE. Galactose SAFE (contrast with galactosemia).',
  'KHK':    'ESSENTIAL FRUCTOSURIA — COMPLETELY BENIGN. Urine: reducing substance POSITIVE (copper reduction), glucose-oxidase NEGATIVE = fructosuria. NO hypoglycemia, NO liver disease. KHK deficiency: fructose never phosphorylated → never enters fructose-1-P pathway → no toxicity. NO TREATMENT. Key DDx from HFI: identical fructosuria but KHK has NO symptoms.',
  'FBP1':   'FRUCTOSE-1,6-BISPHOSPHATASE DEFICIENCY — TRIAD: hypoglycemia + HIGH LACTATE + HIGH KETONES simultaneously on fasting/illness = PATHOGNOMONIC. GLUCAGON TEST: no blood glucose response (gluconeogenesis cannot be activated). HYPERKETOTIC hypoglycemia distinguishes from FAOD (which is hypoketotic). CRITICAL DDx FROM HFI: FRUCTOSE IS NOT TOXIC in FBP1 deficiency — do NOT restrict fructose. Raw cornstarch prevents fasting hypoglycemia.',
  'SLC2A2': 'FANCONI-BICKEL (GLUT2 DEFICIENCY) — PATHOGNOMONIC PARADOX: glucosuria + HYPOGLYCEMIA simultaneously. Normal glucosuria implies hyperglycemia — but here glucose is low or normal (tubular transport lost). Hepatomegaly (glycogen storage) + renal Fanconi (glucosuria+aminoaciduria+phosphaturia) + growth retardation + rickets = PATHOGNOMONIC constellation. ~100 cases worldwide. Galactose-free diet + raw cornstarch + phosphate supplementation.',
};

export default function HereditaryGalactosemiaAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#0d1b2a', minHeight: '100vh', fontSize: 18 }}>Loading Hereditary-Galactosemia-Fructosemia-Sugar-Metabolism-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#0d1b2a', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#e6edf3', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#b71c1c 0%,#1565c0 25%,#6a1b9a 50%,#e65100 75%,#00695c 100%)', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>Hereditary Galactosemia · Fructosemia · Sugar Metabolism Disorders · Newborn Screening · Renal Fanconi</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
          🍬 Hereditary-Galactosemia-Fructosemia-Sugar-Metabolism-Atlas
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
          Complete 8-Gene Reference · GALT · GALK1 · GALE · GALM · ALDOB · KHK · FBP1 · SLC2A2 ·
          320 Patients (8×40) · Seeds 4134–4141
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Total Patients', val: overview?.total_patients },
            { label: 'Genes', val: overview?.genes?.length },
            { label: 'E.coli Sepsis Clue', val: 'GALT (pre-NBS)' },
            { label: 'HFI Loading Test', val: 'NEVER (CI)' },
            { label: 'FBP1 Fructose', val: 'NOT Toxic' },
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
            <h2 style={{ color: '#58a6ff', marginBottom: 16 }}>Atlas Overview — Galactosemia · Fructosemia · Sugar Disorders · 8-Gene Reference</h2>

            {/* Key Clinical Alerts */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 14, marginBottom: 24 }}>
              {[
                { color: '#b71c1c', title: '🚨 GALT — E.coli Sepsis Pre-NBS',
                  body: 'Classic galactosemia: E.coli neonatal sepsis in breastfed infant = PATHOGNOMONIC before NBS returns. Stop breastmilk IMMEDIATELY on positive NBS. 50-70% females develop POI despite treatment — annual gonadotropin surveillance.' },
                { color: '#1565c0', title: '👁️ GALK1 — Cataracts ONLY',
                  body: 'Galactokinase deficiency: BILATERAL CATARACTS ONLY. No intellectual disability, no hepatic disease, no E.coli sepsis. Gal-1-P NORMAL (upstream block). Galactose-free diet → cataracts reverse early.' },
                { color: '#e65100', title: '🚫 ALDOB/HFI — NO Fructose Loading',
                  body: 'HFI: fructose-1-P accumulation → hepatotoxicity + hypoglycemia. A149P = 80% European alleles. FRUCTOSE LOADING TEST ABSOLUTELY CONTRAINDICATED. Fructose-free diet curative. Galactose SAFE.' },
                { color: '#880e4f', title: '⚠️ FBP1 — Fructose NOT Toxic',
                  body: 'FBPase deficiency: gluconeogenesis block. Fasting crisis: hypoglycemia + HIGH lactate + HIGH ketones simultaneously. FRUCTOSE NOT TOXIC (unlike HFI) — no fructose restriction needed. Avoid fasting; raw cornstarch.' },
                { color: '#00695c', title: '🔬 SLC2A2 — Glucosuria+Hypoglycemia Paradox',
                  body: 'Fanconi-Bickel (GLUT2): glucosuria WITH hypoglycemia simultaneously = PATHOGNOMONIC paradox. Hepatomegaly + Fanconi syndrome + rickets. ~100 cases worldwide. Galactose-free + phosphate.' },
                { color: '#2e7d32', title: '✅ KHK — Essential Fructosuria BENIGN',
                  body: 'Fructosuria: urine reducing substance +, glucose-oxidase −. COMPLETELY BENIGN. No hypoglycemia, no liver disease. NO treatment required.' },
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
                      <strong>NBS/Dx:</strong> {s.nbs_marker}
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 12 }}>
                      {[
                        ['NBS Detected', s.nbs_detected_pct + '%'],
                        ['Cataracts', s.cataracts_pct + '%'],
                        ['Liver Disease', s.liver_pct + '%'],
                        ['Hypoglycemia', s.hypoglycemia_pct + '%'],
                        ['POI (Females)', s.poi_pct + '%'],
                        ['Fanconi Renal', s.fanconi_pct + '%'],
                      ].map(([k, v]) => (
                        <div key={k} style={{ background: '#0d1117', borderRadius: 4, padding: '3px 8px' }}>
                          <span style={{ color: '#8b949e' }}>{k}: </span>
                          <span style={{ color: '#e6edf3', fontWeight: 600 }}>{v}</span>
                        </div>
                      ))}
                    </div>
                    {(s.neonatal_sepsis_flag || s.poi_flag || s.fanconi_flag || s.fructose_toxic || s.benign_variant || s.cataracts_only || s.fasting_trigger) && (
                      <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        {s.neonatal_sepsis_flag && <span style={{ background: '#b71c1c', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>E.coli Sepsis</span>}
                        {s.poi_flag && <span style={{ background: '#880e4f', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>POI Females</span>}
                        {s.fanconi_flag && <span style={{ background: '#00695c', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>Fanconi Renal</span>}
                        {s.fructose_toxic && <span style={{ background: '#e65100', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>Fructose Toxic</span>}
                        {s.benign_variant && <span style={{ background: '#2e7d32', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>Benign/Mild</span>}
                        {s.cataracts_only && <span style={{ background: '#1565c0', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>Cataracts Only</span>}
                        {s.fasting_trigger && <span style={{ background: '#f57f17', color: '#fff', padding: '2px 8px', borderRadius: 10, fontSize: 10 }}>Fasting Trigger</span>}
                      </div>
                    )}
                    <div style={{ marginTop: 8, fontSize: 11, color: '#8b949e', borderTop: '1px solid #30363d', paddingTop: 6 }}>
                      <strong>Tx:</strong> {s.diet_type?.substring(0, 100)}{s.diet_type?.length > 100 ? '…' : ''}
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
            <h2 style={{ color: '#58a6ff', marginBottom: 16 }}>Gene Reference Table — Galactosemia-Fructosemia-Sugar-Metabolism Atlas</h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#161b22', borderBottom: '2px solid #30363d' }}>
                    {['Gene', 'Full Name', 'Locus', 'Inh', 'NBS/Dx Marker', 'Key Notes', 'Diet/Treatment'].map(h => (
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
                      <td style={{ padding: '10px 12px', color: '#8b949e', maxWidth: 260, fontSize: 12 }}>{info.note}</td>
                      <td style={{ padding: '10px 12px', color: '#c9d1d9', maxWidth: 180, fontSize: 12 }}>
                        {overview.cohort_summary?.[g]?.diet_type || '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── SUGAR METABOLISM GROUPS TAB ── */}
        {activeTab === 'Sugar Metabolism Groups' && breakdown && (
          <div>
            <h2 style={{ color: '#58a6ff', marginBottom: 16 }}>Sugar Metabolism Groups Breakdown</h2>
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
                    {grp.galt_sepsis_pct != null && <div>GALT Neonatal Sepsis: <strong style={{ color: '#b71c1c' }}>{grp.galt_sepsis_pct}%</strong></div>}
                    {grp.galt_poi_pct != null && <div>GALT POI (females): <strong style={{ color: '#880e4f' }}>{grp.galt_poi_pct}%</strong></div>}
                    {grp.galk_cataracts_pct != null && <div>GALK1 Cataracts: <strong style={{ color: '#1565c0' }}>{grp.galk_cataracts_pct}%</strong></div>}
                    {grp.aldob_liver_pct != null && <div>ALDOB Liver Disease: <strong style={{ color: '#e65100' }}>{grp.aldob_liver_pct}%</strong></div>}
                    {grp.fasting_crisis_pct != null && <div>Fasting Crisis: <strong style={{ color: '#f57f17' }}>{grp.fasting_crisis_pct}%</strong></div>}
                    {grp.fanconi_pct != null && <div>Fanconi Syndrome: <strong style={{ color: '#00695c' }}>{grp.fanconi_pct}%</strong></div>}
                    {grp.aldob_toxic != null && <div>Fructose Toxic: <strong style={{ color: '#e65100' }}>Yes (ALDOB)</strong></div>}
                    {grp.khk_benign != null && <div>KHK Benign: <strong style={{ color: '#2e7d32' }}>Yes (essential fructosuria)</strong></div>}
                    {grp.fructose_safe != null && <div>Fructose SAFE in FBP1: <strong style={{ color: '#2e7d32' }}>Yes (critical DDx)</strong></div>}
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
                    {['Gene', 'N', 'NBS%', 'Cataracts%', 'Liver%', 'Hypoglycemia%', 'Neonatal Sepsis%', 'POI%', 'Fanconi%', 'Fasting Crisis%', 'Diet Response%', 'Avg Onset (mo)', 'Avg Dx Delay (mo)'].map(h => (
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
                        <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>{row.cataracts_pct}%</td>
                        <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>{row.liver_pct}%</td>
                        <td style={{ padding: '10px 12px', color: '#c9d1d9' }}>{row.hypoglycemia_pct}%</td>
                        <td style={{ padding: '10px 12px', color: row.neonatal_sepsis_pct > 0 ? '#b71c1c' : '#c9d1d9', fontWeight: row.neonatal_sepsis_pct > 0 ? 700 : 400 }}>{row.neonatal_sepsis_pct}%</td>
                        <td style={{ padding: '10px 12px', color: row.poi_pct > 0 ? '#880e4f' : '#c9d1d9', fontWeight: row.poi_pct > 0 ? 700 : 400 }}>{row.poi_pct}%</td>
                        <td style={{ padding: '10px 12px', color: row.fanconi_pct > 0 ? '#00695c' : '#c9d1d9', fontWeight: row.fanconi_pct > 0 ? 700 : 400 }}>{row.fanconi_pct}%</td>
                        <td style={{ padding: '10px 12px', color: row.fasting_crisis_pct > 50 ? '#f57f17' : '#c9d1d9', fontWeight: row.fasting_crisis_pct > 50 ? 700 : 400 }}>{row.fasting_crisis_pct}%</td>
                        <td style={{ padding: '10px 12px', color: row.dietary_response_pct > 50 ? '#2e7d32' : '#c9d1d9', fontWeight: row.dietary_response_pct > 50 ? 700 : 400 }}>{row.dietary_response_pct}%</td>
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
                <div style={{ fontWeight: 700, color: '#2ea043', marginBottom: 8 }}>🔬 Cascade NBS/Clinical Diagnostic Algorithm</div>
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
