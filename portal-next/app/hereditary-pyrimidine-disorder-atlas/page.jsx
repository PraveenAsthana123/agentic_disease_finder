'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-pyrimidine-disorder-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'TYMP':  '#01579b',  // deep blue — MNGIE; plasma dThd/dUrd PATHOGNOMONIC; HSCT curative; GI dysmotility pentad
  'DPYD':  '#880e4f',  // deep magenta — DPD deficiency; 5-FU pharmacogenomics CPIC Grade A; uridine antidote
  'DPYS':  '#1b5e20',  // dark green — DHP deficiency; dihydrouracil accumulate; asymptomatic subset; NO 5-FU risk
  'UPB1':  '#e65100',  // deep orange — beta-ureidopropionase; beta-alanine DEFICIENT; GABA mechanism; seizures
  'CAD':   '#4a148c',  // deep purple — de novo pyrimidine steps 1-3; URIDINE CURATIVE; megaloblastic anaemia
  'DHODH': '#b71c1c',  // deep red — Miller syndrome; postaxial limb + craniofacial; leflunomide ABSOLUTELY CI
  'RRM2B': '#006064',  // dark teal — p53R2; mtDNA depletion AR / deletions AD PEO; multi-system
  'TK2':   '#3e2723',  // dark brown — myopathic MDDS; dThd+dCyd deoxynucleoside therapy; His90Asn
};

const GENE_INFO = {
  'TYMP':  { full: 'TYMP / Thymidine Phosphorylase / 482aa', locus: '22q13.33', size: '482 aa / 54 kDa (homodimer; cytoplasmic)', inh: 'AR', disease: 'MNGIE — Mitochondrial NeuroGastroIntestinal Encephalomyopathy; plasma dThd >3 µmol/L + dUrd >5 µmol/L PATHOGNOMONIC; TYMP LOF → dThd/dUrd accumulate → imbalanced mitochondrial dNTP pool → mtDNA depletion/deletions → GI smooth muscle + neuronal dysfunction; PENTAD: GI dysmotility + cachexia + peripheral neuropathy + leukoencephalopathy + CPEO/ptosis; median survival ~35y; HSCT ONLY CURATIVE — normalises dThd/dUrd; AVOID thymidine analogues (AZT, d4T); plasma dThd/dUrd by HPLC-MS diagnostic; TP activity <10% in buffy coat' },
  'DPYD':  { full: 'DPYD / Dihydropyrimidine Dehydrogenase / 1025aa', locus: '1p21.3', size: '1025 aa / 111 kDa (homodimer; NADPH-dependent)', inh: 'AR (complete); het = pharmacogenomics', disease: 'COMPLETE DPD DEFICIENCY: seizures + ID + autism features; urine uracil >80 µmol/mmol Cr PATHOGNOMONIC; PHARMACOGENOMICS (CPIC Grade A): *2A (c.1905+1G>A) het → 50% 5-FU dose reduction; homozygous *2A → AVOID 5-FU/capecitabine; c.2846A>T het → 50% reduction; ANTIDOTE: uridine triacetate (Vistogard) within 96h of 5-FU overdose; plasma uracil >16 ng/mL = phenotypic DPD deficiency marker; MANDATORY pre-treatment DPYD testing in EU (EMSO guideline)' },
  'DPYS':  { full: 'DPYS / Dihydropyrimidinase / 414aa', locus: '8q22.3', size: '414 aa / 46 kDa (homotrimer; zinc metalloenzyme)', inh: 'AR', disease: 'DIHYDROPYRIMIDINASE DEFICIENCY — step 2 catabolism block; dihydrouracil + dihydrothymine accumulate in urine; VARIABLE PENETRANCE: substantial asymptomatic subset (detected by urine OA screening); symptomatic: variable seizures + mild-moderate ID; NO 5-FU pharmacogenomics risk (DPD/DPYD intact); DISTINGUISH FROM DPYD: DPYD accumulates uracil+thymine; DPYS accumulates dihydrouracil+dihydrothymine; ureidopropionic acid ABSENT in DPYS (DPYS block prevents its production)' },
  'UPB1':  { full: 'UPB1 / Beta-Ureidopropionase / 404aa', locus: '22q11.23', size: '404 aa / 44 kDa (homotrimer; PLP-independent)', inh: 'AR', disease: 'BETA-UREIDOPROPIONASE DEFICIENCY — step 3 catabolism block; ureidopropionic acid + ureidoisobutyric acid accumulate; BETA-ALANINE DEFICIENT (not produced) → reduced GABA-A agonism + GABA-T inhibition → seizures (proposed mechanism); seizures + ID + hypotonia; BETA-ALANINE SUPPLEMENTATION 100-200 mg/kg/day → seizure improvement (case reports); DISTINGUISH: UPB1 ureidopropionic acid elevated; DPYS ureidopropionic acid absent; NO 5-FU risk' },
  'CAD':   { full: 'CAD / Trifunctional Pyrimidine Enzyme / 2225aa', locus: '2p23.3', size: '2225 aa / 243 kDa (hexamer; CPS2+ATCase+DHOase)', inh: 'AR', disease: 'CAD DEFICIENCY — de novo pyrimidine synthesis steps 1-3 blocked; uridine deficiency (not accumulation) → epileptic encephalopathy + megaloblastic anaemia + hypersegmented neutrophils; URIDINE SUPPLEMENTATION CURATIVE: seizures cease within 48-72h PATHOGNOMONIC response; triacetyl uridine (TAU) preferred; LIFELONG treatment; urine orotic acid LOW/NORMAL (CONTRAST: UMPS has HIGH orotic acid); plasma dihydroorotate elevated; brain MRI normalises with treatment' },
  'DHODH': { full: 'DHODH / Dihydroorotate Dehydrogenase / 395aa', locus: '16q22.2', size: '395 aa / 43 kDa (monomer; inner mitochondrial membrane; CoQ-linked)', inh: 'AR', disease: 'MILLER SYNDROME (POADS — Postaxial Acrofacial Dysostosis); DHODH LOF → transient embryonic pyrimidine starvation (6-10 weeks gestation) → neural crest + limb bud proliferation impaired; POSTAXIAL LIMB DEFECTS: 4th+5th digit/ray aplasia; forearm/fibula shortening; CRANIOFACIAL: malar hypoplasia + lower eyelid COLOBOMA + micrognathia + cleft palate + downslanting PF; NORMAL INTELLIGENCE (postnatal salvage sufficient); DDx: Nager (preaxial/radial) vs Miller (postaxial/ulnar); LEFLUNOMIDE ABSOLUTELY CONTRAINDICATED' },
  'RRM2B': { full: 'RRM2B / Ribonucleotide Reductase M2B (p53R2) / 351aa', locus: '8q23.1', size: '351 aa / 40 kDa (dimer with R1; p53-inducible)', inh: 'AR (biallelic severe) or AD (het PEO)', disease: 'AR: severe multi-organ MDDS8 (neonatal) — mtDNA depletion in muscle+kidney+liver+heart+brain; lactic acidosis + early lethality; AD: adult-onset CPEO + mtDNA multiple deletions; slowly progressive limb weakness; MECHANISM: p53R2 provides dNTPs for mtDNA REPAIR in G0/G1 non-dividing cells; RRM2B LOF → mitochondrial dNTP starvation → mtDNA depletion (AR) or deletions (AD); AVOID: valproate (mitochondrial toxicity); succinylcholine in anaesthesia' },
  'TK2':   { full: 'TK2 / Thymidine Kinase 2 / 234aa', locus: '16q21', size: '234 aa / 26 kDa (homodimer; mitochondrial matrix)', inh: 'AR', disease: 'TK2 DEFICIENCY — Myopathic MDDS4; TK2 phosphorylates dThd+dCyd in mitochondria → dTTP+dCTP for mtDNA; TK2 LOF → dTTP/dCTP depletion in mitochondria → mtDNA depletion in MUSCLE; CNS SPARED (contrast RRM2B AR); progressive proximal weakness + respiratory failure + CPEO (childhood/adult); p.His90Asn recurrent (>30% alleles); DEOXYNUCLEOSIDE THERAPY: dThd 200 mg/kg/day + dCyd 200 mg/kg/day — DISEASE-MODIFYING; CONTRAST TYMP: TK2 BENEFITS from dThd; TYMP HARMED by dThd (opposite)' },
};

function GeneChip({ gene }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span style={{ background: col, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 12, fontWeight: 700, margin: '0 2px' }}>
      {gene}
    </span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 120 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8' }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HeredPyrimidineAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [selGene, setSelGene] = useState(null);

  useEffect(() => {
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' ? 'breakdown'
      : tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    setLoading(true); setErr(null);
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => {
        if (ep === 'overview') setOverview(data);
        else if (ep === 'breakdown') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setErr(String(e)); setLoading(false); });
  }, [tab]);

  const bg = '#0f172a';
  const card = '#1e293b';
  const accent = '#38bdf8';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: accent, margin: 0 }}>
          🧬 Hereditary Pyrimidine Disorder Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 6 }}>
          Complete 8-Gene Reference · TYMP · DPYD · DPYS · UPB1 · CAD · DHODH · RRM2B · TK2
          · 320 patients (8×40) · seeds 2694–2701
        </div>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          MNGIE (TYMP) · DPD Deficiency / 5-FU Pharmacogenomics (DPYD) · DHP Deficiency (DPYS) ·
          Beta-Ureidopropionase Deficiency (UPB1) · CAD Deficiency (CAD) ·
          Miller Syndrome (DHODH) · p53R2 MDDS8 (RRM2B) · Myopathic MDDS4 (TK2)
        </div>
      </div>

      {/* Gene chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 20 }}>
        {Object.keys(GENE_COLORS).map(g => (
          <button key={g} onClick={() => setSelGene(selGene === g ? null : g)}
            style={{ background: selGene === g ? GENE_COLORS[g] : '#1e293b',
              border: `2px solid ${GENE_COLORS[g]}`, borderRadius: 6, color: '#e2e8f0',
              padding: '4px 12px', cursor: 'pointer', fontSize: 13, fontWeight: 700 }}>
            {g}
          </button>
        ))}
      </div>

      {/* Selected gene info */}
      {selGene && GENE_INFO[selGene] && (
        <div style={{ background: '#1e293b', border: `2px solid ${GENE_COLORS[selGene]}`, borderRadius: 10, padding: 16, marginBottom: 20 }}>
          <div style={{ fontWeight: 700, color: GENE_COLORS[selGene], fontSize: 15, marginBottom: 4 }}>
            {GENE_INFO[selGene].full}
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>
            Locus: {GENE_INFO[selGene].locus} · Size: {GENE_INFO[selGene].size} · Inheritance: {GENE_INFO[selGene].inh}
          </div>
          <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>{GENE_INFO[selGene].disease}</div>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{ background: tab === t ? accent : '#1e293b', color: tab === t ? '#0f172a' : '#94a3b8',
              border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400, fontSize: 13 }}>
            {t}
          </button>
        ))}
      </div>

      {loading && <div style={{ color: '#94a3b8' }}>Loading…</div>}
      {err && <div style={{ color: '#ef4444' }}>Error: {err}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40 cohort" />
            <MetricCard label="Genes" value={overview.genes?.length} sub="Catabolism + Synthesis + Salvage" />
            <MetricCard label="Seeds" value={`${overview.seeds?.[0]}–${overview.seeds?.at(-1)}`} sub="deterministic" />
            <MetricCard label="Critical Distinctions" value={overview.critical_distinctions?.length} warn />
          </div>

          {/* Pathway Categories */}
          <div style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>📋 Pyrimidine Disorder Classification</div>
            {overview.pathway_categories?.map((c, i) => (
              <div key={i} style={{ fontSize: 12, color: '#cbd5e1', borderLeft: `3px solid ${accent}`, paddingLeft: 10, marginBottom: 8 }}>
                {c}
              </div>
            ))}
          </div>

          {/* Critical Distinctions */}
          <div style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#ef4444', marginBottom: 10 }}>⚠️ Critical Diagnostic Distinctions</div>
            {overview.critical_distinctions?.map((d, i) => (
              <div key={i} style={{ fontSize: 12, color: '#fca5a5', borderLeft: '3px solid #ef4444', paddingLeft: 10, marginBottom: 6 }}>
                {d}
              </div>
            ))}
          </div>

          {/* Gene summaries table */}
          <div style={{ background: card, borderRadius: 10, padding: 16 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 12 }}>📊 Gene-Level Statistics (n=40 per gene)</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px' }}>Gene</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Onset yr</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Seizures%</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>GI Dysmot%</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>CPEO%</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Leukoenceph%</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Lactic%</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>RespFail%</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>5-FU Risk%</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #1e293b' }}>
                      <td style={{ padding: '5px 8px' }}><GeneChip gene={s.gene} /></td>
                      <td style={{ textAlign: 'right', padding: '5px 8px', color: '#94a3b8' }}>{s.mean_onset_years}</td>
                      <td style={{ textAlign: 'right', padding: '5px 8px', color: s.seizures_pct > 60 ? '#ef4444' : '#e2e8f0' }}>{s.seizures_pct}%</td>
                      <td style={{ textAlign: 'right', padding: '5px 8px', color: s.gi_dysmotility_pct > 60 ? '#ef4444' : '#e2e8f0' }}>{s.gi_dysmotility_pct}%</td>
                      <td style={{ textAlign: 'right', padding: '5px 8px' }}>{s.cpeo_ptosis_pct}%</td>
                      <td style={{ textAlign: 'right', padding: '5px 8px' }}>{s.leukoenceph_pct}%</td>
                      <td style={{ textAlign: 'right', padding: '5px 8px', color: s.lactic_acidosis_pct > 50 ? '#fbbf24' : '#e2e8f0' }}>{s.lactic_acidosis_pct}%</td>
                      <td style={{ textAlign: 'right', padding: '5px 8px', color: s.resp_failure_pct > 50 ? '#ef4444' : '#e2e8f0' }}>{s.resp_failure_pct}%</td>
                      <td style={{ textAlign: 'right', padding: '5px 8px', color: s.fivefu_toxicity_pct > 50 ? '#ef4444' : '#e2e8f0' }}>{s.fivefu_toxicity_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          {breakdown.breakdown?.map((b, i) => (
            <div key={i} style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 16,
              borderLeft: `4px solid ${GENE_COLORS[b.gene] || accent}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8 }}>
                <div>
                  <GeneChip gene={b.gene} />
                  <span style={{ fontSize: 13, color: '#94a3b8', marginLeft: 10 }}>{b.locus} · {b.protein_size}</span>
                </div>
                <div style={{ display: 'flex', gap: 12, fontSize: 12 }}>
                  <span style={{ color: b.seizures_pct > 60 ? '#ef4444' : '#94a3b8' }}>Sz {b.seizures_pct}%</span>
                  <span style={{ color: b.gi_dysmotility_pct > 60 ? '#ef4444' : '#94a3b8' }}>GI {b.gi_dysmotility_pct}%</span>
                  <span style={{ color: b.lactic_acidosis_pct > 50 ? '#fbbf24' : '#94a3b8' }}>Lact {b.lactic_acidosis_pct}%</span>
                  <span style={{ color: b.resp_failure_pct > 50 ? '#ef4444' : '#94a3b8' }}>Resp {b.resp_failure_pct}%</span>
                </div>
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8, lineHeight: 1.5 }}>
                <strong style={{ color: '#e2e8f0' }}>Disease:</strong> {b.disease_category}
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 6, lineHeight: 1.5 }}>
                <strong style={{ color: '#fbbf24' }}>Pathognomonic:</strong> {b.pathognomonic}
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 6, lineHeight: 1.5 }}>
                <strong style={{ color: '#4ade80' }}>Treatment:</strong> {b.treatment_summary}
              </div>
              <div style={{ display: 'flex', gap: 16, marginTop: 8, fontSize: 11, color: '#64748b' }}>
                {b.fivefu_toxicity_pct > 0 && <span>⚠️ 5-FU risk: {b.fivefu_toxicity_pct}%</span>}
                {b.uridine_response_pct > 0 && <span>💊 Uridine resp: {b.uridine_response_pct}%</span>}
                {b.dnucleoside_treatment_pct > 0 && <span>💊 dNucleoside Rx: {b.dnucleoside_treatment_pct}%</span>}
                {b.postaxial_limb_pct > 0 && <span>🦴 Postaxial limb: {b.postaxial_limb_pct}%</span>}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {/* Diagnostic algorithm */}
          <div style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 12 }}>🔬 Pyrimidine Disorder Diagnostic Algorithm</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
              {[
                { condition: 'GI dysmotility + cachexia + leukoenceph + CPEO PENTAD + plasma dThd/dUrd ELEVATED', gene: 'TYMP', dx: 'MNGIE — HSCT curative; AVOID dThd analogues', col: '#01579b' },
                { condition: 'Urine URACIL + THYMINE elevated + infantile seizures/ID OR 5-FU severe toxicity', gene: 'DPYD', dx: 'DPD deficiency — CPIC Grade A; 50% dose reduction or AVOID 5-FU', col: '#880e4f' },
                { condition: 'Urine DIHYDROURACIL + DIHYDROTHYMINE elevated + variable seizures', gene: 'DPYS', dx: 'DHP deficiency — asymptomatic subset; NO 5-FU pharmacogenomics risk', col: '#1b5e20' },
                { condition: 'Urine UREIDOPROPIONIC ACID elevated + BETA-ALANINE absent + seizures', gene: 'UPB1', dx: 'Beta-ureidopropionase — beta-alanine supplementation', col: '#e65100' },
                { condition: 'Epileptic encephalopathy + megaloblastic anaemia + URIDINE RESPONSE dramatic', gene: 'CAD', dx: 'CAD deficiency — uridine/TAU CURATIVE lifelong; orotic acid LOW', col: '#4a148c' },
                { condition: 'POSTAXIAL limb defects (4th+5th ray) + lower eyelid COLOBOMA + malar hypoplasia', gene: 'DHODH', dx: 'Miller syndrome — NORMAL intelligence; leflunomide ABSOLUTELY CI', col: '#b71c1c' },
                { condition: 'MTDNA DEPLETION <30% in muscle OR mtDNA multiple deletions + PEO', gene: 'RRM2B', dx: 'p53R2 MDDS8 — AR (severe neonatal) or AD (adult PEO)', col: '#006064' },
                { condition: 'Progressive MYOPATHY + respiratory failure + COX-NEGATIVE fibres + CNS SPARED', gene: 'TK2', dx: 'TK2 MDDS4 — dThd+dCyd deoxynucleoside therapy', col: '#3e2723' },
              ].map((row, i) => (
                <div key={i} style={{ background: '#0f172a', borderRadius: 8, padding: 12, borderLeft: `4px solid ${row.col}` }}>
                  <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>{row.condition}</div>
                  <div style={{ fontWeight: 700, fontSize: 13 }}><GeneChip gene={row.gene} /></div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', marginTop: 4 }}>{row.dx}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Critical treatment alerts */}
          <div style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#ef4444', marginBottom: 12 }}>⚠️ Critical Treatment Alerts</div>
            {[
              { alert: 'LEFLUNOMIDE / TERIFLUNOMIDE — ABSOLUTELY CONTRAINDICATED in DHODH', detail: 'DHODH is the pharmacological target of leflunomide. Any residual DHODH activity in Miller syndrome patients will be eliminated. Causes further limb/developmental defects in pregnancy. Do not prescribe.', gene: 'DHODH' },
              { alert: 'THYMIDINE ANALOGUES (AZT, d4T, zalcitabine) — ABSOLUTELY CONTRAINDICATED in TYMP', detail: 'MNGIE already has imbalanced mitochondrial dNTP pool. Thymidine analogues further compound dTTP excess → worsening mtDNA depletion. This is the opposite of TK2 where dThd is therapeutic.', gene: 'TYMP' },
              { alert: 'dThd + dCyd DEOXYNUCLEOSIDE THERAPY — BENEFICIAL in TK2, HARMFUL in TYMP', detail: 'TK2: dThd+dCyd bypass TK2 → replenish mitochondrial dTTP/dCTP → mtDNA maintenance improves. TYMP: additional dThd → already accumulated dThd pool worsens → aggravates MNGIE. Same molecule — opposite effects depending on the gene.', gene: 'TK2' },
              { alert: 'DPYD *2A MANDATORY PRE-5-FU TESTING (EMSO/CPIC Grade A)', detail: 'Heterozygous *2A (c.1905+1G>A): 50% dose reduction of 5-FU/capecitabine + TDM. Homozygous: AVOID fluoropyrimidines entirely. Use raltitrexed or irinotecan-based alternatives. PLASMA URACIL >16 ng/mL as phenotypic screening before genotyping available.', gene: 'DPYD' },
              { alert: 'URIDINE SUPPLEMENTATION — CURATIVE in CAD, NOT INDICATED in DPYD/DPYS/UPB1', detail: 'CAD: de novo pyrimidine synthesis blocked → uridine deficiency → supplement exogenous uridine. DPYD/DPYS/UPB1: pyrimidine CATABOLISM blocked → uracil ACCUMULATES (not deficient) → uridine supplementation would worsen overload. Completely opposite pathomechanism.', gene: 'CAD' },
              { alert: 'VALPROATE — MITOCHONDRIAL TOXICITY in RRM2B and TK2', detail: 'VPA inhibits mitochondrial fatty acid oxidation + complex II. In patients with pre-existing mitochondrial dysfunction (RRM2B/TK2 MDDS), VPA can precipitate hepatotoxicity and metabolic crisis. Use LEV or LTG preferentially.', gene: 'RRM2B' },
            ].map((a, i) => (
              <div key={i} style={{ borderLeft: `4px solid #ef4444`, paddingLeft: 12, marginBottom: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 13, color: '#fca5a5' }}>
                  {a.alert} <GeneChip gene={a.gene} />
                </div>
                <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>{a.detail}</div>
              </div>
            ))}
          </div>

          {/* Sample patients from key genes */}
          {breakdown.breakdown?.filter(b => ['TYMP', 'TK2', 'CAD'].includes(b.gene)).map(b => (
            <div key={b.gene} style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, color: GENE_COLORS[b.gene], marginBottom: 8 }}>
                <GeneChip gene={b.gene} /> Sample Patients (n=5)
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}>
                      <th style={{ textAlign: 'left', padding: 4 }}>ID</th>
                      <th style={{ textAlign: 'right', padding: 4 }}>Onset yr</th>
                      <th style={{ textAlign: 'center', padding: 4 }}>Seizures</th>
                      <th style={{ textAlign: 'center', padding: 4 }}>GI Dysmot</th>
                      <th style={{ textAlign: 'center', padding: 4 }}>CPEO</th>
                      <th style={{ textAlign: 'center', padding: 4 }}>Lactic</th>
                      <th style={{ textAlign: 'center', padding: 4 }}>RespFail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {b.sample_patients?.map((p, j) => (
                      <tr key={j} style={{ borderBottom: '1px solid #0f172a' }}>
                        <td style={{ padding: 4, color: '#64748b' }}>{p.patient_id}</td>
                        <td style={{ textAlign: 'right', padding: 4 }}>{p.age_onset_years}</td>
                        <td style={{ textAlign: 'center', padding: 4 }}>{p.seizures ? '🔴' : '—'}</td>
                        <td style={{ textAlign: 'center', padding: 4 }}>{p.gi_dysmotility ? '🟠' : '—'}</td>
                        <td style={{ textAlign: 'center', padding: 4 }}>{p.cpeo_ptosis ? '👁️' : '—'}</td>
                        <td style={{ textAlign: 'center', padding: 4 }}>{p.lactic_acidosis ? '🟡' : '—'}</td>
                        <td style={{ textAlign: 'center', padding: 4 }}>{p.resp_failure ? '🔴' : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 12 }}>📖 Clinical Glossary</div>
            {Object.entries(definitions.glossary || {}).map(([term, def], i) => (
              <div key={i} style={{ borderBottom: '1px solid #1e293b', paddingBottom: 10, marginBottom: 10 }}>
                <div style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 13 }}>{term}</div>
                <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6, marginTop: 4 }}>{def}</div>
              </div>
            ))}
          </div>
          <div style={{ background: card, borderRadius: 10, padding: 16 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 12 }}>🧬 Full Gene Entries</div>
            {Object.entries(definitions.gene_entries || {}).map(([gene, entry], i) => (
              <div key={i} style={{ borderLeft: `4px solid ${GENE_COLORS[gene] || accent}`, paddingLeft: 12, marginBottom: 16 }}>
                <div style={{ fontWeight: 700, fontSize: 13 }}><GeneChip gene={gene} /> {entry.locus} · {entry.protein_size}</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{entry.inheritance?.substring(0, 200)}…</div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{entry.disease_category?.substring(0, 200)}…</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
