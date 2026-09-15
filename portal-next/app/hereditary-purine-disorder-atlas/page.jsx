'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-purine-disorder-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'HPRT1': '#7b1fa2',  // deep purple — Lesch-Nyhan; SIB PATHOGNOMONIC; hyperuricaemia; allopurinol
  'ADSL':  '#1565c0',  // deep blue — ADSL deficiency; succinylpurines CSF PATHOGNOMONIC; autism+epilepsy
  'ATIC':  '#b71c1c',  // deep red — AICA-ribosiduria; AICAR accumulates; ultraorphan (<5 cases)
  'ADA':   '#00695c',  // dark teal — ADA-SCID; dATP; T+B+NK depleted; STRIMVELIS gene therapy EMA2016
  'PNP':   '#e65100',  // deep orange — PNP deficiency; dGTP; selective T-cell; hypouricaemia PATHOGNOMONIC
  'APRT':  '#4e342e',  // dark brown — 2,8-DHA urolithiasis; radiolucent stones; APRT; allopurinol curative
  'DGUOK': '#1b5e20',  // dark green — hepatocerebral MDDS; mtDNA depletion; rotatory nystagmus; VPA CI
  'PRPS1': '#37474f',  // dark blue-grey — Arts syndrome; PRPP deficiency; SNHL+ataxia+ID+optic atrophy
};

const GENE_INFO = {
  'HPRT1': { full: 'HPRT1 / Hypoxanthine-Guanine Phosphoribosyltransferase / 218aa', locus: 'Xq26.2', size: '218 aa / 25 kDa (homotetramer)', inh: 'XLR', disease: 'LESCH-NYHAN DISEASE — TRIAD: self-injurious behaviour (SIB) + uric acid overproduction + dystonia/neurological; SIB PATHOGNOMONIC — compulsive lip/finger biting; patients request restraints (contra-willful); orange crystals on nappy = urate deposits (first clue); dopaminergic basal ganglia denervation → impaired impulse control; allopurinol controls hyperuricaemia but DOES NOT TREAT NEUROLOGICAL/SIB; gene therapy lentiviral HPRT1 in HSCs (Comet Therapeutics Phase I/II 2024)' },
  'ADSL':  { full: 'ADSL / Adenylosuccinate Lyase / 484aa', locus: '22q13.1', size: '484 aa / 54 kDa (homotetramer; two-reaction)', inh: 'AR', disease: 'ADSL DEFICIENCY — succinylpurines (SAICAr + S-Ado) accumulate → CSF S-Ado:SAICAr PATHOGNOMONIC; NOT on standard metabolic screens (specific succinylpurine HPLC required); Belgian founder Arg426His → milder Type III; severe alleles → Type I/II (profound ID + epilepsy + autism + growth failure); ketogenic diet may reduce seizures + purine synthesis flux; allopurinol reduces SAICAR production (variable benefit)' },
  'ATIC':  { full: 'ATIC / AICAR Transformylase / IMP Cyclohydrolase / 592aa', locus: '2q35', size: '592 aa / 65 kDa (homodimer; bifunctional steps 9-10)', inh: 'AR', disease: 'AICA-RIBOSIDURIA — ATIC LOF → AICAR massively accumulates; urine AICAR PATHOGNOMONIC but only by targeted purine HPLC/MS-MS (NOT standard OA screen); severe refractory epilepsy + profound ID + blindness (optic atrophy) + hypotonia → spasticity; ultraorphan (<5 cases worldwide 2026); METHOTREXATE ABSOLUTELY CONTRAINDICATED (inhibits folate cycle / 10-formyl-THF → worsens step 9 block); uric acid NORMAL/LOW (contrast Lesch-Nyhan)' },
  'ADA':   { full: 'ADA / Adenosine Deaminase / 363aa', locus: '20q13.12', size: '363 aa / 41 kDa (monomer; zinc metalloenzyme)', inh: 'AR', disease: 'ADA-SCID — dATP massively elevated → RRM inhibited → ALL lymphocytes (T+B+NK) deplete; STRIMVELIS EMA 2016 (first approved gene therapy); PEG-ADA (elapegademase) bridges to GT/HSCT; NEUROLOGICAL FEATURES in 40-60% DESPITE immune reconstitution (SNHL 45%, cognitive impairment, ASD 15-20%) — intrinsic neuronal dAdenosine toxicity; AUDIOLOGICAL MONITORING MANDATORY annually in all treated ADA-SCID; STOP PEG-ADA 30-60d before gene therapy; TREC screening on NBS catches presymptomatic' },
  'PNP':   { full: 'PNP / Purine Nucleoside Phosphorylase / 289aa', locus: '14q11.2', size: '289 aa / 32 kDa (homotrimer)', inh: 'AR', disease: 'PNP DEFICIENCY — dGTP elevated → SELECTIVE T-cell immunodeficiency + B cells NORMAL + NK variable (contrast ADA-SCID T+B+NK); HYPOURICAEMIA PATHOGNOMONIC (<1 mg/dL; normal 2.5-7); AUTOIMMUNE features (haemolytic anaemia, ITP) due to unregulated B cells; spastic diplegia + ID in 60-70%; HSCT corrects immunity but neurological features may NOT reverse; FORODESINE (PNP inhibitor used in T-cell lymphoma) ABSOLUTELY CONTRAINDICATED in PNP deficiency; live vaccines contraindicated' },
  'APRT':  { full: 'APRT / Adenine Phosphoribosyltransferase / 180aa', locus: '16q24.3', size: '180 aa / 20 kDa (homodimer; adenine salvage only)', inh: 'AR', disease: '2,8-DIHYDROXYADENINE UROLITHIASIS — APRT LOF → adenine not salvaged → adenine → DHA (via XO); DHA extremely insoluble → precipitates in renal tubules → stones + CKD; RADIOLUCENT on plain X-ray (mistaken for uric acid) but OPAQUE on CT; uric acid NORMAL (critical DDx); ALLOPURINOL CURATIVE — inhibits XO → DHA production ceases; PURELY RENAL — NO neuro/immune features; AZATHIOPRINE + ALLOPURINOL → FATAL myelosuppression (azathioprine metabolised by XO); Japan prevalence 1:27,000 (highest); Type II = milder (p.Met136Thr partial LOF)' },
  'DGUOK': { full: 'DGUOK / Deoxyguanosine Kinase / 277aa', locus: '2p13.1', size: '277 aa / 30 kDa (monomer; mitochondrial matrix)', inh: 'AR', disease: 'HEPATOCEREBRAL MDDS3 — DGUOK LOF → mitochondrial dGTP+dATP deficient → mtDNA depletion in LIVER+BRAIN; ROTATORY NYSTAGMUS early PATHOGNOMONIC feature (distinguishes from other neonatal hepatopathies); combined CI+CIII+CIV deficiency + CII NORMAL = mtDNA depletion pattern; liver transplant corrects hepatic disease but CONTRAINDICATED if neurological features present pre-transplant; VPA ABSOLUTELY CONTRAINDICATED; CONTRAST TK2: TK2 = muscle (pyrimidine); DGUOK = liver+brain (purine); deoxyribonucleoside therapy NOT recommended for DGUOK (contrast TK2)' },
  'PRPS1': { full: 'PRPS1 / Phosphoribosyl Pyrophosphate Synthetase 1 / 318aa', locus: 'Xq22.3', size: '318 aa / 34 kDa (hexamer; PRPP from R5P+ATP)', inh: 'XLR (LOF) / XLR (GOF)', disease: 'ARTS SYNDROME (severe LOF) — PENTAD: profound ID + cerebellar ataxia + severe SNHL + optic atrophy + immunodeficiency; PRPP deficiency → ALL purine+pyrimidine synthesis impaired; inner ear hair cells + optic neurons highest PRPP turnover → SNHL + optic atrophy first; CMTX5 (milder LOF) = neuropathy + SNHL + optic atrophy, intelligence preserved; CONTRAST: PRPS1 SUPERACTIVITY (gain-of-function) → HIGH uric acid + gout + SNHL (OPPOSITE direction); cochlear implants effective for SNHL; HSCT corrects immunodeficiency but NOT neurological/hearing/visual features' },
};

function GeneChip({ gene, active, onClick }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span
      onClick={() => onClick && onClick(gene)}
      style={{
        background: col, color: '#fff', borderRadius: 4,
        padding: '3px 10px', fontSize: 12, fontWeight: 700,
        margin: '0 3px 4px 0', cursor: onClick ? 'pointer' : 'default',
        opacity: active === null || active === gene ? 1 : 0.45,
        border: active === gene ? '2px solid #fff' : '2px solid transparent',
        display: 'inline-block',
      }}
    >{gene}</span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 130 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8' }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function PctBar({ label, pct, color }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#94a3b8', marginBottom: 3 }}>
        <span>{label}</span><span style={{ fontWeight: 700, color: '#e2e8f0' }}>{pct}%</span>
      </div>
      <div style={{ background: '#334155', borderRadius: 4, height: 8 }}>
        <div style={{ background: color || '#38bdf8', width: `${pct}%`, height: 8, borderRadius: 4 }} />
      </div>
    </div>
  );
}

export default function HeredPurineAtlasPage() {
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
  const accent = '#a78bfa';

  const genes = ['HPRT1', 'ADSL', 'ATIC', 'ADA', 'PNP', 'APRT', 'DGUOK', 'PRPS1'];

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: accent, margin: 0 }}>
          🧬 Hereditary Purine Disorder Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 6 }}>
          Complete 8-Gene Reference · HPRT1 · ADSL · ATIC · ADA · PNP · APRT · DGUOK · PRPS1
          · 320 patients (8×40) · seeds 2702–2709
        </div>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          Lesch-Nyhan (HPRT1) · ADSL Deficiency · AICA-Ribosiduria (ATIC) ·
          ADA-SCID (ADA) · PNP Deficiency · 2,8-DHA Urolithiasis (APRT) ·
          Hepatocerebral MDDS (DGUOK) · Arts Syndrome / CMTX5 (PRPS1)
        </div>
      </div>

      {/* Gene chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 20 }}>
        {genes.map(g => (
          <GeneChip key={g} gene={g} active={selGene} onClick={g2 => setSelGene(selGene === g2 ? null : g2)} />
        ))}
        {selGene && (
          <span onClick={() => setSelGene(null)} style={{ fontSize: 11, color: '#64748b', cursor: 'pointer', marginLeft: 8, alignSelf: 'center' }}>
            ✕ clear
          </span>
        )}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              background: tab === t ? accent : '#1e293b',
              color: tab === t ? '#0f172a' : '#94a3b8',
              border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer',
              fontWeight: tab === t ? 700 : 400, fontSize: 13,
            }}
          >{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#64748b' }}>Loading…</div>}
      {err && <div style={{ color: '#ef4444' }}>Error: {err}</div>}

      {/* OVERVIEW */}
      {tab === 'Overview' && overview && !loading && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2702-2709" />
            <MetricCard label="Genes" value={overview.genes?.length} sub="Purine disorder atlas" />
            <MetricCard label="Pathway Categories" value={overview.pathway_categories?.length} />
            <MetricCard label="Critical Distinctions" value={overview.critical_distinctions?.length} warn />
          </div>

          {/* Gene Summaries grid */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: accent, marginBottom: 12 }}>Gene Summary Table</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene', 'Locus', 'Onset (yrs)', 'Self-Inj%', 'UA↑%', 'Dystonia%', 'ID%', 'Seizures%', 'Stones%', 'Immune%', 'Hepato%', 'SNHL%', 'Ataxia%', 'Optic%', 'Neuro%', 'Lactate%'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', color: '#64748b', textAlign: 'left', borderBottom: '1px solid #334155', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.gene_summaries || []).filter(g => !selGene || g.gene === selGene).map(gs => (
                    <tr key={gs.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                      <td style={{ padding: '6px 8px' }}>
                        <GeneChip gene={gs.gene} active={selGene} onClick={g => setSelGene(selGene === g ? null : g)} />
                      </td>
                      <td style={{ padding: '6px 8px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{gs.locus}</td>
                      <td style={{ padding: '6px 8px', color: '#e2e8f0', fontWeight: 700 }}>{gs.mean_onset_years}</td>
                      <td style={{ padding: '6px 8px', color: gs.self_injurious_pct > 50 ? '#ef4444' : '#64748b' }}>{gs.self_injurious_pct}</td>
                      <td style={{ padding: '6px 8px', color: gs.hyperuricaemia_pct > 50 ? '#f97316' : '#64748b' }}>{gs.hyperuricaemia_pct}</td>
                      <td style={{ padding: '6px 8px' }}>{gs.dystonia_pct}</td>
                      <td style={{ padding: '6px 8px', color: gs.id_severe_pct > 60 ? '#f59e0b' : '#64748b' }}>{gs.id_severe_pct}</td>
                      <td style={{ padding: '6px 8px', color: gs.seizures_pct > 50 ? '#f59e0b' : '#64748b' }}>{gs.seizures_pct}</td>
                      <td style={{ padding: '6px 8px', color: gs.renal_stones_pct > 50 ? '#06b6d4' : '#64748b' }}>{gs.renal_stones_pct}</td>
                      <td style={{ padding: '6px 8px', color: gs.immunodeficiency_pct > 50 ? '#ef4444' : '#64748b' }}>{gs.immunodeficiency_pct}</td>
                      <td style={{ padding: '6px 8px', color: gs.hepatopathy_pct > 60 ? '#ef4444' : '#64748b' }}>{gs.hepatopathy_pct}</td>
                      <td style={{ padding: '6px 8px', color: gs.snhl_pct > 50 ? '#a78bfa' : '#64748b' }}>{gs.snhl_pct}</td>
                      <td style={{ padding: '6px 8px' }}>{gs.ataxia_pct}</td>
                      <td style={{ padding: '6px 8px', color: gs.optic_atrophy_pct > 40 ? '#818cf8' : '#64748b' }}>{gs.optic_atrophy_pct}</td>
                      <td style={{ padding: '6px 8px' }}>{gs.neuropathy_pct}</td>
                      <td style={{ padding: '6px 8px', color: gs.lactic_acidosis_pct > 60 ? '#ef4444' : '#64748b' }}>{gs.lactic_acidosis_pct}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pathway Categories */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: accent, marginBottom: 10 }}>Pathway Categories</div>
            {(overview.pathway_categories || []).map((pc, i) => {
              const gene = pc.split('—')[0].split('—')[0].match(/\b(HPRT1|ADSL|ATIC|ADA|PNP|APRT|DGUOK|PRPS1)\b/)?.[1];
              if (selGene && gene && gene !== selGene) return null;
              return (
                <div key={i} style={{ background: card, borderRadius: 8, padding: 12, marginBottom: 8, borderLeft: `4px solid ${GENE_COLORS[gene] || '#64748b'}`, fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
                  {gene && <GeneChip gene={gene} />}
                  <span style={{ marginLeft: 6 }}>{pc}</span>
                </div>
              );
            })}
          </div>

          {/* Critical Distinctions */}
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#ef4444', marginBottom: 10 }}>
              ⚠ Critical Distinctions
            </div>
            {(overview.critical_distinctions || []).map((cd, i) => (
              <div key={i} style={{ background: '#1a0a0a', border: '1px solid #7f1d1d', borderRadius: 8, padding: 12, marginBottom: 8, fontSize: 12, color: '#fca5a5', lineHeight: 1.6 }}>
                {cd}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* GENE TABLE */}
      {tab === 'Gene Table' && breakdown && !loading && (
        <div>
          {(breakdown.genes || []).filter(g => !selGene || g.gene === selGene).map(g => (
            <div key={g.gene} style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 20, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#64748b'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                <GeneChip gene={g.gene} />
                <span style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0' }}>{GENE_INFO[g.gene]?.full}</span>
              </div>
              <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 12 }}>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>Locus: <b style={{ color: '#e2e8f0' }}>{g.locus}</b></span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>Inheritance: <b style={{ color: '#e2e8f0' }}>{GENE_INFO[g.gene]?.inh}</b></span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>Patients: <b style={{ color: '#38bdf8' }}>{g.n_patients}</b></span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>{g.protein_size}</span>
              </div>
              <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, marginBottom: 10 }}>{GENE_INFO[g.gene]?.disease}</div>

              {/* Phenotype bars */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0 20px' }}>
                {[
                  { label: 'Self-Injurious Behaviour', key: 'self_injurious', color: '#ef4444' },
                  { label: 'Hyperuricaemia', key: 'hyperuricaemia', color: '#f97316' },
                  { label: 'Renal Stones', key: 'renal_stones', color: '#06b6d4' },
                  { label: 'Immunodeficiency', key: 'immunodeficiency', color: '#ef4444' },
                  { label: 'Hepatopathy', key: 'hepatopathy', color: '#f59e0b' },
                  { label: 'Seizures', key: 'seizures', color: '#fbbf24' },
                  { label: 'Severe ID', key: 'id_severe', color: '#a78bfa' },
                  { label: 'SNHL', key: 'snhl', color: '#818cf8' },
                  { label: 'Optic Atrophy', key: 'optic_atrophy', color: '#c4b5fd' },
                  { label: 'Ataxia', key: 'ataxia', color: '#34d399' },
                  { label: 'Dystonia', key: 'dystonia', color: '#6ee7b7' },
                  { label: 'Lactic Acidosis', key: 'lactic_acidosis', color: '#f87171' },
                ].map(({ label, key, color }) => {
                  const pts = g.patients || [];
                  const pct = pts.length ? Math.round(100 * pts.filter(p => p[key]).length / pts.length) : 0;
                  if (pct === 0) return null;
                  return <PctBar key={key} label={label} pct={pct} color={color} />;
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* CLINICAL ATLAS */}
      {tab === 'Clinical Atlas' && breakdown && !loading && (
        <div>
          {(breakdown.genes || []).filter(g => !selGene || g.gene === selGene).map(g => (
            <div key={g.gene} style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 20, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#64748b'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                <GeneChip gene={g.gene} />
                <span style={{ fontSize: 15, fontWeight: 700, color: '#e2e8f0' }}>{g.locus} · {GENE_INFO[g.gene]?.inh}</span>
              </div>

              {[
                { label: '🧬 Inheritance & Mechanism', text: g.inheritance },
                { label: '🏥 Disease Category', text: g.disease_category },
                { label: '⚗️ Pathway & Pathomechanism', text: g.disease_pathway },
                { label: '🔬 Pathognomonic Features', text: g.pathognomonic, warn: true },
                { label: '💊 Treatment', text: g.treatment },
              ].map(({ label, text, warn }) => (
                <div key={label} style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: warn ? '#ef4444' : accent, marginBottom: 6 }}>{label}</div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.75, whiteSpace: 'pre-wrap', background: '#0f172a', borderRadius: 6, padding: 10 }}>
                    {text}
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS */}
      {tab === 'Definitions' && definitions && !loading && (
        <div>
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: accent, marginBottom: 12 }}>Per-Gene Reference</div>
            {Object.entries(definitions.gene_entries || {})
              .filter(([gene]) => !selGene || gene === selGene)
              .map(([gene, info]) => (
                <div key={gene} style={{ background: card, borderRadius: 10, padding: 18, marginBottom: 16, borderLeft: `5px solid ${GENE_COLORS[gene] || '#64748b'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                    <GeneChip gene={gene} />
                    <span style={{ fontSize: 12, color: '#94a3b8' }}>{info.locus} · {info.protein_size}</span>
                  </div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7 }}>{info.pathognomonic}</div>
                </div>
              ))}
          </div>

          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: accent, marginBottom: 12 }}>Glossary</div>
            {Object.entries(definitions.glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: card, borderRadius: 8, padding: 14, marginBottom: 10 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#fbbf24', marginBottom: 6 }}>{term}</div>
                <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7 }}>{def}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
