'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-scid-rag-il2rg-ada-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'IL2RG':  '#b71c1c',  // deep red         -- X-SCID most common; PCP infant boy PATHOGNOMONIC; gene therapy approved
  'JAK3':   '#c62828',  // dark red          -- AR phenocopy X-SCID; both sexes; JAK3 Western blot
  'RAG1':   '#1565c0',  // dark blue         -- T-B-NK+ SCID null; Omenn hypomorphic; 11p13 co-test
  'RAG2':   '#1976d2',  // medium blue       -- T-B-NK+ SCID null; Omenn hypomorphic; 11p13 adjacent RAG1
  'DCLRE1C':'#4a148c',  // deep purple       -- Artemis radiosensitive SCID; TBI absolutely CI; Navajo founder
  'ADA':    '#e65100',  // deep orange       -- metabolic dATP toxic; PEG-ADA; Strimvelis gene therapy EU 2016
  'IL7R':   '#00695c',  // dark teal         -- T-B+NK+ SCID; IL-7 T-cell development only; B and NK normal
  'AK2':    '#37474f',  // dark slate        -- Reticular Dysgenesis; SCID+agranulocytosis; most severe; URGENT HSCT
};

const GENE_INFO = {
  'IL2RG':  { full: 'IL2RG-256aa / XLR / Common-gamma-chain-γc / IL-2-4-7-9-15-21-receptor-shared / Xq13.1 / T-B+NK--SCID / PCP-infant-boy-PATHOGNOMONIC / Gene-therapy-Strimvelis-EU2016-approved / BCG-Absolutely-CI',              locus: 'Xq13.1',   size: '256 aa / 29 kDa',   inh: 'X-linked recessive' },
  'JAK3':   { full: 'JAK3-1124aa / AR / Janus-kinase-3-γc-chain-exclusive-kinase / 19p13.11 / T-B+NK--SCID / FEMALE=JAK3-not-IL2RG / AR-both-sexes / JAK3-Western-blot-absent-diagnostic / BCG-Absolutely-CI',                          locus: '19p13.11',  size: '1124 aa / 125 kDa', inh: 'Autosomal recessive' },
  'RAG1':   { full: 'RAG1-1043aa / AR / Recombination-activating-gene-1 / V(D)J-recombination-endonuclease / 11p13 / T-B-NK+-SCID-null / Omenn-syndrome-hypomorphic / RAG2-cotesting-MANDATORY / BCG-Absolutely-CI',                     locus: '11p13',    size: '1043 aa / 119 kDa', inh: 'Autosomal recessive' },
  'RAG2':   { full: 'RAG2-527aa / AR / Recombination-activating-gene-2 / V(D)J-recombination-partner / 11p13 / T-B-NK+-SCID-null / Omenn-syndrome-hypomorphic / RAG1-cotesting-MANDATORY / Same-phenotype-as-RAG1 / BCG-Absolutely-CI',  locus: '11p13',    size: '527 aa / 59 kDa',   inh: 'Autosomal recessive' },
  'DCLRE1C':{ full: 'DCLRE1C-692aa / AR / Artemis-nuclease / V(D)J-hairpin-opening-NHEJ / 10p13 / T-B-NK+-Radiosensitive-SCID / TBI-ABSOLUTELY-CI / RIC-Without-TBI-mandatory / Navajo-p.Thr347Ile-founder / BCG-Absolutely-CI',         locus: '10p13',    size: '692 aa / 77 kDa',   inh: 'Autosomal recessive' },
  'ADA':    { full: 'ADA-363aa / AR / Adenosine-deaminase / Purine-catabolism-dATP-toxic / 20q13.11 / T-B-NK--SCID / PEG-ADA-enzyme-replacement-LIFE-SAVING / Strimvelis-FIRST-GENE-THERAPY-EU2016 / Calcifications-SNHL-50pct-PATHOGNOMONIC', locus: '20q13.11', size: '363 aa / 41 kDa',   inh: 'Autosomal recessive' },
  'IL7R':   { full: 'IL7R-459aa / AR / IL-7-receptor-alpha-chain / T-cell-thymic-development-only / 5p13.2 / T-B+NK+-SCID / T-lymphopenia-ONLY / B-and-NK-NORMAL / NK-present-DDx-from-IL2RG-JAK3 / BCG-Absolutely-CI',                  locus: '5p13.2',   size: '459 aa / 50 kDa',   inh: 'Autosomal recessive' },
  'AK2':    { full: 'AK2-239aa / AR / Adenylate-kinase-2-mitochondrial / Reticular-Dysgenesis / 1p34.1 / T-B-NK--SCID+Agranulocytosis / Most-severe-SCID / Death-weeks-bacterial-sepsis / SNHL-60pct-PATHOGNOMONIC / URGENT-HSCT',         locus: '1p34.1',   size: '239 aa / 26 kDa',   inh: 'Autosomal recessive' },
};

const PHENOTYPE_ROWS = [
  { gene: 'IL2RG',  group: 'γc Chain (IL2RG/JAK3)',      t: '✓✓✓', nk: '✓✓✓', b_absent: '✗',  agranulocytosis: '✗', radiosens: '✗', calcif: '✗', snhl: '±',  gt: '✓✓✓', hsct: '✓✓✓', unique: 'X-SCID MOST COMMON 45-50%: PCP infant boy PATHOGNOMONIC. T-B+NK- SCID (NK absent because IL-15/γc needed for NK). ALL MALES (X-linked). Gene therapy APPROVED (Strimvelis® EU, lentiviral USA). NK absent = key DDx from CD3/IL7R (NK present).' },
  { gene: 'JAK3',   group: 'γc Chain (IL2RG/JAK3)',      t: '✓✓✓', nk: '✓✓✓', b_absent: '✗',  agranulocytosis: '✗', radiosens: '✗', calcif: '✗', snhl: '±',  gt: '✗',    hsct: '✓✓✓', unique: 'AR PHENOCOPY X-SCID: T-B+NK- SCID IDENTICAL to IL2RG BOTH SEXES. FEMALE with T-B+NK- SCID = JAK3. JAK3 Western blot absent = diagnostic. Same clinical urgency as X-SCID. HSCT same approach. No approved gene therapy yet.' },
  { gene: 'RAG1',   group: 'V(D)J Recombination',        t: '✓✓✓', nk: '✗',    b_absent: '✓✓✓', agranulocytosis: '✗', radiosens: '✗', calcif: '✗', snhl: '✗',  gt: '±',    hsct: '✓✓✓', unique: 'T-B-NK+ SCID (NK PRESENT = V(D)J defect): null → complete SCID; hypomorphic → Omenn (erythroderma + eosinophilia + hepatosplenomegaly + oligoclonal T cells). RAG1+RAG2 COTESTING MANDATORY (adjacent 11p13). Omenn: immunosuppression before HSCT.' },
  { gene: 'RAG2',   group: 'V(D)J Recombination',        t: '✓✓✓', nk: '✗',    b_absent: '✓✓✓', agranulocytosis: '✗', radiosens: '✗', calcif: '✗', snhl: '✗',  gt: '±',    hsct: '✓✓✓', unique: 'T-B-NK+ SCID IDENTICAL to RAG1: only sequencing distinguishes. RAG2 null → complete SCID; hypomorphic → Omenn. 11p13 same locus as RAG1 — ALWAYS co-test. RAG2 PHD domain reads H3K4me3 — chromatin targeting for V(D)J. Gene therapy (lentiviral RAG2) in clinical trials.' },
  { gene: 'DCLRE1C',group: 'V(D)J Recombination',        t: '✓✓✓', nk: '✗',    b_absent: '✓✓✓', agranulocytosis: '✗', radiosens: '✓✓✓', calcif: '✗', snhl: '±', gt: '✗',    hsct: '✓',   unique: 'T-B-NK+ SCID + RADIOSENSITIVITY: TBI ABSOLUTELY CONTRAINDICATED. RIC without TBI (busulfan reduced dose) mandatory. Navajo/Apache Native American: p.Thr347Ile founder. ~10% T-B-NK+ SCID. Radiosensitivity colony assay MANDATORY before conditioning.' },
  { gene: 'ADA',    group: 'Metabolic / Cytokine',       t: '✓✓✓', nk: '✓✓✓', b_absent: '✓',   agranulocytosis: '✗', radiosens: '✗', calcif: '✓✓✓', snhl: '✓✓', gt: '✓✓✓', hsct: '✓✓',  unique: 'METABOLIC SCID: dATP toxic kills ALL lymphocytes (T-B-NK-). PERIARTICULAR CALCIFICATIONS + SNHL 50% PATHOGNOMONIC. PEG-ADA enzyme replacement LIFE-SAVING bridge. GENE THERAPY APPROVED (Strimvelis® EU 2016 — FIRST GENE THERAPY EVER APPROVED). Avoids GvHD.' },
  { gene: 'IL7R',   group: 'Metabolic / Cytokine',       t: '✓✓✓', nk: '✗',    b_absent: '✗',   agranulocytosis: '✗', radiosens: '✗', calcif: '✗', snhl: '✗',  gt: '✗',    hsct: '✓✓✓', unique: 'T-B+NK+ SCID: T LYMPHOPENIA ONLY — B and NK COMPLETELY NORMAL. IL-7 required ONLY for thymic T cell development (not B or NK). NK PRESENT = key DDx from IL2RG/JAK3 (NK absent). Next: CD3 flow (IL7R: CD3 present; CD3 complex defects: CD3 absent).' },
  { gene: 'AK2',    group: 'Mitochondrial / Reticular',  t: '✓✓✓', nk: '✓✓✓', b_absent: '✓',   agranulocytosis: '✓✓✓', radiosens: '✗', calcif: '✗', snhl: '✓✓', gt: '✗',   hsct: '✓',   unique: 'RETICULAR DYSGENESIS MOST SEVERE SCID: SCID + AGRANULOCYTOSIS PATHOGNOMONIC. Death from bacterial SEPSIS in weeks (not PCP — no neutrophils). URGENT HSCT in first weeks. SNHL 60% (AK2 in stria vascularis). Broad-spectrum antibiotics + antifungals from birth. No enzyme replacement.' },
];

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info, cohort }) {
  const isGammaC  = ['IL2RG', 'JAK3'].includes(gene);
  const isVDJ     = ['RAG1', 'RAG2', 'DCLRE1C'].includes(gene);
  const isMetabol = ['ADA', 'IL7R'].includes(gene);
  const isMito    = gene === 'AK2';
  const groupLabel = isGammaC  ? 'γc Chain / JAK3 Defect'
    : isVDJ     ? 'V(D)J Recombination Defect'
    : isMetabol ? 'Metabolic / Cytokine Receptor'
    : 'Mitochondrial / Reticular Dysgenesis';
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          <span style={{ fontWeight: 700, fontSize: 18, color }}>{gene}</span>
          <span style={{ marginLeft: 10, fontSize: 12, background: color, color: '#fff', padding: '2px 8px', borderRadius: 4 }}>{groupLabel}</span>
        </div>
        <div style={{ fontSize: 12, color: '#888', textAlign: 'right' }}>
          <div>{info.size} · {info.locus}</div>
          <div>{info.inh}</div>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#bbb', marginTop: 6 }}>{info.full}</div>
      {cohort && (
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Badge text={`T-Absent ${cohort.t_absent_pct}%`}               color={color} />
          <Badge text={`HSCT ${cohort.hsct_pct}%`}                       color='#1565c0' />
          {cohort.agranulocytosis_pct > 0 && <Badge text={`Agranulocytosis ${cohort.agranulocytosis_pct}%`}  color='#37474f' />}
          {cohort.calcifications_pct > 0  && <Badge text={`Calcifications ${cohort.calcifications_pct}%`}   color='#e65100' />}
          {cohort.snhl_pct > 0            && <Badge text={`SNHL ${cohort.snhl_pct}%`}                       color='#5d4037' />}
          {cohort.radiosensitivity_pct > 0 && <Badge text={`Radiosens ${cohort.radiosensitivity_pct}%`}     color='#4a148c' />}
          {cohort.peg_ada_pct > 0         && <Badge text={`PEG-ADA ${cohort.peg_ada_pct}%`}                 color='#e65100' />}
          {cohort.gene_therapy_pct > 0    && <Badge text={`Gene Therapy ${cohort.gene_therapy_pct}%`}       color='#2e7d32' />}
          {cohort.omenn_pct > 0           && <Badge text={`Omenn ${cohort.omenn_pct}%`}                     color='#880e4f' />}
          <Badge text={`Dx age ${cohort.mean_age_at_dx_months}mo`}        color='#455a64' />
        </div>
      )}
    </div>
  );
}

export default function HereditorySCIDRagIL2RGAdaAtlasPage() {
  const [tab, setTab]           = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]         = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov); setBreakdown(bk); setDefs(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ color: '#fff', padding: 32 }}>Loading Hereditary-SCID-RAG-IL2RG-ADA-Atlas…</div>;
  if (error)   return <div style={{ color: '#f44', padding: 32 }}>Error: {error}</div>;

  const stats = overview?.aggregate_stats || {};
  const cohortByGene = {};
  (breakdown?.gene_cohorts || []).forEach(c => { cohortByGene[c.gene] = c; });

  return (
    <div style={{ background: '#121212', minHeight: '100vh', color: '#eee', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: '#1a0000', borderBottom: '2px solid #b71c1c', padding: '20px 32px' }}>
        <h1 style={{ color: '#ef9a9a', fontSize: 22, marginBottom: 4 }}>
          🧬 Hereditary-SCID-RAG-IL2RG-ADA-Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#aaa' }}>
          Complete 8-Gene Classic SCID Reference · IL2RG · JAK3 · RAG1 · RAG2 · DCLRE1C · ADA · IL7R · AK2 ·
          320-Patient Aggregate (8×40) · Seeds 3958-3965
        </div>
        <div style={{ marginTop: 8, display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 12 }}>
          <span style={{ background: '#b71c1c', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>T-Absent {stats.t_absent_pct}%</span>
          <span style={{ background: '#880e4f', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>NK Absent {stats.nk_absent_pct}%</span>
          <span style={{ background: '#1565c0', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>HSCT {stats.hsct_pct}%</span>
          <span style={{ background: '#2e7d32', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Gene Therapy {stats.gene_therapy_pct}%</span>
          <span style={{ background: '#37474f', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Agranulocytosis (AK2) {stats.agranulocytosis_pct}%</span>
          <span style={{ background: '#e65100', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>PEG-ADA (ADA) {stats.peg_ada_pct}%</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid #333', background: '#1a1a1a' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{
              background: tab === t ? '#b71c1c' : 'transparent', color: tab === t ? '#fff' : '#aaa',
              border: 'none', padding: '10px 22px', cursor: 'pointer', fontSize: 13, fontFamily: 'monospace'
            }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: 24 }}>

        {/* ── OVERVIEW TAB ── */}
        {tab === 'Overview' && (
          <div>
            {/* Key Facts */}
            <div style={{ background: '#1a0000', border: '1px solid #b71c1c', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#ef9a9a', marginBottom: 10 }}>Key Clinical Facts — 8 Classic SCID Genes</div>
              {(overview?.key_facts || []).map((f, i) => (
                <div key={i} style={{ fontSize: 12, color: '#ddd', marginBottom: 6, paddingLeft: 12, borderLeft: '3px solid #b71c1c' }}>{f}</div>
              ))}
            </div>

            {/* NK TRIAGE — Critical Clinical Decision Tool */}
            <div style={{ background: '#0a1628', border: '1px solid #1565c0', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 12 }}>NK Cell Count — SCID Triage Algorithm</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                {[
                  { label: 'T-B+NK- SCID', genes: 'IL2RG / JAK3', desc: 'γc chain absent → NK absent (IL-15 needs γc). Male = IL2RG first. Female = JAK3.', color: '#b71c1c' },
                  { label: 'T-B-NK+ SCID', genes: 'RAG1 / RAG2 / DCLRE1C', desc: 'V(D)J defect → B absent too. NK present (germline receptors). Omenn if hypomorphic.', color: '#1565c0' },
                  { label: 'T-B-NK- SCID', genes: 'ADA / AK2', desc: 'ADA: dATP toxic to all lymphocytes. AK2: mitochondrial + agranulocytosis (reticular dysgenesis).', color: '#e65100' },
                  { label: 'T-B+NK+ SCID', genes: 'IL7R (+ CD3 complex)', desc: 'T lymphopenia ONLY. B and NK normal. IL-7 required only for T cell thymic development.', color: '#00695c' },
                ].map(g => (
                  <div key={g.label} style={{ background: '#0d1117', border: `1px solid ${g.color}`, borderRadius: 6, padding: 12 }}>
                    <div style={{ fontWeight: 700, color: g.color, fontSize: 13 }}>{g.label}</div>
                    <div style={{ fontSize: 12, color: '#90caf9', margin: '4px 0' }}>{g.genes}</div>
                    <div style={{ fontSize: 11, color: '#aaa' }}>{g.desc}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Gene Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(480px, 1fr))', gap: 12 }}>
              {(overview?.genes || []).map(gene => (
                <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]}
                  info={GENE_INFO[gene]} cohort={cohortByGene[gene]} />
              ))}
            </div>

            {/* Aggregate Stats */}
            <div style={{ marginTop: 20, background: '#1a1a1a', border: '1px solid #333', borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: '#ef9a9a', marginBottom: 10 }}>Aggregate Statistics — 320 Patients (8×40)</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 8 }}>
                {[
                  ['T-Absent', stats.t_absent_pct, '%', '#b71c1c'],
                  ['NK Absent', stats.nk_absent_pct, '%', '#880e4f'],
                  ['HSCT Done', stats.hsct_pct, '%', '#1565c0'],
                  ['Gene Therapy', stats.gene_therapy_pct, '%', '#2e7d32'],
                  ['Agranulocytosis', stats.agranulocytosis_pct, '%', '#37474f'],
                  ['Calcifications', stats.calcifications_pct, '%', '#e65100'],
                  ['SNHL', stats.snhl_pct, '%', '#5d4037'],
                  ['Radiosensitivity', stats.radiosensitivity_pct, '%', '#4a148c'],
                  ['PEG-ADA Received', stats.peg_ada_pct, '%', '#f57f17'],
                  ['Omenn Syndrome', stats.omenn_syndrome_pct, '%', '#880e4f'],
                  ['BCG Reaction', stats.bcg_reaction_pct, '%', '#b71c1c'],
                ].map(([label, val, unit, color]) => (
                  <div key={label} style={{ background: '#111', border: `1px solid ${color}`, borderRadius: 6, padding: 10 }}>
                    <div style={{ fontSize: 11, color: '#aaa' }}>{label}</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color }}>{val}{unit}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── GENE TABLE TAB ── */}
        {tab === 'Gene Table' && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <thead>
                <tr style={{ background: '#1a0000' }}>
                  {['Gene', 'Group', 'T-Absent', 'NK-Absent', 'B-Absent', 'Agranulocytosis', 'Radiosens', 'Calcif', 'SNHL', 'Gene Rx', 'HSCT', 'Clinical Pearl'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', color: '#ef9a9a', textAlign: 'left', borderBottom: '1px solid #444' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#161616' : '#1a1a1a' }}>
                    <td style={{ padding: '8px 10px', color: GENE_COLORS[row.gene], fontWeight: 700 }}>{row.gene}</td>
                    <td style={{ padding: '8px 10px', color: '#aaa', fontSize: 10 }}>{row.group}</td>
                    <td style={{ padding: '8px 10px', color: '#f44' }}>{row.t}</td>
                    <td style={{ padding: '8px 10px', color: row.nk === '✓✓✓' ? '#f44' : '#4caf50' }}>{row.nk}</td>
                    <td style={{ padding: '8px 10px', color: row.b_absent === '✓✓✓' ? '#f44' : '#4caf50' }}>{row.b_absent}</td>
                    <td style={{ padding: '8px 10px', color: row.agranulocytosis === '✓✓✓' ? '#ff7043' : '#4caf50' }}>{row.agranulocytosis}</td>
                    <td style={{ padding: '8px 10px', color: row.radiosens === '✓✓✓' ? '#ce93d8' : '#4caf50' }}>{row.radiosens}</td>
                    <td style={{ padding: '8px 10px', color: row.calcif === '✓✓✓' ? '#ff8f00' : '#4caf50' }}>{row.calcif}</td>
                    <td style={{ padding: '8px 10px', color: row.snhl.includes('✓') ? '#a5d6a7' : '#4caf50' }}>{row.snhl}</td>
                    <td style={{ padding: '8px 10px', color: row.gt === '✓✓✓' ? '#81c784' : '#888' }}>{row.gt}</td>
                    <td style={{ padding: '8px 10px', color: '#64b5f6' }}>{row.hsct}</td>
                    <td style={{ padding: '8px 10px', color: '#ddd', maxWidth: 320, fontSize: 10 }}>{row.unique}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* ── PATHWAY MAP TAB ── */}
        {tab === 'Pathway Map' && breakdown && (
          <div>
            {/* Cascade Testing */}
            <div style={{ background: '#0a1628', border: '1px solid #1565c0', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 10 }}>Cascade Diagnostic Testing — SCID Workup</div>
              {(breakdown.cascade_testing_order || []).map((step, i) => (
                <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 8, alignItems: 'flex-start' }}>
                  <span style={{ background: '#1565c0', color: '#fff', borderRadius: '50%', width: 22, height: 22, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, flexShrink: 0 }}>{i+1}</span>
                  <span style={{ fontSize: 12, color: '#ddd' }}>{step.replace(/^Step \d+: /, '')}</span>
                </div>
              ))}
            </div>

            {/* Diagnostic Groups */}
            {Object.entries(breakdown.diagnostic_groups || {}).map(([key, grp]) => (
              <div key={key} style={{ background: '#161616', border: '1px solid #333', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
                  {grp.genes.map(g => (
                    <span key={g} style={{ background: GENE_COLORS[g] || '#333', color: '#fff', padding: '2px 10px', borderRadius: 4, fontSize: 12, fontWeight: 700 }}>{g}</span>
                  ))}
                </div>
                <div style={{ fontSize: 12, color: '#90caf9', marginBottom: 6 }}>{grp.mechanism}</div>
                <div style={{ fontSize: 11, color: '#aaa', marginBottom: 4 }}><strong style={{ color: '#ef9a9a' }}>Shared:</strong> {grp.shared_feature}</div>
                <div style={{ fontSize: 11, color: '#aaa' }}><strong style={{ color: '#ef9a9a' }}>DDx:</strong> {grp.key_ddx}</div>
              </div>
            ))}

            {/* Pathognomonic Features */}
            <div style={{ background: '#1a001a', border: '1px solid #880e4f', borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: '#f48fb1', marginBottom: 10 }}>Pathognomonic Features & Clinical Rules</div>
              {Object.entries(breakdown.pathognomonic_features || {}).map(([key, val]) => (
                <div key={key} style={{ marginBottom: 8, paddingLeft: 12, borderLeft: '3px solid #880e4f' }}>
                  <div style={{ fontSize: 10, color: '#888', textTransform: 'uppercase', letterSpacing: 1 }}>{key.replace(/_/g, ' ')}</div>
                  <div style={{ fontSize: 12, color: '#ddd' }}>{val}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {tab === 'Definitions' && defs && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))', gap: 12, marginBottom: 20 }}>
              {(defs.definitions || []).map(d => (
                <div key={d.term} style={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, color: '#ef9a9a', marginBottom: 6 }}>{d.term}</div>
                  <div style={{ fontSize: 12, color: '#ccc', lineHeight: 1.5 }}>{d.definition}</div>
                </div>
              ))}
            </div>

            <div style={{ background: '#0a1628', border: '1px solid #1565c0', borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 10 }}>Key Distinguishers</div>
              {Object.entries(defs.key_distinguishers || {}).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase', letterSpacing: 0.8 }}>{k.replace(/_/g, ' ')}</div>
                  <div style={{ fontSize: 12, color: '#ddd', paddingLeft: 12, borderLeft: '2px solid #1565c0', marginTop: 2 }}>{v}</div>
                </div>
              ))}
            </div>

            <div style={{ background: '#001a00', border: '1px solid #2e7d32', borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: '#a5d6a7', marginBottom: 10 }}>Diagnostic Pathway</div>
              {(defs.diagnostic_pathway || []).map((step, i) => (
                <div key={i} style={{ fontSize: 12, color: '#ddd', marginBottom: 6, paddingLeft: 12, borderLeft: '2px solid #2e7d32' }}>{step}</div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
