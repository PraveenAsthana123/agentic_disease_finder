'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ipex-treg-immune-dysregulation-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'FOXP3':  '#1a237e',  // deep indigo      -- IPEX most common; neonatal triad; HSCT curative; FOXP3-Treg absent
  'CTLA4':  '#0d47a1',  // dark blue         -- CTLA4 haploinsufficiency; abatacept first-line; lymphoproliferation
  'LRBA':   '#1565c0',  // medium blue       -- LRBA deficiency; AR; Evans syndrome; abatacept curative; CTLA4 lysosomal
  'STAT3':  '#006064',  // deep cyan         -- STAT3-GOF; multi-organ autoimmunity; JAK inhibitor ruxolitinib
  'IL2RA':  '#004d40',  // dark teal         -- CD25 deficiency; paradoxical lymphoproliferation; severe IBD; rare
  'AIRE':   '#1b5e20',  // dark green        -- APECED/APS-1; candidiasis+Addison+hypoparathyroidism; anti-IFN autoantibodies
  'CARD11': '#e65100',  // deep orange       -- BENTA disease; severe atopic dermatitis; eosinophilia; IgE markedly elevated
  'BACH2':  '#4a148c',  // deep purple       -- BRIDA; lymphoproliferation+enteropathy+hypogammaglobulinemia; dual defect
};

const GENE_INFO = {
  'FOXP3':  { full: 'FOXP3-256aa / XLR / Forkhead-box-P3 / Treg-master-TF / Xp11.23 / IPEX-Syndrome / NEONATAL-TRIAD-PATHOGNOMONIC / Enteropathy+DM+Eczema / FOXP3-Treg-absent / HSCT-CURATIVE / Death-<2yr-untreated', locus: 'Xp11.23',  size: '256 aa / 47 kDa',  inh: 'X-linked recessive (XLR; males only)' },
  'CTLA4':  { full: 'CTLA4-223aa / AD-LOF / Haploinsufficiency / CTLA4-checkpoint-inhibitory / 2q33.2 / CTLA4-Haploinsufficiency / Lymphoproliferation+CMV+EBV / Abatacept-FIRST-LINE-DRAMATIC / Treg-CTLA4-recycling-defect', locus: '2q33.2',   size: '223 aa / 25 kDa',  inh: 'Autosomal dominant LOF (haploinsufficiency)' },
  'LRBA':   { full: 'LRBA-2863aa / AR / BEACH-domain / CTLA4-lysosomal-degradation-prevented / 4q31.3 / LRBA-Deficiency / Phenocopy-CTLA4-haploinsufficiency / EVANS-SYNDROME-hallmark / Abatacept-CURATIVE / Largest-protein-atlas', locus: '4q31.3',   size: '2863 aa / 319 kDa', inh: 'Autosomal recessive (AR; consanguinity enriches)' },
  'STAT3':  { full: 'STAT3-770aa / AD-GOF / Constitutively-active-STAT3 / IL-10-IL-21-IL-6-overactive / 17q21.2 / STAT3-GOF / Multi-organ-autoimmunity-early / Type-1-DM-prominent / JAK-INHIBITOR-ruxolitinib-DRAMATIC / Complete-remissions', locus: '17q21.2',  size: '770 aa / 92 kDa',  inh: 'Autosomal dominant GOF (de novo common)' },
  'IL2RA':  { full: 'IL2RA-272aa / AR / CD25-IL-2Rα / IL-2-signalling-absent / 10p15.1 / CD25-Deficiency / PARADOXICAL-LYMPHOPROLIFERATION / Severe-IBD+Lung-disease / CD25-absent-flow-DIAGNOSTIC / Very-rare-<30-families', locus: '10p15.1',  size: '272 aa / 30 kDa',  inh: 'Autosomal recessive (AR; very rare)' },
  'AIRE':   { full: 'AIRE-545aa / AR / Autoimmune-regulator / mTEC-central-tolerance / 21q22.3 / APECED-APS-1 / APECED-TRIAD-PATHOGNOMONIC / Candidiasis+Addison+Hypoparathyroidism / Anti-IFN-autoantibodies-COVID-risk / Adrenal-replacement-MANDATORY', locus: '21q22.3',  size: '545 aa / 58 kDa',  inh: 'Autosomal recessive (AR; Finnish/Sardinian founders)' },
  'CARD11': { full: 'CARD11-1154aa / AD-GOF / CBM-complex-scaffold / BCL10-MALT1-NF-kB-constitutive / 7p22.2 / BENTA-Disease / SEVERE-ATOPIC-DERMATITIS-PATHOGNOMONIC / Eosinophilia+IgE-markedly-elevated / B-cell-expansion+T-cell-anergy / Dupilumab-partial', locus: '7p22.2',   size: '1154 aa / 133 kDa', inh: 'Autosomal dominant GOF' },
  'BACH2':  { full: 'BACH2-641aa / AD-LOF / BTB-CNC-TF / Treg+memory-B-cell-differentiation / 6q15 / BRIDA / Lymphoproliferation+Enteropathy+Hypogammaglobulinemia / Dual-Treg+memory-B-defect / IVIG-required / Rare-haploinsufficiency', locus: '6q15',     size: '641 aa / 72 kDa',  inh: 'Autosomal dominant LOF (haploinsufficiency)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'FOXP3',  group: 'Treg Master TF (IPEX)',        autoimmunity: '✓✓✓', lympho: '✓✓',  enteropathy: '✓✓✓', hsct: '✓✓✓', abatacept: '✗',    jaki: '✗',   candidiasis: '±',  ige: '✗',    unique: 'IPEX NEONATAL TRIAD PATHOGNOMONIC: ENTEROPATHY + TYPE-1 DM + ECZEMA. FOXP3-Treg absent. DEATH <2yr without HSCT. HSCT CURATIVE. Bridge: sirolimus/tacrolimus (NOT steroids for gut). XLR — males only.' },
  { gene: 'CTLA4',  group: 'CTLA4 Haploinsufficiency',     autoimmunity: '✓✓✓', lympho: '✓✓✓', enteropathy: '✓✓',  hsct: '±',   abatacept: '✓✓✓', jaki: '±',   candidiasis: '±',  ige: '✗',    unique: 'AD-LOF. Lymphoproliferation + CMV/EBV susceptibility + autoimmunity. ABATACEPT FIRST-LINE DRAMATICALLY EFFECTIVE — shrinks organomegaly within weeks, resolves cytopenias. Treg-CTLA4 recycling defect.' },
  { gene: 'LRBA',   group: 'LRBA/CTLA4 Lysosomal',        autoimmunity: '✓✓✓', lympho: '✓✓✓', enteropathy: '✓✓',  hsct: '±',   abatacept: '✓✓✓', jaki: '±',   candidiasis: '±',  ige: '✗',    unique: 'EVANS SYNDROME (AIHA + ITP combined) HALLMARK. AR. ABATACEPT CURATIVE (exogenous CTLA4-Ig bypasses recycling defect). Phenocopies CTLA4 haploinsufficiency. Largest protein atlas (2863aa).' },
  { gene: 'STAT3',  group: 'STAT3-GOF (JAK-STAT)',         autoimmunity: '✓✓✓', lympho: '✓✓',  enteropathy: '✓✓',  hsct: '±',   abatacept: '±',    jaki: '✓✓✓', candidiasis: '±',  ige: '✗',    unique: 'AD-GOF. EARLY-ONSET MULTI-ORGAN AUTOIMMUNITY: type-1 DM + thyroid + cytopenias + enteropathy + SHORT STATURE. JAK INHIBITOR (ruxolitinib) DRAMATICALLY EFFECTIVE — multiple complete remissions.' },
  { gene: 'IL2RA',  group: 'CD25 Deficiency (IL-2 Signal)', autoimmunity: '✓✓✓', lympho: '✓✓✓', enteropathy: '✓✓✓', hsct: '✓✓',  abatacept: '±',    jaki: '±',   candidiasis: '±',  ige: '✗',    unique: 'AR; very rare (<30 families worldwide). PARADOXICAL LYMPHOPROLIFERATION (IL-2 expands effectors without Tregs). Severe IBD + lung disease. CD25 ABSENT by flow = diagnostic. HSCT for severe cases.' },
  { gene: 'AIRE',   group: 'Central Tolerance (APECED)',   autoimmunity: '✓✓✓', lympho: '±',   enteropathy: '±',    hsct: '✗',   abatacept: '✗',    jaki: '✗',   candidiasis: '✓✓✓', ige: '✗',   unique: 'APECED TRIAD PATHOGNOMONIC: CHRONIC MUCOCUTANEOUS CANDIDIASIS (first) + ADDISON (adrenal replacement MANDATORY) + HYPOPARATHYROIDISM. Anti-IFN-α/ω autoantibodies → COVID-19 severe susceptibility.' },
  { gene: 'CARD11', group: 'BENTA (NF-κB GOF)',            autoimmunity: '✓✓',  lympho: '✓✓✓', enteropathy: '±',    hsct: '±',   abatacept: '±',    jaki: '±',   candidiasis: '±',  ige: '✓✓✓', unique: 'SEVERE ATOPIC DERMATITIS FROM BIRTH PATHOGNOMONIC. EOSINOPHILIA. IgE markedly elevated (>>10,000 IU/mL). B cell expansion + T cell anergy. DUPILUMAB PARTIALLY EFFECTIVE. BENTA disease.' },
  { gene: 'BACH2',  group: 'BRIDA (Treg + Memory B)',      autoimmunity: '✓✓',  lympho: '✓✓✓', enteropathy: '✓✓✓', hsct: '±',   abatacept: '±',    jaki: '±',   candidiasis: '±',  ige: '✗',    unique: 'AD-LOF. DUAL DEFECT: Treg development impaired + memory B cell impaired. Lymphoproliferation + ENTEROPATHY + HYPOGAMMAGLOBULINEMIA. IVIG mandatory. BACH2 unique role in both Treg and memory B cell.' },
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
  const isFOXP3   = gene === 'FOXP3';
  const isCTLA4   = gene === 'CTLA4';
  const isLRBA    = gene === 'LRBA';
  const isSTAT3   = gene === 'STAT3';
  const isIL2RA   = gene === 'IL2RA';
  const isAIRE    = gene === 'AIRE';
  const isCARD11  = gene === 'CARD11';
  const isBACH2   = gene === 'BACH2';
  const groupLabel = isFOXP3  ? 'Treg Master TF (IPEX)'
    : isCTLA4   ? 'CTLA4 Haploinsufficiency'
    : isLRBA    ? 'LRBA/CTLA4-Lysosomal'
    : isSTAT3   ? 'STAT3-GOF (JAK-STAT)'
    : isIL2RA   ? 'CD25 Deficiency'
    : isAIRE    ? 'Central Tolerance (APECED)'
    : isCARD11  ? 'BENTA Disease (NF-κB GOF)'
    : 'BRIDA (Treg + Memory B)';
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
          <Badge text={`Autoimmunity ${cohort.autoimmunity_pct}%`}          color={color} />
          {cohort.lymphoproliferation_pct > 0  && <Badge text={`Lymphoprolif ${cohort.lymphoproliferation_pct}%`} color='#1565c0' />}
          {cohort.enteropathy_pct > 0          && <Badge text={`Enteropathy ${cohort.enteropathy_pct}%`}          color='#4a148c' />}
          {cohort.hsct_pct > 0                 && <Badge text={`HSCT ${cohort.hsct_pct}%`}                        color='#b71c1c' />}
          {cohort.abatacept_pct > 0            && <Badge text={`Abatacept ${cohort.abatacept_pct}%`}              color='#0d47a1' />}
          {cohort.jaki_pct > 0                 && <Badge text={`JAKi ${cohort.jaki_pct}%`}                        color='#006064' />}
          {cohort.candidiasis_pct > 0          && <Badge text={`Candidiasis ${cohort.candidiasis_pct}%`}          color='#1b5e20' />}
          {cohort.t1dm_pct > 0                 && <Badge text={`T1DM ${cohort.t1dm_pct}%`}                        color='#e65100' />}
          {cohort.eczema_pct > 0               && <Badge text={`Eczema ${cohort.eczema_pct}%`}                    color='#880e4f' />}
          <Badge text={`Onset ${cohort.mean_onset_months}mo`}               color='#455a64' />
        </div>
      )}
    </div>
  );
}

export default function HereditaryIPEXTregImmuneDysregulationAtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);

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

  if (loading) return <div style={{ color: '#fff', padding: 32 }}>Loading Hereditary-IPEX-Treg-Immune-Dysregulation-Atlas…</div>;
  if (error)   return <div style={{ color: '#f44', padding: 32 }}>Error: {error}</div>;

  const stats        = overview?.aggregate_stats || {};
  const cohortByGene = {};
  (breakdown?.gene_cohorts || []).forEach(c => { cohortByGene[c.gene] = c; });

  return (
    <div style={{ background: '#0a0a1a', minHeight: '100vh', color: '#eee', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: '#0d0d3a', borderBottom: '2px solid #1a237e', padding: '20px 32px' }}>
        <h1 style={{ color: '#9fa8da', fontSize: 22, marginBottom: 4 }}>
          🧬 Hereditary-IPEX-Treg-Immune-Dysregulation-Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#aaa' }}>
          Complete 8-Gene Regulatory T-Cell Immune Dysregulation Reference · FOXP3 · CTLA4 · LRBA · STAT3 · IL2RA · AIRE · CARD11 · BACH2 ·
          320-Patient Aggregate (8×40) · Seeds 3974-3981
        </div>
        <div style={{ marginTop: 8, display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 12 }}>
          <span style={{ background: '#1a237e', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Autoimmunity {stats.autoimmunity_pct}%</span>
          <span style={{ background: '#1565c0', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Lymphoprolif {stats.lymphoproliferation_pct}%</span>
          <span style={{ background: '#4a148c', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Enteropathy {stats.enteropathy_pct}%</span>
          <span style={{ background: '#b71c1c', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>HSCT {stats.hsct_pct}%</span>
          <span style={{ background: '#0d47a1', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Abatacept {stats.abatacept_pct}%</span>
          <span style={{ background: '#006064', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>JAKi {stats.jaki_pct}%</span>
          <span style={{ background: '#1b5e20', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Candidiasis {stats.candidiasis_pct}%</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid #333', background: '#0d0d3a' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{
              background: tab === t ? '#1a237e' : 'transparent', color: tab === t ? '#fff' : '#aaa',
              border: 'none', padding: '10px 22px', cursor: 'pointer', fontSize: 13, fontFamily: 'monospace'
            }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: 24 }}>

        {/* ── OVERVIEW TAB ── */}
        {tab === 'Overview' && (
          <div>
            {/* Key Facts */}
            <div style={{ background: '#0d0d3a', border: '1px solid #1a237e', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#9fa8da', marginBottom: 10 }}>Key Clinical Facts — 8 IPEX-Spectrum / Treg Immune Dysregulation Diseases</div>
              {(overview?.key_facts || []).map((f, i) => (
                <div key={i} style={{ fontSize: 12, color: '#ddd', marginBottom: 6, paddingLeft: 12, borderLeft: '3px solid #1a237e' }}>{f}</div>
              ))}
            </div>

            {/* Disease Groups by Mechanism */}
            <div style={{ background: '#0a1628', border: '1px solid #1565c0', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 12 }}>Disease Groups by Mechanism</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(270px, 1fr))', gap: 12 }}>
                {[
                  { label: 'Treg Master TF (FOXP3)',      genes: 'FOXP3',         desc: 'XLR; neonatal triad (enteropathy+DM+eczema); FOXP3-Treg absent; HSCT curative; death <2yr untreated.', color: '#1a237e' },
                  { label: 'CTLA4 / LRBA Axis',           genes: 'CTLA4 / LRBA',  desc: 'Insufficient surface CTLA4 (haploinsufficiency or lysosomal degradation); both respond to abatacept dramatically.', color: '#1565c0' },
                  { label: 'STAT3-GOF (JAK-STAT)',         genes: 'STAT3',         desc: 'AD-GOF; multi-organ autoimmunity (DM+thyroid+cytopenias+enteropathy+short stature); JAKi (ruxolitinib) dramatic.', color: '#006064' },
                  { label: 'IL-2 Signalling (CD25)',       genes: 'IL2RA',         desc: 'AR; very rare; CD25 absent → no IL-2 signalling → Treg depletion; paradoxical lymphoproliferation; severe IBD.', color: '#004d40' },
                  { label: 'Central Tolerance (AIRE)',     genes: 'AIRE',          desc: 'AR; APECED triad (candidiasis+Addison+hypoparathyroidism); anti-IFN autoantibodies → COVID-19 susceptibility.', color: '#1b5e20' },
                  { label: 'BENTA (NF-κB GOF)',            genes: 'CARD11',        desc: 'AD-GOF; BCL10-MALT1-NF-κB constitutive; severe atopic dermatitis + eosinophilia + IgE >>10,000; dupilumab.', color: '#e65100' },
                  { label: 'BRIDA (Treg + Memory B)',      genes: 'BACH2',         desc: 'AD-LOF; BACH2 dual role (Treg differentiation + memory B cell); lymphoproliferation + enteropathy + hypogammaglobulinemia.', color: '#4a148c' },
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
            <div style={{ marginTop: 20, background: '#0d0d3a', border: '1px solid #333', borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: '#9fa8da', marginBottom: 10 }}>Aggregate Statistics — 320 Patients (8×40)</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))', gap: 8 }}>
                {[
                  ['Autoimmunity',    stats.autoimmunity_pct,        '%', '#1a237e'],
                  ['Lymphoprolif',    stats.lymphoproliferation_pct,  '%', '#1565c0'],
                  ['Enteropathy',     stats.enteropathy_pct,          '%', '#4a148c'],
                  ['HSCT',            stats.hsct_pct,                 '%', '#b71c1c'],
                  ['Abatacept',       stats.abatacept_pct,            '%', '#0d47a1'],
                  ['JAKi',            stats.jaki_pct,                 '%', '#006064'],
                  ['Candidiasis',     stats.candidiasis_pct,          '%', '#1b5e20'],
                  ['T1DM',            stats.t1dm_pct,                 '%', '#e65100'],
                  ['Eczema',          stats.eczema_pct,               '%', '#880e4f'],
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
                <tr style={{ background: '#0d0d3a' }}>
                  {['Gene', 'Group', 'Autoimmunity', 'Lymphoprolif', 'Enteropathy', 'HSCT', 'Abatacept', 'JAKi', 'Candidiasis', 'IgE↑', 'Clinical Pearl'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', color: '#9fa8da', textAlign: 'left', borderBottom: '1px solid #444' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#161616' : '#0d0d3a' }}>
                    <td style={{ padding: '8px 10px', color: GENE_COLORS[row.gene], fontWeight: 700 }}>{row.gene}</td>
                    <td style={{ padding: '8px 10px', color: '#aaa', fontSize: 10 }}>{row.group}</td>
                    <td style={{ padding: '8px 10px', color: row.autoimmunity === '✓✓✓' ? '#9fa8da' : '#888' }}>{row.autoimmunity}</td>
                    <td style={{ padding: '8px 10px', color: row.lympho      === '✓✓✓' ? '#64b5f6' : '#888' }}>{row.lympho}</td>
                    <td style={{ padding: '8px 10px', color: row.enteropathy === '✓✓✓' ? '#ce93d8' : '#888' }}>{row.enteropathy}</td>
                    <td style={{ padding: '8px 10px', color: row.hsct        === '✓✓✓' ? '#ef9a9a' : '#888' }}>{row.hsct}</td>
                    <td style={{ padding: '8px 10px', color: row.abatacept   === '✓✓✓' ? '#90caf9' : '#888' }}>{row.abatacept}</td>
                    <td style={{ padding: '8px 10px', color: row.jaki        === '✓✓✓' ? '#80cbc4' : '#888' }}>{row.jaki}</td>
                    <td style={{ padding: '8px 10px', color: row.candidiasis === '✓✓✓' ? '#a5d6a7' : '#888' }}>{row.candidiasis}</td>
                    <td style={{ padding: '8px 10px', color: row.ige         === '✓✓✓' ? '#ffcc80' : '#888' }}>{row.ige}</td>
                    <td style={{ padding: '8px 10px', color: '#ddd', maxWidth: 340, fontSize: 10 }}>{row.unique}</td>
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
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 10 }}>Cascade Diagnostic Testing — IPEX-Spectrum Workup</div>
              {(breakdown.cascade_testing_order || []).map((step, i) => (
                <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 8, alignItems: 'flex-start' }}>
                  <span style={{ background: '#1a237e', color: '#fff', borderRadius: '50%', width: 22, height: 22, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, flexShrink: 0 }}>{i+1}</span>
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
                <div style={{ fontSize: 11, color: '#aaa', marginBottom: 4 }}><strong style={{ color: '#9fa8da' }}>Shared:</strong> {grp.shared_feature}</div>
                <div style={{ fontSize: 11, color: '#aaa' }}><strong style={{ color: '#9fa8da' }}>DDx:</strong> {grp.key_ddx}</div>
              </div>
            ))}

            {/* Pathognomonic Features */}
            <div style={{ background: '#0d0d2a', border: '1px solid #1a237e', borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: '#9fa8da', marginBottom: 10 }}>Pathognomonic Features & Clinical Rules</div>
              {Object.entries(breakdown.pathognomonic_features || {}).map(([key, val]) => (
                <div key={key} style={{ marginBottom: 8, paddingLeft: 12, borderLeft: '3px solid #1a237e' }}>
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
                <div key={d.term} style={{ background: '#0d0d3a', border: '1px solid #333', borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, color: '#9fa8da', marginBottom: 6 }}>{d.term}</div>
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

            <div style={{ background: '#0a1a00', border: '1px solid #1b5e20', borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: '#a5d6a7', marginBottom: 10 }}>Treatment Matrix</div>
              {Object.entries(defs.treatment_matrix || {}).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 6, paddingLeft: 12, borderLeft: '2px solid #1b5e20' }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#a5d6a7', textTransform: 'uppercase' }}>{k.replace(/_/g, ' ')}: </span>
                  <span style={{ fontSize: 12, color: '#ddd' }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
