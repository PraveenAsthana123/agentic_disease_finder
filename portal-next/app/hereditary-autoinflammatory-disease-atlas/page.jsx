'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-autoinflammatory-disease-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'MEFV':     '#b71c1c',  // deep red        -- FMF most common; colchicine; amyloidosis PATHOGNOMONIC
  'NLRP3':    '#1565c0',  // dark blue        -- CAPS spectrum; cold-triggered; SNHL; canakinumab FDA
  'TNFRSF1A': '#6a1b9a',  // deep purple      -- TRAPS long episodes; periorbital edema PATHOGNOMONIC
  'MVK':      '#e65100',  // deep orange      -- MKD/HIDS; cervical adenopathy; statins CI; canakinumab
  'PSTPIP1':  '#4e342e',  // dark brown       -- PAPA triad; pyoderma gangrenosum; pathergy
  'NOD2':     '#1b5e20',  // dark green       -- Blau/EOS; granuloma triad; uveitis blindness risk
  'IL1RN':    '#f57f17',  // dark amber       -- DIRA neonatal; anakinra curative; sterile osteomyelitis
  'LPIN2':    '#37474f',  // dark slate       -- Majeed; CRMO + CDA; lipid pathway; Jordanian founder
};

const GENE_INFO = {
  'MEFV':     { full: 'MEFV-781aa / AR+AD / Pyrin-Marenostrin / Pyrin-inflammasome / 16p13.3 / FMF-Most-Common-Monogenic-AID / AMYLOIDOSIS-PATHOGNOMONIC / Colchicine-Lifelong-Mandatory / IL-1-blocker-refractory', locus: '16p13.3',  size: '781 aa / 86 kDa',  inh: 'AR (classical) / AD-LOF (attenuated)' },
  'NLRP3':    { full: 'NLRP3-1036aa / AD-GOF / Cryopyrin-CIAS1 / NLRP3-inflammasome-constitutive / 1q44 / CAPS-FCAS-MWS-NOMID / COLD-URTICARIAL-RASH-PATHOGNOMONIC / SNHL-progressive-MWS / Canakinumab-FDA2016', locus: '1q44',      size: '1036 aa / 118 kDa', inh: 'Autosomal dominant GOF' },
  'TNFRSF1A': { full: 'TNFRSF1A-234aa / AD-GOF / TNFR1-p55 / ER-retention-misfolded / 12p13.31 / TRAPS-Longest-Episodes / PERIORBITAL-EDEMA-PATHOGNOMONIC / Etanercept-NOT-first-line / IL-1-blockers-preferred', locus: '12p13.31', size: '234 aa / 55 kDa',  inh: 'Autosomal dominant GOF' },
  'MVK':      { full: 'MVK-396aa / AR / Mevalonate-kinase / Mevalonate-pathway-step-2 / 12q24.11 / MKD-HIDS / CERVICAL-ADENOPATHY-PATHOGNOMONIC / IgD-elevated / Statins-ABSOLUTELY-CI / Canakinumab-FDA2016', locus: '12q24.11', size: '396 aa / 43 kDa',  inh: 'Autosomal recessive' },
  'PSTPIP1':  { full: 'PSTPIP1-415aa / AD-GOF / CD2BP1 / Pyrin-binding-actin-regulator / 15q24.3 / PAPA-Syndrome / PYODERMA-GANGRENOSUM-PATHOGNOMONIC / Pathergy-positive-90pct / Anakinra-first-line', locus: '15q24.3',  size: '415 aa / 47 kDa',  inh: 'Autosomal dominant GOF' },
  'NOD2':     { full: 'NOD2-1040aa / AD-GOF / CARD15 / MDP-sensing-NF-kB / 16q12.1 / Blau-EOS / GRANULOMATOUS-TRIAD-PATHOGNOMONIC / UVEITIS-blindness-risk / Ophthalmology-MANDATORY / GOF-not-LOF-Crohns', locus: '16q12.1',  size: '1040 aa / 114 kDa', inh: 'Autosomal dominant GOF' },
  'IL1RN':    { full: 'IL1RN-177aa / AR / IL-1Ra-receptor-antagonist / Unopposed-IL-1-signalling / 2q14.1 / DIRA / NEONATAL-ONSET-DAY-1-7 / STERILE-OSTEOMYELITIS-PATHOGNOMONIC / ANAKINRA-CURATIVE / Lifelong-mandatory', locus: '2q14.1',   size: '177 aa / 20 kDa',  inh: 'Autosomal recessive' },
  'LPIN2':    { full: 'LPIN2-890aa / AR / Lipin-2 / Phosphatidic-acid-phosphatase / 18p11.31 / Majeed-Syndrome / CRMO-CDA-Dermatosis-TRIAD-PATHOGNOMONIC / CDA-transfusion-dependent / IL-1-blocker-G-CSF-bisphosphonate', locus: '18p11.31', size: '890 aa / 101 kDa', inh: 'Autosomal recessive' },
};

const PHENOTYPE_ROWS = [
  { gene: 'MEFV',     group: 'Pyrin Inflammasome',      amyloid: '✓✓✓', snhl: '±',    uveitis: '✗',    crmo: '✗',  anaemia: '±',  colchicine: '✓✓✓', il1: '✓',    unique: 'FMF MOST COMMON (Mediterranean). AA AMYLOIDOSIS = preventable tragedy; colchicine lifelong prevents attacks AND amyloidosis. Peritonitis/pleuritis/synovitis 12-72h episodes. M694V homozygous = highest amyloid risk.' },
  { gene: 'NLRP3',    group: 'NLRP3 Inflammasome',      amyloid: '✓✓',  snhl: '✓✓✓', uveitis: '✓',    crmo: '✗',  anaemia: '±',  colchicine: '✗',    il1: '✓✓✓', unique: 'CAPS SPECTRUM (FCAS→MWS→NOMID). COLD-TRIGGERED URTICARIAL RASH PATHOGNOMONIC. MWS: progressive SNHL — audiogram annually. NOMID: neonatal + CNS (papilledema/meningitis). Canakinumab FDA-approved 2016 prevents SNHL progression.' },
  { gene: 'TNFRSF1A', group: 'TNF/ER Stress',            amyloid: '✓✓',  snhl: '±',    uveitis: '✓',    crmo: '✗',  anaemia: '±',  colchicine: '✗',    il1: '✓✓✓', unique: 'TRAPS LONGEST EPISODES (days-weeks). PERIORBITAL EDEMA + MIGRATORY RASH PATHOGNOMONIC. Etanercept (TNFR2) less effective than IL-1 blockers — IL-1 preferred. Amyloidosis risk. R92Q/P46L = low-penetrance variants.' },
  { gene: 'MVK',      group: 'Mevalonate Pathway',       amyloid: '±',   snhl: '✗',    uveitis: '±',    crmo: '✗',  anaemia: '±',  colchicine: '✗',    il1: '✓✓✓', unique: 'MKD/HIDS. CERVICAL ADENOPATHY in children PATHOGNOMONIC. IgD elevated (NOT always). Urine mevalonic acid during fever = diagnostic. STATINS ABSOLUTELY CONTRAINDICATED (worsen pathway). Vaccination triggers attacks. Canakinumab FDA-approved.' },
  { gene: 'PSTPIP1',  group: 'Pyrin-PSTPIP1 Axis',       amyloid: '±',   snhl: '✗',    uveitis: '✗',    crmo: '✗',  anaemia: '±',  colchicine: '✗',    il1: '✓✓✓', unique: 'PAPA TRIAD PATHOGNOMONIC: Pyogenic Arthritis + Pyoderma Gangrenosum + Acne Conglobata. Pathergy test >90% positive. PAMI = severe variant (hepatosplenomegaly). Anakinra first-line; steroids for PG skin lesions. Differential: PASH syndrome.' },
  { gene: 'NOD2',     group: 'NOD2/NF-κB Granuloma',     amyloid: '±',   snhl: '±',    uveitis: '✓✓✓', crmo: '✗',  anaemia: '±',  colchicine: '✗',    il1: '±',    unique: 'BLAU/EOS GRANULOMATOUS TRIAD: UVEITIS (panuveitis→blindness) + BOGGY ARTHRITIS (camptodactyly) + SKIN (ichthyosiform). NOT episodic fever. Ophthalmology MANDATORY. Distinguish from sarcoidosis: no hilar adenopathy; ACE normal; onset <5yr. GOF NOD2 ≠ LOF NOD2 Crohn\'s.' },
  { gene: 'IL1RN',    group: 'IL-1Ra Deficiency',         amyloid: '±',   snhl: '✗',    uveitis: '✗',    crmo: '✓✓✓', anaemia: '±', colchicine: '✗',    il1: '✓✓✓', unique: 'DIRA NEONATAL (day 1-7). STERILE MULTIFOCAL OSTEOMYELITIS + PUSTULOSIS + PERIOSTITIS PATHOGNOMONIC. Anakinra CURATIVE within 24-48h — start empirically. Lifelong daily anakinra mandatory. Distinguish from septic arthritis: culture-negative + periostitis + dramatic anakinra response.' },
  { gene: 'LPIN2',    group: 'Lipid/Lipin Pathway',       amyloid: '±',   snhl: '✗',    uveitis: '±',    crmo: '✓✓✓', anaemia: '✓✓✓', colchicine: '±', il1: '✓✓',  unique: 'MAJEED TRIAD PATHOGNOMONIC: CRMO (bone) + CONGENITAL DYSERYTHROPOIETIC ANEMIA (transfusion-dependent CDA) + Neutrophilic Dermatosis. CDA distinguishes from all other CRMO-MAIDs. Jordanian/Arab consanguineous families. G-CSF + bisphosphonates + IL-1 blocker.' },
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
  const isPyrin    = ['MEFV', 'PSTPIP1'].includes(gene);
  const isNLRP3    = gene === 'NLRP3';
  const isTNF      = gene === 'TNFRSF1A';
  const isMVK      = gene === 'MVK';
  const isNOD2     = gene === 'NOD2';
  const isDIRA     = gene === 'IL1RN';
  const isMajeed   = gene === 'LPIN2';
  const groupLabel = isPyrin  ? 'Pyrin Inflammasome'
    : isNLRP3   ? 'NLRP3 Inflammasome (CAPS)'
    : isTNF     ? 'TNF/ER Stress (TRAPS)'
    : isMVK     ? 'Mevalonate Pathway (MKD)'
    : isNOD2    ? 'NOD2/NF-κB Granuloma (Blau/EOS)'
    : isDIRA    ? 'IL-1Ra Deficiency (DIRA)'
    : isMajeed  ? 'Lipid/Lipin Pathway (Majeed)'
    : 'PAPA Syndrome';
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
          <Badge text={`IL-1 Blocker ${cohort.il1_blocker_pct}%`}      color={color} />
          {cohort.colchicine_pct > 0  && <Badge text={`Colchicine ${cohort.colchicine_pct}%`}    color='#b71c1c' />}
          {cohort.amyloid_pct > 0     && <Badge text={`Amyloid ${cohort.amyloid_pct}%`}           color='#880e4f' />}
          {cohort.snhl_pct > 0        && <Badge text={`SNHL ${cohort.snhl_pct}%`}                 color='#1565c0' />}
          {cohort.uveitis_pct > 0     && <Badge text={`Uveitis ${cohort.uveitis_pct}%`}           color='#1b5e20' />}
          {cohort.crmo_pct > 0        && <Badge text={`CRMO ${cohort.crmo_pct}%`}                 color='#37474f' />}
          {cohort.anaemia_pct > 0     && <Badge text={`Anaemia ${cohort.anaemia_pct}%`}            color='#5d4037' />}
          {cohort.rash_pct > 0        && <Badge text={`Rash/Skin ${cohort.rash_pct}%`}            color='#e65100' />}
          <Badge text={`Onset ${cohort.mean_onset_months}mo`}           color='#455a64' />
        </div>
      )}
    </div>
  );
}

export default function HereditaryAutoinflammatorDiseaseAtlasPage() {
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

  if (loading) return <div style={{ color: '#fff', padding: 32 }}>Loading Hereditary-Autoinflammatory-Disease-Atlas…</div>;
  if (error)   return <div style={{ color: '#f44', padding: 32 }}>Error: {error}</div>;

  const stats        = overview?.aggregate_stats || {};
  const cohortByGene = {};
  (breakdown?.gene_cohorts || []).forEach(c => { cohortByGene[c.gene] = c; });

  return (
    <div style={{ background: '#121212', minHeight: '100vh', color: '#eee', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: '#1a0008', borderBottom: '2px solid #b71c1c', padding: '20px 32px' }}>
        <h1 style={{ color: '#ef9a9a', fontSize: 22, marginBottom: 4 }}>
          🔥 Hereditary-Autoinflammatory-Disease-Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#aaa' }}>
          Complete 8-Gene Monogenic AID Reference · MEFV · NLRP3 · TNFRSF1A · MVK · PSTPIP1 · NOD2 · IL1RN · LPIN2 ·
          320-Patient Aggregate (8×40) · Seeds 3966-3973
        </div>
        <div style={{ marginTop: 8, display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 12 }}>
          <span style={{ background: '#b71c1c', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Amyloidosis Risk {stats.amyloid_pct}%</span>
          <span style={{ background: '#1565c0', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>SNHL {stats.snhl_pct}%</span>
          <span style={{ background: '#1b5e20', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Uveitis {stats.uveitis_pct}%</span>
          <span style={{ background: '#37474f', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>CRMO {stats.crmo_pct}%</span>
          <span style={{ background: '#e65100', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Rash/Skin {stats.rash_pct}%</span>
          <span style={{ background: '#2e7d32', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>IL-1 Blocker {stats.il1_blocker_pct}%</span>
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
            <div style={{ background: '#1a0008', border: '1px solid #b71c1c', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#ef9a9a', marginBottom: 10 }}>Key Clinical Facts — 8 Monogenic Autoinflammatory Diseases</div>
              {(overview?.key_facts || []).map((f, i) => (
                <div key={i} style={{ fontSize: 12, color: '#ddd', marginBottom: 6, paddingLeft: 12, borderLeft: '3px solid #b71c1c' }}>{f}</div>
              ))}
            </div>

            {/* IL-1 Pathway Groups */}
            <div style={{ background: '#0a1628', border: '1px solid #1565c0', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 12 }}>Disease Groups by Mechanism</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(270px, 1fr))', gap: 12 }}>
                {[
                  { label: 'Pyrin Inflammasome',     genes: 'MEFV / PSTPIP1',      desc: 'IL-1β via pyrin assembly. FMF (serositis) vs PAPA (pyoderma/arthritis/acne). Colchicine for FMF.', color: '#b71c1c' },
                  { label: 'NLRP3 Inflammasome',     genes: 'NLRP3',               desc: 'AD-GOF; constitutive IL-1β. CAPS spectrum: FCAS→MWS→NOMID. Cold trigger. Canakinumab.', color: '#1565c0' },
                  { label: 'TNF/ER Stress',          genes: 'TNFRSF1A',            desc: 'ER retention → NF-κB + IL-1β. TRAPS: longest episodes; periorbital edema. IL-1 >> etanercept.', color: '#6a1b9a' },
                  { label: 'Mevalonate Pathway',     genes: 'MVK',                 desc: 'AR; geranylgeranylation defect → IL-1β. Cervical adenopathy. Statins CI. Canakinumab.', color: '#e65100' },
                  { label: 'IL-1Ra Deficiency',      genes: 'IL1RN',               desc: 'Absent IL-1Ra → unopposed IL-1. Neonatal DIRA. Anakinra CURATIVE. Osteomyelitis+pustulosis.', color: '#f57f17' },
                  { label: 'NOD2/NF-κB Granuloma',   genes: 'NOD2',                desc: 'AD-GOF; granulomatous. Blau/EOS: uveitis+arthritis+skin triad. No prominent fever.', color: '#1b5e20' },
                  { label: 'Lipid/Lipin Pathway',    genes: 'LPIN2',               desc: 'AR; PA accumulation → NF-κB. Majeed: CRMO + CDA + dermatosis triad. Consanguineous.', color: '#37474f' },
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
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))', gap: 8 }}>
                {[
                  ['IL-1 Blocker',   stats.il1_blocker_pct,  '%', '#2e7d32'],
                  ['Colchicine',     stats.colchicine_pct,   '%', '#b71c1c'],
                  ['Amyloidosis',    stats.amyloid_pct,      '%', '#880e4f'],
                  ['SNHL',           stats.snhl_pct,         '%', '#1565c0'],
                  ['Uveitis',        stats.uveitis_pct,      '%', '#1b5e20'],
                  ['CRMO',           stats.crmo_pct,         '%', '#37474f'],
                  ['Rash/Skin',      stats.rash_pct,         '%', '#e65100'],
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
                <tr style={{ background: '#1a0008' }}>
                  {['Gene', 'Group', 'Amyloid', 'SNHL', 'Uveitis', 'CRMO', 'Anaemia', 'Colchicine', 'IL-1 Blocker', 'Clinical Pearl'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', color: '#ef9a9a', textAlign: 'left', borderBottom: '1px solid #444' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#161616' : '#1a1a1a' }}>
                    <td style={{ padding: '8px 10px', color: GENE_COLORS[row.gene], fontWeight: 700 }}>{row.gene}</td>
                    <td style={{ padding: '8px 10px', color: '#aaa', fontSize: 10 }}>{row.group}</td>
                    <td style={{ padding: '8px 10px', color: row.amyloid === '✓✓✓' ? '#f44' : '#888' }}>{row.amyloid}</td>
                    <td style={{ padding: '8px 10px', color: row.snhl    === '✓✓✓' ? '#64b5f6' : '#888' }}>{row.snhl}</td>
                    <td style={{ padding: '8px 10px', color: row.uveitis === '✓✓✓' ? '#4caf50' : '#888' }}>{row.uveitis}</td>
                    <td style={{ padding: '8px 10px', color: row.crmo    === '✓✓✓' ? '#90a4ae' : '#888' }}>{row.crmo}</td>
                    <td style={{ padding: '8px 10px', color: row.anaemia === '✓✓✓' ? '#a5d6a7' : '#888' }}>{row.anaemia}</td>
                    <td style={{ padding: '8px 10px', color: row.colchicine === '✓✓✓' ? '#ef9a9a' : '#888' }}>{row.colchicine}</td>
                    <td style={{ padding: '8px 10px', color: row.il1 === '✓✓✓' ? '#81c784' : '#888' }}>{row.il1}</td>
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
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 10 }}>Cascade Diagnostic Testing — MAID Workup</div>
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
              <div style={{ fontWeight: 700, color: '#a5d6a7', marginBottom: 10 }}>Treatment Matrix</div>
              {Object.entries(defs.treatment_matrix || {}).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 6, paddingLeft: 12, borderLeft: '2px solid #2e7d32' }}>
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
