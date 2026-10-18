'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-msmd-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'IFNGR1': '#b71c1c',  // deep red       -- AD partial + AR complete; most common MSMD; serum IFN-γ elevated PATHOGNOMONIC
  'IFNGR2': '#c62828',  // red             -- AR complete; IFNGR1 surface normal; serum IFN-γ elevated; HSCT required
  'IL12B':  '#1565c0',  // deep blue       -- AR; IL-12p40; BCG + Salmonella; IFN-γ therapy effective
  'IL12RB1':'#1a237e',  // deep navy       -- AR; most common AR MSMD worldwide; IL-12Rβ1 absent on flow
  'STAT1':  '#4a148c',  // deep purple     -- AD/AR LOF; MSMD + HSE; opposite of STAT1 GOF (CMC atlas)
  'IRF8':   '#e65100',  // burnt orange    -- AD LOF cDC1 loss; AR LOF total DC absence SCID-severity
  'ISG15':  '#1b5e20',  // deep green      -- AR; CNS calcifications PATHOGNOMONIC; dual MSMD + IFN excess
  'CYBB':   '#4e342e',  // dark brown      -- XL hypomorphic; DHR normal trap; macrophage-selective
};

const GENE_INFO = {
  'IFNGR1':  { full: 'IFNGR1-472aa / AD-Partial+AR-Complete / Most-Common-MSMD / Serum-IFNg-ELEVATED-PATHOGNOMONIC / BCGosis / BCG-ABSOLUTELY-CI / AD=Survivable / AR=HSCT', locus: '6q23.3',   size: '472 aa / 54 kDa', inh: 'AD partial LOF / AR complete LOF' },
  'IFNGR2':  { full: 'IFNGR2-337aa / AR-Complete / IFNg-Receptor-2 / IFNGR1-Surface-NORMAL-Flow-Distinguishes / Serum-IFNg-ELEVATED / HSCT-Required / BCG-CI',                  locus: '21q22.11',  size: '337 aa / 38 kDa', inh: 'AR complete LOF' },
  'IL12B':   { full: 'IL12B-328aa / AR / IL12p40-Shared-IL12-IL23 / BCG+Salmonella / Serum-IFNg-LOW / IFNg-Therapy-EFFECTIVE / Consanguineous-Founder-Mutations',               locus: '5q33.3',    size: '328 aa / 37 kDa', inh: 'AR' },
  'IL12RB1': { full: 'IL12RB1-662aa / AR / Most-Common-AR-MSMD-Worldwide / IL12Rb1-Absent-Flow-Cytometry / BCG+Salmonella / IFNg-Therapy-EFFECTIVE / Better-Prognosis-IFNGR',   locus: '19p13.11',  size: '662 aa / 75 kDa', inh: 'AR' },
  'STAT1':   { full: 'STAT1-LOF-750aa / AD-LOF+AR-LOF / MSMD+HSE-COMBINED / Opposite-STAT1-GOF-CMC-Atlas / AD=MSMD+HSE / AR=Fatal-Viral+Myco / BCG-CI',                        locus: '2q32.2',    size: '750 aa / 87 kDa', inh: 'AD LOF / AR LOF' },
  'IRF8':    { full: 'IRF8-426aa / AD-LOF+AR-LOF / cDC1-BDCA3-XCR1-Absent-Flow / AD=Selective-cDC1-Loss / AR-K108E=Total-DC-Monocyte-Absence-SCID-Severity / HSCT-Urgently',   locus: '16q24.1',   size: '426 aa / 49 kDa', inh: 'AD LOF / AR LOF' },
  'ISG15':   { full: 'ISG15-165aa / AR / CNS-Calcifications-PATHOGNOMONIC / MSMD+CNS-IFN-Excess / USP18-Destabilised / Influenza-Susceptibility / IFNg-Therapy-Effective',       locus: '1p36.33',   size: '165 aa / 18 kDa', inh: 'AR' },
  'CYBB':    { full: 'CYBB-Hypomorphic-570aa / XL / R-MSMD / DHR-NORMAL-DIAGNOSTIC-TRAP / Macrophage-NOX2-Selective / BCGosis / NO-Aspergillus / Males-Only / CYBB-Seq-Required', locus: 'Xp21.1',   size: '570 aa / 91 kDa', inh: 'XL hypomorphic' },
};

const PHENOTYPE_ROWS = [
  { gene: 'IFNGR1', group: 'IFNg-Receptor', ntm: '✓✓✓', bcg: '✓✓✓', tb: '✓✓', salmonella: '±', viral: '±', cns_calcif: '−', hsct: '✓✓', ifng_elev: '✓✓✓', ifng_rx: '±',   unique: 'Serum IFN-γ ELEVATED (pathognomonic receptor defect); AD partial = survivable; AR complete = fatal without HSCT; BCG ABSOLUTELY CI; most common MSMD gene' },
  { gene: 'IFNGR2', group: 'IFNg-Receptor', ntm: '✓✓✓', bcg: '✓✓✓', tb: '✓✓', salmonella: '±', viral: '✓', cns_calcif: '−', hsct: '✓✓✓',ifng_elev: '✓✓✓', ifng_rx: '−',  unique: 'Serum IFN-γ ELEVATED; IFNGR1 surface NORMAL on flow (distinguishes from IFNGR1 defect); ALL AR complete LOF — no AD partial form; HSCT required' },
  { gene: 'IL12B',  group: 'IL12-Axis',     ntm: '✓✓',  bcg: '✓✓✓', tb: '✓✓', salmonella: '✓✓✓',viral: '±',cns_calcif: '−', hsct: '±',   ifng_elev: '−',   ifng_rx: '✓✓✓',unique: 'NON-TYPHOIDAL SALMONELLA BACTEREMIA key feature; serum IFN-γ LOW; IL-12p40 shared with IL-23 → mild Th17 impairment; exogenous IFN-γ DRAMATICALLY EFFECTIVE' },
  { gene: 'IL12RB1',group: 'IL12-Axis',     ntm: '✓✓',  bcg: '✓✓✓', tb: '✓✓', salmonella: '✓✓✓',viral: '±',cns_calcif: '−', hsct: '±',   ifng_elev: '−',   ifng_rx: '✓✓✓',unique: 'MOST COMMON AR MSMD WORLDWIDE; IL-12Rβ1 absent on T/NK surface by flow cytometry; combined IL-12 + IL-23 receptor loss; Salmonella; IFN-γ effective' },
  { gene: 'STAT1',  group: 'STAT1-LOF',     ntm: '✓✓',  bcg: '✓✓',  tb: '✓',  salmonella: '±', viral: '✓✓✓',cns_calcif:'±', hsct: '±',   ifng_elev: '−',   ifng_rx: '±',   unique: 'STAT1 LOF ≠ STAT1 GOF: AD LOF → MSMD + HSE (most common genetic HSE in children); AR LOF → fatal viral+mycobacterial combined; BCG CI; opposite of STAT1 GOF (CMC)' },
  { gene: 'IRF8',   group: 'IRF8-DC',       ntm: '✓✓',  bcg: '✓✓',  tb: '✓',  salmonella: '±', viral: '±', cns_calcif: '−', hsct: '✓✓',  ifng_elev: '−',   ifng_rx: '✓✓',  unique: 'IRF8 AD-LOF (T80A): selective cDC1 (BDCA3+/XCR1+) absent — flow cytometry diagnostic; IRF8 AR-LOF (K108E): total DC+monocyte absence = SCID severity → HSCT urgently' },
  { gene: 'ISG15',  group: 'ISG15-Paradox', ntm: '✓✓',  bcg: '✓✓',  tb: '✓',  salmonella: '±', viral: '✓✓',cns_calcif: '✓✓✓',hsct: '±',ifng_elev: '−',   ifng_rx: '✓✓✓',unique: 'CNS CALCIFICATIONS (basal ganglia/periventricular) PATHOGNOMONIC; ISG15 paradox: MSMD (IFN-γ impaired) + CNS IFN excess (USP18 destabilised); influenza susceptibility' },
  { gene: 'CYBB',   group: 'CYBB-Hypomorph',ntm: '✓✓',  bcg: '✓✓✓', tb: '✓',  salmonella: '±', viral: '±', cns_calcif: '−', hsct: '±',   ifng_elev: '−',   ifng_rx: '±',   unique: 'DHR TEST NORMAL — DIAGNOSTIC TRAP (neutrophil NOX2 preserved); macrophage-specific oxidative burst defect; BCGosis; NO Aspergillus; X-linked males only; CYBB sequencing required' },
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
  const groupLabel = ['IFNGR1','IFNGR2'].includes(gene) ? 'IFN-γ Receptor Defect'
                   : ['IL12B','IL12RB1'].includes(gene) ? 'IL-12 / IL-23 Axis Defect'
                   : gene === 'STAT1' ? 'STAT1 LOF (MSMD + HSE)'
                   : gene === 'IRF8'  ? 'IRF8 DC Deficiency'
                   : gene === 'ISG15' ? 'ISG15 Paradox Defect'
                   : 'CYBB Hypomorphic (R-MSMD)';
  const groupColor = ['IFNGR1','IFNGR2'].includes(gene) ? '#b71c1c'
                   : ['IL12B','IL12RB1'].includes(gene) ? '#1565c0'
                   : gene === 'STAT1' ? '#4a148c'
                   : gene === 'IRF8'  ? '#e65100'
                   : gene === 'ISG15' ? '#1b5e20'
                   : '#4e342e';
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          <span style={{ fontWeight: 700, fontSize: 18, color }}>{gene}</span>
          <span style={{ marginLeft: 10, fontSize: 12, background: groupColor, color: '#fff', padding: '2px 8px', borderRadius: 4 }}>{groupLabel}</span>
        </div>
        <div style={{ fontSize: 12, color: '#888', textAlign: 'right' }}>
          <div>{info.size} · {info.locus}</div>
          <div>{info.inh}</div>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#bbb', marginTop: 6 }}>{info.full}</div>
      {cohort && (
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Badge text={`NTM ${cohort.ntm_pct}%`}          color={color} />
          <Badge text={`BCG disease ${cohort.bcg_disease_pct}%`} color='#880e4f' />
          <Badge text={`TB ${cohort.tb_pct}%`}            color='#795548' />
          <Badge text={`Salmonella ${cohort.salmonella_pct}%`}  color='#1565c0' />
          <Badge text={`Viral ${cohort.viral_suscept_pct}%`}   color='#4a148c' />
          <Badge text={`CNS calcif ${cohort.cns_calcif_pct}%`} color='#1b5e20' />
          <Badge text={`HSCT ${cohort.hsct_pct}%`}        color='#e65100' />
          <Badge text={`IFNg↑ ${cohort.ifng_elevated_pct}%`}   color='#b71c1c' />
          <Badge text={`IFNg-Rx ${cohort.ifng_response_pct}%`} color='#2e7d32' />
          <Badge text={`n=${cohort.n}`}                   color='#555' />
          <Badge text={`dx-lag ${cohort.mean_dx_lag_years}y`}  color='#555' />
        </div>
      )}
      {cohort && (
        <div style={{ marginTop: 8, fontSize: 12, background: 'rgba(255,255,255,0.05)', padding: 8, borderRadius: 4 }}>
          <strong style={{ color: '#ffd54f' }}>Pathognomonic / Key Finding:</strong>{' '}
          <span style={{ color: '#ccc' }}>{cohort.pathognomonic}</span>
        </div>
      )}
    </div>
  );
}

export default function HMSMDAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefs(df);
    }).catch(e => setError(String(e)));
  }, []);

  if (error) return <div style={{ color: 'red', padding: 32 }}>Error: {error}</div>;
  if (!overview) return <div style={{ color: '#ccc', padding: 32 }}>Loading Hereditary-MSMD-Atlas…</div>;

  const cohortMap = {};
  (breakdown?.gene_cohorts || []).forEach(c => { cohortMap[c.gene] = c; });

  return (
    <div style={{ background: '#121212', minHeight: '100vh', color: '#e0e0e0', fontFamily: 'monospace', padding: 24 }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ color: '#ef9a9a', fontSize: 22, marginBottom: 4 }}>
          🧬 Hereditary-MSMD-Atlas
        </h1>
        <div style={{ color: '#90caf9', fontSize: 13 }}>
          Mendelian Susceptibility to Mycobacterial Disease / IFN-γ–IL-12 Axis Defects &nbsp;·&nbsp;
          8 Genes &nbsp;·&nbsp; {overview.total_patients} patients &nbsp;·&nbsp; seeds {overview.seeds}
        </div>
        <div style={{ fontSize: 12, color: '#888', marginTop: 4 }}>
          IUIS PID 2022 Category 7: Susceptibility to Specific Pathogens (Mycobacterial) &nbsp;·&nbsp;
          IFNGR1-IFNGR2-IL12B-IL12RB1-STAT1-IRF8-ISG15-CYBB
        </div>
      </div>

      {/* Alert box */}
      <div style={{ background: '#3e2723', border: '1px solid #b71c1c', borderRadius: 6, padding: 12, marginBottom: 16, fontSize: 12 }}>
        <strong style={{ color: '#ef9a9a' }}>⚠️ BCG ABSOLUTELY CONTRAINDICATED</strong>
        <span style={{ color: '#ffccbc', marginLeft: 8 }}>
          in ALL 8 MSMD genes. BCGosis after BCG vaccination is the most common presenting event.
          Check BCG vaccination status urgently in every suspected MSMD patient.
          IFN-γ SERUM: ELEVATED → IFNGR defect; NORMAL/LOW → IL-12 axis or STAT1/IRF8/ISG15/CYBB.
        </span>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 4, border: 'none', cursor: 'pointer',
            background: tab === t ? '#b71c1c' : '#2a2a2a', color: tab === t ? '#fff' : '#aaa', fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && (
        <div>
          {/* Aggregate stats */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>Aggregate Statistics — {overview.total_patients} patients across 8 genes</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 10 }}>
              {[
                ['NTM Infection', overview.aggregate_stats?.ntm_pct, '#b71c1c'],
                ['BCG Disease', overview.aggregate_stats?.bcg_disease_pct, '#880e4f'],
                ['TB', overview.aggregate_stats?.tb_pct, '#795548'],
                ['Salmonella Bacteremia', overview.aggregate_stats?.salmonella_pct, '#1565c0'],
                ['Viral Susceptibility', overview.aggregate_stats?.viral_suscept_pct, '#4a148c'],
                ['CNS Calcifications', overview.aggregate_stats?.cns_calcif_pct, '#1b5e20'],
                ['HSCT Required', overview.aggregate_stats?.hsct_pct, '#e65100'],
                ['Serum IFN-γ Elevated', overview.aggregate_stats?.ifng_elevated_pct, '#b71c1c'],
                ['IFN-γ Rx Response', overview.aggregate_stats?.ifng_response_pct, '#2e7d32'],
              ].map(([label, val, color]) => (
                <div key={label} style={{ background: '#2a2a2a', borderRadius: 6, padding: 10, textAlign: 'center' }}>
                  <div style={{ fontSize: 22, fontWeight: 700, color }}>{val ?? '—'}%</div>
                  <div style={{ fontSize: 11, color: '#aaa', marginTop: 2 }}>{label}</div>
                </div>
              ))}
            </div>
          </div>
          {/* Gene cards */}
          <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>8 Gene Cohorts (40 patients each)</h3>
          {Object.entries(GENE_INFO).map(([gene, info]) => (
            <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={info} cohort={cohortMap[gene]} />
          ))}
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && (
        <div>
          <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>MSMD Phenotype Matrix — 8 Genes</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1e1e1e' }}>
                  {['Gene','Group','NTM','BCG Dis.','TB','Salmonella','Viral','CNS Calcif.','HSCT','IFNg↑','IFNg-Rx','Key Distinguisher'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', color: '#ef9a9a', borderBottom: '2px solid #333', whiteSpace: 'nowrap', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#181818' : '#202020' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[row.gene], whiteSpace: 'nowrap' }}>{row.gene}</td>
                    <td style={{ padding: '7px 10px', color: '#90caf9', whiteSpace: 'nowrap', fontSize: 11 }}>{row.group}</td>
                    {[row.ntm, row.bcg, row.tb, row.salmonella, row.viral, row.cns_calcif, row.hsct, row.ifng_elev, row.ifng_rx].map((v, j) => (
                      <td key={j} style={{ padding: '7px 10px', textAlign: 'center', color: v === '−' ? '#555' : v?.includes('✓✓✓') ? '#ef9a9a' : v?.includes('✓✓') ? '#ffcc02' : '#aaa' }}>{v}</td>
                    ))}
                    <td style={{ padding: '7px 10px', color: '#ccc', fontSize: 11 }}>{row.unique}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Cascade testing */}
          <div style={{ marginTop: 24, background: '#1e1e1e', borderRadius: 8, padding: 16 }}>
            <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>Cascade Testing Protocol</h3>
            {(overview.cascade_testing_concepts || []).map((step, i) => (
              <div key={i} style={{ padding: '8px 0', borderBottom: i < overview.cascade_testing_concepts.length - 1 ? '1px solid #333' : 'none', fontSize: 13 }}>
                <span style={{ color: '#90caf9', fontWeight: 600 }}>{i + 1}. </span>
                <span style={{ color: '#ccc' }}>{step}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Pathway Map Tab */}
      {tab === 'Pathway Map' && (
        <div>
          <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>IFN-γ / IL-12 Pathway — MSMD Defect Map</h3>
          <div style={{ background: '#1a1a1a', borderRadius: 10, padding: 20, fontSize: 12 }}>
            {/* Pathway diagram as structured text */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ color: '#90caf9', fontWeight: 700, marginBottom: 8 }}>UPSTREAM — IL-12 PRODUCTION AXIS</div>
              <div style={{ paddingLeft: 16, borderLeft: '3px solid #1565c0' }}>
                <div style={{ color: '#ccc', marginBottom: 6 }}>Macrophage/DC phagocytoses mycobacterium → toll-like receptor / NOD2 activation →</div>
                <div style={{ color: '#ffcc02', marginBottom: 4 }}>  IL-12 (p35+p40) + IL-23 (p19+p40) released</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>  ↑ <span style={{ color: '#1565c0', fontWeight: 600 }}>IL12B (p40)</span> deficiency blocks BOTH IL-12 and IL-23 production</div>
                <div style={{ color: '#ccc', marginBottom: 6 }}>→ IL-12/IL-23 binds NK/T cell IL-12Rβ1/IL-12Rβ2 (IL-12) or IL-23R/IL-12Rβ1 (IL-23) →</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>  ↑ <span style={{ color: '#1a237e', fontWeight: 600 }}>IL12RB1</span> deficiency blocks BOTH IL-12 AND IL-23 receptor signalling</div>
                <div style={{ color: '#ffcc02', marginBottom: 4 }}>  → STAT4 phosphorylation → IFN-γ production from NK/T cells</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>  ↑ <span style={{ color: '#1b5e20', fontWeight: 600 }}>ISG15</span> deficiency: reduced secreted ISG15 → NK/CD8+ IFN-γ production impaired</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>  ↑ <span style={{ color: '#e65100', fontWeight: 600 }}>IRF8</span> deficiency: cDC1 absence → IL-12 production reduced → Th1/NK impaired</div>
              </div>
            </div>
            <div style={{ marginBottom: 16 }}>
              <div style={{ color: '#90caf9', fontWeight: 700, marginBottom: 8 }}>CENTRAL — IFN-γ RECEPTOR / STAT1 AXIS</div>
              <div style={{ paddingLeft: 16, borderLeft: '3px solid #b71c1c' }}>
                <div style={{ color: '#ccc', marginBottom: 6 }}>IFN-γ (dimer) → binds IFNGR1/IFNGR2 heterodimer on macrophage surface →</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>  ↑ <span style={{ color: '#b71c1c', fontWeight: 600 }}>IFNGR1</span> deficiency: ligand-binding chain absent/truncated → IFN-γ accumulates (serum elevated)</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>  ↑ <span style={{ color: '#c62828', fontWeight: 600 }}>IFNGR2</span> deficiency: signal-transducing chain absent → same serum IFN-γ elevation; IFNGR1 surface normal</div>
                <div style={{ color: '#ccc', marginBottom: 6 }}>→ JAK1/JAK2 activation → STAT1 phosphorylation (pY701) → STAT1 homodimer (GAF) →</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>  ↑ <span style={{ color: '#4a148c', fontWeight: 600 }}>STAT1 LOF</span>: effector absent → IFN-γ signalling blocked here; ALSO blocks IFN-α/β → HSE risk</div>
                <div style={{ color: '#ffcc02', marginBottom: 4 }}>  → IRF1 / GBP / iNOS / NOX2 gene expression → macrophage mycobacterial killing</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>  ↑ <span style={{ color: '#4e342e', fontWeight: 600 }}>CYBB hypomorphic</span>: NOX2 macrophage-specific activity impaired → killing step failed; neutrophil NOX2 normal → DHR normal</div>
              </div>
            </div>
            <div>
              <div style={{ color: '#90caf9', fontWeight: 700, marginBottom: 8 }}>TRIAGE KEY — Serum IFN-γ Level</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: 12 }}>
                <div style={{ background: '#3e2723', borderRadius: 6, padding: 10 }}>
                  <div style={{ color: '#ef9a9a', fontWeight: 700 }}>IFN-γ ELEVATED</div>
                  <div style={{ color: '#aaa', marginTop: 4 }}>→ IFNGR1 or IFNGR2 defect</div>
                  <div style={{ color: '#ccc', marginTop: 4, fontSize: 11 }}>Flow: IFNGR1 absent → IFNGR1 defect<br/>IFNGR1 present + IFNGR2 absent → IFNGR2 defect</div>
                </div>
                <div style={{ background: '#1a237e33', borderRadius: 6, padding: 10 }}>
                  <div style={{ color: '#90caf9', fontWeight: 700 }}>IFN-γ NORMAL / LOW</div>
                  <div style={{ color: '#aaa', marginTop: 4 }}>→ IL-12 axis / STAT1 / IRF8 / ISG15 / CYBB</div>
                  <div style={{ color: '#ccc', marginTop: 4, fontSize: 11 }}>+ Salmonella → IL12B/IL12RB1<br/>+ HSE → STAT1 LOF<br/>+ CNS calcif → ISG15<br/>+ cDC1 absent (flow) → IRF8<br/>+ DHR normal + male → CYBB hypomorphic</div>
                </div>
              </div>
            </div>
          </div>

          {/* Diagnostic groups */}
          {breakdown && Object.entries(breakdown.diagnostic_groups || {}).map(([key, grp]) => (
            <div key={key} style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginTop: 12 }}>
              <div style={{ color: '#ffd54f', fontWeight: 700, marginBottom: 6 }}>
                {grp.genes?.join(' / ')} — {grp.mechanism?.slice(0, 80)}…
              </div>
              <div style={{ color: '#aaa', fontSize: 12 }}><strong style={{ color: '#90caf9' }}>Shared:</strong> {grp.shared_feature}</div>
              <div style={{ color: '#aaa', fontSize: 12, marginTop: 4 }}><strong style={{ color: '#90caf9' }}>DDx:</strong> {grp.key_ddx}</div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && defs && (
        <div>
          <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>Definitions & Key Distinguishers</h3>
          {/* Key distinguishers */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <h4 style={{ color: '#ffd54f', marginBottom: 10 }}>Key Distinguishers</h4>
            {Object.entries(defs.key_distinguishers || {}).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid #333' }}>
                <div style={{ color: '#90caf9', fontWeight: 600, fontSize: 12 }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ color: '#ccc', fontSize: 12, marginTop: 3 }}>{v}</div>
              </div>
            ))}
          </div>
          {/* Definitions */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <h4 style={{ color: '#ffd54f', marginBottom: 10 }}>Glossary</h4>
            {Object.entries(defs.definitions || {}).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid #333' }}>
                <div style={{ color: '#90caf9', fontWeight: 600, fontSize: 12 }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ color: '#ccc', fontSize: 12, marginTop: 3 }}>{v}</div>
              </div>
            ))}
          </div>
          {/* Diagnostic pathway */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16 }}>
            <h4 style={{ color: '#ffd54f', marginBottom: 10 }}>Diagnostic Pathway</h4>
            {(defs.diagnostic_pathway || []).map((step, i) => (
              <div key={i} style={{ padding: '7px 0', borderBottom: i < defs.diagnostic_pathway.length - 1 ? '1px solid #333' : 'none', fontSize: 13 }}>
                <span style={{ color: '#b71c1c', fontWeight: 700 }}>{step.split(':')[0]}: </span>
                <span style={{ color: '#ccc' }}>{step.split(':').slice(1).join(':')}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
