'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ebv-xlp-nk-t-cell-susceptibility-atlas';
const TABS = ['Overview', 'Gene Table', 'Diagnostic Map', 'Definitions'];

const GENE_COLORS = {
  'SH2D1A':  '#880e4f',  // deep crimson-pink  -- XLP1; NKT absent PATHOGNOMONIC; fatal EBV-HLH; Rituximab+HSCT
  'XIAP':    '#4a148c',  // deep purple         -- XLP2; EBV-HLH + IBD dual PATHOGNOMONIC; XIAP absent flow; iloprost
  'ITK':     '#1a237e',  // dark navy           -- EBV-CID AR; CD4 lymphopenia; both sexes; phospho-PLCg1 assay
  'MAGT1':   '#006064',  // dark teal           -- XMEN syndrome XLR; CD4 <300; NKG2D absent; Mg2+ supp
  'CD27':    '#01579b',  // dark blue           -- CD27 deficiency AR; NK cytotox normal DDx XLP1; CD27 absent flow
  'CD70':    '#b71c1c',  // deep red            -- CD70 deficiency; T-cell exhaustion PATHOGNOMONIC; very high lymphoma risk
  'TNFRSF9': '#1b5e20',  // dark green          -- 4-1BB deficiency; 4-1BB agonists CONTRAINDICATED
  'RASGRP1': '#4e342e',  // dark brown          -- RASGRP1 AR; CID+EBV+autoimmunity UNIQUE; pERK absent
};

const GENE_INFO = {
  'SH2D1A':  { full: 'SH2D1A-128aa / XLR / SAP (SLAM-associated protein) / SLAM-TCR adaptor / Xq25 / XLP1 / NKT-ABSENT-PATHOGNOMONIC / EBV-Fatal-HLH / SAP-Absent-Intracellular-Flow / Rituximab-IMMEDIATELY / HSCT-Curative', locus: 'Xq25',    size: '128 aa / 15 kDa', inh: 'X-linked recessive (XLR; males only)' },
  'XIAP':    { full: 'XIAP-497aa / XLR / X-linked inhibitor of apoptosis / BIR1-BIR2-BIR3-RING / Xq25 / XLP2 / EBV-HLH+IBD-DUAL-PATHOGNOMONIC / XIAP-Absent-Intracellular-Flow / NKT-May-Be-Present-DDx-XLP1 / Iloprost / HSCT', locus: 'Xq25',    size: '497 aa / 56 kDa', inh: 'X-linked recessive (XLR; males only; Xq25 near SH2D1A)' },
  'ITK':     { full: 'ITK-620aa / AR / IL-2-inducible T-cell kinase / Tec family / 5q33.3 / ITK-Deficiency / EBV-CID-Both-Sexes / CD4-Lymphopenia-200-500 / NKT-Reduced / SAP-PRESENT-DDx-XLP1 / Phospho-PLCg1-Defect-TCR-Stimulation', locus: '5q33.3',  size: '620 aa / 72 kDa', inh: 'Autosomal recessive (AR; 5q33.3; both sexes)' },
  'MAGT1':   { full: 'MAGT1-335aa / XLR / Magnesium transporter 1 / ER-Mg2+-channel + OST-subunit / Xq21.1 / XMEN-Syndrome / CD4-Very-Low-<300 / NKG2D-ABSENT-NK-CD8 / Free-Mg2+-Reduced-Diagnostic / Mg2+-Supplementation-Level-B / Hodgkin-Lymphoma', locus: 'Xq21.1',  size: '335 aa / 38 kDa', inh: 'X-linked recessive (XLR; males only; Xq21.1)' },
  'CD27':    { full: 'CD27-260aa / AR / TNF receptor superfamily member 7 / CD27-CD70 co-stimulatory axis / 12p13.31 / CD27-Deficiency / CD27-ABSENT-Surface-Flow / NK-CYTOTOXICITY-NORMAL-DDx-XLP1-XIAP / Absent-Switched-Memory-B-Cells / IVIG+Rituximab+HSCT', locus: '12p13.31', size: '260 aa / 29 kDa', inh: 'Autosomal recessive (AR; 12p13.31; both sexes)' },
  'CD70':    { full: 'CD70-193aa / AR / TNF superfamily member 7 / CD27 ligand / 19p13.3 / CD70-Deficiency / EBV-T-CELL-EXHAUSTION-PATHOGNOMONIC / Very-High-Lymphoma-Risk / CD70-Absent-Post-Stimulation / Chronic-EBV-Viraemia / Valganciclovir-Prophylaxis', locus: '19p13.3',  size: '193 aa / 29 kDa', inh: 'Autosomal recessive (AR; 19p13.3; both sexes)' },
  'TNFRSF9': { full: 'TNFRSF9-255aa / AR / 4-1BB (CD137) / TNF receptor on activated T+NK cells / 1p36.23 / 4-1BB-Deficiency / 4-1BB-ABSENT-Activated-T-NK-Flow / EBV-CD8-Expansion-Defect / 4-1BB-Agonist-Urelumab-Utomilumab-CONTRAINDICATED / HSCT', locus: '1p36.23',  size: '255 aa / 27 kDa', inh: 'Autosomal recessive (AR; 1p36.23; both sexes; rare)' },
  'RASGRP1': { full: 'RASGRP1-797aa / AR / RAS guanyl-releasing protein 1 / DAG-regulated GEF / 15q14 / RASGRP1-Deficiency / AUTOIMMUNITY-OVERLAP-UNIQUE / CID+EBV+AIHA+ANA / Phospho-ERK-Absent-PMA-Stimulation / Hydroxychloroquine-Autoimmune / HSCT', locus: '15q14',    size: '797 aa / 90 kDa', inh: 'Autosomal recessive (AR; 15q14; both sexes)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'SH2D1A',  group: 'XLP1 (X-linked; SAP)',      ebv_hlh: '✓✓✓', lymphoma: '✓✓', ibd: '✗',    nkt_absent: '✓✓✓', flow_test: 'SAP intracellular flow',          xlr: '✓', unique: 'NKT CELLS (CD3+Vα24+Vβ11+) ABSENT PATHOGNOMONIC; SAP absent intracellular flow; fatal EBV-HLH >70% mortality untreated; Rituximab IMMEDIATELY + HLH-2004 protocol; HSCT curative.' },
  { gene: 'XIAP',    group: 'XLP2 (X-linked; BIRC4)',    ebv_hlh: '✓✓✓', lymphoma: '✓',  ibd: '✓✓✓', nkt_absent: '±',    flow_test: 'XIAP intracellular flow',         xlr: '✓', unique: 'EBV-HLH + IBD DUAL PHENOTYPE PATHOGNOMONIC; XIAP absent intracellular flow; NKT cells may be present (DDx XLP1); iloprost for cytopenias; IBD = severe Crohn\'s-like (perianal/fistulae).' },
  { gene: 'ITK',     group: 'AR Tec kinase (CID)',        ebv_hlh: '✓✓',  lymphoma: '✓✓', ibd: '✗',    nkt_absent: '✓✓',  flow_test: 'Phospho-PLCγ1 assay',            xlr: '✗', unique: 'FEMALE CID WITH EBV SUSCEPTIBILITY (AR 5q33.3 — both sexes); CD4 lymphopenia 200-500/μL; SAP PRESENT (DDx XLP1); phospho-PLCγ1 absent after TCR stimulation — functional diagnostic.' },
  { gene: 'MAGT1',   group: 'XMEN Syndrome (XLR)',        ebv_hlh: '✓✓',  lymphoma: '✓✓', ibd: '✗',    nkt_absent: '±',    flow_test: 'NKG2D expression on NK/CD8',     xlr: '✓', unique: 'CD4 VERY LOW (<300/μL) MOST PROMINENT; NKG2D ABSENT on NK/CD8 (Mg2+-dependent glycosylation); FREE Mg2+ REDUCED in T cells; Mg2+ SUPPLEMENTATION may restore NKG2D (Level B); Hodgkin lymphoma.' },
  { gene: 'CD27',    group: 'AR CD27 deficiency',         ebv_hlh: '✓✓',  lymphoma: '✓',  ibd: '✗',    nkt_absent: '✗',    flow_test: 'CD27 surface flow (absent)',      xlr: '✗', unique: 'CD27 ABSENT on T/B/NK surface; NK CYTOTOXICITY NORMAL (DDx XLP1/XIAP); absent switched memory B cells (CD27+IgD- = zero); EBV-triggered aplastic anaemia unique complication.' },
  { gene: 'CD70',    group: 'AR CD70 deficiency',         ebv_hlh: '✓✓',  lymphoma: '✓✓✓',ibd: '✗',   nkt_absent: '✗',    flow_test: 'CD70 post-stimulation flow',     xlr: '✗', unique: 'EBV-SPECIFIC T-CELL EXHAUSTION PATHOGNOMONIC; VERY HIGH LYMPHOMA RISK (Hodgkin + NHL); chronic EBV viraemia; CD70 absent after stimulation; valganciclovir prophylaxis pending HSCT.' },
  { gene: 'TNFRSF9', group: 'AR 4-1BB deficiency',        ebv_hlh: '✓✓',  lymphoma: '✓✓', ibd: '✗',    nkt_absent: '✗',    flow_test: '4-1BB (CD137) post-activation',  xlr: '✗', unique: '4-1BB ABSENT on activated T/NK cells; EBV-specific CD8 expansion defect; NK cytotoxicity reduced; 4-1BB AGONIST ANTIBODIES (urelumab/utomilumab) ABSOLUTELY CONTRAINDICATED.' },
  { gene: 'RASGRP1', group: 'AR RAS/MAPK CID',            ebv_hlh: '✓',   lymphoma: '±',  ibd: '✗',    nkt_absent: '✗',    flow_test: 'Phospho-ERK after PMA/PHA',      xlr: '✗', unique: 'AUTOIMMUNITY OVERLAP UNIQUE (AIHA + ANA + lupus-like alongside CID); absent phospho-ERK after PMA/PHA stimulation; SAP/XIAP/NKG2D/CD27/CD70/4-1BB all NORMAL → RASGRP1 functional assay.' },
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
  const isXLP1    = gene === 'SH2D1A';
  const isXLP2    = gene === 'XIAP';
  const isITK     = gene === 'ITK';
  const isMAGT1   = gene === 'MAGT1';
  const isCD27    = gene === 'CD27';
  const isCD70    = gene === 'CD70';
  const is4BB     = gene === 'TNFRSF9';
  const isRAS     = gene === 'RASGRP1';
  const groupLabel = isXLP1  ? 'XLP1 (X-linked SAP/NKT-Absent-PATHOGNOMONIC)'
    : isXLP2  ? 'XLP2 (X-linked EBV-IBD-Dual-PATHOGNOMONIC)'
    : isITK   ? 'ITK CID (AR; Both-Sexes; CD4-Lymphopenia)'
    : isMAGT1 ? 'XMEN Syndrome (XLR; NKG2D-Absent; Mg2+)'
    : isCD27  ? 'CD27 Deficiency (AR; NK-Cytotoxicity-Normal)'
    : isCD70  ? 'CD70 Deficiency (AR; T-Cell-Exhaustion-PATHOGNOMONIC)'
    : is4BB   ? '4-1BB Deficiency (AR; 4-1BB-Agonists-CONTRAINDICATED)'
    : 'RASGRP1 CID (AR; Autoimmunity-Overlap-UNIQUE)';
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
          <Badge text={`EBV-HLH ${Math.round(cohort.ebv_hlh_pct * 100)}%`}     color={color} />
          <Badge text={`Lymphoma ${Math.round(cohort.lymphoma_pct * 100)}%`}    color='#4a148c' />
          <Badge text={`Rituximab ${Math.round(cohort.rituximab_pct * 100)}%`}  color='#b71c1c' />
          <Badge text={`HSCT ${Math.round(cohort.hsct_pct * 100)}%`}            color='#1b5e20' />
          <Badge text={`Ferritin>10k ${Math.round(cohort.ferritin_10k_pct * 100)}%`} color='#455a64' />
          <Badge text={`Onset ${cohort.median_onset_months}mo`}                 color='#37474f' />
        </div>
      )}
    </div>
  );
}

export default function HereditaryEBVXLPNKTCellSusceptibilityAtlasPage() {
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

  if (loading) return <div style={{ color: '#fff', padding: 32 }}>Loading Hereditary-EBV-XLP-NK-T-Cell-Susceptibility-Atlas…</div>;
  if (error)   return <div style={{ color: '#f44', padding: 32 }}>Error: {error}</div>;

  const stats        = overview?.aggregate_stats || {};
  const cohortByGene = {};
  (breakdown?.gene_cohorts || []).forEach(c => { cohortByGene[c.gene] = c; });

  return (
    <div style={{ background: '#0a0a1a', minHeight: '100vh', color: '#eee', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: '#1a0a0a', borderBottom: '2px solid #880e4f', padding: '20px 32px' }}>
        <h1 style={{ color: '#f48fb1', fontSize: 22, marginBottom: 4 }}>
          🧬 Hereditary-EBV-XLP-NK-T-Cell-Susceptibility-Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#aaa' }}>
          Complete 8-Gene EBV Susceptibility / X-Linked Lymphoproliferative / NK-T Cell PID Reference ·
          SH2D1A · XIAP · ITK · MAGT1 · CD27 · CD70 · TNFRSF9 · RASGRP1 ·
          320-Patient Aggregate (8×40) · Seeds 3990-3997
        </div>
        <div style={{ marginTop: 8, display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 12 }}>
          <span style={{ background: '#880e4f', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>EBV-HLH {Math.round((stats.ebv_hlh_pct || 0) * 100)}%</span>
          <span style={{ background: '#4a148c', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Lymphoma {Math.round((stats.lymphoma_pct || 0) * 100)}%</span>
          <span style={{ background: '#b71c1c', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Rituximab {Math.round((stats.rituximab_pct || 0) * 100)}%</span>
          <span style={{ background: '#1b5e20', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>HSCT {Math.round((stats.hsct_pct || 0) * 100)}%</span>
          <span style={{ background: '#37474f', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Ferritin&gt;10k {Math.round((stats.ferritin_10k_pct || 0) * 100)}%</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid #333', background: '#1a0a0a' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{
              background: tab === t ? '#880e4f' : 'transparent', color: tab === t ? '#fff' : '#aaa',
              border: 'none', padding: '10px 22px', cursor: 'pointer', fontSize: 13, fontFamily: 'monospace'
            }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: 24 }}>

        {/* ── OVERVIEW TAB ── */}
        {tab === 'Overview' && (
          <div>
            {/* Key Facts */}
            <div style={{ background: '#1a0a0a', border: '1px solid #880e4f', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#f48fb1', marginBottom: 10 }}>Key Clinical Facts — 8 EBV-Susceptibility / XLP / NK-T Cell PID Diseases</div>
              {(overview?.key_facts || []).map((f, i) => (
                <div key={i} style={{ fontSize: 12, color: '#ddd', marginBottom: 6, paddingLeft: 12, borderLeft: '3px solid #880e4f' }}>{f}</div>
              ))}
            </div>

            {/* Disease Groups */}
            <div style={{ background: '#0a1628', border: '1px solid #01579b', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 12 }}>Disease Groups by Mechanism and Inheritance</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(270px, 1fr))', gap: 12 }}>
                {[
                  { label: 'XLR — XLP1 (SH2D1A/SAP)',      genes: 'SH2D1A',        desc: 'NKT cells ABSENT PATHOGNOMONIC; fatal EBV-HLH; SAP absent on intracellular flow; Rituximab + HSCT curative.', color: '#880e4f' },
                  { label: 'XLR — XLP2 (XIAP)',             genes: 'XIAP',          desc: 'EBV-HLH + IBD DUAL PATHOGNOMONIC; XIAP absent intracellular flow; NKT may be present; iloprost + HSCT.', color: '#4a148c' },
                  { label: 'XLR — XMEN (MAGT1)',            genes: 'MAGT1',         desc: 'CD4 <300/μL very low; NKG2D absent on NK/CD8; free Mg2+ reduced; Mg2+ supplementation Level B.', color: '#006064' },
                  { label: 'AR — ITK Deficiency',           genes: 'ITK',           desc: 'AR 5q33.3; BOTH SEXES (female presenting CID); CD4 lymphopenia; SAP present DDx XLP1; phospho-PLCγ1 assay.', color: '#1a237e' },
                  { label: 'AR — CD27/CD70 Co-stimulatory', genes: 'CD27 / CD70',   desc: 'CD27-CD70 axis absent; CD27: NK cytotox normal DDx; CD70: T-cell exhaustion PATHOGNOMONIC + very high lymphoma.', color: '#01579b' },
                  { label: 'AR — 4-1BB (TNFRSF9)',          genes: 'TNFRSF9',       desc: 'EBV-specific CD8 expansion defect; 4-1BB absent on activated T/NK; 4-1BB agonists CONTRAINDICATED.', color: '#1b5e20' },
                  { label: 'AR — RASGRP1 CID+Autoimmunity', genes: 'RASGRP1',       desc: 'CID + EBV + AUTOIMMUNITY UNIQUE (AIHA/ANA/lupus-like); phospho-ERK absent; all other EBV-PID markers normal.', color: '#4e342e' },
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
            <div style={{ marginTop: 20, background: '#1a0a0a', border: '1px solid #333', borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: '#f48fb1', marginBottom: 10 }}>Aggregate Statistics — 320 Patients (8×40)</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))', gap: 8 }}>
                {[
                  ['EBV-HLH',   stats.ebv_hlh_pct,   '%', '#880e4f'],
                  ['Lymphoma',  stats.lymphoma_pct,   '%', '#4a148c'],
                  ['Rituximab', stats.rituximab_pct,  '%', '#b71c1c'],
                  ['HSCT',      stats.hsct_pct,       '%', '#1b5e20'],
                  ['Ferritin>10k', stats.ferritin_10k_pct, '%', '#37474f'],
                ].map(([label, val, unit, color]) => (
                  <div key={label} style={{ background: '#111', border: `1px solid ${color}`, borderRadius: 6, padding: 10 }}>
                    <div style={{ fontSize: 11, color: '#aaa' }}>{label}</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color }}>{val ? Math.round(val * 100) : 0}{unit}</div>
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
                <tr style={{ background: '#1a0a0a' }}>
                  {['Gene', 'Group', 'EBV-HLH', 'Lymphoma', 'IBD', 'NKT Absent', 'Flow Test', 'XLR', 'Clinical Pearl'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', color: '#f48fb1', textAlign: 'left', borderBottom: '1px solid #444' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#161616' : '#1a0a0a' }}>
                    <td style={{ padding: '8px 10px', color: GENE_COLORS[row.gene], fontWeight: 700 }}>{row.gene}</td>
                    <td style={{ padding: '8px 10px', color: '#aaa', fontSize: 10 }}>{row.group}</td>
                    <td style={{ padding: '8px 10px', color: row.ebv_hlh  === '✓✓✓' ? '#f48fb1' : '#888' }}>{row.ebv_hlh}</td>
                    <td style={{ padding: '8px 10px', color: row.lymphoma === '✓✓✓' ? '#ce93d8' : '#888' }}>{row.lymphoma}</td>
                    <td style={{ padding: '8px 10px', color: row.ibd      === '✓✓✓' ? '#ffcc80' : '#888' }}>{row.ibd}</td>
                    <td style={{ padding: '8px 10px', color: row.nkt_absent === '✓✓✓' ? '#f48fb1' : '#888' }}>{row.nkt_absent}</td>
                    <td style={{ padding: '8px 10px', color: '#80deea', fontSize: 10 }}>{row.flow_test}</td>
                    <td style={{ padding: '8px 10px', color: row.xlr === '✓' ? '#f48fb1' : '#888' }}>{row.xlr}</td>
                    <td style={{ padding: '8px 10px', color: '#ddd', maxWidth: 340, fontSize: 10 }}>{row.unique}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* ── DIAGNOSTIC MAP TAB ── */}
        {tab === 'Diagnostic Map' && breakdown && (
          <div>
            {/* Cascade Testing */}
            <div style={{ background: '#0a1628', border: '1px solid #01579b', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 10 }}>Cascade Diagnostic Testing — EBV-Susceptibility PID Workup</div>
              {(breakdown.cascade_testing_order || []).map((step, i) => (
                <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 8, alignItems: 'flex-start' }}>
                  <span style={{ background: '#880e4f', color: '#fff', borderRadius: '50%', width: 22, height: 22, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, flexShrink: 0 }}>{i+1}</span>
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
                <div style={{ fontSize: 11, color: '#aaa', marginBottom: 4 }}><strong style={{ color: '#f48fb1' }}>Shared:</strong> {grp.shared_feature}</div>
                <div style={{ fontSize: 11, color: '#aaa' }}><strong style={{ color: '#f48fb1' }}>DDx:</strong> {grp.key_ddx}</div>
              </div>
            ))}

            {/* Pathognomonic Features */}
            <div style={{ background: '#1a0a0a', border: '1px solid #880e4f', borderRadius: 8, padding: 16 }}>
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
                <div key={d.term} style={{ background: '#1a0a0a', border: '1px solid #333', borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, color: '#f48fb1', marginBottom: 6 }}>{d.term}</div>
                  <div style={{ fontSize: 12, color: '#ccc', lineHeight: 1.5 }}>{d.definition}</div>
                </div>
              ))}
            </div>

            <div style={{ background: '#0a1628', border: '1px solid #01579b', borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 10 }}>Key Distinguishers</div>
              {Object.entries(defs.key_distinguishers || {}).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase', letterSpacing: 0.8 }}>{k.replace(/_/g, ' ')}</div>
                  <div style={{ fontSize: 12, color: '#ddd', paddingLeft: 12, borderLeft: '2px solid #01579b', marginTop: 2 }}>{v}</div>
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
