'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-scn-granulocyte-cytotoxic-granule-defect-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Treatment', 'Definitions'];

const GENE_COLORS = {
  'ELANE':  '#b71c1c',  // deep red        -- SCN-I/CyN; 21-day cycling PATHOGNOMONIC; AML risk 15%; G-CSF
  'HAX1':   '#880e4f',  // dark magenta     -- SCN-III Kostmann; RS-variant neurodegeneration; G-CSF HSCT
  'G6PC3':  '#e65100',  // deep orange      -- SCN-IV; venous angiectasias PATHOGNOMONIC; cardiac defects
  'WASP':   '#1a237e',  // dark indigo      -- WAS; microplatelets MPV<7fL PATHOGNOMONIC; HSCT gene-therapy
  'CXCR4':  '#4a148c',  // deep purple      -- WHIM; myelokathexis PATHOGNOMONIC; plerixafor FDA2024
  'LYST':   '#1b5e20',  // dark green       -- CHS; giant granules PBS PATHOGNOMONIC; NK absent; HLH accelerated
  'RAB27A': '#01579b',  // dark teal        -- GS2; silvery hair PATHOGNOMONIC; HLH no neurological; HSCT
  'AP3B1':  '#5d4037',  // dark brown       -- HPS-2; OCA+neutropenia+HLH; absent platelet dense granules PATHOGNOMONIC
};

const GENE_INFO = {
  'ELANE':  { full: 'ELANE-218aa / AD / Neutrophil-elastase / primary-azurophil-granule / 19p13.3 / SCN-I-constant-or-CyN-21day-cycling / AML-risk-15pct-constant-SCN / G-CSF-responsive-90pct / annual-BM-surveillance / CSF3R-AML-transformation', locus: '19p13.3', size: '218 aa / 25 kDa', inh: 'Autosomal dominant (AD; 19p13.3; de novo or inherited; ~1/200,000)' },
  'HAX1':   { full: 'HAX1-279aa / AR / mitochondrial-anti-apoptotic / 1q21.3 / SCN-III-Kostmann / RS-variant-neurodegeneration-isoform1+2 / isoform-1-only-neutropenia-alone / AML-risk-15-20pct / G-CSF-HSCT / HAX1-isoform-characterisation-mandatory', locus: '1q21.3', size: '279 aa / 32 kDa', inh: 'Autosomal recessive (AR; 1q21.3; both sexes; ~1/1,000,000 Kostmann original)' },
  'G6PC3':  { full: 'G6PC3-346aa / AR / glucose-6-phosphatase-ER / 17q21.31 / SCN-IV-Dürr-Boztug / venous-angiectasias-PATHOGNOMONIC / cardiac-defects-50pct / congenital-heart / renal-urogenital / G-CSF-responsive / HSCT-transformation', locus: '17q21.31', size: '346 aa / 39 kDa', inh: 'Autosomal recessive (AR; 17q21.31; both sexes; ultra-rare)' },
  'WASP':   { full: 'WASP-502aa / XLR-LOF or AD-GOF / actin-polymerisation-regulator / Xp11.22 / WAS-LOF-microthrombocytes+eczema+infections / XLN-GOF-neutropenia-only / WASP-protein-flow-severity / HSCT-curative / gene-therapy-lentiviral / microplatelets-MPV-lt7fL-PATHOGNOMONIC', locus: 'Xp11.22', size: '502 aa / 57 kDa', inh: 'X-linked recessive LOF (WAS/XLT; males) or AD GOF (XLN; males; de novo)' },
  'CXCR4':  { full: 'CXCR4-360aa / AD-GOF / chemokine-receptor-SDF1 / 2q22.1 / WHIM-Warts-Hypogammaglobulinaemia-Infections-Myelokathexis / myelokathexis-BM-PATHOGNOMONIC / plerixafor-AMD3100-FDA2024 / p.R334X-European-founder / C-terminal-truncation-GOF', locus: '2q22.1', size: '360 aa / 41 kDa', inh: 'Autosomal dominant (AD; GOF; 2q22.1; ~1/1,000,000 estimated)' },
  'LYST':   { full: 'LYST-3801aa / AR / lysosomal-trafficking-regulator / 1q42.3 / CHS-Chediak-Higashi / giant-azurophil-granules-PBS-PATHOGNOMONIC / NK-cell-absent / accelerated-phase-HLH-EBV / partial-OCA-silvery-hair / HSCT-before-accelerated-phase', locus: '1q42.3', size: '3801 aa / 430 kDa', inh: 'Autosomal recessive (AR; 1q42.3; both sexes; ~1/1,000,000 ultra-rare)' },
  'RAB27A': { full: 'RAB27A-221aa / AR / small-GTPase-vesicle-exocytosis / 15q21.3 / GS2-Griscelli-2 / silvery-hair-large-irregular-clumps-PATHOGNOMONIC / HLH-dominant-80pct / NK-cytotoxicity-absent / NO-giant-PBS-granules / NO-neurological / HSCT-curative', locus: '15q21.3', size: '221 aa / 25 kDa', inh: 'Autosomal recessive (AR; 15q21.3; both sexes; ultra-rare)' },
  'AP3B1':  { full: 'AP3B1-1094aa / AR / AP-3-complex-beta1-LRO-biogenesis / 5q14.1 / HPS-2-Hermansky-Pudlak / OCA+neutropenia+HLH / absent-platelet-dense-granules-EM-PATHOGNOMONIC / NO-severe-pulmonary-fibrosis-unlike-HPS1 / NK-defect-HLH / HSCT-for-HLH', locus: '5q14.1', size: '1094 aa / 121 kDa', inh: 'Autosomal recessive (AR; 5q14.1; both sexes; ultra-rare HPS-2 subtype)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'ELANE',  group: 'SCN-I / Cyclic Neutropenia',      granules_pbs: '✗', hlh: '✗', albinism: '✗', hsct: '±',  rx_drug: 'G-CSF',          unique: '21-DAY CYCLING PATHOGNOMONIC for CyN (vs constant ANC <200 SCN-I); same gene causes both; AML/MDS risk 15% in SCN-I; annual BM surveillance mandatory; G-CSF responsive 90%; 3-week serial ANC to confirm cycling.' },
  { gene: 'HAX1',   group: 'SCN-III Kostmann',                granules_pbs: '✗', hlh: '✗', albinism: '✗', hsct: '✓',  rx_drug: 'G-CSF / HSCT',   unique: 'RS VARIANT: neurodegeneration (epilepsy, ID, regression) when BOTH isoforms 1+2 affected; isoform-1-only = neutropenia ONLY; characterise isoform at diagnosis; AML risk ~15-20%; HSCT for RS-variant or transformation.' },
  { gene: 'G6PC3',  group: 'SCN-IV Dürr-Boztug',             granules_pbs: '✗', hlh: '✗', albinism: '✗', hsct: '±',  rx_drug: 'G-CSF + cardiac', unique: 'VENOUS ANGIECTASIAS PATHOGNOMONIC — visible subcutaneous venous malformations on skin; ONLY SCN subtype with vascular skin findings; cardiac defects ~50%; examine skin + echo at ALL SCN diagnoses.' },
  { gene: 'WASP',   group: 'Wiskott-Aldrich / XLN',          granules_pbs: '✗', hlh: '±', albinism: '✗', hsct: '✓✓', rx_drug: 'HSCT / GT',      unique: 'MICROPLATELETS (MPV <7 fL) PATHOGNOMONIC — combined thrombocytopaenia + TINY platelet size is unique to WAS/XLT; eczema + recurrent infections complete triad in males; WASP protein flow cytometry for diagnosis and severity; HSCT curative.' },
  { gene: 'CXCR4',  group: 'WHIM Syndrome',                  granules_pbs: '✗', hlh: '✗', albinism: '✗', hsct: '±',  rx_drug: 'Plerixafor',     unique: 'MYELOKATHEXIS PATHOGNOMONIC — BM neutrophils with hypersegmented nuclei connected by thin strands; BM hypercellular but peripheral neutropenia; PLERIXAFOR FDA2024 first targeted therapy for primary immunodeficiency; HPV warts refractory.' },
  { gene: 'LYST',   group: 'Chediak-Higashi Syndrome',       granules_pbs: '✓✓✓', hlh: '✓✓✓', albinism: '✓', hsct: '✓✓', rx_drug: 'HSCT urgent',   unique: 'GIANT GRANULES PBS PATHOGNOMONIC — giant azurophil granules visible in ALL granule-containing cells on standard Giemsa PBS; NK cell absent; ACCELERATED PHASE (EBV-HLH) >80% without HSCT; HSCT BEFORE accelerated phase = far better outcomes.' },
  { gene: 'RAB27A', group: 'Griscelli Syndrome Type 2',      granules_pbs: '✗', hlh: '✓✓✓', albinism: '✓', hsct: '✓✓', rx_drug: 'HSCT + HLH-Rx', unique: 'SILVERY HAIR + HLH (NO PBS giant granules, NO neurological) PATHOGNOMONIC for GS2; hair shaft microscopy: large irregular clumps; DISTINGUISH from CHS (PBS granules present in CHS); DISTINGUISH from GS1 (neurological in GS1); HSCT curative for HLH risk.' },
  { gene: 'AP3B1',  group: 'HPS-2 (Hermansky-Pudlak)',      granules_pbs: '✗', hlh: '✓',  albinism: '✓', hsct: '✓',  rx_drug: 'HSCT + DDAVP',  unique: 'ABSENT PLATELET DENSE GRANULES ON EM PATHOGNOMONIC for all HPS; HPS-2 UNIQUE: neutropenia + NK defect + HLH susceptibility; HPS-1 has pulmonary fibrosis (main cause death) instead; OCA + neutropenia + bleeding + HLH = HPS-2 until proven.' },
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
  const groupLabel = {
    ELANE:  'SCN-I-CyclicNeutropenia-21day-PATHOGNOMONIC-AML-15pct-G-CSF-Annual-BM',
    HAX1:   'SCN-III-Kostmann-RS-Variant-Neurodegeneration-isoform-1+2-G-CSF-HSCT',
    G6PC3:  'SCN-IV-VenousAngiectasias-PATHOGNOMONIC-Cardiac-50pct-G6PC3-ER-Stress',
    WASP:   'WAS-Microplatelets-MPV-lt7fL-PATHOGNOMONIC-Eczema-HSCT-XLN-GOF',
    CXCR4:  'WHIM-Myelokathexis-PATHOGNOMONIC-Plerixafor-FDA2024-CXCR4-GOF',
    LYST:   'CHS-GiantGranules-PBS-PATHOGNOMONIC-NK-Absent-HLH-Accelerated-HSCT',
    RAB27A: 'GS2-SilveryHair-PATHOGNOMONIC-HLH-No-PBS-Granules-No-Neurological-HSCT',
    AP3B1:  'HPS2-OCA-Neutropenia-HLH-AbsentDenseGranules-EM-PATHOGNOMONIC-No-Fibrosis',
  }[gene] || gene;

  return (
    <div style={{
      background: '#1a1a2e', border: `2px solid ${color}`, borderRadius: 8,
      padding: 16, marginBottom: 16
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <span style={{
          background: color, color: '#fff', fontWeight: 700, fontSize: 18,
          padding: '4px 12px', borderRadius: 6
        }}>{gene}</span>
        <div>
          <div style={{ color: '#ccc', fontSize: 12 }}>{info.locus} · {info.size}</div>
          <div style={{ color: '#aaa', fontSize: 11 }}>{info.inh}</div>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#e8c270', marginBottom: 6 }}>
        <strong>Group:</strong> {groupLabel}
      </div>
      {cohort && (
        <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#aaa', marginBottom: 6 }}>
          <span>n={cohort.n}</span>
          <span>M:{cohort.male_n} F:{cohort.female_n}</span>
          <span>Onset median: {cohort.median_onset_months}mo</span>
          <span>HSCT rate: {(cohort.hsct_pct * 100).toFixed(0)}%</span>
          <span>Transform: {(cohort.transform_pct * 100).toFixed(0)}%</span>
        </div>
      )}
      <div style={{ fontSize: 11, color: '#9ecfff', marginTop: 6, lineHeight: 1.5 }}>
        <strong>Pathognomonic:</strong> {cohort?.pathognomonic || info.full}
      </div>
      <div style={{ fontSize: 11, color: '#a8e6cf', marginTop: 6, lineHeight: 1.5 }}>
        <strong>Key Rx:</strong> {cohort?.key_drug}
      </div>
    </div>
  );
}

export default function Page() {
  const [tab, setTab]         = useState('Overview');
  const [overview, setOv]     = useState(null);
  const [breakdown, setBk]    = useState(null);
  const [defs, setDefs]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const ep = (tab === 'Definitions') ? 'definitions'
             : (tab === 'Pathway Map' || tab === 'Treatment') ? 'breakdown'
             : 'overview';
    const setter = (tab === 'Definitions') ? setDefs
                 : (tab === 'Pathway Map' || tab === 'Treatment') ? setBk
                 : setOv;
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => { setter(d); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, [tab]);

  const tabStyle = (t) => ({
    padding: '8px 16px', cursor: 'pointer', borderRadius: '6px 6px 0 0',
    background: tab === t ? '#1b5e20' : '#1a1a2e',
    color: '#fff', border: 'none', marginRight: 4, fontWeight: tab === t ? 700 : 400,
  });

  return (
    <div style={{ background: '#0d0d1a', minHeight: '100vh', color: '#f0f0f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ background: 'linear-gradient(135deg,#1b5e20,#4a148c)', borderRadius: 10, padding: 24, marginBottom: 24 }}>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>
            &#x1f9ec; Hereditary-SCN-Granulocyte-Cytotoxic-Granule-Defect-Atlas
          </h1>
          <p style={{ margin: '8px 0 0', fontSize: 13, opacity: 0.9 }}>
            Complete 8-Gene Reference: ELANE-HAX1-G6PC3-WASP-CXCR4-LYST-RAB27A-AP3B1
            · 320 Patients (8×40) · Seeds 4166-4173
          </p>
          <p style={{ margin: '4px 0 0', fontSize: 11, opacity: 0.8 }}>
            SCN-I/CyN (ELANE-21day-cycling) · Kostmann-RS (HAX1-neurodegeneration) · SCN-IV (G6PC3-angiectasias)
            · WAS/XLN (WASP-microplatelets) · WHIM (CXCR4-myelokathexis-Plerixafor-FDA2024)
            · CHS (LYST-giant-granules-HLH) · GS2 (RAB27A-silvery-hair-HLH) · HPS-2 (AP3B1-OCA-neutropenia-HLH)
          </p>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', marginBottom: 0, flexWrap: 'wrap' }}>
          {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
        </div>
        <div style={{ background: '#12122a', borderRadius: '0 8px 8px 8px', padding: 24, minHeight: 400 }}>
          {loading && <div style={{ color: '#aaa' }}>Loading…</div>}
          {error && <div style={{ color: '#f44336' }}>Error: {error}</div>}

          {/* OVERVIEW TAB */}
          {tab === 'Overview' && overview && (
            <div>
              <h2 style={{ color: '#a5d6a7', marginTop: 0 }}>Atlas Overview</h2>
              <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Total Patients</div>
                  <div style={{ color: '#fff', fontSize: 28, fontWeight: 700 }}>{overview.total_patients || 320}</div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Genes</div>
                  <div style={{ color: '#ffb300', fontSize: 28, fontWeight: 700 }}>8</div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 200 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Seeds</div>
                  <div style={{ color: '#80cbc4', fontSize: 20, fontWeight: 700 }}>{overview.seeds || '4166-4173'}</div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Plerixafor Eligible</div>
                  <div style={{ color: '#ce93d8', fontSize: 18, fontWeight: 700 }}>CXCR4-WHIM</div>
                </div>
              </div>

              {/* Pathway group badges */}
              <div style={{ marginBottom: 16 }}>
                {['ELANE','HAX1','G6PC3'].map(g => <Badge key={g} text={`SCN-maturation: ${g}`} color={GENE_COLORS[g]} />)}
                {['WASP'].map(g => <Badge key={g} text={`Actin-cytoskeleton: ${g}`} color={GENE_COLORS[g]} />)}
                {['CXCR4'].map(g => <Badge key={g} text={`Chemokine-retention: ${g}`} color={GENE_COLORS[g]} />)}
                {['LYST','RAB27A','AP3B1'].map(g => <Badge key={g} text={`Granule-biogenesis: ${g}`} color={GENE_COLORS[g]} />)}
              </div>

              {/* HSCT urgency panel */}
              <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                <div style={{ color: '#f44336', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>
                  HSCT Urgency — Critical Timing
                </div>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                  <div style={{ padding: '6px 10px', background: '#2d0000', borderRadius: 6, fontSize: 11, color: '#ff5252' }}>
                    <strong>LYST/CHS — URGENT:</strong> HSCT BEFORE accelerated phase (EBV-HLH); &gt;80% develop HLH without HSCT; monthly EBV surveillance
                  </div>
                  <div style={{ padding: '6px 10px', background: '#2d0000', borderRadius: 6, fontSize: 11, color: '#ff5252' }}>
                    <strong>RAB27A/GS2 — URGENT:</strong> HSCT before HLH trigger; HLH in &gt;80%; EBV surveillance monthly
                  </div>
                  <div style={{ padding: '6px 10px', background: '#1a003d', borderRadius: 6, fontSize: 11, color: '#ce93d8' }}>
                    <strong>WASP/WAS — EARLY:</strong> HSCT at diagnosis if severe; gene therapy option; WASP protein guides timing
                  </div>
                  <div style={{ padding: '6px 10px', background: '#002d00', borderRadius: 6, fontSize: 11, color: '#4caf50' }}>
                    <strong>ELANE/SCN-I — CONDITIONAL:</strong> HSCT only for G-CSF failure or MDS/AML transformation; annual BM surveillance
                  </div>
                </div>
              </div>

              {/* Pathognomonic highlights */}
              {overview.pathognomonic_highlights && (
                <div style={{ marginBottom: 16 }}>
                  <h3 style={{ color: '#a5d6a7', fontSize: 14 }}>Pathognomonic Highlights</h3>
                  <ul style={{ paddingLeft: 20, margin: 0 }}>
                    {overview.pathognomonic_highlights.map((f, i) => (
                      <li key={i} style={{ fontSize: 12, color: '#ccc', marginBottom: 6, lineHeight: 1.5 }}>{f}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Gene cards */}
              <h3 style={{ color: '#a5d6a7', fontSize: 14, marginBottom: 12 }}>Gene Cohort Summaries (40 patients each)</h3>
              {(overview.gene_cohorts || []).map((c) => (
                <GeneCard key={c.gene} gene={c.gene} color={GENE_COLORS[c.gene] || '#555'}
                          info={GENE_INFO[c.gene] || {}} cohort={c} />
              ))}
            </div>
          )}

          {/* GENE TABLE TAB */}
          {tab === 'Gene Table' && overview && (
            <div>
              <h2 style={{ color: '#a5d6a7', marginTop: 0 }}>Gene Comparison Table</h2>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: '#1a1a2e' }}>
                      {['Gene','Group','aa','Locus','kDa','Inheritance','Giant Granules PBS','HLH Risk','Albinism','HSCT','Key Rx','Unique Finding'].map(h => (
                        <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#a5d6a7', borderBottom: '1px solid #333' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {PHENOTYPE_ROWS.map((row, i) => (
                      <tr key={row.gene} style={{ background: i % 2 === 0 ? '#0d0d1a' : '#12122a' }}>
                        <td style={{ padding: '6px', color: GENE_COLORS[row.gene], fontWeight: 700 }}>{row.gene}</td>
                        <td style={{ padding: '6px', color: '#ccc' }}>{row.group}</td>
                        <td style={{ padding: '6px', color: '#aaa' }}>{GENE_INFO[row.gene]?.size?.split('/')[0]}</td>
                        <td style={{ padding: '6px', color: '#9ecfff' }}>{GENE_INFO[row.gene]?.locus}</td>
                        <td style={{ padding: '6px', color: '#aaa' }}>{GENE_INFO[row.gene]?.size?.split('/')[1]}</td>
                        <td style={{ padding: '6px', color: '#ccc', fontSize: 10 }}>{GENE_INFO[row.gene]?.inh?.split('(')[0]}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.granules_pbs === '✓✓✓' ? '#f55' : '#888' }}>{row.granules_pbs}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.hlh === '✓✓✓' ? '#f55' : row.hlh === '✓' ? '#ff8' : row.hlh === '±' ? '#ffb' : '#888' }}>{row.hlh}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.albinism === '✓' ? '#ff8' : '#888' }}>{row.albinism}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.hsct === '✓✓' ? '#f55' : row.hsct === '✓' ? '#ff8' : row.hsct === '±' ? '#ffb' : '#888' }}>{row.hsct}</td>
                        <td style={{ padding: '6px', color: '#a8e6cf', fontSize: 10 }}>{row.rx_drug}</td>
                        <td style={{ padding: '6px', color: '#ccc', fontSize: 10, maxWidth: 300 }}>{row.unique}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* PATHWAY MAP TAB */}
          {tab === 'Pathway Map' && breakdown && (
            <div>
              <h2 style={{ color: '#a5d6a7', marginTop: 0 }}>Granulocyte Pathway Groups & Mechanism Map</h2>

              {/* HLH risk table */}
              {breakdown.hlh_risk_table && (
                <div style={{ background: '#2d0000', border: '1px solid #f44336', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                  <div style={{ color: '#f44336', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>HLH / Transformation Risk by Gene</div>
                  {Object.entries(breakdown.hlh_risk_table).map(([gene, risk]) => (
                    <div key={gene} style={{ display: 'flex', gap: 10, marginBottom: 4, fontSize: 11 }}>
                      <Badge text={gene} color={GENE_COLORS[gene] || '#555'} />
                      <span style={{ color: '#ccc' }}>{risk}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Pathway groups */}
              {Object.entries(breakdown.pathway_groups || {}).map(([grp, data]) => (
                <div key={grp} style={{ background: '#12122a', border: '1px solid #333', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                    <span style={{ color: '#a5d6a7', fontWeight: 700, fontSize: 13 }}>
                      {grp.replace(/_/g, ' ').toUpperCase()}
                    </span>
                    {(data.genes || []).map(g => (
                      <Badge key={g} text={g} color={GENE_COLORS[g] || '#555'} />
                    ))}
                  </div>
                  {data.mechanism && (
                    <div style={{ fontSize: 11, color: '#e8c270', marginBottom: 6 }}>
                      <strong>Mechanism:</strong> {data.mechanism}
                    </div>
                  )}
                  {data.shared_feature && (
                    <div style={{ fontSize: 11, color: '#9ecfff', marginBottom: 6 }}>
                      <strong>Shared:</strong> {data.shared_feature}
                    </div>
                  )}
                  {data.key_ddx && (
                    <div style={{ fontSize: 11, color: '#a8e6cf', lineHeight: 1.5 }}>
                      <strong>DDx:</strong> {data.key_ddx}
                    </div>
                  )}
                </div>
              ))}

              {/* Cascade testing */}
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Cascade Testing Algorithm</h3>
              <ol style={{ paddingLeft: 20, margin: 0 }}>
                {(breakdown.cascade_testing_algorithm || []).map((step, i) => (
                  <li key={i} style={{ fontSize: 11, color: '#ccc', marginBottom: 8, lineHeight: 1.5 }}>{step}</li>
                ))}
              </ol>

              {/* Plerixafor spectrum */}
              {breakdown.plerixafor_spectrum && (
                <div style={{ marginTop: 20 }}>
                  <h3 style={{ color: '#ce93d8', fontSize: 13, marginBottom: 10 }}>Plerixafor (AMD3100) — First Targeted Primary Immunodeficiency Therapy</h3>
                  {Object.entries(breakdown.plerixafor_spectrum).map(([k, v]) => (
                    <div key={k} style={{ background: '#1a003d', borderRadius: 6, padding: 10, marginBottom: 8 }}>
                      <div style={{ color: '#ce93d8', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>
                        {k.replace(/_/g, ' ').toUpperCase()}
                      </div>
                      <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.5 }}>{v}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* TREATMENT TAB */}
          {tab === 'Treatment' && breakdown && (
            <div>
              <h2 style={{ color: '#a5d6a7', marginTop: 0 }}>Treatment Pathways</h2>
              {(breakdown.treatment_pathways || []).map((tp, i) => (
                <div key={i} style={{ background: '#0d1a0d', border: '1px solid #2e7d32', borderRadius: 8, padding: 14, marginBottom: 10 }}>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 6, alignItems: 'center' }}>
                    <Badge text={tp.gene} color={GENE_COLORS[tp.gene] || '#555'} />
                    <span style={{ color: '#a8e6cf', fontWeight: 700, fontSize: 12 }}>{tp.disease}</span>
                  </div>
                  <div style={{ fontSize: 11, color: '#9ecfff', marginBottom: 4 }}>
                    <strong>First-line:</strong> {tp.first_line}
                  </div>
                  {tp.precision_therapy && (
                    <div style={{ fontSize: 11, color: '#ffcc80', marginBottom: 4 }}>
                      <strong>Precision:</strong> {tp.precision_therapy}
                    </div>
                  )}
                  {tp.avoid && (
                    <div style={{ fontSize: 11, color: '#ff5252', marginBottom: 4 }}>
                      <strong>Avoid/CI:</strong> {tp.avoid}
                    </div>
                  )}
                  {tp.monitoring && (
                    <div style={{ fontSize: 11, color: '#aaa' }}>
                      <strong>Monitor:</strong> {tp.monitoring}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* DEFINITIONS TAB */}
          {tab === 'Definitions' && defs && (
            <div>
              <h2 style={{ color: '#a5d6a7', marginTop: 0 }}>Key Definitions</h2>

              {/* Key distinguishers */}
              {defs.key_distinguishers && (
                <div style={{ marginBottom: 20 }}>
                  <h3 style={{ color: '#ffb300', fontSize: 13, marginBottom: 10 }}>Key Distinguishers</h3>
                  {Object.entries(defs.key_distinguishers).map(([k, v]) => (
                    <div key={k} style={{ background: '#1a1a2e', borderRadius: 6, padding: 10, marginBottom: 8 }}>
                      <div style={{ color: '#9ecfff', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>
                        {k.replace(/_/g, ' ').toUpperCase()}
                      </div>
                      <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.5 }}>{v}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Treatment matrix */}
              {defs.treatment_matrix && (
                <div style={{ marginBottom: 20 }}>
                  <h3 style={{ color: '#ffb300', fontSize: 13, marginBottom: 10 }}>Treatment Matrix</h3>
                  {Object.entries(defs.treatment_matrix).map(([k, v]) => (
                    <div key={k} style={{ background: '#0d1a0d', border: '1px solid #2e7d32', borderRadius: 6, padding: 10, marginBottom: 8 }}>
                      <div style={{ color: '#a8e6cf', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>
                        {k.replace(/_/g, ' ').toUpperCase()}
                      </div>
                      <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.5 }}>{v}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Full definitions */}
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Full Definitions</h3>
              {(defs.definitions || []).map((d, i) => (
                <div key={i} style={{ background: '#12122a', borderRadius: 8, padding: 14, marginBottom: 10 }}>
                  <div style={{ color: '#a5d6a7', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{d.term}</div>
                  <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.6 }}>{d.definition}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
