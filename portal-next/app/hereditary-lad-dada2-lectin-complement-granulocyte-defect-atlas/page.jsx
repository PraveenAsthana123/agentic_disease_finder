'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-lad-dada2-lectin-complement-granulocyte-defect-atlas';
const TABS = ['Overview', 'Gene Table', 'Diagnostic Map', 'Definitions'];

const GENE_COLORS = {
  'ITGB2':   '#880e4f',  // dark crimson      -- LAD-I; CD18 absent; delayed cord separation PATHOGNOMONIC; HSCT only curative
  'FERMT3':  '#b71c1c',  // deep red          -- LAD-III; Kindlin-3; combined infections+bleeding PATHOGNOMONIC; HSCT mandatory
  'ADA2':    '#e65100',  // deep orange       -- DADA2; vasculopathy+stroke+livedo PATHOGNOMONIC; anti-TNF curative
  'CEBPE':   '#1a237e',  // dark navy         -- SGD; bilobed nuclei PATHOGNOMONIC; absent lactoferrin; NBT normal
  'CXCR2':   '#0d47a1',  // dark blue         -- SCN6; neutrophil exit failure; G-CSF effective
  'MBL2':    '#1b5e20',  // dark green        -- MBL deficiency; partial immunodeficiency; lectin pathway absent
  'MASP2':   '#33691e',  // dark olive        -- MASP2 deficiency; pyogenic+RA-like; CH50/AH50 normal
  'COLEC11': '#4a148c',  // deep purple       -- 3MC syndrome; craniosynostosis+cleft+hearing loss PATHOGNOMONIC
};

const GENE_INFO = {
  'ITGB2':   { full: 'ITGB2-769aa / AR / Integrin-beta-2 / CD18 / β2-integrin-common-subunit / 21q22.3 / LAD-I-MOST-COMMON-LAD / CD18-ABSENT-Flow-Cytometry-PATHOGNOMONIC / Delayed-Cord-Separation-Omphalitis / Leukocytosis-Without-Pus / Periodontitis / HSCT-ONLY-Curative / Gene-Therapy-Trials', locus: '21q22.3', size: '769 aa / 84 kDa', inh: 'Autosomal recessive (AR; 21q22.3; both sexes; LAD-I)' },
  'FERMT3':  { full: 'FERMT3-667aa / AR / Kindlin-3 / integrin-inside-out-activation / 11q13.1 / LAD-III / CD18-NORMAL-EXPRESSION-Key-DDx-LAD-I / β2-integrin-activation-absent / β3-Platelet-Glanzmann-Like-Bleeding / Combined-Infections-Bleeding-PATHOGNOMONIC / HSCT-MANDATORY', locus: '11q13.1', size: '667 aa / 76 kDa', inh: 'Autosomal recessive (AR; 11q13.1; both sexes; LAD-III; rare)' },
  'ADA2':    { full: 'ADA2-511aa / AR / Adenosine-deaminase-2 / CECR1 / 22q11.1 / DADA2 / PAN-Like-Vasculopathy+Stroke+Livedo-PATHOGNOMONIC / NK-Cell-Lymphopenia / TNF-INHIBITOR-CURATIVE-Start-Immediately / HSCT-Only-Severe-Cytopenias / Georgian-Turkish-Founder', locus: '22q11.1', size: '511 aa / 59 kDa', inh: 'Autosomal recessive (AR; 22q11.1; both sexes; Georgian/Turkish founder variants)' },
  'CEBPE':   { full: 'CEBPE-317aa / AR / C-EBP-epsilon / CCAAT-Enhancer-Binding-Protein / 14q11.2 / Specific-Granule-Deficiency-SGD / Bilobed-PMN-Nuclei-PATHOGNOMONIC / Absent-Lactoferrin-Granule-Assay / NBT-DHR-NORMAL-Not-CGD-KEY-DDx / TMP-SMX+Antifungal / HSCT-Severe', locus: '14q11.2', size: '317 aa / 32 kDa', inh: 'Autosomal recessive (AR; 14q11.2; both sexes; very rare <30 families)' },
  'CXCR2':   { full: 'CXCR2-360aa / AR / IL-8-Receptor-B / CD182 / 2q35 / SCN6 / Neutropenia+Normal-Marrow-Reserve-Exit-Failure / CD182-Absent-Flow / Inflammatory-Skin-Disease / Recurrent-UTI / G-CSF-EFFECTIVE-Bypasses-CXCR2 / Ruxolitinib-Skin / HSCT-Refractory', locus: '2q35', size: '360 aa / 40 kDa', inh: 'Autosomal recessive (AR; 2q35; both sexes; Middle Eastern/European)' },
  'MBL2':    { full: 'MBL2-244aa / AR-structural-variants / Mannose-Binding-Lectin / 10q11.22 / MBL-Deficiency / CH50-NORMAL-AH50-NORMAL-Lectin-Assay-Absent / Partial-Immunodeficiency / Codon-52-54-57-Structural-Variants / rMBL-Replacement-Trials / Prophylactic-Antibiotics', locus: '10q11.22', size: '244 aa / 26 kDa', inh: 'Autosomal recessive (AR; 10q11.22; functional deficiency via structural + promoter variants)' },
  'MASP2':   { full: 'MASP2-686aa / AR / MBL-Associated-Serine-Protease-2 / 1p36.22 / MASP2-Deficiency / C4-C2-Cleavage-Absent / CH50-NORMAL-AH50-NORMAL / Pyogenic-Infections+RA-Like / Narsoplimab-CONTRAINDICATED-LOF / FFP-MASP2-Source / Very-Rare-<30-Families', locus: '1p36.22', size: '686 aa / 76 kDa', inh: 'Autosomal recessive (AR; 1p36.22; both sexes; very rare <30 families globally)' },
  'COLEC11': { full: 'COLEC11-344aa / AR / Collectin-11 / CL-K1 / 2p25.3 / 3MC-Syndrome-Type-2 / Craniosynostosis+Posterior-Embryotoxon+Cleft-Lip-Palate+Hearing-Loss-PATHOGNOMONIC / Neural-Crest-Guidance / COLEC11-vs-MASP1-3MC-Subtypes / Structural-Anomaly-Repair', locus: '2p25.3', size: '344 aa / 38 kDa', inh: 'Autosomal recessive (AR; 2p25.3; both sexes; consanguineous families)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'ITGB2',   group: 'LAD-I (CD18 absent)',       cd18: '✗✗✗', cord_sep: '✓✓✓', bleeding: '✗',   neutro: '✗',   hsct_urg: '✓✓✓', unique: 'CD18 ABSENT on flow cytometry = PATHOGNOMONIC LAD-I; DELAYED CORD SEPARATION (>3-4wk) + omphalitis; LEUKOCYTOSIS 50-100k WITHOUT PUS; periodontitis; G-CSF NOT effective; HSCT = ONLY curative; gene therapy trials.' },
  { gene: 'FERMT3',  group: 'LAD-III (Kindlin-3 LOF)',   cd18: '✓',   cord_sep: '✓✓✓', bleeding: '✓✓✓', neutro: '✗',   hsct_urg: '✓✓✓', unique: 'CD18 EXPRESSION NORMAL (key DDx from LAD-I) — functional activation absent; COMBINED INFECTIONS + GLANZMANN-LIKE BLEEDING PATHOGNOMONIC (β2-integrin adhesion + β3-integrin platelet activation both fail); β2 activation assay absent; HSCT mandatory.' },
  { gene: 'ADA2',    group: 'DADA2 (Vasculopathy)',      cd18: '✓',   cord_sep: '✗',   bleeding: '±',   neutro: '✓',   hsct_urg: '✗',   unique: 'PAN-LIKE VASCULOPATHY + EARLY STROKE (<10yr) + LIVEDO RETICULARIS = PATHOGNOMONIC TRIAD; NK CELL LYMPHOPENIA distinguishes from classic PAN; TNF-INHIBITORS (etanercept/adalimumab) CURATIVE — START IMMEDIATELY; HSCT only for cytopenias.' },
  { gene: 'CEBPE',   group: 'SGD (Specific Granule Def)', cd18: '✓',  cord_sep: '✗',   bleeding: '✗',   neutro: '✗',   hsct_urg: '±',   unique: 'BILOBED PMN NUCLEI on blood film PATHOGNOMONIC; ABSENT LACTOFERRIN by granule assay; NBT/DHR NORMAL (not CGD — primary azurophil granules intact); same pathogens as CGD; TMP-SMX + antifungal prophylaxis same as CGD; HSCT for severe.' },
  { gene: 'CXCR2',   group: 'SCN6 (Chemotaxis failure)', cd18: '✓',   cord_sep: '✗',   bleeding: '✗',   neutro: '✓✓✓', hsct_urg: '✗',   unique: 'PERIPHERAL NEUTROPENIA + NORMAL MARROW RESERVE (exit failure, not production failure); CD182/CXCR2 ABSENT on PMN flow; inflammatory skin disease; recurrent UTIs; G-CSF EFFECTIVE (bypasses CXCR2 via G-CSFR); ruxolitinib for skin.' },
  { gene: 'MBL2',    group: 'MBL Deficiency (Lectin PR)', cd18: '✓',  cord_sep: '✗',   bleeding: '✗',   neutro: '✗',   hsct_urg: '✗',   unique: 'CH50/AH50 NORMAL; MBL-lectin pathway assay (mannan plate) ABSENT; serum MBL < 100 ng/mL; PARTIAL IMMUNODEFICIENCY (significant in infancy/immunosuppression); codon 52/54/57 structural variants; recombinant MBL replacement trials.' },
  { gene: 'MASP2',   group: 'MASP2 Def (Lectin effector)', cd18: '✓', cord_sep: '✗',   bleeding: '✗',   neutro: '✗',   hsct_urg: '✗',   unique: 'MASP2 SERUM UNDETECTABLE; CH50/AH50 NORMAL; C4 deposition on mannan plate = ABSENT; pyogenic infections + RA-LIKE inflammation; NARSOPLIMAB CONTRAINDICATED (inhibitor of absent protein); FFP contains MASP2 for acute infections.' },
  { gene: 'COLEC11', group: '3MC Syndrome (Lectin+Devel)', cd18: '✓', cord_sep: '✗',   bleeding: '✗',   neutro: '✗',   hsct_urg: '✗',   unique: '3MC DEVELOPMENTAL TRIAD PATHOGNOMONIC: craniosynostosis + POSTERIOR EMBRYOTOXON (Axenfeld corneal ring) + cleft lip/palate + hearing loss + caudal dysgenesis; lectin pathway REDUCED; COLEC11 vs MASP1 sequencing (3MC type 2 vs 1); structural anomaly repair.' },
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
  const isITGB2   = gene === 'ITGB2';
  const isFERMT3  = gene === 'FERMT3';
  const isADA2    = gene === 'ADA2';
  const isCEBPE   = gene === 'CEBPE';
  const isCXCR2   = gene === 'CXCR2';
  const isMBL2    = gene === 'MBL2';
  const isMASP2   = gene === 'MASP2';
  const isCOLEC11 = gene === 'COLEC11';
  const groupLabel = isITGB2   ? 'LAD-I (CD18-Absent-PATHOGNOMONIC-Delayed-Cord-HSCT-ONLY)'
    : isFERMT3  ? 'LAD-III (Kindlin-3-CD18-Normal-Combined-Infections-Bleeding-PATHOGNOMONIC-HSCT-Mandatory)'
    : isADA2    ? 'DADA2 (Vasculopathy-Stroke-Livedo-PATHOGNOMONIC-Anti-TNF-Curative)'
    : isCEBPE   ? 'SGD (Bilobed-PMN-Nuclei-PATHOGNOMONIC-Absent-Lactoferrin-NBT-Normal)'
    : isCXCR2   ? 'SCN6 (CXCR2-Absent-Exit-Failure-Normal-Marrow-G-CSF-Effective)'
    : isMBL2    ? 'MBL-Deficiency (Lectin-PR-Partial-Immunodeficiency-CH50-AH50-Normal)'
    : isMASP2   ? 'MASP2-Deficiency (Lectin-Effector-Absent-Narsoplimab-Contraindicated)'
    : 'COLEC11-3MC-Syndrome (Craniosynostosis-Posterior-Embryotoxon-Cleft-PATHOGNOMONIC)';

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
          <span>HSCT: {(cohort.hsct_pct*100).toFixed(0)}%</span>
          <span>Infect: {(cohort.recurrent_infect_pct*100).toFixed(0)}%</span>
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
    const ep = tab === 'Definitions' ? 'definitions'
             : tab === 'Diagnostic Map' ? 'breakdown'
             : 'overview';
    const setter = tab === 'Definitions' ? setDefs
                 : tab === 'Diagnostic Map' ? setBk
                 : setOv;
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => { setter(d); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, [tab]);

  const tabStyle = (t) => ({
    padding: '8px 16px', cursor: 'pointer', borderRadius: '6px 6px 0 0',
    background: tab === t ? '#e65100' : '#1a1a2e',
    color: '#fff', border: 'none', marginRight: 4, fontWeight: tab === t ? 700 : 400,
  });

  return (
    <div style={{ background: '#0d0d1a', minHeight: '100vh', color: '#f0f0f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ background: 'linear-gradient(135deg,#880e4f,#e65100)', borderRadius: 10, padding: 24, marginBottom: 24 }}>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>
            &#x1f9ec; Hereditary-LAD-DADA2-Lectin-Complement-Granulocyte-Defect-Atlas
          </h1>
          <p style={{ margin: '8px 0 0', fontSize: 13, opacity: 0.9 }}>
            Complete 8-Gene Reference: ITGB2-FERMT3-ADA2-CEBPE-CXCR2-MBL2-MASP2-COLEC11
            · 320 Patients (8×40) · Seeds 4006-4013
          </p>
          <p style={{ margin: '4px 0 0', fontSize: 11, opacity: 0.8 }}>
            LAD-I (CD18-Absent-PATHOGNOMONIC) · LAD-III (Kindlin-3-Combined-Infections-Bleeding-PATHOGNOMONIC)
            · DADA2 (Vasculopathy-Stroke-Livedo-Anti-TNF-CURATIVE) · SGD (Bilobed-Nuclei-NBT-NORMAL-PATHOGNOMONIC)
            · CXCR2-SCN6 (Exit-Failure-G-CSF-Effective) · MBL2-MASP2-COLEC11 (Lectin-Pathway-CH50-AH50-Normal)
          </p>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', marginBottom: 0 }}>
          {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
        </div>
        <div style={{ background: '#12122a', borderRadius: '0 8px 8px 8px', padding: 24, minHeight: 400 }}>
          {loading && <div style={{ color: '#aaa' }}>Loading…</div>}
          {error && <div style={{ color: '#f44336' }}>Error: {error}</div>}

          {/* OVERVIEW TAB */}
          {tab === 'Overview' && overview && (
            <div>
              <h2 style={{ color: '#e65100', marginTop: 0 }}>Atlas Overview — {overview.atlas}</h2>
              <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Total Patients</div>
                  <div style={{ color: '#fff', fontSize: 28, fontWeight: 700 }}>{overview.total_n}</div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>HSCT Rate</div>
                  <div style={{ color: '#ff7043', fontSize: 28, fontWeight: 700 }}>
                    {(overview.aggregate_stats?.hsct_pct * 100).toFixed(1)}%
                  </div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Recurrent Infections</div>
                  <div style={{ color: '#ffb300', fontSize: 28, fontWeight: 700 }}>
                    {(overview.aggregate_stats?.recurrent_infections_pct * 100).toFixed(1)}%
                  </div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Seeds</div>
                  <div style={{ color: '#80cbc4', fontSize: 20, fontWeight: 700 }}>{overview.seeds}</div>
                </div>
              </div>

              {/* Group badges */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ marginBottom: 8 }}>
                  {['ITGB2','FERMT3'].map(g => <Badge key={g} text={`LAD: ${g}`} color={GENE_COLORS[g]} />)}
                  {['ADA2'].map(g => <Badge key={g} text={`Vasculopathy: ${g}`} color={GENE_COLORS[g]} />)}
                  {['CEBPE','CXCR2'].map(g => <Badge key={g} text={`Granule/Chemotaxis: ${g}`} color={GENE_COLORS[g]} />)}
                  {['MBL2','MASP2','COLEC11'].map(g => <Badge key={g} text={`Lectin Complement: ${g}`} color={GENE_COLORS[g]} />)}
                </div>
              </div>

              {/* Key facts */}
              <div style={{ marginBottom: 20 }}>
                <h3 style={{ color: '#e65100', fontSize: 14 }}>Key Clinical Facts</h3>
                <ul style={{ paddingLeft: 20, margin: 0 }}>
                  {(overview.key_facts || []).map((f, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#ccc', marginBottom: 6, lineHeight: 1.5 }}>{f}</li>
                  ))}
                </ul>
              </div>

              {/* Gene cards */}
              <h3 style={{ color: '#e65100', fontSize: 14, marginBottom: 12 }}>Gene Cohort Summaries (40 patients each)</h3>
              {(overview.cohort_summaries || []).map((c, i) => (
                <GeneCard key={c.gene} gene={c.gene} color={GENE_COLORS[c.gene] || '#555'}
                          info={GENE_INFO[c.gene] || {}} cohort={c} />
              ))}
            </div>
          )}

          {/* GENE TABLE TAB */}
          {tab === 'Gene Table' && overview && (
            <div>
              <h2 style={{ color: '#e65100', marginTop: 0 }}>Gene Comparison Table</h2>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: '#1a1a2e' }}>
                      {['Gene','Group','aa','Locus','kDa','Inheritance','CD18 Absent','Cord Sep','Bleeding','Neutropenia','HSCT Urg','Unique Finding'].map(h => (
                        <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#e65100', borderBottom: '1px solid #333' }}>{h}</th>
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
                        <td style={{ padding: '6px', textAlign: 'center', color: row.cd18 === '✗✗✗' ? '#f44' : '#4f4' }}>{row.cd18}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.cord_sep.includes('✓') ? '#ffb' : '#888' }}>{row.cord_sep}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.bleeding.includes('✓') ? '#f88' : '#888' }}>{row.bleeding}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.neutro.includes('✓') ? '#ffa' : '#888' }}>{row.neutro}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.hsct_urg.includes('✓✓✓') ? '#f55' : row.hsct_urg.includes('✓') ? '#fa8' : '#888' }}>{row.hsct_urg}</td>
                        <td style={{ padding: '6px', color: '#a8e6cf', fontSize: 10, maxWidth: 300 }}>{row.unique}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* DIAGNOSTIC MAP TAB */}
          {tab === 'Diagnostic Map' && breakdown && (
            <div>
              <h2 style={{ color: '#e65100', marginTop: 0 }}>Diagnostic Map & Cascade Testing</h2>

              {/* Pathognomonic features */}
              <h3 style={{ color: '#ffb300', fontSize: 13, marginBottom: 10 }}>Pathognomonic Features</h3>
              {Object.entries(breakdown.pathognomonic_features || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#1a1a2e', borderRadius: 6, padding: 12, marginBottom: 8 }}>
                  <div style={{ color: '#e65100', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>
                    {k.replace(/_/g, ' ').toUpperCase()}
                  </div>
                  <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.6 }}>{v}</div>
                </div>
              ))}

              {/* Diagnostic groups */}
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Diagnostic Groups</h3>
              {Object.entries(breakdown.diagnostic_groups || {}).map(([grp, data]) => (
                <div key={grp} style={{ background: '#12122a', border: '1px solid #333', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                    <span style={{ color: '#e65100', fontWeight: 700, fontSize: 13 }}>
                      {grp.replace(/_/g, ' ').toUpperCase()}
                    </span>
                    {(data.genes || []).map(g => (
                      <Badge key={g} text={g} color={GENE_COLORS[g] || '#555'} />
                    ))}
                  </div>
                  <div style={{ fontSize: 11, color: '#9ecfff', marginBottom: 6 }}>
                    <strong>Mechanism:</strong> {data.mechanism}
                  </div>
                  <div style={{ fontSize: 11, color: '#a8e6cf', marginBottom: 6 }}>
                    <strong>Shared features:</strong> {data.shared_feature}
                  </div>
                  <div style={{ fontSize: 11, color: '#ffcc80', lineHeight: 1.5 }}>
                    <strong>DDx:</strong> {data.key_ddx}
                  </div>
                </div>
              ))}

              {/* Cascade testing */}
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Cascade Testing Order</h3>
              <ol style={{ paddingLeft: 20, margin: 0 }}>
                {(breakdown.cascade_testing_order || []).map((step, i) => (
                  <li key={i} style={{ fontSize: 11, color: '#ccc', marginBottom: 8, lineHeight: 1.5 }}>{step}</li>
                ))}
              </ol>
            </div>
          )}

          {/* DEFINITIONS TAB */}
          {tab === 'Definitions' && defs && (
            <div>
              <h2 style={{ color: '#e65100', marginTop: 0 }}>Key Definitions & Clinical Protocols</h2>

              {/* Key distinguishers */}
              <h3 style={{ color: '#ffb300', fontSize: 13, marginBottom: 10 }}>Key Distinguishers</h3>
              {Object.entries(defs.key_distinguishers || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#1a1a2e', borderRadius: 6, padding: 10, marginBottom: 8 }}>
                  <div style={{ color: '#9ecfff', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>
                    {k.replace(/_/g, ' ').toUpperCase()}
                  </div>
                  <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.5 }}>{v}</div>
                </div>
              ))}

              {/* Treatment matrix */}
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Treatment Matrix</h3>
              {Object.entries(defs.treatment_matrix || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#0d1a0d', border: '1px solid #2e7d32', borderRadius: 6, padding: 10, marginBottom: 8 }}>
                  <div style={{ color: '#a8e6cf', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>
                    {k.replace(/_/g, ' ').toUpperCase()}
                  </div>
                  <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.5 }}>{v}</div>
                </div>
              ))}

              {/* Full definitions */}
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Full Definitions</h3>
              {(defs.definitions || []).map((d, i) => (
                <div key={i} style={{ background: '#12122a', borderRadius: 8, padding: 14, marginBottom: 10 }}>
                  <div style={{ color: '#e65100', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{d.term}</div>
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
