'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-jak-stat-gof-immune-dysregulation-atlas';
const TABS = ['Overview', 'Gene Table', 'Diagnostic Map', 'Definitions'];

const GENE_COLORS = {
  'STAT1':  '#b71c1c',  // deep red        -- STAT1-GOF; CMC5; esophageal candida; intracranial aneurysm; ruxolitinib
  'STAT3':  '#4a148c',  // deep purple     -- STAT3-GOF; neonatal T1DM; short stature; LPD; ruxolitinib
  'STAT5B': '#880e4f',  // dark crimson    -- STAT5B-GOF; NK-LGL leukemia; eosinophilia; ruxolitinib
  'STAT6':  '#e65100',  // deep orange     -- STAT6-GOF; ultra-severe atopy; IgE >10000; dupilumab; 2022 discovery
  'JAK1':   '#1565c0',  // deep blue       -- JAK1-GOF; multi-STAT; polyarthritis+enteropathy; baricitinib
  'SOCS1':  '#2e7d32',  // dark green      -- SOCS1-LOF haploinsufficiency; SLE+MPN; ruxolitinib; 2021 discovery
  'CARD11': '#f57f17',  // dark amber      -- CARD11-GOF BENTA; B cell expansion; T cell anergy; ibrutinib
  'IL4RA':  '#00838f',  // dark teal       -- IL4RA-GOF; severe atopy; IgE 1000-5000; dupilumab directly blocks receptor
};

const GENE_INFO = {
  'STAT1':  { full: 'STAT1-GOF-750aa / AD / coiled-coil-domain / 2q32.2 / CMC5-Chronic-Mucocutaneous-Candidiasis-type5 / Esophageal-Candida-PATHOGNOMONIC-non-immunocompromised / Th17-deficient / Autoimmunity-35pct / Intracranial-Aneurysm-15pct / Ruxolitinib', locus: '2q32.2', size: '750 aa / 87 kDa', inh: 'Autosomal dominant GOF (AD-GOF; CCD/DBD domain; de novo or familial; CMC5 most common CMC gene)' },
  'STAT3':  { full: 'STAT3-GOF-770aa / AD / SH2-DBD-TAD-domain / 17q21.2 / Multi-Organ-Autoimmune / Early-T1DM-PATHOGNOMONIC / Short-Stature-GH-Resistance / Lymphoproliferation-LGL / Ruxolitinib-Sirolimus', locus: '17q21.2', size: '770 aa / 92 kDa', inh: 'Autosomal dominant GOF (AD-GOF; 17q21.2; de novo majority; multi-organ immune dysregulation)' },
  'STAT5B': { full: 'STAT5B-GOF-787aa / AD / SH2-domain-p.N642H-hotspot / 17q21.2 / NK-LGL-Leukemia-PATHOGNOMONIC / Eosinophilia / Hepatosplenomegaly / DISTINGUISH-from-STAT5B-LOF-GH-resistance / Ruxolitinib', locus: '17q21.2', size: '787 aa / 92 kDa', inh: 'Autosomal dominant GOF (AD-GOF; p.N642H SH2 domain hotspot; NK-LGL leukemia driver)' },
  'STAT6':  { full: 'STAT6-GOF-847aa / AD / DBD-SH2-TAD-domain / 12q13.3 / Ultra-Severe-Atopy / IgE->10000-IU-mL-PATHOGNOMONIC / Neonatal-Eczema / Anaphylaxis-Food-Allergies / Eosinophilia / Dupilumab / 2022-NEJM-discovery', locus: '12q13.3', size: '847 aa / 94 kDa', inh: 'Autosomal dominant GOF (AD-GOF; 12q13.3; predominantly de novo; infantile onset; 2022-2023 discovery)' },
  'JAK1':   { full: 'JAK1-GOF-1142aa / AD / JH2-pseudokinase-domain / 1p31.3 / Multi-STAT-hyperactivation / Polyarthritis-Enteropathy-Eosinophilia-triad / Herpes-viral-susceptibility-paradoxical / Baricitinib-Upadacitinib', locus: '1p31.3', size: '1142 aa / 133 kDa', inh: 'Autosomal dominant GOF (AD-GOF; JH2 pseudokinase domain; constitutive multi-STAT activation)' },
  'SOCS1':  { full: 'SOCS1-LOF-212aa / AD-haploinsufficiency / KIR-SH2-SOCS-box / 16p13.13 / IFN-γ-negative-feedback-loss / SLE-like-ANA-dsDNA / Myeloproliferation-thrombocytosis-PATHOGNOMONIC / CMC-like-STAT1-GOF / Ruxolitinib / 2021-discovery', locus: '16p13.13', size: '212 aa / 24 kDa', inh: 'Autosomal dominant LOF haploinsufficiency (AD-LOF; 16p13.13; monoallelic → 50% SOCS1 → JAK-STAT upregulation; 2021)' },
  'CARD11': { full: 'CARD11-GOF-BENTA-1154aa / AD / coiled-coil-linker-domain / 7p22.2 / B-Cell-Expansion-NF-kB-T-Cell-Anergy / Polyclonal-B-cells->40pct-PATHOGNOMONIC / Splenomegaly / Atopy-IgE / Ibrutinib / DISTINCT-from-CARD11-LOF-CID', locus: '7p22.2', size: '1154 aa / 130 kDa', inh: 'Autosomal dominant GOF (AD-GOF; coiled-coil/linker domain; constitutive NF-κB; BENTA disease; opposite of LOF)' },
  'IL4RA':  { full: 'IL4RA-GOF-825aa / AD / receptor-level-IL4-IL13-STAT6-axis / 16p12.1 / Severe-Atopy / Multiple-Food-Allergies / Asthma / CRSwNP-quadrad / IgE-1000-5000 / Dupilumab-DIRECTLY-BLOCKS-GOF-RECEPTOR-mechanism-specific', locus: '16p12.1', size: '825 aa / 140 kDa', inh: 'Autosomal dominant GOF (AD-GOF; 16p12.1; receptor-level IL-4/IL-13 signaling enhancement; severe atopy)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'STAT1',  group: 'CMC5 (STAT1 GOF)',         pstat: '↑STAT1', cmc: '✓✓✓', autoim: '✓✓', atopy: '✗',   lgl: '✗',   rx: 'Ruxolitinib', unique: 'ESOPHAGEAL CANDIDIASIS non-immunocomp = STAT1-GOF; nail dystrophy + oral thrush; prolonged pSTAT1; intracranial aneurysm 15%; ruxolitinib reverses CMC.' },
  { gene: 'STAT3',  group: 'Multi-organ (STAT3 GOF)',  pstat: '↑STAT3', cmc: '±',   autoim: '✓✓✓', atopy: '±',  lgl: '✓✓', rx: 'Ruxolitinib', unique: 'NEONATAL T1DM + SHORT STATURE = STAT3-GOF; GH resistance (normal GH, low IGF-1); LPD/lymphoma risk; reduced Tregs; ruxolitinib slows T1DM progression.' },
  { gene: 'STAT5B', group: 'NK-LGL leukemia (STAT5B)', pstat: '↑STAT5B', cmc: '✗',  autoim: '✓',  atopy: '✗',   lgl: '✓✓✓', rx: 'Ruxolitinib', unique: 'NK-LGL EXPANSION PATHOGNOMONIC; p.N642H SH2 hotspot (germline + somatic); eosinophilia + hepatosplenomegaly; STAT5B-LOF (opposite: GH resistance) DDx.' },
  { gene: 'STAT6',  group: 'Ultra-atopy (STAT6 GOF)',  pstat: '↑STAT6', cmc: '✗',   autoim: '✗',  atopy: '✓✓✓', lgl: '✗',   rx: 'Dupilumab',  unique: 'IgE >10,000 IU/mL PATHOGNOMONIC; constitutive pSTAT6 without IL-4/IL-13; neonatal eczema + anaphylaxis + eosinophilia; 2022-2023 NEJM discovery.' },
  { gene: 'JAK1',   group: 'Multi-STAT (JAK1 GOF)',    pstat: '↑↑↑',   cmc: '±',   autoim: '✓✓✓', atopy: '±',  lgl: '±',   rx: 'Baricitinib', unique: 'STAT1+STAT3+STAT5 simultaneous elevation; polyarthritis + enteropathy + herpes susceptibility; JH2 pseudokinase domain; upadacitinib preferred (selective JAK1).' },
  { gene: 'SOCS1',  group: 'SLE+MPN (SOCS1 LOF)',      pstat: '↑STAT1', cmc: '✓✓',  autoim: '✓✓✓', atopy: '✗',  lgl: '✗',   rx: 'Ruxolitinib', unique: 'SLE-like + MYELOPROLIFERATION (thrombocytosis) COMBINATION PATHOGNOMONIC; ISG score elevated; SOCS1 haploinsufficiency → ruxolitinib replaces SOCS1 feedback; 2021 discovery.' },
  { gene: 'CARD11', group: 'BENTA (CARD11 GOF)',        pstat: '✗',     cmc: '±',   autoim: '±',  atopy: '✓✓',  lgl: '✗',   rx: 'Ibrutinib',  unique: 'POLYCLONAL B CELL EXPANSION >40% PATHOGNOMONIC; T cell anergy paradox; constitutive NF-κB (not JAK-STAT); ibrutinib reduces B cells; CARD11-LOF (CID) = opposite phenotype DDx.' },
  { gene: 'IL4RA',  group: 'Receptor atopy (IL4RA)',   pstat: '↑STAT6', cmc: '✗',   autoim: '±',  atopy: '✓✓✓', lgl: '✗',   rx: 'Dupilumab',  unique: 'SEVERE AD + FOOD ALLERGIES + ASTHMA + CRSwNP quadrad; IgE 1000-5000 IU/mL; dupilumab DIRECTLY blocks GOF receptor (mechanism-specific); FeNO elevated; Treg reduced.' },
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
    STAT1:  'STAT1-GOF — CMC5 (coiled-coil domain; impaired nuclear dephosphorylation; Th17 suppression; esophageal candida; autoimmunity; ruxolitinib)',
    STAT3:  'STAT3-GOF — Multi-organ autoimmune (SH2/DBD/TAD; impaired SOCS3 feedback; early T1DM; short stature; LPD; reduced Tregs; ruxolitinib)',
    STAT5B: 'STAT5B-GOF — NK-LGL leukemia (p.N642H SH2 hotspot; constitutive pSTAT5B; NK-LGL clonal expansion; eosinophilia; ruxolitinib)',
    STAT6:  'STAT6-GOF — Ultra-severe atopy (DBD/SH2/TAD; constitutive STAT6 without IL-4/IL-13; IgE >10000; infantile eczema; anaphylaxis; dupilumab)',
    JAK1:   'JAK1-GOF — Multi-STAT hyperactivation (JH2 pseudokinase; constitutive JAK1→STAT1+STAT3+STAT5; polyarthritis; enteropathy; baricitinib/upadacitinib)',
    SOCS1:  'SOCS1-LOF haploinsufficiency — SLE+myeloproliferation (KIR/SH2/SOCS-box; reduced IFN-γ feedback; JAK1/2 upregulation; thrombocytosis; ruxolitinib)',
    CARD11: 'CARD11-GOF BENTA — B-cell expansion+NF-κB+T-cell anergy (coiled-coil/linker; constitutive CBM→NF-κB; polyclonal B cells; ibrutinib; ≠LOF/CID)',
    IL4RA:  'IL4RA-GOF — Receptor-level IL-4/IL-13/STAT6 atopy (16p12.1; enhanced IL-4Rα signaling; severe AD+food allergies+asthma+CRSwNP; dupilumab)',
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
        <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#aaa', marginBottom: 6, flexWrap: 'wrap' }}>
          <span>n={cohort.n}</span>
          <span>M:{cohort.male_n} F:{cohort.female_n}</span>
          <span>Onset median: {cohort.median_onset_months}mo</span>
          <span>Ruxolitinib: {(cohort.ruxolitinib_pct*100).toFixed(0)}%</span>
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
    background: tab === t ? '#b71c1c' : '#1a1a2e',
    color: '#fff', border: 'none', marginRight: 4, fontWeight: tab === t ? 700 : 400,
  });

  return (
    <div style={{ background: '#0d0d1a', minHeight: '100vh', color: '#f0f0f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ background: 'linear-gradient(135deg,#b71c1c,#4a148c)', borderRadius: 10, padding: 24, marginBottom: 24 }}>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>
            &#x1f9ec; Hereditary-JAK-STAT-GOF-Immune-Dysregulation-Atlas
          </h1>
          <p style={{ margin: '8px 0 0', fontSize: 13, opacity: 0.9 }}>
            Complete 8-Gene Reference: STAT1-STAT3-STAT5B-STAT6-JAK1-SOCS1-CARD11-IL4RA
            · 320 Patients (8×40) · Seeds 4022-4029
          </p>
          <p style={{ margin: '4px 0 0', fontSize: 11, opacity: 0.8 }}>
            STAT1-GOF (CMC5-Esophageal-Candida-Th17-Deficient-Ruxolitinib) · STAT3-GOF (Neonatal-T1DM-Short-Stature-LPD)
            · STAT5B-GOF (NK-LGL-Leukemia-p.N642H-Hotspot) · STAT6-GOF (Ultra-Atopy-IgE->10000-Dupilumab-2022)
            · JAK1-GOF (Multi-STAT-Polyarthritis-Enteropathy-Baricitinib) · SOCS1-LOF (SLE-MPN-2021)
            · CARD11-GOF-BENTA (B-Expansion-NF-kB-T-Anergy-Ibrutinib) · IL4RA-GOF (Receptor-Atopy-Dupilumab)
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
              <h2 style={{ color: '#b71c1c', marginTop: 0 }}>Atlas Overview — {overview.atlas}</h2>
              <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Total Patients</div>
                  <div style={{ color: '#fff', fontSize: 28, fontWeight: 700 }}>{overview.total_n}</div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Ruxolitinib Use</div>
                  <div style={{ color: '#ff5252', fontSize: 28, fontWeight: 700 }}>
                    {overview.aggregate_stats?.ruxolitinib_pct != null
                      ? (overview.aggregate_stats.ruxolitinib_pct * 100).toFixed(1) + '%'
                      : 'N/A'}
                  </div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>HSCT Rate</div>
                  <div style={{ color: '#ffb300', fontSize: 28, fontWeight: 700 }}>
                    {overview.aggregate_stats?.hsct_pct != null
                      ? (overview.aggregate_stats.hsct_pct * 100).toFixed(1) + '%'
                      : 'N/A'}
                  </div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Seeds</div>
                  <div style={{ color: '#80cbc4', fontSize: 20, fontWeight: 700 }}>{overview.seeds}</div>
                </div>
              </div>

              {/* Therapy group badges */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ marginBottom: 8 }}>
                  {(overview.ruxolitinib_responsive || []).map(g =>
                    <Badge key={g} text={`Ruxolitinib: ${g}`} color={GENE_COLORS[g] || '#555'} />
                  )}
                  {(overview.dupilumab_responsive || []).map(g =>
                    <Badge key={g} text={`Dupilumab: ${g}`} color={GENE_COLORS[g] || '#555'} />
                  )}
                  {(overview.cbm_nfkb_group || []).map(g =>
                    <Badge key={g} text={`Ibrutinib-BENTA: ${g}`} color={GENE_COLORS[g] || '#555'} />
                  )}
                </div>
              </div>

              {/* JAK-STAT subgroups */}
              <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                <div style={{ color: '#b71c1c', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>
                  Targeted Therapy Groups (Key Clinical Distinction)
                </div>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                  <div style={{ padding: '6px 10px', background: '#2d0000', borderRadius: 6, fontSize: 11, color: '#ff8a80' }}>
                    <strong>JAK1/2 Inhibitor (Ruxolitinib):</strong> {(overview.ruxolitinib_responsive || []).join(', ')}
                  </div>
                  <div style={{ padding: '6px 10px', background: '#002d3d', borderRadius: 6, fontSize: 11, color: '#80d8ff' }}>
                    <strong>Anti-IL-4Rα (Dupilumab):</strong> {(overview.dupilumab_responsive || []).join(', ')}
                  </div>
                  <div style={{ padding: '6px 10px', background: '#2d1a00', borderRadius: 6, fontSize: 11, color: '#ffd180' }}>
                    <strong>BTK Inhibitor (Ibrutinib) BENTA:</strong> {(overview.cbm_nfkb_group || []).join(', ')}
                  </div>
                </div>
              </div>

              {/* CMC and atopy groups */}
              {overview.cmc_group && (
                <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                  <div style={{ color: '#b71c1c', fontWeight: 700, fontSize: 12, marginBottom: 6 }}>Clinical Groupings</div>
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                    <div style={{ padding: '4px 8px', background: '#1a000d', borderRadius: 4, fontSize: 11, color: '#f48fb1' }}>
                      <strong>CMC (Candida):</strong> {overview.cmc_group.join(', ')}
                    </div>
                    <div style={{ padding: '4px 8px', background: '#1a1200', borderRadius: 4, fontSize: 11, color: '#fff176' }}>
                      <strong>Atopy:</strong> {(overview.atopy_group || []).join(', ')}
                    </div>
                    <div style={{ padding: '4px 8px', background: '#001a00', borderRadius: 4, fontSize: 11, color: '#a5d6a7' }}>
                      <strong>NK-LGL:</strong> {(overview.nk_lgl_group || []).join(', ')}
                    </div>
                    <div style={{ padding: '4px 8px', background: '#000d1a', borderRadius: 4, fontSize: 11, color: '#90caf9' }}>
                      <strong>Neonatal onset:</strong> {(overview.neonatal_onset || []).join(', ')}
                    </div>
                  </div>
                </div>
              )}

              {/* Key facts */}
              <div style={{ marginBottom: 20 }}>
                <h3 style={{ color: '#b71c1c', fontSize: 14 }}>Key Clinical Facts</h3>
                <ul style={{ paddingLeft: 20, margin: 0 }}>
                  {(overview.key_facts || []).map((f, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#ccc', marginBottom: 6, lineHeight: 1.5 }}>{f}</li>
                  ))}
                </ul>
              </div>

              {/* Gene cards */}
              <h3 style={{ color: '#b71c1c', fontSize: 14, marginBottom: 12 }}>Gene Cohort Summaries (40 patients each)</h3>
              {(overview.cohort_summaries || []).map((c) => (
                <GeneCard key={c.gene} gene={c.gene} color={GENE_COLORS[c.gene] || '#555'}
                          info={GENE_INFO[c.gene] || {}} cohort={c} />
              ))}
            </div>
          )}

          {/* GENE TABLE TAB */}
          {tab === 'Gene Table' && overview && (
            <div>
              <h2 style={{ color: '#b71c1c', marginTop: 0 }}>Gene Comparison Table</h2>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: '#1a1a2e' }}>
                      {['Gene','Group','aa','Locus','kDa','Inheritance','pSTAT','CMC','Autoimmunity','Atopy','LGL/LPD','Rx','Unique Finding'].map(h => (
                        <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#b71c1c', borderBottom: '1px solid #333' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {PHENOTYPE_ROWS.map((row, i) => (
                      <tr key={row.gene} style={{ background: i % 2 === 0 ? '#0d0d1a' : '#12122a' }}>
                        <td style={{ padding: '6px', color: GENE_COLORS[row.gene], fontWeight: 700 }}>{row.gene}</td>
                        <td style={{ padding: '6px', color: '#ccc', fontSize: 10 }}>{row.group}</td>
                        <td style={{ padding: '6px', color: '#aaa' }}>{GENE_INFO[row.gene]?.size?.split('/')[0]}</td>
                        <td style={{ padding: '6px', color: '#9ecfff' }}>{GENE_INFO[row.gene]?.locus}</td>
                        <td style={{ padding: '6px', color: '#aaa' }}>{GENE_INFO[row.gene]?.size?.split('/')[1]}</td>
                        <td style={{ padding: '6px', color: '#ccc', fontSize: 10 }}>{GENE_INFO[row.gene]?.inh?.split('(')[0]}</td>
                        <td style={{ padding: '6px', color: '#f0f', fontWeight: 700 }}>{row.pstat}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.cmc.includes('✓✓✓') ? '#f55' : row.cmc.includes('✓✓') ? '#f88' : row.cmc === '±' ? '#ffd' : '#666' }}>{row.cmc}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.autoim.includes('✓✓✓') ? '#f55' : row.autoim.includes('✓✓') ? '#fa8' : row.autoim.includes('✓') ? '#ffd' : '#666' }}>{row.autoim}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.atopy.includes('✓✓✓') ? '#ff9800' : row.atopy.includes('✓✓') ? '#ffc107' : row.atopy === '±' ? '#ffd' : '#666' }}>{row.atopy}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.lgl.includes('✓✓✓') ? '#f55' : row.lgl.includes('✓✓') ? '#fa8' : row.lgl === '±' ? '#ffd' : '#666' }}>{row.lgl}</td>
                        <td style={{ padding: '6px', color: '#a8e6cf', fontWeight: 700, fontSize: 10 }}>{row.rx}</td>
                        <td style={{ padding: '6px', color: '#ffcc80', fontSize: 10, maxWidth: 300 }}>{row.unique}</td>
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
              <h2 style={{ color: '#b71c1c', marginTop: 0 }}>Diagnostic Map & Cascade Testing</h2>

              {/* Pathognomonic features */}
              <h3 style={{ color: '#ffb300', fontSize: 13, marginBottom: 10 }}>Pathognomonic Features & Key Diagnostics</h3>
              {Object.entries(breakdown.pathognomonic_features || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#1a1a2e', borderRadius: 6, padding: 12, marginBottom: 8 }}>
                  <div style={{ color: '#b71c1c', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>
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
                    <span style={{ color: '#b71c1c', fontWeight: 700, fontSize: 13 }}>
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
              <h2 style={{ color: '#b71c1c', marginTop: 0 }}>Key Definitions & Clinical Protocols</h2>

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
                  <div style={{ color: '#b71c1c', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{d.term}</div>
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
