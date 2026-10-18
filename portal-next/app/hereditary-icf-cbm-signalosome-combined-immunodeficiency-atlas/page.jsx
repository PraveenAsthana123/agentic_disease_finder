'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-icf-cbm-signalosome-combined-immunodeficiency-atlas';
const TABS = ['Overview', 'Gene Table', 'Diagnostic Map', 'Definitions'];

const GENE_COLORS = {
  'DNMT3B': '#1a237e',  // dark navy        -- ICF1; centromeric instability chr1/9/16 PATHOGNOMONIC; facial anomalies; agammaglobulinemia
  'ZBTB24': '#311b92',  // deep indigo       -- ICF2; milder neurological; same chr instability; BTB/POZ zinc-finger
  'CDCA7':  '#4a148c',  // deep purple       -- ICF3; very rare; CDCA7→HELLS pathway; biallelic required
  'HELLS':  '#880e4f',  // dark crimson      -- ICF4; lymphoid helicase; SMARCA6; ZBTB24→CDCA7→HELLS chain
  'CARD11': '#b71c1c',  // deep red          -- CARD11 LOF/AR CID; HSCT URGENT; CRITICAL DDx GOF=BENTA opposite disease
  'BCL10':  '#e65100',  // deep orange       -- BCL10 LOF/AR CID; agammaglobulinemia; very rare; HSCT
  'MALT1':  '#33691e',  // dark olive-green  -- MALT1 LOF/AR Treg dysregulation; colitis PATHOGNOMONIC; paracaspase assay
  'CTPS1':  '#006064',  // dark teal         -- CTPS1 AR; T-cell specific dCTP; EBV/VZV proliferation failure PATHOGNOMONIC
};

const GENE_INFO = {
  'DNMT3B': { full: 'DNMT3B-853aa / AR / DNA methyltransferase 3β / de novo CpG methylation / 20q11.21 / ICF-Type-1-MOST-COMMON-~50pct / Centromeric-Chr1-9-16-INSTABILITY-PATHOGNOMONIC / Facial-Anomalies / Profound-Agammaglobulinemia / IVIG-Lifelong / HSCT-Curative-Rarely-Used', locus: '20q11.21', size: '853 aa / 97 kDa', inh: 'Autosomal recessive (AR; 20q11.21; both sexes; ICF1)' },
  'ZBTB24': { full: 'ZBTB24-570aa / AR / BTB-POZ zinc-finger transcription factor / regulates CDCA7 expression / 6q25.3 / ICF-Type-2-~30pct / Same-Centromeric-Instability-Chr1-9-16 / Milder-Neuro-Than-ICF1 / Hypogammaglobulinemia / IVIG-Lifelong', locus: '6q25.3',  size: '570 aa / 65 kDa', inh: 'Autosomal recessive (AR; 6q25.3; both sexes; ICF2)' },
  'CDCA7':  { full: 'CDCA7-372aa / AR / Cell Division Cycle Associated 7 / links ZBTB24→HELLS pathway / 2q31.1 / ICF-Type-3-Very-Rare-<20-Families / Same-Centromeric-Instability / ICF-Multigene-Panel-Required / Variable-Neuro / IVIG / HSCT-Considered-Severe', locus: '2q31.1',  size: '372 aa / 43 kDa', inh: 'Autosomal recessive (AR; 2q31.1; both sexes; ICF3)' },
  'HELLS':  { full: 'HELLS-838aa / AR / Helicase Lymphoid Specific / SMARCA6 / SWI/SNF family / 10q23.33 / ICF-Type-4-Rarest-~10-Families / Same-Centromeric-Instability / Lymphoid-Specific-Expression / Methylation-Maintenance-Terminal-Effector / IVIG / HSCT-Severe', locus: '10q23.33', size: '838 aa / 97 kDa', inh: 'Autosomal recessive (AR; 10q23.33; both sexes; ICF4)' },
  'CARD11': { full: 'CARD11-1154aa / AR-LOF / CBM signalosome scaffold / CARMA1 / 7p22.2 / CARD11-LOF-CID / Agammaglobulinemia / T-Cell-Anergy / NF-kB-TCR-BCR-Absent / HSCT-URGENT-MANDATORY / CRITICAL-DDx-GOF=BENTA-OPPOSITE-DISEASE / Venetoclax-NOT-for-LOF', locus: '7p22.2',  size: '1154 aa / 130 kDa', inh: 'Autosomal recessive LOF (AR; 7p22.2; biallelic; both sexes — NOT same as AD GOF=BENTA)' },
  'BCL10':  { full: 'BCL10-233aa / AR-LOF / CBM adaptor / caspase-recruitment domain / 1p22.3 / BCL10-LOF-CID / Agammaglobulinemia / T-Cell-B-Cell-Activation-Absent / Very-Rare-<10-Families / IVIG / HSCT-Curative', locus: '1p22.3',  size: '233 aa / 26 kDa', inh: 'Autosomal recessive LOF (AR; 1p22.3; biallelic; both sexes; very rare)' },
  'MALT1':  { full: 'MALT1-824aa / AR-LOF / Paracaspase / mucosa-associated lymphoid tissue 1 / 18q21.31 / MALT1-LOF-CID / Treg-Dysregulation-PATHOGNOMONIC / Colitis-PATHOGNOMONIC-distinguish-IPEX-IL2RA / Paracaspase-Cleavage-Assay / Agammaglobulinemia / HSCT', locus: '18q21.31', size: '824 aa / 92 kDa', inh: 'Autosomal recessive LOF (AR; 18q21.31; biallelic; both sexes)' },
  'CTPS1':  { full: 'CTPS1-591aa / AR / CTP synthase 1 / de novo CTP synthesis in T cells / 1p34.2 / CTPS1-Deficiency / T-Cell-Specific-dCTP-Pool-Depleted / EBV-VZV-Proliferation-Failure-PATHOGNOMONIC / Normal-Resting-Lymphocytes / Elevated-dUMP-Plasma-Diagnostic / HSCT', locus: '1p34.2',  size: '591 aa / 67 kDa', inh: 'Autosomal recessive (AR; 1p34.2; both sexes; Polynesian founder variant common)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'DNMT3B', group: 'ICF1 (DNA methylation)',   agamma: '✓✓✓', chrom_inst: '✓✓✓', facial: '✓✓', neuro: '✓✓',  ivig: '✓✓✓', hsct_urg: '±',   unique: 'CENTROMERIC INSTABILITY CHR 1/9/16 PATHOGNOMONIC on PHA-stimulated karyotype; facial anomalies (epicanthal folds, hypertelorism, macroglossia); profound agammaglobulinemia; B cells present (terminal diff block); IVIG lifelong; HSCT curative rarely needed.' },
  { gene: 'ZBTB24', group: 'ICF2 (BTB/POZ ZF)',        agamma: '✓✓✓', chrom_inst: '✓✓✓', facial: '✓✓', neuro: '✓',   ivig: '✓✓✓', hsct_urg: '±',   unique: 'Same chr 1/9/16 instability pattern as ICF1 — cannot distinguish on karyotype alone; MILDER NEUROLOGICAL involvement than ICF1 (intellectual disability less severe/absent in ~30%); BTB/POZ zinc-finger regulates CDCA7 expression; gene sequencing required for subtyping.' },
  { gene: 'CDCA7',  group: 'ICF3 (CDC-associated 7)',  agamma: '✓✓',  chrom_inst: '✓✓✓', facial: '✓✓', neuro: '±',   ivig: '✓✓',  hsct_urg: '±',   unique: 'VERY RARE (<20 families globally); same centromeric instability; CDCA7 bridges ZBTB24 regulation to HELLS helicase recruitment; ICF multigene panel required (all 4 subtypes); clinical phenotype variable; IVIG supportive.' },
  { gene: 'HELLS',  group: 'ICF4 (Lymphoid helicase)', agamma: '✓✓',  chrom_inst: '✓✓✓', facial: '✓',  neuro: '±',   ivig: '✓✓',  hsct_urg: '±',   unique: 'RAREST ICF (~10 families); "Lymphoid Specific" — highest expression in lymphoid tissue; terminal effector of ZBTB24→CDCA7→HELLS→DNMT3B methylation chain; loss of any link in this chain = ICF; HSCT considered in severe disease.' },
  { gene: 'CARD11', group: 'CARD11 LOF (CBM CID)',     agamma: '✓✓✓', chrom_inst: '✗',   facial: '✗',  neuro: '✗',   ivig: '✓✓✓', hsct_urg: '✓✓✓', unique: 'CARD11 LOF vs GOF CRITICAL SAME-GENE OPPOSITE-DISEASE DDx: LOF (AR biallelic) = profound CID + agammaglobulinemia + T-cell anergy → HSCT URGENT MANDATORY; GOF (AD monoallelic BENTA) = elevated IgE + atopic dermatitis + lymphoproliferation → venetoclax + MALT1 inhibitors. ALWAYS confirm LOF vs GOF before treating.' },
  { gene: 'BCL10',  group: 'BCL10 LOF (CBM CID)',      agamma: '✓✓✓', chrom_inst: '✗',   facial: '✗',  neuro: '✗',   ivig: '✓✓✓', hsct_urg: '✓✓',  unique: 'VERY RARE (<10 families); BCL10 CASPASE RECRUITMENT DOMAIN adaptor links CARD11 scaffold to MALT1 paracaspase; LOF = both B-cell and T-cell activation absent (NF-κB via TCR and BCR); agammaglobulinemia; IVIG + HSCT curative.' },
  { gene: 'MALT1',  group: 'MALT1 LOF (Treg/Colitis)', agamma: '✓✓',  chrom_inst: '✗',   facial: '✗',  neuro: '✗',   ivig: '✓✓',  hsct_urg: '✓✓',  unique: 'TREG DYSREGULATION + COLITIS PATHOGNOMONIC (distinguish from IPEX/IL2RA by MALT1 paracaspase assay); MALT1 LOF = CBM signalosome absent AND Treg development impaired → inflammatory enteropathy; MALT1 paracaspase assay shows absent substrate cleavage — unique diagnostic test.' },
  { gene: 'CTPS1',  group: 'CTPS1 (T-cell dCTP)',      agamma: '✗',   chrom_inst: '✗',   facial: '✗',  neuro: '✗',   ivig: '±',   hsct_urg: '✓',   unique: 'EBV/VZV-SPECIFIC T-CELL PROLIFERATION FAILURE PATHOGNOMONIC; resting lymphocytes NORMAL (DDx SCID); T cells cannot proliferate in response to EBV/VZV (dCTP depletion); elevated plasma dUMP diagnostic; Polynesian founder variant c.1764-2A>G common; HSCT curative.' },
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
  const isDNMT3B = gene === 'DNMT3B';
  const isZBTB24 = gene === 'ZBTB24';
  const isCDCA7  = gene === 'CDCA7';
  const isHELLS  = gene === 'HELLS';
  const isCARD11 = gene === 'CARD11';
  const isBCL10  = gene === 'BCL10';
  const isMALT1  = gene === 'MALT1';
  const isCTPS1  = gene === 'CTPS1';
  const groupLabel = isDNMT3B ? 'ICF1 (Centromeric-Instability-Chr1/9/16-PATHOGNOMONIC)'
    : isZBTB24 ? 'ICF2 (Milder-Neuro; ZBTB24→CDCA7 regulation)'
    : isCDCA7  ? 'ICF3 (Very-Rare; CDCA7→HELLS pathway)'
    : isHELLS  ? 'ICF4 (Rarest; HELLS lymphoid-specific helicase)'
    : isCARD11 ? 'CARD11 LOF CID (HSCT-URGENT; CRITICAL DDx GOF=BENTA)'
    : isBCL10  ? 'BCL10 LOF CID (Very-Rare; T+B activation absent)'
    : isMALT1  ? 'MALT1 LOF (Treg-Dysregulation; Colitis-PATHOGNOMONIC)'
    : 'CTPS1 Deficiency (EBV/VZV-Proliferation-Failure-PATHOGNOMONIC)';
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
          <Badge text={`Agamma ${Math.round((cohort.agamma_pct || 0) * 100)}%`}         color={color} />
          <Badge text={`IVIG ${Math.round((cohort.ivig_pct || 0) * 100)}%`}              color='#1565c0' />
          <Badge text={`HSCT ${Math.round((cohort.hsct_pct || 0) * 100)}%`}              color='#b71c1c' />
          <Badge text={`Infect ${Math.round((cohort.recurrent_infect_pct || 0) * 100)}%`} color='#455a64' />
          <Badge text={`Onset ${cohort.median_onset_months}mo`}                          color='#37474f' />
        </div>
      )}
    </div>
  );
}

export default function HereditaryICFCBMSignalosomeCombinedImmunodeficiencyAtlasPage() {
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

  if (loading) return <div style={{ color: '#fff', padding: 32 }}>Loading Hereditary-ICF-Syndrome-CBM-Signalosome-Combined-Immunodeficiency-Atlas…</div>;
  if (error)   return <div style={{ color: '#f44', padding: 32 }}>Error: {error}</div>;

  const stats        = overview?.aggregate_stats || {};
  const cohortByGene = {};
  (breakdown?.gene_cohorts || []).forEach(c => { cohortByGene[c.gene] = c; });

  return (
    <div style={{ background: '#0a0a1a', minHeight: '100vh', color: '#eee', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: '#0a0a2e', borderBottom: '2px solid #1a237e', padding: '20px 32px' }}>
        <h1 style={{ color: '#90caf9', fontSize: 22, marginBottom: 4 }}>
          🧬 Hereditary-ICF-Syndrome-CBM-Signalosome-Combined-Immunodeficiency-Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#aaa' }}>
          Complete 8-Gene ICF Syndrome · CBM Signalosome · CTPS1 Deficiency Reference ·
          DNMT3B · ZBTB24 · CDCA7 · HELLS · CARD11 · BCL10 · MALT1 · CTPS1 ·
          320-Patient Aggregate (8×40) · Seeds 3998-4005
        </div>
        <div style={{ marginTop: 8, display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 12 }}>
          <span style={{ background: '#1a237e', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Agamma {Math.round((stats.agamma_pct || 0) * 100)}%</span>
          <span style={{ background: '#1565c0', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>IVIG {Math.round((stats.ivig_pct || 0) * 100)}%</span>
          <span style={{ background: '#b71c1c', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>HSCT {Math.round((stats.hsct_pct || 0) * 100)}%</span>
          <span style={{ background: '#37474f', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Recurrent-Infect {Math.round((stats.recurrent_infections_pct || 0) * 100)}%</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid #333', background: '#0a0a2e' }}>
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
            <div style={{ background: '#0a0a2e', border: '1px solid #1a237e', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 10 }}>Key Clinical Facts — 8 ICF/CBM-Signalosome/CTPS1 CID Diseases</div>
              {(overview?.key_facts || []).map((f, i) => (
                <div key={i} style={{ fontSize: 12, color: '#ddd', marginBottom: 6, paddingLeft: 12, borderLeft: '3px solid #1a237e' }}>{f}</div>
              ))}
            </div>

            {/* Disease Groups */}
            <div style={{ background: '#0a1628', border: '1px solid #01579b', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 12 }}>Disease Groups by Mechanism and Pathway</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(270px, 1fr))', gap: 12 }}>
                {[
                  { label: 'ICF1 — DNMT3B (de novo methylation)',  genes: 'DNMT3B',        desc: 'Most common ICF (~50%); centromeric chr1/9/16 instability PATHOGNOMONIC; facial anomalies; profound agammaglobulinemia; B cells present; IVIG lifelong; HSCT curative.', color: '#1a237e' },
                  { label: 'ICF2 — ZBTB24 (CDCA7 regulator)',      genes: 'ZBTB24',        desc: 'Second most common ICF (~30%); same chr1/9/16 instability; milder neurological involvement than ICF1; BTB/POZ zinc-finger regulates CDCA7.', color: '#311b92' },
                  { label: 'ICF3 — CDCA7 (HELLS adaptor)',         genes: 'CDCA7',         desc: 'Very rare (<20 families); same centromeric instability; bridges ZBTB24→HELLS; multigene ICF panel required for subtyping.', color: '#4a148c' },
                  { label: 'ICF4 — HELLS (lymphoid helicase)',      genes: 'HELLS',         desc: 'Rarest ICF (~10 families); SMARCA6/LSH; highest lymphoid expression; terminal effector of ZBTB24→CDCA7→HELLS→DNMT3B chain.', color: '#880e4f' },
                  { label: 'CARD11 LOF — CBM scaffold CID',         genes: 'CARD11',        desc: 'HSCT URGENT MANDATORY; CRITICAL DDx: LOF (AR biallelic CID) vs GOF (AD monoallelic BENTA) are OPPOSITE diseases from same gene; confirm LOF/GOF first.', color: '#b71c1c' },
                  { label: 'BCL10 LOF — CBM adaptor CID',          genes: 'BCL10',         desc: 'Very rare (<10 families); BCL10 CARD/adaptor; T+B activation absent; agammaglobulinemia; IVIG + HSCT curative.', color: '#e65100' },
                  { label: 'MALT1 LOF — Treg dysregulation/Colitis', genes: 'MALT1',       desc: 'Treg dysregulation + colitis PATHOGNOMONIC; distinguish from IPEX/IL2RA by MALT1 paracaspase assay; HSCT corrects both CID and inflammation.', color: '#33691e' },
                  { label: 'CTPS1 — T-cell dCTP/EBV-VZV selective', genes: 'CTPS1',        desc: 'EBV/VZV T-cell proliferation failure PATHOGNOMONIC; resting lymphocytes NORMAL; elevated dUMP plasma diagnostic; Polynesian founder variant common; HSCT.', color: '#006064' },
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
            <div style={{ marginTop: 20, background: '#0a0a2e', border: '1px solid #333', borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 10 }}>Aggregate Statistics — 320 Patients (8×40)</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))', gap: 8 }}>
                {[
                  ['Agamma',        stats.agamma_pct,              '%', '#1a237e'],
                  ['IVIG',          stats.ivig_pct,                '%', '#1565c0'],
                  ['HSCT',          stats.hsct_pct,                '%', '#b71c1c'],
                  ['Recurrent-Inf', stats.recurrent_infections_pct,'%', '#37474f'],
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
                <tr style={{ background: '#0a0a2e' }}>
                  {['Gene', 'Group', 'Agamma', 'Chr Instab', 'Facial', 'Neuro', 'IVIG', 'HSCT Urg', 'Clinical Pearl'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', color: '#90caf9', textAlign: 'left', borderBottom: '1px solid #444' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#161616' : '#0a0a2e' }}>
                    <td style={{ padding: '8px 10px', color: GENE_COLORS[row.gene], fontWeight: 700 }}>{row.gene}</td>
                    <td style={{ padding: '8px 10px', color: '#aaa', fontSize: 10 }}>{row.group}</td>
                    <td style={{ padding: '8px 10px', color: row.agamma    === '✓✓✓' ? '#90caf9' : '#888' }}>{row.agamma}</td>
                    <td style={{ padding: '8px 10px', color: row.chrom_inst === '✓✓✓' ? '#ce93d8' : '#888' }}>{row.chrom_inst}</td>
                    <td style={{ padding: '8px 10px', color: row.facial    === '✓✓'  ? '#ffcc80' : '#888' }}>{row.facial}</td>
                    <td style={{ padding: '8px 10px', color: row.neuro     === '✓✓'  ? '#f48fb1' : '#888' }}>{row.neuro}</td>
                    <td style={{ padding: '8px 10px', color: row.ivig      === '✓✓✓' ? '#80deea' : '#888' }}>{row.ivig}</td>
                    <td style={{ padding: '8px 10px', color: row.hsct_urg  === '✓✓✓' ? '#ef9a9a' : '#888' }}>{row.hsct_urg}</td>
                    <td style={{ padding: '8px 10px', color: '#ddd', maxWidth: 360, fontSize: 10 }}>{row.unique}</td>
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
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 10 }}>Cascade Diagnostic Testing — ICF/CBM/CTPS1 CID Workup</div>
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
                <div style={{ fontSize: 11, color: '#aaa', marginBottom: 4 }}><strong style={{ color: '#90caf9' }}>Shared:</strong> {grp.shared_feature}</div>
                <div style={{ fontSize: 11, color: '#aaa' }}><strong style={{ color: '#90caf9' }}>DDx:</strong> {grp.key_ddx}</div>
              </div>
            ))}

            {/* Pathognomonic Features */}
            <div style={{ background: '#0a0a2e', border: '1px solid #1a237e', borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 10 }}>Pathognomonic Features &amp; Clinical Rules</div>
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
                <div key={d.term} style={{ background: '#0a0a2e', border: '1px solid #333', borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 6 }}>{d.term}</div>
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
