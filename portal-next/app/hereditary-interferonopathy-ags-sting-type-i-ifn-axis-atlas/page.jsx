'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-interferonopathy-ags-sting-type-i-ifn-axis-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'TREX1':    '#b71c1c',  // deep red       -- AGS1; most common; chilblains PATHOGNOMONIC
  'RNASEH2B': '#c62828',  // red            -- AGS2; mildest; may walk; p.A177T founder
  'RNASEH2A': '#e65100',  // burnt orange   -- AGS4; severe; catalytic subunit
  'RNASEH2C': '#f57f17',  // amber          -- AGS3; intermediate; structural subunit
  'SAMHD1':   '#1565c0',  // deep blue      -- AGS5; STROKE RISK; dNTPase; HIV restriction
  'ADAR':     '#4a148c',  // deep purple    -- AGS6; bilateral striatal degeneration PATHOGNOMONIC
  'IFIH1':    '#1b5e20',  // deep green     -- AGS7; MDA5-GOF; Singleton-Merten; mildest neuro
  'STING1':   '#880e4f',  // dark crimson   -- SAVI; lung fibrosis PATHOGNOMONIC; skin necrosis
};

const GENE_INFO = {
  'TREX1':    { full: 'TREX1-314aa / AR+AD / 3p21.31 / AGS1 / cGAS-STING / Chilblains-PATHOGNOMONIC / Intracranial-Calcification / Most-Common-AGS-20pct / Baricitinib+RTIs',            locus: '3p21.31',   size: '314 aa / 35 kDa', inh: 'AR (biallelic) / AD (heterozygous)' },
  'RNASEH2B': { full: 'RNASEH2B-341aa / AR / 13q14.3 / AGS2 / RNase-H2-B-Subunit / PCNA-Binding / Mildest-AGS / p.A177T-Founder / May-Walk / Calcifications / Baricitinib',              locus: '13q14.3',   size: '341 aa / 37 kDa', inh: 'Autosomal recessive' },
  'RNASEH2A': { full: 'RNASEH2A-299aa / AR / 19p13.2 / AGS4 / RNase-H2-A-Subunit / Catalytic / Severe-AGS / Non-Ambulatory / Deep-Microcephaly / Calcifications / Baricitinib',          locus: '19p13.2',   size: '299 aa / 33 kDa', inh: 'Autosomal recessive' },
  'RNASEH2C': { full: 'RNASEH2C-164aa / AR / 11q13.1 / AGS3 / RNase-H2-C-Subunit / Structural / Intermediate-Severity / Calcifications / ISG-Elevated / Baricitinib',                   locus: '11q13.1',   size: '164 aa / 19 kDa', inh: 'Autosomal recessive' },
  'SAMHD1':   { full: 'SAMHD1-626aa / AR / 20q11.23 / AGS5 / dNTPase / HIV-Restriction / Stroke-PATHOGNOMONIC / Calcifications / CLL-Somatic / Anticoagulation / Baricitinib',           locus: '20q11.23',  size: '626 aa / 72 kDa', inh: 'Autosomal recessive' },
  'ADAR':     { full: 'ADAR-1226aa / AR / 1q21.3 / AGS6 / ADAR1 / A-to-I-Editing / p.P193A-Hotspot / Bilateral-Striatal-Degeneration-PATHOGNOMONIC / Dystonia / Baricitinib',            locus: '1q21.3',    size: '1226 aa / 136 kDa', inh: 'Autosomal recessive (biallelic)' },
  'IFIH1':    { full: 'IFIH1-1025aa / AD-GOF / 2q24.2 / AGS7 / MDA5 / dsRNA-Sensor / Singleton-Merten-SinMet1 / Dental-Aortic-Calcification-Glaucoma / T1DM-Risk / Baricitinib',        locus: '2q24.2',    size: '1025 aa / 117 kDa', inh: 'Autosomal dominant GOF' },
  'STING1':   { full: 'STING1-379aa / AD-GOF / 5q31.2 / SAVI / TMEM173 / Lung-Fibrosis-PATHOGNOMONIC / Skin-Necrosis / Vasculopathy / Highest-ISG-Score / Baricitinib+Anti-TNF',        locus: '5q31.2',    size: '379 aa / 42 kDa', inh: 'Autosomal dominant GOF' },
};

const PHENOTYPE_ROWS = [
  { gene: 'TREX1',    group: 'cGAS-Upstream-DNA',   calcif: '✓✓✓', isg: '✓✓✓', csf_ifna: '✓✓✓', chilblains: '✓✓',  micro: '✓✓✓', seizure: '✓✓',  stroke: '±',   lung: '−',   hsct: '−',   unique: 'CHILBLAINS (pernio) IN CHILD WITH NEUROREG = TREX1 PATHOGNOMONIC; most common AGS (20-25%); AD heterozygous → chilblains lupus/SLE/DCM; TREX1 degrades retrotransposon cDNA → LOF → cGAS; baricitinib+RTIs (tenofovir/lamivudine)' },
  { gene: 'RNASEH2B', group: 'RNase-H2-Complex',    calcif: '✓✓',  isg: '✓✓✓', csf_ifna: '✓✓',  chilblains: '±',   micro: '✓✓',  seizure: '✓✓',  stroke: '−',   lung: '−',   hsct: '−',   unique: 'MILDEST AGS — MAY RETAIN INDEPENDENT AMBULATION (unlike other AGS genes); p.A177T Northern European founder; RNASEH2B is PCNA-binding B subunit; ~35% of RNASEH2-subunit AGS; enzyme activity assay confirms complex loss' },
  { gene: 'RNASEH2A', group: 'RNase-H2-Complex',    calcif: '✓✓✓', isg: '✓✓✓', csf_ifna: '✓✓✓', chilblains: '±',   micro: '✓✓✓', seizure: '✓✓',  stroke: '−',   lung: '−',   hsct: '−',   unique: 'SEVERE AGS (AGS4); catalytic A subunit (active site) — A-subunit LOF = most severe RNase H2 loss; non-ambulatory; profound microcephaly; DISTINGUISH: A (severe AGS4) vs B (mild AGS2) vs C (intermediate AGS3) by sequencing' },
  { gene: 'RNASEH2C', group: 'RNase-H2-Complex',    calcif: '✓✓',  isg: '✓✓✓', csf_ifna: '✓✓',  chilblains: '±',   micro: '✓✓',  seizure: '✓✓',  stroke: '−',   lung: '−',   hsct: '−',   unique: 'INTERMEDIATE SEVERITY (AGS3); structural C subunit; cannot distinguish from AGS2/4 by clinical features alone — sequence all three RNASEH2 subunits simultaneously; RNase H2 enzyme activity identical to AGS2/AGS4 (all undetectable)' },
  { gene: 'SAMHD1',   group: 'cGAS-Upstream-dNTP',  calcif: '✓✓',  isg: '✓✓',  csf_ifna: '✓✓',  chilblains: '±',   micro: '✓✓',  seizure: '±',   stroke: '✓✓✓',lung: '−',   hsct: '−',   unique: 'STROKE IS PATHOGNOMONIC (42% stroke history — HIGHEST of all AGS genes); dNTPase restricts HIV/lentiviral replication (HIV-2 Vpx degrades SAMHD1); LOF → elevated dNTPs → LINE-1 cDNA → cGAS; anticoagulation MANDATORY; somatic SAMHD1 mutations in CLL' },
  { gene: 'ADAR',     group: 'RNA-Editing-AGS6',    calcif: '✓✓',  isg: '✓✓✓', csf_ifna: '✓✓✓', chilblains: '±',   micro: '✓✓',  seizure: '✓✓',  stroke: '±',   lung: '−',   hsct: '−',   unique: 'BILATERAL STRIATAL DEGENERATION ON MRI (putamen+caudate) DURING FEBRILE ILLNESS = ADAR-AGS6 PATHOGNOMONIC; p.P193A HOTSPOT (>50% AGS6); DYSTONIA from striatal damage; VACCINATE INFLUENZA (natural infection triggers acute striatal episodes)' },
  { gene: 'IFIH1',    group: 'MDA5-GOF-AGS7',       calcif: '✓✓',  isg: '✓✓',  csf_ifna: '✓✓',  chilblains: '±',   micro: '±',   seizure: '±',   stroke: '±',   lung: '±',   hsct: '−',   unique: 'MILDEST NEUROLOGICAL AGS7 (MDA5-GOF); SINGLETON-MERTEN SinMet1 = DENTAL ANOMALIES + AORTIC VALVE CALCIFICATION + GLAUCOMA PATHOGNOMONIC; T1DM susceptibility alleles; relapsing-remitting course; baricitinib most effective' },
  { gene: 'STING1',   group: 'SAVI-Downstream',     calcif: '±',   isg: '✓✓✓', csf_ifna: '±',   chilblains: '✓✓✓',micro: '±',   seizure: '±',   stroke: '±',   lung: '✓✓✓',hsct: '±',   unique: 'PULMONARY FIBROSIS IN INFANCY = SAVI PATHOGNOMONIC (UNIQUE to STING1-GOF — absent in AGS1-7); SKIN NECROSIS/AVASCULAR NECROSIS digits/nose; HIGHEST ISG of all; DUAL IFN+NF-kB activation; NO calcifications; baricitinib+anti-TNF; lung transplant endpoint' },
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
  const groupLabel = ['TREX1'].includes(gene)               ? 'cGAS-Upstream / DNA Nuclease'
                   : gene === 'SAMHD1'                       ? 'cGAS-Upstream / dNTP Pool'
                   : ['RNASEH2A','RNASEH2B','RNASEH2C'].includes(gene) ? 'RNase H2 Complex (AGS2-4)'
                   : gene === 'ADAR'                         ? 'RNA A-to-I Editing (AGS6)'
                   : gene === 'IFIH1'                        ? 'MDA5 dsRNA Sensor GOF (AGS7)'
                   : 'STING1 GOF — SAVI';

  const agsLabel = gene === 'TREX1' ? 'AGS1' : gene === 'RNASEH2B' ? 'AGS2' :
                   gene === 'RNASEH2C' ? 'AGS3' : gene === 'RNASEH2A' ? 'AGS4' :
                   gene === 'SAMHD1' ? 'AGS5' : gene === 'ADAR' ? 'AGS6' :
                   gene === 'IFIH1' ? 'AGS7' : 'SAVI';

  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          <span style={{ fontWeight: 700, fontSize: 18, color }}>{gene}</span>
          <span style={{ marginLeft: 8, fontSize: 12, background: color, color: '#fff', padding: '2px 8px', borderRadius: 4 }}>{agsLabel}</span>
          <span style={{ marginLeft: 6, fontSize: 12, background: '#333', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>{groupLabel}</span>
        </div>
        <div style={{ fontSize: 12, color: '#888', textAlign: 'right' }}>
          <div>{info.size} · {info.locus}</div>
          <div>{info.inh}</div>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#bbb', marginTop: 6 }}>{info.full}</div>
      {cohort && (
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Badge text={`ISG Elevated ${cohort.isg_elevated_pct?.toFixed(0)}%`}        color={color} />
          <Badge text={`CSF IFN-α ${cohort.csf_ifna_pct?.toFixed(0)}%`}              color='#880e4f' />
          <Badge text={`Calcifications ${cohort.calcification_pct?.toFixed(0)}%`}    color='#795548' />
          <Badge text={`Chilblains ${cohort.chilblains_pct?.toFixed(0)}%`}           color='#e65100' />
          <Badge text={`Seizures ${cohort.seizure_pct?.toFixed(0)}%`}                color='#4a148c' />
          <Badge text={`Stroke ${cohort.stroke_pct?.toFixed(0)}%`}                  color='#c62828' />
          <Badge text={`Lung Fibrosis ${cohort.lung_fibrosis_pct?.toFixed(0)}%`}    color='#880e4f' />
          <Badge text={`Baricitinib ${cohort.baricitinib_pct?.toFixed(0)}%`}        color='#1b5e20' />
          <Badge text={`n=${cohort.n}`}                                              color='#555' />
        </div>
      )}
    </div>
  );
}

export default function Page() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview,   setOverview]   = useState(null);
  const [breakdown,  setBreakdown]  = useState(null);
  const [definitions,setDefinitions]= useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);

  async function load(tab) {
    setLoading(true); setError(null);
    try {
      if (tab === 'Overview' || tab === 'Gene Table') {
        if (!overview) {
          const r = await fetch(`${API}/api/${SLUG}/overview`);
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          setOverview(await r.json());
        }
      }
      if (tab === 'Pathway Map') {
        if (!breakdown) {
          const r = await fetch(`${API}/api/${SLUG}/breakdown`);
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          setBreakdown(await r.json());
        }
      }
      if (tab === 'Definitions') {
        if (!definitions) {
          const r = await fetch(`${API}/api/${SLUG}/definitions`);
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          setDefinitions(await r.json());
        }
      }
    } catch (e) { setError(e.message); }
    setLoading(false);
  }

  useEffect(() => { load(activeTab); }, [activeTab]);

  const tabStyle = (t) => ({
    padding: '8px 18px', cursor: 'pointer', borderBottom: activeTab === t ? '3px solid #b71c1c' : '3px solid transparent',
    color: activeTab === t ? '#b71c1c' : '#aaa', fontWeight: activeTab === t ? 700 : 400, fontSize: 14,
  });

  return (
    <div style={{ background: '#111', minHeight: '100vh', color: '#eee', fontFamily: 'monospace', padding: 24 }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 8 }}>
          <span style={{ fontSize: 28, fontWeight: 700, color: '#b71c1c' }}>🧬</span>
          <div>
            <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#fff' }}>
              Hereditary Interferonopathy — AGS / STING / Type I IFN Axis Atlas
            </h1>
            <div style={{ fontSize: 12, color: '#888', marginTop: 2 }}>
              TREX1 · RNASEH2B · RNASEH2A · RNASEH2C · SAMHD1 · ADAR · IFIH1 · STING1 &nbsp;|&nbsp;
              AGS1-7 + SAVI · 8-Gene Reference · 320 Patients · seeds 4030-4037
            </div>
          </div>
        </div>

        {/* Quick-reference pathway pills */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
          {[
            { label: 'cGAS-STING Pathway',         color: '#b71c1c' },
            { label: 'ISG Score Universal Screen',  color: '#1565c0' },
            { label: 'Baricitinib All 8 Genes',     color: '#1b5e20' },
            { label: 'AGS1-7 + SAVI',               color: '#4a148c' },
            { label: 'Chilblains = AGS Flag',        color: '#e65100' },
            { label: 'SAVI = Lung Fibrosis Only',    color: '#880e4f' },
          ].map(p => (
            <span key={p.label} style={{
              background: p.color, color: '#fff', padding: '3px 10px',
              borderRadius: 12, fontSize: 11, fontWeight: 600
            }}>{p.label}</span>
          ))}
        </div>

        {/* Tab bar */}
        <div style={{ display: 'flex', borderBottom: '1px solid #333', gap: 4 }}>
          {TABS.map(t => <span key={t} style={tabStyle(t)} onClick={() => setActiveTab(t)}>{t}</span>)}
        </div>
      </div>

      {loading && <div style={{ color: '#888', padding: 20 }}>Loading…</div>}
      {error   && <div style={{ color: '#f44', padding: 12 }}>Error: {error}</div>}

      {/* ── OVERVIEW TAB ── */}
      {activeTab === 'Overview' && overview && (
        <div>
          {/* Aggregate stats */}
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 }}>
            {[
              { label: 'ISG Score Elevated', val: overview.aggregate_stats?.isg_elevated_pct?.toFixed(1) + '%', color: '#b71c1c' },
              { label: 'Intracranial Calcif', val: overview.aggregate_stats?.intracranial_calcif_pct?.toFixed(1) + '%', color: '#1565c0' },
              { label: 'Baricitinib Use',     val: overview.aggregate_stats?.baricitinib_pct?.toFixed(1) + '%',        color: '#1b5e20' },
              { label: 'Total Patients',      val: overview.total_n,    color: '#4a148c' },
            ].map(s => (
              <div key={s.label} style={{ background: '#1a1a1a', border: `1px solid ${s.color}`, borderRadius: 8, padding: '12px 20px', minWidth: 140 }}>
                <div style={{ fontSize: 11, color: '#888' }}>{s.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: s.color }}>{s.val}</div>
              </div>
            ))}
          </div>

          {/* AGS type map */}
          {overview.ags_type_map && (
            <div style={{ background: '#1a1a1a', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#b71c1c', marginBottom: 10 }}>AGS Classification Map</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {Object.entries(overview.ags_type_map).map(([type, gene]) => (
                  <div key={type} style={{ background: GENE_COLORS[gene] || '#333', color: '#fff', padding: '6px 14px', borderRadius: 6, fontSize: 12 }}>
                    <span style={{ fontWeight: 700 }}>{type}</span>
                    <span style={{ marginLeft: 6, opacity: 0.85 }}>→ {gene}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Key facts */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#b71c1c', marginBottom: 10 }}>Key Clinical Facts</div>
            {(overview.key_facts || []).map((fact, i) => (
              <div key={i} style={{ background: '#1a1a1a', borderRadius: 6, padding: '10px 14px', marginBottom: 8, fontSize: 12, lineHeight: 1.6, color: '#ccc' }}>
                <span style={{ color: GENE_COLORS[GENES[i]] || '#b71c1c', fontWeight: 700 }}>{GENES[i]}:</span>{' '}
                {fact.replace(/^[A-Z0-9]+[^:]*:\s*/, '')}
              </div>
            ))}
          </div>

          {/* Gene cards (first 4) */}
          <div style={{ fontSize: 13, fontWeight: 700, color: '#b71c1c', marginBottom: 10 }}>Gene Profiles — cGAS-Upstream + RNase H2</div>
          {overview.cohort_summaries?.slice(0, 4).map((cohort, i) => {
            const gene = GENES[i];
            return <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={GENE_INFO[gene]} cohort={cohort} />;
          })}

          <div style={{ fontSize: 13, fontWeight: 700, color: '#b71c1c', marginBottom: 10, marginTop: 8 }}>Gene Profiles — ADAR / IFIH1 / STING1</div>
          {overview.cohort_summaries?.slice(4).map((cohort, i) => {
            const gene = GENES[i + 4];
            return <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={GENE_INFO[gene]} cohort={cohort} />;
          })}
        </div>
      )}

      {/* ── GENE TABLE TAB ── */}
      {activeTab === 'Gene Table' && overview && (
        <div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <thead>
                <tr style={{ background: '#222' }}>
                  {['Gene', 'AGS', 'Locus', 'aa', 'Inh', 'ISG%', 'CSF-IFNα%', 'Calcif%', 'Chilblains%', 'Stroke%', 'LungFib%', 'Baricit%', 'Unique PATHOGNOMONIC'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#b71c1c', borderBottom: '1px solid #333', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => {
                  const cohort = overview.cohort_summaries?.[i];
                  const color  = GENE_COLORS[row.gene] || '#b71c1c';
                  const agsLabel = row.gene === 'TREX1' ? 'AGS1' : row.gene === 'RNASEH2B' ? 'AGS2' :
                                   row.gene === 'RNASEH2C' ? 'AGS3' : row.gene === 'RNASEH2A' ? 'AGS4' :
                                   row.gene === 'SAMHD1' ? 'AGS5' : row.gene === 'ADAR' ? 'AGS6' :
                                   row.gene === 'IFIH1' ? 'AGS7' : 'SAVI';
                  return (
                    <tr key={row.gene} style={{ borderBottom: '1px solid #222', background: i % 2 === 0 ? '#111' : '#161616' }}>
                      <td style={{ padding: '8px 10px', color, fontWeight: 700 }}>{row.gene}</td>
                      <td style={{ padding: '8px 10px', color: '#888' }}>{agsLabel}</td>
                      <td style={{ padding: '8px 10px', color: '#888' }}>{GENE_INFO[row.gene].locus}</td>
                      <td style={{ padding: '8px 10px', color: '#888' }}>{GENE_INFO[row.gene].size.split(' ')[0]}</td>
                      <td style={{ padding: '8px 10px', color: '#888', whiteSpace: 'nowrap', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {GENE_INFO[row.gene].inh.split(' ')[0]}
                      </td>
                      <td style={{ padding: '8px 10px', color: '#e57373', fontWeight: 700 }}>{cohort?.isg_elevated_pct?.toFixed(0)}%</td>
                      <td style={{ padding: '8px 10px', color: '#f48fb1' }}>{cohort?.csf_ifna_pct?.toFixed(0)}%</td>
                      <td style={{ padding: '8px 10px', color: '#ffcc80' }}>{cohort?.calcification_pct?.toFixed(0)}%</td>
                      <td style={{ padding: '8px 10px', color: '#80cbc4' }}>{cohort?.chilblains_pct?.toFixed(0)}%</td>
                      <td style={{ padding: '8px 10px', color: cohort?.stroke_pct > 30 ? '#f44336' : '#aaa', fontWeight: cohort?.stroke_pct > 30 ? 700 : 400 }}>
                        {cohort?.stroke_pct?.toFixed(0)}%
                      </td>
                      <td style={{ padding: '8px 10px', color: cohort?.lung_fibrosis_pct > 50 ? '#e91e63' : '#aaa', fontWeight: cohort?.lung_fibrosis_pct > 50 ? 700 : 400 }}>
                        {cohort?.lung_fibrosis_pct?.toFixed(0)}%
                      </td>
                      <td style={{ padding: '8px 10px', color: '#a5d6a7' }}>{cohort?.baricitinib_pct?.toFixed(0)}%</td>
                      <td style={{ padding: '8px 10px', color: '#ccc', fontSize: 10, maxWidth: 300 }}>{row.unique}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── PATHWAY MAP TAB ── */}
      {activeTab === 'Pathway Map' && breakdown && (
        <div>
          {/* Diagnostic groups */}
          <div style={{ fontSize: 13, fontWeight: 700, color: '#b71c1c', marginBottom: 12 }}>Diagnostic Groups &amp; Pathomechanism</div>
          {Object.entries(breakdown.diagnostic_groups || {}).map(([key, grp]) => (
            <div key={key} style={{ background: '#1a1a1a', borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
                {(grp.genes || []).map(g => (
                  <span key={g} style={{ background: GENE_COLORS[g] || '#333', color: '#fff', padding: '2px 10px', borderRadius: 4, fontSize: 12, fontWeight: 700 }}>{g}</span>
                ))}
              </div>
              <div style={{ fontSize: 11, color: '#aaa', marginBottom: 6 }}><strong style={{ color: '#b71c1c' }}>Mechanism:</strong> {grp.mechanism}</div>
              <div style={{ fontSize: 11, color: '#aaa', marginBottom: 6 }}><strong style={{ color: '#1b5e20' }}>Shared:</strong> {grp.shared_feature}</div>
              <div style={{ fontSize: 11, color: '#aaa' }}><strong style={{ color: '#f57f17' }}>DDx:</strong> {grp.key_ddx}</div>
            </div>
          ))}

          {/* Pathognomonic features */}
          <div style={{ fontSize: 13, fontWeight: 700, color: '#b71c1c', marginBottom: 12, marginTop: 8 }}>Pathognomonic Features &amp; Red Flags</div>
          {Object.entries(breakdown.pathognomonic_features || {}).map(([key, text]) => (
            <div key={key} style={{ background: '#1a1a1a', borderLeft: '3px solid #b71c1c', borderRadius: 6, padding: '10px 14px', marginBottom: 10, fontSize: 11, color: '#ccc', lineHeight: 1.6 }}>
              {text}
            </div>
          ))}

          {/* Cascade testing */}
          <div style={{ fontSize: 13, fontWeight: 700, color: '#b71c1c', marginBottom: 12, marginTop: 8 }}>Cascade Testing Protocol (10 Steps)</div>
          {(breakdown.cascade_testing_order || []).map((step, i) => (
            <div key={i} style={{ background: '#1a1a1a', borderRadius: 6, padding: '10px 14px', marginBottom: 8, fontSize: 11, color: '#ccc', lineHeight: 1.6 }}>
              <span style={{ color: '#b71c1c', fontWeight: 700 }}>Step {i + 1}:</span> {step.replace(/^Step \d+:\s*/, '')}
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {activeTab === 'Definitions' && definitions && (
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#b71c1c', marginBottom: 12 }}>Clinical Definitions</div>
          {(definitions.definitions || []).map((d, i) => (
            <div key={i} style={{ background: '#1a1a1a', borderRadius: 8, padding: 16, marginBottom: 12 }}>
              <div style={{ fontWeight: 700, color: '#b71c1c', fontSize: 13, marginBottom: 6 }}>{d.term}</div>
              <div style={{ fontSize: 11, color: '#bbb', lineHeight: 1.7 }}>{d.definition}</div>
            </div>
          ))}

          {/* Key distinguishers */}
          {definitions.key_distinguishers && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#b71c1c', marginBottom: 10 }}>Key Distinguishers</div>
              {Object.entries(definitions.key_distinguishers).map(([key, text]) => (
                <div key={key} style={{ background: '#1a1a1a', borderLeft: '3px solid #f57f17', borderRadius: 6, padding: '10px 14px', marginBottom: 10, fontSize: 11, color: '#ccc', lineHeight: 1.6 }}>
                  <strong style={{ color: '#f57f17' }}>{key.replace(/_/g, ' / ')}:</strong> {text}
                </div>
              ))}
            </div>
          )}

          {/* Treatment matrix */}
          {definitions.treatment_matrix && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#b71c1c', marginBottom: 10 }}>Treatment Matrix</div>
              {Object.entries(definitions.treatment_matrix).map(([drug, text]) => (
                <div key={drug} style={{ background: '#1a1a1a', borderLeft: '3px solid #1b5e20', borderRadius: 6, padding: '10px 14px', marginBottom: 10, fontSize: 11, color: '#ccc', lineHeight: 1.6 }}>
                  <strong style={{ color: '#1b5e20', textTransform: 'uppercase' }}>{drug}:</strong> {text}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
