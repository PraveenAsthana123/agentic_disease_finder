'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-complement-deficiency-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'C1QA': '#880e4f',  // deep crimson-pink   -- C1q deficiency; SLE 90%; lupus rash PATHOGNOMONIC; anti-dsDNA negative DDx
  'C2':   '#4a148c',  // deep purple          -- C2 most common Western 1:20000; pneumococcal vaccines; penicillin prophylaxis
  'C4A':  '#311b92',  // deep indigo          -- C4A NULL European; partial SLE risk 2-3x; classical pathway amplifier
  'C3':   '#1a237e',  // dark navy            -- C3 central; all pathways blocked; MPGN PATHOGNOMONIC; eculizumab
  'CFB':  '#01579b',  // dark blue            -- Factor B; alternative pathway; Dense Deposit Disease; AP50 zero CH50 normal
  'CFD':  '#006064',  // dark teal            -- Factor D RAREST; North Africa founder; 2x vaccination MANDATORY
  'C5':   '#1b5e20',  // dark green           -- C5 terminal entry; Neisseria PATHOGNOMONIC; eculizumab FDA 2007 CURATIVE
  'CFP':  '#b71c1c',  // deep red             -- Properdin ONLY XLR; males only; fulminant meningococcemia 30% mortality
};

const GENE_INFO = {
  'C1QA': { full: 'C1QA-245aa / AR / C1q-alpha-chain / Classical-pathway-recognition / 1p36.12 / C1q-Deficiency / SLE-90pct-HIGHEST-Autoimmune-Risk / LUPUS-LIKE-RASH-PATHOGNOMONIC / Anti-dsDNA-NEGATIVE-DDx / Hydroxychloroquine / Very-rare-<40-families', locus: '1p36.12',  size: '245 aa / 27 kDa',  inh: 'Autosomal recessive (AR; <40 families worldwide)' },
  'C2':   { full: 'C2-752aa / AR / Classical-lectin-pathway-serine-protease / C3-convertase-C4b2a / 6p21.33 / C2-Deficiency / Most-Common-Complement-Def-Western-1:20000 / Encapsulated-Bacteria-Primary-Risk / SLE-30pct / Pneumococcal-Vaccines-MANDATORY / Penicillin-Prophylaxis-Lifelong', locus: '6p21.33',  size: '752 aa / 102 kDa', inh: 'Autosomal recessive (AR; HLA-A25-B18-DR2 null haplotype)' },
  'C4A':  { full: 'C4A-1744aa / AR+partial-AD / Thioester-complement-protein / Classical-pathway-amplifier / 6p21.33 / C4A-Deficiency / C4A-NULL-Most-Common-MHC-Null-Allele-European / Partial-Deficiency-SLE-2-3x-Risk / Complete-Deficiency-High-SLE-Risk / CNV-Analysis-Required', locus: '6p21.33',  size: '1744 aa / 193 kDa', inh: 'AR (complete); heterozygous C4A*Q0 null allele (partial deficiency)' },
  'C3':   { full: 'C3-1663aa / AR / Central-complement-all-pathways-converge / Most-abundant-complement-protein / 19p13.3 / C3-Deficiency / Most-Severe-Encapsulated-Bacteria / MPGN-PATHOGNOMONIC / All-Pathways-Blocked-CH50+AP50-ZERO / Eculizumab-Emerging-MPGN', locus: '19p13.3',  size: '1663 aa / 187 kDa', inh: 'Autosomal recessive (AR; most severe susceptibility)' },
  'CFB':  { full: 'CFB-764aa / AR / Alternative-pathway-serine-protease / C3bBb-convertase-component / 6p21.33 / Factor-B-Deficiency / Selective-AP-Deficiency / AP50-ZERO-CH50-NORMAL / Recurrent-Neisseria-Meningitidis / MPGN-Type2-Dense-Deposit-Disease / Eculizumab', locus: '6p21.33',  size: '764 aa / 93 kDa',  inh: 'Autosomal recessive (AR; selective AP absent; classical intact)' },
  'CFD':  { full: 'CFD-259aa / AR / Alternative-pathway-rate-limiting-enzyme / Factor-D-Lowest-Plasma-Concentration / 19p13.3 / Factor-D-Deficiency-RAREST / Less-than-15-Families-Worldwide / North-Africa-Founder-Moroccan / 2x-Vaccination-Schedule-MANDATORY / AP50-ZERO', locus: '19p13.3',  size: '259 aa / 24 kDa',  inh: 'Autosomal recessive (AR; RAREST; North Africa founder effect)' },
  'C5':   { full: 'C5-1676aa / AR / Terminal-pathway-entry / MAC-nucleation-C5b / C5a-Anaphylatoxin / 9q21.2 / C5-Deficiency / NEISSERIA-MENINGITIDIS-PATHOGNOMONIC / Eculizumab-FDA2007-CURATIVE / Ravulizumab-Monthly / Vaccinate-Before-Eculizumab-MANDATORY', locus: '9q21.2',   size: '1676 aa / 190 kDa', inh: 'Autosomal recessive (AR; terminal complement pathway absent)' },
  'CFP':  { full: 'CFP-442aa / XLR / Properdin-ONLY-positive-regulator / Stabilises-AP-C3bBb-convertase-10x / Xp11.23 / Properdin-Deficiency / ONLY-XLR-Complement-Deficiency / MALES-ONLY / Fulminant-Meningococcemia-30pct-Mortality / 2x-Vaccination-MANDATORY / Family-Screening-MANDATORY', locus: 'Xp11.23',  size: '442 aa / 53 kDa',  inh: 'X-linked recessive (XLR; ONLY XLR complement deficiency; males only)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'C1QA', group: 'Classical Pathway (C1q)',      sle: '✓✓✓', bacterial: '✓✓',  mpgn: '✓',   neisseria: '±',   eculizumab: '✗',   vaccine2: '✓✓✓', lupus_rash: '✓✓✓', hcq: '✓✓✓', unique: 'LUPUS-LIKE RASH PATHOGNOMONIC + ANTI-dsDNA NEGATIVE DDx from idiopathic SLE. 90% develop SLE = HIGHEST autoimmune risk. Impaired apoptotic clearance. Hydroxychloroquine + vaccination MANDATORY. Very rare (<40 families).' },
  { gene: 'C2',   group: 'Classical (Most Common)',      sle: '✓✓',  bacterial: '✓✓✓', mpgn: '±',   neisseria: '✓',   eculizumab: '✗',   vaccine2: '✓✓✓', lupus_rash: '±',   hcq: '±',   unique: 'MOST COMMON complement deficiency Western 1:20,000. ENCAPSULATED BACTERIA primary risk (S. pneumoniae/H. influenzae). SLE ~30%. PNEUMOCOCCAL VACCINES (PCV13+PPSV23) MANDATORY. PENICILLIN V PROPHYLAXIS lifelong. HLA-A25-B18-DR2 null haplotype.' },
  { gene: 'C4A',  group: 'Classical Pathway Amplifier',  sle: '✓✓',  bacterial: '✓✓',  mpgn: '±',   neisseria: '±',   eculizumab: '✗',   vaccine2: '✓✓',  lupus_rash: '±',   hcq: '✓✓',  unique: 'C4A NULL most common MHC null allele in Europeans (~15% carrier). PARTIAL DEFICIENCY 2-3× SLE risk. Anti-dsDNA may be POSITIVE (unlike C1q — key DDx). CNV analysis required. Complete deficiency: SLE + encapsulated bacteria.' },
  { gene: 'C3',   group: 'Central Complement',           sle: '✓✓',  bacterial: '✓✓✓', mpgn: '✓✓✓', neisseria: '✓✓',  eculizumab: '✓✓',  vaccine2: '✓✓✓', lupus_rash: '±',   hcq: '±',   unique: 'MPGN PATHOGNOMONIC. MOST SEVERE encapsulated bacteria (ALL THREE PATHWAYS BLOCKED). CH50 + AP50 BOTH ZERO. Eculizumab emerging for MPGN. ALL vaccines MANDATORY (PCV13+PPSV23+MenACWY+MenB+Hib). Most severe complement deficiency phenotype.' },
  { gene: 'CFB',  group: 'Alternative Pathway (AP)',     sle: '±',   bacterial: '✓✓',  mpgn: '✓✓✓', neisseria: '✓✓✓', eculizumab: '✓✓✓', vaccine2: '✓✓✓', lupus_rash: '✗',   hcq: '±',   unique: 'MPGN TYPE 2 DENSE DEPOSIT DISEASE characteristic. RECURRENT NEISSERIA MENINGITIDIS. AP50 = ZERO + CH50 = NORMAL distinguishes from classical pathway deficiency. Eculizumab for Dense Deposit Disease.' },
  { gene: 'CFD',  group: 'Alternative Pathway (RAREST)', sle: '±',   bacterial: '✓✓',  mpgn: '✓',   neisseria: '✓✓✓', eculizumab: '±',   vaccine2: '✓✓✓', lupus_rash: '✗',   hcq: '✗',   unique: 'RAREST complement deficiency (<15 families worldwide). NORTH AFRICA FOUNDER EFFECT. RECURRENT NEISSERIA MENINGITIDIS primary. 2× VACCINATION SCHEDULE MANDATORY (MenACWY + MenB dual-dose). AP50 = ZERO; penicillin V prophylaxis.' },
  { gene: 'C5',   group: 'Terminal Pathway Entry',       sle: '±',   bacterial: '✓✓',  mpgn: '±',   neisseria: '✓✓✓', eculizumab: '✓✓✓', vaccine2: '✓✓✓', lupus_rash: '±',   hcq: '±',   unique: 'NEISSERIA MENINGITIDIS PATHOGNOMONIC for terminal complement deficiency. ECULIZUMAB (anti-C5; FDA 2007) CURATIVE. RAVULIZUMAB monthly alternative. CRITICAL: vaccinate with MenACWY + MenB ≥2 weeks BEFORE starting eculizumab.' },
  { gene: 'CFP',  group: 'Properdin (Only XLR)',         sle: '✗',   bacterial: '✓✓',  mpgn: '±',   neisseria: '✓✓✓', eculizumab: '±',   vaccine2: '✓✓✓', lupus_rash: '✗',   hcq: '✗',   unique: 'ONLY XLR COMPLEMENT DEFICIENCY. MALES ONLY. FULMINANT MENINGOCOCCEMIA 30% MORTALITY = most rapidly fatal. PATHOGNOMONIC Neisseria in boys/young men. 2× VACCINATION MANDATORY. FAMILY SCREENING MANDATORY (maternal male relatives).' },
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
  const isC1QA  = gene === 'C1QA';
  const isC2    = gene === 'C2';
  const isC4A   = gene === 'C4A';
  const isC3    = gene === 'C3';
  const isCFB   = gene === 'CFB';
  const isCFD   = gene === 'CFD';
  const isC5    = gene === 'C5';
  const isCFP   = gene === 'CFP';
  const groupLabel = isC1QA  ? 'Classical Pathway (C1q/SLE-90%)'
    : isC2    ? 'Classical Pathway (Most Common)'
    : isC4A   ? 'Classical Pathway Amplifier (C4A-NULL)'
    : isC3    ? 'Central Complement (All Pathways)'
    : isCFB   ? 'Alternative Pathway (Dense Deposit)'
    : isCFD   ? 'Alternative Pathway (RAREST/N.Africa)'
    : isC5    ? 'Terminal Pathway Entry (Eculizumab)'
    : 'Properdin (ONLY XLR Complement)';
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
          <Badge text={`SLE ${cohort.sle_pct}%`}              color={color} />
          {cohort.bacterial_pct > 0     && <Badge text={`Bacterial ${cohort.bacterial_pct}%`}       color='#4a148c' />}
          {cohort.mpgn_pct > 0          && <Badge text={`MPGN ${cohort.mpgn_pct}%`}                color='#01579b' />}
          {cohort.neisseria_pct > 0     && <Badge text={`Neisseria ${cohort.neisseria_pct}%`}       color='#b71c1c' />}
          {cohort.eculizumab_pct > 0    && <Badge text={`Eculizumab ${cohort.eculizumab_pct}%`}    color='#1b5e20' />}
          {cohort.vaccine_2dose_pct > 0 && <Badge text={`Vaccine2x ${cohort.vaccine_2dose_pct}%`}  color='#006064' />}
          {cohort.lupus_rash_pct > 0    && <Badge text={`LupusRash ${cohort.lupus_rash_pct}%`}     color='#880e4f' />}
          {cohort.hcq_pct > 0           && <Badge text={`HCQ ${cohort.hcq_pct}%`}                  color='#311b92' />}
          <Badge text={`Onset ${cohort.mean_onset_months}mo`} color='#455a64' />
        </div>
      )}
    </div>
  );
}

export default function HereditaryComplementDeficiencyAtlasPage() {
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

  if (loading) return <div style={{ color: '#fff', padding: 32 }}>Loading Hereditary-Complement-Deficiency-Atlas…</div>;
  if (error)   return <div style={{ color: '#f44', padding: 32 }}>Error: {error}</div>;

  const stats        = overview?.aggregate_stats || {};
  const cohortByGene = {};
  (breakdown?.gene_cohorts || []).forEach(c => { cohortByGene[c.gene] = c; });

  return (
    <div style={{ background: '#0a0a1a', minHeight: '100vh', color: '#eee', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: '#0d0a2a', borderBottom: '2px solid #880e4f', padding: '20px 32px' }}>
        <h1 style={{ color: '#f48fb1', fontSize: 22, marginBottom: 4 }}>
          🧬 Hereditary-Complement-Deficiency-Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#aaa' }}>
          Complete 8-Gene Complement System Deficiency Reference · C1QA · C2 · C4A · C3 · CFB · CFD · C5 · CFP ·
          320-Patient Aggregate (8×40) · Seeds 3982-3989
        </div>
        <div style={{ marginTop: 8, display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 12 }}>
          <span style={{ background: '#880e4f', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>SLE {stats.sle_pct}%</span>
          <span style={{ background: '#4a148c', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Bacterial {stats.bacterial_pct}%</span>
          <span style={{ background: '#01579b', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>MPGN {stats.mpgn_pct}%</span>
          <span style={{ background: '#b71c1c', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Neisseria {stats.neisseria_pct}%</span>
          <span style={{ background: '#1b5e20', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Eculizumab {stats.eculizumab_pct}%</span>
          <span style={{ background: '#006064', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>Vaccine2x {stats.vaccine_2dose_pct}%</span>
          <span style={{ background: '#311b92', color: '#fff', padding: '2px 8px', borderRadius: 4 }}>HCQ {stats.hcq_pct}%</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid #333', background: '#0d0a2a' }}>
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
            <div style={{ background: '#0d0a2a', border: '1px solid #880e4f', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#f48fb1', marginBottom: 10 }}>Key Clinical Facts — 8 Complement System Deficiency Diseases</div>
              {(overview?.key_facts || []).map((f, i) => (
                <div key={i} style={{ fontSize: 12, color: '#ddd', marginBottom: 6, paddingLeft: 12, borderLeft: '3px solid #880e4f' }}>{f}</div>
              ))}
            </div>

            {/* Disease Groups by Pathway */}
            <div style={{ background: '#0a1628', border: '1px solid #01579b', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 12 }}>Disease Groups by Complement Pathway</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(270px, 1fr))', gap: 12 }}>
                {[
                  { label: 'Classical Pathway (C1q)',        genes: 'C1QA',          desc: 'AR; 90% SLE = HIGHEST autoimmune risk; lupus rash PATHOGNOMONIC; anti-dsDNA NEGATIVE DDx; impaired apoptotic clearance.', color: '#880e4f' },
                  { label: 'Classical Pathway (Most Common)',genes: 'C2',             desc: 'Most common Western 1:20,000; encapsulated bacteria primary risk; SLE 30%; pneumococcal vaccines + penicillin prophylaxis MANDATORY.', color: '#4a148c' },
                  { label: 'Classical Amplifier (C4A-NULL)', genes: 'C4A',            desc: 'C4A NULL most common MHC null allele in Europeans (~15% carrier); partial deficiency 2-3× SLE risk; CNV analysis required.', color: '#311b92' },
                  { label: 'Central Complement (All Paths)', genes: 'C3',             desc: 'All three pathways blocked; most severe encapsulated bacteria; MPGN pathognomonic; eculizumab emerging; ALL vaccines mandatory.', color: '#1a237e' },
                  { label: 'Alternative Pathway (AP)',        genes: 'CFB / CFD',      desc: 'AP50 = ZERO + CH50 = NORMAL distinguishes; Neisseria primary; Dense Deposit Disease; CFD rarest (<15 families); North Africa founder.', color: '#01579b' },
                  { label: 'Terminal Pathway (Eculizumab)',   genes: 'C5',             desc: 'MAC absent → Neisseria PATHOGNOMONIC; eculizumab (anti-C5; FDA 2007) curative; vaccinate BEFORE starting eculizumab.', color: '#1b5e20' },
                  { label: 'Properdin (ONLY XLR)',            genes: 'CFP',            desc: 'ONLY XLR complement deficiency; males only; fulminant meningococcemia 30% mortality; 2× vaccination + family screening MANDATORY.', color: '#b71c1c' },
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
            <div style={{ marginTop: 20, background: '#0d0a2a', border: '1px solid #333', borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: '#f48fb1', marginBottom: 10 }}>Aggregate Statistics — 320 Patients (8×40)</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))', gap: 8 }}>
                {[
                  ['SLE',            stats.sle_pct,           '%', '#880e4f'],
                  ['Bacterial',      stats.bacterial_pct,     '%', '#4a148c'],
                  ['MPGN',           stats.mpgn_pct,          '%', '#01579b'],
                  ['Neisseria',      stats.neisseria_pct,     '%', '#b71c1c'],
                  ['Eculizumab',     stats.eculizumab_pct,    '%', '#1b5e20'],
                  ['Vaccine 2×',     stats.vaccine_2dose_pct, '%', '#006064'],
                  ['Lupus Rash',     stats.lupus_rash_pct,    '%', '#880e4f'],
                  ['HCQ',            stats.hcq_pct,           '%', '#311b92'],
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
                <tr style={{ background: '#0d0a2a' }}>
                  {['Gene', 'Group', 'SLE', 'Bacterial', 'MPGN', 'Neisseria', 'Eculizumab', 'Vaccine 2×', 'Lupus Rash', 'HCQ', 'Clinical Pearl'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', color: '#f48fb1', textAlign: 'left', borderBottom: '1px solid #444' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#161616' : '#0d0a2a' }}>
                    <td style={{ padding: '8px 10px', color: GENE_COLORS[row.gene], fontWeight: 700 }}>{row.gene}</td>
                    <td style={{ padding: '8px 10px', color: '#aaa', fontSize: 10 }}>{row.group}</td>
                    <td style={{ padding: '8px 10px', color: row.sle        === '✓✓✓' ? '#f48fb1' : '#888' }}>{row.sle}</td>
                    <td style={{ padding: '8px 10px', color: row.bacterial  === '✓✓✓' ? '#ce93d8' : '#888' }}>{row.bacterial}</td>
                    <td style={{ padding: '8px 10px', color: row.mpgn       === '✓✓✓' ? '#90caf9' : '#888' }}>{row.mpgn}</td>
                    <td style={{ padding: '8px 10px', color: row.neisseria  === '✓✓✓' ? '#ef9a9a' : '#888' }}>{row.neisseria}</td>
                    <td style={{ padding: '8px 10px', color: row.eculizumab === '✓✓✓' ? '#a5d6a7' : '#888' }}>{row.eculizumab}</td>
                    <td style={{ padding: '8px 10px', color: row.vaccine2   === '✓✓✓' ? '#80deea' : '#888' }}>{row.vaccine2}</td>
                    <td style={{ padding: '8px 10px', color: row.lupus_rash === '✓✓✓' ? '#f48fb1' : '#888' }}>{row.lupus_rash}</td>
                    <td style={{ padding: '8px 10px', color: row.hcq        === '✓✓✓' ? '#b39ddb' : '#888' }}>{row.hcq}</td>
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
            <div style={{ background: '#0a1628', border: '1px solid #01579b', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 10 }}>Cascade Diagnostic Testing — Complement Deficiency Workup</div>
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
            <div style={{ background: '#0d0a2a', border: '1px solid #880e4f', borderRadius: 8, padding: 16 }}>
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
                <div key={d.term} style={{ background: '#0d0a2a', border: '1px solid #333', borderRadius: 8, padding: 14 }}>
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
