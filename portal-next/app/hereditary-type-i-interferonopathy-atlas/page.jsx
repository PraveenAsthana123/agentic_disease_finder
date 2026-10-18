'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-type-i-interferonopathy-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'TREX1':    '#b71c1c',  // deep red       -- AGS1; chilblains PATHOGNOMONIC; most common; cGAS-STING
  'RNASEH2B': '#1565c0',  // deep blue      -- AGS2; mildest; ambulatory; misdiagnosed as CP
  'RNASEH2A': '#0d47a1',  // navy blue      -- AGS4; catalytic; most severe RNASEH2; early calcifications
  'RNASEH2C': '#283593',  // indigo         -- AGS3; Cree founder p.A177T; PCNA-coupling subunit
  'SAMHD1':   '#e65100',  // burnt orange   -- AGS5; SMS1; dental anomalies + aortic calcification PATHOGNOMONIC
  'ADAR':     '#4a148c',  // deep purple    -- AGS6; bilateral striatal necrosis PATHOGNOMONIC; A-to-I editing
  'IFIH1':    '#1b5e20',  // deep green     -- AGS7; AD-GOF MDA5; SMS2; psoriasiform + dental + glaucoma
  'USP18':    '#212121',  // near-black     -- pseudo-TORCH neonatal; fatal; JAKi LIFE-SAVING; IFN score 100x
};

const GENE_INFO = {
  'TREX1':    { full: 'TREX1-314aa / AR+AD / ssDNA-3prime-exonuclease / cGAS-STING / AGS1-Most-Common-25-35pct / Chilblains-PATHOGNOMONIC / AD-monoallelic-SLE-no-brain',          locus: '3p21.31',  size: '314 aa / 36 kDa', inh: 'AR (AGS1) / AD (SLE)' },
  'RNASEH2B': { full: 'RNASEH2B-312aa / AR / RNase-H2-Scaffold-Subunit / AGS2-Mildest / Ambulatory-Majority-UNIQUE / White-Matter-Not-Calcification / Misdiagnosed-CP',              locus: '13q14.3',  size: '312 aa / 33 kDa', inh: 'Autosomal recessive' },
  'RNASEH2A': { full: 'RNASEH2A-299aa / AR / RNase-H2-Catalytic-Subunit / AGS4-Most-Severe-RNASEH2 / Early-Calcifications / Ambulation-Lost / p.G37S-Hypomorph-Milder',             locus: '19p13.2',  size: '299 aa / 33 kDa', inh: 'Autosomal recessive' },
  'RNASEH2C': { full: 'RNASEH2C-164aa / AR / RNase-H2-PCNA-Coupling / AGS3-Intermediate / Cree-Founder-p.A177T-French-Canadian-Inuit / Smallest-Subunit-164aa',                    locus: '11q13.1',  size: '164 aa / 17 kDa', inh: 'Autosomal recessive' },
  'SAMHD1':   { full: 'SAMHD1-626aa / AR / dNTPase+ssDNA-Exonuclease / AGS5-SMS1 / Dental-Anomalies-PATHOGNOMONIC / Aortic-Calcification / Glaucoma / HIV-Restriction-VPHU',       locus: '20q11.23', size: '626 aa / 72 kDa', inh: 'Autosomal recessive' },
  'ADAR':     { full: 'ADAR-1226aa / AR-biallelic-AGS6 / AD-monoallelic-DSH / A-to-I-dsRNA-Editing / MDA5-Suppressor / Bilateral-Striatal-Necrosis-PATHOGNOMONIC / Baricitinib',   locus: '1q21.3',   size: '1226 aa / 136 kDa', inh: 'AR (AGS6) / AD (DSH)' },
  'IFIH1':    { full: 'IFIH1-1025aa / AD-GOF / MDA5-Helicase / AGS7-SMS2 / Psoriasiform-Dental-Glaucoma-JointLaxity-Quad / Single-Allele-Sufficient / JAKi-Effective',             locus: '2q24.2',   size: '1025 aa / 116 kDa', inh: 'Autosomal dominant GOF' },
  'USP18':    { full: 'USP18-371aa / AR / IFN-I-Negative-Regulator / ISG15-Deconjugase / Pseudo-TORCH-Neonatal / Fatal-Without-Ruxolitinib / IFN-Score-100x-Normal',               locus: '22q11.21', size: '371 aa / 43 kDa', inh: 'Autosomal recessive' },
};

const PHENOTYPE_ROWS = [
  { gene: 'TREX1',    group: 'cGAS-STING',       calc: '✓✓✓', csf: '✓✓✓', wm: '✓✓✓', chlbns: '✓✓✓', seiz: '✓✓',  regr: '✓✓✓', isg: '✓✓✓', jaki: '✓✓',  unique: 'CHILBLAINS (PERNIO) PATHOGNOMONIC: cold-triggered violaceous acral lesions in calcifying encephalopathy + IFN score = TREX1; AD monoallelic = SLE only (no brain)' },
  { gene: 'RNASEH2B', group: 'RNase-H2',         calc: '✓✓',  csf: '✓✓',  wm: '✓✓✓', chlbns: '±',   seiz: '✓',   regr: '✓✓',  isg: '✓✓✓', jaki: '✓✓',  unique: 'MILDEST AGS: AMBULATORY MAJORITY UNIQUE among AGS genes; white matter > calcification pattern; most misdiagnosed as cerebral palsy; RNASEH2B only AGS gene where patients preserve walking' },
  { gene: 'RNASEH2A', group: 'RNase-H2',         calc: '✓✓✓', csf: '✓✓✓', wm: '✓✓✓', chlbns: '±',   seiz: '✓✓✓', regr: '✓✓✓', isg: '✓✓✓', jaki: '✓✓',  unique: 'CATALYTIC SUBUNIT: most severe RNASEH2 subunit; complete RNase H2 loss; early extensive calcifications; loss of ambulation common; p.G37S hypomorph = retained partial activity → milder' },
  { gene: 'RNASEH2C', group: 'RNase-H2',         calc: '✓✓',  csf: '✓✓',  wm: '✓✓',  chlbns: '±',   seiz: '✓✓',  regr: '✓✓',  isg: '✓✓✓', jaki: '✓✓',  unique: 'CREE FOUNDER p.A177T: screen FIRST in French-Canadian/Inuit/Cree ancestry before full panel; RNASEH2C couples RNase H2 to PCNA at replication fork; smallest subunit (164aa)' },
  { gene: 'SAMHD1',   group: 'dNTPase',          calc: '✓✓',  csf: '✓✓',  wm: '✓✓',  chlbns: '✓',   seiz: '✓✓',  regr: '✓✓',  isg: '✓✓✓', jaki: '✓✓',  unique: 'SINGLETON-MERTEN TYPE 1: DENTAL ANOMALIES + AORTIC CALCIFICATION PATHOGNOMONIC; panoramic dental X-ray + CT aorta MANDATORY; annual glaucoma; HIV restriction factor (VPHU degrades SAMHD1)' },
  { gene: 'ADAR',     group: 'dsRNA-Editing',    calc: '✓✓✓', csf: '✓✓',  wm: '✓✓✓', chlbns: '±',   seiz: '✓✓✓', regr: '✓✓✓', isg: '✓✓✓', jaki: '✓✓✓', unique: 'BILATERAL STRIATAL NECROSIS PATHOGNOMONIC: putamen/caudate T2-hyperintensity = ADAR; acute paroxysmal dystonia with fever; each crisis → more striatal loss; baricitinib most evidence; monoallelic = DSH (skin only)' },
  { gene: 'IFIH1',    group: 'dsRNA-Sensor-GOF', calc: '✓',   csf: '±',   wm: '✓',   chlbns: '±',   seiz: '±',   regr: '±',   isg: '✓✓✓', jaki: '✓✓',  unique: 'SINGLETON-MERTEN TYPE 2: AD-GOF MDA5; PSORIASIFORM + DENTAL + GLAUCOMA + JOINT LAXITY QUADRAD pathognomonic; mildest neurological of all AGS; primary referral often dermatology; single allele sufficient' },
  { gene: 'USP18',    group: 'IFN-Feedback-LOF', calc: '✓✓✓', csf: '✓✓✓', wm: '✓✓✓', chlbns: '±',   seiz: '✓✓✓', regr: '✓✓✓', isg: '✓✓✓', jaki: '✓✓✓', unique: 'PSEUDO-TORCH NEONATAL — FATAL WITHOUT JAKi: TORCH-negative neonate + IFN score 100x normal = USP18 EMERGENCY; ruxolitinib IMMEDIATELY; ISGylation uncontrolled self-amplification; every day delay = irreversible' },
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
  const groupLabel = gene === 'TREX1'    ? 'cGAS-STING (ssDNA Exonuclease)'
                   : gene === 'RNASEH2B' ? 'RNase H2 Scaffold (Mild)'
                   : gene === 'RNASEH2A' ? 'RNase H2 Catalytic (Severe)'
                   : gene === 'RNASEH2C' ? 'RNase H2 PCNA-Coupling (Cree)'
                   : gene === 'SAMHD1'   ? 'dNTPase / SMS1'
                   : gene === 'ADAR'     ? 'dsRNA A-to-I Editing / AGS6'
                   : gene === 'IFIH1'    ? 'MDA5 GOF / SMS2 / AGS7'
                   : 'IFN-I Feedback LOF / Pseudo-TORCH';
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
          <Badge text={`Calcifications ${cohort.calcifications_pct}%`}     color={color} />
          <Badge text={`CSF Lymphocytosis ${cohort.csf_lymphocytosis_pct}%`} color='#880e4f' />
          <Badge text={`White Matter ${cohort.white_matter_pct}%`}          color='#4e342e' />
          <Badge text={`Chilblains ${cohort.chilblains_pct}%`}              color='#e65100' />
          <Badge text={`Seizures ${cohort.seizures_pct}%`}                  color='#795548' />
          <Badge text={`Regression ${cohort.regression_pct}%`}              color='#4a148c' />
          <Badge text={`ISG Score+ ${cohort.isg_score_elevated_pct}%`}      color='#1b5e20' />
          <Badge text={`JAKi Rx ${cohort.jaki_tx_pct}%`}                   color='#1565c0' />
          <Badge text={`n=${cohort.n}`}                                      color='#555' />
        </div>
      )}
      {cohort && (
        <div style={{ marginTop: 10, fontSize: 12, background: '#1a1a2e', padding: 10, borderRadius: 6, color: '#e8d5b7', borderLeft: `3px solid ${color}` }}>
          <strong style={{ color }}>⚡ PATHOGNOMONIC / KEY:</strong> {cohort.pathognomonic}
        </div>
      )}
    </div>
  );
}

export default function HereditaryTypeIInterferonopathyAtlasPage() {
  const [tab, setTab]         = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefs(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  return (
    <div style={{ background: '#0a0a0f', minHeight: '100vh', color: '#e0e0e0', fontFamily: 'monospace', padding: 20 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ borderBottom: '2px solid #b71c1c', paddingBottom: 16, marginBottom: 20 }}>
          <h1 style={{ color: '#ff8a80', fontSize: 22, margin: 0 }}>
            🧬 Hereditary-Type-I-Interferonopathy-Atlas
          </h1>
          <div style={{ color: '#aaa', fontSize: 12, marginTop: 6 }}>
            Aicardi-Goutières Spectrum / Nucleic Acid Sensing Defects —
            TREX1 · RNASEH2B · RNASEH2A · RNASEH2C · SAMHD1 · ADAR · IFIH1 · USP18
          </div>
          <div style={{ color: '#666', fontSize: 11, marginTop: 4 }}>
            320 Patients · 8 Genes · Seeds 3926–3933 · IUIS 2022 Category 9: Type I Interferonopathy
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: '6px 16px', borderRadius: 4, border: 'none', cursor: 'pointer', fontSize: 13,
              background: tab === t ? '#b71c1c' : '#1a1a2e',
              color: tab === t ? '#fff' : '#aaa',
              fontWeight: tab === t ? 700 : 400,
            }}>{t}</button>
          ))}
        </div>

        {loading && <div style={{ color: '#aaa', padding: 40, textAlign: 'center' }}>Loading atlas data…</div>}
        {error   && <div style={{ color: '#f44', padding: 20 }}>Error: {error}</div>}

        {/* Overview Tab */}
        {!loading && !error && tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 24 }}>
              {[
                { label: 'Total Patients',       value: overview.total_patients },
                { label: 'Calcifications',        value: `${overview.aggregate_stats?.calcifications_pct}%` },
                { label: 'CSF Lymphocytosis',     value: `${overview.aggregate_stats?.csf_lymphocytosis_pct}%` },
                { label: 'White Matter Abn.',     value: `${overview.aggregate_stats?.white_matter_pct}%` },
                { label: 'Chilblains',            value: `${overview.aggregate_stats?.chilblains_pct}%` },
                { label: 'Seizures',              value: `${overview.aggregate_stats?.seizures_pct}%` },
                { label: 'Dev. Regression',       value: `${overview.aggregate_stats?.regression_pct}%` },
                { label: 'ISG Score Elevated',    value: `${overview.aggregate_stats?.isg_score_elevated_pct}%` },
                { label: 'JAKi Treated',          value: `${overview.aggregate_stats?.jaki_tx_pct}%` },
              ].map(({ label, value }) => (
                <div key={label} style={{ background: '#12121e', borderRadius: 6, padding: 12, textAlign: 'center', border: '1px solid #222' }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color: '#ff8a80' }}>{value}</div>
                  <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>{label}</div>
                </div>
              ))}
            </div>

            <div style={{ marginBottom: 24 }}>
              <h3 style={{ color: '#ff8a80', marginBottom: 12 }}>Gene Cohorts (n=40 each)</h3>
              {GENES.map(gene => {
                const cohort = overview.cohort_summaries?.find(c => c.gene === gene);
                return <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={GENE_INFO[gene]} cohort={cohort} />;
              })}
            </div>

            {overview.cascade_testing_concepts && (
              <div style={{ background: '#12121e', borderRadius: 8, padding: 16, border: '1px solid #333' }}>
                <h3 style={{ color: '#ff8a80', marginBottom: 12 }}>Cascade Testing Concepts</h3>
                {overview.cascade_testing_concepts.map((c, i) => (
                  <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 8, fontSize: 12 }}>
                    <span style={{ color: '#b71c1c', minWidth: 20, fontWeight: 700 }}>{i + 1}.</span>
                    <span style={{ color: '#ccc', lineHeight: 1.5 }}>{c}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Gene Table Tab */}
        {!loading && !error && tab === 'Gene Table' && breakdown && (
          <div>
            <div style={{ overflowX: 'auto', marginBottom: 24 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1a1a2e' }}>
                    {['Gene', 'AGS/Syndrome', 'Locus', 'aa/kDa', 'Inh', 'Onset (mo)', 'Dx Lag (yr)', 'Calc%', 'CSF%', 'WM%', 'Chilbls%', 'Seiz%', 'ISG+%', 'JAKi%'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#aaa', borderBottom: '1px solid #333', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.gene_cohorts || []).map((c, i) => (
                    <tr key={c.gene} style={{ background: i % 2 === 0 ? '#0d0d17' : '#111120', borderBottom: '1px solid #222' }}>
                      <td style={{ padding: '8px 10px', color: GENE_COLORS[c.gene], fontWeight: 700 }}>{c.gene}</td>
                      <td style={{ padding: '8px 10px', color: '#ccc', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.syndrome}</td>
                      <td style={{ padding: '8px 10px', color: '#aaa' }}>{c.locus}</td>
                      <td style={{ padding: '8px 10px', color: '#aaa', whiteSpace: 'nowrap' }}>{c.aa}aa/{c.kda}kDa</td>
                      <td style={{ padding: '8px 10px', color: '#aaa', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.inheritance}</td>
                      <td style={{ padding: '8px 10px', color: '#e8d5b7', textAlign: 'center' }}>{c.mean_onset_months}</td>
                      <td style={{ padding: '8px 10px', color: '#e8d5b7', textAlign: 'center' }}>{c.mean_dx_lag_years}</td>
                      <td style={{ padding: '8px 10px', color: '#ff8a80', textAlign: 'center', fontWeight: 700 }}>{c.calcifications_pct}%</td>
                      <td style={{ padding: '8px 10px', color: '#aaa', textAlign: 'center' }}>{c.csf_lymphocytosis_pct}%</td>
                      <td style={{ padding: '8px 10px', color: '#aaa', textAlign: 'center' }}>{c.white_matter_pct}%</td>
                      <td style={{ padding: '8px 10px', color: '#e65100', textAlign: 'center', fontWeight: c.chilblains_pct > 30 ? 700 : 400 }}>{c.chilblains_pct}%</td>
                      <td style={{ padding: '8px 10px', color: '#aaa', textAlign: 'center' }}>{c.seizures_pct}%</td>
                      <td style={{ padding: '8px 10px', color: '#1b5e20', textAlign: 'center', fontWeight: 700 }}>{c.isg_score_elevated_pct}%</td>
                      <td style={{ padding: '8px 10px', color: '#1565c0', textAlign: 'center' }}>{c.jaki_tx_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {breakdown.pathognomonic_features && (
              <div style={{ background: '#12121e', borderRadius: 8, padding: 16, border: '1px solid #333', marginBottom: 16 }}>
                <h3 style={{ color: '#ff8a80', marginBottom: 12 }}>Pathognomonic Features per Gene</h3>
                {Object.entries(breakdown.pathognomonic_features).map(([k, v]) => (
                  <div key={k} style={{ marginBottom: 10, fontSize: 12 }}>
                    <span style={{ color: '#ff8a80', fontWeight: 700 }}>{k.replace(/_/g, ' ')}: </span>
                    <span style={{ color: '#ccc' }}>{v}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Pathway Map Tab */}
        {!loading && !error && tab === 'Pathway Map' && breakdown && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <h3 style={{ color: '#ff8a80', marginBottom: 12 }}>IFN-I Pathway — Gene Groupings by Mechanism</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 12 }}>
                {Object.entries(breakdown.diagnostic_groups || {}).map(([gid, grp]) => (
                  <div key={gid} style={{ background: '#12121e', borderRadius: 8, padding: 14, border: '1px solid #333' }}>
                    <div style={{ fontWeight: 700, color: '#ff8a80', marginBottom: 6, textTransform: 'uppercase', fontSize: 11, letterSpacing: 1 }}>
                      {gid.replace(/_/g, ' ')}
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
                      {grp.genes.map(g => (
                        <span key={g} style={{ background: GENE_COLORS[g], color: '#fff', padding: '2px 8px', borderRadius: 3, fontSize: 12, fontWeight: 700 }}>{g}</span>
                      ))}
                    </div>
                    <div style={{ fontSize: 11, color: '#aaa', marginBottom: 6 }}><strong style={{ color: '#ccc' }}>Mechanism:</strong> {grp.mechanism}</div>
                    <div style={{ fontSize: 11, color: '#aaa', marginBottom: 6 }}><strong style={{ color: '#ccc' }}>Shared:</strong> {grp.shared_feature}</div>
                    <div style={{ fontSize: 11, color: '#888' }}><strong style={{ color: '#ccc' }}>DDx:</strong> {grp.key_ddx}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* IFN pathway diagram */}
            <div style={{ background: '#12121e', borderRadius: 8, padding: 16, border: '1px solid #333' }}>
              <h3 style={{ color: '#ff8a80', marginBottom: 12 }}>IFN-I Signalling Pathway (Simplified)</h3>
              <div style={{ fontFamily: 'monospace', fontSize: 11, color: '#ccc', lineHeight: 2 }}>
                <div>TREX1/SAMHD1/RNASEH2 LOF → Cytosolic ssDNA/R-loops accumulate</div>
                <div style={{ color: '#666' }}>                     ↓</div>
                <div>                         <span style={{ color: '#ff8a80' }}>cGAS senses ssDNA/dsDNA</span> → cGAMP synthesis</div>
                <div style={{ color: '#666' }}>                     ↓</div>
                <div>                         <span style={{ color: '#e65100' }}>STING</span> → TBK1 → IRF3/IRF7 phosphorylation</div>
                <div style={{ color: '#666' }}>                     ↓</div>
                <div>ADAR LOF / IFIH1 GOF → <span style={{ color: '#4a148c' }}>MDA5 detects cytosolic dsRNA</span> → MAVS → IFN-β</div>
                <div style={{ color: '#666' }}>                     ↓</div>
                <div>                     <span style={{ color: '#1b5e20' }}>IFN-α/β secreted</span> → IFNAR receptor on immune cells</div>
                <div style={{ color: '#666' }}>                     ↓</div>
                <div>                     JAK1/TYK2 → STAT1/STAT2 → ISG transcription (ISG score ↑)</div>
                <div style={{ color: '#666' }}>                     ↓</div>
                <div>USP18 (negative feedback) — LOF → <span style={{ color: '#212121', background: '#ff8a80', padding: '0 4px' }}>UNCONTROLLED IFN-I → pseudo-TORCH FATAL</span></div>
                <div style={{ color: '#666' }}>                     ↓</div>
                <div>                     <span style={{ color: '#1565c0' }}>JAKi (ruxolitinib/baricitinib)</span> ← THERAPEUTIC INTERVENTION</div>
              </div>
            </div>
          </div>
        )}

        {/* Definitions Tab */}
        {!loading && !error && tab === 'Definitions' && defs && (
          <div>
            <div style={{ marginBottom: 24 }}>
              {(defs.definitions || []).map((d, i) => (
                <div key={i} style={{ background: '#12121e', borderRadius: 6, padding: 14, marginBottom: 10, border: '1px solid #2a2a40' }}>
                  <div style={{ fontWeight: 700, color: '#ff8a80', marginBottom: 6, fontSize: 13 }}>{d.term}</div>
                  <div style={{ color: '#ccc', fontSize: 12, lineHeight: 1.6 }}>{d.definition}</div>
                </div>
              ))}
            </div>

            {defs.key_distinguishers && (
              <div style={{ background: '#12121e', borderRadius: 8, padding: 16, border: '1px solid #333', marginBottom: 16 }}>
                <h3 style={{ color: '#ff8a80', marginBottom: 12 }}>Key Differential Distinguishers</h3>
                {Object.entries(defs.key_distinguishers).map(([k, v]) => (
                  <div key={k} style={{ marginBottom: 12, fontSize: 12, borderLeft: '3px solid #b71c1c', paddingLeft: 10 }}>
                    <div style={{ color: '#ff8a80', fontWeight: 700, marginBottom: 4 }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ color: '#ccc', lineHeight: 1.5 }}>{v}</div>
                  </div>
                ))}
              </div>
            )}

            {defs.diagnostic_pathway && (
              <div style={{ background: '#12121e', borderRadius: 8, padding: 16, border: '1px solid #333' }}>
                <h3 style={{ color: '#ff8a80', marginBottom: 12 }}>Diagnostic Pathway (8 Steps)</h3>
                {defs.diagnostic_pathway.map((step, i) => (
                  <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 10, fontSize: 12 }}>
                    <span style={{ color: '#b71c1c', fontWeight: 700, minWidth: 58, flexShrink: 0 }}>Step {i + 1}:</span>
                    <span style={{ color: '#ccc', lineHeight: 1.6 }}>{step.replace(/^Step \d+: /, '')}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const GENES = ["TREX1", "RNASEH2B", "RNASEH2A", "RNASEH2C", "SAMHD1", "ADAR", "IFIH1", "USP18"];
