'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-mhc-bare-lymphocyte-syndrome-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'CIITA':   '#1a237e',  // deep indigo      -- BLS-IIA; master MHC-II transactivator; ~20-25% of BLS-II
  'RFXANK':  '#0d47a1',  // dark blue         -- BLS-IIB; Moroccan founder del; MOST COMMON BLS-II
  'RFX5':    '#1565c0',  // medium blue       -- BLS-IIC; DNA-binding subunit; pan-ethnic
  'RFXAP':   '#1976d2',  // blue              -- BLS-IID; rarest BLS-II (<5%)
  'TAP1':    '#b71c1c',  // deep red          -- BLS-I; skin granulomas PATHOGNOMONIC; 6p21.32
  'TAP2':    '#c62828',  // red               -- BLS-I most common; bronchiectasis + sinopulmonary
  'TAPBP':   '#e53935',  // lighter red       -- BLS-I tapasin; rarest + mildest; 6p21.32
  'ZAP70':   '#1b5e20',  // deep green        -- ZAP-70 CID; selective CD8+ absence PATHOGNOMONIC
};

const GENE_INFO = {
  'CIITA':   { full: 'CIITA-1130aa / AR / MHC-II-Master-Transactivator / BLS-IIA / 16p13.13 / CD4+Lymphopenia / HLA-DR-Absent / Cryptosporidium',                      locus: '16p13.13', size: '1130 aa / 130 kDa', inh: 'Autosomal recessive' },
  'RFXANK':  { full: 'RFXANK-260aa / AR / Ankyrin-Repeat / BLS-IIB-Most-Common / 19q13.11 / Moroccan-Founder-33nt-del / >80%-North-African',                           locus: '19q13.11',  size: '260 aa / 33 kDa',  inh: 'Autosomal recessive' },
  'RFX5':    { full: 'RFX5-616aa / AR / X1-Box-DNA-Binding / BLS-IIC / 1q21.3 / Pan-Ethnic / No-Founder-Mutation',                                                     locus: '1q21.3',    size: '616 aa / 67 kDa',  inh: 'Autosomal recessive' },
  'RFXAP':   { full: 'RFXAP-272aa / AR / Acidic-Repeat-Domain / BLS-IID-Rarest / 13q14.2 / <50-Patients-Worldwide / Individual-Kindreds',                              locus: '13q14.2',   size: '272 aa / 36 kDa',  inh: 'Autosomal recessive' },
  'TAP1':    { full: 'TAP1-811aa / AR / ABC-Transporter-ABCB2 / BLS-I / 6p21.32-MHC-Region / CD8+Low / HLA-ABC-Absent / Skin-Granulomas-PATHOGNOMONIC',                locus: '6p21.32',   size: '811 aa / 81 kDa',  inh: 'Autosomal recessive' },
  'TAP2':    { full: 'TAP2-703aa / AR / ABC-Transporter-ABCB3 / BLS-I-Most-Common-70pct / 6p21.32 / Bronchiectasis / Sinopulmonary-GramNeg',                           locus: '6p21.32',   size: '703 aa / 75 kDa',  inh: 'Autosomal recessive' },
  'TAPBP':   { full: 'TAPBP-428aa / AR / Tapasin-Peptide-Editor / BLS-I-Rarest-Mildest / 6p21.32 / Partial-MHC-I-Loading / Adult-Onset-Possible',                     locus: '6p21.32',   size: '428 aa / 48 kDa',  inh: 'Autosomal recessive' },
  'ZAP70':   { full: 'ZAP70-619aa / AR / Syk-Family-Tyrosine-Kinase / ZAP-70-CID / 2q11.2 / CD8-Absent-PATHOGNOMONIC / CD4-Present-Anergic / Mennonite-p.Cys319Arg',  locus: '2q11.2',    size: '619 aa / 70 kDa',  inh: 'Autosomal recessive' },
};

const PHENOTYPE_ROWS = [
  { gene: 'CIITA',   group: 'BLS-II (MHC-II)', cd4lp: '✓✓✓', hladr: '✓✓✓', crypto: '✓✓✓', pcp: '✓✓',  bronch: '✓✓',  hypogam: '✓✓✓', hsct: '✓✓✓', unique: 'BLS-IIA MASTER TRANSACTIVATOR: CD4+ <200/µL + absent HLA-DR on B cells = BLS-II; Cryptosporidium cholangiopathy → liver failure; HSCT URGENTLY before age 2' },
  { gene: 'RFXANK',  group: 'BLS-II (MHC-II)', cd4lp: '✓✓✓', hladr: '✓✓✓', crypto: '✓✓✓', pcp: '✓✓',  bronch: '✓✓',  hypogam: '✓✓✓', hsct: '✓✓✓', unique: 'MOROCCAN FOUNDER p.57_67del PATHOGNOMONIC: >80% North African/Arab alleles; targeted allele test BEFORE full sequencing; most common BLS-II worldwide (~50-60%)' },
  { gene: 'RFX5',    group: 'BLS-II (MHC-II)', cd4lp: '✓✓✓', hladr: '✓✓✓', crypto: '✓✓',  pcp: '✓✓',  bronch: '✓✓',  hypogam: '✓✓✓', hsct: '✓✓',  unique: 'BLS-IIC DNA-BINDING SUBUNIT: pan-ethnic; no founder mutation; compound heterozygotes; ~10-15% of BLS-II; NGS mandatory; phenotype identical to RFXANK/CIITA/RFXAP' },
  { gene: 'RFXAP',   group: 'BLS-II (MHC-II)', cd4lp: '✓✓✓', hladr: '✓✓✓', crypto: '✓✓',  pcp: '✓',   bronch: '✓✓',  hypogam: '✓✓',  hsct: '✓✓',  unique: 'RAREST BLS-II GROUP (<5%, <50 patients worldwide): RFXAP acidic domain; unique structure (no protein family homology); diagnose when RFXANK/RFX5 negative + absent HLA-DR confirmed' },
  { gene: 'TAP1',    group: 'BLS-I (MHC-I)',   cd4lp: '±',   hladr: '✗',   crypto: '✗',   pcp: '±',   bronch: '✓✓',  hypogam: '±',   hsct: '✓',   unique: 'NECROTISING GRANULOMATOUS SKIN LESIONS PATHOGNOMONIC: sterile deep ulcers (midface, extremities) + absent HLA-A/B/C = BLS-I; culture sterile; misdiagnosed as Wegener\'s/Behçet\'s; LATER onset than BLS-II' },
  { gene: 'TAP2',    group: 'BLS-I (MHC-I)',   cd4lp: '±',   hladr: '✗',   crypto: '✗',   pcp: '±',   bronch: '✓✓✓', hypogam: '±',   hsct: '✓',   unique: 'MOST COMMON BLS-I (~70% of BLS-I): TAP2 mutations span entire gene; bronchiectasis + Pseudomonas/atypical mycobacteria; HLA-A/B/C near-absent on flow; sinopulmonary gram-negative dominant' },
  { gene: 'TAPBP',   group: 'BLS-I (MHC-I)',   cd4lp: '±',   hladr: '✗',   crypto: '✗',   pcp: '±',   bronch: '✓',   hypogam: '±',   hsct: '±',   unique: 'RAREST AND MILDEST BLS-I: partial tapasin-independent MHC-I loading persists; HLA-A/B/C reduced (not absent); onset adolescence/adulthood; co-encoded 6p21.32; compound TAP1+TAPBP deletions possible' },
  { gene: 'ZAP70',   group: 'ZAP-70 CID',      cd4lp: '±',   hladr: '✗',   crypto: '±',   pcp: '✓✓',  bronch: '±',   hypogam: '✓',   hsct: '✓✓✓', unique: 'CD8+ ABSENT + CD4+ PRESENT (NON-FUNCTIONAL): ZAP-70 PATHOGNOMONIC — no other PID produces this; Mennonite/Amish p.Cys319Arg founder; autoimmune features; PCP prophylaxis mandatory' },
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
  const groupLabel = ['CIITA','RFXANK','RFX5','RFXAP'].includes(gene)
    ? 'BLS-II (MHC Class II Deficiency)'
    : ['TAP1','TAP2','TAPBP'].includes(gene)
    ? 'BLS-I (MHC Class I Deficiency)'
    : 'ZAP-70 CID (Selective CD8+ Absence)';
  const isBLS2 = ['CIITA','RFXANK','RFX5','RFXAP'].includes(gene);
  const isBLS1 = ['TAP1','TAP2','TAPBP'].includes(gene);
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
          {isBLS2 && <>
            <Badge text={`CD4+ Lymphopenia ${cohort.cd4_lymphopenia_pct}%`}    color={color} />
            <Badge text={`HLA-DR Absent ${cohort.hla_dr_absent_pct}%`}         color='#880e4f' />
            <Badge text={`Cryptosporidium ${cohort.cryptosporidium_pct}%`}     color='#4a148c' />
            <Badge text={`Pneumocystis ${cohort.pneumocystis_pct}%`}           color='#4e342e' />
            <Badge text={`Bronchiectasis ${cohort.bronchiectasis_pct}%`}       color='#e65100' />
            <Badge text={`Hypogamm ${cohort.hypogamm_pct}%`}                  color='#1b5e20' />
            <Badge text={`HSCT ${cohort.hsct_pct}%`}                          color='#1565c0' />
          </>}
          {isBLS1 && <>
            <Badge text={`HLA-A/B/C Low ${cohort.hla_abc_low_pct}%`}          color={color} />
            <Badge text={`Skin Granulomas ${cohort.skin_granulomas_pct}%`}     color='#b71c1c' />
            <Badge text={`Sinopulmonary ${cohort.sinopulmonary_pct}%`}        color='#880e4f' />
            <Badge text={`Bronchiectasis ${cohort.bronchiectasis_pct}%`}      color='#e65100' />
            <Badge text={`Behçet-Like ${cohort.behcet_like_pct}%`}            color='#4a148c' />
            <Badge text={`HSCT ${cohort.hsct_pct}%`}                         color='#1565c0' />
          </>}
          {gene === 'ZAP70' && <>
            <Badge text={`CD8+ Absent ${cohort.cd8_absent_pct}%`}             color={color} />
            <Badge text={`Pneumocystis ${cohort.pneumocystis_pct}%`}          color='#4e342e' />
            <Badge text={`Mennonite Allele ${cohort.mennonite_allele_pct}%`}  color='#795548' />
            <Badge text={`Autoimmune ${cohort.autoimmune_pct}%`}              color='#b71c1c' />
            <Badge text={`HSCT ${cohort.hsct_pct}%`}                         color='#1565c0' />
          </>}
          <Badge text={`n=${cohort.n}`} color='#555' />
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

export default function HereditaryMHCBareLymphocyteSyndromeAtlasPage() {
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
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const cohortMap = breakdown?.gene_cohorts
    ? Object.fromEntries(breakdown.gene_cohorts.map(c => [c.gene, c]))
    : {};

  return (
    <div style={{ background: '#0a0a1a', color: '#e8e8f0', minHeight: '100vh', padding: '24px 32px', fontFamily: 'monospace' }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#90caf9', marginBottom: 6 }}>
          🧬 Hereditary MHC Bare Lymphocyte Syndrome Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#aaa' }}>
          Complete 8-Gene Reference · MHC Class II Deficiency (BLS-II) · MHC Class I Deficiency (BLS-I) · ZAP70 CID<br />
          CIITA · RFXANK · RFX5 · RFXAP · TAP1 · TAP2 · TAPBP · ZAP70 · 320 patients · seeds 3934–3941
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{
              padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: tab === t ? '#1565c0' : '#1e1e3a', color: '#fff', fontSize: 13,
            }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#888' }}>Loading atlas data…</div>}
      {error   && <div style={{ color: '#f44' }}>Error: {error}</div>}

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 24 }}>
            <StatCard label="Total Patients" value={overview.total_patients} color="#90caf9" />
            <StatCard label="Genes Covered"  value={overview.genes_covered?.length} color="#a5d6a7" />
            <StatCard label="HSCT Rate"      value={`${overview.aggregate_stats?.total_hsct_pct}%`} color="#ffcc80" />
            <StatCard label="Seeds"          value={overview.seeds} color="#ce93d8" />
          </div>

          {/* Disease group summary */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16, marginBottom: 24 }}>
            <GroupCard
              title="BLS-II: MHC Class II Deficiency"
              genes={overview.aggregate_stats?.bls2_genes || []}
              color="#1565c0"
              stats={[
                { label: 'CD4+ Lymphopenia', value: `${overview.aggregate_stats?.bls2_cd4_lymphopenia_pct}%` },
                { label: 'Cryptosporidium',  value: `${overview.aggregate_stats?.bls2_cryptosporidium_pct}%` },
                { label: 'n', value: overview.aggregate_stats?.bls2_n },
              ]}
              pathognomonic="CD4+ <200/µL + absent HLA-DR on B cells + Cryptosporidium = BLS-II"
            />
            <GroupCard
              title="BLS-I: MHC Class I Deficiency"
              genes={overview.aggregate_stats?.bls1_genes || []}
              color="#b71c1c"
              stats={[
                { label: 'Skin Granulomas', value: `${overview.aggregate_stats?.bls1_skin_granulomas_pct}%` },
                { label: 'n', value: overview.aggregate_stats?.bls1_n },
              ]}
              pathognomonic="Sterile granulomatous skin lesions + absent HLA-A/B/C = BLS-I"
            />
            <GroupCard
              title="ZAP-70 CID"
              genes={overview.aggregate_stats?.cid_genes || []}
              color="#1b5e20"
              stats={[
                { label: 'CD8+ Absent', value: `${overview.aggregate_stats?.zap_cd8_absent_pct}%` },
                { label: 'n', value: overview.aggregate_stats?.zap_n },
              ]}
              pathognomonic="CD8+ absent + CD4+ present (anergic) = ZAP-70 PATHOGNOMONIC"
            />
          </div>

          {/* Cascade testing */}
          <div style={{ background: '#111130', borderRadius: 8, padding: 16, marginBottom: 24 }}>
            <h3 style={{ color: '#ffcc80', marginBottom: 12 }}>🔬 Cascade Testing Concepts</h3>
            {(overview.cascade_testing_concepts || []).map((c, i) => (
              <div key={i} style={{ fontSize: 12, color: '#ccc', marginBottom: 8, paddingLeft: 12, borderLeft: '2px solid #1565c0' }}>
                {c}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          {/* Phenotype matrix */}
          <div style={{ overflowX: 'auto', marginBottom: 24 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <thead>
                <tr style={{ background: '#1e1e3a' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#90caf9' }}>Gene</th>
                  <th style={{ padding: '8px 6px', textAlign: 'center', color: '#90caf9' }}>Group</th>
                  <th style={{ padding: '8px 6px', textAlign: 'center', color: '#90caf9' }}>CD4+↓</th>
                  <th style={{ padding: '8px 6px', textAlign: 'center', color: '#90caf9' }}>HLA-DR–</th>
                  <th style={{ padding: '8px 6px', textAlign: 'center', color: '#90caf9' }}>Crypto</th>
                  <th style={{ padding: '8px 6px', textAlign: 'center', color: '#90caf9' }}>PCP</th>
                  <th style={{ padding: '8px 6px', textAlign: 'center', color: '#90caf9' }}>Bronch</th>
                  <th style={{ padding: '8px 6px', textAlign: 'center', color: '#90caf9' }}>Hypogam</th>
                  <th style={{ padding: '8px 6px', textAlign: 'center', color: '#90caf9' }}>HSCT</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#90caf9' }}>Unique / Pathognomonic</th>
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#0f0f2a' : '#12122e' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                    <td style={{ padding: '7px 6px', textAlign: 'center', fontSize: 10, color: '#aaa' }}>{row.group}</td>
                    <td style={{ padding: '7px 6px', textAlign: 'center' }}>{row.cd4lp}</td>
                    <td style={{ padding: '7px 6px', textAlign: 'center' }}>{row.hladr}</td>
                    <td style={{ padding: '7px 6px', textAlign: 'center' }}>{row.crypto}</td>
                    <td style={{ padding: '7px 6px', textAlign: 'center' }}>{row.pcp}</td>
                    <td style={{ padding: '7px 6px', textAlign: 'center' }}>{row.bronch}</td>
                    <td style={{ padding: '7px 6px', textAlign: 'center' }}>{row.hypogam}</td>
                    <td style={{ padding: '7px 6px', textAlign: 'center' }}>{row.hsct}</td>
                    <td style={{ padding: '7px 10px', fontSize: 10, color: '#ddd', maxWidth: 260 }}>{row.unique}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Gene cards */}
          {GENES.map(gene => (
            <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={GENE_INFO[gene]} cohort={cohortMap[gene]} />
          ))}
        </div>
      )}

      {/* Pathway Map Tab */}
      {tab === 'Pathway Map' && breakdown && (
        <div>
          {Object.entries(breakdown.diagnostic_groups || {}).map(([groupKey, group]) => (
            <div key={groupKey} style={{ background: '#111130', borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <h3 style={{ color: '#ffcc80', marginBottom: 8, textTransform: 'capitalize' }}>
                {groupKey.replace(/_/g, ' ')}
              </h3>
              <div style={{ fontSize: 12, color: '#90caf9', marginBottom: 6 }}>
                Genes: {(group.genes || []).map(g => (
                  <span key={g} style={{ marginRight: 8, color: GENE_COLORS[g], fontWeight: 700 }}>{g}</span>
                ))}
              </div>
              <div style={{ fontSize: 12, color: '#ccc', marginBottom: 6 }}>
                <strong style={{ color: '#aaa' }}>Mechanism:</strong> {group.mechanism}
              </div>
              <div style={{ fontSize: 12, color: '#ccc', marginBottom: 6 }}>
                <strong style={{ color: '#aaa' }}>Shared feature:</strong> {group.shared_feature}
              </div>
              <div style={{ fontSize: 12, color: '#ccc' }}>
                <strong style={{ color: '#aaa' }}>Key DDx:</strong> {group.key_ddx}
              </div>
              {group.complementation_groups && (
                <div style={{ marginTop: 10 }}>
                  <strong style={{ color: '#aaa', fontSize: 12 }}>Complementation groups:</strong>
                  {Object.entries(group.complementation_groups).map(([grp, desc]) => (
                    <div key={grp} style={{ fontSize: 11, color: '#ddd', marginTop: 4, paddingLeft: 12 }}>
                      <span style={{ color: '#90caf9', fontWeight: 700 }}>{grp}:</span> {desc}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}

          {/* Pathognomonic features */}
          <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>⚡ Pathognomonic Features</h3>
            {Object.entries(breakdown.pathognomonic_features || {}).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 10, paddingLeft: 12, borderLeft: '2px solid #ef9a9a' }}>
                <div style={{ fontSize: 12, color: '#ef9a9a', fontWeight: 700 }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ fontSize: 12, color: '#ddd' }}>{v}</div>
              </div>
            ))}
          </div>

          {/* Diagnostic pathway */}
          {defs?.diagnostic_pathway && (
            <div style={{ background: '#111130', borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#a5d6a7', marginBottom: 12 }}>🔬 Diagnostic Pathway</h3>
              {defs.diagnostic_pathway.map((step, i) => (
                <div key={i} style={{ fontSize: 12, color: '#ccc', marginBottom: 8, paddingLeft: 12, borderLeft: '2px solid #4caf50' }}>
                  {step}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && defs && (
        <div>
          <div style={{ marginBottom: 24 }}>
            {(defs.definitions || []).map((d, i) => (
              <div key={i} style={{ background: '#111130', borderRadius: 8, padding: 14, marginBottom: 12 }}>
                <div style={{ fontWeight: 700, color: '#90caf9', marginBottom: 4 }}>{d.term}</div>
                <div style={{ fontSize: 12, color: '#ccc' }}>{d.definition}</div>
              </div>
            ))}
          </div>

          {defs.key_distinguishers && (
            <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#ffcc80', marginBottom: 12 }}>🔑 Key Distinguishers</h3>
              {Object.entries(defs.key_distinguishers).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 10, paddingLeft: 12, borderLeft: '2px solid #ffcc80' }}>
                  <div style={{ fontSize: 12, color: '#ffcc80', fontWeight: 700 }}>{k.replace(/_/g, ' ')}</div>
                  <div style={{ fontSize: 12, color: '#ddd' }}>{v}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const GENES = ['CIITA', 'RFXANK', 'RFX5', 'RFXAP', 'TAP1', 'TAP2', 'TAPBP', 'ZAP70'];

function StatCard({ label, value, color }) {
  return (
    <div style={{ background: '#111130', borderRadius: 8, padding: '14px 18px', textAlign: 'center' }}>
      <div style={{ fontSize: 24, fontWeight: 700, color }}>{value}</div>
      <div style={{ fontSize: 12, color: '#aaa', marginTop: 4 }}>{label}</div>
    </div>
  );
}

function GroupCard({ title, genes, color, stats, pathognomonic }) {
  return (
    <div style={{ background: '#111130', borderRadius: 8, padding: 14, border: `1px solid ${color}` }}>
      <div style={{ fontWeight: 700, color, marginBottom: 8 }}>{title}</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
        {genes.map(g => (
          <span key={g} style={{ background: GENE_COLORS[g] || color, color: '#fff', padding: '2px 7px', borderRadius: 4, fontSize: 11 }}>{g}</span>
        ))}
      </div>
      {stats.map(s => (
        <div key={s.label} style={{ fontSize: 12, color: '#ccc', marginBottom: 4 }}>
          {s.label}: <span style={{ color, fontWeight: 700 }}>{s.value}</span>
        </div>
      ))}
      <div style={{ fontSize: 11, color: '#aaa', marginTop: 8, fontStyle: 'italic' }}>{pathognomonic}</div>
    </div>
  );
}
