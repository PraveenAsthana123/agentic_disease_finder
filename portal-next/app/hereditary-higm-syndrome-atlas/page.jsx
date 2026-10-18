'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-higm-syndrome-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'CD40LG': '#b71c1c',  // deep red          -- HIGM1; XLR; most common HIGM; CD40L absent
  'CD40':   '#c62828',  // red               -- HIGM3; AR; both sexes; CD40 absent
  'AICDA':  '#1a237e',  // deep indigo       -- HIGM2; AR; GC hyperplasia PATHOGNOMONIC
  'UNG':    '#0d47a1',  // dark blue         -- HIGM5; AR; normal LN; mild phenotype
  'IKBKG':  '#e65100',  // deep orange       -- EDA-ID XLR; anhidrotic ectodermal dysplasia triad
  'MSH5':   '#1b5e20',  // deep green        -- CSR deficiency; MMR; IBD; AR
  'RBCK1':  '#4a148c',  // deep purple       -- LUBAC HOIL-1L; polyglucosan myopathy+IBD PATHOGNOMONIC
  'NFKB2':  '#880e4f',  // deep pink/maroon  -- CVID3 AD-LOF; pituitary stalk thickening PATHOGNOMONIC
};

const GENE_INFO = {
  'CD40LG': { full: 'CD40LG-261aa / XLR / CD40 Ligand (CD154) / HIGM1 / Xq26.3 / Males-Only / CD40L-Absent-Flow-DIAGNOSTIC / PCP-in-Infant-Boy-PATHOGNOMONIC / Cryptosporidium',                     locus: 'Xq26.3',    size: '261 aa / 29 kDa',  inh: 'X-linked recessive' },
  'CD40':   { full: 'CD40-277aa / AR / TNFR-Superfamily-Member-5 / HIGM3 / 20q13.12 / Both-Sexes / CD40-Absent-on-B-cells-Flow / CD40L-NORMAL / Indistinguishable-Phenotype-HIGM1',                    locus: '20q13.12',  size: '277 aa / 30 kDa',  inh: 'Autosomal recessive' },
  'AICDA':  { full: 'AICDA-198aa / AR / Activation-Induced-Cytidine-Deaminase / HIGM2 / 12p13.31 / GC-Hyperplasia-Lymphadenopathy-PATHOGNOMONIC / ALL-Ig-Isotypes-Absent / AID-Absent-Intracellular',  locus: '12p13.31',  size: '198 aa / 24 kDa',  inh: 'Autosomal recessive' },
  'UNG':    { full: 'UNG-313aa / AR / Uracil-DNA-Glycosylase / HIGM5 / 12q24.11 / Normal-LN-Architecture / IgE-Elevated-Paradoxical / Mild-Phenotype-Later-Onset',                                    locus: '12q24.11',  size: '313 aa / 35 kDa',  inh: 'Autosomal recessive' },
  'IKBKG':  { full: 'IKBKG-419aa / XLR / NF-kB-Essential-Modulator-NEMO / EDA-ID / Xq28 / Males-Only / Anhidrosis+Hypodontia+Thin-Hair-EDA-Triad-PATHOGNOMONIC / Mycobacteria+Pyogenic-Combo',        locus: 'Xq28',      size: '419 aa / 48 kDa',  inh: 'X-linked recessive' },
  'MSH5':   { full: 'MSH5-833aa / AR / MutS-Homolog-5 / CSR-Deficiency / 6p21.33-MHC-Region / IBD+Sinopulmonary / Normal-CD4-T-Cells / Selective-Ig-Isotype-Deficiency',                              locus: '6p21.33',   size: '833 aa / 92 kDa',  inh: 'Autosomal recessive' },
  'RBCK1':  { full: 'RBCK1-380aa / AR / LUBAC-HOIL-1L / Linear-Ubiquitin-Assembly / 6p25.2 / Polyglucosan-Myopathy-PATHOGNOMONIC / Amylopectinosis+IBD+Autoinflammation / NF-kB-Pathway',            locus: '6p25.2',    size: '380 aa / 42 kDa',  inh: 'Autosomal recessive' },
  'NFKB2':  { full: 'NFKB2-968aa / AD-LOF / NF-kB2-p100/p52 / CVID3 / 10q24.32 / Pituitary-Stalk-Thickening-PATHOGNOMONIC / Central-Diabetes-Insipidus / Adrenal-Insufficiency',                    locus: '10q24.32',  size: '968 aa / 110 kDa', inh: 'Autosomal dominant (LOF)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'CD40LG', group: 'HIGM1 (CD40/CD40L)', higm: '✓✓✓', crypto: '✓✓✓', pcp: '✓✓✓', gi: '✓✓',  liver: '✓✓',  neuro: '✓✓',  eda: '✗', hsct: '✓✓✓', unique: 'PCP IN INFANT BOY: CD40L-absent on activated CD4+ T cells (flow) = DIAGNOSTIC; Cryptosporidium cholangitis → cirrhosis; males only; most common HIGM worldwide; avoid live vaccines + PCP prophylaxis from birth' },
  { gene: 'CD40',   group: 'HIGM3 (CD40/CD40L)', higm: '✓✓✓', crypto: '✓✓✓', pcp: '✓✓',  gi: '✓✓',  liver: '✓✓',  neuro: '✓',  eda: '✗', hsct: '✓✓✓', unique: 'BOTH SEXES AFFECTED (AR): CD40 absent on B cells (flow); CD40L NORMAL → distinguishes from HIGM1; phenotype indistinguishable; second most common HIGM; HSCT curative' },
  { gene: 'AICDA',  group: 'HIGM2 (AID/UNG)',    higm: '✓✓✓', crypto: '✓',   pcp: '✓',   gi: '✓',   liver: '✓',   neuro: '±',  eda: '✗', hsct: '✓',   unique: 'GC HYPERPLASIA + MASSIVE LYMPHADENOPATHY PATHOGNOMONIC: massive tonsillar/cervical LN despite severe immunodeficiency; all Ig isotypes absent; AID absent on intracellular staining; EBV-driven lymphoma risk' },
  { gene: 'UNG',    group: 'HIGM5 (AID/UNG)',    higm: '✓✓',  crypto: '±',   pcp: '±',   gi: '±',   liver: '±',   neuro: '±',  eda: '✗', hsct: '±',   unique: 'NORMAL LN ARCHITECTURE: no GC hyperplasia (unlike AICDA); paradoxical IgE elevation in some; mildest HIGM phenotype; later onset; IgG/IgA/IgM imbalance; confirm by UNG enzyme assay or WES' },
  { gene: 'IKBKG',  group: 'EDA-ID (NF-kB)',     higm: '✓✓',  crypto: '✓',   pcp: '✓',   gi: '✓',   liver: '±',   neuro: '✓✓', eda: '✓✓✓', hsct: '✓', unique: 'ANHIDROSIS + HYPODONTIA + THIN/SPARSE HAIR EDA TRIAD PATHOGNOMONIC: skin biopsy confirms absent sweat glands; mycobacterial (BCG-osis, MAC) + encapsulated bacteria + pyogenic combo; males only (XLR)' },
  { gene: 'MSH5',   group: 'CSR (Mismatch)',     higm: '✓✓',  crypto: '±',   pcp: '±',   gi: '✓✓',  liver: '±',   neuro: '±',  eda: '✗', hsct: '±',   unique: 'IBD + SINOPULMONARY COMBO: normal CD4+ T cells; selective Ig isotype deficiency; 6p21.33 in MHC region; IBD may precede immunodeficiency diagnosis; no ectodermal features; MMR pathway' },
  { gene: 'RBCK1',  group: 'LUBAC (NF-kB)',      higm: '✓✓',  crypto: '±',   pcp: '±',   gi: '✓✓✓', liver: '±',   neuro: '±',  eda: '✗', hsct: '✓',   unique: 'POLYGLUCOSAN STORAGE MYOPATHY + IBD + AUTOINFLAMMATION PATHOGNOMONIC: amylopectinosis in muscle biopsy; muscle weakness + recurrent fever + IBD; NF-kB hypomorphic; unique LUBAC complex defect' },
  { gene: 'NFKB2',  group: 'CVID3 (NF-kB2)',     higm: '✓✓',  crypto: '±',   pcp: '±',   gi: '±',   liver: '±',   neuro: '✓✓✓', eda: '✗', hsct: '±',  unique: 'PITUITARY STALK THICKENING + CENTRAL DI PATHOGNOMONIC: MRI pituitary → stalk thickening before immunodeficiency symptomatic; CDI + adrenal insufficiency; AD-LOF; IgG low; small deletions in NFKB2 C-terminus' },
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
  const isCD40path = ['CD40LG', 'CD40'].includes(gene);
  const isAIDpath  = ['AICDA', 'UNG'].includes(gene);
  const isNFkB     = ['IKBKG', 'MSH5', 'RBCK1', 'NFKB2'].includes(gene);
  const groupLabel = isCD40path ? 'CD40/CD40L Pathway (HIGM1/3)'
    : isAIDpath ? 'AID/UNG Pathway (HIGM2/5)'
    : 'NF-κB / CSR Pathway';
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
          {isCD40path && <>
            <Badge text={`Elevated IgM ${cohort.elevated_igm_pct}%`}         color={color} />
            <Badge text={`Cryptosporidium ${cohort.cryptosporidium_pct}%`}   color='#4a148c' />
            <Badge text={`Pneumocystis ${cohort.pneumocystis_pct}%`}         color='#4e342e' />
            <Badge text={`Liver Disease ${cohort.liver_disease_pct}%`}       color='#e65100' />
            <Badge text={`HSCT ${cohort.hsct_pct}%`}                        color='#1565c0' />
          </>}
          {isAIDpath && <>
            <Badge text={`Elevated IgM ${cohort.elevated_igm_pct}%`}         color={color} />
            <Badge text={`GC Hyperplasia ${cohort.gc_hyperplasia_pct}%`}    color='#880e4f' />
            <Badge text={`Lymphadenopathy ${cohort.lymphadenopathy_pct}%`}  color='#1b5e20' />
            <Badge text={`Sinopulmonary ${cohort.sinopulmonary_pct}%`}      color='#e65100' />
            <Badge text={`HSCT ${cohort.hsct_pct}%`}                       color='#1565c0' />
          </>}
          {isNFkB && <>
            <Badge text={`IgG Low ${cohort.igg_low_pct}%`}                  color={color} />
            <Badge text={`Recurrent Infect ${cohort.recurrent_infections_pct}%`} color='#880e4f' />
            <Badge text={`Sinopulmonary ${cohort.sinopulmonary_pct}%`}      color='#e65100' />
            <Badge text={`HSCT ${cohort.hsct_pct}%`}                       color='#1565c0' />
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

export default function HereditaryHIGMSyndromeAtlasPage() {
  const [tab, setTab]           = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]         = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

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
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#ef9a9a', marginBottom: 6 }}>
          🧬 Hereditary Hyper-IgM Syndrome &amp; CSR Deficiency Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#aaa' }}>
          Complete 8-Gene Reference · CD40/CD40L Pathway · AID/UNG Pathway · NF-κB / CSR Deficiency<br />
          CD40LG · CD40 · AICDA · UNG · IKBKG · MSH5 · RBCK1 · NFKB2 · 320 patients · seeds 3942–3949
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{
              padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: tab === t ? '#b71c1c' : '#1e1e3a', color: '#fff', fontSize: 13,
            }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#888' }}>Loading atlas data…</div>}
      {error   && <div style={{ color: '#f44' }}>Error: {error}</div>}

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 24 }}>
            <StatCard label="Total Patients" value={overview.total_patients}             color="#ef9a9a" />
            <StatCard label="Genes Covered"  value={overview.genes_covered?.length}      color="#a5d6a7" />
            <StatCard label="HSCT Rate"      value={`${overview.aggregate_stats?.total_hsct_pct}%`} color="#ffcc80" />
            <StatCard label="Seeds"          value={overview.seeds}                      color="#ce93d8" />
          </div>

          {/* Disease group summary */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16, marginBottom: 24 }}>
            <GroupCard
              title="CD40/CD40L Pathway (HIGM1/3)"
              genes={overview.aggregate_stats?.cd40_genes || []}
              color="#b71c1c"
              stats={[
                { label: 'Cryptosporidium', value: `${overview.aggregate_stats?.cd40_cryptosporidium_pct}%` },
                { label: 'Liver Disease',   value: `${overview.aggregate_stats?.cd40_liver_disease_pct}%` },
                { label: 'n', value: overview.aggregate_stats?.cd40_n },
              ]}
              pathognomonic="PCP in infant boy (CD40LG) or both sexes (CD40) + elevated IgM = HIGM1/3"
            />
            <GroupCard
              title="AID/UNG Pathway (HIGM2/5)"
              genes={overview.aggregate_stats?.aid_genes || []}
              color="#1a237e"
              stats={[
                { label: 'GC Hyperplasia', value: `${overview.aggregate_stats?.aid_gc_hyperplasia_pct}%` },
                { label: 'n', value: overview.aggregate_stats?.aid_n },
              ]}
              pathognomonic="GC hyperplasia + massive lymphadenopathy (AICDA) = HIGM2 PATHOGNOMONIC"
            />
            <GroupCard
              title="NF-κB / CSR Pathway"
              genes={overview.aggregate_stats?.nfkb_genes || []}
              color="#e65100"
              stats={[
                { label: 'EDA Triad (IKBKG)', value: `${overview.aggregate_stats?.nfkb_eda_pct}%` },
                { label: 'n', value: overview.aggregate_stats?.nfkb_n },
              ]}
              pathognomonic="EDA triad (IKBKG) / polyglucosan myopathy (RBCK1) / pituitary stalk (NFKB2)"
            />
          </div>

          {/* Cascade testing */}
          <div style={{ background: '#111130', borderRadius: 8, padding: 16, marginBottom: 24 }}>
            <h3 style={{ color: '#ffcc80', marginBottom: 12 }}>🔬 Cascade Testing Concepts</h3>
            {(overview.cascade_testing_concepts || []).map((c, i) => (
              <div key={i} style={{ fontSize: 12, color: '#ccc', marginBottom: 8, paddingLeft: 12, borderLeft: '2px solid #b71c1c' }}>
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
                  <th style={{ padding: '8px 10px', textAlign: 'left',   color: '#ef9a9a' }}>Gene</th>
                  <th style={{ padding: '8px 6px',  textAlign: 'center', color: '#ef9a9a' }}>Group</th>
                  <th style={{ padding: '8px 6px',  textAlign: 'center', color: '#ef9a9a' }}>Hi-IgM</th>
                  <th style={{ padding: '8px 6px',  textAlign: 'center', color: '#ef9a9a' }}>Crypto</th>
                  <th style={{ padding: '8px 6px',  textAlign: 'center', color: '#ef9a9a' }}>PCP</th>
                  <th style={{ padding: '8px 6px',  textAlign: 'center', color: '#ef9a9a' }}>GI</th>
                  <th style={{ padding: '8px 6px',  textAlign: 'center', color: '#ef9a9a' }}>Liver</th>
                  <th style={{ padding: '8px 6px',  textAlign: 'center', color: '#ef9a9a' }}>Neuro</th>
                  <th style={{ padding: '8px 6px',  textAlign: 'center', color: '#ef9a9a' }}>EDA</th>
                  <th style={{ padding: '8px 6px',  textAlign: 'center', color: '#ef9a9a' }}>HSCT</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left',   color: '#ef9a9a' }}>Unique / Pathognomonic</th>
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#0f0f2a' : '#12122e' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                    <td style={{ padding: '7px 6px',  textAlign: 'center', fontSize: 10, color: '#aaa' }}>{row.group}</td>
                    <td style={{ padding: '7px 6px',  textAlign: 'center' }}>{row.higm}</td>
                    <td style={{ padding: '7px 6px',  textAlign: 'center' }}>{row.crypto}</td>
                    <td style={{ padding: '7px 6px',  textAlign: 'center' }}>{row.pcp}</td>
                    <td style={{ padding: '7px 6px',  textAlign: 'center' }}>{row.gi}</td>
                    <td style={{ padding: '7px 6px',  textAlign: 'center' }}>{row.liver}</td>
                    <td style={{ padding: '7px 6px',  textAlign: 'center' }}>{row.neuro}</td>
                    <td style={{ padding: '7px 6px',  textAlign: 'center' }}>{row.eda}</td>
                    <td style={{ padding: '7px 6px',  textAlign: 'center' }}>{row.hsct}</td>
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
              <div style={{ fontSize: 12, color: '#ef9a9a', marginBottom: 6 }}>
                Genes: {(group.genes || []).map(g => (
                  <span key={g} style={{ marginRight: 8, color: GENE_COLORS[g] || '#ccc', fontWeight: 700 }}>{g}</span>
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
                <div style={{ fontWeight: 700, color: '#ef9a9a', marginBottom: 4 }}>{d.term}</div>
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

const GENES = ['CD40LG', 'CD40', 'AICDA', 'UNG', 'IKBKG', 'MSH5', 'RBCK1', 'NFKB2'];

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
