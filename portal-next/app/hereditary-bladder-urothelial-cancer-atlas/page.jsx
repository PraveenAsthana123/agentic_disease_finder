'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-bladder-urothelial-cancer-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MSH2':  '#4a148c',  // deep purple    — Lynch Type 2; urothelial 25% HIGHEST PATHOGNOMONIC; Muir-Torre sebaceous
  'MLH1':  '#1a237e',  // deep indigo    — Lynch Type 1; urothelial 2-4%; BRAF V600E excludes; constitutional methylation
  'MSH6':  '#311b92',  // deep purple-navy — Lynch Type 3; urothelial 7-11%; MSI-L false negative PITFALL; IHC primary
  'BRCA2': '#006064',  // dark teal      — HBOC; bladder 2-4x; cisplatin/platinum sensitive; PARP olaparib
  'RB1':   '#880e4f',  // deep magenta   — Retinoblastoma; secondary bladder SCC 15-20x; BCG CONTRAINDICATED; CDK4/6i INACTIVE
  'TP53':  '#c62828',  // brick red      — LFS; AVOID RADIATION ABSOLUTELY; WBMRI Toronto; radical cystectomy only
  'HRAS':  '#e65100',  // deep orange    — Costello; bladder TCC/RMS adolescent PATHOGNOMONIC; papillomata PATHOGNOMONIC; MEK trametinib
  'CHEK2': '#33691e',  // deep olive     — c.1100delC 1% Northern European; bladder 2-3x moderate; intermediate penetrance
};

const GENE_INFO = {
  'MSH2':  { full: 'Lynch-Type2 / Urothelial-25%-LIFETIME-HIGHEST / Muir-Torre-Sebaceous-Face-PATHOGNOMONIC / EPCAM-3prime-Silencing / Pembrolizumab-MSI-H',  locus: '2p21',    size: '934 aa / 105 kDa',   inh: 'AD LOF' },
  'MLH1':  { full: 'Lynch-Type1 / Urothelial-2-4% / BRAF-V600E-EXCLUDES-Lynch / Constitutional-MLH1-Methylation-Misses-Sequencing / Aspirin-CAPP2-50%',        locus: '3p22.2',  size: '793 aa / 90 kDa',    inh: 'AD LOF' },
  'MSH6':  { full: 'Lynch-Type3 / Urothelial-7-11% / MSI-L-30%-FALSE-NEGATIVE / IHC-PRIMARY-ALWAYS / Endometrial-40-71%-HIGHEST-Lynch',                        locus: '2p16.3',  size: '1360 aa / 160 kDa',  inh: 'AD LOF' },
  'BRCA2': { full: 'HBOC / Bladder-2-4x / Cisplatin-Platinum-Sensitive / PARP-Olaparib-FDA / FA-D1-Biallelic-AVOID-RADIATION',                                 locus: '13q12.3', size: '3418 aa / 384 kDa',  inh: 'AD LOF' },
  'RB1':   { full: 'Bilateral-Retinoblastoma-GERMLINE-PATHOGNOMONIC / Secondary-Bladder-SCC-15-20x-Post-RT / BCG-CONTRAINDICATED / CDK4-6I-INACTIVE',           locus: '13q14.2', size: '928 aa / 110 kDa',   inh: 'AD LOF' },
  'TP53':  { full: 'LFS / Bladder-SCC / AVOID-RADIATION-ABSOLUTELY / WBMRI-Toronto-Annually / Radical-Cystectomy-Not-RT',                                      locus: '17p13.1', size: '393 aa / 43 kDa',    inh: 'AD LOF' },
  'HRAS':  { full: 'Costello-Syndrome / Bladder-TCC-RMS-Adolescent-PATHOGNOMONIC / Papillomata-Perianal-PATHOGNOMONIC / G12S-80% / MEK-Trametinib',             locus: '11p15.5', size: '189 aa / 21 kDa',    inh: 'AD GOF' },
  'CHEK2': { full: 'c.1100delC-1%-Northern-European / Bladder-2-3x-Moderate / Intermediate-Penetrance-NOT-BRCA-Equivalent / PRS-Modifies-Management',          locus: '22q12.1', size: '543 aa / 60 kDa',    inh: 'AD LOF' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color,
      border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 7px',
      fontSize: 11, fontWeight: 600, marginRight: 4,
    }}>{text}</span>
  );
}

export default function HereditaryBladderUrothelialCancerAtlas() {
  const [tab, setTab]                   = useState('Overview');
  const [overview, setOverview]         = useState(null);
  const [breakdown, setBreakdown]       = useState(null);
  const [definitions, setDefinitions]   = useState(null);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState(null);
  const [expandedGene, setExpandedGene] = useState(null);
  const [expandedDef, setExpandedDef]   = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const containerStyle = {
    fontFamily: 'monospace', background: '#0a0a0a', color: '#e0e0e0',
    minHeight: '100vh', padding: '24px',
  };
  const cardStyle = {
    background: '#141414', border: '1px solid #2a2a2a',
    borderRadius: 8, padding: 20, marginBottom: 16,
  };
  const tabStyle = (active) => ({
    padding: '8px 18px', cursor: 'pointer', borderRadius: 4,
    background: active ? '#1a237e' : '#1a1a1a',
    color: active ? '#90caf9' : '#888',
    border: `1px solid ${active ? '#1a237e' : '#333'}`,
    fontFamily: 'monospace', fontSize: 13, marginRight: 6,
  });

  if (loading) return <div style={containerStyle}><p style={{ color: '#888' }}>Loading Hereditary Bladder & Urothelial Cancer Atlas…</p></div>;
  if (error)   return <div style={containerStyle}><p style={{ color: '#ef5350' }}>Error: {error}</p></div>;

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={{ ...cardStyle, borderLeft: '4px solid #4a148c' }}>
        <h1 style={{ color: '#ce93d8', fontSize: 18, margin: '0 0 6px' }}>
          🧬 Hereditary Bladder &amp; Urothelial Cancer Predisposition Atlas
        </h1>
        <div style={{ color: '#888', fontSize: 12 }}>
          Complete 8-Gene Reference · MSH2-MLH1-MSH6-BRCA2-RB1-TP53-HRAS-CHEK2
          {overview && <span> · Seeds {overview.seed_range} · {overview.total_patients} patients (8×40)</span>}
        </div>
        <div style={{ marginTop: 8, fontSize: 11, color: '#aaa' }}>
          <Badge text="MSH2 Urothelial 25% HIGHEST" color="#4a148c" />
          <Badge text="MSH6 MSI-L FALSE-NEGATIVE PITFALL" color="#311b92" />
          <Badge text="RB1 BCG-CONTRAINDICATED" color="#880e4f" />
          <Badge text="TP53 AVOID-RADIATION-ABSOLUTELY" color="#c62828" />
          <Badge text="HRAS Costello PATHOGNOMONIC" color="#e65100" />
        </div>
      </div>

      {/* Tabs */}
      <div style={{ marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} style={tabStyle(tab === t)} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={cardStyle}>
            <h2 style={{ color: '#ce93d8', fontSize: 15, margin: '0 0 12px' }}>
              Atlas Summary — Seeds {overview.seed_range}
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
              {[
                { label: 'Genes', value: overview.gene_count },
                { label: 'Total Patients', value: overview.total_patients },
                { label: 'Per Gene', value: overview.cohort_per_gene },
                { label: 'Highest Urothelial Risk', value: 'MSH2 25%' },
              ].map(s => (
                <div key={s.label} style={{ background: '#1a1a1a', borderRadius: 6, padding: 12, textAlign: 'center' }}>
                  <div style={{ color: '#ce93d8', fontSize: 20, fontWeight: 700 }}>{s.value}</div>
                  <div style={{ color: '#888', fontSize: 11 }}>{s.label}</div>
                </div>
              ))}
            </div>
            {overview.clinical_pearls && (
              <div>
                <div style={{ color: '#ef9a9a', fontSize: 12, fontWeight: 600, marginBottom: 6 }}>CLINICAL PEARLS:</div>
                {overview.clinical_pearls.map((p, i) => (
                  <div key={i} style={{ color: '#ccc', fontSize: 11, marginBottom: 5, paddingLeft: 10, borderLeft: '2px solid #333' }}>
                    {p}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Cancer Risk Summary */}
          {overview.cancer_risks && (
            <div style={cardStyle}>
              <h3 style={{ color: '#ce93d8', fontSize: 14, margin: '0 0 10px' }}>Cancer Risk by Gene</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr>
                      {['Gene', 'Urothelial Risk', 'Other Key Risks', 'Critical Warning'].map(h => (
                        <th key={h} style={{ color: '#888', padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #333' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {overview.genes && overview.genes.map(g => {
                      const cr = overview.cancer_risks[g] || {};
                      const urothelial = cr.urothelial_pct ? `${cr.urothelial_pct}% lifetime`
                        : cr.urothelial_rr ? `${cr.urothelial_rr}x RR`
                        : cr.secondary_bladder_rr ? `${cr.secondary_bladder_rr}x (secondary post-RT)`
                        : cr.bladder_tcc_pct ? `${cr.bladder_tcc_pct}% (TCC/RMS)`
                        : cr.bladder_rr ? `${cr.bladder_rr}x moderate`
                        : '—';
                      const warning = cr.sebaceous_skin ? 'Muir-Torre PATHOGNOMONIC'
                        : cr.braf_v600e_excludes ? 'BRAF V600E → SPORADIC'
                        : cr.msi_l_pitfall ? 'MSI-L FALSE NEGATIVE'
                        : cr.parp_eligible ? 'PARP Olaparib eligible'
                        : cr.bcg_contraindicated ? 'BCG CONTRAINDICATED'
                        : cr.avoid_radiation ? 'AVOID RADIATION ABSOLUTELY'
                        : cr.papillomata_pathognomonic ? 'Papillomata PATHOGNOMONIC'
                        : cr.intermediate_penetrance ? 'Intermediate penetrance'
                        : '—';
                      return (
                        <tr key={g} style={{ borderBottom: '1px solid #1e1e1e' }}>
                          <td style={{ padding: '6px 8px', color: GENE_COLORS[g] || '#aaa', fontWeight: 700 }}>{g}</td>
                          <td style={{ padding: '6px 8px', color: '#90caf9' }}>{urothelial}</td>
                          <td style={{ padding: '6px 8px', color: '#ccc', fontSize: 10 }}>{GENE_INFO[g]?.full?.split('/')[1]?.trim() || ''}</td>
                          <td style={{ padding: '6px 8px', color: '#ef9a9a', fontSize: 10 }}>{warning}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Pathognomonic Features */}
          {overview.pathognomonic_features && (
            <div style={cardStyle}>
              <h3 style={{ color: '#ce93d8', fontSize: 14, margin: '0 0 10px' }}>Pathognomonic Features</h3>
              {Object.entries(overview.pathognomonic_features).map(([gene, feat]) => (
                <div key={gene} style={{ marginBottom: 8, padding: '8px 12px', background: '#1a1a1a', borderRadius: 6, borderLeft: `3px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <span style={{ color: GENE_COLORS[gene] || '#aaa', fontWeight: 700, fontSize: 12 }}>{gene}: </span>
                  <span style={{ color: '#ccc', fontSize: 11 }}>{feat}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <div style={cardStyle}>
            <h2 style={{ color: '#ce93d8', fontSize: 15, margin: '0 0 12px' }}>Per-Gene Cohort Summary</h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr>
                    {['Gene', 'Locus', 'N', 'Mean Age', 'Bladder/Urothelial N', '%', 'Seed', 'Key Distinctions'].map(h => (
                      <th key={h} style={{ color: '#888', padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #333' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.breakdown && breakdown.breakdown.map(row => (
                    <tr key={row.gene} style={{ borderBottom: '1px solid #1e1e1e' }}>
                      <td style={{ padding: '6px 8px', color: GENE_COLORS[row.gene] || '#aaa', fontWeight: 700 }}>{row.gene}</td>
                      <td style={{ padding: '6px 8px', color: '#aaa' }}>{row.locus}</td>
                      <td style={{ padding: '6px 8px', color: '#ccc' }}>{row.n}</td>
                      <td style={{ padding: '6px 8px', color: '#90caf9' }}>{row.mean_age_onset}</td>
                      <td style={{ padding: '6px 8px', color: '#ce93d8' }}>{row.bladder_urothelial_n}</td>
                      <td style={{ padding: '6px 8px', color: '#ce93d8' }}>{row.bladder_urothelial_pct}%</td>
                      <td style={{ padding: '6px 8px', color: '#888' }}>{row.seed}</td>
                      <td style={{ padding: '6px 8px', color: '#aaa', fontSize: 10 }}>{(row.key_distinctions || []).slice(0, 2).join(' / ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Per-gene patient samples */}
          {breakdown.breakdown && breakdown.breakdown.map(row => (
            <div key={row.gene} style={{ ...cardStyle, borderLeft: `4px solid ${GENE_COLORS[row.gene] || '#555'}` }}
              onClick={() => setExpandedGene(expandedGene === row.gene ? null : row.gene)}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', cursor: 'pointer' }}>
                <div>
                  <span style={{ color: GENE_COLORS[row.gene] || '#aaa', fontWeight: 700, fontSize: 14 }}>{row.gene}</span>
                  <span style={{ color: '#888', fontSize: 11, marginLeft: 12 }}>{row.locus} · n={row.n} · mean age {row.mean_age_onset}yr · bladder/urothelial {row.bladder_urothelial_pct}%</span>
                </div>
                <span style={{ color: '#666', fontSize: 12 }}>{expandedGene === row.gene ? '▲' : '▼'}</span>
              </div>
              {expandedGene === row.gene && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ color: '#ef9a9a', fontSize: 11 }}>KEY DISTINCTIONS:</span>
                    <div style={{ color: '#ccc', fontSize: 11 }}>{(row.key_distinctions || []).join(' · ')}</div>
                  </div>
                  <div style={{ overflowX: 'auto', marginTop: 10 }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 10 }}>
                      <thead>
                        <tr>
                          {['Age', 'Cancer', 'Variant', 'Treatment', 'FH', 'Stage'].map(h => (
                            <th key={h} style={{ color: '#888', padding: '4px 6px', textAlign: 'left', borderBottom: '1px solid #222' }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(row.patients || []).slice(0, 10).map((p, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid #1a1a1a' }}>
                            <td style={{ padding: '3px 6px', color: '#90caf9' }}>{p.age_at_diagnosis}</td>
                            <td style={{ padding: '3px 6px', color: '#ce93d8' }}>{p.cancer_type}</td>
                            <td style={{ padding: '3px 6px', color: '#aaa' }}>{p.variant}</td>
                            <td style={{ padding: '3px 6px', color: '#a5d6a7' }}>{p.treatment}</td>
                            <td style={{ padding: '3px 6px', color: '#888' }}>{p.family_history}</td>
                            <td style={{ padding: '3px 6px', color: '#ffcc80' }}>{p.staging}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {(row.patients || []).length > 10 && (
                      <div style={{ color: '#666', fontSize: 10, marginTop: 4 }}>Showing 10 of {row.patients.length} patients</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && overview && (
        <div>
          {overview.genes && overview.genes.map(gene => (
            <div key={gene} style={{
              ...cardStyle,
              borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}`,
            }}>
              <h3 style={{ color: GENE_COLORS[gene] || '#aaa', margin: '0 0 8px', fontSize: 15 }}>
                {gene} — {GENE_INFO[gene]?.inh || ''}
              </h3>
              <div style={{ color: '#aaa', fontSize: 12, marginBottom: 6 }}>
                <span style={{ color: '#777' }}>Locus: </span>{GENE_INFO[gene]?.locus || ''}
                <span style={{ color: '#777', marginLeft: 12 }}>Size: </span>{GENE_INFO[gene]?.size || ''}
              </div>
              <div style={{ color: '#ccc', fontSize: 12, lineHeight: 1.6 }}>{GENE_INFO[gene]?.full || ''}</div>
              <div style={{ color: '#bbb', fontSize: 11, marginTop: 6 }}>{overview.inheritance_modes?.[gene] || ''}</div>
              {overview.key_distinctions?.[gene] && (
                <div style={{ color: '#90caf9', fontSize: 10, marginTop: 4 }}>{overview.key_distinctions[gene]}</div>
              )}
              {overview.surveillance_start_ages?.[gene] && (
                <div style={{ marginTop: 6, color: '#a5d6a7', fontSize: 10 }}>
                  Surveillance: {JSON.stringify(overview.surveillance_start_ages[gene])}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={cardStyle}>
            <h2 style={{ color: '#ce93d8', fontSize: 16, margin: '0 0 12px' }}>
              Clinical Definitions — Seeds {definitions.seed_range}
            </h2>
            {definitions.definitions && definitions.definitions.map((d, i) => (
              <div key={i} style={{
                background: '#1a1a1a', borderRadius: 6, padding: 12, marginBottom: 10,
                borderLeft: `3px solid ${
                  i === 0 ? '#4a148c' : i === 1 ? '#1a237e' : i === 2 ? '#311b92' :
                  i === 3 ? '#006064' : i === 4 ? '#880e4f' : i === 5 ? '#c62828' :
                  i === 6 ? '#e65100' : i === 7 ? '#33691e' : '#555'
                }`,
              }}>
                <div
                  style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between' }}
                  onClick={() => setExpandedDef(expandedDef === i ? null : i)}
                >
                  <span style={{ color: '#ce93d8', fontSize: 13, fontWeight: 600 }}>{d.term}</span>
                  <span style={{ color: '#666', fontSize: 12 }}>{expandedDef === i ? '▲' : '▼'}</span>
                </div>
                {expandedDef === i && (
                  <pre style={{ color: '#ccc', fontSize: 11, whiteSpace: 'pre-wrap', marginTop: 10, lineHeight: 1.7 }}>
                    {d.definition}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
