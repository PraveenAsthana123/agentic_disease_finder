'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-biliary-tract-cancer-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'BRCA1':  '#c62828',  // brick red       — HBOC; iCCA 2-4x; RRSO 35-40yr; olaparib
  'BRCA2':  '#1b5e20',  // deep green      — HBOC; biliary 5-7x HIGHEST; FA-D1; olaparib
  'BAP1':   '#4a148c',  // deep purple     — TPDS; iCCA 40-50% HIGHEST; BAP1-null IHC; AVOID ASBESTOS
  'MSH2':   '#0d47a1',  // deep blue       — Lynch; biliary 2-4%; Muir-Torre PATHOGNOMONIC; EPCAM
  'STK11':  '#e65100',  // deep orange     — PJS; gallbladder 5-13%; pancreatic 30%; macules PATHOGNOMONIC
  'ATM':    '#880e4f',  // deep magenta    — A-T; biliary 2-4x; radiosensitivity ABSOLUTE; ceralasertib
  'CDKN2A': '#006064',  // dark teal       — FAMM; biliary 2-3x; pancreatic 20x; melanoma 25-36%
  'PALB2':  '#33691e',  // deep olive      — HBOC-2; biliary 2-3x; breast 53%; olaparib 82% ORR TBCRC048
};

const GENE_INFO = {
  'BRCA1':  { full: 'HBOC / iCCA-2-4x-RR / Gallbladder-2-3x / RRSO-35-40yr / Olaparib-PARP-Sensitive',                             locus: '17q21.31', size: '1863 aa / 213 kDa', inh: 'AD LOF' },
  'BRCA2':  { full: 'HBOC / iCCA-Gallbladder-5-7x-HIGHEST / FA-D1-Biallelic / RRSO-40-45yr / Olaparib-POLO',                        locus: '13q12.3',  size: '3418 aa / 384 kDa', inh: 'AD LOF' },
  'BAP1':   { full: 'TPDS / iCCA-40-50%-LIFETIME-HIGHEST / BAP1-null-IHC-PATHOGNOMONIC / Mesothelioma-30-60x / AVOID-ASBESTOS',      locus: '3p21.1',   size: '729 aa / 80 kDa',   inh: 'AD LOF' },
  'MSH2':   { full: 'Lynch-T2 / Biliary-2-4%-Lifetime / Muir-Torre-Sebaceous-PATHOGNOMONIC / EPCAM-3prime-Deletion-MLPA-Mandatory',  locus: '2p21',     size: '934 aa / 105 kDa',  inh: 'AD LOF' },
  'STK11':  { full: 'PJS / Gallbladder-5-13%-HIGHEST / Pancreatic-30%-Dominant / Mucocutaneous-Macules-Perioral-PATHOGNOMONIC',      locus: '19p13.3',  size: '433 aa / 48 kDa',   inh: 'AD LOF' },
  'ATM':    { full: 'A-T-Biallelic / Biliary-2-4x-Monoallelic / RADIOSENSITIVITY-ABSOLUTE-Biallelic / Ceralasertib-ATRi-Olaparib',  locus: '11q22.3',  size: '3056 aa / 350 kDa', inh: 'AD/AR LOF' },
  'CDKN2A': { full: 'FAMM / Biliary-Ampullary-2-3x / Pancreatic-20x-DOMINANT / Melanoma-25-36% / CDK4-6i-Pathway',                  locus: '9p21.3',   size: '156 aa / 16 kDa',   inh: 'AD LOF' },
  'PALB2':  { full: 'HBOC-2 / Biliary-2-3x / Breast-53%-Lifetime / Olaparib-TBCRC048-82%-ORR-HIGHEST / FA-N-Biallelic',             locus: '16p12.2',  size: '1186 aa / 131 kDa', inh: 'AD LOF' },
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

export default function HereditaryBiliaryTractCancerAtlas() {
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
    background: active ? '#4a148c' : '#1a1a1a',
    color: active ? '#e1bee7' : '#888',
    border: `1px solid ${active ? '#4a148c' : '#333'}`,
    fontFamily: 'monospace', fontSize: 13, marginRight: 6,
  });

  if (loading) return <div style={containerStyle}><p style={{ color: '#888' }}>Loading Hereditary Biliary Tract Cancer Predisposition Atlas…</p></div>;
  if (error)   return <div style={containerStyle}><p style={{ color: '#ef5350' }}>Error: {error}</p></div>;

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={{ ...cardStyle, borderLeft: '4px solid #4a148c' }}>
        <h1 style={{ color: '#ce93d8', fontSize: 18, margin: '0 0 6px' }}>
          🧬 Hereditary Biliary Tract Cancer Predisposition Atlas
        </h1>
        <div style={{ color: '#888', fontSize: 12 }}>
          Complete 8-Gene Reference · BRCA1-BRCA2-BAP1-MSH2-STK11-ATM-CDKN2A-PALB2
          {overview && <span> · Seeds {overview.seed_range} · {overview.total_patients} patients (8×40)</span>}
        </div>
        <div style={{ marginTop: 8, fontSize: 11, color: '#aaa' }}>
          <Badge text="BAP1 iCCA 40-50% HIGHEST" color="#4a148c" />
          <Badge text="STK11 Gallbladder 5-13%" color="#e65100" />
          <Badge text="BRCA2 Biliary 5-7x HIGHEST-BRCA" color="#1b5e20" />
          <Badge text="BAP1-null IHC PATHOGNOMONIC" color="#4a148c" />
          <Badge text="MSH2 Muir-Torre PATHOGNOMONIC" color="#0d47a1" />
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
                { label: 'Genes', value: overview.genes_n },
                { label: 'Total Patients', value: overview.total_patients },
                { label: 'Biliary N', value: overview.biliary_total_n },
                { label: 'Highest Risk', value: 'BAP1 40-50%' },
              ].map(s => (
                <div key={s.label} style={{ background: '#1a1a1a', borderRadius: 6, padding: 12, textAlign: 'center' }}>
                  <div style={{ color: '#ce93d8', fontSize: 20, fontWeight: 700 }}>{s.value}</div>
                  <div style={{ color: '#888', fontSize: 11 }}>{s.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Gene Risk Table */}
          <div style={cardStyle}>
            <h3 style={{ color: '#ce93d8', fontSize: 14, margin: '0 0 10px' }}>Biliary Tract Risk by Gene</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr>
                    {['Gene', 'Locus', 'Biliary Risk', 'Dominant Other Risk', 'Key Management'].map(h => (
                      <th key={h} style={{ color: '#888', padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #333' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summary && overview.gene_summary.map(g => (
                    <tr key={g.gene} style={{ borderBottom: '1px solid #1e1e1e' }}>
                      <td style={{ padding: '6px 8px', color: GENE_COLORS[g.gene] || '#aaa', fontWeight: 700 }}>{g.gene}</td>
                      <td style={{ padding: '6px 8px', color: '#aaa' }}>{g.locus}</td>
                      <td style={{ padding: '6px 8px', color: '#ce93d8' }}>{g.biliary_pct}% ({g.biliary_n}/{g.n})</td>
                      <td style={{ padding: '6px 8px', color: '#ccc', fontSize: 10 }}>{GENE_INFO[g.gene]?.full?.split('/')[2]?.trim() || ''}</td>
                      <td style={{ padding: '6px 8px', color: '#90caf9', fontSize: 10 }}>{(g.key_distinctions || []).slice(0, 2).join(' · ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Atlas Stats */}
          <div style={cardStyle}>
            <h3 style={{ color: '#ce93d8', fontSize: 14, margin: '0 0 10px' }}>Cohort Statistics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {[
                { label: 'Biliary Cases', value: `${overview.biliary_total_n} (${overview.biliary_total_pct}%)` },
                { label: 'iCCA Cases', value: `${overview.icca_n} (${overview.icca_pct}%)` },
                { label: 'PARP Candidates', value: `${overview.parp_candidate_n} (${overview.parp_candidate_pct}%)` },
              ].map(s => (
                <div key={s.label} style={{ background: '#1a1a1a', borderRadius: 6, padding: 12, textAlign: 'center' }}>
                  <div style={{ color: '#ce93d8', fontSize: 18, fontWeight: 700 }}>{s.value}</div>
                  <div style={{ color: '#888', fontSize: 11 }}>{s.label}</div>
                </div>
              ))}
            </div>
          </div>
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
                    {['Gene', 'Locus', 'N', 'Mean Age', 'Biliary N', 'Biliary %', 'iCCA N', 'Seed', 'Key Distinctions'].map(h => (
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
                      <td style={{ padding: '6px 8px', color: '#ce93d8' }}>{row.biliary_n}</td>
                      <td style={{ padding: '6px 8px', color: '#ce93d8' }}>{row.biliary_pct}%</td>
                      <td style={{ padding: '6px 8px', color: '#b39ddb' }}>{row.icca_n}</td>
                      <td style={{ padding: '6px 8px', color: '#888' }}>{row.seed}</td>
                      <td style={{ padding: '6px 8px', color: '#aaa', fontSize: 10 }}>{(row.key_distinctions || []).slice(0, 2).join(' / ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {breakdown.breakdown && breakdown.breakdown.map(row => (
            <div key={row.gene} style={{ ...cardStyle, borderLeft: `4px solid ${GENE_COLORS[row.gene] || '#555'}` }}
              onClick={() => setExpandedGene(expandedGene === row.gene ? null : row.gene)}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', cursor: 'pointer' }}>
                <div>
                  <span style={{ color: GENE_COLORS[row.gene] || '#aaa', fontWeight: 700, fontSize: 14 }}>{row.gene}</span>
                  <span style={{ color: '#888', fontSize: 11, marginLeft: 12 }}>{row.locus} · n={row.n} · mean age {row.mean_age_onset}yr · biliary {row.biliary_pct}%</span>
                </div>
                <span style={{ color: '#666', fontSize: 12 }}>{expandedGene === row.gene ? '▲' : '▼'}</span>
              </div>
              {expandedGene === row.gene && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ color: '#ce93d8', fontSize: 11 }}>KEY DISTINCTIONS:</span>
                    <div style={{ color: '#ccc', fontSize: 11 }}>{(row.key_distinctions || []).join(' · ')}</div>
                  </div>
                  <div style={{ color: '#aaa', fontSize: 11, marginTop: 6 }}>
                    <span style={{ color: '#777' }}>Pathognomonic: </span>{row.pathognomonic}
                  </div>
                  <div style={{ color: '#90caf9', fontSize: 11, marginTop: 4 }}>
                    <span style={{ color: '#777' }}>Surveillance: </span>{row.surveillance_key}
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
          {overview.genes_detail && overview.genes_detail.map(g => (
            <div key={g.gene} style={{
              ...cardStyle,
              borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}`,
            }}>
              <h3 style={{ color: GENE_COLORS[g.gene] || '#aaa', margin: '0 0 8px', fontSize: 15 }}>
                {g.gene} — {GENE_INFO[g.gene]?.inh || ''}
              </h3>
              <div style={{ color: '#aaa', fontSize: 12, marginBottom: 6 }}>
                <span style={{ color: '#777' }}>Locus: </span>{GENE_INFO[g.gene]?.locus || ''}
                <span style={{ color: '#777', marginLeft: 12 }}>Size: </span>{GENE_INFO[g.gene]?.size || ''}
              </div>
              <div style={{ color: '#ccc', fontSize: 12, lineHeight: 1.6 }}>{GENE_INFO[g.gene]?.full || ''}</div>
              <div style={{ color: '#bbb', fontSize: 11, marginTop: 6 }}>{g.inheritance}</div>
              <div style={{ color: '#90caf9', fontSize: 11, marginTop: 4 }}>{g.cancer_risk}</div>
              <div style={{ color: '#ef9a9a', fontSize: 11, marginTop: 4 }}>
                <span style={{ color: '#777' }}>Pathognomonic: </span>{g.pathognomonic}
              </div>
              <div style={{ color: '#a5d6a7', fontSize: 11, marginTop: 4 }}>
                <span style={{ color: '#777' }}>Surveillance: </span>{g.surveillance_key}
              </div>
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
                  i === 0 ? '#c62828' : i === 1 ? '#1b5e20' : i === 2 ? '#4a148c' :
                  i === 3 ? '#0d47a1' : i === 4 ? '#e65100' : i === 5 ? '#880e4f' :
                  i === 6 ? '#006064' : i === 7 ? '#33691e' : '#555'
                }`,
                cursor: 'pointer',
              }}
                onClick={() => setExpandedDef(expandedDef === i ? null : i)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#ce93d8', fontWeight: 700, fontSize: 13 }}>{d.term}</span>
                  <span style={{ color: '#666', fontSize: 11 }}>{expandedDef === i ? '▲' : '▼'}</span>
                </div>
                {expandedDef === i && (
                  <div style={{ color: '#ccc', fontSize: 11, marginTop: 8, lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{d.definition}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
