'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hcc-liver-cancer-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'FAH':      '#b71c1c',  // deep red        — HT1; HCC EXTREME pediatric; succinylacetone PATHOGNOMONIC
  'ABCB11':   '#1a237e',  // deep indigo     — PFIC2; HCC/CCA children HIGHEST; BSEP-null IHC
  'HFE':      '#e65100',  // deep orange     — HH1; C282Y HCC 20-200x; phlebotomy prevents
  'ATP7B':    '#1b5e20',  // deep green      — Wilson's; Kayser-Fleischer PATHOGNOMONIC
  'SERPINA1': '#4a148c',  // deep purple     — AATD Pi*ZZ; PASD globules PATHOGNOMONIC
  'APC':      '#006064',  // dark teal       — FAP; hepatoblastoma 750-7500x HIGHEST; CHRPE
  'TSC2':     '#880e4f',  // deep magenta    — TSC; hepatic AML 75% bilateral; everolimus FDA
  'SMAD4':    '#33691e',  // deep olive      — JPS-HHT; SMAD4-null IHC PATHOGNOMONIC; hepatic AVMs
};

const GENE_INFO = {
  'FAH':      { full: 'HT1 / HCC-37%-AGE2-EXTREME / Succinylacetone-PATHOGNOMONIC / NTBC-Standard / Liver-Tx-Curative',         locus: '15q25.1',  size: '419 aa / 46 kDa',   inh: 'AR LOF' },
  'ABCB11':   { full: 'PFIC2 / HCC-CCA-Children-HIGHEST / BSEP-null-IHC-PATHOGNOMONIC / GGT-Normal / Odevixibat-IBAT',          locus: '2q31.1',   size: '1321 aa / 146 kDa', inh: 'AR LOF' },
  'HFE':      { full: 'HH1 / C282Y-HCC-20-200x-HIGHEST / Phlebotomy-Prevents-Pre-Cirrhosis / MRI-T2*-Iron / Transferrin-Sat',   locus: '6p21.3',   size: '343 aa / 37 kDa',   inh: 'AR C282Y/H63D' },
  'ATP7B':    { full: "Wilson's / Kayser-Fleischer-PATHOGNOMONIC / Slit-Lamp-MANDATORY / D-Pen-Trientine / HCC-2-5x",            locus: '13q14.3',  size: '1465 aa / 160 kDa', inh: 'AR LOF' },
  'SERPINA1': { full: 'AATD-Pi*ZZ / PASD-Globules-PATHOGNOMONIC / HCC-5-20x / Augmentation-Lung-NOT-Liver / Fazirsiran-Phase3', locus: '14q32.13', size: '418 aa / 52 kDa',   inh: 'AR Pi*ZZ' },
  'APC':      { full: 'FAP-Gardner / Hepatoblastoma-750-7500x-HIGHEST / Liver-USS-Birth-10yr / CHRPE-PATHOGNOMONIC / Colectomy', locus: '5q22.2',   size: '2843 aa / 310 kDa', inh: 'AD LOF' },
  'TSC2':     { full: 'TSC / Hepatic-AML-75%-Bilateral / Everolimus-FDA-Approved / AML-greater-3cm-Threshold / De-Novo-70%',    locus: '16p13.3',  size: '1807 aa / 200 kDa', inh: 'AD LOF' },
  'SMAD4':    { full: 'JPS-HHT / SMAD4-null-IHC-PATHOGNOMONIC / Hepatic-AVMs-Bevacizumab / Gastric-Hamartomas / CRC-39-68%',   locus: '18q21.2',  size: '552 aa / 60 kDa',   inh: 'AD LOF' },
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

export default function HereditaryHCCLiverCancerAtlas() {
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
    background: active ? '#b71c1c' : '#1a1a1a',
    color: active ? '#ffcdd2' : '#888',
    border: `1px solid ${active ? '#b71c1c' : '#333'}`,
    fontFamily: 'monospace', fontSize: 13, marginRight: 6,
  });

  if (loading) return <div style={containerStyle}><p style={{ color: '#888' }}>Loading Hereditary HCC & Primary Liver Cancer Predisposition Atlas…</p></div>;
  if (error)   return <div style={containerStyle}><p style={{ color: '#ef5350' }}>Error: {error}</p></div>;

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={{ ...cardStyle, borderLeft: '4px solid #b71c1c' }}>
        <h1 style={{ color: '#ef9a9a', fontSize: 18, margin: '0 0 6px' }}>
          🧬 Hereditary HCC &amp; Primary Liver Cancer Predisposition Atlas
        </h1>
        <div style={{ color: '#888', fontSize: 12 }}>
          Complete 8-Gene Reference · FAH-ABCB11-HFE-ATP7B-SERPINA1-APC-TSC2-SMAD4
          {overview && <span> · Seeds {overview.seed_range} · {overview.total_patients} patients (8×40)</span>}
        </div>
        <div style={{ marginTop: 8, fontSize: 11, color: '#aaa' }}>
          <Badge text="FAH HCC 37% AGE 2 EXTREME" color="#b71c1c" />
          <Badge text="ABCB11 BSEP-null IHC PATHOGNOMONIC" color="#1a237e" />
          <Badge text="HFE C282Y HCC 20-200x HIGHEST" color="#e65100" />
          <Badge text="APC Hepatoblastoma 750-7500x" color="#006064" />
          <Badge text="ATP7B KF Rings PATHOGNOMONIC" color="#1b5e20" />
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
            <h2 style={{ color: '#ef9a9a', fontSize: 15, margin: '0 0 12px' }}>
              Atlas Summary — Seeds {overview.seed_range}
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
              {[
                { label: 'Genes', value: overview.genes_n },
                { label: 'Total Patients', value: overview.total_patients },
                { label: 'Liver CA N', value: overview.liver_ca_total_n },
                { label: 'Highest Risk', value: `${overview.highest_risk_gene} ${overview.highest_risk_pct}%` },
              ].map(s => (
                <div key={s.label} style={{ background: '#1a1a1a', borderRadius: 6, padding: 12, textAlign: 'center' }}>
                  <div style={{ color: '#ef9a9a', fontSize: 20, fontWeight: 700 }}>{s.value}</div>
                  <div style={{ color: '#888', fontSize: 11 }}>{s.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Gene Risk Table */}
          <div style={cardStyle}>
            <h3 style={{ color: '#ef9a9a', fontSize: 14, margin: '0 0 10px' }}>Liver Cancer Risk by Gene</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr>
                    {['Gene', 'Locus', 'Liver CA Risk', 'Key Syndrome', 'Key Management'].map(h => (
                      <th key={h} style={{ color: '#888', padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #333' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summary && overview.gene_summary.map(g => (
                    <tr key={g.gene} style={{ borderBottom: '1px solid #1e1e1e' }}>
                      <td style={{ padding: '6px 8px', color: GENE_COLORS[g.gene] || '#aaa', fontWeight: 700 }}>{g.gene}</td>
                      <td style={{ padding: '6px 8px', color: '#aaa' }}>{g.locus}</td>
                      <td style={{ padding: '6px 8px', color: '#ef9a9a' }}>{g.liver_ca_pct}% ({g.liver_ca_n}/{g.n})</td>
                      <td style={{ padding: '6px 8px', color: '#ccc', fontSize: 10 }}>{GENE_INFO[g.gene]?.full?.split('/')[0]?.trim() || ''}</td>
                      <td style={{ padding: '6px 8px', color: '#90caf9', fontSize: 10 }}>{(g.key_distinctions || []).slice(0, 2).join(' · ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Atlas Stats */}
          <div style={cardStyle}>
            <h3 style={{ color: '#ef9a9a', fontSize: 14, margin: '0 0 10px' }}>Cohort Statistics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {[
                { label: 'Liver CA Cases', value: `${overview.liver_ca_total_n} (${overview.liver_ca_total_pct}%)` },
                { label: 'Highest Risk Gene', value: overview.highest_risk_gene },
                { label: 'Highest Risk %', value: `${overview.highest_risk_pct}%` },
              ].map(s => (
                <div key={s.label} style={{ background: '#1a1a1a', borderRadius: 6, padding: 12, textAlign: 'center' }}>
                  <div style={{ color: '#ef9a9a', fontSize: 18, fontWeight: 700 }}>{s.value}</div>
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
            <h2 style={{ color: '#ef9a9a', fontSize: 15, margin: '0 0 12px' }}>Per-Gene Cohort Summary</h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr>
                    {['Gene', 'Locus', 'N', 'Mean Age', 'Liver CA N', 'Liver CA %', 'Seed', 'Key Distinctions'].map(h => (
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
                      <td style={{ padding: '6px 8px', color: '#ef9a9a' }}>{row.liver_ca_n}</td>
                      <td style={{ padding: '6px 8px', color: '#ef9a9a' }}>{row.liver_ca_pct}%</td>
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
                  <span style={{ color: '#888', fontSize: 11, marginLeft: 12 }}>{row.locus} · n={row.n} · mean age {row.mean_age_onset}yr · liver CA {row.liver_ca_pct}%</span>
                </div>
                <span style={{ color: '#666', fontSize: 12 }}>{expandedGene === row.gene ? '▲' : '▼'}</span>
              </div>
              {expandedGene === row.gene && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ color: '#ef9a9a', fontSize: 11 }}>KEY DISTINCTIONS:</span>
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
            <h2 style={{ color: '#ef9a9a', fontSize: 16, margin: '0 0 12px' }}>
              Clinical Definitions — Seeds {definitions.seed_range}
            </h2>
            {definitions.definitions && definitions.definitions.map((d, i) => (
              <div key={i} style={{
                background: '#1a1a1a', borderRadius: 6, padding: 12, marginBottom: 10,
                borderLeft: `3px solid ${
                  i === 0 ? '#b71c1c' : i === 1 ? '#1a237e' : i === 2 ? '#e65100' :
                  i === 3 ? '#1b5e20' : i === 4 ? '#4a148c' : i === 5 ? '#006064' :
                  i === 6 ? '#880e4f' : i === 7 ? '#33691e' : '#555'
                }`,
                cursor: 'pointer',
              }}
                onClick={() => setExpandedDef(expandedDef === i ? null : i)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#ef9a9a', fontWeight: 700, fontSize: 13 }}>{d.term}</span>
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
