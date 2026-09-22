'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-head-neck-cancer-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'FANCA':  '#b71c1c',  // deep red       — FA-A; oral/oropharyngeal SCC 700x PATHOGNOMONIC; AVOID RADIATION ABSOLUTELY
  'XPC':    '#e65100',  // deep orange    — XP-C; UV-SCC face 10,000x PATHOGNOMONIC; no neurological; UV avoidance
  'TP53':   '#c62828',  // brick red      — LFS; HNC laryngeal/oral; AVOID RADIATION ABSOLUTELY; WBMRI Toronto
  'CDKN2A': '#1b5e20',  // deep green     — FAMM; oral SCC 10-30x; pancreatic 20x PATHOGNOMONIC; melanoma 25-36x
  'ATM':    '#0d47a1',  // deep blue      — A-T; radiosensitivity PATHOGNOMONIC biallelic; HNC 3-5x monoallelic
  'MSH2':   '#4a148c',  // deep purple    — Lynch/Muir-Torre; sebaceous neoplasms face PATHOGNOMONIC; EPCAM
  'BRCA2':  '#006064',  // dark teal      — HBOC; HNC 2-3x; PARP inhibitor olaparib; platinum-sensitive
  'RECQL4': '#37474f',  // blue-grey      — Rothmund-Thomson; poikiloderma congenitale PATHOGNOMONIC; osteosarcoma
};

const GENE_INFO = {
  'FANCA':  { full: 'FA-A Most-Common-60-70% / Oral-Oropharyngeal-SCC-700x-PATHOGNOMONIC / Age-26yr / AVOID-RADIATION-ABSOLUTELY / BMT',           locus: '16q24.3', size: '1455 aa / 163 kDa', inh: 'AR biallelic LOF' },
  'XPC':    { full: 'XP-C Most-Common-XP / UV-SCC-Face-Scalp-10000x-PATHOGNOMONIC / No-Neurological / Strict-UV-Avoidance-MANDATORY',              locus: '3p25.1',  size: '940 aa / 106 kDa',  inh: 'AR biallelic LOF' },
  'TP53':   { full: 'LFS / HNC-Laryngeal-Oral / AVOID-RADIATION-ABSOLUTELY / WBMRI-Toronto / Surgery+Chemo-Only',                                  locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'CDKN2A': { full: 'FAMM / Oral-Oropharyngeal-SCC-10-30x / Pancreatic-20x-PATHOGNOMONIC / Melanoma-25-36x / CDK4-6i-Emerging',                   locus: '9p21.3',  size: '156 aa (p16)',       inh: 'AD LOF' },
  'ATM':    { full: 'A-T / Cerebellar-Ataxia+Telangiectasias-PATHOGNOMONIC / Radiosensitivity-Absolute-Biallelic / HNC-3-5x-Monoallelic',          locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AR/AD LOF' },
  'MSH2':   { full: 'Lynch-Type2 / Muir-Torre-Sebaceous-Face-PATHOGNOMONIC / Urothelial-15-25% / EPCAM-3prime-Deletion / Aspirin-CAPP2',           locus: '2p21',    size: '934 aa / 105 kDa',  inh: 'AD LOF' },
  'BRCA2':  { full: 'HBOC / HNC-Oropharyngeal-2-3x / PARP-Olaparib-FDA / Platinum-Sensitive / Biallelic-FA-D1',                                   locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF' },
  'RECQL4': { full: 'Rothmund-Thomson / Poikiloderma-Congenitale-PATHOGNOMONIC / SCC-Face-Head / Osteosarcoma-30-40x / RTS-BGS-RAPADILINO',        locus: '8q24.12', size: '1208 aa / 133 kDa', inh: 'AR biallelic LOF' },
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

export default function HereditaryHeadNeckCancerAtlas() {
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
    background: active ? '#1e3a5f' : '#1a1a1a',
    color: active ? '#64b5f6' : '#888',
    border: `1px solid ${active ? '#1e3a5f' : '#333'}`,
    fontFamily: 'monospace', fontSize: 13, marginRight: 6,
  });

  if (loading) return <div style={containerStyle}><p style={{ color: '#888' }}>Loading Hereditary Head & Neck Cancer Atlas…</p></div>;
  if (error)   return <div style={containerStyle}><p style={{ color: '#ef5350' }}>Error: {error}</p></div>;

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={cardStyle}>
        <h1 style={{ color: '#ef9a9a', margin: 0, fontSize: 20 }}>
          🧬 Hereditary Head &amp; Neck Cancer Predisposition Atlas
        </h1>
        <p style={{ color: '#aaa', margin: '8px 0 0', fontSize: 13 }}>
          Complete 8-Gene Reference · FANCA · XPC · TP53 · CDKN2A · ATM · MSH2 · BRCA2 · RECQL4 ·
          320 Patients (8×40) · Seeds 3158-3165
        </p>
        <div style={{ marginTop: 10 }}>
          <Badge text="FANCA: SCC 700× PATHOGNOMONIC"    color="#b71c1c" />
          <Badge text="XPC: UV-SCC Face 10,000×"         color="#e65100" />
          <Badge text="TP53: AVOID RT ABSOLUTELY"         color="#c62828" />
          <Badge text="CDKN2A: Pancreatic 20× PATHOGNOMONIC" color="#1b5e20" />
          <Badge text="ATM: Radiosensitivity PATHOGNOMONIC" color="#0d47a1" />
          <Badge text="MSH2: Sebaceous Face PATHOGNOMONIC" color="#4a148c" />
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
            <h2 style={{ color: '#ef9a9a', fontSize: 16, margin: '0 0 12px' }}>Atlas Summary</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
              {[
                { label: 'Genes', value: overview.total_genes },
                { label: 'Patients', value: overview.total_patients },
                { label: 'Seed Range', value: overview.seed_range },
                { label: 'Cancer Type', value: 'Head & Neck' },
              ].map(({ label, value }) => (
                <div key={label} style={{ background: '#1a1a1a', borderRadius: 6, padding: 12, textAlign: 'center' }}>
                  <div style={{ color: '#ef9a9a', fontSize: 20, fontWeight: 700 }}>{value}</div>
                  <div style={{ color: '#888', fontSize: 11 }}>{label}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={cardStyle}>
            <h2 style={{ color: '#ef9a9a', fontSize: 16, margin: '0 0 12px' }}>8 Genes — Inheritance &amp; Cancer Risk</h2>
            {overview.genes && overview.genes.map(gene => (
              <div key={gene} style={{
                background: '#1a1a1a', borderRadius: 6, padding: 12, marginBottom: 8,
                borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ color: GENE_COLORS[gene] || '#aaa', fontWeight: 700, minWidth: 80 }}>{gene}</span>
                  <span style={{ color: '#bbb', fontSize: 12 }}>{GENE_INFO[gene]?.inh || ''}</span>
                  <span style={{ color: '#777', fontSize: 11 }}>{GENE_INFO[gene]?.locus || ''}</span>
                  <span style={{ color: '#666', fontSize: 11 }}>{GENE_INFO[gene]?.size || ''}</span>
                </div>
                <div style={{ color: '#aaa', fontSize: 12, marginTop: 4 }}>
                  {GENE_INFO[gene]?.full || ''}
                </div>
                <div style={{ color: '#888', fontSize: 11, marginTop: 4 }}>
                  {overview.inheritance_modes?.[gene] || ''}
                </div>
              </div>
            ))}
          </div>

          <div style={cardStyle}>
            <h2 style={{ color: '#ef9a9a', fontSize: 16, margin: '0 0 12px' }}>Key Clinical Rules</h2>
            {overview.key_clinical_rules && overview.key_clinical_rules.map((rule, i) => (
              <div key={i} style={{
                background: '#1a1a1a', borderRadius: 4, padding: '8px 12px', marginBottom: 6,
                borderLeft: `3px solid ${i % 8 === 0 ? '#b71c1c' : i % 8 === 1 ? '#e65100' : i % 8 === 2 ? '#c62828' : i % 8 === 3 ? '#1b5e20' : i % 8 === 4 ? '#0d47a1' : i % 8 === 5 ? '#4a148c' : i % 8 === 6 ? '#006064' : '#37474f'}`,
                fontSize: 12, color: '#ccc',
              }}>
                {rule}
              </div>
            ))}
          </div>

          <div style={cardStyle}>
            <h2 style={{ color: '#ef9a9a', fontSize: 16, margin: '0 0 12px' }}>Gene Panel Note</h2>
            <pre style={{ color: '#bbb', fontSize: 11, whiteSpace: 'pre-wrap', margin: 0 }}>
              {overview.gene_panel_note}
            </pre>
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <div style={cardStyle}>
            <h2 style={{ color: '#ef9a9a', fontSize: 16, margin: '0 0 12px' }}>
              Per-Gene Breakdown — {breakdown.total_patients} Patients ({breakdown.n_genes} × 40)
            </h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1e1e1e' }}>
                    {['Gene', 'Locus', 'n', 'Mean Age', 'HNC %', 'Key Stat 1', 'Key Stat 2', 'Inheritance'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#ef9a9a', borderBottom: '1px solid #333' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.genes && breakdown.genes.map((g, i) => {
                    const statKeys = Object.keys(g).filter(k => k.endsWith('_pct') && k !== 'radiation_avoid_pct');
                    const s1 = statKeys[0] ? `${statKeys[0].replace(/_pct$/, '').replace(/_/g, ' ')}: ${g[statKeys[0]]}%` : '—';
                    const s2 = statKeys[1] ? `${statKeys[1].replace(/_pct$/, '').replace(/_/g, ' ')}: ${g[statKeys[1]]}%` : '—';
                    return (
                      <tr key={g.gene} style={{ background: i % 2 === 0 ? '#111' : '#151515' }}>
                        <td style={{ padding: '7px 10px', color: GENE_COLORS[g.gene] || '#aaa', fontWeight: 700 }}>{g.gene}</td>
                        <td style={{ padding: '7px 10px', color: '#888' }}>{g.locus}</td>
                        <td style={{ padding: '7px 10px', color: '#aaa' }}>{g.n}</td>
                        <td style={{ padding: '7px 10px', color: '#aaa' }}>{g.mean_age_diagnosis}</td>
                        <td style={{ padding: '7px 10px', color: '#ef9a9a', fontWeight: 600 }}>{g.hnc_pct != null ? `${g.hnc_pct}%` : '—'}</td>
                        <td style={{ padding: '7px 10px', color: '#bbb', fontSize: 11 }}>{s1}</td>
                        <td style={{ padding: '7px 10px', color: '#bbb', fontSize: 11 }}>{s2}</td>
                        <td style={{ padding: '7px 10px', color: '#999', fontSize: 11 }}>{g.inheritance?.slice(0, 60)}…</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Gene detail cards */}
          <div style={cardStyle}>
            <h2 style={{ color: '#ef9a9a', fontSize: 16, margin: '0 0 12px' }}>Gene Detail Cards</h2>
            {breakdown.genes && breakdown.genes.map(g => (
              <div key={g.gene} style={{
                background: '#1a1a1a', borderRadius: 6, padding: 14, marginBottom: 10,
                borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}`,
              }}>
                <div
                  style={{ display: 'flex', justifyContent: 'space-between', cursor: 'pointer' }}
                  onClick={() => setExpandedGene(expandedGene === g.gene ? null : g.gene)}
                >
                  <span style={{ color: GENE_COLORS[g.gene] || '#aaa', fontWeight: 700, fontSize: 15 }}>{g.gene}</span>
                  <span style={{ color: '#666', fontSize: 12 }}>{expandedGene === g.gene ? '▲' : '▼'} expand</span>
                </div>
                <div style={{ color: '#bbb', fontSize: 12, marginTop: 4 }}>
                  {g.locus} · n={g.n} · mean age {g.mean_age_diagnosis}yr
                  {g.hnc_pct != null && <> · HNC {g.hnc_pct}%</>}
                </div>
                {expandedGene === g.gene && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ marginBottom: 8 }}>
                      <span style={{ color: '#ef9a9a', fontSize: 11 }}>PATHOGNOMONIC:</span>
                      <div style={{ color: '#ccc', fontSize: 12 }}>{g.pathognomonic}</div>
                    </div>
                    <div style={{ marginBottom: 8 }}>
                      <span style={{ color: '#ef9a9a', fontSize: 11 }}>SURVEILLANCE:</span>
                      <div style={{ color: '#ccc', fontSize: 12 }}>{g.surveillance_key}</div>
                    </div>
                    <div style={{ marginBottom: 8 }}>
                      <span style={{ color: '#ef9a9a', fontSize: 11 }}>INHERITANCE:</span>
                      <div style={{ color: '#bbb', fontSize: 11 }}>{g.inheritance}</div>
                    </div>
                    <div>
                      <span style={{ color: '#ef9a9a', fontSize: 11 }}>PROTEIN:</span>
                      <div style={{ color: '#999', fontSize: 11 }}>{g.protein_summary}</div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
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
                borderLeft: `3px solid ${i === 0 ? '#b71c1c' : i === 1 ? '#e65100' : i === 2 ? '#1b5e20' : i === 3 ? '#0d47a1' : i === 4 ? '#4a148c' : i === 5 ? '#006064' : i === 6 ? '#37474f' : '#555'}`,
              }}>
                <div
                  style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between' }}
                  onClick={() => setExpandedDef(expandedDef === i ? null : i)}
                >
                  <span style={{ color: '#ef9a9a', fontSize: 13, fontWeight: 600 }}>{d.term}</span>
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
