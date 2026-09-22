'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-crc-polyposis-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MLH1':  '#b71c1c',  // deep red      — Lynch 1; most common MMR; CRC+endometrial
  'MSH2':  '#1565c0',  // deep blue     — Lynch 2; Muir-Torre sebaceous; EPCAM silencing
  'MSH6':  '#2e7d32',  // deep green    — Lynch 5; endometrial predominant; MSI-L possible
  'PMS2':  '#e65100',  // deep orange   — Lynch 4; lowest penetrance; CMMRD biallelic
  'APC':   '#6a1b9a',  // deep purple   — FAP; 100% CRC untreated; prophylactic colectomy
  'MUTYH': '#006064',  // dark cyan     — MAP; AR biallelic; Y179C+G396D founders
  'STK11': '#4a148c',  // deep violet   — PJS; lentigines perioral; pancreatic 132x risk
  'SMAD4': '#1b5e20',  // forest green  — JPS; JPS-HHT overlap; echo bubble mandatory
};

const GENE_INFO = {
  'MLH1':  { full: 'Lynch Syndrome 1 / dMMR',                  locus: '3p22.2',  size: '852 aa / 84 kDa',   inh: 'AD LOF' },
  'MSH2':  { full: 'Lynch Syndrome 2 / Muir-Torre / Turcot',   locus: '2p21',    size: '934 aa / 105 kDa',  inh: 'AD LOF' },
  'MSH6':  { full: 'Lynch Syndrome 5 / Endometrial-dominant',  locus: '2p16.3',  size: '1360 aa / 160 kDa', inh: 'AD LOF' },
  'PMS2':  { full: 'Lynch Syndrome 4 / CMMRD (biallelic)',      locus: '7p22.1',  size: '862 aa / 96 kDa',   inh: 'AD LOF (biallelic = CMMRD)' },
  'APC':   { full: 'FAP / AFAP / Gardner / Turcot',            locus: '5q22.2',  size: '2843 aa / 311 kDa', inh: 'AD LOF' },
  'MUTYH': { full: 'MAP (biallelic) / CRC risk (monoallelic)', locus: '1p34.1',  size: '546 aa / 60 kDa',   inh: 'AR (biallelic required for MAP)' },
  'STK11': { full: 'Peutz-Jeghers Syndrome',                   locus: '19p13.3', size: '433 aa / 50 kDa',   inh: 'AD LOF' },
  'SMAD4': { full: 'Juvenile Polyposis / JPS-HHT Overlap',     locus: '18q21.2', size: '552 aa / 60 kDa',   inh: 'AD LOF' },
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

export default function HereditoryCRCPolypisisAtlas() {
  const [tab, setTab]                   = useState('Overview');
  const [overview, setOverview]         = useState(null);
  const [breakdown, setBreakdown]       = useState(null);
  const [definitions, setDefinitions]   = useState(null);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState(null);
  const [expandedGene, setExpandedGene] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const ep = tab === 'Definitions' ? 'definitions' : tab === 'Overview' ? 'overview' : 'breakdown';
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(data => {
        if (tab === 'Overview') setOverview(data);
        else if (tab === 'Definitions') setDefinitions(data);
        else setBreakdown(data);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  const card = (style = {}) => ({
    background: '#1e1e2e', border: '1px solid #333', borderRadius: 8,
    padding: 16, marginBottom: 16, ...style,
  });

  return (
    <div style={{ fontFamily: 'monospace', background: '#0d0d1a', color: '#e0e0e0', minHeight: '100vh', padding: 24 }}>
      {/* Header */}
      <div style={card({ background: '#12122a', borderColor: '#444' })}>
        <h1 style={{ color: '#64b5f6', fontSize: 20, margin: '0 0 6px' }}>
          🧬 Hereditary-CRC-Polyposis-Atlas
        </h1>
        <div style={{ color: '#aaa', fontSize: 12, marginBottom: 10 }}>
          Complete 8-Gene Colorectal Cancer &amp; Polyposis Syndrome Atlas ·
          MLH1-MSH2-MSH6-PMS2-APC-MUTYH-STK11-SMAD4 ·
          320-Patient Aggregate · 8×40 · Seeds 3086–3093
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {Object.entries(GENE_COLORS).map(([g, c]) => (
            <span key={g} style={{
              background: c + '22', color: c, border: `1px solid ${c}55`,
              borderRadius: 4, padding: '3px 10px', fontSize: 12, fontWeight: 700,
            }}>{g}</span>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#1565c0' : '#1e1e2e',
            color: tab === t ? '#fff' : '#aaa',
            border: '1px solid ' + (tab === t ? '#1565c0' : '#333'),
            borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#64b5f6', padding: 20 }}>Loading...</div>}
      {error && <div style={{ color: '#ef5350', padding: 20 }}>Error: {error}</div>}

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))', gap: 12, marginBottom: 16 }}>
            {[
              ['Atlas', overview.atlas],
              ['Genes', overview.total_genes],
              ['Patients', overview.total_patients],
              ['Seeds', overview.seed_range],
            ].map(([k, v]) => (
              <div key={k} style={card({ textAlign: 'center' })}>
                <div style={{ color: '#888', fontSize: 11 }}>{k}</div>
                <div style={{ color: '#64b5f6', fontSize: 15, fontWeight: 700, marginTop: 4 }}>{v}</div>
              </div>
            ))}
          </div>

          {/* Gene loci */}
          <div style={card()}>
            <h3 style={{ color: '#81c784', margin: '0 0 10px', fontSize: 14 }}>Gene Loci &amp; Inheritance</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: 8 }}>
              {Object.entries(overview.inheritance_modes || {}).map(([gene, mode]) => (
                <div key={gene} style={{
                  background: '#151525', border: `1px solid ${GENE_COLORS[gene] || '#444'}44`,
                  borderRadius: 6, padding: 10,
                }}>
                  <div style={{ color: GENE_COLORS[gene] || '#aaa', fontWeight: 700, fontSize: 13, marginBottom: 4 }}>
                    {gene} <span style={{ color: '#888', fontWeight: 400, fontSize: 11 }}>{overview.gene_loci?.[gene]}</span>
                  </div>
                  <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.5 }}>{mode}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Key rules */}
          <div style={card()}>
            <h3 style={{ color: '#ffb74d', margin: '0 0 10px', fontSize: 14 }}>⚠️ Key Clinical Rules</h3>
            {(overview.key_clinical_rules || []).map((r, i) => (
              <div key={i} style={{
                background: '#1a1a2e', borderLeft: '3px solid #ffb74d',
                padding: '6px 10px', marginBottom: 6, fontSize: 12, color: '#e0e0e0',
              }}>{r}</div>
            ))}
          </div>

          {/* Panel note */}
          {overview.gene_panel_note && (
            <div style={card({ borderColor: '#1565c044' })}>
              <h3 style={{ color: '#64b5f6', margin: '0 0 8px', fontSize: 13 }}>Gene Panel &amp; Surveillance Note</h3>
              <div style={{ color: '#bbb', fontSize: 12, lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                {overview.gene_panel_note}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1a1a2e', color: '#888' }}>
                  {['Gene','Locus','N','CRC%','Endometrial%','MSI-H%','Polyps>100%','Duodenal%','Desmoid%','CHRPE%',
                    'Lentigines%','Pancreatic%','Juvenile Polyps%','HHT%','PAVM%','Mean Age Dx','Disease'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #333', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.genes.map(g => (
                  <tr key={g.gene} style={{ borderBottom: '1px solid #222' }}
                    onMouseEnter={e => e.currentTarget.style.background = '#1a1a2e'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                    <td style={{ padding: '7px 10px', color: GENE_COLORS[g.gene] || '#aaa', fontWeight: 700 }}>{g.gene}</td>
                    <td style={{ padding: '7px 10px', color: '#888' }}>{g.locus}</td>
                    <td style={{ padding: '7px 10px' }}>{g.n}</td>
                    <td style={{ padding: '7px 10px', color: g.crc_pct > 50 ? '#ef5350' : '#aaa' }}>{g.crc_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.endometrial_pct > 30 ? '#ff7043' : '#aaa' }}>{g.endometrial_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.msi_h_pct > 80 ? '#66bb6a' : '#aaa' }}>{g.msi_h_pct}%</td>
                    <td style={{ padding: '7px 10px' }}>{g.polyposis_gt100_pct}%</td>
                    <td style={{ padding: '7px 10px' }}>{g.duodenal_adenoma_pct}%</td>
                    <td style={{ padding: '7px 10px' }}>{g.desmoid_pct}%</td>
                    <td style={{ padding: '7px 10px' }}>{g.chrpe_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.lentigines_pct > 50 ? '#ba68c8' : '#aaa' }}>{g.lentigines_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.pancreatic_cancer_pct > 10 ? '#ef5350' : '#aaa' }}>{g.pancreatic_cancer_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.juvenile_polyps_pct > 50 ? '#4db6ac' : '#aaa' }}>{g.juvenile_polyps_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.hht_features_pct > 10 ? '#ff7043' : '#aaa' }}>{g.hht_features_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.pulmonary_avm_pct > 10 ? '#ef5350' : '#aaa' }}>{g.pulmonary_avm_pct}%</td>
                    <td style={{ padding: '7px 10px', color: '#64b5f6' }}>{g.mean_age_dx_yrs} yr</td>
                    <td style={{ padding: '7px 10px', color: '#888', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {g.disease_category}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {breakdown.genes.map(g => (
            <div key={g.gene} style={card({ borderColor: (GENE_COLORS[g.gene] || '#444') + '55' })}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', cursor: 'pointer' }}
                onClick={() => setExpandedGene(expandedGene === g.gene ? null : g.gene)}>
                <div>
                  <span style={{ color: GENE_COLORS[g.gene] || '#aaa', fontWeight: 700, fontSize: 16 }}>{g.gene}</span>
                  <span style={{ color: '#888', fontSize: 12, marginLeft: 10 }}>{g.locus}</span>
                  <span style={{ color: '#666', fontSize: 11, marginLeft: 8 }}>{GENE_INFO[g.gene]?.size}</span>
                  <div style={{ marginTop: 4 }}>
                    <Badge text={GENE_INFO[g.gene]?.inh || 'AD'} color={GENE_COLORS[g.gene] || '#aaa'} />
                    <Badge text={g.disease_category} color="#888" />
                  </div>
                </div>
                <span style={{ color: '#555', fontSize: 18 }}>{expandedGene === g.gene ? '▲' : '▼'}</span>
              </div>

              {/* Quick stats */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 10 }}>
                {[
                  ['CRC', g.crc_pct + '%', g.crc_pct > 50 ? '#ef5350' : '#aaa'],
                  ['Endometrial', g.endometrial_pct + '%', g.endometrial_pct > 30 ? '#ff7043' : '#aaa'],
                  ['MSI-H', g.msi_h_pct + '%', '#66bb6a'],
                  ['Polyps>100', g.polyposis_gt100_pct + '%', '#ba68c8'],
                  ['Duodenal', g.duodenal_adenoma_pct + '%', '#64b5f6'],
                  ['Lentigines', g.lentigines_pct + '%', '#ce93d8'],
                  ['Pancreatic Ca', g.pancreatic_cancer_pct + '%', g.pancreatic_cancer_pct > 10 ? '#ef5350' : '#aaa'],
                  ['Desmoid', g.desmoid_pct + '%', '#ffb74d'],
                  ['HHT', g.hht_features_pct + '%', g.hht_features_pct > 10 ? '#ff7043' : '#aaa'],
                  ['PAVM', g.pulmonary_avm_pct + '%', g.pulmonary_avm_pct > 15 ? '#ef5350' : '#aaa'],
                  ['Mean Age Dx', g.mean_age_dx_yrs + 'yr', '#64b5f6'],
                ].map(([label, val, col]) => (
                  <div key={label} style={{
                    background: '#151525', border: '1px solid #333', borderRadius: 6,
                    padding: '4px 10px', textAlign: 'center',
                  }}>
                    <div style={{ color: '#666', fontSize: 10 }}>{label}</div>
                    <div style={{ color: col, fontSize: 13, fontWeight: 700 }}>{val}</div>
                  </div>
                ))}
              </div>

              {/* Clinical note */}
              <div style={{
                marginTop: 10, background: '#12122a', borderLeft: `3px solid ${GENE_COLORS[g.gene] || '#444'}`,
                padding: '8px 12px', fontSize: 12, color: '#ccc', lineHeight: 1.6,
              }}>
                {g.clinical_note}
              </div>

              {/* Expanded detail */}
              {expandedGene === g.gene && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ color: '#aaa', fontSize: 12, lineHeight: 1.7, marginBottom: 10 }}>
                    <strong style={{ color: '#81c784' }}>Protein:</strong> {g.protein}
                  </div>
                  <div style={{ color: '#aaa', fontSize: 12, lineHeight: 1.7, marginBottom: 10 }}>
                    <strong style={{ color: '#81c784' }}>Inheritance detail:</strong> {g.inheritance}
                  </div>
                  {g.sample_mutations?.length > 0 && (
                    <div>
                      <div style={{ color: '#888', fontSize: 11, marginBottom: 4 }}>Sample mutations observed:</div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {g.sample_mutations.map((m, i) => (
                          <span key={i} style={{
                            background: '#1a1a2e', border: '1px solid #333',
                            borderRadius: 4, padding: '2px 8px', fontSize: 11, color: '#aaa',
                          }}>{m}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {/* Severity */}
                  <div style={{ marginTop: 12 }}>
                    <div style={{ color: '#888', fontSize: 11, marginBottom: 6 }}>Severity distribution:</div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      {[['Severe', g.severe_pct, '#ef5350'], ['Moderate', g.moderate_pct, '#ffb74d'], ['Mild', g.mild_pct, '#66bb6a']].map(([s, v, c]) => (
                        <div key={s} style={{ textAlign: 'center', background: '#151525', border: `1px solid ${c}44`, borderRadius: 6, padding: '4px 12px' }}>
                          <div style={{ color: c, fontSize: 13, fontWeight: 700 }}>{v}%</div>
                          <div style={{ color: '#888', fontSize: 10 }}>{s}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ color: '#888', fontSize: 12, marginBottom: 12 }}>
            {definitions.count} clinical definitions
          </div>
          {definitions.definitions.map((d, i) => (
            <div key={i} style={card()}>
              <div style={{ color: '#64b5f6', fontWeight: 700, fontSize: 13, marginBottom: 8 }}>
                {d.term}
              </div>
              <div style={{ color: '#ccc', fontSize: 12, lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                {d.definition}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
