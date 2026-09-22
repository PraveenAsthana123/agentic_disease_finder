'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-gastric-cancer-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'CDH1':   '#b71c1c',  // deep red       — HDGC; signet-ring cell; prophylactic gastrectomy
  'CTNNA1': '#1565c0',  // deep blue      — HDGC without CDH1; alpha-E-catenin
  'BRCA2':  '#2e7d32',  // deep green     — HBOC; gastric 3-7x; PARP inhibitors
  'PALB2':  '#e65100',  // deep orange    — PALB2/FANCN; breast+pancreatic+gastric
  'ATM':    '#6a1b9a',  // deep purple    — ATM; gastric 2-4x; avoid radiation; HRD
  'TP53':   '#006064',  // dark cyan      — Li-Fraumeni; avoid radiation; WBMRI mandatory
  'RNF43':  '#4a148c',  // deep violet    — Wnt E3 ligase; gastric serrated polyposis
  'POLE':   '#1b5e20',  // forest green   — ultra-hypermutated; TMB-high MSS; immunotherapy
};

const GENE_INFO = {
  'CDH1':   { full: 'Hereditary Diffuse Gastric Cancer / Lobular Breast',  locus: '16q22.1', size: '784 aa / 87 kDa',   inh: 'AD LOF' },
  'CTNNA1': { full: 'HDGC without CDH1 / Alpha-Catenin Deficiency',        locus: '5q31.3',  size: '906 aa / 100 kDa',  inh: 'AD LOF' },
  'BRCA2':  { full: 'HBOC / Gastric / Fanconi FANCD1',                     locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF' },
  'PALB2':  { full: 'PALB2 Hereditary Breast/Pancreatic/Gastric / FANCN',  locus: '16p12.2', size: '1186 aa / 131 kDa', inh: 'AD LOF' },
  'ATM':    { full: 'ATM Heterozygote Hereditary Cancer',                   locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD LOF' },
  'TP53':   { full: 'Li-Fraumeni Syndrome / Gastric-Sarcoma-Breast-Brain', locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'RNF43':  { full: 'Gastric Serrated Polyposis / Wnt-Pathway',            locus: '17q22',   size: '783 aa / 88 kDa',   inh: 'AD LOF' },
  'POLE':   { full: 'POLE Ultra-Hypermutated / TMB-High Immunotherapy',    locus: '12q24.33',size: '2286 aa / 261 kDa', inh: 'AD GOF exonuclease' },
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

export default function HereditaryGastricCancerAtlas() {
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
          🧬 Hereditary-Gastric-Cancer-Atlas
        </h1>
        <div style={{ color: '#aaa', fontSize: 12, marginBottom: 10 }}>
          Complete 8-Gene Hereditary Gastric Cancer Predisposition Atlas ·
          CDH1-CTNNA1-BRCA2-PALB2-ATM-TP53-RNF43-POLE ·
          320-Patient Aggregate · 8×40 · Seeds 3094–3101
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
                  {['Gene','Locus','N','Gastric%','LobularBr%','Breast%','Ovarian%','Pancreatic%',
                    'SigRing%','Intestinal%','GastricPolyps%','ProphGastrect%','PARP%','TMB-High%',
                    'Immuno%','WBMRI%','Mean Age Dx','Disease'].map(h => (
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
                    <td style={{ padding: '7px 10px', color: g.gastric_cancer_pct > 30 ? '#ef5350' : '#aaa' }}>{g.gastric_cancer_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.lobular_breast_pct > 30 ? '#ff7043' : '#aaa' }}>{g.lobular_breast_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.breast_cancer_pct > 30 ? '#ff7043' : '#aaa' }}>{g.breast_cancer_pct}%</td>
                    <td style={{ padding: '7px 10px' }}>{g.ovarian_cancer_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.pancreatic_cancer_pct > 5 ? '#ef5350' : '#aaa' }}>{g.pancreatic_cancer_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.signet_ring_pct > 50 ? '#ce93d8' : '#aaa' }}>{g.signet_ring_pct}%</td>
                    <td style={{ padding: '7px 10px' }}>{g.intestinal_histology_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.gastric_polyps_pct > 50 ? '#4db6ac' : '#aaa' }}>{g.gastric_polyps_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.prophylactic_gastrectomy_pct > 30 ? '#66bb6a' : '#aaa' }}>{g.prophylactic_gastrectomy_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.parp_eligible_pct > 40 ? '#64b5f6' : '#aaa' }}>{g.parp_eligible_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.tmb_high_pct > 80 ? '#a5d6a7' : '#aaa' }}>{g.tmb_high_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.immunotherapy_response_pct > 60 ? '#66bb6a' : '#aaa' }}>{g.immunotherapy_response_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.wbmri_surveillance_pct > 50 ? '#4db6ac' : '#aaa' }}>{g.wbmri_surveillance_pct}%</td>
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
                  ['Gastric Ca', g.gastric_cancer_pct + '%', g.gastric_cancer_pct > 30 ? '#ef5350' : '#aaa'],
                  ['Lobular Br', g.lobular_breast_pct + '%', g.lobular_breast_pct > 30 ? '#ff7043' : '#aaa'],
                  ['Breast Ca', g.breast_cancer_pct + '%', g.breast_cancer_pct > 30 ? '#ff7043' : '#aaa'],
                  ['Ovarian', g.ovarian_cancer_pct + '%', '#ce93d8'],
                  ['Pancreatic', g.pancreatic_cancer_pct + '%', g.pancreatic_cancer_pct > 5 ? '#ef5350' : '#aaa'],
                  ['Signet Ring', g.signet_ring_pct + '%', g.signet_ring_pct > 50 ? '#ba68c8' : '#aaa'],
                  ['Gastric Polyps', g.gastric_polyps_pct + '%', '#4db6ac'],
                  ['Prophyl Gastrect', g.prophylactic_gastrectomy_pct + '%', '#66bb6a'],
                  ['PARP Eligible', g.parp_eligible_pct + '%', '#64b5f6'],
                  ['TMB-High', g.tmb_high_pct + '%', g.tmb_high_pct > 80 ? '#a5d6a7' : '#aaa'],
                  ['Immuno Resp', g.immunotherapy_response_pct + '%', g.immunotherapy_response_pct > 60 ? '#66bb6a' : '#aaa'],
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
