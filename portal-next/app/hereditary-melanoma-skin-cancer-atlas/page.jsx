'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-melanoma-skin-cancer-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'CDKN2A': '#b71c1c',  // deep red       — FAMMM; p16/p14ARF; pancreatic 17x RR
  'CDK4':   '#1565c0',  // deep blue      — CDK4 R24C/H; p16-resistant GOF
  'BAP1':   '#2e7d32',  // deep green     — BAP1-TPDS; uveal melanoma; BAPomas
  'PTCH1':  '#e65100',  // deep orange    — Gorlin/BCNS; BCC; avoid radiation
  'SUFU':   '#6a1b9a',  // deep purple    — Gorlin-like; SHH medulloblastoma
  'MITF':   '#006064',  // dark cyan      — E318K; melanoma-astrocytoma; RCC
  'POT1':   '#4a148c',  // deep violet    — shelterin; familial melanoma; glioma
  'RB1':    '#1b5e20',  // forest green   — hereditary RB; bilateral; secondary sarcoma
};

const GENE_INFO = {
  'CDKN2A': { full: 'FAMMM / Familial Melanoma / Pancreatic 17x RR',      locus: '9p21.3',   size: '156aa(p16)/132aa(p14)',  inh: 'AD LOF' },
  'CDK4':   { full: 'FAMMM Type 2 / p16-Resistant CDK4 R24C/H',           locus: '12q14.1',  size: '303 aa / 33 kDa',         inh: 'AD GOF R24C/H' },
  'BAP1':   { full: 'BAP1-TPDS / Uveal Melanoma 50% / BAPomas',           locus: '3p21.1',   size: '729 aa / 80 kDa',         inh: 'AD LOF' },
  'PTCH1':  { full: 'Gorlin Syndrome BCNS / BCC / OKC / Avoid Radiation', locus: '9q22.32',  size: '1447 aa / 160 kDa',       inh: 'AD LOF' },
  'SUFU':   { full: 'Gorlin-Like SHH / Medulloblastoma >10%',             locus: '10q24.32', size: '484 aa / 54 kDa',         inh: 'AD LOF' },
  'MITF':   { full: 'Melanoma-Astrocytoma Syndrome / E318K / RCC',        locus: '3p14.1',   size: '526 aa / 59 kDa',         inh: 'AD GOF E318K' },
  'POT1':   { full: 'Familial Melanoma 3-4% / Shelterin / Glioma',        locus: '7q31.33',  size: '634 aa / 70 kDa',         inh: 'AD LOF' },
  'RB1':    { full: 'Hereditary Retinoblastoma / Secondary Sarcoma 30-40%', locus: '13q14.2', size: '928 aa / 110 kDa',        inh: 'AD LOF' },
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

export default function HereditaryMelanomaSkinCancerAtlas() {
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
          🧬 Hereditary-Melanoma-Skin-Cancer-Atlas
        </h1>
        <div style={{ color: '#aaa', fontSize: 12, marginBottom: 10 }}>
          Complete 8-Gene Hereditary Melanoma &amp; Skin Cancer Predisposition Atlas ·
          CDKN2A-CDK4-BAP1-PTCH1-SUFU-MITF-POT1-RB1 ·
          320-Patient Aggregate · 8×40 · Seeds 3102–3109
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

      {loading && <div style={{ color: '#64b5f6', padding: 20 }}>Loading…</div>}
      {error && <div style={{ color: '#ef5350', padding: 20 }}>Error: {error}</div>}

      {/* ── Overview ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={card()}>
            <div style={{ color: '#81c784', fontWeight: 700, marginBottom: 8 }}>Atlas Summary</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8 }}>
              {[
                ['Genes', overview.total_genes],
                ['Patients', overview.total_patients],
                ['Seeds', overview.seed_range],
                ['Categories', '8 Distinct Syndromes'],
              ].map(([k, v]) => (
                <div key={k} style={{ background: '#12122a', borderRadius: 6, padding: '10px 14px' }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>{k}</div>
                  <div style={{ color: '#64b5f6', fontSize: 16, fontWeight: 700 }}>{v}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={card()}>
            <div style={{ color: '#81c784', fontWeight: 700, marginBottom: 10 }}>Inheritance Modes</div>
            {Object.entries(overview.inheritance_modes || {}).map(([gene, mode]) => (
              <div key={gene} style={{ marginBottom: 8, paddingBottom: 8, borderBottom: '1px solid #222' }}>
                <span style={{
                  color: GENE_COLORS[gene] || '#aaa', fontWeight: 700, fontSize: 13,
                  marginRight: 10,
                }}>{gene}</span>
                <span style={{ color: '#ccc', fontSize: 12 }}>{mode}</span>
              </div>
            ))}
          </div>

          <div style={card()}>
            <div style={{ color: '#ffb74d', fontWeight: 700, marginBottom: 10 }}>⚠ Key Clinical Rules</div>
            {(overview.key_clinical_rules || []).map((rule, i) => (
              <div key={i} style={{
                marginBottom: 6, paddingLeft: 10, borderLeft: '3px solid #ffb74d33',
                fontSize: 12, color: '#e0e0e0',
              }}>{rule}</div>
            ))}
          </div>

          {overview.gene_panel_note && (
            <div style={card({ borderColor: '#1565c0' })}>
              <div style={{ color: '#64b5f6', fontWeight: 700, marginBottom: 8 }}>Gene Panel Note</div>
              <pre style={{ color: '#ccc', fontSize: 11, whiteSpace: 'pre-wrap', margin: 0 }}>
                {overview.gene_panel_note}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* ── Gene Table ── */}
      {tab === 'Gene Table' && (
        <div>
          {Object.entries(GENE_INFO).map(([gene, info]) => {
            const c = GENE_COLORS[gene] || '#aaa';
            const expanded = expandedGene === gene;
            const bdata = breakdown?.per_gene?.[gene];
            return (
              <div key={gene} style={card({ borderColor: c + '44' })}>
                <div
                  onClick={() => { setExpandedGene(expanded ? null : gene); if (!breakdown) setTab('Gene Table'); }}
                  style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                >
                  <div>
                    <span style={{ color: c, fontWeight: 800, fontSize: 16, marginRight: 12 }}>{gene}</span>
                    <Badge text={info.inh} color={c} />
                    <Badge text={info.locus} color="#555" />
                    <Badge text={info.size} color="#444" />
                    <span style={{ color: '#aaa', fontSize: 12 }}>{info.full}</span>
                  </div>
                  <span style={{ color: '#555', fontSize: 18 }}>{expanded ? '▲' : '▼'}</span>
                </div>
                {expanded && bdata && (
                  <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8 }}>
                    {[
                      ['Melanoma', bdata.melanoma_primary_pct + '%'],
                      ['Uveal Mel.', bdata.uveal_melanoma_pct + '%'],
                      ['Pancreatic', bdata.pancreatic_cancer_pct + '%'],
                      ['BCC', bdata.bcc_pct + '%'],
                      ['Glioma', bdata.glioma_pct + '%'],
                      ['RCC', bdata.rcc_pct + '%'],
                      ['Mesothelioma', bdata.mesothelioma_pct + '%'],
                      ['BAPoma', bdata.bapoma_pct + '%'],
                      ['Retinoblastoma', bdata.retinoblastoma_pct + '%'],
                      ['Medulloblastoma', bdata.medulloblastoma_pct + '%'],
                      ['OKC', bdata.odontogenic_keratocyst_pct + '%'],
                      ['2° Sarcoma', bdata.secondary_sarcoma_pct + '%'],
                      ['Dermoscopy', bdata.dermoscopy_surveillance_pct + '%'],
                      ['Immuno.', bdata.immunotherapy_eligible_pct + '%'],
                      ['Avoid XRT', bdata.avoid_radiation_pct + '%'],
                      ['Mean Age Dx', bdata.mean_age_at_dx + 'yr'],
                    ].map(([k, v]) => (
                      <div key={k} style={{ background: '#12122a', borderRadius: 5, padding: '8px 10px' }}>
                        <div style={{ color: '#888', fontSize: 10 }}>{k}</div>
                        <div style={{ color: c, fontWeight: 700, fontSize: 14 }}>{v}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── Clinical Atlas ── */}
      {tab === 'Clinical Atlas' && (
        <div>
          {!breakdown && (
            <button onClick={() => {
              setLoading(true);
              fetch(`${API}/api/${SLUG}/breakdown`)
                .then(r => r.json())
                .then(d => { setBreakdown(d); setLoading(false); })
                .catch(e => { setError(e.message); setLoading(false); });
            }} style={{
              background: '#1565c0', color: '#fff', border: 'none',
              borderRadius: 6, padding: '8px 20px', cursor: 'pointer', marginBottom: 16,
            }}>Load Breakdown Data</button>
          )}
          {breakdown && (
            <div>
              <div style={card({ background: '#12122a' })}>
                <div style={{ color: '#81c784', fontWeight: 700, marginBottom: 10 }}>Aggregate (320 patients)</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8 }}>
                  {Object.entries(breakdown.aggregate || {}).filter(([k]) => !k.includes('total')).map(([k, v]) => (
                    <div key={k} style={{ background: '#1e1e2e', borderRadius: 5, padding: '8px 10px' }}>
                      <div style={{ color: '#888', fontSize: 10 }}>{k.replace(/_/g,' ').replace('pct','%')}</div>
                      <div style={{ color: '#64b5f6', fontWeight: 700 }}>{typeof v === 'number' ? v + (k.includes('pct') ? '%' : '') : v}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#12122a' }}>
                      {['Gene','Melanoma%','Uveal%','Pancreatic%','BCC%','Glioma%','RCC%','Meso%','BAPoma%','Retino%','MB%','OKC%','2°Sarc%','MeanAge','Severity'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', color: '#81c784', textAlign: 'left', borderBottom: '1px solid #333' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(breakdown.per_gene || {}).map(([gene, d]) => (
                      <tr key={gene} style={{ borderBottom: '1px solid #222' }}>
                        <td style={{ padding: '6px 10px', color: GENE_COLORS[gene] || '#aaa', fontWeight: 700 }}>{gene}</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.melanoma_primary_pct}%</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.uveal_melanoma_pct}%</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.pancreatic_cancer_pct}%</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.bcc_pct}%</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.glioma_pct}%</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.rcc_pct}%</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.mesothelioma_pct}%</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.bapoma_pct}%</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.retinoblastoma_pct}%</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.medulloblastoma_pct}%</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.odontogenic_keratocyst_pct}%</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.secondary_sarcoma_pct}%</td>
                        <td style={{ padding: '6px 10px', color: '#e0e0e0' }}>{d.mean_age_at_dx}yr</td>
                        <td style={{ padding: '6px 10px', color: '#aaa', fontSize: 11 }}>
                          S:{d.severity?.severe_pct}% M:{d.severity?.moderate_pct}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          {(definitions.definitions || []).map((def, i) => (
            <div key={i} style={card()}>
              <div style={{ color: '#ffb74d', fontWeight: 700, fontSize: 13, marginBottom: 8 }}>
                {def.term}
              </div>
              <pre style={{ color: '#ccc', fontSize: 11, whiteSpace: 'pre-wrap', margin: 0, lineHeight: 1.6 }}>
                {def.definition}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
