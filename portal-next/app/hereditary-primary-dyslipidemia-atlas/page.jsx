'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-primary-dyslipidemia-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'LDLR':    '#1565c0',  // deep blue     — FH1 most common, tendon xanthomata PATHOGNOMONIC
  'APOB':    '#b71c1c',  // deep red      — FDB Arg3527Gln European founder, statin-responsive
  'PCSK9':   '#00695c',  // dark teal     — FH3 GOF D374Y / LOF CVD protection 88%
  'LDLRAP1': '#4a148c',  // deep purple   — ARH, lymphocyte LDLR binding normal
  'LIPA':    '#e65100',  // deep orange   — Wolman adrenal calcification / LAL-D sebelipase
  'ABCA1':   '#f57f17',  // amber         — Tangier, orange tonsils PATHOGNOMONIC, HDL near zero
  'ABCG5':   '#2e7d32',  // dark green    — sitosterolaemia, childhood xanthomata, ezetimibe
  'ABCG8':   '#006064',  // dark cyan     — sitosterolaemia type 2, South Asian, D19H gallstones
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

export default function HereditaryPrimaryDyslipidemiaAtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
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

  const cardStyle = {
    background: '#fff', border: '1px solid #e0e0e0',
    borderRadius: 8, padding: 16, marginBottom: 14,
    boxShadow: '0 1px 3px rgba(0,0,0,0.07)',
  };

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto', padding: '20px 16px' }}>
      <div style={{ marginBottom: 18 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#1a1a2e', margin: '0 0 4px' }}>
          🧬 Hereditary Primary Dyslipidemia Atlas
        </h1>
        <p style={{ color: '#555', fontSize: 13, margin: 0 }}>
          Complete 8-Gene Familial Hypercholesterolaemia &amp; Dyslipidaemia Reference · LDLR-APOB-PCSK9-LDLRAP1-LIPA-ABCA1-ABCG5-ABCG8 · 320 Patients · Seeds 3006-3013
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e0e0e0' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 16px', border: 'none', cursor: 'pointer',
            fontWeight: tab === t ? 700 : 400,
            background: tab === t ? '#1565c0' : 'transparent',
            color: tab === t ? '#fff' : '#555',
            borderRadius: '6px 6px 0 0', fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#888', padding: 24 }}>Loading…</div>}
      {error   && <div style={{ color: '#c62828', padding: 12, background: '#fff3f3', borderRadius: 6 }}>Error: {error}</div>}

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && !loading && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 10, marginBottom: 18 }}>
            {[
              { label: 'Total Patients',     value: overview.total_patients },
              { label: 'Genes Covered',      value: overview.gene_count },
              { label: 'Mean LDL-C (mmol/L)',value: overview.aggregate_stats?.mean_ldl_c_mmol_L },
              { label: 'Mean HDL-C (mmol/L)',value: overview.aggregate_stats?.mean_hdl_c_mmol_L },
              { label: 'Premature CVD %',    value: overview.aggregate_stats?.premature_cvd_pct + '%' },
              { label: 'Tendon Xanthomata %',value: overview.aggregate_stats?.tendon_xanthomata_pct + '%' },
              { label: 'On PCSK9i %',        value: overview.aggregate_stats?.on_pcsk9i_pct + '%' },
              { label: 'On Ezetimibe %',     value: overview.aggregate_stats?.on_ezetimibe_pct + '%' },
            ].map(({ label, value }) => (
              <div key={label} style={{ ...cardStyle, textAlign: 'center', padding: 12 }}>
                <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 22, fontWeight: 800, color: '#1565c0' }}>{value}</div>
              </div>
            ))}
          </div>

          <div style={cardStyle}>
            <h3 style={{ fontSize: 14, fontWeight: 700, color: '#333', marginBottom: 10 }}>🧬 Gene Legend</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {overview.genes?.map(g => (
                <Badge key={g} text={g} color={GENE_COLORS[g] || '#555'} />
              ))}
            </div>
          </div>

          <div style={cardStyle}>
            <h3 style={{ fontSize: 14, fontWeight: 700, color: '#333', marginBottom: 10 }}>⚡ Key Clinical Pearls</h3>
            <ul style={{ margin: 0, padding: '0 0 0 18px' }}>
              {overview.key_clinical_pearls?.map((p, i) => (
                <li key={i} style={{ fontSize: 12, color: '#444', marginBottom: 6, lineHeight: 1.5 }}>{p}</li>
              ))}
            </ul>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div style={cardStyle}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: '#333', marginBottom: 8 }}>🧬 Inheritance Spectrum</h3>
              {overview.inheritance_spectrum && Object.entries(overview.inheritance_spectrum).map(([k, genes]) => (
                <div key={k} style={{ marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: '#888', fontWeight: 600, textTransform: 'uppercase' }}>{k.replace(/_/g, ' ')}</span>
                  <div style={{ marginTop: 2 }}>{genes.map(g => <Badge key={g} text={g} color={GENE_COLORS[g] || '#555'} />)}</div>
                </div>
              ))}
            </div>
            <div style={cardStyle}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: '#333', marginBottom: 8 }}>💊 Treatment Highlights</h3>
              {overview.treatment_highlights && Object.entries(overview.treatment_highlights).map(([k, genes]) => (
                <div key={k} style={{ marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: '#888', fontWeight: 600 }}>{k.replace(/_/g, ' ')}</span>
                  <div style={{ marginTop: 2 }}>{genes.map(g => <Badge key={g} text={g} color={GENE_COLORS[g] || '#888'} />)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && !loading && (
        <div>
          {breakdown ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1565c0', color: '#fff' }}>
                    {['Gene','Locus','n','Age Dx','LDL-C','HDL-C','TG','CVD%','Xanth%','PCSK9i%','Ezet%'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.genes?.map((g, i) => (
                    <tr key={g.gene} style={{ background: i % 2 === 0 ? '#f8f9ff' : '#fff', borderBottom: '1px solid #e8e8e8' }}>
                      <td style={{ padding: '7px 10px' }}><Badge text={g.gene} color={GENE_COLORS[g.gene] || '#555'} /></td>
                      <td style={{ padding: '7px 10px', color: '#555' }}>{g.locus}</td>
                      <td style={{ padding: '7px 10px', fontWeight: 600 }}>{g.n_patients}</td>
                      <td style={{ padding: '7px 10px' }}>{g.mean_age_dx}</td>
                      <td style={{ padding: '7px 10px', fontWeight: 600, color: '#b71c1c' }}>{g.mean_ldl_c_mmol_L}</td>
                      <td style={{ padding: '7px 10px', color: '#2e7d32' }}>{g.mean_hdl_c_mmol_L}</td>
                      <td style={{ padding: '7px 10px' }}>{g.mean_triglycerides_mmol_L}</td>
                      <td style={{ padding: '7px 10px' }}>{g.premature_cvd_pct}%</td>
                      <td style={{ padding: '7px 10px' }}>{g.tendon_xanthomata_pct}%</td>
                      <td style={{ padding: '7px 10px' }}>{g.on_pcsk9i_pct}%</td>
                      <td style={{ padding: '7px 10px' }}>{g.on_ezetimibe_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <button onClick={() => { setLoading(true); fetch(`${API}/api/${SLUG}/breakdown`).then(r=>r.json()).then(d=>{setBreakdown(d);setLoading(false);}).catch(e=>{setError(e.message);setLoading(false);}); }} style={{ padding: '10px 20px', background: '#1565c0', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer' }}>Load Gene Table</button>
          )}
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && !loading && (
        <div>
          {breakdown ? (
            <div>
              {breakdown.genes?.map(g => (
                <div key={g.gene} style={{ ...cardStyle, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#1565c0'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }}
                       onClick={() => setExpandedGene(expandedGene === g.gene ? null : g.gene)}>
                    <Badge text={g.gene} color={GENE_COLORS[g.gene] || '#555'} />
                    <span style={{ fontSize: 12, color: '#666' }}>{g.locus}</span>
                    <span style={{ fontSize: 12, color: '#888', flex: 1 }}>{g.disease_category}</span>
                    <span style={{ color: '#aaa', fontSize: 14 }}>{expandedGene === g.gene ? '▲' : '▼'}</span>
                  </div>
                  {expandedGene === g.gene && (
                    <div style={{ marginTop: 12 }}>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(120px,1fr))', gap: 8, marginBottom: 12 }}>
                        {[
                          ['Mean LDL-C', g.mean_ldl_c_mmol_L + ' mmol/L'],
                          ['Mean HDL-C', g.mean_hdl_c_mmol_L + ' mmol/L'],
                          ['Mean TG',    g.mean_triglycerides_mmol_L + ' mmol/L'],
                          ['Premature CVD', g.premature_cvd_pct + '%'],
                          ['Tendon Xanth', g.tendon_xanthomata_pct + '%'],
                          ['Corneal Arcus', g.corneal_arcus_pct + '%'],
                          ...(g.orange_tonsils_pct !== undefined ? [['Orange Tonsils', g.orange_tonsils_pct + '%']] : []),
                          ...(g.adrenal_calc_pct !== undefined ? [['Adrenal Calc', g.adrenal_calc_pct + '%']] : []),
                          ...(g.haemolytic_anaemia_pct !== undefined ? [['Haemolytic Anaemia', g.haemolytic_anaemia_pct + '%']] : []),
                          ...(g.sebelipase_pct !== undefined ? [['On Sebelipase', g.sebelipase_pct + '%']] : []),
                        ].map(([label, val]) => (
                          <div key={label} style={{ background: '#f5f5f5', borderRadius: 6, padding: '8px 10px', textAlign: 'center' }}>
                            <div style={{ fontSize: 10, color: '#888' }}>{label}</div>
                            <div style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{val}</div>
                          </div>
                        ))}
                      </div>
                      <div style={{ fontSize: 11, color: '#555', lineHeight: 1.6, background: '#fafafa', padding: 10, borderRadius: 6 }}>
                        <strong>Inheritance: </strong>{g.inheritance}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <button onClick={() => { setLoading(true); fetch(`${API}/api/${SLUG}/breakdown`).then(r=>r.json()).then(d=>{setBreakdown(d);setLoading(false);}).catch(e=>{setError(e.message);setLoading(false);}); }} style={{ padding: '10px 20px', background: '#1565c0', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer' }}>Load Clinical Atlas</button>
          )}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && !loading && (
        <div>
          <div style={{ marginBottom: 10, fontSize: 12, color: '#888' }}>{definitions.count} clinical definitions</div>
          {definitions.definitions?.map((d, i) => (
            <div key={i} style={{ ...cardStyle, borderLeft: '4px solid #1565c0' }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 8 }}>
                <strong style={{ fontSize: 13, color: '#1a1a2e', flex: 1 }}>{d.term}</strong>
                <div>{d.genes?.map(g => <Badge key={g} text={g} color={GENE_COLORS[g] || '#555'} />)}</div>
              </div>
              <pre style={{ fontSize: 11, color: '#555', whiteSpace: 'pre-wrap', margin: 0, lineHeight: 1.6, fontFamily: 'inherit' }}>{d.definition}</pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
