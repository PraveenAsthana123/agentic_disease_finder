'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-thalassemia-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'HBB':    '#b71c1c',  // deep red      — beta-thalassemia major, HbS/HbE/HbC alleles, Casgevy/Zynteglo
  'HBA2':   '#1565c0',  // deep blue     — alpha-thal, Hb Barts hydrops, CIS vs TRANS critical
  'HBA1':   '#1976d2',  // blue          — alpha-thal tandem, Constant Spring non-deletion
  'ATRX':   '#6a1b9a',  // deep purple   — ATR-X syndrome, XLR non-deletion alpha-thal + ID
  'KLF1':   '#2e7d32',  // dark green    — HPFH6, thalassemia modifier, elevated HbA2 + HbF
  'BCL11A': '#00695c',  // dark teal     — erythroid enhancer, Casgevy CRISPR target, HbF induction
  'CDAN1':  '#e65100',  // deep orange   — CDA-I, H-bridges PATHOGNOMONIC, IFN-alpha, HCC risk 16%
  'SEC23B': '#f57f17',  // amber         — CDA-II HEMPAS, Ham+ PATHOGNOMONIC, binucleate erythroblasts
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

export default function HereditaryThalassemiaAtlasPage() {
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
          🧬 Hereditary Thalassemia Atlas
        </h1>
        <p style={{ color: '#555', fontSize: 13, margin: 0 }}>
          Complete 8-Gene Thalassemia &amp; Hemoglobinopathy Reference · HBB-HBA2-HBA1-ATRX-KLF1-BCL11A-CDAN1-SEC23B · 320 Patients · Seeds 2998-3005
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e0e0e0' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 16px', border: 'none', cursor: 'pointer',
            fontWeight: tab === t ? 700 : 400,
            background: tab === t ? '#b71c1c' : 'transparent',
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
              { label: 'Total Patients',  value: overview.total_patients },
              { label: 'Genes Covered',   value: overview.total_genes },
              { label: 'Seed Range',      value: overview.seed_range },
              { label: 'Atlas',           value: 'Thalassemia' },
            ].map(({ label, value }) => (
              <div key={label} style={{ ...cardStyle, textAlign: 'center', padding: 12 }}>
                <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 18, fontWeight: 800, color: '#b71c1c' }}>{value}</div>
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
            <h3 style={{ fontSize: 14, fontWeight: 700, color: '#333', marginBottom: 10 }}>⚡ Key Clinical Rules</h3>
            <ul style={{ margin: 0, padding: '0 0 0 18px' }}>
              {overview.key_clinical_rules?.map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#444', marginBottom: 6, lineHeight: 1.5 }}>{r}</li>
              ))}
            </ul>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div style={cardStyle}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: '#333', marginBottom: 8 }}>🧬 Inheritance Modes</h3>
              {overview.inheritance_modes && Object.entries(overview.inheritance_modes).map(([gene, mode]) => (
                <div key={gene} style={{ marginBottom: 6 }}>
                  <Badge text={gene} color={GENE_COLORS[gene] || '#555'} />
                  <span style={{ fontSize: 11, color: '#666', marginLeft: 4 }}>{mode}</span>
                </div>
              ))}
            </div>
            <div style={cardStyle}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: '#333', marginBottom: 8 }}>📍 Gene Loci</h3>
              {overview.gene_loci && Object.entries(overview.gene_loci).map(([gene, locus]) => (
                <div key={gene} style={{ display: 'flex', alignItems: 'center', marginBottom: 5 }}>
                  <Badge text={gene} color={GENE_COLORS[gene] || '#555'} />
                  <span style={{ fontSize: 11, color: '#888', marginLeft: 4, fontFamily: 'monospace' }}>{locus}</span>
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
                  <tr style={{ background: '#b71c1c', color: '#fff' }}>
                    {['Gene','Locus','n','Age Dx','Hb g/dL','TDT%','NTDT%','Trait%','Iron%','Spleen%','HbF%'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.genes?.map((g, i) => (
                    <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff8f8' : '#fff', borderBottom: '1px solid #e8e8e8' }}>
                      <td style={{ padding: '7px 10px' }}><Badge text={g.gene} color={GENE_COLORS[g.gene] || '#555'} /></td>
                      <td style={{ padding: '7px 10px', color: '#555' }}>{g.locus}</td>
                      <td style={{ padding: '7px 10px', fontWeight: 600 }}>{g.n_patients}</td>
                      <td style={{ padding: '7px 10px' }}>{g.mean_age_dx}</td>
                      <td style={{ padding: '7px 10px', fontWeight: 600, color: '#b71c1c' }}>{g.mean_hb_g_dL}</td>
                      <td style={{ padding: '7px 10px' }}>{g.tdt_pct}%</td>
                      <td style={{ padding: '7px 10px' }}>{g.ntdt_pct}%</td>
                      <td style={{ padding: '7px 10px' }}>{g.trait_pct}%</td>
                      <td style={{ padding: '7px 10px' }}>{g.iron_overload_pct}%</td>
                      <td style={{ padding: '7px 10px' }}>{g.splenomegaly_pct}%</td>
                      <td style={{ padding: '7px 10px' }}>{g.mean_hbf_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <button onClick={() => { setLoading(true); fetch(`${API}/api/${SLUG}/breakdown`).then(r=>r.json()).then(d=>{setBreakdown(d);setLoading(false);}).catch(e=>{setError(e.message);setLoading(false);}); }} style={{ padding: '10px 20px', background: '#b71c1c', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer' }}>Load Gene Table</button>
          )}
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && !loading && (
        <div>
          {breakdown ? (
            <div>
              {breakdown.genes?.map(g => (
                <div key={g.gene} style={{ ...cardStyle, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#b71c1c'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }}
                       onClick={() => setExpandedGene(expandedGene === g.gene ? null : g.gene)}>
                    <Badge text={g.gene} color={GENE_COLORS[g.gene] || '#555'} />
                    <span style={{ fontSize: 12, color: '#666' }}>{g.locus}</span>
                    <span style={{ fontSize: 12, color: '#888', flex: 1 }}>{g.disease_category?.split(':')[0]}</span>
                    <span style={{ color: '#aaa', fontSize: 14 }}>{expandedGene === g.gene ? '▲' : '▼'}</span>
                  </div>
                  {expandedGene === g.gene && (
                    <div style={{ marginTop: 12 }}>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(120px,1fr))', gap: 8, marginBottom: 12 }}>
                        {[
                          ['Mean Hb (g/dL)',  g.mean_hb_g_dL],
                          ['TDT %',           g.tdt_pct + '%'],
                          ['NTDT %',          g.ntdt_pct + '%'],
                          ['Trait %',         g.trait_pct + '%'],
                          ['Iron Overload %', g.iron_overload_pct + '%'],
                          ['Splenomegaly %',  g.splenomegaly_pct + '%'],
                          ['Mean HbF %',      g.mean_hbf_pct + '%'],
                          ['Mean Age Dx',     g.mean_age_dx + ' yr'],
                        ].map(([label, val]) => (
                          <div key={label} style={{ background: '#f5f5f5', borderRadius: 6, padding: '8px 10px', textAlign: 'center' }}>
                            <div style={{ fontSize: 10, color: '#888' }}>{label}</div>
                            <div style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{val}</div>
                          </div>
                        ))}
                      </div>
                      <div style={{ fontSize: 11, color: '#555', lineHeight: 1.6, background: '#fafafa', padding: 10, borderRadius: 6 }}>
                        <strong>Inheritance: </strong>{g.inheritance?.split('—')[0]}
                      </div>
                      {g.treatment_breakdown && Object.keys(g.treatment_breakdown).length > 0 && (
                        <div style={{ marginTop: 8 }}>
                          <div style={{ fontSize: 11, fontWeight: 600, color: '#333', marginBottom: 4 }}>Treatment Distribution:</div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                            {Object.entries(g.treatment_breakdown).slice(0, 6).map(([tx, n]) => (
                              <span key={tx} style={{ fontSize: 10, background: (GENE_COLORS[g.gene] || '#b71c1c') + '18', color: GENE_COLORS[g.gene] || '#b71c1c', border: `1px solid ${GENE_COLORS[g.gene] || '#b71c1c'}33`, borderRadius: 4, padding: '2px 6px' }}>
                                {tx}: {n}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <button onClick={() => { setLoading(true); fetch(`${API}/api/${SLUG}/breakdown`).then(r=>r.json()).then(d=>{setBreakdown(d);setLoading(false);}).catch(e=>{setError(e.message);setLoading(false);}); }} style={{ padding: '10px 20px', background: '#b71c1c', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer' }}>Load Clinical Atlas</button>
          )}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && !loading && (
        <div>
          <div style={{ marginBottom: 10, fontSize: 12, color: '#888' }}>{definitions.count} clinical definitions</div>
          {definitions.definitions?.map((d, i) => (
            <div key={i} style={{ ...cardStyle, borderLeft: '4px solid #b71c1c' }}>
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
