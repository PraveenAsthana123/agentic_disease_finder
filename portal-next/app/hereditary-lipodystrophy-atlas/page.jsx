'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-lipodystrophy-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'AGPAT2': '#1565c0',  // deep blue     — CGL1, near-complete fat absence, metreleptin SPECIFIC
  'BSCL2':  '#b71c1c',  // deep red      — CGL2 most common+severe, intellectual disability 30%
  'CAV1':   '#00695c',  // dark teal     — CGL3, caveolae ABSENT EM pathognomonic, PAH
  'CAVIN1': '#4a148c',  // deep purple   — CGL4, MYOPATHY + arrhythmia UNIQUE
  'LMNA':   '#e65100',  // deep orange   — FPLD2 Dunnigan, Arg482 hotspot, cardiomyopathy
  'PPARG':  '#2e7d32',  // dark green    — FPLD3, TZD SPECIFIC treatment
  'AKT2':   '#f57f17',  // amber         — FPLD6, severe insulin resistance
  'PLIN1':  '#006064',  // dark cyan     — FPLD4, pancreatitis, unregulated lipolysis
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

export default function HereditaryLipodystrophyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
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
          🧬 Hereditary Lipodystrophy Atlas
        </h1>
        <p style={{ color: '#555', fontSize: 13, margin: 0 }}>
          Complete 8-Gene Lipodystrophy Reference · AGPAT2-BSCL2-CAV1-CAVIN1-LMNA-PPARG-AKT2-PLIN1 · 320 Patients · Seeds 2998-3005
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

      {loading && <div style={{ color: '#888', padding: 20 }}>Loading…</div>}
      {error && <div style={{ color: '#c00', padding: 20 }}>Error: {error}</div>}

      {/* OVERVIEW */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 12, marginBottom: 20 }}>
            {[
              { label: 'Genes', value: overview.total_genes },
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Seed Range', value: overview.seed_range },
              { label: 'CGL Genes (AR)', value: 'AGPAT2, BSCL2, CAV1, CAVIN1' },
              { label: 'FPLD Genes (AD)', value: 'LMNA, PPARG, AKT2, PLIN1' },
              { label: 'Metreleptin Target', value: 'CGL1/2/3/4 + FPLD2' },
            ].map(k => (
              <div key={k.label} style={{ ...cardStyle, textAlign: 'center', padding: 14 }}>
                <div style={{ fontSize: 18, fontWeight: 800, color: '#1565c0' }}>{k.value}</div>
                <div style={{ fontSize: 11, color: '#777', marginTop: 2 }}>{k.label}</div>
              </div>
            ))}
          </div>

          <div style={cardStyle}>
            <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>8 Lipodystrophy Genes — Loci &amp; Inheritance</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {(overview.genes || []).map(g => (
                <div key={g} style={{
                  background: (GENE_COLORS[g] || '#888') + '15',
                  border: `1.5px solid ${(GENE_COLORS[g] || '#888')}55`,
                  borderRadius: 8, padding: '8px 12px', minWidth: 140,
                }}>
                  <div style={{ fontWeight: 800, color: GENE_COLORS[g] || '#333', fontSize: 15 }}>{g}</div>
                  <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>{(overview.gene_loci || {})[g]}</div>
                  <div style={{ fontSize: 10, color: '#777' }}>{((overview.inheritance_modes || {})[g] || '').slice(0, 60)}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={cardStyle}>
            <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 10, color: '#b71c1c' }}>🔑 Key Clinical Rules</h3>
            {(overview.key_clinical_rules || []).map((rule, i) => {
              const ci = rule.indexOf(':');
              const head = ci > -1 ? rule.slice(0, ci) : rule;
              const rest = ci > -1 ? rule.slice(ci + 1) : '';
              return (
                <div key={i} style={{ borderLeft: '3px solid #1565c0', paddingLeft: 10, marginBottom: 8, fontSize: 12 }}>
                  <strong style={{ color: '#1565c0' }}>{head}:</strong>{' '}
                  <span style={{ color: '#444' }}>{rest}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* GENE TABLE */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1565c0', color: '#fff' }}>
                  {['Gene', 'Locus', 'Inheritance', 'Mean TG (mg/dL)', 'Mean Leptin (ng/mL)', 'DM %', 'Pancreatitis %', 'Metreleptin %', 'n'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.genes || []).map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#f8f9fa' : '#fff', cursor: 'pointer' }}
                    onClick={() => setExpandedGene(expandedGene === g.gene ? null : g.gene)}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
                    <td style={{ padding: '7px 10px', color: '#555' }}>{g.locus}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{(g.inheritance || '').split(' ').slice(0,2).join(' ')}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'right', fontWeight: 700,
                      color: g.mean_triglycerides_mg_dL > 500 ? '#b71c1c' : g.mean_triglycerides_mg_dL > 200 ? '#e65100' : '#2e7d32' }}>
                      {g.mean_triglycerides_mg_dL}
                    </td>
                    <td style={{ padding: '7px 10px', textAlign: 'right',
                      color: g.mean_leptin_ng_mL < 2 ? '#b71c1c' : '#2e7d32', fontWeight: 700 }}>
                      {g.mean_leptin_ng_mL}
                    </td>
                    <td style={{ padding: '7px 10px', textAlign: 'right',
                      color: g.dm_prevalence_pct > 70 ? '#b71c1c' : '#555' }}>
                      {g.dm_prevalence_pct}%
                    </td>
                    <td style={{ padding: '7px 10px', textAlign: 'right',
                      color: g.pancreatitis_pct > 40 ? '#b71c1c' : '#555', fontWeight: g.pancreatitis_pct > 40 ? 700 : 400 }}>
                      {g.pancreatitis_pct}%
                    </td>
                    <td style={{ padding: '7px 10px', textAlign: 'right',
                      color: g.metreleptin_treatment_pct > 50 ? '#2e7d32' : '#888', fontWeight: g.metreleptin_treatment_pct > 50 ? 700 : 400 }}>
                      {g.metreleptin_treatment_pct}%
                    </td>
                    <td style={{ padding: '7px 10px', textAlign: 'right', color: '#888' }}>{g.n_patients}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {expandedGene && (() => {
            const g = (breakdown.genes || []).find(x => x.gene === expandedGene);
            if (!g) return null;
            return (
              <div style={{ ...cardStyle, marginTop: 16, borderLeft: `4px solid ${GENE_COLORS[expandedGene] || '#888'}` }}>
                <h3 style={{ color: GENE_COLORS[expandedGene] || '#333', marginTop: 0, fontSize: 15 }}>{expandedGene}</h3>
                <div style={{ fontSize: 11, color: '#555', lineHeight: 1.6, marginBottom: 8, whiteSpace: 'pre-wrap' }}>
                  <strong>Protein/Function:</strong> {g.protein_size}
                </div>
                <div style={{ fontSize: 11, color: '#444', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                  <strong>Disease Category:</strong> {g.disease_category}
                </div>
                <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {[
                    ['CK elevated', g.ck_elevated_pct + '%'],
                    ['PAH', g.pah_pct + '%'],
                    ['ID', g.intellectual_disability_pct + '%'],
                    ['Arrhythmia', g.cardiac_arrhythmia_pct + '%'],
                    ['HOMA-IR', g.mean_homa_ir],
                    ['HbA1c', g.mean_hba1c_pct + '%'],
                    ['Age Dx', g.mean_age_dx + 'y'],
                  ].map(([k,v]) => (
                    <div key={k} style={{ background: '#f5f5f5', borderRadius: 5, padding: '4px 10px', fontSize: 11 }}>
                      <strong>{k}:</strong> {v}
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* CLINICAL ATLAS */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {(breakdown.genes || []).map(g => (
            <div key={g.gene} style={{ ...cardStyle, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#888'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                <h3 style={{ color: GENE_COLORS[g.gene] || '#333', margin: 0, fontSize: 15 }}>
                  {g.gene}
                  <span style={{ fontWeight: 400, fontSize: 12, color: '#555', marginLeft: 8 }}>
                    {g.locus} · {(g.inheritance || '').split(' ').slice(0,2).join(' ')}
                  </span>
                </h3>
                <div style={{ fontSize: 11, color: '#888' }}>n={g.n_patients} · mean age {g.mean_age_dx}y</div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(130px,1fr))', gap: 8, marginBottom: 10 }}>
                {[
                  { label: 'Mean TG (mg/dL)', val: g.mean_triglycerides_mg_dL, highlight: g.mean_triglycerides_mg_dL > 500 ? 'red' : g.mean_triglycerides_mg_dL > 200 ? 'warn' : 'green' },
                  { label: 'Leptin (ng/mL)', val: g.mean_leptin_ng_mL, highlight: g.mean_leptin_ng_mL < 2 ? 'red' : 'green' },
                  { label: 'DM %', val: g.dm_prevalence_pct + '%', highlight: g.dm_prevalence_pct > 70 ? 'red' : 'none' },
                  { label: 'Metreleptin %', val: g.metreleptin_treatment_pct + '%', highlight: g.metreleptin_treatment_pct > 50 ? 'green' : 'none' },
                  { label: 'Pancreatitis %', val: g.pancreatitis_pct + '%', highlight: g.pancreatitis_pct > 40 ? 'red' : 'none' },
                  { label: 'CK elevated %', val: g.ck_elevated_pct + '%', highlight: g.ck_elevated_pct > 50 ? 'red' : 'none' },
                ].map(m => (
                  <div key={m.label} style={{
                    background: m.highlight === 'green' ? '#e8f5e9' : m.highlight === 'red' ? '#ffebee' : m.highlight === 'warn' ? '#fff8e1' : '#f5f5f5',
                    borderRadius: 6, padding: '6px 10px', textAlign: 'center',
                    border: m.highlight === 'green' ? '1px solid #a5d6a7' : m.highlight === 'red' ? '1px solid #ef9a9a' : m.highlight === 'warn' ? '1px solid #ffe082' : '1px solid #e0e0e0',
                  }}>
                    <div style={{ fontWeight: 700, fontSize: 14,
                      color: m.highlight === 'green' ? '#2e7d32' : m.highlight === 'red' ? '#b71c1c' : m.highlight === 'warn' ? '#f57f17' : '#333' }}>{m.val}</div>
                    <div style={{ fontSize: 10, color: '#777' }}>{m.label}</div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 11, color: '#555', lineHeight: 1.6, whiteSpace: 'pre-wrap', borderTop: '1px solid #f0f0f0', paddingTop: 8 }}>
                {(g.inheritance || '').slice(0, 900)}
              </div>
              <div style={{ marginTop: 8 }}>
                <strong style={{ fontSize: 11 }}>Common mutations: </strong>
                {Object.keys(g.mutation_breakdown || {}).slice(0, 4).map(m => (
                  <Badge key={m} text={m} color={GENE_COLORS[g.gene] || '#888'} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS */}
      {tab === 'Definitions' && definitions && (
        <div>
          {(definitions.definitions || []).map((d, i) => (
            <div key={i} style={cardStyle}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: '#1565c0', marginTop: 0, marginBottom: 8 }}>{d.term}</h3>
              <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
                {(d.genes || []).map(g => (
                  <span key={g} style={{
                    background: (GENE_COLORS[g] || '#888') + '22',
                    color: GENE_COLORS[g] || '#333',
                    border: `1px solid ${(GENE_COLORS[g] || '#888')}55`,
                    borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 700,
                  }}>{g}</span>
                ))}
              </div>
              <div style={{ fontSize: 12, color: '#444', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{d.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
