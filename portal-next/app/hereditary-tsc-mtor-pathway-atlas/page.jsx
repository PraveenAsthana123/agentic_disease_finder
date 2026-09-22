'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-tsc-mtor-pathway-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'TSC1':   '#1565c0',  // deep blue   — hamartin; TSC1; cortical tubers; SEGA; everolimus
  'TSC2':   '#6a1b9a',  // deep purple — tuberin; TSC2 MORE SEVERE; PKD; everolimus
  'DEPDC5': '#b71c1c',  // deep red    — GATOR1; FFEVF; SUDEP risk elevated
  'NPRL2':  '#e65100',  // deep orange — GATOR1; focal epilepsy; SUDEP risk
  'NPRL3':  '#1b5e20',  // dark green  — GATOR1; 16p13.3 co-localised with TSC2; ADNFLE-like
  'MTOR':   '#4a148c',  // deep violet — somatic GOF; FCD IIb; balloon cells PATHOGNOMONIC
  'PIK3R2': '#006064',  // dark cyan   — somatic GOF; MCAP; hemisphere asymmetry
  'AKT3':   '#bf360c',  // deep brown  — somatic GOF; hemimegalencephaly; MPPH
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

export default function HereditaryTscMtorPathwayAtlasPage() {
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

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '24px 16px' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, marginBottom: 4 }}>
          🧬 Hereditary TSC-mTOR Pathway Atlas
        </h1>
        <p style={{ color: '#555', marginBottom: 8 }}>
          Complete 8-Gene Reference: TSC1 · TSC2 · DEPDC5 · NPRL2 · NPRL3 · MTOR · PIK3R2 · AKT3
        </p>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <Badge text="320 Patients" color="#1565c0" />
          <Badge text="8 Genes" color="#6a1b9a" />
          <Badge text="Seeds 3062–3069" color="#37474f" />
          <Badge text="TSC + GATOR1 + Somatic mTOR" color="#b71c1c" />
          <Badge text="Everolimus FDA-Approved" color="#1b5e20" />
        </div>
      </div>

      {/* Gene colour chips */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
        {Object.entries(GENE_COLORS).map(([g, c]) => (
          <span key={g} style={{
            background: c + '18', border: `1px solid ${c}44`,
            borderRadius: 20, padding: '3px 12px',
            fontSize: 13, fontWeight: 600, color: c,
          }}>{g}</span>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
              style={{ cursor: 'pointer' }}
            >{t}</button>
          </li>
        ))}
      </ul>

      {loading && <div className="text-center py-5"><div className="spinner-border" /></div>}
      {error && <div className="alert alert-danger">{error}</div>}

      {/* Overview Tab */}
      {!loading && !error && tab === 'Overview' && overview && (
        <div>
          {/* KPI row */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
            {[
              { label: 'Total Genes', value: overview.total_genes, color: '#1565c0' },
              { label: 'Total Patients', value: overview.total_patients, color: '#6a1b9a' },
              { label: 'Seed Range', value: overview.seed_range, color: '#37474f' },
              { label: 'Everolimus-Eligible Genes', value: '2 (TSC1/TSC2)', color: '#1b5e20' },
              { label: 'GATOR1 Complex Genes', value: '3 (DEPDC5/NPRL2/NPRL3)', color: '#b71c1c' },
              { label: 'Somatic mTOR Genes', value: '3 (MTOR/PIK3R2/AKT3)', color: '#4a148c' },
            ].map(k => (
              <div key={k.label} style={{
                background: k.color + '12', border: `1px solid ${k.color}33`,
                borderRadius: 8, padding: '12px 18px', minWidth: 130, textAlign: 'center',
              }}>
                <div style={{ fontSize: 22, fontWeight: 700, color: k.color }}>{k.value}</div>
                <div style={{ fontSize: 11, color: '#666', marginTop: 2 }}>{k.label}</div>
              </div>
            ))}
          </div>

          {/* Inheritance modes */}
          <h5 style={{ fontWeight: 600, marginBottom: 12 }}>Inheritance &amp; Disease Mechanism</h5>
          <div style={{ display: 'grid', gap: 10, marginBottom: 24 }}>
            {Object.entries(overview.inheritance_modes || {}).map(([gene, desc]) => (
              <div key={gene} style={{
                background: (GENE_COLORS[gene] || '#888') + '0d',
                border: `1px solid ${(GENE_COLORS[gene] || '#888')}33`,
                borderRadius: 8, padding: '10px 14px',
              }}>
                <span style={{
                  fontWeight: 700, color: GENE_COLORS[gene] || '#333',
                  fontSize: 14, marginRight: 8,
                }}>{gene}</span>
                <span style={{ fontSize: 13, color: '#444' }}>{desc}</span>
              </div>
            ))}
          </div>

          {/* Key clinical rules */}
          <h5 style={{ fontWeight: 600, marginBottom: 12 }}>Key Clinical Rules</h5>
          <ul style={{ paddingLeft: 20 }}>
            {(overview.key_clinical_rules || []).map((rule, i) => (
              <li key={i} style={{ marginBottom: 8, fontSize: 13, color: '#333' }}>{rule}</li>
            ))}
          </ul>

          {/* Gene panel note */}
          {overview.gene_panel_note && (
            <div style={{
              background: '#e3f2fd', border: '1px solid #90caf9',
              borderRadius: 8, padding: '12px 16px', marginTop: 20,
              fontSize: 13, color: '#1565c0',
            }}>
              <strong>Gene Panel Note:</strong> {overview.gene_panel_note}
            </div>
          )}
        </div>
      )}

      {/* Gene Table Tab */}
      {!loading && !error && tab === 'Gene Table' && breakdown && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f5f5f5' }}>
                {['Gene', 'Locus', 'Patients', 'Epilepsy%', 'Surgery%', 'Everolimus%', 'SUDEP Risk%', 'Deep Seq%', 'Somatic%', 'SEGA%'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #ddd', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(breakdown.genes || []).map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                  <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
                  <td style={{ padding: '7px 10px', color: '#555' }}>{g.locus}</td>
                  <td style={{ padding: '7px 10px' }}>{g.n}</td>
                  <td style={{ padding: '7px 10px' }}>{g.epilepsy_pct ?? g.seizure_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.surgical_candidate_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.everolimus_eligible_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.sudep_risk_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.deep_seq_required_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.somatic_variant_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.sega_pct ?? '—'}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {!loading && !error && tab === 'Clinical Atlas' && breakdown && (
        <div>
          {(breakdown.genes || []).map(g => (
            <div key={g.gene} style={{
              marginBottom: 16,
              border: `1px solid ${(GENE_COLORS[g.gene] || '#888')}44`,
              borderRadius: 10, overflow: 'hidden',
            }}>
              <div
                style={{
                  background: (GENE_COLORS[g.gene] || '#888') + '18',
                  padding: '10px 16px', cursor: 'pointer',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                }}
                onClick={() => setExpandedGene(expandedGene === g.gene ? null : g.gene)}
              >
                <span style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#333', fontSize: 15 }}>
                  {g.gene} <span style={{ fontWeight: 400, fontSize: 12, color: '#666' }}>{g.locus} · n={g.n}</span>
                </span>
                <span style={{ fontSize: 18, color: '#888' }}>{expandedGene === g.gene ? '▲' : '▼'}</span>
              </div>
              {expandedGene === g.gene && (
                <div style={{ padding: '14px 16px', background: '#fff' }}>
                  {/* Stats grid */}
                  <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 14 }}>
                    {[
                      { k: 'epilepsy_pct', label: 'Epilepsy' },
                      { k: 'sega_pct', label: 'SEGA' },
                      { k: 'renal_aml_pct', label: 'Renal AML' },
                      { k: 'lam_pct', label: 'LAM' },
                      { k: 'balloon_cells_pct', label: 'Balloon Cells' },
                      { k: 'hemisphere_asymmetry_pct', label: 'Hemisphere Asymmetry' },
                      { k: 'surgical_candidate_pct', label: 'Surgical Candidate' },
                      { k: 'everolimus_eligible_pct', label: 'Everolimus Eligible' },
                      { k: 'deep_seq_required_pct', label: 'Deep Seq Required' },
                      { k: 'sudep_risk_pct', label: 'SUDEP Risk Flag' },
                      { k: 'somatic_variant_pct', label: 'Somatic Variant' },
                    ].filter(f => g[f.k] != null).map(f => (
                      <div key={f.k} style={{
                        background: (GENE_COLORS[g.gene] || '#888') + '10',
                        border: `1px solid ${(GENE_COLORS[g.gene] || '#888')}30`,
                        borderRadius: 6, padding: '6px 12px', textAlign: 'center',
                      }}>
                        <div style={{ fontSize: 17, fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g[f.k]}%</div>
                        <div style={{ fontSize: 10, color: '#666' }}>{f.label}</div>
                      </div>
                    ))}
                  </div>
                  {/* Clinical notes */}
                  {g.clinical_note && (
                    <p style={{ fontSize: 13, color: '#444', marginBottom: 0 }}>{g.clinical_note}</p>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {!loading && !error && tab === 'Definitions' && definitions && (
        <div>
          {(definitions.definitions || []).map((def, i) => (
            <div key={i} style={{
              marginBottom: 16,
              border: '1px solid #e0e0e0',
              borderRadius: 10, overflow: 'hidden',
            }}>
              <div style={{
                background: '#f5f5f5',
                padding: '10px 16px',
                fontWeight: 600, fontSize: 14,
              }}>
                {def.term || def.title || `Definition ${i + 1}`}
              </div>
              <div style={{ padding: '12px 16px', background: '#fff', fontSize: 13, color: '#333', whiteSpace: 'pre-wrap' }}>
                {def.definition || def.body || def.text || JSON.stringify(def, null, 2)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
