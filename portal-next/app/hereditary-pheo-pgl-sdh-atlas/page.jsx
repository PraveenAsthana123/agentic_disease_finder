'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-pheo-pgl-sdh-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'SDHB':    '#b71c1c',  // deep red       — highest malignant risk 35-40%; Fe-S subunit; PGL4
  'SDHD':    '#1565c0',  // deep blue      — PGL1; paternal imprinting UNIQUE; head/neck
  'SDHC':    '#2e7d32',  // deep green     — PGL3; lowest malignant risk; parasympathetic
  'SDHA':    '#e65100',  // deep orange    — PGL5; largest subunit; GIST co-risk; IHC-specific
  'VHL':     '#6a1b9a',  // deep purple    — VHL disease; ccRCC; HBL; belzutifan FDA2021
  'RET':     '#006064',  // dark cyan      — MEN2A/B; MTC 100%; prophylactic thyroidectomy
  'TMEM127': '#4a148c',  // deep violet    — bilateral adrenal PHEO; mTOR pathway
  'MAX':     '#bf360c',  // deep brown     — paternal imprinting like SDHD; young onset <30yr
};

const GENE_INFO = {
  'SDHB':    { full: 'PGL4 / Hereditary PHEO',          locus: '1p36.13',  size: '280 aa / 32 kDa',   inh: 'AD LOF' },
  'SDHD':    { full: 'PGL1 / Head-Neck PGL (paternal)',  locus: '11q23.1',  size: '159 aa / 17 kDa',   inh: 'AD LOF (paternal imprint)' },
  'SDHC':    { full: 'PGL3 / Head-Neck PGL',            locus: '1q23.3',   size: '169 aa / 19 kDa',   inh: 'AD LOF' },
  'SDHA':    { full: 'PGL5 / PHEO + GIST',              locus: '5p15.33',  size: '621 aa / 70 kDa',   inh: 'AD LOF' },
  'VHL':     { full: 'VHL Disease (ccRCC + HBL + PHEO)', locus: '3p25.3',  size: '213 aa / 24 kDa',   inh: 'AD LOF' },
  'RET':     { full: 'MEN2A / MEN2B (MTC + PHEO)',      locus: '10q11.21', size: '1114 aa / 124 kDa', inh: 'AD GOF' },
  'TMEM127': { full: 'Hereditary PHEO (mTOR pathway)',   locus: '2q11.2',   size: '238 aa / 25 kDa',   inh: 'AD LOF' },
  'MAX':     { full: 'Hereditary PHEO (paternal imprint)', locus: '14q23.3', size: '236 aa / 22 kDa',  inh: 'AD LOF (paternal imprint)' },
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

export default function HereditaryPHEOPGLSDHAtlas() {
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

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '24px 16px' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, marginBottom: 4 }}>
          🧬 Hereditary PHEO-PGL-SDH Atlas
        </h1>
        <p style={{ color: '#555', marginBottom: 8 }}>
          Complete 8-Gene Reference: SDHB · SDHD · SDHC · SDHA · VHL · RET · TMEM127 · MAX
        </p>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <Badge text="320 Patients" color="#1565c0" />
          <Badge text="8 Genes" color="#6a1b9a" />
          <Badge text="Seeds 3078–3085" color="#37474f" />
          <Badge text="SDHB Highest Malignant Risk" color="#b71c1c" />
          <Badge text="Paternal Imprinting: SDHD + MAX" color="#1565c0" />
          <Badge text="Belzutifan FDA2021 (VHL)" color="#6a1b9a" />
          <Badge text="RET Prophylactic Thyroidectomy" color="#006064" />
        </div>
      </div>

      {/* Gene colour chips */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
        {Object.entries(GENE_COLORS).map(([g, c]) => (
          <span key={g} style={{
            background: c + '18', border: `1px solid ${c}44`,
            borderRadius: 20, padding: '3px 12px',
            fontSize: 13, fontWeight: 600, color: c,
          }}>
            {g}
            <span style={{ fontSize: 10, fontWeight: 400, color: '#888', marginLeft: 4 }}>
              {GENE_INFO[g]?.locus}
            </span>
          </span>
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
              { label: 'SDHx Genes', value: '4 (SDHB/SDHD/SDHC/SDHA)', color: '#b71c1c' },
              { label: 'Imprinted Genes', value: '2 (SDHD + MAX)', color: '#1565c0' },
              { label: 'SDHB Malignant Risk', value: '35–40%', color: '#b71c1c' },
            ].map(k => (
              <div key={k.label} style={{
                background: k.color + '12', border: `1px solid ${k.color}33`,
                borderRadius: 8, padding: '12px 18px', minWidth: 140, textAlign: 'center',
              }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: k.color }}>{k.value}</div>
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
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f5f5f5' }}>
                {['Gene', 'Locus', 'Disease', 'Patients', 'Malignant%', 'Extra-Adrenal%', 'Bilateral%', 'Head-Neck%', 'Functional%', 'HTN%', 'SSTR-PET+%', 'Mean Age Dx'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #ddd', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(breakdown.genes || []).map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                  <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
                  <td style={{ padding: '7px 10px', color: '#555' }}>{g.locus}</td>
                  <td style={{ padding: '7px 10px', color: '#555', fontSize: 11 }}>{GENE_INFO[g.gene]?.full || g.disease_category}</td>
                  <td style={{ padding: '7px 10px' }}>{g.n}</td>
                  <td style={{ padding: '7px 10px', fontWeight: g.malignant_pct > 20 ? 700 : 400, color: g.malignant_pct > 20 ? '#b71c1c' : '#333' }}>{g.malignant_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.extra_adrenal_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.bilateral_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.head_neck_pgl_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.functional_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.hypertension_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.sstr_pet_positive_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.mean_age_dx_yrs ?? '—'} yr</td>
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
                  {g.gene}{' '}
                  <span style={{ fontWeight: 400, fontSize: 12, color: '#666' }}>
                    {g.locus} · {GENE_INFO[g.gene]?.full} · n={g.n}
                  </span>
                </span>
                <span style={{ fontSize: 18, color: '#888' }}>{expandedGene === g.gene ? '▲' : '▼'}</span>
              </div>
              {expandedGene === g.gene && (
                <div style={{ padding: '14px 16px', background: '#fff' }}>
                  {/* Stats grid */}
                  <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 14 }}>
                    {[
                      { k: 'malignant_pct',         label: 'Malignant' },
                      { k: 'extra_adrenal_pct',      label: 'Extra-Adrenal' },
                      { k: 'bilateral_pct',          label: 'Bilateral' },
                      { k: 'head_neck_pgl_pct',      label: 'Head/Neck PGL' },
                      { k: 'functional_pct',         label: 'Functional' },
                      { k: 'hypertension_pct',       label: 'Hypertension' },
                      { k: 'gist_pct',               label: 'GIST' },
                      { k: 'rcc_pct',                label: 'ccRCC' },
                      { k: 'hemangioblastoma_pct',   label: 'Hemangioblastoma' },
                      { k: 'retinal_hbl_pct',        label: 'Retinal HBL' },
                      { k: 'pnet_pct',               label: 'pNET' },
                      { k: 'mtc_pct',                label: 'MTC' },
                      { k: 'phpt_pct',               label: 'PHPT' },
                      { k: 'sstr_pet_positive_pct',  label: 'SSTR-PET+' },
                      { k: 'mean_age_dx_yrs',        label: 'Mean Age Dx (yr)' },
                    ].filter(f => g[f.k] != null && g[f.k] !== 0).map(f => (
                      <div key={f.k} style={{
                        background: (GENE_COLORS[g.gene] || '#888') + '10',
                        border: `1px solid ${(GENE_COLORS[g.gene] || '#888')}30`,
                        borderRadius: 6, padding: '6px 12px', textAlign: 'center',
                      }}>
                        <div style={{ fontSize: 17, fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>
                          {f.k === 'mean_age_dx_yrs' ? `${g[f.k]}yr` : `${g[f.k]}%`}
                        </div>
                        <div style={{ fontSize: 10, color: '#666' }}>{f.label}</div>
                      </div>
                    ))}
                  </div>
                  {/* Clinical note */}
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
