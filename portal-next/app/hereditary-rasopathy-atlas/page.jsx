'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-rasopathy-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'PTPN11': '#1565c0',  // deep blue      — SHP2; Noonan 1; ~50-70% all Noonan; PTPN11 first
  'SOS1':   '#2e7d32',  // deep green     — RAS-GEF; Noonan 4; NORMAL IQ unique
  'RAF1':   '#b71c1c',  // deep red       — cRAF; Noonan 5; HCM 75% UNIQUE RASopathy
  'BRAF':   '#e65100',  // deep orange    — B-Raf; CFC type 1; ectodermal triad; severe ID
  'MAP2K1': '#6a1b9a',  // deep purple    — MEK1; CFC type 3; direct MEK inhibitor target
  'HRAS':   '#006064',  // dark cyan      — H-RAS; Costello; loose skin + papillomata; cancer 15%
  'KRAS':   '#4a148c',  // deep violet    — K-RAS; Noonan 3; most severe; AML risk
  'LZTR1':  '#bf360c',  // deep brown     — CUL3 adaptor; Noonan 10; bidirectional AD/AR unique
};

const GENE_INFO = {
  'PTPN11': { full: 'Noonan Syndrome type 1',          locus: '12q24.13', size: '580 aa / 68 kDa', inh: 'AD GOF' },
  'SOS1':   { full: 'Noonan Syndrome type 4',          locus: '2p22.1',   size: '1333 aa / 152 kDa', inh: 'AD GOF' },
  'RAF1':   { full: 'Noonan Syndrome type 5',          locus: '3p25.2',   size: '648 aa / 73 kDa', inh: 'AD GOF' },
  'BRAF':   { full: 'CFC Syndrome type 1',             locus: '7q34',     size: '766 aa / 84 kDa', inh: 'AD GOF' },
  'MAP2K1': { full: 'CFC Syndrome type 3',             locus: '15q22.31', size: '393 aa / 44 kDa', inh: 'AD GOF' },
  'HRAS':   { full: 'Costello Syndrome',               locus: '11p15.5',  size: '189 aa / 21 kDa', inh: 'AD GOF' },
  'KRAS':   { full: 'Noonan Syndrome type 3 / CFC',    locus: '12p12.1',  size: '189 aa / 21 kDa', inh: 'AD GOF' },
  'LZTR1':  { full: 'Noonan Syndrome type 10',         locus: '22q11.21', size: '827 aa / 92 kDa', inh: 'AD or AR (unique)' },
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

export default function HereditaryRASopathyAtlas() {
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
          🧬 Hereditary RASopathy Atlas
        </h1>
        <p style={{ color: '#555', marginBottom: 8 }}>
          Complete 8-Gene Reference: PTPN11 · SOS1 · RAF1 · BRAF · MAP2K1 · HRAS · KRAS · LZTR1
        </p>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <Badge text="320 Patients" color="#1565c0" />
          <Badge text="8 Genes" color="#6a1b9a" />
          <Badge text="Seeds 3070–3077" color="#37474f" />
          <Badge text="Noonan + CFC + Costello" color="#b71c1c" />
          <Badge text="RAS-MAPK Pathway" color="#2e7d32" />
          <Badge text="LZTR1 Bidirectional AD/AR" color="#bf360c" />
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
              { label: 'Noonan Genes', value: '5 (PTPN11/SOS1/RAF1/KRAS/LZTR1)', color: '#1565c0' },
              { label: 'CFC Genes', value: '2 (BRAF/MAP2K1)', color: '#e65100' },
              { label: 'Costello Gene', value: '1 (HRAS)', color: '#006064' },
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
                {['Gene', 'Locus', 'Disease', 'Patients', 'HCM%', 'PS%', 'Short Stature%', 'Ectodermal%', 'Epilepsy%', 'Speech Absent%', 'Cancer Risk%', 'Mean IQ'].map(h => (
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
                  <td style={{ padding: '7px 10px' }}>{g.cardiac_hcm_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.ps_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.short_stature_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.ectodermal_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.epilepsy_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.speech_absent_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.cancer_risk_pct ?? '—'}%</td>
                  <td style={{ padding: '7px 10px' }}>{g.mean_iq ?? '—'}</td>
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
                      { k: 'cardiac_hcm_pct', label: 'HCM' },
                      { k: 'ps_pct', label: 'Pulmonary Stenosis' },
                      { k: 'short_stature_pct', label: 'Short Stature' },
                      { k: 'webbed_neck_pct', label: 'Webbed Neck' },
                      { k: 'ectodermal_pct', label: 'Ectodermal Features' },
                      { k: 'epilepsy_pct', label: 'Epilepsy' },
                      { k: 'speech_absent_pct', label: 'Speech Absent' },
                      { k: 'loose_skin_pct', label: 'Loose Skin' },
                      { k: 'papillomata_pct', label: 'Papillomata' },
                      { k: 'cancer_risk_pct', label: 'Cancer Risk' },
                      { k: 'autism_pct', label: 'Autism Features' },
                      { k: 'mean_iq', label: 'Mean IQ' },
                    ].filter(f => g[f.k] != null).map(f => (
                      <div key={f.k} style={{
                        background: (GENE_COLORS[g.gene] || '#888') + '10',
                        border: `1px solid ${(GENE_COLORS[g.gene] || '#888')}30`,
                        borderRadius: 6, padding: '6px 12px', textAlign: 'center',
                      }}>
                        <div style={{ fontSize: 17, fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>
                          {f.k === 'mean_iq' ? g[f.k] : `${g[f.k]}%`}
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
