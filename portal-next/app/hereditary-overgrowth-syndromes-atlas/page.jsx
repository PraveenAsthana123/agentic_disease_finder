'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-overgrowth-syndromes-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'NSD1':    '#1565c0',  // deep blue      — Sotos; most common overgrowth ~1:14,000
  'EZH2':    '#6a1b9a',  // deep purple    — Weaver; accelerated osseous maturation PATHOGNOMONIC; AML/ALL
  'GPC3':    '#2e7d32',  // dark green     — SGBS1; supernumerary nipples PATHOGNOMONIC; Wilms 10%; XLR
  'CDKN1C':  '#e65100',  // deep orange    — BWS type 3; omphalocele+macroglossia+macrosomia TRIAD
  'PTEN':    '#b71c1c',  // deep red       — PHTS/BRR; macrocephaly >2SD = trigger; breast 85%; ASD 20%
  'SETD2':   '#00695c',  // dark teal      — Luscan-Lumish; Sotos-like + autism 50%; H3K36me3 sole enzyme
  'NFIX':    '#4e342e',  // dark brown     — Malan (LOF) / Marshall-Smith (GOF); same gene opposite phenotypes
  'PIK3CA':  '#f9a825',  // amber          — PROS; mosaic GOF; MCAP/CLOVES/hemimegalencephaly; alpelisib FDA-2022
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

export default function HereditaryOvergrowthSyndromesAtlasPage() {
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
          🧬 Hereditary Overgrowth Syndromes Atlas
        </h1>
        <p style={{ color: '#555', fontSize: 13, margin: 0 }}>
          Complete 8-Gene Overgrowth Genetics Reference · NSD1-EZH2-GPC3-CDKN1C-PTEN-SETD2-NFIX-PIK3CA · 320 Patients · Seeds 3030-3037
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
              { label: 'Total Patients',  value: overview.total_patients },
              { label: 'Genes Covered',   value: overview.total_genes },
              { label: 'Seed Range',      value: overview.seed_range },
              { label: 'Atlas',           value: 'Overgrowth' },
            ].map(({ label, value }) => (
              <div key={label} style={{ ...cardStyle, textAlign: 'center', padding: 12 }}>
                <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 18, fontWeight: 800, color: '#1565c0' }}>{value}</div>
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

          {overview.gene_panel_note && (
            <div style={{ ...cardStyle, background: '#e3f2fd', borderColor: '#90caf9' }}>
              <h3 style={{ fontSize: 13, fontWeight: 700, color: '#1565c0', marginBottom: 6 }}>📋 Gene Panel Note</h3>
              <p style={{ fontSize: 12, color: '#555', margin: 0 }}>{overview.gene_panel_note}</p>
            </div>
          )}
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
                    {['Gene','Locus','n','Severe%','Mod%','Mild%','Mean IQ','Wilms%','Cardiac%','AML%','Autism%','AdvBA%','RaisedICP%','Age Dx(mo)'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.genes?.map((g, i) => (
                    <tr key={g.gene}
                      onClick={() => setExpandedGene(expandedGene === g.gene ? null : g.gene)}
                      style={{
                        background: expandedGene === g.gene ? '#e3f2fd' : (i % 2 === 0 ? '#f8fbff' : '#fff'),
                        borderBottom: '1px solid #e8e8e8', cursor: 'pointer',
                      }}>
                      <td style={{ padding: '7px 10px' }}><Badge text={g.gene} color={GENE_COLORS[g.gene] || '#555'} /></td>
                      <td style={{ padding: '7px 10px', fontFamily: 'monospace', color: '#666' }}>{g.locus}</td>
                      <td style={{ padding: '7px 10px', fontWeight: 700 }}>{g.n_patients}</td>
                      <td style={{ padding: '7px 10px', color: '#c62828', fontWeight: 600 }}>{g.severe_pct}%</td>
                      <td style={{ padding: '7px 10px', color: '#e65100' }}>{g.moderate_pct}%</td>
                      <td style={{ padding: '7px 10px', color: '#388e3c' }}>{g.mild_pct}%</td>
                      <td style={{ padding: '7px 10px', color: g.mean_iq < 70 ? '#c62828' : '#333', fontWeight: g.mean_iq < 70 ? 700 : 400 }}>{g.mean_iq}</td>
                      <td style={{ padding: '7px 10px', color: g.wilms_pct > 0 ? '#c62828' : '#aaa', fontWeight: g.wilms_pct > 5 ? 700 : 400 }}>{g.wilms_pct}%</td>
                      <td style={{ padding: '7px 10px', color: g.cardiac_pct > 0 ? '#1565c0' : '#aaa' }}>{g.cardiac_pct}%</td>
                      <td style={{ padding: '7px 10px', color: g.aml_pct > 0 ? '#6a1b9a' : '#aaa', fontWeight: g.aml_pct > 0 ? 700 : 400 }}>{g.aml_pct}%</td>
                      <td style={{ padding: '7px 10px', color: g.autism_pct > 30 ? '#e65100' : '#666' }}>{g.autism_pct}%</td>
                      <td style={{ padding: '7px 10px', color: g.adv_bone_age_pct > 50 ? '#2e7d32' : '#666' }}>{g.adv_bone_age_pct}%</td>
                      <td style={{ padding: '7px 10px', color: g.raised_icp_pct > 10 ? '#c62828' : '#aaa' }}>{g.raised_icp_pct}%</td>
                      <td style={{ padding: '7px 10px', color: '#666' }}>{g.mean_age_dx_mo}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {expandedGene && breakdown.genes?.find(g => g.gene === expandedGene) && (() => {
                const g = breakdown.genes.find(x => x.gene === expandedGene);
                return (
                  <div style={{ ...cardStyle, marginTop: 12, borderColor: GENE_COLORS[expandedGene] || '#ccc', borderWidth: 2 }}>
                    <h3 style={{ fontSize: 14, fontWeight: 700, color: GENE_COLORS[expandedGene] || '#333', marginBottom: 8 }}>
                      🔬 {expandedGene} — Detail
                    </h3>
                    <div style={{ fontSize: 11, color: '#555', marginBottom: 6 }}>
                      <strong>Inheritance:</strong> {g.inheritance}
                    </div>
                    <div style={{ fontSize: 11, color: '#555', marginBottom: 6 }}>
                      <strong>Disease category:</strong> {g.disease_category}
                    </div>
                    <div style={{ fontSize: 11, color: '#555' }}>
                      <strong>Sample mutations:</strong>{' '}
                      {g.sample_mutations?.map(m => (
                        <span key={m} style={{ background: '#f5f5f5', border: '1px solid #ddd', borderRadius: 3, padding: '1px 5px', marginRight: 4, fontFamily: 'monospace' }}>{m}</span>
                      ))}
                    </div>
                  </div>
                );
              })()}
            </div>
          ) : (
            <div style={{ color: '#888', padding: 24 }}>Click a tab to load data…</div>
          )}
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && !loading && (
        <div>
          {breakdown ? breakdown.genes?.map(g => (
            <div key={g.gene} style={{ ...cardStyle, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#ccc'}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                <div>
                  <Badge text={g.gene} color={GENE_COLORS[g.gene] || '#555'} />
                  <span style={{ fontSize: 11, color: '#888', marginLeft: 6, fontFamily: 'monospace' }}>{g.locus}</span>
                  <span style={{ fontSize: 11, color: '#888', marginLeft: 8 }}>n={g.n_patients}</span>
                </div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {g.wilms_pct > 0 && <Badge text={`Wilms ${g.wilms_pct}%`} color="#c62828" />}
                  {g.aml_pct > 0 && <Badge text={`AML ${g.aml_pct}%`} color="#6a1b9a" />}
                  {g.cardiac_pct > 25 && <Badge text={`Cardiac ${g.cardiac_pct}%`} color="#1565c0" />}
                  {g.autism_pct > 30 && <Badge text={`Autism ${g.autism_pct}%`} color="#e65100" />}
                  {g.adv_bone_age_pct > 50 && <Badge text={`AdvBA ${g.adv_bone_age_pct}%`} color="#2e7d32" />}
                  {g.mean_iq < 70 && <Badge text={`IQ ${g.mean_iq}`} color="#c62828" />}
                </div>
              </div>
              <div style={{ fontSize: 11, color: '#333', lineHeight: 1.6, marginBottom: 6 }}>
                {g.disease_category}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 6, marginTop: 8 }}>
                {[
                  { label: 'Severe', value: g.severe_pct + '%', color: '#c62828' },
                  { label: 'Moderate', value: g.moderate_pct + '%', color: '#e65100' },
                  { label: 'Mild', value: g.mild_pct + '%', color: '#388e3c' },
                  { label: 'Mean IQ', value: g.mean_iq, color: g.mean_iq < 70 ? '#c62828' : '#555' },
                  { label: 'Wilms%', value: g.wilms_pct + '%', color: g.wilms_pct > 5 ? '#c62828' : '#555' },
                ].map(({ label, value, color }) => (
                  <div key={label} style={{ background: '#f9f9f9', borderRadius: 6, padding: '6px 8px', textAlign: 'center' }}>
                    <div style={{ fontSize: 10, color: '#888' }}>{label}</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          )) : <div style={{ color: '#888', padding: 24 }}>Loading…</div>}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && !loading && (
        <div>
          <div style={{ color: '#888', fontSize: 12, marginBottom: 12 }}>
            {definitions.count} clinical definition entries — overgrowth classification, NSD1/EZH2/SETD2 chromatin mechanisms, PIK3CA PROS deep-sequencing, BWS methylation/imprinting, PTEN cancer surveillance algorithm
          </div>
          {definitions.definitions?.map((d, i) => (
            <div key={i} style={cardStyle}>
              <h3 style={{ fontSize: 13, fontWeight: 700, color: '#1a1a2e', marginBottom: 8 }}>
                📘 {d.term}
              </h3>
              <div style={{ marginBottom: 8 }}>
                {d.genes?.map(g => <Badge key={g} text={g} color={GENE_COLORS[g] || '#555'} />)}
              </div>
              <pre style={{
                fontSize: 11, color: '#444', lineHeight: 1.6, margin: 0,
                whiteSpace: 'pre-wrap', fontFamily: 'inherit',
                background: '#f8f8f8', padding: 10, borderRadius: 6,
                border: '1px solid #eee',
              }}>
                {d.definition}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
