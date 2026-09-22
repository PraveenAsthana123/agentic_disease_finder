'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-chromatinopathy-epigenetic-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'KMT2D':  '#1565c0',  // deep blue     — Kabuki 1; arched eyebrows; fingertip pads PATHOGNOMONIC; 60-75% Kabuki
  'KDM6A':  '#4a148c',  // deep purple   — Kabuki 2; X-linked; milder females; more severe males
  'CREBBP': '#b71c1c',  // deep red      — RTS1; broad thumbs/halluces PATHOGNOMONIC; malignancy 10-15%
  'EP300':  '#c62828',  // red           — RTS2; milder CREBBP; same broad thumbs less severe
  'ARID1B': '#2e7d32',  // dark green    — CSS1; absent 5th nail PATHOGNOMONIC; SWI/SNF BAF complex
  'EHMT1':  '#e65100',  // deep orange   — Kleefstra; 9q34.3 deletion; hypotonia+brachycephaly
  'KAT6B':  '#00695c',  // dark teal     — SBBYS/Genitopatellar; absent patella PATHOGNOMONIC
  'KANSL1': '#f57f17',  // amber-gold    — Koolen-de Vries; friendly behaviour PATHOGNOMONIC; 17q21.31
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

export default function HereditaryChromatinopathyEpigeneticAtlasPage() {
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
          🧬 Hereditary Chromatinopathy &amp; Epigenetic Syndrome Atlas
        </h1>
        <p style={{ color: '#555', fontSize: 13, margin: 0 }}>
          Complete 8-Gene Chromatin Modifier Genetics Reference · KMT2D-KDM6A-CREBBP-EP300-ARID1B-EHMT1-KAT6B-KANSL1 · 320 Patients · Seeds 3046-3053
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
              { label: 'Atlas',           value: 'Chromatinopathy' },
            ].map(({ label, value }) => (
              <div key={label} style={{ ...cardStyle, textAlign: 'center', padding: 12 }}>
                <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 18, fontWeight: 800, color: '#1565c0' }}>{value}</div>
              </div>
            ))}
          </div>

          <div style={cardStyle}>
            <h3 style={{ margin: '0 0 10px', fontSize: 14, color: '#1565c0' }}>Gene Loci &amp; Inheritance</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(260px,1fr))', gap: 8 }}>
              {overview.genes && overview.genes.map(g => (
                <div key={g} style={{
                  background: GENE_COLORS[g] + '11',
                  border: `1px solid ${GENE_COLORS[g]}44`,
                  borderRadius: 6, padding: '8px 12px',
                }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[g], fontSize: 13 }}>{g}</div>
                  <div style={{ fontSize: 11, color: '#555', marginTop: 2 }}>
                    {overview.gene_loci?.[g]}
                  </div>
                  <div style={{ fontSize: 11, color: '#444', marginTop: 3 }}>
                    {overview.inheritance_modes?.[g]?.slice(0, 120)}…
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div style={cardStyle}>
            <h3 style={{ margin: '0 0 10px', fontSize: 14, color: '#1565c0' }}>Key Clinical Rules</h3>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {overview.key_clinical_rules?.map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#333', marginBottom: 5, lineHeight: 1.5 }}>{r}</li>
              ))}
            </ul>
          </div>

          {overview.gene_panel_note && (
            <div style={{ ...cardStyle, background: '#e3f2fd', borderColor: '#90caf9' }}>
              <div style={{ fontSize: 12, color: '#1565c0' }}><strong>Gene Panel Note:</strong> {overview.gene_panel_note}</div>
            </div>
          )}
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {(tab === 'Gene Table' || tab === 'Clinical Atlas') && breakdown && !loading && (
        <div>
          {breakdown.genes?.map(g => (
            <div key={g.gene} style={{
              ...cardStyle,
              borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#888'}`,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8 }}>
                <div>
                  <span style={{ fontSize: 16, fontWeight: 800, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</span>
                  <span style={{ fontSize: 12, color: '#888', marginLeft: 8 }}>{g.locus}</span>
                  <div style={{ marginTop: 4 }}>
                    <Badge text={`${g.severe_pct}% severe`}    color="#c62828" />
                    <Badge text={`${g.moderate_pct}% moderate`} color="#e65100" />
                    <Badge text={`${g.mild_pct}% mild`}         color="#2e7d32" />
                    <Badge text={`IQ mean ${g.mean_iq}`}        color="#1565c0" />
                  </div>
                </div>
                <button
                  onClick={() => setExpandedGene(expandedGene === g.gene ? null : g.gene)}
                  style={{ fontSize: 11, padding: '4px 10px', border: '1px solid #ddd',
                    borderRadius: 4, cursor: 'pointer', background: '#f5f5f5' }}>
                  {expandedGene === g.gene ? '▲ Less' : '▼ More'}
                </button>
              </div>

              <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 6, fontSize: 11, color: '#555' }}>
                {g.cardiac_pct > 0 && <span>❤️ Cardiac {g.cardiac_pct}%</span>}
                {g.malignancy_pct > 0 && <span>⚠️ Malignancy {g.malignancy_pct}%</span>}
                {g.absent_nail_pct > 0 && <span>💅 Absent 5th nail {g.absent_nail_pct}%</span>}
                {g.absent_patella_pct > 0 && <span>🦴 Absent patella {g.absent_patella_pct}%</span>}
                {g.corpus_cal_pct > 0 && <span>🧠 CC agenesis {g.corpus_cal_pct}%</span>}
                {g.epilepsy_pct > 0 && <span>⚡ Epilepsy {g.epilepsy_pct}%</span>}
                {g.friendly_pct > 0 && <span>😊 Friendly behaviour {g.friendly_pct}%</span>}
                {g.hypothyroid_pct > 0 && <span>🦋 Hypothyroid {g.hypothyroid_pct}%</span>}
                {g.deletion_9q34_pct > 0 && <span>🔬 9q34.3 del {g.deletion_9q34_pct}%</span>}
                {g.deletion_17q21_pct > 0 && <span>🔬 17q21.31 del {g.deletion_17q21_pct}%</span>}
              </div>

              {expandedGene === g.gene && (
                <div style={{ marginTop: 12, fontSize: 12, color: '#333' }}>
                  <div style={{ background: '#f9f9f9', borderRadius: 6, padding: 10, marginBottom: 8 }}>
                    <strong>Disease Category:</strong> {g.disease_category}
                  </div>
                  <div style={{ marginBottom: 6 }}>
                    <strong>Inheritance:</strong> {g.inheritance}
                  </div>
                  <div>
                    <strong>Sample mutations:</strong> {g.sample_mutations?.join(' · ')}
                  </div>
                  <div style={{ marginTop: 6 }}>
                    <strong>Mean age at diagnosis:</strong> {g.mean_age_dx_mo} months
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && !loading && (
        <div>
          {definitions.definitions?.map((d, i) => (
            <div key={i} style={cardStyle}>
              <h3 style={{ margin: '0 0 6px', fontSize: 14, color: '#1565c0' }}>{d.term}</h3>
              <div style={{ marginBottom: 6 }}>
                {d.genes?.map(g => (
                  <Badge key={g} text={g} color={GENE_COLORS[g] || '#888'} />
                ))}
              </div>
              <div style={{ fontSize: 12, color: '#444', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                {d.definition}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
