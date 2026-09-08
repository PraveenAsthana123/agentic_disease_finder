'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-dystonia-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  TOR1A:  '#1565c0',  // deep blue    — DYT1 most common genetic generalised dystonia
  SGCE:   '#4a148c',  // deep purple  — DYT11 myoclonus-dystonia paternal imprint
  GCH1:   '#2e7d32',  // deep green   — DRD/DYT5a L-DOPA curative TREATABLE
  TH:     '#00695c',  // deep teal    — DRD/DYT5b AR infantile
  KMT2B:  '#e65100',  // deep orange  — DYT28 childhood focal→generalised ID 40%
  THAP1:  '#880e4f',  // deep pink    — DYT6 laryngeal Ashkenazi
  ATP1A3: '#b71c1c',  // deep red     — AHC fever absolute trigger
  ANO3:   '#37474f',  // dark grey    — DYT24 craniocervical tremulous
};

const GENE_INFO = {
  TOR1A:  { full: 'TOR1A / 332aa',   locus: '9q34.11',  size: '332 aa',  inh: 'AD',          disease: 'DYT-TOR1A/DYT1 — GAG DELETION PATHOGNOMONIC 95%+ / MOST COMMON GENETIC GENERALISED DYSTONIA / DBS-GPi >70% / Lower Limb Onset Childhood' },
  SGCE:   { full: 'SGCE / 437aa',    locus: '7q21.3',   size: '437 aa',  inh: 'AD-Pat-Imp',  disease: 'DYT-SGCE/DYT11 — MYOCLONUS PREDOMINATES PATHOGNOMONIC / ALCOHOL RESPONSIVE (diagnostic, not treatment) / PATERNAL IMPRINTING mandatory counselling' },
  GCH1:   { full: 'GCH1 / 250aa',    locus: '14q22.2',  size: '250 aa',  inh: 'AD',          disease: 'DRD/DYT5a/Segawa — DIURNAL VARIATION PATHOGNOMONIC / L-DOPA CURATIVE MUST TRY BEFORE BOTOX / Female 4:1 / TREATABLE' },
  TH:     { full: 'TH / 528aa',      locus: '11p15.5',  size: '528 aa',  inh: 'AR',          disease: 'DRD/DYT5b — CSF HVA+5-HIAA REDUCED PATHOGNOMONIC / AR severe infantile / Pterin NORMAL DDx GCH1 / Oculogyric Crises PATHOGNOMONIC' },
  KMT2B:  { full: 'KMT2B / 3969aa',  locus: '19q13.12', size: '3969 aa', inh: 'AD-de-novo',  disease: 'DYT-KMT2B/DYT28 — FOCAL LOWER LIMB→GENERALISED PATHOGNOMONIC / ID 40% / DBS-GPi HIGHLY EFFECTIVE even with ID / 90% De Novo' },
  THAP1:  { full: 'THAP1 / 213aa',   locus: '8p21.3',   size: '213 aa',  inh: 'AD',          disease: 'DYT-THAP1/DYT6 — LARYNGEAL DYSTONIA PATHOGNOMONIC / Young Adult / Ashkenazi Founder / Botox Thyroarytenoid Gold Standard' },
  ATP1A3: { full: 'ATP1A3 / 1013aa', locus: '19q13.2',  size: '1013 aa', inh: 'AD-de-novo',  disease: 'AHC/CAPOS/RDP — FEVER ABSOLUTE TRIGGER AHC ATTACKS / SLEEP RESOLVES PATHOGNOMONIC / Flunarizine First-Line / FEVER PROTOCOL MANDATORY' },
  ANO3:   { full: 'ANO3 / 913aa',    locus: '11p14.3',  size: '913 aa',  inh: 'AD',          disease: 'DYT-ANO3/DYT24 — CRANIOCERVICAL TREMULOUS DYSTONIA PATHOGNOMONIC / Tremor Distinguishes from THAP1 / Botox + Propranolol' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('MANDATORY') || flag.includes('ABSOLUTE') ? '#880e4f'
    : flag.includes('CURATIVE') || flag.includes('TREATABLE') || flag.includes('FIRST') ? '#2e7d32'
    : flag.includes('CI') || flag.includes('NEVER') || flag.includes('AVOID') ? '#c62828'
    : flag.includes('FOUNDER') || flag.includes('DISTINCTIVE') ? '#00695c'
    : flag.includes('FATAL') || flag.includes('LETHAL') || flag.includes('FEVER') ? '#e65100'
    : flag.includes('DE-NOVO') || flag.includes('DDx') ? '#4a148c'
    : flag.includes('EFFECTIVE') || flag.includes('IMPROVED') ? '#1565c0'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditaryDystoniaAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, def]) => { setOverview(ov); setBreakdown(br); setDefinitions(def); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ fontFamily: 'sans-serif', padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h1 style={{ color: '#1565c0', marginBottom: 4 }}>
        🧬 Hereditary Dystonia Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Hereditary Dystonia Atlas —
        TOR1A (DYT1/GPi) · SGCE (DYT11/Myoclonus/PatImp) · GCH1 (DRD/L-DOPA-Curative) · TH (AR-DRD/Infantile) ·
        KMT2B (DYT28/ID/DBS) · THAP1 (DYT6/Laryngeal/Ashkenazi) · ATP1A3 (AHC/Fever-Protocol) · ANO3 (DYT24/Tremulous)
        &nbsp;|&nbsp; 320 patients · seeds 2158-2165
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            style={{
              padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: activeTab === t ? '#1565c0' : '#e3f2fd',
              color: activeTab === t ? '#fff' : '#1565c0', fontWeight: 600,
            }}>
            {t}
          </button>
        ))}
      </div>

      {loading && <p style={{ color: '#888' }}>Loading dystonia atlas data…</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {/* ── OVERVIEW ── */}
      {activeTab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.n_patients },
              { label: 'Seeds', value: overview.seeds },
              { label: 'Alive %', value: `${overview.alive_pct}%` },
              { label: 'Genes', value: Object.keys(overview.gene_summaries || {}).length },
            ].map(card => (
              <div key={card.label} style={{ background: '#e3f2fd', borderRadius: 8, padding: '14px 18px' }}>
                <div style={{ fontSize: 26, fontWeight: 700, color: '#1565c0' }}>{card.value}</div>
                <div style={{ fontSize: 12, color: '#555' }}>{card.label}</div>
              </div>
            ))}
          </div>

          {/* Key flags */}
          <div style={{ marginBottom: 20 }}>
            <h3 style={{ color: '#1565c0', marginBottom: 8 }}>Key Clinical Flags</h3>
            <div>{(overview.key_flags || []).map(f => <FLAG_BADGE key={f} flag={f} />)}</div>
          </div>

          {/* Gene summary cards */}
          <h3 style={{ color: '#1565c0', marginBottom: 12 }}>Gene Summary</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 14 }}>
            {Object.entries(overview.gene_summaries || {}).map(([gene, s]) => (
              <div key={gene} style={{
                border: `2px solid ${GENE_COLORS[gene] || '#1565c0'}`,
                borderRadius: 8, padding: 14,
              }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#1565c0', fontSize: 16 }}>
                  {gene} <span style={{ fontWeight: 400, fontSize: 13 }}>({s.locus} · {s.protein_size} · {s.inheritance})</span>
                </div>
                <div style={{ fontSize: 12, color: '#333', margin: '6px 0' }}>{s.pathognomonic_summary}</div>
                <div style={{ fontSize: 12 }}>
                  <span style={{ color: '#555' }}>Top etiology: </span>
                  <span style={{ color: GENE_COLORS[gene] || '#1565c0' }}>{s.top_etiology}</span>
                </div>
                <div style={{ fontSize: 12 }}>
                  <span style={{ color: '#555' }}>Top dystonia: </span>{s.top_dystonia_type}
                  &nbsp;|&nbsp;
                  <span style={{ color: '#555' }}>Top trigger: </span>{s.top_trigger}
                </div>
                <div style={{ fontSize: 12, color: '#888' }}>
                  {s.n_patients} pts · alive {s.alive_pct}%
                </div>
              </div>
            ))}
          </div>

          {/* Distributions */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 20 }}>
            <div>
              <h4 style={{ color: '#1565c0' }}>Etiology Distribution</h4>
              {Object.entries(overview.etiology_distribution || {}).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <div style={{ flex: 1, fontSize: 12, color: '#333' }}>{k}</div>
                  <div style={{ width: Math.round(v / 3), height: 14, background: '#1565c0', borderRadius: 3 }} />
                  <div style={{ fontSize: 12, color: '#555', minWidth: 30 }}>{v}</div>
                </div>
              ))}
            </div>
            <div>
              <h4 style={{ color: '#1565c0' }}>Dystonia Type Distribution</h4>
              {Object.entries(overview.dystonia_type_distribution || {}).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <div style={{ flex: 1, fontSize: 12, color: '#333' }}>{k}</div>
                  <div style={{ width: Math.round(v / 3), height: 14, background: '#42a5f5', borderRadius: 3 }} />
                  <div style={{ fontSize: 12, color: '#555', minWidth: 30 }}>{v}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {activeTab === 'Gene Table' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1565c0', color: '#fff' }}>
                {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease / Key Facts'].map(h => (
                  <th key={h} style={{ padding: '8px 12px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(GENE_INFO).map(([gene, info], i) => (
                <tr key={gene} style={{ background: i % 2 === 0 ? '#f5f9ff' : '#fff' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
                  <td style={{ padding: '8px 12px' }}>{info.locus}</td>
                  <td style={{ padding: '8px 12px' }}>{info.size}</td>
                  <td style={{ padding: '8px 12px' }}>{info.inh}</td>
                  <td style={{ padding: '8px 12px', fontSize: 12 }}>{info.disease}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {activeTab === 'Clinical Atlas' && breakdown && (
        <div>
          {Object.entries(breakdown).map(([gene, data]) => (
            <div key={gene} style={{
              border: `2px solid ${GENE_COLORS[gene] || '#1565c0'}`,
              borderRadius: 10, marginBottom: 24, padding: 18,
            }}>
              <h3 style={{ color: GENE_COLORS[gene] || '#1565c0', marginBottom: 4 }}>
                {gene} — {data.locus} · {data.protein_size} · {data.inheritance.split(';')[0]}
              </h3>
              <div style={{ fontSize: 12, color: '#444', marginBottom: 8 }}>{data.protein}</div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div>
                  <strong style={{ color: '#b71c1c' }}>Pathognomonic:</strong>
                  <p style={{ fontSize: 13, color: '#333' }}>{data.pathognomonic}</p>
                  <strong>Treatment:</strong>
                  <p style={{ fontSize: 13, color: '#333' }}>{data.treatment}</p>
                  <strong style={{ color: '#c62828' }}>Contraindications:</strong>
                  <p style={{ fontSize: 13, color: '#555' }}>{data.contraindications}</p>
                </div>
                <div>
                  <strong>Monitoring:</strong>
                  <p style={{ fontSize: 13, color: '#333' }}>
                    {Array.isArray(data.monitoring) ? data.monitoring.join(' · ') : data.monitoring}
                  </p>
                  <strong>Etiologies:</strong>
                  <div>
                    {(data.etiologies || []).map(e => (
                      <div key={e.type} style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '3px 0' }}>
                        <div style={{ width: e.pct * 1.2, height: 12, background: GENE_COLORS[gene] || '#1565c0', borderRadius: 2 }} />
                        <span style={{ fontSize: 12 }}>{e.type} ({e.pct}%)</span>
                      </div>
                    ))}
                  </div>
                  <strong style={{ display: 'block', marginTop: 8 }}>Dystonia Types:</strong>
                  <div>
                    {(data.dystonia_types || []).map(s => (
                      <div key={s.type} style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '3px 0' }}>
                        <div style={{ width: s.pct * 0.8, height: 12, background: '#42a5f5', borderRadius: 2 }} />
                        <span style={{ fontSize: 12 }}>{s.type} ({s.pct}%)</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Key concepts */}
              <div style={{ marginTop: 12 }}>
                <strong>Key Concepts:</strong>
                <div style={{ marginTop: 4 }}>
                  {(data.concepts || []).map(c => <FLAG_BADGE key={c} flag={c} />)}
                </div>
              </div>

              {/* Thresholds */}
              <div style={{ marginTop: 10 }}>
                <strong>Thresholds & Decision Points:</strong>
                <ul style={{ fontSize: 12, color: '#444', margin: '4px 0', paddingLeft: 20 }}>
                  {(data.thresholds || []).map(t => <li key={t}>{t}</li>)}
                </ul>
              </div>

              {/* Lifecycle */}
              <div style={{ marginTop: 10 }}>
                <strong>Lifecycle Stages:</strong>
                <ol style={{ fontSize: 12, color: '#444', margin: '4px 0', paddingLeft: 20 }}>
                  {(data.lifecycle || []).map(l => <li key={l}>{l}</li>)}
                </ol>
              </div>

              {/* References */}
              <div style={{ marginTop: 8 }}>
                <strong>References:</strong>
                <ul style={{ fontSize: 12, color: '#666', margin: '4px 0', paddingLeft: 20 }}>
                  {(data.references || []).map(r => <li key={r}>{r}</li>)}
                </ul>
              </div>

              <div style={{ marginTop: 8, fontSize: 12, color: '#888' }}>
                {data.n_patients} patients · alive {data.alive_pct}%
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {activeTab === 'Definitions' && definitions && (
        <div>
          {Object.entries(definitions).map(([section, entries]) => (
            <div key={section} style={{ marginBottom: 24 }}>
              <h3 style={{ color: '#1565c0', marginBottom: 8 }}>
                {section.replace(/_/g, ' ').toUpperCase()}
              </h3>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <tbody>
                  {Object.entries(entries).map(([term, def], i) => (
                    <tr key={term} style={{ background: i % 2 === 0 ? '#f5f9ff' : '#fff' }}>
                      <td style={{ padding: '7px 12px', fontWeight: 600, color: '#1565c0', width: '26%', verticalAlign: 'top' }}>
                        {term.replace(/_/g, ' ')}
                      </td>
                      <td style={{ padding: '7px 12px', color: '#333' }}>{def}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
