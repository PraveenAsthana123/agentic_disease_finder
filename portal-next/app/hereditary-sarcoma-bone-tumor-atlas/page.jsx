'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-sarcoma-bone-tumor-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'TP53':   '#b71c1c',  // deep red        — Li-Fraumeni; osteosarcoma; AVOID RADIATION
  'RB1':    '#1565c0',  // deep blue       — bilateral retinoblastoma; secondary osteosarcoma
  'DICER1': '#2e7d32',  // deep green      — PPB; ERMS; SLCT; FAPOL
  'EXT1':   '#e65100',  // deep orange     — HME Type 1; chondrosarcoma; higher risk
  'EXT2':   '#f57c00',  // amber           — HME Type 2; milder; same protocol
  'RECQL4': '#6a1b9a',  // deep purple     — RTS2; osteosarcoma 30%; poikiloderma
  'WRN':    '#006064',  // dark cyan       — Werner; adult progeria; bilateral cataracts
  'NF1':    '#1b5e20',  // forest green    — MPNST; most common hereditary sarcoma; FDG-PET
};

const GENE_INFO = {
  'TP53':   { full: 'Li-Fraumeni / Osteosarcoma 28-30% / AVOID RADIATION ABSOLUTELY',  locus: '17p13.1',  size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'RB1':    { full: 'Bilateral Retinoblastoma / Secondary Osteosarcoma 30-40%',          locus: '13q14.2',  size: '928 aa / 110 kDa',  inh: 'AD LOF' },
  'DICER1': { full: 'FAPOL / PPB PATHOGNOMONIC / ERMS-Cervix / SLCT-Ovary',             locus: '14q32.13', size: '1922 aa / 218 kDa', inh: 'AD LOF' },
  'EXT1':   { full: 'HME Type 1 / Multiple Osteochondromas / Chondrosarcoma 1-5%',      locus: '8q24.11',  size: '858 aa / 98 kDa',   inh: 'AD LOF' },
  'EXT2':   { full: 'HME Type 2 / Milder Than EXT1 / Chondrosarcoma ~1%',               locus: '11p11.2',  size: '718 aa / 83 kDa',   inh: 'AD LOF' },
  'RECQL4': { full: 'RTS2 / Osteosarcoma 30% HIGHEST / Poikiloderma PATHOGNOMONIC',     locus: '8q24.12',  size: '1208 aa / 133 kDa', inh: 'AR LOF' },
  'WRN':    { full: 'Werner Syndrome / Bilateral Cataracts <30yr / Sarcoma Adult',       locus: '8p12',     size: '1432 aa / 162 kDa', inh: 'AR LOF' },
  'NF1':    { full: 'NF Type 1 / MPNST 8-13% Most Common Hereditary Sarcoma / FDG-PET', locus: '17q11.2',  size: '2839 aa / 319 kDa', inh: 'AD LOF' },
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

export default function HereditorySarcomaBoneTumorAtlas() {
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
    const endpoints = [
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ];
    Promise.all(endpoints)
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const geneColor = g => GENE_COLORS[g] || '#607d8b';

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#0a0a0a', minHeight: '100vh', color: '#e8e8e8', padding: 24 }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a0000 0%,#001a4d 60%,#001a00 100%)', borderRadius: 12, padding: '28px 32px', marginBottom: 24 }}>
        <div style={{ fontSize: 11, color: '#ef9a9a', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 8 }}>
          Hereditary Disease Atlas · Sarcoma &amp; Bone Tumour Predisposition · 8-Gene Reference
        </div>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800, color: '#fff' }}>
          🧬 Hereditary Sarcoma &amp; Bone Tumour Predisposition Atlas
        </h1>
        <div style={{ marginTop: 10, color: '#b0bec5', fontSize: 13 }}>
          Complete 8-Gene Predisposition Reference · TP53 · RB1 · DICER1 · EXT1 · EXT2 · RECQL4 · WRN · NF1
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(GENE_COLORS).map(([g, c]) => (
            <Badge key={g} text={g} color={c} />
          ))}
        </div>
        <div style={{ marginTop: 10, fontSize: 11, color: '#ef9a9a', fontWeight: 600 }}>
          ⚠ RADIATION AVOIDANCE MANDATE: TP53/LFS · RB1 · NF1 — AVOID RADIATION ABSOLUTELY (radiation-field sarcoma documented)
        </div>
        <div style={{ marginTop: 6, fontSize: 11, color: '#80cbc4' }}>
          320-patient aggregate · 8 × 40 seeds · seeds 3118-3125 · TP53 osteosarcoma-28-30pct-LFS · RB1 secondary-osteosarcoma-30-40pct · DICER1 PPB-PATHOGNOMONIC · EXT1/EXT2 multiple-osteochondromas · RECQL4 RTS2-osteosarcoma-30pct-HIGHEST · WRN Werner-bilateral-cataracts-30yr · NF1 MPNST-8-13pct-FDG-PET-MANDATORY
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#b71c1c' : '#1e1e1e',
            color: tab === t ? '#fff' : '#aaa', fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#90caf9', padding: 40, textAlign: 'center' }}>Loading atlas data…</div>}
      {error   && <div style={{ color: '#ef9a9a', padding: 20, background: '#1a0000', borderRadius: 8 }}>Error: {error}</div>}

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Genes',    val: overview.total_genes },
              { label: 'Total Patients', val: overview.total_patients },
              { label: 'Seed Range',     val: overview.seed_range },
              { label: 'Patients/Gene',  val: 40 },
            ].map(({ label, val }) => (
              <div key={label} style={{ background: '#1e1e1e', borderRadius: 8, padding: '18px 20px', textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 800, color: '#ef9a9a' }}>{val}</div>
                <div style={{ fontSize: 12, color: '#888', marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>

          {/* Gene cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(340px,1fr))', gap: 16, marginBottom: 24 }}>
            {overview.genes.map(g => {
              const color = geneColor(g);
              const info  = GENE_INFO[g] || {};
              const inh   = (overview.inheritance_modes || {})[g] || '';
              return (
                <div key={g} style={{ background: '#1e1e1e', borderRadius: 8, padding: 18, borderLeft: `4px solid ${color}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <span style={{ fontSize: 20, fontWeight: 800, color }}>{g}</span>
                      <span style={{ fontSize: 11, color: '#888', marginLeft: 8 }}>{info.locus}</span>
                    </div>
                    <Badge text={info.inh || 'AD'} color={color} />
                  </div>
                  <div style={{ fontSize: 12, color: '#ccc', marginTop: 6 }}>{info.full}</div>
                  <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>{info.size}</div>
                  {inh && (
                    <div style={{ fontSize: 11, color: '#b0bec5', marginTop: 8, background: '#111', borderRadius: 4, padding: '6px 8px' }}>
                      {inh.substring(0, 220)}{inh.length > 220 ? '…' : ''}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Radiation warning box */}
          <div style={{ background: '#2a0000', border: '2px solid #b71c1c', borderRadius: 8, padding: '14px 20px', marginBottom: 16 }}>
            <div style={{ color: '#ef9a9a', fontWeight: 700, fontSize: 14, marginBottom: 8 }}>
              ☢ RADIATION AVOIDANCE — CRITICAL FOR TP53, RB1, AND NF1
            </div>
            <ul style={{ margin: 0, padding: '0 0 0 18px' }}>
              <li style={{ fontSize: 12, color: '#ffcdd2', marginBottom: 5 }}>
                <strong>TP53/LFS:</strong> Radiation-field sarcoma in LFS survivors documented — AVOID ALL RADIATION; use surgery/proton when possible
              </li>
              <li style={{ fontSize: 12, color: '#ffcdd2', marginBottom: 5 }}>
                <strong>RB1:</strong> In-field osteosarcoma after orbital radiotherapy extremely elevated — IAC preferred over external beam
              </li>
              <li style={{ fontSize: 12, color: '#ffcdd2' }}>
                <strong>NF1:</strong> Radiation-induced MPNST in radiation field documented — proton beam preferred if radiation unavoidable
              </li>
            </ul>
          </div>

          {/* Key clinical rules */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 20, marginBottom: 16 }}>
            <h3 style={{ margin: '0 0 14px', color: '#ef9a9a', fontSize: 15 }}>⚠ Key Clinical Rules</h3>
            <ul style={{ margin: 0, padding: '0 0 0 18px' }}>
              {(overview.key_clinical_rules || []).map((rule, i) => (
                <li key={i} style={{ fontSize: 12, color: '#ccc', marginBottom: 7, lineHeight: 1.5 }}>
                  {rule}
                </li>
              ))}
            </ul>
          </div>

          {/* Gene panel note */}
          {overview.gene_panel_note && (
            <div style={{ background: '#0d1b2a', borderRadius: 8, padding: 16, fontSize: 11, color: '#80cbc4', lineHeight: 1.7 }}>
              <strong style={{ color: '#90caf9' }}>Gene Panel &amp; Decision Tree:</strong>{' '}
              {overview.gene_panel_note}
            </div>
          )}
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          {breakdown.genes.map(g => {
            const color   = geneColor(g.gene);
            const isOpen  = expandedGene === g.gene;
            const info    = GENE_INFO[g.gene] || {};
            return (
              <div key={g.gene} style={{ background: '#1e1e1e', borderRadius: 8, marginBottom: 12, overflow: 'hidden', borderLeft: `4px solid ${color}` }}>
                <div
                  onClick={() => setExpandedGene(isOpen ? null : g.gene)}
                  style={{ padding: '14px 18px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                >
                  <div>
                    <span style={{ fontSize: 17, fontWeight: 700, color }}>{g.gene}</span>
                    <span style={{ fontSize: 12, color: '#888', marginLeft: 10 }}>{g.locus} · {info.full}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 12, alignItems: 'center', fontSize: 12 }}>
                    <span style={{ color: '#aaa' }}>n={g.n}</span>
                    <span style={{ color: '#ef9a9a' }}>Age {g.mean_age_diagnosis}yr</span>
                    <span style={{ color: isOpen ? '#fff' : '#666', fontSize: 16 }}>{isOpen ? '▲' : '▼'}</span>
                  </div>
                </div>
                {isOpen && (
                  <div style={{ padding: '0 18px 18px', borderTop: '1px solid #333' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 10, marginTop: 14 }}>
                      {Object.entries(g)
                        .filter(([k]) => k.endsWith('_pct'))
                        .map(([k, v]) => (
                          <div key={k} style={{ background: '#111', borderRadius: 6, padding: '10px 12px' }}>
                            <div style={{ fontSize: 18, fontWeight: 700, color }}>{v}%</div>
                            <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>
                              {k.replace(/_pct$/, '').replace(/_/g, ' ')}
                            </div>
                          </div>
                        ))}
                    </div>
                    {g.surveillance_key && (
                      <div style={{ marginTop: 12, fontSize: 11, color: '#80cbc4', background: '#0d1b2a', borderRadius: 4, padding: '8px 10px' }}>
                        <strong>Surveillance:</strong> {g.surveillance_key}
                      </div>
                    )}
                    {g.pathognomonic && (
                      <div style={{ marginTop: 8, fontSize: 11, color: '#ffcc80', background: '#1a1000', borderRadius: 4, padding: '8px 10px' }}>
                        <strong>Pathognomonic:</strong> {g.pathognomonic}
                      </div>
                    )}
                    {g.inheritance && (
                      <div style={{ marginTop: 8, fontSize: 11, color: '#b0bec5', background: '#111', borderRadius: 4, padding: '8px 10px', lineHeight: 1.6 }}>
                        <strong>Inheritance:</strong> {g.inheritance}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 20, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 16px', color: '#ef9a9a', fontSize: 15 }}>Syndrome Summary — Hereditary Sarcoma &amp; Bone Tumour Predisposition</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#111' }}>
                    {['Gene', 'Syndrome', 'Locus', 'Size', 'Inheritance', 'Pathognomonic', 'Surveillance Key', 'n', 'Dx Age'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#ef9a9a', borderBottom: '1px solid #333', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.genes.map((g, idx) => {
                    const color = geneColor(g.gene);
                    const info  = GENE_INFO[g.gene] || {};
                    return (
                      <tr key={g.gene} style={{ background: idx % 2 === 0 ? '#181818' : '#1e1e1e' }}>
                        <td style={{ padding: '8px 10px', color, fontWeight: 700 }}>{g.gene}</td>
                        <td style={{ padding: '8px 10px', color: '#ccc', maxWidth: 200 }}>{info.full}</td>
                        <td style={{ padding: '8px 10px', color: '#aaa' }}>{g.locus}</td>
                        <td style={{ padding: '8px 10px', color: '#aaa' }}>{info.size}</td>
                        <td style={{ padding: '8px 10px', color: '#b0bec5' }}>{info.inh}</td>
                        <td style={{ padding: '8px 10px', color: '#ffcc80', fontSize: 11 }}>{g.pathognomonic}</td>
                        <td style={{ padding: '8px 10px', color: '#80cbc4', fontSize: 11 }}>{g.surveillance_key ? g.surveillance_key.split(';')[0] : '—'}</td>
                        <td style={{ padding: '8px 10px', color: '#e0e0e0' }}>{g.n}</td>
                        <td style={{ padding: '8px 10px', color: '#ef9a9a' }}>{g.mean_age_diagnosis}yr</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Protein size reference */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 20 }}>
            <h3 style={{ margin: '0 0 16px', color: '#ef9a9a', fontSize: 15 }}>Protein Size Reference</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 10 }}>
              {Object.entries(GENE_INFO).map(([g, info]) => {
                const color = geneColor(g);
                return (
                  <div key={g} style={{ background: '#111', borderRadius: 6, padding: '12px 14px', borderTop: `3px solid ${color}` }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color }}>{g}</div>
                    <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>{info.size}</div>
                    <div style={{ fontSize: 11, color: '#aaa', marginTop: 2 }}>{info.locus}</div>
                    <div style={{ fontSize: 10, color: '#666', marginTop: 4 }}>{info.inh}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ marginBottom: 12, fontSize: 12, color: '#888' }}>
            {definitions.count} clinical definitions · TP53 LFS sarcoma · RB1 secondary osteosarcoma · DICER1 PPB surveillance · EXT1/EXT2 HME chondrosarcoma · RECQL4 Rothmund-Thomson · WRN Werner syndrome · NF1 MPNST FDG-PET · hereditary sarcoma differential guide
          </div>
          {definitions.definitions.map((d, i) => (
            <div key={i} style={{ background: '#1e1e1e', borderRadius: 8, marginBottom: 12, overflow: 'hidden' }}>
              <div style={{ background: '#b71c1c', padding: '10px 16px', fontSize: 13, fontWeight: 700, color: '#fff' }}>
                {d.term.replace(/-/g, ' ')}
              </div>
              <div style={{ padding: '14px 16px', fontSize: 12, color: '#ccc', lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
                {d.definition}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
