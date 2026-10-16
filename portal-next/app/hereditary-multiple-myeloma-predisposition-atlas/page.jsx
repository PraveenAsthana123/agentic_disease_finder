'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-multiple-myeloma-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'FAM46C': '#4a148c',  // deep purple    — FMM first gene; Ashkenazi S103N; UPR sensitivity
  'BRCA2':  '#1b5e20',  // deep green     — HBOC; HRD olaparib/niraparib; POLO; FA-D1
  'ATM':    '#880e4f',  // deep magenta   — A-T; MM 2x; B-cell NHL; RADIOSENSITIVITY ABSOLUTE
  'TP53':   '#b71c1c',  // deep red       — LFS; del17p worst prognosis; AVOID RT ABSOLUTELY
  'TRAF3':  '#0d47a1',  // deep blue      — NF-κB non-canonical; BCMA upregulated; familial MM
  'CDKN2A': '#006064',  // dark cyan      — FAMM; del9p21 30-40% MM; CDK4/6i INACTIVE p16-null
  'NBN':    '#e65100',  // deep orange    — NBS1; 657del5 Slavic; B-cell NHL 60x HIGHEST
  'PALB2':  '#004d40',  // deep teal      — HBOC-2; TBCRC048 82% ORR; FA-N avoid alkylating
};

const GENE_INFO = {
  'FAM46C': { full: 'FMM / FAM46C-LOF / Familial-MM-First-Gene / Ashkenazi-S103N-1pct / Bortezomib-Sensitivity-UPR',                     locus: '1p12',    size: '155 aa / 17 kDa',    inh: 'LOF (AR/AD debated)' },
  'BRCA2':  { full: 'HBOC / MM-2-3x-Elevated / HRD-Olaparib-Niraparib / POLO-Maintenance / FA-D1-AML-PATHOGNOMONIC-Avoid-Alkylating',    locus: '13q12.3', size: '3418 aa / 384 kDa',  inh: 'AD LOF' },
  'ATM':    { full: 'A-T / MM-2x-Monoallelic / B-Cell-NHL-3-5x / CLL-4-5x / RADIOSENSITIVITY-ABSOLUTE-TBI-AVOID / Ceralasertib-ATRi',   locus: '11q22.3', size: '3056 aa / 350 kDa',  inh: 'AD/AR LOF' },
  'TP53':   { full: 'LFS / del17p-Somatic-10-15pct-WORST / PCL-Risk-Biallelic / AVOID-RADIATION-ABSOLUTELY / Daratumumab-del17p',         locus: '17p13.1', size: '393 aa / 43 kDa',    inh: 'AD LOF' },
  'TRAF3':  { full: 'HMM / NF-κB-Non-Canonical / NIK-Constitutive / BCMA-Upregulated / Teclistamab-Elranatamab-RELEVANT / del14q32-5-8pct', locus: '14q32.32', size: '568 aa / 64 kDa', inh: 'AD LOF' },
  'CDKN2A': { full: 'FAMM / del9p21-30-40pct-MM / CDK4-6i-INACTIVE-p16-Null / Venetoclax-t11-14 / Pancreatic-17-39pct-EUS-40yr',         locus: '9p21.3',  size: '156 aa / 17 kDa',    inh: 'AD LOF' },
  'NBN':    { full: 'NBS / 657del5-Slavic-1-in-150 / B-Cell-NHL-60x-HIGHEST / MM-3x / RADIOSENSITIVITY-Biallelic-ABSOLUTE-CI / MRN-PARP', locus: '8q21.3',  size: '754 aa / 85 kDa',    inh: 'AD/AR LOF' },
  'PALB2':  { full: 'HBOC-2 / TBCRC048-82pct-ORR-HIGHEST / Breast-53pct / FA-N-Avoid-Alkylating / MM-Described / BRCA2-Bridge',          locus: '16p12.2', size: '1186 aa / 131 kDa',  inh: 'AD LOF' },
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

export default function HereditaryMultipleMyelomaPredispositionAtlas() {
  const [tab, setTab]                   = useState('Overview');
  const [overview, setOverview]         = useState(null);
  const [breakdown, setBreakdown]       = useState(null);
  const [definitions, setDefinitions]   = useState(null);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState(null);
  const [expandedGene, setExpandedGene] = useState(null);
  const [expandedDef, setExpandedDef]   = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    const eps = [
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ];
    Promise.all(eps)
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const card = (title, value, sub, color = '#1b5e20') => (
    <div style={{ background: '#fff', border: `2px solid ${color}33`, borderRadius: 8, padding: '14px 18px', minWidth: 160 }}>
      <div style={{ fontSize: 12, color: '#555', marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 26, fontWeight: 700, color }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: '#777', marginTop: 2 }}>{sub}</div>}
    </div>
  );

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', maxWidth: 1400, margin: '0 auto', padding: '20px 24px' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#4a148c 100%)', color: '#fff', borderRadius: 12, padding: '22px 28px', marginBottom: 20 }}>
        <div style={{ fontSize: 11, opacity: 0.8, letterSpacing: 1, textTransform: 'uppercase', marginBottom: 6 }}>Hereditary Cancer Predisposition Atlas</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>🧬 Hereditary Multiple Myeloma Predisposition Atlas</h1>
        <div style={{ fontSize: 13, opacity: 0.85, marginTop: 6 }}>
          Complete 8-Gene Reference · FAM46C · BRCA2 · ATM · TP53 · TRAF3 · CDKN2A · NBN · PALB2 · 320-Patient Aggregate (8×40 seeds 3526–3533)
        </div>
        <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>
          FAM46C-First-FMM-Gene · TRAF3-NF-κB-BCMA-Upregulated · TP53-del17p-WORST-Avoid-RT · CDKN2A-CDK4-6i-INACTIVE · NBN-657del5-Slavic-NHL-60x-HIGHEST
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e0e0e0' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
            background: tab === t ? '#1a237e' : 'transparent', color: tab === t ? '#fff' : '#555',
            borderRadius: '6px 6px 0 0', borderBottom: tab === t ? '2px solid #1a237e' : '2px solid transparent',
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading atlas data…</div>}
      {error   && <div style={{ padding: 20, background: '#fff3e0', border: '1px solid #ff9800', borderRadius: 8, color: '#e65100' }}>⚠ {error}</div>}

      {/* ── OVERVIEW TAB ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 24 }}>
            {card('Total Patients',   overview.total_patients,    '8 genes × 40 patients', '#1a237e')}
            {card('MM Cases',          `${overview.mm_total_n} (${overview.mm_total_pct}%)`, 'Multiple myeloma confirmed', '#4a148c')}
            {card('Mean Age Onset',    `${overview.mean_age_onset} yr`,    'All MM diagnoses', '#006064')}
            {card('del17p (somatic)',  `${overview.del17p_n} (${overview.del17p_pct}%)`, 'Worst-prognosis cytogenetics', '#b71c1c')}
            {card('t(11;14)',          `${overview.t_11_14_n} (${overview.t_11_14_pct}%)`, 'Venetoclax-sensitive subtype', '#0d47a1')}
            {card('HRD Positive',     `${overview.hrd_n} (${overview.hrd_pct}%)`,  'PARP inhibitor eligible basis', '#1b5e20')}
          </div>

          <h3 style={{ marginBottom: 12, color: '#333' }}>Gene Summary — 8 Hereditary MM Predisposition Genes</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(320px,1fr))', gap: 14 }}>
            {overview.gene_summary?.map(gs => {
              const color = GENE_COLORS[gs.gene] || '#333';
              const info  = GENE_INFO[gs.gene] || {};
              return (
                <div key={gs.gene} onClick={() => setExpandedGene(expandedGene === gs.gene ? null : gs.gene)}
                  style={{ background: '#fff', border: `2px solid ${color}44`, borderRadius: 10, padding: '14px 16px', cursor: 'pointer' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <span style={{ fontSize: 18, fontWeight: 800, color }}>{gs.gene}</span>
                      <span style={{ fontSize: 11, color: '#888', marginLeft: 8 }}>{gs.locus}</span>
                      <Badge text={info.inh} color={color} />
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: 22, fontWeight: 700, color }}>{gs.mm_pct}%</div>
                      <div style={{ fontSize: 10, color: '#999' }}>MM rate</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: '#666', margin: '6px 0', fontStyle: 'italic' }}>{info.full}</div>
                  <div style={{ display: 'flex', gap: 12, marginTop: 8, flexWrap: 'wrap' }}>
                    <div><span style={{ fontSize: 10, color: '#999' }}>n </span><strong>{gs.n}</strong></div>
                    <div><span style={{ fontSize: 10, color: '#999' }}>MM </span><strong>{gs.mm_n}</strong></div>
                    <div><span style={{ fontSize: 10, color: '#999' }}>Age </span><strong>{gs.mean_age_onset}yr</strong></div>
                    <div><span style={{ fontSize: 10, color: '#999' }}>del17p </span><strong>{gs.del17p_n}</strong></div>
                    <div><span style={{ fontSize: 10, color: '#999' }}>HRD </span><strong>{gs.hrd_n}</strong></div>
                  </div>
                  {expandedGene === gs.gene && (
                    <div style={{ marginTop: 10, paddingTop: 10, borderTop: `1px solid ${color}33` }}>
                      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 4 }}>Key Distinctions</div>
                      {gs.key_distinctions?.map((d, i) => (
                        <div key={i} style={{ fontSize: 11, color: '#555', marginBottom: 3 }}>• {d}</div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── GENE TABLE TAB ── */}
      {tab === 'Gene Table' && overview && (
        <div>
          <h3 style={{ marginBottom: 12 }}>Complete Gene Reference — Hereditary MM Predisposition (8 Genes)</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene','Locus','Size','Syndrome','Inheritance','MM Risk','Pathognomonic','Key Rule'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overview.genes_detail?.map((gd, idx) => {
                const color = GENE_COLORS[gd.gene] || '#333';
                return (
                  <tr key={gd.gene} style={{ background: idx % 2 === 0 ? '#f8f9ff' : '#fff', verticalAlign: 'top' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 800, color, whiteSpace: 'nowrap' }}>{gd.gene}</td>
                    <td style={{ padding: '8px 10px', color: '#555' }}>{gd.locus}</td>
                    <td style={{ padding: '8px 10px', color: '#555', whiteSpace: 'nowrap' }}>{gd.protein_size?.split('/')[0]?.trim()}</td>
                    <td style={{ padding: '8px 10px', color: '#333', maxWidth: 140 }}>{gd.protein?.split('--').slice(2).join(' ').split('-').slice(0,4).join(' ')}</td>
                    <td style={{ padding: '8px 10px', color: '#555' }}>{gd.inheritance}</td>
                    <td style={{ padding: '8px 10px', color: '#333', maxWidth: 180 }}>{gd.cancer_risk}</td>
                    <td style={{ padding: '8px 10px', color: '#b71c1c', maxWidth: 200, fontWeight: 600 }}>{gd.pathognomonic}</td>
                    <td style={{ padding: '8px 10px', color: '#1a237e', maxWidth: 220 }}>{gd.surveillance_key}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* ── CLINICAL ATLAS TAB ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          <h3 style={{ marginBottom: 12 }}>Per-Gene Clinical Atlas — MM Outcomes (320 Patients, 8×40)</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 16 }}>
            {breakdown.breakdown?.map(b => {
              const color = GENE_COLORS[b.gene] || '#333';
              const mmBarW = Math.round(b.mm_pct);
              const hrdBarW = b.n > 0 ? Math.round(100 * b.hrd_n / b.n) : 0;
              return (
                <div key={b.gene} style={{ background: '#fff', border: `2px solid ${color}44`, borderRadius: 10, padding: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ fontSize: 16, fontWeight: 800, color }}>{b.gene}</span>
                    <span style={{ fontSize: 11, color: '#888' }}>{b.locus} · n={b.n} · age {b.mean_age_onset}yr</span>
                  </div>

                  {/* MM Rate bar */}
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontSize: 10, color: '#777', marginBottom: 2 }}>MM Rate: {b.mm_pct}% ({b.mm_n}/{b.n})</div>
                    <div style={{ height: 8, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{ width: `${mmBarW}%`, height: '100%', background: color, borderRadius: 4 }} />
                    </div>
                  </div>

                  {/* HRD bar */}
                  <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 10, color: '#777', marginBottom: 2 }}>HRD Positive (PARP eligible): {hrdBarW}% ({b.hrd_n}/{b.n})</div>
                    <div style={{ height: 6, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{ width: `${hrdBarW}%`, height: '100%', background: '#1b5e20', borderRadius: 4 }} />
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
                    <div style={{ background: '#fff3e0', borderRadius: 6, padding: '4px 10px', fontSize: 11 }}>
                      <strong>del17p:</strong> {b.del17p_n}/{b.n}
                    </div>
                    <div style={{ background: '#e8eaf6', borderRadius: 6, padding: '4px 10px', fontSize: 11 }}>
                      <strong>t(11;14):</strong> {b.t_11_14_n}/{b.n}
                    </div>
                    <div style={{ background: '#e8f5e9', borderRadius: 6, padding: '4px 10px', fontSize: 11 }}>
                      <strong>PARP elig:</strong> {b.parp_eligible_n}/{b.n}
                    </div>
                  </div>

                  <div style={{ borderTop: `1px solid ${color}22`, paddingTop: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color, marginBottom: 4 }}>Pathognomonic</div>
                    <div style={{ fontSize: 10, color: '#b71c1c', fontWeight: 600 }}>{b.pathognomonic}</div>
                  </div>
                  <div style={{ marginTop: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: '#333', marginBottom: 4 }}>Surveillance</div>
                    <div style={{ fontSize: 10, color: '#555' }}>{b.surveillance_key}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          {/* Key Clinical Distinctions */}
          <div style={{ background: '#e8eaf6', border: '1px solid #3f51b5', borderRadius: 10, padding: '16px 20px', marginBottom: 24 }}>
            <div style={{ fontWeight: 700, color: '#1a237e', marginBottom: 10, fontSize: 14 }}>⚡ Key Clinical Distinctions — Hereditary MM Predisposition</div>
            {definitions.key_clinical_distinctions?.map((d, i) => (
              <div key={i} style={{ fontSize: 12, color: '#333', marginBottom: 6, paddingLeft: 12, borderLeft: '3px solid #3f51b5' }}>
                <strong>{i+1}.</strong> {d}
              </div>
            ))}
          </div>

          {/* Gene Definitions */}
          <h3 style={{ marginBottom: 12 }}>Per-Gene Clinical Definitions</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {definitions.definitions?.map((def, idx) => {
              const gene = def.term.split(' — ')[0];
              const color = GENE_COLORS[gene] || '#333';
              return (
                <div key={idx} onClick={() => setExpandedDef(expandedDef === idx ? null : idx)}
                  style={{ background: '#fff', border: `2px solid ${color}44`, borderRadius: 10, padding: '12px 16px', cursor: 'pointer' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 800, color, fontSize: 15 }}>{def.term}</span>
                    <span style={{ color: '#888', fontSize: 18 }}>{expandedDef === idx ? '▾' : '▸'}</span>
                  </div>
                  {expandedDef === idx && (
                    <pre style={{ marginTop: 10, fontSize: 11, color: '#333', whiteSpace: 'pre-wrap', lineHeight: 1.6,
                      background: `${color}08`, borderRadius: 6, padding: 12, border: `1px solid ${color}22` }}>
                      {def.definition}
                    </pre>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
