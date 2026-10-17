'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-wm-lpl-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MYD88':  '#1a237e',  // deep navy        — L265P >90% WM; NF-κB TLR adaptor; ibrutinib/zanubrutinib
  'CXCR4':  '#880e4f',  // deep magenta     — WHIM 40% WM; SDF-1/CXCL12; plerixafor FDA2024
  'BTK':    '#4a148c',  // deep purple      — XLA; WM drug target; ibrutinib/zanubrutinib/pirtobrutinib
  'ATM':    '#b71c1c',  // deep red         — RADIOSENSITIVITY ABSOLUTE; ceralasertib ATRi; WM 2x
  'TP53':   '#004d40',  // deep teal        — LFS; del17p WORST WM; AVOID RADIATION ABSOLUTELY
  'NBN':    '#e65100',  // deep orange      — NBS1; 657del5 Slavic; NHL 60x HIGHEST; MRN-PARP
  'CARD11': '#006064',  // dark cyan        — BENTA; NF-κB downstream; DLBCL; ibrutinib sensitivity
  'PIK3CD': '#1b5e20',  // deep green       — APDS1; PI3Kδ GOF; leniolisib FDA2023; idelalisib
};

const GENE_INFO = {
  'MYD88':  { full: 'MYD88-L265P / NF-κB-TLR-Adaptor / WM-80pct-Rate / MOST-COMMON-WM-Somatic / Ibrutinib-Zanubrutinib-BTKi',                    locus: '3p22.2',  size: '296 aa / 33 kDa',    inh: 'AD LOF/GOF' },
  'CXCR4':  { full: 'CXCR4WHIM / SDF-1-CXCL12 / WM-40pct-Co-Mutation / Inferior-BTKi / Zanubrutinib+Venetoclax / Plerixafor-FDA2024',             locus: '2q22.1',  size: '360 aa / 41 kDa',    inh: 'AD GOF' },
  'BTK':    { full: 'XLA / BTK-Kinase-Target / Ibrutinib-Zanubrutinib-Acalabrutinib / C481S-Pirtobrutinib / WM-Non-Hereditary-Use',                  locus: 'Xq22.1',  size: '659 aa / 76 kDa',    inh: 'XLR LOF' },
  'ATM':    { full: 'A-T / WM-2x-Elevated / RADIOSENSITIVITY-ABSOLUTE / Ceralasertib-ATRi / RT-Reduce-20-30pct / Busulfan-Conditioning-No-TBI',     locus: '11q22.3', size: '3056 aa / 350 kDa',  inh: 'AD/AR LOF' },
  'TP53':   { full: 'LFS / del17p-WORST-WM / Venetoclax+Zanubrutinib / AVOID-RADIATION-ABSOLUTELY / Daratumumab-del17p / WBMRI-Toronto',             locus: '17p13.1', size: '393 aa / 43 kDa',    inh: 'AD LOF' },
  'NBN':    { full: 'NBS / 657del5-Slavic-1-in-150 / NHL-60x-HIGHEST / WM-3x / RADIOSENSITIVITY-Biallelic / MRN-PARP / Zanubrutinib-Preferred',     locus: '8q21.3',  size: '754 aa / 85 kDa',    inh: 'AD/AR LOF' },
  'CARD11': { full: 'BENTA / NF-κB-Downstream / DLBCL-ibrutinib-Sensitive / WM-B-Cell-NF-kB / BCR-Signalling / CARD11-GOF-LOF-Dual',                locus: '7p22.2',  size: '1154 aa / 133 kDa',  inh: 'AD LOF/GOF' },
  'PIK3CD': { full: 'APDS1 / PI3Kδ-GOF / Leniolisib-FDA2023 / Idelalisib-WM / IgM-High-BM-Infiltration / Acyclovir-MANDATORY-APDS1',               locus: '1p36.22', size: '1044 aa / 119 kDa',  inh: 'AD GOF' },
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

export default function HereditaryWMLPLPredispositionAtlas() {
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

  const card = (title, value, sub, color = '#1a237e') => (
    <div style={{ background: '#fff', border: `2px solid ${color}33`, borderRadius: 8, padding: '14px 18px', minWidth: 160 }}>
      <div style={{ fontSize: 12, color: '#555', marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 26, fontWeight: 700, color }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: '#777', marginTop: 2 }}>{sub}</div>}
    </div>
  );

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', maxWidth: 1400, margin: '0 auto', padding: '20px 24px' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#880e4f 100%)', color: '#fff', borderRadius: 12, padding: '22px 28px', marginBottom: 20 }}>
        <div style={{ fontSize: 11, opacity: 0.8, letterSpacing: 1, textTransform: 'uppercase', marginBottom: 6 }}>Hereditary Cancer Predisposition Atlas</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>🧬 Hereditary Waldenström Macroglobulinemia / LPL Predisposition Atlas</h1>
        <div style={{ fontSize: 13, opacity: 0.85, marginTop: 6 }}>
          Complete 8-Gene Reference · MYD88 · CXCR4 · BTK · ATM · TP53 · NBN · CARD11 · PIK3CD · 320-Patient Aggregate (8×40 seeds 3534–3541)
        </div>
        <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>
          MYD88-L265P-&gt;90%-WM · CXCR4WHIM-40%-Inferior-BTKi · BTK-Drug-Target-Zanubrutinib-ASPEN · NBN-657del5-NHL-60x-HIGHEST · PIK3CD-Leniolisib-FDA2023
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
            {card('Total Patients',          overview.total_patients,              '8 genes × 40 patients',            '#1a237e')}
            {card('WM / LPL Cases',          `${overview.wm_total_n} (${overview.wm_total_pct}%)`, 'WM/LPL confirmed', '#880e4f')}
            {card('Mean Age Onset',          `${overview.mean_age_onset} yr`,      'All WM/LPL diagnoses',             '#006064')}
            {card('MYD88 L265P',             `${overview.myd88_l265p_n} (${overview.myd88_l265p_pct}%)`, 'Somatic BTKi-sensitive driver', '#4a148c')}
            {card('CXCR4 Mutant',            `${overview.cxcr4_n} (${overview.cxcr4_pct}%)`,    'Inferior BTKi → zanu+venetoclax', '#880e4f')}
            {card('Zanubrutinib Eligible',   `${overview.zanubrutinib_eligible_n} (${overview.zanubrutinib_eligible_pct}%)`, 'ASPEN trial superior BTKi', '#1b5e20')}
          </div>

          <h3 style={{ marginBottom: 12, color: '#333' }}>Gene Summary — 8 Hereditary WM/LPL Predisposition Genes</h3>
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
                      <div style={{ fontSize: 22, fontWeight: 700, color }}>{gs.wm_pct}%</div>
                      <div style={{ fontSize: 10, color: '#999' }}>WM rate</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: '#666', margin: '6px 0', fontStyle: 'italic' }}>{info.full}</div>
                  <div style={{ display: 'flex', gap: 12, marginTop: 8, flexWrap: 'wrap' }}>
                    <div><span style={{ fontSize: 10, color: '#999' }}>n </span><strong>{gs.n}</strong></div>
                    <div><span style={{ fontSize: 10, color: '#999' }}>WM </span><strong>{gs.wm_n}</strong></div>
                    <div><span style={{ fontSize: 10, color: '#999' }}>Age </span><strong>{gs.mean_age_onset}yr</strong></div>
                    <div><span style={{ fontSize: 10, color: '#999' }}>MYD88 L265P </span><strong>{gs.myd88_l265p_n}</strong></div>
                    <div><span style={{ fontSize: 10, color: '#999' }}>CXCR4 </span><strong>{gs.cxcr4_n}</strong></div>
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
          <h3 style={{ marginBottom: 12 }}>Complete Gene Reference — Hereditary WM/LPL Predisposition (8 Genes)</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene','Locus','Size','Syndrome','Inheritance','WM Risk','Pathognomonic','Key Rule'].map(h => (
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
                    <td style={{ padding: '8px 10px', color: '#333', maxWidth: 140 }}>{gd.syndrome}</td>
                    <td style={{ padding: '8px 10px', color: '#555' }}>{gd.inheritance}</td>
                    <td style={{ padding: '8px 10px', color: '#333', maxWidth: 180 }}>{gd.wm_risk}</td>
                    <td style={{ padding: '8px 10px', color: '#b71c1c', maxWidth: 200, fontWeight: 600 }}>{gd.pathognomonic}</td>
                    <td style={{ padding: '8px 10px', color: '#1a237e', maxWidth: 220 }}>{gd.key_rule?.substring(0, 120)}{gd.key_rule?.length > 120 ? '…' : ''}</td>
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
          <h3 style={{ marginBottom: 12 }}>Per-Gene Clinical Atlas — WM/LPL Outcomes (320 Patients, 8×40)</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 16 }}>
            {breakdown.breakdown?.map(b => {
              const color = GENE_COLORS[b.gene] || '#333';
              const wmBarW  = Math.round(b.wm_pct);
              const myd88BW = b.n > 0 ? Math.round(100 * b.myd88_l265p_n / b.n) : 0;
              return (
                <div key={b.gene} style={{ background: '#fff', border: `2px solid ${color}44`, borderRadius: 10, padding: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ fontSize: 16, fontWeight: 800, color }}>{b.gene}</span>
                    <span style={{ fontSize: 11, color: '#888' }}>{b.locus} · n={b.n} · age {b.mean_age_onset}yr</span>
                  </div>

                  {/* WM Rate bar */}
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontSize: 10, color: '#777', marginBottom: 2 }}>WM Rate: {b.wm_pct}% ({b.wm_n}/{b.n})</div>
                    <div style={{ height: 8, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{ width: `${wmBarW}%`, height: '100%', background: color, borderRadius: 4 }} />
                    </div>
                  </div>

                  {/* MYD88 L265P bar */}
                  <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 10, color: '#777', marginBottom: 2 }}>MYD88 L265P (BTKi-eligible): {myd88BW}% ({b.myd88_l265p_n}/{b.n})</div>
                    <div style={{ height: 6, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{ width: `${myd88BW}%`, height: '100%', background: '#1a237e', borderRadius: 4 }} />
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
                    <div style={{ background: '#fff3e0', borderRadius: 6, padding: '4px 10px', fontSize: 11 }}>
                      <strong>CXCR4 mut:</strong> {b.cxcr4_n}/{b.n}
                    </div>
                    <div style={{ background: '#e8eaf6', borderRadius: 6, padding: '4px 10px', fontSize: 11 }}>
                      <strong>del6q:</strong> {b.del6q_n}/{b.n}
                    </div>
                    <div style={{ background: '#e8f5e9', borderRadius: 6, padding: '4px 10px', fontSize: 11 }}>
                      <strong>Zanu elig:</strong> {b.zanubrutinib_eligible_n}/{b.n}
                    </div>
                  </div>

                  <div style={{ borderTop: `1px solid ${color}22`, paddingTop: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color, marginBottom: 4 }}>Pathognomonic</div>
                    <div style={{ fontSize: 10, color: '#b71c1c', fontWeight: 600 }}>{b.pathognomonic}</div>
                  </div>
                  <div style={{ marginTop: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: '#333', marginBottom: 4 }}>Key Rule</div>
                    <div style={{ fontSize: 10, color: '#555' }}>{b.key_rule?.substring(0, 160)}{b.key_rule?.length > 160 ? '…' : ''}</div>
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
            <div style={{ fontWeight: 700, color: '#1a237e', marginBottom: 10, fontSize: 14 }}>⚡ Key Clinical Distinctions — Hereditary WM/LPL Predisposition</div>
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
