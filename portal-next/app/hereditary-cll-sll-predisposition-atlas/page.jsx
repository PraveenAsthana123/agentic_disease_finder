'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cll-sll-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ATM':    '#b71c1c',  // deep red         — CLL 4-5x HIGHEST; del11q; RADIOSENSITIVITY ABSOLUTE biallelic
  'TP53':   '#004d40',  // deep teal        — LFS; del17p WORST CLL; AVOID RADIATION ABSOLUTELY
  'POT1':   '#1a237e',  // deep navy        — Familial CLL MOST SPECIFIC; shelterin; 4-6x; younger onset
  'CHEK2':  '#e65100',  // deep orange      — 1100delC European founder 1:200; IVS2+1G>A Ashkenazi; CLL 2-3x
  'NBN':    '#4a148c',  // deep purple      — 657del5 Slavic 1:150; NHL 60x HIGHEST; RADIOSENSITIVITY biallelic
  'BRCA2':  '#006064',  // dark cyan        — HBOC; FA-D1; HRD; PARP olaparib; CLL 2-3x
  'CARD11': '#880e4f',  // deep magenta     — BENTA GOF NF-κB; ibrutinib less effective GOF; MALT1 inhibitors
  'CDKN2A': '#1b5e20',  // deep green       — FAMM; del9p21 CLL 5-10%; p16-null palbociclib INACTIVE
};

const GENE_INFO = {
  'ATM':    { full: 'A-T / CLL-4-5x-HIGHEST / del11q-12pct / RADIOSENSITIVITY-ABSOLUTE-biallelic / Ceralasertib-ATRi / Ibrutinib-del11q',         locus: '11q22.3', size: '3056 aa / 350 kDa',  inh: 'AD/AR LOF' },
  'TP53':   { full: 'LFS / del17p-WORST-CLL / Venetoclax-FDA2016 / AVOID-RADIATION-ABSOLUTELY / Acalabrutinib-ELEVATE / WBMRI-Toronto',             locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'POT1':   { full: 'Familial-CLL-MOST-SPECIFIC / Shelterin-Telomere / 4-6x-CLL-Lifetime / Younger-Onset / Cascade-Testing-MANDATORY',              locus: '7q31.33', size: '634 aa / 71 kDa',   inh: 'AD LOF' },
  'CHEK2':  { full: '1100delC-European-1:200 / IVS2+1G>A-Ashkenazi / CLL-2-3x / ATM-CHEK2-Doubly-Heterozygous-Additive / Partial-HRD',             locus: '22q12.1', size: '543 aa / 61 kDa',   inh: 'AD LOF' },
  'NBN':    { full: 'NBS / 657del5-Slavic-1:150 / NHL-60x-HIGHEST / CLL-3-5x / RADIOSENSITIVITY-Biallelic / MRN-PARP / Zanubrutinib-Preferred',    locus: '8q21.3',  size: '754 aa / 85 kDa',   inh: 'AD/AR LOF' },
  'BRCA2':  { full: 'HBOC / FA-D1-Biallelic-PATHOGNOMONIC / CLL-2-3x / HRD-Positive / Olaparib-PARP / del13q-Proximate / Platinum-Sensitive',      locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF' },
  'CARD11': { full: 'BENTA-GOF-NF-kB / Ibrutinib-Less-Effective-GOF / Venetoclax-Preferred / MALT1-Inhibitor-JNJ / CBM-Complex-BCR',                locus: '7p22.2',  size: '1154 aa / 133 kDa', inh: 'AD LOF/GOF' },
  'CDKN2A': { full: 'FAMM / del9p21-CLL-5-10pct / p16-Null-Palbociclib-INACTIVE / Pancreatic-17-39x-EUS-40yr / Venetoclax-ACTIVE',                  locus: '9p21.3',  size: '156 aa / 17 kDa',   inh: 'AD LOF' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: `${color}18`, color, border: `1px solid ${color}55`,
      borderRadius: 4, fontSize: 10, padding: '2px 7px', marginLeft: 6, fontWeight: 600,
    }}>{text}</span>
  );
}

function card(label, value, sub, color) {
  return (
    <div style={{ background: '#fff', border: `2px solid ${color}33`, borderRadius: 10,
      padding: '14px 18px', minWidth: 140, flex: 1 }}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: 10, color: '#aaa' }}>{sub}</div>
    </div>
  );
}

export default function HereditaryCLLSLLPredispositionAtlasPage() {
  const [tab, setTab]                 = useState('Overview');
  const [overview, setOverview]       = useState(null);
  const [breakdown, setBreakdown]     = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState('');
  const [expandedGene, setExpandedGene] = useState(null);
  const [expandedDef, setExpandedDef]   = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true); setError('');
      try {
        const [ov, br, df] = await Promise.all([
          fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
          fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
          fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
        ]);
        setOverview(ov); setBreakdown(br); setDefinitions(df);
      } catch (e) { setError(e.message); }
      finally { setLoading(false); }
    }
    load();
  }, []);

  return (
    <div style={{ padding: '24px 28px', fontFamily: 'system-ui,sans-serif', background: '#f8f9ff', minHeight: '100vh' }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#b71c1c', marginBottom: 4 }}>
          🧬 Hereditary CLL / SLL Predisposition Atlas
        </h1>
        <div style={{ fontSize: 12, color: '#555' }}>
          Complete 8-Gene Reference — ATM · TP53 · POT1 · CHEK2 · NBN · BRCA2 · CARD11 · CDKN2A
          &nbsp;|&nbsp; 320-patient aggregate (8 × 40, seeds 3542–3549)
          &nbsp;|&nbsp; CLL/SLL hereditary predisposition · ibrutinib · venetoclax · ceralasertib · PARP inhibitors
        </div>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e0e0e0' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
            background: tab === t ? '#b71c1c' : 'transparent', color: tab === t ? '#fff' : '#555',
            borderRadius: '6px 6px 0 0', borderBottom: tab === t ? '2px solid #b71c1c' : '2px solid transparent',
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading atlas data…</div>}
      {error   && <div style={{ padding: 20, background: '#fff3e0', border: '1px solid #ff9800', borderRadius: 8, color: '#e65100' }}>⚠ {error}</div>}

      {/* ── OVERVIEW TAB ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 24 }}>
            {card('Total Patients',          overview.total_patients,                           '8 genes × 40 patients',           '#b71c1c')}
            {card('CLL / SLL Cases',         `${overview.cll_total_n} (${overview.cll_total_pct}%)`, 'CLL/SLL confirmed',          '#004d40')}
            {card('Mean Age Onset',          `${overview.mean_age_onset} yr`,                   'All CLL/SLL diagnoses',           '#1a237e')}
            {card('del11q',                  `${overview.del11q_n} (${overview.del11q_pct}%)`,  'ATM-pathway high-risk',           '#b71c1c')}
            {card('del17p',                  `${overview.del17p_n} (${overview.del17p_pct}%)`,  'TP53 WORST prognosis → venetoclax', '#004d40')}
            {card('Venetoclax Eligible',     `${overview.venetoclax_eligible_n} (${overview.venetoclax_eligible_pct}%)`, 'BCL2i first-line eligible', '#1b5e20')}
          </div>

          <h3 style={{ marginBottom: 12, color: '#333' }}>Gene Summary — 8 Hereditary CLL/SLL Predisposition Genes</h3>
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
                      <div style={{ fontSize: 22, fontWeight: 700, color }}>{gs.cll_pct}%</div>
                      <div style={{ fontSize: 10, color: '#999' }}>CLL rate</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: '#666', margin: '6px 0', fontStyle: 'italic' }}>{info.full}</div>
                  <div style={{ display: 'flex', gap: 12, marginTop: 8, flexWrap: 'wrap' }}>
                    <div><span style={{ fontSize: 10, color: '#999' }}>n </span><strong>{gs.n}</strong></div>
                    <div><span style={{ fontSize: 10, color: '#999' }}>CLL </span><strong>{gs.cll_n}</strong></div>
                    <div><span style={{ fontSize: 10, color: '#999' }}>Age </span><strong>{gs.mean_age_onset}yr</strong></div>
                    <div><span style={{ fontSize: 10, color: '#999' }}>del11q </span><strong>{gs.del11q_n}</strong></div>
                    <div><span style={{ fontSize: 10, color: '#999' }}>del17p </span><strong>{gs.del17p_n}</strong></div>
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
          <h3 style={{ marginBottom: 12 }}>Complete Gene Reference — Hereditary CLL/SLL Predisposition (8 Genes)</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#b71c1c', color: '#fff' }}>
                {['Gene','Locus','Size','Syndrome','Inheritance','CLL Risk','Pathognomonic','Key Rule'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overview.genes_detail?.map((gd, idx) => {
                const color = GENE_COLORS[gd.gene] || '#333';
                return (
                  <tr key={gd.gene} style={{ background: idx % 2 === 0 ? '#fff5f5' : '#fff', verticalAlign: 'top' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 800, color, whiteSpace: 'nowrap' }}>{gd.gene}</td>
                    <td style={{ padding: '8px 10px', color: '#555' }}>{gd.locus}</td>
                    <td style={{ padding: '8px 10px', color: '#555', whiteSpace: 'nowrap' }}>{gd.protein_size?.split('/')[0]?.trim()}</td>
                    <td style={{ padding: '8px 10px', color: '#333', maxWidth: 140 }}>{gd.syndrome}</td>
                    <td style={{ padding: '8px 10px', color: '#555' }}>{gd.inheritance}</td>
                    <td style={{ padding: '8px 10px', color: '#333', maxWidth: 180 }}>{gd.cancer_risk}</td>
                    <td style={{ padding: '8px 10px', color: '#b71c1c', maxWidth: 200, fontWeight: 600 }}>{gd.pathognomonic}</td>
                    <td style={{ padding: '8px 10px', color: '#1a237e', maxWidth: 220 }}>{gd.key_distinctions?.[2]?.substring(0, 120)}{gd.key_distinctions?.[2]?.length > 120 ? '…' : ''}</td>
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
          <h3 style={{ marginBottom: 12 }}>Per-Gene Clinical Atlas — CLL/SLL Outcomes (320 Patients, 8×40)</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 16 }}>
            {breakdown.breakdown?.map(b => {
              const color = GENE_COLORS[b.gene] || '#333';
              const cllBarW   = Math.round(b.cll_pct);
              const del11qBW  = b.n > 0 ? Math.round(100 * b.del11q_n / b.n) : 0;
              const del17pBW  = b.n > 0 ? Math.round(100 * b.del17p_n / b.n) : 0;
              return (
                <div key={b.gene} style={{ background: '#fff', border: `2px solid ${color}44`, borderRadius: 10, padding: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ fontSize: 16, fontWeight: 800, color }}>{b.gene}</span>
                    <span style={{ fontSize: 11, color: '#888' }}>{b.locus} · n={b.n} · age {b.mean_age_onset}yr</span>
                  </div>

                  {/* CLL Rate bar */}
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontSize: 10, color: '#777', marginBottom: 2 }}>CLL Rate: {b.cll_pct}% ({b.cll_n}/{b.n})</div>
                    <div style={{ height: 8, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{ width: `${cllBarW}%`, height: '100%', background: color, borderRadius: 4 }} />
                    </div>
                  </div>

                  {/* del11q bar */}
                  <div style={{ marginBottom: 6 }}>
                    <div style={{ fontSize: 10, color: '#777', marginBottom: 2 }}>del11q (ATM-pathway): {del11qBW}% ({b.del11q_n}/{b.n})</div>
                    <div style={{ height: 6, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{ width: `${del11qBW}%`, height: '100%', background: '#b71c1c', borderRadius: 4 }} />
                    </div>
                  </div>

                  {/* del17p bar */}
                  <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 10, color: '#777', marginBottom: 2 }}>del17p (TP53-WORST): {del17pBW}% ({b.del17p_n}/{b.n})</div>
                    <div style={{ height: 6, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{ width: `${del17pBW}%`, height: '100%', background: '#004d40', borderRadius: 4 }} />
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
                    <div style={{ background: '#fff3e0', borderRadius: 6, padding: '4px 10px', fontSize: 11 }}>
                      <strong>IGHV unmut:</strong> {b.ighv_unmutated_n}/{b.n}
                    </div>
                    <div style={{ background: '#e8eaf6', borderRadius: 6, padding: '4px 10px', fontSize: 11 }}>
                      <strong>HRD+:</strong> {b.hrd_positive_n}/{b.n}
                    </div>
                    <div style={{ background: '#e8f5e9', borderRadius: 6, padding: '4px 10px', fontSize: 11 }}>
                      <strong>Venetoclax:</strong> {b.venetoclax_elig_n}/{b.n}
                    </div>
                  </div>

                  <div style={{ borderTop: `1px solid ${color}22`, paddingTop: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color, marginBottom: 4 }}>Pathognomonic</div>
                    <div style={{ fontSize: 10, color: '#b71c1c', fontWeight: 600 }}>{b.pathognomonic?.substring(0, 180)}{b.pathognomonic?.length > 180 ? '…' : ''}</div>
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
          <div style={{ background: '#ffebee', border: '1px solid #f44336', borderRadius: 10, padding: '16px 20px', marginBottom: 24 }}>
            <div style={{ fontWeight: 700, color: '#b71c1c', marginBottom: 10, fontSize: 14 }}>⚡ Key Clinical Distinctions — Hereditary CLL/SLL Predisposition</div>
            {definitions.key_clinical_distinctions?.map((d, i) => (
              <div key={i} style={{ fontSize: 12, color: '#333', marginBottom: 6, paddingLeft: 12, borderLeft: '3px solid #b71c1c' }}>
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
