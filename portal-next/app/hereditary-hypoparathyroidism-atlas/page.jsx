'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hypoparathyroidism-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'GCM2':    '#1565c0',  // deep blue       — most common isolated HP, master parathyroid TF
  'PTH':     '#0277bd',  // medium blue     — direct PTH gene, isolated HP, rare
  'CASR':    '#b71c1c',  // deep red        — ADH1 GOF, cinacalcet CI, nephrocalcinosis risk
  'GNA11':   '#c62828',  // crimson         — ADH2 GOF, downstream of CaSR
  'TBCE':    '#4a148c',  // deep purple     — HRD/Sanjad-Sakati, AR, Arab founder
  'FAM111A': '#6a1b9a',  // purple          — KCS2, AD GOF, dense bones, normal IQ
  'GATA3':   '#e65100',  // deep orange     — HDR triad, SNHL first feature
  'SOX3':    '#2e7d32',  // dark green      — X-linked, regulatory pitfall, males only
};

const GENE_INFO = {
  'GCM2':    { full: 'GCM2 / Glial Cells Missing 2 / 495aa', locus: '6p24.2',   size: '495 aa / 47 kDa (master parathyroid TF; AD haploinsufficiency or AR biallelic; most common familial isolated HP)', inh: 'AD/AR LOF' },
  'PTH':     { full: 'PTH / Parathyroid Hormone / 115aa preproPTH', locus: '11p15.3', size: '115 aa preproPTH → 84 aa mature / 9.4 kDa (signal peptide or coding mutations → PTH non-secretion; glands present but dysfunctional)', inh: 'AR/AD' },
  'CASR':    { full: 'CASR / Calcium-Sensing Receptor / 1078aa', locus: '3q13.3',  size: '1078 aa / 120 kDa (GPCR Class C; GOF → ADH1; renal CaSR activated → relative hypercalciuria PATHOGNOMONIC; cinacalcet ABSOLUTE CI)', inh: 'AD GOF' },
  'GNA11':   { full: 'GNA11 / Gα11 G-protein / 359aa', locus: '19p13.3',  size: '359 aa / 42 kDa (downstream CaSR; GOF → ADH2; same phenotype as ADH1; cinacalcet ineffective)', inh: 'AD GOF' },
  'TBCE':    { full: 'TBCE / Tubulin-Specific Chaperone E / 527aa', locus: '1q42.3',   size: '527 aa / 59 kDa (microtubule assembly; HRD/Sanjad-Sakati: HP + ID + dysmorphic; Arab founder IVS1-2A>G)', inh: 'AR' },
  'FAM111A': { full: 'FAM111A / FAM111 Protease / 611aa', locus: '11q13.1',  size: '611 aa / 69 kDa (serine protease GOF; KCS2: HP + dense bones + short stature; NORMAL intelligence vs TBCE)', inh: 'AD GOF' },
  'GATA3':   { full: 'GATA3 / GATA-Binding Protein 3 / 444aa', locus: '10p14',    size: '444 aa / 48 kDa (dual ZF TF; HDR = HP + bilateral SNHL + renal anomalies; SNHL often first feature)', inh: 'AD LOF' },
  'SOX3':    { full: 'SOX3 / SRY-Box TF 3 / 446aa', locus: 'Xq27.1',   size: '446 aa / 46 kDa (regulatory region insertion/deletion; X-linked — males only; standard sequencing MISSES)', inh: 'XL' },
};

function GeneChip({ gene, active, onClick }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span
      onClick={() => onClick && onClick(gene)}
      style={{
        background: col, color: '#fff', borderRadius: 4,
        padding: '3px 10px', fontSize: 12, fontWeight: 700,
        margin: '0 3px 4px 0', cursor: onClick ? 'pointer' : 'default',
        opacity: active === null || active === gene ? 1 : 0.45,
        border: active === gene ? '2px solid #fff' : '2px solid transparent',
        display: 'inline-block',
      }}
    >{gene}</span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 130 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HypoparathyroidismAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [activeGene, setActiveGene] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    async function load() {
      setLoading(true); setError('');
      try {
        const [ov, bd, df] = await Promise.all([
          fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
          fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
          fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
        ]);
        setOverview(ov); setBreakdown(bd); setDefinitions(df);
      } catch (e) { setError('API error: ' + e.message); }
      setLoading(false);
    }
    load();
  }, []);

  const style = {
    page:  { background: '#0f172a', color: '#e2e8f0', minHeight: '100vh', padding: '24px 28px', fontFamily: 'system-ui,sans-serif' },
    title: { fontSize: 22, fontWeight: 800, color: '#f8fafc', marginBottom: 4 },
    sub:   { fontSize: 13, color: '#64748b', marginBottom: 20 },
    tabs:  { display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' },
    tab:   (active) => ({
      padding: '7px 18px', borderRadius: 6, cursor: 'pointer', fontSize: 13, fontWeight: 600,
      background: active ? '#1565c0' : '#1e293b',
      color: active ? '#fff' : '#94a3b8',
      border: `1px solid ${active ? '#1565c0' : '#334155'}`,
    }),
    card:  { background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: 16, marginBottom: 14 },
    label: { fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 },
    val:   { fontSize: 14, color: '#f1f5f9' },
    warn:  { background: '#450a0a', border: '1px solid #ef4444', borderRadius: 6, padding: '8px 14px', fontSize: 13, color: '#fca5a5', marginBottom: 10 },
    info:  { background: '#172554', border: '1px solid #1e40af', borderRadius: 6, padding: '8px 14px', fontSize: 13, color: '#93c5fd', marginBottom: 10 },
    ok:    { background: '#052e16', border: '1px solid #16a34a', borderRadius: 6, padding: '8px 14px', fontSize: 13, color: '#86efac', marginBottom: 10 },
  };

  if (loading) return <div style={style.page}><div style={{ color: '#64748b' }}>Loading Hereditary-Hypoparathyroidism-Atlas…</div></div>;
  if (error)   return <div style={style.page}><div style={{ color: '#ef4444' }}>{error}</div></div>;
  if (!overview) return null;

  const gs = overview.gene_summary || [];

  return (
    <div style={style.page}>
      <div style={style.title}>🧬 Hereditary-Hypoparathyroidism-Atlas</div>
      <div style={style.sub}>
        Complete 8-Gene Reference · GCM2 · PTH · CASR (ADH1) · GNA11 (ADH2) · TBCE (HRD) · FAM111A (KCS2) · GATA3 (HDR) · SOX3 (XL) ·
        320 patients · seeds {overview.seeds}
      </div>

      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 20 }}>
        {(overview.genes || []).map(g => (
          <GeneChip key={g} gene={g} active={activeGene} onClick={g2 => setActiveGene(activeGene === g2 ? null : g2)} />
        ))}
      </div>

      <div style={style.tabs}>
        {TABS.map(t => <button key={t} style={style.tab(tab === t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'Overview' && (
        <div>
          <div style={style.warn}>
            ⚠️ ADH1 (CASR GOF): Cinacalcet ABSOLUTE CI — worsens hypocalcaemia. Ca+calcitriol: nephrocalcinosis risk. Use rPTH preferentially.
          </div>
          <div style={style.info}>
            ℹ️ Key discriminator: 24h urine Ca — HIGH in ADH1/ADH2 (CASR/GNA11) despite hypocalcaemia; LOW in all other HP causes.
          </div>
          <div style={style.ok}>
            ✓ SOX3 diagnostic pitfall: Standard gene sequencing MISSES X-linked HP (regulatory region insertion). Request Xq27.1 CNV/MLPA.
          </div>

          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <MetricCard label="Total Patients" value={overview.n_patients} />
            <MetricCard label="Genes" value={overview.n_genes} />
            <MetricCard label="Mean Serum Ca" value={`${overview.aggregate_metrics?.mean_serum_ca_mmol} mmol/L`} warn />
            <MetricCard label="Mean PTH" value={`${overview.aggregate_metrics?.mean_serum_pth_pgml} pg/mL`} warn />
            <MetricCard label="Seizures" value={`${overview.aggregate_metrics?.seizures_pct}%`} warn />
            <MetricCard label="Tetany" value={`${overview.aggregate_metrics?.tetany_pct}%`} />
            <MetricCard label="Nephrocalcinosis" value={`${overview.aggregate_metrics?.nephrocalcinosis_pct}%`} warn />
            <MetricCard label="High Urine Ca (ADH)" value={`${overview.aggregate_metrics?.high_urine_ca_pct}%`} />
          </div>

          <div style={style.card}>
            <div style={style.label}>Key Clinical Discriminators</div>
            {(overview.key_discriminators || []).map((d, i) => (
              <div key={i} style={{ fontSize: 13, color: '#cbd5e1', marginBottom: 6, paddingLeft: 10, borderLeft: '3px solid #1565c0' }}>
                {d}
              </div>
            ))}
          </div>

          <div style={style.card}>
            <div style={style.label}>Gene Summary</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
                  {['Gene','Locus','Inheritance','Mean Ca','Mean PTH','Sz%','High UCa%'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '6px 10px', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {gs.map(g => {
                  const col = GENE_COLORS[g.gene] || '#555';
                  const dim = activeGene && activeGene !== g.gene;
                  return (
                    <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b', opacity: dim ? 0.45 : 1 }}>
                      <td style={{ padding: '7px 10px' }}>
                        <span style={{ background: col, color: '#fff', borderRadius: 3, padding: '2px 8px', fontWeight: 700, fontSize: 11 }}>{g.gene}</span>
                      </td>
                      <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.locus}</td>
                      <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.inheritance}</td>
                      <td style={{ padding: '7px 10px', color: g.mean_ca < 1.8 ? '#ef4444' : '#fbbf24' }}>{g.mean_ca}</td>
                      <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.mean_pth}</td>
                      <td style={{ padding: '7px 10px', color: g.sz_rate_pct > 60 ? '#f87171' : '#94a3b8' }}>{g.sz_rate_pct}%</td>
                      <td style={{ padding: '7px 10px', color: g.high_urine_ca_pct > 50 ? '#ef4444' : '#94a3b8' }}>{g.high_urine_ca_pct}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── GENE TABLE TAB ── */}
      {tab === 'Gene Table' && (
        <div>
          {(breakdown?.genes || []).map(g => {
            const col   = GENE_COLORS[g.gene] || '#555';
            const info  = GENE_INFO[g.gene] || {};
            const dim   = activeGene && activeGene !== g.gene;
            return (
              <div key={g.gene} style={{ ...style.card, opacity: dim ? 0.45 : 1, borderLeft: `4px solid ${col}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <span style={{ background: col, color: '#fff', borderRadius: 4, padding: '3px 12px', fontWeight: 800, fontSize: 13 }}>{g.gene}</span>
                  <span style={{ fontSize: 13, color: '#94a3b8' }}>{info.full}</span>
                  <span style={{ marginLeft: 'auto', fontSize: 11, color: '#64748b' }}>{info.inh} · {info.locus}</span>
                </div>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>{info.size}</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(260px,1fr))', gap: 10 }}>
                  <div>
                    <div style={style.label}>Protein / Mechanism</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'pre-wrap' }}>{g.protein_size?.slice(0, 400)}…</div>
                  </div>
                  <div>
                    <div style={style.label}>Inheritance / Clinical</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'pre-wrap' }}>{g.inheritance?.slice(0, 400)}…</div>
                  </div>
                  <div>
                    <div style={style.label}>Disease Category</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'pre-wrap' }}>{g.disease_category?.slice(0, 400)}…</div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── CLINICAL ATLAS TAB ── */}
      {tab === 'Clinical Atlas' && (
        <div>
          <div style={{ ...style.card, marginBottom: 18 }}>
            <div style={style.label}>Filter by Gene</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
              {(overview.genes || []).map(g => (
                <GeneChip key={g} gene={g} active={activeGene} onClick={g2 => setActiveGene(activeGene === g2 ? null : g2)} />
              ))}
              {activeGene && <button onClick={() => setActiveGene(null)} style={{ background: '#334155', color: '#94a3b8', border: 'none', borderRadius: 4, padding: '3px 10px', fontSize: 12, cursor: 'pointer' }}>Clear</button>}
            </div>
          </div>
          {(breakdown?.genes || [])
            .filter(g => !activeGene || g.gene === activeGene)
            .map(g => {
              const col = GENE_COLORS[g.gene] || '#555';
              const sample = g.patients?.slice(0, 5) || [];
              return (
                <div key={g.gene} style={{ ...style.card, borderLeft: `4px solid ${col}`, marginBottom: 16 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                    <span style={{ background: col, color: '#fff', borderRadius: 4, padding: '3px 12px', fontWeight: 800, fontSize: 13 }}>{g.gene}</span>
                    <span style={{ fontSize: 12, color: '#64748b' }}>{g.n_patients} patients · {GENE_INFO[g.gene]?.locus} · {GENE_INFO[g.gene]?.inh}</span>
                  </div>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
                        {['ID','Sex','Onset(mo)','Ca mmol','PTH pg/mL','Urine Ca','Mg mmol','Sz','Tetany','Nephrocal','Treatment'].map(h => (
                          <th key={h} style={{ textAlign: 'left', padding: '5px 8px' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {sample.map(p => (
                        <tr key={p.id} style={{ borderBottom: '1px solid #1e293b' }}>
                          <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{p.id}</td>
                          <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{p.gender}</td>
                          <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{p.onset_months}</td>
                          <td style={{ padding: '5px 8px', color: p.serum_ca_mmol < 1.75 ? '#ef4444' : '#fbbf24' }}>{p.serum_ca_mmol}</td>
                          <td style={{ padding: '5px 8px', color: p.serum_pth_pgml < 5 ? '#f87171' : '#94a3b8' }}>{p.serum_pth_pgml}</td>
                          <td style={{ padding: '5px 8px', color: p.urine_ca_24h?.startsWith('HIGH') ? '#ef4444' : '#86efac', fontWeight: 600, fontSize: 10 }}>{p.urine_ca_24h}</td>
                          <td style={{ padding: '5px 8px', color: p.serum_mg_mmol < 0.65 ? '#fbbf24' : '#94a3b8' }}>{p.serum_mg_mmol}</td>
                          <td style={{ padding: '5px 8px', color: p.seizures ? '#ef4444' : '#4ade80' }}>{p.seizures ? 'Y' : 'N'}</td>
                          <td style={{ padding: '5px 8px', color: p.tetany ? '#fbbf24' : '#4ade80' }}>{p.tetany ? 'Y' : 'N'}</td>
                          <td style={{ padding: '5px 8px', color: p.nephrocalcinosis ? '#ef4444' : '#4ade80' }}>{p.nephrocalcinosis ? 'Y' : 'N'}</td>
                          <td style={{ padding: '5px 8px', color: '#94a3b8', fontSize: 10 }}>{p.treatment}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {g.n_patients > 5 && <div style={{ fontSize: 11, color: '#64748b', marginTop: 6 }}>Showing 5 of {g.n_patients} patients</div>}
                </div>
              );
            })}
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'Definitions' && (
        <div>
          {(definitions?.definitions || []).map((d, i) => (
            <div key={i} style={style.card}>
              <div style={{ fontWeight: 700, color: '#38bdf8', marginBottom: 8, fontSize: 14 }}>{d.term}</div>
              <div style={{ fontSize: 13, color: '#cbd5e1', whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>{d.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
