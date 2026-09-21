'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-mineralocorticoid-excess-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'HSD11B2': '#b71c1c',  // deep red       — AME, cortisol→MR, THF/THE >10
  'SCNN1B':  '#1565c0',  // deep blue      — Liddle ENaC-β GOF, amiloride curative
  'SCNN1G':  '#0277bd',  // medium blue    — Liddle ENaC-γ GOF, amiloride curative
  'WNK4':    '#1b5e20',  // dark green     — Gordon PHA2B, NCC overactivation
  'WNK1':    '#2e7d32',  // green          — Gordon PHA2A, intronic deletion pitfall
  'KLHL3':   '#4caf50',  // medium green   — Gordon PHA2C, E3 adaptor, AD/AR
  'CUL3':    '#827717',  // dark yellow    — Gordon PHA2E, dominant negative, severe
  'NR3C2':   '#6a1b9a',  // deep purple    — Geller, progesterone agonist, spiro CI
};

const GENE_INFO = {
  'HSD11B2': { full: 'HSD11B2 / 11β-HSD2 / 405aa', locus: '11q22.1', size: '405 aa / 44 kDa (renal cortisol→cortisone inactivator; LOF → cortisol saturates MR → AME; THF/THE ratio >10 PATHOGNOMONIC; AR)', inh: 'AR' },
  'SCNN1B':  { full: 'SCNN1B / ENaC-β / 640aa', locus: '16p12.2', size: '640 aa / 72 kDa (ENaC β subunit; PY-motif GOF → NEDD4-2 cannot bind → ENaC constitutively open; amiloride curative; spiro ineffective)', inh: 'AD GOF' },
  'SCNN1G':  { full: 'SCNN1G / ENaC-γ / 649aa', locus: '16p12.2', size: '649 aa / 74 kDa (ENaC γ subunit; PY-motif truncation; identical Liddle phenotype to SCNN1B; adjacent 16p12.2 — tested together)', inh: 'AD GOF' },
  'WNK4':    { full: 'WNK4 / WNK kinase 4 / 1243aa', locus: '17q21.31', size: '1243 aa / 135 kDa (WNK4; acidic motif missense → KLHL3 cannot bind → WNK4 accumulates → NCC hyperphosphorylation → Gordon PHA2B)', inh: 'AD' },
  'WNK1':    { full: 'WNK1 / WNK kinase 1 / 2382aa', locus: '12p13.33', size: '2382 aa / 251 kDa (WNK1; large intron 1 deletion → KS-WNK1 lost → L-WNK1 dominant → Gordon PHA2A; exon sequencing MISSES)', inh: 'AD' },
  'KLHL3':   { full: 'KLHL3 / Kelch-like 3 / 587aa', locus: '5q31.2', size: '587 aa / 66 kDa (CUL3-RING E3 substrate adaptor for WNK1/4; AD missense milder; AR biallelic severe; Gordon PHA2C)', inh: 'AD/AR' },
  'CUL3':    { full: 'CUL3 / Cullin 3 / 768aa', locus: '2q36.2', size: '768 aa / 89 kDa (CUL3; Δexon9 dominant negative; most severe Gordon PHA2E; extra-renal comorbidities; de novo dominant)', inh: 'AD' },
  'NR3C2':   { full: 'NR3C2 / Mineralocorticoid Receptor / 984aa', locus: '4q31.23', size: '984 aa / 107 kDa (MR; S810L GOF → progesterone full agonist → Geller; spironolactone ABSOLUTE CI; pregnancy HTN crisis)', inh: 'AD GOF' },
};

const SYNDROME_COLORS = {
  'AME':    '#b71c1c',
  'Liddle': '#1565c0',
  'Gordon': '#1b5e20',
  'Geller': '#6a1b9a',
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

function SyndromeTag({ syndrome }) {
  const col = SYNDROME_COLORS[syndrome] || '#555';
  return (
    <span style={{ background: col, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 700 }}>
      {syndrome}
    </span>
  );
}

function MetricCard({ label, value, sub, warn, ok }) {
  const color = warn ? '#ef4444' : ok ? '#22c55e' : '#38bdf8';
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : ok ? '#22c55e' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 130 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function MineralocorticoidExcessAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [activeGene, setActiveGene] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const geneList = overview?.genes || Object.keys(GENE_INFO);

  if (loading) return <div style={{ color: '#94a3b8', padding: 40, textAlign: 'center' }}>Loading Hereditary-Mineralocorticoid-Excess-Atlas…</div>;
  if (error)   return <div style={{ color: '#ef4444', padding: 40 }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'sans-serif', padding: '24px 20px' }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', marginBottom: 6 }}>
          🧬 Hereditary-Mineralocorticoid-Excess-Atlas
        </h1>
        <p style={{ fontSize: 13, color: '#94a3b8', maxWidth: 900 }}>
          Complete 8-gene reference: <b>AME</b> (HSD11B2) · <b>Liddle</b> (SCNN1B, SCNN1G) · <b>Gordon/PHAII</b> (WNK4, WNK1, KLHL3, CUL3) · <b>Geller</b> (NR3C2) —
          hereditary HTN syndromes with distinct mechanisms, treatments, and diagnostic pitfalls.
          320-patient aggregate cohort · seeds 2934–2941.
        </p>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {geneList.map(g => <GeneChip key={g} gene={g} active={activeGene} onClick={g => setActiveGene(prev => prev === g ? null : g)} />)}
        </div>
        {/* Syndrome legend */}
        <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(SYNDROME_COLORS).map(([s, c]) => (
            <span key={s} style={{ background: c, color: '#fff', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700 }}>{s}</span>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #334155', paddingBottom: 8 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#1e40af' : '#1e293b', color: tab === t ? '#fff' : '#94a3b8',
            border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontSize: 13, fontWeight: 600,
          }}>{t}</button>
        ))}
      </div>

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <MetricCard label="Total Patients" value={overview.n_patients} sub="8 × 40, seeds 2934–2941" />
            <MetricCard label="Mean SBP" value={`${overview.aggregate_metrics?.mean_sbp_mmhg} mmHg`} warn={true} />
            <MetricCard label="Mean K+" value={`${overview.aggregate_metrics?.mean_serum_k_mmol} mmol/L`} />
            <MetricCard label="Gordon Phenotype" value={`${overview.aggregate_metrics?.gordon_phenotype_pct}%`} ok={false} sub="WNK4/WNK1/KLHL3/CUL3" />
            <MetricCard label="Low Aldosterone" value={`${overview.aggregate_metrics?.low_aldosterone_pct}%`} sub="AME/Liddle/Geller" />
            <MetricCard label="Severe HyperK" value={`${overview.aggregate_metrics?.severe_hyperkalemia_pct}%`} warn={true} sub="K+ >6.5" />
            <MetricCard label="Preg HTN Crisis" value={`${overview.aggregate_metrics?.bp_crisis_pregnancy_pct}%`} warn={true} sub="Geller females" />
          </div>

          {/* Key clinical rules */}
          <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#fbbf24', marginBottom: 10, fontSize: 14 }}>⚠️ Critical Clinical Rules</div>
            {(overview.key_clinical_rules || []).map((r, i) => (
              <div key={i} style={{ fontSize: 12, color: '#e2e8f0', marginBottom: 6, paddingLeft: 8, borderLeft: '2px solid #fbbf24' }}>
                {r}
              </div>
            ))}
          </div>

          {/* Gene summary table */}
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1e293b' }}>
                  {['Gene', 'Syndrome', 'Locus', 'Inheritance', 'Mean SBP', 'Mean K+', 'Low Aldo%', 'Thiazide Curative', 'Amiloride Curative'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', fontWeight: 600, borderBottom: '1px solid #334155' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.gene_summary || []).map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#131f35', opacity: activeGene && activeGene !== g.gene ? 0.5 : 1 }}>
                    <td style={{ padding: '7px 10px' }}><GeneChip gene={g.gene} /></td>
                    <td style={{ padding: '7px 10px' }}><SyndromeTag syndrome={g.syndrome} /></td>
                    <td style={{ padding: '7px 10px', color: '#94a3b8', fontFamily: 'monospace' }}>{g.locus}</td>
                    <td style={{ padding: '7px 10px', color: '#e2e8f0' }}>{g.inheritance}</td>
                    <td style={{ padding: '7px 10px', color: '#ef4444', fontWeight: 700 }}>{g.mean_sbp}</td>
                    <td style={{ padding: '7px 10px', color: g.mean_k > 5 ? '#ef4444' : '#22c55e', fontWeight: 700 }}>{g.mean_k}</td>
                    <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.low_aldo_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.thiazide_curative ? '#22c55e' : '#475569' }}>{g.thiazide_curative ? '✓ Yes' : '—'}</td>
                    <td style={{ padding: '7px 10px', color: g.amiloride_curative ? '#22c55e' : '#475569' }}>{g.amiloride_curative ? '✓ Yes' : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 14 }}>
          {Object.entries(GENE_INFO)
            .filter(([g]) => !activeGene || g === activeGene)
            .map(([gene, info]) => (
              <div key={gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[gene]}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                  <GeneChip gene={gene} />
                  <span style={{ fontSize: 11, color: '#64748b', fontFamily: 'monospace' }}>{info.locus}</span>
                </div>
                <div style={{ fontSize: 12, color: '#e2e8f0', marginBottom: 4, fontWeight: 600 }}>{info.full}</div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>{info.size}</div>
                <div style={{ fontSize: 11, background: '#0f172a', borderRadius: 4, padding: '4px 8px', color: '#38bdf8', fontWeight: 600 }}>
                  Inheritance: {info.inh}
                </div>
              </div>
            ))}
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {(breakdown.genes || [])
            .filter(g => !activeGene || g.gene === activeGene)
            .map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[g.gene]}` }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ fontSize: 13, color: '#f1f5f9', fontWeight: 600 }}>{g.locus}</span>
                  <span style={{ fontSize: 12, color: '#94a3b8' }}>n={g.n_patients} · SBP={g.mean_sbp} mmHg · K+={g.mean_k} mmol/L</span>
                  {g.gordon_pct === 100 && <span style={{ background: '#1b5e20', color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>Gordon phenotype</span>}
                  {g.thiazide_curative && <span style={{ background: '#164e63', color: '#67e8f9', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>Thiazide curative</span>}
                  {g.amiloride_curative && <span style={{ background: '#1e3a5f', color: '#93c5fd', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>Amiloride curative</span>}
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8, lineHeight: 1.5 }}>{g.protein_size?.slice(0, 300)}…</div>
                <div style={{ fontSize: 11, color: '#cbd5e1', marginBottom: 8, lineHeight: 1.5 }}><b style={{ color: '#fbbf24' }}>Disease:</b> {g.disease_category?.slice(0, 400)}…</div>
                <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.5 }}><b style={{ color: '#34d399' }}>Pathway:</b> {g.disease_pathway?.slice(0, 300)}…</div>

                {/* Treatment distribution */}
                {g.treatment_distribution && (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Treatment distribution (n={g.n_patients}):</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {Object.entries(g.treatment_distribution).sort((a, b) => b[1] - a[1]).map(([tx, cnt]) => (
                        <span key={tx} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 4, padding: '2px 8px', fontSize: 11, color: '#94a3b8' }}>
                          {tx}: <b style={{ color: '#e2e8f0' }}>{cnt}</b>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Sample patients */}
                {g.patients?.length > 0 && (
                  <div style={{ marginTop: 10, overflowX: 'auto' }}>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Sample patients (first 10):</div>
                    <table style={{ fontSize: 11, borderCollapse: 'collapse', width: '100%' }}>
                      <thead>
                        <tr style={{ background: '#0f172a' }}>
                          {['ID', 'Sex', 'Onset(mo)', 'SBP', 'K+', 'HCO3', 'Cl-', 'Aldo↓', 'Gordon', 'Treatment'].map(h => (
                            <th key={h} style={{ padding: '4px 8px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #1e293b' }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {g.patients.map((p, i) => (
                          <tr key={p.id} style={{ background: i % 2 === 0 ? '#0f172a' : '#131f35' }}>
                            <td style={{ padding: '3px 8px', fontFamily: 'monospace', color: '#94a3b8' }}>{p.id}</td>
                            <td style={{ padding: '3px 8px' }}>{p.gender}</td>
                            <td style={{ padding: '3px 8px' }}>{p.onset_months}</td>
                            <td style={{ padding: '3px 8px', color: '#ef4444', fontWeight: 700 }}>{p.sbp_mmhg}</td>
                            <td style={{ padding: '3px 8px', color: p.serum_k_mmol > 5 ? '#ef4444' : p.serum_k_mmol < 3 ? '#f59e0b' : '#e2e8f0', fontWeight: 700 }}>{p.serum_k_mmol}</td>
                            <td style={{ padding: '3px 8px' }}>{p.serum_hco3_mmol}</td>
                            <td style={{ padding: '3px 8px', color: p.serum_cl_mmol > 106 ? '#f59e0b' : '#94a3b8' }}>{p.serum_cl_mmol}</td>
                            <td style={{ padding: '3px 8px', color: p.aldosterone_low ? '#22c55e' : '#94a3b8' }}>{p.aldosterone_low ? 'Yes' : 'No'}</td>
                            <td style={{ padding: '3px 8px', color: p.gordon_phenotype ? '#34d399' : '#475569' }}>{p.gordon_phenotype ? 'Yes' : 'No'}</td>
                            <td style={{ padding: '3px 8px', color: '#94a3b8', fontSize: 10 }}>{p.treatment}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ marginBottom: 12, color: '#64748b', fontSize: 12 }}>{definitions.count} clinical definitions</div>
          {(definitions.terms || []).map((t, i) => (
            <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 12 }}>
              <div style={{ fontWeight: 700, color: '#fbbf24', marginBottom: 8, fontSize: 13 }}>{t.term}</div>
              <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-line' }}>{t.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
