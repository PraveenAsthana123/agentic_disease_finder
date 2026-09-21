'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-primary-hyperparathyroidism-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MEN1':   '#b71c1c',  // deep red       — most common hereditary PHPT
  'CASR':   '#1565c0',  // deep blue      — FHH1 benign, NSHPT emergency
  'CDC73':  '#4a148c',  // deep purple    — HPT-JT, carcinoma 15%
  'GNA11':  '#0277bd',  // dark blue      — FHH2 benign identical to FHH1
  'AP2S1':  '#006064',  // dark cyan      — FHH3 highest Ca²⁺ of FHH
  'RET':    '#e65100',  // deep orange    — MEN2A, pheo exclusion mandatory
  'CDKN1B': '#2e7d32',  // dark green     — MEN4 MEN1-like, p27
  'GCM2':   '#6a1b9a',  // purple         — FIHPT GOF, bidirectional gene
};

const GENE_INFO = {
  'MEN1':   { full: 'MEN1 / Menin / 610aa', locus: '11q13.1', size: '610 aa / 70 kDa (menin — scaffold for H3K4 methyltransferase complex; MEN1 triad: PHPT >90% + pituitary + pancreatic NET; multiglandular; annual Ca²⁺ from age 8)', inh: 'AD LOF' },
  'CASR':   { full: 'CASR / CaSR / 1078aa', locus: '3q13.3', size: '1078 aa / 120 kDa (Ca²⁺-sensing GPCR; heterozygous LOF = FHH1 benign; CCCR <0.01 PATHOGNOMONIC; biallelic LOF = NSHPT neonatal emergency; DO NOT OPERATE FHH)', inh: 'AD LOF' },
  'CDC73':  { full: 'CDC73 / Parafibromin / 531aa', locus: '1q31.2', size: '531 aa / 60 kDa (PAF1 complex subunit; HPT-JT: parathyroid carcinoma 15-20%; ossifying jaw fibroma 50% PATHOGNOMONIC; parafibromin IHC loss = carcinoma marker)', inh: 'AD LOF' },
  'GNA11':  { full: 'GNA11 / Galpha11 / 359aa', locus: '19p13.3', size: '359 aa / 42 kDa (Gα11 Gq-family subunit; FHH2 — identical to FHH1 biochemically; low CCCR; benign; GNA11 GOF = ADH2 hypoparathyroidism — opposite phenotype)', inh: 'AD LOF' },
  'AP2S1':  { full: 'AP2S1 / AP2-sigma1 / 142aa', locus: '19q13.32', size: '142 aa / 17 kDa (AP-2 clathrin adaptor sigma-1 subunit; FHH3 — highest Ca²⁺ of all FHH types; CaSR internalisation defect; cognitive impairment subset; benign)', inh: 'AD LOF' },
  'RET':    { full: 'RET / RET RTK / 1114aa', locus: '10q11.21', size: '1114 aa / 120 kDa (receptor tyrosine kinase; MEN2A GOF C634: MTC+pheo+PHPT; PHEO EXCLUDED BEFORE ANY SURGERY; MEN2B virtually absent PHPT; C634 codon = highest PHPT risk)', inh: 'AD GOF' },
  'CDKN1B': { full: 'CDKN1B / p27Kip1 / 196aa', locus: '12p13.1', size: '196 aa / 22 kDa (CDK2/CDK4 inhibitor p27; MEN4 = MEN1-like without menin; PHPT + pituitary ACTH more common; ~2-3% MEN1-like cases MEN1-negative)', inh: 'AD LOF' },
  'GCM2':   { full: 'GCM2 / GCM2 / 495aa', locus: '6p24.2', size: '495 aa / 47 kDa (master parathyroid TF; GOF → FIHPT multiglandular; same gene as GCM2 LOF = hypoparathyroidism — BIDIRECTIONAL unique endocrine gene)', inh: 'AD GOF' },
};

const SYNDROME_COLORS = {
  'MEN1-Multiglandular':  '#b71c1c',
  'FHH1-Benign':          '#1565c0',
  'HPT-JT-Carcinoma':     '#4a148c',
  'FHH2-Benign':          '#0277bd',
  'FHH3-Benign-HighCa':   '#006064',
  'MEN2A-Mild-PHPT':      '#e65100',
  'MEN4-MEN1-Like':       '#2e7d32',
  'FIHPT-Multiglandular': '#6a1b9a',
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

export default function PrimaryHyperparathyroidismAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [activeGene, setActiveGene] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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

  const genes = Object.keys(GENE_COLORS);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1e293b,#0f172a)', borderBottom: '1px solid #334155', padding: '20px 24px' }}>
        <div style={{ fontSize: 11, color: '#64748b', letterSpacing: 2, marginBottom: 4 }}>HEREDITARY DISEASE ATLAS · ENDOCRINE GENETICS</div>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 800, color: '#f1f5f9' }}>
          🧬 Hereditary Primary Hyperparathyroidism Atlas
        </h1>
        <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 6 }}>
          Complete 8-Gene MEN1 · CASR(FHH1) · CDC73(HPT-JT) · GNA11(FHH2) · AP2S1(FHH3) · RET(MEN2A) · CDKN1B(MEN4) · GCM2(FIHPT) Reference
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {genes.map(g => <GeneChip key={g} gene={g} active={activeGene} onClick={g => setActiveGene(prev => prev === g ? null : g)} />)}
        </div>
      </div>

      {/* Alert banner */}
      <div style={{ background: '#450a0a', borderBottom: '1px solid #7f1d1d', padding: '8px 24px', fontSize: 12, color: '#fca5a5' }}>
        ⚠️ FHH KEY RULE: CASR/GNA11/AP2S1 LOF → CCCR &lt;0.01 → BENIGN FHH → DO NOT OPERATE — surgery does NOT cure FHH and causes permanent hypoparathyroidism · CDC73 carcinoma risk 15% → en-bloc resection
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid #334155', padding: '0 24px', background: '#1e293b' }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              background: 'none', border: 'none', color: tab === t ? '#38bdf8' : '#64748b',
              borderBottom: tab === t ? '2px solid #38bdf8' : '2px solid transparent',
              padding: '10px 16px', cursor: 'pointer', fontSize: 13, fontWeight: tab === t ? 700 : 400,
            }}
          >{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px' }}>
        {loading && <div style={{ color: '#64748b', textAlign: 'center', padding: 40 }}>Loading atlas data…</div>}
        {error && <div style={{ color: '#ef4444', padding: 20, background: '#1e293b', borderRadius: 8 }}>Error: {error}</div>}

        {/* ── OVERVIEW TAB ── */}
        {!loading && tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
              <MetricCard label="Genes" value={overview.n_genes} ok />
              <MetricCard label="Patients" value={overview.n_patients} ok />
              <MetricCard label="Seeds" value={overview.seeds} />
              <MetricCard label="Mean Ca²⁺ (mmol/L)" value={overview.aggregate_metrics?.mean_serum_ca_mmol} warn={overview.aggregate_metrics?.mean_serum_ca_mmol > 2.75} />
              <MetricCard label="Mean PTH (pmol/L)" value={overview.aggregate_metrics?.mean_serum_pth_pmol} warn={overview.aggregate_metrics?.mean_serum_pth_pmol > 9} />
              <MetricCard label="Mean CCCR" value={overview.aggregate_metrics?.mean_cccr} />
              <MetricCard label="FHH (benign) %" value={`${overview.aggregate_metrics?.fhh_benign_pct}%`} ok />
              <MetricCard label="Carcinoma %" value={`${overview.aggregate_metrics?.parathyroid_carcinoma_pct}%`} warn />
              <MetricCard label="Multiglandular %" value={`${overview.aggregate_metrics?.multiglandular_pct}%`} />
              <MetricCard label="Jaw Fibroma %" value={`${overview.aggregate_metrics?.jaw_fibroma_pct}%`} warn />
              <MetricCard label="Nephrolithiasis %" value={`${overview.aggregate_metrics?.nephrolithiasis_pct}%`} />
            </div>

            {/* Syndromes */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, marginBottom: 10, color: '#38bdf8' }}>8 Syndromes — Hereditary PHPT / FHH</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {overview.syndromes?.map((s, i) => (
                  <span key={i} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6, padding: '6px 12px', fontSize: 12 }}>
                    {s}
                  </span>
                ))}
              </div>
            </div>

            {/* Gene Summary Table */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, marginBottom: 12, color: '#38bdf8' }}>Gene Summary (8 × 40 patients)</div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #334155', color: '#64748b' }}>
                      {['Gene','Locus','N','Mean Ca²⁺','Mean PTH','CCCR','Carcinoma%','Multigung%','Jaw%','Nephro%','Syndrome'].map(h => (
                        <th key={h} style={{ textAlign: 'left', padding: '6px 10px', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {overview.gene_summary?.map(g => (
                      <tr key={g.gene} style={{
                        borderBottom: '1px solid #1e293b',
                        background: activeGene && activeGene !== g.gene ? '#0f172a33' : 'transparent',
                        opacity: activeGene && activeGene !== g.gene ? 0.5 : 1,
                        cursor: 'pointer',
                      }} onClick={() => setActiveGene(prev => prev === g.gene ? null : g.gene)}>
                        <td style={{ padding: '6px 10px' }}><GeneChip gene={g.gene} /></td>
                        <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.locus}</td>
                        <td style={{ padding: '6px 10px' }}>{g.n_patients}</td>
                        <td style={{ padding: '6px 10px', color: g.mean_ca > 2.75 ? '#f87171' : '#86efac' }}>{g.mean_ca}</td>
                        <td style={{ padding: '6px 10px', color: g.mean_pth > 9 ? '#fbbf24' : '#86efac' }}>{g.mean_pth}</td>
                        <td style={{ padding: '6px 10px', color: g.mean_cccr < 0.01 ? '#60a5fa' : '#94a3b8' }}>{g.mean_cccr}</td>
                        <td style={{ padding: '6px 10px', color: g.carcinoma_pct > 5 ? '#f87171' : '#94a3b8' }}>{g.carcinoma_pct}%</td>
                        <td style={{ padding: '6px 10px' }}>{g.multigung_pct}%</td>
                        <td style={{ padding: '6px 10px', color: g.jaw_fibroma_pct > 10 ? '#f87171' : '#94a3b8' }}>{g.jaw_fibroma_pct}%</td>
                        <td style={{ padding: '6px 10px' }}>{g.nephro_pct}%</td>
                        <td style={{ padding: '6px 10px' }}><SyndromeTag syndrome={g.syndrome} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Key Clinical Rules */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, marginBottom: 10, color: '#38bdf8' }}>Key Clinical Rules</div>
              {overview.key_clinical_rules?.map((r, i) => (
                <div key={i} style={{ borderLeft: '3px solid #334155', paddingLeft: 12, marginBottom: 10, fontSize: 12, color: '#cbd5e1' }}>
                  {r}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── GENE TABLE TAB ── */}
        {!loading && tab === 'Gene Table' && (
          <div>
            {genes.map(g => {
              const col = GENE_COLORS[g];
              const info = GENE_INFO[g];
              return (
                <div key={g} style={{
                  background: '#1e293b', borderRadius: 8, marginBottom: 16, padding: 16,
                  borderLeft: `4px solid ${col}`,
                  opacity: activeGene && activeGene !== g ? 0.45 : 1,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                    <GeneChip gene={g} />
                    <span style={{ color: '#94a3b8', fontSize: 12 }}>{info?.locus}</span>
                    <span style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 4, padding: '2px 8px', fontSize: 11, color: '#e2e8f0' }}>
                      {info?.inh}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{info?.full}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.6 }}>{info?.size}</div>
                </div>
              );
            })}
          </div>
        )}

        {/* ── CLINICAL ATLAS TAB ── */}
        {!loading && tab === 'Clinical Atlas' && breakdown && (
          <div>
            {breakdown.genes
              ?.filter(g => activeGene === null || g.gene === activeGene)
              .map(g => {
                const col = GENE_COLORS[g.gene] || '#555';
                return (
                  <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, marginBottom: 24, padding: 20, borderLeft: `4px solid ${col}` }}>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 14 }}>
                      <GeneChip gene={g.gene} />
                      <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus}</span>
                      <MetricCard label="Patients" value={g.n_patients} ok />
                      <MetricCard label="Mean Ca²⁺" value={g.mean_ca} warn={g.mean_ca > 2.75} />
                      <MetricCard label="Mean PTH" value={g.mean_pth} warn={g.mean_pth > 9} />
                      <MetricCard label="CCCR" value={g.mean_cccr} ok={g.mean_cccr < 0.01} />
                      <MetricCard label="Carcinoma%" value={`${g.carcinoma_pct}%`} warn={g.carcinoma_pct > 5} />
                      <MetricCard label="Multigung%" value={`${g.multigung_pct}%`} />
                      <MetricCard label="Jaw Fibroma%" value={`${g.jaw_fibroma_pct}%`} warn={g.jaw_fibroma_pct > 10} />
                      <MetricCard label="Nephro%" value={`${g.nephro_pct}%`} />
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 14 }}>
                      <div>
                        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4, fontWeight: 700 }}>PROTEIN / STRUCTURE</div>
                        <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{g.protein_size}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4, fontWeight: 700 }}>DISEASE PATHWAY</div>
                        <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{g.disease_pathway}</div>
                      </div>
                    </div>

                    <div style={{ marginBottom: 14 }}>
                      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4, fontWeight: 700 }}>INHERITANCE / CLINICAL FEATURES</div>
                      <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{g.inheritance}</div>
                    </div>

                    <div style={{ marginBottom: 14 }}>
                      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4, fontWeight: 700 }}>DISEASE CATEGORY / MANAGEMENT</div>
                      <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{g.disease_category}</div>
                    </div>

                    <div>
                      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6, fontWeight: 700 }}>TOP TREATMENTS</div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {g.top_treatments?.map(([tx, n], i) => (
                          <span key={i} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 4, padding: '3px 8px', fontSize: 11 }}>
                            {tx} <span style={{ color: '#64748b' }}>({n})</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {!loading && tab === 'Definitions' && definitions && (
          <div>
            {definitions.definitions?.map((d, i) => (
              <div key={i} style={{ background: '#1e293b', borderRadius: 8, marginBottom: 16, padding: 18, borderLeft: '3px solid #334155' }}>
                <div style={{ fontWeight: 700, color: '#38bdf8', marginBottom: 8, fontSize: 14 }}>{d.term}</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 10 }}>
                  {d.genes?.map(g => <GeneChip key={g} gene={g} />)}
                </div>
                <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{d.definition}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
