'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cah-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'CYP21A2': '#b71c1c',  // deep red       — most common, classic CAH
  'CYP11B1': '#e65100',  // deep orange    — hypertensive virilising CAH
  'HSD3B2':  '#f9a825',  // amber          — all-zone, paradoxical DSD
  'CYP17A1': '#1565c0',  // deep blue      — HTN + absent puberty + 46XY reversal
  'STAR':    '#4a148c',  // deep purple    — lipoid CAH, most severe
  'CYP11A1': '#6a1b9a',  // purple         — P450scc, variable severity
  'POR':     '#2e7d32',  // dark green     — Antley-Bixler, maternal virilisation
  'CYP11B2': '#00695c',  // dark teal      — isolated aldosterone deficiency
};

const GENE_INFO = {
  'CYP21A2': { full: 'CYP21A2 / 21-Hydroxylase / 495aa', locus: '6p21.33', size: '495 aa / 55 kDa (microsomal 21-hydroxylase; MHC-III region; most common CAH ~95%; 17-OHP PATHOGNOMONIC; gene conversion from CYP21A1P pseudogene; MLPA mandatory)', inh: 'AR' },
  'CYP11B1': { full: 'CYP11B1 / 11β-Hydroxylase / 503aa', locus: '8q24.3', size: '503 aa / 56 kDa (mitochondrial 11β-hydroxylase; compound S + DOC accumulate; HYPERTENSION; NO fludrocortisone; 2nd most common ~5%; R448H North African/Sephardic founder)', inh: 'AR' },
  'HSD3B2':  { full: 'HSD3B2 / 3β-HSD type 2 / 372aa', locus: '1p12', size: '372 aa / 42 kDa (3β-hydroxysteroid dehydrogenase; all-zone defect; DHEA accumulates; paradox: 46XX mild virilisation, 46XY undervirilisation; delta-5:delta-4 ratio elevated)', inh: 'AR' },
  'CYP17A1': { full: 'CYP17A1 / 17α-Hydroxylase/17,20-lyase / 508aa', locus: '10q24.32', size: '508 aa / 57 kDa (ER bifunctional; cortisol + sex steroids absent; DOC/corticosterone excess → HTN; 46XY female phenotype + absent puberty; aldosterone suppressed)', inh: 'AR' },
  'STAR':    { full: 'STAR / StAR protein / 285aa', locus: '8p11.23', size: '285 aa / 32 kDa (cholesterol OMM→IMM transport; ALL steroids absent; CT lipid-laden enlarged adrenals PATHOGNOMONIC; 46XY female phenotype; neonatal crisis day 1-4)', inh: 'AR' },
  'CYP11A1': { full: 'CYP11A1 / P450scc / 521aa', locus: '15q24.1', size: '521 aa / 60 kDa (cholesterol → pregnenolone; similar to StAR but adrenals less lipid-laden; complete LOF = severe; partial LOF = late-onset adrenal insufficiency)', inh: 'AR' },
  'POR':     { full: 'POR / P450 Oxidoreductase / 680aa', locus: '7q11.23', size: '680 aa / 77 kDa (electron donor ALL microsomal CYPs; combined CYP21A2+CYP17A1+CYP19A1 block; Antley-Bixler craniosynostosis; MATERNAL VIRILISATION PATHOGNOMONIC)', inh: 'AR' },
  'CYP11B2': { full: 'CYP11B2 / Aldosterone Synthase / 503aa', locus: '8q24.3', size: '503 aa / 56 kDa (zona glomerulosa; CMO I/II; isolated aldosterone deficiency; normal cortisol + 17-OHP; pure salt-wasting; fludrocortisone ONLY; 18-OHB elevated CMO II)', inh: 'AR' },
};

const SYNDROME_COLORS = {
  'Classic-SW/SV/NC':   '#b71c1c',
  'Hypertensive-CAH':   '#e65100',
  'All-Zone-CAH':       '#f9a825',
  'Lipoid-CAH':         '#4a148c',
  'Combined-Block':     '#2e7d32',
  'Isolated-MC-Def':    '#00695c',
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

export default function CAHAtlasPage() {
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

  if (loading) return <div style={{ color: '#94a3b8', padding: 40, textAlign: 'center' }}>Loading Hereditary-CAH-Atlas…</div>;
  if (error)   return <div style={{ color: '#ef4444', padding: 40 }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'sans-serif', padding: '24px 20px' }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', marginBottom: 6 }}>
          🧬 Hereditary-CAH-Atlas
        </h1>
        <p style={{ fontSize: 13, color: '#94a3b8', maxWidth: 900 }}>
          Complete 8-gene steroidogenesis reference: <b>Classic CAH</b> (CYP21A2) · <b>Hypertensive CAH</b> (CYP11B1, CYP17A1) ·
          <b> All-zone CAH</b> (HSD3B2) · <b>Lipoid CAH</b> (STAR, CYP11A1) · <b>Combined block</b> (POR) · <b>Isolated aldosterone deficiency</b> (CYP11B2) —
          hereditary congenital adrenal hyperplasia syndromes with distinct steroid profiles, presentations, and treatments.
          320-patient aggregate cohort · seeds 2942–2949.
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
            <MetricCard label="Total Patients" value={overview.n_patients} sub="8 × 40, seeds 2942–2949" />
            <MetricCard label="Mean SBP" value={`${overview.aggregate_metrics?.mean_sbp_mmhg} mmHg`} />
            <MetricCard label="Mean K+" value={`${overview.aggregate_metrics?.mean_serum_k_mmol} mmol/L`} />
            <MetricCard label="Mean Na+" value={`${overview.aggregate_metrics?.mean_serum_na_mmol} mmol/L`} warn={overview.aggregate_metrics?.mean_serum_na_mmol < 130} />
            <MetricCard label="Salt-Wasting Crisis" value={`${overview.aggregate_metrics?.salt_wasting_crisis_pct}%`} warn={true} sub="neonatal/infant" />
            <MetricCard label="Hypertensive CAH" value={`${overview.aggregate_metrics?.hypertensive_cah_pct}%`} sub="CYP11B1/CYP17A1" />
            <MetricCard label="Virilisation 46XX" value={`${overview.aggregate_metrics?.virilisation_46xx_pct}%`} sub="ambiguous genitalia" />
            <MetricCard label="46XY Sex Reversal" value={`${overview.aggregate_metrics?.sex_reversal_46xy_pct}%`} sub="STAR/CYP17A1/POR" />
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
                  {['Gene', 'Syndrome', 'Locus', 'Mean SBP', 'Mean K+', 'Mean Na+', 'SW Crisis%', 'HTN%', 'Viril 46XX%', '17-OHP>30%'].map(h => (
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
                    <td style={{ padding: '7px 10px', color: g.mean_sbp > 140 ? '#ef4444' : '#e2e8f0', fontWeight: 700 }}>{g.mean_sbp}</td>
                    <td style={{ padding: '7px 10px', color: g.mean_k > 5.5 ? '#ef4444' : g.mean_k < 3.5 ? '#f59e0b' : '#22c55e', fontWeight: 700 }}>{g.mean_k}</td>
                    <td style={{ padding: '7px 10px', color: g.mean_na < 130 ? '#ef4444' : '#94a3b8' }}>{g.mean_na}</td>
                    <td style={{ padding: '7px 10px', color: g.sw_crisis_pct > 50 ? '#ef4444' : '#94a3b8' }}>{g.sw_crisis_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.htn_pct > 40 ? '#f59e0b' : '#475569' }}>{g.htn_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.viril_46xx_pct > 30 ? '#fbbf24' : '#475569' }}>{g.viril_46xx_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.hi_17ohp_pct > 50 ? '#ef4444' : '#475569' }}>{g.hi_17ohp_pct}%</td>
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
                  <span style={{ fontSize: 12, color: '#94a3b8' }}>
                    n={g.n_patients} · SBP={g.mean_sbp} mmHg · K+={g.mean_k} · Na+={g.mean_na} · 17-OHP={g.mean_17ohp} nmol/L
                  </span>
                  {g.sw_crisis_pct > 50 && <span style={{ background: '#7f1d1d', color: '#fca5a5', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>Salt-wasting crisis</span>}
                  {g.htn_pct > 40 && <span style={{ background: '#78350f', color: '#fcd34d', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>Hypertensive CAH</span>}
                  {g.sex_rev_pct > 40 && <span style={{ background: '#312e81', color: '#a5b4fc', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>46XY sex reversal</span>}
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
                          {['ID', 'Sex', 'Onset(d)', 'SBP', 'K+', 'Na+', '17-OHP', 'SW Crisis', 'HTN', 'Sex Rev', 'Treatment'].map(h => (
                            <th key={h} style={{ padding: '4px 8px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #1e293b' }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {g.patients.map((p, i) => (
                          <tr key={p.id} style={{ background: i % 2 === 0 ? '#0f172a' : '#131f35' }}>
                            <td style={{ padding: '3px 8px', fontFamily: 'monospace', color: '#94a3b8' }}>{p.id}</td>
                            <td style={{ padding: '3px 8px' }}>{p.gender}</td>
                            <td style={{ padding: '3px 8px' }}>{p.onset_days}</td>
                            <td style={{ padding: '3px 8px', color: p.sbp_mmhg > 140 ? '#ef4444' : '#e2e8f0', fontWeight: 700 }}>{p.sbp_mmhg}</td>
                            <td style={{ padding: '3px 8px', color: p.serum_k_mmol > 5.5 ? '#ef4444' : p.serum_k_mmol < 3.5 ? '#f59e0b' : '#e2e8f0', fontWeight: 700 }}>{p.serum_k_mmol}</td>
                            <td style={{ padding: '3px 8px', color: p.serum_na_mmol < 130 ? '#ef4444' : '#94a3b8' }}>{p.serum_na_mmol}</td>
                            <td style={{ padding: '3px 8px', color: p.serum_17ohp_nmol > 30 ? '#fbbf24' : '#94a3b8' }}>{p.serum_17ohp_nmol}</td>
                            <td style={{ padding: '3px 8px', color: p.salt_wasting_crisis ? '#ef4444' : '#475569' }}>{p.salt_wasting_crisis ? 'Yes' : 'No'}</td>
                            <td style={{ padding: '3px 8px', color: p.hypertension ? '#f59e0b' : '#475569' }}>{p.hypertension ? 'Yes' : 'No'}</td>
                            <td style={{ padding: '3px 8px', color: p.sex_reversal_46xy ? '#a78bfa' : '#475569' }}>{p.sex_reversal_46xy ? 'Yes' : 'No'}</td>
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
