'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-monogenic-hypertension-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'SCNN1B':  '#1565c0',  // deep blue       — Liddle ENaC-β, PY-motif, amiloride
  'SCNN1G':  '#1976d2',  // medium blue     — Liddle ENaC-γ, same mechanism, test both
  'HSD11B2': '#6a1b9a',  // deep purple     — AME, cortisol-MR agonist, dexamethasone
  'NR3C2':   '#880e4f',  // dark rose       — Geller MR-GOF, pregnancy crisis, avoid progestins
  'WNK4':    '#2e7d32',  // dark green      — Gordon PHAII-B, NCC disinhibition, thiazide response
  'WNK1':    '#1b5e20',  // darker green    — Gordon PHAII-A, intronic deletion, exome misses
  'KLHL3':   '#e65100',  // burnt orange    — Gordon PHAII-D, CRL3 E3 ligase, AR more severe
  'CUL3':    '#bf360c',  // burnt sienna    — Gordon PHAII-E, de novo exon 9 skip, most severe
};

const GENE_INFO = {
  'SCNN1B':  { full: 'SCNN1B / ENaC-β / 669aa', locus: '16p12.2', size: '669 aa / 79 kDa', inh: 'AD', disease: 'Liddle syndrome — PY-motif truncation/missense → Nedd4-2 cannot bind → ENaC accumulates → constitutive Na+ reabsorption; LOW RENIN + LOW ALDOSTERONE + HYPOKALEMIA; AMILORIDE/TRIAMTERENE curative; SPIRONOLACTONE INEFFECTIVE (MR-independent)' },
  'SCNN1G':  { full: 'SCNN1G / ENaC-γ / 649aa', locus: '16p12.2', size: '649 aa / 76 kDa', inh: 'AD', disease: 'Liddle syndrome ENaC-γ — identical mechanism to SCNN1B; SCNN1B and SCNN1G adjacent at 16p12.2 — test BOTH subunits; negative SCNN1B does NOT exclude Liddle; amiloride curative; spironolactone ineffective' },
  'HSD11B2': { full: 'HSD11B2 / 11β-HSD2 / 405aa', locus: '16q22.1', size: '405 aa / 44 kDa', inh: 'AR', disease: 'Apparent Mineralocorticoid Excess (AME) — cortisol-MR activation; URINARY CORTISOL/CORTISONE RATIO >100 PATHOGNOMONIC; dexamethasone suppresses cortisol + spironolactone; liquorice inhibits 11β-HSD2 (acquired AME); AR biallelic' },
  'NR3C2':   { full: 'NR3C2 / MR / 984aa', locus: '4q31.23', size: '984 aa / 107 kDa', inh: 'AD GOF', disease: 'Geller syndrome MR-GOF S810L — DRAMATIC PREGNANCY EXACERBATION PATHOGNOMONIC (progesterone = full MR agonist at S810L); AVOID ALL PROGESTINS; SPIRONOLACTONE WORSENS (MR agonist at S810L); prompt delivery resolves crisis' },
  'WNK4':    { full: 'WNK4 / WNK kinase 4 / 1243aa', locus: '17q21.2', size: '1243 aa / 135 kDa', inh: 'AD', disease: 'Gordon PHAII-B — WNK4 LOF releases NCC → NCC hyperactivation → HYPERKALEMIA + HTN + NORMAL GFR; THIAZIDE DIAGNOSTIC RESPONSE (BP + K normalisation within days PATHOGNOMONIC confirms NCC mechanism)' },
  'WNK1':    { full: 'WNK1 / WNK kinase 1 / 2382aa', locus: '12p13.33', size: '2382 aa / 256 kDa', inh: 'AD', disease: 'Gordon PHAII-A — INTRONIC LARGE DELETION (~41 kb intron 1) — STANDARD EXOME MISSES — WNK1 MLPA/CNV MANDATORY if exome negative; identical Gordon phenotype; thiazide curative' },
  'KLHL3':   { full: 'KLHL3 / Kelch-Like 3 / 587aa', locus: '5q31.2', size: '587 aa / 67 kDa', inh: 'AR/AD', disease: 'Gordon PHAII-D — CRL3^KLHL3 E3 ubiquitin ligase; WNK4/WNK1 not ubiquitinated → accumulate → NCC hyperactivation; AR biallelic = more severe; AD dominant-negative = milder; R528H hotspot; thiazide curative' },
  'CUL3':    { full: 'CUL3 / Cullin-3 / 768aa', locus: '2q36.2', size: '768 aa / 89 kDa', inh: 'AD de novo', disease: 'Gordon PHAII-E — de novo EXON 9 SKIP (dominant-negative CUL3-Δ403-459) → KLHL3 docking impaired → WNK4 accumulates; MOST SEVERE Gordon subtype; NEONATAL presentation; HIGH DE NOVO RATE — no family history does NOT exclude' },
};

function GeneChip({ gene }) {
  return (
    <span style={{
      background: GENE_COLORS[gene] || '#555',
      color: '#fff',
      borderRadius: 4,
      padding: '2px 8px',
      fontSize: 12,
      fontWeight: 700,
      marginRight: 4,
      display: 'inline-block',
    }}>{gene}</span>
  );
}

function MetricCard({ label, value, sub }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', minWidth: 140, flex: '1 1 140px' }}>
      <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#f1f5f9', fontSize: 22, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HeredMonogenicHypertensionAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' || tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    setLoading(true);
    setError(null);
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => {
        if (ep === 'overview') setOverview(d);
        else if (ep === 'breakdown') setBreakdown(d);
        else setDefinitions(d);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  const data = tab === 'Definitions' ? definitions
    : tab === 'Gene Table' || tab === 'Clinical Atlas' ? breakdown
    : overview;

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', fontFamily: 'system-ui,sans-serif', padding: '24px 32px' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ color: '#60a5fa', fontSize: 13, marginBottom: 4 }}>🧬 Hereditary Monogenic Hypertension Atlas</div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: '#f8fafc' }}>
          Hereditary-Monogenic-Hypertension-Atlas
        </h1>
        <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 6 }}>
          Complete 8-Gene Reference — SCNN1B · SCNN1G · HSD11B2 · NR3C2 · WNK4 · WNK1 · KLHL3 · CUL3
        </div>
        <div style={{ color: '#64748b', fontSize: 12, marginTop: 4 }}>
          Liddle (SCNN1B/SCNN1G) · AME (HSD11B2) · Geller MR-GOF (NR3C2) · Gordon PHAII (WNK4/WNK1/KLHL3/CUL3)
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#1d4ed8' : '#1e293b',
            color: tab === t ? '#fff' : '#94a3b8',
            border: 'none', borderRadius: 6, padding: '8px 18px',
            cursor: 'pointer', fontWeight: tab === t ? 700 : 400, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#60a5fa' }}>Loading…</div>}
      {error && <div style={{ color: '#f87171' }}>Error: {error}</div>}

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 28 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40 cohort" />
            <MetricCard label="Genes Covered" value={overview.genes_covered?.length} sub="Monogenic HTN pathways" />
            <MetricCard label="Low Renin" value={`${overview.aggregate_metrics?.low_renin_pct}%`} sub="plasma renin suppressed" />
            <MetricCard label="Low Aldosterone" value={`${overview.aggregate_metrics?.low_aldo_pct}%`} sub="plasma aldosterone suppressed" />
            <MetricCard label="On Amiloride" value={`${overview.aggregate_metrics?.amiloride_triamterene_pct}%`} sub="Liddle ENaC blockade" />
            <MetricCard label="On Thiazide" value={`${overview.aggregate_metrics?.thiazide_pct}%`} sub="Gordon NCC blockade" />
            <MetricCard label="On Dexamethasone" value={`${overview.aggregate_metrics?.dexamethasone_pct}%`} sub="AME cortisol suppression" />
            <MetricCard label="Genetic Diagnosis" value={`${overview.aggregate_metrics?.genetic_diagnosis_pct}%`} sub="confirmed genetically" />
            <MetricCard label="De Novo" value={`${overview.aggregate_metrics?.de_novo_pct}%`} sub="de novo variants" />
          </div>

          {/* Clinical pearls */}
          {overview.clinical_pearls && (
            <div style={{ background: '#1e293b', borderRadius: 10, padding: 16, marginBottom: 24, borderLeft: '4px solid #f59e0b' }}>
              <div style={{ color: '#f59e0b', fontSize: 13, fontWeight: 700, marginBottom: 10 }}>⚡ CLINICAL PEARLS</div>
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {overview.clinical_pearls.map((p, i) => (
                  <li key={i} style={{ color: '#e2e8f0', fontSize: 12, lineHeight: 1.7 }}>{p}</li>
                ))}
              </ul>
            </div>
          )}

          <h2 style={{ color: '#60a5fa', fontSize: 16, marginBottom: 14 }}>Gene Summary</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {overview.gene_summary && Object.values(overview.gene_summary).map(g => (
              <div key={g.gene} style={{
                background: '#1e293b', borderRadius: 10, padding: 16,
                borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
                  <span style={{ color: '#60a5fa', fontSize: 11, marginLeft: 4 }}>{g.hypertension_type}</span>
                </div>
                <div style={{ color: '#f1f5f9', fontSize: 14, fontWeight: 700, marginBottom: 4 }}>{g.disease_category}</div>
                <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8 }}>{g.pathognomonic?.slice(0, 200)}…</div>
                <div style={{ color: '#64748b', fontSize: 11, marginBottom: 6 }}>
                  <b style={{ color: '#60a5fa' }}>Hormone:</b> {g.hormone_profile?.slice(0, 120)}
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#60a5fa' }}>n={g.n_patients}</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#f87171' }}>Avg SBP {g.avg_sbp} mmHg</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fbbf24' }}>K+ {g.avg_serum_k} mmol/L</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#34d399' }}>Amiloride {g.amiloride_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#a78bfa' }}>Thiazide {g.thiazide_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fb923c' }}>Surveillance {g.surveillance_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#94a3b8' }}>Onset {g.avg_age_onset}y</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
                {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease/Syndrome', 'Hormone Profile', 'Avg SBP', 'K+ mmol/L', 'Amiloride%', 'Thiazide%', 'Low Renin%', 'Onset Age'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', borderBottom: '1px solid #334155', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {breakdown.gene_breakdowns?.map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                  <td style={{ padding: '10px 12px', borderBottom: '1px solid #1e293b' }}><GeneChip gene={g.gene} /></td>
                  <td style={{ padding: '10px 12px', color: '#60a5fa', borderBottom: '1px solid #1e293b', whiteSpace: 'nowrap' }}>{g.locus}</td>
                  <td style={{ padding: '10px 12px', color: '#94a3b8', borderBottom: '1px solid #1e293b', whiteSpace: 'nowrap' }}>{g.protein_size}</td>
                  <td style={{ padding: '10px 12px', color: '#e2e8f0', borderBottom: '1px solid #1e293b', whiteSpace: 'nowrap' }}>{g.inheritance?.split(';')[0]?.slice(0, 30)}</td>
                  <td style={{ padding: '10px 12px', color: '#f1f5f9', borderBottom: '1px solid #1e293b', maxWidth: 180 }}>{g.disease_category?.slice(0, 70)}…</td>
                  <td style={{ padding: '10px 12px', color: '#a78bfa', borderBottom: '1px solid #1e293b', maxWidth: 140 }}>{g.hormone_profile?.slice(0, 55)}…</td>
                  <td style={{ padding: '10px 12px', color: '#f87171', borderBottom: '1px solid #1e293b' }}>{g.avg_sbp}</td>
                  <td style={{ padding: '10px 12px', color: '#fbbf24', borderBottom: '1px solid #1e293b' }}>{g.avg_serum_k}</td>
                  <td style={{ padding: '10px 12px', color: '#34d399', borderBottom: '1px solid #1e293b' }}>{g.amiloride_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#818cf8', borderBottom: '1px solid #1e293b' }}>{g.thiazide_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#fb923c', borderBottom: '1px solid #1e293b' }}>{g.low_renin_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>{g.avg_age_onset}y</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {breakdown.gene_breakdowns?.map(g => (
            <div key={g.gene} style={{
              background: '#1e293b', borderRadius: 10, padding: 18,
              borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}`,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                <GeneChip gene={g.gene} />
                <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus} · {g.protein_size} · {g.inheritance?.split(';')[0]}</span>
              </div>
              <div style={{ color: '#f1f5f9', fontSize: 15, fontWeight: 700, marginBottom: 6 }}>{g.disease_category}</div>
              <div style={{ color: '#64748b', fontSize: 12, marginBottom: 6 }}>
                <b style={{ color: '#60a5fa' }}>Hormone Profile:</b> {g.hormone_profile}
              </div>
              <div style={{ color: '#64748b', fontSize: 12, marginBottom: 6 }}>
                <b style={{ color: '#818cf8' }}>HTN Type:</b> {g.hypertension_type}
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#fbbf24', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>PATHOGNOMONIC</div>
                <div style={{ color: '#e2e8f0', fontSize: 12, lineHeight: 1.6 }}>{g.pathognomonic?.slice(0, 450)}…</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#34d399', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>TREATMENT</div>
                <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>{g.treatment?.slice(0, 350)}…</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#60a5fa', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>KEY FEATURES</div>
                <ul style={{ margin: 0, paddingLeft: 18, color: '#94a3b8', fontSize: 12 }}>
                  {g.key_features?.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </div>

              <div>
                <div style={{ color: '#f87171', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>KEY DDx</div>
                <ul style={{ margin: 0, paddingLeft: 18, color: '#94a3b8', fontSize: 12 }}>
                  {g.key_ddx?.map((d, i) => <li key={i}>{d}</li>)}
                </ul>
              </div>

              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#60a5fa' }}>n={g.n_patients}</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#f87171' }}>Avg SBP {g.avg_sbp} mmHg</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fbbf24' }}>K+ {g.avg_serum_k} mmol/L</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#34d399' }}>Amiloride {g.amiloride_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#818cf8' }}>Thiazide {g.thiazide_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fb923c' }}>Low Renin {g.low_renin_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#94a3b8' }}>Onset {g.onset_age?.slice(0, 50)}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Gene entries */}
          {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => (
            <div key={gene} style={{ background: '#1e293b', borderRadius: 10, padding: 18, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <GeneChip gene={gene} />
                <span style={{ color: '#94a3b8', fontSize: 12 }}>{entry.locus} · {entry.protein_size} · {entry.inheritance}</span>
              </div>
              <div style={{ color: '#f1f5f9', fontSize: 14, fontWeight: 700, marginBottom: 6 }}>{entry.disease_name}</div>
              <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 6 }}><b style={{ color: '#60a5fa' }}>Pathway:</b> {entry.disease_pathway?.slice(0, 280)}…</div>
              <div style={{ color: '#e2e8f0', fontSize: 12, marginBottom: 6 }}><b style={{ color: '#fbbf24' }}>Pathognomonic:</b> {entry.pathognomonic?.slice(0, 280)}…</div>
              <div style={{ color: '#94a3b8', fontSize: 12 }}><b style={{ color: '#34d399' }}>Hormone profile:</b> {entry.hormone_profile}</div>
            </div>
          ))}

          {/* Hypertension Glossary */}
          <h3 style={{ color: '#60a5fa', marginTop: 8 }}>Monogenic Hypertension Genetics Glossary</h3>
          {Object.entries(definitions.hypertension_glossary || {}).map(([term, text]) => (
            <div key={term} style={{ background: '#0f172a', borderRadius: 8, padding: 16, borderLeft: '3px solid #3b82f6' }}>
              <div style={{ color: '#60a5fa', fontSize: 13, fontWeight: 700, marginBottom: 6 }}>{term}</div>
              <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>{text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
