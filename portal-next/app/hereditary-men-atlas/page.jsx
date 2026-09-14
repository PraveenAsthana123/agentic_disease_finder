'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-men-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  MEN1:    '#1565c0',  // deep blue       — 3P triad, multiglandular HPT, gastrinoma ZES
  RET:     '#6a1b9a',  // deep purple     — MEN2A/MEN2B, MTC, prophylactic thyroidectomy
  CDKN1B:  '#bf360c',  // burnt sienna    — MEN4, p27/KIP1, MEN1-phenotype MEN1-negative
  AIP:     '#1b5e20',  // dark green      — FIPA, gigantism, SSA-resistant, pegvisomant
  PRKAR1A: '#37474f',  // blue-grey       — Carney complex, PPNAD, paradoxical dexamethasone
  VHL:     '#4a148c',  // dark violet     — hemangioblastoma, RCC, belzutifan HIF-2α
  SDHB:    '#004d40',  // dark teal       — PGL4, extra-adrenal PGL, highest malignancy
  MAX:     '#e65100',  // burnt orange    — PGL5, bilateral adrenal pheo, paternal imprinting
};

const GENE_INFO = {
  MEN1:    { full: 'MEN1 / Menin / 610aa', locus: '11q13.1', size: '610 aa / 67 kDa', inh: 'AD LOF', disease: 'MEN1 syndrome — 3P triad: Parathyroid (>90%, multiglandular) + Pituitary (40%) + Pancreatic NET (40-70%); MULTIGLANDULAR HPT PATHOGNOMONIC; gastrinoma/ZES most common PNET; subtotal parathyroidectomy + thymectomy MANDATORY same operation; annual surveillance from age 5' },
  RET:     { full: 'RET / RET RTK / 1114aa', locus: '10q11.21', size: '1114 aa / 124 kDa', inh: 'AD GOF', disease: 'MEN2A (Cys634): MTC + Pheo + HPT; MEN2B (Met918Thr): MTC + Pheo + mucosal neuromas + Marfanoid; ATA risk D = thyroidectomy within 6 months; CHECK PHEO BEFORE THYROID SURGERY; alpha-blockade mandatory before adrenalectomy' },
  CDKN1B:  { full: 'CDKN1B / p27-KIP1 / 198aa', locus: '12p13.1', size: '198 aa / 22 kDa', inh: 'AD LOF', disease: 'MEN4 — MEN1-like: HPT + pituitary adenoma (ACTH-predominant) + rare PNET; MEN1-phenotype with NEGATIVE MEN1 sequencing = test CDKN1B; lower penetrance than MEN1; same surveillance protocol as MEN1 once positive' },
  AIP:     { full: 'AIP / AIP cochaperone / 330aa', locus: '11q13.2', size: '330 aa / 37 kDa', inh: 'AD LOF', disease: 'FIPA — GH-secreting somatotrophinoma; GIGANTISM in child = AIP FIPA until proven otherwise; young onset (<30y), large macroadenoma; POOR SSA RESPONSE PATHOGNOMONIC; pegvisomant (GH receptor antagonist) preferred; annual MRI surveillance' },
  PRKAR1A: { full: 'PRKAR1A / PKA-R1α / 381aa', locus: '17q24.2', size: '381 aa / 43 kDa', inh: 'AD LOF', disease: 'Carney complex — PPNAD (ACTH-independent Cushing) + cardiac myxoma (any chamber, recurs) + spotty pigmentation (lentigines lips/conjunctiva); PARADOXICAL CORTISOL RISE ON DEXAMETHASONE (Liddle test) PATHOGNOMONIC; annual echocardiography mandatory' },
  VHL:     { full: 'VHL / pVHL / 213aa', locus: '3p25.3', size: '213 aa / 24 kDa', inh: 'AD LOF', disease: 'VHL syndrome — Hemangioblastoma + Clear cell RCC + Pheo + PNET + ELST; RETINAL ANGIOMA first manifestation; belzutifan (HIF-2α inhibitor) FDA approved 2021; type 2C: pheo only; nephron-sparing surgery for RCC <3cm' },
  SDHB:    { full: 'SDHB / SDH iron-sulfur / 280aa', locus: '1p36.13', size: '280 aa / 32 kDa', inh: 'AD LOF', disease: 'PGL4 — Extra-adrenal PGL predominant; HIGHEST MALIGNANCY 30-50%; Carney-Stratakis dyad (GIST+PGL); methoxytyramine elevated (dopaminergic); SDHB IHC loss confirms SDHx; DOTATATE PET/CT for staging; annual surveillance from age 6' },
  MAX:     { full: 'MAX / MYC-assoc factor X / 160aa', locus: '14q23.3', size: '160 aa / 18 kDa', inh: 'AD LOF (paternal imprinting)', disease: 'PGL5 — Bilateral adrenal pheo; PATERNAL IMPRINTING: maternal carriers NOT at risk; adrenaline-secreting (metanephrine elevated); cortical-sparing adrenalectomy preferred; MYC-MAX pathway; lower malignancy than SDHB' },
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

export default function HeredMENAtlasPage() {
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
        <div style={{ color: '#60a5fa', fontSize: 13, marginBottom: 4 }}>🧬 Hereditary Multiple Endocrine Neoplasia Atlas</div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: '#f8fafc' }}>
          Hereditary-Multiple-Endocrine-Neoplasia-Atlas
        </h1>
        <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 6 }}>
          Complete 8-Gene Reference — MEN1 · RET · CDKN1B · AIP · PRKAR1A · VHL · SDHB · MAX
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
            <MetricCard label="Genes Covered" value={overview.genes_covered?.length} sub="MEN + VHL + SDHx pathways" />
            <MetricCard label="Surgery Rate" value={`${overview.aggregate_metrics?.surgery_pct}%`} sub="thyroidectomy / adrenalectomy / PTX" />
            <MetricCard label="Pheo Present" value={`${overview.aggregate_metrics?.pheo_pct}%`} sub="adrenal or extra-adrenal" />
            <MetricCard label="Malignant Disease" value={`${overview.aggregate_metrics?.malignant_pct}%`} sub="MTC / metastatic PGL / RCC" />
            <MetricCard label="Bilateral Disease" value={`${overview.aggregate_metrics?.bilateral_pct}%`} sub="bilateral adrenal / multiglandular" />
            <MetricCard label="Under Surveillance" value={`${overview.aggregate_metrics?.surveillance_pct}%`} sub="active biochemical surveillance" />
          </div>

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
                </div>
                <div style={{ color: '#f1f5f9', fontSize: 14, fontWeight: 700, marginBottom: 4 }}>{g.disease_category}</div>
                <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8 }}>{g.pathognomonic?.slice(0, 200)}…</div>
                <div style={{ color: '#64748b', fontSize: 11, marginBottom: 6 }}>
                  <b style={{ color: '#60a5fa' }}>Hormone:</b> {g.hormone_profile?.slice(0, 120)}
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#60a5fa' }}>n={g.n_patients}</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#f87171' }}>Surgery {g.surgery_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fbbf24' }}>Pheo {g.pheo_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fb923c' }}>Malignant {g.malignant_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#a78bfa' }}>Bilateral {g.bilateral_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#34d399' }}>Surveillance {g.surveillance_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#94a3b8' }}>Avg dx {g.avg_age_dx_years}y</span>
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
                {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease', 'Hormone Profile', 'Malignant%', 'Pheo%', 'Surgery%', 'Bilateral%', 'Surveillance%', 'Avg Age Dx'].map(h => (
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
                  <td style={{ padding: '10px 12px', color: '#e2e8f0', borderBottom: '1px solid #1e293b', whiteSpace: 'nowrap' }}>{g.inheritance?.split(';')[0]?.slice(0, 40)}</td>
                  <td style={{ padding: '10px 12px', color: '#f1f5f9', borderBottom: '1px solid #1e293b', maxWidth: 200 }}>{g.disease_category?.slice(0, 80)}…</td>
                  <td style={{ padding: '10px 12px', color: '#a78bfa', borderBottom: '1px solid #1e293b', maxWidth: 160 }}>{g.hormone_profile?.slice(0, 60)}…</td>
                  <td style={{ padding: '10px 12px', color: '#fb923c', borderBottom: '1px solid #1e293b' }}>{g.malignant_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#fbbf24', borderBottom: '1px solid #1e293b' }}>{g.pheo_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#f87171', borderBottom: '1px solid #1e293b' }}>{g.surgery_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#a78bfa', borderBottom: '1px solid #1e293b' }}>{g.bilateral_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#34d399', borderBottom: '1px solid #1e293b' }}>{g.surveillance_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>{g.avg_age_dx_years}y</td>
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

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#fbbf24', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>PATHOGNOMONIC</div>
                <div style={{ color: '#e2e8f0', fontSize: 12, lineHeight: 1.6 }}>{g.pathognomonic?.slice(0, 400)}…</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#34d399', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>TREATMENT</div>
                <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>{g.treatment?.slice(0, 300)}…</div>
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
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#f87171' }}>Surgery {g.surgery_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fbbf24' }}>Pheo {g.pheo_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fb923c' }}>Malignant {g.malignant_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#a78bfa' }}>Bilateral {g.bilateral_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#34d399' }}>Surveillance {g.surveillance_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#94a3b8' }}>Onset: {g.onset_age?.slice(0, 50)}</span>
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
              <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 6 }}><b style={{ color: '#60a5fa' }}>Pathway:</b> {entry.disease_pathway?.slice(0, 250)}…</div>
              <div style={{ color: '#e2e8f0', fontSize: 12, marginBottom: 6 }}><b style={{ color: '#fbbf24' }}>Pathognomonic:</b> {entry.pathognomonic?.slice(0, 250)}…</div>
              <div style={{ color: '#94a3b8', fontSize: 12 }}><b style={{ color: '#34d399' }}>Hormone profile:</b> {entry.hormone_profile}</div>
            </div>
          ))}

          {/* MEN Glossary */}
          <h3 style={{ color: '#60a5fa', marginTop: 8 }}>MEN Genetics Glossary</h3>
          {Object.entries(definitions.men_glossary || {}).map(([term, text]) => (
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
