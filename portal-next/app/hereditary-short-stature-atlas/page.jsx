'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-short-stature-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  SHOX:   '#1565c0',  // deep blue       — PAR1 pseudoautosomal, LWD Madelung, rGH FDA-approved
  GH1:    '#6a1b9a',  // deep purple     — isolated GHD types IA/IB/II, pituitary somatotroph
  GHR:    '#bf360c',  // burnt sienna    — Laron syndrome, high GH/low IGF-1, mecasermin
  IGF1:   '#1b5e20',  // dark green      — IUGR+SNHL+microcephaly triad, mecasermin
  IGFALS: '#37474f',  // blue-grey       — ALS deficiency, mild, disproportionate IGFBP-3 low
  STAT5B: '#4a148c',  // dark violet     — GH insensitivity + immune dysregulation, varicella
  CUL7:   '#004d40',  // dark teal       — 3-M syndrome, severe proportionate, normal GH axis
  NPR2:   '#e65100',  // burnt orange    — BDE2 short metacarpals AD / ADM acromesomelic AR
};

const GENE_INFO = {
  SHOX:   { full: 'SHOX / Short Stature Homeobox / 758aa', locus: 'Xp22.33/Yp11.32 (PAR1)', size: '758 aa / 29 kDa', inh: 'AD/AR', disease: 'Léri-Weill dyschondrosteosis (LWD, AD) — Madelung deformity PATHOGNOMONIC; Langer mesomelic dysplasia (LMD, AR) — biallelic null severe dwarfism; PAR1 escapes X-inactivation; enhancer deletions exome misses; rGH FDA-approved for SHOX deficiency' },
  GH1:    { full: 'GH1 / Growth Hormone 1 / 22kDa', locus: '17q23.3', size: '217 aa prepro / 22 kDa', inh: 'AR/AD', disease: 'Isolated GH deficiency (IGHD) — Type IA (AR null, anti-GH antibodies on therapy); Type IB (AR, milder); Type II (AD dominant-negative, progressive hypopituitarism); GH stimulation <10 mcg/L on 2 tests; rGH curative for IB/II' },
  GHR:    { full: 'GHR / Growth Hormone Receptor / 638aa', locus: '5p13.1', size: '638 aa / 70 kDa', inh: 'AR', disease: 'Laron syndrome — GH insensitivity; HIGH GH + LOW IGF-1 + LOW IGFBP-3 PATHOGNOMONIC; somatomedin generation test fails; GHBP low (extracellular domain absent); Ecuadorian founder; mecasermin (recombinant IGF-1) — NOT rhGH' },
  IGF1:   { full: 'IGF1 / Insulin-Like Growth Factor 1 / 70aa mature', locus: '12q23.2', size: '195 aa prepro / 7.6 kDa', inh: 'AR', disease: 'IGF-1 deficiency — IUGR + SNHL + microcephaly TRIAD PATHOGNOMONIC; HIGH GH + LOW IGF-1 (similar to Laron but triad unique); prenatal growth failure (GH-independent fetal IGF-1); mecasermin + cochlear implants; very rare' },
  IGFALS: { full: 'IGFALS / Acid Labile Subunit / 605aa', locus: '16p13.3', size: '605 aa / 67 kDa', inh: 'AR', disease: 'ALS deficiency — VERY LOW IGFBP-3 disproportionate to IGF-1 PATHOGNOMONIC; mild short stature (−1 to −3 SDS); pubertal growth spurt PRESERVED; no SNHL/microcephaly; often asymptomatic; tissue paracrine IGF-1 intact explains mild phenotype' },
  STAT5B: { full: 'STAT5B / Signal Transducer 5B / 786aa', locus: '17q21.2', size: '786 aa / 90 kDa', inh: 'AR', disease: 'STAT5B deficiency — GH insensitivity + immune dysregulation; HIGH GH + LOW IGF-1 + SEVERE VARICELLA + T-CELL LYMPHOPENIA PATHOGNOMONIC; GHBP NORMAL (distinguishes from GHR Laron where GHBP low); inflammatory lung disease; HSCT corrects immune defect but not growth' },
  CUL7:   { full: 'CUL7 / Cullin-7 / 1698aa', locus: '6p21.1', size: '1698 aa / 192 kDa', inh: 'AR', disease: '3-M syndrome — PROPORTIONATE severe dwarfism (−8 to −10 SDS); TRIANGULAR FACE + PROMINENT HEELS + SLENDER TUBULAR BONES PATHOGNOMONIC; NORMAL GH/IGF-1 axis; NORMAL intelligence; cellular growth defect, not endocrine; CUL7/OBSL1/CCDC8 heterogeneity' },
  NPR2:   { full: 'NPR2 / Natriuretic Peptide Receptor 2 / 1047aa', locus: '9p13.3', size: '1047 aa / 117 kDa', inh: 'AD/AR', disease: 'AD NPR2 LOF → brachydactyly type E2 (BDE2) — SHORT 4TH METACARPALS PATHOGNOMONIC + mild short stature; AR NPR2 biallelic null → acromesomelic dysplasia Maroteaux (ADM) — SEVERE mesomelic/acromesomelic shortening; vosoritide NOT effective in NPR2 LOF (receptor absent)' },
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

export default function ShortStatureAtlasPage() {
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
        <div style={{ color: '#60a5fa', fontSize: 13, marginBottom: 4 }}>🧬 Hereditary Short Stature Atlas</div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: '#f8fafc' }}>
          Hereditary-Short-Stature-Atlas
        </h1>
        <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 6 }}>
          Complete 8-Gene Reference — SHOX · GH1 · GHR · IGF1 · IGFALS · STAT5B · CUL7 · NPR2
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
            <MetricCard label="Genes Covered" value={overview.genes_covered?.length} sub="GH-axis + skeletal + CNP pathway" />
            <MetricCard label="rGH Therapy" value={`${overview.aggregate_metrics?.rgh_therapy_pct}%`} sub="SHOX / GH1 / CUL7 subset" />
            <MetricCard label="Mecasermin" value={`${overview.aggregate_metrics?.mecasermin_pct}%`} sub="GHR / IGF1 / STAT5B" />
            <MetricCard label="Orthopaedic Surgery" value={`${overview.aggregate_metrics?.orthopedic_surgery_pct}%`} sub="limb lengthening / osteotomy" />
            <MetricCard label="SNHL" value={`${overview.aggregate_metrics?.snhl_pct}%`} sub="IGF1 deficiency predominant" />
            <MetricCard label="Immune Dysregulation" value={`${overview.aggregate_metrics?.immune_dysregulation_pct}%`} sub="STAT5B predominant" />
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
                  <b style={{ color: '#a78bfa' }}>Severity:</b> {g.severity_sds} &nbsp;|&nbsp;
                  <b style={{ color: '#60a5fa' }}>Hormone:</b> {g.hormone_profile}
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#60a5fa' }}>n={g.n_patients}</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#34d399' }}>rGH {g.rgh_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fbbf24' }}>Mecasermin {g.mecasermin_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#f87171' }}>Surgery {g.surgery_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#a78bfa' }}>SNHL {g.snhl_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fb923c' }}>Immune {g.immune_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#94a3b8' }}>Avg dx {g.avg_age_dx_years}y</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#e2e8f0' }}>Avg HtSDS {g.avg_height_sds}</span>
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
                {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease', 'GH Profile', 'Severity (SDS)', 'rGH%', 'Meca%', 'Surgery%', 'SNHL%', 'Immune%', 'Avg Age Dx'].map(h => (
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
                  <td style={{ padding: '10px 12px', color: '#fbbf24', borderBottom: '1px solid #1e293b', whiteSpace: 'nowrap' }}>{g.severity_sds?.split('(')[0]?.trim()}</td>
                  <td style={{ padding: '10px 12px', color: '#34d399', borderBottom: '1px solid #1e293b' }}>{g.rgh_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#fb923c', borderBottom: '1px solid #1e293b' }}>{g.mecasermin_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#f87171', borderBottom: '1px solid #1e293b' }}>{g.surgery_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#a78bfa', borderBottom: '1px solid #1e293b' }}>{g.snhl_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#f97316', borderBottom: '1px solid #1e293b' }}>{g.immune_pct}%</td>
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
                <b style={{ color: '#a78bfa' }}>Severity:</b> {g.severity_sds} &nbsp;|&nbsp;
                <b style={{ color: '#60a5fa' }}>GH Profile:</b> {g.hormone_profile}
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
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#34d399' }}>rGH {g.rgh_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fb923c' }}>Mecasermin {g.mecasermin_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#f87171' }}>Surgery {g.surgery_pct}%</span>
                {g.gh_deficiency && <span style={{ background: '#1d4ed8', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fff' }}>GH deficient</span>}
                {g.mecasermin_indicated && <span style={{ background: '#7c3aed', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fff' }}>Mecasermin Rx</span>}
                {g.autosomal_recessive_risk && <span style={{ background: '#065f46', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fff' }}>AR risk</span>}
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fbbf24' }}>Onset: {g.onset_age?.slice(0, 40)}</span>
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

          {/* Glossary */}
          <h3 style={{ color: '#60a5fa', marginTop: 8 }}>Short Stature Genetics Glossary</h3>
          {Object.entries(definitions.short_stature_glossary || {}).map(([term, text]) => (
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
