'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-thyroid-disorder-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'TSHR':   '#1565c0',  // deep blue       — FNAH, GOF, suppressed TSH, negative autoantibodies
  'DUOX2':  '#2e7d32',  // dark green      — most common CH dyshormonogenesis, transient CH
  'TPO':    '#6a1b9a',  // deep purple     — total organification defect, perchlorate >90%
  'TG':     '#e65100',  // burnt orange    — undetectable TG despite goitre — paradox
  'SLC5A5': '#37474f',  // blue-grey       — ITD, absent RAI uptake <5%, NIS absent
  'FOXE1':  '#880e4f',  // dark rose       — Bamforth-Lazarus, thyroid agenesis, cleft palate
  'PAX8':   '#004d40',  // dark teal       — thyroid dysgenesis, variable expressivity, AD
  'NKX2-1': '#bf360c',  // burnt sienna    — BLT syndrome, benign hereditary chorea, RDS
};

const GENE_INFO = {
  'TSHR':   { full: 'TSHR / TSH Receptor / 764aa', locus: '14q31.1', size: '764 aa / 87 kDa', inh: 'AD GOF / AR LOF', disease: 'FNAH: suppressed TSH + elevated FT4/FT3 + NEGATIVE TRAb/TPO-Ab = TSHR GOF PATHOGNOMONIC; FGH: hCG-sensitive, gestational thyrotoxicosis; TSH resistance LOF: elevated TSH, normal FT4; radioiodine NOT curative for FNAH; methimazole or total thyroidectomy' },
  'DUOX2':  { full: 'DUOX2 / Dual Oxidase 2 / 1548aa', locus: '15q21.1', size: '1548 aa / 175 kDa', inh: 'AR', disease: 'Most common hereditary CH from dyshormonogenesis (~15-20% of CH); perchlorate discharge >10% = partial organification defect; TRANSIENT CH: monoallelic DUOX2 — trial L-T4 cessation at age 3; DUOXA2 maturation factor = same phenotype' },
  'TPO':    { full: 'TPO / Thyroid Peroxidase / 933aa', locus: '2p25.3', size: '933 aa / 103 kDa', inh: 'AR', disease: 'TOTAL organification defect; perchlorate discharge >90% PATHOGNOMONIC; large goitre at birth; permanent CH — no transient cases; Pendred DDx (SLC26A4): EVA + SNHL absent in TPO' },
  'TG':     { full: 'TG / Thyroglobulin / 2768aa', locus: '8q24.22', size: '2768 aa / 330 kDa', inh: 'AR', disease: 'CH + goitre + UNDETECTABLE/LOW serum TG (paradox) PATHOGNOMONIC; most common goitrous CH in iodine-sufficient countries; TG = largest secreted protein; perchlorate mildly positive (15-30%); RAI uptake normal' },
  'SLC5A5': { full: 'SLC5A5 / NIS / 643aa', locus: '19p13.11', size: '643 aa / 70 kDa', inh: 'AR', disease: 'Iodide transport defect (ITD); 123I uptake <5% at 24h despite elevated TSH PATHOGNOMONIC; pertechnetate also absent; saliva/serum iodide ratio <30 (normal >40); radioiodine NOT effective; iodide supplementation does NOT help' },
  'FOXE1':  { full: 'FOXE1 / TTF-2 / 373aa', locus: '9q22.33', size: '373 aa / 42 kDa', inh: 'AR', disease: 'Bamforth-Lazarus: thyroid agenesis + cleft palate + bifid epiglottis + spiky hair TETRAD PATHOGNOMONIC (<50 cases); neonatal emergency: choanal atresia hypoxia + profound CH; IV levothyroxine + airway management simultaneously' },
  'PAX8':   { full: 'PAX8 / Paired Box 8 / 450aa', locus: '2q13', size: '450 aa / 48 kDa', inh: 'AD', disease: 'Thyroid dysgenesis (hypoplasia/ectopy/athyreosis); VARIABLE EXPRESSIVITY — affected parent may be euthyroid (clinical trap); ectopic lingual thyroid — DO NOT excise (only thyroid tissue); positive family history; Tc scan mandatory' },
  'NKX2-1': { full: 'NKX2-1 / TTF-1/TITF1 / 371aa', locus: '14q13.3', size: '371 aa / 41 kDa', inh: 'AD', disease: 'Brain-Lung-Thyroid (BLT) syndrome: benign hereditary chorea (onset 1-5y, NO cognitive decline) + neonatal RDS (surfactant deficiency) + CH TRIAD PATHOGNOMONIC; avoid tetrabenazine; TTF-1 = lung adenocarcinoma IHC marker (somatic)' },
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

export default function HeredThyroidDisorderAtlasPage() {
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
        <div style={{ color: '#60a5fa', fontSize: 13, marginBottom: 4 }}>🧬 Hereditary Thyroid Disorder Atlas</div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: '#f8fafc' }}>
          Hereditary-Thyroid-Disorder-Atlas
        </h1>
        <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 6 }}>
          Complete 8-Gene Reference — TSHR · DUOX2 · TPO · TG · SLC5A5 · FOXE1 · PAX8 · NKX2-1
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
            <MetricCard label="Genes Covered" value={overview.genes_covered?.length} sub="Thyroid disorder pathways" />
            <MetricCard label="On Levothyroxine" value={`${overview.aggregate_metrics?.levothyroxine_pct}%`} sub="CH / hypothyroid treatment" />
            <MetricCard label="Thyroid Dysgenesis" value={`${overview.aggregate_metrics?.dysgenesis_pct}%`} sub="hypoplasia / ectopy / agenesis" />
            <MetricCard label="Surgery Rate" value={`${overview.aggregate_metrics?.surgery_pct}%`} sub="thyroidectomy / airway" />
            <MetricCard label="Goitre Present" value={`${overview.aggregate_metrics?.goitre_pct}%`} sub="dyshormonogenesis goitre" />
            <MetricCard label="Under Surveillance" value={`${overview.aggregate_metrics?.surveillance_pct}%`} sub="active monitoring" />
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
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#34d399' }}>Levo {g.levothyroxine_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fbbf24' }}>Goitre {g.goitre_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#f87171' }}>Surgery {g.surgery_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#a78bfa' }}>Dysgenesis {g.dysgenesis_pct}%</span>
                  <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fb923c' }}>Surveillance {g.surveillance_pct}%</span>
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
                {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease', 'Hormone Profile', 'Levo%', 'Goitre%', 'Surgery%', 'Dysgenesis%', 'Surveillance%', 'Avg Age Dx'].map(h => (
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
                  <td style={{ padding: '10px 12px', color: '#34d399', borderBottom: '1px solid #1e293b' }}>{g.levothyroxine_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#fbbf24', borderBottom: '1px solid #1e293b' }}>{g.goitre_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#f87171', borderBottom: '1px solid #1e293b' }}>{g.surgery_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#a78bfa', borderBottom: '1px solid #1e293b' }}>{g.dysgenesis_pct}%</td>
                  <td style={{ padding: '10px 12px', color: '#fb923c', borderBottom: '1px solid #1e293b' }}>{g.surveillance_pct}%</td>
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
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#34d399' }}>Levo {g.levothyroxine_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fbbf24' }}>Goitre {g.goitre_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#f87171' }}>Surgery {g.surgery_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#a78bfa' }}>Dysgenesis {g.dysgenesis_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fb923c' }}>Surveillance {g.surveillance_pct}%</span>
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

          {/* Thyroid Glossary */}
          <h3 style={{ color: '#60a5fa', marginTop: 8 }}>Thyroid Genetics Glossary</h3>
          {Object.entries(definitions.thyroid_glossary || {}).map(([term, text]) => (
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
