'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-neonatal-diabetes-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'KCNJ11': '#b71c1c',  // deep red      — Kir6.2, most common PNDM, sulfonylurea curative
  'ABCC8':  '#1565c0',  // deep blue     — SUR1, TNDM/PNDM, sulfonylurea
  'INS':    '#e65100',  // deep orange   — ER stress, insulin only
  'EIF2AK3':'#2e7d32',  // dark green    — PERK/WRS, AR, epiphyseal dysplasia + liver
  'FOXP3':  '#6a1b9a',  // deep purple   — IPEX, XLR, HSCT only cure
  'RFX6':   '#00695c',  // dark teal     — Mitchell-Riley, AR, intestinal atresia
  'GLIS3':  '#4527a0',  // deep indigo   — ND + CH + CHD + renal cysts
  'PDX1':   '#c62828',  // crimson       — pancreatic agenesis, glucagon pen FAILS
};

const GENE_INFO = {
  'KCNJ11':  { full: 'KCNJ11 / Kir6.2 / 390aa', locus: '11p15.1', size: '390 aa / 43 kDa (K-ATP pore; GOF → channel stays open → no depolarisation → no insulin; PNDM/TNDM/DEND; sulfonylurea CURATIVE ~90%; AD de novo)', inh: 'AD GOF' },
  'ABCC8':   { full: 'ABCC8 / SUR1 / 1581aa', locus: '11p15.1', size: '1581 aa / 177 kDa (K-ATP regulatory; GOF → no insulin secretion; TNDM/PNDM; sulfonylurea ~75-80%; AR LOF = OPPOSITE = congenital hyperinsulinism)', inh: 'AD GOF' },
  'INS':     { full: 'INS / Preproinsulin / 110aa', locus: '11p15.5', size: '110 aa → 51 aa mature insulin (AD missense → ER stress → β-cell apoptosis; PNDM; insulin ONLY; no SU response; no exocrine insufficiency)', inh: 'AD' },
  'EIF2AK3': { full: 'EIF2AK3 / PERK / 1116aa', locus: '2p11.2', size: '1116 aa / 126 kDa (ER stress sensor; PERK LOF → WRS triad: ND + epiphyseal dysplasia + liver failure; AR; consanguineous; most common AR ND)', inh: 'AR LOF' },
  'FOXP3':   { full: 'FOXP3 / Treg master TF / 431aa', locus: 'Xp11.23', size: '431 aa / 47 kDa (Treg master regulator; LOF → IPEX: Immune dysregulation + Polyendocrinopathy + Enteropathy + X-linked; islet Ab POSITIVE; HSCT only cure)', inh: 'XLR' },
  'RFX6':    { full: 'RFX6 / Winged-helix TF / 890aa', locus: '6q22.31', size: '890 aa / 99 kDa (pancreatic endocrine development TF; LOF → Mitchell-Riley: ND + hypothyroidism + intestinal atresia + gallbladder agenesis; AR)', inh: 'AR LOF' },
  'GLIS3':   { full: 'GLIS3 / GLI-similar ZF3 / 829aa', locus: '9p24.2', size: '829 aa / 95 kDa (β-cell + thyroid + kidney + heart + liver TF; AR LOF → ND + congenital hypothyroidism + CHD + polycystic kidneys + liver fibrosis)', inh: 'AR LOF' },
  'PDX1':    { full: 'PDX1 / Pancreas master TF / 283aa', locus: '13q12.2', size: '283 aa / 32 kDa (AR homozygous → pancreatic agenesis: complete absent exocrine + endocrine; glucagon pen FAILS; PERT mandatory; AD het = MODY4)', inh: 'AR LOF' },
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

export default function HereditaryNeonatalDiabetesAtlas() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [activeGene, setActiveGene] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const [ovRes, bkRes, dfRes] = await Promise.all([
          fetch(`${API}/api/${SLUG}/overview`),
          fetch(`${API}/api/${SLUG}/breakdown`),
          fetch(`${API}/api/${SLUG}/definitions`),
        ]);
        setOverview(await ovRes.json());
        setBreakdown(await bkRes.json());
        setDefs(await dfRes.json());
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const geneList = Object.keys(GENE_COLORS);

  if (loading) return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#94a3b8', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18 }}>
      Loading Neonatal Diabetes Atlas…
    </div>
  );
  if (error) return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#ef4444', padding: 32, fontSize: 16 }}>
      Error: {error}
    </div>
  );

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)', borderBottom: '1px solid #334155', padding: '24px 32px' }}>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4, letterSpacing: 2, textTransform: 'uppercase' }}>
          Hereditary Disease Atlas · Endocrine · Monogenic Diabetes
        </div>
        <h1 style={{ fontSize: 24, fontWeight: 800, color: '#f1f5f9', margin: 0 }}>
          🧬 Hereditary Neonatal Diabetes Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 6 }}>
          Complete 8-Gene Monogenic Neonatal/Infancy-Onset Diabetes Reference · KCNJ11-ABCC8-INS-EIF2AK3-FOXP3-RFX6-GLIS3-PDX1
        </div>
        <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 0 }}>
          {geneList.map(g => (
            <GeneChip key={g} gene={g} active={activeGene} onClick={g2 => setActiveGene(activeGene === g2 ? null : g2)} />
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#1e293b', borderBottom: '1px solid #334155', display: 'flex', padding: '0 32px' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: 'none', border: 'none', color: tab === t ? '#38bdf8' : '#94a3b8',
            borderBottom: tab === t ? '2px solid #38bdf8' : '2px solid transparent',
            padding: '12px 20px', cursor: 'pointer', fontSize: 14, fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 32px' }}>
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2894-2901" />
              <MetricCard label="Genes Covered" value={overview.total_genes} sub="K-ATP / ER-stress / IPEX / Dev-TF" />
              <MetricCard label="SU-Curative Genes" value="2" sub="KCNJ11 + ABCC8" warn={false} />
              <MetricCard label="Onset Criterion" value="<6 mo" sub="Mandatory genetic test" warn={true} />
            </div>

            <div style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 12, fontSize: 16 }}>🔑 Key Clinical Facts</div>
              {overview.key_facts && overview.key_facts.map((f, i) => (
                <div key={i} style={{ padding: '6px 10px', marginBottom: 6, background: '#0f172a', borderRadius: 6, fontSize: 13, color: '#cbd5e1', borderLeft: '3px solid #38bdf8' }}>
                  {f}
                </div>
              ))}
            </div>

            <div style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 12, fontSize: 16 }}>🩺 Diagnostic Algorithm</div>
              <div style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.8, fontFamily: 'monospace' }}>
                {overview.diagnostic_algorithm}
              </div>
            </div>

            <div style={{ background: '#1e293b', borderRadius: 10, padding: 20 }}>
              <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 12, fontSize: 16 }}>📊 Categories</div>
              {overview.categories && Object.entries(overview.categories).map(([cat, genes]) => (
                <div key={cat} style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{cat}</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                    {genes.map(g => <GeneChip key={g} gene={g} active={null} />)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === 'Gene Table' && overview && (
          <div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
                  {['Gene', 'Locus', 'Inh.', 'Patients', 'Avg Onset (wks)', 'Avg HbA1c (%)', 'SU Response', 'Autoantibodies'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', borderBottom: '1px solid #334155', fontSize: 11, textTransform: 'uppercase' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {overview.gene_rows && overview.gene_rows.map(row => {
                  const info = GENE_INFO[row.gene] || {};
                  return (
                    <tr key={row.gene} style={{ borderBottom: '1px solid #1e293b', background: activeGene === row.gene ? '#1a2744' : 'transparent' }}
                      onClick={() => setActiveGene(activeGene === row.gene ? null : row.gene)}>
                      <td style={{ padding: '10px 12px' }}>
                        <GeneChip gene={row.gene} active={null} />
                        <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{info.full}</div>
                      </td>
                      <td style={{ padding: '10px 12px', color: '#94a3b8' }}>{row.locus}</td>
                      <td style={{ padding: '10px 12px', color: '#94a3b8' }}>{info.inh}</td>
                      <td style={{ padding: '10px 12px', color: '#38bdf8', fontWeight: 700 }}>{row.patients}</td>
                      <td style={{ padding: '10px 12px', color: '#a3e635' }}>{row.avg_onset_weeks}</td>
                      <td style={{ padding: '10px 12px', color: '#fbbf24' }}>{row.avg_hba1c_pct}</td>
                      <td style={{ padding: '10px 12px' }}>
                        <span style={{ background: row.sulfonylurea_response ? '#14532d' : '#450a0a', color: row.sulfonylurea_response ? '#86efac' : '#fca5a5', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 700 }}>
                          {row.sulfonylurea_response ? `YES ~${row.su_response_pct}%` : 'NO'}
                        </span>
                      </td>
                      <td style={{ padding: '10px 12px' }}>
                        <span style={{ background: row.autoantibodies === 'POSITIVE' ? '#7c2d12' : '#0c4a6e', color: row.autoantibodies === 'POSITIVE' ? '#fed7aa' : '#7dd3fc', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 700 }}>
                          {row.autoantibodies}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            {breakdown.genes && breakdown.genes
              .filter(g => activeGene === null || g.gene === activeGene)
              .map(entry => (
                <div key={entry.gene} style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 20, borderLeft: `4px solid ${GENE_COLORS[entry.gene] || '#555'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                    <GeneChip gene={entry.gene} active={null} />
                    <span style={{ color: '#94a3b8', fontSize: 13 }}>{entry.locus}</span>
                    <span style={{ color: '#64748b', fontSize: 12 }}>{GENE_INFO[entry.gene]?.inh}</span>
                  </div>
                  {[
                    ['Protein / Function', entry.protein_size],
                    ['Inheritance', entry.inheritance],
                    ['Disease Category', entry.disease_category],
                    ['Pathway', entry.disease_pathway],
                    ['Pathognomonic / Characteristic', entry.pathognomonic],
                    ['Treatment', entry.treatment],
                  ].map(([label, text]) => text && (
                    <div key={label} style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>{label}</div>
                      <div style={{ fontSize: 13, color: '#cbd5e1', lineHeight: 1.7, background: '#0f172a', borderRadius: 6, padding: '8px 12px', fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>{text}</div>
                    </div>
                  ))}
                </div>
              ))}
            {activeGene && (
              <button onClick={() => setActiveGene(null)} style={{ background: '#334155', color: '#94a3b8', border: 'none', borderRadius: 6, padding: '8px 16px', cursor: 'pointer', fontSize: 13 }}>
                Show all genes
              </button>
            )}
          </div>
        )}

        {tab === 'Definitions' && defs && (
          <div>
            {defs.definitions && defs.definitions.map(d => (
              <div key={d.term} style={{ background: '#1e293b', borderRadius: 8, padding: '16px 20px', marginBottom: 12, borderLeft: '3px solid #0ea5e9' }}>
                <div style={{ fontWeight: 700, color: '#38bdf8', fontSize: 14, marginBottom: 8 }}>{d.term}</div>
                <div style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.7 }}>{d.definition}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
