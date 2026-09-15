'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-platelet-disorders-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ITGA2B': '#b71c1c',  // deep red    — GT type A, alphaIIb, most common GT
  'ITGB3':  '#1565c0',  // deep blue   — GT type A/B, beta3, alphaIIbbeta3 complex
  'GP1BA':  '#7b1fa2',  // deep purple — BSS, GPIbalpha, giant platelets
  'GP9':    '#2e7d32',  // dark green  — BSS, GPIX, GPIb-IX-V complex
  'MYH9':   '#e65100',  // deep orange — MYH9-RD, Döhle inclusions, NMHC-IIA
  'ANKRD26':'#00695c',  // dark teal   — THC2, 5\'UTR mutations, AML risk 5%
  'RUNX1':  '#37474f',  // dark slate  — FPD/AML, AML 35-40%, dense granule defect
  'GFI1B':  '#4527a0',  // deep indigo — GFI1B-RD, CD42b on RBCs PATHOGNOMONIC
};

const GENE_INFO = {
  'ITGA2B': { full: 'ITGA2B / Integrin alpha-IIb / 1039aa', locus: '17q21.31', size: '1039 aa / 117 kDa (heavy + light chain; alphaIIbbeta3 integrin; fibrinogen receptor)', inh: 'AR/AD' },
  'ITGB3':  { full: 'ITGB3 / Integrin beta-3 / 788aa', locus: '17q21.32', size: '788 aa / 87 kDa (beta3 subunit; alphaIIbbeta3 + alphaVbeta3; fibrinogen/vWF receptor)', inh: 'AR/AD' },
  'GP1BA':  { full: 'GP1BA / Glycoprotein Ib alpha / 626aa', locus: '17p13.2', size: '626 aa / 70 kDa (GPIbalpha; leucine-rich repeat; vWF-A1 domain receptor)', inh: 'AR/AD' },
  'GP9':    { full: 'GP9 / Glycoprotein IX / 160aa', locus: '3q21.3', size: '160 aa / 17 kDa (GPIX; GPIb-IX-V complex; vWF binding co-receptor)', inh: 'AR' },
  'MYH9':   { full: 'MYH9 / Myosin-9 / Non-Muscle Heavy Chain IIA / 1960aa', locus: '22q12.3-q13.1', size: '1960 aa / 227 kDa (non-muscle myosin IIA heavy chain; ATPase motor; cytoskeletal)', inh: 'AD' },
  'ANKRD26':{ full: 'ANKRD26 / Ankyrin Repeat Domain 26 / 1710aa', locus: '10p12.1', size: '1710 aa / 192 kDa (5\' UTR mutations only; RUNX1/FLI1 silencing; MK differentiation regulator)', inh: 'AD' },
  'RUNX1':  { full: 'RUNX1 / Runt-Related Transcription Factor 1 / 453aa', locus: '21q22.12', size: '453 aa / 49 kDa (runt domain TF; master haematopoiesis regulator; dense granule defect)', inh: 'AD' },
  'GFI1B':  { full: 'GFI1B / Growth Factor Independence 1B / 330aa', locus: '9q34.13', size: '330 aa / 37 kDa (zinc-finger transcriptional repressor; MK/erythroid lineage; CD42b on RBCs)', inh: 'AD/AR' },
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

export default function HereditaryPlateletDisordersAtlas() {
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
      Loading Platelet Disorders Atlas…
    </div>
  );
  if (error) return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#ef4444', padding: 32, fontSize: 16 }}>
      Error: {error}
    </div>
  );

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%)', borderBottom: '1px solid #1e3a5f', padding: '20px 28px 14px' }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: '#38bdf8', letterSpacing: 1 }}>
          🧬 Hereditary Platelet Disorders Atlas
        </div>
        <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>
          Complete 8-Gene GT / BSS / MYH9-RD / FPD-AML / GFI1B Reference — ITGA2B · ITGB3 · GP1BA · GP9 · MYH9 · ANKRD26 · RUNX1 · GFI1B
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          {geneList.map(g => (
            <GeneChip key={g} gene={g} active={activeGene} onClick={g => setActiveGene(activeGene === g ? null : g)} />
          ))}
        </div>
        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginTop: 14 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              background: tab === t ? '#38bdf8' : '#1e293b',
              color: tab === t ? '#0f172a' : '#94a3b8',
              border: 'none', borderRadius: 4, padding: '5px 14px', fontSize: 12,
              fontWeight: tab === t ? 700 : 400, cursor: 'pointer',
            }}>{t}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: '20px 28px' }}>

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2830-2837" />
              <MetricCard label="Genes Covered" value={overview.genes?.length} sub="GT / BSS / MYH9-RD / FPD / GFI1B" />
              <MetricCard label="Inheritance Types" value="AR/AD/XLR" sub="all modes represented" />
              <MetricCard label="Unique Mechanisms" value="5" sub="alphaIIbbeta3 / GPIb-IX-V / NMHC / MK-diff / GFI1B" />
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 10 }}>
              Gene Summary — Platelet Count, Bleeding Score &amp; AML Risk
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene','Locus','Inh','Pts','Median Plt (×10⁹/L)','Median Bleed Score','Alloimmunised %','Giant Plt %','AML/MDS %','HSCT %'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', color: '#38bdf8', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.filter(g => !activeGene || g.gene === activeGene).map((g, i) => (
                    <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1a2332', cursor: 'pointer' }}
                        onClick={() => setActiveGene(activeGene === g.gene ? null : g.gene)}>
                      <td style={{ padding: '5px 8px' }}>
                        <GeneChip gene={g.gene} active={activeGene} />
                      </td>
                      <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{g.locus}</td>
                      <td style={{ padding: '5px 8px', color: '#fbbf24' }}>{GENE_INFO[g.gene]?.inh}</td>
                      <td style={{ padding: '5px 8px' }}>{g.n_patients}</td>
                      <td style={{ padding: '5px 8px', color: g.median_platelet_count < 30 ? '#ef4444' : '#fbbf24' }}>
                        {g.median_platelet_count}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.median_bleeding_score > 5 ? '#ef4444' : '#fbbf24' }}>
                        {g.median_bleeding_score}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_alloimmunised > 20 ? '#ef4444' : '#fbbf24' }}>
                        {g.pct_alloimmunised}%
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_giant_platelets > 0 ? '#fbbf24' : '#475569' }}>
                        {g.pct_giant_platelets > 0 ? `${g.pct_giant_platelets}%` : '—'}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_aml_mds > 10 ? '#ef4444' : g.pct_aml_mds > 0 ? '#fbbf24' : '#475569' }}>
                        {g.pct_aml_mds > 0 ? `${g.pct_aml_mds}%` : '—'}
                      </td>
                      <td style={{ padding: '5px 8px' }}>{g.pct_hsct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginTop: 24, marginBottom: 10 }}>
              Pathway Categories
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 12 }}>
              {overview.pathway_categories?.map((cat, i) => (
                <div key={i} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, color: '#fbbf24', fontSize: 12, marginBottom: 6 }}>{cat.pathway}</div>
                  <div style={{ marginBottom: 6, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                    {cat.genes?.map(g => <GeneChip key={g} gene={g} active={null} />)}
                  </div>
                  <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{cat.note}</div>
                </div>
              ))}
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginTop: 24, marginBottom: 10 }}>
              Critical Distinctions
            </div>
            {overview.critical_distinctions?.map((d, i) => (
              <div key={i} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6, padding: '8px 12px', marginBottom: 6, fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>
                ⚡ {d}
              </div>
            ))}
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && (
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 14 }}>
              Gene Reference — Protein, Locus, Inheritance
            </div>
            {geneList.filter(g => !activeGene || g === activeGene).map(g => (
              <div key={g} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[g]}44`, borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <GeneChip gene={g} active={null} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{GENE_INFO[g]?.locus}</span>
                  <span style={{ color: '#fbbf24', fontSize: 12, fontWeight: 700 }}>{GENE_INFO[g]?.inh}</span>
                </div>
                <div style={{ fontSize: 12, color: '#e2e8f0', marginBottom: 4 }}>{GENE_INFO[g]?.full}</div>
                <div style={{ fontSize: 11, color: '#64748b' }}>{GENE_INFO[g]?.size}</div>
              </div>
            ))}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            {breakdown.genes?.filter(g => !activeGene || g.gene === activeGene).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[g.gene]}55`, borderRadius: 10, padding: 18, marginBottom: 18 }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10 }}>
                  <GeneChip gene={g.gene} active={null} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus}</span>
                  <span style={{ color: '#fbbf24', fontSize: 11 }}>{GENE_INFO[g.gene]?.inh}</span>
                  <span style={{ color: '#64748b', fontSize: 11 }}>{g.n_patients} patients</span>
                </div>
                {[['Protein', g.protein_size], ['Inheritance', g.inheritance], ['Disease', g.disease_category],
                  ['Pathway', g.disease_pathway], ['Pathognomonic', g.pathognomonic], ['Treatment', g.treatment]].map(([label, val]) => (
                  <div key={label} style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, color: '#38bdf8', fontWeight: 700, marginBottom: 2 }}>{label.toUpperCase()}</div>
                    <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{val}</div>
                  </div>
                ))}
                {g.patients && g.patients.length > 0 && (
                  <div>
                    <div style={{ fontSize: 11, color: '#38bdf8', fontWeight: 700, marginBottom: 6 }}>SAMPLE PATIENTS (first 5)</div>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                        <thead>
                          <tr style={{ background: '#0f172a' }}>
                            {['ID','Sex','Age Dx','Plt ×10⁹/L','Bleed Score','Alloimmunised','AML/MDS','HSCT'].map(h => (
                              <th key={h} style={{ padding: '4px 8px', color: '#64748b', textAlign: 'left' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {g.patients.map((p, i) => (
                            <tr key={p.patient_id} style={{ background: i % 2 === 0 ? '#1a2332' : '#0f172a' }}>
                              <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{p.patient_id}</td>
                              <td style={{ padding: '3px 8px' }}>{p.sex}</td>
                              <td style={{ padding: '3px 8px' }}>{p.age_at_diagnosis_years}y</td>
                              <td style={{ padding: '3px 8px', color: p.platelet_count < 30 ? '#ef4444' : '#fbbf24' }}>{p.platelet_count}</td>
                              <td style={{ padding: '3px 8px', color: p.bleeding_score_isth > 5 ? '#ef4444' : '#fbbf24' }}>{p.bleeding_score_isth}</td>
                              <td style={{ padding: '3px 8px', color: p.alloimmunised ? '#ef4444' : '#475569' }}>{p.alloimmunised ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px', color: p.aml_mds_event ? '#ef4444' : '#475569' }}>{p.aml_mds_event ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px' }}>{p.hsct_performed ? 'Yes' : 'No'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && defs && (
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 14 }}>
              Glossary — GT / BSS / MYH9-RD / FPD-AML / GFI1B / alphaIIbbeta3 / GPIb-IX-V
            </div>
            {defs.definitions?.map((d, i) => (
              <div key={i} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: '12px 16px', marginBottom: 10 }}>
                <div style={{ fontWeight: 700, color: '#fbbf24', fontSize: 13, marginBottom: 6 }}>{d.term}</div>
                <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{d.definition}</div>
              </div>
            ))}
            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginTop: 20, marginBottom: 10 }}>
              Standards &amp; References
            </div>
            {defs.standards?.map((s, i) => (
              <div key={i} style={{ background: '#1e293b', border: '1px solid #1e3a5f', borderRadius: 6, padding: '6px 12px', marginBottom: 6, fontSize: 11, color: '#94a3b8' }}>
                📋 {s}
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  );
}
