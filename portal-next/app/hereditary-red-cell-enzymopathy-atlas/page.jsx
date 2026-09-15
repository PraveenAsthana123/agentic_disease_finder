'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-red-cell-enzymopathy-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'G6PD':  '#b71c1c',  // deep red    — most common, X-linked
  'PKLR':  '#1565c0',  // deep blue   — most common CNSHA, mitapivat
  'HK1':   '#7b1fa2',  // deep purple — 2,3-DPG low DDx
  'GPI':   '#2e7d32',  // dark green  — 2nd most common CNSHA
  'PGK1':  '#e65100',  // deep orange — XLR triad CNSHA+myopathy+CNS
  'TPI1':  '#00695c',  // dark teal   — most severe, neurodegeneration
  'ALDOA': '#37474f',  // dark slate  — triad CNSHA+myopathy+ID
  'PFKM':  '#4527a0',  // deep indigo — Tarui, no second wind
};

const GENE_INFO = {
  'G6PD':  { full: 'G6PD / Glucose-6-Phosphate Dehydrogenase / 515aa', locus: 'Xq28', size: '515 aa / 59 kDa (PPP step 1; NADPH production; oxidative defence; ~500M worldwide; WHO class I-V)', inh: 'XLR' },
  'PKLR':  { full: 'PKLR / Pyruvate Kinase L-R isoform / 574aa', locus: '1q22', size: '574 aa / 62 kDa (glycolysis step 10; ATP production; 2,3-DPG elevated; mitapivat FDA 2022; R479W most common)', inh: 'AR' },
  'HK1':   { full: 'HK1 / Hexokinase 1 / 917aa', locus: '10q22.1', size: '917 aa / 100 kDa (glycolysis step 1; HK1-E RBC isoform; 2,3-DPG LOW — DDx from PKLR; severe CNSHA)', inh: 'AR' },
  'GPI':   { full: 'GPI / Glucose-6-Phosphate Isomerase / 558aa', locus: '19q13.11', size: '558 aa / 63 kDa (glycolysis step 2; G6P→F6P; 2nd most common CNSHA; French p.Arg347His; also AMF in tumours)', inh: 'AR' },
  'PGK1':  { full: 'PGK1 / Phosphoglycerate Kinase 1 / 417aa', locus: 'Xq21.1', size: '417 aa / 45 kDa (glycolysis step 7; first ATP-generating step; XLR; UNIQUE TRIAD: CNSHA+myopathy+CNS)', inh: 'XLR' },
  'TPI1':  { full: 'TPI1 / Triosephosphate Isomerase 1 / 286aa', locus: '12p13.31', size: '286 aa / 27 kDa subunit; homodimer (glycolysis step 5; DHAP→GAP; DHAP neurotoxic; most severe; fatal childhood)', inh: 'AR' },
  'ALDOA': { full: 'ALDOA / Aldolase A / 364aa', locus: '16p11.2', size: '364 aa / 39 kDa (glycolysis step 4; F-1,6-BP→DHAP+GAP; muscle/brain/RBC isoform; CNSHA+myopathy+ID triad)', inh: 'AR' },
  'PFKM':  { full: 'PFKM / Phosphofructokinase M-subunit / 780aa', locus: '12q13.11', size: '780 aa / 85 kDa (glycolysis step 3; M4 in muscle = complete loss; L4 in RBC preserved → mild hemolysis; Tarui GSD VII)', inh: 'AR' },
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

export default function HereditaryRedCellEnzymopathyAtlas() {
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
      Loading Red Cell Enzymopathy Atlas…
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
          🧬 Hereditary Red Cell Enzymopathy Atlas
        </div>
        <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>
          Complete 8-Gene PPP / Glycolysis Enzymopathy Reference — G6PD · PKLR · HK1 · GPI · PGK1 · TPI1 · ALDOA · PFKM
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
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2838-2845" />
              <MetricCard label="Genes Covered" value={overview.genes?.length} sub="PPP / Glycolysis enzymopathies" />
              <MetricCard label="Inheritance Types" value="AR/XLR" sub="AR (6 genes) + XLR (G6PD, PGK1)" />
              <MetricCard label="Pathway Classes" value="3" sub="PPP / Upper Glycolysis / Central Glycolysis" />
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 10 }}>
              Gene Summary — Hb, Reticulocytes, LDH, Neuro%, Myopathy%
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene','Locus','Inh','Pts','Median Hb (g/dL)','Mean Retic %','Mean LDH (U/L)','Splenomegaly %','Neuro %','Myopathy %','Transfusion %','Enzyme Act %'].map(h => (
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
                      <td style={{ padding: '5px 8px', color: g.median_hb_g_dl < 9 ? '#ef4444' : '#fbbf24' }}>
                        {g.median_hb_g_dl}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.mean_retic_pct > 15 ? '#ef4444' : '#fbbf24' }}>
                        {g.mean_retic_pct}%
                      </td>
                      <td style={{ padding: '5px 8px', color: g.mean_ldh_u_l > 600 ? '#ef4444' : '#fbbf24' }}>
                        {g.mean_ldh_u_l}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_splenomegaly > 70 ? '#ef4444' : '#fbbf24' }}>
                        {g.pct_splenomegaly}%
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_neuro > 50 ? '#ef4444' : g.pct_neuro > 0 ? '#fbbf24' : '#475569' }}>
                        {g.pct_neuro > 0 ? `${g.pct_neuro}%` : '—'}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_myopathy > 50 ? '#ef4444' : g.pct_myopathy > 0 ? '#fbbf24' : '#475569' }}>
                        {g.pct_myopathy > 0 ? `${g.pct_myopathy}%` : '—'}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_transfusion_dependent > 30 ? '#ef4444' : '#fbbf24' }}>
                        {g.pct_transfusion_dependent}%
                      </td>
                      <td style={{ padding: '5px 8px', color: g.mean_enzyme_activity_pct < 15 ? '#ef4444' : '#fbbf24' }}>
                        {g.mean_enzyme_activity_pct}%
                      </td>
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
                            {['ID','Sex','Age Dx','Hb g/dL','Retic %','LDH U/L','Bili µmol/L','Enzyme %','Spleen','Neuro','Myopathy','Transfusion'].map(h => (
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
                              <td style={{ padding: '3px 8px', color: p.hemoglobin_g_dl < 9 ? '#ef4444' : '#fbbf24' }}>{p.hemoglobin_g_dl}</td>
                              <td style={{ padding: '3px 8px', color: p.reticulocyte_pct > 15 ? '#ef4444' : '#fbbf24' }}>{p.reticulocyte_pct}%</td>
                              <td style={{ padding: '3px 8px', color: p.ldh_u_l > 600 ? '#ef4444' : '#fbbf24' }}>{p.ldh_u_l}</td>
                              <td style={{ padding: '3px 8px', color: p.bilirubin_umol_l > 80 ? '#ef4444' : '#fbbf24' }}>{p.bilirubin_umol_l}</td>
                              <td style={{ padding: '3px 8px', color: p.enzyme_activity_pct_normal < 15 ? '#ef4444' : '#fbbf24' }}>{p.enzyme_activity_pct_normal}%</td>
                              <td style={{ padding: '3px 8px', color: p.splenomegaly ? '#ef4444' : '#475569' }}>{p.splenomegaly ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px', color: p.neuro_involvement ? '#ef4444' : '#475569' }}>{p.neuro_involvement ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px', color: p.myopathy ? '#ef4444' : '#475569' }}>{p.myopathy ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px', color: p.transfusion_dependent ? '#ef4444' : '#475569' }}>{p.transfusion_dependent ? 'Yes' : 'No'}</td>
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
              Glossary — CNSHA / Favism / Heinz Bodies / 2,3-DPG Paradox / Aplastic Crisis / DHAP Toxicity / Mitapivat / Ischemic Forearm Test / No Second Wind / G6PD Assay Timing
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
