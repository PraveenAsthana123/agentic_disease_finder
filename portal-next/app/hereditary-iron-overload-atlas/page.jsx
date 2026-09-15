'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-iron-overload-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'HFE':     '#b71c1c',  // deep red      — classic HH, C282Y/H63D, most common
  'HJV':     '#1565c0',  // deep blue     — juvenile HH type 2A, BMP co-receptor
  'HAMP':    '#2e7d32',  // dark green    — hepcidin, most severe juvenile HH
  'TFR2':    '#6a1b9a',  // deep purple   — HH type 3, TFR2 sensor
  'SLC40A1': '#e65100',  // deep orange   — ferroportin disease, type 4A/4B
  'CP':      '#00695c',  // dark teal     — aceruloplasminemia, brain iron, neurological
  'TMPRSS6': '#c62828',  // crimson       — IRIDA, excess hepcidin, oral iron fails
  'BMP6':    '#4527a0',  // deep indigo   — HH type 5, BMP6 liver-specific signal
};

const GENE_INFO = {
  'HFE':     { full: 'HFE / HFE Protein / 343aa', locus: '6p21.3', size: '343 aa / 28 kDa (MHC class I-like; C282Y/C282Y = 80-90% of HH; C282Y/H63D compound het; TS >45% fasting = screening test; phlebotomy curative)', inh: 'AR' },
  'HJV':     { full: 'HJV / Hemojuvelin / 426aa', locus: '1q21.1', size: '426 aa / 50 kDa (BMP co-receptor; juvenile HH type 2A; onset teens-20s; cardiomyopathy + hypogonadism dominate; most severe HH overall)', inh: 'AR' },
  'HAMP':    { full: 'HAMP / Hepcidin / 84aa', locus: '19q13.12', size: '84 aa / 9 kDa (hepcidin master hormone; juvenile HH type 2B; highest ferritins seen in haematology; most severe form of HH)', inh: 'AR' },
  'TFR2':    { full: 'TFR2 / Transferrin Receptor 2 / 801aa', locus: '7q22.1', size: '801 aa / 89 kDa (iron sensor; HH type 3; adult onset milder than HJV/HAMP; same phenotype as HFE but no C282Y)', inh: 'AR' },
  'SLC40A1': { full: 'SLC40A1 / Ferroportin / 571aa', locus: '2q32.2', size: '571 aa / 62 kDa (iron exporter; AD; Type 4A = macrophage iron trap — HIGH FERRITIN, LOW TS; Type 4B GOF = classical HH pattern)', inh: 'AD' },
  'CP':      { full: 'CP / Ceruloplasmin / 1065aa', locus: '3q24', size: '1065 aa / 122 kDa (ferroxidase; aceruloplasminemia; LOW serum iron + LOW TS despite MASSIVE brain/liver iron overload; neurodegeneration)', inh: 'AR' },
  'TMPRSS6': { full: 'TMPRSS6 / Matriptase-2 / 811aa', locus: '22q12.3', size: '811 aa / 90 kDa (IRIDA; cleaves hemojuvelin → hepcidin suppressed normally; loss → excess hepcidin → oral iron futile; IV iron works)', inh: 'AR' },
  'BMP6':    { full: 'BMP6 / Bone Morphogenetic Protein 6 / 513aa', locus: '6p24.3', size: '513 aa / 60 kDa (liver-specific BMP6 activates SMAD → hepcidin; HH type 5 adult-onset; phenotype milder than HJV/HAMP)', inh: 'AR' },
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

export default function HereditaryIronOverloadAtlas() {
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
      Loading Iron Overload Atlas…
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
      <div style={{ background: 'linear-gradient(135deg, #1a0a00 0%, #0f172a 100%)', borderBottom: '1px solid #7c2d12', padding: '20px 28px 14px' }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: '#fb923c', letterSpacing: 1 }}>
          🧬 Hereditary Iron Overload Atlas
        </div>
        <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>
          Complete 8-Gene Hereditary Haemochromatosis / Ferroportin Disease / Aceruloplasminemia / IRIDA Reference — HFE · HJV · HAMP · TFR2 · SLC40A1 · CP · TMPRSS6 · BMP6
        </div>
        <div style={{ marginTop: 6, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <div style={{ padding: '5px 10px', background: '#7c2d12', borderRadius: 4, display: 'inline-block', fontSize: 11, color: '#fb923c', fontWeight: 700 }}>
            ⚡ SLC40A1-Type4A: HIGH FERRITIN + LOW TS — macrophage iron trap (OPPOSITE of classic HH)
          </div>
          <div style={{ padding: '5px 10px', background: '#1e3a5f', borderRadius: 4, display: 'inline-block', fontSize: 11, color: '#7dd3fc', fontWeight: 700 }}>
            ⚠ TMPRSS6-IRIDA: ORAL IRON FAILS — parenteral IV iron mandatory
          </div>
          <div style={{ padding: '5px 10px', background: '#1a2e1a', borderRadius: 4, display: 'inline-block', fontSize: 11, color: '#86efac', fontWeight: 700 }}>
            ⚠ CP-Aceruloplasminemia: LOW serum Fe + LOW TS despite MASSIVE brain iron overload
          </div>
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
              background: tab === t ? '#fb923c' : '#1e293b',
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
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2854-2861" />
              <MetricCard label="Genes Covered" value={overview.genes?.length} sub="HH / Ferroportin / IRIDA / Aceruloplasminemia" />
              <MetricCard label="Inheritance" value="AR (7) + AD (1)" sub="SLC40A1 is the only AD gene" />
              <MetricCard label="Disorder Classes" value="4" sub="Classic HH / Ferroportin / CP / IRIDA" />
              <MetricCard label="Oral Iron Fails" value="TMPRSS6" sub="IRIDA — IV iron mandatory" warn />
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#fb923c', marginBottom: 10 }}>
              Gene Summary — Ferritin, Transferrin Sat%, Liver Iron Concentration, Organ Involvement
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene','Locus','Inh','Pts','Median Ferritin (µg/L)','Mean TS%','Mean LIC (mg Fe/g)','Cirrhosis%','Cardiac%','Diabetes%','Hypogonadism%','Neurological%'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', color: '#fb923c', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
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
                      <td style={{ padding: '5px 8px', color: g.median_ferritin_ug_l > 5000 ? '#ef4444' : g.median_ferritin_ug_l > 1000 ? '#f97316' : '#fbbf24' }}>
                        {g.median_ferritin_ug_l?.toLocaleString()}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.mean_transferrin_saturation_pct > 75 ? '#ef4444' : g.mean_transferrin_saturation_pct < 20 ? '#34d399' : '#fbbf24' }}>
                        {g.mean_transferrin_saturation_pct}%
                        {g.mean_transferrin_saturation_pct < 20 ? ' ↓IRIDA' : ''}
                        {g.gene === 'SLC40A1' && g.mean_transferrin_saturation_pct < 40 ? ' ↓Type4A' : ''}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.mean_liver_iron_concentration > 30 ? '#ef4444' : g.mean_liver_iron_concentration > 15 ? '#f97316' : '#fbbf24' }}>
                        {g.mean_liver_iron_concentration}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_cirrhosis > 20 ? '#ef4444' : '#fbbf24' }}>
                        {g.pct_cirrhosis}%
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_cardiac > 30 ? '#ef4444' : '#94a3b8' }}>
                        {g.pct_cardiac > 0 ? `${g.pct_cardiac}%` : '—'}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_diabetes > 30 ? '#ef4444' : '#94a3b8' }}>
                        {g.pct_diabetes > 0 ? `${g.pct_diabetes}%` : '—'}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_hypogonadism > 30 ? '#ef4444' : '#94a3b8' }}>
                        {g.pct_hypogonadism > 0 ? `${g.pct_hypogonadism}%` : '—'}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_neurological > 0 ? '#ef4444' : '#475569' }}>
                        {g.pct_neurological > 0 ? `${g.pct_neurological}% ⚠` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#fb923c', marginTop: 24, marginBottom: 10 }}>
              Disorder Categories
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 12 }}>
              {overview.disorder_categories?.map((cat, i) => (
                <div key={i} style={{ background: '#1e293b', border: `1px solid ${i === 2 ? '#14532d' : i === 3 ? '#1e3a5f' : '#334155'}`, borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, color: i === 2 ? '#86efac' : i === 3 ? '#7dd3fc' : '#fbbf24', fontSize: 12, marginBottom: 6 }}>{cat.category}</div>
                  <div style={{ marginBottom: 6, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                    {cat.genes?.map(g => <GeneChip key={g} gene={g} active={null} />)}
                  </div>
                  <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{cat.note}</div>
                </div>
              ))}
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#fb923c', marginTop: 24, marginBottom: 10 }}>
              Critical Distinctions
            </div>
            {overview.critical_distinctions?.map((d, i) => (
              <div key={i} style={{
                background: '#1e293b',
                border: `1px solid ${d.includes('IRIDA') || d.includes('ORAL IRON') ? '#1e3a5f' : d.includes('ACERULOPLASMINEMIA') || d.includes('CP') ? '#14532d' : '#334155'}`,
                borderRadius: 6, padding: '8px 12px', marginBottom: 6,
                fontSize: 11, color: '#cbd5e1', lineHeight: 1.6
              }}>
                {d.includes('IRIDA') || d.includes('ORAL IRON') ? '💉 ' : d.includes('ACERULOPLASMINEMIA') || d.includes('CP') ? '🧠 ' : '⚡ '}{d}
              </div>
            ))}
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && (
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#fb923c', marginBottom: 14 }}>
              Gene Reference — Protein, Locus, Inheritance, Clinical Role
            </div>
            {geneList.filter(g => !activeGene || g === activeGene).map(g => (
              <div key={g} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[g]}44`, borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
                  <GeneChip gene={g} active={null} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{GENE_INFO[g]?.locus}</span>
                  <span style={{ color: '#fbbf24', fontSize: 12, fontWeight: 700 }}>{GENE_INFO[g]?.inh}</span>
                  {g === 'TMPRSS6' && (
                    <span style={{ background: '#1e3a5f', color: '#7dd3fc', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      IRIDA — IV IRON ONLY
                    </span>
                  )}
                  {g === 'CP' && (
                    <span style={{ background: '#14532d', color: '#86efac', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      NEURODEGENERATION
                    </span>
                  )}
                  {g === 'SLC40A1' && (
                    <span style={{ background: '#7c2d12', color: '#fb923c', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      AD — TYPE4A LOW TS
                    </span>
                  )}
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
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' }}>
                  <GeneChip gene={g.gene} active={null} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus}</span>
                  <span style={{ color: '#fbbf24', fontSize: 11 }}>{GENE_INFO[g.gene]?.inh}</span>
                  <span style={{ color: '#64748b', fontSize: 11 }}>{g.n_patients} patients</span>
                  {g.gene === 'TMPRSS6' && (
                    <span style={{ background: '#1e3a5f', color: '#7dd3fc', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      ⚠ ORAL IRON FUTILE — IV IRON MANDATORY
                    </span>
                  )}
                  {g.gene === 'CP' && (
                    <span style={{ background: '#14532d', color: '#86efac', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      ⚠ LOW Fe + LOW TS DESPITE IRON OVERLOAD
                    </span>
                  )}
                </div>
                {[['Protein', g.protein_size], ['Inheritance', g.inheritance], ['Disease', g.disease_category],
                  ['Pathway', g.disease_pathway], ['Pathognomonic', g.pathognomonic], ['Treatment', g.treatment]].map(([label, val]) => (
                  <div key={label} style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, color: '#fb923c', fontWeight: 700, marginBottom: 2 }}>{label.toUpperCase()}</div>
                    <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{val}</div>
                  </div>
                ))}
                {g.patients && g.patients.length > 0 && (
                  <div>
                    <div style={{ fontSize: 11, color: '#fb923c', fontWeight: 700, marginBottom: 6 }}>SAMPLE PATIENTS (first 5)</div>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                        <thead>
                          <tr style={{ background: '#0f172a' }}>
                            {['ID','Sex','Age Dx','Hb g/dL','Ferritin µg/L','TS%','LIC mg/g','Fibrosis','Cardiac','Hypogonadism','Diabetes','Neuro','Phlebotomy Units','Cirrhosis'].map(h => (
                              <th key={h} style={{ padding: '4px 8px', color: '#64748b', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {g.patients.slice(0, 5).map((p, i) => (
                            <tr key={p.patient_id} style={{ background: i % 2 === 0 ? '#1a2332' : '#0f172a' }}>
                              <td style={{ padding: '3px 8px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{p.patient_id}</td>
                              <td style={{ padding: '3px 8px' }}>{p.sex}</td>
                              <td style={{ padding: '3px 8px' }}>{p.age_at_diagnosis_years}y</td>
                              <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{p.hemoglobin_g_dl}</td>
                              <td style={{ padding: '3px 8px', color: p.serum_ferritin_ug_l > 5000 ? '#ef4444' : p.serum_ferritin_ug_l > 1000 ? '#f97316' : '#fbbf24' }}>
                                {p.serum_ferritin_ug_l?.toLocaleString()}
                              </td>
                              <td style={{ padding: '3px 8px', color: p.transferrin_saturation_pct > 75 ? '#ef4444' : p.transferrin_saturation_pct < 20 ? '#34d399' : '#fbbf24' }}>
                                {p.transferrin_saturation_pct}%
                              </td>
                              <td style={{ padding: '3px 8px', color: p.liver_iron_concentration > 30 ? '#ef4444' : p.liver_iron_concentration > 15 ? '#f97316' : '#fbbf24' }}>
                                {p.liver_iron_concentration}
                              </td>
                              <td style={{ padding: '3px 8px', color: p.fibrosis_stage >= 3 ? '#ef4444' : p.fibrosis_stage >= 1 ? '#fbbf24' : '#475569' }}>
                                F{p.fibrosis_stage}
                              </td>
                              <td style={{ padding: '3px 8px', color: p.cardiac_involvement ? '#ef4444' : '#475569' }}>{p.cardiac_involvement ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px', color: p.hypogonadism ? '#ef4444' : '#475569' }}>{p.hypogonadism ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px', color: p.diabetes ? '#fbbf24' : '#475569' }}>{p.diabetes ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px', color: p.neurological_involvement ? '#ef4444' : '#475569' }}>{p.neurological_involvement ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{p.phlebotomy_units_removed}</td>
                              <td style={{ padding: '3px 8px', color: p.cirrhosis ? '#ef4444' : '#475569' }}>{p.cirrhosis ? 'Yes' : 'No'}</td>
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
            <div style={{ fontSize: 14, fontWeight: 700, color: '#fb923c', marginBottom: 14 }}>
              Glossary — Hepcidin-BMP-SMAD / Transferrin Saturation / C282Y-H63D Penetrance / Phlebotomy Protocol / Ferroportin Type4A-4B / IRIDA / Aceruloplasminemia / Juvenile HH
            </div>
            {defs.definitions?.map((d, i) => (
              <div key={i} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: '12px 16px', marginBottom: 10 }}>
                <div style={{ fontWeight: 700, color: '#fbbf24', fontSize: 13, marginBottom: 6 }}>{d.term}</div>
                <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{d.definition}</div>
              </div>
            ))}
            <div style={{ fontSize: 14, fontWeight: 700, color: '#fb923c', marginTop: 20, marginBottom: 10 }}>
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
