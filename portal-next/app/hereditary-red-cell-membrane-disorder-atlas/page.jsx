'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-red-cell-membrane-disorder-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ANK1':   '#b71c1c',  // deep red     — most common HS, ankyrin-spectrin anchor
  'SPTA1':  '#1565c0',  // deep blue    — HPP/HE, LELY modifier
  'SPTB':   '#2e7d32',  // dark green   — HE2/HS2, spectrin beta
  'SLC4A1': '#6a1b9a',  // deep purple  — Band 3, SAO malaria protection, dRTA
  'EPB42':  '#e65100',  // deep orange  — Protein 4.2, HS5, Japanese/Mediterranean
  'EPB41':  '#00695c',  // dark teal    — Protein 4.1R, HE1, SE Asian, GPC
  'PIEZO1': '#c62828',  // crimson      — DHS1, SPLENECTOMY CI, fatal thrombosis
  'KCNN4':  '#4527a0',  // deep indigo  — Gardos channel DHS2, senicapoc direct target
};

const GENE_INFO = {
  'ANK1':   { full: 'ANK1 / Ankyrin-1 / 1881aa', locus: '8p11.21', size: '1881 aa / 206 kDa (anchors spectrin to Band 3; most common HS ~60-65%; microspherocytes; splenectomy curative)', inh: 'AD' },
  'SPTA1':  { full: 'SPTA1 / Spectrin alpha-1 / 2429aa', locus: '1q23.1', size: '2429 aa / 281 kDa (biallelic → HPP; LELY modifier; heat sensitivity 45-46°C pathognomonic HPP)', inh: 'AR/AD' },
  'SPTB':   { full: 'SPTB / Spectrin beta / 2137aa', locus: '14q23.3', size: '2137 aa / 246 kDa (HE2 most common; AD elliptocytosis; self-association domain mutations → impaired tetramer)', inh: 'AD' },
  'SLC4A1': { full: 'SLC4A1 / Band 3 (AE1) / 911aa', locus: '17q21.31', size: '911 aa / 102 kDa (HS3 AD; SAO 27-bp del; AR dRTA-HA; CO2/Cl-/HCO3- exchanger; SAO EMA INCREASED)', inh: 'AD/AR' },
  'EPB42':  { full: 'EPB42 / Protein 4.2 / 721aa', locus: '15q15.2', size: '721 aa / 77 kDa (stabilises AE1-ankyrin; HS5 AR; absent band 4.2 SDS-PAGE diagnostic; Japanese p.Ala142Thr)', inh: 'AR' },
  'EPB41':  { full: 'EPB41 / Protein 4.1R / 823aa', locus: '1p35.3', size: '823 aa / 80 kDa (junctional complex; HE1 AD; SE Asian founder; GPC reduced; EBA-140 malaria invasion blocked)', inh: 'AD' },
  'PIEZO1': { full: 'PIEZO1 / Mechanosensitive channel / 2521aa', locus: '16q24.3', size: '2521 aa / 286 kDa (DHS1 GOF AD; SPLENECTOMY CI — fatal thrombosis; stomatocytes; MCHC >36; senicapoc)', inh: 'AD GOF' },
  'KCNN4':  { full: 'KCNN4 / Gardos channel (KCa3.1) / 427aa', locus: '19q13.31', size: '427 aa / 47 kDa (DHS2 GOF AD; Gardos channel; SPLENECTOMY CI; senicapoc direct target; pseudohyperkalemia)', inh: 'AD GOF' },
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

export default function HereditaryRedCellMembraneDisorderAtlas() {
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
      Loading Red Cell Membrane Disorder Atlas…
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
      <div style={{ background: 'linear-gradient(135deg, #1a0a0a 0%, #0f172a 100%)', borderBottom: '1px solid #3d1515', padding: '20px 28px 14px' }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: '#fca5a5', letterSpacing: 1 }}>
          🧬 Hereditary Red Cell Membrane Disorder Atlas
        </div>
        <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>
          Complete 8-Gene HS / HE / DHS Reference — ANK1 · SPTA1 · SPTB · SLC4A1 · EPB42 · EPB41 · PIEZO1 · KCNN4
        </div>
        <div style={{ marginTop: 6, padding: '6px 10px', background: '#7f1d1d', borderRadius: 4, display: 'inline-block', fontSize: 11, color: '#fca5a5', fontWeight: 700 }}>
          ⚠ SPLENECTOMY ABSOLUTELY CONTRAINDICATED in PIEZO1-DHS1 + KCNN4-DHS2 — fatal thrombosis risk
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
              background: tab === t ? '#fca5a5' : '#1e293b',
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
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2846-2853" />
              <MetricCard label="Genes Covered" value={overview.genes?.length} sub="HS / HE / DHS / SAO disorders" />
              <MetricCard label="Inheritance Types" value="AD/AR/AR-AR" sub="AD (6) + AR (2: EPB42, SPTA1-HPP)" />
              <MetricCard label="Disorder Classes" value="3" sub="HS (vertical) / HE (horizontal) / DHS (cation)" />
              <MetricCard label="Splenectomy CI" value="PIEZO1+KCNN4" sub="Fatal thrombosis — absolute CI" warn />
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#fca5a5', marginBottom: 10 }}>
              Gene Summary — Hb, Retic, LDH, MCHC, EMA%, Splenomegaly%
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene','Locus','Inh','Pts','Median Hb (g/dL)','Mean Retic %','Mean LDH (U/L)','Mean MCHC','EMA % ctrl','Splenomegaly %','RBC Dehydrated %','Transfusion %'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', color: '#fca5a5', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
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
                      <td style={{ padding: '5px 8px', color: g.mean_mchc_g_dl > 36 ? '#ef4444' : '#fbbf24' }}>
                        {g.mean_mchc_g_dl}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.mean_ema_pct > 100 ? '#34d399' : g.mean_ema_pct < 80 ? '#ef4444' : '#fbbf24' }}>
                        {g.mean_ema_pct}%
                        {g.mean_ema_pct > 100 ? ' ↑SAO' : ''}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_splenomegaly > 70 ? '#ef4444' : '#fbbf24' }}>
                        {g.pct_splenomegaly}%
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_dehydrated > 0 ? '#f97316' : '#475569' }}>
                        {g.pct_dehydrated > 0 ? `${g.pct_dehydrated}% ⚠` : '—'}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_transfusion_dependent > 20 ? '#ef4444' : '#fbbf24' }}>
                        {g.pct_transfusion_dependent}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#fca5a5', marginTop: 24, marginBottom: 10 }}>
              Disorder Categories
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 12 }}>
              {overview.disorder_categories?.map((cat, i) => (
                <div key={i} style={{ background: '#1e293b', border: `1px solid ${i === 2 ? '#7f1d1d' : '#334155'}`, borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, color: i === 2 ? '#fca5a5' : '#fbbf24', fontSize: 12, marginBottom: 6 }}>{cat.category}</div>
                  <div style={{ marginBottom: 6, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                    {cat.genes?.map(g => <GeneChip key={g} gene={g} active={null} />)}
                  </div>
                  <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{cat.note}</div>
                </div>
              ))}
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#fca5a5', marginTop: 24, marginBottom: 10 }}>
              Critical Distinctions
            </div>
            {overview.critical_distinctions?.map((d, i) => (
              <div key={i} style={{
                background: '#1e293b',
                border: `1px solid ${d.includes('SPLENECTOMY') || d.includes('fatal') ? '#7f1d1d' : '#334155'}`,
                borderRadius: 6, padding: '8px 12px', marginBottom: 6,
                fontSize: 11, color: '#cbd5e1', lineHeight: 1.6
              }}>
                {d.includes('SPLENECTOMY') || d.includes('fatal') ? '🚫 ' : '⚡ '}{d}
              </div>
            ))}
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && (
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#fca5a5', marginBottom: 14 }}>
              Gene Reference — Protein, Locus, Inheritance
            </div>
            {geneList.filter(g => !activeGene || g === activeGene).map(g => (
              <div key={g} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[g]}44`, borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <GeneChip gene={g} active={null} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{GENE_INFO[g]?.locus}</span>
                  <span style={{ color: '#fbbf24', fontSize: 12, fontWeight: 700 }}>{GENE_INFO[g]?.inh}</span>
                  {(g === 'PIEZO1' || g === 'KCNN4') && (
                    <span style={{ background: '#7f1d1d', color: '#fca5a5', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      SPLENECTOMY CI
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
                  {(g.gene === 'PIEZO1' || g.gene === 'KCNN4') && (
                    <span style={{ background: '#7f1d1d', color: '#fca5a5', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      ⚠ SPLENECTOMY ABSOLUTELY CONTRAINDICATED
                    </span>
                  )}
                </div>
                {[['Protein', g.protein_size], ['Inheritance', g.inheritance], ['Disease', g.disease_category],
                  ['Pathway', g.disease_pathway], ['Pathognomonic', g.pathognomonic], ['Treatment', g.treatment]].map(([label, val]) => (
                  <div key={label} style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, color: '#fca5a5', fontWeight: 700, marginBottom: 2 }}>{label.toUpperCase()}</div>
                    <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{val}</div>
                  </div>
                ))}
                {g.patients && g.patients.length > 0 && (
                  <div>
                    <div style={{ fontSize: 11, color: '#fca5a5', fontWeight: 700, marginBottom: 6 }}>SAMPLE PATIENTS (first 5)</div>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                        <thead>
                          <tr style={{ background: '#0f172a' }}>
                            {['ID','Sex','Age Dx','Hb g/dL','Retic %','LDH U/L','Bili µmol/L','MCHC g/dL','Osm Frag','EMA %ctrl','Morphology','Spleen','Gallstones','Transfusion'].map(h => (
                              <th key={h} style={{ padding: '4px 8px', color: '#64748b', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {g.patients.map((p, i) => (
                            <tr key={p.patient_id} style={{ background: i % 2 === 0 ? '#1a2332' : '#0f172a' }}>
                              <td style={{ padding: '3px 8px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{p.patient_id}</td>
                              <td style={{ padding: '3px 8px' }}>{p.sex}</td>
                              <td style={{ padding: '3px 8px' }}>{p.age_at_diagnosis_years}y</td>
                              <td style={{ padding: '3px 8px', color: p.hemoglobin_g_dl < 9 ? '#ef4444' : '#fbbf24' }}>{p.hemoglobin_g_dl}</td>
                              <td style={{ padding: '3px 8px', color: p.reticulocyte_pct > 15 ? '#ef4444' : '#fbbf24' }}>{p.reticulocyte_pct}%</td>
                              <td style={{ padding: '3px 8px', color: p.ldh_u_l > 600 ? '#ef4444' : '#fbbf24' }}>{p.ldh_u_l}</td>
                              <td style={{ padding: '3px 8px', color: p.bilirubin_umol_l > 80 ? '#ef4444' : '#fbbf24' }}>{p.bilirubin_umol_l}</td>
                              <td style={{ padding: '3px 8px', color: p.mchc_g_dl > 36 ? '#ef4444' : '#fbbf24' }}>{p.mchc_g_dl}</td>
                              <td style={{ padding: '3px 8px', color: p.osmotic_fragility === 'increased' ? '#ef4444' : p.osmotic_fragility === 'decreased' ? '#34d399' : '#94a3b8' }}>
                                {p.osmotic_fragility}
                              </td>
                              <td style={{ padding: '3px 8px', color: p.ema_binding_pct_control > 100 ? '#34d399' : p.ema_binding_pct_control < 80 ? '#ef4444' : '#fbbf24' }}>
                                {p.ema_binding_pct_control}%
                              </td>
                              <td style={{ padding: '3px 8px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{p.morphology}</td>
                              <td style={{ padding: '3px 8px', color: p.splenomegaly ? '#ef4444' : '#475569' }}>{p.splenomegaly ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px', color: p.gallstones ? '#fbbf24' : '#475569' }}>{p.gallstones ? 'Yes' : 'No'}</td>
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
            <div style={{ fontSize: 14, fontWeight: 700, color: '#fca5a5', marginBottom: 14 }}>
              Glossary — HS/EMA/Osmotic-Fragility / HPP-Heat-Sensitivity / LELY / DHS-Pseudohyperkalemia / Splenectomy-CI / SAO / Senicapoc / Gardos
            </div>
            {defs.definitions?.map((d, i) => (
              <div key={i} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: '12px 16px', marginBottom: 10 }}>
                <div style={{ fontWeight: 700, color: '#fbbf24', fontSize: 13, marginBottom: 6 }}>{d.term}</div>
                <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{d.definition}</div>
              </div>
            ))}
            <div style={{ fontSize: 14, fontWeight: 700, color: '#fca5a5', marginTop: 20, marginBottom: 10 }}>
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
