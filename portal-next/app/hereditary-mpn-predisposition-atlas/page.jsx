'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-mpn-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'JAK2':   '#b71c1c',  // deep red      — V617F driver, PV/ET/PMF
  'CALR':   '#1565c0',  // deep blue     — CALR del52/ins5, ET/PMF
  'MPL':    '#2e7d32',  // dark green    — MPL W515, ET/PMF
  'SH2B3':  '#6a1b9a',  // deep purple   — LNK, germline predisposition
  'EPOR':   '#e65100',  // deep orange   — familial erythrocytosis type 1
  'VHL':    '#00695c',  // dark teal     — Chuvash / VHL syndrome
  'EPAS1':  '#c62828',  // crimson       — HIF-2α, ECYT4 + paraganglioma
  'THPO':   '#4527a0',  // deep indigo   — familial ET, elevated TPO
};

const GENE_INFO = {
  'JAK2':  { full: 'JAK2 / Janus Kinase 2 / 1132aa', locus: '9p24.1', size: '1132 aa / 130 kDa (V617F somatic: PV 95%, ET 55%, PMF 65%; 46/1 haplotype germline predisposition; aquagenic pruritus pathognomonic in PV; ruxolitinib first-line PMF)', inh: 'AD somatic' },
  'CALR':  { full: 'CALR / Calreticulin / 400aa', locus: '19p13.2', size: '400 aa / 46 kDa (ER chaperone; exon 9 frameshift; Type 1 del52 = MF risk; Type 2 ins5 = benign ET; activates MPL constitutively; neoantigen therapeutic target)', inh: 'AD somatic' },
  'MPL':   { full: 'MPL / Thrombopoietin Receptor / 635aa', locus: '1p34.2', size: '635 aa / 75 kDa (TPOR; W515L/K somatic ET/PMF; P106L germline familial ET; TPO elevated; CAMT when LOF; TPO-mimetics CONTRAINDICATED)', inh: 'AD somatic/germline' },
  'SH2B3': { full: 'SH2B3 / LNK / 575aa', locus: '12q24.12', size: '575 aa / 68 kDa (LNK; JAK2 negative regulator; germline LOF → MPN predisposition; autoimmune co-morbidities: T1D, coeliac, RA; somatic biallelic → blast transformation)', inh: 'AD germline' },
  'EPOR':  { full: 'EPOR / Erythropoietin Receptor / 508aa', locus: '19p13.2', size: '508 aa / 55 kDa (C-terminal truncation → removes SHP-1/SOCS3 docking → EPO hypersensitivity; EPO SUPPRESSED like PV but JAK2 NEGATIVE; pure erythrocytosis; AD familial)', inh: 'AD' },
  'VHL':   { full: 'VHL / Von Hippel-Lindau / 213aa', locus: '3p25.3', size: '213 aa / 24 kDa (HIF-1/2α E3 ubiquitin ligase; Chuvash p.R200W AR = erythrocytosis + elevated EPO; VHL syndrome AD = RCC + haemangioblastoma + phaeochromocytoma; belzutifan FDA2021)', inh: 'AD/AR' },
  'EPAS1': { full: 'EPAS1 / HIF-2α / 870aa', locus: '2p21', size: '870 aa / 97 kDa (HIF-2α GOF → resists PHD2 hydroxylation → EPO elevated; ECYT4; paraganglioma-polycythaemia somatic mosaic; EPAS1 + VHL = EPO elevated, JAK2 negative; belzutifan target)', inh: 'AD GOF' },
  'THPO':  { full: 'THPO / Thrombopoietin / 353aa', locus: '3q27.1', size: '353 aa / 35 kDa (5-UTR mutations → uORF disruption → excess TPO; familial ET; serum THPO ELEVATED; JAK2/CALR/MPL all NEGATIVE; standard NGS MISSES 5-UTR — request specifically)', inh: 'AD' },
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

export default function HereditaryMPNPredispositionAtlas() {
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
      Loading MPN Predisposition Atlas…
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
      <div style={{ background: 'linear-gradient(135deg, #1a0020 0%, #0f172a 100%)', borderBottom: '1px solid #6b21a8', padding: '20px 28px 14px' }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: '#c084fc', letterSpacing: 1 }}>
          🧬 Hereditary MPN Predisposition Atlas
        </div>
        <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>
          Complete 8-Gene Hereditary MPN / Familial Erythrocytosis / Familial ET Reference — JAK2 · CALR · MPL · SH2B3 · EPOR · VHL · EPAS1 · THPO
        </div>
        <div style={{ marginTop: 6, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <div style={{ padding: '5px 10px', background: '#4c1d95', borderRadius: 4, display: 'inline-block', fontSize: 11, color: '#c084fc', fontWeight: 700 }}>
            ⚡ JAK2 V617F: AQUAGENIC PRURITUS = PV-PATHOGNOMONIC (absent in secondary erythrocytosis)
          </div>
          <div style={{ padding: '5px 10px', background: '#1e3a5f', borderRadius: 4, display: 'inline-block', fontSize: 11, color: '#7dd3fc', fontWeight: 700 }}>
            ⚠ EPOR / VHL / EPAS1: EPO pattern distinguishes — EPOR suppressed; VHL/EPAS1 elevated
          </div>
          <div style={{ padding: '5px 10px', background: '#1a2e1a', borderRadius: 4, display: 'inline-block', fontSize: 11, color: '#86efac', fontWeight: 700 }}>
            ⚠ THPO 5-UTR: standard NGS MISSES these mutations — request specifically
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
              background: tab === t ? '#c084fc' : '#1e293b',
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
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2862-2869" />
              <MetricCard label="Genes Covered" value={overview.genes?.length} sub="MPN Drivers / Predisposition / Erythrocytosis / Familial ET" />
              <MetricCard label="MPN Subtypes" value="PV/ET/PMF" sub="+ Familial Erythrocytosis Types 1/3/4" />
              <MetricCard label="Disorder Classes" value="5" sub="Clonal / Predisposition / Low-EPO / High-EPO / Familial-ET" />
              <MetricCard label="JAK Inhibitor" value="Ruxolitinib" sub="First-line PMF; PV-HU-refractory" warn={false} />
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#c084fc', marginBottom: 10 }}>
              Gene Summary — Hb, Platelets, EPO, Thrombosis%, Aquagenic Pruritus%, Spleen
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene','Locus','Inh','Pts','Mean Hb (g/dL)','Mean Plt (×10⁹)','Mean EPO (mIU/mL)','Thrombosis%','Aquagenic%','Spleen cm'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', color: '#c084fc', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
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
                      <td style={{ padding: '5px 8px', color: '#fbbf24', fontSize: 10 }}>{GENE_INFO[g.gene]?.inh}</td>
                      <td style={{ padding: '5px 8px' }}>{g.n_patients}</td>
                      <td style={{ padding: '5px 8px', color: g.mean_hb_g_dl > 17 ? '#ef4444' : '#fbbf24' }}>
                        {g.mean_hb_g_dl}
                        {g.mean_hb_g_dl > 17 ? ' ↑' : ''}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.mean_platelets_per_nl > 600 ? '#ef4444' : g.mean_platelets_per_nl > 450 ? '#f97316' : '#94a3b8' }}>
                        {g.mean_platelets_per_nl?.toLocaleString()}
                        {g.mean_platelets_per_nl > 600 ? ' ↑↑' : ''}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.mean_epo_miu_ml < 6 ? '#ef4444' : g.mean_epo_miu_ml > 30 ? '#34d399' : '#fbbf24' }}>
                        {g.mean_epo_miu_ml}
                        {g.mean_epo_miu_ml < 6 ? ' ↓LOW' : g.mean_epo_miu_ml > 30 ? ' ↑HIGH' : ''}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_thrombosis > 25 ? '#ef4444' : '#fbbf24' }}>
                        {g.pct_thrombosis}%
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_aquagenic > 0 ? '#ef4444' : '#475569' }}>
                        {g.pct_aquagenic > 0 ? `${g.pct_aquagenic}% ⚡` : '—'}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.mean_spleen_cm > 15 ? '#ef4444' : g.mean_spleen_cm > 12 ? '#f97316' : '#94a3b8' }}>
                        {g.mean_spleen_cm} cm
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#c084fc', marginTop: 24, marginBottom: 10 }}>
              Disorder Categories
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 12 }}>
              {overview.disorder_categories?.map((cat, i) => (
                <div key={i} style={{ background: '#1e293b', border: `1px solid ${i === 0 ? '#7c2d12' : i === 1 ? '#4c1d95' : i === 2 ? '#1e3a5f' : i === 3 ? '#14532d' : '#334155'}`, borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, color: i === 0 ? '#fb923c' : i === 1 ? '#c084fc' : i === 2 ? '#7dd3fc' : i === 3 ? '#86efac' : '#fbbf24', fontSize: 12, marginBottom: 6 }}>{cat.category}</div>
                  <div style={{ marginBottom: 6, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                    {cat.genes?.map(g => <GeneChip key={g} gene={g} active={null} />)}
                  </div>
                  <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{cat.note}</div>
                </div>
              ))}
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#c084fc', marginTop: 24, marginBottom: 10 }}>
              Critical Distinctions
            </div>
            {overview.critical_distinctions?.map((d, i) => (
              <div key={i} style={{
                background: '#1e293b',
                border: `1px solid ${d.includes('AQUAGENIC') ? '#7c2d12' : d.includes('THPO') || d.includes('5-UTR') ? '#4c1d95' : d.includes('RUXOLITINIB') ? '#1e3a5f' : '#334155'}`,
                borderRadius: 6, padding: '8px 12px', marginBottom: 6,
                fontSize: 11, color: '#cbd5e1', lineHeight: 1.6
              }}>
                {d.includes('AQUAGENIC') ? '⚡ ' : d.includes('THPO') || d.includes('5-UTR') ? '🔬 ' : d.includes('RUXOLITINIB') ? '⚠ ' : '▶ '}{d}
              </div>
            ))}
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && (
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#c084fc', marginBottom: 14 }}>
              Gene Reference — Protein, Locus, Inheritance, Clinical Role
            </div>
            {geneList.filter(g => !activeGene || g === activeGene).map(g => (
              <div key={g} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[g]}44`, borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
                  <GeneChip gene={g} active={null} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{GENE_INFO[g]?.locus}</span>
                  <span style={{ color: '#fbbf24', fontSize: 12, fontWeight: 700 }}>{GENE_INFO[g]?.inh}</span>
                  {g === 'JAK2' && (
                    <span style={{ background: '#7c2d12', color: '#fb923c', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      V617F PATHOGNOMONIC — AQUAGENIC PRURITUS
                    </span>
                  )}
                  {g === 'THPO' && (
                    <span style={{ background: '#4c1d95', color: '#c084fc', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      5-UTR — STANDARD NGS MISSES
                    </span>
                  )}
                  {(g === 'VHL' || g === 'EPAS1') && (
                    <span style={{ background: '#14532d', color: '#86efac', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      EPO ELEVATED — HIF PATHWAY
                    </span>
                  )}
                  {g === 'EPOR' && (
                    <span style={{ background: '#1e3a5f', color: '#7dd3fc', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      EPO SUPPRESSED — JAK2 NEGATIVE
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
                  {g.gene === 'JAK2' && (
                    <span style={{ background: '#7c2d12', color: '#fb923c', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      ⚡ AQUAGENIC PRURITUS = PV PATHOGNOMONIC
                    </span>
                  )}
                  {g.gene === 'THPO' && (
                    <span style={{ background: '#4c1d95', color: '#c084fc', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      ⚠ 5-UTR — REQUEST SPECIFIC SEQUENCING
                    </span>
                  )}
                  {(g.gene === 'VHL' || g.gene === 'EPAS1') && (
                    <span style={{ background: '#14532d', color: '#86efac', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      ⚠ EPO ELEVATED — HIF PATHWAY
                    </span>
                  )}
                </div>
                {[['Protein', g.protein_size], ['Inheritance', g.inheritance], ['Disease', g.disease_category],
                  ['Pathway', g.disease_pathway], ['Pathognomonic', g.pathognomonic], ['Treatment', g.treatment]].map(([label, val]) => (
                  <div key={label} style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, color: '#c084fc', fontWeight: 700, marginBottom: 2 }}>{label.toUpperCase()}</div>
                    <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{val}</div>
                  </div>
                ))}
                {g.patients && g.patients.length > 0 && (
                  <div>
                    <div style={{ fontSize: 11, color: '#c084fc', fontWeight: 700, marginBottom: 6 }}>SAMPLE PATIENTS (first 5)</div>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                        <thead>
                          <tr style={{ background: '#0f172a' }}>
                            {['ID','Sex','Age Dx','Subtype','Hb g/dL','Plt ×10⁹','EPO mIU/mL','JAK2 VAF%','BM Fibr','Spleen cm','Thrombosis','Aquagenic','Treatment'].map(h => (
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
                              <td style={{ padding: '3px 8px', color: '#94a3b8', whiteSpace: 'nowrap', fontSize: 9 }}>{p.subtype}</td>
                              <td style={{ padding: '3px 8px', color: p.hemoglobin_g_dl > 17 ? '#ef4444' : '#fbbf24' }}>
                                {p.hemoglobin_g_dl}
                              </td>
                              <td style={{ padding: '3px 8px', color: p.platelets_per_nl > 600 ? '#ef4444' : p.platelets_per_nl > 450 ? '#f97316' : '#94a3b8' }}>
                                {p.platelets_per_nl?.toLocaleString()}
                              </td>
                              <td style={{ padding: '3px 8px', color: p.serum_epo_miu_ml < 6 ? '#ef4444' : p.serum_epo_miu_ml > 30 ? '#34d399' : '#fbbf24' }}>
                                {p.serum_epo_miu_ml}
                              </td>
                              <td style={{ padding: '3px 8px', color: p.jak2_v617f_vaf_pct > 50 ? '#ef4444' : '#94a3b8' }}>
                                {p.jak2_v617f_vaf_pct != null ? `${p.jak2_v617f_vaf_pct}%` : '—'}
                              </td>
                              <td style={{ padding: '3px 8px', color: p.bone_marrow_fibrosis_grade >= 2 ? '#ef4444' : p.bone_marrow_fibrosis_grade >= 1 ? '#fbbf24' : '#475569' }}>
                                MF-{p.bone_marrow_fibrosis_grade}
                              </td>
                              <td style={{ padding: '3px 8px', color: p.spleen_size_cm_bcm > 15 ? '#ef4444' : '#94a3b8' }}>
                                {p.spleen_size_cm_bcm} cm
                              </td>
                              <td style={{ padding: '3px 8px', color: p.thrombosis_history ? '#ef4444' : '#475569' }}>{p.thrombosis_history ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px', color: p.aquagenic_pruritus ? '#ef4444' : '#475569' }}>{p.aquagenic_pruritus ? 'Yes ⚡' : 'No'}</td>
                              <td style={{ padding: '3px 8px', color: '#94a3b8', whiteSpace: 'nowrap', fontSize: 9 }}>{p.treatment}</td>
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
            <div style={{ fontSize: 14, fontWeight: 700, color: '#c084fc', marginBottom: 14 }}>
              Glossary — JAK2-V617F-Allele-Burden / CALR-Type1-vs-Type2 / Aquagenic-Pruritus / Triple-Negative-MPN / Erythrocytosis-EPO-Algorithm / Ruxolitinib-Withdrawal / Acquired-VWD / Belzutifan / THPO-5-UTR
            </div>
            {defs.definitions?.map((d, i) => (
              <div key={i} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: '12px 16px', marginBottom: 10 }}>
                <div style={{ fontWeight: 700, color: '#fbbf24', fontSize: 13, marginBottom: 6 }}>{d.term}</div>
                <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{d.definition}</div>
              </div>
            ))}
            <div style={{ fontSize: 14, fontWeight: 700, color: '#c084fc', marginTop: 20, marginBottom: 10 }}>
              Standards &amp; References
            </div>
            {defs.standards?.map((s, i) => (
              <div key={i} style={{ background: '#1e293b', border: '1px solid #4c1d95', borderRadius: 6, padding: '6px 12px', marginBottom: 6, fontSize: 11, color: '#94a3b8' }}>
                📋 {s}
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  );
}
