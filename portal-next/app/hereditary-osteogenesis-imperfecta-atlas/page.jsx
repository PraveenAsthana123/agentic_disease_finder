'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-osteogenesis-imperfecta-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'COL1A1':  '#b71c1c',  // deep red         — AD OI types I-IV, most common
  'COL1A2':  '#c62828',  // red               — AD OI types II-IV / arthrochalasis EDS
  'IFITM5':  '#e65100',  // burnt orange      — OI type V hyperplastic callus + IOM calcification
  'SERPINF1':'#f57f17',  // amber             — OI type VI fish-scale lamellae, PEDF absent
  'CRTAP':   '#1565c0',  // deep blue         — OI type VII rhizomelia, prolyl-3-OH complex
  'LEPRE1':  '#2e7d32',  // dark green        — OI type VIII West African founder
  'PPIB':    '#6a1b9a',  // deep purple       — OI type IX cyclophilin B ternary complex
  'FKBP10':  '#00695c',  // teal              — Bruck syndrome 1 contractures + OI
};

const GENE_INFO = {
  'COL1A1':  { full: 'COL1A1 / Collagen α1(I) / 1464aa', locus: '17q21.33', size: '1464 aa / 138 kDa', inh: 'AD', disease: 'OI types I/II/III/IV — NULL→type I mild (haploinsufficiency); GLYCINE SUBSTITUTION→types II-IV severe (dominant-negative); TYPE II LETHAL PERINATAL crumpled long bones; BISPHOSPHONATES cornerstone' },
  'COL1A2':  { full: 'COL1A2 / Collagen α2(I) / 1366aa', locus: '7q21.3', size: '1366 aa / 129 kDa', inh: 'AD', disease: 'OI types II/III/IV; Biallelic splice→Arthrochalasis EDS (severe joint laxity + OI); COL1A2 NULL frameshift = NOT OI (alpha1 homotrimers compensate — CRITICAL distinction)' },
  'IFITM5':  { full: 'IFITM5 / BRIL / 132aa', locus: '11p15.5', size: '132 aa / 14 kDa', inh: 'AD', disease: 'OI type V — HYPERPLASTIC CALLUS PATHOGNOMONIC (misdiagnosed as osteosarcoma); CALCIFICATION INTEROSSEOUS MEMBRANE PATHOGNOMONIC; collagen biochemistry NORMAL — missed by standard OI panel; recurrent c.-14C>T in >95%' },
  'SERPINF1':{ full: 'SERPINF1 / PEDF / 418aa', locus: '17p13.3', size: '418 aa / 46 kDa', inh: 'AR', disease: 'OI type VI — FISH-SCALE BONE LAMELLAE on biopsy PATHOGNOMONIC; SERUM PEDF UNDETECTABLE screening test; collagen NORMAL; DENOSUMAB superior to bisphosphonates' },
  'CRTAP':   { full: 'CRTAP / Cartilage-Associated Protein / 339aa', locus: '3p22.3', size: '339 aa / 37 kDa', inh: 'AR', disease: 'OI type VII — RHIZOMELIA (proximal limb shortening) PATHOGNOMONIC; OVERMODIFIED COLLAGEN SDS-PAGE; NULL=LETHAL PERINATAL; Cree-Oji-Cree founder p.Arg462X' },
  'LEPRE1':  { full: 'LEPRE1 / P3H1 / 736aa', locus: '1p34.2', size: '736 aa / 84 kDa', inh: 'AR', disease: 'OI type VIII — WEST AFRICAN FOUNDER p.Trp339Ter (>80%); OVERMODIFIED COLLAGEN same as CRTAP; NEONATAL LETHAL; BIOCHEMICAL DIAGNOSIS essential before WES in West African families' },
  'PPIB':    { full: 'PPIB / Cyclophilin B / 212aa', locus: '15q22.31', size: '212 aa / 24 kDa', inh: 'AR', disease: 'OI type IX — P3H1 ENZYME ACTIVITY NORMAL (CypB is isomerase not hydroxylase — DDx from LEPRE1); overmodified collagen; consanguineous Middle East/Asia families' },
  'FKBP10':  { full: 'FKBP10 / FKBP65 / 582aa', locus: '17q21.2', size: '582 aa / 65 kDa', inh: 'AR', disease: 'OI type XI / Bruck syndrome 1 — CONTRACTURES AT BIRTH + OI PATHOGNOMONIC; PTERYGIUM (joint skin webbing); Turkish founder c.831+1G>T; LH2/PLOD2 cross-link defect mechanism' },
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

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', minWidth: 130, flex: '1 1 130px' }}>
      <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div style={{ color: warn ? '#f87171' : '#f1f5f9', fontSize: 22, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HeredOsteogenesisImperfectaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [selGene, setSelGene] = useState(null);

  useEffect(() => {
    setLoading(true);
    setErr(null);
    const ep = tab === 'Definitions' ? 'definitions' : tab === 'Gene Table' ? 'breakdown' : tab === 'Clinical Atlas' ? 'breakdown' : 'overview';
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => {
        if (tab === 'Overview') setOverview(d);
        else if (tab === 'Gene Table' || tab === 'Clinical Atlas') setBreakdown(d);
        else setDefinitions(d);
      })
      .catch(e => setErr(e.message))
      .finally(() => setLoading(false));
  }, [tab]);

  const bg = '#0f172a';
  const card = '#1e293b';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#f1f5f9', fontFamily: 'system-ui,sans-serif', padding: '0 0 40px' }}>
      {/* Header */}
      <div style={{ background: '#1e293b', borderBottom: '1px solid #334155', padding: '18px 28px' }}>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>🧬 Hereditary Disease Atlas</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#f1f5f9' }}>
          Hereditary Osteogenesis Imperfecta Atlas
        </h1>
        <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 4 }}>
          Complete 8-Gene Bone Fragility &amp; Collagen-I Deficiency Reference ·{' '}
          {['COL1A1','COL1A2','IFITM5','SERPINF1','CRTAP','LEPRE1','PPIB','FKBP10'].map(g => (
            <GeneChip key={g} gene={g} />
          ))}
        </div>
        <div style={{ marginTop: 6, fontSize: 11, color: '#475569' }}>
          320 patients · 8 × 40 · seeds 2550-2557 · OI types I–IV (COL1A1/COL1A2) · OI type V (IFITM5 hyperplastic callus) · OI type VI (SERPINF1 fish-scale lamellae) · OI types VII–IX (CRTAP/LEPRE1/PPIB prolyl-3-OH complex) · Bruck syndrome 1 (FKBP10)
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, padding: '14px 28px 0', borderBottom: '1px solid #334155' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#b71c1c' : 'transparent',
            color: tab === t ? '#fff' : '#94a3b8',
            border: 'none', borderRadius: '6px 6px 0 0', padding: '8px 18px',
            cursor: 'pointer', fontWeight: tab === t ? 700 : 400, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 28px' }}>
        {loading && <div style={{ color: '#64748b' }}>Loading…</div>}
        {err && <div style={{ color: '#f87171' }}>Error: {err}</div>}

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40" />
              <MetricCard label="Genes Covered" value={overview.n_genes} sub="seeds 2550-2557" />
              <MetricCard label="Lethal Perinatal" value={`${overview.aggregate_metrics?.lethal_pct}%`} sub="types II/VII/VIII" warn />
              <MetricCard label="On Bisphosphonate" value={`${overview.aggregate_metrics?.on_bisphosphonate_pct}%`} sub="pamidronate/zoledronate" />
              <MetricCard label="Blue Sclerae" value={`${overview.aggregate_metrics?.blue_sclera_pct}%`} sub="types I/II mainly" />
              <MetricCard label="Hearing Loss" value={`${overview.aggregate_metrics?.hearing_impairment_pct}%`} sub="esp. type I COL1A1" />
              <MetricCard label="Dentinogenesis Imperfecta" value={`${overview.aggregate_metrics?.dentinogenesis_imperfecta_pct}%`} sub="50% in COL1A1/2 OI" />
              <MetricCard label="Hyperplastic Callus" value={`${overview.aggregate_metrics?.hyperplastic_callus_pct}%`} sub="OI type V IFITM5" />
              <MetricCard label="Contractures at Birth" value={`${overview.aggregate_metrics?.contractures_pct}%`} sub="Bruck/FKBP10" />
              <MetricCard label="Avg BMD Z-score" value={overview.aggregate_metrics?.avg_bmd_zscore} sub="DEXA" warn />
            </div>

            {/* Disease class grid */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>8 Disease Classes</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(300px,1fr))', gap: 10 }}>
                {overview.disease_classes?.map((cls, i) => {
                  const gene = Object.keys(GENE_COLORS)[i];
                  return (
                    <div key={cls} style={{ background: card, borderRadius: 8, padding: '12px 14px', borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                      <div style={{ color: '#f1f5f9', fontSize: 13, fontWeight: 600 }}>{cls}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Gene summary table */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Gene Summary</div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#334155', color: '#94a3b8' }}>
                      {['Gene','Locus','Inh.','Disease','Lethal%','Blue Sclerae%','Hearing%','DI%','BiP%','Callus%','Contractures%','BMD Z'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {overview.gene_summary?.map(g => (
                      <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                        <td style={{ padding: '8px 10px' }}><GeneChip gene={g.gene} /></td>
                        <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                        <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.inheritance?.split('(')[0]?.trim()}</td>
                        <td style={{ padding: '8px 10px', color: '#f1f5f9', maxWidth: 180 }}>{g.disease_name?.split(' —')[0]}</td>
                        <td style={{ padding: '8px 10px', color: g.lethal_pct > 20 ? '#f87171' : '#94a3b8' }}>{g.lethal_pct}%</td>
                        <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.blue_sclera_pct}%</td>
                        <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.hearing_impairment_pct}%</td>
                        <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.dentinogenesis_imperfecta_pct}%</td>
                        <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.on_bisphosphonate_pct}%</td>
                        <td style={{ padding: '8px 10px', color: g.hyperplastic_callus_pct > 50 ? '#fb923c' : '#94a3b8' }}>{g.hyperplastic_callus_pct}%</td>
                        <td style={{ padding: '8px 10px', color: g.contractures_at_birth_pct > 50 ? '#fb923c' : '#94a3b8' }}>{g.contractures_at_birth_pct}%</td>
                        <td style={{ padding: '8px 10px', color: '#f87171' }}>{g.avg_bmd_zscore}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Clinical pearls */}
            <div style={{ background: card, borderRadius: 8, padding: '16px 20px' }}>
              <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>Clinical Pearls</div>
              {overview.clinical_pearls?.map((p, i) => (
                <div key={i} style={{ marginBottom: 8, fontSize: 13, color: '#e2e8f0', display: 'flex', gap: 8 }}>
                  <span style={{ color: '#b71c1c', fontWeight: 800 }}>⬥</span>
                  <span>{p}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && breakdown && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))', gap: 8, marginBottom: 20 }}>
              {breakdown.gene_breakdowns?.map(g => (
                <div key={g.gene}
                  onClick={() => setSelGene(selGene === g.gene ? null : g.gene)}
                  style={{ background: selGene === g.gene ? GENE_COLORS[g.gene] + '33' : card, borderRadius: 8, padding: '12px 14px', cursor: 'pointer', border: `1px solid ${selGene === g.gene ? GENE_COLORS[g.gene] : '#334155'}` }}>
                  <GeneChip gene={g.gene} />
                  <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 4 }}>{g.locus} · {g.protein_size}</div>
                  <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{g.inheritance?.split(';')[0]}</div>
                </div>
              ))}
            </div>

            {selGene && (() => {
              const g = breakdown.gene_breakdowns?.find(x => x.gene === selGene);
              if (!g) return null;
              const info = GENE_INFO[selGene];
              return (
                <div style={{ background: card, borderRadius: 10, padding: '20px 24px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                    <GeneChip gene={g.gene} />
                    <div>
                      <div style={{ color: '#f1f5f9', fontWeight: 700, fontSize: 16 }}>{info?.full}</div>
                      <div style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus} · {g.protein_size} · {g.inheritance?.split(';')[0]}</div>
                    </div>
                  </div>

                  {/* Key metrics */}
                  <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
                    {[
                      { label: 'Patients', value: g.n_patients },
                      { label: 'Lethal %', value: `${g.lethal_pct}%` },
                      { label: 'Blue Sclerae', value: `${g.blue_sclera_pct}%` },
                      { label: 'Hearing Loss', value: `${g.hearing_impairment_pct}%` },
                      { label: 'DI', value: `${g.dentinogenesis_imperfecta_pct}%` },
                      { label: 'BiP', value: `${g.on_bisphosphonate_pct}%` },
                      { label: 'Callus', value: `${g.hyperplastic_callus_pct}%` },
                      { label: 'Contractures', value: `${g.contractures_at_birth_pct}%` },
                      { label: 'BMD Z', value: g.avg_bmd_zscore },
                      { label: 'Avg Fx', value: g.avg_fractures },
                      { label: 'Scoliosis', value: `${g.scoliosis_pct}%` },
                      { label: 'Wormian Bones', value: `${g.wormian_bones_pct}%` },
                    ].map(m => (
                      <div key={m.label} style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 90 }}>
                        <div style={{ color: '#64748b', fontSize: 10 }}>{m.label}</div>
                        <div style={{ color: '#f1f5f9', fontWeight: 700 }}>{m.value}</div>
                      </div>
                    ))}
                  </div>

                  {/* Pathognomonic */}
                  <div style={{ background: '#0f172a', borderRadius: 6, padding: '10px 14px', marginBottom: 12, borderLeft: `3px solid ${GENE_COLORS[selGene]}` }}>
                    <div style={{ color: '#94a3b8', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Pathognomonic</div>
                    <div style={{ color: '#fbbf24', fontSize: 12, fontWeight: 600 }}>{g.pathognomonic}</div>
                  </div>

                  {/* Key features */}
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ color: '#94a3b8', fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>Key Features</div>
                    {g.key_features?.map((f, i) => (
                      <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 4, fontSize: 12, color: '#e2e8f0' }}>
                        <span style={{ color: GENE_COLORS[selGene] }}>▸</span><span>{f}</span>
                      </div>
                    ))}
                  </div>

                  {/* Treatment */}
                  <div style={{ background: '#0f172a', borderRadius: 6, padding: '10px 14px', marginBottom: 12 }}>
                    <div style={{ color: '#94a3b8', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Treatment</div>
                    <div style={{ color: '#e2e8f0', fontSize: 12 }}>{g.treatment}</div>
                  </div>

                  {/* DDx */}
                  <div>
                    <div style={{ color: '#94a3b8', fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>Key DDx</div>
                    {g.key_ddx?.map((d, i) => (
                      <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 4, fontSize: 12, color: '#94a3b8' }}>
                        <span style={{ color: '#f87171' }}>⚠</span><span>{d}</span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            {breakdown.gene_breakdowns?.map(g => (
              <div key={g.gene} style={{ background: card, borderRadius: 10, padding: '18px 22px', marginBottom: 14, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                  <GeneChip gene={g.gene} />
                  <div style={{ color: '#f1f5f9', fontWeight: 700 }}>{g.disease_category?.split(' —')[0]}</div>
                  <div style={{ marginLeft: 'auto', color: '#64748b', fontSize: 11 }}>{g.locus} · {g.inheritance?.split(';')[0]}</div>
                </div>
                <div style={{ color: '#fbbf24', fontSize: 12, marginBottom: 8, fontStyle: 'italic' }}>{g.pathognomonic?.substring(0, 180)}…</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 11, color: '#94a3b8' }}>
                  <span>Lethal: <b style={{ color: g.lethal_pct > 20 ? '#f87171' : '#f1f5f9' }}>{g.lethal_pct}%</b></span>
                  <span>BiP: <b style={{ color: '#f1f5f9' }}>{g.on_bisphosphonate_pct}%</b></span>
                  <span>Blue sclerae: <b style={{ color: '#f1f5f9' }}>{g.blue_sclera_pct}%</b></span>
                  <span>Hearing: <b style={{ color: '#f1f5f9' }}>{g.hearing_impairment_pct}%</b></span>
                  <span>DI: <b style={{ color: '#f1f5f9' }}>{g.dentinogenesis_imperfecta_pct}%</b></span>
                  <span>Callus: <b style={{ color: g.hyperplastic_callus_pct > 50 ? '#fb923c' : '#f1f5f9' }}>{g.hyperplastic_callus_pct}%</b></span>
                  <span>Contractures: <b style={{ color: g.contractures_at_birth_pct > 50 ? '#fb923c' : '#f1f5f9' }}>{g.contractures_at_birth_pct}%</b></span>
                  <span>BMD Z: <b style={{ color: '#f87171' }}>{g.avg_bmd_zscore}</b></span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            {/* Gene entries */}
            <div style={{ marginBottom: 24 }}>
              <div style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>Gene Definitions</div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
                {Object.keys(definitions.gene_entries || {}).map(g => (
                  <button key={g} onClick={() => setSelGene(selGene === g ? null : g)} style={{
                    background: GENE_COLORS[g] || '#334155',
                    color: '#fff', border: 'none', borderRadius: 4,
                    padding: '4px 12px', cursor: 'pointer', fontWeight: 700, fontSize: 12,
                    opacity: selGene && selGene !== g ? 0.5 : 1,
                  }}>{g}</button>
                ))}
              </div>

              {Object.entries(definitions.gene_entries || {}).filter(([g]) => !selGene || selGene === g).map(([gene, d]) => (
                <div key={gene} style={{ background: card, borderRadius: 8, padding: '16px 20px', marginBottom: 10, borderLeft: `3px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <GeneChip gene={gene} />
                    <div style={{ color: '#f1f5f9', fontWeight: 700 }}>{d.disease_name?.split(' —')[0] || d.full_name}</div>
                    <div style={{ marginLeft: 'auto', color: '#64748b', fontSize: 11 }}>{d.locus} · {d.protein_size} · {d.inheritance}</div>
                  </div>
                  <div style={{ color: '#fbbf24', fontSize: 12, marginBottom: 8 }}>{d.pathognomonic}</div>
                  <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8 }}><b style={{ color: '#e2e8f0' }}>Treatment:</b> {d.treatment}</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, fontSize: 11, marginBottom: 8 }}>
                    <div><span style={{ color: '#64748b' }}>Lethal:</span> <span style={{ color: d.lethal_pct > 20 ? '#f87171' : '#cbd5e1' }}>{d.lethal_pct}%</span></div>
                    <div><span style={{ color: '#64748b' }}>Blue sclerae:</span> <span style={{ color: '#cbd5e1' }}>{d.blue_sclera_pct}%</span></div>
                    <div><span style={{ color: '#64748b' }}>Hearing:</span> <span style={{ color: '#cbd5e1' }}>{d.hearing_impairment_pct}%</span></div>
                    <div><span style={{ color: '#64748b' }}>DI:</span> <span style={{ color: '#cbd5e1' }}>{d.dentinogenesis_imperfecta_pct}%</span></div>
                    <div><span style={{ color: '#64748b' }}>Hyperplastic callus:</span> <span style={{ color: d.hyperplastic_callus_pct > 50 ? '#fb923c' : '#cbd5e1' }}>{d.hyperplastic_callus_pct}%</span></div>
                    <div><span style={{ color: '#64748b' }}>Contractures:</span> <span style={{ color: d.contractures_at_birth_pct > 50 ? '#fb923c' : '#cbd5e1' }}>{d.contractures_at_birth_pct}%</span></div>
                  </div>
                  <div>
                    <div style={{ color: '#94a3b8', fontSize: 10, textTransform: 'uppercase', marginBottom: 4 }}>Key DDx</div>
                    {d.key_ddx?.map((ddx, i) => (
                      <div key={i} style={{ fontSize: 11, color: '#94a3b8', marginBottom: 2 }}>⚠ {ddx}</div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* OI Glossary */}
            <div>
              <div style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>OI Glossary</div>
              {Object.entries(definitions.oi_glossary || {}).map(([term, def]) => (
                <div key={term} style={{ background: '#0f172a', borderRadius: 8, padding: '14px 18px', marginBottom: 8 }}>
                  <div style={{ color: '#fbbf24', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{term}</div>
                  <div style={{ color: '#e2e8f0', fontSize: 12, lineHeight: 1.6 }}>{def}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
