'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-congenital-hyperinsulinism-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ABCC8':   '#1565c0',  // deep blue     — SUR1, K-ATP, most common CHI
  'KCNJ11':  '#0277bd',  // medium blue   — Kir6.2, K-ATP pore
  'GLUD1':   '#2e7d32',  // dark green    — GDH, HI/HA, NH3 elevated
  'GCK':     '#e65100',  // deep orange   — glucokinase GOF, set-point
  'HADH':    '#6a1b9a',  // deep purple   — SCHAD, 3-OH-glutarate
  'HNF4A':   '#c62828',  // crimson       — biphasic CHI → MODY1
  'UCP2':    '#00695c',  // dark teal     — mild CHI, spontaneous resolution
  'SLC16A1': '#4527a0',  // deep indigo   — EIHI, exercise-induced, promoter
};

const GENE_INFO = {
  'ABCC8':   { full: 'ABCC8 / SUR1 / 1582aa', locus: '11p15.1', size: '1582 aa / 177 kDa (SUR1 — K-ATP regulatory subunit; K-ATP octamer (SUR1·Kir6.2)₄; LOF → K-ATP constitutively closed → unregulated insulin; 18F-DOPA PET focal/diffuse)', inh: 'AR/AD' },
  'KCNJ11':  { full: 'KCNJ11 / Kir6.2 / 390aa', locus: '11p15.1', size: '390 aa / 43 kDa (Kir6.2 — K-ATP pore subunit; ATP binds Kir6.2 directly; LOF → K-ATP absent → CHI; GOF → neonatal diabetes; same locus ABCC8)', inh: 'AR/AD' },
  'GLUD1':   { full: 'GLUD1 / GDH1 / 558aa', locus: '10q23.3', size: '558 aa / 61 kDa (GDH — mitochondrial glutamate dehydrogenase; GOF → GTP/ATP allosteric inhibition lost → excess ATP → CHI + NH3 → HI/HA syndrome)', inh: 'AD GOF' },
  'GCK':     { full: 'GCK / Glucokinase / 465aa', locus: '7p13', size: '465 aa / 52 kDa (hexokinase IV; glucose sensor; GOF → lowered BG set-point for insulin secretion; LOF → MODY2; same gene opposite phenotype)', inh: 'AD GOF' },
  'HADH':    { full: 'HADH / SCHAD / 314aa', locus: '4q25', size: '314 aa / 34 kDa (short-chain 3-OH-acyl-CoA dehydrogenase; inhibits GDH; LOF → GDH disinhibited → CHI; urine 3-OH-glutaric acid PATHOGNOMONIC)', inh: 'AR' },
  'HNF4A':   { full: 'HNF4A / NR2A1 / 474aa', locus: '20q13.12', size: '474 aa / 53 kDa (nuclear receptor TF; drives ABCC8/KCNJ11 expression; LOF → fewer K-ATP → neonatal CHI then adult MODY1 — biphasic)', inh: 'AD LOF' },
  'UCP2':    { full: 'UCP2 / Uncoupling Protein 2 / 309aa', locus: '11q13.4', size: '309 aa / 34 kDa (IMM uncoupling protein; GOF → mild CHI via ATP/K-ATP dysregulation; spontaneous resolution by 3-6 years)', inh: 'AD GOF' },
  'SLC16A1': { full: 'SLC16A1 / MCT1 / 478aa', locus: '17q24.2', size: '478 aa / 52 kDa (MCT1 — normally ABSENT from beta-cells; promoter GOF → ectopic beta-cell MCT1 → exercise pyruvate → K-ATP closes → insulin → EIHI)', inh: 'AD GOF (promoter)' },
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

export default function HereditaryCongenitalHyperinsulinismAtlas() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [activeGene, setActiveGene] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setLoading(true);
    setErr(null);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  const genes = overview?.genes || [];

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #1e293b 100%)', padding: '24px 32px', borderBottom: '1px solid #1565c0' }}>
        <div style={{ fontSize: 11, color: '#90caf9', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 6 }}>
          Hereditary Disease Atlas · Endocrinology · Congenital Hyperinsulinism
        </div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#f1f5f9' }}>
          🧬 Hereditary Congenital Hyperinsulinism Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 4 }}>
          Complete 8-Gene CHI Reference · ABCC8 · KCNJ11 · GLUD1 · GCK · HADH · HNF4A · UCP2 · SLC16A1 · Seeds 2910–2917
        </div>
        <div style={{ marginTop: 10 }}>
          {genes.map(g => <GeneChip key={g} gene={g} active={activeGene} onClick={setActiveGene} />)}
          {activeGene && <span onClick={() => setActiveGene(null)} style={{ cursor: 'pointer', fontSize: 11, color: '#94a3b8', marginLeft: 8 }}>[clear]</span>}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid #1e293b', padding: '0 32px', background: '#0f172a' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: 'none', border: 'none', color: tab === t ? '#60a5fa' : '#64748b',
            borderBottom: tab === t ? '2px solid #60a5fa' : '2px solid transparent',
            padding: '10px 18px', cursor: 'pointer', fontFamily: 'monospace', fontSize: 13, fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {/* Body */}
      <div style={{ padding: '24px 32px' }}>
        {loading && <div style={{ color: '#64748b' }}>Loading atlas data…</div>}
        {err && <div style={{ color: '#ef4444' }}>Error: {err}</div>}

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
              <MetricCard label="Total genes" value={overview.total_genes} />
              <MetricCard label="Total patients" value={overview.total_patients} sub="8 × 40 cohort" />
              <MetricCard label="Seeds" value="2910–2917" sub="hereditary CHI" />
            </div>
            <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 20, fontSize: 13, lineHeight: 1.6, color: '#94a3b8' }}>
              {overview.description}
            </div>

            {/* Categories */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>Mechanistic Categories</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                {Object.entries(overview.categories || {}).map(([cat, gList]) => (
                  <div key={cat} style={{ background: '#1e293b', borderRadius: 6, padding: '8px 14px', fontSize: 12 }}>
                    <div style={{ color: '#94a3b8', marginBottom: 4 }}>{cat}</div>
                    <div>{gList.map(g => <GeneChip key={g} gene={g} active={null} />)}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Key facts */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>Key Clinical Facts</div>
              {overview.key_facts?.map((f, i) => (
                <div key={i} style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6, padding: '3px 0', borderBottom: '1px solid #0f172a' }}>
                  <span style={{ color: '#60a5fa', marginRight: 8 }}>▸</span>{f}
                </div>
              ))}
            </div>

            {/* Diagnostic algorithm */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>Diagnostic Algorithm</div>
              <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{overview.diagnostic_algorithm}</div>
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && overview && (
          <div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #334155' }}>
                  {['Gene', 'Locus', 'Inheritance', 'Patients', 'Avg BGL Nadir (mmol/L)', 'Avg GIR (mg/kg/min)', 'Diazoxide Responsive %', 'Macrosomia %'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {overview.gene_rows?.filter(r => !activeGene || r.gene === activeGene).map((r, i) => (
                  <tr key={r.gene} style={{ borderBottom: '1px solid #1e293b', background: i % 2 === 0 ? '#0f172a' : '#111827' }}>
                    <td style={{ padding: '8px 10px' }}><GeneChip gene={r.gene} active={null} /></td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{r.locus}</td>
                    <td style={{ padding: '8px 10px', color: '#a5b4fc' }}>{GENE_INFO[r.gene]?.inh || '—'}</td>
                    <td style={{ padding: '8px 10px', color: '#38bdf8' }}>{r.patients}</td>
                    <td style={{ padding: '8px 10px', color: '#f87171' }}>{r.avg_bgl_nadir_mmol_l}</td>
                    <td style={{ padding: '8px 10px', color: '#f59e0b' }}>{r.avg_gir_mg_kg_min}</td>
                    <td style={{ padding: '8px 10px', color: r.diazoxide_responsive_pct > 80 ? '#4ade80' : '#ef4444' }}>
                      {r.diazoxide_responsive_pct}%
                    </td>
                    <td style={{ padding: '8px 10px', color: r.macrosomia_pct > 40 ? '#fbbf24' : '#64748b' }}>
                      {r.macrosomia_pct}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            {breakdown.genes?.filter(g => !activeGene || g.gene === activeGene).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <GeneChip gene={g.gene} active={null} />
                  <span style={{ fontSize: 12, color: '#64748b' }}>{GENE_INFO[g.gene]?.full}</span>
                  <span style={{ fontSize: 11, color: '#475569', marginLeft: 'auto' }}>seed {g.seed} · n={g.patient_count}</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div>
                    <div style={{ fontSize: 10, color: '#475569', textTransform: 'uppercase', marginBottom: 4 }}>Disease / Category</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{g.disease_category?.slice(0, 380)}{g.disease_category?.length > 380 ? '…' : ''}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: '#475569', textTransform: 'uppercase', marginBottom: 4 }}>Treatment</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{g.treatment?.slice(0, 380)}{g.treatment?.length > 380 ? '…' : ''}</div>
                  </div>
                </div>
                <div style={{ marginTop: 10 }}>
                  <div style={{ fontSize: 10, color: '#475569', textTransform: 'uppercase', marginBottom: 4 }}>Pathognomonic</div>
                  <div style={{ fontSize: 11, color: '#fcd34d', lineHeight: 1.5 }}>{g.pathognomonic}</div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            {definitions.definitions?.map((d, i) => (
              <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#60a5fa', marginBottom: 6 }}>{d.term}</div>
                <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7 }}>{d.definition}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
