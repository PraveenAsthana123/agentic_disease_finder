'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-thyroid-dyshormonogenesis-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'TPO':     '#1565c0',  // deep blue      — peroxidase, organification, most common
  'TSHR':    '#0277bd',  // medium blue    — TSH receptor resistance, no goiter
  'TG':      '#2e7d32',  // dark green     — thyroglobulin scaffold, low TG paradox
  'SLC5A5':  '#e65100',  // deep orange    — NIS, absent scan uptake
  'DUOX2':   '#6a1b9a',  // deep purple    — H2O2 generator, transient/permanent
  'DUOXA2':  '#ad1457',  // deep pink      — DUOX2 chaperone, under-recognised
  'SLC26A4': '#c62828',  // crimson        — Pendred, SNHL + Mondini/EVA
  'DEHAL1':  '#4527a0',  // deep indigo    — iodotyrosine recycling, late NBS miss
};

const GENE_INFO = {
  'TPO':     { full: 'TPO / Thyroid Peroxidase / 933aa', locus: '2p25.3', size: '933 aa / 103 kDa (haem enzyme; iodination + coupling; LOF → organification defect; perchlorate discharge POSITIVE; most common ~25%)', inh: 'AR' },
  'TSHR':    { full: 'TSHR / TSH Receptor / 764aa', locus: '14q31.1', size: '764 aa / 84 kDa (7TM GPCR; TSH binds → Gsα → cAMP → thyroid growth/function; LOF → resistance; NO GOITER; hypoplastic in-situ gland)', inh: 'AR LOF' },
  'TG':      { full: 'TG / Thyroglobulin / 2768aa', locus: '8q24.22', size: '2768 aa / 330 kDa (scaffold for T3/T4 synthesis; MIT+DIT → T3/T4 on TG; LOF → misfolded/ER-retained → low serum TG despite goiter — PARADOX)', inh: 'AR' },
  'SLC5A5':  { full: 'SLC5A5 / NIS / 643aa', locus: '19p13.11', size: '643 aa / 70 kDa (13TM; Na+:I− co-transporter; concentrates iodide into thyroid; LOF → absent scan uptake; normal urine iodide PATHOGNOMONIC)', inh: 'AR' },
  'DUOX2':   { full: 'DUOX2 / Dual Oxidase 2 / 1548aa', locus: '15q15.3', size: '1548 aa / 178 kDa (NADPH oxidase; generates H2O2 for TPO; biallelic = permanent CH; monoallelic = transient CH — most common transient CH cause)', inh: 'AR/heterozyg' },
  'DUOXA2':  { full: 'DUOXA2 / DUOX2 Maturation Factor / 320aa', locus: '15q15.3', size: '320 aa / 36 kDa (ER chaperone for DUOX2; DUOX2 cannot reach apical membrane without DUOXA2; biallelic = identical to DUOX2 null)', inh: 'AR' },
  'SLC26A4': { full: 'SLC26A4 / Pendrin / 780aa', locus: '7q22.3', size: '780 aa / 86 kDa (anion exchanger Cl−/HCO3−/I−; thyroid apical + inner ear endolymphatic sac; Pendred syndrome: CH + SNHL + Mondini/EVA)', inh: 'AR' },
  'DEHAL1':  { full: 'DEHAL1 / Iodotyrosine Dehalogenase / 289aa', locus: '6q25.1', size: '289 aa / 33 kDa (FMN reductase; deiodates MIT/DIT → recycles iodide; LOF → urine MIT/DIT elevated PATHOGNOMONIC; NBS may miss late-onset CH)', inh: 'AR' },
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

export default function HereditaryThyroidDyshormonogenesisAtlas() {
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
      <div style={{ background: 'linear-gradient(135deg, #1a237e 0%, #01579b 50%, #1e293b 100%)', padding: '24px 32px', borderBottom: '1px solid #0277bd' }}>
        <div style={{ fontSize: 11, color: '#81d4fa', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 6 }}>
          Hereditary Disease Atlas · Endocrinology · Thyroid Dyshormonogenesis
        </div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#f1f5f9' }}>
          🧬 Hereditary Thyroid Dyshormonogenesis Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 4 }}>
          Complete 8-Gene CH Reference · TPO · TSHR · TG · SLC5A5 · DUOX2 · DUOXA2 · SLC26A4 · DEHAL1 · Seeds 2918–2925
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
              <MetricCard label="Seeds" value="2918–2925" sub="hereditary thyroid" />
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
            {overview.diagnostic_algorithm && (
              <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 20 }}>
                <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>Diagnostic Algorithm</div>
                <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.8, whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                  {overview.diagnostic_algorithm}
                </div>
              </div>
            )}

            {/* Gene summary table */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>Gene Summary</div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#1e293b' }}>
                      {['Gene', 'Locus', 'Patients', 'Avg TSH (mU/L)', 'Goiter %', 'NBS Detected %', 'Perchlorate +ve %'].map(h => (
                        <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#64748b', fontWeight: 600, borderBottom: '1px solid #334155' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {overview.gene_rows?.map(row => (
                      <tr key={row.gene} style={{ borderBottom: '1px solid #1e293b', opacity: activeGene && activeGene !== row.gene ? 0.4 : 1 }}>
                        <td style={{ padding: '8px 12px' }}><GeneChip gene={row.gene} active={activeGene} onClick={setActiveGene} /></td>
                        <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{row.locus}</td>
                        <td style={{ padding: '8px 12px', color: '#38bdf8' }}>{row.patients}</td>
                        <td style={{ padding: '8px 12px', color: '#fbbf24' }}>{row.avg_tsh_mu_l}</td>
                        <td style={{ padding: '8px 12px', color: row.goiter_pct > 50 ? '#f87171' : '#94a3b8' }}>{row.goiter_pct}%</td>
                        <td style={{ padding: '8px 12px', color: row.nbs_detected_pct > 80 ? '#4ade80' : '#fbbf24' }}>{row.nbs_detected_pct}%</td>
                        <td style={{ padding: '8px 12px', color: row.perchlorate_positive_pct > 50 ? '#fb923c' : '#94a3b8' }}>{row.perchlorate_positive_pct}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && breakdown && (
          <div>
            <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 16 }}>
              Gene-Level Reference ({breakdown.count} genes)
            </div>
            {(breakdown.genes || [])
              .filter(g => !activeGene || g.gene === activeGene)
              .map(g => (
                <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                    <GeneChip gene={g.gene} active={null} />
                    <span style={{ fontSize: 12, color: '#64748b' }}>{GENE_INFO[g.gene]?.full}</span>
                    <span style={{ fontSize: 11, color: '#475569' }}>{g.locus}</span>
                    <span style={{ fontSize: 11, color: '#38bdf8', marginLeft: 'auto' }}>{g.patient_count} patients · seed {g.seed}</span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: 12 }}>
                    <div>
                      <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>PROTEIN</div>
                      <div style={{ color: '#94a3b8', lineHeight: 1.5 }}>{GENE_INFO[g.gene]?.size}</div>
                    </div>
                    <div>
                      <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>INHERITANCE</div>
                      <div style={{ color: '#94a3b8', lineHeight: 1.5 }}>{GENE_INFO[g.gene]?.inh}</div>
                    </div>
                  </div>
                </div>
              ))}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            {(breakdown.genes || [])
              .filter(g => !activeGene || g.gene === activeGene)
              .map(g => (
                <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 20, marginBottom: 20, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                    <GeneChip gene={g.gene} active={null} />
                    <span style={{ fontSize: 14, fontWeight: 700, color: '#f1f5f9' }}>{GENE_INFO[g.gene]?.full}</span>
                    <span style={{ fontSize: 11, color: '#38bdf8', marginLeft: 'auto' }}>{g.locus} · {g.patient_count} pts</span>
                  </div>
                  {[
                    { label: 'Protein / Function', text: g.protein_size },
                    { label: 'Inheritance & Genetics', text: g.inheritance },
                    { label: 'Disease Category', text: g.disease_category },
                    { label: 'Disease Pathway', text: g.disease_pathway },
                    { label: 'Pathognomonic Pattern', text: g.pathognomonic, highlight: true },
                    { label: 'Treatment', text: g.treatment },
                  ].map(({ label, text, highlight }) => (
                    <div key={label} style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>{label}</div>
                      <div style={{
                        fontSize: 12, color: highlight ? '#fbbf24' : '#94a3b8',
                        lineHeight: 1.6, background: '#0f172a', borderRadius: 4, padding: '8px 12px',
                        border: highlight ? '1px solid #f59e0b40' : '1px solid #1e293b',
                        whiteSpace: 'pre-wrap',
                      }}>{text}</div>
                    </div>
                  ))}
                </div>
              ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 16 }}>
              Clinical Glossary ({definitions.definitions?.length} entries)
            </div>
            {definitions.definitions?.map((d, i) => (
              <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#60a5fa', marginBottom: 8 }}>{d.term}</div>
                <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{d.definition}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
