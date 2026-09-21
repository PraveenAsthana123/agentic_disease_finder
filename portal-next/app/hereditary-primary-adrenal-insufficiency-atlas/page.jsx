'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-primary-adrenal-insufficiency-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'AIRE':   '#7b1fa2',  // deep purple   — APS-1, autoimmune, AIRE TF
  'NR0B1':  '#1565c0',  // deep blue     — X-linked AHC + IHH, DAX1
  'AAAS':   '#2e7d32',  // dark green    — Triple A, nuclear pore
  'MC2R':   '#b71c1c',  // deep red      — FGD1, ACTH receptor
  'MRAP':   '#e65100',  // deep orange   — FGD2, MC2R chaperone
  'NNT':    '#00695c',  // dark teal     — FGD5, mitochondrial NADPH
  'TXNRD2': '#4527a0',  // deep indigo   — FGD4, thioredoxin reductase
  'ABCD1':  '#c62828',  // crimson       — X-ALD, VLCFA, cerebral ALD
};

const GENE_INFO = {
  'AIRE':   { full: 'AIRE / Autoimmune Regulator / 545aa', locus: '21q22.3', size: '545 aa / 60 kDa (central tolerance TF in mTECs; LOF → peripheral antigens not thymic-presented → multi-organ autoimmunity; APS-1/APECED)', inh: 'AR' },
  'NR0B1':  { full: 'NR0B1 / DAX1 / 470aa', locus: 'Xp21.2', size: '470 aa / 51 kDa (orphan nuclear receptor; adrenal cortex + gonads + pituitary; LOF → adrenal aplasia + IHH in males)', inh: 'XLR' },
  'AAAS':   { full: 'AAAS / Aladin / 546aa', locus: '12q13.13', size: '546 aa / 60 kDa (WD40 nuclear pore complex; LOF → alacrima + achalasia + ACTH-resistant PAI + progressive neurodegeneration)', inh: 'AR' },
  'MC2R':   { full: 'MC2R / ACTH Receptor / 297aa', locus: '18p11.21', size: '297 aa / 33 kDa (7TM GPCR; only binds ACTH; LOF → Gs not activated → cAMP absent → cortisol absent; FGD1; aldosterone preserved)', inh: 'AR' },
  'MRAP':   { full: 'MRAP / MC2R Accessory Protein / 227aa', locus: '21q22.11', size: '227 aa / 25 kDa (MC2R chaperone; escorts MC2R to plasma membrane; LOF → MC2R ER-trapped → no ACTH response; FGD2)', inh: 'AR' },
  'NNT':    { full: 'NNT / Nicotinamide Nucleotide Transhydrogenase / 1086aa', locus: '5p12', size: '1086 aa / 114 kDa (IMM NADPH regenerator; LOF → mitochondrial oxidative stress → adrenocortical apoptosis; FGD5)', inh: 'AR' },
  'TXNRD2': { full: 'TXNRD2 / Thioredoxin Reductase 2 / 524aa', locus: '22q11.21', size: '524 aa / 57 kDa (selenoprotein; reduces TXN2; LOF → PRDX3 inactivated → H₂O₂ → adrenocortical cell death; FGD4 + cardiomyopathy)', inh: 'AR' },
  'ABCD1':  { full: 'ABCD1 / ALDP / 745aa', locus: 'Xq28', size: '745 aa / 84 kDa (peroxisomal VLCFA transporter; LOF → C26:0 accumulates → adrenal + CNS demyelination; X-ALD; HSCT if LOES ≤ 9)', inh: 'XLR' },
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

export default function HereditaryPrimaryAdrenalInsufficiencyAtlas() {
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
      <div style={{ background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e293b 100%)', padding: '24px 32px', borderBottom: '1px solid #4f46e5' }}>
        <div style={{ fontSize: 11, color: '#a5b4fc', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 6 }}>
          Hereditary Disease Atlas · Endocrinology · Primary Adrenal Insufficiency
        </div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#f1f5f9' }}>
          🧬 Hereditary Primary Adrenal Insufficiency Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 4 }}>
          Complete 8-Gene Non-CAH PAI Reference · AIRE · NR0B1 · AAAS · MC2R · MRAP · NNT · TXNRD2 · ABCD1 · Seeds 2902–2909
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
            background: 'none', border: 'none', color: tab === t ? '#818cf8' : '#64748b',
            borderBottom: tab === t ? '2px solid #818cf8' : '2px solid transparent',
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
              <MetricCard label="Seeds" value="2902–2909" sub="hereditary PAI" />
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
                  <span style={{ color: '#818cf8', marginRight: 8 }}>▸</span>{f}
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
                  {['Gene', 'Locus', 'Inheritance', 'Patients', 'Avg ACTH (pmol/L)', 'Mineralocorticoid Affected', 'Glucocorticoid-Only %'].map(h => (
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
                    <td style={{ padding: '8px 10px', color: '#f59e0b' }}>{r.avg_acth_pmol_l}</td>
                    <td style={{ padding: '8px 10px', color: r.mineralocorticoid_affected ? '#ef4444' : '#4ade80' }}>
                      {r.mineralocorticoid_affected ? 'YES (±both)' : 'PRESERVED'}
                    </td>
                    <td style={{ padding: '8px 10px', color: r.glucocorticoid_only_pct > 80 ? '#4ade80' : '#f59e0b' }}>
                      {r.glucocorticoid_only_pct}%
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
                    <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{g.disease_category?.slice(0, 350)}{g.disease_category?.length > 350 ? '…' : ''}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: '#475569', textTransform: 'uppercase', marginBottom: 4 }}>Treatment</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{g.treatment?.slice(0, 350)}{g.treatment?.length > 350 ? '…' : ''}</div>
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
                <div style={{ fontSize: 13, fontWeight: 700, color: '#818cf8', marginBottom: 6 }}>{d.term}</div>
                <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7 }}>{d.definition}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
