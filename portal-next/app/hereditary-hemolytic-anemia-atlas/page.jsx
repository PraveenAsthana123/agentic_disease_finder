'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hemolytic-anemia-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'HBB':    '#b71c1c',  // deep red — sickle cell
  'HBA1':   '#880e4f',  // dark pink — alpha-thal
  'G6PD':   '#e65100',  // burnt orange — enzymopathy
  'PKLR':   '#f57f17',  // amber — PK deficiency
  'ANK1':   '#1565c0',  // deep blue — spherocytosis
  'SLC4A1': '#1b5e20',  // dark green — SAO/HS2
  'SPTA1':  '#4a148c',  // deep purple — HE/HPP
  'KCNN4':  '#006064',  // teal — DHS/xerocytosis
};

const GENE_INFO = {
  'HBB':    { full: 'HBB / Beta-Globin / 147aa', locus: '11p15.4', size: '147 aa / 16 kDa', inh: 'AR/codominant', disease: 'Sickle cell disease (SCD) + beta-thalassemia — HbS p.Glu6Val; hydroxyurea reduces VOC 45%; Casgevy (CRISPR HbF) FDA 2023; NBS MANDATORY' },
  'HBA1':   { full: 'HBA1 / Alpha-Globin / 142aa', locus: '16p13.3', size: '142 aa / 16 kDa', inh: 'AR (deletional)', disease: 'Alpha-thalassemia — --/-- = Hb Bart\'s hydrops fetalis (FATAL); HbH disease; MLPA/GAP-PCR MANDATORY (standard PCR misses deletions)' },
  'G6PD':   { full: 'G6PD / Glucose-6-Phosphate Dehydrogenase / 515aa', locus: 'Xq28', size: '515 aa / 59 kDa', inh: 'XLR', disease: 'G6PD deficiency — most common RBC enzymopathy globally (~400M); bite cells + Heinz bodies on PBS; test 3 months AFTER crisis (reticulocytes falsely normal)' },
  'PKLR':   { full: 'PKLR / Pyruvate Kinase LR / 574aa', locus: '1q22', size: '574 aa / 62 kDa', inh: 'AR', disease: 'PK deficiency — echinocytes on PBS (pathognomonic); 2,3-DPG elevated; mitapivat (Pyrukynd) FDA 2022 — FIRST oral disease-modifying therapy' },
  'ANK1':   { full: 'ANK1 / Ankyrin-1 / 1881aa', locus: '8p11.21', size: '1881 aa / 206 kDa', inh: 'AD', disease: 'Hereditary spherocytosis type 1 — most common HS (40-65%); EMA binding test PATHOGNOMONIC; Parvovirus B19 aplastic crisis; vaccination MANDATORY before splenectomy' },
  'SLC4A1': { full: 'SLC4A1 / AE1 / Band 3 / 911aa', locus: '17q21.31', size: '911 aa / 102 kDa', inh: 'AD/AR', disease: 'HS type 2 + Southeast Asian ovalocytosis (SAO) — SAO PROTECTS against cerebral malaria; osmotic fragility NORMAL/DECREASED in SAO (opposite of classic HS); AR stomatocytosis: splenectomy CONTRAINDICATED' },
  'SPTA1':  { full: 'SPTA1 / Alpha-Spectrin I / 2429aa', locus: '1q23.1', size: '2429 aa / 280 kDa', inh: 'AR', disease: 'Hereditary elliptocytosis (HE) + pyropoikilocytosis (HPP) — alphaLELY allele critical modifier; thermal lability test 45°C (HPP fragments); EMA test NORMAL (distinguishes from HS)' },
  'KCNN4':  { full: 'KCNN4 / SK4 Gardos Channel / 427aa', locus: '19q13.31', size: '427 aa / 48 kDa', inh: 'AD (GOF)', disease: 'Dehydrated hereditary stomatocytosis (DHS/xerocytosis) — MCHC ELEVATED (>36) pathognomonic; SPLENECTOMY ABSOLUTELY CONTRAINDICATED (life-threatening post-splenectomy thromboembolism)' },
};

function GeneChip({ gene }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span style={{ background: col, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 12, fontWeight: 700, margin: '0 2px' }}>
      {gene}
    </span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  const bg = '#1e293b';
  const border = warn ? '#ef4444' : '#334155';
  return (
    <div style={{ background: bg, border: `1px solid ${border}`, borderRadius: 8, padding: '12px 16px', minWidth: 120 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8' }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HereditaryHaemolyticAnaemiaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [selGene, setSelGene] = useState(null);

  useEffect(() => {
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' ? 'breakdown'
      : tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    setLoading(true);
    setErr(null);
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => {
        if (ep === 'overview') setOverview(data);
        else if (ep === 'breakdown') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setErr(String(e)); setLoading(false); });
  }, [tab]);

  const bg = '#0f172a';
  const card = '#1e293b';
  const accent = '#b71c1c';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ background: card, borderRadius: 12, padding: '20px 24px', marginBottom: 20, borderLeft: `4px solid ${accent}` }}>
          <h1 style={{ margin: 0, fontSize: 20, color: '#f87171' }}>🧬 Hereditary Hemolytic Anemia Atlas</h1>
          <p style={{ margin: '6px 0 0', color: '#94a3b8', fontSize: 13 }}>
            Complete 8-Gene Reference — HBB · HBA1 · G6PD · PKLR · ANK1 · SLC4A1 · SPTA1 · KCNN4 &nbsp;|&nbsp; 320 patients · Seeds 2582–2589
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 10 }}>
            {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{ background: tab === t ? accent : card, color: tab === t ? '#fff' : '#94a3b8', border: 'none', borderRadius: 6, padding: '8px 18px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400 }}>
              {t}
            </button>
          ))}
        </div>

        {loading && <div style={{ color: '#94a3b8', padding: 20 }}>Loading…</div>}
        {err && <div style={{ color: '#ef4444', padding: 20 }}>Error: {err}</div>}

        {/* ── OVERVIEW ── */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 genes × 40" />
              <MetricCard label="Genes" value={overview.n_genes} sub="Haemolytic Anemia" />
              <MetricCard label="Seeds" value={overview.seeds} />
              <MetricCard label="Avg Hgb (g/dL)" value={overview.aggregate_metrics?.avg_hgb_g_dl ?? '—'} sub="steady state" />
              <MetricCard label="Transfusion Dep." value={`${overview.aggregate_metrics?.transfusion_pct ?? '—'}%`} warn={overview.aggregate_metrics?.transfusion_pct > 30} />
              <MetricCard label="Splenomegaly" value={`${overview.aggregate_metrics?.splenomegaly_pct ?? '—'}%`} />
            </div>

            <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <h3 style={{ color: '#f87171', margin: '0 0 12px' }}>Disease Classes</h3>
              {overview.disease_classes?.map((dc, i) => (
                <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #334155', fontSize: 13 }}>
                  <GeneChip gene={dc.split(' — ')[0].trim()} />
                  <span style={{ marginLeft: 8, color: '#cbd5e1' }}>{dc.split(' — ').slice(1).join(' — ')}</span>
                </div>
              ))}
            </div>

            <div style={{ background: card, borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#f87171', margin: '0 0 12px' }}>Clinical Pearls</h3>
              {overview.clinical_pearls?.map((pearl, i) => (
                <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid #334155', fontSize: 13, color: '#cbd5e1', lineHeight: 1.5 }}>
                  <span style={{ color: '#fbbf24', marginRight: 8 }}>▸</span>{pearl}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── GENE TABLE ── */}
        {tab === 'Gene Table' && breakdown && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1e3a5f' }}>
                  {['Gene', 'Locus', 'Inheritance', 'Avg Hgb', 'Retic %', 'Transfusion %', 'Splenomegaly %', 'Splenectomy %', 'Iron Overload %'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#7dd3fc', borderBottom: '1px solid #334155' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.gene_breakdowns?.map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#1e293b' : '#1a2540', cursor: 'pointer' }}
                    onClick={() => setSelGene(selGene === g.gene ? null : g.gene)}>
                    <td style={{ padding: '8px 10px', fontWeight: 700 }}><GeneChip gene={g.gene} /></td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8', fontSize: 11 }}>{g.inheritance?.split(';')[0]}</td>
                    <td style={{ padding: '8px 10px', color: '#38bdf8' }}>{g.avg_hgb_g_dl}</td>
                    <td style={{ padding: '8px 10px', color: '#34d399' }}>{g.avg_reticulocyte_pct}%</td>
                    <td style={{ padding: '8px 10px', color: g.transfusion_pct > 30 ? '#f87171' : '#e2e8f0' }}>{g.transfusion_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{g.splenomegaly_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{g.splenectomy_pct}%</td>
                    <td style={{ padding: '8px 10px', color: g.iron_overload_pct > 20 ? '#fbbf24' : '#e2e8f0' }}>{g.iron_overload_pct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {selGene && breakdown.gene_breakdowns?.filter(g => g.gene === selGene).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginTop: 16 }}>
                <h3 style={{ color: GENE_COLORS[g.gene] || '#38bdf8', margin: '0 0 10px' }}>
                  {GENE_INFO[g.gene]?.full || g.gene}
                </h3>
                <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8 }}>
                  <strong style={{ color: '#e2e8f0' }}>Locus:</strong> {g.locus} &nbsp;|&nbsp;
                  <strong style={{ color: '#e2e8f0' }}>Size:</strong> {g.protein_size} &nbsp;|&nbsp;
                  <strong style={{ color: '#e2e8f0' }}>Inheritance:</strong> {g.inheritance?.split(';')[0]}
                </div>
                <div style={{ fontSize: 12, color: '#fbbf24', marginBottom: 8 }}>
                  <strong>Pathognomonic:</strong> {g.pathognomonic?.split(';')[0]}
                </div>
                <div style={{ fontSize: 12, color: '#86efac', marginBottom: 8 }}>
                  <strong>Treatment:</strong> {g.treatment?.split(';')[0]}
                </div>
                <div style={{ marginTop: 10 }}>
                  <strong style={{ fontSize: 12, color: '#e2e8f0' }}>Key Features:</strong>
                  <ul style={{ margin: '6px 0 0 16px', padding: 0 }}>
                    {g.key_features?.map((f, i) => (
                      <li key={i} style={{ fontSize: 11, color: '#cbd5e1', marginBottom: 4 }}>{f}</li>
                    ))}
                  </ul>
                </div>
                <div style={{ marginTop: 10 }}>
                  <strong style={{ fontSize: 12, color: '#e2e8f0' }}>Key DDx:</strong>
                  <ul style={{ margin: '6px 0 0 16px', padding: 0 }}>
                    {g.key_ddx?.map((d, i) => (
                      <li key={i} style={{ fontSize: 11, color: '#f87171', marginBottom: 4 }}>{d}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── CLINICAL ATLAS ── */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 16 }}>
            {breakdown.gene_breakdowns?.map(g => (
              <div key={g.gene} style={{ background: card, borderRadius: 8, padding: 16, borderTop: `3px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ fontSize: 11, color: '#64748b' }}>{g.locus} · {g.inheritance?.split(';')[0]?.trim()}</span>
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8 }}>{g.disease_category?.split(';')[0]}</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
                  <div style={{ background: '#0f172a', borderRadius: 4, padding: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: '#38bdf8' }}>{g.avg_hgb_g_dl}</div>
                    <div style={{ fontSize: 10, color: '#64748b' }}>Avg Hgb g/dL</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 4, padding: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: '#34d399' }}>{g.avg_reticulocyte_pct}%</div>
                    <div style={{ fontSize: 10, color: '#64748b' }}>Retic %</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 4, padding: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: g.transfusion_pct > 30 ? '#f87171' : '#e2e8f0' }}>{g.transfusion_pct}%</div>
                    <div style={{ fontSize: 10, color: '#64748b' }}>Transfusion dep.</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 4, padding: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: '#fbbf24' }}>{g.splenomegaly_pct}%</div>
                    <div style={{ fontSize: 10, color: '#64748b' }}>Splenomegaly</div>
                  </div>
                </div>
                <div style={{ fontSize: 11, color: '#fbbf24', background: '#1e1a00', borderRadius: 4, padding: '4px 8px', marginBottom: 6 }}>
                  ⚠ {g.pathognomonic?.split(';')[0]?.trim()?.slice(0, 120)}
                </div>
                <div style={{ fontSize: 11, color: '#86efac' }}>
                  Rx: {g.treatment?.split(';')[0]?.trim()?.slice(0, 100)}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── DEFINITIONS ── */}
        {tab === 'Definitions' && definitions && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12, marginBottom: 20 }}>
              {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => (
                <div key={gene} style={{ background: card, borderRadius: 8, padding: 14, borderLeft: `3px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#38bdf8', marginBottom: 4 }}>{gene}</div>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>{entry.locus} · {entry.protein_size} · {entry.inheritance}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>{entry.disease_name?.split(';')[0]}</div>
                  <div style={{ fontSize: 11, color: '#fbbf24', marginTop: 6 }}>
                    Hgb: {entry.hgb_g_dl_median} g/dL · Retic: {entry.reticulocyte_pct_median}% · Transfusion: {entry.transfusion_pct}%
                  </div>
                  {entry.splenectomy_ci && (
                    <div style={{ fontSize: 11, color: '#ef4444', fontWeight: 700, marginTop: 4 }}>
                      ⛔ SPLENECTOMY ABSOLUTELY CONTRAINDICATED
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div style={{ background: card, borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#f87171', margin: '0 0 12px' }}>Haemolytic Anemia Glossary</h3>
              {Object.entries(definitions.hemolytic_anemia_glossary || {}).map(([term, def]) => (
                <div key={term} style={{ marginBottom: 16, borderBottom: '1px solid #334155', paddingBottom: 12 }}>
                  <div style={{ fontWeight: 700, color: '#7dd3fc', marginBottom: 6, fontSize: 13 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>{def}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
