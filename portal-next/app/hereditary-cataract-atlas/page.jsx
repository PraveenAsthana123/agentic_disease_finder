'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cataract-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  CRYAA:  '#1a237e',  // deep indigo       — αA-crystallin, most common crystallin, PSC/zonular AD
  CRYAB:  '#6a1b9a',  // deep purple        — αB-crystallin, R120G MFM+DCM+cataract triad
  GJA8:   '#1565c0',  // deep blue          — Connexin 50, nuclear pulverulent, AD congenital
  GJA3:   '#0277bd',  // steel blue         — Connexin 46, cerulean/total nuclear, AD congenital
  MIP:    '#004d40',  // dark teal          — Aquaporin-0/AQP0, lamellar/zonular, AD
  EPHA2:  '#e65100',  // burnt orange       — EphA2 RTK, cortical/PSC, AD/AR, GWAS risk
  NHS:    '#b71c1c',  // deep red           — Nance-Horan Syndrome, XLR nuclear + dental anomalies
  FYCO1:  '#33691e',  // dark olive green   — FYCO1 autophagy adaptor, AR founder mutations
};

const GENE_INFO = {
  CRYAA:  { full: 'CRYAA / αA-Crystallin / HspB4 / 173aa', locus: '21q22.3', size: '173 aa / 20 kDa', inh: 'AD/AR', disease: 'Cataract-9 — MOST COMMON crystallin cataract gene; AD: posterior subcapsular or zonular-lamellar; R116C most common AD variant; AR biallelic: CATARACT + MICROPHTHALMIA + IRIS COLOBOMA triad pathognomonic' },
  CRYAB:  { full: 'CRYAB / αB-Crystallin / HspB5 / 175aa', locus: '11q23.1', size: '175 aa / 20 kDa', inh: 'AD/AR', disease: 'Cataract-16 + MFM2/DCM — R120G PATHOGNOMONIC: desmin-related myopathy + dilated cardiomyopathy + posterior cataract TRIAD; fibrillar protein aggregates in cardiac/skeletal muscle biopsy pathognomonic; ubiquitous (DDx CRYAA lens-only)' },
  GJA8:   { full: 'GJA8 / Connexin-50 / Cx50 / 440aa', locus: '1q21.1', size: '440 aa / 50 kDa', inh: 'AD', disease: 'Cataract-1 — NUCLEAR PULVERULENT / total nuclear pathognomonic; lens fiber cell gap junction; W45S most common (South Asian), P88S (European); congenital onset; no systemic disease; dominant-negative Cx50' },
  GJA3:   { full: 'GJA3 / Connexin-46 / Cx46 / 435aa', locus: '13q12.11', size: '435 aa / 46 kDa', inh: 'AD', disease: 'Cataract-14 — CERULEAN (blue-dot) or total nuclear pathognomonic; Cx46 only in lens fiber cells (not epithelium); N188T most common; Ca2+ homeostasis critical; GJA3+GJA8 double KO = complete nuclear opacity' },
  MIP:    { full: 'MIP / Aquaporin-0 / AQP0 / 263aa', locus: '12q13.3', size: '263 aa / 28 kDa', inh: 'AD', disease: 'Cataract-15 — LAMELLAR (ZONULAR) pathognomonic: discrete shell of opacity within clear lens; most abundant lens fiber membrane protein (45%); dual water channel + cell adhesion function; T138R most common; incomplete penetrance' },
  EPHA2:  { full: 'EPHA2 / Ephrin Receptor A2 / 976aa', locus: '1p36.13', size: '976 aa / 108 kDa', inh: 'AD/AR', disease: 'Cataract-6 — CORTICAL WEDGE/SPOKE or PSC; most common GWAS gene for age-related cortical cataract; R721Q most prevalent European AD variant; regulates lens epithelial-to-fiber differentiation (EphA2-EphrinA5); AR biallelic: congenital nuclear' },
  NHS:    { full: 'NHS / Nance-Horan Syndrome Protein / 1630aa', locus: 'Xp22.13', size: '1630 aa / 177 kDa', inh: 'XLR', disease: 'Nance-Horan Syndrome — DENSE NUCLEAR CATARACT IN HEMIZYGOUS MALES AT BIRTH PATHOGNOMONIC; DENTAL: supplemental maxillary incisors + screwdriver-shaped teeth; carrier females: posterior sutural opacities only; intellectual disability 30-50%' },
  FYCO1:  { full: 'FYCO1 / FYVE-Coiled-Coil Autophagy Adaptor / 1478aa', locus: '3p21.31', size: '1478 aa / 167 kDa', inh: 'AR', disease: 'Cataract-18 — TOTAL NUCLEAR AR; most common AR cataract in consanguineous South Asian/Middle Eastern/East Asian families; p.Gln762Ter Pakistani founder; autophagy adaptor (crystallin clearance mechanism); no systemic disease' },
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

function MetricCard({ label, value, sub }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', minWidth: 140, flex: '1 1 140px' }}>
      <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#f1f5f9', fontSize: 22, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function CataractAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' || tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    setLoading(true);
    setError(null);
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => {
        if (ep === 'overview') setOverview(d);
        else if (ep === 'breakdown') setBreakdown(d);
        else setDefinitions(d);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  const data = tab === 'Definitions' ? definitions
    : tab === 'Gene Table' || tab === 'Clinical Atlas' ? breakdown
    : overview;

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', fontFamily: 'system-ui,sans-serif', padding: '24px 32px' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ color: '#60a5fa', fontSize: 13, marginBottom: 4 }}>🧬 Hereditary Cataract Atlas</div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: '#f8fafc' }}>
          Hereditary-Cataract-Atlas
        </h1>
        <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 6 }}>
          Complete 8-Gene Reference — CRYAA · CRYAB · GJA8 · GJA3 · MIP · EPHA2 · NHS · FYCO1
        </div>
        <div style={{ color: '#64748b', fontSize: 12, marginTop: 4 }}>
          Crystallins (αA/αB) · Connexins (Cx50/Cx46) · Aquaporin-0 · EphA2 RTK · Nance-Horan XLR · FYCO1 AR Autophagy
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#3b82f6' : '#1e293b',
            color: tab === t ? '#fff' : '#94a3b8',
            border: 'none', borderRadius: 6, padding: '8px 18px',
            cursor: 'pointer', fontWeight: tab === t ? 700 : 400, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#60a5fa' }}>Loading…</div>}
      {error && <div style={{ color: '#f87171' }}>Error: {error}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 28 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub={`seeds ${overview.seeds}`} />
            <MetricCard label="Genes Covered" value={overview.genes_covered?.length} sub="8 hereditary cataract genes" />
            <MetricCard label="Dense Cataract" value={`${overview.aggregate_metrics?.dense_cataract_pct}%`} sub="visually significant opacity" />
            <MetricCard label="Surgery Rate" value={`${overview.aggregate_metrics?.surgery_performed_pct}%`} sub="phacoemulsification/lensectomy" />
            <MetricCard label="Amblyopia" value={`${overview.aggregate_metrics?.amblyopia_pct}%`} sub="despite treatment" />
            <MetricCard label="Systemic" value={`${overview.aggregate_metrics?.systemic_involvement_pct}%`} sub="CRYAB cardiac/muscle or NHS dental/ID" />
            <MetricCard label="BCVA <6/60" value={`${overview.aggregate_metrics?.bcva_worse_than_6_60_pct}%`} sub="legally blind / uncorrectable" />
            <MetricCard label="Consanguinity" value={`${overview.aggregate_metrics?.consanguineous_family_pct}%`} sub="FYCO1-enriched AR families" />
          </div>

          <h2 style={{ color: '#e2e8f0', fontSize: 17, marginBottom: 16 }}>Gene Summary</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(420px,1fr))', gap: 14 }}>
            {Object.values(overview.gene_summary || {}).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, padding: 16, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ color: '#94a3b8', fontSize: 11 }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
                </div>
                <div style={{ color: '#f1f5f9', fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{g.disease_category}</div>
                <div style={{ color: '#64748b', fontSize: 11, marginBottom: 8 }}>
                  Morphology: <span style={{ color: '#94a3b8' }}>{g.morphology}</span> ·
                  Onset: <span style={{ color: '#94a3b8' }}>{g.onset_age}</span> ·
                  Family: <span style={{ color: '#94a3b8' }}>{g.gene_family}</span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, fontSize: 11 }}>
                  <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4 }}>
                    Dense {g.dense_pct}%
                  </span>
                  <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4 }}>
                    Surgery {g.surgery_pct}%
                  </span>
                  <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4 }}>
                    Amblyopia {g.amblyopia_pct}%
                  </span>
                  <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4 }}>
                    BCVA&lt;6/60 {g.bcva_poor_pct}%
                  </span>
                  <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4 }}>
                    Avg dx {g.avg_age_dx_years}y
                  </span>
                  {g.consanguineous_pct > 0 && (
                    <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4, color: '#86efac' }}>
                      Consanguinity {g.consanguineous_pct}%
                    </span>
                  )}
                </div>
                {g.systemic_involvement && (
                  <div style={{ marginTop: 6, fontSize: 11, color: '#fbbf24', fontWeight: 600 }}>
                    ⚠ Systemic involvement: {g.gene === 'CRYAB' ? 'Cardiomyopathy + Myopathy (R120G)' : 'Dental anomalies + Intellectual disability (NHS/XLR)'}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1e293b' }}>
                {['Gene', 'Locus', 'Size', 'Inh', 'Morphology', 'Onset', 'Dense%', 'Surgery%', 'Amblyopia%', 'BCVA<6/60%', 'Avg Dx Age', 'Systemic', 'Urgency'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', borderBottom: '1px solid #334155' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(breakdown.gene_breakdowns || []).map(g => (
                <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                  <td style={{ padding: '8px 10px' }}><GeneChip gene={g.gene} /></td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.protein_size}</td>
                  <td style={{ padding: '8px 10px', color: '#fbbf24', fontWeight: 700 }}>{g.inheritance?.split(' ')[0]}</td>
                  <td style={{ padding: '8px 10px', color: '#e2e8f0', maxWidth: 160 }}>{g.morphology}</td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.onset_age}</td>
                  <td style={{ padding: '8px 10px', color: '#f1f5f9', fontWeight: 600 }}>{g.dense_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#f1f5f9' }}>{g.surgery_pct}%</td>
                  <td style={{ padding: '8px 10px', color: g.amblyopia_pct > 30 ? '#f87171' : '#f1f5f9' }}>{g.amblyopia_pct}%</td>
                  <td style={{ padding: '8px 10px', color: g.bcva_poor_pct > 20 ? '#f87171' : '#f1f5f9' }}>{g.bcva_poor_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.avg_age_dx_years}y</td>
                  <td style={{ padding: '8px 10px', color: g.systemic_involvement ? '#fbbf24' : '#4ade80' }}>{g.systemic_involvement ? '⚠ Yes' : 'No'}</td>
                  <td style={{ padding: '8px 10px', color: g.surgical_urgency?.startsWith('Extremely') ? '#f87171' : g.surgical_urgency?.startsWith('Urgent') ? '#fbbf24' : '#94a3b8', fontSize: 11 }}>{g.surgical_urgency}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {(breakdown.gene_breakdowns || []).map(g => (
            <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, padding: 20, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                <GeneChip gene={g.gene} />
                <span style={{ color: '#e2e8f0', fontWeight: 700, fontSize: 15 }}>{g.gene_family}</span>
                <span style={{ color: '#64748b', fontSize: 11 }}>{g.locus} · {g.protein_size} · {g.inheritance?.split(';')[0]}</span>
              </div>
              <div style={{ color: '#cbd5e1', fontSize: 13, marginBottom: 8 }}>
                <strong style={{ color: '#f1f5f9' }}>Disease:</strong> {g.disease_category}
              </div>
              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#60a5fa', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Morphology / Onset / Urgency</div>
                <div style={{ color: '#94a3b8', fontSize: 12 }}>
                  <strong>Morphology:</strong> {g.morphology} &nbsp;·&nbsp;
                  <strong>Onset:</strong> {g.onset_age} &nbsp;·&nbsp;
                  <strong>Urgency:</strong> <span style={{ color: g.surgical_urgency?.startsWith('Extremely') ? '#f87171' : g.surgical_urgency?.startsWith('Urgent') ? '#fbbf24' : '#94a3b8' }}>{g.surgical_urgency}</span>
                </div>
              </div>
              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#60a5fa', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Pathognomonic Features</div>
                <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.6 }}>{g.pathognomonic}</div>
              </div>
              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#60a5fa', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Key Features</div>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {(g.key_features || []).map((f, i) => (
                    <li key={i} style={{ color: '#94a3b8', fontSize: 12, marginBottom: 2 }}>{f}</li>
                  ))}
                </ul>
              </div>
              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#fbbf24', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Key DDx</div>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {(g.key_ddx || []).map((d, i) => (
                    <li key={i} style={{ color: '#94a3b8', fontSize: 12, marginBottom: 2 }}>{d}</li>
                  ))}
                </ul>
              </div>
              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#4ade80', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Treatment</div>
                <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>{g.treatment?.slice(0, 500)}…</div>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, fontSize: 11 }}>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4 }}>n={g.n_patients}</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4 }}>Dense {g.dense_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4 }}>Surgery {g.surgery_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4 }}>Amblyopia {g.amblyopia_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4 }}>BCVA&lt;6/60 {g.bcva_poor_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4 }}>Avg dx {g.avg_age_dx_years}y</span>
                {g.systemic_involvement && (
                  <span style={{ background: '#7c2d12', padding: '3px 10px', borderRadius: 4, color: '#fbbf24' }}>
                    ⚠ Systemic
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h2 style={{ color: '#e2e8f0', fontSize: 17, marginBottom: 16 }}>Gene Entries</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginBottom: 28 }}>
            {Object.values(definitions.gene_entries || {}).map(e => (
              <div key={e.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[e.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <GeneChip gene={e.gene} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{e.locus} · {e.protein_size} · {e.inheritance}</span>
                </div>
                <div style={{ color: '#f1f5f9', fontWeight: 700, fontSize: 14, marginBottom: 4 }}>{e.disease_name}</div>
                <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.6, marginBottom: 6 }}>
                  <strong style={{ color: '#60a5fa' }}>Pathway:</strong> {e.disease_pathway?.slice(0, 400)}…
                </div>
                <div style={{ color: '#e2e8f0', fontSize: 12, lineHeight: 1.6, marginBottom: 6 }}>
                  <strong style={{ color: '#f59e0b' }}>Pathognomonic:</strong> {e.pathognomonic?.slice(0, 350)}…
                </div>
                <div style={{ color: '#94a3b8', fontSize: 12 }}>
                  <strong>Morphology:</strong> {e.morphology} ·
                  <strong> Onset:</strong> {e.onset_age} ·
                  <strong> Urgency:</strong> {e.surgical_urgency} ·
                  <strong> Family:</strong> {e.gene_family}
                </div>
              </div>
            ))}
          </div>

          <h2 style={{ color: '#e2e8f0', fontSize: 17, marginBottom: 16 }}>Cataract Genetics Glossary</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {Object.entries(definitions.cataract_glossary || {}).map(([title, text]) => (
              <div key={title} style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
                <div style={{ color: '#60a5fa', fontWeight: 700, fontSize: 13, marginBottom: 8 }}>{title}</div>
                <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.7 }}>{text}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
