'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-corneal-dystrophy-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  TGFBI:   '#6a1b9a',  // deep purple    — multiple stromal/Bowman, most common gene, variant-specific
  SLC4A11: '#0d47a1',  // navy           — CHED2, congenital endothelial, AR, bilateral at birth
  ZEB1:    '#1b5e20',  // dark green     — PPCD3 most common PPCD, endothelial metaplasia
  OVOL2:   '#004d40',  // dark teal      — PPCD1 regulatory variants, promoter sequencing needed
  TCF4:    '#bf360c',  // burnt sienna   — Fuchs FECD3 most common adult CD, CTG18.1 repeat
  COL8A2:  '#880e4f',  // deep rose      — Fuchs FECD1 early-onset L450W Q455K
  VSX1:    '#e65100',  // burnt orange   — keratoconus KTCN1 + PPCD2, cross-linking halts
  KRT12:   '#37474f',  // blue-grey      — Meesmann epithelial, microcysts, mild prognosis
};

const GENE_INFO = {
  TGFBI:   { full: 'TGFBI / keratoepithelin / 683aa', locus: '5q31.1', size: '683 aa / 68 kDa', inh: 'AD', disease: 'Multiple TGFBI-related dystrophies — variant-specific: R124C→LCD-I (lattice/amyloid), R555W→GCD1 (granular/hyaline), R124H→Avellino (mixed), R555Q→Thiel-Behnke (honeycomb), R124L→Reis-Bücklers (Bowman); most common corneal dystrophy gene worldwide; PTK for anterior; recurs in graft' },
  SLC4A11: { full: 'SLC4A11 / NaBC1 / 891aa', locus: '20p13', size: '891 aa / 99 kDa', inh: 'AR (biallelic)', disease: 'CHED2 — Congenital Hereditary Endothelial Dystrophy type 2; BILATERAL DIFFUSE GROUND-GLASS CORNEAL OPACITY AT BIRTH PATHOGNOMONIC; nystagmus + photophobia in neonates; normal IOP (DDx from congenital glaucoma); DMEK/DSAEK curative; Harboyan syndrome (CHED2 + SNHL) in some' },
  ZEB1:    { full: 'ZEB1 / ZFHX1B / 1124aa', locus: '10p11.22', size: '1124 aa / 125 kDa', inh: 'AD (haploinsufficiency)', disease: 'PPCD3 — most common PPCD gene; band-like vesicular lesions on endothelium PATHOGNOMONIC; iridocorneal adhesions 25-30% → secondary glaucoma; bilateral (DDx ICE syndrome unilateral); endothelial-to-epithelial metaplasia; early-onset Fuchs association in carriers' },
  OVOL2:   { full: 'OVOL2 / OVO-like ZF2 / 323aa', locus: '20p13', size: '323 aa / 36 kDa', inh: 'AD (regulatory region)', disease: 'PPCD1 — regulatory/promoter variants (NOT coding): standard exome misses PPCD1; promoter/5-UTR sequencing mandatory; clinically identical to PPCD3; iridocorneal adhesions possible; bilateral; conservative management often adequate; DMEK/DSAEK for oedema' },
  TCF4:    { full: 'TCF4 / E2-2 / ITF2 / 667aa', locus: '18q21.2', size: '667 aa / 73 kDa', inh: 'AD (CTG18.1 repeat)', disease: 'FECD3 — Fuchs endothelial type 3; most common hereditary adult corneal dystrophy; CTG18.1 trinucleotide repeat >40 in 79% of Fuchs; DESCEMET GUTTAE PATHOGNOMONIC; morning worsening (diurnal variation); DMEK gold standard; RNA gain-of-function; female:male 3:1; leading corneal transplant indication worldwide' },
  COL8A2:  { full: 'COL8A2 / collagen VIII α2 / 703aa', locus: '1p34.3', size: '703 aa / 78 kDa', inh: 'AD', disease: 'FECD1 — early-onset Fuchs; guttae onset <40 years PATHOGNOMONIC for COL8A2 (vs TCF4 >40y); L450W and Q455K most common variants; DMEK in 4th decade; ER stress UPR pathway; also PPCD2; does not recur in donor graft; cataract combined with DMEK if co-existing' },
  VSX1:    { full: 'VSX1 / Visual System Homeobox 1 / 365aa', locus: '20p11.21', size: '365 aa / 41 kDa', inh: 'AD (variable penetrance)', disease: 'KTCN1 — keratoconus; CORNEAL THINNING + IRREGULAR ASTIGMATISM + FLEISCHER RING (iron) PATHOGNOMONIC; Vogt striae; corneal hydrops emergency; cross-linking (CXL) halts progression >90%; DALK preferred for transplant; eye rubbing strictly contraindicated; VSX1 = 5-10% of familial keratoconus' },
  KRT12:   { full: 'KRT12 / Keratin 12 / 505aa', locus: '17q21.2', size: '505 aa / 55 kDa', inh: 'AD (dominant-negative)', disease: 'Meesmann epithelial corneal dystrophy (MECD); most common hereditary epithelial dystrophy; INTRAEPITHELIAL MICROCYSTS THROUGHOUT ENTIRE CORNEA from limbus to limbus PATHOGNOMONIC; PAS+ glycogen in cysts; onset neonatal/infancy; usually MILD — good visual prognosis; also KRT3 (12q13) causes same phenotype' },
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

export default function CornealDystrophyAtlasPage() {
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
        <div style={{ color: '#60a5fa', fontSize: 13, marginBottom: 4 }}>🧬 Hereditary Corneal Dystrophy Atlas</div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: '#f8fafc' }}>
          Hereditary-Corneal-Dystrophy-Atlas
        </h1>
        <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 6 }}>
          Complete 8-Gene Reference — TGFBI · SLC4A11 · ZEB1 · OVOL2 · TCF4 · COL8A2 · VSX1 · KRT12
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 24, borderBottom: '1px solid #1e293b', paddingBottom: 8 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#3b82f6' : 'transparent',
            color: tab === t ? '#fff' : '#94a3b8',
            border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontSize: 13, fontWeight: 600,
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#60a5fa' }}>Loading…</div>}
      {error && <div style={{ color: '#f87171' }}>Error: {error}</div>}

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40 cohorts" />
            <MetricCard label="Corneal Transplant" value={`${overview.aggregate_metrics.transplant_pct}%`} sub="DMEK/DSAEK/DALK/PKP" />
            <MetricCard label="Corneal Oedema" value={`${overview.aggregate_metrics.corneal_oedema_pct}%`} sub="endothelial failure" />
            <MetricCard label="Recurrent Erosions" value={`${overview.aggregate_metrics.recurrent_erosions_pct}%`} sub="epithelial/stromal" />
            <MetricCard label="Glaucoma" value={`${overview.aggregate_metrics.glaucoma_pct}%`} sub="PPCD iridocorneal" />
            <MetricCard label="PTK" value={`${overview.aggregate_metrics.ptk_pct}%`} sub="TGFBI anterior lesions" />
            <MetricCard label="CXL" value={`${overview.aggregate_metrics.cxl_pct}%`} sub="keratoconus cross-link" />
          </div>

          <h3 style={{ color: '#60a5fa', marginBottom: 12 }}>Gene Summary</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 12 }}>
            {overview.gene_summary && Object.values(overview.gene_summary).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
                </div>
                <div style={{ color: '#f1f5f9', fontSize: 13, marginBottom: 6, fontWeight: 600 }}>{g.disease_category.split('—')[0].trim()}</div>
                <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 8 }}>Layer: {g.corneal_layer}</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4, fontSize: 11, color: '#60a5fa' }}>Transplant {g.transplant_pct}%</span>
                  {g.erosion_pct > 10 && <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4, fontSize: 11, color: '#fbbf24' }}>Erosions {g.erosion_pct}%</span>}
                  {g.glaucoma_pct > 5 && <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4, fontSize: 11, color: '#f87171' }}>Glaucoma {g.glaucoma_pct}%</span>}
                  <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4, fontSize: 11, color: '#a78bfa' }}>Dx age {g.avg_age_dx_years}y</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Gene</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Locus</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Size</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Inheritance</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Layer</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Disease</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Onset</th>
                <th style={{ padding: '10px 12px', textAlign: 'center' }}>Transplant%</th>
                <th style={{ padding: '10px 12px', textAlign: 'center' }}>Erosions%</th>
                <th style={{ padding: '10px 12px', textAlign: 'center' }}>Glaucoma%</th>
              </tr>
            </thead>
            <tbody>
              {breakdown.gene_breakdowns.map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b', borderBottom: '1px solid #334155' }}>
                  <td style={{ padding: '10px 12px' }}><GeneChip gene={g.gene} /></td>
                  <td style={{ padding: '10px 12px', color: '#94a3b8' }}>{g.locus}</td>
                  <td style={{ padding: '10px 12px', color: '#94a3b8' }}>{g.protein_size}</td>
                  <td style={{ padding: '10px 12px', color: '#94a3b8' }}>{g.inheritance.split(';')[0]}</td>
                  <td style={{ padding: '10px 12px', color: '#60a5fa', fontSize: 11 }}>{g.corneal_layer.split('+')[0].trim()}</td>
                  <td style={{ padding: '10px 12px', color: '#f1f5f9', maxWidth: 220 }}>{g.disease_category.split('—')[0].trim()}</td>
                  <td style={{ padding: '10px 12px', color: '#a78bfa', fontSize: 11 }}>{g.onset_age}</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center', color: '#60a5fa', fontWeight: 700 }}>{g.transplant_pct}%</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center', color: '#fbbf24' }}>{g.erosion_pct}%</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center', color: '#f87171' }}>{g.glaucoma_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {breakdown.gene_breakdowns.map(g => (
            <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, padding: 20, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                <GeneChip gene={g.gene} />
                <span style={{ color: '#f1f5f9', fontWeight: 700, fontSize: 15 }}>{g.disease_category.split('—')[0].trim()}</span>
                <span style={{ color: '#64748b', fontSize: 12 }}>· {g.locus} · {g.protein_size} · {g.inheritance.split(';')[0]}</span>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#60a5fa', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>PATHOGNOMONIC</div>
                <div style={{ color: '#e2e8f0', fontSize: 13 }}>{g.pathognomonic.slice(0, 350)}{g.pathognomonic.length > 350 ? '…' : ''}</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#34d399', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>TREATMENT</div>
                <div style={{ color: '#94a3b8', fontSize: 12 }}>{g.treatment.slice(0, 280)}{g.treatment.length > 280 ? '…' : ''}</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#fbbf24', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>KEY FEATURES</div>
                <ul style={{ margin: 0, paddingLeft: 18, color: '#cbd5e1', fontSize: 12 }}>
                  {g.key_features.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#f87171', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>KEY DDx</div>
                <ul style={{ margin: 0, paddingLeft: 18, color: '#94a3b8', fontSize: 12 }}>
                  {g.key_ddx.map((d, i) => <li key={i}>{d}</li>)}
                </ul>
              </div>

              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#60a5fa' }}>n={g.n_patients}</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#a78bfa' }}>Transplant {g.transplant_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fbbf24' }}>Erosions {g.erosion_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#f87171' }}>Glaucoma {g.glaucoma_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#34d399' }}>Layer: {g.corneal_layer.split('+')[0]}</span>
                {g.recurrence_in_graft && <span style={{ background: '#7c3aed', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fff' }}>Recurs in graft</span>}
                {g.ptk_effective && <span style={{ background: '#1d4ed8', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fff' }}>PTK effective</span>}
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#64748b' }}>IC3D: {g.ic3d_category.split('(')[0].trim()}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Gene entries */}
          {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => (
            <div key={gene} style={{ background: '#1e293b', borderRadius: 10, padding: 18, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <GeneChip gene={gene} />
                <span style={{ color: '#94a3b8', fontSize: 12 }}>{entry.locus} · {entry.protein_size} · {entry.inheritance}</span>
              </div>
              <div style={{ color: '#f1f5f9', fontSize: 14, fontWeight: 700, marginBottom: 6 }}>{entry.disease_name}</div>
              <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 6 }}><b style={{ color: '#60a5fa' }}>Pathway:</b> {entry.disease_pathway.slice(0, 250)}…</div>
              <div style={{ color: '#e2e8f0', fontSize: 12 }}><b style={{ color: '#fbbf24' }}>Pathognomonic:</b> {entry.pathognomonic.slice(0, 250)}…</div>
            </div>
          ))}

          {/* Glossary */}
          <h3 style={{ color: '#60a5fa', marginTop: 8 }}>Corneal Dystrophy Glossary</h3>
          {Object.entries(definitions.corneal_dystrophy_glossary || {}).map(([term, text]) => (
            <div key={term} style={{ background: '#0f172a', borderRadius: 8, padding: 16, borderLeft: '3px solid #3b82f6' }}>
              <div style={{ color: '#60a5fa', fontSize: 13, fontWeight: 700, marginBottom: 6 }}>{term}</div>
              <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>{text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
