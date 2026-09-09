'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-vitreoretinopathy-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  COL2A1:  '#1a237e',  // deep indigo       — Stickler Type 1, optically empty vitreous, most common
  COL11A1: '#1565c0',  // deep blue          — Stickler Type 2, fibrillar vitreous, SNHL prominent
  VCAN:    '#004d40',  // dark teal          — Wagner syndrome, pure ocular, synchysis + veils
  FZD4:    '#880e4f',  // deep rose          — FEVR Type 1, most common FEVR gene, Wnt receptor
  NDP:     '#b71c1c',  // deep red           — Norrie Disease, congenital blindness boys, XLR
  LRP5:    '#e65100',  // burnt orange       — FEVR4/OPPG, pseudoglioma + low bone mass AR
  TSPAN12: '#4a148c',  // deep purple        — FEVR Type 5, incomplete penetrance, Wnt scaffold
  ZNF408:  '#33691e',  // dark olive green   — FEVR Type 6, PFV/PHPV, persistent hyaloid
};

const GENE_INFO = {
  COL2A1:  { full: 'COL2A1 / Collagen Type II α1 / 1487aa', locus: '12q13.11', size: '1487 aa / 141 kDa', inh: 'AD', disease: 'Stickler Syndrome Type 1 — MOST COMMON inherited vitreoretinopathy; TYPE 1 OPTICALLY EMPTY VITREOUS PATHOGNOMONIC on biomicroscopy; systemic: midface hypoplasia + cleft palate + SNHL + arthropathy; retinal detachment 30-70% lifetime; prophylactic 360° retinopexy mandatory' },
  COL11A1: { full: 'COL11A1 / Collagen Type XI α1 / 1837aa', locus: '1p21.1', size: '1837 aa / 186 kDa', inh: 'AD', disease: 'Stickler Syndrome Type 2 — TYPE 2 FIBRILLAR BEADED VITREOUS PATHOGNOMONIC (DDx COL2A1 optically empty); more severe/earlier SNHL than Type 1; Marshall syndrome overlap (same gene, exon-skip splice → ectodermal + intracranial calcifications)' },
  VCAN:    { full: 'VCAN / Versican / 3396aa', locus: '5q14.2', size: '3396 aa / 370 kDa', inh: 'AD (splice-site)', disease: 'Wagner Syndrome — PURE OCULAR vitreoretinopathy; VITREOUS SYNCHYSIS + FIBROVASCULAR VEILS PATHOGNOMONIC; NO systemic features (DDx Stickler: no deafness, no arthropathy); progressive choroidal atrophy + pigmentary retinopathy; intronic splice-site variants (exon 7/8)' },
  FZD4:    { full: 'FZD4 / Frizzled-4 / 537aa', locus: '11q14.2', size: '537 aa / 57 kDa', inh: 'AD', disease: 'FEVR Type 1 (EVR1) — MOST COMMON FEVR GENE (~40-50%); AVASCULAR PERIPHERAL RETINA ON FFA PATHOGNOMONIC; term babies (DDx ROP); Stage 1-5 spectrum; Wnt receptor for Norrin; family FFA screening mandatory — asymptomatic carriers treated' },
  NDP:     { full: 'NDP / Norrin / 133aa', locus: 'Xp11.4', size: '133 aa / 15 kDa', inh: 'XLR', disease: 'Norrie Disease / FEVR Type 2 — CONGENITAL BILATERAL LEUKOCORIA IN BOYS PATHOGNOMONIC; born blind (pseudoglioma); SNHL 35%; intellectual disability/psychiatric 25-35%; carrier females: mild FEVR-like FFA changes; Norrin = upstream FZD4 ligand; most severe in atlas' },
  LRP5:    { full: 'LRP5 / LDL Receptor-Related Protein 5 / 1615aa', locus: '11q13.2', size: '1615 aa / 179 kDa', inh: 'AD/AR', disease: 'FEVR Type 4 (EVR4) AD-LOF / Osteoporosis-Pseudoglioma (OPPG) AR-biallelic-LOF; PSEUDOGLIOMA + SEVERE JUVENILE OSTEOPOROSIS PATHOGNOMONIC for OPPG; AD GOF → HIGH BONE DENSITY (no eye disease); same gene, three opposite phenotypes' },
  TSPAN12: { full: 'TSPAN12 / Tetraspanin-12 / 305aa', locus: '7q31.31', size: '305 aa / 32 kDa', inh: 'AD (incomplete penetrance)', disease: 'FEVR Type 5 (EVR5) — INCOMPLETE PENETRANCE ~50%: obligate carrier parent may have normal FFA; milder FEVR phenotype; TSPAN12 scaffold increases Norrin-FZD4 binding 10-fold; avascular peripheral retina; fibrovascular proliferation in subset' },
  ZNF408:  { full: 'ZNF408 / Zinc Finger Protein 408 / 720aa', locus: '11p11.2', size: '720 aa / 81 kDa', inh: 'AD', disease: 'FEVR Type 6 (EVR6) / PFV — PERSISTENT FETAL VASCULATURE + AVASCULAR PERIPHERAL RETINA COMBINATION PATHOGNOMONIC; failure of hyaloid vasculature regression; retrolental fibrovascular stalk; most phenotypically variable FEVR gene; distinct from Wnt-pathway FEVR genes' },
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

export default function VitreoretinopathyAtlasPage() {
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
        <div style={{ color: '#60a5fa', fontSize: 13, marginBottom: 4 }}>🧬 Hereditary Vitreoretinopathy Atlas</div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: '#f8fafc' }}>
          Hereditary-Vitreoretinopathy-Atlas
        </h1>
        <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 6 }}>
          Complete 8-Gene Reference — COL2A1 · COL11A1 · VCAN · FZD4 · NDP · LRP5 · TSPAN12 · ZNF408
        </div>
        <div style={{ color: '#64748b', fontSize: 12, marginTop: 4 }}>
          Stickler Syndrome (Type 1 & 2) · Wagner Syndrome · FEVR (Types 1, 2, 4, 5, 6) · Norrie Disease · OPPG · PFV/PHPV
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
            <MetricCard label="Retinal Detachment" value={`${overview.aggregate_metrics.retinal_detachment_pct}%`} sub="lifetime risk" />
            <MetricCard label="Avascular Retina" value={`${overview.aggregate_metrics.avascular_peripheral_retina_pct}%`} sub="FEVR/NDP/LRP5/ZNF408" />
            <MetricCard label="Hearing Loss" value={`${overview.aggregate_metrics.hearing_loss_pct}%`} sub="Stickler/Norrie" />
            <MetricCard label="Prophylactic Laser" value={`${overview.aggregate_metrics.prophylactic_laser_pct}%`} sub="retinopexy/FEVR laser" />
            <MetricCard label="Surgical Intervention" value={`${overview.aggregate_metrics.surgical_intervention_pct}%`} sub="vitreoretinal surgery" />
            <MetricCard label="VA < 6/60" value={`${overview.aggregate_metrics.bcva_worse_than_6_60_pct}%`} sub="legal blindness" />
          </div>

          <h3 style={{ color: '#60a5fa', marginBottom: 12 }}>Gene Summary</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 12 }}>
            {overview.gene_summary && Object.values(overview.gene_summary).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
                </div>
                <div style={{ color: '#f1f5f9', fontSize: 13, marginBottom: 4, fontWeight: 600 }}>{g.disease_category.split('—')[0].trim()}</div>
                <div style={{ color: '#60a5fa', fontSize: 11, marginBottom: 6 }}>Vitreous: {g.vitreous_phenotype}</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4, fontSize: 11, color: '#f87171' }}>RD {g.rd_pct}%</span>
                  {g.avascular_pct > 50 && <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4, fontSize: 11, color: '#fbbf24' }}>Avascular {g.avascular_pct}%</span>}
                  {g.hearing_loss_pct > 5 && <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4, fontSize: 11, color: '#a78bfa' }}>Hearing {g.hearing_loss_pct}%</span>}
                  <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4, fontSize: 11, color: '#34d399' }}>Dx {g.avg_age_dx_years}y</span>
                  {g.systemic_involvement && <span style={{ background: '#7c3aed', padding: '2px 8px', borderRadius: 4, fontSize: 11, color: '#fff' }}>Systemic</span>}
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
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Inheritance</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Disease</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Vitreous</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Onset</th>
                <th style={{ padding: '10px 12px', textAlign: 'center' }}>RD%</th>
                <th style={{ padding: '10px 12px', textAlign: 'center' }}>Avascular%</th>
                <th style={{ padding: '10px 12px', textAlign: 'center' }}>Hearing%</th>
                <th style={{ padding: '10px 12px', textAlign: 'center' }}>Blind%</th>
              </tr>
            </thead>
            <tbody>
              {breakdown.gene_breakdowns.map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b', borderBottom: '1px solid #334155' }}>
                  <td style={{ padding: '10px 12px' }}><GeneChip gene={g.gene} /></td>
                  <td style={{ padding: '10px 12px', color: '#94a3b8' }}>{g.locus}</td>
                  <td style={{ padding: '10px 12px', color: '#94a3b8' }}>{g.inheritance.split(';')[0]}</td>
                  <td style={{ padding: '10px 12px', color: '#f1f5f9', maxWidth: 180 }}>{g.disease_category.split('—')[0].trim()}</td>
                  <td style={{ padding: '10px 12px', color: '#60a5fa', fontSize: 11, maxWidth: 160 }}>{g.vitreous_phenotype}</td>
                  <td style={{ padding: '10px 12px', color: '#a78bfa', fontSize: 11 }}>{g.onset_age}</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center', color: '#f87171', fontWeight: 700 }}>{g.rd_pct}%</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center', color: '#fbbf24' }}>{g.avascular_pct}%</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center', color: '#a78bfa' }}>{g.hearing_loss_pct}%</td>
                  <td style={{ padding: '10px 12px', textAlign: 'center', color: '#ef4444', fontWeight: 700 }}>{g.blind_pct}%</td>
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

              <div style={{ marginBottom: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <span style={{ background: '#0f172a', padding: '2px 8px', borderRadius: 4, fontSize: 11, color: '#60a5fa' }}>
                  Vitreous: {g.vitreous_phenotype}
                </span>
                {g.systemic_involvement && <span style={{ background: '#7c3aed', padding: '2px 8px', borderRadius: 4, fontSize: 11, color: '#fff' }}>Systemic</span>}
                {g.hearing_loss_type !== 'None' && <span style={{ background: '#1d4ed8', padding: '2px 8px', borderRadius: 4, fontSize: 11, color: '#fff' }}>Hearing: {g.hearing_loss_type}</span>}
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#60a5fa', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>PATHOGNOMONIC</div>
                <div style={{ color: '#e2e8f0', fontSize: 13 }}>{g.pathognomonic.slice(0, 380)}{g.pathognomonic.length > 380 ? '…' : ''}</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#34d399', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>TREATMENT</div>
                <div style={{ color: '#94a3b8', fontSize: 12 }}>{g.treatment.slice(0, 300)}{g.treatment.length > 300 ? '…' : ''}</div>
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
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#f87171' }}>RD {g.rd_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fbbf24' }}>Avascular {g.avascular_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#ef4444' }}>Blind {g.blind_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#a78bfa' }}>Laser {g.laser_pct}%</span>
                <span style={{ background: '#0f172a', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#64748b' }}>RD risk: {g.retinal_detachment_risk_pct}</span>
                {g.prophylactic_retinopexy_indicated && <span style={{ background: '#1d4ed8', padding: '3px 10px', borderRadius: 4, fontSize: 11, color: '#fff' }}>Prophylactic laser</span>}
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
              <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 6 }}><b style={{ color: '#60a5fa' }}>Pathway:</b> {entry.disease_pathway.slice(0, 280)}…</div>
              <div style={{ color: '#e2e8f0', fontSize: 12 }}><b style={{ color: '#fbbf24' }}>Pathognomonic:</b> {entry.pathognomonic.slice(0, 280)}…</div>
            </div>
          ))}

          {/* Glossary */}
          <h3 style={{ color: '#60a5fa', marginTop: 8 }}>Vitreoretinopathy Glossary</h3>
          {Object.entries(definitions.vitreoretinopathy_glossary || {}).map(([term, text]) => (
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
