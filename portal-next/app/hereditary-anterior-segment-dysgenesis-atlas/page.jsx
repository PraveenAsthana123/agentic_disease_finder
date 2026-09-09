'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-anterior-segment-dysgenesis-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PITX2:  '#1a237e',  // deep indigo       — ARS1, NCC homeodomain TF, dental+umbilical
  FOXC1:  '#4a148c',  // deep purple        — ARS3, forkhead TF, cardiac defects
  PAX6:   '#1565c0',  // deep blue          — Aniridia, master eye TF, WAGR, foveal hypoplasia
  FOXE3:  '#0277bd',  // steel blue         — ASD2, lens vesicle separation, Peters anomaly
  B3GLCT: '#004d40',  // dark teal          — Peters Plus, TSR-glucosyltransferase, tetrad
  PXDN:   '#6a1b9a',  // medium purple      — ASD7, peroxidasin, collagen IV sulfilimine
  HCCS:   '#b71c1c',  // deep red           — MIDAS/MLS, XLD lethal males, Blaschko skin
  CYP1B1: '#e65100',  // burnt orange       — Peters type 2 / PCG, steroid metabolism, surgical emergency
};

const GENE_INFO = {
  PITX2:  { full: 'PITX2 / Paired-Like Homeodomain 2 / 317aa', locus: '4q25', size: '317 aa / 35 kDa', inh: 'AD', disease: 'Axenfeld-Rieger Syndrome Type 1 (ARS1) — IRIDOCORNEAL STRANDS + POSTERIOR EMBRYOTOXON + IRIS HYPOPLASIA TRIAD PATHOGNOMONIC; glaucoma 50% lifetime; DENTAL hypodontia/peg teeth + UMBILICAL redundant skin; NCC haploinsufficiency; most common ARS gene (50-70%)' },
  FOXC1:  { full: 'FOXC1 / Forkhead Box C1 / 553aa', locus: '6p25.3', size: '553 aa / 60 kDa', inh: 'AD', disease: 'Axenfeld-Rieger Syndrome Type 3 (ARS3) — same ARS triad as PITX2; CARDIAC SEPTAL DEFECTS 8% (DDx PITX2 <3%); FOXC1 DUPLICATION → iridogoniodysgenesis without systemic (MLPA required); SNHL subset; digenic FOXC1+PITX2 → more severe' },
  PAX6:   { full: 'PAX6 / Paired Box 6 / 422aa', locus: '11p13', size: '422 aa / 46 kDa', inh: 'AD', disease: 'Aniridia type II — BILATERAL NEAR-TOTAL IRIS ABSENCE + FOVEAL HYPOPLASIA + PENDULAR NYSTAGMUS TRIAD PATHOGNOMONIC; WAGR deletion (11p13): Wilms tumour 45-60% + GU anomalies + ID — CHROMOSOME MICROARRAY MANDATORY; progressive limbal stem cell keratopathy; glaucoma 30-50%' },
  FOXE3:  { full: 'FOXE3 / Forkhead Box E3 / 338aa', locus: '1p33', size: '338 aa / 38 kDa', inh: 'AD/AR', disease: 'Anterior Segment Dysgenesis type 2 (ASD2) — Peters anomaly with LENS TOUCH (anterior capsule-cornea adhesion) PATHOGNOMONIC; AR biallelic severe: PRIMARY APHAKIA + sclerocornea + microphthalmia + coloboma; AD hypomorphic: congenital cataract + microcornea; lens vesicle separation failure' },
  B3GLCT: { full: 'B3GLCT / Beta-3-Glucosyltransferase / 498aa', locus: '13q12.3', size: '498 aa / 57 kDa', inh: 'AR', disease: 'Peters Plus Syndrome — CORNEAL CLOUDING + SHORT STATURE + INTELLECTUAL DISABILITY + CLEFT LIP/PALATE = DIAGNOSTIC TETRAD PATHOGNOMONIC; IVS8+1G>A founder mutation 60% European alleles; TSR-glucosylation defect; Peters anomaly universal (100%); COMP misfolding → short stature; normal transferrin IEF (not classic CDG)' },
  PXDN:   { full: 'PXDN / Peroxidasin / 1479aa', locus: '2p25.3', size: '1479 aa / 165 kDa', inh: 'AR', disease: 'Anterior Segment Dysgenesis type 7 (ASD7) — BILATERAL CONGENITAL CORNEAL OPACIFICATION TO TOTAL SCLEROCORNEA AT BIRTH PATHOGNOMONIC; collagen IV sulfilimine crosslink deficiency (unique enzymatic mechanism); consanguineous Turkish/Pakistani/North African; no systemic features; Boston KPro for total sclerocornea' },
  HCCS:   { full: 'HCCS / Holocytochrome C-Type Synthase / 295aa', locus: 'Xp22.2', size: '295 aa / 33 kDa', inh: 'XLD', disease: 'MIDAS/MLS Syndrome — PERIOCULAR FACIAL LINEAR SKIN DEFECTS (Blaschko lines) + MICROPHTHALMIA/ANOPHTHALMIA IN FEMALES PATHOGNOMONIC; X-LINKED DOMINANT LETHAL IN HEMIZYGOUS MALES; cardiac defects 20%; CNS anomalies (ACC) 30%; holocytochrome c synthase (Complex III OXPHOS); X-inactivation mosaicism drives phenotype' },
  CYP1B1: { full: 'CYP1B1 / Cytochrome P450 1B1 / 543aa', locus: '2p22.2', size: '543 aa / 60 kDa', inh: 'AR', disease: 'Primary Congenital Glaucoma (PCG/GLC3A) + Peters Anomaly type 2 — BUPHTHALMOS + HAAB STRIAE + EXCESS TEARING AT BIRTH = PCG TRIAD PATHOGNOMONIC; SURGICAL EMERGENCY — goniotomy within days; most common PCG gene worldwide (up to 95% consanguineous); G61E (European), R368H (Middle Eastern/Indian); steroid/RA metabolism defect → NCC toxicity' },
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

export default function ASDAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const ep = tab === 'Overview' || tab === 'Gene Table' ? 'overview'
             : tab === 'Clinical Atlas' ? 'breakdown'
             : 'definitions';
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(data => {
        if (ep === 'overview') setOverview(data);
        else if (ep === 'breakdown') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  const dark = { background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', fontFamily: 'system-ui,sans-serif', padding: '24px 32px' };
  const tabBar = { display: 'flex', gap: 8, marginBottom: 24, borderBottom: '1px solid #334155', paddingBottom: 8 };
  const tabBtn = (active) => ({
    background: active ? '#3b82f6' : 'transparent',
    color: active ? '#fff' : '#94a3b8',
    border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontWeight: active ? 700 : 400,
  });

  return (
    <div style={dark}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', marginBottom: 4 }}>
          🧬 Hereditary Anterior Segment Dysgenesis Atlas
        </h1>
        <p style={{ color: '#94a3b8', fontSize: 14, margin: 0 }}>
          Complete 8-Gene Reference — PITX2 · FOXC1 · PAX6 · FOXE3 · B3GLCT · PXDN · HCCS · CYP1B1<br />
          320-Patient Aggregate Cohort (8×40) · Seeds 2390–2397 · Peters Anomaly · Axenfeld-Rieger · Aniridia · PCG
        </p>
      </div>

      {/* Gene chips */}
      <div style={{ marginBottom: 20, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
      </div>

      <div style={tabBar}>
        {TABS.map(t => (
          <button key={t} style={tabBtn(tab === t)} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#94a3b8' }}>Loading…</div>}
      {error && <div style={{ color: '#f87171' }}>Error: {error}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 genes × 40" />
            <MetricCard label="Glaucoma Risk" value={`${overview.aggregate_metrics?.glaucoma_pct ?? '--'}%`} sub="lifetime avg" />
            <MetricCard label="Corneal Opacity" value={`${overview.aggregate_metrics?.corneal_opacity_pct ?? '--'}%`} sub="across atlas" />
            <MetricCard label="Surgery Rate" value={`${overview.aggregate_metrics?.surgery_performed_pct ?? '--'}%`} sub="keratoplasty / goniotomy" />
            <MetricCard label="Systemic Involvement" value={`${overview.aggregate_metrics?.systemic_involvement_pct ?? '--'}%`} sub="dental/cardiac/CNS" />
            <MetricCard label="VA < 6/18" value={`${overview.aggregate_metrics?.va_worse_than_6_18_pct ?? '--'}%`} sub="visual impairment" />
          </div>

          <h3 style={{ color: '#93c5fd', marginBottom: 12 }}>Gene Summary</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            {overview.gene_summary && Object.values(overview.gene_summary).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 14, minWidth: 260, flex: '1 1 260px', borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ color: '#94a3b8', fontSize: 11 }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
                </div>
                <div style={{ color: '#e2e8f0', fontSize: 12, marginBottom: 6, lineHeight: 1.4 }}>
                  {g.disease_category}
                </div>
                <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>{g.onset_age}</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 6 }}>
                  <span style={{ color: '#fbbf24', fontSize: 11 }}>Glaucoma {g.glaucoma_pct}%</span>
                  <span style={{ color: '#60a5fa', fontSize: 11 }}>Corneal {g.corneal_opacity_pct}%</span>
                  <span style={{ color: '#34d399', fontSize: 11 }}>Surgery {g.surgery_pct}%</span>
                  <span style={{ color: '#f87171', fontSize: 11 }}>VA↓ {g.va_poor_pct}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && overview && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1e293b' }}>
                {['Gene','Locus','Size','Inh','Disease','Onset','Glaucoma %','Corneal %','Surgery %','VA↓ %','Systemic','Urgency'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', color: '#93c5fd', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overview.gene_summary && Object.values(overview.gene_summary).map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b', borderBottom: '1px solid #334155' }}>
                  <td style={{ padding: '8px 10px' }}><GeneChip gene={g.gene} /></td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.locus}</td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.protein_size}</td>
                  <td style={{ padding: '8px 10px', color: '#fbbf24', whiteSpace: 'nowrap' }}>{g.inheritance}</td>
                  <td style={{ padding: '8px 10px', color: '#e2e8f0', maxWidth: 200 }}>{g.disease_category}</td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.onset_age}</td>
                  <td style={{ padding: '8px 10px', color: '#fbbf24', textAlign: 'center' }}>{g.glaucoma_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#60a5fa', textAlign: 'center' }}>{g.corneal_opacity_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#34d399', textAlign: 'center' }}>{g.surgery_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#f87171', textAlign: 'center' }}>{g.va_poor_pct}%</td>
                  <td style={{ padding: '8px 10px', color: g.systemic_involvement ? '#f87171' : '#94a3b8' }}>{g.systemic_involvement ? 'Yes' : 'No'}</td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8', fontSize: 11 }}>{g.surgical_urgency}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ marginTop: 24 }}>
            <h3 style={{ color: '#93c5fd', marginBottom: 12 }}>Gene Reference Detail</h3>
            {Object.entries(GENE_INFO).map(([gene, info]) => (
              <div key={gene} style={{ background: '#1e293b', borderRadius: 8, padding: 14, marginBottom: 10, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <GeneChip gene={gene} />
                  <span style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 600 }}>{info.full}</span>
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>· {info.locus} · {info.size} · {info.inh}</span>
                </div>
                <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5 }}>{info.disease}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {breakdown.gene_breakdowns?.map(g => (
            <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, padding: 18, marginBottom: 16, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                <GeneChip gene={g.gene} />
                <span style={{ color: '#e2e8f0', fontWeight: 700 }}>{g.disease_category}</span>
                <span style={{ color: '#94a3b8', fontSize: 12 }}>· {g.locus} · {g.protein_size} · {g.inheritance}</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                <div>
                  <div style={{ color: '#fbbf24', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>PATHOGNOMONIC</div>
                  <div style={{ color: '#e2e8f0', fontSize: 12, lineHeight: 1.5 }}>{g.pathognomonic}</div>
                </div>
                <div>
                  <div style={{ color: '#60a5fa', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>KEY DDx</div>
                  <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5 }}>{g.key_ddx}</div>
                </div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#34d399', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>TREATMENT / MANAGEMENT</div>
                <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5, whiteSpace: 'pre-line' }}>{g.treatment}</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#a78bfa', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>KEY FEATURES</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {g.key_features?.map((f, i) => (
                    <span key={i} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 4, padding: '2px 8px', fontSize: 11, color: '#cbd5e1' }}>{f}</span>
                  ))}
                </div>
              </div>

              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', borderTop: '1px solid #334155', paddingTop: 10, marginTop: 8 }}>
                <span style={{ color: '#fbbf24', fontSize: 12 }}>Glaucoma {g.glaucoma_pct}%</span>
                <span style={{ color: '#60a5fa', fontSize: 12 }}>Corneal opacity {g.corneal_opacity_pct}%</span>
                <span style={{ color: '#34d399', fontSize: 12 }}>Surgery {g.surgery_pct}%</span>
                <span style={{ color: '#f87171', fontSize: 12 }}>VA↓ {g.va_poor_pct}%</span>
                <span style={{ color: '#94a3b8', fontSize: 12 }}>Consanguineous {g.consanguineous_pct}%</span>
                <span style={{ color: '#94a3b8', fontSize: 12 }}>Systemic: {g.systemic_involvement ? 'Yes' : 'No'}</span>
                <span style={{ color: '#64748b', fontSize: 11 }}>Urgency: {g.surgical_urgency}</span>
              </div>

              {g.sample_patients?.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>Sample Patients (n=3)</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {g.sample_patients.map(p => (
                      <div key={p.id} style={{ background: '#0f172a', borderRadius: 6, padding: '6px 10px', fontSize: 11, color: '#cbd5e1' }}>
                        <b>{p.id}</b> · Glaucoma: {p.glaucoma ? 'Y' : 'N'} · Corneal: {p.corneal_opacity ? 'Y' : 'N'} · Surg: {p.surgery_performed ? 'Y' : 'N'} · VA↓: {p.va_poor ? 'Y' : 'N'}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ color: '#93c5fd', marginBottom: 12 }}>Gene Definitions</h3>
          {Object.values(definitions.gene_entries || {}).map(e => (
            <div key={e.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 14, marginBottom: 12, borderLeft: `4px solid ${GENE_COLORS[e.gene] || '#555'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                <GeneChip gene={e.gene} />
                <span style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 600 }}>{e.full_name}</span>
                <span style={{ color: '#94a3b8', fontSize: 11 }}>· {e.locus} · {e.protein_size} · {e.inheritance}</span>
              </div>
              <div style={{ color: '#fbbf24', fontSize: 11, marginBottom: 4 }}>{e.disease_name}</div>
              <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5, marginBottom: 6 }}>{e.disease_pathway}</div>
              <div style={{ color: '#a78bfa', fontSize: 11, marginBottom: 2, fontWeight: 700 }}>PATHOGNOMONIC</div>
              <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5, marginBottom: 6 }}>{e.pathognomonic}</div>
              <div style={{ color: '#34d399', fontSize: 11, fontWeight: 700, marginBottom: 2 }}>TREATMENT</div>
              <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5 }}>{e.treatment}</div>
            </div>
          ))}

          <h3 style={{ color: '#93c5fd', marginTop: 24, marginBottom: 12 }}>ASD Glossary</h3>
          {Object.entries(definitions.asd_glossary || {}).map(([term, def]) => (
            <div key={term} style={{ background: '#1e293b', borderRadius: 8, padding: 14, marginBottom: 10 }}>
              <div style={{ color: '#fbbf24', fontSize: 12, fontWeight: 700, marginBottom: 6 }}>{term}</div>
              <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.6 }}>{def}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
