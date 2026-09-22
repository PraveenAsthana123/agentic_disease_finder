'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hypophosphatemic-rickets-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'PHEX':    '#1565c0',  // deep blue     — XLH, most common, burosumab
  'FGF23':   '#b71c1c',  // deep red      — ADHR GOF / Tumoral Calcinosis LOF
  'DMP1':    '#4a148c',  // deep purple   — ARHR1, enthesopathy distinctive
  'ENPP1':   '#e65100',  // deep orange   — GACI + ARHR2, etidronate
  'CLCN5':   '#00695c',  // dark teal     — Dent 1, LMW proteinuria
  'OCRL':    '#2e7d32',  // dark green    — Lowe syndrome / Dent 2
  'SLC34A3': '#f9a825',  // amber         — HHRH, suppressed PTH + high 1,25D
  'CYP27B1': '#6a1b9a',  // purple        — VDDR1, calcitriol curative
};

const GENE_INFO = {
  'PHEX': {
    full: 'PHEX / Phosphate-Regulating Endopeptidase / 749aa',
    locus: 'Xp22.11',
    size: '749 aa / 86 kDa (osteoblast/osteocyte metalloendopeptidase; cleaves FGF23; LOF → iFGF23 HIGH → phosphaturia + low 1,25D; XLH 1:20,000; burosumab FDA 2018; dental abscesses without caries PATHOGNOMONIC; XL)',
    inh: 'XL',
  },
  'FGF23': {
    full: 'FGF23 / Fibroblast Growth Factor 23 / 251aa',
    locus: '12p13.32',
    size: '251 aa / 26 kDa (phosphatonin; RXXR cleavage site; GOF=ADHR: cleavage-resistant iFGF23 high, intermittent, iron triggers; LOF=Tumoral Calcinosis: HYPERPHOSPHATAEMIA + periarticular calcifications OPPOSITE phenotype; AD GOF / AR LOF)',
    inh: 'AD GOF / AR LOF',
  },
  'DMP1': {
    full: 'DMP1 / Dentin Matrix Protein 1 / 473aa',
    locus: '4q22.1',
    size: '473 aa / 54 kDa (SIBLING family; osteocyte FGF23 suppressor; LOF → FGF23 overproduction → ARHR1; enthesopathy + periosteal reactions DISTINCTIVE in adults; dentinogenesis imperfecta; AR)',
    inh: 'AR',
  },
  'ENPP1': {
    full: 'ENPP1 / Ectonucleotide Pyrophosphatase/Phosphodiesterase 1 / 925aa',
    locus: '6q23.2',
    size: '925 aa / 100 kDa (ATP→AMP+PPi; PPi inhibits calcification; LOF → PPi absent → GACI neonatal arterial calcification + ARHR2; calcification paradox: vessels calcify + bones rickets; etidronate FIRST-LINE GACI; AR)',
    inh: 'AR',
  },
  'CLCN5': {
    full: 'CLCN5 / Chloride Voltage-Gated Channel 5 / 746aa',
    locus: 'Xp11.23',
    size: '746 aa / 83 kDa (CLC-5 endosomal H+/Cl- exchanger; acidifies endosomes; LOF → megalin-cubilin recycling impaired → LMW proteinuria CARDINAL; Dent 1; hypercalciuria; nephrocalcinosis; progressive CKD; XL)',
    inh: 'XL',
  },
  'OCRL': {
    full: 'OCRL / Oculocerebrorenal Protein of Lowe / 901aa',
    locus: 'Xq26.1',
    size: '901 aa / 105 kDa (PI(4,5)P2 5-phosphatase; endosomal trafficking; LOF → Lowe syndrome: cataracts+ID+Fanconi; Dent 2: males without eye/brain; female carriers: lens opacities slit-lamp >90%; XL)',
    inh: 'XL',
  },
  'SLC34A3': {
    full: 'SLC34A3 / Sodium-Phosphate Cotransporter IIc / 599aa',
    locus: '9q34.3',
    size: '599 aa / 68 kDa (NaPi-IIc; proximal tubule Pi reabsorption; LOF → FGF23-INDEPENDENT phosphate wasting; HHRH: suppressed PTH + HIGH 1,25D + hypercalciuria PATHOGNOMONIC; phosphate-alone treats; AR)',
    inh: 'AR',
  },
  'CYP27B1': {
    full: 'CYP27B1 / 25-Hydroxyvitamin D 1-Alpha-Hydroxylase / 508aa',
    locus: '12q14.1',
    size: '508 aa / 55 kDa (mitochondrial CYP450; converts 25D→1,25D; LOF → VDDR1: 25D NORMAL + 1,25D VERY LOW; calcitriol CURATIVE; vitamin D3 INEFFECTIVE; secondary HPT; AR)',
    inh: 'AR',
  },
};

export default function HereditaryHypophosphatemicRicketsAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function load(t) {
    setLoading(true); setError(null);
    try {
      if (t === 'Overview' && !overview) {
        const r = await fetch(`${API}/api/${SLUG}/overview`);
        setOverview(await r.json());
      } else if ((t === 'Gene Table' || t === 'Clinical Atlas') && !breakdown) {
        const r = await fetch(`${API}/api/${SLUG}/breakdown`);
        setBreakdown(await r.json());
      } else if (t === 'Definitions' && !definitions) {
        const r = await fetch(`${API}/api/${SLUG}/definitions`);
        setDefinitions(await r.json());
      }
    } catch (e) { setError(String(e)); }
    setLoading(false);
  }

  useEffect(() => { load('Overview'); }, []);
  useEffect(() => { load(tab); }, [tab]);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: '#fbbf24', marginBottom: 4 }}>
          🧬 Hereditary Hypophosphatemic Rickets Atlas
        </div>
        <div style={{ fontSize: 12, color: '#94a3b8' }}>
          Complete 8-Gene FGF23 / Phosphate-Wasting Rickets Reference — PHEX · FGF23 · DMP1 · ENPP1 · CLCN5 · OCRL · SLC34A3 · CYP27B1 — 320 patients, seeds 2974–2981
        </div>
      </div>

      {/* ── TAB BAR ── */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
            background: tab === t ? '#fbbf24' : '#1e293b', color: tab === t ? '#0f172a' : '#94a3b8',
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#fbbf24' }}>Loading…</div>}
      {error   && <div style={{ color: '#ef4444' }}>Error: {error}</div>}

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 12, marginBottom: 20 }}>
            {[
              { label: 'Genes', value: overview.total_genes },
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Seed Range', value: overview.seed_range },
              { label: 'FGF23-Dependent', value: 4 },
              { label: 'FGF23-Independent', value: 2 },
              { label: 'Tubular Dent', value: 2 },
            ].map(kpi => (
              <div key={kpi.label} style={{ background: '#1e293b', borderRadius: 8, padding: 14, textAlign: 'center' }}>
                <div style={{ fontSize: 22, fontWeight: 800, color: '#fbbf24' }}>{kpi.value}</div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{kpi.label}</div>
              </div>
            ))}
          </div>

          {/* Gene chips */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
            {(overview.genes || []).map(g => (
              <div key={g} style={{ background: GENE_COLORS[g] || '#334155', borderRadius: 6, padding: '4px 12px', fontSize: 11, fontWeight: 700, color: '#fff' }}>
                {g} — {GENE_INFO[g]?.locus || '?'} — {GENE_INFO[g]?.inh || '?'}
              </div>
            ))}
          </div>

          {/* Key clinical rules */}
          <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, color: '#fbbf24', marginBottom: 10, fontSize: 13 }}>Key Clinical Rules</div>
            {(overview.key_clinical_rules || []).map((r, i) => (
              <div key={i} style={{ fontSize: 11, color: '#cbd5e1', marginBottom: 6, paddingLeft: 12, borderLeft: '3px solid #fbbf24', lineHeight: 1.6 }}>
                {r}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#1e293b' }}>
                {['Gene', 'Locus', 'Inh', 'N', 'Mean Age Dx', 'Mean Pi', 'Pi Low%', 'FGF23 Hi%', 'LMW Prot%', 'Nephrocalc%', 'Low 1,25D%', 'Supp PTH%', 'Top Treatment'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', color: '#fbbf24', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(breakdown.genes || []).map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                  <td style={{ padding: '6px 10px', color: GENE_COLORS[g.gene] || '#e2e8f0', fontWeight: 700 }}>{g.gene}</td>
                  <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.locus}</td>
                  <td style={{ padding: '6px 10px', color: '#cbd5e1' }}>{GENE_INFO[g.gene]?.inh || '?'}</td>
                  <td style={{ padding: '6px 10px', color: '#e2e8f0' }}>{g.n_patients}</td>
                  <td style={{ padding: '6px 10px', color: '#e2e8f0' }}>{g.mean_age_dx}</td>
                  <td style={{ padding: '6px 10px', color: g.mean_phosphate < 0.65 ? '#ef4444' : '#94a3b8' }}>{g.mean_phosphate}</td>
                  <td style={{ padding: '6px 10px', color: g.phosphate_low_pct > 80 ? '#ef4444' : '#94a3b8' }}>{g.phosphate_low_pct}%</td>
                  <td style={{ padding: '6px 10px', color: g.fgf23_high_pct > 70 ? '#fbbf24' : '#94a3b8' }}>{g.fgf23_high_pct}%</td>
                  <td style={{ padding: '6px 10px', color: g.lmw_prot_pct > 70 ? '#22d3ee' : '#94a3b8' }}>{g.lmw_prot_pct}%</td>
                  <td style={{ padding: '6px 10px', color: g.nephrocalc_pct > 50 ? '#f97316' : '#94a3b8' }}>{g.nephrocalc_pct}%</td>
                  <td style={{ padding: '6px 10px', color: g.low_1_25d_pct > 70 ? '#a78bfa' : '#94a3b8' }}>{g.low_1_25d_pct}%</td>
                  <td style={{ padding: '6px 10px', color: g.suppressed_pth_pct > 70 ? '#34d399' : '#94a3b8' }}>{g.suppressed_pth_pct}%</td>
                  <td style={{ padding: '6px 10px', color: '#64748b', fontSize: 10 }}>
                    {Object.entries(g.treatment_breakdown || {}).sort((a,b) => b[1]-a[1])[0]?.[0] || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {(breakdown.genes || []).map((g, i) => (
            <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#475569'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                <span style={{ background: GENE_COLORS[g.gene] || '#334155', color: '#fff', fontWeight: 800, borderRadius: 6, padding: '3px 10px', fontSize: 13 }}>{g.gene}</span>
                <span style={{ color: '#94a3b8', fontSize: 11 }}>{g.locus}</span>
                <span style={{ color: '#64748b', fontSize: 11 }}>{GENE_INFO[g.gene]?.inh || '?'}</span>
                <span style={{ color: '#475569', fontSize: 11 }}>n={g.n_patients}</span>
              </div>

              {/* KPI strip */}
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
                {[
                  { label: 'Age Dx', value: g.mean_age_dx + 'y', color: '#fbbf24' },
                  { label: 'Pi (mmol/L)', value: g.mean_phosphate, color: '#ef4444' },
                  { label: 'FGF23 Hi', value: g.fgf23_high_pct + '%', color: '#fbbf24' },
                  { label: 'LMW Prot', value: g.lmw_prot_pct + '%', color: '#22d3ee' },
                  { label: 'Nephrocalc', value: g.nephrocalc_pct + '%', color: '#f97316' },
                  { label: 'Low 1,25D', value: g.low_1_25d_pct + '%', color: '#a78bfa' },
                  { label: 'Supp PTH', value: g.suppressed_pth_pct + '%', color: '#34d399' },
                ].map(kpi => (
                  <div key={kpi.label} style={{ background: '#0f172a', borderRadius: 6, padding: '4px 10px', fontSize: 10 }}>
                    <span style={{ color: '#64748b' }}>{kpi.label}: </span>
                    <span style={{ color: kpi.color, fontWeight: 700 }}>{kpi.value}</span>
                  </div>
                ))}
              </div>

              {/* Inheritance summary */}
              <div style={{ fontSize: 11, color: '#cbd5e1', whiteSpace: 'pre-line', lineHeight: 1.7, marginBottom: 8, maxHeight: 220, overflowY: 'auto' }}>
                {g.inheritance}
              </div>

              {/* Patient micro-table (first 8) */}
              {g.patients && g.patients.length > 0 && (
                <div style={{ overflowX: 'auto', marginTop: 8 }}>
                  <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                    <thead>
                      <tr style={{ background: '#0f172a' }}>
                        {['ID', 'Age Dx', 'Pi (mmol/L)', 'FGF23 Hi', 'LMW Prot', 'Nephrocalc', 'Low 1,25D', 'Supp PTH', 'Treatment'].map(h => (
                          <th key={h} style={{ padding: '3px 8px', color: '#64748b', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {g.patients.slice(0, 8).map(p => (
                        <tr key={p.id}>
                          <td style={{ padding: '3px 8px', color: '#475569' }}>{p.id}</td>
                          <td style={{ padding: '3px 8px', color: '#e2e8f0' }}>{p.age_at_diagnosis}</td>
                          <td style={{ padding: '3px 8px', color: p.phosphate_low ? '#ef4444' : '#22d3ee' }}>{p.serum_phosphate_mmol_L}</td>
                          <td style={{ padding: '3px 8px', color: p.fgf23_high ? '#fbbf24' : '#475569' }}>{p.fgf23_high ? 'Yes' : 'No'}</td>
                          <td style={{ padding: '3px 8px', color: p.lmw_proteinuria ? '#22d3ee' : '#475569' }}>{p.lmw_proteinuria ? 'Yes' : 'No'}</td>
                          <td style={{ padding: '3px 8px', color: p.nephrocalcinosis ? '#f97316' : '#475569' }}>{p.nephrocalcinosis ? 'Yes' : 'No'}</td>
                          <td style={{ padding: '3px 8px', color: p.low_1_25d ? '#a78bfa' : '#475569' }}>{p.low_1_25d ? 'Yes' : 'No'}</td>
                          <td style={{ padding: '3px 8px', color: p.suppressed_pth ? '#34d399' : '#475569' }}>{p.suppressed_pth ? 'Yes' : 'No'}</td>
                          <td style={{ padding: '3px 8px', color: '#94a3b8', fontSize: 10 }}>{p.treatment}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ marginBottom: 12, color: '#64748b', fontSize: 12 }}>{definitions.count} clinical definitions</div>
          {(definitions.definitions || []).map((t, i) => (
            <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 12 }}>
              <div style={{ fontWeight: 700, color: '#fbbf24', marginBottom: 8, fontSize: 13 }}>{t.term}</div>
              <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
                {(t.genes || []).map(g => (
                  <span key={g} style={{ background: GENE_COLORS[g] || '#334155', color: '#fff', fontSize: 10, borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>{g}</span>
                ))}
              </div>
              <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-line' }}>{t.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
