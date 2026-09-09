'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  CASR:  '#1565c0',  // deep blue — calcium sensor; FHH1/NSHPT/ADH1 triple phenotype
  PTH:   '#2e7d32',  // dark green — PTH hormone; isolated familial hypoparathyroidism
  GATA3: '#6a1b9a',  // deep purple — HDR/Barakat; hypoparathyroidism + deafness + renal
  GCM2:  '#e65100',  // deep amber — GCM2 TF; most common genetic isolated HP
  GNA11: '#880e4f',  // dark pink — FHH2/ADH2; CASR downstream effector
  AP2S1: '#37474f',  // slate — FHH3; R15 hotspot; cinacalcet-responsive
  GNAS:  '#4e342e',  // brown — PHP1a/AHO; maternal imprint; PTH resistance
  CDC73: '#b71c1c',  // dark red — HPT-JT; jaw fibroma PATHOGNOMONIC; carcinoma 15-20%
};

const GENE_DISEASE = {
  CASR:  'FHH1/NSHPT/ADH1 (AD LOF/AR biallelic LOF/AD GOF) — MOST COMMON genetic hypercalcaemia; FECa <0.01 CRITICAL; surgery CI in FHH1; NSHPT neonatal emergency',
  PTH:   'Isolated familial hypoparathyroidism (AR/AD) — prepro-PTH signal peptide mutations; severe neonatal hypocalcaemia; calcitriol + calcium lifelong',
  GATA3: 'HDR/Barakat syndrome (AD haploinsufficiency) — Hypoparathyroidism + bilateral SNHL + renal dysplasia TRIAD PATHOGNOMONIC; check Ca²⁺ before audiometry',
  GCM2:  'Isolated familial hypoparathyroidism (AR LOF/AD GOF) — most commonly mutated gene in isolated HP; GCM2 GOF → familial primary HPT',
  GNA11: 'FHH2/ADH2 (AD LOF/GOF) — CASR downstream Gα11 effector; phenocopies CASR; FHH2 benign; ADH2 hypocalcaemia + hypercalciuria',
  AP2S1: 'FHH3 (AD) — R15L/R15C/R15H hotspot; impairs CASR internalisation; cinacalcet-RESPONSIVE unlike FHH1/FHH2; milder hypercalcaemia',
  GNAS:  'PHP1a/pseudoPHP/PHP1b (AD imprinted) — Maternal LOF: PTH resistance + AHO; Paternal LOF: AHO only no resistance; GNAS methylation → PHP1b',
  CDC73: 'HPT-JT syndrome (AD LOF) — ossifying jaw fibromas PATHOGNOMONIC; parathyroid carcinoma 15-20% HIGHEST hereditary risk; annual surveillance mandatory',
};

const INHERITANCE_MAP = {
  CASR:  'AD-LOF/AR-biallelic/AD-GOF', PTH:   'AR/AD', GATA3: 'AD-haploinsuff',
  GCM2:  'AR-LOF/AD-GOF',              GNA11: 'AD-LOF/AD-GOF', AP2S1: 'AD',
  GNAS:  'AD-imprinted',               CDC73: 'AD-LOF',
};

const CA_GROUP = {
  CASR:  'Calcium-Sensing Receptor / G-protein 7TM (3q21.1)',
  PTH:   'PTH Neuropeptide / Parathyroid (11p15.3)',
  GATA3: 'Zinc-Finger TF / Neural Crest (10p14)',
  GCM2:  'GCM TF / Parathyroid Development (6p24.2)',
  GNA11: 'Gα11 Subunit / CASR Effector (19p13.3)',
  AP2S1: 'AP2 Adaptor / CASR Internalisation (19q13.32)',
  GNAS:  'Gsα Subunit / cAMP / Imprinted (20q13.32)',
  CDC73: 'Parafibromin / Tumour Suppressor (1q31.2)',
};

export default function HereditaryCalciumHomeostasisAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedGene, setSelectedGene] = useState('CASR');

  useEffect(() => {
    async function load() {
      try {
        const [ov, bk, df] = await Promise.all([
          fetch(`${API}/api/hereditary-calcium-homeostasis-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/hereditary-calcium-homeostasis-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/hereditary-calcium-homeostasis-atlas/definitions`).then(r => r.json()),
        ]);
        setOverview(ov); setBreakdown(bk); setDefinitions(df);
      } catch (e) { setError(e.message); }
      finally { setLoading(false); }
    }
    load();
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#1565c0' }}>Loading Hereditary Calcium Homeostasis Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  const accentColor = '#1565c0';
  const genes = Object.keys(GENE_COLORS);
  const gd = breakdown?.breakdown_by_gene?.[selectedGene];

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', maxWidth: 1200, margin: '0 auto', padding: 20 }}>
      {/* Header */}
      <div style={{ background: `linear-gradient(135deg,${accentColor} 0%,#880e4f 100%)`, borderRadius: 12, padding: '24px 32px', marginBottom: 24, color: '#fff' }}>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 700 }}>Hereditary Calcium Homeostasis Atlas</h1>
        <div style={{ opacity: 0.85, marginTop: 6, fontSize: 13 }}>
          Complete 8-Gene Reference · CASR · PTH · GATA3 · GCM2 · GNA11 · AP2S1 · GNAS · CDC73
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12 }}>
          <span style={{ background: 'rgba(255,255,255,0.18)', borderRadius: 6, padding: '4px 10px' }}>
            320 Patients · 8 × 40 · Seeds 2494–2501
          </span>
          <span style={{ background: 'rgba(255,255,255,0.18)', borderRadius: 6, padding: '4px 10px' }}>
            FHH · NSHPT · ADH · Isolated HP · HDR · PHP · HPT-JT
          </span>
          <span style={{ background: 'rgba(255,255,255,0.18)', borderRadius: 6, padding: '4px 10px' }}>
            FECa · Cinacalcet · Teriparatide · Parathyroid Carcinoma Surveillance
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '2px solid #e3f2fd', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{
              padding: '10px 20px', border: 'none', borderRadius: '8px 8px 0 0',
              background: tab === t ? accentColor : '#e3f2fd',
              color: tab === t ? '#fff' : '#555', cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
              borderBottom: tab === t ? `3px solid ${accentColor}` : '3px solid transparent',
              fontSize: 14,
            }}>{t}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Genes Covered', value: overview.genes_covered },
              { label: 'Cohort / Gene', value: 40 },
              { label: 'Seed Range', value: overview.seed_range },
            ].map(m => (
              <div key={m.label} style={{ background: '#e3f2fd', borderRadius: 10, padding: '16px 20px', textAlign: 'center', border: `1px solid #90caf9` }}>
                <div style={{ fontSize: 28, fontWeight: 800, color: accentColor }}>{m.value}</div>
                <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>{m.label}</div>
              </div>
            ))}
          </div>

          {/* Emergency rules */}
          <div style={{ background: '#fce4ec', borderRadius: 10, padding: 20, marginBottom: 20, border: '1px solid #f48fb1' }}>
            <div style={{ fontWeight: 700, color: '#b71c1c', marginBottom: 10, fontSize: 15 }}>Emergency / Key Rules</div>
            {overview.key_emergency_rules?.map((r, i) => (
              <div key={i} style={{ padding: '5px 0', borderBottom: '1px solid #f8bbd0', fontSize: 13, color: '#4a0e2b' }}>
                ⚠ {r}
              </div>
            ))}
          </div>

          {/* Cohort cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14 }}>
            {overview.cohort_breakdown?.map(c => (
              <div key={c.gene}
                style={{ borderRadius: 10, border: `2px solid ${GENE_COLORS[c.gene] || '#ccc'}`, background: '#fff', padding: 14, cursor: 'pointer' }}
                onClick={() => { setSelectedGene(c.gene); setTab('Clinical Atlas'); }}>
                <div style={{ fontWeight: 800, fontSize: 18, color: GENE_COLORS[c.gene] || '#333' }}>{c.gene}</div>
                <div style={{ fontSize: 11, color: '#666', marginBottom: 6 }}>{c.locus} · {c.protein_size}</div>
                <div style={{ fontSize: 11, color: '#888', lineHeight: 1.4 }}>{c.disease_summary?.slice(0, 90)}…</div>
                <div style={{ marginTop: 8, fontSize: 11, color: '#999' }}>
                  {c.patients} patients · avg dx age {c.avg_age_at_dx} yr
                </div>
              </div>
            ))}
          </div>

          {/* Diagnostic tests */}
          <div style={{ marginTop: 24, background: '#e8f5e9', borderRadius: 10, padding: 20, border: '1px solid #a5d6a7' }}>
            <div style={{ fontWeight: 700, color: '#1b5e20', marginBottom: 10, fontSize: 15 }}>Key Diagnostic Tests</div>
            {overview.key_diagnostic_tests?.map((t, i) => (
              <div key={i} style={{ padding: '4px 0', borderBottom: '1px solid #c8e6c9', fontSize: 13, color: '#2e7d32' }}>
                🔬 {t}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: accentColor, color: '#fff' }}>
                {['Gene', 'Locus', 'Protein', 'Inheritance', 'Group / Pathway', 'Key Rule'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, fontSize: 12 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {genes.map((gene, idx) => {
                const bkg = idx % 2 === 0 ? '#e3f2fd' : '#fff';
                const gdata = breakdown?.breakdown_by_gene?.[gene];
                return (
                  <tr key={gene} style={{ background: bkg, cursor: 'pointer' }}
                    onClick={() => { setSelectedGene(gene); setTab('Clinical Atlas'); }}>
                    <td style={{ padding: '10px 12px', fontWeight: 700, color: GENE_COLORS[gene], whiteSpace: 'nowrap' }}>{gene}</td>
                    <td style={{ padding: '10px 12px', color: '#555', whiteSpace: 'nowrap' }}>{gdata?.locus || '—'}</td>
                    <td style={{ padding: '10px 12px', color: '#666', maxWidth: 120 }}>{gdata?.protein_size || '—'}</td>
                    <td style={{ padding: '10px 12px', fontSize: 11 }}><span style={{ background: '#bbdefb', color: accentColor, padding: '2px 6px', borderRadius: 4, whiteSpace: 'nowrap' }}>{INHERITANCE_MAP[gene]}</span></td>
                    <td style={{ padding: '10px 12px', color: '#555', maxWidth: 200, fontSize: 12 }}>{CA_GROUP[gene]}</td>
                    <td style={{ padding: '10px 12px', color: '#333', maxWidth: 220, fontSize: 12 }}>{GENE_DISEASE[gene]?.slice(0, 100)}…</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && (
        <div style={{ display: 'flex', gap: 20 }}>
          {/* Gene selector */}
          <div style={{ width: 160, flexShrink: 0 }}>
            {genes.map(gene => (
              <button key={gene} onClick={() => setSelectedGene(gene)}
                style={{
                  display: 'block', width: '100%', marginBottom: 6, padding: '8px 12px',
                  background: selectedGene === gene ? GENE_COLORS[gene] : '#e3f2fd',
                  color: selectedGene === gene ? '#fff' : GENE_COLORS[gene],
                  border: `2px solid ${GENE_COLORS[gene]}`, borderRadius: 8,
                  cursor: 'pointer', fontWeight: selectedGene === gene ? 700 : 500, fontSize: 13, textAlign: 'left',
                }}>{gene}</button>
            ))}
          </div>

          {/* Gene detail */}
          {gd && (
            <div style={{ flex: 1 }}>
              <div style={{ borderRadius: 10, background: GENE_COLORS[selectedGene], color: '#fff', padding: '16px 20px', marginBottom: 16 }}>
                <div style={{ fontSize: 22, fontWeight: 800 }}>{selectedGene}</div>
                <div style={{ fontSize: 12, opacity: 0.85, marginTop: 4 }}>{gd.locus} · {gd.protein_size} · {INHERITANCE_MAP[selectedGene]}</div>
                <div style={{ fontSize: 13, marginTop: 8, opacity: 0.9 }}>{gd.disease_category}</div>
              </div>

              {[
                { label: 'Inheritance & Mechanism', text: gd.inheritance },
                { label: 'Disease Pathway', text: gd.disease_pathway },
                { label: 'Pathognomonic Pearls', text: gd.pathognomonic },
                { label: 'Treatment', text: gd.treatment },
                { label: 'DDx', text: gd.key_ddx },
                { label: 'Cascade Testing', text: gd.cascade_testing },
                { label: 'Emergency Protocol', text: gd.emergency_protocol },
              ].map(s => s.text && (
                <div key={s.label} style={{ marginBottom: 12, background: '#e3f2fd', borderRadius: 8, padding: '12px 16px', border: `1px solid #90caf9` }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[selectedGene], marginBottom: 6, fontSize: 13 }}>{s.label}</div>
                  <div style={{ fontSize: 13, color: '#333', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{s.text}</div>
                </div>
              ))}

              {/* Key features */}
              {gd.key_features?.length > 0 && (
                <div style={{ background: '#e8eaf6', borderRadius: 8, padding: '12px 16px', marginBottom: 12 }}>
                  <div style={{ fontWeight: 700, color: '#1a237e', marginBottom: 8, fontSize: 13 }}>Key Features</div>
                  {gd.key_features.map((f, i) => (
                    <div key={i} style={{ padding: '3px 0', fontSize: 13, color: '#283593' }}>• {f}</div>
                  ))}
                </div>
              )}

              {/* Patient distribution */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginTop: 12 }}>
                {[
                  { label: 'Presentations', data: gd.presentation_distribution },
                  { label: 'Managements', data: gd.management_distribution },
                  { label: 'Outcomes', data: gd.outcome_distribution },
                ].map(({ label, data }) => data && (
                  <div key={label} style={{ background: '#fff', border: '1px solid #90caf9', borderRadius: 8, padding: 12 }}>
                    <div style={{ fontWeight: 700, fontSize: 12, color: GENE_COLORS[selectedGene], marginBottom: 8 }}>{label}</div>
                    {Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([k, v]) => (
                      <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, padding: '2px 0', borderBottom: '1px solid #e3f2fd' }}>
                        <span style={{ color: '#555', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{k.replaceAll('_', ' ')}</span>
                        <span style={{ fontWeight: 700, color: GENE_COLORS[selectedGene] }}>{v}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ background: '#e3f2fd', borderRadius: 10, padding: '12px 18px', marginBottom: 20, border: `1px solid #90caf9` }}>
            <div style={{ fontWeight: 700, color: accentColor, fontSize: 15 }}>{definitions.atlas_domain}</div>
            <div style={{ marginTop: 6, fontSize: 12, color: '#666' }}>
              8 Genes: CASR · PTH · GATA3 · GCM2 · GNA11 · AP2S1 · GNAS · CDC73 · 320 patients modelled
            </div>
          </div>

          {/* Clinical pearls */}
          {definitions.clinical_pearls?.length > 0 && (
            <div style={{ background: '#fce4ec', borderRadius: 10, padding: 18, marginBottom: 20, border: '1px solid #f48fb1' }}>
              <div style={{ fontWeight: 700, color: '#880e4f', marginBottom: 10, fontSize: 14 }}>Clinical Pearls & Key Rules</div>
              {definitions.clinical_pearls.map((d, i) => (
                <div key={i} style={{ padding: '5px 0', borderBottom: '1px solid #f8bbd0', fontSize: 13, color: '#4a0e2b' }}>
                  💎 {d}
                </div>
              ))}
            </div>
          )}

          {/* Key definitions */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {definitions.key_definitions && Object.entries(definitions.key_definitions).map(([key, val]) => (
              <div key={key} style={{ background: '#e3f2fd', borderRadius: 8, padding: '14px 16px', border: '1px solid #90caf9' }}>
                <div style={{ fontWeight: 700, color: accentColor, marginBottom: 8, fontSize: 13 }}>
                  {key.replaceAll('_', ' ')}
                </div>
                <div style={{ fontSize: 12, color: '#333', lineHeight: 1.6 }}>{val}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
