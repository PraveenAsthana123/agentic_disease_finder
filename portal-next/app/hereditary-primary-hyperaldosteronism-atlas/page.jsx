'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  KCNJ5:   '#1565c0',  // deep blue — Kir3.4/GIRK4; FH3 most common hereditary PA; somatic APA 35-40%
  CLCN2:   '#2e7d32',  // dark green — CLC-2 chloride channel; FH2; adult onset bilateral
  CACNA1H: '#6a1b9a',  // deep purple — Cav3.2 T-type Ca²⁺; FH4 + PASNA; childhood seizures
  CACNA1D: '#880e4f',  // dark pink — Cav1.3 L-type Ca²⁺; PASNA + SNHL + cardiac; somatic APA
  ATP1A1:  '#e65100',  // deep amber — Na+/K+ ATPase α1; severe early-onset; cortisol cosecretion
  ATP2B3:  '#37474f',  // slate — PMCA3; X-linked; male predominant somatic APA
  ARMC5:   '#4e342e',  // brown — armadillo repeat; BMAH tumour suppressor; food-dependent cortisol
  PRKACA:  '#b71c1c',  // dark red — PKA-Cα; bilateral BAH Cushing's + aldosterone; Leu206Arg somatic
};

const GENE_DISEASE = {
  KCNJ5:   'FH3 (AD GOF) + somatic APA 35-40% — Kir3.4 selectivity filter G151R/L168R; childhood bilateral or adult unilateral APA; female predominant somatic',
  CLCN2:   'FH2 (AD GOF) — CLC-2 chloride channel; bilateral PA adult onset; milder phenotype; normo/mildly hypokalaemic; MRA responsive',
  CACNA1H: 'FH4/PASNA (AD GOF) — Cav3.2 T-type Ca²⁺ channel; childhood PA + epilepsy + neurodevelopmental delay; bilateral hyperplasia',
  CACNA1D: 'PASNA (AD GOF) + somatic APA ~10% — Cav1.3 L-type Ca²⁺; PA + seizures + SNHL + sinus bradycardia; de novo common',
  ATP1A1:  'Severe early-onset PA (germline rare) + somatic APA ~6% — Na+/K+ATPase α1; second most common somatic after KCNJ5; cortisol cosecretion',
  ATP2B3:  'Somatic APA ~2% X-linked (germline rare) — PMCA3 Ca²⁺ extrusion pump; male predominant; small CT-occult APA; X-linked pedigree',
  ARMC5:   'BMAH tumour suppressor (AD LOF two-hit) — bilateral macronodular adrenal hyperplasia; food-dependent cortisol PATHOGNOMONIC; ~50% familial BMAH',
  PRKACA:  'PKA-Cα GOF — bilateral BAH Cushing syndrome ± aldosterone; somatic Leu206Arg most common overt adrenal Cushing's mutation; perioperative cortisol mandatory',
};

const INHERITANCE_MAP = {
  KCNJ5:   'AD-GOF/Somatic',  CLCN2:   'AD-GOF',
  CACNA1H: 'AD-GOF/De-Novo',  CACNA1D: 'AD-GOF/Somatic',
  ATP1A1:  'AD-Germline/Somatic', ATP2B3: 'X-Linked/Somatic',
  ARMC5:   'AD-LOF-Two-Hit',  PRKACA:  'AD-GOF/Somatic',
};

const PATHWAY_GROUP = {
  KCNJ5:   'K+ Channel / Selectivity Filter (11q24.3)',
  CLCN2:   'Cl⁻ Channel / ZG Depolarisation (3q27.3)',
  CACNA1H: 'T-type Ca²⁺ Channel Cav3.2 (16p13.3)',
  CACNA1D: 'L-type Ca²⁺ Channel Cav1.3 (3p14.3)',
  ATP1A1:  'Na+/K+ ATPase α1 / Ion Gradient (1p13.1)',
  ATP2B3:  'Plasma Membrane Ca²⁺ Pump PMCA3 (Xq28)',
  ARMC5:   'Armadillo Repeat / Tumour Suppressor (16p11.2)',
  PRKACA:  'PKA Catalytic α / cAMP Pathway (19p13.12)',
};

export default function HereditaryPrimaryHyperaldosteronismAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedGene, setSelectedGene] = useState('KCNJ5');

  useEffect(() => {
    async function load() {
      try {
        const [ov, bk, df] = await Promise.all([
          fetch(`${API}/api/hereditary-primary-hyperaldosteronism-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/hereditary-primary-hyperaldosteronism-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/hereditary-primary-hyperaldosteronism-atlas/definitions`).then(r => r.json()),
        ]);
        setOverview(ov); setBreakdown(bk); setDefinitions(df);
      } catch (e) { setError(e.message); }
      finally { setLoading(false); }
    }
    load();
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#1565c0' }}>Loading Hereditary Primary Hyperaldosteronism Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  const accentColor = '#1565c0';
  const genes = Object.keys(GENE_COLORS);
  const gd = breakdown?.breakdown_by_gene?.[selectedGene];

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', maxWidth: 1200, margin: '0 auto', padding: 20 }}>
      {/* Header */}
      <div style={{ background: `linear-gradient(135deg,${accentColor} 0%,#b71c1c 100%)`, borderRadius: 12, padding: '24px 32px', marginBottom: 24, color: '#fff' }}>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 700 }}>Hereditary Primary Hyperaldosteronism Atlas</h1>
        <div style={{ opacity: 0.85, marginTop: 6, fontSize: 13 }}>
          Complete 8-Gene Reference · KCNJ5 · CLCN2 · CACNA1H · CACNA1D · ATP1A1 · ATP2B3 · ARMC5 · PRKACA
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12 }}>
          <span style={{ background: 'rgba(255,255,255,0.18)', borderRadius: 6, padding: '4px 10px' }}>
            320 Patients · 8 × 40 · Seeds 2502–2509
          </span>
          <span style={{ background: 'rgba(255,255,255,0.18)', borderRadius: 6, padding: '4px 10px' }}>
            FH2 · FH3 · FH4 · PASNA · BMAH · Somatic APAs
          </span>
          <span style={{ background: 'rgba(255,255,255,0.18)', borderRadius: 6, padding: '4px 10px' }}>
            ARR · AVS · MRA · Adrenalectomy · Cortisol Cosecretion
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '2px solid #e3f2fd', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', border: 'none', background: tab === t ? accentColor : '#e3f2fd',
            color: tab === t ? '#fff' : '#1565c0', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            fontWeight: tab === t ? 700 : 400, fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      {/* === OVERVIEW === */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 16, marginBottom: 28 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Genes Covered', value: overview.genes_covered },
              { label: 'Cohort/Gene', value: 40 },
              { label: 'Seed Range', value: overview.seed_range },
            ].map(s => (
              <div key={s.label} style={{ background: '#e3f2fd', borderRadius: 10, padding: 18, textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: accentColor }}>{s.value}</div>
                <div style={{ fontSize: 12, color: '#555', marginTop: 4 }}>{s.label}</div>
              </div>
            ))}
          </div>

          <div style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h3 style={{ color: accentColor, margin: '0 0 12px' }}>Domain</h3>
            <p style={{ margin: 0, color: '#333', lineHeight: 1.6 }}>{overview.domain}</p>
          </div>

          {/* Gene cohort cards */}
          <h3 style={{ color: accentColor, margin: '0 0 12px' }}>Gene Cohort Summary</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(260px,1fr))', gap: 14, marginBottom: 28 }}>
            {overview.cohorts?.map(c => (
              <div key={c.gene} style={{ background: '#fff', border: `2px solid ${GENE_COLORS[c.gene] || '#ccc'}`, borderRadius: 10, padding: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontWeight: 700, color: GENE_COLORS[c.gene] || '#333', fontSize: 16 }}>{c.gene}</span>
                  <span style={{ background: GENE_COLORS[c.gene] || '#ccc', color: '#fff', borderRadius: 6, padding: '2px 8px', fontSize: 11 }}>{c.patients} pts</span>
                </div>
                <div style={{ fontSize: 11, color: '#666', marginBottom: 6 }}>{c.locus} · {c.protein_size}</div>
                <div style={{ fontSize: 11, color: '#888', marginBottom: 6 }}>Avg dx age: {c.avg_age_at_dx}y</div>
                <div style={{ fontSize: 11, color: '#444', lineHeight: 1.4 }}>{c.disease_summary.substring(0, 100)}…</div>
              </div>
            ))}
          </div>

          {/* Emergency rules */}
          <div style={{ background: '#ffebee', border: '2px solid #b71c1c', borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h3 style={{ color: '#b71c1c', margin: '0 0 10px' }}>🚨 Emergency Rules</h3>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {overview.key_emergency_rules?.map((r, i) => <li key={i} style={{ color: '#b71c1c', marginBottom: 6, fontSize: 13 }}>{r}</li>)}
            </ul>
          </div>

          {/* Diagnostic tests */}
          <div style={{ background: '#f3e5f5', border: '1px solid #6a1b9a', borderRadius: 10, padding: 20 }}>
            <h3 style={{ color: '#6a1b9a', margin: '0 0 10px' }}>🔬 Key Diagnostic Tests</h3>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {overview.key_diagnostic_tests?.map((t, i) => <li key={i} style={{ color: '#4a148c', marginBottom: 6, fontSize: 13 }}>{t}</li>)}
            </ul>
          </div>
        </div>
      )}

      {/* === GENE TABLE === */}
      {tab === 'Gene Table' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: accentColor, color: '#fff' }}>
                {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease / Syndrome', 'Pathway Group'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {genes.map((g, idx) => (
                <tr key={g} style={{ background: idx % 2 === 0 ? '#f5f5f5' : '#fff', cursor: 'pointer' }}
                    onClick={() => { setSelectedGene(g); setTab('Clinical Atlas'); }}>
                  <td style={{ padding: '8px 12px', fontWeight: 700, color: GENE_COLORS[g] }}>{g}</td>
                  <td style={{ padding: '8px 12px', color: '#555' }}>{breakdown?.breakdown_by_gene?.[g]?.locus}</td>
                  <td style={{ padding: '8px 12px', color: '#555', whiteSpace: 'nowrap' }}>{breakdown?.breakdown_by_gene?.[g]?.protein_size}</td>
                  <td style={{ padding: '8px 12px' }}><span style={{ background: `${GENE_COLORS[g]}22`, color: GENE_COLORS[g], borderRadius: 4, padding: '2px 6px', fontSize: 11 }}>{INHERITANCE_MAP[g]}</span></td>
                  <td style={{ padding: '8px 12px', color: '#333', maxWidth: 280 }}>{GENE_DISEASE[g]}</td>
                  <td style={{ padding: '8px 12px', color: '#666' }}>{PATHWAY_GROUP[g]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 12, fontSize: 12, color: '#888' }}>Click a row to view full gene profile in Clinical Atlas tab.</div>
        </div>
      )}

      {/* === CLINICAL ATLAS === */}
      {tab === 'Clinical Atlas' && (
        <div style={{ display: 'flex', gap: 20 }}>
          {/* Gene selector */}
          <div style={{ minWidth: 140 }}>
            {genes.map(g => (
              <button key={g} onClick={() => setSelectedGene(g)} style={{
                display: 'block', width: '100%', marginBottom: 6, padding: '8px 12px',
                background: selectedGene === g ? GENE_COLORS[g] : '#f5f5f5',
                color: selectedGene === g ? '#fff' : GENE_COLORS[g],
                border: `2px solid ${GENE_COLORS[g]}`, borderRadius: 8, cursor: 'pointer',
                fontWeight: 700, fontSize: 13,
              }}>{g}</button>
            ))}
          </div>

          {/* Gene detail */}
          {gd && (
            <div style={{ flex: 1 }}>
              <div style={{ background: GENE_COLORS[selectedGene], borderRadius: 10, padding: '16px 20px', marginBottom: 16, color: '#fff' }}>
                <h2 style={{ margin: 0, fontSize: 22 }}>{gd.gene}</h2>
                <div style={{ opacity: 0.85, marginTop: 4, fontSize: 12 }}>{gd.protein}</div>
                <div style={{ marginTop: 8, fontSize: 12 }}>
                  <span style={{ background: 'rgba(255,255,255,0.2)', borderRadius: 4, padding: '2px 8px', marginRight: 8 }}>{gd.locus}</span>
                  <span style={{ background: 'rgba(255,255,255,0.2)', borderRadius: 4, padding: '2px 8px', marginRight: 8 }}>{gd.protein_size}</span>
                  <span style={{ background: 'rgba(255,255,255,0.2)', borderRadius: 4, padding: '2px 8px' }}>n={gd.cohort_size}</span>
                </div>
              </div>

              {/* Disease category */}
              <div style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4 style={{ margin: '0 0 8px', color: GENE_COLORS[selectedGene] }}>Disease Category</h4>
                <p style={{ margin: 0, fontSize: 13, color: '#333', lineHeight: 1.6 }}>{gd.disease_category}</p>
              </div>

              {/* Inheritance */}
              <div style={{ background: '#e8f5e9', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4 style={{ margin: '0 0 8px', color: '#2e7d32' }}>Inheritance & Mechanism</h4>
                <p style={{ margin: 0, fontSize: 13, color: '#1b5e20', lineHeight: 1.6 }}>{gd.inheritance}</p>
              </div>

              {/* Disease pathway */}
              <div style={{ background: '#e3f2fd', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4 style={{ margin: '0 0 8px', color: '#1565c0' }}>Disease Pathway</h4>
                <p style={{ margin: 0, fontSize: 13, color: '#0d47a1', lineHeight: 1.6 }}>{gd.disease_pathway}</p>
              </div>

              {/* Pathognomonic / Clinical Pearls */}
              <div style={{ background: '#fff8e1', border: '2px solid #f9a825', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4 style={{ margin: '0 0 8px', color: '#e65100' }}>⚡ Clinical Pearls & Pathognomonic Features</h4>
                <p style={{ margin: 0, fontSize: 13, color: '#bf360c', lineHeight: 1.6 }}>{gd.pathognomonic}</p>
              </div>

              {/* Key features */}
              <div style={{ background: '#f3e5f5', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4 style={{ margin: '0 0 8px', color: '#6a1b9a' }}>Key Clinical Features</h4>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {gd.key_features?.map((f, i) => <li key={i} style={{ fontSize: 13, color: '#4a148c', marginBottom: 4 }}>{f}</li>)}
                </ul>
              </div>

              {/* Treatment */}
              <div style={{ background: '#e8f5e9', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4 style={{ margin: '0 0 8px', color: '#2e7d32' }}>Treatment</h4>
                <p style={{ margin: 0, fontSize: 13, color: '#1b5e20', lineHeight: 1.6 }}>{gd.treatment}</p>
              </div>

              {/* Emergency protocol */}
              <div style={{ background: '#ffebee', border: '2px solid #b71c1c', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4 style={{ margin: '0 0 8px', color: '#b71c1c' }}>🚨 Emergency Protocol</h4>
                <p style={{ margin: 0, fontSize: 13, color: '#b71c1c', lineHeight: 1.6 }}>{gd.emergency_protocol}</p>
              </div>

              {/* DDx */}
              {gd.key_ddx && (
                <div style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                  <h4 style={{ margin: '0 0 8px', color: GENE_COLORS[selectedGene] }}>Key Differential Diagnoses</h4>
                  {Object.entries(gd.key_ddx).map(([k, v]) => (
                    <div key={k} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid #eee' }}>
                      <div style={{ fontWeight: 600, color: '#333', fontSize: 13, marginBottom: 4 }}>{k.replace(/_/g, ' ')}</div>
                      <div style={{ fontSize: 12, color: '#555', lineHeight: 1.5 }}>{v}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Systemic involvement */}
              <div style={{ background: '#e0f2f1', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4 style={{ margin: '0 0 8px', color: '#00695c' }}>Systemic Involvement</h4>
                <p style={{ margin: 0, fontSize: 13, color: '#004d40', lineHeight: 1.6 }}>{gd.systemic_involvement}</p>
              </div>

              {/* Cascade testing */}
              <div style={{ background: '#e8eaf6', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4 style={{ margin: '0 0 8px', color: '#283593' }}>Cascade Family Testing</h4>
                <p style={{ margin: 0, fontSize: 13, color: '#1a237e', lineHeight: 1.6 }}>{gd.cascade_testing}</p>
              </div>

              {/* Cohort stats */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginTop: 12 }}>
                {[
                  { label: 'Presentations', data: gd.presentation_distribution },
                  { label: 'Management', data: gd.management_distribution },
                  { label: 'Outcomes', data: gd.outcome_distribution },
                ].map(({ label, data }) => (
                  <div key={label} style={{ background: '#f5f5f5', borderRadius: 8, padding: 14 }}>
                    <h5 style={{ margin: '0 0 8px', color: '#555', fontSize: 12 }}>{label}</h5>
                    {data && Object.entries(data).map(([k, v]) => (
                      <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
                        <span style={{ color: '#666', maxWidth: '75%' }}>{k.replace(/_/g, ' ')}</span>
                        <span style={{ fontWeight: 600, color: GENE_COLORS[selectedGene] }}>{v}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* === DEFINITIONS === */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ background: '#e3f2fd', borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <h3 style={{ margin: 0, color: accentColor }}>{definitions.atlas_domain}</h3>
          </div>

          {/* Key definitions */}
          {Object.entries(definitions.key_definitions || {}).map(([k, v]) => (
            <div key={k} style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8, padding: 16, marginBottom: 14 }}>
              <h4 style={{ margin: '0 0 8px', color: accentColor, fontSize: 14 }}>{k.replace(/_/g, ' ')}</h4>
              <p style={{ margin: 0, fontSize: 13, color: '#333', lineHeight: 1.6 }}>{v}</p>
            </div>
          ))}

          {/* Clinical pearls */}
          <div style={{ background: '#fff8e1', border: '2px solid #f9a825', borderRadius: 10, padding: 20, marginTop: 20 }}>
            <h3 style={{ color: '#e65100', margin: '0 0 12px' }}>⚡ Clinical Pearls</h3>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {definitions.clinical_pearls?.map((p, i) => (
                <li key={i} style={{ color: '#bf360c', marginBottom: 8, fontSize: 13, lineHeight: 1.5 }}>{p}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
