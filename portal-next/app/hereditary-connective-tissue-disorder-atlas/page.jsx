'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  FBN1:   '#7c3aed',  // purple  — Marfan; aortic root; ectopia lentis; losartan; Revised Ghent 2010
  COL3A1: '#dc2626',  // red     — vEDS; arterial rupture NO warning; celiprolol ONLY; NO elective surgery
  TGFBR2: '#ea580c',  // orange  — Loeys-Dietz LDS2; surgery 4.0 cm; whole-body imaging; bifid uvula
  COL1A1: '#0f766e',  // teal    — OI; bisphosphonates; blue sclerae; wormian bones; glycine vs null
  ELN:    '#ca8a04',  // amber   — SVAS/Williams; 7q11.23 deletion; elfin facies; anaesthetic risk
  ABCC6:  '#2563eb',  // blue    — PXE; angioid streaks PATHOGNOMONIC; anti-VEGF; vitamin K2; NO trauma
  COL5A1: '#be185d',  // pink    — cEDS; atrophic scars required; molluscoid pseudotumours; no aortic risk
  TNXB:   '#16a34a',  // green   — TNX-EDS; serum TNX only EDS biomarker; CAH-X adrenal; fibril spacing
};

const GENE_DISEASE = {
  FBN1:   'AD Marfan-Syndrome — FBN1-2871aa — 15q21.1 — Aortic-Root-Dilation — Ectopia-Lentis-UPWARD-TEMPORAL — Losartan-ARB — Elective-Root-Repair-4.5-5.0cm — Revised-Ghent-2010',
  COL3A1: 'AD Vascular-EDS — COL3A1-1466aa — 2q32.2 — Arterial-Rupture-WITHOUT-Warning — Celiprolol-Level-B-ONLY — NO-Elective-Surgery-ABSOLUTE-CI — Most-Lethal-EDS-Median-48yr',
  TGFBR2: 'AD Loeys-Dietz-LDS2 — TGFBR2-592aa — 3p24.1 — Surgery-4.0cm-NOT-4.5cm — Whole-Body-Imaging-Mandatory — Hypertelorism-Bifid-Uvula-Triad — Arterial-Tortuosity',
  COL1A1: 'AD Osteogenesis-Imperfecta — COL1A1-1464aa — 17q21.33 — Bisphosphonates-Fracture-40-50pct — Blue-Sclerae-Type-I — Wormian-Bones-Skull-XR — Glycine-Sub-Severe-Null-Mild',
  ELN:    'AD/deletion SVAS-Williams-Beuren — ELN-786aa — 7q11.23 — 7q11.23-Deletion-WBS-25-Genes — Isolated-SVAS-Point-Mutation — Elfin-Facies-Hypercalcemia — Anaesthetic-Sudden-Death-Risk',
  ABCC6:  'AR Pseudoxanthoma-Elasticum — ABCC6-1503aa — 16p13.1 — Angioid-Streaks-PATHOGNOMONIC — Anti-VEGF-CNV — Vitamin-K2-MK7 — Avoid-Ocular-Trauma-ABSOLUTE — Premature-PAD',
  COL5A1: 'AD Classical-EDS-cEDS — COL5A1-2836aa — 9q34.3 — Atrophic-Scars-REQUIRED-Diagnosis — Molluscoid-Pseudotumours-PATHOGNOMONIC — Gorlin-Sign — 50pct-De-Novo — No-Aortic-Risk',
  TNXB:   'AR/AD-haplo TNX-EDS — TNXB-4243aa — 6p21.3 — Serum-TNX-ONLY-EDS-Biomarker — ZERO=Homozygous-LOF-50pct=Haploinsufficiency — CAH-X-CYP21A2-Contiguous-Adrenal-Crisis',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#94a3b8' }}>Loading…</div>;
}

function ErrorBox({ msg }) {
  return (
    <div style={{ padding: '1rem', background: '#450a0a', borderRadius: 8, color: '#fca5a5', margin: '1rem 0' }}>
      Error: {msg}
    </div>
  );
}

function KPI({ label, value, color }) {
  return (
    <div style={{
      background: '#1e293b', borderRadius: 10, padding: '1rem 1.2rem',
      borderLeft: `4px solid ${color || '#6366f1'}`, minWidth: 160,
    }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#a5b4fc' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{label}</div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 6, padding: '2px 8px', fontSize: 11, fontWeight: 600,
    }}>{text}</span>
  );
}

export default function HereditaryConnectiveTissueDisorderAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/hereditary-connective-tissue-disorder-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-connective-tissue-disorder-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-connective-tissue-disorder-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const tabStyle = (t) => ({
    padding: '0.5rem 1.1rem', borderRadius: 8, cursor: 'pointer', fontWeight: 600,
    fontSize: 13, border: 'none',
    background: tab === t ? '#6366f1' : '#1e293b',
    color: tab === t ? '#fff' : '#94a3b8',
  });

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'system-ui,sans-serif', padding: '1.5rem' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>

        {/* Header */}
        <div style={{ marginBottom: '1.5rem' }}>
          <h1 style={{ fontSize: 22, fontWeight: 800, color: '#f8fafc', margin: 0 }}>
            🧬 Hereditary Connective Tissue Disorder Atlas
          </h1>
          <p style={{ color: '#64748b', fontSize: 12, margin: '4px 0 0' }}>
            Complete 8-Gene Atlas — FBN1/Marfan · COL3A1/vEDS · TGFBR2/LDS2 · COL1A1/OI · ELN/SVAS · ABCC6/PXE · COL5A1/cEDS · TNXB/TNX-EDS — 320 patients · seeds 1942–1949
          </p>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 8, marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
        </div>

        {loading && <Loading />}
        {error && <ErrorBox msg={error} />}

        {/* ── OVERVIEW TAB ── */}
        {tab === 'Overview' && overview && (
          <div>
            {/* KPI row */}
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: '1.5rem' }}>
              <KPI label="Total Patients" value={overview.aggregate_stats.total_patients} color="#6366f1" />
              <KPI label="Genes Covered" value={overview.aggregate_stats.genes_covered} color="#22d3ee" />
              <KPI label="Avg Age at Dx (yr)" value={overview.aggregate_stats.avg_age_at_diagnosis_yr} color="#f59e0b" />
              <KPI label="Severe Cases %" value={`${overview.aggregate_stats.severe_cases_pct}%`} color="#ef4444" />
              <KPI label="Seed Range" value={overview.aggregate_stats.seed_range} color="#a78bfa" />
            </div>

            {/* Gene chips */}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: '1.5rem' }}>
              {overview.gene_summary.map(g => (
                <div key={g.gene} style={{
                  background: GENE_COLORS[g.gene] + '22', border: `1px solid ${GENE_COLORS[g.gene]}55`,
                  borderRadius: 8, padding: '0.5rem 0.9rem',
                }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene], fontSize: 13 }}>{g.gene}</div>
                  <div style={{ fontSize: 10, color: '#94a3b8' }}>{g.locus} · {g.protein_size} · {g.inheritance}</div>
                  <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{g.n_patients} patients</div>
                </div>
              ))}
            </div>

            {/* Key distinctions */}
            <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem 1.2rem' }}>
              <div style={{ fontWeight: 700, color: '#f8fafc', marginBottom: 8, fontSize: 13 }}>
                ⚠️ Key Clinical Distinctions
              </div>
              {overview.key_clinical_distinctions.map((d, i) => {
                const gene = d.split('-')[0];
                const color = GENE_COLORS[gene] || '#6366f1';
                return (
                  <div key={i} style={{
                    borderLeft: `3px solid ${color}`, paddingLeft: 10, marginBottom: 8,
                    fontSize: 12, color: '#cbd5e1', lineHeight: 1.5,
                  }}>
                    {d}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── GENE TABLE TAB ── */}
        {tab === 'Gene Table' && breakdown && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1e293b' }}>
                  {['Gene', 'Disease', 'Locus', 'Size', 'Inheritance', 'Patients', 'Severe'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.genes.map((g, i) => {
                  const color = GENE_COLORS[g.gene] || '#6366f1';
                  return (
                    <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1a2234', borderBottom: '1px solid #1e293b' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 700, color }}>{g.gene}</td>
                      <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.alt_name}</td>
                      <td style={{ padding: '8px 10px', color: '#94a3b8', fontFamily: 'monospace' }}>{g.locus}</td>
                      <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.protein_size}</td>
                      <td style={{ padding: '8px 10px' }}><Badge text={g.inheritance} color={color} /></td>
                      <td style={{ padding: '8px 10px', color: '#f8fafc', fontWeight: 600 }}>{g.n_patients}</td>
                      <td style={{ padding: '8px 10px', color: '#ef4444' }}>{g.severity_distribution?.severe ?? 0}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* ── CLINICAL ATLAS TAB ── */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {breakdown.genes.map(g => {
              const color = GENE_COLORS[g.gene] || '#6366f1';
              return (
                <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, padding: '1rem 1.2rem', borderLeft: `4px solid ${color}` }}>
                  <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', flexWrap: 'wrap', marginBottom: 6 }}>
                    <span style={{ fontWeight: 800, color, fontSize: 16 }}>{g.gene}</span>
                    <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.alt_name}</span>
                    <Badge text={g.inheritance} color={color} />
                    <span style={{ color: '#64748b', fontSize: 11, fontFamily: 'monospace' }}>{g.locus} · {g.protein_size}</span>
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>{GENE_DISEASE[g.gene]}</div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
                    <div style={{ background: '#0f172a', borderRadius: 6, padding: '0.6rem 0.8rem' }}>
                      <div style={{ fontSize: 10, color: '#475569', fontWeight: 700, marginBottom: 3 }}>AGE OF ONSET</div>
                      <div style={{ fontSize: 11, color: '#cbd5e1' }}>{g.age_of_onset}</div>
                    </div>
                    <div style={{ background: '#0f172a', borderRadius: 6, padding: '0.6rem 0.8rem' }}>
                      <div style={{ fontSize: 10, color: '#475569', fontWeight: 700, marginBottom: 3 }}>KEY BIOMARKER</div>
                      <div style={{ fontSize: 11, color: '#cbd5e1' }}>{g.key_biomarker}</div>
                    </div>
                  </div>

                  <div style={{ background: '#0f172a', borderRadius: 6, padding: '0.6rem 0.8rem', marginBottom: 8 }}>
                    <div style={{ fontSize: 10, color: '#475569', fontWeight: 700, marginBottom: 3 }}>PATHOGNOMONIC / DDx</div>
                    <div style={{ fontSize: 11, color: '#fde68a' }}>{g.pathognomonic}</div>
                  </div>

                  <div style={{ background: '#0f172a', borderRadius: 6, padding: '0.6rem 0.8rem', marginBottom: 8 }}>
                    <div style={{ fontSize: 10, color: '#475569', fontWeight: 700, marginBottom: 3 }}>TREATMENT</div>
                    <div style={{ fontSize: 11, color: '#bbf7d0' }}>{g.treatment}</div>
                  </div>

                  <div>
                    <div style={{ fontSize: 10, color: '#475569', fontWeight: 700, marginBottom: 4 }}>CRITICAL FLAGS</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                      {g.critical_flags.map((f, i) => (
                        <div key={i} style={{
                          fontSize: 11, color: '#fca5a5', background: '#450a0a44',
                          borderRadius: 4, padding: '3px 7px', borderLeft: `2px solid ${color}`,
                        }}>⚑ {f}</div>
                      ))}
                    </div>
                  </div>

                  <div style={{ marginTop: 8 }}>
                    <div style={{ fontSize: 10, color: '#475569', fontWeight: 700, marginBottom: 4 }}>SAMPLE PATIENTS (5/{g.n_patients})</div>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {g.patients.map(p => (
                        <div key={p.patient_id} style={{
                          background: '#0f172a', borderRadius: 6, padding: '4px 8px', fontSize: 10,
                          border: `1px solid ${color}33`,
                        }}>
                          <span style={{ color: '#94a3b8' }}>Age {p.age}</span>
                          {' · '}
                          <span style={{ color: p.severity === 'severe' ? '#ef4444' : p.severity === 'moderate' ? '#f59e0b' : '#22d3ee' }}>
                            {p.severity}
                          </span>
                          {' · '}
                          <span style={{ color: '#cbd5e1' }}>{p.key_feature}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {tab === 'Definitions' && definitions && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Gene definitions */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {definitions.genes.map(g => {
                const color = GENE_COLORS[g.gene] || '#6366f1';
                return (
                  <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem 1rem', borderLeft: `3px solid ${color}` }}>
                    <div style={{ fontWeight: 700, color, fontSize: 14, marginBottom: 2 }}>{g.gene} — {g.alt_name}</div>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4, fontFamily: 'monospace' }}>
                      {g.locus} · {g.protein_size} · {g.inheritance}
                    </div>
                    <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>{g.definition}</div>
                    <div style={{ fontSize: 10, color: '#475569', marginBottom: 4 }}><strong>Age of onset:</strong> {g.age_of_onset}</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      {g.critical_flags.map((f, i) => (
                        <div key={i} style={{ fontSize: 10, color: '#fca5a5', paddingLeft: 8, borderLeft: `1px solid ${color}55` }}>
                          {f}
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Glossary */}
            <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem 1.2rem' }}>
              <div style={{ fontWeight: 700, color: '#f8fafc', marginBottom: 10, fontSize: 14 }}>📖 Glossary</div>
              {Object.entries(definitions.glossary).map(([term, def]) => (
                <div key={term} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid #0f172a' }}>
                  <div style={{ fontWeight: 700, color: '#a5b4fc', fontSize: 12, marginBottom: 3 }}>{term}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.6 }}>{def}</div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
