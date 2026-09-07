'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PKD1:    '#dc2626',  // red    — ADPKD1; 85%; tolvaptan FDA2018; ESRD 54 yr
  PKD2:    '#2563eb',  // blue   — ADPKD2; 15%; milder; ESRD 74 yr
  PKHD1:   '#7c3aed',  // purple — ARPKD; CHF always; Potter sequence
  MUC1:    '#ea580c',  // orange — ADTKD-MUC1; NGS misses VNTR; toxic MUC1-fs
  UMOD:    '#0f766e',  // teal   — ADTKD-UMOD; gout teens PATHOGNOMONIC; allopurinol
  REN:     '#ca8a04',  // amber  — ADTKD-REN; childhood anemia; NO ACE-i/ARB
  HNF1B:   '#16a34a',  // green  — RCAD; MLPA mandatory; SU INEFFECTIVE; MODY5
  DNAJB11: '#be185d',  // pink   — ADTKD-DNAJB11; newest 2018; atypical polycystic
};

const GENE_DISEASE = {
  PKD1:    'AD ADPKD1 Polycystin-1 — PKD1-4304aa — 16p13.3 — Most-Common-85pct-ADPKD — ESRD-Median-54yr — Tolvaptan-V2R-FDA2018-FIRST-ADPKD — ICA-4x-Risk — Mayo-1C-1E-Rapid',
  PKD2:    'AD ADPKD2 Polycystin-2/TRPP2 — PKD2-968aa — 4q22.1 — 15pct-ADPKD-MILDER — ESRD-Median-74yr-20yr-Later-PKD1 — Tolvaptan-If-Rapid-1C-1E — Genetic-Test-Distinguishes-PKD1-PKD2',
  PKHD1:   'AR ARPKD Fibrocystin — PKHD1-4074aa — 6p21.2 — Most-Common-Inherited-Renal-Cystic-Children-1:20000 — CHF-ALWAYS-Portal-HTN — Collecting-Duct-Ectasia-NOT-Balloon-Cysts — Potter-Sequence-Neonatal',
  MUC1:    'AD ADTKD-MUC1 Mucin-1 — MUC1-1255aa — 1q22 — STANDARD-NGS-MISSES-VNTR — Tubulointerstitial-NO-Cysts — MUC1-fs-Toxic-ER-Stress — Long-Read-Assay-Required — ESRD-5th-6th-Decade',
  UMOD:    'AD ADTKD-UMOD Uromodulin — UMOD-640aa — 16p12.3 — Gout-Hyperuricemia-Teens-20s-PATHOGNOMONIC — Most-Abundant-Urinary-Protein — Allopurinol-Early — Medullary-Cysts-MRI-Better — ZP-Domain-Cysteine',
  REN:     'AD ADTKD-REN Renin — REN-406aa — 1q32.1 — Childhood-Anemia-Hyperkalemia-Low-BP — Low-Renin-State — NO-ACE-I-ARB-Worsen-K — Rarest-ADTKD — ESA-Required — Childhood-Clue',
  HNF1B:   'AD RCAD HNF1beta — HNF1B-557aa — 17q12 — MLPA-Mandatory-50pct-17q12-Deletion — MODY5-Insulin-Required-SU-INEFFECTIVE — Pancreatic-Hypoplasia-PERT — Hypomagnesaemia-TRIAD — Bicornuate-Uterus-30pct',
  DNAJB11: 'AD ADTKD-DNAJB11 DnaJ-B11 — DNAJB11-354aa — 3q27.3 — Newest-ADTKD-2018 — Atypical-Polycystic-Mimics-PKD1-PKD2 — PKD1-PKD2-Negative-Next-DNAJB11 — ER-Co-Chaperone — ESRD-6th-7th-Decade',
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

export default function HereditaryRenalCysticDiseaseAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/hereditary-renal-cystic-disease-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-renal-cystic-disease-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-renal-cystic-disease-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const pageStyle = { background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace' };
  const headerStyle = { background: '#1e293b', borderBottom: '1px solid #334155', padding: '1rem 2rem' };
  const tabBarStyle = { display: 'flex', gap: 4, padding: '0.75rem 2rem', background: '#1e293b', borderBottom: '1px solid #334155' };

  return (
    <div style={pageStyle}>
      <div style={headerStyle}>
        <h1 style={{ margin: 0, fontSize: 20, color: '#38bdf8' }}>
          🧬 Hereditary-Renal-Cystic-Disease-Atlas
        </h1>
        <p style={{ margin: '4px 0 0', fontSize: 12, color: '#94a3b8' }}>
          Complete 8-Gene Hereditary Renal Cystic Disease Atlas —
          PKD1/Polycystin1-4304aa-16p13.3-AD-ADPKD1-85pct-Tolvaptan-FDA2018 ·
          PKD2/TRPP2-968aa-4q22.1-AD-ADPKD2-MILDER-ESRD-74yr ·
          PKHD1/Fibrocystin-4074aa-6p21.2-AR-ARPKD-CHF-ALWAYS ·
          MUC1-1255aa-1q22-AD-ADTKD-NGS-MISSES-VNTR ·
          UMOD-640aa-16p12.3-AD-Gout-Teens-PATHOGNOMONIC ·
          REN-406aa-1q32.1-AD-Childhood-Anemia-NO-ACE-I ·
          HNF1B-557aa-17q12-AD-RCAD-MLPA-Mandatory-SU-INEFFECTIVE ·
          DNAJB11-354aa-3q27.3-AD-Newest-ADTKD-2018 ·
          320-Patient-Aggregate-8x40-seeds-1918–1925
        </p>
      </div>

      <div style={tabBarStyle}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#2563eb' : '#334155',
            color: '#e2e8f0', border: 'none', borderRadius: 6,
            padding: '6px 16px', cursor: 'pointer', fontSize: 13, fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '1.5rem 2rem' }}>
        {loading && <Loading />}
        {error && <ErrorBox msg={error} />}

        {/* ── OVERVIEW TAB ── */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: '1.5rem' }}>
              <KPI label="Total Patients" value={overview.aggregate_stats.total_patients} color="#38bdf8" />
              <KPI label="Genes Covered" value={overview.aggregate_stats.genes_covered} color="#a78bfa" />
              <KPI label="Avg Age (yr)" value={overview.aggregate_stats.avg_age_at_diagnosis_yr} color="#34d399" />
              <KPI label="Severe Cases %" value={`${overview.aggregate_stats.severe_cases_pct}%`} color="#f87171" />
              <KPI label="Seeds" value={overview.aggregate_stats.seed_range} color="#fbbf24" />
            </div>

            <h3 style={{ color: '#94a3b8', fontSize: 13, marginBottom: 8 }}>KEY CLINICAL DISTINCTIONS</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: '1.5rem' }}>
              {overview.key_clinical_distinctions.map((d, i) => {
                const gene = d.split('-')[0];
                const color = GENE_COLORS[gene] || '#6366f1';
                return (
                  <div key={i} style={{
                    background: '#1e293b', borderRadius: 8, padding: '8px 12px',
                    borderLeft: `3px solid ${color}`, fontSize: 12, color: '#cbd5e1',
                  }}>
                    {d}
                  </div>
                );
              })}
            </div>

            <h3 style={{ color: '#94a3b8', fontSize: 13, marginBottom: 8 }}>GENE REGISTRY</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 12 }}>
              {overview.gene_summary.map(g => (
                <div key={g.gene} style={{
                  background: '#1e293b', borderRadius: 10, padding: '1rem',
                  borderTop: `3px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', fontSize: 16 }}>{g.gene}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{g.alt_name}</div>
                  <div style={{ marginTop: 6, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    <Badge text={g.locus} color={GENE_COLORS[g.gene] || '#6366f1'} />
                    <Badge text={g.protein_size} color="#64748b" />
                    <Badge text={g.inheritance} color={g.inheritance === 'AR' ? '#f97316' : '#22d3ee'} />
                    <Badge text={`n=${g.n_patients}`} color="#475569" />
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 6 }}>{GENE_DISEASE[g.gene]}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── GENE TABLE TAB ── */}
        {tab === 'Gene Table' && breakdown && (
          <div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
                  {['Gene', 'Alt Name', 'Locus', 'Size', 'Inh.', 'Onset', 'n', 'Mild', 'Mod', 'Severe'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #334155' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.genes.map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.alt_name}</td>
                    <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.locus}</td>
                    <td style={{ padding: '8px 10px', color: '#64748b' }}>{g.protein_size}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <Badge text={g.inheritance} color={g.inheritance === 'AR' ? '#f97316' : '#22d3ee'} />
                    </td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8', maxWidth: 200, fontSize: 11 }}>{g.age_of_onset}</td>
                    <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.n_patients}</td>
                    <td style={{ padding: '8px 10px', color: '#34d399' }}>{g.severity_distribution.mild || 0}</td>
                    <td style={{ padding: '8px 10px', color: '#fbbf24' }}>{g.severity_distribution.moderate || 0}</td>
                    <td style={{ padding: '8px 10px', color: '#f87171' }}>{g.severity_distribution.severe || 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* ── CLINICAL ATLAS TAB ── */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {breakdown.genes.map(g => (
              <div key={g.gene} style={{
                background: '#1e293b', borderRadius: 12, padding: '1.2rem',
                borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <span style={{ fontSize: 20, fontWeight: 800, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
                  <span style={{ fontSize: 13, color: '#94a3b8' }}>{g.alt_name}</span>
                  <Badge text={g.locus} color={GENE_COLORS[g.gene] || '#6366f1'} />
                  <Badge text={g.protein_size} color="#64748b" />
                  <Badge text={g.inheritance} color={g.inheritance === 'AR' ? '#f97316' : '#22d3ee'} />
                </div>

                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600, marginBottom: 4 }}>KEY BIOMARKER</div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.5 }}>{g.key_biomarker}</div>
                </div>

                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600, marginBottom: 4 }}>PATHOGNOMONIC</div>
                  <div style={{ fontSize: 12, color: '#fde68a', lineHeight: 1.5 }}>{g.pathognomonic}</div>
                </div>

                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600, marginBottom: 4 }}>TREATMENT</div>
                  <div style={{ fontSize: 12, color: '#86efac', lineHeight: 1.5 }}>{g.treatment}</div>
                </div>

                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600, marginBottom: 4 }}>CRITICAL FLAGS</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {g.critical_flags.map((flag, fi) => (
                      <div key={fi} style={{
                        background: '#0f172a', borderRadius: 6, padding: '5px 10px',
                        fontSize: 11, color: '#fca5a5', borderLeft: `2px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                      }}>{flag}</div>
                    ))}
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600, marginBottom: 4 }}>SAMPLE PATIENTS</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {(g.patients || []).slice(0, 5).map((p, pi) => (
                      <div key={pi} style={{
                        background: '#0f172a', borderRadius: 6, padding: '6px 10px', fontSize: 11,
                        border: `1px solid ${p.severity === 'severe' ? '#dc2626' : p.severity === 'moderate' ? '#ca8a04' : '#16a34a'}44`,
                      }}>
                        <div style={{ color: '#94a3b8' }}>{p.patient_id}</div>
                        <div style={{ color: '#cbd5e1' }}>age {p.age} · {p.severity}</div>
                        <div style={{ color: '#64748b' }}>{p.key_feature}</div>
                        <div style={{ color: '#60a5fa', fontSize: 10 }}>{p.current_therapy}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {tab === 'Definitions' && definitions && (
          <div>
            <h3 style={{ color: '#94a3b8', fontSize: 13, marginBottom: 12 }}>GENE DEFINITIONS</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: '2rem' }}>
              {definitions.genes.map(g => (
                <div key={g.gene} style={{
                  background: '#1e293b', borderRadius: 10, padding: '1rem',
                  borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', fontSize: 15, marginBottom: 4 }}>
                    {g.gene} — {g.alt_name}
                  </div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
                    <Badge text={g.locus} color={GENE_COLORS[g.gene] || '#6366f1'} />
                    <Badge text={g.protein_size} color="#64748b" />
                    <Badge text={g.inheritance} color={g.inheritance === 'AR' ? '#f97316' : '#22d3ee'} />
                    <Badge text={g.age_of_onset} color="#475569" />
                  </div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>{g.definition}</div>
                </div>
              ))}
            </div>

            <h3 style={{ color: '#94a3b8', fontSize: 13, marginBottom: 12 }}>GLOSSARY</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {Object.entries(definitions.glossary || {}).map(([term, def]) => (
                <div key={term} style={{
                  background: '#1e293b', borderRadius: 8, padding: '8px 12px',
                  borderLeft: '3px solid #475569',
                }}>
                  <div style={{ fontWeight: 700, color: '#fbbf24', fontSize: 12, marginBottom: 4 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5 }}>{def}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
