'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  FKRP:    '#7c3aed',  // purple  — LGMD2I; L276I founder; most common AR LGMD N.Europe; cardiac
  FKTN:    '#dc2626',  // red     — Fukuyama CMD; Japan; SVA founder; cardiac transplant
  POMT1:   '#0f766e',  // teal    — Walker-Warburg type 1; most lethal; lissencephaly
  POMT2:   '#2563eb',  // blue    — Walker-Warburg type 2; Dandy-Walker; cobblestone
  POMGNT1: '#ea580c',  // orange  — MEB; progressive myopia key clue; cerebellar cysts
  LARGE1:  '#ca8a04',  // amber   — MDC1D; white matter; most severe cognition
  ISPD:    '#be185d',  // pink    — WWS+cardiomyopathy; Leigh-like MRI trap; Arg272Cys
  GMPPB:   '#16a34a',  // green   — LGMD2T; NMJ; pyridostigmine responsive; normal brain
};

const GENE_DISEASE = {
  FKRP:    'AR LGMD2I/MDC1C — FKRP-495aa — 19q13.32 — L276I-Caucasian-Founder-80pct — Most-Common-AR-LGMD-Northern-Europe — Cardiomyopathy-DCM-20-30pct-MANDATORY-Cardiac-Surveillance — Steroids-NOT-Effective',
  FKTN:    'AR Fukuyama-CMD/LGMD2M — FKTN-461aa — 9q31.2 — 3kb-SVA-Retrotransposon-Founder-87pct-Japanese — Cobblestone-Lissencephaly — Cardiomyopathy-2nd-3rd-Decade-Cardiac-Transplant — SVA-NOT-Detected-by-Standard-WES',
  POMT1:   'AR Walker-Warburg-WWS1/LGMD2K — POMT1-747aa — 9q34.13 — Most-Severe-Dystroglycanopathy — Cobblestone-Lissencephaly — Eye-Malformations — Median-Survival-<3yr — Mild-Alleles-LGMD2K',
  POMT2:   'AR Walker-Warburg-WWS2/LGMD2N — POMT2-750aa — 14q24.3 — Clinically-Indistinguishable-from-POMT1-Gene-Testing-Essential — Dandy-Walker-Malformation — Cobblestone-Lissencephaly',
  POMGNT1: 'AR Muscle-Eye-Brain-MEB/Santavuori — POMGNT1-740aa — 1p34.1 — Progressive-HIGH-Myopia-PATHOGNOMONIC-Clue — Flat-ERG — Cerebellar-Cysts — Better-Survival-Than-WWS — DISTINGUISH-from-FCMD',
  LARGE1:  'AR MDC1D/LGMD2L — LARGE1-756aa — 22q12.3 — Most-Severe-Intellectual-Disability — Periventricular-White-Matter-Leukodystrophy — Cerebellar-Cysts — LARGE2-Does-NOT-Compensate',
  ISPD:    'AR WWS-Subtype/LGMD — ISPD-352aa — 7p21.2 — CDP-Ribitol-Pyrophosphorylase — Cardiomyopathy-Distinctive — Leigh-Like-MRI-Trap-Metabolic-Screen-NORMAL — Arg272Cys-Northern-European-Founder',
  GMPPB:   'AR LGMD2T-Myasthenic-Dystroglycanopathy — GMPPB-395aa — 3p24.3 — Fluctuating-Weakness-Decrement-EMG — Pyridostigmine-RESPONSIVE-ONLY-Dystroglycanopathy-with-NMJ — Normal-Intelligence-Normal-Brain-MRI — CK-Elevated-5-50x',
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

export default function HereditaryDystroglycanopathyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/hereditary-dystroglycanopathy-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-dystroglycanopathy-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-dystroglycanopathy-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const tabStyle = (t) => ({
    padding: '0.5rem 1.2rem', cursor: 'pointer', borderRadius: '6px 6px 0 0',
    fontWeight: tab === t ? 700 : 400,
    background: tab === t ? '#1e40af' : '#1e293b',
    color: tab === t ? '#fff' : '#94a3b8',
    border: 'none', fontSize: 13,
  });

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', padding: '1.5rem' }}>
      <div style={{ maxWidth: 1400, margin: '0 auto' }}>

        {/* Header */}
        <div style={{ marginBottom: '1.5rem' }}>
          <h1 style={{ fontSize: 22, fontWeight: 800, color: '#7c3aed', margin: 0 }}>
            🧬 Hereditary-Dystroglycanopathy-Atlas
          </h1>
          <p style={{ color: '#94a3b8', margin: '0.3rem 0 0', fontSize: 13 }}>
            Complete 8-Gene Alpha-Dystroglycan O-Glycosylation Disorder Atlas —
            FKRP · FKTN · POMT1 · POMT2 · POMGNT1 · LARGE1 · ISPD · GMPPB —
            320 patients · seeds 1958–1965
          </p>
          <p style={{ color: '#64748b', margin: '0.2rem 0 0', fontSize: 11 }}>
            Spectrum: Walker-Warburg (lethal) → Fukuyama CMD → MEB → MDC1D → LGMD2I (FKRP, most common AR LGMD N.Europe) → LGMD2T (GMPPB, pyridostigmine responsive)
          </p>
        </div>

        {loading && <Loading />}
        {error && <ErrorBox msg={error} />}

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 0, flexWrap: 'wrap' }}>
          {TABS.map(t => (
            <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>
          ))}
        </div>
        <div style={{ background: '#1e293b', borderRadius: '0 8px 8px 8px', padding: '1.5rem' }}>

          {/* ── OVERVIEW TAB ── */}
          {tab === 'Overview' && overview && (
            <div>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: '1.5rem' }}>
                <KPI label="Total Patients" value={overview.total_patients} color="#7c3aed" />
                <KPI label="CMD (severe)" value={overview.cmd_patients} color="#dc2626" />
                <KPI label="LGMD (milder)" value={overview.lgmd_patients} color="#2563eb" />
                <KPI label="Cardiomyopathy" value={overview.cardiomyopathy_patients} color="#ea580c" />
                <KPI label="Brain Abnormalities" value={overview.brain_abnormality_patients} color="#0f766e" />
                <KPI label="Respiratory Support" value={overview.respiratory_support_patients} color="#be185d" />
                <KPI label="Ambulation Preserved" value={overview.ambulation_preserved_patients} color="#16a34a" />
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <h3 style={{ color: '#7c3aed', fontSize: 14, marginBottom: 8 }}>Pathway</h3>
                <div style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem 1rem', fontSize: 12, color: '#cbd5e1' }}>
                  {overview.pathway}
                </div>
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <h3 style={{ color: '#7c3aed', fontSize: 14, marginBottom: 8 }}>Key Clinical Insight</h3>
                <div style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem 1rem', fontSize: 12, color: '#cbd5e1' }}>
                  {overview.key_clinical_insight}
                </div>
              </div>

              <div>
                <h3 style={{ color: '#7c3aed', fontSize: 14, marginBottom: 8 }}>Genes in This Atlas</h3>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {overview.genes?.map(g => (
                    <Badge key={g} text={g} color={GENE_COLORS[g] || '#6366f1'} />
                  ))}
                </div>
                <div style={{ marginTop: '1rem', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {overview.genes?.map(g => (
                    <div key={g} style={{
                      background: '#0f172a', borderRadius: 8, padding: '0.5rem 0.8rem',
                      borderLeft: `3px solid ${GENE_COLORS[g] || '#6366f1'}`, minWidth: 100,
                    }}>
                      <div style={{ fontWeight: 700, color: GENE_COLORS[g] || '#a5b4fc', fontSize: 13 }}>{g}</div>
                      <div style={{ color: '#94a3b8', fontSize: 10 }}>{overview.gene_patient_counts?.[g]} pts</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── GENE TABLE TAB ── */}
          {tab === 'Gene Table' && breakdown && (
            <div>
              <h3 style={{ color: '#7c3aed', fontSize: 14, marginBottom: '1rem' }}>
                Per-Gene Breakdown — Dystroglycanopathy Spectrum
              </h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#0f172a', color: '#64748b' }}>
                      {['Gene', 'Locus', 'Size', 'Inh.', 'N', 'CMD%', 'LGMD%', 'Cardiac%', 'Amb.%', 'Disease / Key Fact'].map(h => (
                        <th key={h} style={{ padding: '0.5rem 0.7rem', textAlign: 'left', whiteSpace: 'nowrap', fontSize: 11 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.values(breakdown).map((g, i) => (
                      <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                        <td style={{ padding: '0.5rem 0.7rem', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.locus}</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.protein_size}</td>
                        <td style={{ padding: '0.5rem 0.7rem' }}>
                          <Badge text={g.inheritance} color={g.inheritance === 'AR' ? '#16a34a' : g.inheritance === 'AD' ? '#2563eb' : '#7c3aed'} />
                        </td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#e2e8f0', fontWeight: 600 }}>{g.n_patients}</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#dc2626' }}>{g.cmd_pct}%</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#2563eb' }}>{g.lgmd_pct}%</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#ea580c' }}>{g.cardiomyopathy_pct}%</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#16a34a' }}>{g.ambulation_preserved_pct}%</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#cbd5e1', fontSize: 10, maxWidth: 300 }}>
                          {GENE_DISEASE[g.gene] || g.alt_name}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── CLINICAL ATLAS TAB ── */}
          {tab === 'Clinical Atlas' && breakdown && (
            <div>
              <h3 style={{ color: '#7c3aed', fontSize: 14, marginBottom: '1rem' }}>
                Clinical Atlas — 8 Dystroglycanopathy Genes
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                {Object.values(breakdown).map(g => (
                  <div key={g.gene} style={{
                    background: '#0f172a', borderRadius: 10,
                    borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                    padding: '1rem 1.2rem',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
                      <span style={{ fontSize: 16, fontWeight: 800, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
                      <Badge text={g.inheritance} color={g.inheritance === 'AR' ? '#16a34a' : '#2563eb'} />
                      <span style={{ color: '#64748b', fontSize: 11 }}>{g.locus} · {g.protein_size}</span>
                      <span style={{ color: '#64748b', fontSize: 11 }}>{g.n_patients} patients</span>
                    </div>
                    <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 8 }}>{g.alt_name}</div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10 }}>
                      {[
                        { label: 'Age of Onset', val: g.age_of_onset, color: '#7c3aed' },
                        { label: 'Key Biomarker', val: g.key_biomarker, color: '#0f766e' },
                        { label: 'Pathognomonic / DDx', val: g.pathognomonic, color: '#dc2626' },
                        { label: 'Treatment', val: g.treatment, color: '#2563eb' },
                      ].map(({ label, val, color }) => (
                        <div key={label} style={{ background: '#1e293b', borderRadius: 8, padding: '0.7rem' }}>
                          <div style={{ fontSize: 10, color, fontWeight: 700, marginBottom: 4, textTransform: 'uppercase' }}>{label}</div>
                          <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.5 }}>{val}</div>
                        </div>
                      ))}
                    </div>

                    <div style={{ marginTop: 10 }}>
                      <div style={{ fontSize: 10, color: '#ca8a04', fontWeight: 700, marginBottom: 6, textTransform: 'uppercase' }}>
                        ⚠ Critical Flags
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                        {g.critical_flags?.map((flag, fi) => (
                          <div key={fi} style={{
                            background: '#1e293b', borderRadius: 6, padding: '0.4rem 0.7rem',
                            borderLeft: '3px solid #ca8a04', fontSize: 11, color: '#fef3c7',
                          }}>
                            {flag}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── DEFINITIONS TAB ── */}
          {tab === 'Definitions' && definitions && (
            <div>
              <h3 style={{ color: '#7c3aed', fontSize: 14, marginBottom: '1rem' }}>
                Gene Definitions & Glossary
              </h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.7rem', marginBottom: '1.5rem' }}>
                {Object.entries(definitions.gene_definitions || {}).map(([gene, def]) => (
                  <div key={gene} style={{
                    background: '#0f172a', borderRadius: 8, padding: '0.8rem 1rem',
                    borderLeft: `3px solid ${GENE_COLORS[gene] || '#6366f1'}`,
                  }}>
                    <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc', fontSize: 13, marginBottom: 4 }}>
                      {gene} — {def.locus} · {def.protein_size} · {def.inheritance}
                    </div>
                    <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{def.protein}</div>
                  </div>
                ))}
              </div>

              <h3 style={{ color: '#7c3aed', fontSize: 14, marginBottom: '1rem' }}>Glossary</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1.5rem' }}>
                {Object.entries(definitions.glossary || {}).map(([term, def]) => (
                  <div key={term} style={{ background: '#0f172a', borderRadius: 8, padding: '0.7rem 1rem' }}>
                    <div style={{ fontWeight: 700, color: '#c084fc', fontSize: 12, marginBottom: 2 }}>{term}</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{def}</div>
                  </div>
                ))}
              </div>

              <h3 style={{ color: '#7c3aed', fontSize: 14, marginBottom: '1rem' }}>Clinical Pearls</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {definitions.clinical_pearls?.map((pearl, i) => (
                  <div key={i} style={{
                    background: '#0f172a', borderRadius: 8, padding: '0.7rem 1rem',
                    borderLeft: '3px solid #16a34a', fontSize: 11, color: '#bbf7d0', lineHeight: 1.5,
                  }}>
                    {pearl}
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
