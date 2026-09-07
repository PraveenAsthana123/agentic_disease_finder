'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ARSA:   '#dc2626',  // red    — MLD; sulfatide accumulation; Libmeldy EMA2020; pseudodeficiency pitfall
  GALC:   '#2563eb',  // blue   — Krabbe; psychosine toxic; HSCT pre-symptomatic only; NBS mandatory
  PLP1:   '#7c3aed',  // purple — PMD; duplication 70%; nystagmus at birth; MLPA mandatory; X-linked
  ABCD1:  '#ea580c',  // orange — X-ALD; VLCFA; Skysona FDA2022; adrenal crisis; PHT absolute CI
  ASPA:   '#0f766e',  // teal   — Canavan; NAA elevated MRS; U-fibres; Ashkenazi founder
  GFAP:   '#ca8a04',  // amber  — Alexander; GOF not LOF; Rosenthal fibres; frontal WM; de novo
  EIF2B5: '#be185d',  // pink   — VWM; stress triggered; ISR hypersensitivity; ISRIB; ovarioleukodystrophy
  POLR3A: '#16a34a',  // green  — HLD7; dental abnormalities; hypomyelination; intronic splice; slow
};

const GENE_DISEASE = {
  ARSA:   'AR MLD Metachromatic-Leukodystrophy — ARSA-507aa — 22q13.33 — Sulfatide-Accumulation — Libmeldy-EMA2020-First-Gene-Therapy — Tigroid-MRI — Pseudodeficiency-N350S-I179S-Pitfall',
  GALC:   'AR Krabbe Globoid-Cell-Leukodystrophy — GALC-669aa — 14q31.3 — Psychosine-Toxic-Nanomolar — HSCT-Pre-Symptomatic-ONLY — NBS-Mandatory — Globoid-Cells-PATHOGNOMONIC',
  PLP1:   'X-linked PMD Pelizaeus-Merzbacher — PLP1-276aa — Xq22.2 — Duplication-70pct-Most-Common — MLPA-Mandatory — Nystagmus-Birth-EARLIEST — Diffuse-Hypomyelination',
  ABCD1:  'X-linked X-ALD Adrenoleukodystrophy — ABCD1-745aa — Xq28 — VLCFA-C26-Elevated — CCALD-Narrow-Window-Loes9 — Skysona-FDA2022 — Adrenal-Insufficiency-71pct — PHT-ABSOLUTE-CI',
  ASPA:   'AR Canavan — ASPA-313aa — 17p13.2 — NAA-Elevated-MOST-SPECIFIC-MRS — U-Fibres-EARLY — Macrocephaly-Birth — Ashkenazi-Founder-p.Glu285Ala-p.Tyr231X',
  GFAP:   'AD Alexander-Disease — GFAP-432aa — 17q21.31 — ALL-GOF-NOT-LOF-CRITICAL — Rosenthal-Fibres-PATHOGNOMONIC — Frontal-WM-Dominant — CSF-GFAP-Elevated — De-Novo-Most',
  EIF2B5: 'AR VWM Vanishing-White-Matter — EIF2B5-712aa — 3q27.1 — ISR-Hypersensitivity — Stress-Triggered-EMERGENCY — FLAIR-WM-CSF-Signal — ISRIB-Trials — Ovarioleukodystrophy',
  POLR3A: 'AR HLD7 POLR3-Related — POLR3A-1390aa — 10q22.3 — Dental-Abnormalities-PATHOGNOMONIC-Clue — Hypomyelination-NOT-Demyelination — Intronic-Splice-c.1909+22G>A-Genome-Seq-Required',
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

export default function HereditaryLeukodystrophyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/hereditary-leukodystrophy-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-leukodystrophy-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-leukodystrophy-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const stats = overview?.aggregate_stats;

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1e1b4b 0%,#0f172a 100%)', padding: '2rem', borderBottom: '1px solid #334155' }}>
        <div style={{ fontSize: 11, color: '#818cf8', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 6 }}>
          🧬 Hereditary Leukodystrophy Atlas
        </div>
        <h1 style={{ margin: 0, fontSize: 'clamp(1.1rem,2.5vw,1.6rem)', fontWeight: 800, color: '#e0e7ff', lineHeight: 1.3 }}>
          Hereditary-Leukodystrophy-Atlas — Complete 8-Gene Hereditary White Matter Disorders Reference
        </h1>
        <p style={{ margin: '0.5rem 0 0', color: '#94a3b8', fontSize: 13 }}>
          ARSA · GALC · PLP1 · ABCD1 · ASPA · GFAP · EIF2B5 · POLR3A &nbsp;|&nbsp; 320-Patient Aggregate · Seeds 1934–1941
        </p>
        <div style={{ marginTop: '0.75rem', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Badge text="MLD" color="#dc2626" />
          <Badge text="Krabbe" color="#2563eb" />
          <Badge text="PMD" color="#7c3aed" />
          <Badge text="X-ALD" color="#ea580c" />
          <Badge text="Canavan" color="#0f766e" />
          <Badge text="Alexander" color="#ca8a04" />
          <Badge text="VWM" color="#be185d" />
          <Badge text="POLR3-Related" color="#16a34a" />
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, padding: '1rem 2rem 0', borderBottom: '1px solid #1e293b' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '0.5rem 1rem', borderRadius: '6px 6px 0 0', border: 'none', cursor: 'pointer',
            background: tab === t ? '#1e293b' : 'transparent',
            color: tab === t ? '#818cf8' : '#64748b', fontWeight: tab === t ? 700 : 400,
            fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '2rem' }}>
        {loading && <Loading />}
        {error && <ErrorBox msg={error} />}

        {/* ── Overview ── */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: '2rem' }}>
              <KPI label="Total Patients" value={stats?.total_patients} color="#818cf8" />
              <KPI label="Genes Covered" value={stats?.genes_covered} color="#34d399" />
              <KPI label="Avg Age at Dx (yr)" value={stats?.avg_age_at_diagnosis_yr} color="#f59e0b" />
              <KPI label="Severe Cases %" value={stats?.severe_cases_pct + '%'} color="#f87171" />
              <KPI label="Seed Range" value={stats?.seed_range} color="#a78bfa" />
            </div>

            {/* Gene quick-reference */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(260px,1fr))', gap: 12, marginBottom: '2rem' }}>
              {overview.gene_summary?.map(g => (
                <div key={g.gene} style={{
                  background: '#1e293b', borderRadius: 10, padding: '1rem',
                  borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', fontSize: 16 }}>{g.gene}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{g.alt_name}</div>
                  <div style={{ marginTop: 6, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <Badge text={g.locus} color="#6366f1" />
                    <Badge text={g.protein_size} color="#0ea5e9" />
                    <Badge text={g.inheritance} color="#10b981" />
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 6 }}>{g.n_patients} patients</div>
                </div>
              ))}
            </div>

            {/* Key distinctions */}
            <div style={{ background: '#1e293b', borderRadius: 12, padding: '1.5rem' }}>
              <div style={{ fontWeight: 700, color: '#f59e0b', marginBottom: '1rem', fontSize: 14 }}>
                ⚡ Critical Clinical Distinctions
              </div>
              {overview.key_clinical_distinctions?.map((d, i) => {
                const [key, ...rest] = d.split(':');
                return (
                  <div key={i} style={{ marginBottom: 10, display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                    <span style={{ background: '#f59e0b22', color: '#f59e0b', borderRadius: 4, padding: '2px 7px', fontSize: 10, fontWeight: 700, flexShrink: 0, marginTop: 2 }}>
                      {key}
                    </span>
                    <span style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>: {rest.join(':')}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Gene Table ── */}
        {tab === 'Gene Table' && breakdown && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
                  {['Gene', 'Disease', 'Locus', 'Size', 'Inh.', 'Age of Onset', 'Key Biomarker', 'Patients'].map(h => (
                    <th key={h} style={{ padding: '0.7rem 0.8rem', textAlign: 'left', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.genes?.map((g, i) => (
                  <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b', background: i % 2 ? '#0f1729' : 'transparent' }}>
                    <td style={{ padding: '0.7rem 0.8rem', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', whiteSpace: 'nowrap' }}>{g.gene}</td>
                    <td style={{ padding: '0.7rem 0.8rem', color: '#cbd5e1', fontSize: 11, maxWidth: 200 }}>{g.alt_name}</td>
                    <td style={{ padding: '0.7rem 0.8rem', color: '#818cf8', whiteSpace: 'nowrap' }}>{g.locus}</td>
                    <td style={{ padding: '0.7rem 0.8rem', color: '#64748b', whiteSpace: 'nowrap' }}>{g.protein_size}</td>
                    <td style={{ padding: '0.7rem 0.8rem' }}>
                      <Badge text={g.inheritance} color="#10b981" />
                    </td>
                    <td style={{ padding: '0.7rem 0.8rem', color: '#94a3b8', fontSize: 11, maxWidth: 200 }}>{g.age_of_onset?.slice(0, 80)}…</td>
                    <td style={{ padding: '0.7rem 0.8rem', color: '#94a3b8', fontSize: 11, maxWidth: 220 }}>{g.key_biomarker?.slice(0, 90)}…</td>
                    <td style={{ padding: '0.7rem 0.8rem', color: '#64748b', textAlign: 'center' }}>{g.n_patients}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* ── Clinical Atlas ── */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div style={{ display: 'grid', gap: 20 }}>
            {breakdown.genes?.map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 12, padding: '1.5rem', borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#6366f1'}` }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap', marginBottom: '0.75rem' }}>
                  <span style={{ fontSize: 20, fontWeight: 800, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
                  <span style={{ color: '#94a3b8', fontSize: 13 }}>{g.alt_name}</span>
                  <Badge text={g.locus} color="#6366f1" />
                  <Badge text={g.protein_size} color="#0ea5e9" />
                  <Badge text={g.inheritance} color="#10b981" />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: 12, marginBottom: '1rem' }}>
                  {[
                    { label: 'Age of Onset', value: g.age_of_onset, color: '#f59e0b' },
                    { label: 'Key Biomarker', value: g.key_biomarker, color: '#0ea5e9' },
                    { label: 'Pathognomonic', value: g.pathognomonic, color: '#a78bfa' },
                    { label: 'Treatment', value: g.treatment, color: '#34d399' },
                  ].map(item => (
                    <div key={item.label} style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem' }}>
                      <div style={{ fontSize: 10, color: item.color, fontWeight: 700, letterSpacing: 1, marginBottom: 4, textTransform: 'uppercase' }}>
                        {item.label}
                      </div>
                      <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{item.value}</div>
                    </div>
                  ))}
                </div>

                {/* Critical flags */}
                <div style={{ borderTop: '1px solid #334155', paddingTop: '0.75rem' }}>
                  <div style={{ fontSize: 10, color: '#f87171', fontWeight: 700, letterSpacing: 1, marginBottom: 6, textTransform: 'uppercase' }}>
                    Critical Flags
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {g.critical_flags?.map((flag, fi) => {
                      const [code, ...desc] = flag.split(':');
                      return (
                        <div key={fi} style={{ display: 'flex', gap: 8, alignItems: 'flex-start', fontSize: 11 }}>
                          <span style={{ background: '#f8717122', color: '#f87171', borderRadius: 4, padding: '1px 6px', fontSize: 10, fontWeight: 700, whiteSpace: 'nowrap', flexShrink: 0 }}>
                            {code}
                          </span>
                          <span style={{ color: '#94a3b8', lineHeight: 1.5 }}>: {desc.join(':')}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Sample patients */}
                {g.patients?.length > 0 && (
                  <div style={{ marginTop: '0.75rem', borderTop: '1px solid #334155', paddingTop: '0.75rem' }}>
                    <div style={{ fontSize: 10, color: '#64748b', fontWeight: 600, marginBottom: 6 }}>SAMPLE PATIENTS (5 of {g.n_patients})</div>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {g.patients.map(p => (
                        <div key={p.patient_id} style={{ background: '#0f172a', borderRadius: 6, padding: '0.4rem 0.7rem', fontSize: 11 }}>
                          <span style={{ color: '#818cf8' }}>{p.patient_id}</span>
                          <span style={{ color: '#64748b' }}> · age {p.age} · {p.severity} · {p.key_feature}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* ── Definitions ── */}
        {tab === 'Definitions' && definitions && (
          <div>
            <div style={{ display: 'grid', gap: 16, marginBottom: '2rem' }}>
              {definitions.genes?.map(g => (
                <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, padding: '1.2rem', borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}` }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', fontSize: 15, marginBottom: 4 }}>
                    {g.gene} — {g.alt_name}
                  </div>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                    <Badge text={g.locus} color="#6366f1" />
                    <Badge text={g.protein_size} color="#0ea5e9" />
                    <Badge text={g.inheritance} color="#10b981" />
                  </div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7 }}>{g.definition}</div>
                </div>
              ))}
            </div>

            {/* Glossary */}
            <div style={{ background: '#1e293b', borderRadius: 12, padding: '1.5rem' }}>
              <div style={{ fontWeight: 700, color: '#818cf8', marginBottom: '1rem', fontSize: 14 }}>📖 Clinical Glossary</div>
              {Object.entries(definitions.glossary || {}).map(([term, def]) => (
                <div key={term} style={{ marginBottom: 14, borderBottom: '1px solid #334155', paddingBottom: 14 }}>
                  <div style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 12, marginBottom: 4 }}>{term}</div>
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
