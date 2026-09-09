'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  SLC12A1: '#dc2626',  // red    — Bartter type 1; NKCC2; neonatal severe
  KCNJ1:   '#2563eb',  // blue   — Bartter type 2; ROMK; transient hyperK
  CLCNKB:  '#7c3aed',  // purple — Bartter type 3; most common; complete deletion
  BSND:    '#ea580c',  // orange — Bartter type 4; barttin; SNHL
  SLC12A3: '#16a34a',  // green  — Gitelman; hypoMg; low urine Ca
  CLCN5:   '#0f766e',  // teal   — Dent 1; ClC-5; LMW proteinuria; XLR
  OCRL:    '#ca8a04',  // amber  — Lowe syndrome; cataracts + ID + Fanconi; XLR
  AGXT:    '#be185d',  // pink   — PH1; systemic oxalosis; lumasiran; OLT
};

const GENE_DISEASE = {
  SLC12A1: 'AR Bartter-Type-1 — SLC12A1/NKCC2-1099aa — 15q21.1 — Na-K-2Cl-Cotransporter-2-TAL-Apical — Neonatal-Severe-Polyhydramnios-Salt-Wasting — Hypercalciuria-Nephrocalcinosis — Indomethacin-KCl-NaCl',
  KCNJ1:   'AR Bartter-Type-2 — KCNJ1/ROMK-391aa — 11q24.3 — ROMK-TAL-K-Recycling-Plus-CCD-K-Secretion — Transient-Neonatal-HyperK-PATHOGNOMONIC-Then-HypoK — Biphasic-K-Course — Amiloride-Preferred',
  CLCNKB:  'AR Bartter-Type-3-MOST-COMMON — CLCNKB/ClC-Kb-687aa — 1p36.13 — Basolateral-Cl-Channel-TAL-DCT — Complete-Deletion-MLPA-Mandatory — Classic-Childhood-Milder — Phenotypic-Overlap-Gitelman',
  BSND:    'AR Bartter-Type-4-Plus-SNHL-PATHOGNOMONIC — BSND/Barttin-320aa — 1p32.3 — Beta-Subunit-ClC-Ka-AND-ClC-Kb — Stria-Vascularis-Marginal-Cells — Cochlear-Implant-Early — ABR-Neonatal-Mandatory',
  SLC12A3: 'AR Gitelman-1:40000 — SLC12A3/NCCT-1021aa — 16q13 — Na-Cl-Cotransporter-DCT-Thiazide-Target — HypoMg-Low-Urine-Ca-PATHOGNOMONIC — Chondrocalcinosis-QTc-Long-Term — Mg-Supplementation-Lifelong',
  CLCN5:   'XLR Dent-Disease-1 — CLCN5/ClC-5-746aa — Xp11.23 — Endosomal-Cl-H-Exchanger-Proximal-Tubule — LMW-Proteinuria-Beta-2-MG-PATHOGNOMONIC-Dipstick-Negative — Nephrocalcinosis-Nephrolithiasis — Thiazide-Stones',
  OCRL:    'XLR Lowe-Syndrome — OCRL/OCRL1-901aa — Xq26.1 — PI45P2-5-Phosphatase-Endosomal-Golgi — Cataracts-ID-Fanconi-RTA-TRIAD-PATHOGNOMONIC — Congenital-Cataracts-Lens-Extraction-Weeks — Dent-2-Allelic-Milder',
  AGXT:    'AR Primary-Hyperoxaluria-Type-1 — AGXT/PH1-392aa — 2q37.3 — Alanine-Glyoxylate-Aminotransferase-Peroxisomal — Systemic-Oxalosis-Bones-Heart-Retina — Lumasiran-FDA2020 — Combined-LKT-CURATIVE',
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

export default function HereditaryRenalTubularAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/hereditary-renal-tubular-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-renal-tubular-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-renal-tubular-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const bg = '#0f172a', card = '#1e293b', border = '#334155', text = '#e2e8f0', muted = '#94a3b8';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: text, fontFamily: 'system-ui,sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#0c4a6e,#075985,#1e293b)', padding: '2rem 2rem 1.5rem' }}>
        <div style={{ fontSize: 11, color: '#38bdf8', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 8 }}>
          Hereditary Renal Tubular Atlas — Bartter · Gitelman · Dent · Lowe · PH1
        </div>
        <h1 style={{ fontSize: 26, fontWeight: 800, margin: 0, color: '#e0f2fe' }}>
          Hereditary Renal Tubular Disorders Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#7dd3fc', marginTop: 6 }}>
          Complete 8-Gene Hereditary Renal Tubular Disorders Atlas — 320 patients · seeds 1910–1917
        </div>
        <div style={{ fontSize: 11, color: muted, marginTop: 4 }}>
          SLC12A1/NKCC2 (Bartter-1) · KCNJ1/ROMK (Bartter-2) · CLCNKB/ClC-Kb (Bartter-3) ·
          BSND/Barttin (Bartter-4 + SNHL) · SLC12A3/NCCT (Gitelman) · CLCN5/ClC-5 (Dent-1) ·
          OCRL/OCRL1 (Lowe) · AGXT (PH1 · lumasiran · OLT)
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, padding: '0 2rem', background: '#0f172a', borderBottom: `1px solid ${border}` }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '10px 18px', background: 'none', border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: 600,
            color: tab === t ? '#38bdf8' : muted,
            borderBottom: tab === t ? '2px solid #38bdf8' : '2px solid transparent',
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '2rem' }}>
        {loading && <Loading />}
        {error && <ErrorBox msg={error} />}

        {/* OVERVIEW */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
              <KPI label="Total patients" value={overview.aggregate_stats?.total_patients} color="#38bdf8" />
              <KPI label="Genes covered" value={overview.aggregate_stats?.genes_covered} color="#a78bfa" />
              <KPI label="Avg age at diagnosis (yr)" value={overview.aggregate_stats?.avg_age_at_diagnosis_yr} color="#34d399" />
              <KPI label="Severe cases %" value={`${overview.aggregate_stats?.severe_cases_pct}%`} color="#f87171" />
              <KPI label="Seeds" value={overview.aggregate_stats?.seed_range} color="#fb923c" />
            </div>

            <div style={{ background: card, borderRadius: 10, padding: '1.2rem', marginBottom: 20, border: `1px solid ${border}` }}>
              <div style={{ fontSize: 11, color: '#38bdf8', fontWeight: 700, marginBottom: 8 }}>ATLAS SUBTITLE</div>
              <div style={{ fontSize: 11, color: muted, lineHeight: 1.8, wordBreak: 'break-word' }}>{overview.subtitle}</div>
            </div>

            <div style={{ background: card, borderRadius: 10, padding: '1.2rem', marginBottom: 20, border: `1px solid ${border}` }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8', marginBottom: 12 }}>
                Key Clinical Distinctions — Renal Tubular Disorders
              </div>
              {(overview.key_clinical_distinctions || []).map((d, i) => (
                <div key={i} style={{
                  padding: '8px 12px', marginBottom: 6, borderRadius: 6,
                  background: '#0f172a', borderLeft: '3px solid #38bdf8',
                  fontSize: 11, color: text, lineHeight: 1.6,
                }}>{d}</div>
              ))}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 16 }}>
              {(overview.gene_summary || []).map(g => (
                <div key={g.gene} style={{
                  background: card, borderRadius: 10, padding: '1.2rem',
                  border: `1px solid ${GENE_COLORS[g.gene] || border}44`,
                  borderTop: `3px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontWeight: 800, fontSize: 16, color: GENE_COLORS[g.gene] || text }}>{g.gene}</span>
                    <Badge text={g.inheritance} color={g.inheritance === 'XLR' ? '#f59e0b' : '#38bdf8'} />
                  </div>
                  <div style={{ fontSize: 11, color: muted, marginBottom: 6 }}>{g.alt_name} · {g.locus} · {g.protein_size}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}><b style={{ color: '#7dd3fc' }}>Onset:</b> {g.age_of_onset}</div>
                  <div style={{ fontSize: 11, color: text, marginBottom: 6, lineHeight: 1.5 }}>
                    <b style={{ color: '#38bdf8' }}>Pathognomonic:</b> {g.pathognomonic?.slice(0, 200)}{g.pathognomonic?.length > 200 ? '…' : ''}
                  </div>
                  <div style={{ fontSize: 11, color: muted }}>n = {g.n_patients} patients</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* GENE TABLE */}
        {tab === 'Gene Table' && (
          <div>
            <div style={{ marginBottom: 16, fontSize: 13, color: muted }}>
              8 genes — Bartter types 1–4 · Gitelman · Dent-1 · Lowe · PH1
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene', 'Alias', 'Locus', 'Size', 'Inh.', 'Onset', 'Disease OMIM'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#38bdf8', borderBottom: `1px solid ${border}`, fontSize: 11 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(GENE_DISEASE).map(([gene, desc], i) => {
                    const g = overview?.gene_summary?.find(x => x.gene === gene) || {};
                    return (
                      <tr key={gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                        <td style={{ padding: '8px 12px', color: GENE_COLORS[gene], fontWeight: 700 }}>{gene}</td>
                        <td style={{ padding: '8px 12px', color: muted }}>{g.alt_name || '—'}</td>
                        <td style={{ padding: '8px 12px', color: text }}>{g.locus || '—'}</td>
                        <td style={{ padding: '8px 12px', color: text }}>{g.protein_size || '—'}</td>
                        <td style={{ padding: '8px 12px' }}>
                          <Badge text={g.inheritance || '—'} color={g.inheritance === 'XLR' ? '#f59e0b' : '#38bdf8'} />
                        </td>
                        <td style={{ padding: '8px 12px', color: muted, fontSize: 11 }}>{g.age_of_onset?.slice(0,40) || '—'}</td>
                        <td style={{ padding: '8px 12px', color: text, fontSize: 10, maxWidth: 320 }}>{desc.slice(0, 120)}…</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* CLINICAL ATLAS */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {breakdown.genes?.map(g => (
              <div key={g.gene} style={{
                background: card, borderRadius: 10, padding: '1.5rem',
                border: `1px solid ${border}`,
                borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
                  <div>
                    <span style={{ fontWeight: 800, fontSize: 18, color: GENE_COLORS[g.gene] || text }}>{g.gene}</span>
                    <span style={{ marginLeft: 10, color: muted, fontSize: 13 }}>{g.alt_name} · {g.locus} · {g.protein_size}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <Badge text={g.inheritance} color={g.inheritance === 'XLR' ? '#f59e0b' : '#38bdf8'} />
                    <Badge text={`n=${g.n_patients}`} color="#64748b" />
                    {Object.entries(g.severity_distribution || {}).map(([sev, cnt]) => (
                      <Badge key={sev} text={`${sev}: ${cnt}`}
                        color={sev === 'severe' ? '#dc2626' : sev === 'moderate' ? '#f59e0b' : '#16a34a'} />
                    ))}
                  </div>
                </div>

                <div style={{ fontSize: 11, color: muted, marginBottom: 8 }}>
                  <b style={{ color: '#7dd3fc' }}>Onset:</b> {g.age_of_onset}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                  <div style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem' }}>
                    <div style={{ fontSize: 11, color: '#38bdf8', fontWeight: 700, marginBottom: 6 }}>KEY BIOMARKER</div>
                    <div style={{ fontSize: 11, color: text, lineHeight: 1.6 }}>{g.key_biomarker}</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem' }}>
                    <div style={{ fontSize: 11, color: '#f472b6', fontWeight: 700, marginBottom: 6 }}>PATHOGNOMONIC</div>
                    <div style={{ fontSize: 11, color: text, lineHeight: 1.6 }}>{g.pathognomonic}</div>
                  </div>
                </div>

                <div style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem', marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: '#34d399', fontWeight: 700, marginBottom: 6 }}>TREATMENT</div>
                  <div style={{ fontSize: 11, color: text, lineHeight: 1.6 }}>{g.treatment}</div>
                </div>

                <div>
                  <div style={{ fontSize: 11, color: '#fb923c', fontWeight: 700, marginBottom: 8 }}>CRITICAL FLAGS</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {(g.critical_flags || []).map((flag, i) => (
                      <div key={i} style={{
                        padding: '6px 10px', background: '#1e293b',
                        borderLeft: '3px solid #fb923c', borderRadius: 4,
                        fontSize: 10, color: text, lineHeight: 1.5,
                      }}>{flag}</div>
                    ))}
                  </div>
                </div>

                {g.patients && g.patients.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ fontSize: 11, color: muted, marginBottom: 6 }}>Sample patients (first 5 of {g.n_patients}):</div>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                        <thead>
                          <tr>
                            {['ID', 'Age (yr)', 'Sex', 'Severity', 'Locus'].map(h => (
                              <th key={h} style={{ padding: '4px 8px', color: muted, textAlign: 'left', borderBottom: `1px solid ${border}` }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {g.patients.map(p => (
                            <tr key={p.patient_id}>
                              <td style={{ padding: '4px 8px', color: GENE_COLORS[g.gene] }}>{p.patient_id}</td>
                              <td style={{ padding: '4px 8px', color: text }}>{p.age_at_diagnosis_yr}</td>
                              <td style={{ padding: '4px 8px', color: muted }}>{p.sex}</td>
                              <td style={{ padding: '4px 8px' }}>
                                <Badge text={p.severity}
                                  color={p.severity === 'severe' ? '#dc2626' : p.severity === 'moderate' ? '#f59e0b' : '#16a34a'} />
                              </td>
                              <td style={{ padding: '4px 8px', color: muted }}>{p.locus}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS */}
        {tab === 'Definitions' && definitions && (
          <div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginBottom: 32 }}>
              {definitions.genes?.map(g => (
                <div key={g.gene} style={{
                  background: card, borderRadius: 10, padding: '1.2rem',
                  border: `1px solid ${border}`,
                  borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                }}>
                  <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8, flexWrap: 'wrap' }}>
                    <span style={{ fontWeight: 800, fontSize: 15, color: GENE_COLORS[g.gene] || text }}>{g.gene}</span>
                    <span style={{ color: muted, fontSize: 12 }}>{g.alt_name}</span>
                    <Badge text={g.locus} color="#64748b" />
                    <Badge text={g.protein_size} color="#64748b" />
                    <Badge text={g.inheritance} color={g.inheritance === 'XLR' ? '#f59e0b' : '#38bdf8'} />
                  </div>
                  <div style={{ fontSize: 11, color: muted, marginBottom: 8 }}>{g.age_of_onset}</div>
                  <div style={{ fontSize: 11, color: text, lineHeight: 1.7, marginBottom: 12 }}>{g.definition}</div>
                  {g.critical_flags?.length > 0 && (
                    <div>
                      <div style={{ fontSize: 10, color: '#fb923c', fontWeight: 700, marginBottom: 6 }}>CRITICAL FLAGS</div>
                      {g.critical_flags.map((f, i) => (
                        <div key={i} style={{
                          fontSize: 10, color: text, padding: '4px 8px',
                          borderLeft: '2px solid #fb923c', marginBottom: 3, lineHeight: 1.5,
                        }}>{f}</div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div style={{ background: card, borderRadius: 10, padding: '1.2rem', border: `1px solid ${border}` }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8', marginBottom: 12 }}>Glossary</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 12 }}>
                {Object.entries(definitions.glossary || {}).map(([term, def]) => (
                  <div key={term} style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem' }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: '#7dd3fc', marginBottom: 4 }}>{term}</div>
                    <div style={{ fontSize: 11, color: muted, lineHeight: 1.6 }}>{def}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
