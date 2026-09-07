'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  NF1:   '#dc2626',  // red    — NF1/Von Recklinghausen; RAS-GAP; selumetinib
  NF2:   '#2563eb',  // blue   — NF2; bilateral VS PATHOGNOMONIC; bevacizumab
  TSC1:  '#7c3aed',  // purple — TSC1; cardiac rhabdomyoma PATHOGNOMONIC; everolimus
  TSC2:  '#ea580c',  // orange — TSC2; MORE SEVERE; LAM; sirolimus MILES
  VHL:   '#0f766e',  // teal   — VHL; clear cell RCC; belzutifan HIF-2α FDA2021
  PTCH1: '#ca8a04',  // amber  — Gorlin; calcified falx; RT ABSOLUTELY CI; vismodegib
  STK11: '#16a34a',  // green  — PJS; lentigines PATHOGNOMONIC; breast 45%
  PTEN:  '#be185d',  // pink   — PHTS/Cowden; macrocephaly PATHOGNOMONIC; breast 85%
};

const GENE_DISEASE = {
  NF1:   'AD NF1 Von-Recklinghausen — NF1/Neurofibromin1-2839aa — 17q11.2 — RAS-GAP-LOF-RAS-Constitutive — Selumetinib-MEK1/2-FDA2020-FIRST-NF1-Therapy — MPNST-8-13pct-Plexiform — Lisch-Nodules-Adults-PATHOGNOMONIC',
  NF2:   'AD NF2 Bilateral-VS-Syndrome — NF2/Merlin-595aa — 22q12.2 — FERM-Tumor-Suppressor — Bilateral-Vestibular-Schwannomas-PATHOGNOMONIC-95pct — Bevacizumab-VEGF-FIRST-Systemic — Meningiomas-50-75pct',
  TSC1:  'AD TSC1 Hamartin — TSC1/Hamartin-1164aa — 9q34.13 — mTORC1-Inhibitor-Rheb-GAP — Cardiac-Rhabdomyoma-PATHOGNOMONIC-Neonate — Everolimus-SEGA-AML-LAM — Vigabatrin-IS-FIRST-LINE',
  TSC2:  'AD TSC2 Tuberin MORE-SEVERE — TSC2/Tuberin-1807aa — 16p13.3 — GAP-Catalytic-Subunit — LAM-Predominantly-TSC2-MILES-Sirolimus — TSC2/PKD1-Contiguous — AML-GT4cm-Embolization',
  VHL:   'AD VHL Von-Hippel-Lindau — VHL/pVHL-213aa — 3p25.3 — E3-Ubiquitin-Ligase-HIF1a-HIF2a — Clear-Cell-RCC-PATHOGNOMONIC — Belzutifan-HIF-2a-FDA2021-FIRST-VHL-Targeted — Alpha-Before-Beta-Pheo',
  PTCH1: 'AD Gorlin NBCCS — PTCH1/Patched1-1447aa — 9q22.32 — Hedgehog-SMO-Inhibitor — Multiple-BCCs-LT20yr-PATHOGNOMONIC — Calcified-Falx-XR-PATHOGNOMONIC — RADIATION-ABSOLUTELY-CI — Vismodegib-SMO-FDA',
  STK11: 'AD Peutz-Jeghers — STK11/LKB1-433aa — 19p13.3 — AMPK-Activation-Energy-Sensor — Mucocutaneous-Lentigines-Lips-Buccal-PATHOGNOMONIC — GI-Hamartomas-Intussusception — Breast-45pct-Pancreatic-11-36pct',
  PTEN:  'AD PHTS Cowden — PTEN-403aa — 10q23.31 — PI3K-AKT-mTOR-PIP3-Phosphatase — Macrocephaly-HC-GT97th-PATHOGNOMONIC — Trichilemmoma-Histology-PATHOGNOMONIC — Breast-85pct-SAME-As-BRCA1',
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

export default function HereditaryHamartomaSyndromeAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/hereditary-hamartoma-syndrome-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-hamartoma-syndrome-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-hamartoma-syndrome-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const bg = '#0f172a', card = '#1e293b', border = '#334155', text = '#e2e8f0', muted = '#94a3b8';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: text, fontFamily: 'system-ui,sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a1a2e,#16213e,#0f3460,#1e293b)', padding: '2rem 2rem 1.5rem' }}>
        <div style={{ fontSize: 11, color: '#7dd3fc', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 8 }}>
          Hereditary Hamartoma Syndrome Atlas — Phakomatoses & Tumor Predisposition
        </div>
        <h1 style={{ fontSize: 26, fontWeight: 800, margin: 0, color: '#e0f2fe' }}>
          Hereditary Hamartoma Syndrome Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#bae6fd', marginTop: 6 }}>
          Complete 8-Gene Hereditary Hamartoma & Phakomatosis Syndrome Atlas — 320 patients · seeds 1910–1917
        </div>
        <div style={{ fontSize: 11, color: muted, marginTop: 4 }}>
          NF1 (selumetinib FDA2020) · NF2 (bilateral VS) · TSC1 · TSC2 (LAM sirolimus) · VHL (belzutifan FDA2021) · PTCH1/Gorlin (RT-CI) · STK11/PJS · PTEN/Cowden (breast 85%)
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, padding: '1rem 2rem 0', borderBottom: `1px solid ${border}` }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: '8px 8px 0 0',
            background: tab === t ? card : 'transparent',
            border: tab === t ? `1px solid ${border}` : '1px solid transparent',
            borderBottom: tab === t ? `1px solid ${card}` : 'none',
            color: tab === t ? '#e0f2fe' : muted, cursor: 'pointer', fontSize: 13, fontWeight: 600,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '1.5rem 2rem' }}>
        {loading && <Loading />}
        {error && <ErrorBox msg={error} />}

        {/* ── OVERVIEW ── */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
              <KPI label="Total Patients" value={overview.aggregate_stats?.total_patients} color="#6366f1" />
              <KPI label="Genes Covered" value={overview.aggregate_stats?.genes_covered} color="#818cf8" />
              <KPI label="Avg Age Dx (yr)" value={overview.aggregate_stats?.avg_age_at_diagnosis_yr} color="#38bdf8" />
              <KPI label="Severe Cases (%)" value={`${overview.aggregate_stats?.severe_cases_pct}%`} color="#f43f5e" />
              <KPI label="Seed Range" value={overview.aggregate_stats?.seed_range} color="#22d3ee" />
            </div>

            {/* Gene cards */}
            <h3 style={{ color: '#bae6fd', marginBottom: 12 }}>Gene Reference Cards</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(320px,1fr))', gap: 14 }}>
              {overview.gene_summary?.map(g => (
                <div key={g.gene} style={{
                  background: card, borderRadius: 10, padding: '1rem',
                  borderTop: `3px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                    <div>
                      <span style={{ fontSize: 18, fontWeight: 800, color: GENE_COLORS[g.gene] || '#bae6fd' }}>{g.gene}</span>
                      {g.alt_name && g.alt_name !== g.gene && (
                        <span style={{ fontSize: 12, color: muted, marginLeft: 6 }}>/ {g.alt_name}</span>
                      )}
                    </div>
                    <Badge text={g.inheritance} color={GENE_COLORS[g.gene] || '#6366f1'} />
                  </div>
                  <div style={{ fontSize: 11, color: muted, marginBottom: 6 }}>
                    {g.locus} · {g.protein_size} · {g.age_of_onset}
                  </div>
                  <div style={{ fontSize: 11, color: '#e0f2fe', marginBottom: 6, lineHeight: 1.5 }}>
                    <strong style={{ color: GENE_COLORS[g.gene] }}>Pathognomonic:</strong> {g.pathognomonic}
                  </div>
                  <div style={{ fontSize: 11, color: '#cbd5e1', marginBottom: 8, lineHeight: 1.5 }}>
                    <strong>Tx:</strong> {g.treatment?.substring(0, 120)}…
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {g.critical_flags?.slice(0, 3).map((f, i) => (
                      <span key={i} style={{
                        background: '#0c1a2e', borderRadius: 4, padding: '2px 6px',
                        fontSize: 10, color: '#7dd3fc', border: '1px solid #1e3a5f',
                      }}>{f.split(' — ')[0]}</span>
                    ))}
                  </div>
                  <div style={{ marginTop: 8, fontSize: 11, color: muted }}>n = {g.n_patients} patients</div>
                </div>
              ))}
            </div>

            {/* Key clinical distinctions */}
            <div style={{ marginTop: 24, background: card, borderRadius: 10, padding: '1.2rem' }}>
              <h3 style={{ color: '#bae6fd', marginBottom: 12, fontSize: 15 }}>Key Clinical Distinctions</h3>
              {overview.key_clinical_distinctions?.map((d, i) => (
                <div key={i} style={{
                  padding: '8px 12px', marginBottom: 6, background: '#0f172a',
                  borderRadius: 6, borderLeft: '3px solid #0284c7', fontSize: 12, color: '#cbd5e1',
                }}>{d}</div>
              ))}
            </div>
          </div>
        )}

        {/* ── GENE TABLE ── */}
        {tab === 'Gene Table' && breakdown && (
          <div>
            <h3 style={{ color: '#bae6fd', marginBottom: 14 }}>Hamartoma Syndrome Gene Reference Table</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#0c1a2e' }}>
                    {['Gene', 'Alt Name', 'Locus', 'Size', 'Inh', 'Onset', 'Key Biomarker', 'Pathognomonic', 'n'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#7dd3fc', borderBottom: `1px solid ${border}`, whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.genes?.map((g, i) => (
                    <tr key={g.gene} style={{ background: i % 2 === 0 ? card : bg }}>
                      <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || text }}>{g.gene}</td>
                      <td style={{ padding: '8px 10px', color: muted }}>{g.alt_name || '—'}</td>
                      <td style={{ padding: '8px 10px', color: muted }}>{g.locus}</td>
                      <td style={{ padding: '8px 10px', color: muted, whiteSpace: 'nowrap' }}>{g.protein_size}</td>
                      <td style={{ padding: '8px 10px' }}><Badge text={g.inheritance} color={GENE_COLORS[g.gene] || '#6366f1'} /></td>
                      <td style={{ padding: '8px 10px', color: '#e0f2fe', fontSize: 11 }}>{g.age_of_onset}</td>
                      <td style={{ padding: '8px 10px', color: '#cbd5e1', fontSize: 11, maxWidth: 220 }}>{g.key_biomarker?.substring(0, 100)}…</td>
                      <td style={{ padding: '8px 10px', color: '#bae6fd', fontSize: 11, maxWidth: 220 }}>{g.pathognomonic?.substring(0, 100)}…</td>
                      <td style={{ padding: '8px 10px', color: muted }}>{g.n_patients}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── CLINICAL ATLAS ── */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            <h3 style={{ color: '#bae6fd', marginBottom: 14 }}>Hamartoma Syndrome Clinical Atlas — Per-Gene Detail</h3>
            {breakdown.genes?.map(g => (
              <div key={g.gene} style={{
                background: card, borderRadius: 10, padding: '1.2rem', marginBottom: 16,
                borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <div>
                    <span style={{ fontSize: 20, fontWeight: 800, color: GENE_COLORS[g.gene] || '#bae6fd' }}>{g.gene}</span>
                    {g.alt_name && g.alt_name !== g.gene && (
                      <span style={{ fontSize: 14, color: muted, marginLeft: 8 }}>/ {g.alt_name}</span>
                    )}
                    <span style={{ fontSize: 12, color: muted, marginLeft: 12 }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
                  </div>
                  <Badge text={`n=${g.n_patients}`} color={GENE_COLORS[g.gene] || '#6366f1'} />
                </div>
                <div style={{ fontSize: 12, color: muted, marginBottom: 8 }}>
                  <strong style={{ color: '#e0f2fe' }}>Onset:</strong> {g.age_of_onset}
                </div>
                <div style={{ fontSize: 12, color: '#e0f2fe', marginBottom: 8, lineHeight: 1.6 }}>
                  <strong>Disease:</strong> {GENE_DISEASE[g.gene] || ''}
                </div>
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Key Biomarker:</div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.5 }}>{g.key_biomarker}</div>
                </div>
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Pathognomonic:</div>
                  <div style={{ fontSize: 12, color: '#bae6fd', lineHeight: 1.5 }}>{g.pathognomonic}</div>
                </div>
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Treatment:</div>
                  <div style={{ fontSize: 12, color: '#7dd3fc', lineHeight: 1.5 }}>{g.treatment}</div>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>Critical Flags:</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {g.critical_flags?.map((f, i) => (
                      <div key={i} style={{
                        background: '#0f172a', borderRadius: 6, padding: '6px 10px',
                        borderLeft: `2px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                        fontSize: 11, color: '#e0f2fe',
                      }}>{f}</div>
                    ))}
                  </div>
                </div>
                <div style={{ marginTop: 10 }}>
                  <div style={{ fontSize: 11, color: muted, marginBottom: 4 }}>Severity distribution (n={g.n_patients}):</div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    {Object.entries(g.severity_distribution || {}).map(([s, n]) => (
                      <Badge key={s} text={`${s}: ${n}`} color={s === 'severe' ? '#ef4444' : s === 'moderate' ? '#f59e0b' : '#22c55e'} />
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── DEFINITIONS ── */}
        {tab === 'Definitions' && definitions && (
          <div>
            <h3 style={{ color: '#bae6fd', marginBottom: 14 }}>Hamartoma Syndrome Gene Definitions & Glossary</h3>
            {definitions.genes?.map(g => (
              <div key={g.gene} style={{
                background: card, borderRadius: 10, padding: '1.2rem', marginBottom: 14,
                borderTop: `3px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[g.gene] || '#bae6fd' }}>
                    {g.gene}{g.alt_name && g.alt_name !== g.gene ? ` / ${g.alt_name}` : ''}
                  </span>
                  <span style={{ fontSize: 11, color: muted }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
                </div>
                <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                  {g.definition}
                </div>
              </div>
            ))}
            <div style={{ marginTop: 20, background: card, borderRadius: 10, padding: '1.2rem' }}>
              <h4 style={{ color: '#bae6fd', marginBottom: 12 }}>Hamartoma Syndrome Glossary</h4>
              {Object.entries(definitions.glossary || {}).map(([term, def]) => (
                <div key={term} style={{ marginBottom: 10, borderBottom: `1px solid ${border}`, paddingBottom: 10 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#7dd3fc', marginBottom: 3 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#cbd5e1' }}>{def}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
