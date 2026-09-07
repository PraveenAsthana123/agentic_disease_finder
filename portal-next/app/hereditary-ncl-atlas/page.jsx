'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  CLN1:  '#dc2626',  // red    — INCL; PPT1; GRODs; rapid death; no therapy
  CLN2:  '#2563eb',  // blue   — Brineura FDA2017; ICV ERT; curvilinear bodies
  CLN3:  '#7c3aed',  // purple — JNCL Batten; vision FIRST; sea-blue histiocytes
  CLN5:  '#16a34a',  // green  — Finnish variant; TPP1 normal; p.Y392X
  CLN6:  '#ea580c',  // orange — Kufs adult NCL; no vision loss; Sri Lankan
  CLN7:  '#0f766e',  // teal   — MFSD8; Turkish founder; fingerprint EM
  CLN8:  '#ca8a04',  // amber  — EPMR; slowest NCL; Finnish p.Arg24Gly
  CLN10: '#be185d',  // pink   — CTSD; congenital lethal; aspartyl protease
};

const GENE_DISEASE = {
  CLN1:  'AR INCL — CLN1/PPT1-306aa — 1p34.2 — PPT1-Palmitoyl-Protein-Thioesterase — GRODs-Granular-Osmiophilic-Deposits-EM-PATHOGNOMONIC — Onset-6-18m — No-FDA-Therapy — Death-5-10yr',
  CLN2:  'AR Late-Infantile-NCL — CLN2/TPP1-563aa — 11p15.4 — TPP1-Tripeptidyl-Peptidase-1 — Cerliponase-Alfa-Brineura-ICV-FDA2017-ONLY-NCL-ERT — Curvilinear-Bodies-EM — Giant-VEPs-1-2Hz-PATHOGNOMONIC',
  CLN3:  'AR JNCL-Batten — CLN3-438aa — 16p12.1 — Battenin-Lysosomal-Membrane — Vision-Loss-FIRST-5yr-Before-Seizures-PATHOGNOMONIC — Sea-Blue-Histiocytes-BM-PATHOGNOMONIC — 1-kb-Deletion-73pct',
  CLN5:  'AR Finnish-Variant-LI-NCL — CLN5-407aa — 13q22.3 — ER-Golgi-Protein-Not-Enzyme — TPP1-Normal-KEY-DISTINCTION — p.Y392X-Finnish-Founder — Mixed-Fingerprint-Rectilinear-EM',
  CLN6:  'AR Variant-LI-NCL / Kufs-A — CLN6/Linclin-311aa — 15q23 — ER-Membrane-Protein — Kufs-Adult-NCL-NO-Vision-Loss — Sri-Lankan-Romani-Costa-Rican-Founders — Gene-Therapy-Trial-Active',
  CLN7:  'AR Late-Infantile-Variant — CLN7/MFSD8-518aa — 4q28.2 — Lysosomal-MFS-Transporter — Turkish-Founder-c.103C>T-p.Arg35Trp — Fingerprint-EM-Predominant — Normal-TPP1-PPT1-Enzymes',
  CLN8:  'AR EPMR-Northern-Epilepsy — CLN8-286aa — 8p23.3 — ER-Membrane-Protein — Finnish-Founder-p.Arg24Gly — SLOWEST-NCL-Survival-35-50yr — PME-Cognitive-Decline-Vision-Preserved-EPMR',
  CLN10: 'AR Congenital-NCL — CLN10/CTSD-412aa — 11p15.5 — Cathepsin-D-Aspartyl-Protease — Null-Congenital-Lethal-Seizures-Microcephaly-Death-Days-Weeks — p.Y199C-Juvenile-Slower — ONLY-NCL-Aspartyl-Protease',
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

export default function HereditaryNCLAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/hereditary-ncl-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-ncl-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-ncl-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const bg = '#0f172a', card = '#1e293b', border = '#334155', text = '#e2e8f0', muted = '#94a3b8';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: text, fontFamily: 'system-ui,sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1e1b4b,#312e81,#1e293b)', padding: '2rem 2rem 1.5rem' }}>
        <div style={{ fontSize: 11, color: '#818cf8', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 8 }}>
          Hereditary NCL Atlas — Neuronal Ceroid Lipofuscinosis
        </div>
        <h1 style={{ fontSize: 26, fontWeight: 800, margin: 0, color: '#e0e7ff' }}>
          Hereditary NCL (Batten Disease) Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#a5b4fc', marginTop: 6 }}>
          Complete 8-Gene Hereditary Neuronal Ceroid Lipofuscinosis Atlas — 320 patients · seeds 1902–1909
        </div>
        <div style={{ fontSize: 11, color: muted, marginTop: 4 }}>
          CLN1/PPT1 · CLN2/TPP1 (Brineura FDA2017) · CLN3 (JNCL) · CLN5 · CLN6 · CLN7/MFSD8 · CLN8 (EPMR) · CLN10/CTSD
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
            color: tab === t ? '#e0e7ff' : muted, cursor: 'pointer', fontSize: 13, fontWeight: 600,
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
              <KPI label="Avg Age Dx (yr)" value={overview.aggregate_stats?.avg_age_at_diagnosis_yr} color="#a5b4fc" />
              <KPI label="Severe Cases (%)" value={`${overview.aggregate_stats?.severe_cases_pct}%`} color="#f43f5e" />
              <KPI label="Seed Range" value={overview.aggregate_stats?.seed_range} color="#22d3ee" />
            </div>

            {/* Gene cards */}
            <h3 style={{ color: '#c7d2fe', marginBottom: 12 }}>Gene Reference Cards</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(320px,1fr))', gap: 14 }}>
              {overview.gene_summary?.map(g => (
                <div key={g.gene} style={{
                  background: card, borderRadius: 10, padding: '1rem',
                  borderTop: `3px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                    <div>
                      <span style={{ fontSize: 18, fontWeight: 800, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
                      {g.alt_name && g.alt_name !== g.gene && (
                        <span style={{ fontSize: 12, color: muted, marginLeft: 6 }}>/ {g.alt_name}</span>
                      )}
                    </div>
                    <Badge text={g.inheritance} color={GENE_COLORS[g.gene] || '#6366f1'} />
                  </div>
                  <div style={{ fontSize: 11, color: muted, marginBottom: 6 }}>
                    {g.locus} · {g.protein_size} · {g.age_of_onset}
                  </div>
                  <div style={{ fontSize: 11, color: '#e0e7ff', marginBottom: 6, lineHeight: 1.5 }}>
                    <strong style={{ color: GENE_COLORS[g.gene] }}>Pathognomonic:</strong> {g.pathognomonic}
                  </div>
                  <div style={{ fontSize: 11, color: '#cbd5e1', marginBottom: 8, lineHeight: 1.5 }}>
                    <strong>Tx:</strong> {g.treatment?.substring(0, 120)}…
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {g.critical_flags?.slice(0, 3).map((f, i) => (
                      <span key={i} style={{
                        background: '#1e1b4b', borderRadius: 4, padding: '2px 6px',
                        fontSize: 10, color: '#a5b4fc', border: '1px solid #312e81',
                      }}>{f.split(' — ')[0]}</span>
                    ))}
                  </div>
                  <div style={{ marginTop: 8, fontSize: 11, color: muted }}>n = {g.n_patients} patients</div>
                </div>
              ))}
            </div>

            {/* Key clinical distinctions */}
            <div style={{ marginTop: 24, background: card, borderRadius: 10, padding: '1.2rem' }}>
              <h3 style={{ color: '#c7d2fe', marginBottom: 12, fontSize: 15 }}>Key Clinical Distinctions</h3>
              {overview.key_clinical_distinctions?.map((d, i) => (
                <div key={i} style={{
                  padding: '8px 12px', marginBottom: 6, background: '#0f172a',
                  borderRadius: 6, borderLeft: '3px solid #4f46e5', fontSize: 12, color: '#cbd5e1',
                }}>{d}</div>
              ))}
            </div>
          </div>
        )}

        {/* ── GENE TABLE ── */}
        {tab === 'Gene Table' && breakdown && (
          <div>
            <h3 style={{ color: '#c7d2fe', marginBottom: 14 }}>NCL Gene Reference Table</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1e1b4b' }}>
                    {['Gene', 'Alt Name', 'Locus', 'Size', 'Inh', 'Onset', 'Key Biomarker', 'Pathognomonic', 'n'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#a5b4fc', borderBottom: `1px solid ${border}`, whiteSpace: 'nowrap' }}>{h}</th>
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
                      <td style={{ padding: '8px 10px', color: '#e0e7ff', fontSize: 11 }}>{g.age_of_onset}</td>
                      <td style={{ padding: '8px 10px', color: '#cbd5e1', fontSize: 11, maxWidth: 220 }}>{g.key_biomarker?.substring(0, 100)}…</td>
                      <td style={{ padding: '8px 10px', color: '#c7d2fe', fontSize: 11, maxWidth: 220 }}>{g.pathognomonic?.substring(0, 100)}…</td>
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
            <h3 style={{ color: '#c7d2fe', marginBottom: 14 }}>NCL Clinical Atlas — Per-Gene Detail</h3>
            {breakdown.genes?.map(g => (
              <div key={g.gene} style={{
                background: card, borderRadius: 10, padding: '1.2rem', marginBottom: 16,
                borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <div>
                    <span style={{ fontSize: 20, fontWeight: 800, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
                    {g.alt_name && g.alt_name !== g.gene && (
                      <span style={{ fontSize: 14, color: muted, marginLeft: 8 }}>/ {g.alt_name}</span>
                    )}
                    <span style={{ fontSize: 12, color: muted, marginLeft: 12 }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
                  </div>
                  <Badge text={`n=${g.n_patients}`} color={GENE_COLORS[g.gene] || '#6366f1'} />
                </div>
                <div style={{ fontSize: 12, color: muted, marginBottom: 8 }}>
                  <strong style={{ color: '#e0e7ff' }}>Onset:</strong> {g.age_of_onset}
                </div>
                <div style={{ fontSize: 12, color: '#e0e7ff', marginBottom: 8, lineHeight: 1.6 }}>
                  <strong>Disease:</strong> {GENE_DISEASE[g.gene] || ''}
                </div>
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Key Biomarker:</div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.5 }}>{g.key_biomarker}</div>
                </div>
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Pathognomonic:</div>
                  <div style={{ fontSize: 12, color: '#c7d2fe', lineHeight: 1.5 }}>{g.pathognomonic}</div>
                </div>
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Treatment:</div>
                  <div style={{ fontSize: 12, color: '#a5b4fc', lineHeight: 1.5 }}>{g.treatment}</div>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>Critical Flags:</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {g.critical_flags?.map((f, i) => (
                      <div key={i} style={{
                        background: '#0f172a', borderRadius: 6, padding: '6px 10px',
                        borderLeft: `2px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                        fontSize: 11, color: '#e0e7ff',
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
            <h3 style={{ color: '#c7d2fe', marginBottom: 14 }}>NCL Gene Definitions & Glossary</h3>
            {definitions.genes?.map(g => (
              <div key={g.gene} style={{
                background: card, borderRadius: 10, padding: '1.2rem', marginBottom: 14,
                borderTop: `3px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>
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
              <h4 style={{ color: '#c7d2fe', marginBottom: 12 }}>NCL Glossary</h4>
              {Object.entries(definitions.glossary || {}).map(([term, def]) => (
                <div key={term} style={{ marginBottom: 10, borderBottom: `1px solid ${border}`, paddingBottom: 10 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#a5b4fc', marginBottom: 3 }}>{term}</div>
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
