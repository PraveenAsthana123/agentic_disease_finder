'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  CFH:  '#dc2626',  // red    — Factor H; most common aHUS gene 25-30%; high transplant recurrence
  CFI:  '#2563eb',  // blue   — Factor I; C3 very low; secondary immunodeficiency if homozygous
  C3:   '#7c3aed',  // purple — C3 central hub; C3G + aHUS; GOF/LOF both disease-causing
  CFB:  '#ea580c',  // orange — Factor B; GOF mutations; Asp279Gly activating; iptacopan target
  CD46: '#0f766e',  // teal   — MCP; youngest onset; BEST transplant; PE ineffective; ~50% penetrance
  THBD: '#ca8a04',  // amber  — thrombomodulin; complement + coagulation bridge; PAH overlap
  DGKE: '#be185d',  // pink   — DGK-epsilon; complement-INDEPENDENT; eculizumab FAILS; infantile
  C5:   '#16a34a',  // green  — C5; eculizumab direct target; Arg885His = eculizumab resistant
};

const GENE_DISEASE = {
  CFH:  'AD/AR aHUS1 Factor-H — CFH-1231aa — 1q31.3 — Most-Common-25-30pct-Complement-aHUS — C3G-AMD — SCR19-20-Host-Self-Recognition — High-Relapse-Transplant-Eculizumab-Mandatory',
  CFI:  'AD/AR aHUS2 Factor-I — CFI-583aa — 4q25 — 10pct-Complement-aHUS — C3-Very-Low-Secondary-Immunodeficiency-If-Homozygous — Serine-Protease-C3b-C4b-Cleavage — Meningococcal-Prophylaxis',
  C3:   'AD/AR C3G-aHUS Component-C3 — C3-1663aa — 19p13.3 — Central-Hub-All-Pathways — GOF-C3G-aHUS-LOF-C3-Deficiency — DDD-Dense-Deposits-EM-PATHOGNOMONIC — Avacopan-Pegcetacoplan-Pipeline',
  CFB:  'AD aHUS Factor-B — CFB-739aa — 6p21.33 — GOF-Asp279Gly-Activating — C3bBb-Alternative-Convertase — Iptacopan-Oral-Factor-B-Inhibitor-FDA2023-PNH — MHC-Region-MLPA-Required',
  CD46: 'AD aHUS3 MCP — CD46-347aa — 1q32.2 — 15pct-aHUS-Youngest-Onset-Median-4-8yr — BEST-Transplant-NO-Recurrence-Donor-Kidney-Normal-CD46 — PE-INEFFECTIVE-Membrane-Protein — 50pct-Penetrance',
  THBD: 'AD/AR aHUS Thrombomodulin — THBD-557aa — 20p11.21 — Rare-5pct — Complement-Coagulation-Bridge-Protein-C-CFI-Cofactor — CAPS-DDx-APLA-Mandatory — PAH-Overlap — FFP-Dual-Benefit',
  DGKE: 'AR aHUS COMPLEMENT-INDEPENDENT DGK-epsilon — DGKE-520aa — 17q22 — Eculizumab-FAILS-DO-NOT-USE — Infantile-ALWAYS-Before-2yr — Normal-Complement-All-Parameters — Heavy-Proteinuria-Plus-HUS-PATHOGNOMONIC',
  C5:   'AD C5-Component Eculizumab-Target — C5-1676aa — 9q33.2 — Anti-C5-mAb-Direct-Target — Arg885His-Japanese-ECULIZUMAB-RESISTANT-Switch-Ravulizumab — LOF-Absent-MAC-Neisseria-Recurrent — MAC-C5b-9',
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

export default function HereditaryComplementDisorderAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/hereditary-complement-disorder-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-complement-disorder-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-complement-disorder-atlas/definitions`).then(r => r.json()),
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
          🧬 Hereditary Complement Disorder Atlas
        </div>
        <h1 style={{ margin: 0, fontSize: 'clamp(1.1rem,2.5vw,1.6rem)', fontWeight: 800, color: '#e0e7ff', lineHeight: 1.3 }}>
          Hereditary-Complement-Disorder-Atlas — Complete 8-Gene aHUS / C3G Reference
        </h1>
        <p style={{ margin: '0.5rem 0 0', color: '#94a3b8', fontSize: 13 }}>
          CFH · CFI · C3 · CFB · CD46/MCP · THBD · DGKE · C5 &nbsp;|&nbsp; 320-Patient Aggregate · Seeds 1926–1933
        </p>
        <div style={{ marginTop: '0.75rem', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Badge text="aHUS" color="#6366f1" />
          <Badge text="C3 Glomerulopathy" color="#8b5cf6" />
          <Badge text="Eculizumab / Ravulizumab" color="#10b981" />
          <Badge text="DGKE — Complement-Independent" color="#be185d" />
          <Badge text="C5 Arg885His Resistant" color="#f59e0b" />
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid #334155', background: '#1e293b' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '0.75rem 1.25rem', background: 'none', border: 'none', cursor: 'pointer',
            color: tab === t ? '#818cf8' : '#64748b', fontWeight: tab === t ? 700 : 400,
            borderBottom: tab === t ? '2px solid #818cf8' : '2px solid transparent',
            fontSize: 13, transition: 'all 0.15s',
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '1.5rem', maxWidth: 1400, margin: '0 auto' }}>
        {loading && <Loading />}
        {error && <ErrorBox msg={error} />}

        {/* ── OVERVIEW ── */}
        {tab === 'Overview' && !loading && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: '1.5rem' }}>
              <KPI label="Total Patients" value={stats.total_patients} color="#6366f1" />
              <KPI label="Genes Covered" value={stats.genes_covered} color="#10b981" />
              <KPI label="Avg Age at Dx (yr)" value={stats.avg_age_at_diagnosis_yr} color="#f59e0b" />
              <KPI label="Severe Cases %" value={`${stats.severe_cases_pct}%`} color="#ef4444" />
              <KPI label="Seeds" value={stats.seed_range} color="#8b5cf6" />
            </div>

            {/* Gene colour key */}
            <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
              <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10, fontWeight: 600 }}>GENE COLOUR KEY</div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {Object.entries(GENE_COLORS).map(([gene, color]) => (
                  <span key={gene} style={{
                    background: color + '22', color, border: `1px solid ${color}55`,
                    borderRadius: 6, padding: '3px 10px', fontSize: 12, fontWeight: 700,
                  }}>{gene}</span>
                ))}
              </div>
            </div>

            {/* Gene summary table */}
            <div style={{ background: '#1e293b', borderRadius: 10, overflow: 'auto', marginBottom: '1.5rem' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#0f172a' }}>
                    {['Gene', 'Alt Name', 'Locus', 'Size', 'Inheritance', 'Patients'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', color: '#94a3b8', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summary.map((g, i) => (
                    <tr key={g.gene} style={{ borderTop: '1px solid #334155', background: i % 2 ? '#1e293b' : '#16213e' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#e2e8f0' }}>{g.gene}</td>
                      <td style={{ padding: '8px 12px', color: '#cbd5e1' }}>{g.alt_name}</td>
                      <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{g.locus}</td>
                      <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{g.protein_size}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <Badge text={g.inheritance}
                          color={g.inheritance.startsWith('AR') ? '#8b5cf6' : g.inheritance === 'AD' ? '#f59e0b' : '#6366f1'} />
                      </td>
                      <td style={{ padding: '8px 12px', color: '#e2e8f0', fontWeight: 600 }}>{g.n_patients}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Key clinical distinctions */}
            <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem' }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#818cf8', marginBottom: 12 }}>
                ⚡ KEY CLINICAL DISTINCTIONS
              </div>
              {overview.key_clinical_distinctions.map((d, i) => {
                const [flag, ...rest] = d.split(':');
                const isAlert = flag.includes('ECULIZUMAB-FAILS') || flag.includes('RESISTANT') || flag.includes('INEFFECTIVE');
                return (
                  <div key={i} style={{
                    borderLeft: `3px solid ${isAlert ? '#ef4444' : '#6366f1'}`,
                    marginBottom: 8, padding: '6px 10px',
                    background: isAlert ? '#450a0a22' : '#0f172a',
                    borderRadius: '0 6px 6px 0',
                  }}>
                    <span style={{ color: isAlert ? '#f87171' : '#818cf8', fontWeight: 700, fontSize: 11 }}>{flag}:</span>
                    <span style={{ color: '#cbd5e1', fontSize: 12 }}>{rest.join(':')}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── GENE TABLE ── */}
        {tab === 'Gene Table' && !loading && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 16 }}>
            {Object.entries(GENE_DISEASE).map(([gene, disease]) => (
              <div key={gene} style={{
                background: '#1e293b', borderRadius: 10,
                borderLeft: `4px solid ${GENE_COLORS[gene] || '#6366f1'}`,
                padding: '1rem',
              }}>
                <div style={{ fontWeight: 800, fontSize: 16, color: GENE_COLORS[gene], marginBottom: 6 }}>{gene}</div>
                <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>
                  {disease.split(' — ').map((part, i) => (
                    <div key={i} style={{ marginBottom: 2 }}>
                      {i === 0 ? <strong style={{ color: '#cbd5e1' }}>{part}</strong> : part}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── CLINICAL ATLAS ── */}
        {tab === 'Clinical Atlas' && !loading && breakdown && (
          <div>
            {breakdown.genes.map(g => (
              <div key={g.gene} style={{
                background: '#1e293b', borderRadius: 10, padding: '1.25rem',
                marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
              }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10 }}>
                  <span style={{ fontSize: 18, fontWeight: 800, color: GENE_COLORS[g.gene] }}>{g.gene}</span>
                  <span style={{ color: '#94a3b8', fontSize: 13 }}>{g.alt_name}</span>
                  <Badge text={g.inheritance}
                    color={g.inheritance.startsWith('AR') ? '#8b5cf6' : '#f59e0b'} />
                  <Badge text={g.locus} color="#475569" />
                  <Badge text={g.protein_size} color="#334155" />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 10 }}>
                  <div>
                    <div style={{ fontSize: 10, color: '#64748b', fontWeight: 600, marginBottom: 4 }}>AGE OF ONSET</div>
                    <div style={{ fontSize: 12, color: '#cbd5e1' }}>{g.age_of_onset}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: '#64748b', fontWeight: 600, marginBottom: 4 }}>KEY BIOMARKER</div>
                    <div style={{ fontSize: 12, color: '#cbd5e1' }}>{g.key_biomarker}</div>
                  </div>
                </div>

                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 10, color: '#64748b', fontWeight: 600, marginBottom: 4 }}>PATHOGNOMONIC</div>
                  <div style={{ fontSize: 12, color: '#fde68a', background: '#422006', borderRadius: 6, padding: '6px 10px' }}>
                    {g.pathognomonic}
                  </div>
                </div>

                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 10, color: '#64748b', fontWeight: 600, marginBottom: 4 }}>TREATMENT</div>
                  <div style={{ fontSize: 12, color: '#bbf7d0', background: '#052e16', borderRadius: 6, padding: '6px 10px' }}>
                    {g.treatment}
                  </div>
                </div>

                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 10, color: '#64748b', fontWeight: 600, marginBottom: 6 }}>CRITICAL FLAGS</div>
                  {g.critical_flags.map((cf, i) => {
                    const [flag, ...rest] = cf.split(':');
                    const isAlert = flag.includes('FAILS') || flag.includes('RESISTANT') || flag.includes('INEFFECTIVE') || flag.includes('PITFALL');
                    return (
                      <div key={i} style={{
                        borderLeft: `2px solid ${isAlert ? '#ef4444' : '#6366f1'}`,
                        marginBottom: 4, padding: '4px 8px', fontSize: 11,
                        background: isAlert ? '#450a0a11' : 'transparent',
                        borderRadius: '0 4px 4px 0',
                      }}>
                        <span style={{ color: isAlert ? '#f87171' : '#818cf8', fontWeight: 700 }}>{flag}:</span>
                        <span style={{ color: '#94a3b8' }}>{rest.join(':')}</span>
                      </div>
                    );
                  })}
                </div>

                {/* Severity distribution */}
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 11, color: '#64748b' }}>Severity: </span>
                  {Object.entries(g.severity_distribution).map(([sev, cnt]) => (
                    <Badge key={sev} text={`${sev}: ${cnt}`}
                      color={sev === 'severe' ? '#ef4444' : sev === 'moderate' ? '#f59e0b' : '#10b981'} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── DEFINITIONS ── */}
        {tab === 'Definitions' && !loading && definitions && (
          <div>
            {/* Gene definitions */}
            <div style={{ marginBottom: '2rem' }}>
              <h2 style={{ color: '#818cf8', fontSize: 14, fontWeight: 700, marginBottom: 12 }}>GENE DEFINITIONS</h2>
              {definitions.genes.map(g => (
                <div key={g.gene} style={{
                  background: '#1e293b', borderRadius: 10, padding: '1.25rem',
                  marginBottom: 12, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontWeight: 800, fontSize: 15, color: GENE_COLORS[g.gene] }}>{g.gene}</span>
                    <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.alt_name}</span>
                    <Badge text={g.inheritance}
                      color={g.inheritance.startsWith('AR') ? '#8b5cf6' : '#f59e0b'} />
                    <Badge text={g.locus} color="#475569" />
                  </div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, marginBottom: 10 }}>
                    {g.definition}
                  </div>
                  <div style={{ fontSize: 10, color: '#64748b', fontWeight: 600, marginBottom: 6 }}>CRITICAL FLAGS</div>
                  {g.critical_flags.map((cf, i) => {
                    const [flag, ...rest] = cf.split(':');
                    const isAlert = flag.includes('FAILS') || flag.includes('RESISTANT') || flag.includes('INEFFECTIVE');
                    return (
                      <div key={i} style={{
                        borderLeft: `2px solid ${isAlert ? '#ef4444' : '#6366f1'}`,
                        marginBottom: 3, padding: '3px 8px', fontSize: 11,
                        background: isAlert ? '#450a0a11' : 'transparent',
                        borderRadius: '0 4px 4px 0',
                      }}>
                        <span style={{ color: isAlert ? '#f87171' : '#818cf8', fontWeight: 700 }}>{flag}:</span>
                        <span style={{ color: '#94a3b8' }}>{rest.join(':')}</span>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>

            {/* Glossary */}
            <div>
              <h2 style={{ color: '#818cf8', fontSize: 14, fontWeight: 700, marginBottom: 12 }}>GLOSSARY</h2>
              {Object.entries(definitions.glossary).map(([term, def]) => (
                <div key={term} style={{
                  background: '#1e293b', borderRadius: 8, padding: '1rem',
                  marginBottom: 10, borderLeft: '3px solid #334155',
                }}>
                  <div style={{ fontWeight: 700, color: '#818cf8', fontSize: 13, marginBottom: 6 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>{def}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
