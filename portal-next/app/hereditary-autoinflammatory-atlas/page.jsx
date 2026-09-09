'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-autoinflammatory-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  MEFV:     '#b71c1c',  // deep red        — FMF, most common autoinflammatory, Mediterranean/Middle Eastern
  TNFRSF1A: '#1565c0',  // deep blue        — TRAPS, longest attacks, periorbital edema, highest amyloid
  NLRP3:    '#0d47a1',  // navy             — CAPS spectrum, FCAS/MWS/NOMID, cold-triggered, canakinumab
  MVK:      '#283593',  // dark indigo      — MKD/HIDS, vaccination-triggered, urinary mevalonate
  PSTPIP1:  '#1b5e20',  // dark green       — PAPA, pyogenic arthritis + pyoderma + acne triad
  NOD2:     '#6a1b9a',  // deep purple      — Blau syndrome, granulomatous uveitis, GOF vs Crohn's LOF
  IL1RN:    '#e65100',  // burnt orange     — DIRA, neonatal, anakinra curative, lethal without treatment
  IL36RN:   '#004d40',  // dark teal        — DITRA, GPP, spesolimab FDA 2022
};

const GENE_INFO = {
  MEFV:     { full: 'MEFV/Pyrin / 781aa', locus: '16p13.3', size: '781 aa / 95 kDa', inh: 'AR (incomplete penetrance)', disease: 'FMF — MOST COMMON autoinflammatory worldwide; short self-limiting fever 12-72h + serositis PATHOGNOMONIC; AA amyloidosis M694V homozygotes; colchicine 1-2mg/day LIFELONG first-line; anakinra/canakinumab for colchicine-resistant' },
  TNFRSF1A: { full: 'TNFRSF1A/TNFR1 / 455aa', locus: '12p13.31', size: '455 aa / 50 kDa', inh: 'AD', disease: 'TRAPS — LONGEST fever attacks >7 days; migratory centrifugal rash + periorbital edema + myalgia PATHOGNOMONIC; HIGHEST AA amyloid risk; IL-1 blockade superior to etanercept; low-penetrance p.R92Q milder' },
  NLRP3:    { full: 'NLRP3/Cryopyrin / 1036aa', locus: '1q44', size: '1036 aa / 118 kDa', inh: 'AD GOF', disease: 'CAPS — cold-triggered urticaria rash PATHOGNOMONIC; spectrum FCAS<MWS<NOMID; sensorineural hearing loss MWS/NOMID; NOMID neonatal chronic meningitis; canakinumab FDA 2009 first-line DRAMATIC response' },
  MVK:      { full: 'MVK/Mevalonate Kinase / 396aa', locus: '12q24.11', size: '396 aa / 45 kDa', inh: 'AR', disease: 'MKD/HIDS — vaccination-triggered attacks PATHOGNOMONIC; elevated urinary mevalonate during attacks biochemical PATHOGNOMONIC; high IgD >100 IU/mL; Dutch/European p.Val377Ile founder; anakinra/canakinumab effective' },
  PSTPIP1:  { full: 'PSTPIP1 / 416aa', locus: '15q24.3', size: '416 aa / 47 kDa', inh: 'AD', disease: 'PAPA Syndrome — sterile destructive arthritis + pyoderma gangrenosum + cystic acne TRIAD PATHOGNOMONIC; p.A230T/p.E250K founders; PSTPIP1 binds pyrin; IL-1 blockade most effective; pathergy pyoderma — avoid surgery' },
  NOD2:     { full: 'NOD2/CARD15 / 1040aa', locus: '16q12.1', size: '1040 aa / 115 kDa', inh: 'AD GOF', disease: 'Blau Syndrome — granulomatous uveitis + polyarthritis + skin rash TRIAD PATHOGNOMONIC; <4 years onset; NOD2 GOF (NOT same as Crohn\'s LOF variants); uveitis most severe — blindness risk; adalimumab/infliximab most effective' },
  IL1RN:    { full: 'IL1RN/IL-1Ra / 177aa', locus: '2q14.1', size: '177 aa / 25 kDa', inh: 'AR', disease: 'DIRA — NEONATAL onset within weeks PATHOGNOMONIC; multifocal osteomyelitis + periostitis + sterile pustulosis TRIAD; anakinra CURATIVE within 24-48h; WITHOUT TREATMENT LETHAL; Puerto Rican/Dutch founder deletions' },
  IL36RN:   { full: 'IL36RN/IL-36Ra / 155aa', locus: '2q14.1', size: '155 aa / 17 kDa', inh: 'AR', disease: 'DITRA — generalised pustular psoriasis (GPP) PATHOGNOMONIC; IL-36Ra absent -> IL-36α/β/γ unopposed in keratinocytes; p.Ser113Leu Mediterranean/Asian founder; spesolimab (anti-IL-36R, FDA 2022) first approved targeted therapy' },
};

function GeneChip({ gene }) {
  return (
    <span style={{
      background: GENE_COLORS[gene] || '#555',
      color: '#fff',
      borderRadius: 4,
      padding: '2px 8px',
      fontSize: 12,
      fontWeight: 700,
      marginRight: 4,
      display: 'inline-block',
    }}>{gene}</span>
  );
}

function MetricCard({ label, value, sub }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', minWidth: 140, flex: '1 1 140px' }}>
      <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#f1f5f9', fontSize: 22, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function AutoinflammatoryAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' || tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    setLoading(true);
    setError(null);
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => {
        if (ep === 'overview') setOverview(d);
        else if (ep === 'breakdown') setBreakdown(d);
        else setDefinitions(d);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui,sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1b5e20,#0d47a1)', padding: '24px 32px' }}>
        <div style={{ fontSize: 11, color: '#a7f3d0', letterSpacing: 2, marginBottom: 6 }}>
          HEREDITARY DISEASE ATLAS · AUTOINFLAMMATORY
        </div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: '#fff' }}>
          Hereditary-Autoinflammatory-Syndrome-Atlas
        </h1>
        <div style={{ color: '#cbd5e1', marginTop: 6, fontSize: 14 }}>
          Complete 8-Gene FMF/TRAPS/CAPS/MKD/PAPA/Blau/DIRA/DITRA Autoinflammatory Reference
          &nbsp;·&nbsp; 320 Patients · Seeds 2342–2349
        </div>
        {/* Gene chips */}
        <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {Object.keys(GENE_INFO).map(g => <GeneChip key={g} gene={g} />)}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid #1e293b', background: '#0f172a', padding: '0 32px' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: 'none', border: 'none', color: tab === t ? '#4ade80' : '#64748b',
            borderBottom: tab === t ? '2px solid #4ade80' : '2px solid transparent',
            padding: '12px 18px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
            fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '28px 32px', maxWidth: 1200 }}>
        {loading && <div style={{ color: '#64748b' }}>Loading…</div>}
        {error && <div style={{ color: '#f87171' }}>Error: {error}</div>}

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <h2 style={{ color: '#f8fafc', marginTop: 0 }}>{overview.atlas}</h2>
            <p style={{ color: '#94a3b8' }}>{overview.subtitle}</p>

            {/* Aggregate metrics */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 28 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40 cohort" />
              <MetricCard label="Avg Attack Duration" value={`${overview.aggregate_metrics?.avg_attack_duration_days}d`} sub="across all genes" />
              <MetricCard label="IL-1 Response" value={`${overview.aggregate_metrics?.il1_response_pct}%`} sub="across cohorts" />
              <MetricCard label="Amyloid Risk" value={`${overview.aggregate_metrics?.amyloid_risk_pct}%`} sub="FMF+TRAPS highest" />
              <MetricCard label="Vaccination Trigger" value={`${overview.aggregate_metrics?.vaccination_trigger_pct}%`} sub="MVK enriched" />
              <MetricCard label="Cold Trigger" value={`${overview.aggregate_metrics?.cold_trigger_pct}%`} sub="CAPS enriched" />
              <MetricCard label="Remission on Biologics" value={`${overview.aggregate_metrics?.remission_on_biologics_pct}%`} sub="all cohorts" />
            </div>

            {/* Gene summary cards */}
            <h3 style={{ color: '#cbd5e1', borderBottom: '1px solid #1e293b', paddingBottom: 8 }}>Per-Gene Summary</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {Object.entries(overview.gene_summary || {}).map(([gene, info]) => (
                <div key={gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                    <GeneChip gene={gene} />
                    <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{info.disease_category}</span>
                    <span style={{ color: '#64748b', fontSize: 12 }}>· {info.locus} · {info.protein_size} · {info.inheritance}</span>
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: 13, marginBottom: 8 }}>{info.pathognomonic}</div>
                  <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, color: '#64748b' }}>
                    <span>Avg attack: <b style={{ color: '#f59e0b' }}>{info.avg_attack_duration_days}d</b></span>
                    <span>IL-1 response: <b style={{ color: '#34d399' }}>{info.il1_response_pct}%</b></span>
                    <span>Amyloid risk: <b style={{ color: '#f87171' }}>{info.amyloid_risk_pct}%</b></span>
                    <span>Vaccination trigger: <b style={{ color: '#60a5fa' }}>{info.vaccination_trigger_pct}%</b></span>
                    <span>Cold trigger: <b style={{ color: '#a78bfa' }}>{info.cold_trigger_pct}%</b></span>
                    <span>Remission/biologics: <b style={{ color: '#4ade80' }}>{info.remission_biologics_pct}%</b></span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && breakdown && (
          <div>
            <h2 style={{ color: '#f8fafc', marginTop: 0 }}>Gene Reference Table</h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
                    {['Gene', 'Locus', 'Size', 'Inh', 'Disease', 'Key Pathognomonic', 'Amyloid%', 'IL-1%', 'Vax%', 'Cold%'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.gene_breakdowns || []).map((row, i) => (
                    <tr key={row.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b', borderBottom: '1px solid #334155' }}>
                      <td style={{ padding: '9px 12px' }}><GeneChip gene={row.gene} /></td>
                      <td style={{ padding: '9px 12px', color: '#94a3b8' }}>{row.locus}</td>
                      <td style={{ padding: '9px 12px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{row.protein_size}</td>
                      <td style={{ padding: '9px 12px', color: '#fbbf24' }}>{row.inheritance?.split(';')[0]?.split('(')[0]?.trim()}</td>
                      <td style={{ padding: '9px 12px', color: '#e2e8f0', maxWidth: 150 }}>{row.disease_category?.split('--')[0]?.trim()}</td>
                      <td style={{ padding: '9px 12px', color: '#94a3b8', maxWidth: 220, fontSize: 12 }}>{row.pathognomonic?.slice(0, 110)}…</td>
                      <td style={{ padding: '9px 12px', color: row.amyloid_risk_pct > 10 ? '#f87171' : '#475569' }}>{row.amyloid_risk_pct}%</td>
                      <td style={{ padding: '9px 12px', color: '#34d399' }}>{row.il1_response_pct}%</td>
                      <td style={{ padding: '9px 12px', color: row.vaccination_trigger_pct > 30 ? '#60a5fa' : '#475569' }}>{row.vaccination_trigger_pct}%</td>
                      <td style={{ padding: '9px 12px', color: row.cold_trigger_pct > 30 ? '#a78bfa' : '#475569' }}>{row.cold_trigger_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            <h2 style={{ color: '#f8fafc', marginTop: 0 }}>Clinical Atlas — Per-Gene Detail</h2>
            {(breakdown.gene_breakdowns || []).map(entry => (
              <div key={entry.gene} style={{ background: '#1e293b', borderRadius: 10, padding: 20, marginBottom: 20, borderLeft: `5px solid ${GENE_COLORS[entry.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <GeneChip gene={entry.gene} />
                  <span style={{ color: '#f1f5f9', fontWeight: 700, fontSize: 16 }}>
                    {GENE_INFO[entry.gene]?.full}
                  </span>
                  <span style={{ color: '#64748b', fontSize: 13 }}>· {entry.inheritance?.split(';')[0]}</span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                  <div>
                    <div style={{ color: '#64748b', fontSize: 11, marginBottom: 3 }}>DISEASE CATEGORY</div>
                    <div style={{ color: '#fbbf24', fontWeight: 600 }}>{entry.disease_category}</div>
                  </div>
                  <div>
                    <div style={{ color: '#64748b', fontSize: 11, marginBottom: 3 }}>ONSET</div>
                    <div style={{ color: '#e2e8f0' }}>{entry.onset_age}</div>
                  </div>
                </div>

                <div style={{ background: '#0f172a', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                  <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>PATHOGNOMONIC / DIAGNOSTIC</div>
                  <div style={{ color: '#94a3b8', fontSize: 13 }}>{entry.pathognomonic}</div>
                </div>

                <div style={{ background: '#0f172a', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                  <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>TREATMENT</div>
                  <div style={{ color: '#86efac', fontSize: 13 }}>{entry.treatment}</div>
                </div>

                <div style={{ background: '#0f172a', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                  <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>KEY DDx</div>
                  <div style={{ color: '#fca5a5', fontSize: 13 }}>{entry.key_ddx}</div>
                </div>

                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
                  {(entry.key_features || []).map((f, i) => (
                    <span key={i} style={{ background: '#1e3a5f', borderRadius: 4, padding: '3px 10px', fontSize: 12, color: '#93c5fd' }}>{f}</span>
                  ))}
                </div>

                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, color: '#64748b', marginTop: 4 }}>
                  <span>Amyloid risk: <b style={{ color: entry.amyloid_risk?.includes('HIGH') ? '#f87171' : '#94a3b8' }}>{entry.amyloid_risk}</b></span>
                  <span>Colchicine: <b style={{ color: entry.colchicine_response?.includes('EXCELLENT') ? '#34d399' : '#94a3b8' }}>{entry.colchicine_response}</b></span>
                  <span>IL-1 response: <b style={{ color: '#4ade80' }}>{entry.il1_response}</b></span>
                  <span>Common trigger: <b style={{ color: '#a78bfa' }}>{entry.attack_trigger_common}</b></span>
                </div>

                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, marginTop: 8, paddingTop: 8, borderTop: '1px solid #1e293b' }}>
                  <span style={{ color: '#64748b' }}>n={entry.n_patients} patients</span>
                  <span style={{ color: '#f59e0b' }}>Avg attack: {entry.avg_attack_duration_days}d</span>
                  <span style={{ color: '#f87171' }}>Amyloid: {entry.amyloid_risk_pct}%</span>
                  <span style={{ color: '#34d399' }}>IL-1 response: {entry.il1_response_pct}%</span>
                  <span style={{ color: '#60a5fa' }}>Vax trigger: {entry.vaccination_trigger_pct}%</span>
                  <span style={{ color: '#a78bfa' }}>Cold trigger: {entry.cold_trigger_pct}%</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            <h2 style={{ color: '#f8fafc', marginTop: 0 }}>Autoinflammatory Clinical Definitions & Glossary</h2>

            {/* Gene definitions */}
            <h3 style={{ color: '#cbd5e1' }}>Gene Definitions</h3>
            {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => (
              <div key={gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 12, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <GeneChip gene={gene} />
                  <span style={{ color: '#f1f5f9', fontWeight: 600 }}>{entry.disease_name}</span>
                  <span style={{ color: '#64748b', fontSize: 12 }}>· {entry.locus} · {entry.protein_size} · {entry.inheritance}</span>
                </div>
                <div style={{ color: '#94a3b8', fontSize: 13, marginBottom: 6 }}><b style={{ color: '#fbbf24' }}>Pathognomonic:</b> {entry.pathognomonic?.slice(0, 250)}…</div>
                <div style={{ color: '#86efac', fontSize: 13, marginBottom: 6 }}><b>Treatment:</b> {entry.treatment}</div>
                <div style={{ color: '#64748b', fontSize: 12 }}><b>DDx:</b> {entry.key_ddx}</div>
              </div>
            ))}

            {/* Glossary */}
            <h3 style={{ color: '#cbd5e1', marginTop: 28 }}>Autoinflammatory Glossary</h3>
            {Object.entries(definitions.autoinflammatory_glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 10 }}>
                <div style={{ color: '#fbbf24', fontWeight: 600, marginBottom: 6 }}>{term}</div>
                <div style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.6 }}>{def}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
