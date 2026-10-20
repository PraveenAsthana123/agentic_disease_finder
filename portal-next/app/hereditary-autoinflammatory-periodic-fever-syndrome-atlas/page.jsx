'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-autoinflammatory-periodic-fever-syndrome-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MEFV':    '#b71c1c',  // deep red      — FMF most common; colchicine
  'NLRP3':   '#1565c0',  // deep blue     — CAPS; IL-1 blockade transformative
  'MVK':     '#2e7d32',  // dark green    — HIDS; mevalonic acid biomarker
  'TNFRSF1A':'#6a1b9a',  // deep purple   — TRAPS; migratory myalgia; long attacks
  'NLRC4':   '#e65100',  // deep orange   — NLRC4-MAS; extreme ferritin+IL-18
  'PSTPIP1': '#00695c',  // dark teal     — PAPA; PG surgery absolute CI
  'NOD2':    '#4a148c',  // indigo        — Blau; granulomatous triad; no lung
  'IL36RN':  '#f9a825',  // amber         — DITRA/GPP; spesolimab FDA 2022
};

const GENE_INFO = {
  'MEFV':     { full: 'MEFV / Pyrin / 781aa', locus: '16p13.3', size: '781 aa / 86 kDa (Pyrin inflammasome sensor; detects RhoA GTPase inactivation; neutrophils/monocytes/serosal cells; AR biallelic = classic FMF; AD monoallelic = milder autoinflammatory; M694V most severe; colchicine stabilizes microtubules → prevents pyrin activation)', inh: 'AR / AD' },
  'NLRP3':    { full: 'NLRP3 / Cryopyrin / 1036aa', locus: '1q44', size: '1036 aa / 118 kDa (NLRP3 inflammasome; detects ATP/uric acid/K+ efflux/silica; NLRP3+ASC+caspase-1 → IL-1β+IL-18; AD GOF; CAPS spectrum FCAS→MWS→NOMID; SNHL (MWS); neonatal onset (NOMID); canakinumab FDA 2009)', inh: 'AD GOF' },
  'MVK':      { full: 'MVK / Mevalonate Kinase / 396aa', locus: '12q24.11', size: '396 aa / 42 kDa (mevalonate → 5-phosphomevalonate; isoprenoid pathway; MVK LOF → isoprenoid deficiency → Rac1/NLRP3 → IL-1β; AR; HIDS (partial) vs MA (complete); urinary mevalonic acid PATHOGNOMONIC during attacks)', inh: 'AR' },
  'TNFRSF1A': { full: 'TNFRSF1A / TNFR1 / 235aa', locus: '12p13.31', size: '235 aa / 26 kDa (TNFR1/p55; NF-κB + apoptosis; structural Cys variants → ER retention → reduced sTNFR1 shedding → enhanced TNF signaling; AD; TRAPS attacks 7-21+ days DISTINGUISHING; canakinumab FDA 2016; NOT infliximab/adalimumab)', inh: 'AD' },
  'NLRC4':    { full: 'NLRC4 / IPAF / 1024aa', locus: '2p22.3', size: '1024 aa / 115 kDa (NLRC4 inflammasome; NAIP co-receptor; detects flagellin/T3SS; AD GOF/somatic; NLRC4-MAS; extreme ferritin >10,000 + IL-18 >10,000 pg/mL PATHOGNOMONIC; IL-18 blockade tadekinig alfa; ruxolitinib)', inh: 'AD GOF / Somatic GOF' },
  'PSTPIP1':  { full: 'PSTPIP1 / CD2BP1 / 415aa', locus: '15q24.3', size: '415 aa / 48 kDa (binds pyrin → hyperactivates pyrin inflammasome → excess IL-1β; AD GOF; PAPA triad: sterile pyogenic arthritis + pyoderma gangrenosum + cystic acne; PG pathergy → surgery ABSOLUTE CI; anakinra first-line)', inh: 'AD GOF' },
  'NOD2':     { full: 'NOD2 / CARD15 / 1040aa', locus: '16q12.1', size: '1040 aa / 115 kDa (intracellular PRR; detects MDP from peptidoglycan; NOD2+ASC+procaspase-1; AD GOF (NACHT domain); Blau/EOS granulomatous triad: arthritis+uveitis+skin; NO lung; onset <5yr; TNF inhibitors first-line; camptodactyly early sign)', inh: 'AD GOF' },
  'IL36RN':   { full: 'IL36RN / IL-36Ra / 155aa', locus: '2q14.1', size: '155 aa / 17 kDa (endogenous antagonist of IL-36R/IL1RL2; IL36RN LOF → unchecked IL-36α/β/γ → NF-κB in keratinocytes → neutrophil recruitment → pustules; AR biallelic; DITRA/GPP; Kogoj microabscess PATHOGNOMONIC; spesolimab FDA 2022)', inh: 'AR' },
};

const PATHOGNOMONIC = {
  'MEFV':     'Peritonitis + erysipelas-like erythema lower leg; colchicine 90% response; M694V homozygous = highest AA amyloid risk',
  'NLRP3':    'Urticaria + SNHL (MWS); patella enlargement (NOMID); cold-triggered (FCAS); IL-18 elevated; canakinumab FDA 2009',
  'MVK':      'Cervical lymphadenopathy + elevated IgD (>100 IU/mL) + urinary mevalonic acid during attacks PATHOGNOMONIC',
  'TNFRSF1A': 'Migratory myalgia + periorbital edema + attacks 7-21+ days; soluble sTNFR1 REDUCED during attacks PATHOGNOMONIC',
  'NLRC4':    'Free serum IL-18 >10,000 pg/mL PATHOGNOMONIC; extreme ferritin >10,000; NLRC4-MAS; enterocolitis; IL-18 blockade',
  'PSTPIP1':  'PAPA triad: sterile arthritis + pyoderma gangrenosum + cystic acne; surgery ABSOLUTE CI (pathergy); anakinra first-line',
  'NOD2':     'Granulomatous triad: arthritis + uveitis + skin rash; NO lung involvement; onset <5yr; camptodactyly EARLY SIGN',
  'IL36RN':   'Generalized sterile pustules + Kogoj microabscess + IL36RN biallelic; spesolimab (anti-IL-36R) FDA 2022 first GPP treatment',
};

export default function HereditaryAutoinflammPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedDef, setExpandedDef] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#1a1a2e', minHeight: '100vh', fontSize: 18 }}>Loading Hereditary-Autoinflammatory-Periodic-Fever-Syndrome-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#1a1a2e', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#e6edf3', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#b71c1c 0%,#e65100 40%,#1565c0 100%)', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>Autoinflammatory Disease · Inflammasome Disorders · Periodic Fever Syndromes · IL-1/IL-18/IL-36 Axis</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
          🧬 Hereditary-Autoinflammatory-Periodic-Fever-Syndrome-Atlas
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
          Complete 8-Gene Reference · MEFV · NLRP3 · MVK · TNFRSF1A · NLRC4 · PSTPIP1 · NOD2 · IL36RN ·
          320 Patients (8×40) · Seeds 4078–4085
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Total Patients', val: overview?.total_patients },
            { label: 'Genes', val: overview?.genes?.length },
            { label: 'Colchicine Effective', val: 'MEFV only (pyrin)' },
            { label: 'NLRC4 IL-18', val: '>10,000 pg/mL PATHOGNOMONIC' },
            { label: 'Spesolimab FDA', val: '2022 (GPP first-specific)' },
          ].map(b => (
            <div key={b.label} style={{ background: 'rgba(255,255,255,0.12)', borderRadius: 8, padding: '6px 14px' }}>
              <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.65)', textTransform: 'uppercase' }}>{b.label}</div>
              <div style={{ fontSize: 15, fontWeight: 700, color: '#fff' }}>{b.val ?? '…'}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', borderBottom: '1px solid #30363d', background: '#161b22', padding: '0 24px' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            style={{ padding: '12px 20px', border: 'none', background: 'none', cursor: 'pointer',
              color: activeTab === t ? '#58a6ff' : '#8b949e',
              borderBottom: activeTab === t ? '2px solid #58a6ff' : '2px solid transparent',
              fontWeight: activeTab === t ? 600 : 400, fontSize: 14 }}>
            {t}
          </button>
        ))}
      </div>

      <div style={{ padding: '24px 32px' }}>

        {/* ── OVERVIEW TAB ── */}
        {activeTab === 'Overview' && overview && (
          <div>
            <div style={{ marginBottom: 20, padding: '14px 18px', background: '#161b22', borderRadius: 8, border: '1px solid #30363d' }}>
              <div style={{ fontWeight: 700, color: '#58a6ff', marginBottom: 8, fontSize: 15 }}>Atlas Summary</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 8 }}>
                {Object.entries(overview.key_facts || {}).map(([k, v]) => (
                  <div key={k} style={{ background: '#0d1117', borderRadius: 6, padding: '8px 12px', border: '1px solid #21262d' }}>
                    <div style={{ fontSize: 10, color: '#8b949e', textTransform: 'uppercase', marginBottom: 3 }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 12, color: '#e6edf3' }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 14 }}>
              {(overview.genes || []).map(g => {
                const c = overview.cohort_summary?.[g];
                if (!c) return null;
                return (
                  <div key={g} style={{ background: '#161b22', borderRadius: 10, border: `2px solid ${GENE_COLORS[g] || '#30363d'}`, padding: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                      <div>
                        <span style={{ fontWeight: 700, fontSize: 18, color: GENE_COLORS[g] || '#58a6ff' }}>{g}</span>
                        <span style={{ marginLeft: 8, fontSize: 12, color: '#8b949e' }}>{c.locus}</span>
                      </div>
                      <span style={{ fontSize: 11, color: '#8b949e', background: '#0d1117', padding: '2px 8px', borderRadius: 10 }}>{c.inheritance}</span>
                    </div>
                    <div style={{ fontSize: 12, color: '#c9d1d9', marginBottom: 6 }}>{c.syndrome}</div>
                    <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 8 }}>{GENE_INFO[g]?.size}</div>
                    <div style={{ background: '#21262d', borderRadius: 6, padding: '8px', fontSize: 11, color: '#f0b429', marginBottom: 10 }}>
                      {PATHOGNOMONIC[g]}
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                      {[
                        ['Serositis', c.serositis_pct + '%'],
                        ['AA Amyloid Risk', c.aa_amyloid_pct + '%'],
                        ['Colchicine Response', c.colchicine_response_pct + '%'],
                        ['Attack Duration', c.attack_duration_hr + ' hr'],
                      ].map(([label, val]) => (
                        <div key={label} style={{ background: '#0d1117', borderRadius: 4, padding: '5px 8px' }}>
                          <div style={{ fontSize: 9, color: '#8b949e', textTransform: 'uppercase' }}>{label}</div>
                          <div style={{ fontSize: 13, fontWeight: 700, color: '#e6edf3' }}>{val}</div>
                        </div>
                      ))}
                    </div>
                    <div style={{ marginTop: 8, fontSize: 11, color: '#8b949e' }}>
                      <span style={{ color: '#c9d1d9', fontWeight: 600 }}>Treatment: </span>{c.key_treatment}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── GENE TABLE TAB ── */}
        {activeTab === 'Gene Table' && overview && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#161b22', borderBottom: '2px solid #30363d' }}>
                  {['Gene', 'Locus', 'Size', 'Inh.', 'Syndrome', 'Serositis%', 'AA Amyloid%', 'Colchicine%', 'Attack Duration', 'Treatment'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', color: '#8b949e', fontWeight: 600, textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.genes || []).map((g, i) => {
                  const c = overview.cohort_summary?.[g];
                  if (!c) return null;
                  return (
                    <tr key={g} style={{ background: i % 2 === 0 ? '#0d1117' : '#161b22', borderBottom: '1px solid #21262d' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 700, color: GENE_COLORS[g] || '#58a6ff' }}>{g}</td>
                      <td style={{ padding: '8px 12px', color: '#8b949e' }}>{c.locus}</td>
                      <td style={{ padding: '8px 12px', color: '#c9d1d9', whiteSpace: 'nowrap' }}>{c.size_aa}aa / {c.kda}kDa</td>
                      <td style={{ padding: '8px 12px', color: '#8b949e', fontSize: 11 }}>{c.inheritance}</td>
                      <td style={{ padding: '8px 12px', color: '#c9d1d9', fontSize: 11 }}>{c.syndrome}</td>
                      <td style={{ padding: '8px 12px', color: c.serositis_pct >= 80 ? '#3fb950' : '#e6edf3', fontWeight: 600 }}>{c.serositis_pct}%</td>
                      <td style={{ padding: '8px 12px', color: c.aa_amyloid_pct >= 20 ? '#ff7b72' : '#e6edf3', fontWeight: c.aa_amyloid_pct >= 20 ? 700 : 400 }}>{c.aa_amyloid_pct}%</td>
                      <td style={{ padding: '8px 12px', color: c.colchicine_response_pct >= 80 ? '#3fb950' : c.colchicine_response_pct < 15 ? '#8b949e' : '#e6edf3', fontWeight: c.colchicine_response_pct >= 80 ? 700 : 400 }}>{c.colchicine_response_pct}%</td>
                      <td style={{ padding: '8px 12px', color: '#e6edf3', whiteSpace: 'nowrap' }}>{c.attack_duration_hr}h</td>
                      <td style={{ padding: '8px 12px', color: '#c9d1d9', fontSize: 11, maxWidth: 200 }}>{c.key_treatment}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* ── CLINICAL ATLAS TAB ── */}
        {activeTab === 'Clinical Atlas' && breakdown && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#58a6ff', marginBottom: 12, fontSize: 15 }}>Pathway / Mechanism Groups</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
                {Object.entries(breakdown.pathway_groups || {}).map(([key, grp]) => (
                  <div key={key} style={{ background: '#161b22', borderRadius: 8, border: '1px solid #30363d', padding: '14px' }}>
                    <div style={{ fontWeight: 600, color: '#e6edf3', marginBottom: 6, fontSize: 13 }}>
                      {grp.genes?.join(' · ')}
                    </div>
                    <div style={{ fontSize: 12, color: '#c9d1d9', marginBottom: 8 }}>{grp.description}</div>
                    <div style={{ fontSize: 11, color: '#f0b429', background: '#0d1117', padding: '6px 8px', borderRadius: 4 }}>
                      🔬 {grp.key_test}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ fontWeight: 700, color: '#58a6ff', marginBottom: 12, fontSize: 15 }}>Gene Clinical Data (Cohort of 40 per gene)</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#161b22', borderBottom: '2px solid #30363d' }}>
                    {['Gene', 'n', 'Serositis%', 'AA Amyloid%', 'Colchicine%', 'Mean Age Onset'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', color: '#8b949e', fontWeight: 600, textAlign: 'left', whiteSpace: 'nowrap', fontSize: 11 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview?.genes || []).map((g, i) => {
                    const s = breakdown.cohort_stats?.[g];
                    if (!s) return null;
                    return (
                      <tr key={g} style={{ background: i % 2 === 0 ? '#0d1117' : '#161b22', borderBottom: '1px solid #21262d' }}>
                        <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g] || '#58a6ff' }}>{g}</td>
                        <td style={{ padding: '7px 10px', color: '#8b949e' }}>{s.n}</td>
                        <td style={{ padding: '7px 10px', color: s.serositis_n / s.n >= 0.7 ? '#3fb950' : '#e6edf3', fontWeight: s.serositis_n / s.n >= 0.7 ? 700 : 400 }}>{Math.round(s.serositis_n / s.n * 100)}%</td>
                        <td style={{ padding: '7px 10px', color: s.aa_amyloid_n / s.n >= 0.2 ? '#ff7b72' : '#e6edf3', fontWeight: s.aa_amyloid_n / s.n >= 0.2 ? 700 : 400 }}>{Math.round(s.aa_amyloid_n / s.n * 100)}%</td>
                        <td style={{ padding: '7px 10px', color: s.colchicine_resp_n / s.n >= 0.8 ? '#3fb950' : '#8b949e', fontWeight: s.colchicine_resp_n / s.n >= 0.8 ? 700 : 400 }}>{Math.round(s.colchicine_resp_n / s.n * 100)}%</td>
                        <td style={{ padding: '7px 10px' }}>{s.mean_age_onset}yr</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {breakdown && (
              <div style={{ marginTop: 24 }}>
                <div style={{ fontWeight: 700, color: '#58a6ff', marginBottom: 10, fontSize: 15 }}>Gene Summary Cards</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 8 }}>
                  {Object.entries(breakdown.gene_summary || {}).map(([g, gs]) => (
                    <div key={g} style={{ background: '#161b22', borderRadius: 6, padding: '10px 14px', border: `1px solid ${GENE_COLORS[g] || '#30363d'}` }}>
                      <div style={{ fontWeight: 700, color: GENE_COLORS[g] || '#58a6ff', marginBottom: 4 }}>{g}</div>
                      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 6 }}>{gs.syndrome}</div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11 }}>
                        <div><span style={{ color: '#8b949e' }}>Serositis: </span><span style={{ color: '#e6edf3' }}>{gs.serositis_pct}%</span></div>
                        <div><span style={{ color: '#8b949e' }}>AA Amyloid: </span><span style={{ color: gs.aa_amyloid_pct >= 20 ? '#ff7b72' : '#e6edf3', fontWeight: gs.aa_amyloid_pct >= 20 ? 700 : 400 }}>{gs.aa_amyloid_pct}%</span></div>
                        <div><span style={{ color: '#8b949e' }}>Colchicine: </span><span style={{ color: gs.colchicine_resp_pct >= 80 ? '#3fb950' : '#8b949e', fontWeight: gs.colchicine_resp_pct >= 80 ? 700 : 400 }}>{gs.colchicine_resp_pct}%</span></div>
                        <div><span style={{ color: '#8b949e' }}>Mean Age: </span><span style={{ color: '#e6edf3' }}>{gs.mean_age_onset}yr</span></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {activeTab === 'Definitions' && definitions && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#58a6ff', marginBottom: 12, fontSize: 15 }}>Treatment Matrix</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 8 }}>
                {Object.entries(definitions.treatment_matrix || {}).map(([k, v]) => (
                  <div key={k} style={{ background: '#161b22', borderRadius: 6, padding: '10px 14px', border: '1px solid #30363d' }}>
                    <div style={{ fontSize: 11, color: '#8b949e', textTransform: 'uppercase', marginBottom: 4 }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 12, color: '#e6edf3' }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ marginBottom: 20 }}>
              <div style={{ fontWeight: 700, color: '#58a6ff', marginBottom: 10, fontSize: 15 }}>Key Distinguishers</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 8 }}>
                {Object.entries(definitions.key_distinguishers || {}).map(([k, v]) => (
                  <div key={k} style={{ background: '#161b22', borderRadius: 6, padding: '10px 14px', border: '1px solid #21262d' }}>
                    <div style={{ fontSize: 11, color: '#f0b429', textTransform: 'uppercase', marginBottom: 4 }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 12, color: '#e6edf3' }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ fontWeight: 700, color: '#58a6ff', marginBottom: 10, fontSize: 15 }}>Clinical Definitions</div>
            {(definitions.definitions || []).map((def, idx) => (
              <div key={idx} style={{ background: '#161b22', borderRadius: 8, border: '1px solid #30363d', marginBottom: 8, overflow: 'hidden' }}>
                <button onClick={() => setExpandedDef(expandedDef === idx ? null : idx)}
                  style={{ width: '100%', textAlign: 'left', padding: '12px 16px', background: 'none', border: 'none', cursor: 'pointer',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 600, color: '#e6edf3', fontSize: 13 }}>{def.term}</span>
                  <span style={{ color: '#58a6ff', fontSize: 16 }}>{expandedDef === idx ? '▲' : '▼'}</span>
                </button>
                {expandedDef === idx && (
                  <div style={{ padding: '0 16px 14px', fontSize: 12, color: '#c9d1d9', lineHeight: 1.65, borderTop: '1px solid #21262d' }}>
                    <div style={{ paddingTop: 10 }}>{def.definition}</div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  );
}
