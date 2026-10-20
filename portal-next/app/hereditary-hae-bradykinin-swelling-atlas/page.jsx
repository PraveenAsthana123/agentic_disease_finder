'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hae-bradykinin-swelling-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'SERPING1': '#b71c1c',  // deep red     — HAE type I/II; C1-INH; most common
  'F12':      '#1565c0',  // deep blue    — HAE type III; estrogen; OCP CI
  'PLG':      '#2e7d32',  // dark green   — HAE type IV; K311E; facial attacks
  'ANGPT1':   '#6a1b9a',  // deep purple  — HAE type V; Tie2 axis
  'KNG1':     '#e65100',  // deep orange  — HAE type VI; HMWK platform
  'MYOF':     '#00695c',  // dark teal    — HAE-nCI-MYOF; endothelial repair
  'HS3ST6':   '#4a148c',  // indigo       — HAE-nCI; heparan sulfate F12 platform
  'KLKB1':    '#f9a825',  // amber        — Fletcher factor; prolonged PTT; AR
};

const GENE_INFO = {
  'SERPING1': { full: 'SERPING1 / C1-Inhibitor / 478aa', locus: '11q12.1', size: '478 aa / 53 kDa (Serpin G1; C1-INH; inhibits C1s + C1r + FXIIa + pKal + FXIa; HAE type I (C1-INH level low) + type II (level normal, function low); AD LOF; C4 chronically low PATHOGNOMONIC; prevalence 1:50,000)', inh: 'AD LOF' },
  'F12':      { full: 'F12 / Factor XII / Hageman / 596aa', locus: '5q35.3', size: '596 aa / 68 kDa (contact activation initiator; F12 GOF variants K371/R353/T328; estrogen ERE in F12 promoter; HAE type III; women >> men; OCP ABSOLUTE CI; normal C4 + C1-INH; icatibant effective)', inh: 'AD GOF' },
  'PLG':      { full: 'PLG / Plasminogen / 810aa', locus: '6q26', size: '810 aa / 92 kDa (zymogen of plasmin; K311E kringle-3 variant; HAE type IV 2016; young women; facial attacks dominant; normal C4/C1-INH/F12; icatibant responsive; PLG activity assay)', inh: 'AD LOF' },
  'ANGPT1':   { full: 'ANGPT1 / Angiopoietin-1 / 498aa', locus: '8q23.1', size: '498 aa / 57 kDa (Tie2 agonist; vascular endothelial stability; ANGPT1 LOF → Tie2 destabilization → increased permeability; HAE type V 2020; bradykinin amplifies Ang2/Tie2 instability; normal C4/C1-INH)', inh: 'AD LOF' },
  'KNG1':     { full: 'KNG1 / HMW-Kininogen / 644aa', locus: '3q27.3', size: '644 aa / 88 kDa (HMWK; bradykinin substrate; FXIIa+KLKB1 cleave HMWK at Arg362-Ser363 → bradykinin released; KNG1 LOF → dysregulated contact platform; HAE type VI 2021; HMWK antigen/function assay required)', inh: 'AD LOF' },
  'MYOF':     { full: 'MYOF / Myoferlin / 2061aa', locus: '10p12.1', size: '2061 aa / 234 kDa (ferlin family; endothelial caveolae/membrane repair; VEGFR2 recycling; Weibel-Palade body exocytosis; MYOF LOF → prolonged bradykinin signaling; HAE-nCI 2020; milder attacks; AR biallelic = LGMD)', inh: 'AD LOF' },
  'HS3ST6':   { full: 'HS3ST6 / Heparan Sulfate 3-O-ST6 / 367aa', locus: '1p35.3', size: '367 aa / 42 kDa (heparan sulfate 3-O-sulfation; F12-HSPG binding platform; HS3ST6 LOF → dysregulated F12 contact activation termination; XLD pattern; 2021; normal C4/C1-INH; estrogen sensitive)', inh: 'XLD' },
  'KLKB1':    { full: 'KLKB1 / Prekallikrein / Fletcher / 619aa', locus: '4q35.2', size: '619 aa / 88 kDa (plasma prekallikrein; bound to HMWK; FXIIa → pKal → bradykinin from HMWK; AR LOF; Fletcher factor deficiency; markedly prolonged PTT without bleeding CLASSIC; HAE-like attacks subset; KLKB1 activity assay)', inh: 'AR LOF' },
};

const PATHOGNOMONIC = {
  'SERPING1': 'C4 CHRONICALLY LOW between attacks (even in remission) — best single screening test; C3 normal; C1q normal; laryngeal attacks 40% lifetime; icatibant + lanadelumab',
  'F12':      'HAE type III: NORMAL C4 + C1-INH; OCP/estrogen ABSOLUTE TRIGGER; women >> men; facial attacks dominant; F12 K371/R353 variants; icatibant effective',
  'PLG':      'HAE type IV: K311E kringle-3 variant; NORMAL C4 + C1-INH + F12; young women; FACIAL attacks most prominent (tongue/lip); icatibant responsive; PLG sequencing',
  'ANGPT1':   'HAE type V: Ang1/Tie2 vascular stability axis; NORMAL C4/C1-INH; bradykinin-mediated via endothelial priming; icatibant responsive; ANGPT1 sequencing on full panel',
  'KNG1':     'HAE type VI: HMW-kininogen (bradykinin substrate) AD LOF; NORMAL C4/C1-INH/F12; HMWK antigen + function assay + KNG1 sequencing required; icatibant responsive',
  'MYOF':     'HAE-nCI-MYOF: MILDER attacks; NORMAL C4/C1-INH/F12/PLG/ANGPT1; endothelial membrane repair defect; AD LOF; AR biallelic = LGMD (myopathy, not HAE)',
  'HS3ST6':   'HAE-nCI-HS3ST6: heparan sulfate F12 binding platform; XLD; NORMAL C4/C1-INH; estrogen sensitive; no father-to-son transmission if truly X-linked; genetic panel',
  'KLKB1':    'Fletcher factor deficiency: MARKEDLY PROLONGED PTT (>100s) WITHOUT BLEEDING — classic teaching case; HAE-like attacks (subset); KLKB1 activity assay; AR LOF',
};

export default function HereditaryHAEBradykininPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#1a1a2e', minHeight: '100vh', fontSize: 18 }}>Loading Hereditary-HAE-Bradykinin-Swelling-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#1a1a2e', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#e6edf3', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#b71c1c 0%,#1565c0 50%,#2e7d32 100%)', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>Hereditary Angioedema · Contact Activation · Kallikrein-Kinin System · Bradykinin-Mediated Swelling</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
          🧬 Hereditary-HAE-Bradykinin-Swelling-Atlas
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
          Complete 8-Gene Reference · SERPING1 · F12 · PLG · ANGPT1 · KNG1 · MYOF · HS3ST6 · KLKB1 ·
          320 Patients (8×40) · Seeds 4086–4093
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Total Patients', val: overview?.total_patients },
            { label: 'Genes', val: overview?.genes?.length },
            { label: 'C4 Screen (SERPING1)', val: 'Chronically low PATHOGNOMONIC' },
            { label: 'Antihistamines', val: 'INEFFECTIVE (bradykinin)' },
            { label: 'OCP in HAE', val: 'ABSOLUTE CONTRAINDICATION' },
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
              <div style={{ fontWeight: 700, color: '#58a6ff', marginBottom: 8, fontSize: 15 }}>Atlas Summary — Key Clinical Facts</div>
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
                        <span style={{ marginLeft: 8, fontSize: 12, color: '#8b949e' }}>HAE {c.hae_type}</span>
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
                        ['Laryngeal Risk', c.laryngeal_pct + '%'],
                        ['Abdominal Attack', c.abdominal_pct + '%'],
                        ['Icatibant Response', c.icatibant_resp_pct + '%'],
                        ['Mean Age Onset', c.mean_age_onset + ' yr'],
                      ].map(([label, val]) => (
                        <div key={label} style={{ background: '#0d1117', borderRadius: 4, padding: '5px 8px' }}>
                          <div style={{ fontSize: 9, color: '#8b949e', textTransform: 'uppercase' }}>{label}</div>
                          <div style={{ fontSize: 13, fontWeight: 700, color: '#e6edf3' }}>{val}</div>
                        </div>
                      ))}
                    </div>
                    <div style={{ marginTop: 8, display: 'flex', gap: 6, fontSize: 11, flexWrap: 'wrap' }}>
                      {c.c4_low_pct > 80 && <span style={{ background: '#b71c1c', color: '#fff', padding: '2px 6px', borderRadius: 4 }}>C4 Low</span>}
                      {c.estrogen_sens_pct > 60 && <span style={{ background: '#6a1b9a', color: '#fff', padding: '2px 6px', borderRadius: 4 }}>Estrogen-sensitive</span>}
                      {c.surgical_misdiag_pct > 15 && <span style={{ background: '#e65100', color: '#fff', padding: '2px 6px', borderRadius: 4 }}>Surgical trap</span>}
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
                  {['Gene', 'HAE Type', 'Inh.', 'Syndrome', 'Laryngeal%', 'Abdominal%', 'Icatibant%', 'Estrogen%', 'Mean Age Onset'].map(h => (
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
                      <td style={{ padding: '8px 12px', color: '#c9d1d9' }}>{c.hae_type}</td>
                      <td style={{ padding: '8px 12px', color: '#8b949e', fontSize: 11 }}>{c.inheritance}</td>
                      <td style={{ padding: '8px 12px', color: '#c9d1d9', fontSize: 11 }}>{c.syndrome}</td>
                      <td style={{ padding: '8px 12px', color: c.laryngeal_pct >= 35 ? '#ff7b72' : '#e6edf3', fontWeight: c.laryngeal_pct >= 35 ? 700 : 400 }}>{c.laryngeal_pct}%</td>
                      <td style={{ padding: '8px 12px', color: c.abdominal_pct >= 60 ? '#f0b429' : '#e6edf3', fontWeight: c.abdominal_pct >= 60 ? 700 : 400 }}>{c.abdominal_pct}%</td>
                      <td style={{ padding: '8px 12px', color: c.icatibant_resp_pct >= 85 ? '#3fb950' : '#e6edf3', fontWeight: c.icatibant_resp_pct >= 85 ? 700 : 400 }}>{c.icatibant_resp_pct}%</td>
                      <td style={{ padding: '8px 12px', color: c.estrogen_sens_pct >= 80 ? '#6a1b9a' : '#e6edf3', fontWeight: c.estrogen_sens_pct >= 80 ? 700 : 400 }}>{c.estrogen_sens_pct}%</td>
                      <td style={{ padding: '8px 12px' }}>{c.mean_age_onset}yr</td>
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
                    {['Gene', 'n', 'Laryngeal%', 'Abdominal%', 'C4 Low%', 'Icatibant%', 'Estrogen%', 'Mean Age Onset'].map(h => (
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
                        <td style={{ padding: '7px 10px', color: s.laryngeal_n / s.n >= 0.35 ? '#ff7b72' : '#e6edf3', fontWeight: s.laryngeal_n / s.n >= 0.35 ? 700 : 400 }}>{Math.round(s.laryngeal_n / s.n * 100)}%</td>
                        <td style={{ padding: '7px 10px', color: s.abdominal_n / s.n >= 0.60 ? '#f0b429' : '#e6edf3', fontWeight: s.abdominal_n / s.n >= 0.60 ? 700 : 400 }}>{Math.round(s.abdominal_n / s.n * 100)}%</td>
                        <td style={{ padding: '7px 10px', color: s.c4_low_n / s.n >= 0.80 ? '#ff7b72' : '#8b949e', fontWeight: s.c4_low_n / s.n >= 0.80 ? 700 : 400 }}>{Math.round(s.c4_low_n / s.n * 100)}%</td>
                        <td style={{ padding: '7px 10px', color: s.icatibant_resp_n / s.n >= 0.85 ? '#3fb950' : '#e6edf3', fontWeight: s.icatibant_resp_n / s.n >= 0.85 ? 700 : 400 }}>{Math.round(s.icatibant_resp_n / s.n * 100)}%</td>
                        <td style={{ padding: '7px 10px', color: s.estrogen_sensitive_n / s.n >= 0.80 ? '#6a1b9a' : '#e6edf3' }}>{Math.round(s.estrogen_sensitive_n / s.n * 100)}%</td>
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
                        <div><span style={{ color: '#8b949e' }}>Laryngeal: </span><span style={{ color: gs.laryngeal_pct >= 35 ? '#ff7b72' : '#e6edf3', fontWeight: gs.laryngeal_pct >= 35 ? 700 : 400 }}>{gs.laryngeal_pct}%</span></div>
                        <div><span style={{ color: '#8b949e' }}>Icatibant: </span><span style={{ color: gs.icatibant_resp_pct >= 85 ? '#3fb950' : '#e6edf3', fontWeight: gs.icatibant_resp_pct >= 85 ? 700 : 400 }}>{gs.icatibant_resp_pct}%</span></div>
                        <div><span style={{ color: '#8b949e' }}>Abdominal: </span><span style={{ color: '#e6edf3' }}>{gs.abdominal_pct}%</span></div>
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
