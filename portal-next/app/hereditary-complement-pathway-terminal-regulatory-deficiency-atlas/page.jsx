'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-complement-pathway-terminal-regulatory-deficiency-atlas';
const TABS = ['Overview', 'Gene Table', 'Complement Pathways', 'Definitions'];

const GENE_COLORS = {
  'C3':   '#1a237e',  // deep navy       — AR; central node ALL pathways; most severe
  'C1QB': '#6a1b9a',  // deep purple     — AR; C1q; strongest SLE predisposition
  'C2':   '#4a148c',  // dark purple     — AR; most common Caucasians; HLA-DR2
  'C5':   '#b71c1c',  // deep red        — AR; terminal; Neisseria; mandatory vaccine
  'C6':   '#c62828',  // red             — AR; terminal; African descent; adult onset
  'CFH':  '#e65100',  // deep orange     — AD/AR; Factor H; aHUS; eculizumab FDA
  'CFI':  '#f57f17',  // amber           — AR; Factor I; C3 hyperconversion; paradox
  'CFP':  '#1b5e20',  // dark green      — XL AR; Properdin; only X-linked; males fatal
};

const GENE_INFO = {
  'C3':   { full: 'C3 / Complement Component 3 / 1663aa',                      locus: '19p13.3',  size: '1663 aa / 185 kDa (central node ALL 3 pathways; opsonin C3b; anaphylatoxin C3a; AR LOF; ALL pathways fail; severe encapsulated bacteria + IC disease; CH50+AH50 absent; C3 absent)',                                         inh: 'AR LOF' },
  'C1QB': { full: 'C1QB / Complement C1q Subunit B / 253aa',                    locus: '1p36.11',  size: '253 aa / 27 kDa (C1q beta chain; 18-mer A/B/C heterotrimer bouquet; apoptotic cell clearance; classical pathway initiator; AR LOF; >90% SLE; CH50 absent + AH50 NORMAL)',                                                        inh: 'AR LOF' },
  'C2':   { full: 'C2 / Complement Component 2 / 752aa',                        locus: '6p21.32',  size: '752 aa / 102 kDa (CP/MBL C3 convertase serine protease; type I null allele 28-bp deletion; HLA-DR2 linked; MOST COMMON Caucasian complement deficiency; CH50 absent + AH50 NORMAL + C4 NORMAL)',                                  inh: 'AR LOF' },
  'C5':   { full: 'C5 / Complement Component 5 / 1676aa',                       locus: '9q33.2',   size: '1676 aa / 190 kDa (terminal pathway gateway; C5 convertase cleavage → C5b + C5a; MAC initiation; AR LOF; specifically Neisseria; CH50+AH50 absent; C3 NORMAL; MenACWY+MenB MANDATORY)',                                         inh: 'AR LOF' },
  'C6':   { full: 'C6 / Complement Component 6 / 913aa',                        locus: '5p13.1',   size: '913 aa / 105 kDa (MAC structural; C5b binds C6 → stable complex; terminal; AR LOF; Neisseria; African descent overrepresented; adult first presentation common; CH50+AH50 absent; C3/C5 NORMAL)',                              inh: 'AR LOF' },
  'CFH':  { full: 'CFH / Complement Factor H / 1231aa',                         locus: '1q31.3',   size: '1231 aa / 150 kDa (20 CCP domains; AP C3 convertase decay; host cell protection; AD/AR LOF; aHUS TMA; C3 low secondary; AH50 absent/reduced; CH50 normal; eculizumab FDA 2011; AMD Y402H polymorphism)',                     inh: 'AD/AR LOF' },
  'CFI':  { full: 'CFI / Complement Factor I / 583aa',                          locus: '4q25',     size: '583 aa / 88 kDa (serine protease; cleaves C3b → iC3b; requires CFH/MCP/CR1 cofactors; AR LOF; C3 hyperconversion → secondary C3 depletion; infections + autoimmunity PARADOX; C3 low; CH50+AH50 absent)',                   inh: 'AR LOF' },
  'CFP':  { full: 'CFP / Complement Factor P / Properdin / 442aa',              locus: 'Xp11.3',   size: '442 aa / 53 kDa (only positive complement regulator; stabilizes C3bBb AP convertase ×10 half-life; X-LINKED AR; ONLY X-LINKED complement deficiency; males: fulminant meningococcal; CH50 NORMAL + AH50 ABSENT PATHOGNOMONIC)', inh: 'XL AR' },
};

const PATHOGNOMONIC = {
  'C3':   'ALL pathways fail (CH50+AH50 absent + C3 absent); severe encapsulated bacteria (S.pneumo, H.flu, Neisseria) + immune complex disease (SLE-like, MPGN). Most severe complement deficiency. C3NeF autoAb → acquired C3 deficiency with partial lipodystrophy.',
  'C1QB': '>90% develop SLE-like disease — STRONGEST SLE PREDISPOSITION of any PID. C1q clears apoptotic cells → C1q LOF: nuclear antigen release → anti-dsDNA. CH50 ABSENT + AH50 NORMAL (AP intact). Test all 3 genes: C1QA/C1QB/C1QC.',
  'C2':   'MOST COMMON complement deficiency in Caucasians (carrier ~1/100). Type I null allele (28-bp deletion, HLA-DR2). SLE-like 50%; mild infections (S.pneumo, H.flu). CH50 absent + AH50 NORMAL + C4 NORMAL + C2 absent.',
  'C5':   'SPECIFICALLY Neisseria meningitidis + gonorrhoeae (terminal pathway defect; MAC is primary anti-Neisseria mechanism). CH50+AH50 absent + C3 NORMAL. Meningococcal vaccine MenACWY + MenB MANDATORY every 5yr. Penicillin V prophylaxis.',
  'C6':   'Terminal; Neisseria susceptibility. OVERREPRESENTED in African descent (South African founder p.Gln417Stop). CH50+AH50 absent + C3 NORMAL + C5 NORMAL (C6 absent distinguishes from C5 def). Adult first presentation common.',
  'CFH':  'aHUS (TMA: MAHA + thrombocytopenia + AKI) from uncontrolled AP on endothelium. CH50 NORMAL + AH50 absent/reduced + C3 LOW (secondary). ECULIZUMAB FDA 2011; RAVULIZUMAB FDA 2019. MenACWY+MenB+penicillin mandatory while on eculizumab.',
  'CFI':  'PARADOX: BOTH infections (secondary C3 depletion) AND autoimmunity (SLE-like, anti-dsDNA). C3 LOW (secondary hyperconversion — gene normal). Diagnose by CFI antigen + C3b cleavage assay. FFP breaks amplification loop in acute crisis.',
  'CFP':  'ONLY X-LINKED complement deficiency. CH50 NORMAL + AH50 ABSENT = PATHOGNOMONIC (only complement deficiency with normal CH50 and absent AH50). C3 NORMAL. Males: fulminant meningococcal (Waterhouse-Friderichsen). Vaccinate ALL male relatives BEFORE first episode.',
};

export default function HereditaryComplementPathwayAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#0d1b2a', minHeight: '100vh', fontSize: 18 }}>Loading Hereditary-Complement-Pathway-Terminal-Regulatory-Deficiency-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#0d1b2a', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#e6edf3', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#6a1b9a 30%,#b71c1c 60%,#1b5e20 100%)', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>Hereditary Complement Pathway · Terminal Complement · Alternative Pathway Regulation · Primary Immunodeficiency</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
          🧬 Hereditary-Complement-Pathway-Terminal-Regulatory-Deficiency-Atlas
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
          Complete 8-Gene Reference · C3 · C1QB · C2 · C5 · C6 · CFH · CFI · CFP ·
          320 Patients (8×40) · Seeds 4110–4117
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Total Patients', val: overview?.total_patients },
            { label: 'Genes', val: overview?.genes?.length },
            { label: 'Most Common (Caucasian)', val: 'C2 deficiency' },
            { label: 'Strongest SLE link', val: 'C1q (C1QB)' },
            { label: 'Only X-linked', val: 'Properdin (CFP)' },
          ].map(({ label, val }) => (
            <div key={label} style={{ background: 'rgba(255,255,255,0.15)', borderRadius: 8, padding: '6px 14px' }}>
              <div style={{ fontSize: 11, opacity: 0.8 }}>{label}</div>
              <div style={{ fontWeight: 700, fontSize: 15 }}>{val ?? '…'}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#161b22', borderBottom: '1px solid #30363d', display: 'flex', gap: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            style={{ padding: '10px 20px', background: activeTab === t ? '#1f2d3d' : 'transparent',
              color: activeTab === t ? '#58a6ff' : '#8b949e', border: 'none', borderBottom: activeTab === t ? '2px solid #58a6ff' : '2px solid transparent',
              cursor: 'pointer', fontSize: 14 }}>
            {t}
          </button>
        ))}
      </div>

      <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>

        {/* OVERVIEW TAB */}
        {activeTab === 'Overview' && overview && (
          <div>
            {/* Key facts */}
            <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 10, padding: 20, marginBottom: 24 }}>
              <h3 style={{ margin: '0 0 14px', color: '#58a6ff', fontSize: 16 }}>⚡ Key Clinical Pearls — Complement Pathway Deficiencies</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(340px,1fr))', gap: 12 }}>
                {Object.entries(overview.key_facts || {}).map(([k, v]) => (
                  <div key={k} style={{ background: '#0d1117', border: '1px solid #21262d', borderRadius: 8, padding: '10px 14px' }}>
                    <div style={{ fontSize: 12, color: '#f0883e', fontWeight: 600, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.5 }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 13, color: '#cdd9e5', lineHeight: 1.5 }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Gene cards */}
            <h3 style={{ color: '#58a6ff', fontSize: 16, marginBottom: 16 }}>Gene-Specific Cohort Summary (8 Genes × 40 Patients)</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(310px,1fr))', gap: 16 }}>
              {(overview.genes || []).map(gene => {
                const cs = overview.cohort_summary?.[gene];
                const info = GENE_INFO[gene];
                const color = GENE_COLORS[gene] || '#444';
                return (
                  <div key={gene} style={{ background: '#161b22', border: `1px solid ${color}44`, borderRadius: 10, overflow: 'hidden' }}>
                    <div style={{ background: color, padding: '10px 14px' }}>
                      <div style={{ fontWeight: 700, fontSize: 15, color: '#fff' }}>{gene}</div>
                      <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.8)', marginTop: 2 }}>{info?.inh} · {info?.locus}</div>
                    </div>
                    <div style={{ padding: '12px 14px' }}>
                      <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 8 }}>{info?.full}</div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
                        {[
                          ['Infections', `${cs?.infect_pct}%`],
                          ['SLE-like', `${cs?.sle_pct}%`],
                          ['Neisseria', cs?.neisseria_suscept ? `${cs?.neisseria_pct}%` : 'Low'],
                          ['aHUS', cs?.aHUS_risk ? `${cs?.aHUS_pct}%` : '—'],
                          ['Eculizumab', cs?.eculizumab ? '✅ FDA' : '—'],
                          ['X-Linked', cs?.x_linked ? '✅ Males' : '—'],
                        ].map(([lbl, val]) => (
                          <div key={lbl} style={{ background: '#0d1117', borderRadius: 6, padding: '5px 8px' }}>
                            <div style={{ fontSize: 10, color: '#8b949e' }}>{lbl}</div>
                            <div style={{ fontSize: 13, fontWeight: 600, color: String(val).includes('✅') ? '#3fb950' : '#e6edf3' }}>{val}</div>
                          </div>
                        ))}
                      </div>
                      <div style={{ background: '#0d1117', borderRadius: 6, padding: '8px 10px', fontSize: 12, color: '#e6edf3', lineHeight: 1.5, borderLeft: `3px solid ${color}` }}>
                        {PATHOGNOMONIC[gene]}
                      </div>
                      <div style={{ marginTop: 8, fontSize: 11, color: '#8b949e' }}>
                        Mean onset: <b>{cs?.mean_age_onset} yr</b> · Meningococcal vax: <b>{cs?.meningococcal_vaccine ? '✅ MANDATORY' : 'Not primary'}</b>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {activeTab === 'Gene Table' && overview && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#161b22' }}>
                  {['Gene', 'Size', 'Locus', 'Inheritance', 'Syndrome', 'Infect', 'SLE-like', 'Neisseria', 'aHUS', 'Eculizumab', 'X-Linked', 'Mean Onset'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', color: '#8b949e', fontWeight: 600, fontSize: 12, borderBottom: '1px solid #30363d', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.genes || []).map((gene, i) => {
                  const cs = overview.cohort_summary?.[gene];
                  const color = GENE_COLORS[gene] || '#444';
                  return (
                    <tr key={gene} style={{ background: i % 2 === 0 ? '#0d1117' : '#161b22', borderBottom: '1px solid #21262d' }}>
                      <td style={{ padding: '9px 12px', fontWeight: 700, color: color }}>{gene}</td>
                      <td style={{ padding: '9px 12px', color: '#8b949e', fontSize: 12 }}>{GENE_INFO[gene]?.size?.split('/')[0]?.trim()}</td>
                      <td style={{ padding: '9px 12px', color: '#8b949e', fontSize: 12 }}>{cs?.locus}</td>
                      <td style={{ padding: '9px 12px', color: '#e6edf3', fontSize: 12 }}>{cs?.inheritance}</td>
                      <td style={{ padding: '9px 12px', color: '#cdd9e5', fontSize: 12, maxWidth: 200 }}>{cs?.syndrome?.split('-').slice(0,5).join('-')}</td>
                      <td style={{ padding: '9px 12px', color: '#3fb950', fontWeight: 600 }}>{cs?.infect_pct}%</td>
                      <td style={{ padding: '9px 12px', color: cs?.sle_pct > 40 ? '#f0883e' : '#8b949e' }}>{cs?.sle_pct}%</td>
                      <td style={{ padding: '9px 12px', color: cs?.neisseria_suscept ? '#f85149' : '#8b949e', fontWeight: cs?.neisseria_suscept ? 700 : 400 }}>{cs?.neisseria_suscept ? `✅ ${cs?.neisseria_pct}%` : '—'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.aHUS_risk ? '#f0883e' : '#8b949e', fontWeight: cs?.aHUS_risk ? 700 : 400 }}>{cs?.aHUS_risk ? `⚠ ${cs?.aHUS_pct}%` : '—'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.eculizumab ? '#3fb950' : '#8b949e', fontWeight: cs?.eculizumab ? 700 : 400 }}>{cs?.eculizumab ? '✅ FDA' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.x_linked ? '#58a6ff' : '#8b949e', fontWeight: cs?.x_linked ? 700 : 400 }}>{cs?.x_linked ? '✅ XL' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: '#e6edf3' }}>{cs?.mean_age_onset} yr</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div style={{ marginTop: 20 }}>
              <h3 style={{ color: '#58a6ff', fontSize: 15, marginBottom: 12 }}>Gene Detail — Protein & Pathway Reference</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(320px,1fr))', gap: 12 }}>
                {Object.entries(GENE_INFO).map(([gene, info]) => (
                  <div key={gene} style={{ background: '#161b22', border: `1px solid ${GENE_COLORS[gene]}44`, borderRadius: 8, padding: '12px 14px' }}>
                    <div style={{ fontWeight: 700, color: GENE_COLORS[gene], marginBottom: 4 }}>{gene}</div>
                    <div style={{ fontSize: 12, color: '#cdd9e5', lineHeight: 1.5 }}>{info.size}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* COMPLEMENT PATHWAYS TAB */}
        {activeTab === 'Complement Pathways' && breakdown && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(380px,1fr))', gap: 20, marginBottom: 28 }}>
              {Object.entries(breakdown.pathway_groups || {}).map(([groupKey, group]) => (
                <div key={groupKey} style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 10, padding: 18 }}>
                  <div style={{ fontWeight: 700, color: '#58a6ff', fontSize: 15, marginBottom: 8, textTransform: 'capitalize' }}>{groupKey.replace(/_/g, ' ')}</div>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
                    {(group.genes || []).map(g => (
                      <span key={g} style={{ background: GENE_COLORS[g] || '#333', color: '#fff', borderRadius: 5, padding: '2px 8px', fontSize: 12, fontWeight: 700 }}>{g}</span>
                    ))}
                  </div>
                  <div style={{ fontSize: 13, color: '#cdd9e5', lineHeight: 1.6, marginBottom: 10 }}>{group.description}</div>
                  <div style={{ background: '#0d1117', borderRadius: 6, padding: '8px 10px', fontSize: 12, color: '#f0883e' }}>
                    <b>Key Test:</b> {group.key_test}
                  </div>
                </div>
              ))}
            </div>
            <h3 style={{ color: '#58a6ff', fontSize: 15, marginBottom: 12 }}>Gene Summary Stats</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 12 }}>
              {Object.entries(breakdown.gene_summary || {}).map(([gene, gs]) => (
                <div key={gene} style={{ background: '#161b22', border: `1px solid ${GENE_COLORS[gene]}44`, borderRadius: 8, padding: 12 }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[gene], fontSize: 14, marginBottom: 6 }}>{gene}</div>
                  <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 4 }}>n={gs.n} · Seed {gs.seed}</div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Infections: <b>{gs.infect_pct}%</b></div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>SLE-like: <b>{gs.sle_pct}%</b></div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Neisseria: <b>{gs.neisseria_pct}%</b></div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Mean onset: <b>{gs.mean_age_onset} yr</b></div>
                  <div style={{ fontSize: 12, marginTop: 4, color: gs.eculizumab ? '#3fb950' : '#8b949e' }}>Eculizumab: <b>{gs.eculizumab ? '✅ FDA' : '—'}</b></div>
                  <div style={{ fontSize: 12, color: gs.x_linked ? '#58a6ff' : '#8b949e' }}>X-Linked: <b>{gs.x_linked ? '✅ Males' : '—'}</b></div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {activeTab === 'Definitions' && definitions && (
          <div>
            <div style={{ display: 'grid', gap: 12 }}>
              {(definitions.definitions || []).map((def, i) => (
                <div key={i} style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 10, overflow: 'hidden' }}>
                  <button onClick={() => setExpandedDef(expandedDef === i ? null : i)}
                    style={{ width: '100%', textAlign: 'left', padding: '14px 18px', background: 'transparent', border: 'none', cursor: 'pointer',
                      color: '#58a6ff', fontSize: 14, fontWeight: 600, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>{def.term}</span>
                    <span style={{ color: '#8b949e', fontSize: 18, marginLeft: 12 }}>{expandedDef === i ? '▲' : '▼'}</span>
                  </button>
                  {expandedDef === i && (
                    <div style={{ padding: '0 18px 16px', fontSize: 13, color: '#cdd9e5', lineHeight: 1.75, borderTop: '1px solid #21262d' }}>
                      <div style={{ marginTop: 12 }}>{def.definition}</div>
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div style={{ marginTop: 28 }}>
              <h3 style={{ color: '#58a6ff', fontSize: 15, marginBottom: 12 }}>Key Distinguishers</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(330px,1fr))', gap: 10 }}>
                {Object.entries(definitions.key_distinguishers || {}).map(([k, v]) => (
                  <div key={k} style={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, padding: '10px 14px' }}>
                    <div style={{ fontSize: 11, color: '#f0883e', fontWeight: 700, marginBottom: 4, textTransform: 'uppercase' }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 13, color: '#cdd9e5', lineHeight: 1.5 }}>{v}</div>
                  </div>
                ))}
              </div>
              <h3 style={{ color: '#58a6ff', fontSize: 15, margin: '24px 0 12px' }}>Treatment Matrix</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))', gap: 10 }}>
                {Object.entries(definitions.treatment_matrix || {}).map(([k, v]) => (
                  <div key={k} style={{ background: '#0d1117', border: '1px solid #30363d', borderRadius: 8, padding: '10px 14px' }}>
                    <div style={{ fontSize: 12, color: '#3fb950', fontWeight: 700, marginBottom: 4 }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                    <div style={{ fontSize: 12, color: '#cdd9e5', lineHeight: 1.5 }}>{v}</div>
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
