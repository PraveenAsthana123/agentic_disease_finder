'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hse-tlr3-encephalitis-susceptibility-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'TLR3':    '#b71c1c',  // deep red      — TLR3 most common ~10% sporadic HSE Casanova 2006
  'UNC93B1': '#1565c0',  // deep blue     — UNC93B1 all endosomal TLRs 3/7/8/9 lost
  'TICAM1':  '#2e7d32',  // dark green    — TICAM1/TRIF TLR3 adaptor bridge
  'TRAF3':   '#6a1b9a',  // deep purple   — TRAF3 IRF3/IRF7 bridge + myeloma risk
  'TBK1':    '#e65100',  // deep orange   — TBK1 dual HSE + ALS/FTD PATHOGNOMONIC
  'IRF3':    '#00695c',  // dark teal     — IRF3 IFN-β terminal transcription factor
  'IFNAR1':  '#4a148c',  // indigo        — IFNAR1 live vaccine CI PATHOGNOMONIC
  'DBR1':    '#f9a825',  // amber         — DBR1 brainstem/rhombencephalitis PATHOGNOMONIC
};

const GENE_INFO = {
  'TLR3':    { full: 'TLR3 / Toll-Like Receptor 3 / 904aa', locus: '4q35.1', size: '904 aa / 104 kDa (endosomal dsRNA sensor; recognizes HSV-1 dsRNA replication intermediates in neurons/astrocytes/pDCs; AD partial LOF; ~10% sporadic childhood HSE; temporal lobe predilection; poly(I:C)-inducible IFN-β reduced)', inh: 'AD (partial LOF)' },
  'UNC93B1': { full: 'UNC93B1 / UNC-93 Homolog B1 / 598aa', locus: '11q13.1', size: '598 aa / 65 kDa (12-pass ER transmembrane chaperone; escorts ALL endosomal TLRs 3/7/8/9 from ER → endolysosomes; AR complete LOF; BROADEST endosomal TLR defect; all 4 stimuli absent in PBMCs; HSE primary manifestation despite global endosomal TLR loss)', inh: 'AR (complete LOF)' },
  'TICAM1':  { full: 'TICAM1 / TRIF / TIR-Domain Adaptor Molecule 1 / 712aa', locus: '19p13.3', size: '712 aa / 79 kDa (SOLE adaptor for TLR3; TLR3→TRIF→TRAF3→TBK1→IRF3→IFN-β; also TLR4 MyD88-independent arm; AD partial LOF; HSE-specific; no bacterial susceptibility; phospho-IRF3 reduced on poly(I:C))', inh: 'AD (partial LOF)' },
  'TRAF3':   { full: 'TRAF3 / TNF Receptor-Associated Factor 3 / 568aa', locus: '14q32.32', size: '568 aa / 64 kDa (E3 ubiquitin ligase; bridges TRIF→TBK1 AND cGAS-STING→TBK1 AND RIG-I-MAVS→TBK1; convergence point for all innate antiviral sensors; AD partial LOF; germline het = HSE; somatic biallelic = multiple myeloma 15-20%; annual SPEP surveillance from age 45)', inh: 'AD (partial LOF)' },
  'TBK1':    { full: 'TBK1 / TANK-Binding Kinase 1 / 729aa', locus: '12q14.2', size: '729 aa / 84 kDa (master antiviral kinase; phosphorylates IRF3-Ser396+IRF7; also phosphorylates p62/SQSTM1+OPTN for selective autophagy; AD partial LOF; DUAL PHENOTYPE: childhood HSE + adult ALS/FTD PATHOGNOMONIC; TBK1-ALS ~1% familial ALS; family ALS = TBK1 first)', inh: 'AD (partial LOF)' },
  'IRF3':    { full: 'IRF3 / Interferon Regulatory Factor 3 / 427aa', locus: '19q13.33', size: '427 aa / 47 kDa (terminal TF of TLR3 cascade; TBK1 phosphorylates Ser396 → IRF3 dimerization → nuclear translocation → IFN-β transcription; constitutively expressed; AD partial LOF; phospho-IRF3 nuclear translocation reduced; broader viral susceptibility via cGAS/RIG-I convergence)', inh: 'AD (partial LOF)' },
  'IFNAR1':  { full: 'IFNAR1 / IFN-α/β Receptor Chain 1 / 557aa', locus: '21q22.11', size: '557 aa / 64 kDa (TYK2-associated IFN-α/β receptor chain 1; JAK1:IFNAR2 + TYK2:IFNAR1 → STAT1/STAT2 phosphorylation → ISGF3 → ISGs; AR complete LOF; IFN PRODUCED but cell CANNOT RESPOND; live vaccines ABSOLUTELY CI: YFV FATAL, MMR→measles encephalitis)', inh: 'AR (complete LOF)' },
  'DBR1':    { full: 'DBR1 / RNA Lariat Debranching Enzyme 1 / 431aa', locus: '2q24.2', size: '431 aa / 50 kDa (2\'-5\' RNA lariat debranching enzyme; cleaves spliceosomal lariat introns after splicing; AR complete LOF; RNA lariat ACCUMULATION; PATHOGNOMONIC: brainstem/RHOMBENCEPHALITIS not temporal lobe; CN palsies VI/VII/IX/X; respiratory failure 45%; temporal lobe SPARED)', inh: 'AR (complete LOF)' },
};

const PATHOGNOMONIC = {
  'TLR3':    'Temporal lobe HSE ~10% sporadic cases; poly(I:C)-inducible IFN-β REDUCED in fibroblasts; AD partial LOF',
  'UNC93B1': 'ALL 4 endosomal TLRs absent: poly(I:C)+R848+CpG → IFN ABSENT; TLR4 (LPS) = NORMAL — 4-stimulus PBMC panel pathognomonic',
  'TICAM1':  'TLR3 AND TLR4-late IFN-β both reduced (TRIF shared adaptor); TLR4-early NF-κB (MyD88) normal — distinguishes from IRAK4/MyD88',
  'TRAF3':   'IFN-β reduced to poly(I:C) AND cGAMP (STING) — broader than TLR3-only defects; germline = HSE; somatic biallelic = myeloma',
  'TBK1':    'Family ALS/FTD + HSE child = TBK1 FIRST — ONLY HSE gene causing adult-onset ALS/FTD (TBK1 autophagy + antiviral dual role)',
  'IRF3':    'Phospho-IRF3 (Ser396) REDUCED by Western blot; IRF3 nuclear translocation ABSENT by immunofluorescence after poly(I:C)',
  'IFNAR1':  'YFV live vaccine FATAL; MMR → measles vaccine encephalitis; ALL live vaccines absolutely contraindicated — unique in this atlas',
  'DBR1':    'BRAINSTEM/rhombencephalitis — temporal lobe SPARED; CN palsies VI/VII/IX/X; respiratory failure 45%; RNA lariat accumulation detectable',
};

export default function HereditaryHSETLR3Page() {
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

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#1a1a2e', minHeight: '100vh', fontSize: 18 }}>Loading Hereditary-HSE-TLR3-Encephalitis-Susceptibility-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#1a1a2e', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#e6edf3', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#b71c1c 0%,#4a148c 50%,#1565c0 100%)', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>Primary Immunodeficiency · TLR3 Pathway · HSE Susceptibility · Innate Antiviral Immunity</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
          🧬 Hereditary-HSE-TLR3-Encephalitis-Susceptibility-Atlas
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
          Complete 8-Gene Reference · TLR3 · UNC93B1 · TICAM1 · TRAF3 · TBK1 · IRF3 · IFNAR1 · DBR1 ·
          320 Patients (8×40) · Seeds 4070–4077
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Total Patients', val: overview?.total_patients },
            { label: 'Genes', val: overview?.genes?.length },
            { label: 'Acyclovir Urgency', val: '≤48h → 70% survival' },
            { label: 'DBR1 Location', val: 'Brainstem (not temporal)' },
            { label: 'IFNAR1 Vaccines', val: 'YFV + MMR CONTRAINDICATED' },
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
                    <div style={{ fontSize: 12, color: '#c9d1d9', marginBottom: 8 }}>{c.hse_type}</div>
                    <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 10 }}>{GENE_INFO[g]?.size}</div>
                    <div style={{ background: '#21262d', borderRadius: 6, padding: '8px', fontSize: 11, color: '#f0b429', marginBottom: 10 }}>
                      {PATHOGNOMONIC[g]}
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                      {[
                        ['Temporal Lobe HSE', c.hse_temporal_lobe_pct + '%'],
                        ['Acyclovir Response', c.acyclovir_response_pct + '%'],
                        ['Recurrence Risk', c.recurrence_pct + '%'],
                        ['Neuro Sequelae', c.neurological_sequelae_pct + '%'],
                        ['IFN-β Reduced', c.ifn_beta_reduced_pct + '%'],
                        ['Brainstem HSE', c.brainstem_pct + '%'],
                      ].map(([label, val]) => (
                        <div key={label} style={{ background: '#0d1117', borderRadius: 4, padding: '5px 8px' }}>
                          <div style={{ fontSize: 9, color: '#8b949e', textTransform: 'uppercase' }}>{label}</div>
                          <div style={{ fontSize: 14, fontWeight: 700, color: '#e6edf3' }}>{val}</div>
                        </div>
                      ))}
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
                  {['Gene', 'Locus', 'Size', 'Inh.', 'HSE Type', 'Temp Lobe%', 'Brainstem%', 'Recurrence%', 'Sequelae%', 'IFN-β Reduced%', 'Untreated Mortality%'].map(h => (
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
                      <td style={{ padding: '8px 12px', color: '#c9d1d9', fontSize: 11, maxWidth: 200 }}>{c.hse_type}</td>
                      <td style={{ padding: '8px 12px', color: c.hse_temporal_lobe_pct >= 80 ? '#3fb950' : c.hse_temporal_lobe_pct < 30 ? '#f0b429' : '#e6edf3', fontWeight: 600 }}>{c.hse_temporal_lobe_pct}%</td>
                      <td style={{ padding: '8px 12px', color: c.brainstem_pct >= 80 ? '#ff7b72' : '#e6edf3', fontWeight: c.brainstem_pct >= 80 ? 700 : 400 }}>{c.brainstem_pct}%</td>
                      <td style={{ padding: '8px 12px', color: '#e6edf3' }}>{c.recurrence_pct}%</td>
                      <td style={{ padding: '8px 12px', color: '#e6edf3' }}>{c.neurological_sequelae_pct}%</td>
                      <td style={{ padding: '8px 12px', color: c.ifn_beta_reduced_pct === 0 ? '#f0b429' : '#e6edf3', fontStyle: c.ifn_beta_reduced_pct === 0 ? 'italic' : 'normal' }}>{c.ifn_beta_reduced_pct === 0 ? 'PRODUCED (receptor defect)' : c.ifn_beta_reduced_pct + '%'}</td>
                      <td style={{ padding: '8px 12px', color: c.mortality_untreated_pct >= 75 ? '#ff7b72' : '#e6edf3', fontWeight: 600 }}>{c.mortality_untreated_pct}%</td>
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
              <div style={{ fontWeight: 700, color: '#58a6ff', marginBottom: 12, fontSize: 15 }}>Pathway Groups</div>
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
                    {['Gene', 'n', 'Temporal%', 'Brainstem%', 'Recurrence%', 'Sequelae%', 'IFN-β↓n', 'PLEDs EEG', 'CSF+n', 'VZV CNS', 'Live Vax', 'Resp Fail', 'CN Palsy', 'ALS Family', 'Mean Age', 'Days→Acyc', 'MRI Score'].map(h => (
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
                        <td style={{ padding: '7px 10px', color: s.temporal_lobe_n / s.n < 0.3 ? '#f0b429' : '#3fb950', fontWeight: 600 }}>{Math.round(s.temporal_lobe_n / s.n * 100)}%</td>
                        <td style={{ padding: '7px 10px', color: s.brainstem_n / s.n > 0.5 ? '#ff7b72' : '#e6edf3', fontWeight: s.brainstem_n / s.n > 0.5 ? 700 : 400 }}>{Math.round(s.brainstem_n / s.n * 100)}%</td>
                        <td style={{ padding: '7px 10px' }}>{Math.round(s.recurrence_n / s.n * 100)}%</td>
                        <td style={{ padding: '7px 10px' }}>{Math.round(s.sequelae_n / s.n * 100)}%</td>
                        <td style={{ padding: '7px 10px' }}>{s.ifn_reduced_n}</td>
                        <td style={{ padding: '7px 10px' }}>{s.pleds_n}</td>
                        <td style={{ padding: '7px 10px' }}>{s.csf_positive_n}</td>
                        <td style={{ padding: '7px 10px' }}>{s.vzv_cns_n}</td>
                        <td style={{ padding: '7px 10px', color: s.live_vax_enceph_n > 0 ? '#ff7b72' : '#e6edf3', fontWeight: s.live_vax_enceph_n > 0 ? 700 : 400 }}>{s.live_vax_enceph_n}</td>
                        <td style={{ padding: '7px 10px', color: s.resp_failure_n > 5 ? '#ff7b72' : '#e6edf3' }}>{s.resp_failure_n}</td>
                        <td style={{ padding: '7px 10px', color: s.cn_palsy_n > 10 ? '#ff7b72' : '#e6edf3' }}>{s.cn_palsy_n}</td>
                        <td style={{ padding: '7px 10px', color: s.als_family_n > 0 ? '#f0b429' : '#e6edf3' }}>{s.als_family_n}</td>
                        <td style={{ padding: '7px 10px' }}>{s.mean_age_onset}yr</td>
                        <td style={{ padding: '7px 10px', color: s.mean_days_to_acyclovir > 4 ? '#ff7b72' : '#3fb950' }}>{s.mean_days_to_acyclovir}d</td>
                        <td style={{ padding: '7px 10px' }}>{s.mean_mri_severity}/10</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Key distinguishers */}
            {breakdown && (
              <div style={{ marginTop: 24 }}>
                <div style={{ fontWeight: 700, color: '#58a6ff', marginBottom: 10, fontSize: 15 }}>Key Distinguishers</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 8 }}>
                  {Object.entries(breakdown.gene_summary || {}).map(([g, gs]) => (
                    <div key={g} style={{ background: '#161b22', borderRadius: 6, padding: '10px 14px', border: `1px solid ${GENE_COLORS[g] || '#30363d'}` }}>
                      <div style={{ fontWeight: 700, color: GENE_COLORS[g] || '#58a6ff', marginBottom: 6 }}>{g}</div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11 }}>
                        <div><span style={{ color: '#8b949e' }}>Temporal Lobe: </span><span style={{ color: '#e6edf3' }}>{gs.temporal_lobe_pct}%</span></div>
                        <div><span style={{ color: '#8b949e' }}>Brainstem: </span><span style={{ color: gs.brainstem_pct > 50 ? '#ff7b72' : '#e6edf3', fontWeight: gs.brainstem_pct > 50 ? 700 : 400 }}>{gs.brainstem_pct}%</span></div>
                        <div><span style={{ color: '#8b949e' }}>Recurrence: </span><span style={{ color: '#e6edf3' }}>{gs.recurrence_pct}%</span></div>
                        <div><span style={{ color: '#8b949e' }}>Sequelae: </span><span style={{ color: '#e6edf3' }}>{gs.sequelae_pct}%</span></div>
                        <div><span style={{ color: '#8b949e' }}>Mean Age: </span><span style={{ color: '#e6edf3' }}>{gs.mean_age_onset}yr</span></div>
                        <div><span style={{ color: '#8b949e' }}>MRI Score: </span><span style={{ color: '#e6edf3' }}>{gs.mean_mri_severity}/10</span></div>
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
