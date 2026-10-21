'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-pku-biopterin-deficiency-atlas';
const TABS = ['Overview', 'Gene Table', 'BH4 & Neurotransmitter Groups', 'Definitions'];

const GENE_COLORS = {
  'PAH':     '#1a237e',  // deep navy        — AR; Classic PKU; Phe elevated NBS; R408W; sapropterin
  'PTS':     '#b71c1c',  // deep red         — AR; PTPS; most common BH4D; triple therapy MANDATORY
  'GCH1':    '#4a148c',  // deep violet      — AR+AD; Segawa DRD1; diurnal variation; female 3x
  'QDPR':    '#880e4f',  // dark magenta     — AR; DHPR; brain calcification; folinic acid
  'PCBD1':   '#2e7d32',  // dark green       — AR; primapterinuria; BENIGN transient
  'SPR':     '#e65100',  // deep orange      — AR; NBS MISSES; CSF sepiapterin; DRD
  'DNAJC12': '#00695c',  // dark teal        — AR; HSP40 chaperone; pterins NORMAL
  'TH':      '#1565c0',  // deep blue        — AR; oculogyric crises; L-DOPA CURATIVE; 5-HIAA normal
};

const GENE_INFO = {
  'PAH':     { full: 'PAH / Phenylalanine Hydroxylase / Phenylketonuria (PKU) / 452aa', locus: '12q23.2', inh: 'AR LOF',
               note: '452aa/52kDa; BH4-dependent Phe→Tyr; LOF→Phe accumulates→brain damage; R408W most common N-European (30-40%); NBS detects all variants; sapropterin (Kuvan/BH4) responsive in 30-50% (missense alleles); diet + BH4; maternal PKU: Phe>360 in pregnancy → embryopathy (microcephaly+CHD); neuropsychiatric despite treatment; first ever NBS disease (Guthrie 1963)' },
  'PTS':     { full: 'PTS / 6-Pyruvoyltetrahydropterin Synthase / PTPS Deficiency / 145aa', locus: '11q23.1', inh: 'AR LOF',
               note: '145aa/17kDa; most common BH4 deficiency (>60%); neopterin VERY HIGH + biopterin VERY LOW = PATHOGNOMONIC; NBS: Phe elevated; DHPR normal; TRIPLE THERAPY ALL MANDATORY: BH4 + L-DOPA/carbidopa + 5-HTP; Phe-diet ALONE = FATAL (missing neurotransmitter replacement); CSF monitoring essential' },
  'GCH1':    { full: 'GCH1 / GTP Cyclohydrolase I / Segawa DRD1 (AD) or Severe BH4D (AR) / 250aa', locus: '14q22.2', inh: 'AD (Segawa) / AR (severe)',
               note: '250aa/28kDa; SAME GENE — TWO COMPLETELY DIFFERENT DISEASES by inheritance; AD LOF = Segawa DRD1: diurnal variation dystonia, DRAMATICALLY L-DOPA responsive, NBS NORMAL, female 3:1 ratio (imprinting); AR LOF = severe BH4 deficiency, all pterins absent, Phe elevated; Segawa: small L-DOPA doses = essentially CURATIVE; diurnal variation = worse afternoon/evening, better morning' },
  'QDPR':    { full: 'QDPR / Dihydropteridine Reductase / DHPR Deficiency / 244aa', locus: '4p15.32', inh: 'AR LOF',
               note: '244aa/26kDa; BH4 regeneration failure; Phe may normalize on diet BUT NEURODEGENERATION CONTINUES; BRAIN CALCIFICATION (basal ganglia) = PATHOGNOMONIC; DHPR enzyme absent on DBS = PATHOGNOMONIC; FOLINIC ACID MANDATORY (cerebral folate deficiency); BH4+L-DOPA+5-HTP+folinic acid ALL needed; biopterin HIGH (accumulates); Middle Eastern/Mediterranean enrichment' },
  'PCBD1':   { full: 'PCBD1 / Pterin-4α-Carbinolamine Dehydratase / Primapterinuria / 104aa', locus: '10q22.1', inh: 'AR LOF',
               note: '104aa/12kDa; BENIGN — transient hyperphenylalaninemia; 7-biopterin (PRIMAPTERIN) HIGH = PATHOGNOMONIC; DHPR normal; resolves spontaneously 6-12 months; NO TREATMENT; ~30 families; critical to identify to AVOID unnecessary BH4/L-DOPA/5-HTP; key DDx from PTPS (neopterin HIGH, no primapterin)' },
  'SPR':     { full: 'SPR / Sepiapterin Reductase / Sepiapterin Reductase Deficiency / 261aa', locus: '2p14', inh: 'AR LOF',
               note: '261aa/29kDa; NBS MISSES (Phe NORMAL) — diagnosed LATE; CSF: sepiapterin VERY HIGH + HVA LOW + 5-HIAA LOW = PATHOGNOMONIC; blood/urine pterins not helpful (only CSF); diurnal variation DRD + hypotonia + ID; L-DOPA+5-HTP DRAMATIC response; often misdiagnosed as CP; average dx delay 5-10 years; <100 cases' },
  'DNAJC12': { full: 'DNAJC12 / DnaJ HSP40 Chaperone / DNAJC12 Deficiency / 198aa', locus: '10q21.3', inh: 'AR LOF',
               note: '198aa/22kDa; HSP40 chaperone required for PAH+TH+TPH stability; Phe elevated (NBS detects); PTERINS NORMAL + BH4 LOADING NORMAL = key DDx from BH4 deficiencies; often misclassified as BH4-non-responsive PAH; progressive neurodegeneration; gene sequencing when Phe elevated + pterins normal + BH4 normal; recently described 2017; <50 cases' },
  'TH':      { full: 'TH / Tyrosine Hydroxylase / Tyrosine Hydroxylase Deficiency / 528aa', locus: '11p15.5', inh: 'AR LOF',
               note: '528aa/60kDa; rate-limiting catecholamine synthesis step; NBS MISSES (Phe NORMAL); OCULOGYRIC CRISES = highly characteristic; CSF: HVA very low + MHPG very low + 5-HIAA NORMAL (key DDx: serotonin intact); L-DOPA/carbidopa ESSENTIALLY CURATIVE within days; TYPE A severe neonatal (Gln381Lys) vs TYPE B milder DRD-like (Glu328Lys); DO NOT delay L-DOPA trial' },
};

const PATHOGNOMONIC = {
  'PAH':     'PHENYLKETONURIA — Elevated Phe + elevated Phe/Tyr ratio on NBS. FIRST disease identified by NBS (Guthrie 1963). R408W = most common N-European allele (30-40%). SAPROPTERIN (BH4/Kuvan): 30-50% of patients with missense alleles respond (≥30% Phe reduction at 24h). Neuropsychiatric sequelae (anxiety, ADHD) occur despite treatment. MATERNAL PKU: Phe >360 µmol/L in pregnancy → microcephaly + CHD in fetus — diet must be strict BEFORE conception.',
  'PTS':     'PTPS DEFICIENCY — PATHOGNOMONIC PTERIN PROFILE: neopterin VERY HIGH + biopterin VERY LOW in blood/urine + DHPR activity NORMAL. CRITICAL: PTERIN PROFILE + DHPR ASSAY MANDATORY on ALL NBS Phe elevations — Phe dietary restriction ALONE = FATAL (neurotransmitters not replaced). TRIPLE THERAPY: BH4 (reduces Phe) + L-DOPA/carbidopa (dopamine) + 5-HTP (serotonin) — ALL THREE simultaneously.',
  'GCH1':    'GCH1 — UNIQUE BIOLOGY: SAME GENE, OPPOSITE INHERITANCE, COMPLETELY DIFFERENT DISEASES. AD LOF (Segawa DRD1): DIURNAL VARIATION (worst afternoon/evening, best morning = PATHOGNOMONIC for DRD) + DRAMATIC L-DOPA RESPONSE (1-3 mg/kg) + FEMALE PREDOMINANCE 3:1 (imprinting). NBS NORMAL (Phe normal in heterozygous). Any child with gait abnormality + diurnal variation → L-DOPA TRIAL IS BOTH DIAGNOSTIC AND CURATIVE.',
  'QDPR':    'DHPR DEFICIENCY — DHPR ENZYME ABSENT on dried blood spot = PATHOGNOMONIC. BRAIN CALCIFICATION (basal ganglia + white matter) on CT/MRI = PATHOGNOMONIC complication. CRITICAL TRAP: Phe may normalize on dietary restriction giving FALSE REASSURANCE but neurodegeneration and calcification CONTINUE without neurotransmitter replacement + FOLINIC ACID. Biopterin accumulates (HIGH). FOLINIC ACID (5-formylTHF) is MANDATORY — regular folic acid does NOT cross blood-brain barrier.',
  'PCBD1':   'PRIMAPTERINURIA — PATHOGNOMONIC: 7-biopterin (primapterin) HIGH in urine. BENIGN AND TRANSIENT — resolves spontaneously by 6-12 months. DHPR activity NORMAL. NO TREATMENT REQUIRED. Critical to identify and communicate benign nature to prevent unnecessary BH4/L-DOPA/5-HTP prescription. Only ~30 families described worldwide.',
  'SPR':     'SEPIAPTERIN REDUCTASE DEFICIENCY — NBS COMPLETELY NORMAL (Phe is NORMAL). DIAGNOSED LATE (average 5-10 year delay). PATHOGNOMONIC: CSF sepiapterin VERY HIGH — ONLY in CSF, not blood/urine. CSF HVA LOW + 5-HIAA LOW. Diurnal variation dystonia + hypotonia + ID, often misdiagnosed as cerebral palsy for years. L-DOPA + 5-HTP = DRAMATIC RESPONSE. DDx from TH: 5-HIAA is LOW in SPR (vs NORMAL in TH).',
  'DNAJC12': 'DNAJC12 DEFICIENCY — The DIAGNOSTIC TRAP: Phe elevated (NBS detects) BUT pterins are NORMAL AND BH4-loading test is NORMAL — distinguishes from ALL known BH4 deficiencies. HSP40 chaperone for PAH, TH, TPH. Progressive neurodegeneration despite apparent BH4 non-responsiveness. Misclassified as "atypical PAH" or "BH4-non-responsive HPA." DNAJC12 gene sequencing is required when Phe elevated + pterins normal + BH4 load normal.',
  'TH':      'TYROSINE HYDROXYLASE DEFICIENCY — NBS COMPLETELY NORMAL (Phe is NORMAL). OCULOGYRIC CRISES (sustained forced upward eye deviation) = HIGHLY CHARACTERISTIC. CSF: HVA VERY LOW + MHPG VERY LOW + 5-HIAA NORMAL (serotonin intact — key DDx from PTPS/SPR where 5-HIAA also low). L-DOPA/carbidopa ESSENTIALLY CURATIVE — improvement within DAYS. DO NOT wait for molecular confirmation to start L-DOPA trial.',
};

export default function HereditaryPKUBiopterinAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#0d1b2a', minHeight: '100vh', fontSize: 18 }}>Loading Hereditary-PKU-Biopterin-Deficiency-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#0d1b2a', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#e6edf3', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#b71c1c 20%,#4a148c 40%,#880e4f 60%,#e65100 80%,#1565c0 100%)', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>Phenylketonuria · BH4 Deficiency · Neurotransmitter Synthesis · Newborn Screening · DOPA-Responsive Dystonia</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
          🧬 Hereditary-PKU-Biopterin-Deficiency-Atlas
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
          Complete 8-Gene Reference · PAH · PTS · GCH1 · QDPR · PCBD1 · SPR · DNAJC12 · TH ·
          320 Patients (8×40) · Seeds 4142–4149
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Total Patients', val: overview?.total_patients },
            { label: 'Genes', val: overview?.genes?.length },
            { label: 'Most Common BH4D', val: 'PTS (>60%)' },
            { label: 'NBS Misses', val: 'SPR, TH, GCH1-AD' },
            { label: 'DHPR Folinic Acid', val: 'QDPR MANDATORY' },
          ].map(({ label, val }) => (
            <div key={label} style={{ background: 'rgba(0,0,0,0.3)', borderRadius: 8, padding: '6px 14px', fontSize: 13 }}>
              <span style={{ color: 'rgba(255,255,255,0.6)' }}>{label}: </span>
              <span style={{ color: '#fff', fontWeight: 600 }}>{val ?? '…'}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Tab Bar */}
      <div style={{ background: '#161b22', borderBottom: '1px solid #30363d', display: 'flex', gap: 0, overflowX: 'auto' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)} style={{
            padding: '12px 20px', border: 'none', background: 'none', cursor: 'pointer',
            color: activeTab === t ? '#58a6ff' : '#8b949e',
            borderBottom: activeTab === t ? '2px solid #58a6ff' : '2px solid transparent',
            fontWeight: activeTab === t ? 600 : 400, fontSize: 14, whiteSpace: 'nowrap',
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>

        {/* ── OVERVIEW TAB ── */}
        {activeTab === 'Overview' && overview && (
          <div>
            <div style={{ marginBottom: 24, padding: 20, background: '#161b22', borderRadius: 10, border: '1px solid #30363d' }}>
              <h2 style={{ margin: '0 0 8px', fontSize: 17, color: '#58a6ff' }}>Atlas Summary</h2>
              <p style={{ margin: 0, color: '#8b949e', fontSize: 14, lineHeight: 1.6 }}>
                {overview.subtitle}
              </p>
            </div>

            {/* Gene Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 16, marginBottom: 28 }}>
              {GENES.map(g => {
                const s = overview.cohort_summary?.[g];
                if (!s) return null;
                return (
                  <div key={g} style={{ background: '#161b22', border: `1px solid ${GENE_COLORS[g]}44`, borderRadius: 10, padding: 16, borderLeft: `4px solid ${GENE_COLORS[g]}` }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                      <div>
                        <span style={{ fontWeight: 700, fontSize: 16, color: GENE_COLORS[g] }}>{g}</span>
                        <span style={{ marginLeft: 8, fontSize: 12, color: '#8b949e' }}>{s.locus} · {s.size_aa}aa · {s.inheritance}</span>
                      </div>
                      <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: s.benign_variant ? '#1b4332' : '#1c1917', color: s.benign_variant ? '#6ee7b7' : '#fca5a5', border: `1px solid ${s.benign_variant ? '#064e3b' : '#7f1d1d'}` }}>
                        {s.benign_variant ? 'BENIGN' : 'PATHOGENIC'}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 10, lineHeight: 1.5 }}>
                      {GENE_INFO[g]?.note?.slice(0, 160)}…
                    </div>

                    {/* Key stats */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                      {[
                        { label: 'NBS Detected', val: `${s.nbs_detected_pct}%`, flag: s.nbs_detected_flag },
                        { label: 'Hyperphe', val: `${s.hyperphe_pct}%`, flag: s.hyperphe_flag },
                        { label: 'Neurodegeneration', val: `${s.neurodegeneration_pct}%`, flag: s.neurodegeneration_flag },
                        { label: 'DOPA-Resp Dystonia', val: `${s.dopa_resp_pct}%`, flag: s.dopa_resp_flag },
                        { label: 'Brain Calcification', val: `${s.brain_calc_pct}%`, flag: s.brain_calc_flag },
                        { label: 'BH4 Responsive', val: `${s.bh4_resp_pct}%`, flag: s.bh4_responsive_flag },
                        { label: 'NT Replacement', val: `${s.nt_replace_pct}%`, flag: s.nt_replacement_flag },
                        { label: 'Folinic Acid', val: `${s.folinic_acid_pct}%`, flag: s.folinic_acid_flag },
                      ].map(({ label, val, flag }) => (
                        <div key={label} style={{ background: '#0d1117', borderRadius: 6, padding: '5px 8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: 11, color: '#8b949e' }}>{label}</span>
                          <span style={{ fontSize: 12, fontWeight: 600, color: flag ? '#58a6ff' : '#6e7681' }}>{val}</span>
                        </div>
                      ))}
                    </div>

                    {/* NBS flag */}
                    {!s.nbs_detected_flag && (
                      <div style={{ marginTop: 8, padding: '4px 10px', background: '#7c2d12', borderRadius: 6, fontSize: 11, color: '#fed7aa' }}>
                        ⚠ NBS MISSES this gene (Phe normal or AD variant)
                      </div>
                    )}
                    {s.folinic_acid_flag && (
                      <div style={{ marginTop: 6, padding: '4px 10px', background: '#1e3a5f', borderRadius: 6, fontSize: 11, color: '#93c5fd' }}>
                        💊 FOLINIC ACID MANDATORY (cerebral folate deficiency)
                      </div>
                    )}
                    {s.nt_replacement_flag && (
                      <div style={{ marginTop: 6, padding: '4px 10px', background: '#3b1f00', borderRadius: 6, fontSize: 11, color: '#fde68a' }}>
                        🔬 Neurotransmitter replacement required (diet alone insufficient)
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Key Facts */}
            <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 10, padding: 20 }}>
              <h3 style={{ margin: '0 0 16px', fontSize: 16, color: '#e6edf3' }}>Key Clinical Facts</h3>
              <div style={{ display: 'grid', gap: 12 }}>
                {Object.entries(overview.key_facts || {}).map(([k, v]) => (
                  <div key={k} style={{ padding: 14, background: '#0d1117', borderRadius: 8, borderLeft: '3px solid #58a6ff' }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#58a6ff', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      {k.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontSize: 13, color: '#c9d1d9', lineHeight: 1.6 }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── GENE TABLE TAB ── */}
        {activeTab === 'Gene Table' && overview && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#161b22', borderBottom: '2px solid #30363d' }}>
                  {['Gene', 'Disease', 'aa/kDa', 'Locus', 'Inh', 'NBS', 'Phe↑', 'Neurodegeneration', 'DOPA-DRD', 'BH4-Resp', 'NT Replace', 'Folinic', 'Brain Calc', 'Benign', 'Key Mutation'].map(h => (
                    <th key={h} style={{ padding: '10px 10px', textAlign: 'left', color: '#58a6ff', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {GENES.map((g, idx) => {
                  const s = overview.cohort_summary?.[g];
                  if (!s) return null;
                  const info = GENE_INFO[g];
                  return (
                    <tr key={g} style={{ borderBottom: '1px solid #21262d', background: idx % 2 === 0 ? '#0d1117' : '#161b22' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g] }}>{g}</td>
                      <td style={{ padding: '8px 10px', color: '#c9d1d9', maxWidth: 180, fontSize: 12 }}>{s.syndrome?.split('-').slice(0,4).join(' ')}</td>
                      <td style={{ padding: '8px 10px', color: '#8b949e', whiteSpace: 'nowrap' }}>{s.size_aa}aa</td>
                      <td style={{ padding: '8px 10px', color: '#8b949e', fontFamily: 'monospace' }}>{s.locus}</td>
                      <td style={{ padding: '8px 10px', color: '#8b949e' }}>{s.inheritance}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><span style={{ color: s.nbs_detected_flag ? '#3fb950' : '#f85149', fontWeight: 700 }}>{s.nbs_detected_flag ? '✓' : '✗'}</span></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><span style={{ color: s.hyperphe_flag ? '#3fb950' : '#6e7681', fontWeight: 700 }}>{s.hyperphe_flag ? '✓' : '—'}</span></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><span style={{ color: s.neurodegeneration_flag ? '#f85149' : '#6e7681', fontWeight: 700 }}>{s.neurodegeneration_flag ? '✓' : '—'}</span></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><span style={{ color: s.dopa_resp_flag ? '#e3b341' : '#6e7681', fontWeight: 700 }}>{s.dopa_resp_flag ? '✓' : '—'}</span></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><span style={{ color: s.bh4_responsive_flag ? '#3fb950' : '#6e7681', fontWeight: 700 }}>{s.bh4_responsive_flag ? '✓' : '—'}</span></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><span style={{ color: s.nt_replacement_flag ? '#f85149' : '#6e7681', fontWeight: 700 }}>{s.nt_replacement_flag ? '✓' : '—'}</span></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><span style={{ color: s.folinic_acid_flag ? '#f85149' : '#6e7681', fontWeight: 700 }}>{s.folinic_acid_flag ? '✓' : '—'}</span></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><span style={{ color: s.brain_calc_flag ? '#f85149' : '#6e7681', fontWeight: 700 }}>{s.brain_calc_flag ? '✓' : '—'}</span></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}><span style={{ color: s.benign_variant ? '#3fb950' : '#6e7681', fontWeight: 700 }}>{s.benign_variant ? 'BENIGN' : '—'}</span></td>
                      <td style={{ padding: '8px 10px', color: '#8b949e', maxWidth: 200, fontSize: 11 }}>{s.key_mutation?.slice(0, 80)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            {/* Pathognomonic findings */}
            <div style={{ marginTop: 28 }}>
              <h3 style={{ margin: '0 0 16px', fontSize: 15, color: '#e6edf3' }}>Pathognomonic Findings & Clinical Clues</h3>
              <div style={{ display: 'grid', gap: 12 }}>
                {GENES.map(g => (
                  <div key={g} style={{ padding: 14, background: '#161b22', borderRadius: 8, borderLeft: `4px solid ${GENE_COLORS[g]}` }}>
                    <div style={{ fontWeight: 700, color: GENE_COLORS[g], marginBottom: 4 }}>{g}</div>
                    <div style={{ fontSize: 13, color: '#c9d1d9', lineHeight: 1.6 }}>{PATHOGNOMONIC[g]}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── BH4 & NEUROTRANSMITTER GROUPS TAB ── */}
        {activeTab === 'BH4 & Neurotransmitter Groups' && breakdown && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(420px,1fr))', gap: 20 }}>
              {Object.entries(breakdown.groups || {}).map(([grpKey, grp]) => (
                <div key={grpKey} style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 10, padding: 18 }}>
                  <h3 style={{ margin: '0 0 6px', fontSize: 15, color: '#58a6ff' }}>
                    {grpKey.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </h3>
                  <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 12 }}>{grp.mechanism}</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
                    {(grp.genes || []).map(g => (
                      <span key={g} style={{ padding: '3px 10px', borderRadius: 12, background: `${GENE_COLORS[g]}22`, color: GENE_COLORS[g], fontSize: 12, fontWeight: 600, border: `1px solid ${GENE_COLORS[g]}55` }}>{g}</span>
                    ))}
                  </div>
                  <div style={{ fontSize: 13, color: '#8b949e' }}>
                    <span style={{ color: '#e6edf3' }}>{grp.n_patients}</span> patients
                  </div>
                  {grp.diet_alone_insufficient && (
                    <div style={{ marginTop: 8, padding: '4px 10px', background: '#7c1d1d', borderRadius: 6, fontSize: 11, color: '#fca5a5' }}>
                      ⚠ Diet alone INSUFFICIENT — neurotransmitter replacement required
                    </div>
                  )}
                  {grp.nbs_detects_none && (
                    <div style={{ marginTop: 8, padding: '4px 10px', background: '#78350f', borderRadius: 6, fontSize: 11, color: '#fcd34d' }}>
                      ⚠ NBS MISSES all three DRD genes when Phe is normal
                    </div>
                  )}
                  {grp.folinic_acid_mandatory && (
                    <div style={{ marginTop: 8, padding: '4px 10px', background: '#1e3a5f', borderRadius: 6, fontSize: 11, color: '#93c5fd' }}>
                      💊 Folinic acid mandatory for {(grp.genes || []).join(', ')}
                    </div>
                  )}
                  {grp.pcbd1_benign && (
                    <div style={{ marginTop: 8, padding: '4px 10px', background: '#1b4332', borderRadius: 6, fontSize: 11, color: '#6ee7b7' }}>
                      ✓ PCBD1 = BENIGN (transient, no treatment)
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Gene breakdown stats */}
            <div style={{ marginTop: 28, background: '#161b22', border: '1px solid #30363d', borderRadius: 10, padding: 20 }}>
              <h3 style={{ margin: '0 0 16px', fontSize: 15, color: '#e6edf3' }}>Gene-Level Statistics (40 patients each)</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #30363d' }}>
                      {['Gene', 'NBS%', 'Phe↑%', 'Neuro%', 'DRD%', 'Brain Calc%', 'BH4 Resp%', 'NT Rx%', 'Folinic%', 'Diet Rx%', 'Diurnal%', 'Oculogyric%', 'Onset(m)', 'DxDelay(m)'].map(h => (
                        <th key={h} style={{ padding: '8px 8px', color: '#58a6ff', fontWeight: 600, textAlign: 'right', whiteSpace: 'nowrap' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {GENES.map((g, idx) => {
                      const s = breakdown.gene_breakdown?.[g];
                      if (!s) return null;
                      return (
                        <tr key={g} style={{ borderBottom: '1px solid #21262d', background: idx % 2 === 0 ? '#0d1117' : '#161b22' }}>
                          <td style={{ padding: '6px 8px', fontWeight: 700, color: GENE_COLORS[g] }}>{g}</td>
                          <td style={{ padding: '6px 8px', color: '#c9d1d9', textAlign: 'right' }}>{s.nbs_detected_pct}%</td>
                          <td style={{ padding: '6px 8px', color: '#c9d1d9', textAlign: 'right' }}>{s.hyperphe_pct}%</td>
                          <td style={{ padding: '6px 8px', color: '#f85149', textAlign: 'right' }}>{s.neurodegeneration_pct}%</td>
                          <td style={{ padding: '6px 8px', color: '#e3b341', textAlign: 'right' }}>{s.dopa_resp_pct}%</td>
                          <td style={{ padding: '6px 8px', color: '#f85149', textAlign: 'right' }}>{s.brain_calc_pct}%</td>
                          <td style={{ padding: '6px 8px', color: '#3fb950', textAlign: 'right' }}>{s.bh4_resp_pct}%</td>
                          <td style={{ padding: '6px 8px', color: '#fca5a5', textAlign: 'right' }}>{s.nt_replace_pct}%</td>
                          <td style={{ padding: '6px 8px', color: '#93c5fd', textAlign: 'right' }}>{s.folinic_acid_pct}%</td>
                          <td style={{ padding: '6px 8px', color: '#c9d1d9', textAlign: 'right' }}>{s.diet_rx_pct}%</td>
                          <td style={{ padding: '6px 8px', color: '#e3b341', textAlign: 'right' }}>{s.diurnal_var_pct}%</td>
                          <td style={{ padding: '6px 8px', color: '#f85149', textAlign: 'right' }}>{s.oculogyric_pct}%</td>
                          <td style={{ padding: '6px 8px', color: '#8b949e', textAlign: 'right' }}>{s.mean_age_onset_m}</td>
                          <td style={{ padding: '6px 8px', color: '#8b949e', textAlign: 'right' }}>{s.mean_dx_delay_m}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {activeTab === 'Definitions' && definitions && (
          <div>
            <div style={{ display: 'grid', gap: 12 }}>
              {GENES.map(g => {
                const d = definitions.definitions?.[g];
                if (!d) return null;
                const expanded = expandedDef === g;
                return (
                  <div key={g} style={{ background: '#161b22', border: `1px solid ${GENE_COLORS[g]}44`, borderRadius: 10, overflow: 'hidden' }}>
                    <button onClick={() => setExpandedDef(expanded ? null : g)} style={{
                      width: '100%', padding: '16px 20px', background: 'none', border: 'none', cursor: 'pointer',
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center', textAlign: 'left',
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                        <span style={{ fontWeight: 700, fontSize: 16, color: GENE_COLORS[g] }}>{g}</span>
                        <span style={{ fontSize: 13, color: '#8b949e' }}>{d.full_name} · {d.protein_size} · {d.locus}</span>
                        <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4, background: '#1c1917', color: '#fca5a5', border: '1px solid #7f1d1d' }}>{d.inheritance}</span>
                      </div>
                      <span style={{ color: '#58a6ff', fontSize: 18 }}>{expanded ? '▲' : '▼'}</span>
                    </button>
                    {expanded && (
                      <div style={{ padding: '0 20px 20px', borderTop: `1px solid ${GENE_COLORS[g]}33` }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16, marginTop: 16 }}>
                          {[
                            ['OMIM Gene', d.omim_gene],
                            ['OMIM Disease', d.omim_disease],
                            ['Inheritance', d.inheritance],
                            ['Locus', d.locus],
                          ].map(([label, val]) => (
                            <div key={label} style={{ background: '#0d1117', borderRadius: 6, padding: '8px 12px' }}>
                              <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 2 }}>{label}</div>
                              <div style={{ fontSize: 13, color: '#e6edf3', fontWeight: 500 }}>{val}</div>
                            </div>
                          ))}
                        </div>
                        {[
                          { label: 'Molecular Mechanism', val: d.mechanism_detail, color: '#58a6ff' },
                          { label: 'Pathognomonic Features', val: d.pathognomonic, color: '#e3b341' },
                          { label: 'Treatment', val: d.treatment, color: '#3fb950' },
                          { label: 'Surveillance', val: d.surveillance, color: '#79c0ff' },
                          { label: 'Key DDx', val: d.key_ddx, color: '#ff7b72' },
                        ].map(({ label, val, color }) => (
                          <div key={label} style={{ padding: '10px 14px', background: '#0d1117', borderRadius: 8, marginBottom: 8, borderLeft: `3px solid ${color}` }}>
                            <div style={{ fontSize: 12, fontWeight: 600, color, marginBottom: 4 }}>{label}</div>
                            <div style={{ fontSize: 13, color: '#c9d1d9', lineHeight: 1.6 }}>{val}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
