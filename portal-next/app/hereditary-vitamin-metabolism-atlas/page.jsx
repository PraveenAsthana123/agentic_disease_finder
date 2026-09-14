'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-vitamin-metabolism-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'BTD':     '#1a237e',  // deep indigo — Biotinidase deficiency; biotin recycling; alopecia+SNHL; biotin curative
  'HLCS':    '#880e4f',  // deep magenta — Holocarboxylase synthetase; neonatal MCD; hyperammonaemia; biotin 10-40mg
  'SLC19A3': '#b71c1c',  // deep red — BTBGD; thiamine transporter 2; stress-triggered BG lesions; biotin+thiamine
  'MTHFR':   '#e65100',  // deep orange — Severe MTHFR; homocystinuria type 4; LOW methionine; betaine+5-MTHF
  'SLC52A2': '#1b5e20',  // dark green — RTD2/BVVL2; riboflavin transporter; SNHL+pontobulbar; riboflavin
  'FLAD1':   '#4a148c',  // deep purple — FAD synthase; MADD-like; lipid storage myopathy; riboflavin-responsive
  'TCN2':    '#006064',  // dark teal — Transcobalamin II; B12 cellular delivery; NORMAL B12 trap; parenteral B12
  'AMN':     '#3e2723',  // dark brown — IGS-2; selective B12 malabsorption; tubular proteinuria; IM B12
};

const GENE_INFO = {
  'BTD':     { full: 'BTD / Biotinidase / 543aa', locus: '3p25.1', size: '543 aa / 67 kDa', inh: 'AR', disease: 'Biotinidase deficiency; biotin recycling defect; alopecia + perioral dermatitis + SNHL (40-75%) + seizures; lactic acidosis + organic aciduria; BIOTIN 5-10 mg/day CURATIVE; NBS by fluorimetric BTD assay; p.Asp444His most common severe allele; untreated: optic atrophy + hearing loss + neurodevelopmental regression' },
  'HLCS':    { full: 'HLCS / Holocarboxylase Synthetase / 726aa', locus: '21q22.13', size: '726 aa / 82 kDa', inh: 'AR', disease: 'Holocarboxylase synthetase deficiency; neonatal multiple carboxylase deficiency (MCD); neonatal crisis: metabolic acidosis + hyperammonaemia in first 24-72 hours; triple organic acid pattern (3-methylcrotonylglycinuria + lactic + methylcitric); C5-OH + C3 on NBS; BIOTIN 10-40 mg/day; Km-mutant alleles (p.Leu216Arg) respond well; functionally null = poor outcome' },
  'SLC19A3': { full: 'SLC19A3 / Thiamine Transporter 2 (ThTr2) / 500aa', locus: '2q36.3', size: '500 aa / 56 kDa', inh: 'AR', disease: 'Biotin-thiamine responsive basal ganglia disease (BTBGD/THMD2); thiamine transporter 2 deficiency; BILATERAL SYMMETRIC BG MRI LESIONS (caudate+putamen) + cortical ribbon T2 changes PATHOGNOMONIC; STRESS-TRIGGERED encephalopathy (fever/vaccination); BOTH biotin + thiamine required; p.Glu320Gln Saudi founder ~60%; IV thiamine before glucose in crisis' },
  'MTHFR':   { full: 'MTHFR / Methylenetetrahydrofolate Reductase / 698aa', locus: '1p36.22', size: '698 aa / 74 kDa', inh: 'AR (severe biallelic only)', disease: 'Severe MTHFR deficiency / Homocystinuria type 4; LOW METHIONINE + elevated homocysteine PATHOGNOMONIC DDx FROM CBS (CBS = HIGH methionine); neonatal encephalopathy + white matter disease; BETAINE (primary) + 5-MTHF + hydroxocobalamin; CRITICAL: C677T/A1298C are population POLYMORPHISMS — NOT rare disease' },
  'SLC52A2': { full: 'SLC52A2 / Riboflavin Transporter 2 (RFT2) / 460aa', locus: '8q24.13', size: '460 aa / 50 kDa', inh: 'AR', disease: 'Riboflavin transporter deficiency type 2 (RTD2) / Brown-Vialetto-Van Laere syndrome 2 (BVVL2); SNHL (often first) + PONTOBULBAR PALSY (VII/IX/X/XII) + axonal neuropathy + respiratory failure; MADD-like acylcarnitines; RIBOFLAVIN 10-40 mg/kg/day CAN REVERSE established SNHL — unique treatment response; start empirically on SNHL+pontobulbar combination' },
  'FLAD1':   { full: 'FLAD1 / FAD Synthase / 644aa', locus: '1q21.3', size: '644 aa / 72 kDa', inh: 'AR', disease: 'FAD synthase deficiency; riboflavin-responsive MADD-like; LIPID STORAGE MYOPATHY on muscle biopsy (oil red O excess lipid); exercise intolerance + proximal myopathy + rhabdomyolysis; MADD acylcarnitines (C6-C18); riboflavin 10-40 mg/kg/day often dramatically effective; fasting avoidance critical; distinguish from SLC52A2: no SNHL/pontobulbar palsy' },
  'TCN2':    { full: 'TCN2 / Transcobalamin II / 427aa', locus: '22q12.2', size: '427 aa / 46 kDa', inh: 'AR', disease: 'Transcobalamin II deficiency; B12 cellular delivery defect; NORMAL/HIGH serum B12 PATHOGNOMONIC TRAP — B12 appears normal but all cells B12-deficient; neonatal megaloblastic anaemia + FTT; low holotranscobalamin (active B12) is KEY diagnostic marker; mild MMA + homocysteine; HYDROXOCOBALAMIN IM bypasses TCII; lifelong parenteral B12' },
  'AMN':     { full: 'AMN / Amnionless / 453aa', locus: '14q32.32', size: '453 aa / 50 kDa', inh: 'AR', disease: 'Imerslund-Gräsbeck syndrome type 2 (IGS-2); cubam complex dysfunction (AMN = cubilin trafficking chaperone); selective intestinal B12 malabsorption; LOW serum B12 (contrast TCN2); tubular proteinuria 50% (β2-microglobulin/RBP — low-grade, NOT nephrotic); ORAL B12 NOT absorbed (even with IF); Schilling NOT corrected by IF; IM hydroxocobalamin CURATIVE; lifelong parenteral B12' },
};

function GeneChip({ gene }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span style={{ background: col, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 12, fontWeight: 700, margin: '0 2px' }}>
      {gene}
    </span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 120 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8' }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HereditaryVitaminMetabolismAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [selGene, setSelGene] = useState(null);

  useEffect(() => {
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' ? 'breakdown'
      : tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    setLoading(true); setErr(null);
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => {
        if (ep === 'overview') setOverview(data);
        else if (ep === 'breakdown') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setErr(String(e)); setLoading(false); });
  }, [tab]);

  const bg = '#0f172a';
  const card = '#1e293b';
  const accent = '#38bdf8';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1400, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: accent, marginBottom: 6 }}>
            🧬 Hereditary Vitamin Metabolism Atlas
          </div>
          <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>
            8-Gene Reference · BTD · HLCS · SLC19A3 · MTHFR · SLC52A2 · FLAD1 · TCN2 · AMN · 320 patients (8×40) · seeds 2638–2645
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
            {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
          </div>
          {/* Tabs */}
          <div style={{ display: 'flex', gap: 8, borderBottom: '1px solid #334155', paddingBottom: 0, marginTop: 12 }}>
            {TABS.map(t => (
              <button key={t} onClick={() => setTab(t)} style={{
                background: tab === t ? accent : 'transparent',
                color: tab === t ? '#0f172a' : '#94a3b8',
                border: 'none', padding: '6px 16px', borderRadius: '4px 4px 0 0',
                fontWeight: tab === t ? 700 : 400, cursor: 'pointer', fontSize: 13,
              }}>{t}</button>
            ))}
          </div>
        </div>

        {loading && <div style={{ color: '#94a3b8', padding: 24 }}>Loading...</div>}
        {err && <div style={{ color: '#ef4444', padding: 12 }}>Error: {err}</div>}

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 genes × 40" />
              <MetricCard label="Genes" value={overview.n_genes} sub="BTD–AMN" />
              <MetricCard label="Hearing Loss%" value={`${overview.aggregate_stats?.overall_hearing_loss_pct}%`} />
              <MetricCard label="Metabolic Crisis%" value={`${overview.aggregate_stats?.overall_metabolic_crisis_pct}%`} />
              <MetricCard label="Vitamin Responsive%" value={`${overview.aggregate_stats?.overall_vitamin_responsive_pct}%`} />
              <MetricCard label="Seeds" value={overview.seeds} sub="sequential" />
            </div>

            {/* Disease Classes */}
            <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>Disease Classes by Gene</div>
              {overview.disease_classes?.map((dc, i) => (
                <div key={i} style={{ fontSize: 12, color: '#cbd5e1', borderBottom: '1px solid #0f172a', padding: '5px 0', lineHeight: 1.5 }}>
                  {dc}
                </div>
              ))}
            </div>

            {/* Key Clinical Distinctions */}
            <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>Key Clinical Distinctions</div>
              {overview.key_clinical_distinctions?.map((kd, i) => (
                <div key={i} style={{ fontSize: 12, color: '#cbd5e1', borderBottom: '1px solid #0f172a', padding: '6px 0', lineHeight: 1.5 }}>
                  {kd}
                </div>
              ))}
            </div>

            {/* Gene Summary Table */}
            <div style={{ background: card, borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>Gene Summary — 40 Patients Each</div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ color: '#64748b' }}>
                    <th style={{ textAlign: 'left', padding: '4px 8px' }}>Gene</th>
                    <th style={{ textAlign: 'left', padding: '4px 8px' }}>Locus</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>N</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Avg Onset (yr)</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Hearing Loss%</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Metabolic Crisis%</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Vit Responsive%</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.map((gs, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#0f172a' : card }}>
                      <td style={{ padding: '4px 8px' }}><GeneChip gene={gs.gene} /></td>
                      <td style={{ padding: '4px 8px', color: '#94a3b8' }}>{gs.locus}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.n_patients}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.avg_onset_age}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.hearing_loss_pct}%</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.metabolic_crisis_pct}%</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right', color: '#86efac' }}>{gs.vitamin_treatment_responsive_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && breakdown && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
            {breakdown.gene_breakdowns?.map((gb, i) => {
              const info = GENE_INFO[gb.gene] || {};
              return (
                <div key={i} style={{ background: card, borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[gb.gene] || '#555'}` }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[gb.gene] || accent, fontSize: 15, marginBottom: 4 }}>
                    {gb.gene}
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>
                    {info.locus} · {info.size} · {info.inh}
                  </div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 8, lineHeight: 1.5 }}>
                    {info.disease}
                  </div>
                  <div style={{ display: 'flex', gap: 12, fontSize: 11, color: '#94a3b8', flexWrap: 'wrap' }}>
                    <span>Onset: {gb.avg_onset_age}yr</span>
                    <span>N: {gb.n_patients}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            {/* Gene selector */}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
              {breakdown.gene_breakdowns?.map(gb => (
                <button key={gb.gene} onClick={() => setSelGene(gb.gene === selGene ? null : gb.gene)}
                  style={{
                    background: selGene === gb.gene ? GENE_COLORS[gb.gene] : 'transparent',
                    color: selGene === gb.gene ? '#fff' : '#94a3b8',
                    border: `1px solid ${GENE_COLORS[gb.gene] || '#334155'}`,
                    padding: '4px 12px', borderRadius: 4, cursor: 'pointer', fontSize: 12, fontWeight: 600,
                  }}>{gb.gene}</button>
              ))}
            </div>

            {breakdown.gene_breakdowns?.filter(gb => !selGene || gb.gene === selGene).map((gb, i) => (
              <div key={i} style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[gb.gene] || '#555'}` }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gb.gene] || accent, fontSize: 16, marginBottom: 8 }}>
                  {gb.gene} — {gb.disease_category?.split(';')[0]}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                  <div>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>INHERITANCE / PREVALENCE</div>
                    <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.5 }}>{gb.inheritance}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>PATHOGNOMONIC FEATURES</div>
                    <div style={{ fontSize: 12, color: '#fbbf24', lineHeight: 1.5 }}>{gb.pathognomonic?.substring(0, 400)}...</div>
                  </div>
                </div>
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>TREATMENT</div>
                  <div style={{ fontSize: 12, color: '#86efac', lineHeight: 1.5 }}>{gb.treatment?.substring(0, 500)}...</div>
                </div>
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>KEY FEATURES</div>
                  {gb.key_features?.map((kf, j) => (
                    <div key={j} style={{ fontSize: 12, color: '#e2e8f0', padding: '2px 0' }}>• {kf}</div>
                  ))}
                </div>
                <div>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>KEY DDx</div>
                  {gb.key_ddx?.map((kd, j) => (
                    <div key={j} style={{ fontSize: 12, color: '#f87171', padding: '2px 0' }}>• {kd}</div>
                  ))}
                </div>
                {/* Feature rates */}
                {gb.feature_rates && Object.keys(gb.feature_rates).length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>COHORT FEATURE RATES (N=40)</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {Object.entries(gb.feature_rates).map(([feat, pct]) => (
                        <span key={feat} style={{
                          background: pct > 70 ? '#1a3a2a' : pct > 40 ? '#1e2a1e' : '#1e293b',
                          border: `1px solid ${pct > 70 ? '#22c55e' : pct > 40 ? '#86efac' : '#334155'}`,
                          borderRadius: 4, padding: '2px 6px', fontSize: 11, color: '#cbd5e1'
                        }}>
                          {feat.replace(/_/g, ' ')}: {pct}%
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            {/* Gene entries */}
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontWeight: 700, color: accent, fontSize: 15, marginBottom: 12 }}>Gene Reference Entries</div>
              {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => (
                <div key={gene} style={{ background: card, borderRadius: 8, padding: 14, marginBottom: 12, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || accent, marginBottom: 4 }}>{gene} — {entry.disease_name}</div>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>{entry.locus} · {entry.protein_size} · {entry.inheritance}</div>
                  <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 6 }}>{entry.disease_pathway?.substring(0, 300)}...</div>
                  <div style={{ fontSize: 12, color: '#fbbf24', lineHeight: 1.5 }}>{entry.pathognomonic?.substring(0, 250)}...</div>
                </div>
              ))}
            </div>

            {/* Vitamin Metabolism Glossary */}
            <div style={{ fontWeight: 700, color: accent, fontSize: 15, marginBottom: 12 }}>Vitamin Metabolism Glossary</div>
            {Object.entries(definitions.vitamin_metabolism_glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: card, borderRadius: 8, padding: 14, marginBottom: 12 }}>
                <div style={{ fontWeight: 700, color: '#7dd3fc', marginBottom: 6 }}>{term}</div>
                <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{def}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
