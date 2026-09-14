'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hypertriglyceridemia-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'LPL':     '#0d47a1',  // deep blue — most common FCS; postheparin LPL near-zero PATHOGNOMONIC; cream plasma
  'APOC2':   '#880e4f',  // deep magenta — FCS Type Ib; APOC2 correction test POSITIVE PATHOGNOMONIC; FFP first-line
  'APOA5':   '#1b5e20',  // dark green — Type V (VLDL+chylomicrons); fibrates more effective; pregnancy risk
  'GPIHBP1': '#e65100',  // deep orange — postheparin LPL NORMAL paradox; acquired autoantibody form
  'LMF1':    '#4a148c',  // deep purple — combined LPL+HL both low PATHOGNOMONIC; LMF1 chaperone
  'APOC3':   '#b71c1c',  // deep red — LPL inhibitor; volanesorsen EMA2019; platelet monitoring MANDATORY
  'ANGPTL3': '#006064',  // dark teal — pan-lipase inhibitor LOF=all lipids low; evinacumab FDA2021 HoFH
  'LIPC':    '#3e2723',  // dark brown — hepatic lipase; IDL elevated; HDL paradox; CVD high; palmar xanthomas
};

const GENE_INFO = {
  'LPL':     { full: 'LPL / Lipoprotein Lipase / 475aa', locus: '8p21.3', size: '475 aa / 53 kDa (homodimer, GPI-anchored via GPIHBP1)', inh: 'AR', disease: 'FAMILIAL CHYLOMICRONEMIA SYNDROME (FCS) TYPE I — most common FCS (~70%); TG >2000 mg/dL (often 5000-18,000); CREAM LAYER PATHOGNOMONIC; postheparin LPL activity near-zero PATHOGNOMONIC; APOC2 correction test NEGATIVE; pancreatitis 75%; eruptive xanthomas; lipaemia retinalis; NO excess CVD (chylomicrons too large for artery wall); VOLANESORSEN EMA2019 (APOC3 ASO reduces LPL inhibitor); ultra-low fat diet <20g/day mandatory; French-Canadian founder mutation common' },
  'APOC2':   { full: 'APOC2 / Apolipoprotein C-II / 101aa', locus: '19q13.32', size: '101 aa / 9 kDa (exchangeable apolipoprotein, LPL obligate cofactor)', inh: 'AR', disease: 'FCS TYPE Ib — LPL obligate cofactor deficiency; TG >2000 mg/dL; cream plasma; postheparin LPL activity low; APOC2 CORRECTION TEST POSITIVE = PATHOGNOMONIC (add exogenous APOC2 → LPL activity restores); FFP FIRST-LINE acute (provides APOC2 → 50-90% TG fall in 24-48h); more predictable FFP response than LPL FCS; ultra-low fat diet; volanesorsen (partial — APOC3 relief); fibrates/omega-3 ineffective for homozygous' },
  'APOA5':   { full: 'APOA5 / Apolipoprotein A-V / 366aa', locus: '11q23.3', size: '366 aa / 41 kDa (liver-secreted, HSPG-binding, LPL enhancer)', inh: 'AR (biallelic severe); heterozygous = susceptibility', disease: 'HYPERLIPOPROTEINEMIA TYPE V (biallelic) — BOTH chylomicrons + VLDL elevated (Type V = Type I + IV); TG 500-8000 mg/dL; fibrates MORE EFFECTIVE than in LPL/APOC2 FCS (PPARα upregulates APOA5); common variants: -1131T>C (European, 4x TG risk), p.Gly19Arg (East Asian, 4x TG); PREGNANCY RISK: heterozygous APOA5 + pregnancy → gestational FCS + pancreatitis; secondary triggers (OCP, alcohol, T2DM) amplify severity; APOA5 dual mechanism: activates LPL + promotes hepatic TRL remnant uptake via HSPG' },
  'GPIHBP1': { full: 'GPIHBP1 / GPI-Anchored HDL-Binding Protein 1 / 184aa', locus: '8q24.13', size: '184 aa / 22 kDa (GPI-anchored endothelial, Ly6/uPAR domain)', inh: 'AR (hereditary); autoimmune (acquired)', disease: 'FCS — GPIHBP1 SHUTTLE DEFECT; TG >2000 mg/dL; POSTHEPARIN LPL ACTIVITY NORMAL = KEY DIAGNOSTIC PARADOX (LPL released by heparin but cannot function in vivo — not transported to capillary lumen); ACQUIRED FORM: anti-GPIHBP1 IgG autoantibodies — adult onset, paraneoplastic/autoimmune; immunosuppression (steroids/rituximab) dramatically effective for acquired form; Ly6 domain mutations impair LPL binding; diagnosis = GPIHBP1 sequencing + anti-GPIHBP1 ELISA' },
  'LMF1':    { full: 'LMF1 / Lipase Maturation Factor 1 / 567aa', locus: '16p13.3', size: '567 aa / 65 kDa (multi-pass ER transmembrane chaperone)', inh: 'AR', disease: 'COMBINED LIPASE DEFICIENCY — ER chaperone for LPL + HL + EL maturation; POSTHEPARIN LPL + HL BOTH LOW = PATHOGNOMONIC (contrast pure LPL FCS where HL NORMAL); TG 1200-9000 mg/dL; IDL/remnant accumulation (from HL deficiency); higher CVD risk than pure FCS (IDL atherogenic); SALT-RESISTANT HL ASSAY: HL low at 1M NaCl + LPL low at 0M NaCl → LMF1; very rare; cld mouse model defined LMF1 biology; limited fibrate benefit (lipases still misfolded)' },
  'APOC3':   { full: 'APOC3 / Apolipoprotein C-III / 99aa', locus: '19q13.32', size: '99 aa / 9 kDa (exchangeable apolipoprotein; LPL inhibitor + hepatic TRL uptake blocker)', inh: 'No simple AR; dose-effect dominant-like', disease: 'FAMILIAL HYPERTRIGLYCERIDAEMIA (FHTG) — LPL inhibitor; TG 400-6000 mg/dL; VLDL elevated (Type IV); CVD risk (VLDL remnants atherogenic); APOC3 level >15 mg/dL with severe HTG; VOLANESORSEN (Waylivra EMA2019): 285mg SC weekly; TG 70-80% reduction; PLATELET MONITORING MANDATORY (thrombocytopenia 40%); R19X natural LOF = 40% lower TG + 40% lower CVD; LOF validates APOC3 as therapeutic target; alcohol + insulin resistance major APOC3 upregulators; T2DM control essential' },
  'ANGPTL3': { full: 'ANGPTL3 / Angiopoietin-Like Protein 3 / 460aa', locus: '1p31.3', size: '460 aa / 54 kDa (liver-secreted; N-terminal coiled-coil + C-terminal fibrinogen-like)', inh: 'AR LOF = familial combined hypolipidaemia; therapeutic target (evinacumab)', disease: 'FAMILIAL COMBINED HYPOLIPIDAEMIA (biallelic LOF) — ALL lipid fractions very low (TG <40, LDL <50, HDL <25 mg/dL); benign phenotype; NO CVD + NO pancreatitis; validates ANGPTL3 as safe target; EVINACUMAB (Evkeeza FDA2021): anti-ANGPTL3 mAb 15mg/kg IV monthly for HoFH; LDL reduction ~47% additional; UNIQUE: LDLR-independent LDL lowering (works in LDLR-null patients failing statins+PCSK9i+apheresis); ANGPTL3 high in metabolic syndrome → contributes to common HTG; ELEBSIRAN (siRNA, Phase 3); VUPANORSEN (ASO)' },
  'LIPC':    { full: 'LIPC / Hepatic Lipase / 499aa', locus: '15q22.1', size: '499 aa / 53 kDa (GPI-anchored liver sinusoidal endothelium; salt-resistant lipase)', inh: 'AR', disease: 'HEPATIC LIPASE DEFICIENCY — TG 200-2000 mg/dL (less severe than FCS); IDL + chylomicron remnants elevated (ATHEROGENIC); HDL-C PARADOXICALLY HIGH (HDL2 not converted to HDL3); PALMAR XANTHOMAS (orange palm creases — IDL deposits); HIGH CVD RISK (IDL atherogenic); POSTHEPARIN HL LOW + LPL NORMAL (distinguish from LMF1 where BOTH low); SALT-RESISTANT LIPASE (HL at 1M NaCl) absent; LIPC -514C>T common promoter variant reduces HL 20-30%; differentiate from Type III: APOE ε2/ε2 in Type III vs normal APOE + LIPC mutations in HL def; statins + fibrates primary treatment' },
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

export default function HeredHTGAtlasPage() {
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

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: accent, margin: 0 }}>
          🧬 Hereditary Hypertriglyceridemia &amp; Familial Chylomicronemia Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 6 }}>
          Complete 8-Gene Reference · LPL · APOC2 · APOA5 · GPIHBP1 · LMF1 · APOC3 · ANGPTL3 · LIPC
          · 320 patients (8×40) · seeds 2686–2693
        </div>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          FCS Type I (LPL) · FCS Type Ib (APOC2) · Hyperlipoproteinemia V (APOA5) ·
          GPIHBP1 FCS · Combined Lipase Deficiency (LMF1) · FHTG (APOC3) ·
          Familial Combined Hypolipidemia (ANGPTL3) · Hepatic Lipase Deficiency (LIPC)
        </div>
      </div>

      {/* Gene chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 20 }}>
        {Object.keys(GENE_COLORS).map(g => (
          <button key={g} onClick={() => setSelGene(selGene === g ? null : g)}
            style={{ background: selGene === g ? GENE_COLORS[g] : '#1e293b',
              border: `2px solid ${GENE_COLORS[g]}`, borderRadius: 6, color: '#e2e8f0',
              padding: '4px 12px', cursor: 'pointer', fontSize: 13, fontWeight: 700 }}>
            {g}
          </button>
        ))}
      </div>

      {/* Selected gene info */}
      {selGene && GENE_INFO[selGene] && (
        <div style={{ background: '#1e293b', border: `2px solid ${GENE_COLORS[selGene]}`, borderRadius: 10, padding: 16, marginBottom: 20 }}>
          <div style={{ fontWeight: 700, color: GENE_COLORS[selGene], fontSize: 15, marginBottom: 4 }}>
            {GENE_INFO[selGene].full}
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>
            Locus: {GENE_INFO[selGene].locus} · Size: {GENE_INFO[selGene].size} · Inheritance: {GENE_INFO[selGene].inh}
          </div>
          <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>{GENE_INFO[selGene].disease}</div>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{ background: tab === t ? accent : '#1e293b', color: tab === t ? '#0f172a' : '#94a3b8',
              border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400, fontSize: 13 }}>
            {t}
          </button>
        ))}
      </div>

      {loading && <div style={{ color: '#94a3b8' }}>Loading…</div>}
      {err && <div style={{ color: '#ef4444' }}>Error: {err}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40 cohort" />
            <MetricCard label="Genes" value={overview.genes?.length} sub="FCS + HTG + Hypolipidemia" />
            <MetricCard label="Seeds" value={`${overview.seeds?.[0]}–${overview.seeds?.at(-1)}`} sub="deterministic" />
            <MetricCard label="Critical Distinctions" value={overview.critical_distinctions?.length} warn />
          </div>

          {/* FCS Classification */}
          <div style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>📋 FCS & HTG Classification</div>
            {overview.fcs_classification?.map((c, i) => (
              <div key={i} style={{ fontSize: 12, color: '#cbd5e1', borderLeft: `3px solid ${accent}`, paddingLeft: 10, marginBottom: 8 }}>
                {c}
              </div>
            ))}
          </div>

          {/* Critical Distinctions */}
          <div style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#ef4444', marginBottom: 10 }}>⚠️ Critical Diagnostic Distinctions</div>
            {overview.critical_distinctions?.map((d, i) => (
              <div key={i} style={{ fontSize: 12, color: '#fca5a5', borderLeft: '3px solid #ef4444', paddingLeft: 10, marginBottom: 6 }}>
                {d}
              </div>
            ))}
          </div>

          {/* Gene summaries table */}
          <div style={{ background: card, borderRadius: 10, padding: 16 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 12 }}>📊 Gene-Level Statistics (n=40 per gene)</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px' }}>Gene</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Mean TG (mg/dL)</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Pancreatitis%</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Xanthoma%</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>CVD%</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Lipaemia%</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px' }}>↓LPL?</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px' }}>↓HL?</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Onset yr</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #1e293b' }}>
                      <td style={{ padding: '5px 8px' }}><GeneChip gene={s.gene} /></td>
                      <td style={{ textAlign: 'right', padding: '5px 8px', color: s.mean_tg_mg_dL > 2000 ? '#ef4444' : s.mean_tg_mg_dL > 500 ? '#fbbf24' : '#4ade80' }}>{s.mean_tg_mg_dL.toLocaleString()}</td>
                      <td style={{ textAlign: 'right', padding: '5px 8px', color: s.pancreatitis_pct > 50 ? '#ef4444' : '#e2e8f0' }}>{s.pancreatitis_pct}%</td>
                      <td style={{ textAlign: 'right', padding: '5px 8px' }}>{s.eruptive_xanthoma_pct}%</td>
                      <td style={{ textAlign: 'right', padding: '5px 8px', color: s.cvd_event_pct > 30 ? '#ef4444' : '#e2e8f0' }}>{s.cvd_event_pct}%</td>
                      <td style={{ textAlign: 'right', padding: '5px 8px' }}>{s.lipaemia_retinalis_pct}%</td>
                      <td style={{ textAlign: 'center', padding: '5px 8px' }}>{s.postheparin_lpl_low ? '🔴 LOW' : '🟢 NORMAL'}</td>
                      <td style={{ textAlign: 'center', padding: '5px 8px' }}>{s.postheparin_hl_low ? '🔴 LOW' : '🟢 NORMAL'}</td>
                      <td style={{ textAlign: 'right', padding: '5px 8px', color: '#94a3b8' }}>{s.mean_onset_years}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          {breakdown.breakdown?.map((b, i) => (
            <div key={i} style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 16,
              borderLeft: `4px solid ${GENE_COLORS[b.gene] || accent}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8 }}>
                <div>
                  <GeneChip gene={b.gene} />
                  <span style={{ fontSize: 13, color: '#94a3b8', marginLeft: 10 }}>{b.locus} · {b.protein_size}</span>
                </div>
                <div style={{ display: 'flex', gap: 12, fontSize: 12 }}>
                  <span style={{ color: b.mean_tg_mg_dL > 2000 ? '#ef4444' : '#fbbf24' }}>TG {b.mean_tg_mg_dL?.toFixed(0)} mg/dL avg</span>
                  <span style={{ color: '#94a3b8' }}>Panc {b.pancreatitis_pct}%</span>
                  <span style={{ color: b.cvd_event_pct > 30 ? '#ef4444' : '#94a3b8' }}>CVD {b.cvd_event_pct}%</span>
                  <span style={{ color: '#94a3b8' }}>HDL {b.mean_hdl_c_mg_dL?.toFixed(0)}</span>
                </div>
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8, lineHeight: 1.5 }}>
                <strong style={{ color: '#e2e8f0' }}>Disease:</strong> {b.disease_category}
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 6, lineHeight: 1.5 }}>
                <strong style={{ color: '#fbbf24' }}>Pathognomonic:</strong> {b.pathognomonic}
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 6, lineHeight: 1.5 }}>
                <strong style={{ color: '#4ade80' }}>Treatment:</strong> {b.treatment_summary}
              </div>
              <div style={{ display: 'flex', gap: 16, marginTop: 8, fontSize: 11, color: '#64748b' }}>
                <span>⬇️LPL: {b.postheparin_lpl_low ? 'LOW' : 'NORMAL'}</span>
                <span>⬇️HL: {b.postheparin_hl_low ? 'LOW' : 'NORMAL'}</span>
                <span>FFP resp: {b.ffp_response_pct}%</span>
                <span>Volanesorsen candidate: {b.volanesorsen_candidate_pct}%</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {/* Postheparin Assay Decision Tree */}
          <div style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 12 }}>🔬 Postheparin Assay Diagnostic Algorithm</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
              {[
                { condition: 'LPL LOW + HL NORMAL + APOC2 correction NEGATIVE', gene: 'LPL', dx: 'FCS Type I — most common', col: '#0d47a1' },
                { condition: 'LPL LOW + HL NORMAL + APOC2 correction POSITIVE', gene: 'APOC2', dx: 'FCS Type Ib — FFP most effective', col: '#880e4f' },
                { condition: 'LPL NORMAL + HL NORMAL + TG >2000', gene: 'GPIHBP1', dx: 'GPIHBP1 FCS — or check anti-GPIHBP1 Ab', col: '#e65100' },
                { condition: 'LPL LOW + HL LOW (salt-resistant)', gene: 'LMF1', dx: 'Combined Lipase Deficiency', col: '#4a148c' },
                { condition: 'LPL REDUCED (not absent) + Type V phenotype', gene: 'APOA5', dx: 'APOA5 biallelic — fibrates more effective', col: '#1b5e20' },
                { condition: 'LPL NORMAL + HL LOW + IDL elevated', gene: 'LIPC', dx: 'Hepatic Lipase Deficiency — CVD risk', col: '#3e2723' },
                { condition: 'TG moderately-severely elevated + APOC3 >15 mg/dL', gene: 'APOC3', dx: 'FHTG — volanesorsen target', col: '#b71c1c' },
                { condition: 'ALL lipids very low (TG <40, LDL <50, HDL <25)', gene: 'ANGPTL3', dx: 'Familial Combined Hypolipidemia — benign', col: '#006064' },
              ].map((row, i) => (
                <div key={i} style={{ background: '#0f172a', borderRadius: 8, padding: 12, borderLeft: `4px solid ${row.col}` }}>
                  <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>{row.condition}</div>
                  <div style={{ fontWeight: 700, fontSize: 13 }}><GeneChip gene={row.gene} /></div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', marginTop: 4 }}>{row.dx}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Treatment alerts */}
          <div style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#ef4444', marginBottom: 12 }}>⚠️ Critical Treatment Alerts</div>
            {[
              { alert: 'VOLANESORSEN — PLATELET COUNT MANDATORY', detail: 'Thrombocytopenia in up to 40% patients. Monitor platelets before each injection. Hold <75,000/µL. STOP <50,000/µL.', gene: 'APOC3' },
              { alert: 'FIBRATES + GEMFIBROZIL + STATIN — RHABDOMYOLYSIS RISK', detail: 'Prefer fenofibrate + statin (lower rhabdomyolysis risk). Avoid gemfibrozil + statin combination.', gene: 'LIPC' },
              { alert: 'OCP (ETHINYLOESTRADIOL) IN APOA5 CARRIERS', detail: 'Oestrogens increase VLDL secretion → can precipitate severe gestational or OCP-induced pancreatitis. Switch to progestogen-only or IUD.', gene: 'APOA5' },
              { alert: 'EVINACUMAB — LDLR-INDEPENDENT MECHANISM', detail: 'Works even in LDLR-null HoFH patients. Do not assume ineffective because LDLR is absent. LDL reduction ~47% additional.', gene: 'ANGPTL3' },
              { alert: 'FFP IN APOC2 FCS — MOST EFFECTIVE ACUTE INTERVENTION', detail: 'Provides functional APOC2 cofactor → LPL activates → TG falls 50-90% in 24-48h. More predictable than in LPL FCS.', gene: 'APOC2' },
              { alert: 'GPIHBP1 PARADOX — DO NOT BE REASSURED BY NORMAL POSTHEPARIN LPL', detail: 'Normal postheparin LPL activity does NOT exclude FCS. GPIHBP1 deficiency releases LPL but LPL cannot reach capillary lumen in vivo.', gene: 'GPIHBP1' },
            ].map((a, i) => (
              <div key={i} style={{ borderLeft: `4px solid #ef4444`, paddingLeft: 12, marginBottom: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 13, color: '#fca5a5' }}>
                  {a.alert} <GeneChip gene={a.gene} />
                </div>
                <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>{a.detail}</div>
              </div>
            ))}
          </div>

          {/* Sample patients */}
          {breakdown.breakdown?.slice(0, 3).map(b => (
            <div key={b.gene} style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, color: GENE_COLORS[b.gene], marginBottom: 8 }}>
                <GeneChip gene={b.gene} /> Sample Patients (n=5)
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ color: '#94a3b8', borderBottom: '1px solid #334155' }}>
                      <th style={{ textAlign: 'left', padding: 4 }}>ID</th>
                      <th style={{ textAlign: 'right', padding: 4 }}>TG</th>
                      <th style={{ textAlign: 'right', padding: 4 }}>HDL</th>
                      <th style={{ textAlign: 'right', padding: 4 }}>LDL</th>
                      <th style={{ textAlign: 'center', padding: 4 }}>Pancreatitis</th>
                      <th style={{ textAlign: 'center', padding: 4 }}>Xanthoma</th>
                      <th style={{ textAlign: 'center', padding: 4 }}>CVD</th>
                    </tr>
                  </thead>
                  <tbody>
                    {b.sample_patients?.map((p, j) => (
                      <tr key={j} style={{ borderBottom: '1px solid #0f172a' }}>
                        <td style={{ padding: 4, color: '#64748b' }}>{p.patient_id}</td>
                        <td style={{ textAlign: 'right', padding: 4, color: p.tg_mg_dL > 2000 ? '#ef4444' : '#fbbf24' }}>{p.tg_mg_dL.toLocaleString()}</td>
                        <td style={{ textAlign: 'right', padding: 4 }}>{p.hdl_c_mg_dL}</td>
                        <td style={{ textAlign: 'right', padding: 4 }}>{p.ldl_c_mg_dL}</td>
                        <td style={{ textAlign: 'center', padding: 4 }}>{p.pancreatitis_episode ? '🔴' : '—'}</td>
                        <td style={{ textAlign: 'center', padding: 4 }}>{p.eruptive_xanthoma ? '🟡' : '—'}</td>
                        <td style={{ textAlign: 'center', padding: 4 }}>{p.cvd_event ? '❤️' : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ background: card, borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 12 }}>📖 Clinical Glossary</div>
            {Object.entries(definitions.glossary || {}).map(([term, def], i) => (
              <div key={i} style={{ borderBottom: '1px solid #1e293b', paddingBottom: 10, marginBottom: 10 }}>
                <div style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 13 }}>{term}</div>
                <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6, marginTop: 4 }}>{def}</div>
              </div>
            ))}
          </div>
          <div style={{ background: card, borderRadius: 10, padding: 16 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 12 }}>🧬 Full Gene Entries</div>
            {Object.entries(definitions.gene_entries || {}).map(([gene, entry], i) => (
              <div key={i} style={{ borderLeft: `4px solid ${GENE_COLORS[gene] || accent}`, paddingLeft: 12, marginBottom: 16 }}>
                <div style={{ fontWeight: 700, fontSize: 13 }}><GeneChip gene={gene} /> {entry.locus} · {entry.protein_size}</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{entry.inheritance?.substring(0, 200)}…</div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{entry.disease_category?.substring(0, 200)}…</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
