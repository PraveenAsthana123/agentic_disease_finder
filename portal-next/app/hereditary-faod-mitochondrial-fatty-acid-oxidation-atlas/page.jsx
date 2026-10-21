'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-faod-mitochondrial-fatty-acid-oxidation-atlas';
const TABS = ['Overview', 'Gene Table', 'Metabolic Groups', 'Definitions'];

const GENE_COLORS = {
  'ACADM':    '#1565c0',  // deep blue       — AR; MCAD; most common FAOD
  'ACADVL':   '#6a1b9a',  // deep purple     — AR; VLCAD; cardiac/myopathic
  'HADHA':    '#880e4f',  // dark magenta    — AR; LCHAD; retinopathy+AFLP
  'CPT1A':    '#1b5e20',  // dark green      — AR; CPT1A; liver; Arctic P479L
  'CPT2':     '#e65100',  // deep orange     — AR; CPT2; rhabdomyolysis; anesthesia CI
  'SLC25A20': '#b71c1c',  // deep red        — AR; CACT; neonatal; hyperammonemia
  'HADHB':    '#4a148c',  // darker purple   — AR; MTP-beta; complete MTP
  'HADH':     '#00695c',  // dark teal       — AR; SCHAD; hyperinsulinism
};

const GENE_INFO = {
  'ACADM':    { full: 'ACADM / Medium-Chain Acyl-CoA Dehydrogenase / MCAD / 421aa', locus: '1p31.1',  size: '421 aa / 47 kDa (homotetrameric FAD-linked; C6-C12 acyl-CoA; most common FAOD; K304E >50% European alleles; C8↑ NBS; hexanoylglycine+suberylglycine urine OA; SUDI in 20-25% undiagnosed; avoid fasting; no fat restriction)',                                                inh: 'AR LOF' },
  'ACADVL':   { full: 'ACADVL / Very-Long-Chain Acyl-CoA Dehydrogenase / VLCAD / 655aa', locus: '17p13.1', size: '655 aa / 73 kDa (homodimer inner MMM; C14-C20 acyl-CoA; C14:1↑ NBS; 3 forms: neonatal HCM / infantile / adult rhabdomyolysis; MCT oil bypasses block; riboflavin minor benefit)',                                                                                    inh: 'AR LOF' },
  'HADHA':    { full: 'HADHA / LCHAD-MTP Alpha-Subunit / 763aa',                    locus: '3p25.3',  size: '763 aa / 83 kDa (MTP alpha = enoyl-CoA hydratase + LCHAD; c.1528G>A = p.E510Q = isolated LCHAD; C16-OH+C18-OH↑; retinopathy+neuropathy PATHOGNOMONIC; maternal AFLP; DHA slows retinopathy)',                                                                              inh: 'AR LOF' },
  'CPT1A':    { full: 'CPT1A / Carnitine Palmitoyltransferase 1A / Liver / 773aa',  locus: '11q13.3', size: '773 aa / 88 kDa (outer MMM; liver isoform; rate-limiting LCFAO step; C0/(C16+C18) >100 PATHOGNOMONIC; NO cardiac disease; Arctic P479L cold-adaptive; do not give L-carnitine)',                                                                                           inh: 'AR LOF' },
  'CPT2':     { full: 'CPT2 / Carnitine Palmitoyltransferase 2 / 658aa',            locus: '1p32.3',  size: '658 aa / 74 kDa (inner MMM; S113L = adult myopathic; C16+C18↑; 3 forms: neonatal lethal brain dysgenesis / infantile / adult rhabdomyolysis; volatile anesthetics+succinylcholine CI; bezafibrate)',                                                                       inh: 'AR LOF' },
  'SLC25A20': { full: 'SLC25A20 / CACT / Carnitine Acylcarnitine Translocase / 301aa', locus: '3p21.31', size: '301 aa / 33 kDa (inner MMM antiporter; acylcarnitine in ↔ free carnitine out; C16+C18 MARKEDLY↑ + C0 LOW; secondary hyperammonemia; neonatal cardiomyopathy; IV carnitine + NH3 scavengers emergency)',                                                                inh: 'AR LOF' },
  'HADHB':    { full: 'HADHB / MTP Beta-Subunit / Thiolase / 474aa',               locus: '2p23.3',  size: '474 aa / 51 kDa (MTP beta = long-chain 3-ketoacyl-CoA thiolase; complete MTP deficiency all 3 activities; C16-OH+C18-OH↑; retinopathy milder than LCHAD; maternal AFLP; DHA + MCT)',                                                                                       inh: 'AR LOF' },
  'HADH':     { full: 'HADH / SCHAD / Short-Chain 3-Hydroxyacyl-CoA Dehydrogenase / 261aa', locus: '4q25',   size: '261 aa / 27 kDa (mitochondrial homotetrameric; C4-C6; INHIBITS GDH; LOF → unrestrained GDH → hyperinsulinism; C4-OH↑; 3-OH-glutaric aciduria; diazoxide-responsive; no hyperammonemia)',                                                                           inh: 'AR LOF' },
};

const PATHOGNOMONIC = {
  'ACADM':    'MOST COMMON FAOD in Caucasians (1/10,000–15,000). C8 acylcarnitine NBS + hexanoylglycine + suberylglycine + phenylpropionylglycine (urine OA TRIAD). SUDI in 20-25% undiagnosed first event. K304E >50% European alleles. EXCELLENT prognosis with NBS + fasting avoidance. No fat restriction.',
  'ACADVL':   'C14:1 acylcarnitine NBS marker. 3 FORMS: neonatal severe (HCM + hypoglycemia), infantile hepatic, adult myopathic rhabdomyolysis (most common in NBS era). Cardiomyopathy in neonatal/infantile. MCT oil supplements (bypasses VLCAD block). Monitor echo annually. Riboflavin 100 mg/day (minor benefit).',
  'HADHA':    'ONLY FAOD causing RETINOPATHY + PERIPHERAL NEUROPATHY. Maternal AFLP/HELLP in CARRIER mothers of LCHAD-deficient fetuses. C16-OH + C18-OH + C18:1-OH (3-hydroxy species). c.1528G>A (p.E510Q) = isolated LCHAD. DHA supplementation slows retinopathy. MCT oil. Ophthalmologic monitoring MANDATORY (ERG + fundus annually).',
  'CPT1A':    'UNIQUE inverted acylcarnitine: C0↑ + C16/C18↓ → C0/(C16+C18) >100 = PATHOGNOMONIC. No cardiac disease (liver isoform only). Arctic P479L: most homozygous Inuit/Indigenous ASYMPTOMATIC (cold-adaptive). DO NOT give L-carnitine. Hepatomegaly + hypoketotic hypoglycemia only.',
  'CPT2':     'MOST COMMON LCFAO transport disorder. Adult S113L: rhabdomyolysis from exercise/fever/cold/anesthesia. CK >10,000–100,000 IU/L. VOLATILE ANESTHETICS + SUCCINYLCHOLINE ABSOLUTELY CONTRAINDICATED → total IV anesthesia. BEZAFIBRATE 400mg/day reduces attacks. Neonatal form: BRAIN DYSGENESIS (neuronal migration defect) PATHOGNOMONIC.',
  'SLC25A20': 'Markedly elevated C16+C18 + LOW C0 (carnitine trap) + SECONDARY HYPERAMMONEMIA (reduced CPS1 → NH3 200-400 µmol/L; mimics urea cycle disorder). Neonatal cardiomyopathy. EMERGENCY: IV dextrose + IV L-carnitine 100-400 mg/kg/day + NH3 scavengers. Highest long-chain acylcarnitines of all FAOD.',
  'HADHB':    'Complete MTP deficiency (all 3 activities: enoyl-CoA hydratase + LCHAD + thiolase). Retinopathy present (milder than isolated LCHAD). Maternal AFLP/HELLP in carrier mothers. C16-OH+C18-OH+C18:1-OH. Distinguish from HADHA by: HADHB = thiolase activity also absent; HADHA c.1528G>A = LCHAD-specific.',
  'HADH':     'ONLY FAOD causing HYPERINSULINISM (not hypoketotic hypoglycemia from FAO block). SCHAD inhibits GDH — LOF → unrestrained GDH → insulin secretion. C4-OH↑ + urine 3-OH-glutaric acid. NO hyperammonemia (distinguishes from GLUD1 HI/HA). DIAZOXIDE-RESPONSIVE. Leucine-sensitive. No cardiac/retinal/neurologic involvement.',
};

export default function HereditaryFAODAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#0d1b2a', minHeight: '100vh', fontSize: 18 }}>Loading Hereditary-FAOD-Mitochondrial-Fatty-Acid-Oxidation-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#0d1b2a', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#e6edf3', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1565c0 0%,#6a1b9a 25%,#880e4f 50%,#e65100 75%,#00695c 100%)', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>Hereditary Fatty Acid Oxidation Disorders · Mitochondrial Beta-Oxidation · Carnitine Transport · Newborn Screening</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
          🧬 Hereditary-FAOD-Mitochondrial-Fatty-Acid-Oxidation-Atlas
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
          Complete 8-Gene Reference · ACADM · ACADVL · HADHA · CPT1A · CPT2 · SLC25A20 · HADHB · HADH ·
          320 Patients (8×40) · Seeds 4118–4125
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Total Patients', val: overview?.total_patients },
            { label: 'Genes', val: overview?.genes?.length },
            { label: 'Most Common', val: 'MCAD (ACADM)' },
            { label: 'Only FAOD + Retinopathy', val: 'LCHAD (HADHA)' },
            { label: 'Only FAOD + Hyperinsulinism', val: 'SCHAD (HADH)' },
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
              <h3 style={{ margin: '0 0 14px', color: '#58a6ff', fontSize: 16 }}>⚡ Key Clinical Pearls — Fatty Acid Oxidation Disorders</h3>
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
                          ['Hypoketotic Hypo', cs?.hypo_pct ? `${cs.hypo_pct}%` : (cs?.hyperinsulinism_flag ? '—' : '—')],
                          ['Cardiomyopathy', cs?.cardiomyopathy ? `${cs?.cardiac_pct}%` : '—'],
                          ['Rhabdomyolysis', cs?.rhabdo_pct > 0 ? `${cs.rhabdo_pct}%` : '—'],
                          ['Retinopathy', cs?.retinopathy_flag ? `${cs?.retinopathy_pct}%` : '—'],
                          ['Hyperinsulinism', cs?.hyperinsulinism_flag ? `${cs?.hyperinsulinism_pct}%` : '—'],
                          ['Maternal AFLP', cs?.maternal_aflp_flag ? `${cs?.maternal_aflp_pct}%` : '—'],
                        ].map(([lbl, val]) => (
                          <div key={lbl} style={{ background: '#0d1117', borderRadius: 6, padding: '5px 8px' }}>
                            <div style={{ fontSize: 10, color: '#8b949e' }}>{lbl}</div>
                            <div style={{ fontSize: 13, fontWeight: 600, color: val === '—' ? '#444' : '#e6edf3' }}>{val}</div>
                          </div>
                        ))}
                      </div>
                      <div style={{ background: '#0d1117', borderRadius: 6, padding: '8px 10px', fontSize: 12, color: '#e6edf3', lineHeight: 1.5, borderLeft: `3px solid ${color}` }}>
                        {PATHOGNOMONIC[gene]}
                      </div>
                      <div style={{ marginTop: 8, fontSize: 11, color: '#8b949e' }}>
                        NBS marker: <b style={{ color: '#58a6ff' }}>{cs?.nbs_marker}</b> ·
                        Key mutation: <b>{cs?.key_mutation?.split(';')[0]}</b>
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
                  {['Gene', 'Size', 'Locus', 'Inheritance', 'NBS Marker', 'Hypoketotic Hypo', 'Cardiac', 'Rhabdo', 'Retinopathy', 'HI', 'Neonatal Lethal', 'Exercise Trigger', 'Maternal AFLP', 'Mean Onset (m)'].map(h => (
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
                      <td style={{ padding: '9px 12px', color: '#8b949e', fontSize: 12 }}>{cs?.size_aa} aa</td>
                      <td style={{ padding: '9px 12px', color: '#8b949e', fontSize: 12 }}>{cs?.locus}</td>
                      <td style={{ padding: '9px 12px', color: '#e6edf3', fontSize: 12 }}>{cs?.inheritance}</td>
                      <td style={{ padding: '9px 12px', color: '#58a6ff', fontSize: 12, fontWeight: 600 }}>{cs?.nbs_marker}</td>
                      <td style={{ padding: '9px 12px', color: cs?.hypo_pct > 0 ? '#3fb950' : (cs?.hyperinsulinism_flag ? '#00695c' : '#444'), fontWeight: cs?.hypo_pct > 0 ? 700 : 400 }}>{cs?.hypo_pct > 0 ? `${cs.hypo_pct}%` : (cs?.hyperinsulinism_flag ? 'HI' : '—')}</td>
                      <td style={{ padding: '9px 12px', color: cs?.cardiomyopathy ? '#f0883e' : '#444', fontWeight: cs?.cardiomyopathy ? 700 : 400 }}>{cs?.cardiomyopathy ? `⚠ ${cs.cardiac_pct}%` : '—'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.rhabdo_pct > 0 ? '#f85149' : '#444', fontWeight: cs?.rhabdo_pct > 0 ? 700 : 400 }}>{cs?.rhabdo_pct > 0 ? `${cs.rhabdo_pct}%` : '—'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.retinopathy_flag ? '#e040fb' : '#444', fontWeight: cs?.retinopathy_flag ? 700 : 400 }}>{cs?.retinopathy_flag ? `✅ ${cs.retinopathy_pct}%` : '—'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.hyperinsulinism_flag ? '#00e5ff' : '#444', fontWeight: cs?.hyperinsulinism_flag ? 700 : 400 }}>{cs?.hyperinsulinism_flag ? '✅ HI' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.neonatal_lethal ? '#f85149' : '#444', fontWeight: cs?.neonatal_lethal ? 700 : 400 }}>{cs?.neonatal_lethal ? '⚠ LETHAL' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.exercise_trigger ? '#f0883e' : '#444', fontWeight: cs?.exercise_trigger ? 700 : 400 }}>{cs?.exercise_trigger ? '⚠ Yes' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.maternal_aflp_flag ? '#f0883e' : '#444', fontWeight: cs?.maternal_aflp_flag ? 700 : 400 }}>{cs?.maternal_aflp_flag ? '⚠ AFLP' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: '#e6edf3' }}>{cs?.mean_age_onset_m}</td>
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

        {/* METABOLIC GROUPS TAB */}
        {activeTab === 'Metabolic Groups' && breakdown && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(380px,1fr))', gap: 20, marginBottom: 28 }}>
              {Object.entries(breakdown.metabolic_groups || {}).map(([groupKey, group]) => (
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
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Hypo: <b>{gs.hypo_pct}%</b></div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Cardiac: <b>{gs.cardiac_pct}%</b></div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Rhabdo: <b>{gs.rhabdo_pct}%</b></div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Retinopathy: <b>{gs.retinopathy_pct}%</b></div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>HI: <b>{gs.hyperinsulinism_pct}%</b></div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Mean onset: <b>{gs.mean_age_onset_m} mo</b></div>
                  <div style={{ fontSize: 11, color: '#8b949e', marginTop: 4 }}>NBS: {gs.nbs_marker}</div>
                  {gs.exercise_trigger && <div style={{ fontSize: 11, color: '#f0883e', marginTop: 2 }}>⚠ Exercise trigger</div>}
                  {gs.maternal_aflp && <div style={{ fontSize: 11, color: '#f0883e', marginTop: 2 }}>⚠ Maternal AFLP</div>}
                  {gs.diazoxide_rx && <div style={{ fontSize: 11, color: '#00695c', marginTop: 2 }}>💊 Diazoxide Rx</div>}
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
