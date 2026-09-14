'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-chi-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ABCC8':   '#1a237e',  // deep indigo — SUR1 KATP; most common CHI; diazoxide-unresponsive (AR); focal vs diffuse; 18F-DOPA PET
  'KCNJ11':  '#880e4f',  // deep magenta — Kir6.2; CHI (LOF) + NDM/DEND (GOF); sulphonylurea switch for NDM
  'HADH':    '#4a148c',  // deep purple — SCHAD; protein-sensitive CHI; C4-OH acylcarnitine; NO hyperammonaemia; GDH inhibitor
  'GLUD1':   '#b71c1c',  // deep red — GDH GOF; HI/HA syndrome; hyperammonaemia; leucine-sensitive; valproate CI
  'GCK':     '#1b5e20',  // dark green — glucokinase set-point CHI; diazoxide-responsive; AD; MODY2 same gene LOF
  'HNF4A':   '#e65100',  // deep orange — macrosomia + CHI in infancy → MODY1 in adulthood; dual phenotype
  'INSR':    '#006064',  // dark teal — Donohue/Leprechaunism; extreme hyperinsulinaemia; acanthosis nigricans; elfin facies
  'SLC16A1': '#3e2723',  // dark brown — MCT1 ectopic; EIHI; exercise-triggered ONLY; normal at rest; anaerobic provocation
};

const GENE_INFO = {
  'ABCC8':   { full: 'ABCC8 / Sulphonylurea Receptor 1 (SUR1) / 1581aa', locus: '11p15.1', size: '1581 aa / 177 kDa', inh: 'AR/AD', disease: 'Most common CHI (40-45%); KATP channel regulatory subunit; AR biallelic = severe diffuse/focal CHI, diazoxide-UNRESPONSIVE; AD monoallelic = milder, often diazoxide-responsive; FOCAL CHI: 18F-DOPA PET-CT gold standard; somatic 11p15 LOH + paternal germline ABCC8 → focal clone surgically curable (limited resection); DIFFUSE: near-total pancreatectomy; c.3992-9G>A Ashkenazi founder; HIGH GLUCOSE INFUSION RATE >8-10 mg/kg/min hallmark' },
  'KCNJ11':  { full: 'KCNJ11 / Inward Rectifier K+ Channel Kir6.2 / 390aa', locus: '11p15.1', size: '390 aa / 43 kDa', inh: 'AR/AD', disease: 'DUAL DISEASE: LOF (biallelic AR) → CHI (identical to ABCC8); GOF (dominant activating) → NDM/DEND syndrome; DEND = Developmental delay + Epilepsy + Neonatal Diabetes; sulphonylurea switch from insulin → DRAMATIC improvement in NDM + neurological features; p.R201H, p.V59M common DEND; screen KCNJ11+ABCC8 in ALL NDM <6 months before committing to insulin' },
  'HADH':    { full: 'HADH / Short-Chain L-3-Hydroxyacyl-CoA Dehydrogenase (SCHAD) / 314aa', locus: '4q25', size: '314 aa / 34 kDa', inh: 'AR', disease: 'SCHAD deficiency; protein-sensitive CHI; unique mechanism — HADH physically inhibits GDH (GLUD1); LOF → unregulated GDH → excess insulin after protein meals; PROTEIN-SENSITIVE hypoglycaemia; C4-OH (3-hydroxybutyrylcarnitine) elevated on acylcarnitine panel; DIAZOXIDE-RESPONSIVE (KATP intact); NO hyperammonaemia (DDx from GLUD1); low-protein diet + diazoxide; good prognosis' },
  'GLUD1':   { full: 'GLUD1 / Glutamate Dehydrogenase 1 (GDH) / 558aa', locus: '10q23.3', size: '558 aa / 56 kDa (hexamer)', inh: 'AD GOF', disease: 'HI/HA SYNDROME: hyperinsulinism + hyperammonaemia (PATHOGNOMONIC combination; no other CHI cause produces both); GDH constitutively overactive; protein-sensitive + leucine-sensitive; ammonia 50-200 μmol/L (usually asymptomatic); diazoxide-responsive; EPILEPSY in ~50%; VALPROATE ABSOLUTELY CONTRAINDICATED (elevates ammonia); protein aversion behaviour clue; de novo or familial AD' },
  'GCK':     { full: 'GCK / Glucokinase (Beta-Cell Glucose Sensor) / 465aa', locus: '7p13', size: '465 aa / 52 kDa', inh: 'AD GOF/AR LOF', disease: 'DUAL PHENOTYPE: GOF activating → CHI (set-point shifted left, fasting glucose 2.5-3.5 mmol/L); heterozygous LOF → MODY2 (fasting glucose 5.4-8.3 mmol/L, benign, no treatment); homozygous LOF → permanent NDM; GCK GOF CHI diazoxide-responsive; usually diffuse; AD family history; some cases remit; MODY2 DDx: fasting glucose elevated (MODY2) vs fasting glucose LOW (GOF CHI) — same gene, opposite disease' },
  'HNF4A':   { full: 'HNF4A / Hepatocyte Nuclear Factor 4 Alpha / 474aa', locus: '20q13.12', size: '474 aa / 53 kDa', inh: 'AD', disease: 'DUAL TEMPORAL PHENOTYPE: neonatal/infantile macrosomia + diazoxide-responsive CHI → spontaneous resolution → adult MODY1 (progressive insulin secretory defect); MACROSOMIA +500-600g birthweight — clue for HNF4A/HNF1A CHI; CHI typically resolves by 1 year; MODY1 onset 20-50yr; hepatic abnormalities (elevated transaminases, low LDL/apoB); renal Fanconi in some; sulphonylurea highly effective for MODY1 phase; screen family' },
  'INSR':    { full: 'INSR / Insulin Receptor Tyrosine Kinase / 1382aa', locus: '19p13.2', size: '1382 aa / 155 kDa (α2β2)', inh: 'AR/AD', disease: 'SEVERITY SPECTRUM: AR severe LOF = Donohue syndrome (Leprechaunism) — elfin facies, extreme growth retardation, insulin >1000-100,000 mU/L PATHOGNOMONIC, paradoxical post-prandial hypoglycaemia + fasting hyperglycaemia; Rabson-Mendenhall = moderate; Type A = mild (young women, PCOS-like, acanthosis); ACANTHOSIS NIGRICANS universal; IGF-1 (mecasermin) partially bypasses INSR; glucose infusion can PARADOXICALLY WORSEN hypoglycaemia' },
  'SLC16A1': { full: 'SLC16A1 / Monocarboxylate Transporter 1 (MCT1) / 465aa', locus: '1p13.2', size: '465 aa / 43 kDa', inh: 'AD', disease: 'EIHI (exercise-induced hyperinsulinism); <30 families worldwide; MCT1 normally NOT expressed in beta-cells; promoter/regulatory mutations → ectopic MCT1 in beta-cells; anaerobic exercise → blood pyruvate/lactate rises → enters beta-cell via MCT1 → ATP → insulin spike → hypoglycaemia; NORMAL standard CHI workup at rest; ANAEROBIC EXERCISE PROVOCATION TEST required for diagnosis; avoid intense exercise; pre-exercise carbohydrate; diazoxide partial effect' },
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

export default function HereditoryCHIAtlasPage() {
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
      <h1 style={{ fontSize: 22, fontWeight: 700, color: accent, marginBottom: 4 }}>
        🧬 Hereditary Congenital Hyperinsulinism Atlas
      </h1>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 16 }}>
        ABCC8 · KCNJ11 · HADH · GLUD1 · GCK · HNF4A · INSR · SLC16A1 — 8 genes, 320 patients (8×40), seeds 2654–2661
      </p>

      {/* Gene chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 20 }}>
        {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? accent : card,
            color: tab === t ? '#0f172a' : '#94a3b8',
            border: 'none', borderRadius: 6, padding: '7px 16px',
            fontWeight: 600, fontSize: 13, cursor: 'pointer',
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#94a3b8' }}>Loading…</div>}
      {err && <div style={{ color: '#ef4444' }}>Error: {err}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2654–2661" />
            <MetricCard label="Genes" value={overview.genes?.length} sub="KATP · GDH · SCHAD · GCK · HNF4A · INSR · MCT1" />
            <MetricCard label="Diazoxide Responders" value={overview.diazoxide_responders?.length} sub="HADH GLUD1 GCK HNF4A SLC16A1" />
            <MetricCard label="Diazoxide NON-Responders" value={overview.diazoxide_non_responders?.length} sub="ABCC8(AR) KCNJ11(AR) INSR" warn />
          </div>

          <div style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h2 style={{ color: accent, fontSize: 15, marginBottom: 14 }}>CHI Mechanisms — 7 Distinct Pathways</h2>
            <ol style={{ color: '#cbd5e1', fontSize: 13, lineHeight: 1.9, paddingLeft: 18 }}>
              {overview.chi_mechanisms?.map((m, i) => <li key={i}>{m}</li>)}
            </ol>
          </div>

          <div style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h2 style={{ color: accent, fontSize: 15, marginBottom: 14 }}>Gene Summary Statistics (40 patients each)</h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #334155' }}>
                    {['Gene', 'Onset (yr)', 'Insulin Peak (mU/L)', 'Macrosomia %', 'Diazoxide Resp %', 'Hyperammon %', 'Exercise-Trig %', 'Surgery %'].map(h => (
                      <th key={h} style={{ color: '#94a3b8', padding: '6px 8px', textAlign: 'left' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.map(s => (
                    <tr key={s.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                      <td style={{ padding: '6px 8px' }}><GeneChip gene={s.gene} /></td>
                      <td style={{ padding: '6px 8px', color: '#e2e8f0' }}>{s.mean_onset_years}</td>
                      <td style={{ padding: '6px 8px', color: s.gene === 'INSR' ? '#ef4444' : '#e2e8f0', fontWeight: s.gene === 'INSR' ? 700 : 400 }}>{s.mean_insulin_peak_mU_L}</td>
                      <td style={{ padding: '6px 8px', color: s.macrosomia_pct > 60 ? '#fbbf24' : '#e2e8f0' }}>{s.macrosomia_pct}%</td>
                      <td style={{ padding: '6px 8px', color: s.diazoxide_responsive_pct < 40 ? '#ef4444' : '#4ade80', fontWeight: 600 }}>{s.diazoxide_responsive_pct}%</td>
                      <td style={{ padding: '6px 8px', color: s.hyperammonaemia_pct > 80 ? '#fbbf24' : '#e2e8f0', fontWeight: s.hyperammonaemia_pct > 80 ? 700 : 400 }}>{s.hyperammonaemia_pct}%</td>
                      <td style={{ padding: '6px 8px', color: s.exercise_triggered_pct > 80 ? '#a78bfa' : '#e2e8f0', fontWeight: s.exercise_triggered_pct > 80 ? 700 : 400 }}>{s.exercise_triggered_pct}%</td>
                      <td style={{ padding: '6px 8px', color: s.surgery_required_pct > 40 ? '#ef4444' : '#e2e8f0' }}>{s.surgery_required_pct}%</td>
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
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
            {Object.keys(GENE_INFO).map(g => (
              <button key={g} onClick={() => setSelGene(selGene === g ? null : g)} style={{
                background: selGene === g ? (GENE_COLORS[g] || '#555') : card,
                color: '#fff', border: 'none', borderRadius: 6,
                padding: '5px 12px', fontSize: 12, fontWeight: 700, cursor: 'pointer',
              }}>{g}</button>
            ))}
          </div>
          {(selGene ? [GENE_INFO[selGene]] : Object.entries(GENE_INFO).map(([k, v]) => ({ gene: k, ...v }))).map((info, idx) => {
            const gene = selGene || Object.keys(GENE_INFO)[idx];
            const gi = selGene ? info : info;
            const gn = selGene ? selGene : gene;
            const gdata = GENE_INFO[gn];
            const col = GENE_COLORS[gn] || '#555';
            return (
              <div key={gn} style={{ background: card, borderLeft: `4px solid ${col}`, borderRadius: 8, padding: 18, marginBottom: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                  <GeneChip gene={gn} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{gdata.full}</span>
                  <span style={{ color: '#64748b', fontSize: 11 }}>{gdata.locus} · {gdata.size} · {gdata.inh}</span>
                </div>
                <p style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.8, margin: 0 }}>{gdata.disease}</p>
              </div>
            );
          })}
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {breakdown.gene_entries?.map(entry => (
            <div key={entry.gene} style={{ background: card, borderLeft: `4px solid ${GENE_COLORS[entry.gene] || '#555'}`, borderRadius: 8, padding: 18, marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                <GeneChip gene={entry.gene} />
                <span style={{ color: '#94a3b8', fontSize: 12 }}>{entry.locus} · {entry.protein_size} · {entry.gene}</span>
              </div>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 14 }}>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 110 }}>
                  <div style={{ color: '#38bdf8', fontSize: 18, fontWeight: 700 }}>{entry.macrosomia_pct}%</div>
                  <div style={{ color: '#64748b', fontSize: 11 }}>Macrosomia</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 110 }}>
                  <div style={{ color: entry.diazoxide_responsive_pct < 40 ? '#ef4444' : '#4ade80', fontSize: 18, fontWeight: 700 }}>{entry.diazoxide_responsive_pct}%</div>
                  <div style={{ color: '#64748b', fontSize: 11 }}>Diazoxide Resp</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 110 }}>
                  <div style={{ color: entry.hyperammonaemia_pct > 70 ? '#fbbf24' : '#e2e8f0', fontSize: 18, fontWeight: 700 }}>{entry.hyperammonaemia_pct}%</div>
                  <div style={{ color: '#64748b', fontSize: 11 }}>Hyperammon</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 110 }}>
                  <div style={{ color: entry.exercise_triggered_pct > 70 ? '#a78bfa' : '#e2e8f0', fontSize: 18, fontWeight: 700 }}>{entry.exercise_triggered_pct}%</div>
                  <div style={{ color: '#64748b', fontSize: 11 }}>Exercise-Trig</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 110 }}>
                  <div style={{ color: '#e2e8f0', fontSize: 18, fontWeight: 700 }}>{entry.epilepsy_pct}%</div>
                  <div style={{ color: '#64748b', fontSize: 11 }}>Epilepsy</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 110 }}>
                  <div style={{ color: entry.surgery_required_pct > 40 ? '#ef4444' : '#e2e8f0', fontSize: 18, fontWeight: 700 }}>{entry.surgery_required_pct}%</div>
                  <div style={{ color: '#64748b', fontSize: 11 }}>Surgery Req</div>
                </div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#fbbf24', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>🔬 Pathognomonic / Diagnosis</div>
                <p style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.8, margin: 0 }}>{entry.pathognomonic}</p>
              </div>
              <div>
                <div style={{ color: '#4ade80', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>💊 Treatment</div>
                <p style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.8, margin: 0 }}>{entry.treatment}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h2 style={{ color: accent, fontSize: 15, marginBottom: 14 }}>Per-Gene Pathognomonic Summary</h2>
            {Object.entries(definitions.gene_entries || {}).map(([g, text]) => (
              <div key={g} style={{ marginBottom: 16, borderLeft: `3px solid ${GENE_COLORS[g] || '#555'}`, paddingLeft: 12 }}>
                <GeneChip gene={g} />
                <p style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.8, marginTop: 6 }}>{text}</p>
              </div>
            ))}
          </div>

          <div style={{ background: card, borderRadius: 10, padding: 20 }}>
            <h2 style={{ color: accent, fontSize: 15, marginBottom: 14 }}>CHI Glossary</h2>
            {Object.entries(definitions.chi_glossary || {}).map(([term, def]) => (
              <div key={term} style={{ marginBottom: 20, borderBottom: '1px solid #334155', paddingBottom: 16 }}>
                <div style={{ color: '#fbbf24', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{term}</div>
                <p style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.8, margin: 0 }}>{def}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
