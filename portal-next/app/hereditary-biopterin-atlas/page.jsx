'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-biopterin-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'GCH1':   '#1a237e',  // deep indigo — most famous BH4 gene; AR: BH4def+HPA; AD: DRD/Segawa; diurnal fluctuation PATHOGNOMONIC
  'PTS':    '#880e4f',  // deep magenta — most common BH4 deficiency 60-75%; peripheral vs central subtype; CSF mandatory
  'QDPR':   '#b71c1c',  // deep red — BH4 recycling defect; folinic acid MANDATORY; basal ganglia calcification; secondary folate
  'PCBD1':  '#1b5e20',  // dark green — BENIGN transient HPA; primapterinuria PATHOGNOMONIC; no NT deficiency; no treatment
  'SPR':    '#e65100',  // deep orange — NBS ALWAYS MISSED; normal Phe; profoundly low CSF HVA+5HIAA; liver-brain asymmetry
  'DNAJC12':'#4a148c',  // deep purple — novel 2017; PAH+AADC cochaperone; BH4 NORMAL; HPA+NT deficiency; older panels miss
  'TH':     '#006064',  // dark teal — dopamine deficiency; BH4 normal; HVA low + 5-HIAA NORMAL; L-DOPA near-complete remission
  'DDC':    '#3e2723',  // dark brown — AADC; oculogyric crises PATHOGNOMONIC 55-85%; 3-OMD elevated; gene therapy Upstaza EU2022
};

const GENE_INFO = {
  'GCH1':   { full: 'GCH1 / GTPCH-I / GTP Cyclohydrolase I / 252aa', locus: '14q22.2', size: '252 aa / 28 kDa (homodecamer)', inh: 'AR/AD', disease: 'DUAL DISEASE — same gene: AR: severe BH4 deficiency + HPA + combined dopamine+serotonin deficiency — sapropterin + L-DOPA + 5-HTP + Phe restriction; AD: DOPA-RESPONSIVE DYSTONIA (DRD/Segawa disease, OMIM 128230) — haploinsufficiency; DIURNAL FLUCTUATION PATHOGNOMONIC (worse PM, better AM after sleep); L-DOPA DRAMATIC RESPONSE at low doses (near-complete remission); Phe NORMAL in AD-DRD (NBS negative); family history AD ~50%; must exclude DRD in ALL atypical cerebral palsy — treatable condition' },
  'PTS':    { full: 'PTS / PTPS / 6-Pyruvoyltetrahydropterin Synthase / 145aa', locus: '11q23.1', size: '145 aa / 16 kDa (homotrimer)', inh: 'AR', disease: 'MOST COMMON BH4 DEFICIENCY (60-75%); CRITICAL SUBTYPE DISTINCTION requiring CSF: PERIPHERAL (~30%): CSF HVA+5-HIAA NORMAL → sapropterin + diet ONLY; CENTRAL (~70%): CSF HVA+5-HIAA LOW → L-DOPA + 5-HTP + sapropterin MANDATORY; URINE PTERINS: neopterin VERY HIGH + biopterin VERY LOW (step 2 blocked → upstream neopterin accumulates); ALL PTS detected on NBS by elevated Phe; RULE: CSF BEFORE L-DOPA (L-DOPA raises HVA → masks diagnostic pattern); BH4 loading test: >30% Phe drop at 4-8h' },
  'QDPR':   { full: 'QDPR / DHPR / Dihydropteridine Reductase / 244aa', locus: '4p15.32', size: '244 aa / 25.7 kDa (homotetramer)', inh: 'AR', disease: 'BH4 RECYCLING DEFECT — qBH2 accumulates; SECONDARY FOLATE DEFICIENCY (qBH2 inhibits DHFR → CSF 5-MTHF critically low); FOLINIC ACID MANDATORY alongside BH4 — progressive neurodegeneration without it; BASAL GANGLIA CALCIFICATION (bilateral, symmetric, ~50%); DHPR DBS ASSAY: ZERO ACTIVITY = PATHOGNOMONIC (only BH4 gene with direct DBS enzyme assay); urine pterins: NORMAL biopterin + NORMAL neopterin (distinguishes from GCH1 and PTS); METHOTREXATE ABSOLUTE CI; FOLIC ACID WRONG — use FOLINIC ACID (bypasses blocked DHFR)' },
  'PCBD1':  { full: 'PCBD1 / PCD / DCoH / Pterin-4-Carbinolamine Dehydratase 1 / 104aa', locus: '10q22.1', size: '104 aa / 12 kDa (homotetramer)', inh: 'AR', disease: 'BENIGN TRANSIENT HPA — most important clinical fact: NO neurological sequelae; PRIMAPTERINURIA: 7-biopterin (primapterin) ELEVATED = PATHOGNOMONIC; DHPR DBS NORMAL (distinguishes from QDPR); CSF neurotransmitters NORMAL (no NT deficiency); HPA resolves spontaneously in weeks-months WITHOUT treatment; DUAL ROLE: PCD enzyme (pterin recycling) + DCoH (HNF1α transcription factor cofactor); severe mutations → HNF1A-like MODY; L-DOPA and 5-HTP CONTRAINDICATED (neurotransmitters normal — would cause iatrogenic excess); distinguish from PTS: primapterin HIGH vs neopterin HIGH in PTS' },
  'SPR':    { full: 'SPR / Sepiapterin Reductase / 261aa', locus: '2p13.2', size: '261 aa / 28 kDa (homodimer)', inh: 'AR', disease: 'NORMAL PHENYLALANINE — ALWAYS MISSED BY NBS: liver uses alternative carbonyl reductase 1 pathway → hepatic BH4 partial → PAH active → Phe NORMAL; brain has NO alternative → BH4 absent → TH/TPH fail → PROFOUND dopamine+serotonin deficiency; CSF HVA+5-HIAA PROFOUNDLY LOW; CSF sepiapterin ELEVATED (PATHOGNOMONIC for SPR); severe progressive dystonia + parkinsonism + oculomotor + intellectual disability; L-DOPA PARTIAL response (needs ALL THREE: L-DOPA + 5-HTP + sapropterin); KEY: child with dystonia + NORMAL Phe + profoundly low CSF NT → SUSPECT SPR' },
  'DNAJC12':{ full: 'DNAJC12 / DnaJ Cochaperone / PAH+AADC Cochaperone / 198aa', locus: '10q21.3', size: '198 aa / 23 kDa', inh: 'AR', disease: 'NOVEL 2017 (Blau et al.) — may be absent on older gene panels; BH4 COMPLETELY NORMAL (not a pterin gene — chaperone defect); mechanism: PAH + AADC misfolding due to HSP70 cochaperone loss; HPA (elevated Phe) + NT deficiency (CSF HVA+5-HIAA low) despite NORMAL pterins; BH4 loading test: VARIABLE PARTIAL response (PAH stabilisation by pharmacological chaperone); CRITICAL TRAP: partial BH4 response → misdiagnosed as BH4-responsive PKU → NT deficiency missed → progressive neurological injury; RULE: normal pterins + NT deficiency = DNAJC12 or TH or DDC; ensure DNAJC12 is on your HPA panel' },
  'TH':     { full: 'TH / Tyrosine Hydroxylase / Rate-Limiting Dopamine Synthesis / 498aa', locus: '11p15.5', size: '498 aa / 59 kDa (homotetramer)', inh: 'AR', disease: 'NOT a BH4 gene — BH4-DEPENDENT enzyme; BH4 NORMAL; Phe NORMAL (NBS negative); CSF FINGERPRINT: HVA VERY LOW + 5-HIAA NORMAL (isolated dopamine deficiency — serotonin intact because TPH uses normal BH4); BH4 NORMAL; TYPE A (DRD-B): diurnal dystonia (less fluctuation than GCH1-AD); TYPE B: infantile parkinsonism — neonatal/infancy onset, hypokinetic-rigid, ptosis, oculomotor; L-DOPA HIGHLY RESPONSIVE — near-complete remission even in severe type B; DO NOT confuse with GCH1-AD: TH = BH4 normal + 5-HIAA normal + AR; GCH1-AD = BH4 borderline + 5-HIAA mildly low + AD; include TH in all DRD/movement disorder panels' },
  'DDC':    { full: 'DDC / AADC / Aromatic L-Amino Acid Decarboxylase / 480aa', locus: '7p12.3', size: '480 aa / 54 kDa (homodimer)', inh: 'AR', disease: 'FINAL STEP both dopamine AND serotonin synthesis (convergence point); OCULOGYRIC CRISES PATHOGNOMONIC (55-85%) — episodic forced sustained upward eye deviation; CSF FINGERPRINT: HVA ABSENT + 5-HIAA ABSENT + 3-OMD (3-O-methyldopa) ELEVATED (L-DOPA methylated by COMT); AADC PLASMA ENZYME ACTIVITY: ZERO; PLP (pyridoxal-5-phosphate) dependent — pyridoxine trial in all; L-DOPA NOT HELPFUL (cannot convert L-DOPA); treatment: pyridoxine + selegiline (MAO-B inhibitor) + bromocriptine/pramipexole (dopamine agonists); GENE THERAPY: UPSTAZA (eladocagene exuparvovec) — EMA/MHRA approved 2022; bilateral putamen AAV2-hAADC injection; >80% OGC resolution; age 18m-6y eligibility' },
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

export default function HereditaryBiopterinAtlasPage() {
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
        🧬 Hereditary Biopterin (BH4) Metabolism Atlas
      </h1>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 16 }}>
        GCH1 · PTS · QDPR · PCBD1 · SPR · DNAJC12 · TH · DDC — 8 genes, 320 patients (8×40), seeds 2670–2677
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
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2670–2677" />
            <MetricCard label="Atlas Genes" value={overview.genes?.length} sub="BH4 synthesis · recycling · cochaperone · BH4-dependent" />
            <MetricCard label="NBS-Missed Genes" value={overview.nbs_gap_genes?.length} sub="SPR · TH · DDC — Phe NORMAL" warn />
            <MetricCard label="Gene Therapy" value={overview.gene_therapy_approved?.length} sub="DDC — Upstaza EU/UK 2022" />
          </div>

          <div style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h2 style={{ color: accent, fontSize: 15, marginBottom: 14 }}>BH4 Pathway Classification — 7 Categories</h2>
            <ol style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 2, paddingLeft: 18 }}>
              {overview.bh4_pathway_classification?.map((m, i) => <li key={i}>{m}</li>)}
            </ol>
          </div>

          <div style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h2 style={{ color: accent, fontSize: 15, marginBottom: 10 }}>Clinical Highlights</h2>
            <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 2 }}>
              <div><span style={{ color: '#4ade80', fontWeight: 700 }}>L-DOPA HIGHLY RESPONSIVE:</span> {overview.ldopa_highly_responsive?.join(', ')}</div>
              <div><span style={{ color: '#f87171', fontWeight: 700 }}>NBS GAP (Phe normal, always missed):</span> {overview.normal_phe_genes?.join(', ')}</div>
              <div><span style={{ color: '#fbbf24', fontWeight: 700 }}>FOLINIC ACID MANDATORY:</span> {overview.folinic_acid_mandatory?.join(', ')}</div>
              <div><span style={{ color: '#a78bfa', fontWeight: 700 }}>CSF MANDATORY GENES:</span> {overview.csf_mandatory_genes?.join(', ')}</div>
              <div><span style={{ color: '#38bdf8', fontWeight: 700 }}>GENE THERAPY APPROVED:</span> {overview.gene_therapy_approved?.join(', ')}</div>
            </div>
          </div>

          <div style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h2 style={{ color: accent, fontSize: 15, marginBottom: 14 }}>Gene Summary Statistics (40 patients each)</h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #334155' }}>
                    {['Gene', 'Onset (yr)', 'Phe Peak (µmol/L)', 'NBS Detected %', 'BH4 Resp %', 'CSF HVA Low %', 'CSF 5HIAA Low %', 'Dystonia %', 'OGC %', 'L-DOPA Resp %'].map(h => (
                      <th key={h} style={{ color: '#94a3b8', padding: '6px 8px', textAlign: 'left' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.map(s => (
                    <tr key={s.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                      <td style={{ padding: '6px 8px' }}><GeneChip gene={s.gene} /></td>
                      <td style={{ padding: '6px 8px', color: '#e2e8f0' }}>{s.mean_onset_years}</td>
                      <td style={{ padding: '6px 8px', color: s.mean_phe_umol_L > 400 ? '#ef4444' : s.mean_phe_umol_L < 70 ? '#4ade80' : '#fbbf24', fontWeight: 600 }}>{s.mean_phe_umol_L}</td>
                      <td style={{ padding: '6px 8px', color: s.nbs_detected_pct < 10 ? '#ef4444' : '#4ade80', fontWeight: s.nbs_detected_pct < 10 ? 700 : 400 }}>{s.nbs_detected_pct}%</td>
                      <td style={{ padding: '6px 8px', color: s.bh4_responsive_pct > 70 ? '#4ade80' : '#e2e8f0', fontWeight: 600 }}>{s.bh4_responsive_pct}%</td>
                      <td style={{ padding: '6px 8px', color: s.csf_hva_low_pct > 70 ? '#fbbf24' : '#e2e8f0', fontWeight: s.csf_hva_low_pct > 70 ? 700 : 400 }}>{s.csf_hva_low_pct}%</td>
                      <td style={{ padding: '6px 8px', color: s.csf_5hiaa_low_pct > 70 ? '#fbbf24' : '#e2e8f0' }}>{s.csf_5hiaa_low_pct}%</td>
                      <td style={{ padding: '6px 8px', color: s.dystonia_movement_disorder_pct > 80 ? '#a78bfa' : '#e2e8f0' }}>{s.dystonia_movement_disorder_pct}%</td>
                      <td style={{ padding: '6px 8px', color: s.oculogyric_crises_pct > 50 ? '#ef4444' : '#e2e8f0', fontWeight: s.oculogyric_crises_pct > 50 ? 700 : 400 }}>{s.oculogyric_crises_pct}%</td>
                      <td style={{ padding: '6px 8px', color: s.ldopa_responsive_pct > 80 ? '#4ade80' : '#e2e8f0', fontWeight: 600 }}>{s.ldopa_responsive_pct}%</td>
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
          {(selGene ? [selGene] : Object.keys(GENE_INFO)).map(gn => {
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
                  <div style={{ color: entry.mean_phe_umol_L < 70 ? '#4ade80' : entry.mean_phe_umol_L > 400 ? '#ef4444' : '#fbbf24', fontSize: 18, fontWeight: 700 }}>{entry.mean_phe_umol_L}</div>
                  <div style={{ color: '#64748b', fontSize: 11 }}>Phe µmol/L (mean)</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 110 }}>
                  <div style={{ color: entry.nbs_detected_pct < 10 ? '#ef4444' : '#4ade80', fontSize: 18, fontWeight: 700 }}>{entry.nbs_detected_pct}%</div>
                  <div style={{ color: '#64748b', fontSize: 11 }}>NBS Detected</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 110 }}>
                  <div style={{ color: entry.csf_hva_low_pct > 70 ? '#fbbf24' : '#e2e8f0', fontSize: 18, fontWeight: 700 }}>{entry.csf_hva_low_pct}%</div>
                  <div style={{ color: '#64748b', fontSize: 11 }}>CSF HVA Low</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 110 }}>
                  <div style={{ color: entry.csf_5hiaa_low_pct > 70 ? '#fbbf24' : '#e2e8f0', fontSize: 18, fontWeight: 700 }}>{entry.csf_5hiaa_low_pct}%</div>
                  <div style={{ color: '#64748b', fontSize: 11 }}>CSF 5-HIAA Low</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 110 }}>
                  <div style={{ color: entry.oculogyric_crises_pct > 50 ? '#ef4444' : '#e2e8f0', fontSize: 18, fontWeight: 700 }}>{entry.oculogyric_crises_pct}%</div>
                  <div style={{ color: '#64748b', fontSize: 11 }}>Oculogyric Crises</div>
                </div>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 110 }}>
                  <div style={{ color: entry.ldopa_responsive_pct > 80 ? '#4ade80' : '#e2e8f0', fontSize: 18, fontWeight: 700 }}>{entry.ldopa_responsive_pct}%</div>
                  <div style={{ color: '#64748b', fontSize: 11 }}>L-DOPA Resp</div>
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
            <h2 style={{ color: accent, fontSize: 15, marginBottom: 14 }}>BH4 Metabolism Glossary</h2>
            {Object.entries(definitions.bh4_glossary || {}).map(([term, def]) => (
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
