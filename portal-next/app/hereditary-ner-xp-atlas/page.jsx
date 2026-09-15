'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ner-xp-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'XPA':   '#b71c1c',  // deep red — XP-A most severe; both GGR+TCR abolished; neurodegeneration; De Sanctis-Cacchione
  'ERCC3': '#6a1b9a',  // deep purple — XPB/TFIIH; TTD tiger-tail; XP+CS overlap; ultraorphan
  'XPC':   '#0277bd',  // deep blue — most common non-Japanese; GGR only; NO neurodegeneration; skin cancer
  'ERCC2': '#1565c0',  // blue — XPD/TFIIH; widest spectrum: XP/CS/TTD; ERCC2 most common TTD gene
  'DDB2':  '#2e7d32',  // dark green — XP-E mildest; CPD sensor; UDS 30-50% normal; later cancer
  'ERCC4': '#e65100',  // deep orange — XPF-ERCC1; FANCQ allelic; bone marrow failure; ICL repair; cisplatin CI
  'ERCC5': '#37474f',  // dark slate — XPG; 3' incision + TFIIH anchor; XPG null = worst XP+CS; toxic intermediate
  'POLH':  '#00695c',  // dark teal — XP-Variant; NORMAL UDS; Pol-eta TLS; CPD bypass; delayed cancer; Japan common
};

const GENE_INFO = {
  'XPA':   { full: 'XPA / DNA Damage Verification Protein XPA / 273aa', locus: '9q22.33', size: '273 aa / 31 kDa (zinc-finger; GGR+TCR scaffold)', inh: 'AR', disease: 'XERODERMA PIGMENTOSUM GROUP A — MOST SEVERE: both GGR+TCR abolished (XPA required for both subpathways); near-zero UDS (0-4%); UV hypersensitivity from age 1-2 (acute sunburn from minimal UV); skin cancers (BCC/SCC/melanoma) by 1st decade; SEVERE NEURODEGENERATION: progressive SNHL + cerebellar ataxia + areflexia + intellectual decline + dementia; Japanese founder Arg228Ter most common XP-A allele; PHOTOSENSITISING DRUGS ABSOLUTELY CI (tetracyclines, fluoroquinolones, HCT, amiodarone, NSAIDs, phenothiazines)' },
  'ERCC3': { full: 'ERCC3 / XPB TFIIH 3′→5′ Helicase / 782aa', locus: '2q21.3', size: '782 aa / 89 kDa (TFIIH helicase + transcription)', inh: 'AR', disease: 'XP GROUP B / TTD / XP+CS OVERLAP — ULTRAORPHAN: TFIIH XPB subunit; helicase alleles → XP-B/XP+CS; stability alleles → TTD (brittle hair tiger-tail banding PATHOGNOMONIC); TFIIH dual function = NER + RNA Pol II initiation; TTD mechanism: TFIIH reduced → impaired transcription of sulphur-rich hair proteins; NO SKIN CANCER in TTD (NER partially retained + TCR intact); <30 XP-B patients worldwide; Pro131Thr = only known XP-B mutation' },
  'XPC':   { full: 'XPC / GGR Damage-Recognition Initiator / 940aa', locus: '3p25.1', size: '940 aa / 106 kDa (GGR initiator; RAD23B-CETN2 complex)', inh: 'AR', disease: 'XERODERMA PIGMENTOSUM GROUP C — MOST COMMON non-Japanese XP (USA/Europe ~50%): GGR defect ONLY (TCR intact) → CUTANEOUS DISEASE, NO NEURODEGENERATION; severe skin cancers from 1st decade; UDS 5-25%; North African founder c.1643_1644del (p.Trp548Ter); XPC senses helix distortion on undamaged strand opposite lesion; DDB2 (UV-DDB) assists XPC at CPD; ocular: pterygium + squamous cell carcinoma conjunctiva; XPC gene therapy mRNA-LNP in clinical trials 2024' },
  'ERCC2': { full: 'ERCC2 / XPD TFIIH 5′→3′ Helicase / 760aa', locus: '19q13.32', size: '760 aa / 87 kDa (TFIIH 5′→3′ helicase; lesion verifier)', inh: 'AR', disease: 'XP GROUP D / TTD / XP+CS / COFS — WIDEST SPECTRUM: XPD helicase stalls at lesion → lesion verification; helicase mutations → XP ± CS; stability mutations → TTD; Arg683Trp/Lys751Gln → mild XP-D (most common Europe); severe alleles → XP+CS with intracranial calcifications; TTD-ERCC2 alleles reduce TFIIH level → transcription impaired; ERCC2 accounts for ~50% of all TTD; TFIIH protein level normal in XP-D (helicase inactive) vs reduced in TTD-ERCC2 (complex unstable)' },
  'DDB2':  { full: 'DDB2 / UV-DDB CPD Sensor XPE / 428aa', locus: '11p11.2', size: '428 aa / 48 kDa (WD40 β-propeller; CRL4DDB2 E3 ligase)', inh: 'AR', disease: 'XERODERMA PIGMENTOSUM GROUP E — MILDEST CLASSIC XP: DDB2 is CPD-specific recognition factor of UV-DDB (DDB1-DDB2); DDB2 LOF → CPD poorly sensed → GGR slowed; 6-4PP directly sensed by XPC → less affected; UDS 30-50% normal (HIGHEST among NER-defective XP); mild-moderate UV sensitivity; skin cancers present (later onset); NO neurodegeneration; p53 target gene (DDB2 upregulated by p53 after UV); rarest XP group (~3-5%); Lys244Glu = original XP-E mutation' },
  'ERCC4': { full: 'ERCC4 / XPF-ERCC1 5′ Endonuclease / 916aa', locus: '16p13.12', size: '916 aa / 104 kDa (XPF catalytic; heterodimer with ERCC1 79 kDa)', inh: 'AR', disease: 'XP GROUP F / FANCONI ANEMIA Q — ALLELIC COMPLEXITY: XPF-ERCC1 5\' incision in NER AND ICL repair; mild alleles (Arg788Trp) → XP-F (mild UV sensitivity + later cancer); ICL alleles → FANCQ (bone marrow failure + DEB chromosomal fragility PATHOGNOMONIC); FANCQ: DEB test POSITIVE; ERCC4 progeroid: severe alleles → lipodystrophy + premature ageing; CISPLATIN CAUTION in ERCC4 (XPF-ERCC1 repairs cisplatin crosslinks); FANCQ: HSCT for bone marrow failure; ALKYLATING CHEMOTHERAPY CI in FA' },
  'ERCC5': { full: 'ERCC5 / XPG 3′ Endonuclease / 1186aa', locus: '13q33.1', size: '1186 aa / 133 kDa (FEN1 superfamily; TFIIH anchor)', inh: 'AR', disease: 'XP GROUP G / XPG+CS — WORST COMBINED PHENOTYPE: XPG makes 3\' incision AND anchors TFIIH at repair site; XPG null → TFIIH not retained → TCR also fails (despite CSB intact) → severe neurodegeneration + CS features; severe XPG null: profound ID + cachectic dwarfism + SNHL + retinal degeneration + intracranial calcifications; worst XP+CS prognosis (death 1st-2nd decade); partial ERCC5 function → milder XP-G alone; XPG null toxic intermediate hypothesis: open NER bubble (no 3\' cut) → apoptosis' },
  'POLH':  { full: 'POLH / Pol η Y-family TLS Polymerase / 713aa', locus: '6p21.1', size: '713 aa / 78 kDa (Y-family TLS polymerase; PCNA PIP-box C-terminal)', inh: 'AR', disease: 'XERODERMA PIGMENTOSUM VARIANT (XP-V) — UNIQUE: NORMAL NER (NORMAL UDS ~100%); TLS defect only; Pol η inserts AA opposite CPD (error-free bypass); POLH LOF → error-prone TLS pols at CPD → C→T/CC→TT UV mutations → delayed cancer (2nd-3rd decade vs 1st); MILDER UV sensitivity; NO neurodegeneration; Japan most common XP group (30-40%); Arg412Stop = most common allele worldwide; MISDIAGNOSIS COMMON — normal UDS; Pol η also bypasses cisplatin GG-adducts → CISPLATIN sensitivity in POLH LOF' },
};

function GeneChip({ gene, active, onClick }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span
      onClick={() => onClick && onClick(gene)}
      style={{
        background: col, color: '#fff', borderRadius: 4,
        padding: '3px 10px', fontSize: 12, fontWeight: 700,
        margin: '0 3px 4px 0', cursor: onClick ? 'pointer' : 'default',
        opacity: active === null || active === gene ? 1 : 0.45,
        border: active === gene ? '2px solid #fff' : '2px solid transparent',
        display: 'inline-block',
      }}
    >{gene}</span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 130 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8' }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function PctBar({ label, pct, color }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#94a3b8', marginBottom: 3 }}>
        <span>{label}</span><span style={{ fontWeight: 700, color: '#e2e8f0' }}>{pct}%</span>
      </div>
      <div style={{ background: '#334155', borderRadius: 4, height: 8 }}>
        <div style={{ background: color || '#38bdf8', width: `${pct}%`, height: 8, borderRadius: 4 }} />
      </div>
    </div>
  );
}

export default function HeredNerXpAtlasPage() {
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
  const accent = '#f59e0b';  // amber — UV/DNA damage theme

  const genes = ['XPA', 'ERCC3', 'XPC', 'ERCC2', 'DDB2', 'ERCC4', 'ERCC5', 'POLH'];

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: accent, margin: 0 }}>
          ☀ Hereditary NER/XP Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 6 }}>
          Complete 8-Gene Nucleotide Excision Repair Reference · XPA · ERCC3 · XPC · ERCC2 · DDB2 · ERCC4 · ERCC5 · POLH
          · 320 patients (8×40) · seeds 2710–2717
        </div>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          XP-A (XPA) · XP-B/TTD (ERCC3) · XP-C (XPC) · XP-D/TTD (ERCC2) ·
          XP-E (DDB2) · XP-F/FANCQ (ERCC4) · XP-G+CS (ERCC5) · XP-Variant (POLH)
        </div>
      </div>

      {/* Gene chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 20 }}>
        {genes.map(g => (
          <GeneChip key={g} gene={g} active={selGene} onClick={g2 => setSelGene(selGene === g2 ? null : g2)} />
        ))}
        {selGene && (
          <span onClick={() => setSelGene(null)} style={{ fontSize: 11, color: '#64748b', cursor: 'pointer', marginLeft: 8, alignSelf: 'center' }}>
            ✕ clear
          </span>
        )}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              background: tab === t ? accent : '#1e293b',
              color: tab === t ? '#0f172a' : '#94a3b8',
              border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer',
              fontWeight: tab === t ? 700 : 400, fontSize: 13,
            }}
          >{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#64748b' }}>Loading…</div>}
      {err && <div style={{ color: '#ef4444' }}>Error: {err}</div>}

      {/* OVERVIEW */}
      {tab === 'Overview' && overview && !loading && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2710-2717" />
            <MetricCard label="Genes" value={overview.genes?.length} sub="NER/XP atlas" />
            <MetricCard label="NER Pathway Groups" value={overview.pathway_categories?.length} />
            <MetricCard label="Critical Distinctions" value={overview.critical_distinctions?.length} warn />
          </div>

          {/* Gene Summaries grid */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: accent, marginBottom: 12 }}>Gene Summary Table</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene', 'Locus', 'Onset (yrs)', 'UV Sens%', 'Skin Ca%', 'Neurodegenr%', 'SNHL%', 'Ataxia%', 'ID%', 'Seizures%', 'Eye%', 'Brittle Hair%', 'BMF%', 'Progeroid%'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', color: '#64748b', textAlign: 'left', borderBottom: '1px solid #334155', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.gene_summaries || []).filter(gs => !selGene || gs.gene === selGene).map(gs => (
                    <tr key={gs.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                      <td style={{ padding: '6px 8px' }}>
                        <GeneChip gene={gs.gene} active={selGene} onClick={g => setSelGene(selGene === g ? null : g)} />
                      </td>
                      <td style={{ padding: '6px 8px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{gs.locus}</td>
                      <td style={{ padding: '6px 8px', color: '#e2e8f0', fontWeight: 700 }}>{gs.mean_onset_years}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_uv_sensitivity > 90 ? '#f59e0b' : '#64748b' }}>{gs.pct_uv_sensitivity}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_skin_cancer > 70 ? '#ef4444' : '#64748b' }}>{gs.pct_skin_cancer}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_neurodegeneration > 50 ? '#ef4444' : '#64748b' }}>{gs.pct_neurodegeneration}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_snhl > 50 ? '#a78bfa' : '#64748b' }}>{gs.pct_snhl}</td>
                      <td style={{ padding: '6px 8px' }}>{gs.pct_ataxia}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_id_severe > 60 ? '#f59e0b' : '#64748b' }}>{gs.pct_id_severe}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_seizures > 30 ? '#f59e0b' : '#64748b' }}>{gs.pct_seizures}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_eye_involvement > 70 ? '#38bdf8' : '#64748b' }}>{gs.pct_eye_involvement}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_brittle_hair > 20 ? '#a78bfa' : '#64748b' }}>{gs.pct_brittle_hair}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_bone_marrow_failure > 20 ? '#ef4444' : '#64748b' }}>{gs.pct_bone_marrow_failure}</td>
                      <td style={{ padding: '6px 8px', color: gs.pct_progeroid > 15 ? '#f97316' : '#64748b' }}>{gs.pct_progeroid}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pathway Categories */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: accent, marginBottom: 10 }}>NER Pathway Categories</div>
            {(overview.pathway_categories || []).map((pc, i) => (
              <div key={i} style={{ background: card, borderRadius: 8, padding: 12, marginBottom: 8, borderLeft: `4px solid ${accent}`, fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
                <div style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 4 }}>{pc.pathway}</div>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 4 }}>
                  {(pc.genes || []).map(g => <GeneChip key={g} gene={g} />)}
                </div>
                <div style={{ color: '#94a3b8' }}>{pc.note}</div>
              </div>
            ))}
          </div>

          {/* Critical Distinctions */}
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#ef4444', marginBottom: 10 }}>
              ⚠ Critical Distinctions
            </div>
            {(overview.critical_distinctions || []).map((cd, i) => (
              <div key={i} style={{ background: '#1a0a0a', border: '1px solid #7f1d1d', borderRadius: 8, padding: 12, marginBottom: 8, fontSize: 12, color: '#fca5a5', lineHeight: 1.6 }}>
                {cd}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* GENE TABLE */}
      {tab === 'Gene Table' && breakdown && !loading && (
        <div>
          {(breakdown.genes || []).filter(g => !selGene || g.gene === selGene).map(g => (
            <div key={g.gene} style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 20, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#64748b'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                <GeneChip gene={g.gene} />
                <span style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0' }}>{GENE_INFO[g.gene]?.full}</span>
              </div>
              <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 12 }}>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>Locus: <b style={{ color: '#e2e8f0' }}>{g.locus}</b></span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>Inheritance: <b style={{ color: '#e2e8f0' }}>{GENE_INFO[g.gene]?.inh}</b></span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>Patients: <b style={{ color: '#38bdf8' }}>{g.n_patients}</b></span>
                <span style={{ fontSize: 12, color: '#94a3b8' }}>{g.protein_size}</span>
              </div>
              <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, marginBottom: 10 }}>{GENE_INFO[g.gene]?.disease}</div>

              {/* Phenotype bars */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0 20px' }}>
                {[
                  { label: 'UV Sensitivity', key: 'pct_uv_sensitivity', color: '#f59e0b' },
                  { label: 'Skin Cancer', key: 'pct_skin_cancer', color: '#ef4444' },
                  { label: 'Neurodegeneration', key: 'pct_neurodegeneration', color: '#a78bfa' },
                  { label: 'SNHL', key: 'pct_snhl', color: '#818cf8' },
                  { label: 'Ataxia', key: 'pct_ataxia', color: '#6366f1' },
                  { label: 'ID (Severe)', key: 'pct_id_severe', color: '#f97316' },
                  { label: 'Seizures', key: 'pct_seizures', color: '#fbbf24' },
                  { label: 'Eye Involvement', key: 'pct_eye_involvement', color: '#38bdf8' },
                  { label: 'Brittle Hair (TTD)', key: 'pct_brittle_hair', color: '#c084fc' },
                  { label: 'Bone Marrow Failure', key: 'pct_bone_marrow_failure', color: '#ef4444' },
                  { label: 'Progeroid Features', key: 'pct_progeroid', color: '#fb923c' },
                ].map(({ label, key, color }) => {
                  const pct = g[key] ?? 0;
                  return pct > 0 ? (
                    <PctBar key={key} label={label} pct={pct} color={color} />
                  ) : null;
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* CLINICAL ATLAS */}
      {tab === 'Clinical Atlas' && breakdown && !loading && (
        <div>
          {(breakdown.genes || []).filter(g => !selGene || g.gene === selGene).map(g => (
            <div key={g.gene} style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 24, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#64748b'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                <GeneChip gene={g.gene} />
                <span style={{ fontSize: 16, fontWeight: 700, color: accent }}>{g.gene}</span>
                <span style={{ fontSize: 12, color: '#64748b' }}>{g.locus} · {g.protein_size}</span>
              </div>

              {[
                { title: 'Inheritance / Biology', content: g.inheritance },
                { title: 'Disease Category', content: g.disease_category },
                { title: 'NER Pathway Mechanism', content: g.disease_pathway },
                { title: 'Pathognomonic Features', content: g.pathognomonic },
                { title: 'Treatment', content: g.treatment },
              ].map(({ title, content }) => (
                <div key={title} style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#38bdf8', marginBottom: 4 }}>{title}</div>
                  <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7, background: '#0f172a', borderRadius: 6, padding: '10px 14px' }}>
                    {content}
                  </div>
                </div>
              ))}

              {/* Patient sample table */}
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b', marginBottom: 6 }}>
                  Patient Cohort Sample (first 10 of {g.n_patients})
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                    <thead>
                      <tr style={{ background: '#0f172a' }}>
                        {['ID', 'Sex', 'Onset', 'Age', 'UV', 'Cancer', 'Neuro', 'SNHL', 'Ataxia', 'ID', 'Seiz', 'Eye', 'Hair', 'BMF', 'Prog'].map(h => (
                          <th key={h} style={{ padding: '4px 6px', color: '#475569', textAlign: 'left', borderBottom: '1px solid #1e293b' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(g.patients || []).slice(0, 10).map(p => (
                        <tr key={p.patient_id} style={{ borderBottom: '1px solid #1e293b' }}>
                          <td style={{ padding: '3px 6px', color: '#64748b' }}>{p.patient_id}</td>
                          <td style={{ padding: '3px 6px' }}>{p.sex}</td>
                          <td style={{ padding: '3px 6px', color: '#e2e8f0' }}>{p.age_onset_years}y</td>
                          <td style={{ padding: '3px 6px', color: '#94a3b8' }}>{p.age_current_years}y</td>
                          {[
                            [p.uv_sensitivity, '#f59e0b'],
                            [p.skin_cancer, '#ef4444'],
                            [p.neurodegeneration, '#a78bfa'],
                            [p.snhl, '#818cf8'],
                            [p.ataxia, '#6366f1'],
                            [p.id_severe, '#f97316'],
                            [p.seizures, '#fbbf24'],
                            [p.eye_involvement, '#38bdf8'],
                            [p.brittle_hair, '#c084fc'],
                            [p.bone_marrow_failure, '#ef4444'],
                            [p.progeroid, '#fb923c'],
                          ].map(([val, col], j) => (
                            <td key={j} style={{ padding: '3px 6px', color: val ? col : '#334155' }}>
                              {val ? '✓' : '·'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS */}
      {tab === 'Definitions' && definitions && !loading && (
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: accent, marginBottom: 16 }}>
            NER/XP Glossary & Standards
          </div>
          {Object.entries(definitions.glossary || {}).map(([term, def]) => (
            <div key={term} style={{ background: card, borderRadius: 8, padding: 14, marginBottom: 12, borderLeft: '4px solid #334155' }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8', marginBottom: 6 }}>{term}</div>
              <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7 }}>{def}</div>
            </div>
          ))}

          <div style={{ fontSize: 14, fontWeight: 700, color: accent, marginBottom: 12, marginTop: 24 }}>
            Standards & References
          </div>
          {(definitions.standards || []).map((s, i) => (
            <div key={i} style={{ background: '#1a2332', borderRadius: 6, padding: '8px 14px', marginBottom: 6, fontSize: 12, color: '#64748b' }}>
              {s}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
