'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-lymphoid-malignancy-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ATM':   '#7b1fa2',  // deep purple    — DDR kinase, A-T, CLL/MCL
  'TP53':  '#b71c1c',  // deep red        — Li-Fraumeni, guardian of genome
  'CHEK2': '#1565c0',  // deep blue       — checkpoint kinase 2, CLL/breast
  'PAX5':  '#2e7d32',  // dark green      — B-cell identity TF, familial B-ALL
  'IKZF1': '#e65100',  // deep orange     — Ikaros, B-cell development, Ph-like ALL
  'POT1':  '#00695c',  // dark teal       — shelterin, familial CLL/melanoma
  'RUNX1': '#c62828',  // crimson         — FPD/AML, haematopoietic master TF
  'STAT3': '#4527a0',  // deep indigo     — JAK-STAT, T-LGL, GOF lymphoma
};

const GENE_INFO = {
  'ATM':   { full: 'ATM / Ataxia-Telangiectasia-Mutated / 3056aa', locus: '11q22.3', size: '3056 aa / 350 kDa (PI3K-like kinase; MRN-activated DSB sensor; het = 2-4× CLL/MCL/lymphoma; hom = A-T: cerebellar ataxia + telangiectasias + IgA deficiency + radiosensitivity; del(11q22) = 25-35% CLL)', inh: 'AD (het) / AR (hom A-T)' },
  'TP53':  { full: 'TP53 / Tumour Protein p53 / 393aa', locus: '17p13.1', size: '393 aa / 43 kDa (Guardian of genome; Li-Fraumeni Syndrome; hypodiploid B-ALL in child → germline TP53 >90%; del(17p13) CLL ultra-poor risk; whole-body MRI annually; AVOID RT; APR-246 investigational)', inh: 'AD' },
  'CHEK2': { full: 'CHEK2 / Checkpoint Kinase 2 / 543aa', locus: '22q12.1', size: '543 aa / 60 kDa (ATM substrate; FHA + kinase domain; I157T Eastern European founder ~5% Poland; 1100delC NW European; 2-3× CLL + 4× colorectal; moderate-risk counselling — not BRCA-equivalent)', inh: 'AD' },
  'PAX5':  { full: 'PAX5 / Paired Box 5 / BSAP / 391aa', locus: '9p13.2', size: '391 aa / 47 kDa (B-cell identity master TF; G183S most recurrent familial B-ALL; dominant negative mechanism; buccal DNA required to distinguish germline vs somatic; sibling donor exclusion; incomplete penetrance ~20-30%)', inh: 'AD' },
  'IKZF1': { full: 'IKZF1 / Ikaros / 519aa', locus: '7p12.2', size: '519 aa / 58 kDa (Ikaros ZF TF; B-cell commitment; IK6 dominant-negative isoform (del exons 4-7); B-lymphopenia + IVIg; IKZF1plus = ultra-high risk ALL → HSCT CR1; Ph-like ALL: add TKI/ruxolitinib)', inh: 'AD' },
  'POT1':  { full: 'POT1 / Protection of Telomeres 1 / 634aa', locus: '7q31.33', size: '634 aa / 71 kDa (Shelterin; OB-fold ss-TTAGGG binding; telomere cap; POT1-TPP1 heterodimer; familial CLL 3-5% + melanoma + angiosarcoma + glioma; CLL + melanoma in same patient → POT1 germline)', inh: 'AD' },
  'RUNX1': { full: 'RUNX1 / AML1 / CBFα2 / 453aa', locus: '21q22.12', size: '453 aa / 50 kDa (Runt-domain TF; CBFβ partner; FPD/AML: thrombocytopenia 50-150k + plt function defect → 35-44% MDS/AML + 15-20% ALL; ITP misdiagnosis — avoid IVIg/steroids; sibling donor exclusion mandatory)', inh: 'AD' },
  'STAT3': { full: 'STAT3 / Signal Transducer and Activator of Transcription 3 / 770aa', locus: '17q21.2', size: '770 aa / 92 kDa (SH2-domain TF; JAK-STAT3 pathway; Y640F GOF = T-LGL + neutropenia + autoimmune cytopenias; MTX/CsA first-line; ruxolitinib emerging; LOF = Hyper-IgE (opposite phenotype))', inh: 'AD GOF' },
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
      <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HereditaryLymphoidMalignancyPredispositionAtlas() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [activeGene, setActiveGene] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const [ovRes, bkRes, dfRes] = await Promise.all([
          fetch(`${API}/api/${SLUG}/overview`),
          fetch(`${API}/api/${SLUG}/breakdown`),
          fetch(`${API}/api/${SLUG}/definitions`),
        ]);
        setOverview(await ovRes.json());
        setBreakdown(await bkRes.json());
        setDefs(await dfRes.json());
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) return <div style={{ color: '#94a3b8', padding: 40 }}>Loading Hereditary-Lymphoid-Malignancy-Predisposition-Atlas…</div>;
  if (error) return <div style={{ color: '#ef4444', padding: 40 }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1e293b,#0f172a)', borderBottom: '1px solid #334155', padding: '20px 32px' }}>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>HEREDITARY LYMPHOID MALIGNANCY PREDISPOSITION</div>
        <div style={{ fontSize: 20, fontWeight: 700, color: '#f1f5f9' }}>
          🧬 Hereditary-Lymphoid-Malignancy-Predisposition-Atlas
        </div>
        <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
          Complete 8-Gene Germline CLL / ALL / NHL / T-LGL / Li-Fraumeni Haematologic Malignancy Reference
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} active={null} />)}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 2, padding: '0 32px', background: '#1e293b', borderBottom: '1px solid #334155' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#0f172a' : 'transparent',
            color: tab === t ? '#38bdf8' : '#64748b',
            border: 'none', borderBottom: tab === t ? '2px solid #38bdf8' : '2px solid transparent',
            padding: '10px 18px', cursor: 'pointer', fontSize: 13, fontWeight: 600,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 32px' }}>
        {/* ── OVERVIEW TAB ── */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40 aggregate" />
              <MetricCard label="Genes" value={overview.n_genes} sub="seeds 2878-2885" />
              <MetricCard label="DDR / Genome Guardian" value="ATM · TP53 · CHEK2" sub="3 genes" />
              <MetricCard label="B-Cell Dev TF" value="PAX5 · IKZF1" sub="familial B-ALL" />
              <MetricCard label="Telomere Protection" value="POT1" sub="familial CLL/melanoma" />
              <MetricCard label="Haematopoietic TF" value="RUNX1" sub="FPD/AML thrombocytopenia" />
              <MetricCard label="JAK-STAT" value="STAT3 GOF" sub="T-LGL leukaemia" />
            </div>

            {/* Clinical alerts */}
            <div style={{ background: '#1e293b', border: '1px solid #ef4444', borderRadius: 8, padding: '14px 18px', marginBottom: 20 }}>
              <div style={{ color: '#ef4444', fontWeight: 700, fontSize: 13, marginBottom: 8 }}>⚠ KEY CLINICAL ALERTS</div>
              {overview.key_clinical_alerts?.map((a, i) => (
                <div key={i} style={{ color: '#fca5a5', fontSize: 12, marginBottom: 4 }}>• {a}</div>
              ))}
            </div>

            {/* Pathway categories */}
            {overview.pathway_summary && (
              <div style={{ marginBottom: 20 }}>
                <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8, fontWeight: 700 }}>PATHWAY CATEGORIES</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                  {Object.entries(overview.pathway_summary).filter(([, genes]) => genes.length > 0).map(([path, genes]) => (
                    <div key={path} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6, padding: '8px 14px' }}>
                      <div style={{ color: '#38bdf8', fontSize: 11, fontWeight: 700 }}>{path.replace(/_/g, ' ')}</div>
                      <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                        {genes.map(g => <GeneChip key={g} gene={g} active={null} />)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Gene summary table */}
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene', 'Locus', 'Protein Summary', 'N', 'Median Age', '%F', 'Mean Hb', 'Mean PLT', '%HSCT', '%Remission'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', borderBottom: '1px solid #334155' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.map((g, i) => (
                    <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                      <td style={{ padding: '7px 10px' }}><GeneChip gene={g.gene} active={null} /></td>
                      <td style={{ padding: '7px 10px', color: '#64748b' }}>{g.locus}</td>
                      <td style={{ padding: '7px 10px', color: '#94a3b8', maxWidth: 220 }}>{g.protein_summary}</td>
                      <td style={{ padding: '7px 10px', color: '#38bdf8' }}>{g.n_patients}</td>
                      <td style={{ padding: '7px 10px', color: '#e2e8f0' }}>{g.median_age_dx}y</td>
                      <td style={{ padding: '7px 10px', color: '#e2e8f0' }}>{g.pct_female}%</td>
                      <td style={{ padding: '7px 10px', color: g.mean_hb_gdl < 9 ? '#ef4444' : '#e2e8f0' }}>{g.mean_hb_gdl}</td>
                      <td style={{ padding: '7px 10px', color: g.mean_platelets_k < 80 ? '#ef4444' : '#e2e8f0' }}>{g.mean_platelets_k}k</td>
                      <td style={{ padding: '7px 10px', color: '#f59e0b' }}>{g.pct_hsct}%</td>
                      <td style={{ padding: '7px 10px', color: '#4ade80' }}>{g.pct_remission}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── GENE TABLE TAB ── */}
        {tab === 'Gene Table' && (
          <div>
            <div style={{ marginBottom: 14, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              <span style={{ color: '#94a3b8', fontSize: 12, marginRight: 6 }}>Filter:</span>
              <button onClick={() => setActiveGene(null)} style={{ background: activeGene === null ? '#38bdf8' : '#1e293b', color: activeGene === null ? '#0f172a' : '#94a3b8', border: '1px solid #334155', borderRadius: 4, padding: '3px 10px', fontSize: 11, cursor: 'pointer' }}>All</button>
              {Object.keys(GENE_COLORS).map(g => (
                <button key={g} onClick={() => setActiveGene(activeGene === g ? null : g)} style={{ background: activeGene === g ? GENE_COLORS[g] : '#1e293b', color: '#fff', border: `1px solid ${GENE_COLORS[g]}`, borderRadius: 4, padding: '3px 10px', fontSize: 11, cursor: 'pointer' }}>{g}</button>
              ))}
            </div>

            {Object.entries(GENE_INFO)
              .filter(([g]) => !activeGene || g === activeGene)
              .map(([gene, info]) => (
                <div key={gene} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[gene]}33`, borderRadius: 8, padding: '16px 20px', marginBottom: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                    <GeneChip gene={gene} active={null} />
                    <span style={{ color: '#94a3b8', fontSize: 12 }}>{info.full}</span>
                    <span style={{ color: '#64748b', fontSize: 11, marginLeft: 6 }}>{info.locus}</span>
                    <span style={{ color: '#64748b', fontSize: 11, marginLeft: 6, background: '#0f172a', padding: '2px 8px', borderRadius: 4 }}>{info.inh}</span>
                  </div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>{info.size}</div>
                  {breakdown && breakdown[gene] && (
                    <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                      <div style={{ background: '#0f172a', borderRadius: 6, padding: '10px 14px' }}>
                        <div style={{ color: '#38bdf8', fontSize: 11, fontWeight: 700, marginBottom: 6 }}>PATHOGNOMONIC</div>
                        <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>{breakdown[gene].pathognomonic?.slice(0, 400)}…</div>
                      </div>
                      <div style={{ background: '#0f172a', borderRadius: 6, padding: '10px 14px' }}>
                        <div style={{ color: '#4ade80', fontSize: 11, fontWeight: 700, marginBottom: 6 }}>TREATMENT</div>
                        <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>{breakdown[gene].treatment?.slice(0, 400)}…</div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
          </div>
        )}

        {/* ── CLINICAL ATLAS TAB ── */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            <div style={{ marginBottom: 14, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              <span style={{ color: '#94a3b8', fontSize: 12, marginRight: 6 }}>Gene:</span>
              <button onClick={() => setActiveGene(null)} style={{ background: activeGene === null ? '#38bdf8' : '#1e293b', color: activeGene === null ? '#0f172a' : '#94a3b8', border: '1px solid #334155', borderRadius: 4, padding: '3px 10px', fontSize: 11, cursor: 'pointer' }}>All</button>
              {Object.keys(GENE_COLORS).map(g => (
                <button key={g} onClick={() => setActiveGene(activeGene === g ? null : g)} style={{ background: activeGene === g ? GENE_COLORS[g] : '#1e293b', color: '#fff', border: `1px solid ${GENE_COLORS[g]}`, borderRadius: 4, padding: '3px 10px', fontSize: 11, cursor: 'pointer' }}>{g}</button>
              ))}
            </div>

            {Object.entries(breakdown)
              .filter(([g]) => !activeGene || g === activeGene)
              .map(([gene, data]) => {
                const pts = data.patients || [];
                const shown = pts.slice(0, 15);
                return (
                  <div key={gene} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[gene]}44`, borderRadius: 8, padding: '14px 18px', marginBottom: 16 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                      <GeneChip gene={gene} active={null} />
                      <span style={{ color: '#94a3b8', fontSize: 12 }}>{data.n_patients} patients</span>
                    </div>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                        <thead>
                          <tr style={{ background: '#0f172a' }}>
                            {['ID', 'Age', 'Sex', 'Hb', 'PLT', 'Mono', 'Diagnosis', 'Treatment', 'HSCT', 'Outcome'].map(h => (
                              <th key={h} style={{ padding: '5px 8px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #334155' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {shown.map((p, i) => (
                            <tr key={p.id} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                              <td style={{ padding: '4px 8px', color: '#64748b' }}>{p.id}</td>
                              <td style={{ padding: '4px 8px' }}>{p.age_at_dx}y</td>
                              <td style={{ padding: '4px 8px', color: p.sex === 'F' ? '#f472b6' : '#60a5fa' }}>{p.sex}</td>
                              <td style={{ padding: '4px 8px', color: p.hb_gdl < 8 ? '#ef4444' : '#e2e8f0' }}>{p.hb_gdl}</td>
                              <td style={{ padding: '4px 8px', color: p.platelets_k < 50 ? '#ef4444' : '#e2e8f0' }}>{p.platelets_k}k</td>
                              <td style={{ padding: '4px 8px', color: p.monocytes_abs < 0.05 ? '#ef4444' : '#e2e8f0' }}>{p.monocytes_abs}</td>
                              <td style={{ padding: '4px 8px', color: '#f59e0b', maxWidth: 140 }}>{p.primary_dx}</td>
                              <td style={{ padding: '4px 8px', color: '#94a3b8', maxWidth: 140 }}>{p.treatment}</td>
                              <td style={{ padding: '4px 8px', color: p.hsct === 'Yes' ? '#4ade80' : '#64748b' }}>{p.hsct}</td>
                              <td style={{ padding: '4px 8px', color: p.outcome === 'remission' ? '#4ade80' : p.outcome === 'deceased' ? '#ef4444' : '#94a3b8' }}>{p.outcome}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {pts.length > 15 && <div style={{ color: '#64748b', fontSize: 11, marginTop: 6 }}>… and {pts.length - 15} more patients</div>}
                    </div>
                  </div>
                );
              })}
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {tab === 'Definitions' && defs && (
          <div>
            <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 16 }}>
              {Object.keys(defs.glossary || {}).length} definitions · Hereditary-Lymphoid-Malignancy-Predisposition-Atlas
            </div>
            {Object.entries(defs.glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6, padding: '12px 16px', marginBottom: 10 }}>
                <div style={{ color: '#38bdf8', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{term.replace(/-/g, ' ')}</div>
                <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{def}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
