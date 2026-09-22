'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-gh-igf1-axis-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'AIP':    '#b71c1c',  // deep red     — FIPA somatotropinoma, pituitary adenoma
  'GHR':    '#e65100',  // deep orange  — Laron syndrome, GH insensitivity
  'IGF1':   '#f9a825',  // amber        — IGF1 LOF, severe pre/postnatal growth failure
  'IGF1R':  '#1565c0',  // deep blue    — haploinsufficiency, mild-moderate short stature
  'STAT5B': '#4a148c',  // deep purple  — GH+immune combined deficiency
  'IGFALS': '#2e7d32',  // dark green   — IGF acid labile subunit deficiency
  'PAPPA2': '#00695c',  // dark teal    — high total IGF-1, low free IGF-1
  'GPR101': '#6a1b9a',  // purple       — X-LAG, infant-onset gigantism
};

const GENE_INFO = {
  'AIP': {
    full: 'AIP / Aryl Hydrocarbon Receptor Interacting Protein / 330aa',
    locus: '11q13.2',
    size: '330 aa / 37 kDa (AhR co-chaperone; LOF → somatotropinoma; FIPA syndrome; AD incomplete penetrance ~30%; young-onset gigantism; macroadenoma; octreotide resistance; AIP mutation search in any young pituitary adenoma)',
    inh: 'AD',
  },
  'GHR': {
    full: 'GHR / Growth Hormone Receptor / 638aa',
    locus: '5p13.1',
    size: '638 aa / 70 kDa (single-pass class I cytokine receptor; extracellular domain deletion → Laron syndrome; very low IGF-1; very high GH; rhGH INEFFECTIVE; rhIGF-1 treatment; E180 splice = exon 3 deletion = partial insensitivity)',
    inh: 'AR',
  },
  'IGF1': {
    full: 'IGF1 / Insulin-like Growth Factor 1 / 195aa',
    locus: '12q23.2',
    size: '195 aa prepropeptide / 7.5 kDa mature (70aa); LOF → severe prenatal + postnatal growth failure; sensorineural hearing loss; intellectual disability; high GH; very low IGF-1; partial exon deletions possible',
    inh: 'AR',
  },
  'IGF1R': {
    full: 'IGF1R / Insulin-like Growth Factor 1 Receptor / 1367aa',
    locus: '15q26.3',
    size: '1367 aa / 155 kDa (RTK; alpha/beta heterodimer; haploinsufficiency → moderate short stature; SGA + poor catch-up; mild features; 15q26 deletion → check contiguous gene syndrome; heterozygous usually sufficient)',
    inh: 'AD',
  },
  'STAT5B': {
    full: 'STAT5B / Signal Transducer and Activator of Transcription 5B / 787aa',
    locus: '17q21.2',
    size: '787 aa / 90 kDa (GH signalling transcription factor; LOF → GH insensitivity + immune dysregulation; eczema/autoimmune; lung disease; lymphopenia; unique DUAL GH+immune phenotype; IGFALS and GHR expression both impaired)',
    inh: 'AR',
  },
  'IGFALS': {
    full: 'IGFALS / Insulin-like Growth Factor Acid Labile Subunit / 605aa',
    locus: '16p13.3',
    size: '605 aa / 66 kDa (ALS; 18 LRR; ternary complex with IGF-1 + IGFBP-3; LOF → ALS deficiency; low IGF-1 and IGFBP-3; MILD short stature only; pubertal delay; delayed bone age; no treatment required usually)',
    inh: 'AR',
  },
  'PAPPA2': {
    full: 'PAPPA2 / Pregnancy-associated Plasma Protein A2 / 1791aa',
    locus: '1q25.2',
    size: '1791 aa / 200 kDa (IGFBP-3/5 protease; cleaves to release free IGF-1; LOF → high TOTAL IGF-1, low FREE IGF-1; short stature; hyperostosis; thin cortex; PAPPA2 deficiency = IGF bioavailability defect despite normal total IGF-1)',
    inh: 'AR',
  },
  'GPR101': {
    full: 'GPR101 / G Protein-Coupled Receptor 101 / 506aa',
    locus: 'Xq26.3',
    size: '506 aa / 56 kDa (orphan GPCR; Xq26.3 microduplication → X-LAG; infant-onset gigantism before age 3; mixed GH/PRL adenoma or hyperplasia; highest GH ever recorded; FEMALES worse; sporadic or germline; pasireotide + pegvisomant; surgery + radiotherapy)',
    inh: 'XL',
  },
};

const SYNDROME_COLORS = {
  'FIPA-Somatotropinoma': '#b71c1c',
  'Laron-GH-Insensitivity': '#e65100',
  'IGF1-LOF-Severe': '#f9a825',
  'IGF1R-Haploinsuff': '#1565c0',
  'GH+Immune-Combined': '#4a148c',
  'ALS-Deficiency': '#2e7d32',
  'IGF-Bioavail-Defect': '#00695c',
  'X-LAG-Gigantism': '#6a1b9a',
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

function SyndromeTag({ syndrome }) {
  const col = SYNDROME_COLORS[syndrome] || '#555';
  return (
    <span style={{ background: col, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 700 }}>
      {syndrome}
    </span>
  );
}

function MetricCard({ label, value, sub, warn, ok }) {
  const color = warn ? '#ef4444' : ok ? '#22c55e' : '#38bdf8';
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : ok ? '#22c55e' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 130 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function GHIgf1AtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [activeGene, setActiveGene] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const geneList = overview?.genes || Object.keys(GENE_INFO);

  if (loading) return <div style={{ color: '#94a3b8', padding: 40, textAlign: 'center' }}>Loading Hereditary-GH-IGF1-Axis-Atlas…</div>;
  if (error)   return <div style={{ color: '#ef4444', padding: 40 }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'sans-serif', padding: '24px 20px' }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', marginBottom: 6 }}>
          🧬 Hereditary-GH-IGF1-Axis-Atlas
        </h1>
        <p style={{ fontSize: 13, color: '#94a3b8', maxWidth: 900 }}>
          Complete 8-gene GH/IGF-1 axis reference: <b>FIPA somatotropinoma</b> (AIP) · <b>Laron syndrome</b> (GHR) ·
          <b> IGF1 LOF severe</b> (IGF1) · <b>IGF1R haploinsufficiency</b> (IGF1R) ·
          <b> GH+immune combined deficiency</b> (STAT5B) · <b>ALS deficiency</b> (IGFALS) ·
          <b> IGF bioavailability defect</b> (PAPPA2) · <b>X-LAG infant gigantism</b> (GPR101) —
          hereditary growth disorders spanning GH excess and GH/IGF-1 deficiency with distinct biochemical fingerprints.
          320-patient aggregate cohort · seeds 2958–2965.
        </p>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {geneList.map(g => <GeneChip key={g} gene={g} active={activeGene} onClick={g => setActiveGene(prev => prev === g ? null : g)} />)}
        </div>
        <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(SYNDROME_COLORS).map(([s, c]) => (
            <span key={s} style={{ background: c, color: '#fff', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700 }}>{s}</span>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #334155', paddingBottom: 8 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#1e40af' : '#1e293b', color: tab === t ? '#fff' : '#94a3b8',
            border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontSize: 13, fontWeight: 600,
          }}>{t}</button>
        ))}
      </div>

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <MetricCard label="Total Patients" value={overview.n_patients} sub="8 × 40, seeds 2958–2965" />
            <MetricCard label="Mean Height SDS" value={overview.aggregate_metrics?.mean_height_sds} warn={overview.aggregate_metrics?.mean_height_sds < -2} />
            <MetricCard label="Mean IGF-1 SDS" value={overview.aggregate_metrics?.mean_igf1_sds} warn={overview.aggregate_metrics?.mean_igf1_sds < -2} />
            <MetricCard label="Mean GH ng/mL" value={overview.aggregate_metrics?.mean_gh_ng_ml} ok={overview.aggregate_metrics?.mean_gh_ng_ml < 1} />
            <MetricCard label="GH Excess %" value={`${overview.aggregate_metrics?.gh_excess_pct}%`} warn={true} sub="AIP/GPR101" />
            <MetricCard label="rhIGF-1 Treated %" value={`${overview.aggregate_metrics?.rhigf1_treated_pct}%`} sub="Laron/STAT5B" />
            <MetricCard label="Octreotide Resistant %" value={`${overview.aggregate_metrics?.octreotide_resistant_pct}%`} warn={true} sub="AIP macroadenoma" />
            <MetricCard label="Immune Dysregulation %" value={`${overview.aggregate_metrics?.immune_dysreg_pct}%`} sub="STAT5B" />
          </div>

          {/* Key clinical rules */}
          <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#fbbf24', marginBottom: 10, fontSize: 14 }}>⚠️ Critical Clinical Rules</div>
            {(overview.key_clinical_rules || []).map((r, i) => (
              <div key={i} style={{ fontSize: 12, color: '#e2e8f0', marginBottom: 6, paddingLeft: 8, borderLeft: '2px solid #fbbf24' }}>
                {r}
              </div>
            ))}
          </div>

          {/* Gene summary table */}
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1e293b' }}>
                  {['Gene', 'Syndrome', 'Locus', 'Ht SDS', 'IGF-1 SDS', 'GH ng/mL', 'GH Excess%', 'Immune%', 'rhIGF-1%', 'Oct Resist%'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', fontWeight: 600, borderBottom: '1px solid #334155' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.gene_summary || []).map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#131f35', opacity: activeGene && activeGene !== g.gene ? 0.5 : 1 }}>
                    <td style={{ padding: '7px 10px' }}><GeneChip gene={g.gene} /></td>
                    <td style={{ padding: '7px 10px' }}><SyndromeTag syndrome={g.syndrome} /></td>
                    <td style={{ padding: '7px 10px', color: '#94a3b8', fontFamily: 'monospace' }}>{g.locus}</td>
                    <td style={{ padding: '7px 10px', color: g.mean_height_sds < -2.5 ? '#ef4444' : g.mean_height_sds > 2 ? '#22c55e' : '#e2e8f0', fontWeight: 700 }}>{g.mean_height_sds}</td>
                    <td style={{ padding: '7px 10px', color: g.mean_igf1_sds < -2 ? '#ef4444' : g.mean_igf1_sds > 2 ? '#f59e0b' : '#e2e8f0', fontWeight: 700 }}>{g.mean_igf1_sds}</td>
                    <td style={{ padding: '7px 10px', color: g.mean_gh_ng_ml > 5 ? '#f59e0b' : '#94a3b8' }}>{g.mean_gh_ng_ml}</td>
                    <td style={{ padding: '7px 10px', color: g.gh_excess_pct > 30 ? '#ef4444' : '#475569' }}>{g.gh_excess_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.immune_dysreg_pct > 30 ? '#a78bfa' : '#475569' }}>{g.immune_dysreg_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.rhigf1_pct > 30 ? '#38bdf8' : '#475569' }}>{g.rhigf1_pct}%</td>
                    <td style={{ padding: '7px 10px', color: g.oct_resist_pct > 30 ? '#ef4444' : '#475569' }}>{g.oct_resist_pct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 14 }}>
          {Object.entries(GENE_INFO)
            .filter(([g]) => !activeGene || g === activeGene)
            .map(([gene, info]) => (
              <div key={gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[gene]}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                  <GeneChip gene={gene} />
                  <span style={{ fontSize: 11, color: '#64748b', fontFamily: 'monospace' }}>{info.locus}</span>
                </div>
                <div style={{ fontSize: 12, color: '#e2e8f0', marginBottom: 4, fontWeight: 600 }}>{info.full}</div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>{info.size}</div>
                <div style={{ fontSize: 11, background: '#0f172a', borderRadius: 4, padding: '4px 8px', color: '#38bdf8', fontWeight: 600 }}>
                  Inheritance: {info.inh}
                </div>
              </div>
            ))}
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {(breakdown.genes || [])
            .filter(g => !activeGene || g.gene === activeGene)
            .map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[g.gene]}` }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ fontSize: 13, color: '#f1f5f9', fontWeight: 600 }}>{g.locus}</span>
                  <span style={{ fontSize: 12, color: '#94a3b8' }}>
                    n={g.n_patients} · Ht SDS={g.mean_height_sds} · IGF-1 SDS={g.mean_igf1_sds} · GH={g.mean_gh_ng_ml} ng/mL
                  </span>
                  {g.gh_excess_pct > 30 && <span style={{ background: '#7f1d1d', color: '#fca5a5', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>GH excess</span>}
                  {g.immune_dysreg_pct > 30 && <span style={{ background: '#312e81', color: '#a5b4fc', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>Immune dysreg</span>}
                  {g.oct_resist_pct > 30 && <span style={{ background: '#78350f', color: '#fcd34d', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>Oct resistant</span>}
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8, lineHeight: 1.5 }}>{g.protein_size?.slice(0, 300)}…</div>
                <div style={{ fontSize: 11, color: '#cbd5e1', marginBottom: 8, lineHeight: 1.5 }}><b style={{ color: '#fbbf24' }}>Disease:</b> {g.disease_category?.slice(0, 400)}…</div>
                <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.5 }}><b style={{ color: '#34d399' }}>Pathway:</b> {g.disease_pathway?.slice(0, 300)}…</div>

                {g.treatment_distribution && (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Treatment distribution (n={g.n_patients}):</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {Object.entries(g.treatment_distribution).sort((a, b) => b[1] - a[1]).map(([tx, cnt]) => (
                        <span key={tx} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 4, padding: '2px 8px', fontSize: 11, color: '#94a3b8' }}>
                          {tx}: <b style={{ color: '#e2e8f0' }}>{cnt}</b>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {g.patients?.length > 0 && (
                  <div style={{ marginTop: 10, overflowX: 'auto' }}>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Sample patients (first 10):</div>
                    <table style={{ fontSize: 11, borderCollapse: 'collapse', width: '100%' }}>
                      <thead>
                        <tr style={{ background: '#0f172a' }}>
                          {['ID', 'Sex', 'Age', 'Ht SDS', 'IGF-1 SDS', 'GH', 'GH Excess', 'Immune', 'Oct Resist', 'Treatment'].map(h => (
                            <th key={h} style={{ padding: '4px 8px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #1e293b' }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {g.patients.map((p, i) => (
                          <tr key={p.id} style={{ background: i % 2 === 0 ? '#0f172a' : '#131f35' }}>
                            <td style={{ padding: '3px 8px', fontFamily: 'monospace', color: '#94a3b8' }}>{p.id}</td>
                            <td style={{ padding: '3px 8px' }}>{p.gender}</td>
                            <td style={{ padding: '3px 8px' }}>{p.age_yr}</td>
                            <td style={{ padding: '3px 8px', color: p.height_sds < -2.5 ? '#ef4444' : p.height_sds > 2 ? '#22c55e' : '#e2e8f0', fontWeight: 700 }}>{p.height_sds}</td>
                            <td style={{ padding: '3px 8px', color: p.igf1_sds < -2 ? '#ef4444' : p.igf1_sds > 2 ? '#f59e0b' : '#e2e8f0', fontWeight: 700 }}>{p.igf1_sds}</td>
                            <td style={{ padding: '3px 8px', color: p.gh_ng_ml > 5 ? '#f59e0b' : '#94a3b8' }}>{p.gh_ng_ml}</td>
                            <td style={{ padding: '3px 8px', color: p.gh_excess ? '#ef4444' : '#475569' }}>{p.gh_excess ? 'Yes' : 'No'}</td>
                            <td style={{ padding: '3px 8px', color: p.immune_dysregulation ? '#a78bfa' : '#475569' }}>{p.immune_dysregulation ? 'Yes' : 'No'}</td>
                            <td style={{ padding: '3px 8px', color: p.octreotide_resistant ? '#f59e0b' : '#475569' }}>{p.octreotide_resistant ? 'Yes' : 'No'}</td>
                            <td style={{ padding: '3px 8px', color: '#94a3b8', fontSize: 10 }}>{p.treatment}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ marginBottom: 12, color: '#64748b', fontSize: 12 }}>{definitions.count} clinical definitions</div>
          {(definitions.terms || []).map((t, i) => (
            <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 12 }}>
              <div style={{ fontWeight: 700, color: '#fbbf24', marginBottom: 8, fontSize: 13 }}>{t.term}</div>
              <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-line' }}>{t.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
