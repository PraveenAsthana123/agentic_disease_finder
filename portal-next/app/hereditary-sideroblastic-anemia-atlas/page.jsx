'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-sideroblastic-anemia-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ALAS2':    '#b71c1c',  // deep red   — most common XLSA, pyridoxine-responsive, rate-limiting haem enzyme
  'SLC25A38': '#1565c0',  // deep blue  — AR, severe neonatal, glycine importer, pyridoxine-non-responsive
  'GLRX5':    '#2e7d32',  // dark green  — Fe-S cluster scaffold, FECH cofactor, IRP1 dysregulation
  'HSPA9':    '#6a1b9a',  // deep purple — mortalin/GRP75, Fe-S delivery chaperone
  'ABCB7':    '#e65100',  // deep orange — X-linked ataxia (XLSA/A), Fe-S export to cytoplasm
  'PUS1':     '#00695c',  // dark teal   — MLASA1, mt-tRNA pseudouridylation, OXPHOS
  'YARS2':    '#37474f',  // dark slate  — MLASA2, mt-TyrRS, p.F52L hotspot, severe cardiomyopathy
  'TRNT1':    '#4527a0',  // deep indigo — SIFD, CCA-adding enzyme, immunodeficiency unique
};

const GENE_INFO = {
  'ALAS2':    { full: 'ALAS2 / 5-Aminolevulinate Synthase 2 / 587aa', locus: 'Xp11.21', size: '587 aa / 65 kDa (rate-limiting haem enzyme; PLP-dependent)', inh: 'XLR/GOF-AD' },
  'SLC25A38': { full: 'SLC25A38 / Mitochondrial Glycine Importer / 344aa', locus: '3p22.1', size: '344 aa / 38 kDa (mitochondrial glycine import; ALAS2 substrate supply)', inh: 'AR' },
  'GLRX5':    { full: 'GLRX5 / Glutaredoxin 5 / 157aa', locus: '14q32.13', size: '157 aa / 18 kDa ([2Fe-2S] scaffold; FECH cofactor delivery; IRP1 regulator)', inh: 'AR' },
  'HSPA9':    { full: 'HSPA9 / Mortalin·GRP75 / 679aa', locus: '5q31.2', size: '679 aa / 70 kDa (Hsp70 chaperone; Fe-S transfer from ISCU; multifunctional)', inh: 'AR' },
  'ABCB7':    { full: 'ABCB7 / ABC Transporter B7 / 752aa', locus: 'Xq13.3', size: '752 aa / 80 kDa (Fe-S export to cytoplasm; CIA pathway; XLSA/A ataxia)', inh: 'XLR' },
  'PUS1':     { full: 'PUS1 / Pseudouridine Synthase 1 / 445aa', locus: '12q24.33', size: '445 aa / 51 kDa (mt-tRNA pseudouridylation; OXPHOS support; MLASA1)', inh: 'AR' },
  'YARS2':    { full: 'YARS2 / Mitochondrial TyrRS / 477aa', locus: '12p11.21', size: '477 aa / 54 kDa (mt-tRNA(Tyr) aminoacylation; p.F52L hotspot; MLASA2)', inh: 'AR' },
  'TRNT1':    { full: 'TRNT1 / CCA-Adding Enzyme / 405aa', locus: '3p26.2', size: '405 aa / 45 kDa (universal tRNA 3′-CCA maintenance; SIFD syndrome)', inh: 'AR' },
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

export default function HereditorySideroblasticAnemiaAtlas() {
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

  const genes = overview?.genes || Object.keys(GENE_COLORS);

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#4a0000,#1a0030)', padding: '20px 28px 16px', borderBottom: '1px solid #1e293b' }}>
        <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>&#x1f9ec; Hereditary-Sideroblastic-Anemia-Atlas · Complete-8-Gene-Sideroblastic-Anaemia-Reference</div>
        <h1 style={{ fontSize: 20, fontWeight: 800, margin: 0, color: '#f8fafc' }}>
          Hereditary Sideroblastic Anemia Atlas
        </h1>
        <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
          ALAS2 · SLC25A38 · GLRX5 · HSPA9 · ABCB7 · PUS1 · YARS2 · TRNT1 &nbsp;|&nbsp; 320 patients · seeds 2814-2821
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          {genes.map(g => (
            <GeneChip key={g} gene={g} active={activeGene} onClick={g2 => setActiveGene(activeGene === g2 ? null : g2)} />
          ))}
        </div>
        {/* Tabs */}
        <div style={{ marginTop: 14, display: 'flex', gap: 4 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{ background: tab === t ? '#7c3aed' : '#1e293b', color: '#e2e8f0', border: 'none', borderRadius: 6, padding: '5px 14px', fontSize: 12, cursor: 'pointer', fontWeight: tab === t ? 700 : 400 }}>
              {t}
            </button>
          ))}
        </div>
      </div>

      <div style={{ padding: '20px 28px' }}>
        {loading && <div style={{ color: '#94a3b8', fontSize: 13 }}>Loading atlas data…</div>}
        {error && <div style={{ color: '#ef4444', fontSize: 13 }}>Error: {error}</div>}

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 genes × 40 each" />
              <MetricCard label="Gene Count" value={overview.genes?.length} sub="haem/Fe-S/MLASA/SIFD" />
              <MetricCard label="Seeds" value={overview.seeds} sub="deterministic cohort" />
              <MetricCard label="Pyridoxine-Responsive" value="ALAS2 only" sub="60-70%; others non-responsive" warn />
              <MetricCard label="Pathognomonic Syndromes" value="3" sub="XLSA/A · MLASA · SIFD" />
            </div>

            {/* Pathway categories */}
            {overview.pathway_categories?.map((cat, ci) => (
              <div key={ci} style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', marginBottom: 12, border: '1px solid #334155' }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#38bdf8', marginBottom: 8 }}>{cat.pathway}</div>
                <div style={{ marginBottom: 8 }}>
                  {cat.genes?.map(g => <GeneChip key={g} gene={g} active={null} />)}
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.6 }}>{cat.note}</div>
              </div>
            ))}

            {/* Critical distinctions */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', marginTop: 16, border: '1px solid #7c3aed' }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#a78bfa', marginBottom: 10 }}>Critical Clinical Distinctions</div>
              {overview.critical_distinctions?.map((d, i) => (
                <div key={i} style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6, paddingLeft: 12, borderLeft: '2px solid #7c3aed', lineHeight: 1.6 }}>
                  {d}
                </div>
              ))}
            </div>

            {/* Gene summary table */}
            <div style={{ marginTop: 20, overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene', 'Locus', 'Inh.', 'N', 'Median Hb (g/dL)', 'Ring SB %', 'B6 Resp %', 'Transfusion %', 'HSCT %', 'Syndromic %'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #334155' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.map((g, i) => (
                    <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1a2540' }}>
                      <td style={{ padding: '7px 10px' }}>
                        <GeneChip gene={g.gene} active={activeGene} onClick={gn => setActiveGene(activeGene === gn ? null : gn)} />
                      </td>
                      <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.locus}</td>
                      <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{GENE_INFO[g.gene]?.inh || '—'}</td>
                      <td style={{ padding: '7px 10px', color: '#e2e8f0' }}>{g.n_patients}</td>
                      <td style={{ padding: '7px 10px', color: '#e2e8f0' }}>{g.median_hb}</td>
                      <td style={{ padding: '7px 10px', color: '#fbbf24' }}>{g.mean_ring_sideroblast_pct}%</td>
                      <td style={{ padding: '7px 10px', color: g.pct_pyridoxine_response > 0 ? '#34d399' : '#ef4444' }}>
                        {g.pct_pyridoxine_response}%
                      </td>
                      <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.pct_transfusion_dep}%</td>
                      <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.pct_hsct}%</td>
                      <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.pct_syndromic_feature}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && (
          <div>
            <div style={{ marginBottom: 14, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {genes.map(g => (
                <GeneChip key={g} gene={g} active={activeGene} onClick={g2 => setActiveGene(activeGene === g2 ? null : g2)} />
              ))}
              {activeGene && <button onClick={() => setActiveGene(null)} style={{ background: '#334155', color: '#e2e8f0', border: 'none', borderRadius: 4, padding: '3px 10px', fontSize: 11, cursor: 'pointer' }}>Clear</button>}
            </div>
            {Object.entries(GENE_INFO)
              .filter(([g]) => !activeGene || g === activeGene)
              .map(([g, info]) => (
                <div key={g} style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', marginBottom: 10, borderLeft: `4px solid ${GENE_COLORS[g]}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                    <GeneChip gene={g} active={null} />
                    <span style={{ fontSize: 12, color: '#94a3b8' }}>{info.locus}</span>
                    <span style={{ fontSize: 11, color: '#64748b' }}>|</span>
                    <span style={{ fontSize: 11, color: '#7c3aed' }}>{info.inh}</span>
                  </div>
                  <div style={{ fontSize: 12, color: '#e2e8f0', marginBottom: 4 }}>{info.full}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>{info.size}</div>
                </div>
              ))}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            <div style={{ marginBottom: 14, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {genes.map(g => (
                <GeneChip key={g} gene={g} active={activeGene} onClick={g2 => setActiveGene(activeGene === g2 ? null : g2)} />
              ))}
              {activeGene && <button onClick={() => setActiveGene(null)} style={{ background: '#334155', color: '#e2e8f0', border: 'none', borderRadius: 4, padding: '3px 10px', fontSize: 11, cursor: 'pointer' }}>Clear filter</button>}
            </div>
            {breakdown.genes
              ?.filter(g => !activeGene || g.gene === activeGene)
              .map(g => (
                <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, marginBottom: 16, border: `1px solid ${GENE_COLORS[g.gene]}44` }}>
                  <div style={{ background: GENE_COLORS[g.gene] + '22', padding: '12px 18px', borderRadius: '10px 10px 0 0', borderBottom: `1px solid ${GENE_COLORS[g.gene]}44` }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <GeneChip gene={g.gene} active={null} />
                      <span style={{ fontSize: 11, color: '#94a3b8' }}>{g.locus}</span>
                      <span style={{ fontSize: 11, color: '#7c3aed' }}>{GENE_INFO[g.gene]?.inh}</span>
                      <span style={{ fontSize: 11, color: '#64748b' }}>N={g.n_patients}</span>
                    </div>
                    <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{g.protein_size}</div>
                  </div>
                  <div style={{ padding: '14px 18px' }}>
                    {[
                      { label: 'Disease Category', val: g.disease_category, col: '#fbbf24' },
                      { label: 'Inheritance & Mechanism', val: g.inheritance, col: '#38bdf8' },
                      { label: 'Molecular Pathway', val: g.disease_pathway, col: '#a78bfa' },
                      { label: 'Pathognomonic Features', val: g.pathognomonic, col: '#34d399' },
                      { label: 'Treatment', val: g.treatment, col: '#f87171' },
                    ].map(({ label, val, col }) => (
                      <div key={label} style={{ marginBottom: 12 }}>
                        <div style={{ fontSize: 11, fontWeight: 700, color: col, marginBottom: 4 }}>{label}</div>
                        <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{val}</div>
                      </div>
                    ))}

                    {/* Sample patients */}
                    {g.patients?.length > 0 && (
                      <div style={{ marginTop: 14 }}>
                        <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', marginBottom: 6 }}>Sample Patients (first 5 of 40)</div>
                        <div style={{ overflowX: 'auto' }}>
                          <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                            <thead>
                              <tr>
                                {['ID', 'Sex', 'Age Dx (y)', 'Hb (g/dL)', 'MCV (fL)', 'Ferritin', 'Ring SB%', 'B6 Resp', 'Transfusion', 'HSCT'].map(h => (
                                  <th key={h} style={{ padding: '4px 8px', color: '#475569', textAlign: 'left', borderBottom: '1px solid #334155' }}>{h}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {g.patients.map(p => (
                                <tr key={p.patient_id} style={{ borderBottom: '1px solid #1e293b' }}>
                                  <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{p.patient_id}</td>
                                  <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{p.sex}</td>
                                  <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{p.age_at_diagnosis_years}</td>
                                  <td style={{ padding: '3px 8px', color: '#e2e8f0' }}>{p.hb_gdl}</td>
                                  <td style={{ padding: '3px 8px', color: '#e2e8f0' }}>{p.mcv_fl}</td>
                                  <td style={{ padding: '3px 8px', color: '#fbbf24' }}>{p.ferritin_ngml}</td>
                                  <td style={{ padding: '3px 8px', color: '#fbbf24' }}>{p.ring_sideroblast_pct}%</td>
                                  <td style={{ padding: '3px 8px', color: p.pyridoxine_response ? '#34d399' : '#ef4444' }}>
                                    {p.pyridoxine_response ? 'Yes' : 'No'}
                                  </td>
                                  <td style={{ padding: '3px 8px', color: p.transfusion_dependent ? '#f87171' : '#34d399' }}>
                                    {p.transfusion_dependent ? 'Yes' : 'No'}
                                  </td>
                                  <td style={{ padding: '3px 8px', color: p.hsct_performed ? '#38bdf8' : '#64748b' }}>
                                    {p.hsct_performed ? 'Yes' : 'No'}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && defs && (
          <div>
            <div style={{ marginBottom: 14, fontSize: 12, color: '#64748b' }}>
              {defs.gene_count} genes · {defs.definitions?.length} definitions · seeds {defs.seeds}
            </div>
            {defs.definitions?.map((def, i) => (
              <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: '12px 18px', marginBottom: 10, borderLeft: '4px solid #7c3aed' }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#a78bfa', marginBottom: 6 }}>{def.term}</div>
                <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{def.definition}</div>
              </div>
            ))}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: '12px 18px', marginTop: 16, border: '1px solid #334155' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#64748b', marginBottom: 8 }}>Standards &amp; References</div>
              {defs.standards?.map((s, i) => (
                <div key={i} style={{ fontSize: 11, color: '#64748b', marginBottom: 4, paddingLeft: 8, borderLeft: '2px solid #334155' }}>{s}</div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
