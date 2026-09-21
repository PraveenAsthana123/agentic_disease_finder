'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-aml-mds-jmml-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'GATA2':  '#b71c1c',  // deep red       — GATA2 deficiency / MonoMAC / MDS
  'DDX41':  '#1565c0',  // deep blue      — adult AML germline, RNA helicase
  'CEBPA':  '#2e7d32',  // dark green     — familial AML biallelic bZIP
  'ETV6':   '#6a1b9a',  // deep purple    — ETS TF, thrombocytopenia + ALL/AML
  'SAMD9':  '#e65100',  // deep orange    — MIRAGE, monosomy 7 GOF
  'SAMD9L': '#00695c',  // dark teal      — ataxia-pancytopenia GOF
  'PTPN11': '#c62828',  // crimson        — SHP2, Noonan-JMML RASopathy
  'CBL':    '#4527a0',  // deep indigo    — E3 ligase, CBL-JMML UPD11q
};

const GENE_INFO = {
  'GATA2':  { full: 'GATA2 / GATA-Binding-Protein-2 / 480aa', locus: '3q21.3', size: '480 aa / 50 kDa (ZnF TF; HSC master regulator; LOF haploinsufficiency; MonoMAC: monocytes near-absent + NTM; Emberger: lymphedema + MDS + SNHL; DCML deficiency; monosomy 7 MDS → AML; HSCT only cure)', inh: 'AD' },
  'DDX41':  { full: 'DDX41 / DEAD-Box-Helicase-41 / 622aa', locus: '5q35.3', size: '622 aa / 68 kDa (RNA helicase + innate immune STING dsDNA sensor; most common adult AML germline ~4%; D140Gfs germline + R525H somatic biallelic; male predominance; sibling donor exclusion critical)', inh: 'AD' },
  'CEBPA':  { full: 'CEBPA / C/EBP-alpha / 358aa', locus: '19q13.11', size: '358 aa / 42 kDa (bZIP myeloid TF; germline N-terminal frameshift + somatic C-terminal bZIP = biallelic AML; ELN favourable; CR1 >90%; buccal germline testing mandatory in all biallelic CEBPA AML)', inh: 'AD' },
  'ETV6':   { full: 'ETV6 / TEL / 452aa', locus: '12p13.2', size: '452 aa / 57 kDa (ETS TF; PNT + ETS domains; ETS dominant-negative; thrombocytopenia 30-150k lifelong (ITP misdiagnosis); dense granule release defect; 15-35% lifetime ALL/AML/MDS)', inh: 'AD' },
  'SAMD9':  { full: 'SAMD9 / Sterile-Alpha-Motif-Domain-9 / 1589aa', locus: '7q21.2', size: '1589 aa / 170 kDa (IFN-inducible antiproliferative; GOF = MIRAGE: Myelodysplasia + Infections + Restriction + Adrenal + Genital + Enteropathy; monosomy 7 = BM somatic rescue; adrenal crisis at birth if missed)', inh: 'AD GOF' },
  'SAMD9L': { full: 'SAMD9L / Sterile-Alpha-Motif-Domain-9-Like / 1589aa', locus: '7q21.2', size: '1589 aa / 170 kDa (SAMD9 paralogue; cerebellar Purkinje high expression → ataxia; ataxia-pancytopenia syndrome + monosomy 7; HSCT corrects BM NOT ataxia — critical counselling)', inh: 'AD GOF' },
  'PTPN11': { full: 'PTPN11 / SHP2 / 593aa', locus: '12q24.13', size: '593 aa / 68 kDa (SH2-PTP; N-SH2 autoinhibition; GOF = RAS-MAPK hyperactivation; Noonan syndrome (50%) + JMML (25% of JMML); GM-CSF hypersensitivity; HbF elevated; spontaneous remission in Noonan-JMML 20-30%)', inh: 'AD GOF' },
  'CBL':    { full: 'CBL / Casitas-B-Lineage-Lymphoma / 906aa', locus: '11q23.3', size: '906 aa / 100 kDa (RING E3 ubiquitin ligase; RTK degradation; germline LOF + UPD11q23 somatic = CBL-JMML; UPD11q = near-pathognomonic; vasculitis unique to CBL; spontaneous remission 30-50%)', inh: 'AD' },
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

export default function HereditaryAMLMDSJMMLPredispositionAtlas() {
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

  if (loading) return <div style={{ color: '#94a3b8', padding: 40 }}>Loading Hereditary-AML-MDS-JMML-Predisposition-Atlas…</div>;
  if (error) return <div style={{ color: '#ef4444', padding: 40 }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1e293b,#0f172a)', borderBottom: '1px solid #334155', padding: '20px 32px' }}>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>HEREDITARY MYELOID / LYMPHOID MALIGNANCY PREDISPOSITION</div>
        <div style={{ fontSize: 20, fontWeight: 700, color: '#f1f5f9' }}>
          🧬 Hereditary-AML-MDS-JMML-Predisposition-Atlas
        </div>
        <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
          Complete 8-Gene Germline AML / MDS / ALL / JMML / Monosomy-7 Syndrome Reference
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
              <MetricCard label="Genes" value={overview.n_genes} sub="seeds 2870-2877" />
              <MetricCard label="AML Predisposition" value="GATA2 · DDX41 · CEBPA" sub="3 genes" />
              <MetricCard label="JMML" value="PTPN11 · CBL" sub="RASopathy-driven" />
              <MetricCard label="Monosomy 7" value="GATA2 · SAMD9 · SAMD9L" sub="somatic adaptation" />
              <MetricCard label="ALL Predisposition" value="ETV6" sub="thrombocytopenia + ALL" />
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
              {Object.keys(defs.glossary || {}).length} definitions · Hereditary-AML-MDS-JMML-Predisposition-Atlas
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
