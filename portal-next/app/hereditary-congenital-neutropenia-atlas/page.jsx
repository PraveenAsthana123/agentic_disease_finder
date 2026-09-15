'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-congenital-neutropenia-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ELANE':  '#b71c1c',  // deep red   — most common SCN1, AD, ER stress/UPR
  'HAX1':   '#1565c0',  // deep blue  — Kostmann AR, neurological isoform
  'G6PC3':  '#7b1fa2',  // deep purple — Dursun SCN4, cardiac/urogenital/ear
  'WAS':    '#2e7d32',  // dark green  — WAS/XLN, XLR, actin cytoskeleton
  'CXCR4':  '#e65100',  // deep orange — WHIM GOF, myelokathexis, plerixafor
  'GFI1':   '#00695c',  // dark teal   — SCN2 DN, ELANE repressor
  'VPS45':  '#37474f',  // dark slate  — SCN5 AR, BM fibrosis, nephromegaly
  'JAGN1':  '#4527a0',  // deep indigo — SCN7 AR, glycosylation, G-CSF resistance
};

const GENE_INFO = {
  'ELANE':  { full: 'ELANE / Neutrophil Elastase / 267aa', locus: '19p13.3', size: '267 aa / 29 kDa (serine protease; azurophil granule; ER-stress apoptosis)', inh: 'AD' },
  'HAX1':   { full: 'HAX1 / HCLS1-Associated Protein X-1 / 279aa', locus: '1q21.3', size: '279 aa / 32 kDa (mitochondrial/ER anti-apoptotic adaptor; isoform-dependent neurology)', inh: 'AR' },
  'G6PC3':  { full: 'G6PC3 / Glucose-6-Phosphatase Catalytic-3 / 346aa', locus: '17q21.31', size: '346 aa / 37 kDa (ER glucose-6-phosphatase; 9-TM; Dursun syndrome)', inh: 'AR' },
  'WAS':    { full: 'WAS / Wiskott-Aldrich Syndrome Protein / 502aa', locus: 'Xp11.22', size: '502 aa / 53 kDa (haematopoietic Arp2/3 activator; LOF=WAS; GOF=XLN)', inh: 'XLR' },
  'CXCR4':  { full: 'CXCR4 / CXC Chemokine Receptor 4 / 360aa', locus: '2q22.1', size: '360 aa / 39 kDa (7-TM GPCR; SDF-1 receptor; GOF C-terminal truncation)', inh: 'AD' },
  'GFI1':   { full: 'GFI1 / Growth Factor Independence 1 / 422aa', locus: '1p22.1', size: '422 aa / 46 kDa (zinc-finger transcriptional repressor; ELANE repressor; DN-SCN2)', inh: 'AD' },
  'VPS45':  { full: 'VPS45 / Vacuolar Protein Sorting-45 / 578aa', locus: '1q21.2', size: '578 aa / 65 kDa (SM-protein; SNARE vesicle fusion; lysosomal trafficking)', inh: 'AR' },
  'JAGN1':  { full: 'JAGN1 / Jagunal Homolog 1 / 183aa', locus: '3p25.3', size: '183 aa / 21 kDa (ER N-glycosylation QC; G-CSFR misglycosylated; G-CSF resistance)', inh: 'AR' },
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

export default function HereditaryCongenitalNeutropeniaAtlas() {
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

  const geneList = Object.keys(GENE_COLORS);

  if (loading) return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#94a3b8', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18 }}>
      Loading Congenital Neutropenia Atlas…
    </div>
  );
  if (error) return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#ef4444', padding: 32, fontSize: 16 }}>
      Error: {error}
    </div>
  );

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%)', borderBottom: '1px solid #1e3a5f', padding: '20px 28px 14px' }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: '#38bdf8', letterSpacing: 1 }}>
          🧬 Hereditary Congenital Neutropenia Atlas
        </div>
        <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>
          Complete 8-Gene SCN / WHIM / Kostmann / WAS-XLN Reference — ELANE · HAX1 · G6PC3 · WAS · CXCR4 · GFI1 · VPS45 · JAGN1
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          {geneList.map(g => (
            <GeneChip key={g} gene={g} active={activeGene} onClick={g => setActiveGene(activeGene === g ? null : g)} />
          ))}
        </div>
        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginTop: 14 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              background: tab === t ? '#38bdf8' : '#1e293b',
              color: tab === t ? '#0f172a' : '#94a3b8',
              border: 'none', borderRadius: 4, padding: '5px 14px', fontSize: 12,
              fontWeight: tab === t ? 700 : 400, cursor: 'pointer',
            }}>{t}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: '20px 28px' }}>

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2822-2829" />
              <MetricCard label="Genes Covered" value={overview.genes?.length} sub="SCN1/3/4/5/7 + WAS/WHIM/GFI1" />
              <MetricCard label="Inheritance Types" value="AD/AR/XLR" sub="all modes represented" />
              <MetricCard label="Unique Mechanisms" value="5" sub="ER stress / egress / actin / glycosyl / vesicle" />
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 10 }}>
              Gene Summary — G-CSF Response &amp; AML Risk
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene','Locus','Inh','Pts','Median ANC','Inf/yr','G-CSF Resp %','AML/MDS %','HSCT %','Warts %','BM Fibr %','Neuro %'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', color: '#38bdf8', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.filter(g => !activeGene || g.gene === activeGene).map((g, i) => (
                    <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1a2332', cursor: 'pointer' }}
                        onClick={() => setActiveGene(activeGene === g.gene ? null : g.gene)}>
                      <td style={{ padding: '5px 8px' }}>
                        <GeneChip gene={g.gene} active={activeGene} />
                      </td>
                      <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{g.locus}</td>
                      <td style={{ padding: '5px 8px', color: '#fbbf24' }}>{GENE_INFO[g.gene]?.inh}</td>
                      <td style={{ padding: '5px 8px' }}>{g.n_patients}</td>
                      <td style={{ padding: '5px 8px', color: g.median_anc_nadir < 0.1 ? '#ef4444' : '#fbbf24' }}>
                        {g.median_anc_nadir}
                      </td>
                      <td style={{ padding: '5px 8px' }}>{g.mean_infections_per_year}</td>
                      <td style={{ padding: '5px 8px', color: g.pct_gcsf_response < 60 ? '#ef4444' : '#4ade80' }}>
                        {g.pct_gcsf_response}%
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_aml_mds > 20 ? '#ef4444' : '#fbbf24' }}>
                        {g.pct_aml_mds}%
                      </td>
                      <td style={{ padding: '5px 8px' }}>{g.pct_hsct}%</td>
                      <td style={{ padding: '5px 8px', color: g.pct_warts > 0 ? '#fbbf24' : '#475569' }}>
                        {g.pct_warts > 0 ? `${g.pct_warts}%` : '—'}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_bm_fibrosis > 0 ? '#ef4444' : '#475569' }}>
                        {g.pct_bm_fibrosis > 0 ? `${g.pct_bm_fibrosis}%` : '—'}
                      </td>
                      <td style={{ padding: '5px 8px', color: g.pct_neuro_phenotype > 0 ? '#fbbf24' : '#475569' }}>
                        {g.pct_neuro_phenotype > 0 ? `${g.pct_neuro_phenotype}%` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginTop: 24, marginBottom: 10 }}>
              Pathway Categories
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 12 }}>
              {overview.pathway_categories?.map((cat, i) => (
                <div key={i} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, color: '#fbbf24', fontSize: 12, marginBottom: 6 }}>{cat.pathway}</div>
                  <div style={{ marginBottom: 6, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                    {cat.genes?.map(g => <GeneChip key={g} gene={g} active={null} />)}
                  </div>
                  <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>{cat.note}</div>
                </div>
              ))}
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginTop: 24, marginBottom: 10 }}>
              Critical Distinctions
            </div>
            {overview.critical_distinctions?.map((d, i) => (
              <div key={i} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6, padding: '8px 12px', marginBottom: 6, fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>
                ⚡ {d}
              </div>
            ))}
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && (
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 14 }}>
              Gene Reference — Protein, Locus, Inheritance
            </div>
            {geneList.filter(g => !activeGene || g === activeGene).map(g => (
              <div key={g} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[g]}44`, borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <GeneChip gene={g} active={null} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{GENE_INFO[g]?.locus}</span>
                  <span style={{ color: '#fbbf24', fontSize: 12, fontWeight: 700 }}>{GENE_INFO[g]?.inh}</span>
                </div>
                <div style={{ fontSize: 12, color: '#e2e8f0', marginBottom: 4 }}>{GENE_INFO[g]?.full}</div>
                <div style={{ fontSize: 11, color: '#64748b' }}>{GENE_INFO[g]?.size}</div>
              </div>
            ))}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            {breakdown.genes?.filter(g => !activeGene || g.gene === activeGene).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[g.gene]}55`, borderRadius: 10, padding: 18, marginBottom: 18 }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10 }}>
                  <GeneChip gene={g.gene} active={null} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus}</span>
                  <span style={{ color: '#fbbf24', fontSize: 11 }}>{GENE_INFO[g.gene]?.inh}</span>
                  <span style={{ color: '#64748b', fontSize: 11 }}>{g.n_patients} patients</span>
                </div>
                {[['Protein', g.protein_size], ['Inheritance', g.inheritance], ['Disease', g.disease_category],
                  ['Pathway', g.disease_pathway], ['Pathognomonic', g.pathognomonic], ['Treatment', g.treatment]].map(([label, val]) => (
                  <div key={label} style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, color: '#38bdf8', fontWeight: 700, marginBottom: 2 }}>{label.toUpperCase()}</div>
                    <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{val}</div>
                  </div>
                ))}
                {g.patients && g.patients.length > 0 && (
                  <div>
                    <div style={{ fontSize: 11, color: '#38bdf8', fontWeight: 700, marginBottom: 6 }}>SAMPLE PATIENTS (first 5)</div>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                        <thead>
                          <tr style={{ background: '#0f172a' }}>
                            {['ID','Sex','Age Dx','ANC Nadir','Inf/yr','G-CSF Resp','AML/MDS','HSCT'].map(h => (
                              <th key={h} style={{ padding: '4px 8px', color: '#64748b', textAlign: 'left' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {g.patients.map((p, i) => (
                            <tr key={p.patient_id} style={{ background: i % 2 === 0 ? '#1a2332' : '#0f172a' }}>
                              <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{p.patient_id}</td>
                              <td style={{ padding: '3px 8px' }}>{p.sex}</td>
                              <td style={{ padding: '3px 8px' }}>{p.age_at_diagnosis_years}y</td>
                              <td style={{ padding: '3px 8px', color: p.anc_nadir_per_uL < 0.1 ? '#ef4444' : '#fbbf24' }}>{p.anc_nadir_per_uL}</td>
                              <td style={{ padding: '3px 8px' }}>{p.infections_per_year}</td>
                              <td style={{ padding: '3px 8px', color: p.gcsf_response ? '#4ade80' : '#ef4444' }}>{p.gcsf_response ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px', color: p.aml_mds_event ? '#ef4444' : '#475569' }}>{p.aml_mds_event ? 'Yes' : 'No'}</td>
                              <td style={{ padding: '3px 8px' }}>{p.hsct_performed ? 'Yes' : 'No'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && defs && (
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 14 }}>
              Glossary — SCN / WHIM / Kostmann / WAS-XLN / Myelokathexis / Plerixafor
            </div>
            {defs.definitions?.map((d, i) => (
              <div key={i} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: '12px 16px', marginBottom: 10 }}>
                <div style={{ fontWeight: 700, color: '#fbbf24', fontSize: 13, marginBottom: 6 }}>{d.term}</div>
                <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{d.definition}</div>
              </div>
            ))}
            <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginTop: 20, marginBottom: 10 }}>
              Standards &amp; References
            </div>
            {defs.standards?.map((s, i) => (
              <div key={i} style={{ background: '#1e293b', border: '1px solid #1e3a5f', borderRadius: 6, padding: '6px 12px', marginBottom: 6, fontSize: 11, color: '#94a3b8' }}>
                📋 {s}
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  );
}
