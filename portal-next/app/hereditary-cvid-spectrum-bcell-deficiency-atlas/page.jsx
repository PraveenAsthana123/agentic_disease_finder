'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cvid-spectrum-bcell-deficiency-atlas';
const TABS = ['Overview', 'Gene Table', 'B-cell Pathway', 'Definitions'];

const GENE_COLORS = {
  'ICOS':       '#1b5e20',  // dark green     — AR; germinal center; T follicular helper
  'TNFRSF13B':  '#0d47a1',  // deep blue      — AR/AD; TACI; most common CVID gene
  'TNFRSF13C':  '#1565c0',  // mid blue       — AR; BAFF-R; B-cell survival
  'CD19':       '#6a1b9a',  // deep purple    — AR; B-cell co-receptor; normal B+agammaglobulinemia
  'CD81':       '#4a148c',  // darker purple  — AR; TAPA1; secondary CD19 loss
  'LRBA':       '#b71c1c',  // deep red       — AR; CTLA4 trafficking; abatacept
  'CTLA4':      '#e65100',  // deep orange    — AD LOF; Treg dysfunction; abatacept
  'PIK3CD':     '#f57f17',  // amber          — AD GOF; APDS1; leniolisib FDA2023
};

const GENE_INFO = {
  'ICOS':      { full: 'ICOS / Inducible T-Cell Costimulator / 199aa',         locus: '2q33.3',  size: '199 aa / 23 kDa (CD28-family costimulator; TFH differentiation; germinal center required; AR LOF; ALL isotypes low; NO memory B cells; Grimbacher 2003)',                                       inh: 'AR LOF' },
  'TNFRSF13B': { full: 'TNFRSF13B / TACI / Transmembrane Activator / 293aa',   locus: '17p11.2', size: '293 aa / 32 kDa (APRIL+BAFF receptor; class switch recombination; C104R+A181E key variants; most prevalent CVID gene 10-15%; biallelic=complete CVID; heterozygous=incomplete penetrance)',         inh: 'AR LOF / AD partial LOF' },
  'TNFRSF13C': { full: 'TNFRSF13C / BAFF-R / BAFF Receptor / 184aa',           locus: '22q13.1', size: '184 aa / 20 kDa (BAFF-exclusive receptor; non-canonical NF-κB; transitional→naive B cell survival; IgM decreases first; transitional B cells INCREASED; mildest CVID phenotype)',                  inh: 'AR LOF' },
  'CD19':      { full: 'CD19 / B-Lymphocyte Antigen CD19 / 556aa',              locus: '16p11.2', size: '556 aa / 95 kDa (co-receptor signaling subunit; BCR amplification ×1000; PI3K recruitment; AR LOF; NORMAL B cell count + severe agammaglobulinemia PATHOGNOMONIC; DDx XLA absent B cells)',         inh: 'AR LOF' },
  'CD81':      { full: 'CD81 / TAPA-1 / Tetraspanin / 236aa',                   locus: '11p15.5', size: '236 aa / 26 kDa (tetraspanin; CD19 surface stability/trafficking; AR LOF; secondary CD19 loss; check CD81 BEFORE CD19 in flow panel; CD81 absent on B+T cells)',                                    inh: 'AR LOF' },
  'LRBA':      { full: 'LRBA / LPS-Responsive Beige-Like Anchor / 2863aa',      locus: '4q31.3',  size: '2863 aa / 319 kDa (BEACH domain; recycling endosome; escorts CTLA4 from lysosomes to surface; AR LOF; CTLA4 absent on Tregs; abatacept dramatically effective; severe combined immune dysregulation)', inh: 'AR LOF' },
  'CTLA4':     { full: 'CTLA4 / Cytotoxic T-Lymphocyte Antigen 4 / 223aa',      locus: '2q33.2',  size: '223 aa / 25 kDa (B7 competitor; trans-endocytosis; Treg function; AD LOF haploinsufficiency; CTLA4 ~50% on Tregs; organ infiltration GLILD+gut+brain; abatacept first-line; AD fathers transmit)',   inh: 'AD LOF' },
  'PIK3CD':    { full: 'PIK3CD / PI3Kδ Catalytic Subunit / 1044aa',             locus: '1p36.22', size: '1044 aa / 119 kDa (PI3Kδ GOF E1021K; constitutive PIP3; Akt/mTOR; senescent CD8+CD57+CD28− PATHOGNOMONIC; CD21lo B cells; herpesvirus CMV/EBV; leniolisib FDA2023)',                                 inh: 'AD GOF' },
};

const PATHOGNOMONIC = {
  'ICOS':      'NO germinal centers → NO memory B cells; IgG/IgA/IgM ALL low; TFH absent; ICOS stimulation assay (anti-CD28 signal absent); AR; HSCT can cure',
  'TNFRSF13B': 'MOST PREVALENT CVID GENE (10-15%); APRIL-switch assay fails; biallelic=complete CVID; heterozygous=incomplete penetrance; C104R+A181E diagnostic; autoimmune complications 20-25%',
  'TNFRSF13C': 'IgM DECREASES FIRST (unique); transitional B cells INCREASED; mature naive B cells reduced; milder phenotype — some asymptomatic; BAFF-R surface flow diagnostic',
  'CD19':      'NORMAL B CELL COUNT + SEVERE AGAMMAGLOBULINEMIA — PATHOGNOMONIC DDx from XLA (XLA has absent B cells); CD19 flow absent; BCR signaling amplification lost',
  'CD81':      'CD81 (TAPA-1) tetraspanin required for CD19 surface stability → secondary CD19 loss on flow; CD81 staining absent; PATHWAY: CD81 → CD19 → BCR amplification',
  'LRBA':      'CTLA4 ABSENT ON TREG SURFACE (PATHOGNOMONIC); LRBA western blot absent; ABATACEPT (CTLA4-Ig) DRAMATICALLY EFFECTIVE; severe: IBD + autoimmunity + lymphoproliferation',
  'CTLA4':     'CTLA4 SURFACE EXPRESSION < 50% ON TREGS (AD); T cell organ infiltration WITHOUT infection; lung (GLILD) + gut + brain; ABATACEPT first-line; AD so fathers transmit',
  'PIK3CD':    'SENESCENT CD8+CD57+CD28− T CELLS PATHOGNOMONIC; CMV/EBV viremia; CD21lo B cells; LENIOLISIB FDA2023 (first PI3Kδ inhibitor for PID); bronchiectasis develops early',
};

export default function HereditoryCVIDSpectrumAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#1a1a2e', minHeight: '100vh', fontSize: 18 }}>Loading Hereditary-CVID-Spectrum-B-Cell-Deficiency-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#1a1a2e', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#e6edf3', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1b5e20 0%,#0d47a1 35%,#6a1b9a 65%,#e65100 100%)', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>Hereditary CVID-Spectrum · B-Cell Deficiency · Primary Antibody Deficiency · Immune Dysregulation</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
          🧬 Hereditary-CVID-Spectrum-B-Cell-Deficiency-Atlas
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
          Complete 8-Gene Reference · ICOS · TACI · BAFF-R · CD19 · CD81 · LRBA · CTLA4 · PIK3CD ·
          320 Patients (8×40) · Seeds 4102–4109
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Total Patients', val: overview?.total_patients },
            { label: 'Genes', val: overview?.genes?.length },
            { label: 'TACI', val: 'Most prevalent CVID gene' },
            { label: 'Abatacept-responsive', val: 'LRBA + CTLA4' },
            { label: 'PI3Kδ inhibitor', val: 'Leniolisib FDA2023' },
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
              <h3 style={{ margin: '0 0 14px', color: '#58a6ff', fontSize: 16 }}>⚡ Key Clinical Pearls — CVID-Spectrum / B-Cell Deficiency</h3>
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
                          ['Invasive Bacterial', `${cs?.invasive_bact_pct}%`],
                          ['Mycobacterial', `${cs?.mycobacterial_pct}%`],
                          ['Lymphoproliferation', cs?.lymphoproliferation ? 'YES' : 'Low'],
                          ['Autoinflammatory', `${cs?.autoinflammatory_pct}%`],
                          ['Abatacept', cs?.abatacept ? '✅ Indicated' : '—'],
                          ['Leniolisib', cs?.leniolisib ? '✅ FDA2023' : '—'],
                        ].map(([lbl, val]) => (
                          <div key={lbl} style={{ background: '#0d1117', borderRadius: 6, padding: '5px 8px' }}>
                            <div style={{ fontSize: 10, color: '#8b949e' }}>{lbl}</div>
                            <div style={{ fontSize: 13, fontWeight: 600, color: String(val).includes('✅') ? '#3fb950' : '#e6edf3' }}>{val}</div>
                          </div>
                        ))}
                      </div>
                      <div style={{ background: '#0d1117', borderRadius: 6, padding: '8px 10px', fontSize: 12, color: '#e6edf3', lineHeight: 1.5, borderLeft: `3px solid ${color}` }}>
                        {PATHOGNOMONIC[gene]}
                      </div>
                      <div style={{ marginTop: 8, fontSize: 11, color: '#8b949e' }}>Mean onset: <b>{cs?.mean_age_onset} yr</b> · HSCT: <b>{cs?.hsct_indicated ? 'Indicated' : 'Usually not'}</b></div>
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
                  {['Gene', 'Size', 'Locus', 'Inheritance', 'Syndrome', 'Invs Bact', 'Lymphoprolif', 'Abatacept', 'Leniolisib', 'HSCT', 'Mean Onset'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', color: '#8b949e', fontWeight: 600, fontSize: 12, borderBottom: '1px solid #30363d', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.genes || []).map((gene, i) => {
                  const cs = overview.cohort_summary?.[gene];
                  const info = GENE_INFO[gene];
                  const color = GENE_COLORS[gene] || '#444';
                  return (
                    <tr key={gene} style={{ background: i % 2 === 0 ? '#0d1117' : '#161b22', borderBottom: '1px solid #21262d' }}>
                      <td style={{ padding: '9px 12px', fontWeight: 700, color: color }}>{gene}</td>
                      <td style={{ padding: '9px 12px', color: '#8b949e', fontSize: 12 }}>{PROFILES_INFO[gene]?.size_aa ?? '—'} aa</td>
                      <td style={{ padding: '9px 12px', color: '#8b949e', fontSize: 12 }}>{cs?.locus}</td>
                      <td style={{ padding: '9px 12px', color: '#e6edf3', fontSize: 12 }}>{cs?.inheritance}</td>
                      <td style={{ padding: '9px 12px', color: '#cdd9e5', fontSize: 12, maxWidth: 200 }}>{cs?.syndrome}</td>
                      <td style={{ padding: '9px 12px', color: '#3fb950', fontWeight: 600 }}>{cs?.invasive_bact_pct}%</td>
                      <td style={{ padding: '9px 12px', color: cs?.lymphoproliferation_pct > 30 ? '#f0883e' : '#8b949e' }}>{cs?.lymphoproliferation_pct}%</td>
                      <td style={{ padding: '9px 12px', color: cs?.abatacept ? '#3fb950' : '#8b949e', fontWeight: cs?.abatacept ? 700 : 400 }}>{cs?.abatacept ? '✅ Yes' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.leniolisib ? '#f0883e' : '#8b949e', fontWeight: cs?.leniolisib ? 700 : 400 }}>{cs?.leniolisib ? '✅ FDA2023' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.hsct_indicated ? '#f0883e' : '#8b949e', fontWeight: cs?.hsct_indicated ? 700 : 400 }}>{cs?.hsct_indicated ? 'Yes' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: '#e6edf3' }}>{cs?.mean_age_onset} yr</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div style={{ marginTop: 20 }}>
              <h3 style={{ color: '#58a6ff', fontSize: 15, marginBottom: 12 }}>Gene Detail — Protein & Locus Reference</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))', gap: 12 }}>
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

        {/* B-CELL PATHWAY TAB */}
        {activeTab === 'B-cell Pathway' && breakdown && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(380px,1fr))', gap: 20, marginBottom: 28 }}>
              {Object.entries(breakdown.pathway_groups || {}).map(([groupKey, group]) => (
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
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Invasive bact: <b>{gs.invasive_bact_pct}%</b></div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Lymphoprolif: <b>{gs.lymphoproliferation_pct}%</b></div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Mean onset: <b>{gs.mean_age_onset} yr</b></div>
                  <div style={{ fontSize: 12, color: gs.abatacept ? '#3fb950' : '#8b949e', marginTop: 4 }}>Abatacept: <b>{gs.abatacept ? '✅ Yes' : '—'}</b></div>
                  <div style={{ fontSize: 12, color: gs.leniolisib ? '#f0883e' : '#8b949e' }}>Leniolisib: <b>{gs.leniolisib ? '✅ FDA2023' : '—'}</b></div>
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

const PROFILES_INFO = {
  'ICOS':      { size_aa: 199 },
  'TNFRSF13B': { size_aa: 293 },
  'TNFRSF13C': { size_aa: 184 },
  'CD19':      { size_aa: 556 },
  'CD81':      { size_aa: 236 },
  'LRBA':      { size_aa: 2863 },
  'CTLA4':     { size_aa: 223 },
  'PIK3CD':    { size_aa: 1044 },
};
