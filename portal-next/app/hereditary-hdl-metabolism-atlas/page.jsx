'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hdl-metabolism-atlas';
const TABS = ['Overview', 'Gene Breakdown', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ABCA1':  '#b71c1c',  // deep red — Tangier disease; orange tonsils PATHOGNOMONIC; HDL=0; RCT initiator
  'APOA1':  '#e65100',  // deep orange — ApoA-I Milano paradox; LCAT activator; ABCA1 substrate; RCT scaffold
  'LCAT':   '#f57f17',  // amber — sole HDL CE esterification; FLD triad; Fish-eye disease; Lp-X
  'LIPC':   '#1b5e20',  // dark forest green — hepatic lipase; high HDL+TG paradox; post-heparin PHLA
  'CETP':   '#004d40',  // dark teal — CETP deficiency; Asp442Gly Japanese founder; centenarian longevity
  'APOE':   '#4a148c',  // deep purple — Type III HLP; palmar xanthomas PATHOGNOMONIC; ε2/ε2 + second hit
  'SCARB1': '#37474f',  // dark slate — SR-BI; RCT final step; high HDL + premature CAD; adrenal crisis
  'LIPG':   '#006064',  // dark cyan — endothelial lipase; inflammation-HDL drop; phospholipase A1 on HDL
};

const GENE_INFO = {
  'ABCA1':  { full: 'ABCA1 / ATP-Binding Cassette Transporter A1 / 2058aa', locus: '9q31.1',  size: '2058 aa / 254 kDa (12-TM ABC transporter; flippase FC+PL; ABCA1-substrate = lipid-poor ApoA-I; rate-limiting nascent HDL disc)',     inh: 'AR (Tangier) / AD (FHA)' },
  'APOA1':  { full: 'APOA1 / Apolipoprotein A-I / 267aa',                   locus: '11q23.3', size: '267 aa / 28 kDa (major HDL structural protein 70-80%; LCAT obligate activator; ABCA1 substrate; Milano Arg173Cys = low HDL no CAD)', inh: 'AD (dominant-negative / LOF)' },
  'LCAT':   { full: 'LCAT / Lecithin-Cholesterol Acyltransferase / 440aa',  locus: '16q22.1', size: '440 aa / 50 kDa (serine esterase; ApoA-I-activated; FC+PC → CE; HDL disc → sphere; FLD=complete/FED=partial LCAT)',              inh: 'AR' },
  'LIPC':   { full: 'LIPC / Hepatic Lipase / 476aa',                        locus: '15q21.3', size: '476 aa / 53 kDa (TG-lipase + phospholipase A1; GPI-anchored hepatocyte; HDL-2→HDL-3 remodelling; IDL remnant clearance)',          inh: 'AR' },
  'CETP':   { full: 'CETP / Cholesteryl Ester Transfer Protein / 493aa',    locus: '16q21',   size: '493 aa / 53 kDa (boomerang lipid transfer; CE HDL→VLDL/LDL exchange for TG; 40-70% HDL-CE catabolism; Asp442Gly Japanese 7%)',    inh: 'AD (LOF) / AR (complete)' },
  'APOE':   { full: 'APOE / Apolipoprotein E / 317aa',                      locus: '19q13.32',size: '317 aa / 36 kDa (ε2/ε3/ε4 isoforms; LDLR+LRP1+HSPG ligand; remnant clearance; ε2/ε2+second hit=Type III HLP; ε4=Alzheimer risk)', inh: 'AD (ε2/ε2 requires second hit)' },
  'SCARB1': { full: 'SCARB1 / Scavenger Receptor Class B Type I / 509aa',   locus: '12q24.31',size: '509 aa / 57 kDa (selective CE uptake from HDL; no endocytosis; hepatic RCT final step; adrenal/gonadal steroidogenesis supply)', inh: 'AR' },
  'LIPG':   { full: 'LIPG / Endothelial Lipase / 500aa',                    locus: '18q21.1', size: '500 aa / 68 kDa (vascular endothelium GPI-anchored; phospholipase A1 on HDL; inflammation-induced; EL surge → HDL-C drop in sepsis)', inh: 'AR (LOF → elevated HDL)' },
};

function GeneChip({ gene, active, onClick }) {
  return (
    <button
      onClick={() => onClick(gene)}
      style={{
        background: active ? GENE_COLORS[gene] : '#263238',
        color: '#fff',
        border: `2px solid ${GENE_COLORS[gene]}`,
        borderRadius: 8,
        padding: '6px 14px',
        margin: 4,
        cursor: 'pointer',
        fontWeight: active ? 700 : 400,
        fontSize: 13,
        transition: 'all 0.15s',
      }}
    >
      {gene}
    </button>
  );
}

function StatBadge({ label, value, color }) {
  return (
    <div style={{ background: '#1e2a31', border: `1px solid ${color || '#37474f'}`, borderRadius: 8, padding: '10px 14px', minWidth: 100, textAlign: 'center', margin: 4 }}>
      <div style={{ color: color || '#90a4ae', fontSize: 10, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#fff', fontSize: 20, fontWeight: 700 }}>{value}</div>
    </div>
  );
}

function Section({ title, children, color }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ color: color || '#90a4ae', fontWeight: 700, fontSize: 13, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>{title}</div>
      {children}
    </div>
  );
}

function ClinicalText({ text }) {
  if (!text) return null;
  return (
    <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12, fontSize: 12, color: '#cfd8dc', lineHeight: 1.7, whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
      {text}
    </div>
  );
}

export default function HHDLAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [activeGene, setActiveGene] = useState('ABCA1');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, []);

  const genes = Object.keys(GENE_COLORS);

  if (loading) return <div style={{ background: '#0d1b21', minHeight: '100vh', color: '#90a4ae', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18 }}>Loading Hereditary HDL-Metabolism Atlas…</div>;
  if (error) return <div style={{ background: '#0d1b21', minHeight: '100vh', color: '#ef9a9a', padding: 32, fontSize: 14 }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1b21', minHeight: '100vh', color: '#cfd8dc', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: '#0a1520', borderBottom: '2px solid #1e3a4a', padding: '18px 28px' }}>
        <div style={{ fontSize: 22, fontWeight: 700, color: '#e0f7fa', marginBottom: 4 }}>
          🧬 Hereditary-HDL-Metabolism-Atlas
        </div>
        <div style={{ fontSize: 12, color: '#90a4ae' }}>
          Complete 8-Gene HDL Metabolism &amp; Reverse Cholesterol Transport Reference &nbsp;·&nbsp;
          ABCA1-APOA1-LCAT-LIPC-CETP-APOE-SCARB1-LIPG &nbsp;·&nbsp; 320 patients · seeds 2774-2781
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', background: '#0f2029', borderBottom: '1px solid #1e3a4a', padding: '0 28px' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: 'none', border: 'none', borderBottom: tab === t ? '3px solid #26c6da' : '3px solid transparent',
            color: tab === t ? '#26c6da' : '#78909c', padding: '12px 18px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 28px' }}>

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
              <StatBadge label="TOTAL PATIENTS" value={overview.total_patients} color="#26c6da" />
              <StatBadge label="GENES" value={overview.genes?.length} color="#4dd0e1" />
              <StatBadge label="SEEDS" value={overview.seeds} color="#80cbc4" />
              <StatBadge label="AVG HDL-C (mg/dL)" value={overview.summary?.avg_hdl_c_mgdl} color="#81d4fa" />
              <StatBadge label="AVG LDL-C (mg/dL)" value={overview.summary?.avg_ldl_c_mgdl} color="#ce93d8" />
              <StatBadge label="AVG TG (mg/dL)" value={overview.summary?.avg_tg_mgdl} color="#ffcc02" />
            </div>

            <Section title="RCT Pathway Steps (all 7 genes mapped)" color="#26c6da">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 10 }}>
                {Object.entries(overview.rct_pathway_steps || {}).map(([step, desc]) => (
                  <div key={step} style={{ background: '#1e2a31', borderRadius: 8, padding: 10, borderLeft: `3px solid #26c6da` }}>
                    <div style={{ color: '#26c6da', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>{step.replace(/_/g,' ').toUpperCase()}</div>
                    <div style={{ fontSize: 12, color: '#b0bec5' }}>{desc}</div>
                  </div>
                ))}
              </div>
            </Section>

            <Section title="Pathognomonic Signs by Gene" color="#ef9a9a">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 8 }}>
                {Object.entries(overview.pathognomonic_signs || {}).map(([gene, sign]) => (
                  <div key={gene} style={{ background: '#1e2a31', borderRadius: 8, padding: 10, borderLeft: `3px solid ${GENE_COLORS[gene] || '#546e7a'}` }}>
                    <div style={{ color: GENE_COLORS[gene] || '#90a4ae', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>{gene}</div>
                    <div style={{ fontSize: 11, color: '#cfd8dc' }}>{sign}</div>
                  </div>
                ))}
              </div>
            </Section>

            <Section title="Per-Gene Summary" color="#80cbc4">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 8 }}>
                {Object.entries(overview.summary?.per_gene || {}).map(([gene, info]) => (
                  <div key={gene} style={{ background: '#1e2a31', borderRadius: 8, padding: 12, borderLeft: `3px solid ${GENE_COLORS[gene] || '#546e7a'}` }}>
                    <div style={{ color: GENE_COLORS[gene] || '#90a4ae', fontWeight: 700, fontSize: 14, marginBottom: 6 }}>{gene}</div>
                    <div style={{ fontSize: 11, color: '#90a4ae' }}>n={info.n} · {info.locus}</div>
                    <div style={{ fontSize: 11, color: '#b0bec5', marginTop: 4 }}>{info.protein_size}</div>
                    <div style={{ fontSize: 11, color: '#cfd8dc', marginTop: 4 }}>
                      HDL: {info.avg_hdl} · LDL: {info.avg_ldl} · TG: {info.avg_tg} mg/dL
                    </div>
                  </div>
                ))}
              </div>
            </Section>

            <Section title="Cascade Testing Protocol" color="#ce93d8">
              <ClinicalText text={overview.cascade_testing} />
            </Section>
          </div>
        )}

        {/* GENE BREAKDOWN TAB */}
        {tab === 'Gene Breakdown' && breakdown && (
          <div>
            <div style={{ marginBottom: 16 }}>
              {genes.map(g => <GeneChip key={g} gene={g} active={activeGene === g} onClick={setActiveGene} />)}
            </div>
            {breakdown[activeGene] && (() => {
              const g = breakdown[activeGene];
              return (
                <div>
                  <div style={{ background: GENE_COLORS[activeGene], borderRadius: 10, padding: '12px 18px', marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
                    <div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: '#fff' }}>{g.gene}</div>
                      <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.85)', marginTop: 2 }}>{g.locus} · {g.protein_size}</div>
                    </div>
                    <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: 8, padding: '8px 14px', textAlign: 'center' }}>
                      <div style={{ color: '#fff', fontSize: 22, fontWeight: 700 }}>{g.n_patients}</div>
                      <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: 10 }}>PATIENTS</div>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
                    <div>
                      <Section title="Inheritance" color={GENE_COLORS[activeGene]}>
                        <ClinicalText text={g.inheritance} />
                      </Section>
                      <Section title="Disease Pathway" color={GENE_COLORS[activeGene]}>
                        <ClinicalText text={g.disease_pathway} />
                      </Section>
                    </div>
                    <div>
                      <Section title="Disease Category" color={GENE_COLORS[activeGene]}>
                        <ClinicalText text={g.disease_category} />
                      </Section>
                      <Section title="Pathognomonic Signs" color="#ef9a9a">
                        <ClinicalText text={g.pathognomonic} />
                      </Section>
                    </div>
                  </div>

                  <Section title="Key Facts" color="#80cbc4">
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {(g.key_facts || []).map(f => (
                        <span key={f} style={{ background: '#1e2a31', border: `1px solid ${GENE_COLORS[activeGene]}`, borderRadius: 6, padding: '4px 10px', fontSize: 11, color: '#cfd8dc' }}>{f}</span>
                      ))}
                    </div>
                  </Section>

                  <Section title="Treatment" color="#a5d6a7">
                    <ClinicalText text={g.treatment} />
                  </Section>

                  <Section title="Sample Patients (n=5 of 40)" color="#90a4ae">
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                        <thead>
                          <tr style={{ background: '#1e2a31' }}>
                            {['Patient ID','Age','Sex','HDL-C','LDL-C','TG','TC','ApoA-I','Key Phenotype'].map(h => (
                              <th key={h} style={{ padding: '8px 10px', color: '#80cbc4', textAlign: 'left', borderBottom: '1px solid #263238' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {(g.sample_patients || []).map((p, i) => (
                            <tr key={i} style={{ borderBottom: '1px solid #1e2a31', background: i % 2 === 0 ? '#131f27' : '#0d1b21' }}>
                              <td style={{ padding: '6px 10px', color: '#cfd8dc' }}>{p.patient_id}</td>
                              <td style={{ padding: '6px 10px', color: '#cfd8dc' }}>{p.age}</td>
                              <td style={{ padding: '6px 10px', color: '#cfd8dc' }}>{p.sex}</td>
                              <td style={{ padding: '6px 10px', color: '#4dd0e1', fontWeight: 600 }}>{p.hdl_c}</td>
                              <td style={{ padding: '6px 10px', color: '#cfd8dc' }}>{p.ldl_c}</td>
                              <td style={{ padding: '6px 10px', color: '#cfd8dc' }}>{p.tg}</td>
                              <td style={{ padding: '6px 10px', color: '#cfd8dc' }}>{p.total_cholesterol}</td>
                              <td style={{ padding: '6px 10px', color: '#cfd8dc' }}>{p.apoa1_mgdl}</td>
                              <td style={{ padding: '6px 10px', color: '#ffcc02', fontSize: 10 }}>{p.key_phenotype}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Section>
                </div>
              );
            })()}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            <div style={{ marginBottom: 16, color: '#90a4ae', fontSize: 13 }}>
              All 8 genes — complete protein atlas with locus, size, inheritance, and clinical classification
            </div>
            {genes.map(gene => {
              const g = breakdown[gene];
              if (!g) return null;
              const info = GENE_INFO[gene];
              return (
                <div key={gene} style={{ background: '#131f27', borderRadius: 10, padding: '14px 18px', marginBottom: 14, borderLeft: `4px solid ${GENE_COLORS[gene]}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 10 }}>
                    <div>
                      <span style={{ color: GENE_COLORS[gene], fontWeight: 700, fontSize: 16 }}>{gene}</span>
                      <span style={{ color: '#78909c', fontSize: 12, marginLeft: 10 }}>{info?.locus}</span>
                      <span style={{ color: '#78909c', fontSize: 12, marginLeft: 10 }}>{info?.inh}</span>
                    </div>
                    <span style={{ background: '#1e2a31', borderRadius: 6, padding: '3px 10px', fontSize: 11, color: '#80cbc4' }}>{g.n_patients} patients · seed {g.seed}</span>
                  </div>
                  <div style={{ fontSize: 12, color: '#90a4ae', marginBottom: 6 }}>{info?.size}</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 8 }}>
                    {(g.key_facts || []).map(f => (
                      <span key={f} style={{ background: '#1e2a31', border: `1px solid ${GENE_COLORS[gene]}33`, borderRadius: 5, padding: '3px 8px', fontSize: 11, color: '#b0bec5' }}>{f}</span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            <div style={{ marginBottom: 14, color: '#90a4ae', fontSize: 13 }}>
              {Object.keys(definitions.terms || {}).length} terms — HDL metabolism, RCT, lipid transfer, pathognomonic signs, pharmacogenomics
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 10 }}>
              {Object.entries(definitions.terms || {}).map(([term, def]) => (
                <div key={term} style={{ background: '#131f27', borderRadius: 8, padding: 12, borderLeft: '3px solid #26c6da' }}>
                  <div style={{ color: '#26c6da', fontWeight: 700, fontSize: 12, marginBottom: 5 }}>{term.replace(/_/g, ' ')}</div>
                  <div style={{ fontSize: 11, color: '#b0bec5', lineHeight: 1.6 }}>{def}</div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
