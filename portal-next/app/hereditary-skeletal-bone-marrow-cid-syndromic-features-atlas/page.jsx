'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-skeletal-bone-marrow-cid-syndromic-features-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'SBDS':     '#b71c1c',  // deep red       — SDS type 1, ribosome biogenesis, EPI+neutropenia+metaphyseal
  'RMRP':     '#e65100',  // deep orange    — cartilage-hair hypoplasia, short stature, sparse hair
  'AK2':      '#1565c0',  // deep blue      — reticular dysgenesis, total leukocyte aplasia, SNHL
  'SMARCAL1': '#4a148c',  // deep purple    — SIOD, FSGS+T-lymphopenia+spondyloepiphyseal
  'CORO1A':   '#00695c',  // dark teal      — Coronin-1A deficiency, thymic egress, EBV susceptibility
  'ARPC1B':   '#f9a825',  // amber          — ARPC1B deficiency, eosinophilia PATHOGNOMONIC, vasculitis
  'WIPF1':    '#2e7d32',  // dark green     — WIP deficiency, WAS phenocopy, microplatelets
  'DNAJC21':  '#6a1b9a',  // purple         — SDS type 2, DNAJC21, exocrine pancreatic+neutropenia
};

const GENE_INFO = {
  'SBDS':     { full: 'SBDS / Shwachman-Bodian-Diamond Syndrome Protein / 255aa', locus: '7q11.21', size: '255 aa / 29 kDa (ribosome biogenesis factor; 60S maturation; eIF6 release; pseudogene SBDSP caution in sequencing; SDS type 1)', inh: 'AR' },
  'RMRP':     { full: 'RMRP / RNA Component of Mitochondrial RNA-Processing Endoribonuclease / non-coding RNA', locus: '9p13.3', size: 'Non-coding RNA (265 nt; RNase MRP RNA subunit; pre-rRNA processing + mitochondrial DNA replication + cell cycle regulation; CHH)', inh: 'AR' },
  'AK2':      { full: 'AK2 / Adenylate Kinase 2 / 239aa', locus: '1p35.1', size: '239 aa / 26 kDa (mitochondrial intermembrane space; 2ADP↔ATP+AMP; hematopoietic progenitor energy; reticular dysgenesis)', inh: 'AR' },
  'SMARCAL1': { full: 'SMARCAL1 / SWI/SNF-Related Chromatin Regulator Subfamily A-like 1 / 954aa', locus: '2q34', size: '954 aa / 106 kDa (HARP helicase; stalled replication fork rescue; fork reversal; spondyloepiphyseal dysplasia+FSGS+T-lymphopenia; SIOD)', inh: 'AR' },
  'CORO1A':   { full: 'CORO1A / Coronin 1A / 461aa', locus: '16p11.2', size: '461 aa / 51 kDa (WD40-repeat actin-regulatory protein; thymic T-cell egress; Ca2+-NFAT signaling; T lymphopenia+enlarged thymus)', inh: 'AR' },
  'ARPC1B':   { full: 'ARPC1B / Actin-Related Protein 2/3 Complex Subunit B / 370aa', locus: '7q22.1', size: '370 aa / 40 kDa (Arp2/3 complex γ-subunit; branched actin nucleation; immune synapse; eosinophilia PATHOGNOMONIC+vasculitis+IBD)', inh: 'AR' },
  'WIPF1':    { full: 'WIPF1 / WASp-Interacting Protein Family Member 1 / 503aa', locus: '2q31.1', size: '503 aa / 54 kDa (WIP; stabilizes WASp; recruits WASp to TCR; secondary WASp loss; WAS phenocopy; microplatelets PATHOGNOMONIC)', inh: 'AR' },
  'DNAJC21':  { full: 'DNAJC21 / DnaJ Heat Shock Protein Family Member C21 / 647aa', locus: '5p13.2', size: '647 aa / 73 kDa (Hsp40 co-chaperone; 60S ribosome biogenesis; cooperates with EFL1+SBDS; SDS type 2; riboflavin NOT responsive)', inh: 'AR' },
};

const PATHOGNOMONIC = {
  'SBDS':     'Fecal elastase-1 < 100 μg/g (EPI) + metaphyseal dysplasia + neutropenia — SDS TRIAD; i(7q) on marrow cytogenetics for MDS transition',
  'RMRP':     'Short limb dwarfism + sparse fine hair + T lymphopenia — CHH TRIAD; VZV disproportionately severe (more than any other T cell deficiency)',
  'AK2':      'TOTAL leukocyte aplasia (zero lymphocytes + zero myeloid + zero monocytes) + sensorineural deafness — MOST SEVERE SCID; SNHL does not improve post-HSCT',
  'SMARCAL1': 'Spondyloepiphyseal dysplasia (flat vertebrae on X-ray) + steroid-RESISTANT FSGS + T lymphopenia — SIOD TRIAD; Moyamoya-like cerebrovascular disease 40%',
  'CORO1A':   'ENLARGED thymus on imaging + profound T lymphopenia — PARADOX; T cells trapped in thymus cannot exit; EBV susceptibility >80%',
  'ARPC1B':   'EOSINOPHILIA > 1×10⁹/L + immunodeficiency + vasculitis — PATHOGNOMONIC for ARPC1B (absent in WAS/WIPF1); WASp protein PRESENT',
  'WIPF1':    'WASp FLOW ABSENT + WAS GENE NORMAL + microplatelets (MPV < 6 fL) — WASp secondarily absent due to WIP loss; sequence WIPF1 when this pattern',
  'DNAJC21':  'Clinically identical to SDS type 1 (EPI + neutropenia + metaphyseal dysplasia); SBDS biallelic mutations EXCLUDED; riboflavin-unresponsive',
};

function GeneChip({ gene, active, onClick }) {
  return (
    <button
      onClick={() => onClick(gene)}
      style={{
        background: active ? GENE_COLORS[gene] : '#1e293b',
        color: '#fff',
        border: `2px solid ${GENE_COLORS[gene]}`,
        borderRadius: 8,
        padding: '6px 14px',
        margin: 4,
        cursor: 'pointer',
        fontWeight: active ? 700 : 400,
        fontSize: 13,
      }}
    >
      {gene}
    </button>
  );
}

export default function HereditarySkeletalBoneMarrowCIDSyndromicFeaturesAtlas() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeGene, setActiveGene] = useState('SBDS');

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

  const genes = ['SBDS', 'RMRP', 'AK2', 'SMARCAL1', 'CORO1A', 'ARPC1B', 'WIPF1', 'DNAJC21'];

  if (loading) return <div style={{ padding: 40, color: '#94a3b8', fontFamily: 'monospace' }}>Loading atlas data…</div>;
  if (error) return <div style={{ padding: 40, color: '#f87171', fontFamily: 'monospace' }}>Error: {error}</div>;

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)', borderBottom: '1px solid #334155', padding: '24px 32px' }}>
        <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 2, marginBottom: 8 }}>
          Hereditary Immunodeficiency Atlas · IUIS 2022 Category II
        </div>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#f8fafc', margin: '0 0 6px' }}>
          🧬 Hereditary-Skeletal-Bone-Marrow-CID-Syndromic-Features-Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8' }}>
          Complete 8-Gene Reference — SBDS · RMRP · AK2 · SMARCAL1 · CORO1A · ARPC1B · WIPF1 · DNAJC21
          &nbsp;|&nbsp; 320 patients (8 × 40) · seeds 4046–4053
        </div>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          Shwachman-Diamond (SDS1+2) · Cartilage-Hair Hypoplasia · Reticular Dysgenesis · SIOD · Coronin-1A · ARPC1B · WIP Deficiency
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid #334155', padding: '0 32px' }}>
        {TABS.map((t, i) => (
          <button key={i} onClick={() => setTab(i)} style={{
            background: 'none', border: 'none', color: tab === i ? '#38bdf8' : '#64748b',
            padding: '12px 20px', cursor: 'pointer', fontWeight: tab === i ? 700 : 400,
            borderBottom: tab === i ? '2px solid #38bdf8' : '2px solid transparent',
            fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 32px' }}>

        {/* Tab 0: Overview */}
        {tab === 0 && overview && (
          <div>
            {/* Gene chips */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>SELECT GENE</div>
              <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                {genes.map(g => <GeneChip key={g} gene={g} active={activeGene === g} onClick={setActiveGene} />)}
              </div>
            </div>

            {/* Active gene card */}
            {activeGene && overview.cohorts && overview.cohorts[activeGene] && (
              <div style={{ background: '#1e293b', borderRadius: 12, border: `2px solid ${GENE_COLORS[activeGene]}`, padding: 20, marginBottom: 20 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
                  <div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: GENE_COLORS[activeGene] }}>{activeGene}</div>
                    <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>{GENE_INFO[activeGene]?.full}</div>
                    <div style={{ fontSize: 11, color: '#64748b' }}>Locus: {GENE_INFO[activeGene]?.locus} · {GENE_INFO[activeGene]?.inh} · {GENE_INFO[activeGene]?.size}</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 8, padding: '8px 16px', fontSize: 13, color: '#fbbf24', border: '1px solid #f59e0b' }}>
                    ⚠ {PATHOGNOMONIC[activeGene]}
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginTop: 16 }}>
                  {[
                    ['Patients', overview.cohorts[activeGene].n],
                    ['Neutropenia %', `${overview.cohorts[activeGene].neutropenia_pct}%`],
                    ['EPI %', `${overview.cohorts[activeGene].pancreatic_insuff_pct}%`],
                    ['Skeletal Dyspl. %', `${overview.cohorts[activeGene].skel_dysplasia_pct}%`],
                    ['HSCT Rate %', `${overview.cohorts[activeGene].hsct_pct}%`],
                    ['5yr Survival %', `${overview.cohorts[activeGene].survival_5yr_pct}%`],
                    ['Leukemia Risk %', `${overview.cohorts[activeGene].leukemia_risk_pct}%`],
                    ['Short Stature %', `${overview.cohorts[activeGene].short_stature_pct}%`],
                  ].map(([label, val]) => (
                    <div key={label} style={{ background: '#0f172a', borderRadius: 8, padding: '10px 14px', textAlign: 'center' }}>
                      <div style={{ fontSize: 18, fontWeight: 700, color: GENE_COLORS[activeGene] }}>{val}</div>
                      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Key facts */}
            {overview.key_facts && (
              <div style={{ background: '#1e293b', borderRadius: 12, padding: 20, marginBottom: 20 }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 12 }}>KEY CLINICAL FACTS</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10 }}>
                  {Object.entries(overview.key_facts).map(([k, v]) => (
                    <div key={k} style={{ background: '#0f172a', borderRadius: 8, padding: '10px 14px' }}>
                      <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>{k.replace(/_/g, ' ')}</div>
                      <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.5 }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Pathway overview */}
            {overview.pathway_overview && (
              <div style={{ background: '#1e293b', borderRadius: 12, padding: 20, border: '1px solid #334155' }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#a78bfa', marginBottom: 10 }}>PATHWAY OVERVIEW</div>
                <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7 }}>{overview.pathway_overview}</div>
              </div>
            )}
          </div>
        )}

        {/* Tab 1: Gene Table */}
        {tab === 1 && breakdown && (
          <div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene', 'Disease', 'Locus/Size/Inh', 'n', 'Neutropenia %', 'EPI %', 'Skeletal %', 'HSCT %', '5yr Surv %', 'Leukemia %', 'Short Stature %'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #334155' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {genes.map(g => {
                    const c = breakdown.gene_summary?.[g] || {};
                    const info = GENE_INFO[g] || {};
                    const diseaseMap = {
                      'SBDS': 'Shwachman-Diamond Syndrome Type 1',
                      'RMRP': 'Cartilage-Hair Hypoplasia (CHH)',
                      'AK2': 'Reticular Dysgenesis',
                      'SMARCAL1': 'Schimke Immuno-Osseous Dysplasia (SIOD)',
                      'CORO1A': 'Coronin-1A Deficiency',
                      'ARPC1B': 'ARPC1B Deficiency (Arp2/3 Complex)',
                      'WIPF1': 'WIP/WIPF1 Deficiency (WAS Phenocopy)',
                      'DNAJC21': 'Shwachman-Diamond Syndrome Type 2',
                    };
                    return (
                      <tr key={g} style={{ borderBottom: '1px solid #1e293b', cursor: 'pointer' }}
                        onClick={() => { setActiveGene(g); setTab(0); }}>
                        <td style={{ padding: '10px 12px', fontWeight: 700, color: GENE_COLORS[g] }}>{g}</td>
                        <td style={{ padding: '10px 12px', color: '#cbd5e1', fontSize: 11 }}>{diseaseMap[g]}</td>
                        <td style={{ padding: '10px 12px', color: '#94a3b8', fontSize: 10 }}>{info.locus} · {info.inh}</td>
                        <td style={{ padding: '10px 12px', color: '#94a3b8' }}>{c.n}</td>
                        <td style={{ padding: '10px 12px', color: c.neutropenia_pct > 70 ? '#f87171' : '#94a3b8' }}>{c.neutropenia_pct}%</td>
                        <td style={{ padding: '10px 12px', color: c.pancreatic_insuff_pct > 50 ? '#fb923c' : '#94a3b8' }}>{c.pancreatic_insuff_pct}%</td>
                        <td style={{ padding: '10px 12px', color: c.skel_dysplasia_pct > 80 ? '#facc15' : '#94a3b8' }}>{c.skel_dysplasia_pct}%</td>
                        <td style={{ padding: '10px 12px', color: c.hsct_pct > 60 ? '#38bdf8' : '#94a3b8' }}>{c.hsct_pct}%</td>
                        <td style={{ padding: '10px 12px', color: c.survival_5yr_pct < 70 ? '#f87171' : '#4ade80' }}>{c.survival_5yr_pct}%</td>
                        <td style={{ padding: '10px 12px', color: c.leukemia_risk_pct > 15 ? '#f87171' : '#94a3b8' }}>{c.leukemia_risk_pct}%</td>
                        <td style={{ padding: '10px 12px', color: c.short_stature_pct > 80 ? '#a78bfa' : '#94a3b8' }}>{c.short_stature_pct}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 2: Clinical Atlas */}
        {tab === 2 && breakdown && breakdown.diagnostic_groups && (
          <div>
            {Object.entries(breakdown.diagnostic_groups).map(([key, grp]) => (
              <div key={key} style={{ background: '#1e293b', borderRadius: 12, padding: 20, marginBottom: 16, border: '1px solid #334155' }}>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
                  {grp.genes.map(g => (
                    <span key={g} style={{ background: GENE_COLORS[g], color: '#fff', padding: '2px 10px', borderRadius: 12, fontSize: 12, fontWeight: 700 }}>{g}</span>
                  ))}
                </div>
                <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10, lineHeight: 1.6 }}>
                  <strong style={{ color: '#38bdf8' }}>Mechanism: </strong>{grp.mechanism}
                </div>
                <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10, lineHeight: 1.6 }}>
                  <strong style={{ color: '#4ade80' }}>Shared Features: </strong>{grp.shared_feature}
                </div>
                <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6, background: '#0f172a', padding: 12, borderRadius: 8 }}>
                  <strong style={{ color: '#fbbf24' }}>DDx Key: </strong>{grp.key_ddx}
                </div>
              </div>
            ))}

            {/* Key distinguishers */}
            {breakdown.key_distinguishers && (
              <div style={{ background: '#1e293b', borderRadius: 12, padding: 20, border: '1px solid #a78bfa' }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#a78bfa', marginBottom: 12 }}>KEY DISTINGUISHERS</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 10 }}>
                  {Object.entries(breakdown.key_distinguishers).map(([k, v]) => (
                    <div key={k} style={{ background: '#0f172a', borderRadius: 8, padding: '10px 14px' }}>
                      <div style={{ fontSize: 11, color: '#a78bfa', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>{k.replace(/_/g, ' ')}</div>
                      <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.5 }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Definitions */}
        {tab === 3 && definitions && (
          <div>
            {definitions.definitions?.map((def, i) => (
              <div key={i} style={{ background: '#1e293b', borderRadius: 12, padding: 20, marginBottom: 12, border: '1px solid #334155' }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#38bdf8', marginBottom: 8 }}>{def.term}</div>
                <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7 }}>{def.definition}</div>
              </div>
            ))}

            {definitions.treatment_matrix && (
              <div style={{ background: '#1e293b', borderRadius: 12, padding: 20, border: '1px solid #22c55e', marginTop: 16 }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#22c55e', marginBottom: 12 }}>TREATMENT MATRIX</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10 }}>
                  {Object.entries(definitions.treatment_matrix).map(([k, v]) => (
                    <div key={k} style={{ background: '#0f172a', borderRadius: 8, padding: '10px 14px' }}>
                      <div style={{ fontSize: 11, color: '#22c55e', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>{k.replace(/_/g, ' ')}</div>
                      <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.5 }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
