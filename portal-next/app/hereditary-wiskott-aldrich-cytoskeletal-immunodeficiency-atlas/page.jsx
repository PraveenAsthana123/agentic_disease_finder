'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-wiskott-aldrich-cytoskeletal-immunodeficiency-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'WAS':       '#b71c1c',  // deep red      -- classic WAS triad; small platelets PATHOGNOMONIC; EBV lymphoma; XLN-GOF
  'WIPF1':     '#c62828',  // red           -- WAS phenocopy; WASP absent + WAS gene normal; AR (females affected)
  'ARPC1B':    '#e65100',  // burnt orange  -- Arp2/3; eosinophilia + IBD + vasculitis; delta-granule defect
  'CORONIN1A': '#1565c0',  // deep blue     -- T-B+NK+ SCID; thymic shadow enlarged PATHOGNOMONIC; Swiss-type
  'RAC2':      '#4a148c',  // deep purple   -- GOF-D57N neonatal SCID; cord delay PATHOGNOMONIC; LOF neutrophil
  'RHOH':      '#1b5e20',  // deep green    -- HPV warts PATHOGNOMONIC; Kaposi; EBV; ultra-rare
  'RLTPR':     '#0d47a1',  // navy          -- CARMIL2; CMC + colitis + EBV; CD28 assay; Treg defect
  'STK4':      '#4e342e',  // dark brown    -- STK4/MST1; naive T-lymphopenia + cardiac ASD/VSD PATHOGNOMONIC
};

const GENE_INFO = {
  'WAS':       { full: 'WAS-502aa / XLR / WASp / Small-Platelets-MPV-LT7-PATHOGNOMONIC / Eczema / PID / EBV-Lymphoma-13-22pct / XLN-GOF-Neutropenia-AML-30pct / HSCT-Early',    locus: 'Xp11.22',   size: '502 aa / 53 kDa', inh: 'X-linked recessive' },
  'WIPF1':     { full: 'WIPF1-503aa / AR / WIP-WASp-Stabiliser / WASP-ABSENT-WAS-GENE-NORMAL-TRAP / WAS-Phenocopy-Identical-Triad / Females-Affected / HSCT-Curative',            locus: '2q31.1',    size: '503 aa / 54 kDa', inh: 'Autosomal recessive' },
  'ARPC1B':    { full: 'ARPC1B-323aa / AR / Arp2-3-Complex / Eosinophilia-GT1500-PATHOGNOMONIC / Early-IBD-Colitis / Vasculitis / Delta-Granule-Defect / JAKi-Ruxolitinib',        locus: '7q22.1',    size: '323 aa / 47 kDa', inh: 'Autosomal recessive' },
  'CORONIN1A': { full: 'CORONIN1A-461aa / AR / T-B+NK+-SCID / Thymic-Shadow-Enlarged-PATHOGNOMONIC / Swiss-Type / Thymic-Egress-Block / Autophagy-Defect / p.R18C-Swiss-Founder', locus: '16p11.2',   size: '461 aa / 51 kDa', inh: 'Autosomal recessive' },
  'RAC2':      { full: 'RAC2-192aa / AD-GOF+AR-LOF / D57N-Constitutively-Active / Cord-Delay-PATHOGNOMONIC / Neonatal-Sepsis / LAD-SCID-Overlap / Leukocytosis / HSCT-Birth',     locus: '22q13.1',   size: '192 aa / 21 kDa', inh: 'AD GOF / AR LOF' },
  'RHOH':      { full: 'RHOH-191aa / AR / Ultra-Rare-LT30-Families / HPV-Warts-Hundreds-PATHOGNOMONIC / Kaposi-Sarcoma-HHV8 / EBV-LPD / Molluscum / TCR-Proximal-Signalling',     locus: '4p13',      size: '191 aa / 21 kDa', inh: 'Autosomal recessive' },
  'RLTPR':     { full: 'CARMIL2-1113aa / AR / CD28-Actin-Scaffold / CMC-Oral-Oesophageal / Colitis-VERY-EARLY / EBV-LPD / Treg-Deficient / CD28-Assay-IL2-Diagnostic / HSCT',     locus: '16q22.1',   size: '1113 aa / 126 kDa', inh: 'Autosomal recessive' },
  'STK4':      { full: 'STK4-487aa / AR / MST1-Hippo-Kinase / Naive-T-Lymphopenia / Cardiac-ASD-VSD-40-60pct-PATHOGNOMONIC / Echo-MANDATORY / NK-Reduced-CMV / HSCT-Curative',    locus: '20q11.2',   size: '487 aa / 56 kDa', inh: 'Autosomal recessive' },
};

const PHENOTYPE_ROWS = [
  { gene: 'WAS',       group: 'WASp-Family',   thrombo: '✓✓✓', eczema: '✓✓✓', pid: '✓✓✓', ebv: '✓✓',  ibd: '−',   vascul: '±', cardiac: '−',  hpv: '−',  hsct: '✓✓✓', unique: 'SMALL PLATELETS (MPV < 7 fL) PATHOGNOMONIC; X-linked males only; WASP absent on flow; EBV lymphoma 13-22%; XLN-GOF isolated neutropenia + AML; HSCT before EBV' },
  { gene: 'WIPF1',     group: 'WASp-Family',   thrombo: '✓✓✓', eczema: '✓✓✓', pid: '✓✓✓', ebv: '✓✓',  ibd: '−',   vascul: '±', cardiac: '−',  hpv: '−',  hsct: '✓✓✓', unique: 'WASP ABSENT + WAS GENE NORMAL = diagnostic trap; AR (females affected unlike WAS X-linked); clinical triad identical to WAS; sequence WIPF1 immediately if WASP absent and WAS gene normal' },
  { gene: 'ARPC1B',   group: 'Arp2/3',        thrombo: '✓✓',  eczema: '✓✓',  pid: '✓✓✓', ebv: '±',   ibd: '✓✓✓',vascul: '✓✓✓',cardiac: '−', hpv: '−',  hsct: '✓✓',  unique: 'EOSINOPHILIA > 1500/µL PATHOGNOMONIC triple: PID + IBD + vasculitis; platelet delta-granule defect (storage pool); WASP normal; JAKi (ruxolitinib) controls inflammation pre-HSCT' },
  { gene: 'CORONIN1A',group: 'Thymic-Egress',  thrombo: '−',   eczema: '−',   pid: '✓✓✓', ebv: '✓✓',  ibd: '−',   vascul: '−', cardiac: '−',  hpv: '−',  hsct: '✓✓✓', unique: 'ENLARGED THYMIC SHADOW on CXR in T-cell-lymphopenic infant = PATHOGNOMONIC; T-B+NK+ SCID; most SCID has absent thymic shadow; Swiss-type; p.R18C founder; autophagy defect' },
  { gene: 'RAC2',     group: 'Rho-GTPase',    thrombo: '±',   eczema: '−',   pid: '✓✓✓', ebv: '±',   ibd: '−',   vascul: '−', cardiac: '−',  hpv: '−',  hsct: '✓✓✓', unique: 'DELAYED CORD SEPARATION (> day 14) + NEONATAL SEPSIS + LEUKOCYTOSIS = RAC2-D57N GOF PATHOGNOMONIC; mimics LAD-I but CD18 normal; T-cell lymphopenic; HSCT urgently from birth' },
  { gene: 'RHOH',     group: 'Rho-GTPase',    thrombo: '−',   eczema: '−',   pid: '✓✓',  ebv: '✓✓✓', ibd: '−',   vascul: '−', cardiac: '−',  hpv: '✓✓✓',hsct: '✓✓',  unique: 'SEVERE RECALCITRANT HPV WARTS (hundreds, plantar/periungual, childhood onset) PATHOGNOMONIC; Kaposi sarcoma (HHV8); EBV LPD; Molluscum; ultra-rare (< 30 families); TCR signalling defect' },
  { gene: 'RLTPR',    group: 'CD28-Actin',    thrombo: '−',   eczema: '±',   pid: '✓✓',  ebv: '✓✓',  ibd: '✓✓✓',vascul: '±', cardiac: '−',  hpv: '−',  hsct: '✓✓',  unique: 'CD28 STIMULATION ASSAY: isolated CD28 ligation → reduced IL-2 = CARMIL2 PATHOGNOMONIC; CMC + colitis + EBV LPD triple; TREG DEFICIENCY; CD28-actin mechanism (NOT IL-17 pathway); Abatacept NOT effective (unlike LRBA)' },
  { gene: 'STK4',     group: 'Hippo-Kinase',  thrombo: '−',   eczema: '−',   pid: '✓✓✓', ebv: '±',   ibd: '−',   vascul: '−', cardiac: '✓✓✓',hpv: '−',  hsct: '✓✓',  unique: 'CONGENITAL HEART DISEASE (ASD/VSD) + NAIVE T-LYMPHOPENIA + PULMONARY INFECTIONS = STK4 PATHOGNOMONIC TRIAD; cardiac defect in 40-60%; echocardiogram MANDATORY; HSCT corrects immune defect but NOT cardiac' },
];

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info, cohort }) {
  const groupLabel = ['WAS','WIPF1'].includes(gene) ? 'WASp Family'
                   : gene === 'ARPC1B'    ? 'Arp2/3 Complex'
                   : gene === 'CORONIN1A' ? 'Thymic Egress Defect'
                   : gene === 'RAC2'      ? 'Rho GTPase (GOF/LOF)'
                   : gene === 'RHOH'      ? 'Rho GTPase (TCR Signalling)'
                   : gene === 'RLTPR'     ? 'CD28 Actin Scaffold'
                   : 'Hippo Pathway Kinase';
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          <span style={{ fontWeight: 700, fontSize: 18, color }}>{gene}</span>
          <span style={{ marginLeft: 10, fontSize: 12, background: color, color: '#fff', padding: '2px 8px', borderRadius: 4 }}>{groupLabel}</span>
        </div>
        <div style={{ fontSize: 12, color: '#888', textAlign: 'right' }}>
          <div>{info.size} · {info.locus}</div>
          <div>{info.inh}</div>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#bbb', marginTop: 6 }}>{info.full}</div>
      {cohort && (
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Badge text={`Thrombocytopenia ${cohort.thrombocytopenia_pct}%`}  color={color} />
          <Badge text={`Eczema ${cohort.eczema_pct}%`}                      color='#880e4f' />
          <Badge text={`PID ${cohort.pid_pct}%`}                            color='#795548' />
          <Badge text={`EBV LPD ${cohort.ebv_lymphoma_pct}%`}               color='#4a148c' />
          <Badge text={`Autoimmune/IBD ${cohort.autoimmune_pct}%`}           color='#e65100' />
          <Badge text={`HSCT ${cohort.hsct_pct}%`}                          color='#1b5e20' />
          <Badge text={`Small PLT MPV ${cohort.small_plt_mpv_pct}%`}         color='#b71c1c' />
          <Badge text={`n=${cohort.n}`}                                      color='#555' />
          <Badge text={`dx-lag ${cohort.mean_dx_lag_years}y`}               color='#555' />
        </div>
      )}
      {cohort && (
        <div style={{ marginTop: 8, fontSize: 12, background: 'rgba(255,255,255,0.05)', padding: 8, borderRadius: 4 }}>
          <strong style={{ color: '#ffd54f' }}>Pathognomonic / Key Finding:</strong>{' '}
          <span style={{ color: '#ccc' }}>{cohort.pathognomonic}</span>
        </div>
      )}
    </div>
  );
}

export default function HWACytoskeletalAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefs(df);
    }).catch(e => setError(String(e)));
  }, []);

  if (error) return <div style={{ color: 'red', padding: 32 }}>Error: {error}</div>;
  if (!overview) return <div style={{ color: '#ccc', padding: 32 }}>Loading Hereditary-Wiskott-Aldrich-Cytoskeletal-Immunodeficiency-Atlas…</div>;

  const cohortMap = {};
  (breakdown?.gene_cohorts || []).forEach(c => { cohortMap[c.gene] = c; });

  return (
    <div style={{ background: '#121212', minHeight: '100vh', color: '#e0e0e0', fontFamily: 'monospace', padding: 24 }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ color: '#ef9a9a', fontSize: 22, marginBottom: 4 }}>
          🧬 Hereditary-Wiskott-Aldrich-Cytoskeletal-Immunodeficiency-Atlas
        </h1>
        <div style={{ color: '#90caf9', fontSize: 13 }}>
          Actin/Cytoskeletal Defects Affecting Immune Cells / WASp–Arp2/3–Rho GTPase–CD28–Hippo &nbsp;·&nbsp;
          8 Genes &nbsp;·&nbsp; {overview.total_patients} patients &nbsp;·&nbsp; seeds {overview.seeds}
        </div>
        <div style={{ fontSize: 12, color: '#888', marginTop: 4 }}>
          IUIS PID 2022 Categories 1, 2, 6 &nbsp;·&nbsp;
          WAS-WIPF1-ARPC1B-CORONIN1A-RAC2-RHOH-CARMIL2-STK4
        </div>
      </div>

      {/* Alert box */}
      <div style={{ background: '#3e2723', border: '1px solid #b71c1c', borderRadius: 6, padding: 12, marginBottom: 16, fontSize: 12 }}>
        <strong style={{ color: '#ef9a9a' }}>⚠️ BCG ABSOLUTELY CONTRAINDICATED</strong>
        <span style={{ color: '#ffccbc', marginLeft: 8 }}>
          in ALL 8 cytoskeletal immunodeficiency genes. SMALL PLATELETS (MPV &lt; 7 fL) — first test in
          suspected WAS/WIPF1. WASP absent + WAS gene normal → WIPF1. Enlarged thymic shadow + T-lymphopenia → CORONIN1A.
          Cord delay + neonatal sepsis → RAC2-D57N. HPV warts → RHOH. Cardiac ASD/VSD + naive T-lymphopenia → STK4.
        </span>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 4, border: 'none', cursor: 'pointer',
            background: tab === t ? '#b71c1c' : '#2a2a2a', color: tab === t ? '#fff' : '#aaa', fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && (
        <div>
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>Aggregate Statistics — {overview.total_patients} patients across 8 genes</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 10 }}>
              {[
                ['Thrombocytopenia', overview.aggregate_stats?.thrombocytopenia_pct, '#b71c1c'],
                ['Eczema', overview.aggregate_stats?.eczema_pct, '#880e4f'],
                ['PID (recurrent infections)', overview.aggregate_stats?.pid_pct, '#795548'],
                ['EBV Lymphoproliferation', overview.aggregate_stats?.ebv_lymphoma_pct, '#4a148c'],
                ['Autoimmune / IBD', overview.aggregate_stats?.autoimmune_pct, '#e65100'],
                ['HSCT Required', overview.aggregate_stats?.hsct_pct, '#1b5e20'],
                ['Small Plt MPV < 7 fL', overview.aggregate_stats?.small_plt_mpv_pct, '#b71c1c'],
              ].map(([label, val, color]) => (
                <div key={label} style={{ background: '#2a2a2a', borderRadius: 6, padding: 10, textAlign: 'center' }}>
                  <div style={{ fontSize: 22, fontWeight: 700, color }}>{val ?? '—'}%</div>
                  <div style={{ fontSize: 11, color: '#aaa', marginTop: 2 }}>{label}</div>
                </div>
              ))}
            </div>
          </div>
          <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>8 Gene Cohorts (40 patients each)</h3>
          {Object.entries(GENE_INFO).map(([gene, info]) => (
            <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={info} cohort={cohortMap[gene]} />
          ))}
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && (
        <div>
          <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>Cytoskeletal Immunodeficiency Phenotype Matrix — 8 Genes</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1e1e1e' }}>
                  {['Gene','Group','Thrombo','Eczema','PID','EBV LPD','IBD','Vasculitis','Cardiac','HPV Warts','HSCT','Key Distinguisher'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', color: '#ef9a9a', borderBottom: '2px solid #333', whiteSpace: 'nowrap', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#181818' : '#202020' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[row.gene], whiteSpace: 'nowrap' }}>{row.gene}</td>
                    <td style={{ padding: '7px 10px', color: '#90caf9', whiteSpace: 'nowrap', fontSize: 11 }}>{row.group}</td>
                    {[row.thrombo, row.eczema, row.pid, row.ebv, row.ibd, row.vascul, row.cardiac, row.hpv, row.hsct].map((v, j) => (
                      <td key={j} style={{ padding: '7px 10px', textAlign: 'center', color: v === '−' ? '#555' : v?.includes('✓✓✓') ? '#ef9a9a' : v?.includes('✓✓') ? '#ffcc02' : '#aaa' }}>{v}</td>
                    ))}
                    <td style={{ padding: '7px 10px', color: '#ccc', fontSize: 11 }}>{row.unique}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ marginTop: 24, background: '#1e1e1e', borderRadius: 8, padding: 16 }}>
            <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>Cascade Testing Protocol</h3>
            {(overview.cascade_testing_concepts || []).map((step, i) => (
              <div key={i} style={{ padding: '8px 0', borderBottom: i < overview.cascade_testing_concepts.length - 1 ? '1px solid #333' : 'none', fontSize: 13 }}>
                <span style={{ color: '#90caf9', fontWeight: 600 }}>{i + 1}. </span>
                <span style={{ color: '#ccc' }}>{step}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Pathway Map Tab */}
      {tab === 'Pathway Map' && (
        <div>
          <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>Actin/Cytoskeletal Immunodeficiency — Molecular Pathway Map</h3>
          <div style={{ background: '#1a1a1a', borderRadius: 10, padding: 20, fontSize: 12 }}>
            <div style={{ marginBottom: 16 }}>
              <div style={{ color: '#90caf9', fontWeight: 700, marginBottom: 8 }}>WASP FAMILY — Actin Nucleation via Arp2/3</div>
              <div style={{ paddingLeft: 16, borderLeft: '3px solid #b71c1c' }}>
                <div style={{ color: '#ccc', marginBottom: 6 }}>CDC42-GTP → WASP GBD domain relieves autoinhibition → VCA domain activates Arp2/3 → de novo actin branch → immune synapse / phagocytic cup / platelet spreading</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>↑ <span style={{ color: '#b71c1c', fontWeight: 600 }}>WAS LOF</span>: WASP absent → Arp2/3 not activated → immunological synapse fails → T/NK/platelet defects</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>↑ <span style={{ color: '#c62828', fontWeight: 600 }}>WIPF1 LOF</span>: WIP absent → WASP degraded despite normal WAS gene → functionally identical to WAS-null</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>↑ <span style={{ color: '#e65100', fontWeight: 600 }}>ARPC1B LOF</span>: Arp2/3 complex destabilised → actin branching impaired + innate inflammasome dysregulated → PID + eosinophilia + IBD + vasculitis</div>
              </div>
            </div>
            <div style={{ marginBottom: 16 }}>
              <div style={{ color: '#90caf9', fontWeight: 700, marginBottom: 8 }}>THYMIC EGRESS — Coronin-1A F-Actin Dynamics</div>
              <div style={{ paddingLeft: 16, borderLeft: '3px solid #1565c0' }}>
                <div style={{ color: '#ccc', marginBottom: 6 }}>Mature thymocyte → Coronin-1A regulates cortical F-actin dynamics → vesicle trafficking → thymocyte exits thymus via blood vessels</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>↑ <span style={{ color: '#1565c0', fontWeight: 600 }}>CORONIN1A LOF</span>: F-actin disorganised → thymocytes trapped in thymus → enlarged thymus on CXR → T-B+NK+ SCID in periphery</div>
              </div>
            </div>
            <div style={{ marginBottom: 16 }}>
              <div style={{ color: '#90caf9', fontWeight: 700, marginBottom: 8 }}>RHO GTPASE AXIS — Neutrophil + Lymphocyte Defects</div>
              <div style={{ paddingLeft: 16, borderLeft: '3px solid #4a148c' }}>
                <div style={{ color: '#ccc', marginBottom: 6 }}>GEF activates RAC2 (GTP-bound) → lamellipodia / NADPH oxidase assembly / actin polymerisation in neutrophils + lymphocytes</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>↑ <span style={{ color: '#4a148c', fontWeight: 600 }}>RAC2-D57N GOF</span>: constitutively GTP-bound → paradoxical dysfunction → neutrophil/T-cell combined defect → neonatal SCID + cord delay</div>
                <div style={{ color: '#ccc', marginBottom: 6, marginTop: 6 }}>RhoH → ZAP70 recruitment to CD3ζ ITAM → LAT phosphorylation → TCR proximal signalling</div>
                <div style={{ color: '#aaa', fontSize: 11 }}>↑ <span style={{ color: '#1b5e20', fontWeight: 600 }}>RHOH LOF</span>: TCR signal attenuated → T-cell lymphopenia → HPV warts + Kaposi + EBV LPD (viral susceptibility dominant)</div>
              </div>
            </div>
            <div>
              <div style={{ color: '#90caf9', fontWeight: 700, marginBottom: 8 }}>CD28 ACTIN SCAFFOLD (CARMIL2) + HIPPO KINASE (STK4)</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div style={{ paddingLeft: 12, borderLeft: '3px solid #0d47a1' }}>
                  <div style={{ color: '#ccc', marginBottom: 6 }}>CD28 engagement → CARMIL2/RLTPR recruits WAVE2 via IQGAP1 → actin remodelling at CD28 microcluster → full T-cell activation (IL-2, Treg generation)</div>
                  <div style={{ color: '#aaa', fontSize: 11 }}>↑ <span style={{ color: '#0d47a1', fontWeight: 600 }}>CARMIL2 LOF</span>: CD28 co-stimulation impaired → reduced Treg → colitis + CMC + EBV LPD; CD28 assay IL-2 diagnostic</div>
                </div>
                <div style={{ paddingLeft: 12, borderLeft: '3px solid #4e342e' }}>
                  <div style={{ color: '#ccc', marginBottom: 6 }}>STK4 (MST1) → phospho-FOXO3a → nuclear Bcl-2 upregulation → naive T-cell survival (anti-apoptotic); cardiac development (Hippo-YAP axis)</div>
                  <div style={{ color: '#aaa', fontSize: 11 }}>↑ <span style={{ color: '#4e342e', fontWeight: 600 }}>STK4 LOF</span>: naive T-cell apoptosis → CD45RA+ lymphopenia; cardiac defect ASD/VSD 40-60% (structural, not corrected by HSCT)</div>
                </div>
              </div>
            </div>
          </div>

          {breakdown && Object.entries(breakdown.diagnostic_groups || {}).map(([key, grp]) => (
            <div key={key} style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginTop: 12 }}>
              <div style={{ color: '#ffd54f', fontWeight: 700, marginBottom: 6 }}>
                {grp.genes?.join(' / ')} — {grp.mechanism?.slice(0, 80)}…
              </div>
              <div style={{ color: '#aaa', fontSize: 12 }}><strong style={{ color: '#90caf9' }}>Shared:</strong> {grp.shared_feature}</div>
              <div style={{ color: '#aaa', fontSize: 12, marginTop: 4 }}><strong style={{ color: '#90caf9' }}>DDx:</strong> {grp.key_ddx}</div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && defs && (
        <div>
          <h3 style={{ color: '#ef9a9a', marginBottom: 12 }}>Definitions & Key Distinguishers</h3>
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <h4 style={{ color: '#ffd54f', marginBottom: 10 }}>Key Distinguishers</h4>
            {Object.entries(defs.key_distinguishers || {}).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid #333' }}>
                <div style={{ color: '#90caf9', fontWeight: 600, fontSize: 12 }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ color: '#ccc', fontSize: 12, marginTop: 3 }}>{v}</div>
              </div>
            ))}
          </div>
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <h4 style={{ color: '#ffd54f', marginBottom: 10 }}>Glossary</h4>
            {(defs.definitions || []).map((item, i) => (
              <div key={i} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: i < defs.definitions.length - 1 ? '1px solid #333' : 'none' }}>
                <div style={{ color: '#90caf9', fontWeight: 600, fontSize: 12 }}>{item.term}</div>
                <div style={{ color: '#ccc', fontSize: 12, marginTop: 3 }}>{item.definition}</div>
              </div>
            ))}
          </div>
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 16 }}>
            <h4 style={{ color: '#ffd54f', marginBottom: 10 }}>Diagnostic Pathway</h4>
            {(defs.diagnostic_pathway || []).map((step, i) => (
              <div key={i} style={{ padding: '7px 0', borderBottom: i < defs.diagnostic_pathway.length - 1 ? '1px solid #333' : 'none', fontSize: 13 }}>
                <span style={{ color: '#b71c1c', fontWeight: 700 }}>{step.split(':')[0]}: </span>
                <span style={{ color: '#ccc' }}>{step.split(':').slice(1).join(':')}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
