'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cd3-tcr-orai1-stim1-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'CD3D':  '#0d47a1',  // deep navy blue    -- CD3 delta; T-B+NK+ SCID; CD3 absent flow
  'CD3E':  '#1565c0',  // dark blue         -- CD3 epsilon; Morocco Cys68Tyr founder
  'CD3G':  '#1976d2',  // medium blue       -- CD3 gamma; MILDER partial T-lymphopenia
  'CD247': '#0277bd',  // light deep blue   -- CD3 zeta; ULTRA-RARE <30 families; Turkey Arg103Ter
  'LCK':   '#00695c',  // dark teal         -- Lck p56; LOF=lymphopenia GOF=lymphoproliferation OPPOSITE
  'LAT':   '#00796b',  // teal              -- Linker for Activation of T cells; surface CD3 PRESENT
  'ORAI1': '#6a1b9a',  // deep purple       -- CRAC channel pore; SOCE; ANHIDROSIS+MYOPATHY PATHOGNOMONIC
  'STIM1': '#7b1fa2',  // purple            -- ER Ca2+ sensor; TUBULAR AGGREGATE MYOPATHY+CARDIAC ECG MANDATORY
};

const GENE_INFO = {
  'CD3D':  { full: 'CD3D-171aa / AR / CD3-delta / TCR-CD3-complex-subunit / 11q23.3 / T-B+NK+-SCID / Surface-CD3-Absent-Flow-DIAGNOSTIC / BCG-Absolutely-CI / HSCT-Myeloablative-Curative',             locus: '11q23.3',  size: '171 aa / 19 kDa',  inh: 'Autosomal recessive' },
  'CD3E':  { full: 'CD3E-207aa / AR / CD3-epsilon / TCR-CD3-complex-subunit / 11q23.3 / T-B+NK+-SCID / Morocco-p.Cys68Tyr-Founder-Mutation / Surface-CD3-Absent-Flow / Identical-Phenotype-CD3D',        locus: '11q23.3',  size: '207 aa / 23 kDa',  inh: 'Autosomal recessive' },
  'CD3G':  { full: 'CD3G-182aa / AR / CD3-gamma / TCR-CD3-complex-subunit / 11q23.3 / MILDER-PARTIAL-T-lymphopenia-NOT-complete-SCID / Residual-T-10-30pct / CD3-REDUCED-not-absent / Later-Onset',      locus: '11q23.3',  size: '182 aa / 20 kDa',  inh: 'Autosomal recessive' },
  'CD247': { full: 'CD247-164aa / AR / CD3-zeta / TCR-CD3-complex-subunit / 1q24.2 / T-B+NK+-SCID / ULTRA-RARE-<30-families-worldwide / Turkey-p.Arg103Ter-Founder / Surface-CD3-Absent-Flow',           locus: '1q24.2',   size: '164 aa / 18 kDa',  inh: 'Autosomal recessive' },
  'LCK':   { full: 'LCK-509aa / AR-LOF + AD-GOF / p56-Lck-Src-family-kinase / CD4-CD8-coreceptor-signaling / 1p35.1 / LOF=T-lymphopenia / GOF=lymphoproliferation / OPPOSITE-PHENOTYPES-SAME-GENE',     locus: '1p35.1',   size: '509 aa / 56 kDa',  inh: 'AR (LOF) / AD (GOF)' },
  'LAT':   { full: 'LAT-262aa / AR / Linker-for-Activation-of-T-cells / TCR-adapter-scaffold / 16q24.1 / T-SCID-like / Surface-CD3-PRESENT-but-TCR-downstream-absent / pTyr132-LAT-assay-DIAGNOSTIC',    locus: '16q24.1',  size: '262 aa / 36 kDa',  inh: 'Autosomal recessive' },
  'ORAI1': { full: 'ORAI1-301aa / AR / CRAC-channel-pore-subunit / Store-Operated-Ca2+-Entry-SOCE / 12q24.31 / Combined-ID+ANHIDROSIS+MYOPATHY-PATHOGNOMONIC / Tuebingen-p.Arg91Trp-European-Founder',   locus: '12q24.31', size: '301 aa / 33 kDa',  inh: 'Autosomal recessive' },
  'STIM1': { full: 'STIM1-685aa / AR / ER-Ca2+-sensor-CRAC-activator / SOCE-pathway / 11p15.5 / Tubular-Aggregate-Myopathy-PATHOGNOMONIC / Cardiac-Arrhythmia-Annual-ECG-MANDATORY / Hypocalcemia',      locus: '11p15.5',  size: '685 aa / 77 kDa',  inh: 'Autosomal recessive' },
};

const PHENOTYPE_ROWS = [
  { gene: 'CD3D',  group: 'CD3 Complex (CD3D/E/G/Z)', tcell: '✓✓✓', cd3flow: '✓✓✓', partial: '✗',  lck: '✗', lat: '✗', soce: '✗', anhid: '✗', myop: '✗', hsct: '✓✓✓', unique: 'T-B+NK+ SCID: surface CD3 absent on ALL T cells by flow cytometry = DIAGNOSTIC; BCG absolutely CI; myeloablative HSCT curative; 11q23.3 locus same as CD3E/CD3G; lymphopenia severe from birth; PCP/CMV prophylaxis mandatory' },
  { gene: 'CD3E',  group: 'CD3 Complex (CD3D/E/G/Z)', tcell: '✓✓✓', cd3flow: '✓✓✓', partial: '✗',  lck: '✗', lat: '✗', soce: '✗', anhid: '✗', myop: '✗', hsct: '✓✓✓', unique: 'MOROCCAN FOUNDER p.Cys68Tyr: enriched in North African families; phenotype identical to CD3D; surface CD3 absent (flow); same 11q23.3 locus as CD3D; consider founder testing in Moroccan ancestry before full WES; CD3E most common CD3 complex defect overall' },
  { gene: 'CD3G',  group: 'CD3 Complex (CD3D/E/G/Z)', tcell: '✓✓',  cd3flow: '✓',   partial: '✓✓✓', lck: '✗', lat: '✗', soce: '✗', anhid: '✗', myop: '✗', hsct: '±',   unique: 'MILDER PARTIAL NOT COMPLETE SCID: residual T cells 10-30%; CD3 surface REDUCED not absent; later onset recurrent infections rather than neonatal SCID; may be misclassified as combined immunodeficiency; CD3G partial LOF due to hypomorphic variants; gene therapy candidate' },
  { gene: 'CD247', group: 'CD3 Complex (CD3D/E/G/Z)', tcell: '✓✓✓', cd3flow: '✓✓✓', partial: '✗',  lck: '✗', lat: '✗', soce: '✗', anhid: '✗', myop: '✗', hsct: '✓✓✓', unique: 'ULTRA-RARE FEWER THAN 30 FAMILIES WORLDWIDE: Turkish p.Arg103Ter founder; phenotype identical to CD3D SCID; surface CD3 absent (flow); 1q24.2 locus; zeta-chain essential for TCR surface expression; consider in consanguineous Turkish families; WES/WGS required for diagnosis' },
  { gene: 'LCK',   group: 'TCR Signaling (LCK/LAT)',  tcell: '✓✓',  cd3flow: '±',   partial: '±',  lck: '✓✓✓', lat: '✗', soce: '✗', anhid: '✗', myop: '✗', hsct: '±',   unique: 'LOF vs GOF OPPOSITE PHENOTYPES SAME GENE: AR-LOF = T-lymphopenia + hypogammaglobulinemia (immunodeficiency); AD-GOF = lymphoproliferation + autoimmunity (splenomegaly, cytopenias); CRITICAL to distinguish GOF from LOF before HSCT — GOF may respond to mTOR inhibition instead' },
  { gene: 'LAT',   group: 'TCR Signaling (LCK/LAT)',  tcell: '✓✓',  cd3flow: '✗',   partial: '±',  lck: '✗', lat: '✓✓✓', soce: '✗', anhid: '✗', myop: '✗', hsct: '✓',   unique: 'SURFACE CD3 PRESENT BUT TCR DOWNSTREAM ABSENT: unique among TCR defects — normal surface CD3/TCR counts by flow, but signaling absent; pTyr132-LAT phosphoflow assay DIAGNOSTIC (no downstream phosphorylation after TCR stimulation); easy to miss on standard lymphocyte subsets alone' },
  { gene: 'ORAI1', group: 'SOCE (ORAI1/STIM1)',       tcell: '✓✓',  cd3flow: '✗',   partial: '✗',  lck: '✗', lat: '✗', soce: '✓✓✓', anhid: '✓✓✓', myop: '✓✓', hsct: '✓', unique: 'ANHIDROSIS + MYOPATHY + COMBINED ID PATHOGNOMONIC TRIAD: Tübingen p.Arg91Trp European founder; anhidrosis due to absent sweat gland secretion (SOCE required for exocrine glands); CRAC current assay in T cells DIAGNOSTIC; muscle biopsy tubular aggregates; HSCT corrects ID but not myopathy/anhidrosis' },
  { gene: 'STIM1', group: 'SOCE (ORAI1/STIM1)',       tcell: '✓✓',  cd3flow: '✗',   partial: '✗',  lck: '✗', lat: '✗', soce: '✓✓✓', anhid: '±',  myop: '✓✓✓', hsct: '✓', unique: 'TUBULAR AGGREGATE MYOPATHY + CARDIAC ARRHYTHMIA: annual ECG MANDATORY — life-threatening conduction defects; STIM1 is ER Ca2+ sensor that activates ORAI1 CRAC channel; hypocalcemia in some; platelet dysfunction (SOCE needed for platelet activation); HSCT corrects ID not myopathy/cardiac risk' },
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
  const isCD3  = ['CD3D', 'CD3E', 'CD3G', 'CD247'].includes(gene);
  const isTCR  = ['LCK', 'LAT'].includes(gene);
  const isSOCE = ['ORAI1', 'STIM1'].includes(gene);
  const groupLabel = isCD3  ? 'CD3 Complex Assembly'
    : isTCR  ? 'TCR Proximal Kinase / Adapter'
    : 'SOCE Calcium Entry (CRAC)';
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
          {isCD3 && <>
            <Badge text={`T-Lymphopenia ${cohort.t_lymphopenia_pct}%`}           color={color} />
            <Badge text={`CD3-Absent ${cohort.cd3_absent_flow_pct ?? cohort.cd3_reduced_flow_pct ?? '—'}%`} color='#4a148c' />
            <Badge text={`Opportunistic ${cohort.opportunistic_inf_pct}%`}       color='#4e342e' />
            <Badge text={`HSCT ${cohort.hsct_pct}%`}                            color='#1565c0' />
          </>}
          {isTCR && <>
            <Badge text={`T-Lymphopenia ${cohort.t_lymphopenia_pct}%`}           color={color} />
            <Badge text={`Opportunistic ${cohort.opportunistic_inf_pct}%`}       color='#4e342e' />
            <Badge text={`HSCT ${cohort.hsct_pct}%`}                            color='#1565c0' />
          </>}
          {isSOCE && <>
            <Badge text={`Anhidrosis ${cohort.anhidrosis_pct}%`}                color='#e65100' />
            <Badge text={`Myopathy ${cohort.myopathy_pct}%`}                    color='#880e4f' />
            <Badge text={`HSCT ${cohort.hsct_pct}%`}                            color='#1565c0' />
          </>}
          <Badge text={`n=${cohort.n}`} color='#37474f' />
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{ background: '#1e1e2e', border: `1px solid ${color || '#444'}`, borderRadius: 8, padding: '14px 18px', minWidth: 140 }}>
      <div style={{ fontSize: 26, fontWeight: 700, color: color || '#fff' }}>{value}</div>
      <div style={{ fontSize: 13, color: '#ccc' }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function GroupCard({ group, data, color }) {
  if (!data) return null;
  return (
    <div style={{ background: '#1a1a2a', border: `1px solid ${color}`, borderRadius: 8, padding: 14, marginBottom: 14 }}>
      <div style={{ fontWeight: 700, color, marginBottom: 8 }}>{group}</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, fontSize: 12 }}>
        {Object.entries(data).map(([k, v]) => (
          <span key={k} style={{ background: '#2a2a3a', padding: '3px 8px', borderRadius: 4, color: '#ddd' }}>
            <span style={{ color }}>{k}:</span> {typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(1)) : String(v)}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function HreditaryCD3TCROrai1Stim1Atlas() {
  const [tab, setTab]         = useState('Overview');
  const [overview, setOverview]     = useState(null);
  const [breakdown, setBreakdown]   = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const headerStyle = { background: 'linear-gradient(135deg,#0d47a1 0%,#6a1b9a 100%)', padding: '24px 32px', borderRadius: 10, marginBottom: 24 };

  if (loading) return <div style={{ padding: 40, color: '#ccc', fontFamily: 'monospace' }}>Loading Hereditary CD3/TCR/ORAI1/STIM1 Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f66', fontFamily: 'monospace' }}>Error: {error}</div>;

  const cohortMap = {};
  (breakdown?.genes || []).forEach(g => { cohortMap[g.gene] = g; });

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#111', color: '#eee', minHeight: '100vh', padding: 24 }}>
      {/* Header */}
      <div style={headerStyle}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#fff' }}>
          🧬 Hereditary CD3/TCR/ORAI1/STIM1 Atlas
        </h1>
        <div style={{ color: '#b3d4ff', marginTop: 6, fontSize: 13 }}>
          CD3D · CD3E · CD3G · CD247 · LCK · LAT · ORAI1 · STIM1 — 8-Gene Reference · 320 Patients (40 per gene) · Seeds 3950–3957
        </div>
        <div style={{ color: '#c3b1e1', marginTop: 4, fontSize: 12 }}>
          CD3 Complex Assembly | TCR Proximal Kinase/Adapter | SOCE Calcium Entry (CRAC) — BCG Absolutely Contraindicated All 8 Genes
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontWeight: 600,
            background: tab === t ? '#0d47a1' : '#222', color: tab === t ? '#fff' : '#aaa'
          }}>{t}</button>
        ))}
      </div>

      {/* Tab: Overview */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 24 }}>
            <StatCard label="Total Patients"       value={overview.total_patients}      color='#42a5f5' />
            <StatCard label="Genes"                value={overview.total_genes}         color='#66bb6a' sub='8-gene atlas' />
            <StatCard label="Median Age Dx (mo)"  value={overview.median_age_dx_mo}    color='#ffa726' sub='months' />
            <StatCard label="SCID (T-B+NK+)"      value={`${overview.scid_pct}%`}      color='#ef5350' sub='CD3 complex defects' />
            <StatCard label="CD3 Flow Dx"         value={`${overview.cd3_flow_diagnostic_pct}%`} color='#ab47bc' sub='flow cytometry confirmed' />
            <StatCard label="Opportunistic Inf"   value={`${overview.opportunistic_inf_pct}%`}   color='#26c6da' />
            <StatCard label="BCG-osis"            value={`${overview.bcg_osis_pct}%`}  color='#ff7043' sub='BCG absolutely CI all 8' />
            <StatCard label="HSCT"                value={`${overview.hsct_pct}%`}      color='#26a69a' />
            <StatCard label="Anhidrosis (SOCE)"   value={`${overview.anhidrosis_pct}%`} color='#ec407a' sub='ORAI1/STIM1' />
            <StatCard label="Myopathy (SOCE)"     value={`${overview.myopathy_pct}%`}  color='#8d6e63' sub='tubular aggregate' />
          </div>

          <h3 style={{ color: '#b3d4ff' }}>Mechanistic Groups</h3>
          {breakdown?.diagnostic_groups && Object.entries(breakdown.diagnostic_groups).map(([grp, data]) => (
            <GroupCard key={grp} group={grp.replace(/_/g,' ').toUpperCase()} data={data}
              color={grp.includes('cd3') ? '#1565c0' : grp.includes('tcr') ? '#00796b' : '#7b1fa2'} />
          ))}

          <h3 style={{ color: '#c3b1e1', marginTop: 20 }}>Pathognomonic Features</h3>
          {(breakdown?.pathognomonic_features || []).map((f, i) => (
            <div key={i} style={{ background: '#1a1a2a', borderLeft: `4px solid ${f.color || '#6a1b9a'}`, padding: '10px 14px', marginBottom: 8, borderRadius: 4 }}>
              <span style={{ fontWeight: 700, color: f.color || '#c3b1e1' }}>{f.gene}: </span>
              <span style={{ fontSize: 13 }}>{f.feature}</span>
            </div>
          ))}

          <h3 style={{ color: '#80cbc4', marginTop: 20 }}>Cascade Testing Order</h3>
          <ol style={{ paddingLeft: 20, color: '#ccc', fontSize: 13, lineHeight: 1.8 }}>
            {(breakdown?.cascade_testing_order || []).map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ol>
        </div>
      )}

      {/* Tab: Gene Table */}
      {tab === 'Gene Table' && (
        <div>
          <p style={{ color: '#aaa', fontSize: 13, marginBottom: 16 }}>
            CD3D/E/G/Z = T-B+NK+ SCID (CD3G milder); LCK = LOF vs GOF opposite; LAT = surface CD3 present but downstream absent; ORAI1/STIM1 = SOCE defect with anhidrosis/myopathy. BCG absolutely CI all 8.
          </p>
          {Object.entries(GENE_INFO).map(([gene, info]) => (
            <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={info} cohort={cohortMap[gene]} />
          ))}
        </div>
      )}

      {/* Tab: Pathway Map */}
      {tab === 'Pathway Map' && (
        <div>
          <h3 style={{ color: '#b3d4ff' }}>T-Cell Activation Pathway — 3 Mechanistic Groups</h3>

          {/* CD3 Complex */}
          <div style={{ background: '#0a1929', border: '2px solid #1565c0', borderRadius: 10, padding: 18, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#42a5f5', fontSize: 16, marginBottom: 10 }}>Group 1 — CD3 Complex Assembly (CD3D / CD3E / CD3G / CD247)</div>
            <div style={{ fontSize: 13, color: '#ccc', lineHeight: 1.7 }}>
              TCR-αβ heterodimer pairs with <strong style={{color:'#42a5f5'}}>CD3δε + CD3γε + CD3ζζ</strong> to form the TCR–CD3 surface complex.
              Loss of any CD3 subunit prevents surface assembly → <strong style={{color:'#ef5350'}}>T-B+NK+ SCID</strong> (T cells absent; B/NK cells present).
              <br/><br/>
              <strong>Flow Dx:</strong> CD3δ/CD3ε/CD3γ LOF → surface CD3 <em>absent</em> (flow). CD3G hypomorphic → surface CD3 <em>reduced</em> (partial).
              CD247 (ζ-chain) loss → CD3 absent; ultra-rare (&lt;30 families).
              <br/>
              <strong>BCG:</strong> Absolutely CI — live attenuated mycobacterium → BCG-osis + dissemination.
            </div>
            <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {['CD3D','CD3E','CD3G','CD247'].map(g => (
                <div key={g} style={{ background: GENE_COLORS[g], color:'#fff', padding:'6px 12px', borderRadius:6, fontSize:12, fontWeight:700 }}>
                  {g} · {GENE_INFO[g].locus}
                </div>
              ))}
            </div>
          </div>

          {/* TCR Proximal Signaling */}
          <div style={{ background: '#0a1f1a', border: '2px solid #00796b', borderRadius: 10, padding: 18, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#4db6ac', fontSize: 16, marginBottom: 10 }}>Group 2 — TCR Proximal Kinase / Adapter (LCK / LAT)</div>
            <div style={{ fontSize: 13, color: '#ccc', lineHeight: 1.7 }}>
              TCR engagement → <strong style={{color:'#4db6ac'}}>LCK</strong> (CD4/CD8-associated kinase) phosphorylates CD3ζ ITAMs →
              <strong style={{color:'#4db6ac'}}> LAT</strong> (adapter scaffold) recruits PLCγ1, Grb2, SOS → downstream Ca²⁺/MAPK/NF-κB.
              <br/><br/>
              <strong>LCK LOF vs GOF:</strong> AR-LOF → T-lymphopenia + combined ID (treat with HSCT).
              AD-GOF → lymphoproliferation + autoimmunity (cytopenia, splenomegaly — consider mTOR inhibition FIRST before HSCT).
              <br/>
              <strong>LAT:</strong> Surface CD3 present (normal by flow!) but pTyr132-LAT phosphoflow after TCR stimulation = <em>absent</em> → DIAGNOSTIC. Easy to miss on routine immunophenotyping.
            </div>
            <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {['LCK','LAT'].map(g => (
                <div key={g} style={{ background: GENE_COLORS[g], color:'#fff', padding:'6px 12px', borderRadius:6, fontSize:12, fontWeight:700 }}>
                  {g} · {GENE_INFO[g].locus}
                </div>
              ))}
            </div>
          </div>

          {/* SOCE */}
          <div style={{ background: '#1a0a29', border: '2px solid #7b1fa2', borderRadius: 10, padding: 18, marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#ce93d8', fontSize: 16, marginBottom: 10 }}>Group 3 — Store-Operated Ca²⁺ Entry / SOCE (ORAI1 / STIM1)</div>
            <div style={{ fontSize: 13, color: '#ccc', lineHeight: 1.7 }}>
              TCR signaling depletes ER Ca²⁺ stores → <strong style={{color:'#ce93d8'}}>STIM1</strong> (ER Ca²⁺ sensor) translocates to ER–PM junctions → gates
              <strong style={{color:'#ce93d8'}}> ORAI1</strong> (CRAC channel pore) → Ca²⁺ influx → NFAT activation → T-cell gene transcription.
              <br/><br/>
              <strong>ORAI1:</strong> Tübingen p.Arg91Trp European founder. Triad: combined ID + <em>anhidrosis</em> (sweat gland SOCE required) + <em>myopathy</em>.
              CRAC current patch-clamp assay in T cells DIAGNOSTIC. HSCT corrects ID but NOT anhidrosis/myopathy.
              <br/>
              <strong>STIM1:</strong> Tubular aggregate myopathy (muscle biopsy PATHOGNOMONIC) + <em>cardiac arrhythmia</em> → annual ECG MANDATORY.
              Platelet dysfunction (SOCE required for platelet Ca²⁺ signaling).
            </div>
            <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {['ORAI1','STIM1'].map(g => (
                <div key={g} style={{ background: GENE_COLORS[g], color:'#fff', padding:'6px 12px', borderRadius:6, fontSize:12, fontWeight:700 }}>
                  {g} · {GENE_INFO[g].locus}
                </div>
              ))}
            </div>
          </div>

          {/* Phenotype comparison table */}
          <h3 style={{ color: '#b3d4ff', marginTop: 24 }}>Comparative Phenotype Matrix</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1a1a3a' }}>
                  {['Gene','Group','T-Cell ↓','CD3 Flow','Partial','LCK LOF/GOF','LAT phos','SOCE','Anhidrosis','Myopathy','HSCT','Key Distinguisher'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', color: '#aaa', textAlign: 'left', borderBottom: '1px solid #333' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map(r => (
                  <tr key={r.gene} style={{ borderBottom: '1px solid #222' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[r.gene] }}>{r.gene}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#aaa' }}>{r.group}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.tcell}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.cd3flow}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.partial}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.lck}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.lat}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.soce}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.anhid}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.myop}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.hsct}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#ccc', maxWidth: 300 }}>{r.unique}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab: Definitions */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ color: '#b3d4ff' }}>Clinical Definitions</h3>
          {(definitions.clinical_definitions || []).map((d, i) => (
            <div key={i} style={{ background: '#1a1a2a', border: '1px solid #333', borderRadius: 8, padding: 14, marginBottom: 10 }}>
              <div style={{ fontWeight: 700, color: '#42a5f5', marginBottom: 4 }}>{d.term}</div>
              <div style={{ fontSize: 13, color: '#ccc' }}>{d.definition}</div>
              {d.clinical_relevance && <div style={{ fontSize: 12, color: '#888', marginTop: 4 }}>Clinical: {d.clinical_relevance}</div>}
            </div>
          ))}

          <h3 style={{ color: '#c3b1e1', marginTop: 24 }}>Key Distinguishers</h3>
          {definitions.key_distinguishers && Object.entries(definitions.key_distinguishers).map(([gene, val]) => (
            <div key={gene} style={{ display: 'flex', gap: 12, background: '#1a1a2a', borderRadius: 6, padding: '10px 14px', marginBottom: 8, alignItems: 'flex-start' }}>
              <span style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#42a5f5', minWidth: 60 }}>{gene}</span>
              <span style={{ fontSize: 13, color: '#ccc' }}>{val}</span>
            </div>
          ))}

          <h3 style={{ color: '#80cbc4', marginTop: 24 }}>Diagnostic Pathway</h3>
          <ol style={{ paddingLeft: 20, color: '#ccc', fontSize: 13, lineHeight: 2 }}>
            {(definitions.diagnostic_pathway || []).map((step, i) => (
              <li key={i}>{step}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
