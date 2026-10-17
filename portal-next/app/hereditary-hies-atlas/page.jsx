'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hies-atlas';
const TABS = ['Overview', 'Gene Table', 'HIES Classification Map', 'Definitions'];

const GENE_COLORS = {
  'STAT3':  '#b71c1c',  // deep red         -- AD-HIES most common; retained teeth PATHOGNOMONIC; Th17 absent
  'DOCK8':  '#0d47a1',  // deep blue         -- AR-HIES viral infections; NK defect; lymphoma risk
  'PGM3':   '#1b5e20',  // dark green        -- AR-HIES skeletal dysplasia; neutropenia; CDG mechanism
  'CARD11': '#4a148c',  // deep purple       -- AD-GOF BENTA; B-cell expansion; NF-kB constitutive
  'IL6ST':  '#e65100',  // burnt orange      -- gp130 deficiency; coronary aneurysm PATHOGNOMONIC
  'ZNF341': '#880e4f',  // deep pink/magenta -- AR-HIES STAT3 promoter activator; stat3 seq negative
  'IL21R':  '#004d40',  // deep teal         -- AR CID; cryptosporidium liver PATHOGNOMONIC; B-cell loss
  'TYK2':   '#33691e',  // olive green       -- AR LOF; mycobacterial + viral + candida; multi-cytokine
};

const GENE_INFO = {
  'STAT3':  { full: 'STAT3-770aa / AD-HIES-Job-Syndrome / Th17-absent-cold-abscesses / retained-primary-teeth-PATHOGNOMONIC / NIH-score-GT40 / pneumatoceles', locus: '17q21.2', size: '770 aa / 88 kDa', inh: 'AD-DN' },
  'DOCK8':  { full: 'DOCK8-2099aa / AR-HIES-Type2 / viral-HSV-HPV-molluscum-DOMINANT / NK-cytotoxicity-severely-impaired / lymphoma-EBV-10-15pct / HSCT-curative', locus: '9p24.3', size: '2099 aa / 237 kDa', inh: 'AR' },
  'PGM3':   { full: 'PGM3-686aa / AR-HIES-Type3-CDG / skeletal-dysplasia-spondyloepiphyseal-PATHOGNOMONIC / neutropenia-chronic / eosinophilia / UDP-GlcNAc-deficiency', locus: '6q14.1', size: '686 aa / 77 kDa', inh: 'AR' },
  'CARD11': { full: 'CARD11-1154aa / AD-GOF-BENTA / B-cell-expansion-lymphocytosis-DISTINGUISHER / NF-kB-constitutive / T-cell-anergy / lymphoma-risk', locus: '7p22.2', size: '1154 aa / 133 kDa', inh: 'AD-GOF' },
  'IL6ST':  { full: 'IL6ST-gp130-918aa / AR-gp130-deficiency / coronary-artery-aneurysm-PATHOGNOMONIC / multi-cytokine-IL6-IL11-IL27-LIF / echo-annually-mandatory', locus: '5q11.2', size: '918 aa / 103 kDa', inh: 'AR' },
  'ZNF341': { full: 'ZNF341-844aa / AR-HIES-STAT3-promoter-activator / STAT3-protein-reduced-gene-intact / AR-inheritance-STAT3-seqneg / consanguineous', locus: '20q11.21', size: '844 aa / 95 kDa', inh: 'AR' },
  'IL21R':  { full: 'IL21R-538aa / AR-CID / cryptosporidium-biliary-sclerosing-cholangitis-PATHOGNOMONIC / progressive-B-cell-loss / HSCT-before-liver-failure', locus: '16p12.1', size: '538 aa / 59 kDa', inh: 'AR' },
  'TYK2':   { full: 'TYK2-1187aa / AR-LOF / combined-IL12-IFNalpha-IL23-signaling-loss / mycobacterial-MSMD-viral-candida / BCG-ABSOLUTELY-CI / P1104A-hypomorphic', locus: '19p13.2', size: '1187 aa / 134 kDa', inh: 'AR' },
};

const PHENOTYPE_ROWS = [
  { gene: 'STAT3',  inh: 'AD',    retained_teeth: '✓✓✓', cold_abscess: '✓✓✓', viral: '±',    food_allergy: '±',   coronary: '−',   lymphoma: '±',  neutropenia: '−',  unique: 'Retained teeth PATHOGNOMONIC; NIH score ≥40; pneumatoceles' },
  { gene: 'DOCK8',  inh: 'AR',    retained_teeth: '−',   cold_abscess: '±',   viral: '✓✓✓', food_allergy: '✓✓✓',coronary: '−',   lymphoma: '✓✓', neutropenia: '−',  unique: 'Viral (HSV/HPV) + food allergy + NK defect + lymphoma 10-15%' },
  { gene: 'PGM3',   inh: 'AR',    retained_teeth: '±',   cold_abscess: '±',   viral: '±',    food_allergy: '±',   coronary: '−',   lymphoma: '−',  neutropenia: '✓✓✓',unique: 'Skeletal dysplasia PATHOGNOMONIC + neutropenia (CDG)' },
  { gene: 'CARD11', inh: 'AD-GOF',retained_teeth: '−',   cold_abscess: '±',   viral: '±',    food_allergy: '✓✓', coronary: '−',   lymphoma: '±',  neutropenia: '−',  unique: 'B-cell LYMPHOCYTOSIS distinguisher; NF-κB constitutive' },
  { gene: 'IL6ST',  inh: 'AR',    retained_teeth: '✓✓',  cold_abscess: '✓',   viral: '±',    food_allergy: '±',   coronary: '✓✓✓',lymphoma: '−',  neutropenia: '−',  unique: 'Coronary aneurysm PATHOGNOMONIC; echo mandatory; multi-cytokine' },
  { gene: 'ZNF341', inh: 'AR',    retained_teeth: '✓✓✓', cold_abscess: '✓✓',  viral: '±',    food_allergy: '±',   coronary: '−',   lymphoma: '−',  neutropenia: '−',  unique: 'AR phenocopy of STAT3; STAT3 seq negative; protein level assay' },
  { gene: 'IL21R',  inh: 'AR',    retained_teeth: '−',   cold_abscess: '±',   viral: '±',    food_allergy: '±',   coronary: '−',   lymphoma: '±',  neutropenia: '−',  unique: 'Cryptosporidium biliary disease PATHOGNOMONIC; HSCT before liver failure' },
  { gene: 'TYK2',   inh: 'AR',    retained_teeth: '−',   cold_abscess: '±',   viral: '✓✓✓', food_allergy: '±',   coronary: '−',   lymphoma: '±',  neutropenia: '−',  unique: 'Mycobacterial + viral combined; BCG ABSOLUTELY-CI; P1104A hypomorphic' },
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
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ background: color, color: '#fff', borderRadius: 6, padding: '6px 14px', fontWeight: 700, fontSize: 18 }}>{gene}</div>
        <div>
          <div style={{ fontWeight: 600 }}>{cohort?.syndrome || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#444', marginBottom: 8 }}>{info?.full?.replace(/-/g, ' ')}</div>
      {cohort?.pathognomonic && (
        <div style={{ background: '#fff3cd', border: '1px solid #ffc107', borderRadius: 4, padding: '6px 10px', fontSize: 12, marginBottom: 8 }}>
          <strong>PATHOGNOMONIC:</strong> {cohort.pathognomonic}
        </div>
      )}
      {cohort?.first_line_treatment && (
        <div style={{ background: '#e8f5e9', border: '1px solid #4caf50', borderRadius: 4, padding: '6px 10px', fontSize: 12, marginBottom: 8 }}>
          <strong>Treatment:</strong> {cohort.first_line_treatment}
        </div>
      )}
      {cohort?.critical_rules?.length > 0 && (
        <div style={{ marginTop: 8 }}>
          {cohort.critical_rules.map(r => (
            <Badge key={r} text={r.replace(/-/g, ' ')} color={color} />
          ))}
        </div>
      )}
      {cohort?.n && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
          <Badge text={`n=${cohort.n}`} color={color} />
          <Badge text={`onset ${cohort.mean_onset_age}yr`} color="#555" />
          <Badge text={`dx-lag ${cohort.mean_dx_lag_yrs}yr`} color="#888" />
          {cohort.ige_elevated_n > 0 && <Badge text={`IgE↑ ${cohort.ige_elevated_n}/40`} color="#b71c1c" />}
          {cohort.eosinophilia_n > 0 && <Badge text={`Eos ${cohort.eosinophilia_n}/40`} color="#e65100" />}
          {cohort.cold_abscess_n > 0 && <Badge text={`ColdAbs ${cohort.cold_abscess_n}/40`} color="#1565c0" />}
          {cohort.retained_teeth_n > 0 && <Badge text={`RetainedTeeth ${cohort.retained_teeth_n}/40`} color="#880e4f" />}
          {cohort.viral_infection_n > 0 && <Badge text={`Viral ${cohort.viral_infection_n}/40`} color="#0d47a1" />}
          {cohort.food_allergy_n > 0 && <Badge text={`FoodAllergy ${cohort.food_allergy_n}/40`} color="#4a148c" />}
          {cohort.lymphoma_n > 0 && <Badge text={`Lymphoma ${cohort.lymphoma_n}/40`} color="#c62828" />}
          {cohort.coronary_aneurysm_n > 0 && <Badge text={`CoronaryAn ${cohort.coronary_aneurysm_n}/40`} color="#e65100" />}
          {cohort.hsct_benefit_n > 0 && <Badge text={`HSCT-ben ${cohort.hsct_benefit_n}/40`} color="#1b5e20" />}
        </div>
      )}
    </div>
  );
}

export default function HereditaryHIESAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefs(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#888' }}>Loading Hereditary HIES Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  const cohortByGene = {};
  (breakdown?.genes || []).forEach(g => { cohortByGene[g.gene] = g; });

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #b71c1c 0%, #0d47a1 40%, #4a148c 70%, #e65100 100%)', borderRadius: 10, padding: '20px 28px', marginBottom: 24, color: '#fff' }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>🧬 Hereditary HIES Atlas</h1>
        <div style={{ marginTop: 6, fontSize: 13, opacity: 0.9 }}>
          Hyper-IgE Syndrome &amp; Related PIDs · Complete 8-Gene Reference · STAT3 · DOCK8 · PGM3 · CARD11 · IL6ST · ZNF341 · IL21R · TYK2 · {overview?.total_patients} patients · seeds {overview?.seeds}
        </div>
        <div style={{ marginTop: 4, fontSize: 12, opacity: 0.8 }}>
          Th17-Absent · Cold Abscesses · Retained Teeth · Viral NK-Defect · Skeletal Dysplasia · Coronary Aneurysm · Cryptosporidium · Mycobacterial
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button
            key={t}
            onClick={() => setTab(i)}
            style={{
              padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 14, fontWeight: 600,
              background: tab === i ? '#b71c1c' : '#f5f5f5',
              color: tab === i ? '#fff' : '#333',
            }}
          >{t}</button>
        ))}
      </div>

      {/* Tab 0: Overview */}
      {tab === 0 && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#b71c1c' },
              { label: 'Genes Covered', value: overview.genes?.length, color: '#4a148c' },
              { label: 'Cohort per Gene', value: overview.cohort_size, color: '#0d47a1' },
              { label: 'Seed Range', value: overview.seeds, color: '#1b5e20' },
            ].map(k => (
              <div key={k.label} style={{ background: '#fff', border: `2px solid ${k.color}`, borderRadius: 8, padding: 16, textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 800, color: k.color }}>{k.value}</div>
                <div style={{ fontSize: 13, color: '#666', marginTop: 4 }}>{k.label}</div>
              </div>
            ))}
          </div>

          <h2 style={{ color: '#b71c1c', marginBottom: 12 }}>Key Clinical Rules</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 12, marginBottom: 24 }}>
            {Object.entries(overview.key_facts || {}).map(([k, v]) => (
              <div key={k} style={{ background: '#fff8e1', border: '1px solid #ffa000', borderRadius: 6, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#e65100', textTransform: 'uppercase', marginBottom: 4 }}>
                  {k.replace(/_/g, ' ')}
                </div>
                <div style={{ fontSize: 13 }}>{v}</div>
              </div>
            ))}
          </div>

          <h2 style={{ color: '#b71c1c', marginBottom: 12 }}>Diagnostic Approach — Step-by-Step</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 12, marginBottom: 24 }}>
            {Object.entries(overview.diagnostic_approach || {}).map(([k, v]) => (
              <div key={k} style={{ background: '#e8f5e9', border: '1px solid #43a047', borderRadius: 6, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#1b5e20', textTransform: 'uppercase', marginBottom: 4 }}>
                  {k.replace(/_/g, ' ')}
                </div>
                <div style={{ fontSize: 13 }}>{v}</div>
              </div>
            ))}
          </div>

          <h2 style={{ color: '#b71c1c', marginBottom: 12 }}>Gene Cohorts Overview</h2>
          {overview.cohorts?.map(c => (
            <GeneCard
              key={c.gene}
              gene={c.gene}
              color={GENE_COLORS[c.gene] || '#555'}
              info={GENE_INFO[c.gene]}
              cohort={c}
            />
          ))}
        </div>
      )}

      {/* Tab 1: Gene Table */}
      {tab === 1 && breakdown && (
        <div>
          <h2 style={{ color: '#b71c1c', marginBottom: 16 }}>Gene Reference Table</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#b71c1c', color: '#fff' }}>
                  {['#', 'Gene', 'Protein / Size', 'Locus', 'Inh.', 'n', 'Onset', 'Dx-Lag', 'IgE↑', 'Eos', 'ColdAbs', 'RetainedTeeth', 'Viral', 'FoodAllergy', 'Lymphoma', 'CoronaryAn', 'HSCT-ben'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.genes?.map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '7px 10px', color: '#888' }}>{g.rank}</td>
                    <td style={{ padding: '7px 10px' }}>
                      <span style={{ background: GENE_COLORS[g.gene] || '#555', color: '#fff', borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>{g.gene}</span>
                    </td>
                    <td style={{ padding: '7px 10px', fontSize: 11, color: '#444' }}>{g.aa}aa / {g.mw_kda}kDa</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{g.locus}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{g.inheritance}-{g.mechanism}</td>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#555' }}>{g.n}</td>
                    <td style={{ padding: '7px 10px' }}>{g.mean_onset_age}yr</td>
                    <td style={{ padding: '7px 10px' }}>{g.mean_dx_lag_yrs}yr</td>
                    <td style={{ padding: '7px 10px', color: '#b71c1c', fontWeight: 700 }}>{g.ige_elevated_n}</td>
                    <td style={{ padding: '7px 10px', color: '#e65100' }}>{g.eosinophilia_n}</td>
                    <td style={{ padding: '7px 10px', color: '#1565c0' }}>{g.cold_abscess_n}</td>
                    <td style={{ padding: '7px 10px', color: '#880e4f', fontWeight: g.retained_teeth_n > 20 ? 700 : 400 }}>{g.retained_teeth_n}</td>
                    <td style={{ padding: '7px 10px', color: '#0d47a1', fontWeight: g.viral_infection_n > 25 ? 700 : 400 }}>{g.viral_infection_n}</td>
                    <td style={{ padding: '7px 10px', color: '#4a148c' }}>{g.food_allergy_n}</td>
                    <td style={{ padding: '7px 10px', color: '#c62828' }}>{g.lymphoma_n}</td>
                    <td style={{ padding: '7px 10px', color: '#e65100', fontWeight: g.coronary_aneurysm_n > 0 ? 700 : 400 }}>{g.coronary_aneurysm_n}</td>
                    <td style={{ padding: '7px 10px', color: '#1b5e20' }}>{g.hsct_benefit_n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ marginTop: 20 }}>
            <h3 style={{ color: '#b71c1c', marginBottom: 12 }}>Critical Rules per Gene</h3>
            {breakdown.genes?.map(g => (
              <div key={g.gene} style={{ marginBottom: 14, padding: 12, background: '#fafafa', borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#555', marginBottom: 6 }}>{g.gene} — {g.syndrome?.split('/')[0]}</div>
                <div style={{ marginBottom: 6, fontSize: 12, color: '#444' }}>
                  <strong>PATHOGNOMONIC:</strong> {g.pathognomonic}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {g.critical_rules?.map(r => (
                    <Badge key={r} text={r.replace(/-/g, ' ')} color={GENE_COLORS[g.gene] || '#555'} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 2: HIES Classification Map */}
      {tab === 2 && (
        <div>
          <h2 style={{ color: '#b71c1c', marginBottom: 16 }}>HIES Differential Diagnosis Classification</h2>

          <div style={{ background: '#fff3e0', border: '2px solid #ff6f00', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#e65100', margin: '0 0 10px' }}>AD vs AR HIES — Inheritance Separator</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10 }}>
              {[
                { type: 'AD-HIES', genes: 'STAT3, CARD11-GOF', features: 'De novo or family; STAT3: retained teeth, skeletal, pneumatoceles; CARD11: B-cell expansion; no typical AR family structure', color: '#b71c1c' },
                { type: 'AR-HIES (immune)', genes: 'DOCK8, IL6ST, ZNF341, IL21R, TYK2', features: 'Consanguineous families enriched; siblings at 25% risk; phenotype varies widely by gene (viral/skeletal/coronary/liver)', color: '#0d47a1' },
                { type: 'AR-HIES (CDG)', genes: 'PGM3', features: 'Hypo-glycosylation mechanism; skeletal dysplasia + neutropenia distinguish; glucosamine supplementation under study', color: '#1b5e20' },
              ].map(p => (
                <div key={p.type} style={{ background: '#fff', border: `2px solid ${p.color}`, borderRadius: 6, padding: 12 }}>
                  <div style={{ fontWeight: 700, color: p.color, marginBottom: 4 }}>{p.type}: {p.genes}</div>
                  <div style={{ fontSize: 13 }}>{p.features}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ overflowX: 'auto', marginBottom: 24 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#b71c1c', color: '#fff' }}>
                  {['Gene', 'Inh.', 'Retained Teeth', 'Cold Abscess', 'Viral Dominant', 'Food Allergy', 'Coronary An.', 'Lymphoma', 'Neutropenia', 'Unique Feature'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff', borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '7px 10px' }}>
                      <span style={{ background: GENE_COLORS[row.gene] || '#555', color: '#fff', borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>{row.gene}</span>
                    </td>
                    <td style={{ padding: '7px 10px', fontSize: 11, fontWeight: 600 }}>{row.inh}</td>
                    {[row.retained_teeth, row.cold_abscess, row.viral, row.food_allergy, row.coronary, row.lymphoma, row.neutropenia].map((val, vi) => (
                      <td key={vi} style={{ padding: '7px 10px', color: val === '✓✓✓' ? '#b71c1c' : (val === '✓✓' ? '#e65100' : (val === '✓' ? '#1b5e20' : (val === '−' ? '#ccc' : '#888'))), fontWeight: val.includes('✓') ? 700 : 400, fontSize: 15 }}>{val}</td>
                    ))}
                    <td style={{ padding: '7px 10px', fontSize: 11, color: '#333', maxWidth: 220 }}>{row.unique}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ background: '#fce4ec', border: '2px solid #c62828', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <h3 style={{ color: '#c62828', margin: '0 0 10px' }}>⚠️ Critical Safety &amp; Diagnostic Rules</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>STAT3 vs ZNF341:</strong> Classic HIES + AR inheritance + STAT3 sequencing NEGATIVE → STAT3 protein level (reduced in ZNF341) → ZNF341 sequencing. Never stop at STAT3-negative without checking ZNF341.</li>
              <li><strong>DOCK8 HSCT:</strong> STRONGLY recommended early — curative for immune phenotype. Prevents EBV lymphoma (10-15% risk). NK cytotoxicity and T-cell function restored. DO NOT delay HSCT for DOCK8.</li>
              <li><strong>IL6ST coronary:</strong> Echo MANDATORY annually. Coronary artery aneurysms in infancy are Kawasaki-like and require aspirin + IVIG per Kawasaki protocol. ECG and echo from diagnosis.</li>
              <li><strong>IL21R liver urgency:</strong> Cryptosporidium biliary disease is progressive and irreversible. HSCT must happen BEFORE sclerosing cholangitis becomes end-stage liver disease. Urgency cannot be overstated.</li>
              <li><strong>TYK2 — BCG ABSOLUTELY CONTRAINDICATED:</strong> TYK2-deficient infants given BCG develop disseminated mycobacterial disease. Do NOT vaccinate with BCG until TYK2 deficiency excluded in high-risk infants.</li>
              <li><strong>CARD11-GOF lymphoma:</strong> B-cell clonal expansion + persistent uncontrolled NF-κB → lymphoma risk. Annual lymphoma surveillance (LDH, neck/abdominal ultrasound or CT). Avoid broad immunosuppression.</li>
              <li><strong>PGM3 neutropenia:</strong> G-CSF titrated to ANC >1000. Neutropenia predisposes to bacterial infections on top of HIES features. Unlike STAT3-HIES which has normal neutrophil count.</li>
            </ul>
          </div>
        </div>
      )}

      {/* Tab 3: Definitions */}
      {tab === 3 && defs && (
        <div>
          <h2 style={{ color: '#b71c1c', marginBottom: 16 }}>Clinical Definitions</h2>
          {Object.entries(defs.definitions || {}).map(([title, text]) => (
            <div key={title} style={{ marginBottom: 20, padding: 16, background: '#fafafa', borderRadius: 8, borderLeft: '4px solid #b71c1c' }}>
              <h3 style={{ color: '#b71c1c', margin: '0 0 10px', fontSize: 15 }}>{title}</h3>
              <p style={{ margin: 0, fontSize: 13, lineHeight: 1.7, color: '#333' }}>{text}</p>
            </div>
          ))}

          <div style={{ marginTop: 24, padding: 16, background: '#e3f2fd', borderRadius: 8 }}>
            <h3 style={{ color: '#1565c0', margin: '0 0 10px' }}>Standards &amp; References</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13 }}>
              {defs.standards?.map(s => <li key={s} style={{ marginBottom: 6 }}>{s}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
