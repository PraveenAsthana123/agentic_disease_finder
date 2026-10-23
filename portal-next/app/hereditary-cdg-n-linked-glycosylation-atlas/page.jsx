'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cdg-n-linked-glycosylation-atlas';
const TABS = ['Overview', 'Gene Table', 'IEF Algorithm', 'Treatment', 'Definitions'];

const GENE_COLORS = {
  'PMM2':    '#1b5e20',  // deep green       — CDG-Ia; most common; inverted nipples + peau d'orange PATHOGNOMONIC
  'MPI':     '#0d47a1',  // deep blue         — CDG-Ib; NO neurological; mannose CURATIVE
  'ALG6':    '#4a148c',  // deep purple       — CDG-Ic; 2nd most common; milder
  'SRD5A3':  '#e65100',  // burnt orange      — CDG-Iq; ichthyosis + cerebellar PATHOGNOMONIC
  'SLC35C1': '#880e4f',  // deep crimson      — CDG-IIc/LADII; Bombay blood group PATHOGNOMONIC
  'PGM1':    '#37474f',  // dark slate        — PGM1-CDG; mixed glycosylation; galactose+mannose RESPONSIVE
  'TMEM165': '#004d40',  // dark teal         — TMEM165-CDG; Golgi Mn pump; manganese responsive
  'ATP6AP1': '#bf360c',  // dark ember        — ATP6AP1-CDG; XLR; liver dominant; copper abnormal
};

const GENE_INFO = {
  'PMM2':    { full: 'PMM2-259aa / AR / Phosphomannomutase-2 / 16p13.2 / CDG-Ia / Most-Common-65-70pct / InvertedNipples+PeauDOrange-PATHOGNOMONIC-Infancy / CerebellarHypoplasia / Transferrin-IEF-Type1 / Mannose-NOT-EFFECTIVE / p.R141H-50pct-European-AlleleFreq-Never-Homozygous-Viable', locus: '16p13.2', size: '259 aa / 28 kDa', inh: 'AR; Most common CDG (65-70%); inverted nipples + abnormal subcutaneous fat pads (peau d\'orange) in infancy = PATHOGNOMONIC; cerebellar hypoplasia in >90%; coagulopathy (Factor XI/antithrombin deficiency); p.R141H never viable as homozygote — always compound heterozygous; mannose supplement NOT effective for PMM2-CDG; IEF type 1 same as MPI — enzyme assay distinguishes' },
  'MPI':     { full: 'MPI-423aa / AR / Mannose-Phosphate-Isomerase / 15q24.1 / CDG-Ib / THE-Treatable-CDG / NO-Neurological-PATHOGNOMONIC-Key-DDx-PMM2 / Hepatic-Fibrosis-Dominant / Protein-Losing-Enteropathy / Mannose-1g-kg-CURATIVE / Transferrin-IEF-Type1-same-as-PMM2', locus: '15q24.1', size: '423 aa / 47 kDa', inh: 'AR; THE treatable CDG; NO neurological involvement = PATHOGNOMONIC (distinguishes from all other CDG types); hepatic fibrosis + portal hypertension; protein-losing enteropathy; coagulopathy; recurrent hypoglycemia; SAME IEF type 1 as PMM2 — ALWAYS do MPI enzyme assay in type 1 IEF without cerebellar; oral mannose 1g/kg/day CURATIVE for hepatopathy/enteropathy/coagulopathy' },
  'ALG6':    { full: 'ALG6-507aa / AR / Glucosyltransferase-I / 1p31.3 / CDG-Ic / 2nd-Most-Common-after-PMM2 / Milder-Phenotype / Epilepsy-More-Prominent-20-30pct / No-Inverted-Nipples-DDx-PMM2 / No-Cerebellar-Hypoplasia-Many / Transferrin-IEF-Type1 / No-Approved-Therapy', locus: '1p31.3', size: '507 aa / 57 kDa', inh: 'AR; Second most common CDG; milder than PMM2; NO inverted nipples (distinguishes from PMM2); epilepsy more prominent (20-30%); cerebellar hypoplasia absent in many; coagulopathy common; IEF type 1; no approved therapy; glucosylation of LLO first step — calnexin/calreticulin QC escape' },
  'SRD5A3':  { full: 'SRD5A3-318aa / AR / Polyprenol-Reductase / 4q11 / CDG-Iq / Ichthyosis-PATHOGNOMONIC / CerebellarHypoplasia / OcularColoboma-50-70pct / DolicholDeficiency / Polyprenol-Dolichol-Ratio-INVERTED / Transferrin-IEF-Type1 / Skin-Emollients', locus: '4q11', size: '318 aa / 35 kDa', inh: 'AR; polyprenol reductase — converts polyprenol to dolichol (dolichol = LLO carrier); ichthyosis (lamellar scaling) + cerebellar hypoplasia = PATHOGNOMONIC combination for SRD5A3-CDG; ocular coloboma in 50-70%; plasma polyprenol/dolichol-18 ratio inverted; IEF type 1; skin emollients; no pharmacotherapy' },
  'SLC35C1': { full: 'SLC35C1-364aa / AR / GDP-Fucose-Transporter-1 / 11p11.2 / CDG-IIc-LADII-RamonSyndrome / BombayBloodGroup-PATHOGNOMONIC / Anti-H-Antibody / CD15s-Absent-Neutrophils / LAD-II-Phenotype / OralFucose-Variant-Dependent / Transferrin-IEF-Type2 / BloodBank-ALERT-MANDATORY', locus: '11p11.2', size: '364 aa / 40 kDa', inh: 'AR; Bombay (h/h) blood group = PATHOGNOMONIC (no H antigen on RBCs — incompatible with all standard ABO types); CD15s absent on neutrophils (flowcytometry); LAD-II phenotype (recurrent bacterial infections, leukocytosis); IEF type 2; oral fucose responsive in p.T308R variant; CRITICAL: blood bank alert for all admissions; intellectual disability; short stature' },
  'PGM1':    { full: 'PGM1-562aa / AR / Phosphoglucomutase-1 / 1p31.3 / PGM1-CDG / Mixed-N-O-Glycosylation / BifidUvula-CleftPalate-50-60pct / ExerciseIntolerance-Rhabdomyolysis / DilatedCardiomyopathy-20-30pct / Mannose-AND-Galactose-RESPONSIVE / Mixed-IEF-Pattern-1plus2 / Echocardiogram-MANDATORY', locus: '1p31.3', size: '562 aa / 63 kDa', inh: 'AR; mixed N+O linked glycosylation defect; bifid uvula/cleft palate (50-60%) + rhabdomyolysis + mixed IEF pattern = essentially diagnostic combination; dilated cardiomyopathy 20-30% — echocardiogram MANDATORY; D-galactose 1-1.5g/kg/day + mannose 0.5g/kg/day RESPONSIVE; avoid prolonged fasting (hypoglycemia + myopathy); CK monitoring' },
  'TMEM165': { full: 'TMEM165-324aa / AR / Golgi-Mn-Ca-Transporter / 4q12 / TMEM165-CDG / Transferrin-IEF-Type2-Golgi / Manganese-Supplementation-RESPONSIVE / Moderate-ID / Hypotonia / Skeletal-Dysplasia-Some / SeekSpecialistMetabolicCenter', locus: '4q12', size: '324 aa / 36 kDa', inh: 'AR; Golgi Ca²⁺/Mn²⁺ pump deficiency; IEF type 2 (Golgi defect); manganese (MnCl₂) supplementation corrects transferrin IEF in ~40-70% of patients — FIRST CDG where mineral supplementation corrects biochemistry; moderate ID; hypotonia; skeletal dysplasia in some; specialist metabolic center referral for Mn dosing' },
  'ATP6AP1': { full: 'ATP6AP1-394aa / XLR / V-ATPase-Accessory-Subunit-AC45 / Xq28 / ATP6AP1-CDG / X-Linked-Males-Predominantly / Liver-Disease-DOMINANT / HepatosplenomegalyPrimary / LowCeruloplasmin-CopperAbnormal / Wilson-DDx / Immunodeficiency / Transferrin-IEF-Type2 / Liver-Transplant-Curative', locus: 'Xq28', size: '394 aa / 45 kDa', inh: 'XLR; V-ATPase accessory subunit; X-linked — males primarily affected; liver disease DOMINANT presenting feature; hepatosplenomegaly + hepatic fibrosis; low ceruloplasmin + hepatic copper accumulation (mimics Wilson disease — ATP7B normal distinguishes); immunodeficiency (recurrent infections); IEF type 2; liver transplantation curative for hepatic disease and copper abnormality; IMPORTANT: transferrin IEF type 2 in male with liver disease + low ceruloplasmin → ATP6AP1-CDG until proven otherwise' },
};

const PHENOTYPE_ROWS = [
  { gene: 'PMM2',    group: 'ER Defect — CDG-Ia',    ief: 'Type 1', inverted_nipples: '✓', cerebellar: '✓ (>90%)', treatable: '✗', pathognomonic: 'Inverted nipples + peau d\'orange fat pads (infancy)', rx: 'Supportive; physio; coag monitoring' },
  { gene: 'MPI',     group: 'ER Defect — CDG-Ib',    ief: 'Type 1', inverted_nipples: '✗', cerebellar: '✗ (NONE)', treatable: '✓', pathognomonic: 'NO neurological; mannose CURATIVE; hepatic/enteropathy', rx: 'Oral mannose 1g/kg/day CURATIVE' },
  { gene: 'ALG6',    group: 'ER Defect — CDG-Ic',    ief: 'Type 1', inverted_nipples: '✗', cerebellar: '+/− (mild)', treatable: '✗', pathognomonic: 'No inverted nipples (DDx PMM2); epilepsy 20-30%', rx: 'AEDs; supportive' },
  { gene: 'SRD5A3',  group: 'ER Defect — CDG-Iq',    ief: 'Type 1', inverted_nipples: '✗', cerebellar: '✓', treatable: '✗', pathognomonic: 'Ichthyosis + cerebellar hypoplasia PATHOGNOMONIC; coloboma 50-70%', rx: 'Skin emollients; supportive' },
  { gene: 'SLC35C1', group: 'Golgi Defect — CDG-IIc', ief: 'Type 2', inverted_nipples: '✗', cerebellar: '✗', treatable: '✓', pathognomonic: 'Bombay blood group PATHOGNOMONIC; CD15s absent; LAD-II', rx: 'Oral fucose (variant-dependent); blood bank ALERT' },
  { gene: 'PGM1',    group: 'Mixed N+O — PGM1-CDG',  ief: 'Mixed 1+2', inverted_nipples: '✗', cerebellar: '✗', treatable: '✓', pathognomonic: 'Bifid uvula/cleft palate + rhabdomyolysis + mixed IEF', rx: 'D-galactose + mannose; echo MANDATORY' },
  { gene: 'TMEM165', group: 'Golgi Defect',           ief: 'Type 2', inverted_nipples: '✗', cerebellar: '✗', treatable: '✓', pathognomonic: 'IEF type 2; Mn²⁺ supplementation corrects IEF', rx: 'Mn supplementation (specialist center)' },
  { gene: 'ATP6AP1', group: 'Golgi Defect — XLR',    ief: 'Type 2', inverted_nipples: '✗', cerebellar: '✗', treatable: '✓', pathognomonic: 'XLR males; liver dominant; low ceruloplasmin (Wilson DDx)', rx: 'Liver transplant; copper monitoring' },
];

const SEED_BASE_DISPLAY = {
  'PMM2': 4246, 'MPI': 4247, 'ALG6': 4248, 'SRD5A3': 4249,
  'SLC35C1': 4250, 'PGM1': 4251, 'TMEM165': 4252, 'ATP6AP1': 4253,
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info }) {
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 12, marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          <span style={{ background: color, color: '#fff', fontWeight: 700, padding: '3px 10px', borderRadius: 5, fontSize: 15 }}>{gene}</span>
          <span style={{ marginLeft: 8, fontSize: 12, color: '#aaa' }}>{info.locus} · {info.size} · {info.inh}</span>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#ccc', marginTop: 6, lineHeight: 1.5 }}>{info.full}</div>
    </div>
  );
}

export default function HereditoryCDGNLinkedAtlasPage() {
  const [tab, setTab]             = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefs(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const tabStyle = (t) => ({
    padding: '6px 14px', marginRight: 4, borderRadius: '4px 4px 0 0', cursor: 'pointer',
    background: tab === t ? '#1b5e20' : '#333', color: '#fff', border: 'none', fontSize: 13,
  });

  if (loading) return <div style={{ background: '#111', color: '#fff', minHeight: '100vh', padding: 24 }}>Loading CDG Atlas…</div>;
  if (error)   return <div style={{ background: '#111', color: '#f66', minHeight: '100vh', padding: 24 }}>Error: {error}</div>;

  return (
    <div style={{ background: '#111', color: '#eee', minHeight: '100vh', padding: 16, fontFamily: 'monospace' }}>
      <h1 style={{ color: '#81c784', fontSize: 20, marginBottom: 4 }}>
        🧬 Hereditary-CDG-N-Linked-Glycosylation-Atlas — Complete 8-Gene Reference
      </h1>
      <div style={{ fontSize: 12, color: '#aaa', marginBottom: 12 }}>
        PMM2 · MPI · ALG6 · SRD5A3 · SLC35C1 · PGM1 · TMEM165 · ATP6AP1 ·
        seeds {SEED_BASE_DISPLAY['PMM2']}–{SEED_BASE_DISPLAY['ATP6AP1']} ·
        Transferrin-IEF-Primary-Split · MPI-Mannose-CURATIVE · Bombay-Blood-Group-SLC35C1-PATHOGNOMONIC
      </div>

      <div style={{ marginBottom: 0 }}>
        {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      <div style={{ background: '#1a1a1a', borderRadius: '0 4px 4px 4px', padding: 16 }}>

        {/* ── OVERVIEW ── */}
        {tab === 'Overview' && overview && (
          <div>
            <h2 style={{ color: '#81c784', fontSize: 16 }}>CDG N-Linked Atlas Overview — {overview.total_patients} Patients · 8 Genes</h2>
            <div style={{ fontSize: 13, color: '#aaa', marginBottom: 12 }}>{overview.diagnostic_triage}</div>

            {/* Treatable genes callout */}
            <div style={{ background: '#0d3b14', border: '1px solid #4caf50', borderRadius: 6, padding: 10, marginBottom: 14 }}>
              <strong style={{ color: '#81c784' }}>⚡ Treatable CDG Genes: </strong>
              <span style={{ color: '#e8f5e9', fontSize: 13 }}>{overview.treatable_genes?.join(' · ')}</span>
              <div style={{ fontSize: 12, color: '#a5d6a7', marginTop: 4 }}>
                MPI → oral mannose CURATIVE | SLC35C1 → oral fucose (variant-dependent) | PGM1 → mannose + galactose RESPONSIVE | TMEM165 → Mn supplementation | ATP6AP1 → liver transplant
              </div>
            </div>

            {/* IEF groups */}
            {overview.ief_groups && Object.entries(overview.ief_groups).map(([grpKey, grp]) => (
              <div key={grpKey} style={{ marginBottom: 12, background: '#222', borderRadius: 6, padding: 10 }}>
                <div style={{ color: '#ffb74d', fontWeight: 700, marginBottom: 4 }}>
                  {grpKey === 'type1_pattern' ? '🔵 Transferrin IEF Type 1 (ER Defect)' :
                   grpKey === 'type2_pattern' ? '🟠 Transferrin IEF Type 2 (Golgi Defect)' :
                   '🟣 Mixed IEF Pattern (N+O Glycosylation)'}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 4 }}>
                  {grp.genes?.map(g => (
                    <span key={g} style={{ background: GENE_COLORS[g], color: '#fff', padding: '2px 10px', borderRadius: 4, fontSize: 13, fontWeight: 700 }}>{g}</span>
                  ))}
                </div>
                <div style={{ fontSize: 12, color: '#ccc' }}>{grp.description}</div>
                <div style={{ fontSize: 11, color: '#ff8a65', marginTop: 3 }}>⚠ {grp.notes}</div>
              </div>
            ))}

            {/* Summary table */}
            <div style={{ overflowX: 'auto', marginTop: 12 }}>
              <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#333' }}>
                    {['Gene','Locus','aa','kDa','Inh','Treatable','Key Pathognomonic','Mechanism','N'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', textAlign: 'left', color: '#81c784', borderBottom: '1px solid #444' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.summary_table?.map((row, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#1e1e1e' : '#222' }}>
                      <td style={{ padding: '5px 8px', fontWeight: 700, color: GENE_COLORS[row.gene] || '#fff' }}>{row.gene}</td>
                      <td style={{ padding: '5px 8px', color: '#aaa' }}>{row.locus}</td>
                      <td style={{ padding: '5px 8px' }}>{row.aa}</td>
                      <td style={{ padding: '5px 8px' }}>{row.kda}</td>
                      <td style={{ padding: '5px 8px' }}>{row.inheritance}</td>
                      <td style={{ padding: '5px 8px', color: row.treatable ? '#81c784' : '#ef9a9a' }}>{row.treatable ? '✓ YES' : '✗'}</td>
                      <td style={{ padding: '5px 8px', maxWidth: 280, fontSize: 11 }}>{row.pathognomonic}</td>
                      <td style={{ padding: '5px 8px', fontSize: 11, color: '#aaa' }}>{row.mechanism}</td>
                      <td style={{ padding: '5px 8px' }}>{row.n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── GENE TABLE ── */}
        {tab === 'Gene Table' && (
          <div>
            <h2 style={{ color: '#81c784', fontSize: 16 }}>CDG Gene Reference — 8 Genes</h2>
            {Object.keys(GENE_INFO).map(gene => (
              <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={GENE_INFO[gene]} />
            ))}

            <h3 style={{ color: '#ffb74d', marginTop: 16 }}>Phenotype Comparison Table</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#333' }}>
                    {['Gene','Group','IEF Pattern','Inv. Nipples','Cerebellar','Treatable','Key Pathognomonic','Treatment'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', textAlign: 'left', color: '#81c784', borderBottom: '1px solid #444' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {PHENOTYPE_ROWS.map((row, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#1e1e1e' : '#222' }}>
                      <td style={{ padding: '5px 8px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                      <td style={{ padding: '5px 8px', fontSize: 11, color: '#aaa' }}>{row.group}</td>
                      <td style={{ padding: '5px 8px', fontWeight: 700 }}>{row.ief}</td>
                      <td style={{ padding: '5px 8px' }}>{row.inverted_nipples}</td>
                      <td style={{ padding: '5px 8px' }}>{row.cerebellar}</td>
                      <td style={{ padding: '5px 8px', color: row.treatable.includes('✓') ? '#81c784' : '#ef9a9a' }}>{row.treatable}</td>
                      <td style={{ padding: '5px 8px', maxWidth: 260, fontSize: 11 }}>{row.pathognomonic}</td>
                      <td style={{ padding: '5px 8px', fontSize: 11, color: '#80cbc4' }}>{row.rx}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── IEF ALGORITHM ── */}
        {tab === 'IEF Algorithm' && defs && (
          <div>
            <h2 style={{ color: '#81c784', fontSize: 16 }}>Transferrin IEF Cascade Algorithm</h2>
            <div style={{ background: '#0a2e0a', border: '1px solid #4caf50', borderRadius: 8, padding: 14, marginBottom: 16, fontSize: 13, lineHeight: 1.7 }}>
              {defs.primary_cascade_split?.split('. ').map((step, i) => (
                <div key={i} style={{ marginBottom: 4, paddingLeft: step.startsWith('STEP') ? 0 : 16 }}>
                  {step.startsWith('STEP') ? <strong style={{ color: '#81c784' }}>{step}.</strong> : <span style={{ color: '#ccc' }}>• {step}.</span>}
                </div>
              ))}
            </div>

            {/* IEF notes */}
            {defs.ief_notes && (
              <div style={{ background: '#1a1a2e', border: '1px solid #7c4dff', borderRadius: 6, padding: 12, marginBottom: 14 }}>
                <h3 style={{ color: '#ce93d8', margin: '0 0 8px' }}>Transferrin IEF Notes</h3>
                {Object.entries(defs.ief_notes).map(([k, v]) => (
                  <div key={k} style={{ fontSize: 12, marginBottom: 4 }}>
                    <span style={{ color: '#7c4dff', fontWeight: 700 }}>{k.replace(/_/g, ' ')}: </span>
                    <span style={{ color: '#ccc' }}>{v}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Critical distinctions */}
            {defs.key_distinguishers?.map((kd, i) => (
              <div key={i} style={{ background: '#222', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                <div style={{ color: '#ffb74d', fontWeight: 700, marginBottom: 6, fontSize: 14 }}>{kd.feature}</div>
                <div style={{ fontSize: 12, color: '#aaa', marginBottom: 4 }}>{kd.significance}</div>
                {Object.entries(kd).filter(([k]) => !['feature','significance'].includes(k)).map(([k, v]) => (
                  <div key={k} style={{ fontSize: 12, marginBottom: 2 }}>
                    <span style={{ color: '#81c784', fontWeight: 700 }}>{k.replace(/_/g, ' ')}: </span>
                    <span style={{ color: '#ddd' }}>{v}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}

        {/* ── TREATMENT ── */}
        {tab === 'Treatment' && breakdown && (
          <div>
            <h2 style={{ color: '#81c784', fontSize: 16 }}>CDG Treatment Matrix</h2>
            <div style={{ fontSize: 12, color: '#ff8a65', marginBottom: 12 }}>
              ⚡ 4 genes are treatable: MPI (mannose), SLC35C1 (fucose), PGM1 (mannose+galactose), TMEM165 (manganese), ATP6AP1 (liver transplant) — early diagnosis critical
            </div>
            {defs?.treatment_matrix?.map((t, i) => (
              <div key={i} style={{ background: '#1e2e1e', border: '1px solid #4caf50', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                <div style={{ color: '#81c784', fontWeight: 700, fontSize: 14, marginBottom: 6 }}>{t.intervention}</div>
                {Object.entries(t).filter(([k]) => k !== 'intervention').map(([k, v]) => (
                  <div key={k} style={{ fontSize: 12, marginBottom: 3 }}>
                    <span style={{ color: '#ffb74d', fontWeight: 700, textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}: </span>
                    <span style={{ color: '#ddd' }}>{v}</span>
                  </div>
                ))}
              </div>
            ))}

            <h3 style={{ color: '#ffb74d', marginTop: 16 }}>Gene Profiles with Treatment Response</h3>
            {breakdown.gene_profiles?.map((gp, i) => (
              <div key={i} style={{ background: '#222', border: `1px solid ${GENE_COLORS[gp.gene] || '#555'}`, borderRadius: 6, padding: 12, marginBottom: 10 }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gp.gene] || '#fff', fontSize: 14, marginBottom: 6 }}>
                  {gp.gene} — {gp.locus} — {gp.aa}aa — {gp.inheritance}
                </div>
                <div style={{ fontSize: 12, color: '#ffb74d', marginBottom: 6 }}>IEF: {gp.ief_pattern}</div>
                <div style={{ fontSize: 12, color: '#80cbc4', marginBottom: 6 }}>Key Labs: {gp.key_labs}</div>
                <div style={{ fontSize: 12, color: '#a5d6a7', marginBottom: 4 }}>Pathognomonic: {gp.pathognomonic}</div>
                {gp.profile_extra && Object.entries(gp.profile_extra).map(([k, v]) => (
                  <div key={k} style={{ fontSize: 11, color: '#aaa', marginBottom: 2 }}>
                    <span style={{ color: '#ce93d8' }}>{k.replace(/_/g, ' ')}: </span>{typeof v === 'number' ? `${v}%` : v}
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}

        {/* ── DEFINITIONS ── */}
        {tab === 'Definitions' && defs && (
          <div>
            <h2 style={{ color: '#81c784', fontSize: 16 }}>CDG Definitions & Key Concepts</h2>

            <div style={{ background: '#0a2e0a', border: '1px solid #4caf50', borderRadius: 6, padding: 12, marginBottom: 14 }}>
              <strong style={{ color: '#81c784' }}>CDG Registry: </strong>
              <span style={{ color: '#a5d6a7', fontSize: 13 }}>{defs.cdg_registry}</span>
            </div>

            {defs.key_distinguishers?.map((kd, i) => (
              <div key={i} style={{ background: '#222', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                <div style={{ color: '#ffb74d', fontWeight: 700, marginBottom: 4, fontSize: 14 }}>{kd.feature}</div>
                <div style={{ fontSize: 12, color: '#888', marginBottom: 4 }}>Significance: {kd.significance}</div>
                {Object.entries(kd).filter(([k]) => !['feature','significance'].includes(k)).map(([k, v]) => (
                  <div key={k} style={{ fontSize: 12, marginBottom: 2 }}>
                    <span style={{ color: '#80cbc4', fontWeight: 700 }}>{k.replace(/_/g, ' ')}: </span>
                    <span style={{ color: '#ddd' }}>{v}</span>
                  </div>
                ))}
              </div>
            ))}

            <div style={{ fontSize: 12, color: '#666', marginTop: 16 }}>
              Seeds: {defs.seeds} · Atlas: {defs.atlas_id}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
