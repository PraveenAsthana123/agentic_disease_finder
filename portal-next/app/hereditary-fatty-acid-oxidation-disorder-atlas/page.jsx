'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-fatty-acid-oxidation-disorder-atlas';
const TABS = ['Overview', 'Gene Table', 'FAO Map', 'Treatment', 'Definitions'];

const GENE_COLORS = {
  'ACADVL':   '#1a237e',  // deep navy       — VLCAD; C14:1 PATHOGNOMONIC; cardiac+hepatic+myopathic 3 phenotypes
  'ACADM':    '#1b5e20',  // dark green       — MCAD; most common FAOD; C8; hypoketotic hypoglycemia; NBS hero
  'HADHA':    '#880e4f',  // deep crimson     — LCHAD/TFP; 3-OH-C16; retinopathy PATHOGNOMONIC; maternal AFLP
  'ACADS':    '#37474f',  // dark slate       — SCAD; C4; often benign; riboflavin; SCAD polymorphisms
  'CPT1A':    '#4a148c',  // deep purple      — CPT1A; C0 HIGH (opposite!); Arctic founder p.P479L
  'CPT2':     '#e65100',  // burnt orange     — CPT2; C16/C18; myopathic adolescent rhabdomyolysis; p.S113L
  'SLC25A20': '#b71c1c',  // dark red         — CACT; severe neonatal arrhythmias; C16/C18 elevated
  'HADH':     '#004d40',  // dark teal        — SCHAD; protein-sensitive hyperinsulinism; GDH disinhibition; diazoxide
};

const GENE_INFO = {
  'ACADVL':   { full: 'ACADVL-655aa / AR / Very-Long-Chain-Acyl-CoA-Dehydrogenase / Long-chain-FA-oxidation-first-step / 17p13.1 / VLCAD-Deficiency / C14:1-ELEVATED-PATHOGNOMONIC / THREE-phenotypes-cardiac+hepatic+myopathic / NBS-DETECTABLE / Avoid-fasting', locus: '17p13.1', size: '655 aa / 70 kDa', inh: 'AR; outer mitochondrial membrane enzyme; C14:1 (tetradecenoylcarnitine) PATHOGNOMONIC marker; 3 phenotypes by residual activity: severe cardiac (neonatal), hepatic (childhood), myopathic (adult); NBS-detected; retinopathy ABSENT (DDx LCHAD)' },
  'ACADM':    { full: 'ACADM-421aa / AR / Medium-Chain-Acyl-CoA-Dehydrogenase / C4-C12-oxidation / 1p31.1 / MCAD-Deficiency / MOST-COMMON-FAOD-1:10000-15000 / C8-ELEVATED-PATHOGNOMONIC / Hypoketotic-Hypoglycemia / NBS-DETECTED / p.K329E-founder-80pct-N.European', locus: '1p31.1', size: '421 aa / 45 kDa', inh: 'AR; most common FAOD; C8 (octanoylcarnitine) MARKEDLY elevated PATHOGNOMONIC; hypoketotic hypoglycemia = NO ketones during hypoglycemia (pathognomonic); p.K329E >80% N.European alleles; NBS has virtually eliminated mortality; avoid fasting' },
  'HADHA':    { full: 'HADHA-763aa / AR / Mitochondrial-Trifunctional-Protein-Alpha-Subunit / LCHAD+hydratase+thiolase / 2p23.3 / LCHAD-TFP-Deficiency / 3-OH-C16-ELEVATED-PATHOGNOMONIC / RETINOPATHY-PATHOGNOMONIC / Maternal-AFLP-20-25pct / Peripheral-neuropathy / DHA-supplementation', locus: '2p23.3', size: '763 aa / 83 kDa', inh: 'AR; alpha subunit of mitochondrial trifunctional protein (MTP); 3-OH-C16 PATHOGNOMONIC; retinopathy (pigmentary) PATHOGNOMONIC — unique among FAODs; peripheral neuropathy; rhabdomyolysis; MATERNAL AFLP/HELLP in 20-25% of mothers carrying LCHAD-affected fetus; DHA supplementation mandatory' },
  'ACADS':    { full: 'ACADS-412aa / AR / Short-Chain-Acyl-CoA-Dehydrogenase / C4-C6-butyryl-CoA-oxidation / 12q24.31 / SCAD-Deficiency / C4-butyrylcarnitine-ELEVATED / EthylmalonicAcid-urine / Often-BENIGN-most-NBS-detected-asymptomatic / Riboflavin-cofactor', locus: '12q24.31', size: '412 aa / 44 kDa', inh: 'AR; C4 (butyrylcarnitine) elevated; ethylmalonic acid in urine; OFTEN BENIGN — most NBS-detected patients asymptomatic; common SCAD polymorphisms p.511C>T/p.625G>A reduce activity but not disease-causing alone; riboflavin trial; avoid over-treatment' },
  'CPT1A':    { full: 'CPT1A-773aa / AR / Carnitine-Palmitoyltransferase-1A-Liver / Rate-limiting-carnitine-shuttle-step1 / 11q13.3 / CPT1A-Deficiency / C0-VERY-HIGH-LC-AC-LOW-PATHOGNOMONIC-OPPOSITE / Arctic-founder-p.P479L-up-to-70pct-carrier-Inuit / Hypoketotic-Hypoglycemia / Hepatomegaly', locus: '11q13.3', size: '773 aa / 88 kDa', inh: 'AR; C0 VERY HIGH + long-chain AC LOW = OPPOSITE pattern vs all other long-chain FAODs; p.P479L Arctic founder (Inuit/First Nations carrier up to 70%); hypoketotic hypoglycemia + hepatomegaly; renal tubular acidosis occasional; NBS C0 elevation in Arctic populations often p.P479L' },
  'CPT2':     { full: 'CPT2-658aa / AR / Carnitine-Palmitoyltransferase-2 / Carnitine-shuttle-step2-inner-membrane / 1p32.3 / CPT2-Deficiency-MOST-COMMON-CPT / Myopathic-Adolescent-Rhabdomyolysis / p.S113L-founder-60-70pct-myopathic / C16-C18-elevated-C0-low / Bezafibrate-experimental', locus: '1p32.3', size: '658 aa / 74 kDa', inh: 'AR; 3 forms: lethal neonatal (rare), severe infantile (rare), myopathic (>75%); myopathic: adolescent rhabdomyolysis triggered by exercise/cold/fasting; p.S113L founder 60-70% myopathic alleles; C16/C18/C18:1 elevated + C0 low; bezafibrate experimental (increases residual activity)' },
  'SLC25A20': { full: 'SLC25A20-301aa / AR / Carnitine-Acylcarnitine-Translocase-CACT / Inner-membrane-exchanger-free-carnitine-out-acylcarnitine-in / 3p21.31 / CACT-Deficiency / SEVERE-NEONATAL-Cardiac-ARRHYTHMIAS-primary-cause-death / C16-C18-HIGH-C0-VERY-LOW / Hyperammonemia-secondary', locus: '3p21.31', size: '301 aa / 33 kDa', inh: 'AR; severe neonatal onset within 24-48h; cardiac ARRHYTHMIAS primary cause of death (not cardiomyopathy); C16/C18/C18:1 very elevated + C0 very low; hyperammonemia secondary (liver); molecular testing required to distinguish from CPT2 neonatal (similar acylcarnitine profile)' },
  'HADH':     { full: 'HADH-314aa / AR / Short-Chain-3-Hydroxyacyl-CoA-Dehydrogenase-SCHAD / Beta-oxidation-short-chain-step3 / 4q25 / SCHAD-Deficiency / PROTEIN-SENSITIVE-HYPERINSULINISM-PATHOGNOMONIC / ONLY-FAOD-causing-hyperinsulinism / GDH-disinhibition-mechanism / NORMAL-AMMONIA-DDx-GLUD1 / C4-OH-3-hydroxybutyrylcarnitine / Diazoxide-RESPONSIVE', locus: '4q25', size: '314 aa / 35 kDa', inh: 'AR; UNIQUE: ONLY FAOD causing hyperinsulinism; HADH normally inhibits GDH in beta cells; HADH LOF -> GDH uninhibited -> leucine-stimulated excess insulin -> PROTEIN-SENSITIVE hypoglycemia; NORMAL AMMONIA distinguishes from GLUD1-HI (hyperammonemia-HI); C4-OH elevated; diazoxide RESPONSIVE' },
};

const PHENOTYPE_ROWS = [
  { gene: 'ACADVL',   group: 'Long-chain FAO — VLCAD Deficiency',       acylcarnitine: 'C14:1 ↑',      nbs: '✓', neonatal: '~', pathognomonic: 'C14:1 elevated; 3 phenotypes cardiac/hepatic/myopathic', rx: 'Avoid fasting; MCT limited; echo',         unique: 'THREE phenotypes: severe cardiac (neonatal, <10% activity), hepatic (childhood), myopathic (adult, rhabdomyolysis); C14:1 most specific marker; retinopathy ABSENT (DDx LCHAD); echo mandatory in cardiac form' },
  { gene: 'ACADM',    group: 'Medium-chain FAO — MCAD Deficiency',      acylcarnitine: 'C8 ↑↑',        nbs: '✓', neonatal: '~', pathognomonic: 'C8 dominant; hypoketotic hypoglycemia; NBS success', rx: 'Avoid fasting; 10% dextrose IV illness',   unique: 'MOST COMMON FAOD; NO KETONES during hypoglycemia PATHOGNOMONIC; p.K329E >80% N.European; NBS transformed outcome; no fat restriction needed; carnitine controversial' },
  { gene: 'HADHA',    group: 'Long-chain FAO — LCHAD/TFP Deficiency',   acylcarnitine: '3-OH-C16 ↑',   nbs: '✓', neonatal: '~', pathognomonic: 'Retinopathy; maternal AFLP; 3-OH-C16', rx: 'Low-LC fat + MCT; DHA; ophthalmology',    unique: 'RETINOPATHY PATHOGNOMONIC (unique among FAODs); peripheral neuropathy; rhabdomyolysis; MATERNAL AFLP/HELLP 20-25% (retrograde diagnosis); DHA reduces retinal degeneration; low-long-chain fat + MCT' },
  { gene: 'ACADS',    group: 'Short-chain FAO — SCAD Deficiency',       acylcarnitine: 'C4 ↑',         nbs: '✓', neonatal: '✗', pathognomonic: 'C4 elevated; EMA urine; often benign',   rx: 'Riboflavin; avoid over-treatment',       unique: 'OFTEN CLINICALLY BENIGN — most asymptomatic; common polymorphisms p.511C>T/p.625G>A (NOT disease-causing alone); DDx MADD (multiple acylcarnitines); riboflavin trial; do NOT restrict fat in asymptomatic' },
  { gene: 'CPT1A',    group: 'Carnitine Shuttle — CPT1A Deficiency',    acylcarnitine: 'C0 ↑↑ (OPPOSITE)', nbs: '~', neonatal: '~', pathognomonic: 'C0 VERY HIGH + LC-AC LOW (opposite pattern)',rx: 'Avoid fasting; low-LC fat; cornstarch',  unique: 'C0 HIGH + LC-AC LOW = OPPOSITE of other long-chain FAODs; p.P479L Arctic founder (Inuit carrier up to 70%); adaptive variant — cultural counseling required; hypoketotic hypoglycemia + hepatomegaly' },
  { gene: 'CPT2',     group: 'Carnitine Shuttle — CPT2 Deficiency',     acylcarnitine: 'C16/C18 ↑',    nbs: '~', neonatal: '~', pathognomonic: 'Myopathic adolescent rhabdomyolysis; p.S113L', rx: 'Avoid exercise/cold; high-carb diet',    unique: 'MYOPATHIC form (>75%): adolescent/adult exercise + cold + fasting -> rhabdomyolysis + myoglobinuria + renal failure; p.S113L founder; bezafibrate experimental; IV hydration for acute rhabdo' },
  { gene: 'SLC25A20', group: 'Carnitine Shuttle — CACT Deficiency',     acylcarnitine: 'C16/C18 ↑↑',   nbs: '✓', neonatal: '✓', pathognomonic: 'Neonatal cardiac arrhythmias; C16/C18 high', rx: 'IV glucose + antiarrhythmic; carnitine IV',unique: 'SEVERE NEONATAL; cardiac ARRHYTHMIAS primary cause of death (not cardiomyopathy like VLCAD); hyperammonemia secondary; molecular testing distinguishes from CPT2 neonatal; poor prognosis even with treatment' },
  { gene: 'HADH',     group: 'Short-chain FAO — SCHAD/Hyperinsulinism', acylcarnitine: 'C4-OH ↑',       nbs: '~', neonatal: '~', pathognomonic: 'Protein-sensitive HI; NORMAL ammonia; GDH disinhibition', rx: 'Diazoxide; restrict protein excess',    unique: 'ONLY FAOD with hyperinsulinism; HADH inhibits GDH in beta cells; LOF -> GDH active -> leucine-stimulated insulin excess; NORMAL AMMONIA (DDx GLUD1-HI has NH3 elevated); diazoxide RESPONSIVE; protein-sensitive not fasting-sensitive' },
];

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info, geneData, cohortData }) {
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 12, marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          <span style={{ background: color, color: '#fff', fontWeight: 700, padding: '3px 10px', borderRadius: 5, fontSize: 15 }}>{gene}</span>
          <span style={{ marginLeft: 10, fontSize: 12, color: '#555' }}>{geneData?.disease ?? info?.full?.split('/')[2] ?? ''}</span>
        </div>
        <span style={{ fontSize: 11, color: '#888' }}>seed {SEED_BASE_DISPLAY[gene] ?? '—'} · n=40</span>
      </div>
      <div style={{ marginTop: 8, fontSize: 12, color: '#333' }}>{info?.full}</div>
      <div style={{ marginTop: 6, fontSize: 12 }}>
        <strong>Locus:</strong> {info?.locus} &nbsp;|&nbsp;
        <strong>Size:</strong> {info?.size} &nbsp;|&nbsp;
        <strong>Inheritance:</strong> {info?.inh?.split(';')[0]}
      </div>
      {geneData && (
        <div style={{ marginTop: 6, fontSize: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Badge text={`Group: ${geneData.faod_group ?? '—'}`} color={color} />
          <Badge text={`Key AC: ${geneData.key_acylcarnitine ?? '—'}`} color="#1565c0" />
          <Badge text={`NBS: ${geneData.nbs_detected ? '✓ detected' : '✗ varies'}`} color={geneData.nbs_detected ? '#1565c0' : '#757575'} />
          {geneData.has_retinopathy    && <Badge text="RETINOPATHY UNIQUE" color="#880e4f" />}
          {geneData.has_rhabdomyolysis && <Badge text="Rhabdomyolysis risk" color="#e65100" />}
          {geneData.has_cardiomyopathy && <Badge text="Cardiomyopathy" color="#b71c1c" />}
          {geneData.has_hyperinsulinism&& <Badge text="HYPERINSULINISM UNIQUE" color="#004d40" />}
          {geneData.is_often_benign    && <Badge text="Often benign" color="#388e3c" />}
          {geneData.arctic_founder     && <Badge text="Arctic founder p.P479L" color="#4a148c" />}
          {cohortData && <Badge text={`Mean glucose: ${cohortData.mean_glucose} mg/dL`} color={cohortData.mean_glucose < 40 ? '#b71c1c' : '#37474f'} />}
          {cohortData && <Badge text={`Severe: ${Math.round((cohortData.pct_severe ?? 0) * 100)}%`} color={cohortData.pct_severe > 0.4 ? '#b71c1c' : '#388e3c'} />}
          {geneData.pathognomonic && (
            <div style={{ width: '100%', marginTop: 4, background: '#f5f5f5', padding: '4px 8px', borderRadius: 4, fontSize: 11, color: '#333' }}>
              <strong>⚡ Pathognomonic:</strong> {geneData.pathognomonic?.split(';')[0]}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Seed display map
const SEED_BASE_DISPLAY = {
  'ACADVL': 4222, 'ACADM': 4223, 'HADHA': 4224, 'ACADS': 4225,
  'CPT1A': 4226, 'CPT2': 4227, 'SLC25A20': 4228, 'HADH': 4229,
};

export default function HeredFAODAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 32, color: '#aaa' }}>Loading Hereditary FAOD Atlas…</div>;
  if (error)   return <div style={{ padding: 32, color: '#f44' }}>Error: {error}</div>;

  const geneMap = {};
  (overview?.gene_table ?? []).forEach(g => { geneMap[g.gene] = g; });

  return (
    <div style={{ fontFamily: 'monospace', background: '#0a0f14', color: '#e0e0e0', minHeight: '100vh', padding: 24 }}>
      {/* Header */}
      <div style={{ marginBottom: 18 }}>
        <h1 style={{ fontSize: 22, color: '#64b5f6', margin: 0, fontWeight: 800 }}>
          Hereditary Fatty Acid Oxidation Disorder Atlas
        </h1>
        <div style={{ fontSize: 12, color: '#888', marginTop: 4 }}>
          8-Gene Reference · ACADVL · ACADM · HADHA · ACADS · CPT1A · CPT2 · SLC25A20 · HADH
          &nbsp;·&nbsp; seeds {overview?.seeds} &nbsp;·&nbsp; {overview?.total_patients} patients
        </div>
        <div style={{ fontSize: 11, color: '#aaa', marginTop: 4 }}>
          Acylcarnitine-profile-driven diagnosis · Fasting-avoidance primary management · MCAD most common FAOD
          · LCHAD retinopathy pathognomonic · HADH hyperinsulinism unique
        </div>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 18, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 700,
            background: tab === t ? '#1a237e' : '#1a1f2b', color: tab === t ? '#90caf9' : '#888',
          }}>{t}</button>
        ))}
      </div>

      {/* === OVERVIEW TAB === */}
      {tab === 'Overview' && (
        <div>
          {/* Cascade decision tree */}
          <div style={{ background: '#111827', border: '1px solid #1a237e', borderRadius: 8, padding: 14, marginBottom: 18 }}>
            <div style={{ fontWeight: 700, fontSize: 13, color: '#90caf9', marginBottom: 8 }}>⚡ Primary Cascade: Acylcarnitine Profile Algorithm</div>
            {(overview?.primary_cascade_split ?? '').split('. ').map((s, i) => s.trim() && (
              <div key={i} style={{ marginBottom: 4, paddingLeft: 12, borderLeft: '3px solid #1a237e', fontSize: 12 }}>
                <span style={{ color: '#90caf9', fontWeight: 700 }}>{s.match(/STEP \d+:/)?.[0] ?? '▸'}</span>
                <span style={{ color: '#ccc' }}> {s.replace(/STEP \d+: ?/, '')}</span>
              </div>
            ))}
          </div>

          {/* FAO group summary badges */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 10, marginBottom: 18 }}>
            {[
              { label: 'Long-chain FAO', genes: overview?.long_chain_group ?? [], color: '#1a237e' },
              { label: 'Medium-chain FAO', genes: overview?.medium_chain_group ?? [], color: '#1b5e20' },
              { label: 'Short-chain FAO', genes: overview?.short_chain_group ?? [], color: '#37474f' },
              { label: 'Carnitine Shuttle', genes: overview?.carnitine_shuttle ?? [], color: '#4a148c' },
            ].map(grp => (
              <div key={grp.label} style={{ background: '#1a1f2b', border: `1px solid ${grp.color}`, borderRadius: 6, padding: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: '#90caf9', marginBottom: 4 }}>{grp.label}</div>
                {grp.genes.map(g => (
                  <Badge key={g} text={g} color={GENE_COLORS[g] ?? grp.color} />
                ))}
              </div>
            ))}
          </div>

          {/* Gene cards */}
          {Object.entries(GENE_COLORS).map(([gene, color]) => (
            <GeneCard key={gene} gene={gene} color={color} info={GENE_INFO[gene]}
              geneData={geneMap[gene]}
              cohortData={(overview?.gene_table ?? []).find(g => g.gene === gene)} />
          ))}
        </div>
      )}

      {/* === GENE TABLE TAB === */}
      {tab === 'Gene Table' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#90caf9' }}>
                {['Gene','Disease','Locus','aa','kDa','Inh','Group','Key AC','NBS','Retinopathy','Rhabdo','HI','Benign','Pathognomonic','OMIM'].map(h => (
                  <th key={h} style={{ padding: '6px 8px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(overview?.gene_table ?? []).map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#111' : '#161b22', borderBottom: '1px solid #1a1f2b' }}>
                  <td style={{ padding: '5px 8px', fontWeight: 700, color: GENE_COLORS[g.gene] ?? '#90caf9' }}>{g.gene}</td>
                  <td style={{ padding: '5px 8px', fontSize: 10 }}>{g.disease}</td>
                  <td style={{ padding: '5px 8px' }}>{g.locus}</td>
                  <td style={{ padding: '5px 8px' }}>{g.aa}</td>
                  <td style={{ padding: '5px 8px' }}>{g.kda}</td>
                  <td style={{ padding: '5px 8px' }}>{g.inheritance}</td>
                  <td style={{ padding: '5px 8px', fontSize: 10 }}>{g.faod_group}</td>
                  <td style={{ padding: '5px 8px', color: '#90caf9' }}>{g.key_acylcarnitine}</td>
                  <td style={{ padding: '5px 8px', color: g.nbs_detected ? '#4caf50' : '#757575' }}>{g.nbs_detected ? '✓' : '—'}</td>
                  <td style={{ padding: '5px 8px', color: g.has_retinopathy ? '#e91e63' : '#555' }}>{g.has_retinopathy ? '✓ UNIQUE' : '—'}</td>
                  <td style={{ padding: '5px 8px', color: g.has_rhabdomyolysis ? '#ff9800' : '#555' }}>{g.has_rhabdomyolysis ? '✓' : '—'}</td>
                  <td style={{ padding: '5px 8px', color: g.has_hyperinsulinism ? '#00bcd4' : '#555' }}>{g.has_hyperinsulinism ? '✓ UNIQUE' : '—'}</td>
                  <td style={{ padding: '5px 8px', color: g.is_often_benign ? '#4caf50' : '#555' }}>{g.is_often_benign ? 'Often' : '—'}</td>
                  <td style={{ padding: '5px 8px', fontSize: 10, maxWidth: 200 }}>{g.pathognomonic?.substring(0, 80)}…</td>
                  <td style={{ padding: '5px 8px', color: '#64b5f6' }}>{g.omim_disease}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* === FAO MAP TAB === */}
      {tab === 'FAO Map' && (
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, color: '#90caf9', marginBottom: 12 }}>
            Acylcarnitine Profile → Gene Identification Algorithm
          </div>
          {Object.entries(breakdown?.acylcarnitine_algorithm ?? {}).map(([key, val]) => (
            <div key={key} style={{ background: '#111827', border: `2px solid ${GENE_COLORS[val.gene] ?? '#333'}`, borderRadius: 8, padding: 12, marginBottom: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap' }}>
                <span style={{ background: GENE_COLORS[val.gene] ?? '#333', color: '#fff', fontWeight: 700, padding: '2px 10px', borderRadius: 4, fontSize: 13 }}>
                  {val.gene}
                </span>
                <span style={{ color: '#90caf9', fontWeight: 700, fontSize: 13 }}>{key}</span>
              </div>
              <div style={{ marginTop: 8, fontSize: 12, color: '#ccc' }}>{val.action}</div>
            </div>
          ))}

          {/* FAO group details */}
          <div style={{ marginTop: 20, fontWeight: 700, fontSize: 14, color: '#90caf9', marginBottom: 12 }}>
            Functional FAO Groups
          </div>
          {Object.entries(breakdown?.faod_groups ?? {}).map(([grpId, grp]) => (
            <div key={grpId} style={{ background: '#1a1f2b', border: '1px solid #1a237e', borderRadius: 8, padding: 12, marginBottom: 10 }}>
              <div style={{ fontWeight: 700, color: '#64b5f6', textTransform: 'uppercase', fontSize: 12, marginBottom: 6 }}>{grpId}</div>
              <div style={{ fontSize: 12, color: '#ccc', marginBottom: 6 }}>{grp.description}</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {(grp.genes ?? []).map(g => <Badge key={g} text={g} color={GENE_COLORS[g] ?? '#333'} />)}
              </div>
              {grp.key_acylcarnitines && (
                <div style={{ marginTop: 6, fontSize: 11, color: '#90caf9' }}>
                  Key markers: {Object.entries(grp.key_acylcarnitines).map(([g, ac]) => `${g}→${ac}`).join(', ')}
                </div>
              )}
              {grp.c0_pattern && (
                <div style={{ marginTop: 6, fontSize: 11, color: '#ff9800' }}>{grp.c0_pattern}</div>
              )}
            </div>
          ))}

          {/* Distinguishing features table */}
          <div style={{ marginTop: 20 }}>
            <div style={{ fontWeight: 700, fontSize: 14, color: '#90caf9', marginBottom: 10 }}>Distinguishing Features by Gene</div>
            {(breakdown?.distinguishing_features ?? []).map(row => (
              <div key={row.gene} style={{ display: 'flex', gap: 10, padding: '8px 0', borderBottom: '1px solid #1a1f2b', flexWrap: 'wrap' }}>
                <span style={{ background: GENE_COLORS[row.gene] ?? '#333', color: '#fff', fontWeight: 700,
                  padding: '2px 8px', borderRadius: 4, fontSize: 12, minWidth: 80, textAlign: 'center' }}>{row.gene}</span>
                <div style={{ flex: 1, fontSize: 12 }}>
                  <div style={{ color: '#90caf9' }}>{row.feature}</div>
                  <div style={{ color: '#aaa', marginTop: 2 }}>→ {row.action}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* === TREATMENT TAB === */}
      {tab === 'Treatment' && (
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, color: '#90caf9', marginBottom: 12 }}>FAOD Treatment Matrix</div>

          {/* Critical rules */}
          <div style={{ background: '#1a0000', border: '2px solid #b71c1c', borderRadius: 8, padding: 14, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, color: '#ef5350', fontSize: 13, marginBottom: 8 }}>⚠ Universal FAOD Principles</div>
            <div style={{ fontSize: 12, color: '#ffcdd2' }}>
              <div>• <strong>AVOID FASTING</strong> — primary management for ALL FAODs (except HADH protein restriction)</div>
              <div>• <strong>EMERGENCY CARD</strong> — every FAOD patient carries IV dextrose 10% emergency protocol for anesthesia/illness</div>
              <div>• <strong>NEVER NPO for surgery</strong> without IV glucose running before anesthesia induction</div>
              <div>• <strong>MCAD: no fat restriction</strong> — medium-chain fat intake is NOT the problem (oxidation is); restricting fat worsens nutrition</div>
              <div>• <strong>LCHAD: ophthalmology annually</strong> — retinopathy is silent until advanced; DHA supplementation from diagnosis</div>
              <div>• <strong>CPT2 myopathic: avoid exercise triggers</strong> — not primarily a fasting issue; cold + prolonged exercise = rhabdomyolysis</div>
              <div>• <strong>HADH: restrict protein, not fat</strong> — protein (leucine) triggers hyperinsulinism; fasting is NOT the mechanism</div>
            </div>
          </div>

          {(definitions?.treatment_matrix ?? []).map((tx, i) => (
            <div key={i} style={{ background: '#111827', border: '1px solid #1a237e', borderRadius: 8, padding: 14, marginBottom: 12 }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#64b5f6', marginBottom: 6 }}>{tx.intervention}</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
                <div><span style={{ color: '#90caf9' }}>Indications: </span>{tx.indications}</div>
                <div><span style={{ color: '#90caf9' }}>Mechanism: </span>{tx.mechanism}</div>
                <div><span style={{ color: '#90caf9' }}>Dose: </span>{tx.dose}</div>
                <div><span style={{ color: tx.contraindicated?.toLowerCase().includes('not') || tx.contraindicated?.toLowerCase().includes('contro') ? '#ef5350' : '#aaa' }}>
                  CI/Caution: </span>{tx.contraindicated}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* === DEFINITIONS TAB === */}
      {tab === 'Definitions' && (
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, color: '#90caf9', marginBottom: 12 }}>Key Clinical Distinctions — FAOD Atlas</div>
          {(definitions?.key_distinguishers ?? []).map((d, i) => (
            <div key={i} style={{ background: '#111827', border: '1px solid #1a237e', borderRadius: 8, padding: 14, marginBottom: 12 }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#64b5f6', marginBottom: 8 }}>{d.name}</div>
              <div style={{ fontSize: 12, color: '#ccc', lineHeight: 1.7 }}>{d.description}</div>
            </div>
          ))}
        </div>
      )}

      <div style={{ marginTop: 24, fontSize: 10, color: '#555' }}>
        seeds {overview?.seeds} · {overview?.total_patients} synthetic patients · 8x40 · {ATLAS_NAME} · academic reference only
      </div>
    </div>
  );
}

const ATLAS_NAME = 'Hereditary-Fatty-Acid-Oxidation-Disorder-Atlas';
