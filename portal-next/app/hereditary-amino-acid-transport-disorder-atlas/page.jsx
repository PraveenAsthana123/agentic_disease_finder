'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-amino-acid-transport-disorder-atlas';
const TABS = ['Overview', 'Gene Table', 'Urine AA Cascade', 'Treatment', 'Definitions'];

const GENE_COLORS = {
  'SLC6A19':  '#1a237e',  // deep navy        — Hartnup; neutral aminoaciduria; pellagra; nicotinamide CURATIVE
  'SLC3A1':   '#b71c1c',  // deep crimson      — Cystinuria type A; rBAT; cystine stones; tiopronin
  'SLC7A9':   '#880e4f',  // deep magenta      — Cystinuria type B; b0+AT; heterozygote intermediate
  'SLC7A7':   '#e65100',  // burnt orange      — LPI; alveolar proteinosis PATHOGNOMONIC; citrulline
  'SLC17A5':  '#1b5e20',  // deep forest       — Salla/ISSD; sialin; Finnish founder; free sialic acid
  'SLC6A8':   '#4a148c',  // deep purple       — Creatine transporter; XLR; absent brain creatine MRS
  'SLC6A5':   '#37474f',  // dark slate        — GlyT2; hyperekplexia; non-habituating startle; clonazepam
  'SLC25A22': '#004d40',  // dark teal         — GC1; neonatal myoclonic EE; burst-suppression; KD
};

const GENE_INFO = {
  'SLC6A19':  { full: 'SLC6A19-634aa / AR / B0AT1-NeutralAATransporter / 5p15.33 / Hartnup-Disease / PellagraLike-PhotosensitiveRash-PATHOGNOMONIC / IntermittentCerebellarAtaxia / NeutralAminoaciduria-UrineChromatography / NicotinamideSupplementCURATIVE / IncompletePenetrance-ManyAsymptomatic / HighProteinDiet', locus: '5p15.33', size: '634 aa / 72 kDa', inh: 'AR; Hartnup disease; B0AT1 neutral amino acid transporter (intestine + kidney); pellagra-like photosensitive rash + intermittent ataxia; URINE NEUTRAL AMINOACIDURIA (all monoamino-monocarboxylic AAs elevated) PATHOGNOMONIC; nicotinamide 50-100mg/day CURATIVE; incomplete penetrance — dietary tryptophan deficiency needed to trigger symptoms' },
  'SLC3A1':   { full: 'SLC3A1-685aa / AR / rBAT-HeavyChain / 2p21 / CystinuriaTypeA / CystineUrolithiasis-PATHOGNOMONIC / DibasicAminoaciduria-Cys+Lys+Arg+Orn / CyanideNitroprussidePositive / TypeA-Heterozygotes-ASYMPTOMATIC / Tiopronin-THIOLA-preferred / KCitrate-Alkalinization / StaghornCalculi', locus: '2p21', size: '685 aa / 79 kDa', inh: 'AR (pure); rBAT heavy chain of b0,+ transporter; cystinuria type A; cystine urolithiasis PATHOGNOMONIC; dibasic AA elevated in urine; cyanide-nitroprusside test positive; TYPE A HETEROZYGOTES COMPLETELY ASYMPTOMATIC; tiopronin (THIOLA) preferred chelation; hydration 3L+ + potassium citrate pH >7.5' },
  'SLC7A9':   { full: 'SLC7A9-487aa / AR / b0+AT-LightChain / 19q13.11 / CystinuriaTypeB / IdenticalPhenotype-TypeA / KEY-TypeB-Heterozygotes-MAY-Have-IntermediateAminoaciduria-SomeStones / MediterraneanFounder-p.A182T / SameTreatment-TypeA / CarrierCounseling-Different', locus: '19q13.11', size: '487 aa / 55 kDa', inh: 'AR; b0,+AT light chain (catalytic subunit); cystinuria type B; IDENTICAL phenotype to type A; KEY DISTINCTION: SLC7A9 heterozygotes may show INTERMEDIATE aminoaciduria and 10-20% develop stones (incomplete dominance) — carrier urological follow-up may be needed; Mediterranean founder p.A182T' },
  'SLC7A7':   { full: 'SLC7A7-511aa / AR / y+LAT1-DibasicEfflux / 14q11.2 / LysinuricProteinIntolerance-LPI / AlveolarProteinosis-PAP-60-70pct-PATHOGNOMONIC / PostProteinHyperammonemia / DibasicAAHighUrineLowPlasma-InversePATTERN / OroticAcidElevated / HLH-20pct-Risk / Citrulline-FIRSTLINE / HCQ-PAP / Osteoporosis-LysineDeficiency', locus: '14q11.2', size: '511 aa / 57 kDa', inh: 'AR; y+LAT1 dibasic AA efflux transporter; LPI; ALVEOLAR PROTEINOSIS (PAP) 60-70% PATHOGNOMONIC (lung in amino acid disorder!); post-protein hyperammonemia; urine dibasic AA HIGH, plasma paradoxically LOW; orotic acid elevated; HLH 20% risk; L-citrulline supplement first-line; hydroxychloroquine for PAP; osteoporosis from lysine deficiency' },
  'SLC17A5':  { full: 'SLC17A5-495aa / AR / Sialin-LysosomalSialicAcidTransporter / 6q13 / SallaDisease+ISSD-SameGene / UrineFreeSialicAcid-100-1000x-ELEVATED-PATHOGNOMONIC / FinnishFounder-p.R39C-Salla-Mild / SevereMutations-ISSD-Lethal-Infancy / WMDysmyelinationMRI / p.R39C-Homozygous-Salla-IQ30-50-AdultSurvival / NoApprovedTherapy', locus: '6q13', size: '495 aa / 54 kDa', inh: 'AR; sialin lysosomal sialic acid exporter; Salla disease (mild, Finnish founder p.R39C) or ISSD (severe, lethal infancy); URINE FREE SIALIC ACID 100-1000x elevated PATHOGNOMONIC; Finnish p.R39C = Salla (IQ 30-50, ambulatory adults); severe mutations = ISSD (hydrops, death infancy); white matter dysmyelination; no disease-modifying therapy' },
  'SLC6A8':   { full: 'SLC6A8-636aa / XLR / CrT1-BloodBrainBarrierCreatineTransporter / Xq28 / CreatineTransporterDeficiency-CTD / AbsentBrainCreatine-3.03ppm-1H-MRS-PATHOGNOMONIC / UrineCr-CrRatioElevated / XLR-MalesSeverelyAffected / CreatineSupplementDOESNOTCrossBBB-CriticalTrap / SpeechDelayDispropritionate / BehavioralASD / GenetherapyTrials', locus: 'Xq28', size: '636 aa / 70 kDa', inh: 'XLR; CrT1 brain creatine transporter; CTD; ABSENT BRAIN CREATINE AT 3.03ppm ON 1H-MRS PATHOGNOMONIC; X-linked — males severely affected (ID/epilepsy/speech delay); CRITICAL: creatine supplement DOES NOT improve brain MRS (cannot cross BBB); urine creatine/creatinine ratio elevated; gene therapy trials ongoing' },
  'SLC6A5':   { full: 'SLC6A5-799aa / AR / GlyT2-PresynapticGlycineTransporter / 11p15.1 / HypereklexiaType3 / NON-HabituatingStartleReflexPATHOGNOMONIC / NeonatalHypertonia / NoseTapReflex / ToniApnea-LIFETHREATENING / EEG-NORMAL-NotEpilepsy / Clonazepam-HIGHLY-EFFECTIVE-FirstLine / HeimlichManeuver-LIFE-SAVING-Parents-Must-Learn / GLRA1-GLRB-DDx', locus: '11p15.1', size: '799 aa / 87 kDa', inh: 'AR; GlyT2 presynaptic glycine reuptake transporter; hyperekplexia type 3; NON-HABITUATING EXAGGERATED STARTLE PATHOGNOMONIC (normal startle habituates after 3-5; hyperekplexia does NOT); neonatal tonic apnea LIFE-THREATENING; EEG NORMAL (not epilepsy); clonazepam first-line HIGHLY EFFECTIVE; HEIMLICH MANEUVER for apnea LIFE-SAVING — parents must demonstrate before discharge' },
  'SLC25A22': { full: 'SLC25A22-323aa / AR / GC1-MitochondrialGlutamateCarrier1 / 11p15.5 / EarlyMyoclonicEncephalopathy / NeonatalMyoclonus-Day0-3ONSET / BurstSuppression-then-Hypsarrhythmia-PATHOGNOMONIC-TRANSITION / LargelyRefractory-AEDs / KetogenicDiet-PartialResponse-30-50pct / PyridoxineMandatoryExcludePDE / OrganicAcidsNormal-DDx-NKH / ProfoundID-NoIndependentFunction', locus: '11p15.5', size: '323 aa / 33 kDa', inh: 'AR; GC1 mitochondrial glutamate carrier; neonatal epileptic encephalopathy; NEONATAL MYOCLONUS (day 0-3) + BURST-SUPPRESSION → HYPSARRHYTHMIA TRANSITION PATHOGNOMONIC; largely AED-refractory; ketogenic diet partial response 30-50%; pyridoxine 100mg IV MANDATORY (exclude PDE); organic acids normal (DDx from organic acidemias); profound ID, no independent function' },
};

const PHENOTYPE_ROWS = [
  { gene: 'SLC6A19',  group: 'Intestinal/Renal — Neutral AA',  urine_aa: 'Neutral ↑',       plasma_aa: 'Low neutral AAs',    treatable: '✓', pathognomonic: 'Photosensitive rash + intermittent ataxia + urine neutral aminoaciduria', rx: 'Nicotinamide 50-100mg/day CURATIVE; high-protein diet' },
  { gene: 'SLC3A1',   group: 'Renal — Dibasic AA (TypeA)',     urine_aa: 'Dibasic ↑ (Cys+)', plasma_aa: 'Normal',              treatable: '✓', pathognomonic: 'Cystine urolithiasis; cyanide-nitroprusside+; TypeA het ASYMPTOMATIC', rx: 'Tiopronin + hydration 3L+ + KCitrate pH>7.5' },
  { gene: 'SLC7A9',   group: 'Renal — Dibasic AA (TypeB)',     urine_aa: 'Dibasic ↑ (Cys+)', plasma_aa: 'Normal',              treatable: '✓', pathognomonic: 'Same as TypeA; TypeB het may have intermediate aminoaciduria + stones', rx: 'Same as SLC3A1; carrier urological surveillance' },
  { gene: 'SLC7A7',   group: 'Intestinal — Dibasic Efflux',   urine_aa: 'Dibasic ↑',        plasma_aa: 'Dibasic LOW (INVERSE)', treatable: '✓', pathognomonic: 'Alveolar proteinosis 60-70% PATHOGNOMONIC; post-protein NH3↑; HLH risk', rx: 'L-citrulline first-line; low protein; HCQ for PAP; HLH protocol if triggered' },
  { gene: 'SLC17A5',  group: 'Lysosomal — Free Sialic Acid',  urine_aa: 'Free sialic ↑↑',   plasma_aa: 'Normal',              treatable: '✗', pathognomonic: 'Urine free sialic acid 100-1000x; p.R39C=Salla (mild); severe=ISSD (lethal)', rx: 'Supportive; AEDs; physiotherapy (Salla); palliative (ISSD)' },
  { gene: 'SLC6A8',   group: 'CNS — Creatine Transport (XLR)', urine_aa: 'Cr/CrRatio ↑',    plasma_aa: 'Normal/high Cr',      treatable: '✗', pathognomonic: 'Absent brain creatine on 1H-MRS PATHOGNOMONIC; XLR; creatine supplement FAILS', rx: 'No effective therapy; gene therapy investigational; AEDs; behavioral therapy' },
  { gene: 'SLC6A5',   group: 'CNS — Glycine Reuptake',        urine_aa: 'Normal',            plasma_aa: 'Normal',              treatable: '✓', pathognomonic: 'Non-habituating startle; neonatal tonic apnea; EEG NORMAL (not epilepsy)', rx: 'Clonazepam URGENTLY; Heimlich maneuver LIFE-SAVING; reduce startling stimuli' },
  { gene: 'SLC25A22', group: 'Mitochondria — Glutamate',      urine_aa: 'Normal',            plasma_aa: 'Normal',              treatable: '✗', pathognomonic: 'Neonatal myoclonus day 0-3 + burst-suppression → hypsarrhythmia transition', rx: 'Pyridoxine 100mg IV FIRST; ketogenic diet partial; AEDs largely refractory' },
];

const SEED_BASE_DISPLAY = {
  'SLC6A19': 4254, 'SLC3A1': 4255, 'SLC7A9': 4256, 'SLC7A7': 4257,
  'SLC17A5': 4258, 'SLC6A8': 4259, 'SLC6A5': 4260, 'SLC25A22': 4261,
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

export default function HereditaryAATransportAtlasPage() {
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
    background: tab === t ? '#1a237e' : '#333', color: '#fff', border: 'none', fontSize: 13,
  });

  if (loading) return <div style={{ background: '#111', color: '#fff', minHeight: '100vh', padding: 24 }}>Loading AA Transport Atlas…</div>;
  if (error)   return <div style={{ background: '#111', color: '#f66', minHeight: '100vh', padding: 24 }}>Error: {error}</div>;

  return (
    <div style={{ background: '#111', color: '#eee', minHeight: '100vh', padding: 16, fontFamily: 'monospace' }}>
      <h1 style={{ color: '#90caf9', fontSize: 20, marginBottom: 4 }}>
        🧬 Hereditary-Amino-Acid-Transport-Disorder-Atlas — Complete 8-Gene Reference
      </h1>
      <div style={{ fontSize: 12, color: '#aaa', marginBottom: 12 }}>
        SLC6A19 · SLC3A1 · SLC7A9 · SLC7A7 · SLC17A5 · SLC6A8 · SLC6A5 · SLC25A22 ·
        seeds {SEED_BASE_DISPLAY['SLC6A19']}–{SEED_BASE_DISPLAY['SLC25A22']} ·
        Urine-AA-Chromatography-Primary-Split · Cystinuria-Tiopronin · LPI-PAP-Citrulline · Hyperekplexia-Clonazepam
      </div>

      <div style={{ marginBottom: 0 }}>
        {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      <div style={{ background: '#1a1a1a', borderRadius: '0 4px 4px 4px', padding: 16 }}>

        {/* ── OVERVIEW ── */}
        {tab === 'Overview' && overview && (
          <div>
            <h2 style={{ color: '#90caf9', fontSize: 16 }}>AA Transport Atlas Overview — {overview.total_patients} Patients · 8 Genes</h2>
            <div style={{ fontSize: 13, color: '#aaa', marginBottom: 12 }}>{overview.diagnostic_triage}</div>

            {/* Treatable genes callout */}
            <div style={{ background: '#0d1b3e', border: '1px solid #42a5f5', borderRadius: 6, padding: 10, marginBottom: 14 }}>
              <strong style={{ color: '#90caf9' }}>⚡ Treatable Transport Disorders: </strong>
              <span style={{ color: '#e3f2fd', fontSize: 13 }}>{overview.treatable_genes?.join(' · ')}</span>
              <div style={{ fontSize: 12, color: '#90caf9', marginTop: 4 }}>
                SLC6A19 → nicotinamide CURATIVE | SLC3A1/SLC7A9 → tiopronin + alkalinization | SLC7A7 → citrulline | SLC6A5 → clonazepam HIGHLY EFFECTIVE
              </div>
            </div>

            {/* Transport groups */}
            {overview.transport_groups && Object.entries(overview.transport_groups).map(([grpKey, grp]) => (
              <div key={grpKey} style={{ marginBottom: 12, background: '#222', borderRadius: 6, padding: 10 }}>
                <div style={{ color: '#ffb74d', fontWeight: 700, marginBottom: 4 }}>
                  {grpKey === 'renal_intestinal' ? '🔵 Renal/Intestinal Amino Acid Transport' :
                   grpKey === 'lysosomal' ? '🟢 Lysosomal Transport (Sialic Acid)' :
                   '🟣 CNS/Mitochondrial Transport'}
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
                      <th key={h} style={{ padding: '6px 8px', textAlign: 'left', color: '#90caf9', borderBottom: '1px solid #444' }}>{h}</th>
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
            <h2 style={{ color: '#90caf9', fontSize: 16 }}>AA Transport Gene Reference — 8 Genes</h2>
            {Object.keys(GENE_INFO).map(gene => (
              <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={GENE_INFO[gene]} />
            ))}

            <h3 style={{ color: '#ffb74d', marginTop: 16 }}>Phenotype Comparison Table</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#333' }}>
                    {['Gene','Group','Urine AA','Plasma AA','Treatable','Key Pathognomonic','Treatment'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', textAlign: 'left', color: '#90caf9', borderBottom: '1px solid #444' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {PHENOTYPE_ROWS.map((row, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#1e1e1e' : '#222' }}>
                      <td style={{ padding: '5px 8px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                      <td style={{ padding: '5px 8px', fontSize: 11, color: '#aaa' }}>{row.group}</td>
                      <td style={{ padding: '5px 8px', fontWeight: 700 }}>{row.urine_aa}</td>
                      <td style={{ padding: '5px 8px', color: row.plasma_aa.includes('LOW') ? '#ef9a9a' : '#ccc' }}>{row.plasma_aa}</td>
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

        {/* ── URINE AA CASCADE ── */}
        {tab === 'Urine AA Cascade' && defs && (
          <div>
            <h2 style={{ color: '#90caf9', fontSize: 16 }}>Urine Amino Acid Cascade Algorithm</h2>
            <div style={{ background: '#0a1929', border: '1px solid #42a5f5', borderRadius: 8, padding: 14, marginBottom: 16, fontSize: 13, lineHeight: 1.7 }}>
              {defs.primary_cascade_split?.split('. ').map((step, i) => (
                <div key={i} style={{ marginBottom: 4, paddingLeft: step.startsWith('STEP') ? 0 : 16 }}>
                  {step.startsWith('STEP') ? <strong style={{ color: '#90caf9' }}>{step}.</strong> : <span style={{ color: '#ccc' }}>• {step}.</span>}
                </div>
              ))}
            </div>

            {/* Urine AA cascade boxes */}
            {defs.urine_aa_cascade && (
              <div style={{ background: '#1a1a2e', border: '1px solid #7c4dff', borderRadius: 6, padding: 12, marginBottom: 14 }}>
                <h3 style={{ color: '#ce93d8', margin: '0 0 8px' }}>Urine AA Chromatography Quick Reference</h3>
                {Object.entries(defs.urine_aa_cascade).map(([k, v]) => (
                  <div key={k} style={{ fontSize: 12, marginBottom: 4 }}>
                    <span style={{ color: '#7c4dff', fontWeight: 700 }}>{k.replace(/_/g, ' ').toUpperCase()}: </span>
                    <span style={{ color: '#ccc' }}>{v}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Key distinguishers */}
            {defs.key_distinguishers?.map((kd, i) => (
              <div key={i} style={{ background: '#222', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                <div style={{ color: '#ffb74d', fontWeight: 700, marginBottom: 6, fontSize: 14 }}>{kd.feature}</div>
                <div style={{ fontSize: 12, color: '#aaa', marginBottom: 4 }}>{kd.significance}</div>
                {Object.entries(kd).filter(([k]) => !['feature','significance'].includes(k)).map(([k, v]) => (
                  <div key={k} style={{ fontSize: 12, marginBottom: 2 }}>
                    <span style={{ color: '#90caf9', fontWeight: 700 }}>{k.replace(/_/g, ' ')}: </span>
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
            <h2 style={{ color: '#90caf9', fontSize: 16 }}>Treatment Matrix</h2>
            <div style={{ fontSize: 12, color: '#ff8a65', marginBottom: 12 }}>
              ⚡ Treatable: SLC6A19 (nicotinamide CURATIVE), SLC3A1/SLC7A9 (tiopronin+alkalinization), SLC7A7 (citrulline), SLC6A5 (clonazepam HIGHLY EFFECTIVE) — diagnose early
            </div>
            {defs?.treatment_matrix?.map((t, i) => (
              <div key={i} style={{ background: '#1e2e3e', border: '1px solid #42a5f5', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                <div style={{ color: '#90caf9', fontWeight: 700, fontSize: 14, marginBottom: 6 }}>{t.intervention}</div>
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
                <div style={{ fontSize: 12, color: '#ffb74d', marginBottom: 6 }}>Neurological: {gp.neurological_pct}%</div>
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
            <h2 style={{ color: '#90caf9', fontSize: 16 }}>AA Transport Disorder Definitions</h2>

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
