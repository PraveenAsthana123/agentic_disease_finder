'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-pyridoxine-dependent-molybdenum-cofactor-sulfite-oxidase-epilepsy-atlas';
const TABS = ['Overview', 'Gene Table', 'Cascade Algorithm', 'Treatment', 'Definitions'];

const GENE_COLORS = {
  'ALDH7A1': '#1b5e20',  // deep green       — PDE1; antiquitin; AASA/pipecolic acid; pyridoxine CURATIVE
  'PNPO':    '#1a237e',  // deep navy         — PNPO deficiency; PLP ONLY; NOT pyridoxine; neonatal EE
  'PLPBP':   '#4a148c',  // deep purple       — PLPHP deficiency; both B6 forms; milder
  'ALDH4A1': '#e65100',  // burnt orange      — P5CDH; hyperprolinemia II; P5C inactivates PLP
  'MOCS1':   '#880e4f',  // deep crimson      — MoCoD type A; cPMP absent; fosdenopterin FDA-approved
  'MOCS2':   '#37474f',  // dark slate        — MoCoD type B; cPMP elevated; fosdenopterin FAILS
  'GPHN':    '#004d40',  // dark teal         — MoCoD type C; DUAL role: MoCo + inhibitory synapse
  'SUOX':    '#bf360c',  // dark ember        — isolated SUOX; sulfite positive; uric acid NORMAL
};

const GENE_INFO = {
  'ALDH7A1': { full: 'ALDH7A1-539aa / AR / Antiquitin / Alpha-AASA-dehydrogenase / 5q31.2 / PDE1-Pyridoxine-Dependent-Epilepsy / AASA+PipecolicAcid-PATHOGNOMONIC-even-on-treatment / Pyridoxine100mgIV-trial / LysineRestriction-improves-neurodevelopment', locus: '5q31.2', size: '539 aa / 58 kDa', inh: 'AR; AASA + pipecolic acid remain elevated even on pyridoxine (permanent biomarkers); pyridoxine 100 mg IV trial stops seizures (also diagnostic); lysine restriction + arginine supplementation improves neurodevelopment significantly; prenatal seizures reported; P6C (from AASA) inactivates PLP via Knoevenagel condensation' },
  'PNPO':    { full: 'PNPO-261aa / AR / Pyridox(am)ine-5-phosphate-Oxidase / Converts-PNP-PMP-to-PLP / 17q21.32 / PNPO-Deficiency / PLP-ONLY-NOT-Pyridoxine-CRITICAL / NeonatalEpilepticEncephalopathy / CSF-pyridoxamine-elevated / Threonine-elevated-CSF / RiboflavinAdjunct', locus: '17q21.32', size: '261 aa / 30 kDa', inh: 'AR; CRITICAL DISTINCTION: PLP ONLY works — pyridoxine WILL NOT stop seizures; PNPO cannot convert pyridoxine to PLP; delaying PLP treatment causes irreversible brain damage; neonatal epileptic encephalopathy; CSF pyridoxamine/PNP elevated; threonine elevated in CSF; monitor LFTs on PLP' },
  'PLPBP':   { full: 'PLPBP-321aa / AR / PLP-Homeostasis-Protein / PLPHP-Deficiency / 8p11.23 / Both-Pyridoxine-AND-PLP-Work / Neonatal-to-early-infantile / AASA-NEGATIVE / Milder-than-PNPO / B6-Vitamer-Profiling-Diagnostic / FormerlyPROSC', locus: '8p11.23', size: '321 aa / 35 kDa', inh: 'AR; both pyridoxine AND PLP work (unlike PNPO where only PLP works); generally milder phenotype; AASA negative (distinguishes from ALDH7A1); no specific pathognomonic metabolite; diagnosis requires PLPBP sequencing + B6 vitamer profiling; formerly called PROSC deficiency' },
  'ALDH4A1': { full: 'ALDH4A1-563aa / AR / P5C-Dehydrogenase / P5CDH / 1p36.13 / HyperprolinemiaII / P5C-Accumulates-Inactivates-PLP / PlasmaProline-GT1000umolL / UrinaryP5C-Elevated / Pyridoxine-Partial-Response / IncompletePenetrance-Neurological', locus: '1p36.13', size: '563 aa / 62 kDa', inh: 'AR; proline catabolism: proline → P5C (PRODH) → glutamate (ALDH4A1); P5C accumulates when ALDH4A1 deficient; P5C inactivates PLP (same Knoevenagel mechanism as P6C in ALDH7A1); plasma proline markedly elevated (>1000 µmol/L); urinary P5C elevated (distinguishes hyperprolinemia II from type I/PRODH); incomplete neurological penetrance' },
  'MOCS1':   { full: 'MOCS1-481aa / AR / cPMP-Synthesis / MOCS1A+MOCS1B / 6p21.2 / MoCoD-TypeA / SulfitePositive-UricAcidLow-PATHOGNOMONIC / cPMP-ABSENT-CRITICAL / Fosdenopterin-FDA2021-TypeA-ONLY / EctopiaLentis / NeonatalSeizures-Severe', locus: '6p21.2', size: '481 aa / 54 kDa', inh: 'AR; MoCoD type A — cPMP absent in plasma (MOCS1 cannot make cPMP); sulfite POSITIVE + uric acid VERY LOW pathognomonic; ectopia lentis; neonatal refractory seizures; fosdenopterin (NULIBRY) FDA-approved 2021 ONLY for type A; start within days of birth; urinary sulfite decreases on treatment' },
  'MOCS2':   { full: 'MOCS2-362aa / AR / MPT-Synthase / MOCS2A+MOCS2B-OverlappingORFs / 5q11.2 / MoCoD-TypeB / SulfitePositive-UricAcidLow / cPMP-ELEVATED-KeyDDx-TypeA / Fosdenopterin-DOES-NOT-WORK / GeneTherepy-Investigational / SamePhenotype-MOCS1', locus: '5q11.2', size: '362 aa / 41 kDa', inh: 'AR; MoCoD type B — cPMP ELEVATED (MPT synthase deficient, substrate accumulates); same biochemistry as type A (sulfite positive, uric acid low) but DIFFERENT treatment; fosdenopterin DOES NOT WORK (cannot process cPMP without MPT synthase); two overlapping ORFs (MOCS2A + MOCS2B); gene therapy preclinical/investigational' },
  'GPHN':    { full: 'GPHN-736aa / AR / Gephyrin / MoCoD-TypeC-PLUS-InhibitorySynapse-Scaffold / 14q23.3 / DUAL-DISEASE-MECHANISM / MoCoD-TypeC-SulfitePositive-UricAcidLow / OR-EpilepsyAutism-WithoutMoCoD / GABA-A-GlyR-Anchor / Fosdenopterin-FAILS-TypeC', locus: '14q23.3', size: '736 aa / 83 kDa', inh: 'AR; DUAL FUNCTION: (1) MoCo synthesis (G-domain): molybdate insertase; LOF → MoCoD type C (sulfite positive, uric acid low, severe neonatal); fosdenopterin DOES NOT WORK for type C; (2) inhibitory synapse scaffold (E-domain): GABA-A and glycine receptor anchor; scaffolding-only variants → epilepsy + autism WITHOUT MoCoD biochemistry (sulfite negative, uric acid normal)' },
  'SUOX':    { full: 'SUOX-545aa / AR / Sulfite-Oxidase / Sulfite-to-Sulfate / 12q13.11 / IsolatedSulfiteOxidaseDeficiency / SulfitePositive-UricAcidNORMAL-CRITICAL-DDx / S-Sulfocysteine-PATHOGNOMONIC / NoFosdenopterin / LowSulfurDiet / NeonatalSeizures-EctopiaLentis', locus: '12q13.11', size: '545 aa / 60 kDa', inh: 'AR; isolated SUOX deficiency — sulfite POSITIVE + uric acid NORMAL (xanthine dehydrogenase intact, MoCo is present); this is THE CRITICAL DDx from all MoCoD types; S-sulfocysteine elevated and pathognomonic; neonatal seizures; ectopia lentis less common than MoCoD; no FDA-approved therapy; low-sulfur amino acid diet reduces substrate; fosdenopterin NOT indicated (MoCo is normal)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'ALDH7A1', group: 'B6-Responsive — PDE1 (Antiquitin)',        sulfite: '—',  uricacid: 'nl', cPMP: 'nl',      pathognomonic: 'AASA + pipecolic acid elevated even on Rx; pyridoxine 100mg IV trial',          rx: 'Pyridoxine 30mg/kg/day + lysine restriction + arginine',   unique: 'AASA/pipecolic acid persist on treatment = permanent markers; P6C inactivates PLP; lysine restriction improves neurodevelopment; prenatal seizures; most common B6-responsive epilepsy worldwide' },
  { gene: 'PNPO',    group: 'B6-Responsive — PNPO Deficiency',           sulfite: '—',  uricacid: 'nl', cPMP: 'nl',      pathognomonic: 'PLP ONLY works — pyridoxine fails; CSF pyridoxamine/PNP elevated',              rx: 'PLP 30-50mg/kg/day (NOT pyridoxine); monitor LFTs',         unique: 'LIFE-SAVING DISTINCTION: pyridoxine WILL NOT stop seizures; delaying PLP causes irreversible brain damage; neonatal EE; CSF threonine elevated (PLP-dependent threonine aldolase); riboflavin adjunct in hypomorphic variants' },
  { gene: 'PLPBP',   group: 'B6-Responsive — PLPHP Deficiency',          sulfite: '—',  uricacid: 'nl', cPMP: 'nl',      pathognomonic: 'Both pyridoxine AND PLP work; AASA negative; sulfite negative',                   rx: 'Pyridoxine or PLP (both effective); 15-30mg/kg/day',        unique: 'Milder than PNPO; no specific pathognomonic metabolite (unlike ALDH7A1 AASA); B6 vitamer profiling + sequencing for diagnosis; formerly called PROSC deficiency; PLPBP = PLP buffer/homeostasis protein' },
  { gene: 'ALDH4A1', group: 'B6-Related — Hyperprolinemia II (P5CDH)',   sulfite: '—',  uricacid: 'nl', cPMP: 'nl',      pathognomonic: 'Plasma proline >1000µmol/L; urinary P5C elevated; incomplete neurological penetrance', rx: 'Pyridoxine (adjunct); low-proline diet',                   unique: 'P5CDH deficiency; proline catabolism block; P5C inactivates PLP (same mechanism as ALDH7A1-P6C); incomplete penetrance — some patients asymptomatic; elevated P5C (not just proline) distinguishes HPII from PRODH-hyperprolinemia I (type I: no P5C, schizophrenia association, 22q11)' },
  { gene: 'MOCS1',   group: 'MoCoD Type A — cPMP Synthesis',             sulfite: '↑↑', uricacid: 'LOW', cPMP: 'ABSENT', pathognomonic: 'Sulfite + + uric acid LOW; cPMP ABSENT; fosdenopterin FDA-approved',               rx: 'Fosdenopterin (NULIBRY) daily IV — START IMMEDIATELY',      unique: 'MoCoD type A: cPMP absent in plasma (nothing for fosdenopterin to build on at the first step); fosdenopterin provides exogenous cPMP → intact MOCS2+GPHN process it; FDA-approved 2021; ectopia lentis; start within days of birth for best neurological benefit; monitor urinary sulfite' },
  { gene: 'MOCS2',   group: 'MoCoD Type B — MPT Synthase',               sulfite: '↑↑', uricacid: 'LOW', cPMP: 'HIGH',   pathognomonic: 'Sulfite + + uric acid LOW; cPMP ELEVATED — KEY DDx from MOCS1',                   rx: 'Supportive; low-sulfur diet; gene therapy investigational', unique: 'MoCoD type B: cPMP elevated (MPT synthase absent; cPMP substrate accumulates); fosdenopterin USELESS (cPMP cannot be processed); same severe neonatal phenotype as MOCS1; overlapping ORF structure (MOCS2A + MOCS2B — variants in either cause disease); gene therapy trials underway' },
  { gene: 'GPHN',    group: 'MoCoD Type C — Gephyrin (DUAL)',            sulfite: '↑/—', uricacid: 'LOW/nl', cPMP: 'nl',  pathognomonic: 'MoCoD type C: sulfite + uric acid LOW; OR epilepsy+ASD: sulfite negative uric acid normal', rx: 'MoCoD type C: supportive; scaffolding variants: AED + GABA-modulating', unique: 'DUAL DISEASE GENE: full LOF → MoCoD type C (identical severity to MOCS1/MOCS2); scaffolding-domain-only missense → epilepsy+autism WITHOUT MoCoD biochemistry; gephyrin = master GABA-A + glycine receptor anchor; GPHN CNVs → ASD; always check sulfite+uric acid in GPHN variant patient' },
  { gene: 'SUOX',    group: 'Isolated Sulfite Oxidase Deficiency',        sulfite: '↑↑', uricacid: 'NORMAL', cPMP: 'nl', pathognomonic: 'Sulfite POSITIVE + uric acid NORMAL — THE critical DDx from MoCoD; S-sulfocysteine', rx: 'Low-sulfur amino acid diet; supportive; no approved pharmacotherapy', unique: 'Isolated SUOX — MoCo is PRESENT and functional (xanthine dehydrogenase intact → uric acid normal); only sulfite oxidase step is deficient; S-sulfocysteine elevated (same as MoCoD); no fosdenopterin indication; XDH-derived uric acid is the key split between SUOX and all MoCoD types' },
];

const SEED_BASE_DISPLAY = {
  'ALDH7A1': 4238, 'PNPO': 4239, 'PLPBP': 4240, 'ALDH4A1': 4241,
  'MOCS1': 4242, 'MOCS2': 4243, 'GPHN': 4244, 'SUOX': 4245,
};

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
          <span style={{ marginLeft: 8, color: '#555', fontSize: 12 }}>{info.locus} · {info.size} · {info.inh}</span>
        </div>
        <span style={{ color: '#888', fontSize: 11 }}>seed {SEED_BASE_DISPLAY[gene]}</span>
      </div>
      <div style={{ marginTop: 6, fontSize: 12, color: '#333', lineHeight: 1.5 }}>{info.full}</div>
      {geneData && (
        <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          <Badge text={`n=${geneData.n || 40}`} color={color} />
          <Badge text={`seed ${SEED_BASE_DISPLAY[gene]}`} color="#777" />
          {geneData.pathognomonic && <Badge text={geneData.pathognomonic.slice(0, 55) + '…'} color="#555" />}
        </div>
      )}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div>Loading...</div>;
  const groups = data.groups || {};
  return (
    <div>
      <div style={{ background: '#f8f9fa', border: '1px solid #ddd', borderRadius: 8, padding: 14, marginBottom: 16 }}>
        <h3 style={{ margin: '0 0 8px', color: '#1b5e20' }}>8-Gene Atlas Summary</h3>
        <p style={{ margin: '0 0 6px', color: '#555', fontSize: 13 }}>
          <strong>Total cohort:</strong> {data.total_n || 320} patients · <strong>Seeds:</strong> {data.seeds || '4238-4245'}
        </p>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 6 }}>
          {(groups.b6_responsive || []).map(g => <Badge key={g} text={g} color={GENE_COLORS[g] || '#555'} />)}
          {(groups['mocoд_types'] || []).map(g => <Badge key={g} text={g} color={GENE_COLORS[g] || '#555'} />)}
          {(groups.sulfite_oxidase || []).map(g => <Badge key={g} text={g} color={GENE_COLORS[g] || '#555'} />)}
        </div>
      </div>

      <div style={{ background: '#fff3e0', border: '1px solid #ff8f00', borderRadius: 8, padding: 12, marginBottom: 14 }}>
        <h4 style={{ margin: '0 0 6px', color: '#e65100' }}>Primary Cascade Split</h4>
        <p style={{ margin: 0, fontSize: 12, color: '#4a148c', lineHeight: 1.6 }}>{data.primary_cascade_split}</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 10 }}>
        {Object.keys(GENE_COLORS).map(gene => {
          const cohort = (data.cohort_summaries || []).find(c => c.gene === gene);
          return (
            <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={GENE_INFO[gene]} geneData={cohort} />
          );
        })}
      </div>

      {data.universal_drug_contraindications && (
        <div style={{ background: '#ffebee', border: '1px solid #c62828', borderRadius: 8, padding: 12, marginTop: 14 }}>
          <h4 style={{ margin: '0 0 6px', color: '#c62828' }}>Key Treatment Notes</h4>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#333', lineHeight: 1.7 }}>
            {data.universal_drug_contraindications.map((ci, i) => <li key={i}>{ci}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

function GeneTableTab() {
  return (
    <div>
      <h3 style={{ marginTop: 0, color: '#1a237e' }}>Gene Reference Table</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#1a237e', color: '#fff' }}>
              {['Gene', 'Size', 'Locus', 'Inh', 'Group', 'Sulfite', 'Uric Acid', 'cPMP', 'Pathognomonic', 'Treatment', 'Key Unique Features'].map(h => (
                <th key={h} style={{ padding: '6px 8px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {PHENOTYPE_ROWS.map((row, i) => (
              <tr key={row.gene} style={{ background: i % 2 === 0 ? '#f8f8f8' : '#fff' }}>
                <td style={{ padding: '6px 8px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                <td style={{ padding: '6px 8px', whiteSpace: 'nowrap' }}>{GENE_INFO[row.gene].size}</td>
                <td style={{ padding: '6px 8px', whiteSpace: 'nowrap' }}>{GENE_INFO[row.gene].locus}</td>
                <td style={{ padding: '6px 8px' }}>{GENE_INFO[row.gene].inh.split(';')[0]}</td>
                <td style={{ padding: '6px 8px', fontSize: 11 }}>{row.group}</td>
                <td style={{ padding: '6px 8px', fontWeight: 700, color: row.sulfite === '↑↑' ? '#c62828' : row.sulfite === '↑/—' ? '#e65100' : '#1b5e20' }}>{row.sulfite || '—'}</td>
                <td style={{ padding: '6px 8px', fontWeight: 700, color: row.uricacid === 'LOW' ? '#c62828' : row.uricacid === 'NORMAL' ? '#1b5e20' : '#555' }}>{row.uricacid}</td>
                <td style={{ padding: '6px 8px', fontWeight: 700, color: row.cPMP === 'ABSENT' ? '#880e4f' : row.cPMP === 'HIGH' ? '#c62828' : '#555' }}>{row.cPMP}</td>
                <td style={{ padding: '6px 8px', fontSize: 11 }}>{row.pathognomonic}</td>
                <td style={{ padding: '6px 8px', fontSize: 11 }}>{row.rx}</td>
                <td style={{ padding: '6px 8px', fontSize: 11, color: '#333' }}>{row.unique}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CascadeTab({ data }) {
  const cascadeText = data?.primary_cascade_split || '';
  const steps = cascadeText.split(/(?=STEP\s+\d+|KEY RULE)/g).filter(Boolean);
  return (
    <div>
      <h3 style={{ marginTop: 0, color: '#880e4f' }}>Neonatal Epileptic Encephalopathy — Diagnostic Cascade</h3>

      <div style={{ background: '#fce4ec', border: '2px solid #880e4f', borderRadius: 8, padding: 14, marginBottom: 14 }}>
        <h4 style={{ margin: '0 0 10px', color: '#880e4f' }}>Two Distinct Pathways</h4>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div style={{ background: '#fff', borderRadius: 6, padding: 10, border: '2px solid #1b5e20' }}>
            <div style={{ fontWeight: 700, color: '#1b5e20', marginBottom: 6 }}>Urine Sulfite NEGATIVE → B6/PLP Trial</div>
            <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, lineHeight: 1.7 }}>
              <li><strong>Pyridoxine 100mg IV stops seizures</strong> → ALDH7A1 (PDE1) — check AASA + pipecolic acid</li>
              <li><strong>PLP works but NOT pyridoxine</strong> → PNPO — CSF pyridoxamine/threonine elevated</li>
              <li><strong>Both work</strong> → PLPBP — vitamer profiling + sequencing</li>
              <li><strong>Plasma proline &gt;1000</strong> → ALDH4A1 (hyperprolinemia II) — check P5C</li>
            </ul>
          </div>
          <div style={{ background: '#fff', borderRadius: 6, padding: 10, border: '2px solid #880e4f' }}>
            <div style={{ fontWeight: 700, color: '#880e4f', marginBottom: 6 }}>Urine Sulfite POSITIVE → Uric Acid</div>
            <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, lineHeight: 1.7 }}>
              <li><strong>Uric acid LOW → MoCoD:</strong>
                <ul style={{ paddingLeft: 14 }}>
                  <li>cPMP ABSENT → MOCS1 type A → <strong style={{ color: '#880e4f' }}>FOSDENOPTERIN IMMEDIATELY</strong></li>
                  <li>cPMP ELEVATED → MOCS2 type B → supportive (fosdenopterin FAILS)</li>
                  <li>Sequencing → GPHN type C → supportive (fosdenopterin FAILS)</li>
                </ul>
              </li>
              <li><strong>Uric acid NORMAL → isolated SUOX</strong> — low-sulfur diet; no fosdenopterin</li>
            </ul>
          </div>
        </div>
      </div>

      <div style={{ background: '#e8eaf6', border: '1px solid #3f51b5', borderRadius: 8, padding: 12, marginBottom: 12 }}>
        <h4 style={{ margin: '0 0 6px', color: '#1a237e' }}>Step-by-Step Protocol</h4>
        {steps.map((s, i) => (
          <div key={i} style={{ marginBottom: 8, padding: 8, background: '#fff', borderRadius: 6, fontSize: 12, lineHeight: 1.5 }}>
            {s.trim()}
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 10 }}>
        {PHENOTYPE_ROWS.map(row => (
          <div key={row.gene} style={{ border: `2px solid ${GENE_COLORS[row.gene]}`, borderRadius: 8, padding: 10 }}>
            <div style={{ fontWeight: 700, color: GENE_COLORS[row.gene], marginBottom: 4 }}>{row.gene}</div>
            <div style={{ fontSize: 11, color: '#555', marginBottom: 4 }}>{row.group}</div>
            <div style={{ fontSize: 11, color: '#c62828', marginBottom: 4 }}><strong>Pathognomonic:</strong> {row.pathognomonic}</div>
            <div style={{ fontSize: 11, color: '#1b5e20' }}><strong>Treatment:</strong> {row.rx}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TreatmentTab({ data }) {
  const matrix = data?.treatment_matrix || [];
  return (
    <div>
      <h3 style={{ marginTop: 0, color: '#1a237e' }}>Treatment Matrix</h3>
      {matrix.map((t, i) => (
        <div key={i} style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12, marginBottom: 10 }}>
          <div style={{ fontWeight: 700, color: '#1a237e', fontSize: 14, marginBottom: 6 }}>{t.intervention}</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
            <div><strong>Indications:</strong> {t.indications}</div>
            <div><strong>Mechanism:</strong> {t.mechanism}</div>
            <div><strong>Dose:</strong> {t.dose}</div>
            <div style={{ color: '#c62828' }}><strong>Contraindicated/Notes:</strong> {t.contraindicated}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  const defs = data?.key_distinguishers || [];
  const [open, setOpen] = useState(null);
  return (
    <div>
      <h3 style={{ marginTop: 0, color: '#1a237e' }}>Key Clinical Definitions & Distinguishers</h3>
      {defs.map((d, i) => (
        <div key={i} style={{ border: '1px solid #ddd', borderRadius: 8, marginBottom: 6, overflow: 'hidden' }}>
          <div
            style={{ padding: '10px 14px', cursor: 'pointer', background: open === i ? '#e8eaf6' : '#f8f9fa',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            onClick={() => setOpen(open === i ? null : i)}
          >
            <span style={{ fontWeight: 600, fontSize: 13, color: '#1a237e' }}>{d.name}</span>
            <span style={{ fontSize: 16 }}>{open === i ? '▲' : '▼'}</span>
          </div>
          {open === i && (
            <div style={{ padding: '10px 14px', borderTop: '1px solid #ddd', fontSize: 12, lineHeight: 1.6 }}>
              <p style={{ margin: '0 0 8px', color: '#333' }}>{d.definition}</p>
              <div style={{ background: '#e8f5e9', borderRadius: 4, padding: '6px 10px', color: '#1b5e20', fontSize: 11 }}>
                <strong>Clinical relevance:</strong> {d.clinical_relevance}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function HereditaryPDMCOSEpilepsyAtlasPage() {
  const [tab, setTab]         = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    if (tab === 'Overview' && !overview) fetchData('overview', setOverview);
    if ((tab === 'Gene Table' || tab === 'Cascade Algorithm') && !breakdown) fetchData('breakdown', setBreakdown);
    if (tab === 'Treatment' && !definitions) fetchData('definitions', setDefinitions);
    if (tab === 'Definitions' && !definitions) fetchData('definitions', setDefinitions);
  }, [tab]);

  async function fetchData(endpoint, setter) {
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API}/api/${SLUG}/${endpoint}`);
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      setter(await r.json());
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }

  const renderTab = () => {
    if (loading) return <div style={{ textAlign: 'center', padding: 40, color: '#888' }}>Loading…</div>;
    if (error)   return <div style={{ color: '#c62828', padding: 16 }}>Error: {error}</div>;
    switch (tab) {
      case 'Overview':         return <OverviewTab data={overview} />;
      case 'Gene Table':       return <GeneTableTab />;
      case 'Cascade Algorithm': return <CascadeTab data={breakdown} />;
      case 'Treatment':        return <TreatmentTab data={definitions} />;
      case 'Definitions':      return <DefinitionsTab data={definitions} />;
      default:                 return null;
    }
  };

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '16px' }}>
      <div style={{ background: 'linear-gradient(135deg, #1b5e20 0%, #4a148c 50%, #880e4f 100%)', color: '#fff', borderRadius: 10, padding: '18px 22px', marginBottom: 16 }}>
        <h1 style={{ margin: '0 0 6px', fontSize: 20 }}>
          Hereditary Pyridoxine-Dependent · Molybdenum Cofactor · Sulfite-Oxidase Epilepsy Atlas
        </h1>
        <p style={{ margin: '0 0 4px', opacity: 0.9, fontSize: 13 }}>
          Complete 8-Gene Reference · ALDH7A1 · PNPO · PLPBP · ALDH4A1 · MOCS1 · MOCS2 · GPHN · SUOX
        </p>
        <p style={{ margin: 0, opacity: 0.75, fontSize: 11 }}>
          320-Patient Aggregate · 8×40 seeds 4238-4245 · B6-Responsive Epilepsies + MoCoD Types A/B/C + Isolated SUOX
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 16, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '7px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13,
            background: tab === t ? '#1a237e' : '#e8eaf6', color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400
          }}>{t}</button>
        ))}
      </div>

      {renderTab()}
    </div>
  );
}
