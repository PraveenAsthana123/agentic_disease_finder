'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-urea-cycle-disorder-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Treatment', 'Definitions'];

const GENE_COLORS = {
  'CPS1':     '#1b5e20',  // dark green      — most severe neonatal; orotic acid ABSENT; NCG trial NAGS DDx
  'OTC':      '#4a148c',  // deep purple     — X-linked most common; orotic acid MARKEDLY ELEVATED PATHOGNOMONIC
  'ASS1':     '#1a237e',  // deep navy       — CTLN1; citrulline >1000 umol/L PATHOGNOMONIC; NBS-detectable
  'ASL':      '#880e4f',  // deep crimson    — argininosuccinic acid PATHOGNOMONIC; hair fragility; HTN persists post-LTx
  'ARG1':     '#b71c1c',  // dark red        — spastic diplegia NOT hyperammonemia PATHOGNOMONIC; arginine RESTRICT
  'NAGS':     '#e65100',  // burnt orange    — rarest UCD; carglumic acid (NCG) COMPLETELY CURATIVE
  'SLC25A15': '#37474f',  // dark blue-grey  — HHH syndrome; homocitrulline PATHOGNOMONIC; French Canadian founder
  'SLC25A13': '#004d40',  // dark teal       — citrin CTLN2; food cravings PATHOGNOMONIC; East Asian; urgent LTx
};

const GENE_INFO = {
  'CPS1':     { full: 'CPS1-1500aa / AR / Carbamoyl-phosphate-synthetase-1 / First+rate-limiting-urea-cycle-step / 2q35 / CPS1-Deficiency / Orotic-acid-ABSENT-PATHOGNOMONIC / NCG-trial-MANDATORY / LTx-curative', locus: '2q35', size: '1500 aa / 165 kDa', inh: 'AR; first mitochondrial urea cycle step; requires NAG as activator; orotic acid ABSENT (block before cytoplasmic carbamoyl phosphate); ALWAYS trial NCG empirically to exclude NAGS; LTx curative' },
  'OTC':      { full: 'OTC-354aa / XLR / Ornithine-transcarbamylase / Second-urea-cycle-step / Xp11.4 / OTC-Deficiency / MOST-COMMON-UCD / Orotic-acid-MARKEDLY-ELEVATED-PATHOGNOMONIC / AllopurinolTest-carriers / LTx-curative', locus: 'Xp11.4', size: '354 aa / 39 kDa', inh: 'XLR; most common UCD (~1:14,000-40,000 males); carbamoyl phosphate overflows to cytoplasm -> orotic acid MARKEDLY elevated; female carriers variable (lyonization); allopurinol loading test detects carriers; LTx curative' },
  'ASS1':     { full: 'ASS1-412aa / AR / Argininosuccinate-synthetase-1 / Third-urea-cycle-step / 9q34.11 / Citrullinemia-Type-1-CTLN1 / Citrulline-GT1000-PATHOGNOMONIC / NBS-DETECTABLE / Arginine-MANDATORY / LTx-curative', locus: '9q34.11', size: '412 aa / 46 kDa', inh: 'AR; cytoplasmic step; citrulline MARKEDLY elevated (>1000 umol/L neonatal) PATHOGNOMONIC; NBS-detectable by tandem MS; NO argininosuccinic acid (DDx from ASL); arginine supplementation CRITICAL; LTx curative' },
  'ASL':      { full: 'ASL-464aa / AR / Argininosuccinate-lyase / Fourth-urea-cycle-step / 7q11.21 / Argininosuccinic-Aciduria / ArgininosuccinicAcid-PATHOGNOMONIC / HairFragility-PATHOGNOMONIC / Hypertension-NO-synthesis / LTx-corrects-NH3-NOT-hair-HTN', locus: '7q11.21', size: '464 aa / 52 kDa', inh: 'AR; argininosuccinic acid in plasma/urine PATHOGNOMONIC; hair fragility (trichorrhexis nodosa) PATHOGNOMONIC in survivors; hypertension via NO synthesis defect; LTx corrects NH3 but NOT hair fragility or HTN (extrahepatic defect persists)' },
  'ARG1':     { full: 'ARG1-330aa / AR / Arginase-1 / Fifth+final-urea-cycle-step / 6q23.2 / Arginemia / SPASTIC-DIPLEGIA-not-hyperammonemia-PATHOGNOMONIC / Arginine-GT200-PATHOGNOMONIC / Arginine-RESTRICTION-diet / No-scavengers-typically', locus: '6q23.2', size: '330 aa / 35 kDa', inh: 'AR; UNIQUE phenotype -- spastic diplegia NOT hyperammonemia PATHOGNOMONIC; NH3 mild/absent; arginine MARKEDLY elevated; RESTRICT arginine (OPPOSITE of other UCDs); often misdiagnosed as cerebral palsy; onset 2-4 years' },
  'NAGS':     { full: 'NAGS-534aa / AR / N-acetylglutamate-synthase / NAG-synthesis / 17q21.31 / NAGS-Deficiency / RAREST-UCD-~50-worldwide / Carglumic-acid-NCG-COMPLETELY-CURATIVE-PATHOGNOMONIC / CPS1-phenocopy-orotic-ABSENT / EMPIRICAL-NCG-MANDATORY', locus: '17q21.31', size: '534 aa / 58 kDa', inh: 'AR; produces NAG (obligatory CPS1 activator); rarest UCD (~50 cases worldwide); BIOCHEMICALLY IDENTICAL to CPS1 (orotic acid absent); NCG (carglumic acid) completely curative -- NH3 normalization confirms NAGS; ALWAYS give NCG empirically in any CPS1-like presentation' },
  'SLC25A15': { full: 'SLC25A15-301aa / AR / ORNT1-Ornithine-Citrulline-Antiporter / Mitochondrial-ornithine-transport / 13q14.11 / HHH-Syndrome / Homocitrulline-urine-PATHOGNOMONIC / Hyperornithinemia+Hyperammonemia+Homocitrullinuria-TRIAD / FrenchCanadian-pF188del-founder / Citrulline-supplement-bypasses-ORNT1', locus: '13q14.11', size: '301 aa / 34 kDa', inh: 'AR; mitochondrial ornithine/citrulline antiporter; HHH syndrome triad; homocitrullinuria PATHOGNOMONIC (not seen in other UCDs); neurological degeneration dominates; French Canadian founder p.F188del; citrulline supplement bypasses ORNT1 block' },
  'SLC25A13': { full: 'SLC25A13-675aa / AR / Citrin-AGC2-Aspartate-Glutamate-Carrier / Mitochondrial-malate-aspartate-shuttle / 7q21.3 / Citrin-Deficiency-NICCD-CTLN2 / FoodCravings-carbohydrate-aversion-PATHOGNOMONIC-CTLN2 / EastAsian-founder / IV-glycerol-glucose-ABSOLUTELY-CI / Urgent-LTx-CTLN2', locus: '7q21.3', size: '675 aa / 74 kDa', inh: 'AR; mitochondrial aspartate/glutamate carrier; NICCD (neonatal cholestasis, self-resolves) + CTLN2 (adult encephalopathy); food cravings (protein/fat, carbohydrate aversion) PATHOGNOMONIC for CTLN2 in East Asian adults; IV glycerol/high-glucose ABSOLUTELY CONTRAINDICATED; urgent LTx for CTLN2' },
};

const PHENOTYPE_ROWS = [
  { gene: 'CPS1',     group: 'Proximal UCD — CPS1 Deficiency',       orotic: 'ABSENT', nbs: '✗', neonatal: '✓', pathognomonic: 'Orotic acid absent + NH3 >1000',   rx: 'NCG trial → LTx',             unique: 'Orotic acid ABSENT (block before cytoplasmic CP pool); ALWAYS trial NCG (carglumic acid) empirically -- if NH3 normalizes = NAGS; if no response = CPS1 -> liver transplant' },
  { gene: 'OTC',      group: 'Proximal UCD — OTC Deficiency (XLR)',  orotic: 'HIGH',   nbs: '✗', neonatal: '✓', pathognomonic: 'Orotic acid MARKEDLY elevated',    rx: 'Benzoate/phenylbutyrate+Arg', unique: 'MOST COMMON UCD; X-linked; carbamoyl phosphate overflows to cytoplasm -> orotic acid MARKEDLY elevated PATHOGNOMONIC vs CPS1; female carriers variable; allopurinol test for carriers' },
  { gene: 'ASS1',     group: 'Distal UCD — CTLN1',                   orotic: 'normal', nbs: '✓', neonatal: '✓', pathognomonic: 'Citrulline >1000 umol/L (pure)',   rx: 'Arginine supplement + LTx',   unique: 'NBS-DETECTABLE; citrulline >1000 umol/L WITHOUT argininosuccinate = ASS1 (pure citrullinemia); arginine supplementation CRITICAL; LTx curative' },
  { gene: 'ASL',      group: 'Distal UCD — Argininosuccinic Aciduria',orotic: 'normal', nbs: '✓', neonatal: '✓', pathognomonic: 'Argininosuccinic acid in plasma/urine', rx: 'Arginine supplement + LTx', unique: 'HAIR FRAGILITY trichorrhexis nodosa PATHOGNOMONIC in survivors; HYPERTENSION via NO synthesis defect; LTx corrects NH3 NOT hair fragility/HTN (extrahepatic defect persists)' },
  { gene: 'ARG1',     group: 'Distal UCD — Arginemia',               orotic: 'normal', nbs: '✓', neonatal: '✗', pathognomonic: 'Spastic diplegia + arginine HIGH', rx: 'RESTRICT arginine diet',      unique: 'UNIQUE: spastic diplegia NOT hyperammonemia (NH3 mild/absent); arginine MARKEDLY elevated; RESTRICT arginine (OPPOSITE other UCDs); cerebral palsy misdiagnosis at ages 2-4yr; no scavengers typically' },
  { gene: 'NAGS',     group: 'Activator UCD — NAGS Deficiency',      orotic: 'ABSENT', nbs: '✗', neonatal: '✓', pathognomonic: 'CPS1 phenocopy; NCG CURATIVE',    rx: 'NCG (carglumic acid) CURATIVE', unique: 'RAREST UCD (~50 worldwide); biochemically IDENTICAL to CPS1; carglumic acid NCG COMPLETELY CURATIVE PATHOGNOMONIC response; ALWAYS give NCG empirically in CPS1-like; NH3 normalization confirms NAGS' },
  { gene: 'SLC25A15', group: 'Transport UCD — HHH Syndrome',         orotic: 'mild↑',  nbs: '✗', neonatal: '✗', pathognomonic: 'Homocitrulline urine PATHOGNOMONIC', rx: 'Ornithine restrict + Cit supp', unique: 'HHH TRIAD: hyperornithinemia + hyperammonemia + homocitrullinuria; homocitrulline PATHOGNOMONIC (not in other UCDs); NEUROLOGICAL DEGENERATION (cerebellar ataxia) dominates; French Canadian p.F188del founder' },
  { gene: 'SLC25A13', group: 'Transport UCD — Citrin Deficiency',    orotic: 'normal', nbs: '✓*', neonatal: '✗', pathognomonic: 'Food cravings (CTLN2); NBS citrulline+galactose (NICCD)', rx: 'CTLN2: urgent LTx; NICCD: MCT diet', unique: 'TWO PHENOTYPES: NICCD (neonatal cholestasis, self-resolves) + CTLN2 (adult encephalopathy); food cravings (protein/fat craving + carbohydrate AVERSION) PATHOGNOMONIC CTLN2; East Asian founder; IV glycerol/glucose ABSOLUTELY CONTRAINDICATED' },
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
        <span style={{ fontSize: 11, color: '#888' }}>seed {GENE_COLORS[gene] ? (Object.keys(GENE_COLORS).indexOf(gene) + 4214) : '—'} · n=40</span>
      </div>
      <div style={{ marginTop: 8, fontSize: 12, color: '#333' }}>{info?.full}</div>
      <div style={{ marginTop: 6, fontSize: 12 }}>
        <strong>Locus:</strong> {info?.locus} &nbsp;|&nbsp;
        <strong>Size:</strong> {info?.size} &nbsp;|&nbsp;
        <strong>Inheritance:</strong> {info?.inh?.split(';')[0]}
      </div>
      {geneData && (
        <div style={{ marginTop: 6, fontSize: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Badge text={`Group: ${geneData.ucd_group ?? '—'}`} color={color} />
          <Badge text={`NBS: ${geneData.nbs_detected ? '✓ detected' : '✗ missed'}`} color={geneData.nbs_detected ? '#1565c0' : '#757575'} />
          <Badge text={`Orotic: ${geneData.orotic_absent ? 'ABSENT' : geneData.orotic_elevated ? 'HIGH' : 'normal'}`} color={geneData.orotic_absent ? '#b71c1c' : geneData.orotic_elevated ? '#6a1b9a' : '#388e3c'} />
          <Badge text={`Neonatal crisis: ${geneData.neonatal_crisis ? '✓' : '✗'}`} color={geneData.neonatal_crisis ? '#b71c1c' : '#757575'} />
          {geneData.carglumate_curative && <Badge text="NCG CURATIVE" color="#e65100" />}
          {cohortData && <Badge text={`Mean NH3: ${cohortData.mean_nh3} umol/L`} color={cohortData.mean_nh3 > 500 ? '#b71c1c' : '#37474f'} />}
          {cohortData && <Badge text={`Severe: ${Math.round((cohortData.pct_severe ?? 0) * 100)}%`} color={cohortData.pct_severe > 0.4 ? '#b71c1c' : '#388e3c'} />}
          {geneData.inheritance === 'XLR' && <Badge text="X-LINKED" color="#4a148c" />}
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

export default function HeredUreaCycleDisorderAtlasPage() {
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
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Urea Cycle Disorder Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  // Build lookup maps from API data
  const geneMap = {};
  (overview?.gene_table ?? []).forEach(g => { geneMap[g.gene] = g; });
  const cohortMap = {};
  (breakdown?.cohort_summary ?? []).forEach(c => { cohortMap[c.gene] = c; });

  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: 1100, margin: '0 auto', padding: '20px 16px' }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
        🧬 Hereditary-Urea-Cycle-Disorder-Atlas
      </h1>
      <p style={{ fontSize: 13, color: '#555', marginBottom: 4 }}>
        Complete 8-Gene Reference · CPS1 · OTC · ASS1 · ASL · ARG1 · NAGS · SLC25A15 · SLC25A13
      </p>
      <p style={{ fontSize: 12, color: '#777', marginBottom: 16 }}>
        Seeds {overview?.seeds} · {overview?.total_patients} patients · Urea Cycle & Nitrogen Disposal Disorders
      </p>

      {/* Orotic Acid Algorithm Banner */}
      <div style={{ background: '#e8f5e9', border: '2px solid #1b5e20', borderRadius: 8, padding: '10px 16px', marginBottom: 12, fontSize: 13 }}>
        <strong>⚡ EMPIRICAL NCG RULE (NAGS vs CPS1):</strong>{' '}
        Severe hyperammonemia + orotic acid ABSENT → start carglumic acid (NCG) 100 mg/kg IMMEDIATELY — DO NOT wait for molecular results. NH3 normalization confirms NAGS (completely curative). No response → CPS1 → liver transplant workup.
      </div>
      <div style={{ background: '#e8eaf6', border: '2px solid #3949ab', borderRadius: 8, padding: '10px 16px', marginBottom: 16, fontSize: 13 }}>
        <strong>⚡ Primary Cascade Split:</strong>{' '}
        Orotic acid ABSENT (CPS1/NAGS) → trial NCG · Orotic acid HIGH (OTC) → benzoate/phenylbutyrate+arginine · Normal orotic (ASS1/ASL/ARG1/SLC25A15/SLC25A13) → plasma amino acids
        <div style={{ marginTop: 6, fontSize: 12, color: '#555' }}>
          <strong>NBS-detected:</strong> {(overview?.nbs_detectable_genes ?? []).join(', ')}&nbsp;&nbsp;|&nbsp;&nbsp;
          <strong>Orotic absent:</strong> {(overview?.orotic_absent_genes ?? []).join(', ')}&nbsp;&nbsp;|&nbsp;&nbsp;
          <strong>X-linked:</strong> {(overview?.xlinked_genes ?? []).join(', ')}&nbsp;&nbsp;|&nbsp;&nbsp;
          <strong>NCG curative:</strong> NAGS only
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 20, border: 'none', cursor: 'pointer', fontSize: 13,
            background: tab === t ? '#1b5e20' : '#e8eaf6', color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'Overview' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))', gap: 12, marginBottom: 20 }}>
            {[
              { label: 'Total Patients', value: overview?.total_patients },
              { label: 'Genes', value: (overview?.genes ?? []).length },
              { label: 'Seeds', value: overview?.seeds },
              { label: 'Proximal UCD', value: (overview?.proximal_ucd ?? []).join(', ') },
              { label: 'Distal UCD', value: (overview?.distal_ucd ?? []).join(', ') },
              { label: 'NBS-Detected', value: (overview?.nbs_detectable_genes ?? []).join(', ') },
              { label: 'Orotic Absent', value: (overview?.orotic_absent_genes ?? []).join(', ') },
              { label: 'X-Linked', value: (overview?.xlinked_genes ?? []).join(', ') },
            ].map(({ label, value }) => (
              <div key={label} style={{ background: '#f5f5f5', borderRadius: 8, padding: '10px 14px', textAlign: 'center' }}>
                <div style={{ fontSize: 14, fontWeight: 700 }}>{value}</div>
                <div style={{ fontSize: 11, color: '#666', marginTop: 2 }}>{label}</div>
              </div>
            ))}
          </div>
          <h3 style={{ fontSize: 16, marginBottom: 10 }}>All 8 Genes</h3>
          {Object.entries(GENE_COLORS).map(([g, col]) => (
            <GeneCard key={g} gene={g} color={col} info={GENE_INFO[g]} geneData={geneMap[g]} cohortData={cohortMap[g]} />
          ))}
        </div>
      )}

      {/* ── Gene Table Tab ── */}
      {tab === 'Gene Table' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1b5e20', color: '#fff' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Gene</th>
                <th style={{ padding: '8px 10px' }}>UCD Group</th>
                <th style={{ padding: '8px 10px' }}>Orotic Acid</th>
                <th style={{ padding: '8px 10px' }}>NBS</th>
                <th style={{ padding: '8px 10px' }}>Neonatal Crisis</th>
                <th style={{ padding: '8px 10px' }}>Key Pathognomonic</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Treatment</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Key Distinguisher</th>
              </tr>
            </thead>
            <tbody>
              {PHENOTYPE_ROWS.map((row, i) => (
                <tr key={row.gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{row.group}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 700,
                    color: row.orotic === 'ABSENT' ? '#b71c1c' : row.orotic === 'HIGH' ? '#6a1b9a' : '#388e3c' }}>{row.orotic}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', color: row.nbs === '✓' || row.nbs === '✓*' ? '#1565c0' : '#888', fontWeight: 700 }}>{row.nbs}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', color: row.neonatal === '✓' ? '#b71c1c' : '#888', fontWeight: 700 }}>{row.neonatal}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11, fontWeight: 600, color: '#1a237e' }}>{row.pathognomonic}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{row.rx}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{row.unique}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Pathway Map Tab ── */}
      {tab === 'Pathway Map' && (
        <div>
          <h3 style={{ fontSize: 15, marginBottom: 12 }}>UCD Groups — Metabolic Classification</h3>
          {Object.entries(breakdown?.ucd_groups ?? {}).map(([key, group]) => {
            const groupColorMap = {
              proximal_ucd: '#1b5e20', distal_ucd: '#1a237e',
              activator_ucd: '#e65100', transport_ucd: '#004d40',
            };
            const groupLabelMap = {
              proximal_ucd: '🟢 Proximal UCD (CPS1 + OTC)',
              distal_ucd: '🔵 Distal UCD (ASS1 + ASL + ARG1)',
              activator_ucd: '🟠 Activator UCD (NAGS)',
              transport_ucd: '⚫ Transport UCD (SLC25A15 + SLC25A13)',
            };
            const col = groupColorMap[key] || '#555';
            return (
              <div key={key} style={{ marginBottom: 16, border: `2px solid ${col}`, borderRadius: 8, padding: 16 }}>
                <h4 style={{ fontSize: 14, fontWeight: 700, marginBottom: 8, color: col }}>{groupLabelMap[key] || key}</h4>
                <div style={{ marginBottom: 6 }}><strong>Genes:</strong> {(group.genes ?? []).join(', ')}</div>
                <div style={{ fontSize: 13, marginBottom: 4 }}><strong>Description:</strong> {group.description}</div>
                {group.orotic_rule && <div style={{ fontSize: 13 }}><strong>Orotic acid rule:</strong> {group.orotic_rule}</div>}
                {group.nbs_detectable && <div style={{ fontSize: 13, color: '#1565c0', fontWeight: 600 }}>All genes NBS-detectable (tandem MS)</div>}
                {group.carglumate_critical && <div style={{ fontSize: 13, color: '#e65100', fontWeight: 600 }}>NCG (carglumic acid) COMPLETELY CURATIVE — EMPIRICAL USE MANDATORY</div>}
                {group.ethnic_founders && <div style={{ fontSize: 13 }}><strong>Ethnic founders:</strong> {group.ethnic_founders.join('; ')}</div>}
              </div>
            );
          })}

          <h3 style={{ fontSize: 15, marginTop: 20, marginBottom: 12 }}>Orotic Acid Diagnostic Algorithm</h3>
          {Object.entries(breakdown?.orotic_acid_algorithm ?? {}).map(([key, alg]) => {
            const algColorMap = { absent_low: '#b71c1c', markedly_elevated: '#6a1b9a', mild_normal: '#1565c0' };
            const algLabelMap = { absent_low: 'ABSENT / LOW → CPS1 or NAGS', markedly_elevated: 'MARKEDLY ELEVATED → OTC', mild_normal: 'NORMAL / MILD → Distal/Transport UCDs' };
            return (
              <div key={key} style={{ marginBottom: 12, border: `2px solid ${algColorMap[key] || '#555'}`, borderRadius: 8, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6, color: algColorMap[key] }}>{algLabelMap[key] || key}</div>
                <div style={{ fontSize: 13, marginBottom: 4 }}><strong>Genes:</strong> {(alg.genes ?? []).join(', ')}</div>
                <div style={{ fontSize: 13, marginBottom: 4 }}><strong>Action:</strong> {alg.action}</div>
                {alg.urgency && <div style={{ fontSize: 13, fontWeight: 700, color: '#b71c1c' }}>⚡ {alg.urgency}</div>}
              </div>
            );
          })}

          <h3 style={{ fontSize: 15, marginTop: 20, marginBottom: 12 }}>Distinguishing Features by Gene</h3>
          {(breakdown?.distinguishing_features ?? []).map((df, i) => (
            <div key={df.gene} style={{ display: 'flex', gap: 12, padding: '8px 12px', background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee', borderRadius: 4, marginBottom: 4 }}>
              <span style={{ background: GENE_COLORS[df.gene] || '#555', color: '#fff', fontWeight: 700, padding: '2px 8px', borderRadius: 4, fontSize: 12, minWidth: 80, textAlign: 'center' }}>{df.gene}</span>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#1a237e', marginBottom: 2 }}>{df.feature}</div>
                <div style={{ fontSize: 12, color: '#555' }}><strong>Action:</strong> {df.action}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Treatment Tab ── */}
      {tab === 'Treatment' && (
        <div>
          {breakdown?.treatment_principles && (
            <div style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: 15, marginBottom: 10 }}>Treatment Principles</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(280px,1fr))', gap: 12 }}>
                <div style={{ background: '#fce4ec', border: '2px solid #b71c1c', borderRadius: 8, padding: 12 }}>
                  <div style={{ fontWeight: 700, color: '#b71c1c', marginBottom: 6 }}>Emergency</div>
                  <div style={{ fontSize: 13 }}>{breakdown.treatment_principles.emergency}</div>
                </div>
                <div style={{ background: '#e8f5e9', border: '2px solid #1b5e20', borderRadius: 8, padding: 12 }}>
                  <div style={{ fontWeight: 700, color: '#1b5e20', marginBottom: 6 }}>Maintenance</div>
                  <div style={{ fontSize: 13 }}>{breakdown.treatment_principles.maintenance}</div>
                </div>
              </div>
              <h4 style={{ fontSize: 13, marginTop: 16, marginBottom: 8 }}>Gene-Specific Rules</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(280px,1fr))', gap: 10 }}>
                {Object.entries(breakdown.treatment_principles.gene_specific ?? {}).map(([gene, rule]) => (
                  <div key={gene} style={{ border: `2px solid ${GENE_COLORS[gene] || '#999'}`, borderRadius: 8, padding: 10 }}>
                    <div style={{ fontWeight: 700, color: GENE_COLORS[gene], marginBottom: 4, fontSize: 13 }}>{gene}</div>
                    <div style={{ fontSize: 12 }}>{rule}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 14, fontSize: 13, display: 'flex', gap: 20, flexWrap: 'wrap' }}>
                <div><strong style={{ color: '#1b5e20' }}>LTx fully curative:</strong> {(breakdown.treatment_principles.transplant_curative ?? []).join(', ')}</div>
                <div><strong style={{ color: '#e65100' }}>LTx partial benefit:</strong> {(breakdown.treatment_principles.transplant_partial ?? []).join(', ')}</div>
              </div>
            </div>
          )}

          <h3 style={{ fontSize: 15, marginBottom: 10 }}>Treatment Matrix</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1b5e20', color: '#fff' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Intervention</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Indications</th>
                <th style={{ padding: '8px 10px' }}>Mechanism</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Dose</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Cautions / CI</th>
              </tr>
            </thead>
            <tbody>
              {(definitions?.treatment_matrix ?? []).map((tm, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700, fontSize: 11 }}>{tm.intervention}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{tm.indications}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{tm.mechanism}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{tm.dose}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11, color: '#b71c1c' }}>{tm.contraindicated}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'Definitions' && (
        <div>
          <div style={{ background: '#e8eaf6', borderRadius: 8, padding: '10px 16px', marginBottom: 16, fontSize: 13 }}>
            <strong>Primary Cascade Split:</strong>{' '}{definitions?.primary_cascade_split ?? '—'}
          </div>

          <h3 style={{ fontSize: 15, marginBottom: 10 }}>Key Distinguishers</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(320px,1fr))', gap: 12, marginBottom: 20 }}>
            {(definitions?.key_distinguishers ?? []).map((kd, i) => (
              <div key={i} style={{ background: '#f3f4f6', borderRadius: 8, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4, color: '#1b5e20' }}>{kd.name}</div>
                <div style={{ fontSize: 12, color: '#333' }}>{kd.description}</div>
              </div>
            ))}
          </div>

          <h3 style={{ fontSize: 15, marginBottom: 10 }}>UCD Group Summary</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', gap: 12 }}>
            {Object.entries(definitions?.ucd_groups ?? {}).map(([group, info]) => {
              const groupColorMap2 = { proximal: '#1b5e20', distal: '#1a237e', activator: '#e65100', transport: '#004d40' };
              const col = groupColorMap2[group] || '#555';
              return (
                <div key={group} style={{ background: '#fff', border: `1px solid ${col}`, borderRadius: 8, padding: 12 }}>
                  <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6, textTransform: 'capitalize', color: col }}>
                    {group.toUpperCase()} UCD
                  </div>
                  <div style={{ fontSize: 12, marginBottom: 4 }}><strong>Genes:</strong> {(info.genes ?? []).join(', ')}</div>
                  <div style={{ fontSize: 12, color: '#555' }}><strong>Orotic:</strong> {info.orotic}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div style={{ marginTop: 24, fontSize: 11, color: '#aaa', textAlign: 'center' }}>
        Hereditary-Urea-Cycle-Disorder-Atlas · Seeds {overview?.seeds} ·
        CPS1/OTC (Proximal) + ASS1/ASL/ARG1 (Distal) + NAGS (Activator) + SLC25A15/SLC25A13 (Transport) ·
        {overview?.total_patients} patients
      </div>
    </div>
  );
}
