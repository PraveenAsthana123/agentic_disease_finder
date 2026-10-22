'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-thiamine-biotin-lipoate-cofactor-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Treatment', 'Definitions'];

const GENE_COLORS = {
  'SLC19A3': '#b71c1c',  // dark red         — BTBGD; bilateral BG necrosis PATHOGNOMONIC; biotin+thiamine CURATIVE
  'SLC19A2': '#1565c0',  // dark blue         — TRMA Rogers; megaloblastic+DM+SNHL triad; SNHL IRREVERSIBLE
  'TPK1':    '#4527a0',  // deep purple       — THMD1; TPP synthesis blocked; neonatal lactic acidosis; thiamine
  'SLC25A19':'#1b5e20',  // dark green        — Amish microcephaly; 2-OG aciduria; p.G177A founder
  'BTD':     '#e65100',  // deep orange       — biotinidase; NBS-detectable; biotin 5 mg CURATIVE
  'HLCS':    '#f57f17',  // amber-900         — MCD1; BTD activity NORMAL; neonatal; biotin-responsive
  'LIAS':    '#37474f',  // blue-grey         — lipoate synthesis; ALL 4 complexes; Leigh-like; KD contraindicated
  'LIPT1':   '#546e7a',  // blue-grey-lighter — lipoate attachment; BCKDH spared (vs LIAS); PDH+alphaKGDH+GCS
};

const GENE_INFO = {
  'SLC19A3':  { full: 'SLC19A3-496aa / AR / THTR2-Thiamine-Transporter-2 / Brain-specific-thiamine+biotin-transport / 2q37.3 / BTBGD-Bilateral-BG-PATHOGNOMONIC / Blood-thiamine-NORMAL / Biotin+Thiamine-COMPLETE-REVERSAL / Empirical-treatment-DO-NOT-WAIT-genetics', locus: '2q37.3', size: '496 aa / 55 kDa', inh: 'AR; THTR2 brain-specific thiamine+biotin transport; BTBGD; blood thiamine NORMAL (brain-selective); bilateral BG necrosis MRI; biotin 10 mg + thiamine 300 mg EMPIRICALLY' },
  'SLC19A2':  { full: 'SLC19A2-497aa / AR / THTR1-Thiamine-Transporter-1 / Intestinal+erythroid-thiamine / 1q24.2 / TRMA-Rogers-Syndrome / Megaloblastic-anemia+DM+SNHL-TRIAD / Autoantibodies-NEGATIVE / Thiamine-25-100mg / SNHL-IRREVERSIBLE', locus: '1q24.2', size: '497 aa / 55 kDa', inh: 'AR; THTR1 intestinal+erythroid transport; TRMA megaloblastic+DM+SNHL triad; blood thiamine LOW; SNHL irreversible regardless of treatment' },
  'TPK1':     { full: 'TPK1-262aa / AR / Thiamine-pyrophosphokinase-1 / Thiamine→TPP-phosphorylation / 7q35 / THMD1 / Neonatal-encephalopathy+lactic-acidosis / Combined-TPP-dependent-failure-PDH+alphaKGDH+TKT+BCKDH / Thiamine-200-600mg', locus: '7q35', size: '262 aa / 30 kDa', inh: 'AR; thiamine→TPP phosphorylation blocked; THMD1 neonatal lactic acidosis; all TPP-dependent enzymes fail; thiamine 200-600 mg/day' },
  'SLC25A19': { full: 'SLC25A19-320aa / AR / DNC-Mitochondrial-TPP+dNTP-transporter / 17q25.3 / Amish-Microcephaly-607196 / THMD4 / HC-minus7to10SD-congenital / 2-Oxoglutarate-MARKEDLY-ELEVATED / p.G177A-Amish-founder / Prenatal-mandatory', locus: '17q25.3', size: '320 aa / 35 kDa', inh: 'AR; mitochondrial TPP+dNTP transporter; Amish microcephaly (HC -7 to -10 SD); THMD4 milder; 2-OG aciduria; p.G177A Amish founder' },
  'BTD':      { full: 'BTD-543aa / AR / Biotinidase / Biocytin-biotin-recycling / 3p25.1 / MCD-late-onset / NBS-DETECTABLE-fluorometric-DBS / Biotin-5-10mg-COMPLETELY-CURATIVE / Dermatitis+alopecia+optic-atrophy-PATHOGNOMONIC-triad / 1in60000', locus: '3p25.1', size: '543 aa / 60 kDa', inh: 'AR; biotinidase recycling failure; NBS-detectable (DBS enzyme assay); 1:60,000; biotin 5-10 mg completely curative if early; SNHL+optic atrophy IRREVERSIBLE if delayed' },
  'HLCS':     { full: 'HLCS-726aa / AR / Holocarboxylase-synthetase / Biotin-attachment-to-PCC+MCC+PC+ACC / 21q22.13 / MCD1-neonatal / BTD-activity-NORMAL-CRITICAL-DDx / NBS-MISSES / Biotin-10-20mg / Neonatal-ketoacidosis+lactic-acidosis+rash+alopecia', locus: '21q22.13', size: '726 aa / 80 kDa', inh: 'AR; biotin attachment failure to all 4 carboxylases; BTD activity NORMAL (DDx from BTD); neonatal onset; NBS misses; biotin 10-20 mg/day' },
  'LIAS':     { full: 'LIAS-370aa / AR / Lipoic-acid-synthase / De-novo-lipoate-synthesis-radical-SAM / 4q32.3 / ALL-4-lipoate-complexes-PDH+alphaKGDH+BCKDH+GCS / Leigh-like / Lactic+2OG+glycine+alloisoleucine / KD-CONTRAINDICATED / ALA-60-100mg-kg', locus: '4q32.3', size: '370 aa / 42 kDa', inh: 'AR; de novo lipoate synthesis (radical SAM); ALL 4 complexes (PDH+alphaKGDH+BCKDH+GCS) deficient; KD CONTRAINDICATED; ALA 60-100 mg/kg/day' },
  'LIPT1':    { full: 'LIPT1-376aa / AR / Lipoyl-transferase-1 / Lipoate-attachment-to-PDH-E2+alphaKGDH-E2+GCS-H-protein / 2q11.2 / PDH+alphaKGDH+GCS-deficient / BCKDH-SPARED-vs-LIAS / Alloisoleucine-LESS-elevated / KD-CONTRAINDICATED / ALA+thiamine', locus: '2q11.2', size: '376 aa / 42 kDa', inh: 'AR; lipoate attachment to PDH/alphaKGDH/GCS; BCKDH relatively spared (vs LIAS — key DDx); KD CONTRAINDICATED; ALA + thiamine' },
};

const PHENOTYPE_ROWS = [
  { gene: 'SLC19A3', group: 'Thiamine — BTBGD',            nbs: '✗', lactic: '±',  biomarker: 'Bilateral BG MRI',       vitamin: 'Biotin 10mg+Thiamine 300mg', unique: 'Bilateral caudate+putamen MRI PATHOGNOMONIC; blood thiamine NORMAL (brain-specific); fever-triggered; EMPIRICAL TREATMENT before genetics; COMPLETE MRI reversal; biotin+thiamine BOTH required' },
  { gene: 'SLC19A2', group: 'Thiamine — TRMA Rogers',       nbs: '✗', lactic: '—',  biomarker: 'Blood thiamine LOW',      vitamin: 'Thiamine 25-100mg',          unique: 'Megaloblastic anemia+non-autoimmune DM+SNHL triad PATHOGNOMONIC; autoantibodies NEGATIVE; anemia+DM reverse; SNHL IRREVERSIBLE — hearing aids at diagnosis; ring sideroblasts BM' },
  { gene: 'TPK1',    group: 'Thiamine — THMD1',             nbs: '✗', lactic: '↑↑', biomarker: 'Combined TPP enzymes low', vitamin: 'Thiamine 200-600mg',         unique: 'All TPP-dependent enzymes fail (PDH+alphaKGDH+TKT+BCKDH); neonatal/infantile lactic acidosis; Leigh-like BG; high-dose thiamine empirically in ANY infant with lactic acidosis+BG disease' },
  { gene: 'SLC25A19',group: 'Thiamine — Amish micro/THMD4', nbs: '✗', lactic: '↑',  biomarker: '2-OG aciduria MARKEDLY ↑', vitamin: 'Thiamine 100-300mg',         unique: '2-oxoglutarate MARKEDLY elevated PATHOGNOMONIC; Amish microcephaly HC -7 to -10 SD + p.G177A; prenatal testing mandatory Amish families; THMD4 milder partial thiamine response' },
  { gene: 'BTD',     group: 'Biotin — late-onset MCD',      nbs: '✓', lactic: '—',  biomarker: 'BTD enzyme <10%',         vitamin: 'Biotin 5-10mg',             unique: 'NBS-DETECTABLE; 1:60,000; BTD enzyme activity DBS <10% (profound) 10-30% (partial); dermatitis+alopecia+optic atrophy triad PATHOGNOMONIC; COMPLETELY CURATIVE if before optic/hearing damage; biotin lab-test interference FDA 2019 warning' },
  { gene: 'HLCS',    group: 'Biotin — neonatal MCD1',       nbs: '✗', lactic: '↑↑', biomarker: 'BTD activity NORMAL',     vitamin: 'Biotin 10-20mg',            unique: 'BTD activity NORMAL — CRITICAL DDx from BTD; neonatal ketoacidosis; 3-OHisovalerate+3-MCG+propionate ALL elevated; NBS misses; biotin 10-20 mg rapidly curative; biotin overrides reduced HLCS affinity by substrate saturation' },
  { gene: 'LIAS',    group: 'Lipoate — synthesis defect',   nbs: '✗', lactic: '↑↑', biomarker: 'ALL 4 complexes low',     vitamin: 'ALA 60-100mg/kg+thiamine',  unique: 'ALL 4 lipoate-dependent complexes (PDH+alphaKGDH+BCKDH+GCS) simultaneously severely deficient PATHOGNOMONIC; alloisoleucine elevated (BCKDH); Leigh MRI; KD ABSOLUTELY CONTRAINDICATED; ALA partially effective' },
  { gene: 'LIPT1',   group: 'Lipoate — attachment defect',  nbs: '✗', lactic: '↑↑', biomarker: 'BCKDH relatively spared', vitamin: 'ALA 60-100mg/kg+thiamine',  unique: 'PDH+alphaKGDH+GCS deficient but BCKDH relatively SPARED (vs LIAS — discriminating DDx); alloisoleucine LESS elevated; neonatal encephalopathy; KD CONTRAINDICATED; ALA+thiamine; molecular diagnosis required' },
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
  const groupLabel = {
    SLC19A3:  'BTBGD-BilatBG-PATHOGNOMONIC-Biotin+Thiamine-COMPLETE-REVERSAL-EMPIRICAL',
    SLC19A2:  'TRMA-Rogers-Megaloblastic+DM+SNHL-TRIAD-SNHL-IRREVERSIBLE-Thiamine',
    TPK1:     'THMD1-NeonatalLacticAcidosis-TPP-AllComplexesFail-Thiamine-200-600mg',
    SLC25A19: 'AmishMicrocephaly-THMD4-2OG-Aciduria-pG177A-Founder-PrenatalMandatory',
    BTD:      'BiotinidaseMCD-NBS-DETECTABLE-Biotin5mg-CURATIVE-Dermatitis-Alopecia-OpticAtrophy',
    HLCS:     'MCD1-NeonatalKetoacidosis-BTD-NORMAL-NBS-MISSES-Biotin10mg-Responsive',
    LIAS:     'LipoateSynthesis-ALL4Complexes-LeighLike-KD-CONTRAINDICATED-ALA',
    LIPT1:    'LipoateAttachment-BCKDH-SPARED-PDH+alphaKGDH+GCS-KD-CONTRAINDICATED',
  }[gene] || gene;

  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 12, marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          <span style={{ background: color, color: '#fff', fontWeight: 700, padding: '3px 10px', borderRadius: 5, fontSize: 15 }}>{gene}</span>
          <span style={{ marginLeft: 10, fontSize: 12, color: '#555' }}>{groupLabel}</span>
        </div>
        <span style={{ fontSize: 11, color: '#888' }}>seed {cohort?.seed ?? '—'} · n={cohort?.n ?? 40}</span>
      </div>
      <div style={{ marginTop: 8, fontSize: 12, color: '#333' }}>{info?.full}</div>
      <div style={{ marginTop: 6, fontSize: 12 }}>
        <strong>Locus:</strong> {info?.locus} &nbsp;|&nbsp;
        <strong>Size:</strong> {info?.size} &nbsp;|&nbsp;
        <strong>Inheritance:</strong> {info?.inh}
      </div>
      {cohort && (
        <div style={{ marginTop: 6, fontSize: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Badge text={`Cofactor: ${cohort.cofactor_group ?? '—'}`} color={color} />
          <Badge text={`NBS: ${cohort.nbs_detected_pct > 0.5 ? '✓ detected' : '✗ missed'}`} color={cohort.nbs_detected_pct > 0.5 ? '#1565c0' : '#757575'} />
          <Badge text={`Tx response: ${Math.round((cohort.treatment_response_pct ?? 0) * 100)}%`} color={cohort.treatment_response_pct > 0.7 ? '#2e7d32' : cohort.treatment_response_pct > 0.4 ? '#e65100' : '#b71c1c'} />
          {cohort.lactic_acidosis_pct > 0.3 && <Badge text={`Lactic acidosis: ${Math.round(cohort.lactic_acidosis_pct * 100)}%`} color="#6a1b9a" />}
          {cohort.mri_bg_lesions_pct > 0.3 && <Badge text={`MRI BG lesions: ${Math.round(cohort.mri_bg_lesions_pct * 100)}%`} color="#1a237e" />}
          {cohort.megaloblastic_pct > 0.3 && <Badge text={`Megaloblastic: ${Math.round(cohort.megaloblastic_pct * 100)}%`} color="#0d47a1" />}
          {cohort.diabetes_pct > 0.3 && <Badge text={`Diabetes: ${Math.round(cohort.diabetes_pct * 100)}%`} color="#4a148c" />}
          {cohort.hearing_loss_pct > 0.2 && <Badge text={`SNHL: ${Math.round(cohort.hearing_loss_pct * 100)}%`} color="#37474f" />}
          {cohort.dermatitis_pct > 0.2 && <Badge text={`Dermatitis: ${Math.round(cohort.dermatitis_pct * 100)}%`} color="#e65100" />}
          {cohort.alopecia_pct > 0.2 && <Badge text={`Alopecia: ${Math.round(cohort.alopecia_pct * 100)}%`} color="#bf360c" />}
          <Badge text={`Onset: ~${cohort.median_onset_months}mo`} color="#37474f" />
          <Badge text={`Reversibility: ${cohort.reversible_with_treatment}`} color={cohort.reversible_with_treatment?.toLowerCase().includes('completely') ? '#2e7d32' : '#e65100'} />
        </div>
      )}
    </div>
  );
}

export default function HeredThiamineBiotinLipoateCofactorAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Thiamine-Biotin-Lipoate Cofactor Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  const cohortMap = {};
  (overview?.gene_cohorts ?? []).forEach(c => { cohortMap[c.gene] = c; });

  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: 1100, margin: '0 auto', padding: '20px 16px' }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
        🧬 Hereditary-Thiamine-Biotin-Lipoate-Cofactor-Transport-Metabolism-Atlas
      </h1>
      <p style={{ fontSize: 13, color: '#555', marginBottom: 4 }}>
        Complete 8-Gene Reference · SLC19A3 · SLC19A2 · TPK1 · SLC25A19 · BTD · HLCS · LIAS · LIPT1
      </p>
      <p style={{ fontSize: 12, color: '#777', marginBottom: 16 }}>
        Seeds {overview?.seeds} · {overview?.total_patients} patients · Water-Soluble Vitamin Cofactor Disorders (Thiamine + Biotin + Lipoate)
      </p>

      {/* DDx Banner */}
      <div style={{ background: '#fce4ec', border: '2px solid #b71c1c', borderRadius: 8, padding: '10px 16px', marginBottom: 12, fontSize: 13 }}>
        <strong>⚡ EMPIRICAL TREATMENT RULE (SLC19A3/BTBGD):</strong>{' '}
        Bilateral BG lesions on MRI during febrile encephalopathy → start biotin 10 mg + thiamine 300 mg IMMEDIATELY — DO NOT wait for genetics. MRI lesions CAN completely reverse.
      </div>
      <div style={{ background: '#e8eaf6', border: '2px solid #3949ab', borderRadius: 8, padding: '10px 16px', marginBottom: 16, fontSize: 13 }}>
        <strong>⚡ Primary DDx Split:</strong>{' '}
        {overview?.atlas_summary?.critical_ddx_rule ?? 'Thiamine (SLC19A3/SLC19A2/TPK1/SLC25A19) vs Biotin (BTD/HLCS) vs Lipoate (LIAS/LIPT1)'}
        <div style={{ marginTop: 6, fontSize: 12, color: '#555' }}>
          <strong>NBS detects:</strong> {(overview?.atlas_summary?.nbs_detected_genes ?? []).join(', ') || 'BTD only (fluorometric DBS assay)'}&nbsp;&nbsp;|&nbsp;&nbsp;
          <strong>Completely reversible:</strong> {(overview?.atlas_summary?.completely_reversible ?? []).join(', ')}&nbsp;&nbsp;|&nbsp;&nbsp;
          <strong>KD contraindicated:</strong> LIAS, LIPT1, TPK1 (PDH-dependent patients)
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 20, border: 'none', cursor: 'pointer', fontSize: 13,
            background: tab === t ? '#b71c1c' : '#e8eaf6', color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'Overview' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))', gap: 12, marginBottom: 20 }}>
            {[
              { label: 'Total Patients', value: overview?.total_patients },
              { label: 'Genes', value: overview?.genes_count },
              { label: 'Seeds', value: overview?.seeds },
              { label: 'Thiamine Group', value: (overview?.atlas_summary?.thiamine_group_genes ?? []).length },
              { label: 'Biotin Group', value: (overview?.atlas_summary?.biotin_group_genes ?? []).length },
              { label: 'Lipoate Group', value: (overview?.atlas_summary?.lipoate_group_genes ?? []).length },
              { label: 'NBS-Detected', value: (overview?.atlas_summary?.nbs_detected_genes ?? []).join(', ') || 'BTD' },
              { label: 'All AR', value: overview?.atlas_summary?.all_ar_inheritance ? 'Yes' : '—' },
            ].map(({ label, value }) => (
              <div key={label} style={{ background: '#f5f5f5', borderRadius: 8, padding: '10px 14px', textAlign: 'center' }}>
                <div style={{ fontSize: 18, fontWeight: 700 }}>{value}</div>
                <div style={{ fontSize: 11, color: '#666', marginTop: 2 }}>{label}</div>
              </div>
            ))}
          </div>
          <h3 style={{ fontSize: 16, marginBottom: 10 }}>All 8 Genes</h3>
          {Object.entries(GENE_COLORS).map(([g, col]) => (
            <GeneCard key={g} gene={g} color={col} info={GENE_INFO[g]} cohort={cohortMap[g]} />
          ))}
        </div>
      )}

      {/* ── Gene Table Tab ── */}
      {tab === 'Gene Table' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#b71c1c', color: '#fff' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Gene</th>
                <th style={{ padding: '8px 10px' }}>Group / Disease</th>
                <th style={{ padding: '8px 10px' }}>NBS</th>
                <th style={{ padding: '8px 10px' }}>Lactic Acidosis</th>
                <th style={{ padding: '8px 10px' }}>Key Biomarker</th>
                <th style={{ padding: '8px 10px' }}>Vitamin Rx</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Key Distinguisher</th>
              </tr>
            </thead>
            <tbody>
              {PHENOTYPE_ROWS.map((row, i) => (
                <tr key={row.gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{row.group}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', color: row.nbs === '✓' ? '#2e7d32' : '#888', fontWeight: row.nbs === '✓' ? 700 : 400 }}>{row.nbs}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', color: row.lactic === '↑↑' ? '#b71c1c' : row.lactic === '↑' ? '#e65100' : '#888', fontWeight: row.lactic !== '—' ? 700 : 400 }}>{row.lactic}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11, fontWeight: 600, color: '#1a237e' }}>{row.biomarker}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{row.vitamin}</td>
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
          {Object.entries(breakdown?.pathway_groups ?? {}).map(([key, group]) => {
            const colorMap = { thiamine: '#b71c1c', biotin: '#e65100', lipoate: '#37474f' };
            const groupColor = key.includes('thiamine') ? colorMap.thiamine : key.includes('biotin') ? colorMap.biotin : colorMap.lipoate;
            const groupLabel = key.includes('thiamine') ? '🔴 Thiamine Transport/Metabolism' : key.includes('biotin') ? '🟠 Biotin Recycling/Attachment' : '⚫ Lipoate Synthesis/Attachment';
            return (
              <div key={key} style={{ marginBottom: 20, border: `2px solid ${groupColor}`, borderRadius: 8, padding: 16 }}>
                <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8, color: groupColor }}>{groupLabel}</h3>
                <div style={{ marginBottom: 8 }}><strong>Genes:</strong> {(group.genes ?? []).join(', ')}</div>
                <div style={{ marginBottom: 6, fontSize: 13 }}><strong>Mechanism:</strong> {group.mechanism}</div>
                <div style={{ marginBottom: 6, fontSize: 13 }}><strong>Shared Feature:</strong> {group.shared_feature}</div>
                <div style={{ fontSize: 13, color: '#555' }}><strong>Key DDx:</strong> {group.key_ddx}</div>
              </div>
            );
          })}

          <h3 style={{ fontSize: 15, marginBottom: 10 }}>10-Step Diagnostic Cascade</h3>
          <ol style={{ paddingLeft: 20, fontSize: 13 }}>
            {(breakdown?.diagnostic_cascade ?? []).map((step, i) => (
              <li key={i} style={{ marginBottom: 8, lineHeight: 1.5 }}>{step}</li>
            ))}
          </ol>

          {breakdown?.empirical_treatment_rule && (
            <div style={{ background: '#fce4ec', border: '2px solid #b71c1c', borderRadius: 8, padding: 12, marginTop: 16, fontSize: 13 }}>
              <strong>⚡ Empirical Treatment Rule:</strong> {breakdown.empirical_treatment_rule}
            </div>
          )}
          {breakdown?.keto_diet_contraindications && (
            <div style={{ background: '#fff8e1', border: '2px solid #f57f17', borderRadius: 8, padding: 12, marginTop: 10, fontSize: 13 }}>
              <strong>⚠️ Ketogenic Diet Contraindication:</strong> {breakdown.keto_diet_contraindications}
            </div>
          )}
        </div>
      )}

      {/* ── Treatment Tab ── */}
      {tab === 'Treatment' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))', gap: 16, marginBottom: 20 }}>
            {(breakdown?.treatment_pathways ?? []).map(tp => (
              <div key={tp.gene} style={{ border: `2px solid ${GENE_COLORS[tp.gene] || '#999'}`, borderRadius: 8, padding: 14 }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[tp.gene], marginBottom: 6, fontSize: 14 }}>
                  {tp.gene} — {tp.disease}
                </div>
                <div style={{ fontSize: 12, marginBottom: 4 }}><strong>First-line:</strong> {tp.first_line}</div>
                <div style={{ fontSize: 12, marginBottom: 4 }}><strong>Precision:</strong> {tp.precision_therapy}</div>
                <div style={{ fontSize: 12, marginBottom: 4, color: '#b71c1c' }}><strong>⚠ Avoid:</strong> {tp.avoid}</div>
                <div style={{ fontSize: 12, color: '#555' }}><strong>Monitor:</strong> {tp.monitoring}</div>
              </div>
            ))}
          </div>

          <h3 style={{ fontSize: 15, marginBottom: 10 }}>Treatment Matrix</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#37474f', color: '#fff' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Intervention</th>
                <th style={{ padding: '8px 10px' }}>Mechanism</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Dose</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Notes/Cautions</th>
              </tr>
            </thead>
            <tbody>
              {(definitions?.treatment_matrix ?? []).map((tm, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700 }}>{tm.intervention}</td>
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
                <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4, color: '#b71c1c' }}>{kd.name}</div>
                <div style={{ fontSize: 12, color: '#333' }}>{kd.description}</div>
              </div>
            ))}
          </div>

          <h3 style={{ fontSize: 15, marginBottom: 10 }}>Cofactor Groups</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(280px,1fr))', gap: 12 }}>
            {Object.entries(definitions?.cofactor_groups ?? {}).map(([group, info]) => (
              <div key={group} style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6, textTransform: 'capitalize', color: group === 'thiamine' ? '#b71c1c' : group === 'biotin' ? '#e65100' : '#37474f' }}>
                  {group.toUpperCase()} group
                </div>
                <div style={{ fontSize: 12, marginBottom: 4 }}><strong>Genes:</strong> {(info.genes ?? []).join(', ')}</div>
                <div style={{ fontSize: 12, color: '#555' }}><strong>Treatment:</strong> {info.treatment}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ marginTop: 24, fontSize: 11, color: '#aaa', textAlign: 'center' }}>
        Hereditary-Thiamine-Biotin-Lipoate-Cofactor-Atlas · Seeds {overview?.seeds} ·
        SLC19A3/SLC19A2/TPK1/SLC25A19 (Thiamine) + BTD/HLCS (Biotin) + LIAS/LIPT1 (Lipoate) ·
        {overview?.total_patients} patients
      </div>
    </div>
  );
}
