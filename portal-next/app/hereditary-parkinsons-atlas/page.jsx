'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  LRRK2:   '#1a237e',  // deep indigo — most common genetic PD, kinase trials
  PRKN:    '#1b5e20',  // deep green — most common AR PD, MLPA mandatory
  PINK1:   '#004d40',  // deep teal — mitophagy pathway, exercise therapeutic
  SNCA:    '#4a148c',  // deep purple — alpha-synuclein, gene dosage
  'DJ-1':  '#37474f',  // dark slate — rare, mild, oxidative stress sensor
  GBA:     '#e65100',  // deep orange — most common risk gene, ambroxol
  VPS35:   '#880e4f',  // deep magenta — retromer D620N, late onset
  ATP13A2: '#bf360c',  // deep burnt orange — Kufor-Rakeb, MRI iron PATHOGNOMONIC
};

const GENE_DISEASE = {
  LRRK2:   'PARK8 AD — G2019S Kinase Inhibitor Trials — Ashkenazi/North-African Mandatory Testing — Incomplete Penetrance 25-80%',
  PRKN:    'PARK2 AR Most-Common-AR-PD — MLPA MANDATORY Exon Rearrangements 50% Missed by Sequencing — Early Dyskinesias Expected',
  PINK1:   'PARK6 AR — PINK1-Parkin Mitophagy Pathway — Similar PRKN Clinically — Exercise Therapeutic — Psychiatric Screen',
  SNCA:    'PARK1/4 AD — Gene Dosage: Duplication Mild / Triplication Severe+Dementia — A53T Greek Founder — Anti-Synuclein Trials',
  'DJ-1':  'PARK7 AR Rare — Mild Tremor Slow Progression — L166P Most Common — Oxidative Stress Sensor — MRI Normal',
  GBA:     'Most Common PD Risk Gene — Heterozygous 5-8x Risk NOT Gaucher — Biallelic = Gaucher+PD — Ambroxol Trial — Rapid Cognition',
  VPS35:   'PARK17 AD — D620N Sole Pathogenic Variant 95% — Retromer Complex — Late Onset — Similar Sporadic PD',
  ATP13A2: 'PARK9 AR Kufor-Rakeb — MRI Iron Putamen PATHOGNOMONIC — Pyramidal Signs — Supranuclear Gaze Palsy — Dementia Prominent',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /AVOID|CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|MISSES|PATHOGNOMONIC|NOT.Gaucher/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE|REQUIRED|PROTOCOL|STAT|FIRST|TRIAL|PHASE|ENROL|ELIGIBLE|GENE.THERAPY/i.test(text);
  const bg = isCI ? '#b71c1c' : isWarning ? '#e65100' : '#1565c0';
  return (
    <div style={{
      background: bg, color: '#fff', borderRadius: 6, padding: '6px 12px',
      marginBottom: 8, fontSize: 13, lineHeight: 1.4,
    }}>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const { aggregate_stats: s, top_alerts, gene_summaries: genes } = data;

  const statRows = [
    ['Total patients', s.total_patients],
    ['Mean diagnostic delay (all genes)', `${s.mean_dx_delay_months} mo`],
    ['Levodopa response (any gene)', `${s.levodopa_response_pct}%`],
    ['MLPA performed (PRKN/SNCA/VPS35)', `${s.mlpa_performed_pct}%`],
    ['Exercise programme enrolled', `${s.exercise_enrolled_pct}%`],
    ['Trial eligible (any)', `${s.trial_eligible_pct}%`],
    ['ATP13A2 — MRI iron in basal ganglia', `${s.mri_iron_abnormal_pct_atp13a2}%`],
    ['PRKN/PINK1 — early dyskinesias', `${s.early_dyskinesias_pct_prkn_pink1}%`],
    ['LRRK2 — G2019S variant', `${s.lrrk2_g2019s_pct}%`],
    ['LRRK2 — kinase inhibitor trial eligible', `${s.lrrk2_kinase_trial_eligible_pct}%`],
    ['LRRK2 — levodopa excellent response', `${s.lrrk2_levodopa_excellent_pct}%`],
    ['LRRK2 — tremor predominant', `${s.lrrk2_tremor_predominant_pct}%`],
    ['LRRK2 — penetrance counselled', `${s.lrrk2_penetrance_counselled_pct}%`],
    ['LRRK2 — cascade tested', `${s.lrrk2_cascade_tested_pct}%`],
    ['PRKN — exon rearrangement', `${s.prkn_exon_rearrangement_pct}%`],
    ['PRKN — MLPA performed', `${s.prkn_mlpa_performed_pct}%`],
    ['PRKN — levodopa response', `${s.prkn_levodopa_response_pct}%`],
    ['PRKN — early dyskinesias', `${s.prkn_early_dyskinesias_pct}%`],
    ['PRKN — dopamine agonist first', `${s.prkn_dopamine_agonist_first_pct}%`],
    ['PRKN — exercise enrolled', `${s.prkn_exercise_enrolled_pct}%`],
    ['PINK1 — levodopa response', `${s.pink1_levodopa_response_pct}%`],
    ['PINK1 — psychiatric comorbidity', `${s.pink1_psychiatric_comorbidity_pct}%`],
    ['PINK1 — exercise enrolled', `${s.pink1_exercise_enrolled_pct}%`],
    ['PINK1 — MRI normal', `${s.pink1_mri_normal_pct}%`],
    ['PINK1 — sleep benefit', `${s.pink1_sleep_benefit_pct}%`],
    ['SNCA — duplication (2 copies)', `${s.snca_duplication_pct}%`],
    ['SNCA — triplication (3 copies)', `${s.snca_triplication_pct}%`],
    ['SNCA — dementia developed', `${s.snca_dementia_developed_pct}%`],
    ['SNCA — MLPA performed', `${s.snca_mlpa_performed_pct}%`],
    ['SNCA — anti-synuclein trial eligible', `${s.snca_anti_synuclein_trial_pct}%`],
    ['DJ-1 — L166P variant', `${s.dj1_l166p_pct}%`],
    ['DJ-1 — levodopa response', `${s.dj1_levodopa_response_pct}%`],
    ['DJ-1 — MRI normal', `${s.dj1_mri_normal_pct}%`],
    ['DJ-1 — very slow progression', `${s.dj1_slow_progression_pct}%`],
    ['GBA — N370S variant', `${s.gba_n370s_pct}%`],
    ['GBA — L444P severe variant', `${s.gba_l444p_pct}%`],
    ['GBA — heterozygous (PD risk)', `${s.gba_heterozygous_pct}%`],
    ['GBA — ambroxol trial eligible', `${s.gba_ambroxol_trial_pct}%`],
    ['GBA — annual MoCA performed', `${s.gba_moca_annual_pct}%`],
    ['GBA — cognitive decline present', `${s.gba_cognitive_decline_pct}%`],
    ['VPS35 — D620N variant', `${s.vps35_d620n_pct}%`],
    ['VPS35 — levodopa response', `${s.vps35_levodopa_response_pct}%`],
    ['VPS35 — cascade tested', `${s.vps35_cascade_tested_pct}%`],
    ['VPS35 — late onset (>50)', `${s.vps35_late_onset_pct}%`],
    ['ATP13A2 — MRI iron basal ganglia', `${s.atp13a2_mri_iron_pct}%`],
    ['ATP13A2 — pyramidal signs', `${s.atp13a2_pyramidal_signs_pct}%`],
    ['ATP13A2 — supranuclear gaze palsy', `${s.atp13a2_supranuclear_gaze_pct}%`],
    ['ATP13A2 — dementia prominent', `${s.atp13a2_dementia_pct}%`],
    ['ATP13A2 — A746T Jordanian founder', `${s.atp13a2_a746t_pct}%`],
  ];

  return (
    <div>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>{data.title || 'Hereditary Parkinson\'s Disease Atlas'}</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>{data.subtitle || '8 genes · 320 patients · seeds 1502–1509'}</p>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        {genes.map(g => (
          <div key={g.gene} style={{
            background: GENE_COLORS[g.gene] || '#1565c0', color: '#fff', borderRadius: 8,
            padding: '10px 16px', minWidth: 120,
          }}>
            <div style={{ fontWeight: 700, fontSize: 15 }}>{g.gene}</div>
            <div style={{ fontSize: 11, opacity: 0.85 }}>{g.locus} · {g.aa}</div>
            <div style={{ fontSize: 11, opacity: 0.85 }}>{(g.inheritance || '').split('—')[0].trim()}</div>
            <div style={{ fontSize: 11, opacity: 0.9, marginTop: 4 }}>{g.n_patients} pts</div>
          </div>
        ))}
      </div>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Cohort Statistics</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, marginBottom: 24 }}>
        <tbody>
          {statRows.map(([label, val]) => (
            <tr key={label} style={{ borderBottom: '1px solid #eee' }}>
              <td style={{ padding: '5px 8px', color: '#333' }}>{label}</td>
              <td style={{ padding: '5px 8px', fontWeight: 600, color: '#1a237e', textAlign: 'right' }}>{val}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>
        Critical Alerts ({top_alerts.length})
      </h3>
      {top_alerts.map((a, i) => <AlertBadge key={i} text={a} />)}
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;

  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Per-Gene Breakdown</h2>
      {breakdown.map(g => (
        <div key={g.gene} style={{
          border: `2px solid ${GENE_COLORS[g.gene] || '#1565c0'}`,
          borderRadius: 10, marginBottom: 24, overflow: 'hidden',
        }}>
          <div style={{
            background: GENE_COLORS[g.gene] || '#1565c0', color: '#fff',
            padding: '10px 16px',
          }}>
            <span style={{ fontWeight: 700, fontSize: 16 }}>{g.gene}</span>
            <span style={{ marginLeft: 12, fontSize: 13 }}>{g.protein}</span>
          </div>
          <div style={{ padding: 16 }}>
            <p style={{ fontSize: 12, color: '#555', marginBottom: 8 }}>
              <strong>Locus:</strong> {g.locus} · <strong>Size:</strong> {g.aa} ({g.kDa}) ·
              <strong> OMIM gene:</strong> {g.omim_gene} · <strong>Disease:</strong> {g.omim_disease}
            </p>
            <p style={{ fontSize: 12, color: '#555', marginBottom: 8 }}>
              <strong>Inheritance:</strong> {g.inheritance}
            </p>
            <p style={{ fontSize: 12, color: '#444', marginBottom: 8 }}>
              <strong>Mean onset:</strong> {g.mean_onset_years} yr ·
              <strong> Mean dx delay:</strong> {g.mean_dx_delay_months} mo ·
              <strong> M:</strong> {g.sex_distribution?.M} / <strong>F:</strong> {g.sex_distribution?.F}
            </p>
            <p style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>{(g.alias || '').slice(0, 400)}…</p>
            <details style={{ marginTop: 8 }}>
              <summary style={{ cursor: 'pointer', fontSize: 12, color: '#1a237e', marginBottom: 6 }}>
                Gene-Class Mechanistic Detail
              </summary>
              <p style={{ fontSize: 12, color: '#444', marginTop: 6 }}>{g.gene_class}</p>
            </details>
            <div style={{ marginTop: 10 }}>
              <strong style={{ fontSize: 12 }}>Aetiology distribution:</strong>
              {Object.entries(g.etiology_counts || {}).map(([et, cnt]) => (
                <div key={et} style={{ fontSize: 11, color: '#555', marginTop: 2 }}>
                  {et}: <strong>{cnt}</strong>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 10 }}>
              {(g.key_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;

  const rows = breakdown.map(g => ({
    gene: g.gene,
    locus: g.locus,
    aa: g.aa,
    inh: (g.inheritance || '').split('—')[0].trim(),
    disease: GENE_DISEASE[g.gene] || '',
    pts: g.n_patients,
    onset: `${g.mean_onset_years}yr`,
  }));

  return (
    <div style={{ overflowX: 'auto' }}>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 12 }}>Clinical Atlas Summary</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#1a237e', color: '#fff' }}>
            {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease / Key Rule', 'Pts', 'Onset'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.gene} style={{ background: i % 2 === 0 ? '#f8f8f8' : '#fff' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[r.gene] || '#1565c0' }}>{r.gene}</td>
              <td style={{ padding: '7px 10px' }}>{r.locus}</td>
              <td style={{ padding: '7px 10px' }}>{r.aa}</td>
              <td style={{ padding: '7px 10px' }}>{r.inh}</td>
              <td style={{ padding: '7px 10px', fontSize: 11 }}>{r.disease}</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{r.pts}</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{r.onset}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginTop: 24, marginBottom: 12 }}>
        Precision Treatment & Investigation Matrix
      </h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#4a148c', color: '#fff' }}>
            {['Gene', 'AVOID / Contraindicated', 'MANDATORY Investigation / Treatment', 'Special Rule'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[
            ['LRRK2',   'Assuming 100% penetrance pre-test counselling', 'Population-targeted testing (Ashkenazi/N.African) · Kinase inhibitor trial enrolment · Annual UPDRS', 'G2019S: penetrance 25% at 59y, 80% at 79y — NOT deterministic; urine pRab10 as trial biomarker'],
            ['PRKN',    'Missing exon rearrangements (sequencing alone) · Levodopa as first-line in young', 'MLPA / aCGH for exon copy number · Dopamine agonist first (<50y) · Structured exercise programme', 'Exon deletions/duplications = 50% of variants — WES and Sanger are BLIND to these; must perform MLPA'],
            ['PINK1',   'No specific contraindications', 'Aerobic exercise programme · Psychiatric screening (anxiety/depression) · MLPA if single variant found', 'Clinical phenotype nearly identical to PRKN — distinguish only by genetics; MRI normal (if iron seen, reconsider)'],
            ['SNCA',    'Missing copy number (sequencing alone misses multiplications)', 'MLPA or digital PCR for copy number · Annual MoCA (triplication) · Anti-synuclein trial enrolment', 'Gene dosage = phenotype: 3 copies typical PD; 4 copies severe PD+dementia (DLB-like); A53T Greek founder'],
            ['DJ-1',    'No specific CI; avoid unnecessary polypharmacy', 'Levodopa (excellent response) · Aerobic exercise (antioxidant) · Oxidative stress research panels', 'Mildest AR PD; MRI normal — iron on MRI rules out DJ-1, prompts ATP13A2/NBIA workup'],
            ['GBA',     'Assuming heterozygous = Gaucher disease (it does NOT)', 'GBA testing ALL PD patients · Annual MoCA · Ambroxol trial enrolment · Gaucher specialist if biallelic', 'GBA heterozygous = PD risk modifier only (5-8x) NOT Gaucher; L444P > N370S cognitive severity'],
            ['VPS35',   'Missing D620N with standard panel (check VPS35 specifically)', 'Targeted D620N testing · Retromer pathway research trials · First-degree cascade testing', 'D620N = 95% of VPS35-PD; late onset; clinically indistinguishable from sporadic PD — genetics only distinguisher'],
            ['ATP13A2', 'Assuming typical PD without MRI (iron is diagnostic) · Missing pyramidal exam', 'MRI brain (T2/T2* for iron) MANDATORY · Eye movement exam (supranuclear gaze) · Annual cognitive + pyramidal', 'Iron in putamen/caudate T2 PATHOGNOMONIC; pyramidal + oculomotor + dementia = KRS triad; MRI DDx from PRKN/PINK1'],
          ].map(([gene, avoid, mandatory, special], i) => (
            <tr key={gene} style={{ background: i % 2 === 0 ? '#e8eaf6' : '#fff' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[gene] || '#1565c0' }}>{gene}</td>
              <td style={{ padding: '7px 10px', color: '#b71c1c', fontWeight: 600 }}>{avoid}</td>
              <td style={{ padding: '7px 10px', color: '#1b5e20', fontWeight: 600 }}>{mandatory}</td>
              <td style={{ padding: '7px 10px', color: '#555' }}>{special}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.clinical_definitions || data.definitions || [];
  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Clinical Definitions</h2>
      {defs.map((d, i) => (
        <div key={i} style={{
          border: '1px solid #e0e0e0', borderRadius: 8,
          marginBottom: 16, padding: 16,
        }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, color: '#1a237e', marginBottom: 8 }}>
            {i + 1}. {d.term}
          </h3>
          <p style={{ fontSize: 13, color: '#444', lineHeight: 1.6 }}>{d.definition}</p>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryParkinsonsAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-parkinsons-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if ((tab === 'Gene Table' || tab === 'Clinical Atlas') && !breakdown) {
      fetch(`${API}/api/hereditary-parkinsons-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
    if (tab === 'Definitions' && !definitions) {
      fetch(`${API}/api/hereditary-parkinsons-atlas/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
  }, [tab, breakdown, definitions]);

  return (
    <div style={{ padding: '1.5rem', fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ marginBottom: 8 }}>
        <span style={{
          background: '#1a237e', color: '#fff', borderRadius: 6,
          padding: '4px 12px', fontSize: 12, fontWeight: 600,
        }}>
          Hereditary Parkinson&#39;s Disease Atlas
        </span>
        <span style={{ marginLeft: 10, fontSize: 12, color: '#888' }}>
          8 genes · 320 patients · seeds 1502–1509
        </span>
      </div>
      <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
        Hereditary-Parkinson&#39;s-Atlas — Complete 8-Gene Hereditary Parkinson&#39;s Disease Reference
      </h1>
      <p style={{ fontSize: 13, color: '#666', marginBottom: 16 }}>
        LRRK2 · PRKN · PINK1 · SNCA · DJ-1 · GBA · VPS35 · ATP13A2
        — kinase inhibitor trials, MLPA mandatory, mitophagy pathway, gene dosage, ambroxol, retromer, Kufor-Rakeb MRI iron
      </p>

      {error && (
        <div style={{ background: '#ffebee', border: '1px solid #ef9a9a', borderRadius: 6, padding: 12, marginBottom: 16 }}>
          Error: {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: tab === t ? '#1a237e' : '#f0f0f0',
              color: tab === t ? '#fff' : '#333',
              fontWeight: tab === t ? 700 : 400,
              fontSize: 13,
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Gene Table' && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
