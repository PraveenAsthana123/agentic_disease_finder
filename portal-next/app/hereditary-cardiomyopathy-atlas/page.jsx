'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  MYBPC3: '#1565c0',  // deep blue — most common HCM, MyBP-C thick filament
  MYH7:   '#b71c1c',  // deep red — 2nd HCM, dominant negative, Arg403Gln malignant
  TNNT2:  '#e65100',  // deep orange — troponin T, high SCD low LVH
  LMNA:   '#4a148c',  // deep purple — DCM + CCD, nuclear envelope, Padua score
  SCN5A:  '#006064',  // dark cyan — Nav1.5, Brugada, LQTS3, flecainide ABSOLUTE CI
  DSP:    '#2e7d32',  // deep green — desmoplakin, ARVC left-dominant, woolly hair
  PKP2:   '#f57f17',  // amber — plakophilin-2, most common ARVC, epsilon wave
  PLN:    '#4e342e',  // deep brown — phospholamban, Arg14del Dutch founder, transplant
};

const GENE_DISEASE = {
  MYBPC3: 'HCM AD Haploinsufficiency — Most Common 35-40% — Mavacamten FDA 2022 Obstructive HCM — ICD SCD Risk Score — Incomplete Penetrance 5th-6th Decade',
  MYH7:   'HCM AD Dominant Negative — 2nd Most Common 25-35% — Arg403Gln Malignant High SCD — Mavacamten Obstructive — Myectomy Preferred Young',
  TNNT2:  'HCM+DCM AD — Arg92Trp HIGH SCD Despite MILD LVH — Cardiac MRI LGE Mandatory — Dual Phenotype HCM/DCM Same Gene',
  LMNA:   'DCM+CCD AD — ICD Padua Score ≥4 Regardless LVEF — CCD Precedes DCM — AF High Stroke Risk — Non-Missense Highest Risk',
  SCN5A:  'Brugada/LQTS3/DCM AD — Flecainide ABSOLUTE CI Brugada — Fever Trigger — Quinidine Brugada — Mexiletine LQTS3 — Brugadrugs.org Mandatory',
  DSP:    'ARVC Left-Dominant AD — Woolly Hair + Keratoderma PATHOGNOMONIC — LV LGE Characteristic — Carvajal Biallelic DCM — Exercise MANDATORY Restriction',
  PKP2:   'ARVC RV-Dominant AD — Most Common ARVC 40% — Epsilon Wave Major Criterion — LBBB VT — Task Force Criteria — Exercise ABSOLUTE Prohibition',
  PLN:    'DCM AD — Arg14del Dutch/Belgian Founder 1/400-800 NL — SERCA2a Constitutive Inhibition — ICD Mandatory — Cardiac Transplantation Frequent',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /AVOID|CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|MISSES|EMERGENCY|NOT.Routine|OPPOSITE|PATHOGNOMONIC/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE|REQUIRED|PROTOCOL|STAT|FIRST|TRIAL|PHASE|ENROL|ELIGIBLE|Preferred|RESTRICTION|SCORE|Padua/i.test(text);
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
  const { aggregate_stats: s, top_alerts, genes } = data;

  const statRows = [
    ['Total patients', s.total_patients],
    ['Mean diagnostic delay (all genes)', `${s.mean_dx_delay_months} mo`],
    ['ICD implanted (any gene)', `${s.icd_implanted_pct}%`],
    ['Exercise restricted', `${s.exercise_restricted_pct}%`],
    ['Cascade tested', `${s.cascade_tested_pct}%`],
    ['Cardiac MRI performed', `${s.cardiac_mri_done_pct}%`],
    ['LGE detected (any pattern)', `${s.lge_detected_pct}%`],
    // MYBPC3
    ['MYBPC3 — obstructive HCM (LVOTO >30 mmHg)', `${s.mybpc3_obstructive_pct}%`],
    ['MYBPC3 — mavacamten prescribed', `${s.mybpc3_mavacamten_pct}%`],
    ['MYBPC3 — ICD implanted', `${s.mybpc3_icd_pct}%`],
    ['MYBPC3 — beta-blocker prescribed', `${s.mybpc3_beta_blocker_pct}%`],
    ['MYBPC3 — septal reduction therapy', `${s.mybpc3_septal_reduction_pct}%`],
    // MYH7
    ['MYH7 — Arg403Gln malignant variant', `${s.myh7_arg403gln_pct}%`],
    ['MYH7 — obstructive HCM', `${s.myh7_obstructive_pct}%`],
    ['MYH7 — mavacamten prescribed', `${s.myh7_mavacamten_pct}%`],
    ['MYH7 — ICD implanted', `${s.myh7_icd_pct}%`],
    ['MYH7 — extensive LGE on CMR', `${s.myh7_lge_extensive_pct}%`],
    // TNNT2
    ['TNNT2 — mild LVH (<15 mm) with HCM', `${s.tnnt2_mild_lvh_pct}%`],
    ['TNNT2 — high SCD risk classification', `${s.tnnt2_high_scd_pct}%`],
    ['TNNT2 — extensive LGE on CMR', `${s.tnnt2_lge_extensive_pct}%`],
    ['TNNT2 — ICD implanted', `${s.tnnt2_icd_pct}%`],
    ['TNNT2 — DCM sacubitril/valsartan', `${s.tnnt2_dcm_sacubitril_pct}%`],
    // LMNA
    ['LMNA — CCD precedes DCM', `${s.lmna_ccd_precedes_dcm_pct}%`],
    ['LMNA — AV block documented', `${s.lmna_av_block_pct}%`],
    ['LMNA — pacemaker implanted', `${s.lmna_pacemaker_pct}%`],
    ['LMNA — ICD implanted', `${s.lmna_icd_pct}%`],
    ['LMNA — Padua score ≥4', `${s.lmna_padua_score_4plus_pct}%`],
    ['LMNA — atrial fibrillation', `${s.lmna_af_pct}%`],
    ['LMNA — midwall LGE on CMR', `${s.lmna_lge_midwall_pct}%`],
    ['LMNA — non-missense (truncating) variant', `${s.lmna_non_missense_pct}%`],
    // SCN5A
    ['SCN5A — Brugada syndrome phenotype', `${s.scn5a_brugada_pct}%`],
    ['SCN5A — LQTS3 phenotype', `${s.scn5a_lqts3_pct}%`],
    ['SCN5A — flecainide received (should be 0)', `${s.scn5a_flecainide_received_pct}%`],
    ['SCN5A — fever-triggered VF', `${s.scn5a_fever_triggered_pct}%`],
    ['SCN5A — quinidine prescribed (Brugada)', `${s.scn5a_quinidine_pct}%`],
    ['SCN5A — mexiletine prescribed (LQTS3)', `${s.scn5a_mexiletine_pct}%`],
    // DSP
    ['DSP — woolly hair', `${s.dsp_woolly_hair_pct}%`],
    ['DSP — palmoplantar keratoderma', `${s.dsp_keratoderma_pct}%`],
    ['DSP — LV involvement', `${s.dsp_lv_involvement_pct}%`],
    ['DSP — LV LGE on CMR', `${s.dsp_lv_lge_pct}%`],
    ['DSP — Carvajal (biallelic)', `${s.dsp_carvajal_pct}%`],
    ['DSP — exercise restricted', `${s.dsp_exercise_restricted_pct}%`],
    // PKP2
    ['PKP2 — epsilon wave present', `${s.pkp2_epsilon_wave_pct}%`],
    ['PKP2 — RV dysfunction', `${s.pkp2_rv_dysfunction_pct}%`],
    ['PKP2 — LBBB-morphology VT', `${s.pkp2_lbbb_vt_pct}%`],
    ['PKP2 — Task Force Criteria positive', `${s.pkp2_task_force_positive_pct}%`],
    ['PKP2 — exercise restricted', `${s.pkp2_exercise_restricted_pct}%`],
    ['PKP2 — MLPA performed', `${s.pkp2_mlpa_done_pct}%`],
    // PLN
    ['PLN — Arg14del Dutch/Belgian founder', `${s.pln_arg14del_pct}%`],
    ['PLN — ICD implanted', `${s.pln_icd_pct}%`],
    ['PLN — transplant listed', `${s.pln_transplant_listed_pct}%`],
    ['PLN — transplanted', `${s.pln_transplanted_pct}%`],
    ['PLN — sacubitril/valsartan', `${s.pln_sacubitril_pct}%`],
    ['PLN — non-sustained VT', `${s.pln_nsvt_pct}%`],
  ];

  return (
    <div>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>
        {data.title || 'Hereditary Cardiomyopathy Atlas'}
      </h2>
      <p style={{ color: '#555', marginBottom: 16 }}>
        {data.subtitle || '8 genes · 320 patients · seeds 1518–1525'}
      </p>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        {(genes || []).map(g => (
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
              <td style={{ padding: '5px 8px', fontWeight: 600, color: '#1565c0', textAlign: 'right' }}>{val}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>
        Critical Alerts ({(top_alerts || []).length})
      </h3>
      {(top_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
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
              <summary style={{ cursor: 'pointer', fontSize: 12, color: '#1565c0', marginBottom: 6 }}>
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
          <tr style={{ background: '#1565c0', color: '#fff' }}>
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
        Precision Treatment &amp; Investigation Matrix
      </h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#b71c1c', color: '#fff' }}>
            {['Gene', 'AVOID / Contraindicated', 'MANDATORY Investigation / Treatment', 'Special Rule'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[
            ['MYBPC3', 'Competitive sport · Positive inotropes · Nitrates in obstructive HCM (reduce preload → worsen LVOTO)', 'Beta-blocker FIRST LINE · Mavacamten if LVOTO >30 mmHg · ICD per SCD risk score · Cardiac MRI LGE · Annual family cascade screening', 'Mavacamten EF monitoring mandatory — stop if EF <50%; incomplete penetrance means normal echo at 30 does NOT exclude future HCM'],
            ['MYH7', 'Digoxin (positive inotrope worsens LVOTO) · Dihydropyridine CCB in obstructive HCM · Alcohol ablation preferred over myectomy (myectomy is superior in young patients)', 'Beta-blocker FIRST LINE · Mavacamten obstructive HCM · Septal myectomy preferred young patients · ICD per SCD risk · Annual CMR LGE', 'Arg403Gln = malignant — lower ICD threshold; rod domain variants can cause DCM not HCM — full characterisation mandatory'],
            ['TNNT2', 'Wall thickness alone for SCD risk assessment (MISLEADING in TNNT2 — mild LVH high SCD) · Assuming DCM rules out HCM (same gene, both phenotypes)', 'Cardiac MRI LGE MANDATORY (TNNT2-HCM even mild LVH) · ICD lower threshold · SCD risk score must include LGE · Standard HFrEF GDMT for TNNT2-DCM', 'Arg92Trp: near-normal echo + extensive LGE + high SCD risk — do not be falsely reassured by wall thickness'],
            ['LMNA', 'Waiting for LVEF <35% before ICD if Padua ≥4 · Class I AAD (flecainide) in LMNA — proarrhythmic · Rhythm control deferred in LMNA-AF', 'Padua score calculation mandatory · ICD at Padua ≥4 regardless LVEF · Annual ECG CCD monitoring · CMR LGE · Anticoagulate AF · Cascade urgent', 'Non-missense (truncating) = highest Padua risk; CCD often precedes DCM — may need pacing years before DCM develops'],
            ['SCN5A', 'Flecainide/propafenone/ajmaline ABSOLUTE CI in Brugada · Symptomatic treatment of fever without antipyretics · Prescribing ANY drug without Brugadrugs.org check', 'Brugadrugs.org check before any new prescription · Antipyretics immediately for fever · ICD for symptomatic Brugada · Quinidine Brugada · Mexiletine LQTS3 · Ajmaline challenge monitored setting only', 'Ajmaline/flecainide IS the Brugada diagnostic provocation test but in monitored setting with defibrillator READY — the drug that provokes diagnosis is the same drug that is ABSOLUTELY CI for ongoing therapy'],
            ['DSP', 'Exercise (strongest ARVC trigger even in DSP phenotype-negative gene-positive) · Flecainide/Class IC in ARVC · Assuming normal RV = normal DSP (LV is primary)', 'Cardiac MRI with LGE MANDATORY (LV LGE characteristic) · Exercise restriction IMMEDIATE · ICD if LV LGE + dysfunction + NSVT · Task Force Criteria applied to LV not just RV · Skin/hair examination mandatory', 'Woolly hair + keratoderma = DSP until proven otherwise; biallelic DSP = Carvajal (DCM not ARVC) — completely different management'],
            ['PKP2', 'Exercise (most potent PKP2-ARVC disease trigger and arrhythmia risk) · Flecainide/sotalol in ARVC · Relying on standard echo alone (CMR required)', 'Exercise restriction ABSOLUTE (gene-positive regardless phenotype) · Signal-averaged ECG for epsilon wave · CMR RV characterisation · MLPA if seronegative strong phenotype · ICD primary prevention per risk score · Amiodarone if AAD needed', 'Epsilon wave (signal-averaged ECG V1-V3) = MAJOR Task Force Criterion; LBBB VT = RV origin; PKP2 truncating = highest risk'],
            ['PLN', 'Waiting for LVEF <35% before ICD in PLN (NSVT is sufficient ICD indication) · Withholding GDMT (sacubitril/valsartan mandatory) · Assuming stable disease trajectory in PLN (progressive)', 'ICD mandatory at LVEF <35% or NSVT · Maximally tolerated sacubitril/valsartan + carvedilol + eplerenone + SGLT2i · Early transplant listing discussion · LVAD bridge if needed · Cascade testing ALL first-degree relatives in Dutch/Belgian families', 'Arg14del Dutch founder: in Netherlands/Belgium, screen all first-degree relatives urgently; homozygous = neonatal lethal DCM — emergency transplant'],
          ].map(([gene, avoid, mandatory, special], i) => (
            <tr key={gene} style={{ background: i % 2 === 0 ? '#fce4ec' : '#fff' }}>
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
          <h3 style={{ fontSize: 14, fontWeight: 700, color: '#1565c0', marginBottom: 8 }}>
            {i + 1}. {d.term}
          </h3>
          <p style={{ fontSize: 13, color: '#444', lineHeight: 1.6 }}>{d.definition}</p>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryCardiomyopathyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-cardiomyopathy-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if ((tab === 'Gene Table' || tab === 'Clinical Atlas') && !breakdown) {
      fetch(`${API}/api/hereditary-cardiomyopathy-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
    if (tab === 'Definitions' && !definitions) {
      fetch(`${API}/api/hereditary-cardiomyopathy-atlas/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
  }, [tab, breakdown, definitions]);

  return (
    <div style={{ padding: '1.5rem', fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ marginBottom: 8 }}>
        <span style={{
          background: '#1565c0', color: '#fff', borderRadius: 6,
          padding: '4px 12px', fontSize: 12, fontWeight: 600,
        }}>
          Hereditary Cardiomyopathy Atlas
        </span>
        <span style={{ marginLeft: 10, fontSize: 12, color: '#888' }}>
          8 genes · 320 patients · seeds 1518–1525
        </span>
      </div>
      <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
        Hereditary-Cardiomyopathy-Atlas — Complete 8-Gene Hereditary Cardiomyopathy Reference
      </h1>
      <p style={{ fontSize: 13, color: '#666', marginBottom: 16 }}>
        MYBPC3 · MYH7 · TNNT2 · LMNA · SCN5A · DSP · PKP2 · PLN
        — HCM Mavacamten FDA 2022, LMNA Padua Score ICD, Brugada Flecainide ABSOLUTE CI,
        DSP Woolly Hair PATHOGNOMONIC, ARVC Exercise Restriction, PLN Arg14del Dutch Founder Transplant
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
              background: tab === t ? '#1565c0' : '#f0f0f0',
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
