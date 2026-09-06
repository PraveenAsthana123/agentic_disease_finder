'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  F5:        '#7b1fa2',  // deep purple — Factor V Leiden, most common, OCP 35x
  F2:        '#1565c0',  // deep blue — Prothrombin G20210A, 3'UTR mechanism
  SERPINC1:  '#b71c1c',  // deep red — AT-III, highest risk, heparin resistance
  PROC:      '#e65100',  // deep orange — Protein C, warfarin necrosis MANDATORY
  PROS1:     '#2e7d32',  // deep green — Protein S, OCP confound, Type I/II/III
  MTHFR:    '#00695c',  // dark teal — MTHFR, hyperhomocysteinaemia, NOT routine
  THBD:      '#4527a0',  // deep indigo — Thrombomodulin, rare, aHUS overlap
  SERPINE1:  '#6d4c41',  // deep brown — PAI-1, fibrinolysis, bleeding vs thrombosis
};

const GENE_DISEASE = {
  F5:       'FVL AD — APC Resistance p.R534Q — OCP 35x Risk ABSOLUTE CI — Homozygous 50-80x Lifelong Anticoagulation — DOAC Preferred',
  F2:       'Prothrombin G20210A AD — 3\'UTR mRNA Stability 30% Elevated Prothrombin — WES May Miss 3\'UTR — 2-5x VTE Risk',
  SERPINC1: 'AT-III Deficiency AD — Highest Single-Gene VTE Risk 10-50x — Type II HBS Heparin Resistance AT-Concentrate MANDATORY',
  PROC:     'Protein C AD — Warfarin Skin Necrosis Heparin-Bridge MANDATORY — Neonatal Purpura Fulminans Homozygous EMERGENCY',
  PROS1:    'Protein S AD — OCP Dramatically Reduces Free PS Test-Off-OCP — Pregnancy Test 3M Postpartum — Type I/II/III Classification',
  MTHFR:   'MTHFR AR C677T Thermolabile — Hyperhomocysteinaemia — Folate 5-MTHF — NOT Routine VTE Screen per NICE',
  THBD:     'Thrombomodulin AD Rare — PC Activation Impaired TM-Thrombin Complex — aHUS Complement Overlap Eculizumab',
  SERPINE1: 'PAI-1 4G/4G Elevated Impaired Fibrinolysis VTE — Complete Deficiency AR = BLEEDING Opposite Phenotype — NOT Routine Screen',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /AVOID|CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|MISSES|EMERGENCY|NOT.Routine|OPPOSITE/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE|REQUIRED|PROTOCOL|STAT|FIRST|TRIAL|PHASE|ENROL|ELIGIBLE|CONCENTRATE|BRIDGE/i.test(text);
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
    ['PE event (any form)', `${s.pe_event_pct}%`],
    ['DVT (isolated)', `${s.dvt_event_pct}%`],
    ['Anticoagulation — DOAC', `${s.anticoagulation_on_doac_pct}%`],
    ['Warfarin skin necrosis (all genes)', `${s.warfarin_necrosis_pct}%`],
    ['OCP-associated VTE', `${s.ocp_associated_pct}%`],
    ['Pregnancy loss', `${s.pregnancy_loss_pct}%`],
    ['Heparin resistance (SERPINC1 Type II HBS)', `${s.heparin_resistance_pct}%`],
    ['Cascade tested (any gene)', `${s.cascade_tested_pct}%`],
    ['Homocysteine elevated (any gene)', `${s.homocysteine_elevated_pct}%`],
    ['MTHFR — folate supplemented', `${s.mthfr_folate_supplemented_pct}%`],
    // F5
    ['F5 — OCP-associated event', `${s.f5_ocp_associated_pct}%`],
    ['F5 — Homozygous Leiden', `${s.f5_homozygous_leiden_pct}%`],
    ['F5 — APC resistance ratio tested', `${s.f5_apc_resistance_tested_pct}%`],
    ['F5 — cascade tested', `${s.f5_cascade_tested_pct}%`],
    ['F5 — DOAC prescribed', `${s.f5_doac_prescribed_pct}%`],
    // F2
    ['F2 — OCP-associated event', `${s.f2_ocp_associated_pct}%`],
    ['F2 — WES missed G20210A (3\'UTR)', `${s.f2_wes_missed_pct}%`],
    ['F2 — targeted F2 assay performed', `${s.f2_targeted_assay_pct}%`],
    ['F2 — cascade tested', `${s.f2_cascade_tested_pct}%`],
    // SERPINC1
    ['SERPINC1 — Type II HBS heparin resistance', `${s.serpinc1_heparin_resistance_pct}%`],
    ['SERPINC1 — AT-III concentrate used', `${s.serpinc1_at_concentrate_used_pct}%`],
    ['SERPINC1 — functional chromogenic assay', `${s.serpinc1_functional_assay_pct}%`],
    ['SERPINC1 — acquired deficiency excluded', `${s.serpinc1_acquired_excluded_pct}%`],
    ['SERPINC1 — Type I quantitative', `${s.serpinc1_type_i_pct}%`],
    // PROC
    ['PROC — warfarin skin necrosis', `${s.proc_warfarin_necrosis_pct}%`],
    ['PROC — heparin bridge given', `${s.proc_heparin_bridge_given_pct}%`],
    ['PROC — neonatal purpura fulminans', `${s.proc_neonatal_purpura_pct}%`],
    ['PROC — chromogenic PC assay', `${s.proc_chromogenic_assay_pct}%`],
    ['PROC — pregnancy loss', `${s.proc_pregnancy_loss_pct}%`],
    // PROS1
    ['PROS1 — OCP confound at time of testing', `${s.pros1_ocp_confound_pct}%`],
    ['PROS1 — tested off OCP (correct)', `${s.pros1_tested_off_ocp_pct}%`],
    ['PROS1 — free PS measured', `${s.pros1_free_ps_tested_pct}%`],
    ['PROS1 — total PS measured', `${s.pros1_total_ps_tested_pct}%`],
    ['PROS1 — MLPA performed', `${s.pros1_mlpa_performed_pct}%`],
    ['PROS1 — Type III (low free only)', `${s.pros1_type_iii_pct}%`],
    // MTHFR
    ['MTHFR — homocysteine elevated', `${s.mthfr_homocysteine_elevated_pct}%`],
    ['MTHFR — 5-MTHF prescribed', `${s.mthfr_five_mthf_prescribed_pct}%`],
    ['MTHFR — B12 checked', `${s.mthfr_b12_checked_pct}%`],
    ['MTHFR — B12 deficient', `${s.mthfr_b12_deficient_pct}%`],
    ['MTHFR — NICE guideline discussed', `${s.mthfr_nice_discussed_pct}%`],
    // THBD
    ['THBD — aHUS/TMA overlap', `${s.thbd_ahus_overlap_pct}%`],
    ['THBD — complement workup done', `${s.thbd_complement_workup_pct}%`],
    ['THBD — eculizumab eligible', `${s.thbd_eculizumab_eligible_pct}%`],
    ['THBD — specialist confirmed pathogenicity', `${s.thbd_specialist_confirmed_pct}%`],
    // SERPINE1
    ['SERPINE1 — 4G/4G homozygous thrombophilic', `${s.serpine1_four_g_four_g_pct}%`],
    ['SERPINE1 — complete deficiency (bleeding)', `${s.serpine1_bleeding_phenotype_pct}%`],
    ['SERPINE1 — PAI-1 activity tested', `${s.serpine1_pai1_activity_tested_pct}%`],
    ['SERPINE1 — metabolic syndrome comorbidity', `${s.serpine1_metabolic_syndrome_pct}%`],
  ];

  return (
    <div>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>{data.title || 'Hereditary Thrombophilia Atlas'}</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>{data.subtitle || '8 genes · 320 patients · seeds 1510–1517'}</p>

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
              <td style={{ padding: '5px 8px', fontWeight: 600, color: '#7b1fa2', textAlign: 'right' }}>{val}</td>
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
              <summary style={{ cursor: 'pointer', fontSize: 12, color: '#7b1fa2', marginBottom: 6 }}>
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
          <tr style={{ background: '#7b1fa2', color: '#fff' }}>
            {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease / Key Rule', 'Pts', 'Onset'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.gene} style={{ background: i % 2 === 0 ? '#f8f8f8' : '#fff' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[r.gene] || '#7b1fa2' }}>{r.gene}</td>
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
          <tr style={{ background: '#b71c1c', color: '#fff' }}>
            {['Gene', 'AVOID / Contraindicated', 'MANDATORY Investigation / Treatment', 'Special Rule'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[
            ['F5',       'Estrogen-containing OCP (35x VTE risk with FVL) · Assuming APC resistance without DNA confirmation', 'APC resistance ratio APTT-based · DNA confirmation p.R534Q · LMWH in pregnancy · Cascade testing first-degree relatives · DOAC preferred long-term', 'Heterozygous 5-7x; homozygous 50-80x — lifelong anticoagulation post-first-event; OCP ABSOLUTE CI; DOAC non-inferior to warfarin'],
            ['F2',       'Estrogen-containing OCP (amplified thrombotic risk) · Standard WES alone (may miss 3\'UTR)', 'Targeted F2 G20210A assay or comprehensive panel · Progesterone-only contraception · LMWH in pregnancy · Cascade first-degree relatives', 'G20210A is in the 3\'UTR — standard WES may not report it; targeted assay essential; 2-5x VTE risk; lower risk than FVL'],
            ['SERPINC1', 'Heparin monotherapy in Type II HBS (ineffective) · Antigenic-only AT-III assay (misses Type II)', 'Functional (chromogenic) AT-III assay FIRST · AT-III concentrate (Thrombate III/ATryn) perioperatively · Exclude acquired causes before hereditary diagnosis', 'Highest single-gene VTE risk (10-50x); Type II HBS = heparin resistance — AT-III concentrate MANDATORY; lifelong anticoagulation post-first-event'],
            ['PROC',     'Warfarin WITHOUT heparin bridge (skin necrosis) · Clot-based Protein C assay (misses Type II)', 'LMWH bridge MANDATORY with warfarin initiation ≥5 days · Chromogenic PC assay · Ceprotin for neonatal purpura · Heparin + PC concentrate in homozygous neonate EMERGENCY', 'Warfarin skin necrosis mechanism: PC short half-life (6-8h) drops before procoagulant factors — DOAC avoids entirely; neonatal purpura fulminans = EMERGENCY'],
            ['PROS1',    'Testing ON OCP (false-positive PS deficiency) · Testing during pregnancy (physiologically low) · Antigenic-only assay (misses Type II)', 'Test ≥3 months after stopping OCP · Test ≥3 months postpartum · Measure free AND total PS + functional APC cofactor · MLPA if sequencing non-diagnostic', 'OCP dramatically reduces free PS → false-positive test common; Type III (low free only) is commonest; MLPA for large deletions'],
            ['MTHFR',   'MTHFR genotyping as routine thrombophilia screen (NICE: NOT recommended) · Treating genotype without measuring homocysteine', 'Measure plasma homocysteine (>15 μmol/L = risk) · 5-MTHF 400-800 μg/day if elevated · Check B12 and B6 · Recheck homocysteine 3 months post-supplementation', 'MTHFR genotype alone is NOT a thrombophilia test; measure plasma homocysteine; NICE/BCSH explicitly recommend AGAINST routine MTHFR testing'],
            ['THBD',     'Assuming THBD variant pathogenic without specialist review · Missing TMA/aHUS in THBD patient', 'Specialist haematologist confirmation of pathogenicity · Functional TM-PC activation assay · Complement panel (C3/C4/CH50/FH/anti-FH) if TMA · Eculizumab if complement-mediated TMA', 'THBD is rare; plasma PC may be normal — functional TM-mediated PC activation assay needed; aHUS/TMA phenotype requires complement workup'],
            ['SERPINE1', 'Interpreting 4G/5G polymorphism as definitive thrombophilia screen result · Missing opposite bleeding phenotype of complete PAI-1 deficiency', 'Plasma PAI-1 activity (chromogenic, not antigen) · Metabolic syndrome management · Antifibrinolytics (tranexamic acid) for complete PAI-1 deficiency bleeding', '4G/4G = elevated PAI-1 → impaired fibrinolysis → modest thrombotic risk; complete deficiency (biallelic frameshift) = SEVERE BLEEDING — opposite phenotype'],
          ].map(([gene, avoid, mandatory, special], i) => (
            <tr key={gene} style={{ background: i % 2 === 0 ? '#fce4ec' : '#fff' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[gene] || '#7b1fa2' }}>{gene}</td>
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
          <h3 style={{ fontSize: 14, fontWeight: 700, color: '#7b1fa2', marginBottom: 8 }}>
            {i + 1}. {d.term}
          </h3>
          <p style={{ fontSize: 13, color: '#444', lineHeight: 1.6 }}>{d.definition}</p>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryThrombophiliaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-thrombophilia-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if ((tab === 'Gene Table' || tab === 'Clinical Atlas') && !breakdown) {
      fetch(`${API}/api/hereditary-thrombophilia-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
    if (tab === 'Definitions' && !definitions) {
      fetch(`${API}/api/hereditary-thrombophilia-atlas/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
  }, [tab, breakdown, definitions]);

  return (
    <div style={{ padding: '1.5rem', fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ marginBottom: 8 }}>
        <span style={{
          background: '#7b1fa2', color: '#fff', borderRadius: 6,
          padding: '4px 12px', fontSize: 12, fontWeight: 600,
        }}>
          Hereditary Thrombophilia Atlas
        </span>
        <span style={{ marginLeft: 10, fontSize: 12, color: '#888' }}>
          8 genes · 320 patients · seeds 1510–1517
        </span>
      </div>
      <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
        Hereditary-Thrombophilia-Atlas — Complete 8-Gene Hereditary Thrombophilia Reference
      </h1>
      <p style={{ fontSize: 13, color: '#666', marginBottom: 16 }}>
        F5 · F2 · SERPINC1 · PROC · PROS1 · MTHFR · THBD · SERPINE1
        — Factor V Leiden OCP 35x CI, AT-III heparin resistance, Protein C warfarin necrosis, Protein S OCP confound, MTHFR not routine, thrombomodulin aHUS, PAI-1 bleeding vs thrombosis
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
              background: tab === t ? '#7b1fa2' : '#f0f0f0',
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
