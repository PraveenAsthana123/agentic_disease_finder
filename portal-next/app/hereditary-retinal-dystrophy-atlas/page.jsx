'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  RPGR:   '#4a148c',  // deep purple — XLRP, ORF15 diagnostic gap
  USH2A:  '#1a237e',  // deep indigo — Usher 2A, dual sensory
  ABCA4:  '#e65100',  // deep orange — Stargardt, light restriction
  RHO:    '#880e4f',  // deep magenta — most common ADRP, P23H
  PRPF31: '#1b5e20',  // deep green — reduced penetrance RP11
  CRB1:   '#37474f',  // dark slate — RP12/LCA8, PPRPE sign
  CNGB3:  '#006064',  // deep teal — achromatopsia, stable, gene therapy
  BEST1:  '#bf360c',  // deep burnt orange — Best VMD2, EOG diagnostic
};

const GENE_DISEASE = {
  RPGR:   'RP3 XLR — ORF15 WES MISSES 50% — Specific Sequencing Mandatory — Females 20-25% Symptomatic — No Approved Therapy',
  USH2A:  'Usher 2A AR — RP + Congenital SNHL Moderate-Severe — NO Vestibular Dysfunction — c.2299delG — Joint ENT-Ophtho',
  ABCA4:  'Stargardt AR — Most Common Hereditary MD — FAF Diagnostic — AVOID Light + AVOID Vitamin A — N1868I Hypomorphic',
  RHO:    'RP4 AD — Most Common ADRP — P23H Class II Dominant Negative — Vitamin A 15000 IU Recommended — Avoid Beta-Carotene Smokers',
  PRPF31: 'RP11 AD — REDUCED PENETRANCE 60-80% — CNOT3 Modifier — Unaffected Carriers Common — Counselling Critical',
  CRB1:   'RP12/LCA8 AR — PPRPE PATHOGNOMONIC — Thick Retina OCT — NOT Luxturna (RPE65 Only) — Coats-Like Complication',
  CNGB3:  'Achromatopsia AR — STABLE No Blindness — FL-41 Lenses — Absent Photopic + Normal Scotopic ERG — Gene Therapy Phase 2/3',
  BEST1:  'Best VMD2 AD — EOG Arden Ratio PATHOGNOMONIC (Carriers Too) — ERG NORMAL — CNV Anti-VEGF — AR Bestrophinopathy',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /AVOID|CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|MISSES|NOT\s+Luxturna|STABLE|PATHOGNOMONIC|ABNORMAL/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE|REQUIRED|PROTOCOL|STAT|FIRST|TRIAL|PHASE|GENE.THERAPY|ELIGIBLE/i.test(text);
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
    ['RPGR — ORF15 variant (WES may miss)', `${s.rpgr_orf15_variant_pct}%`],
    ['RPGR — standard WES missed causal variant', `${s.rpgr_standard_wes_missed_pct}%`],
    ['RPGR — UV protection counselled', `${s.rpgr_uv_protection_counselled_pct}%`],
    ['RPGR — gene therapy trial referred', `${s.rpgr_trial_referred_pct}%`],
    ['USH2A — congenital SNHL', `${s.usha_snhl_congenital_pct}%`],
    ['USH2A — cochlear implant received', `${s.usha_ci_received_pct}%`],
    ['USH2A — joint ENT-ophthalmology clinic', `${s.usha_joint_clinic_pct}%`],
    ['USH2A — c.2299delG European founder variant', `${s.usha_c2299delG_pct}%`],
    ['USH2A — Vitamin A 15000 IU prescribed', `${s.usha_vit_a_pct}%`],
    ['ABCA4 — Stargardt phenotype', `${s.abca4_stargardt_pct}%`],
    ['ABCA4 — FAF performed', `${s.abca4_faf_performed_pct}%`],
    ['ABCA4 — FAF classic pisciform pattern', `${s.abca4_faf_classic_pct}%`],
    ['ABCA4 — light restriction counselled', `${s.abca4_light_restricted_pct}%`],
    ['ABCA4 — Vitamin A appropriately avoided', `${s.abca4_vit_a_avoided_pct}%`],
    ['ABCA4 — N1868I hypomorphic allele present', `${s.abca4_n1868i_pct}%`],
    ['RHO — P23H North American hotspot', `${s.rho_p23h_pct}%`],
    ['RHO — Class II misfolding (dominant negative)', `${s.rho_class_ii_pct}%`],
    ['RHO — Vitamin A 15000 IU prescribed', `${s.rho_vit_a_pct}%`],
    ['PRPF31 — reduced penetrance counselled', `${s.prpf31_penetrance_counselled_pct}%`],
    ['PRPF31 — unaffected carrier identified in family', `${s.prpf31_unaffected_carrier_family_pct}%`],
    ['PRPF31 — CNOT3 modifier testing offered', `${s.prpf31_cnot3_offered_pct}%`],
    ['CRB1 — PPRPE sign present on fundoscopy', `${s.crb1_pprpe_present_pct}%`],
    ['CRB1 — PPRPE sign noted/documented', `${s.crb1_pprpe_noted_pct}%`],
    ['CRB1 — thick retina on OCT', `${s.crb1_thick_oct_pct}%`],
    ['CRB1 — erroneously referred for Luxturna', `${s.crb1_luxturna_erroneously_referred_pct}%`],
    ['CNGB3 — complete achromatopsia', `${s.cngb3_complete_pct}%`],
    ['CNGB3 — FL-41 tinted lenses prescribed', `${s.cngb3_fl41_pct}%`],
    ['CNGB3 — scotopic ERG normal (confirmed)', `${s.cngb3_scotopic_normal_pct}%`],
    ['CNGB3 — gene therapy trial eligible', `${s.cngb3_gene_therapy_eligible_pct}%`],
    ['CNGB3 — counselled condition is stable (no blindness)', `${s.cngb3_counselled_stable_pct}%`],
    ['BEST1 — EOG performed', `${s.best1_eog_performed_pct}%`],
    ['BEST1 — EOG Arden ratio abnormal', `${s.best1_eog_abnormal_pct}%`],
    ['BEST1 — ERG normal (pan-retinal intact)', `${s.best1_erg_normal_pct}%`],
    ['BEST1 — CNV complication', `${s.best1_cnv_pct}%`],
    ['BEST1 — anti-VEGF for CNV prescribed', `${s.best1_anti_vegf_pct}%`],
    ['Cross-gene — gene therapy referred/enrolled', `${s.any_gene_therapy_referred_pct}%`],
    ['Cross-gene — annual ERG monitoring', `${s.any_annual_erg_pct}%`],
    ['Cross-gene — Vitamin A supplementation', `${s.any_vit_a_prescribed_pct}%`],
    ['Mean diagnostic delay (all genes)', `${s.mean_dx_delay_months} mo`],
  ];

  return (
    <div>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>{data.title}</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>{data.subtitle}</p>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        {genes.map(g => (
          <div key={g.gene} style={{
            background: GENE_COLORS[g.gene], color: '#fff', borderRadius: 8,
            padding: '10px 16px', minWidth: 120,
          }}>
            <div style={{ fontWeight: 700, fontSize: 15 }}>{g.gene}</div>
            <div style={{ fontSize: 11, opacity: 0.85 }}>{g.locus} · {g.aa}</div>
            <div style={{ fontSize: 11, opacity: 0.85 }}>{g.inheritance.split('—')[0].trim()}</div>
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
          border: `2px solid ${GENE_COLORS[g.gene]}`,
          borderRadius: 10, marginBottom: 24, overflow: 'hidden',
        }}>
          <div style={{
            background: GENE_COLORS[g.gene], color: '#fff',
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
              <strong> M:</strong> {g.sex_distribution.M} / <strong>F:</strong> {g.sex_distribution.F}
            </p>
            <p style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>{g.alias.slice(0, 400)}…</p>
            <details style={{ marginTop: 8 }}>
              <summary style={{ cursor: 'pointer', fontSize: 12, color: '#1565c0', marginBottom: 6 }}>
                Gene-Class Mechanistic Detail
              </summary>
              <p style={{ fontSize: 12, color: '#444', marginTop: 6 }}>{g.gene_class}</p>
            </details>
            <div style={{ marginTop: 10 }}>
              <strong style={{ fontSize: 12 }}>Aetiology distribution:</strong>
              {Object.entries(g.etiology_counts).map(([et, cnt]) => (
                <div key={et} style={{ fontSize: 11, color: '#555', marginTop: 2 }}>
                  {et}: <strong>{cnt}</strong>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 10 }}>
              {g.key_alerts.map((a, i) => <AlertBadge key={i} text={a} />)}
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
    inh: g.inheritance.split('—')[0].trim(),
    disease: GENE_DISEASE[g.gene],
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
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[r.gene] }}>{r.gene}</td>
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
            {['Gene', 'AVOID / Contraindicated', 'MANDATORY Investigation / Use', 'Special Rule'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[
            ['RPGR',   'High-dose Vit A (unproven); beta-carotene if smoker', 'ORF15-specific sequencing (not WES alone) · UV400 lenses · Annual ERG', 'Standard WES misses ORF15 frameshifts in 50% — specific protocol mandatory'],
            ['USH2A',  'Delay cochlear implant referral; late low-vision intervention', 'Audiogram + CI evaluation · Joint ENT-ophtho clinic · Orientation & mobility', 'First gene to test in AR-RP + SNHL; Vitamin A 15000 IU evidence-based'],
            ['ABCA4',  'Vitamin A (worsens A2E accumulation) · Excessive light', 'FAF (fundus autofluorescence) · UV400 lenses · Light restriction counselling', 'N1868I alone benign; only pathogenic in compound het with severe allele'],
            ['RHO',    'Vitamin A restriction (needed for chromophore) · Beta-carotene in smokers', 'Vitamin A 15000 IU/day (Berson evidence) · LFTs annual · Annual ERG+VF', 'P23H Class II dominant negative — gene therapy must knock down mutant allele'],
            ['PRPF31', 'Assume penetrance = 100% (it is 60-80%)', 'Genetic counselling for reduced penetrance · Predictive testing relatives · Annual ERG', 'CNOT3 modifier — unaffected carriers real — must explain before predictive testing'],
            ['CRB1',   'Referring for Luxturna (voretigene) — RPE65 only', 'Fundoscopy for PPRPE · OCT (thick retina) · Natural history study enrolment', 'PPRPE sign pathognomonic — document on fundoscopy; monitor for Coats-like exudates'],
            ['CNGB3',  'Telling patient they will go blind (rods intact)', 'FL-41 lenses · Dark wraparound sunglasses · Gene therapy trial enrolment', 'Stable condition — no progression to blindness — normal scotopic ERG distinguishes'],
            ['BEST1',  'Relying on ERG alone (ERG normal in Best) · Skipping carrier EOG', 'EOG Arden ratio · Relatives EOG screening · OCT-A if CNV suspected', 'EOG abnormal in ALL carriers; CNV → anti-VEGF (bevacizumab/aflibercept) responsive'],
          ].map(([gene, avoid, mandatory, special], i) => (
            <tr key={gene} style={{ background: i % 2 === 0 ? '#f3e5f5' : '#fff' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
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
  const { definitions } = data;
  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Clinical Definitions</h2>
      {definitions.map((d, i) => (
        <div key={i} style={{
          border: '1px solid #e0e0e0', borderRadius: 8,
          marginBottom: 16, padding: 16,
        }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, color: '#4a148c', marginBottom: 8 }}>
            {i + 1}. {d.term}
          </h3>
          <p style={{ fontSize: 13, color: '#444', lineHeight: 1.6 }}>{d.definition}</p>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryRetinalDystrophyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-retinal-dystrophy-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 'Gene Table' && !breakdown) {
      fetch(`${API}/api/hereditary-retinal-dystrophy-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
    if (tab === 'Definitions' && !definitions) {
      fetch(`${API}/api/hereditary-retinal-dystrophy-atlas/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
    if (tab === 'Clinical Atlas' && !breakdown) {
      fetch(`${API}/api/hereditary-retinal-dystrophy-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
  }, [tab, breakdown, definitions]);

  return (
    <div style={{ padding: '1.5rem', fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ marginBottom: 8 }}>
        <span style={{
          background: '#4a148c', color: '#fff', borderRadius: 6,
          padding: '4px 12px', fontSize: 12, fontWeight: 600,
        }}>
          Hereditary Retinal Dystrophy Atlas
        </span>
        <span style={{ marginLeft: 10, fontSize: 12, color: '#888' }}>
          8 genes · 320 patients · seeds 1494–1501
        </span>
      </div>
      <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
        Hereditary-Retinal-Dystrophy-Atlas — Complete 8-Gene Inherited Retinal Disease Reference
      </h1>
      <p style={{ fontSize: 13, color: '#666', marginBottom: 16 }}>
        RPGR · USH2A · ABCA4 · RHO · PRPF31 · CRB1 · CNGB3 · BEST1
        — ORF15 diagnostic gap, Usher dual-sensory, Stargardt light restriction, ADRP, reduced-penetrance RP11, LCA8/RP12, Achromatopsia stability, Best EOG
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
              background: tab === t ? '#4a148c' : '#f0f0f0',
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
