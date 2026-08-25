'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Metabolic', 'Treatments & Genetics', 'Definitions'];

// PWS colour scheme — warm orange/amber (hyperphagia/metabolic) + navy (imprinting/genetics)
const ACCENT  = '#e65100';   // deep orange — hyperphagia / metabolic crisis
const ACCENT2 = '#bf360c';   // burnt orange — paternal imprinting / deletion
const ACCENT3 = '#1b5e20';   // deep green — GH therapy / effective treatments
const ACCENT4 = '#b71c1c';   // deep red — high risk / psychosis / VPA risk
const ACCENT5 = '#0d47a1';   // dark blue — genetics / mechanism / SNORD116
const ACCENT6 = '#4a148c';   // purple — imprinting / 15q11 (shared with Angelman)
const ACCENT7 = '#37474f';   // dark slate — epidemiology
const ACCENT8 = '#00695c';   // dark teal — GH/carbetocin/therapy

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function PctBar({ label, pct, color = ACCENT }) {
  const numPct = typeof pct === 'string' ? parseInt(pct) : pct;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${numPct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function InfoBox({ title, children, color = ACCENT }) {
  return (
    <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold" style={{ color }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: '0.75rem' }}>
      {text}
    </span>
  );
}

export default function PWSPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview]     = useState(null);
  const [breakdown, setBreakdown]   = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [ov, br, df] = await Promise.all([
          fetch(`${API}/api/pws/overview`).then(r => r.json()),
          fetch(`${API}/api/pws/breakdown`).then(r => r.json()),
          fetch(`${API}/api/pws/definitions`).then(r => r.json()),
        ]);
        setOverview(ov); setBreakdown(br); setDefinitions(df);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <div className="p-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-2">Loading PWS dashboard…</p></div>;
  if (error)   return <div className="p-4 alert alert-danger">Error: {error}</div>;

  const ov = overview;
  const br = breakdown;
  const df = definitions;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="row mb-3">
        <div className="col">
          <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
            🧬 Prader-Willi Syndrome (PWS) Dashboard
          </h2>
          <p className="text-muted small mb-1">
            <strong>Locus:</strong> 15q11.2-q13 — PATERNAL expression required |{' '}
            <strong>OMIM:</strong> #176270 |{' '}
            <strong>Key genes:</strong> SNORD116 · SNRPN · MKRN3 · MAGEL2 · NDN |{' '}
            <strong>Prevalence:</strong> ~1:10,000–30,000 |{' '}
            <strong>Seed:</strong> {ov.seed} · n={ov.n_patients} patients
          </p>
          <div className="d-flex flex-wrap gap-1">
            <Badge text="15q11.2-q13 PATERNAL" color={ACCENT2} />
            <Badge text="SNORD116 Principal Driver" color={ACCENT5} />
            <Badge text="Same Locus as Angelman" color={ACCENT6} />
            <Badge text="Hyperphagia → Obesity" color={ACCENT} />
            <Badge text="GH Therapy Level A" color={ACCENT3} />
            <Badge text="Epilepsy ~10-15%" color={ACCENT7} />
            <Badge text="mUPD15 Psychosis Risk 20-30%" color={ACCENT4} />
          </div>
        </div>
        <div className="col-auto">
          <Link href="/" className="btn btn-sm btn-outline-secondary">← Home</Link>
        </div>
      </div>

      {/* Critical alert — same locus as Angelman */}
      <div className="alert alert-warning py-2 mb-3" style={{ borderLeft: `4px solid ${ACCENT6}` }}>
        <strong>⚠️ SAME LOCUS AS ANGELMAN (15q11-q13) — OPPOSITE PARENT:</strong>{' '}
        PATERNAL deletion/UPD = PWS (hyperphagia + obesity + minimal epilepsy ~{ov.epilepsy_summary.epilepsy_pct}%).{' '}
        MATERNAL deletion/UPD = Angelman (severe epilepsy ~{ov.epilepsy_summary.vs_angelman_pct}% + ataxia + absent speech).
        GH therapy (Level A) and FOOD SECURITY are the two non-negotiable pillars of PWS management.
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
              style={tab === t ? { color: ACCENT, borderBottomColor: ACCENT, fontWeight: 600 } : {}}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ── */}
      {tab === 'Overview' && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Patients" value={ov.n_patients} color={ACCENT7} />
            <KPI label="Epilepsy %" value={`${ov.epilepsy_summary.epilepsy_pct}%`} color={ACCENT} />
            <KPI label="vs Angelman Epilepsy" value={`${ov.epilepsy_summary.vs_angelman_pct}%`} color={ACCENT6} />
            <KPI label="On GH Therapy" value={`${ov.gh_therapy.gh_pct}%`} color={ACCENT3} />
            <KPI label="Obesity (BMI≥30)" value={`${ov.metabolic_features.obesity_pct}%`} color={ACCENT} />
            <KPI label="Avg BMI" value={ov.avg_bmi} color={ACCENT2} />
            <KPI label="Avg IQ" value={ov.avg_iq_estimate} color={ACCENT5} />
            <KPI label="Avg Hypotonia (0-10)" value={ov.avg_hypotonia_score_0_10} color={ACCENT7} />
            <KPI label="mUPD15 Psychosis Risk" value={`${ov.behavioral_features.psychosis_risk_n}/${ov.n_patients}`} color={ACCENT4} />
            <KPI label="Skin Picking" value={`${ov.behavioral_features.skin_picking_n}/${ov.n_patients}`} color={ACCENT} />
            <KPI label="Sleep Apnea" value={`${ov.clinical_features.sleep_apnea_n}/${ov.n_patients}`} color={ACCENT7} />
            <KPI label="Avg Dx Delay (mo)" value={ov.avg_diagnosis_delay_months} color={ACCENT4} />
          </div>

          <div className="row g-3">
            {/* Genetic mechanism distribution */}
            <div className="col-md-6">
              <InfoBox title="Genetic Mechanism Distribution (n=40)" color={ACCENT2}>
                {Object.entries(ov.mechanism_distribution).map(([mech, cnt]) => (
                  <PctBar key={mech}
                    label={`${mech} (n=${cnt})`}
                    pct={Math.round(cnt / ov.n_patients * 100)}
                    color={mech.includes('Type 1') ? ACCENT2 : mech.includes('Type 2') ? ACCENT : mech.includes('UPD') ? ACCENT5 : mech.includes('Imprinting') ? ACCENT6 : ACCENT7}
                  />
                ))}
                <p className="text-muted small mt-2 mb-0">
                  Deletion class (Type 1+2): {Math.round((ov.mechanism_distribution['Deletion 15q11.2-q13 Type 1 (bp1-bp3, ~6 Mb)'] + ov.mechanism_distribution['Deletion 15q11.2-q13 Type 2 (bp2-bp3, ~5 Mb)']) / ov.n_patients * 100)}% |
                  mUPD15: {Math.round(ov.mechanism_distribution['Maternal UPD15 (mUPD15)'] / ov.n_patients * 100)}% |
                  IC defect: {Math.round(ov.mechanism_distribution['Imprinting centre defect'] / ov.n_patients * 100)}%
                </p>
              </InfoBox>
            </div>

            {/* Behavioral features */}
            <div className="col-md-6">
              <InfoBox title="Behavioral & Psychiatric Features" color={ACCENT4}>
                <PctBar label="Skin picking (self-injury)" pct={Math.round(ov.behavioral_features.skin_picking_n/ov.n_patients*100)} color={ACCENT} />
                <PctBar label="Temper tantrums" pct={Math.round(ov.behavioral_features.temper_tantrums_n/ov.n_patients*100)} color={ACCENT2} />
                <PctBar label="OCD-like rigidity / food obsession" pct={Math.round(ov.behavioral_features.ocd_like_n/ov.n_patients*100)} color={ACCENT5} />
                <PctBar label="ASD features (mUPD15 enriched)" pct={Math.round(ov.behavioral_features.asd_features_n/ov.n_patients*100)} color={ACCENT6} />
                <PctBar label="Psychosis risk (mUPD15 class)" pct={Math.round(ov.behavioral_features.psychosis_risk_n/ov.n_patients*100)} color={ACCENT4} />
                <p className="small text-danger mt-2 mb-0">
                  ⚠️ mUPD15 class: 20-30% lifetime psychosis risk — highest of any UPD syndrome
                </p>
              </InfoBox>
            </div>

            {/* Clinical features */}
            <div className="col-md-6">
              <InfoBox title="Clinical Features" color={ACCENT}>
                <PctBar label="Obesity (BMI ≥30)" pct={ov.metabolic_features.obesity_pct} color={ACCENT} />
                <PctBar label="GH therapy ongoing" pct={ov.gh_therapy.gh_pct} color={ACCENT3} />
                <PctBar label="Sleep apnea (OSA/central)" pct={Math.round(ov.clinical_features.sleep_apnea_n/ov.n_patients*100)} color={ACCENT7} />
                <PctBar label="Excessive daytime sleepiness" pct={Math.round(ov.clinical_features.eds_n/ov.n_patients*100)} color={ACCENT7} />
                <PctBar label="Scoliosis" pct={Math.round(ov.clinical_features.scoliosis_n/ov.n_patients*100)} color={ACCENT5} />
                <PctBar label="Type 2 diabetes (BMI≥35 subset)" pct={Math.round(ov.metabolic_features.t2dm_n/ov.n_patients*100)} color={ACCENT4} />
                <PctBar label="Carbetocin trial participants" pct={Math.round(ov.clinical_features.carbetocin_trial_n/ov.n_patients*100)} color={ACCENT8} />
              </InfoBox>
            </div>

            {/* Epilepsy vs Angelman */}
            <div className="col-md-6">
              <InfoBox title="Epilepsy: PWS vs Angelman Syndrome (Same Locus!)" color={ACCENT6}>
                <PctBar label="PWS epilepsy prevalence" pct={ov.epilepsy_summary.epilepsy_pct} color={ACCENT7} />
                <PctBar label="Angelman epilepsy prevalence" pct={ov.epilepsy_summary.vs_angelman_pct} color={ACCENT6} />
                <div className="mt-2 p-2 rounded" style={{ background: '#fff3e0' }}>
                  <p className="small mb-1"><strong>PWS Epilepsy Profile:</strong></p>
                  <ul className="small mb-0">
                    <li>~10-15% have epilepsy (vs Angelman ~85%)</li>
                    <li>Types: focal &gt; generalized; mostly well-controlled</li>
                    <li>No pathognomonic EEG pattern (vs Angelman high-amplitude delta)</li>
                    <li>Standard AEDs effective (LEV, LTG preferred; VPA high risk)</li>
                    <li>DRE very uncommon in PWS; common in Angelman</li>
                  </ul>
                </div>
              </InfoBox>
            </div>

            {/* Key exam facts */}
            <div className="col-12">
              <InfoBox title="Key Exam Facts (15 Critical Points)" color={ACCENT5}>
                <ol className="mb-0 small">
                  {ov.key_exam_facts.map((f, i) => (
                    <li key={i} className="mb-1"
                      style={{ color: f.includes('ABSOLUTE') || f.includes('HIGH RISK') || f.includes('PSYCHOSIS') ? ACCENT4 :
                               f.includes('Level A') || f.includes('Level B') ? ACCENT3 :
                               f.includes('SNORD116') || f.includes('IMPRINTING') ? ACCENT5 :
                               f.includes('SAME LOCUS') || f.includes('Angelman') ? ACCENT6 : 'inherit' }}>
                      {f}
                    </li>
                  ))}
                </ol>
              </InfoBox>
            </div>
          </div>
        </div>
      )}

      {/* ── Patients & Metabolic Tab ── */}
      {tab === 'Patients & Metabolic' && br && (
        <div>
          {/* By mechanism summary */}
          <InfoBox title="Phenotype by Genetic Mechanism" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered mb-0 small">
                <thead style={{ backgroundColor: ACCENT2, color: '#fff' }}>
                  <tr>
                    <th>Mechanism</th><th>n</th><th>Epilepsy%</th>
                    <th>Obesity%</th><th>Psychosis%</th><th>ASD%</th>
                    <th>GH%</th><th>Avg BMI</th><th>Avg IQ</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(br.by_mechanism).map(([mech, row]) => (
                    <tr key={mech}>
                      <td className="fw-bold">{mech.replace('Deletion 15q11.2-q13 ', 'Del ')}</td>
                      <td>{row.n}</td>
                      <td style={{ color: row.epilepsy_pct > 20 ? ACCENT4 : ACCENT7 }}>{row.epilepsy_pct}%</td>
                      <td style={{ color: row.obesity_pct > 50 ? ACCENT4 : ACCENT7 }}>{row.obesity_pct}%</td>
                      <td style={{ color: row.psychosis_risk_pct > 10 ? ACCENT4 : ACCENT7 }}>{row.psychosis_risk_pct}%</td>
                      <td>{row.asd_pct}%</td>
                      <td style={{ color: ACCENT3 }}>{row.gh_therapy_pct}%</td>
                      <td style={{ color: row.avg_bmi >= 30 ? ACCENT4 : ACCENT7 }}>{row.avg_bmi}</td>
                      <td>{row.avg_iq}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </InfoBox>

          {/* Clinical summary bars */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <InfoBox title="Clinical Feature Rates (Cohort)" color={ACCENT}>
                <PctBar label="Epilepsy" pct={br.clinical_summary.pct_epilepsy} color={ACCENT} />
                <PctBar label="Obesity (BMI≥30)" pct={br.clinical_summary.pct_obesity} color={ACCENT2} />
                <PctBar label="GH therapy" pct={br.clinical_summary.pct_gh_therapy} color={ACCENT3} />
                <PctBar label="Sleep apnea" pct={br.clinical_summary.pct_sleep_apnea} color={ACCENT7} />
                <PctBar label="Skin picking" pct={br.clinical_summary.pct_skin_picking} color={ACCENT} />
                <PctBar label="Temper tantrums" pct={br.clinical_summary.pct_temper_tantrums} color={ACCENT2} />
                <PctBar label="Psychosis risk" pct={br.clinical_summary.pct_psychosis_risk} color={ACCENT4} />
                <PctBar label="ASD features" pct={br.clinical_summary.pct_asd_features} color={ACCENT5} />
                <PctBar label="Scoliosis" pct={br.clinical_summary.pct_scoliosis} color={ACCENT7} />
                <div className="mt-2 small text-muted">Avg BMI: {br.clinical_summary.avg_bmi} · Avg IQ est: {br.clinical_summary.avg_iq}</div>
              </InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="AED Usage (Epilepsy Patients)" color={ACCENT5}>
                {Object.keys(br.aed_counts).length === 0 ? (
                  <p className="text-muted small">No patients with AED (all seizure-free in this subset)</p>
                ) : (
                  Object.entries(br.aed_counts).map(([aed, cnt]) => (
                    <PctBar key={aed}
                      label={`${aed} ${aed === 'VPA' ? '(HIGH RISK in PWS)' : aed === 'LEV' ? '(Preferred)' : aed === 'LTG' ? '(Preferred)' : ''}`}
                      pct={Math.round(cnt / ov.epilepsy_summary.any_epilepsy_n * 100)}
                      color={aed === 'VPA' ? ACCENT4 : aed === 'LEV' ? ACCENT3 : aed === 'LTG' ? ACCENT3 : ACCENT7}
                    />
                  ))
                )}
                <InfoBox title="EEG Pattern Distribution" color={ACCENT6}>
                  {Object.entries(br.eeg_counts).map(([pat, cnt]) => (
                    <PctBar key={pat}
                      label={`${pat} (n=${cnt})`}
                      pct={Math.round(cnt / ov.n_patients * 100)}
                      color={pat.includes('Normal') && !pat.includes('IED') ? ACCENT3 : ACCENT7}
                    />
                  ))}
                  <p className="small text-muted mt-1 mb-0">
                    No pathognomonic EEG pattern in PWS (contrast Angelman: high-amplitude delta 200-500 µV, 2-3 Hz)
                  </p>
                </InfoBox>
              </InfoBox>
            </div>
          </div>

          {/* Patient table */}
          <InfoBox title={`All ${br.patients.length} Patients`} color={ACCENT7}>
            <div className="table-responsive" style={{ maxHeight: 420 }}>
              <table className="table table-sm table-striped table-hover small mb-0">
                <thead className="sticky-top" style={{ backgroundColor: ACCENT7, color: '#fff' }}>
                  <tr>
                    <th>ID</th><th>Mechanism</th><th>BMI</th><th>IQ</th>
                    <th>GH Tx</th><th>Epilepsy</th><th>AED</th>
                    <th>Psychosis</th><th>ASD</th><th>Skin Pick</th>
                    <th>EEG</th><th>Dx Delay(mo)</th>
                  </tr>
                </thead>
                <tbody>
                  {br.patients.map(p => (
                    <tr key={p.id}>
                      <td className="fw-bold">{p.id}</td>
                      <td style={{ maxWidth: 160, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {p.mechanism.replace('Deletion 15q11.2-q13 ', 'Del ')}
                      </td>
                      <td style={{ color: p.bmi >= 30 ? ACCENT4 : ACCENT3 }}>{p.bmi}</td>
                      <td>{p.iq_estimate}</td>
                      <td style={{ color: p.gh_therapy ? ACCENT3 : ACCENT7 }}>{p.gh_therapy ? '✓' : '–'}</td>
                      <td style={{ color: p.has_epilepsy ? ACCENT4 : ACCENT3 }}>
                        {p.has_epilepsy ? (p.seizure_types.join(', ') || '?') : '✓ seizure-free'}
                      </td>
                      <td style={{ color: p.aed_used === 'VPA' ? ACCENT4 : p.aed_used !== 'None' ? ACCENT3 : ACCENT7 }}>
                        {p.aed_used}
                      </td>
                      <td style={{ color: p.psychosis_risk ? ACCENT4 : ACCENT7 }}>
                        {p.psychosis_risk ? '⚠️ Risk' : '–'}
                      </td>
                      <td>{p.asd_features ? '✓' : '–'}</td>
                      <td style={{ color: p.skin_picking ? ACCENT : ACCENT7 }}>{p.skin_picking ? '✓' : '–'}</td>
                      <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {p.eeg_pattern}
                      </td>
                      <td>{(p.diagnosis_age_months - p.onset_months).toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </InfoBox>
        </div>
      )}

      {/* ── Treatments & Genetics Tab ── */}
      {tab === 'Treatments & Genetics' && (
        <div className="row g-3">
          <div className="col-md-6">
            <InfoBox title="Treatment Hierarchy (PWS-specific)" color={ACCENT3}>
              {[
                { tx: 'GH therapy (Genotropin)', level: 'Level A', note: 'FDA 2000; start infancy; +10-12 cm stature; improves body composition, energy', risk: 'Safe', color: ACCENT3 },
                { tx: 'Dietary restriction + food security', level: 'Level A', note: 'Locked refrigerators/pantries; 1000-1200 kcal/day structured; non-negotiable', risk: 'Essential', color: ACCENT3 },
                { tx: 'Carbetocin (oxytocin agonist)', level: 'Level B', note: 'Phase III CARE-PWS trial 2026; targets hypothalamic satiety; investigational', risk: 'Investigational', color: ACCENT8 },
                { tx: 'Melatonin', level: 'Level B', note: 'Central hypoventilation + sleep disturbance; 2-6 mg nocte', risk: 'Safe', color: ACCENT3 },
                { tx: 'SSRI (fluoxetine/sertraline)', level: 'Level B', note: 'Skin picking, OCD-like behavior, emotional lability; 30-40% response', risk: 'Monitor', color: ACCENT7 },
                { tx: 'LEV (levetiracetam)', level: 'Level B', note: 'First-line AED for epilepsy; broad-spectrum; weight-neutral', risk: 'Safe', color: ACCENT3 },
                { tx: 'LTG (lamotrigine)', level: 'Level B', note: 'Generalised seizures; safe; weight-neutral; preferred over VPA in PWS', risk: 'Safe', color: ACCENT3 },
                { tx: 'GnRH analog (leuprolide)', level: 'Level A (CPP)', note: 'Central precocious puberty subset; MKRN3 LOF mechanism', risk: 'Safe', color: ACCENT5 },
                { tx: 'Metformin', level: 'Level B', note: 'Insulin resistance / pre-T2DM; weight-neutral; renal monitoring', risk: 'Monitor', color: ACCENT7 },
                { tx: 'Aripiprazole/risperidone', level: 'Level B', note: 'Aggressive behavior, psychosis (especially mUPD15); lower metabolic risk vs typical APs', risk: 'Monitor BMI', color: ACCENT7 },
                { tx: 'VPA (valproate)', level: 'Level B (epilepsy)', note: 'HIGH RISK: weight gain + hepatosteatosis hepatotoxicity in obese PWS; last resort', risk: '⚠️ HIGH RISK', color: ACCENT4 },
                { tx: 'Topiramate', level: 'Level C', note: 'Epilepsy + weight-neutral; CAUTION: cognitive effects in IDD', risk: 'CAUTION', color: ACCENT2 },
              ].map(({ tx, level, note, risk, color }) => (
                <div key={tx} className="mb-2 p-2 rounded" style={{ background: '#f8f9fa' }}>
                  <div className="d-flex justify-content-between align-items-start">
                    <div>
                      <span className="fw-bold small">{tx}</span>{' '}
                      <span className="badge ms-1" style={{ backgroundColor: color }}>{level}</span>
                    </div>
                    <span className="badge" style={{ backgroundColor: color, opacity: 0.8 }}>{risk}</span>
                  </div>
                  <p className="text-muted mb-0" style={{ fontSize: '0.72rem' }}>{note}</p>
                </div>
              ))}
            </InfoBox>
          </div>

          <div className="col-md-6">
            <InfoBox title="Drug Risks — PWS-Specific" color={ACCENT4}>
              {[
                { drug: 'VPA (Valproate)', risk: 'HIGH RISK', reason: 'Weight gain worsens core disease; obesity → hepatic steatosis → hepatotoxicity; hyperammonemia at high doses; use only if no alternative' },
                { drug: 'Typical antipsychotics (haloperidol, etc.)', risk: 'HIGH RISK', reason: 'Severe weight gain + metabolic syndrome + EPS; prefer atypical APs (aripiprazole)' },
                { drug: 'Benzodiazepines', risk: 'CAUTION', reason: 'Respiratory depression risk in central hypoventilation + OSA; use cautiously; avoid clonazepam long-term' },
                { drug: 'GH therapy in untreated obesity', risk: 'CAUTION (avoid)', reason: 'Sudden death risk in severe obesity + uncontrolled respiratory failure; screen with PSG before GH' },
                { drug: 'Topiramate', risk: 'CAUTION', reason: 'Cognitive impairment in already intellectually impaired patients; not first-line; weight-neutral advantage does not outweigh cognitive risk' },
                { drug: 'CBZ/OXC (carbamazepine/oxcarbazepine)', risk: 'Relative caution', reason: 'NOT absolutely contraindicated in PWS (unlike Angelman); can worsen hyponatremia; limited efficacy for typical PWS seizure types; NOT first choice' },
                { drug: 'Estrogen / combined OCP', risk: 'CAUTION', reason: 'Thromboembolism risk in obese PWS females; thrombophilia screen before; prefer progesterone-only' },
                { drug: 'Corticosteroids', risk: 'HIGH RISK', reason: 'Severe weight gain; hyperglycemia in T2DM-prone; adrenal function may already be impaired' },
              ].map(({ drug, risk, reason }) => (
                <div key={drug} className="mb-2 p-2 rounded" style={{ background: '#fff3f3', borderLeft: `3px solid ${ACCENT4}` }}>
                  <div className="fw-bold small">{drug}
                    <span className="badge ms-1" style={{ backgroundColor: risk.includes('HIGH') ? ACCENT4 : ACCENT2 }}>{risk}</span>
                  </div>
                  <p className="text-muted mb-0" style={{ fontSize: '0.72rem' }}>{reason}</p>
                </div>
              ))}
            </InfoBox>

            <InfoBox title="Genetic Mechanisms — 5 Types" color={ACCENT2}>
              {[
                { mech: 'Deletion Type 1 (bp1-bp3, ~6 Mb)', pct: '40%', note: 'Slightly more severe behavioral; de novo' },
                { mech: 'Deletion Type 2 (bp2-bp3, ~5 Mb)', pct: '27%', note: 'Most common single deletion type; de novo' },
                { mech: 'Maternal UPD15 (mUPD15)', pct: '28%', note: '20-30% psychosis risk; more ASD; slightly milder hypotonia' },
                { mech: 'Imprinting centre defect', pct: '~2%', note: 'IC epimutation or microdeletion; up to 50% recurrence if inherited' },
                { mech: 'Chromosomal translocation', pct: '<1%', note: 'Karyotype required; parent karyotyping for recurrence risk' },
              ].map(({ mech, pct, note }) => (
                <div key={mech} className="mb-1 d-flex justify-content-between align-items-start small">
                  <span className="fw-bold">{mech}</span>
                  <span className="text-muted ms-2" style={{ minWidth: 30 }}>{pct}</span>
                  <span className="text-muted small ms-2" style={{ fontSize: '0.7rem' }}>{note}</span>
                </div>
              ))}
            </InfoBox>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'Definitions' && df && (
        <div>
          <div className="row mb-3 g-2">
            <div className="col-md-3">
              <div className="card shadow-sm p-3" style={{ borderTop: `4px solid ${ACCENT}` }}>
                <div className="fw-bold small">{df.disease_name}</div>
                <div className="text-muted small">{df.key_genes}</div>
                <div className="text-muted small">Locus: {df.locus}</div>
                <div className="text-muted small">OMIM: {df.omim_disease}</div>
                <div className="text-muted small">{df.inheritance}</div>
              </div>
            </div>
          </div>
          <div className="row g-3">
            {Object.entries(df.terms).map(([key, val]) => (
              <div className="col-md-6" key={key}>
                <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${ACCENT5}` }}>
                  <div className="card-body">
                    <h6 className="card-title fw-bold small" style={{ color: ACCENT5 }}>
                      {key.replace(/_/g, ' ')}
                    </h6>
                    <p className="card-text small text-muted mb-0">{val}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
