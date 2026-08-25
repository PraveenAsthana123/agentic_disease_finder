'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & EEG', 'Treatments & Genetics', 'Definitions'];

// UBE3A / Angelman Syndrome colour scheme — purple/violet (imprinting / happy demeanor)
const ACCENT  = '#4a148c';   // deep purple — UBE3A imprinting / identity
const ACCENT2 = '#7b1fa2';   // purple — deletion class
const ACCENT3 = '#1b5e20';   // deep green — KEY POSITIVES / effective treatments
const ACCENT4 = '#b71c1c';   // deep red — absolute CI / CBZ worsening
const ACCENT5 = '#0d47a1';   // dark blue — genetics / mechanism
const ACCENT6 = '#004d40';   // dark teal — EEG / neurological
const ACCENT7 = '#37474f';   // dark slate — epidemiology
const ACCENT8 = '#e65100';   // orange — fenfluramine / therapy

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
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: '0.72rem' }}>
      {text}
    </span>
  );
}

// ── Overview tab ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  const md  = data.mechanism_distribution || {};
  const ef  = data.epilepsy_features || {};
  const cf  = data.clinical_features || {};
  const eeg = data.eeg_summary || {};
  const tr  = data.treatment_summary || {};

  return (
    <div>
      {/* KPIs */}
      <div className="row g-3 mb-4">
        <KPI label="Patients"    value={data.n_patients}  color={ACCENT}  />
        <KPI label="Seed"        value={data.seed}         color={ACCENT7} />
        <KPI label="OMIM Gene"   value={data.omim_gene}    color={ACCENT5} />
        <KPI label="Locus"       value={data.locus}        color={ACCENT2} />
        <KPI label="Inheritance" value="Maternal LOF"      color={ACCENT7} />
        <KPI label="Prevalence"  value="~1:12-20K"         color={ACCENT8} />
      </div>

      {/* Primary banner */}
      <div className="alert mb-4" style={{ backgroundColor: '#f3e5f5', borderLeft: `5px solid ${ACCENT}` }}>
        <h6 className="fw-bold mb-1" style={{ color: ACCENT }}>
          Angelman Syndrome (UBE3A) — GENOMIC IMPRINTING: Only Maternal UBE3A Expressed in Neurons
        </h6>
        <p className="mb-0 small">
          Paternal UBE3A silenced in neurons by <strong>SNHG14/UBE3A-ATS antisense lncRNA</strong>.
          Maternal UBE3A LOF → zero UBE3A in neurons → Angelman Syndrome. Four mechanisms:
          deletion 15q11-q13 (65-70%), UBE3A mutation (10-15%), UPD15 (3-7%), IC defect (2-3%).
          <strong> ~85% have epilepsy</strong> — myoclonic, GTCS, atonic drops, West syndrome.
          {' '}<strong>CBZ/OXC ABSOLUTE CI</strong> (worsens myoclonic/atonic).
          {' '}<strong>EEG: high-amplitude rhythmic delta 2-3 Hz</strong> — characteristic/pathognomonic.
          {' '}Fenfluramine FDA-approved 2023. ASO UBE3A-ATS Phase I/II ongoing.
        </p>
      </div>

      {/* Mechanism distribution */}
      <InfoBox title="Genetic Mechanism Distribution" color={ACCENT5}>
        <div className="row">
          {Object.entries(md).map(([mech, n]) => (
            <div key={mech} className="col-md-6 mb-2">
              <PctBar
                label={mech}
                pct={Math.round(n / data.n_patients * 100)}
                color={
                  mech.includes('Deletion') ? ACCENT2 :
                  mech.includes('point')    ? ACCENT5 :
                  mech.includes('UPD')      ? ACCENT3 :
                  mech.includes('Imprint')  ? ACCENT6 : ACCENT7
                }
              />
              <div className="text-muted small">{n} patients</div>
            </div>
          ))}
        </div>
        <div className="mt-2">
          <Badge text="Deletion 67% — MOST SEVERE; OCA2 co-deleted → hypopig" color={ACCENT2} />
          <Badge text="UPD15 MILDEST — may have 1-2 words; ASD-prominent" color={ACCENT3} />
          <Badge text="IC defect — BEST ASO therapy candidate" color={ACCENT6} />
        </div>
      </InfoBox>

      {/* Epilepsy features */}
      <InfoBox title="Epilepsy Features — ~85% Prevalence, Multiple Types" color={ACCENT4}>
        <div className="row">
          {Object.entries(ef).map(([feat, n]) => (
            <div key={feat} className="col-6 col-md-4 mb-2">
              <div className="d-flex justify-content-between small">
                <span style={{
                  color: feat.includes('ci_violated') ? ACCENT4 :
                         feat.includes('myoclonic') ? ACCENT2 : 'inherit'
                }}>
                  {feat.replace(/_/g, ' ')}
                </span>
                <span className="fw-bold" style={{
                  color: feat.includes('ci_violated') ? ACCENT4 : ACCENT
                }}>
                  {n}/{data.n_patients}
                </span>
              </div>
              <div className="progress" style={{ height: 6 }}>
                <div className="progress-bar" style={{
                  width: `${n / data.n_patients * 100}%`,
                  backgroundColor: feat.includes('ci_violated') ? ACCENT4 :
                                   feat.includes('myoclonic') ? ACCENT2 : ACCENT,
                }} />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-2">
          <Badge text="Myoclonic 60% — most common; clonazepam + VPA" color={ACCENT2} />
          <Badge text="GTCS 45%" color={ACCENT4} />
          <Badge text="Atonic drops 30% — fall injuries" color={ACCENT} />
          <Badge text="West Syndrome 20% — infantile spasms onset" color={ACCENT5} />
          <Badge text="CBZ ABSOLUTE CI — worsens myoclonic status" color={ACCENT4} />
        </div>
      </InfoBox>

      {/* EEG summary */}
      <InfoBox title="EEG — CHARACTERISTIC Pattern (near-pathognomonic in context)" color={ACCENT6}>
        <div className="row">
          <div className="col-md-6">
            <div className="mb-2">
              <div className="fw-bold small" style={{ color: ACCENT6 }}>Average EEG Amplitude</div>
              <div className="fs-2 fw-bold" style={{ color: ACCENT6 }}>{eeg.avg_amplitude_uv} µV</div>
              <div className="text-muted small">Normal background &lt;150 µV; AS 200-500 µV — markedly elevated</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="mb-2">
              <div className="fw-bold small" style={{ color: ACCENT6 }}>Typical Pattern</div>
              <div className="text-muted small">{eeg.typical_pattern}</div>
            </div>
          </div>
        </div>
        <div className="mt-2">
          <Badge text="High-amplitude rhythmic delta 2-3 Hz — DIAGNOSTIC CLUE" color={ACCENT6} />
          <Badge text="Notched delta (delta + superimposed spikes)" color={ACCENT5} />
          <Badge text="Occipital dominance — eye-closure sensitive" color={ACCENT7} />
          <Badge text="Runs diffuse delta-theta in older patients" color={ACCENT7} />
        </div>
      </InfoBox>

      {/* Clinical features */}
      <InfoBox title="Clinical Features (Non-Epileptic)" color={ACCENT7}>
        <div className="row">
          {Object.entries(cf).map(([feat, n]) => (
            <div key={feat} className="col-6 col-md-3 mb-2">
              <div className="d-flex justify-content-between small">
                <span>{feat.replace(/_/g, ' ')}</span>
                <span className="fw-bold" style={{ color: ACCENT }}>{n}/{data.n_patients}</span>
              </div>
              <div className="progress" style={{ height: 6 }}>
                <div className="progress-bar" style={{ width: `${n / data.n_patients * 100}%`, backgroundColor: ACCENT7 }} />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-2">
          <Badge text="Sleep disturbance ~90% — melatonin Level A" color={ACCENT7} />
          <Badge text="Absent speech — ALL deletion class" color={ACCENT2} />
          <Badge text="Hypopigmentation — deletion class only (OCA2)" color={ACCENT5} />
          <Badge text="Happy demeanor + hand flapping + fascination with water" color={ACCENT3} />
        </div>
      </InfoBox>

      {/* Key exam facts */}
      <InfoBox title="Highest-Yield Exam Facts" color={ACCENT}>
        <ol className="mb-0 small">
          {(data.key_exam_facts || []).map((f, i) => (
            <li key={i} className="mb-1"
              style={{
                color: f.includes('ABSOLUTE') ? ACCENT4 :
                       f.includes('IMPRINTING') || f.includes('MATERNAL') ? ACCENT5 :
                       f.includes('EEG') ? ACCENT6 :
                       f.includes('CBZ') || f.includes('WORSENS') ? ACCENT4 :
                       f.includes('HAPPY') || f.includes('ANGELIC') ? ACCENT3 :
                       f.includes('PWS') || f.includes('SAME LOCUS') ? ACCENT2 :
                       f.includes('FDA') || f.includes('Fenfluramine') ? ACCENT8 :
                       f.includes('ASO') || f.includes('UBE3A-ATS') ? ACCENT6 :
                       f.includes('NBS') || f.includes('diagnosis') ? ACCENT7 : 'inherit'
              }}
            >{f}</li>
          ))}
        </ol>
      </InfoBox>
    </div>
  );
}

// ── Patients & EEG tab ────────────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  const pts = data.patients || [];
  const bm  = data.by_mechanism || {};
  const sc  = data.seizure_counts || {};
  const ep  = data.eeg_pattern_freq || {};
  const cs  = data.clinical_summary || {};

  return (
    <div>
      {/* Clinical summary KPIs */}
      <div className="row g-3 mb-4">
        <KPI label="Epilepsy Rate" value={`${cs.pct_epilepsy}%`}        color={ACCENT4} />
        <KPI label="Sleep Disturbed" value={`${cs.pct_sleep_disturbed}%`} color={ACCENT7} />
        <KPI label="Any Speech" value={`${cs.pct_any_speech}%`}          color={ACCENT3} />
        <KPI label="Hypopig (del)" value={`${cs.pct_hypopig}%`}          color={ACCENT2} />
        <KPI label="Microcephaly" value={`${cs.pct_microcephaly}%`}      color={ACCENT5} />
        <KPI label="Avg EEG µV" value={cs.avg_eeg_amp_uv}               color={ACCENT6} />
      </div>

      {/* By mechanism */}
      <InfoBox title="Clinical Profile by Genetic Mechanism" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead>
              <tr style={{ backgroundColor: '#f3e5f5' }}>
                <th>Mechanism</th><th>N</th><th>Epilepsy%</th><th>Myoclonic%</th>
                <th>West%</th><th>Speech%</th><th>Hypopig%</th>
                <th>Avg EEG µV</th><th>Avg Ataxia/10</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(bm).map(([mech, d]) => (
                <tr key={mech}>
                  <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{mech}</td>
                  <td>{d.n}</td>
                  <td style={{ color: d.epilepsy_pct > 80 ? ACCENT4 : 'inherit' }}>{d.epilepsy_pct}%</td>
                  <td>{d.myoclonic_pct}%</td>
                  <td>{d.west_pct}%</td>
                  <td style={{ color: d.any_speech_pct > 30 ? ACCENT3 : ACCENT4 }}>{d.any_speech_pct}%</td>
                  <td style={{ color: d.hypopig_pct > 40 ? ACCENT2 : ACCENT3 }}>{d.hypopig_pct}%</td>
                  <td style={{ color: d.avg_eeg_amp_uv > 300 ? ACCENT6 : 'inherit' }}>{d.avg_eeg_amp_uv}</td>
                  <td>{d.avg_ataxia_score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2">
          <Badge text="Deletion = most severe; highest EEG amplitude; no speech; hypopig" color={ACCENT2} />
          <Badge text="UPD15 = mildest; possible speech; ASD features" color={ACCENT3} />
        </div>
      </InfoBox>

      {/* Seizure counts */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <InfoBox title="Seizure Type Distribution" color={ACCENT4}>
            {Object.entries(sc).sort((a, b) => b[1] - a[1]).map(([type, cnt]) => (
              <PctBar key={type} label={type} pct={Math.round(cnt / pts.length * 100)}
                color={type.includes('Myoclonic') ? ACCENT2 :
                       type.includes('Status') || type.includes('NCSE') ? ACCENT4 :
                       type.includes('Atonic') ? ACCENT : ACCENT7} />
            ))}
            <div className="mt-2">
              <Badge text="Multiple seizure types co-occur — polytherapy often needed" color={ACCENT4} />
            </div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="EEG Pattern Frequency (across 40 patients)" color={ACCENT6}>
            {Object.entries(ep).sort((a, b) => b[1] - a[1]).map(([pat, cnt]) => (
              <PctBar key={pat} label={pat} pct={Math.round(cnt / pts.length * 100)} color={ACCENT6} />
            ))}
            <div className="mt-2">
              <Badge text="High-amplitude delta 2-3 Hz most common" color={ACCENT6} />
              <Badge text="In right context: near-pathognomonic" color={ACCENT5} />
            </div>
          </InfoBox>
        </div>
      </div>

      {/* Patient table */}
      <InfoBox title="Individual Patient Records (40-Patient Cohort, Seed 287)" color={ACCENT7}>
        <div className="table-responsive" style={{ maxHeight: 420 }}>
          <table className="table table-sm table-striped small mb-0">
            <thead className="sticky-top" style={{ backgroundColor: '#f3e5f5' }}>
              <tr>
                <th>ID</th><th>Mechanism</th><th>Severity</th>
                <th>Seizures</th><th>Onset (mo)</th><th>EEG µV</th>
                <th>Speech</th><th>Hypopig</th><th>Sleep±</th>
                <th>VPA Resp</th><th>CBZ±</th>
              </tr>
            </thead>
            <tbody>
              {pts.map(p => (
                <tr key={p.id}>
                  <td className="font-monospace small">{p.id}</td>
                  <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    color: p.mechanism.includes('Deletion') ? ACCENT2 :
                           p.mechanism.includes('UPD')      ? ACCENT3 : ACCENT5 }}>
                    {p.mechanism}</td>
                  <td style={{ color: p.severity === 'Severe' ? ACCENT4 : p.severity.includes('Mild') ? ACCENT3 : 'inherit' }}>
                    {p.severity}</td>
                  <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {p.seizure_types.join(', ') || '—'}</td>
                  <td>{p.onset_age_months}</td>
                  <td style={{ color: p.eeg_amp_uv > 350 ? ACCENT6 : 'inherit' }}>{p.eeg_amp_uv}</td>
                  <td style={{ color: p.has_any_speech ? ACCENT3 : ACCENT4 }}>{p.has_any_speech ? `${p.word_count}w` : 'None'}</td>
                  <td style={{ color: p.hypopigmentation ? ACCENT2 : ACCENT3 }}>{p.hypopigmentation ? '✓' : '—'}</td>
                  <td style={{ color: p.sleep_disturbed ? ACCENT4 : ACCENT3 }}>{p.sleep_disturbed ? '✓' : '—'}</td>
                  <td style={{ color: p.vpa_response === 'Good' ? ACCENT3 : p.vpa_response === 'Partial' ? ACCENT8 : ACCENT7 }}>
                    {p.vpa_response}</td>
                  <td style={{ color: p.cbz_worsened ? ACCENT4 : ACCENT3 }}>{p.cbz_worsened ? '⚠ WORSE' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </InfoBox>
    </div>
  );
}

// ── Treatments & Genetics tab ─────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  const tr = data.treatment_summary || {};
  const n  = (data.patients || []).length;

  const treatments = [
    { label: "Valproate (VPA)", level: "Level A", note: "First line — myoclonic + GTCS + absence; CAUTION: avoid VPA+LTG combo (SJS risk)", color: ACCENT3, ci: false },
    { label: "Clonazepam", level: "Level A", note: "Highly effective for myoclonic seizures; tolerance risk long-term; useful adjunct", color: ACCENT3, ci: false },
    { label: "Levetiracetam (LEV)", level: "Level B", note: "Broad-spectrum; safe; preferred if VPA contraindicated or poorly tolerated", color: ACCENT3, ci: false },
    { label: "Topiramate (TPM)", level: "Level B", note: "Focal + generalised; cognitive side-effects concerning in AS (cognitive baseline poor)", color: ACCENT5, ci: false },
    { label: "Clobazam", level: "Level B", note: "Adjunct for myoclonic; tolerance develops; useful for clusters", color: ACCENT5, ci: false },
    { label: "Lamotrigine (LTG)", level: "Level B", note: "ALONE: effective for myoclonic; AVOID with VPA (SJS risk — up to 3x dose increase)", color: ACCENT5, ci: false },
    { label: "Fenfluramine (FFA)", level: "Level A (FDA 2023)", note: "FDA-approved 2023 for AS; serotonin + sigma-1 agonist; reduces all seizure types; cardiac echo required", color: ACCENT8, ci: false },
    { label: "Melatonin", level: "Level A", note: "Severe sleep disturbance ~90% AS; 2-10 mg nocte; first-line for sleep", color: ACCENT7, ci: false },
    { label: "Cannabidiol (CBD)", level: "Level C", note: "Adjunct for drug-resistant seizures; evidence accumulating for AS specifically", color: ACCENT7, ci: false },
    { label: "ASO UBE3A-ATS knockdown", level: "Investigational", note: "Silences paternal SNHG14/UBE3A-ATS → de-represses paternal UBE3A in neurons; Phase I/II", color: ACCENT6, ci: false },
    { label: "Carbamazepine (CBZ) / OXC", level: "ABSOLUTE CI", note: "Worsens myoclonic and atonic seizures; precipitates myoclonic status epilepticus", color: ACCENT4, ci: true },
    { label: "VPA + LTG combination", level: "HIGH RISK", note: "Stevens-Johnson syndrome risk; VPA doubles LTG plasma level; avoid co-prescription", color: ACCENT4, ci: true },
    { label: "Vigabatrin / Tiagabine", level: "CAUTION", note: "May worsen myoclonic/absence — avoid unless focal component confirmed", color: ACCENT4, ci: true },
  ];

  const variants = [
    { v: "Deletion bp1-bp3 (~5 Mb)", freq: "~70% of deletions", domain: "15q11.2-q13 incl OCA2", sev: "Most severe; hypopig; deepest phenotype" },
    { v: "Deletion bp2-bp3 (~3 Mb)", freq: "~30% of deletions", domain: "15q12-q13", sev: "Severe; may have milder features than bp1-bp3" },
    { v: "p.Arg67Trp", freq: "Notable", domain: "HECT N-lobe", sev: "Classic AS; HECT domain structural disruption" },
    { v: "p.Trp608Ter (null)", freq: "Notable", domain: "HECT domain null", sev: "Severe classic AS; premature stop → NMD" },
    { v: "p.Glu550Lys", freq: "Notable", domain: "HECT catalytic Cys region", sev: "LOF at ubiquitin-transfer active site" },
    { v: "c.IVS7+1G>A (splice null)", freq: "Notable", domain: "Splice site — null", sev: "Transcript degraded; complete LOF" },
    { v: "Paternal UPD15", freq: "3-7% of AS", domain: "Whole chromosome 15 — UPD", sev: "Mildest; no deletion; methylation abnormal" },
    { v: "IC deletion (maternal)", freq: "2-3% of AS", domain: "Imprinting centre 15q11", sev: "Mild-moderate; best ASO candidate" },
  ];

  return (
    <div>
      {/* Treatment summary KPIs */}
      <div className="row g-3 mb-4">
        <KPI label="Fenfluramine Used"   value={tr.fenfluramine_used_n}       color={ACCENT8} />
        <KPI label="LEV Used"            value={tr.lev_used_n}                color={ACCENT3} />
        <KPI label="Clonazepam Used"     value={tr.clonazepam_used_n}         color={ACCENT3} />
        <KPI label="Melatonin (sleep)"   value={tr.melatonin_sleep_n}         color={ACCENT7} />
        <KPI label="CBZ CI Violated"     value={tr.cbz_absolute_ci_violated_n} color={ACCENT4} />
        <KPI label="VPA Good Response"   value={tr.vpa_good_response_n}       color={ACCENT3} />
      </div>

      {/* CI warning */}
      <div className="alert" style={{ backgroundColor: '#ffebee', borderLeft: `5px solid ${ACCENT4}` }}>
        <h6 className="fw-bold mb-1" style={{ color: ACCENT4 }}>
          ⚠ ABSOLUTE CI: CBZ/OXC — Worsens Myoclonic/Atonic Seizures in Angelman Syndrome
        </h6>
        <p className="mb-0 small">
          Carbamazepine (CBZ) and oxcarbazepine (OXC) are sodium-channel blockers. In AS,
          they <strong>paradoxically WORSEN myoclonic and atonic seizures</strong> — the predominant
          seizure types. Patients misdiagnosed as focal epilepsy before AS diagnosis and prescribed
          CBZ can deteriorate into myoclonic status epilepticus.{' '}
          <strong>VPA + LTG: HIGH RISK</strong> — VPA inhibits LTG glucuronidation → plasma LTG
          doubles → Stevens-Johnson syndrome. Use LTG ALONE (safe and effective in AS) if VPA not used.
        </p>
      </div>

      {/* Treatment table */}
      <InfoBox title="Treatment Evidence Levels — Angelman Syndrome Epilepsy" color={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead>
              <tr style={{ backgroundColor: '#f3e5f5' }}>
                <th>Treatment</th><th>Evidence Level</th><th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {treatments.map(t => (
                <tr key={t.label} style={{ backgroundColor: t.ci ? '#fff8f8' : 'transparent' }}>
                  <td className="fw-bold" style={{ color: t.color }}>{t.label}</td>
                  <td>
                    <span className="badge" style={{ backgroundColor: t.color }}>{t.level}</span>
                  </td>
                  <td style={{ color: t.ci ? ACCENT4 : 'inherit' }}>{t.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </InfoBox>

      {/* Fenfluramine detail */}
      <InfoBox title="Fenfluramine (FDA-Approved 2023 for AS) — Mechanism & Monitoring" color={ACCENT8}>
        <div className="row">
          <div className="col-md-6">
            <p className="small mb-1"><strong>Mechanism:</strong> Serotonin-releasing agent + sigma-1 receptor agonist.</p>
            <p className="small mb-1"><strong>Dose:</strong> 0.1–0.7 mg/kg/day; max 26 mg/day.</p>
            <p className="small mb-1"><strong>Efficacy:</strong> ~25-30% seizure frequency reduction across types (BUTTERFLY trials).</p>
          </div>
          <div className="col-md-6">
            <p className="small mb-1"><strong>Cardiac monitoring MANDATORY:</strong> Echo + ECG at baseline, 6 months, annually (historical valvulopathy concern at obesity doses).</p>
            <p className="small mb-1"><strong>BMI monitoring:</strong> Weight loss common; monitor growth in paediatric patients.</p>
          </div>
        </div>
        <div className="mt-1">
          <Badge text="BUTTERFLY-1 + BUTTERFLY-2 trials (Lagae 2022)" color={ACCENT8} />
          <Badge text="First FDA-approved AS-specific AED" color={ACCENT8} />
          <Badge text="Cardiac echo mandatory — baseline + 6mo + annual" color={ACCENT4} />
        </div>
      </InfoBox>

      {/* Variant / mechanism table */}
      <InfoBox title="Genetic Variants & Mechanisms (Representative)" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead>
              <tr style={{ backgroundColor: '#f3e5f5' }}>
                <th>Variant / Mechanism</th><th>Frequency</th><th>Domain / Location</th><th>Severity</th>
              </tr>
            </thead>
            <tbody>
              {variants.map(v => (
                <tr key={v.v}>
                  <td className="fw-bold" style={{ color: ACCENT5 }}>{v.v}</td>
                  <td>{v.freq}</td>
                  <td>{v.domain}</td>
                  <td>{v.sev}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-2">
          <Badge text="Deletion MOST COMMON (67%); confirm MATERNAL origin for point mutations" color={ACCENT2} />
          <Badge text="UPD15 = NO deletion; methylation abnormal; SNP array required" color={ACCENT3} />
          <Badge text="IC defect = BEST ASO candidate (imprinting mechanism still present)" color={ACCENT6} />
        </div>
      </InfoBox>

      {/* Diagnosis workup */}
      <InfoBox title="Diagnostic Workup Algorithm" color={ACCENT7}>
        <ol className="small mb-0">
          <li><strong>Clinical suspicion</strong> — severe ID + absent speech + ataxia + happy demeanor + characteristic EEG</li>
          <li><strong>DNA methylation study (SNRPN locus 15q11-q13)</strong> — FIRST MOLECULAR TEST</li>
          <li>If ABNORMAL methylation → <strong>CMA/FISH</strong> (deletion?) → <strong>SNP array</strong> (UPD?) → <strong>IC sequencing</strong> (IC defect?)</li>
          <li>If NORMAL methylation → <strong>UBE3A gene sequencing</strong> (confirm maternal origin of any variant)</li>
          <li>If both NORMAL → consider other NDDs; long-read sequencing research protocol</li>
        </ol>
        <div className="mt-2">
          <Badge text="Methylation study = FIRST test (detects deletion, UPD, IC defect)" color={ACCENT5} />
          <Badge text="UBE3A sequencing = SECOND test (normal methylation → UBE3A mutation?)" color={ACCENT6} />
          <Badge text="Mean diagnosis delay ~2.5 years from symptom onset" color={ACCENT7} />
        </div>
      </InfoBox>
    </div>
  );
}

// ── Definitions tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  const terms = data.terms || {};
  return (
    <div>
      <InfoBox title="Disease Overview" color={ACCENT}>
        <p className="small mb-0">
          <strong>{data.disease_name}</strong> ·{' '}
          Gene: <strong>{data.gene}</strong> · Locus: {data.locus} ·{' '}
          {data.omim_gene} · {data.omim_disease} · Inheritance: {data.inheritance}
        </p>
      </InfoBox>
      {Object.entries(terms).map(([key, text]) => (
        <InfoBox key={key} title={key.replace(/_/g, ' ')} color={
          key.includes('CBZ') || key.includes('LTG') ? ACCENT4 :
          key.includes('Imprint') || key.includes('MATERNAL') || key.includes('Four_mech') ? ACCENT5 :
          key.includes('EEG') ? ACCENT6 :
          key.includes('Fenfluramine') || key.includes('ASO') ? ACCENT8 :
          key.includes('PWS') || key.includes('UPD') ? ACCENT2 :
          key.includes('Methylation') || key.includes('Diagnosis') ? ACCENT7 :
          ACCENT
        }>
          <p className="small mb-0">{text}</p>
        </InfoBox>
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function UBE3APage() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview,  setOverview]  = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs,      setDefs]      = useState(null);
  const [error,     setError]     = useState(null);

  useEffect(() => {
    const base = `${API}/api/ube3a`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefs(df); })
      .catch(e => setError(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      {/* Breadcrumb */}
      <nav aria-label="breadcrumb" className="mb-3">
        <ol className="breadcrumb">
          <li className="breadcrumb-item"><Link href="/">Home</Link></li>
          <li className="breadcrumb-item"><Link href="/expert-dashboards">Expert Dashboards</Link></li>
          <li className="breadcrumb-item active">Angelman Syndrome (UBE3A)</li>
        </ol>
      </nav>

      {/* Header */}
      <div className="mb-3" style={{ borderLeft: `5px solid ${ACCENT}`, paddingLeft: 16 }}>
        <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
          Angelman Syndrome — UBE3A / E6-AP Ubiquitin Ligase Deficiency
        </h2>
        <p className="text-muted mb-0 small">
          Genomic imprinting epileptic encephalopathy · 15q11.2-q13 · Maternal LOF only · ~1:12,000–20,000 ·
          OMIM #105830 · EEG: high-amplitude rhythmic delta · Epilepsy ~85% · CBZ ABSOLUTE CI ·
          Fenfluramine FDA-approved 2023 · ASO therapy Phase I/II
        </p>
      </div>

      {error && <div className="alert alert-danger">API error: {error}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${activeTab === i ? 'active fw-bold' : ''}`}
              style={activeTab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setActiveTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* Tab content */}
      {activeTab === 0 && <OverviewTab     data={overview}  />}
      {activeTab === 1 && <PatientsTab     data={breakdown} />}
      {activeTab === 2 && <TreatmentsTab   data={breakdown} />}
      {activeTab === 3 && <DefinitionsTab  data={defs}      />}
    </div>
  );
}
