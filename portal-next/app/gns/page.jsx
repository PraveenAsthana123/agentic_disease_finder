'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#263238';   // dark blue-grey slate — MPS IIID rarest Sanfilippo / GNS
const ACCENT2 = '#c62828';   // dark red — HIGH RISK / contraindications
const ACCENT3 = '#e65100';   // deep orange — CAUTION / relative CI
const ACCENT4 = '#4527a0';   // deep indigo-purple — investigational / preclinical only
const ACCENT5 = '#4a148c';   // deep purple — melatonin / sleep-seizure nexus
const ACCENT6 = '#2e7d32';   // green — SAFE alternatives / LEV

const ETIOLOGY_COLORS = {
  'Compound Heterozygous — Null + Missense': '#263238',
  'Consanguineous Homozygous — Missense (Middle Eastern/South Asian Founder-enriched)': '#37474f',
  'Biallelic Null — Frameshift/Nonsense (Severe Phenotype)': '#c62828',
  'Attenuated — Biallelic Missense (Residual Activity ≥8%)': '#4527a0',
  'Deep Intronic / Regulatory Variant + Null (Cryptic Splicing Subset)': '#00695c',
};

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
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

function Badge({ text, color = ACCENT }) {
  return (
    <span className="badge me-1 mb-1" style={{ background: color, fontSize: '0.72rem' }}>{text}</span>
  );
}

function SectionCard({ title, color = ACCENT, children }) {
  return (
    <div className="card shadow-sm mb-4">
      <div className="card-header text-white fw-bold" style={{ background: color }}>{title}</div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function TreatmentCard({ item }) {
  const ev = item.evidence || 'C';
  const lvlColor = ev === 'Level A' ? ACCENT4
    : ev === 'Level B' ? ACCENT
    : ev === 'Investigational' ? '#4527a0'
    : ACCENT3;
  return (
    <div className="col-md-6 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-header text-white small fw-bold" style={{ background: lvlColor }}>
          {ev} — {item.treatment}
        </div>
        <div className="card-body small">
          <p className="mb-1">{item.notes}</p>
          {item.contraindications && (
            <p className="mb-0 text-danger"><strong>Contraindications:</strong> {item.contraindications}</p>
          )}
        </div>
      </div>
    </div>
  );
}

function CICard({ item }) {
  const riskColor = (item.risk || '').includes('HIGH RISK') ? ACCENT2
    : (item.risk || '').includes('RELATIVE CI') ? ACCENT3
    : (item.risk || '').includes('ABSOLUTE CI') ? '#b71c1c'
    : (item.risk || '').includes('CAUTION') ? '#f57c00'
    : (item.risk || '').includes('AVOID') ? '#880e4f'
    : (item.risk || '').includes('NOT INDICATED') ? '#546e7a'
    : '#546e7a';
  return (
    <div className="col-md-6 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-header text-white small fw-bold" style={{ background: riskColor }}>
          {item.drug} — {item.risk}
        </div>
        <div className="card-body small">
          <p className="mb-1"><strong>Mechanism:</strong> {item.mechanism}</p>
          <p className="mb-0 text-success"><strong>Alternative:</strong> {item.alternative}</p>
        </div>
      </div>
    </div>
  );
}

// ── Overview Tab ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading…</div>;
  const seizureTypes = data.by_seizure_type || {};
  const topTriggers  = data.top_triggers || {};
  return (
    <>
      <div className="alert mb-4 text-white fw-bold" style={{ background: ACCENT }}>
        🧬 GNS / MPS IIID — Sanfilippo Syndrome D (N-Acetylglucosamine-6-Sulfate Sulfatase Deficiency)
        &nbsp;|&nbsp; {data.locus} &nbsp;|&nbsp; {data.inheritance}
      </div>

      {/* Rarest subtype + no advanced GT warning */}
      <div className="alert border-2 mb-4" style={{ borderColor: ACCENT2, background: '#ffebee' }}>
        <strong style={{ color: ACCENT2 }}>⚠️ RAREST MPS III SUBTYPE — NO APPROVED THERAPY + NO ADVANCED-PHASE GENE THERAPY (2026):</strong>
        <div className="mt-1 small">
          MPS IIID has NO approved ERT and NO advanced-phase gene therapy trials (contrast: MPS IIIA
          has OAV-101 Phase II/III; IIIB has AAV9-NAGLU Phase I/II; IIIC has AAV9-HGSNAT Phase I).
          HSCT is NOT recommended. Refer all families to <strong>NIH Natural History Study / MPS Society
          International Registry</strong>. Only ~30–40 cases published worldwide.
        </div>
      </div>

      {/* MPS IIID distinguishing box */}
      {data.distinguishing_from_mps_iiia_iiib_iiic && (
        <div className="alert border-2 mb-4" style={{ borderColor: ACCENT, background: '#eceff1' }}>
          <strong style={{ color: ACCENT }}>🔬 DISTINGUISHING MPS IIID FROM IIIA/B/C:</strong>
          <div className="mt-1 small">{data.distinguishing_from_mps_iiia_iiib_iiic}</div>
        </div>
      )}

      {/* Melatonin sleep-seizure nexus */}
      <div className="alert border-2 mb-4" style={{ borderColor: ACCENT5, background: '#f3e5f5' }}>
        <strong style={{ color: ACCENT5 }}>🌙 MELATONIN — SLEEP-SEIZURE NEXUS (Level B):</strong>
        <div className="mt-1 small">
          Sleep disorder (80–85%) is the cardinal earliest symptom in MPS IIID. Hypothalamic SCN
          HS accumulation disrupts circadian rhythm → melatonin deficiency. Initiate melatonin
          5–10 mg nocte from behavioral phase onset. Sleep-triggered seizure clusters are the
          primary precipitant. Act BEFORE AEDs in new behavioral phase presentations.
        </div>
      </div>

      {/* KPI row */}
      <div className="row g-2 mb-4">
        <KPI label="Cohort" value={data.cohort_size} color={ACCENT} />
        <KPI label="Epilepsy overall" value={`${data.epilepsy_prevalence_pct}%`} color={ACCENT2} />
        <KPI label="Drug-resistant" value={`${data.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="Sleep disorder" value="80–85%" color={ACCENT5} />
        <KPI label="GT trials" value="Preclinical" color={ACCENT4} />
        <KPI label="Rarest MPS III" value="~1–5%" color={ACCENT2} />
      </div>

      {/* Disease overview + warning */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Disease Overview" color={ACCENT}>
            <p className="small mb-1"><strong>Mechanism:</strong> {data.disease_mechanism}</p>
            <p className="small mb-1"><strong>Onset:</strong> {data.epilepsy_onset}</p>
            <p className="small mb-1"><strong>Seizure types:</strong> {(data.cardinal_seizure_types || []).join(', ')}</p>
            <p className="small mb-0"><strong>Cardinal trigger:</strong> {data.cardinal_trigger}</p>
          </SectionCard>
          <SectionCard title="Seizure Type Prevalence" color={ACCENT2}>
            {Object.entries(seizureTypes).map(([type, pct], i) => (
              <PctBar key={i} label={type} pct={pct} color={ACCENT2} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Key Pharmacological Warning" color={ACCENT2}>
            <p className="small mb-0">{data.key_warning}</p>
          </SectionCard>
          <SectionCard title="Top Seizure Triggers" color={ACCENT3}>
            {Object.entries(topTriggers).map(([trigger, pct], i) => (
              <PctBar key={i} label={trigger.split(' (')[0].substring(0, 50)} pct={pct} color={ACCENT3} />
            ))}
          </SectionCard>
          <SectionCard title="High-Risk / CI Drugs" color={ACCENT2}>
            {(data.absolute_ci_drugs || []).map((d, i) => (
              <div key={i} className="mb-1 small"><Badge text="CI / RISK" color={ACCENT2} /> {d}</div>
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Lifecycle */}
      {data.lifecycle && (
        <SectionCard title="Disease Lifecycle (5 Stages)" color={ACCENT}>
          <div className="row">
            {data.lifecycle.map((l, i) => (
              <div key={i} className="col-md-6 mb-3">
                <div className="card h-100 border-0" style={{ background: '#eceff1' }}>
                  <div className="card-body small">
                    <div className="fw-bold" style={{ color: ACCENT }}>{l.stage}</div>
                    <p className="mb-0 mt-1">{l.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Key concepts */}
      {data.key_concepts && (
        <SectionCard title="Key Concepts (18)" color={ACCENT4}>
          <ul className="mb-0 small">
            {data.key_concepts.map((c, i) => <li key={i} className="mb-1">{c}</li>)}
          </ul>
        </SectionCard>
      )}
    </>
  );
}

// ── Patients & Etiology Tab ───────────────────────────────────────────────────
function EtiologyTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading…</div>;
  const { etiologies = [], patients = [], cohort_summary = {} } = data;
  return (
    <>
      <SectionCard title="Cohort Summary" color={ACCENT}>
        <div className="row">
          <div className="col-md-4 mb-2 text-center">
            <div className="fw-bold fs-5" style={{ color: ACCENT2 }}>{cohort_summary.drug_resistant_pct}%</div>
            <div className="small text-muted">Drug resistant ({cohort_summary.drug_resistant_n}/{cohort_summary.total_patients})</div>
          </div>
          <div className="col-md-4 mb-2 text-center">
            <div className="fw-bold fs-5" style={{ color: ACCENT5 }}>{cohort_summary.sleep_disorder_pct}%</div>
            <div className="small text-muted">Sleep disorder ({cohort_summary.sleep_disorder_n}/{cohort_summary.total_patients})</div>
          </div>
          <div className="col-md-4 mb-2 text-center">
            <div className="fw-bold fs-5" style={{ color: ACCENT4 }}>{cohort_summary.on_melatonin_pct}%</div>
            <div className="small text-muted">On melatonin ({cohort_summary.on_melatonin_n}/{cohort_summary.total_patients})</div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Genotype Distribution — 40 Patients (5 Etiologies)" color={ACCENT}>
        {etiologies.map((e, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#eceff1' }}>
            <div className="d-flex justify-content-between">
              <span className="fw-bold small" style={{ color: ETIOLOGY_COLORS[e.name] || ACCENT }}>{e.name}</span>
              <Badge text={`${e.pct}%`} color={ETIOLOGY_COLORS[e.name] || ACCENT} />
            </div>
            <div className="progress my-1" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, background: ETIOLOGY_COLORS[e.name] || ACCENT }} />
            </div>
            <p className="small mb-1 text-muted">{e.onset}</p>
            <p className="small mb-1">{e.notes}</p>
            <p className="small mb-0 fw-bold" style={{ color: ACCENT4 }}>Key finding: {e.key_finding}</p>
          </div>
        ))}
      </SectionCard>

      {patients.length > 0 && (
        <SectionCard title="Per-Patient Table (40 Patients)" color={ACCENT}>
          <div style={{ overflowX: 'auto' }}>
            <table className="table table-sm table-striped small mb-0">
              <thead>
                <tr>
                  <th>ID</th><th>Etiology</th><th>Onset (yrs)</th>
                  <th>Seizure Types</th><th>Primary AED</th>
                  <th>Drug Response</th><th>Sleep Disorder</th><th>Melatonin</th><th>Stage</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i}>
                    <td className="fw-bold" style={{ color: ACCENT }}>{p.patient_id}</td>
                    <td style={{ maxWidth: 140, whiteSpace: 'normal' }}>{p.etiology.split('—')[0].trim()}</td>
                    <td>{p.age_onset_seizures_yrs}</td>
                    <td>{(p.seizure_types || []).join(', ')}</td>
                    <td>{p.primary_aed?.split('(')[0].trim()}</td>
                    <td>
                      <Badge text={p.drug_response}
                        color={p.drug_resistant ? ACCENT2 : p.drug_response === 'Controlled' ? ACCENT6 : ACCENT3} />
                    </td>
                    <td>{p.sleep_disorder ? '✓' : '—'}</td>
                    <td>{p.on_melatonin ? '✓' : '—'}</td>
                    <td style={{ maxWidth: 120, whiteSpace: 'normal', fontSize: '0.7rem' }}>
                      {p.lifecycle_stage?.split('(')[0].trim()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}
    </>
  );
}

// ── Seizures & Triggers Tab ───────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading…</div>;
  const { seizure_types = [], triggers = [], monitoring = [] } = data;
  return (
    <>
      <SectionCard title="Seizure Types (5)" color={ACCENT2}>
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between align-items-center">
              <span className="fw-bold small">{s.type}</span>
              <Badge text={`${s.pct}%`} color={ACCENT2} />
            </div>
            <div className="progress my-1" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${s.pct}%`, background: ACCENT2 }} />
            </div>
            <p className="small text-muted mb-0"><strong>EEG:</strong> {s.eeg}</p>
            <p className="small mb-0">{s.notes}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (7)" color={ACCENT3}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between align-items-center">
              <span className="fw-bold small">{t.trigger}</span>
              <Badge text={`${t.pct}%`} color={t.pct >= 80 ? ACCENT2 : ACCENT3} />
            </div>
            <div className="progress my-1" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${t.pct}%`, background: t.pct >= 80 ? ACCENT2 : ACCENT3 }} />
            </div>
          </div>
        ))}
      </SectionCard>

      {monitoring.length > 0 && (
        <SectionCard title="Monitoring Protocol (12 Items)" color={ACCENT5}>
          <ul className="mb-0 small">
            {monitoring.map((m, i) => <li key={i} className="mb-1">{m}</li>)}
          </ul>
        </SectionCard>
      )}
    </>
  );
}

// ── Treatments Tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading…</div>;
  const { treatments = [], contraindications = [], thresholds = {} } = data;
  return (
    <>
      <div className="alert border-2 mb-4" style={{ borderColor: ACCENT2, background: '#ffebee' }}>
        <strong style={{ color: ACCENT2 }}>NO APPROVED DISEASE-MODIFYING THERAPY + NO ADVANCED GENE THERAPY TRIALS (2026):</strong>
        <div className="mt-1 small">
          MPS IIID (GNS) has no ERT and no advanced-phase gene therapy trials. HSCT NOT recommended.
          AED management is purely symptomatic. Refer all patients to NIH Natural History Study and
          MPS Society International Registry for future trial eligibility. GNS preclinical (KO mouse +
          canine) programs exist — AAV-GNS IT delivery in early development only.
        </div>
      </div>

      <SectionCard title="AEDs & Disease-Modifying Treatments (8)" color={ACCENT}>
        <div className="row">
          {treatments.map((t, i) => <TreatmentCard key={i} item={t} />)}
        </div>
      </SectionCard>

      <SectionCard title="Drug Safety — Contraindications & Cautions (6)" color={ACCENT2}>
        <div className="row">
          {contraindications.map((c, i) => <CICard key={i} item={c} />)}
        </div>
      </SectionCard>

      {thresholds && Object.keys(thresholds).length > 0 && (
        <SectionCard title="Clinical Thresholds" color={ACCENT5}>
          <div className="row">
            {Object.entries(thresholds).map(([k, v], i) => (
              <div key={i} className="col-md-6 mb-2 small">
                <strong style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}:</strong>{' '}
                <Badge text={String(v)} color={ACCENT3} />
              </div>
            ))}
          </div>
        </SectionCard>
      )}
    </>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading…</div>;
  const {
    diagnostic_algorithm = [],
    definitions = [],
    key_concepts = [],
    standards = [],
    differential_diagnosis = [],
    pharmacological_distinctions = [],
  } = data;
  return (
    <>
      <SectionCard title="Diagnostic Algorithm (8 Steps)" color={ACCENT4}>
        <ol className="mb-0 small">
          {diagnostic_algorithm.map((step, i) => <li key={i} className="mb-2">{step}</li>)}
        </ol>
      </SectionCard>

      {pharmacological_distinctions.length > 0 && (
        <SectionCard title="Key Pharmacological Distinctions (12)" color={ACCENT2}>
          <ol className="mb-0 small">
            {pharmacological_distinctions.map((d, i) => <li key={i} className="mb-2">{d}</li>)}
          </ol>
        </SectionCard>
      )}

      {definitions.length > 0 && (
        <SectionCard title="GNS / MPS IIID Glossary (10 Terms)" color={ACCENT}>
          {definitions.map((d, i) => (
            <div key={i} className="mb-3 p-2 rounded" style={{ background: '#eceff1' }}>
              <div className="fw-bold small" style={{ color: ACCENT }}>{d.term}</div>
              <p className="small mb-0 mt-1">{d.definition}</p>
            </div>
          ))}
        </SectionCard>
      )}

      {differential_diagnosis.length > 0 && (
        <SectionCard title="Differential Diagnosis (6 Conditions)" color={ACCENT3}>
          {differential_diagnosis.map((d, i) => (
            <div key={i} className="mb-3 p-2 rounded" style={{ background: '#fff3e0' }}>
              <div className="fw-bold small" style={{ color: ACCENT3 }}>{d.condition}</div>
              <p className="small mb-1 mt-1">{d.distinguishing}</p>
              {d.key_test && (
                <p className="small mb-0 text-muted"><strong>Key test:</strong> {d.key_test}</p>
              )}
            </div>
          ))}
        </SectionCard>
      )}

      {key_concepts.length > 0 && (
        <SectionCard title="Key Concepts (18)" color={ACCENT5}>
          <ul className="mb-0 small">
            {key_concepts.map((c, i) => <li key={i} className="mb-1">{c}</li>)}
          </ul>
        </SectionCard>
      )}

      {standards.length > 0 && (
        <SectionCard title="Standards & References" color="#546e7a">
          <ul className="mb-0 small">
            {standards.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
          </ul>
        </SectionCard>
      )}
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function GnsPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/gns/overview`).then(r => r.json()).then(setOverview).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/gns/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/gns/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
        🧬 GNS — MPS IIID / Sanfilippo Syndrome D (N-Acetylglucosamine-6-Sulfate Sulfatase Deficiency)
      </h2>
      <p className="text-muted small mb-3">
        OMIM 252940 / 607664 · 12q14.3 · AR biallelic LOF · RAREST MPS III (~1–5%) ·
        No approved ERT (2026) · No advanced-phase gene therapy trials · HSCT NOT recommended ·
        40-patient cohort · ~30–40 cases worldwide · NIH Natural History Study referral
      </p>

      {err && <div className="alert alert-danger">API error: {err}</div>}

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { borderBottomColor: ACCENT, color: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <EtiologyTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
