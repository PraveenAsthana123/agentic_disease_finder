'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1b5e20';   // deep teal — heparan sulfate / CNS-dominant LSD
const ACCENT2 = '#c62828';   // dark red — HIGH RISK / contraindications
const ACCENT3 = '#e65100';   // deep orange — CAUTION / relative CI
const ACCENT4 = '#0277bd';   // dark blue — gene therapy / investigational
const ACCENT5 = '#4a148c';   // deep purple — melatonin / sleep-seizure nexus
const ACCENT6 = '#558b2f';   // olive green — SAFE alternatives / LEV

const ETIOLOGY_COLORS = {
  'Compound Heterozygous — Null + Missense': '#1b5e20',
  'Homozygous Dutch Founder — p.Arg643Cys': '#2e7d32',
  'Biallelic Null / Homozygous Truncating': '#c62828',
  'Greek Founder — p.Arg626stop (Homozygous / Compound Het)': '#e65100',
  'Attenuated — Missense/Missense with Residual Activity': '#0277bd',
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
  const lvlColor = item.level === 'A' ? ACCENT4
    : item.level === 'B' ? ACCENT
    : item.level === 'D (not recommended)' ? '#b71c1c'
    : item.level === 'Investigational' ? '#6a1b9a'
    : ACCENT3;
  const levelLabel = item.level === 'Investigational' ? 'Investigational'
    : item.level === 'D (not recommended)' ? 'NOT Recommended'
    : `Level ${item.level}`;
  return (
    <div className="col-md-6 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-header text-white small fw-bold" style={{ background: lvlColor }}>
          {levelLabel} — {item.treatment}
        </div>
        <div className="card-body small">
          <p className="mb-1"><strong>Indication:</strong> {item.indication}</p>
          <p className="mb-1"><strong>Mechanism:</strong> {item.mechanism}</p>
          <p className="mb-1 text-info"><strong>Monitoring:</strong> {item.monitoring}</p>
          {item.caution && <p className="mb-0 text-warning"><strong>Caution:</strong> {item.caution}</p>}
        </div>
      </div>
    </div>
  );
}

function CICard({ item }) {
  const riskColor = item.level?.includes('HIGH RISK') ? ACCENT2
    : item.level?.includes('RELATIVE CI') ? ACCENT3
    : item.level?.includes('ABSOLUTE CI') ? '#b71c1c'
    : item.level?.includes('CAUTION') ? '#f57c00'
    : item.level?.includes('AVOID') ? '#880e4f'
    : item.level?.includes('NOT INDICATED') ? '#546e7a'
    : '#546e7a';
  return (
    <div className="col-md-6 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-header text-white small fw-bold" style={{ background: riskColor }}>
          {item.drug} — {item.level}
        </div>
        <div className="card-body small">
          <p className="mb-1"><strong>Reason:</strong> {item.reason}</p>
          <p className="mb-0 text-success"><strong>Safe alternative:</strong> {item.safe_alternative}</p>
        </div>
      </div>
    </div>
  );
}

// ── Overview Tab ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading…</div>;
  const seizureTypes = data.by_seizure_type || {};
  const topTriggers = data.top_triggers || {};
  return (
    <>
      <div className="alert mb-4 text-white fw-bold" style={{ background: ACCENT }}>
        🧬 SGSH / MPS IIIA — Sanfilippo Syndrome A (Heparan Sulfate Sulfamidase Deficiency)
        &nbsp;|&nbsp; {data.locus} &nbsp;|&nbsp; {data.inheritance}
      </div>

      {/* No approved therapy warning */}
      <div className="alert border-2 mb-4" style={{ borderColor: ACCENT2, background: '#ffebee' }}>
        <strong style={{ color: ACCENT2 }}>⚠️ NO APPROVED DISEASE-MODIFYING THERAPY (2026):</strong>
        <div className="mt-1 small">
          Unlike MPS I (laronidase), MPS II (idursulfase), MPS IVA (elosulfase), and MPS VI (galsulfase),
          MPS IIIA has NO approved ERT. Enrol eligible patients in <strong>OAV-101 (tralesinagene aparvovec) intracranial AAV10 gene therapy
          trial NCT02716246</strong> (Lysogene / Nationwide Children's Hospital). HSCT is NOT recommended
          in established MPS IIIA (unlike MPS I/II — neurological decline continues post-HSCT in MPS III).
        </div>
      </div>

      {/* Melatonin sleep-seizure nexus box */}
      <div className="alert border-2 mb-4" style={{ borderColor: ACCENT5, background: '#f3e5f5' }}>
        <strong style={{ color: ACCENT5 }}>🌙 MELATONIN — SLEEP-SEIZURE NEXUS (Level B):</strong>
        <div className="mt-1 small">
          Sleep disorder (90%+) is the cardinal trigger for seizure clusters in MPS IIIA — hypothalamic
          SCN HS accumulation disrupts circadian rhythm → sleep fragmentation → lowered seizure threshold.
          Melatonin 2–10 mg at bedtime is strongly indicated in ALL MPS IIIA patients with sleep disorder
          (Level B). No dependency, no withdrawal seizures, no seizure threshold effect.
        </div>
      </div>

      {/* KPI row */}
      <div className="row g-2 mb-4">
        <KPI label="Cohort" value={data.cohort_size} color={ACCENT} />
        <KPI label="Epilepsy overall" value={`${data.epilepsy_prevalence_pct}%`} color={ACCENT2} />
        <KPI label="Drug-resistant" value={`${data.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="Sleep disorder" value="90%+" color={ACCENT5} />
        <KPI label="Gene therapy" value={data.gene_therapy_phase?.split(' ')[0] || 'Phase I/II'} color={ACCENT4} />
        <KPI label="No ERT approved" value="2026" color={ACCENT2} />
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
              <PctBar key={i} label={trigger.split(' (')[0].substring(0, 45)} pct={pct} color={ACCENT3} />
            ))}
          </SectionCard>
          <SectionCard title="High-Risk / CI Drugs" color={ACCENT2}>
            {(data.absolute_ci_drugs || []).map((d, i) => (
              <div key={i} className="mb-1 small"><Badge text="HIGH RISK / CI" color={ACCENT2} /> {d}</div>
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
                <div className="card h-100 border-0" style={{ background: '#e0f2f1' }}>
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
        <SectionCard title="Key Concepts (15)" color={ACCENT4}>
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
          <div className="col-md-4 mb-2">
            <div className="text-center">
              <div className="fw-bold fs-5" style={{ color: ACCENT2 }}>{cohort_summary.drug_resistant_pct}%</div>
              <div className="small text-muted">Drug resistant ({cohort_summary.drug_resistant_n}/{cohort_summary.total_patients})</div>
            </div>
          </div>
          <div className="col-md-4 mb-2">
            <div className="text-center">
              <div className="fw-bold fs-5" style={{ color: ACCENT5 }}>{cohort_summary.sleep_disorder_pct}%</div>
              <div className="small text-muted">Sleep disorder ({cohort_summary.sleep_disorder_n}/{cohort_summary.total_patients})</div>
            </div>
          </div>
          <div className="col-md-4 mb-2">
            <div className="text-center">
              <div className="fw-bold fs-5" style={{ color: ACCENT4 }}>{cohort_summary.on_melatonin_pct}%</div>
              <div className="small text-muted">On melatonin ({cohort_summary.on_melatonin_n}/{cohort_summary.total_patients})</div>
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Genotype Distribution — 40 Patients" color={ACCENT}>
        {etiologies.map((e, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#e0f2f1' }}>
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
      <SectionCard title="Seizure Types" color={ACCENT2}>
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between align-items-center">
              <span className="fw-bold small">{s.type}</span>
              <Badge text={`${s.pct}%`} color={ACCENT2} />
            </div>
            <div className="progress my-1" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${s.pct}%`, background: ACCENT2 }} />
            </div>
            <p className="small text-muted mb-0">{s.subtype}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers" color={ACCENT3}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between align-items-center">
              <span className="fw-bold small">{t.trigger}</span>
              <Badge text={`${t.pct}%`} color={t.pct >= 80 ? ACCENT2 : ACCENT3} />
            </div>
            <div className="progress my-1" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${t.pct}%`, background: t.pct >= 80 ? ACCENT2 : ACCENT3 }} />
            </div>
            <p className="small text-muted mb-0">{t.notes}</p>
          </div>
        ))}
      </SectionCard>

      {monitoring.length > 0 && (
        <SectionCard title="Monitoring Protocol (14 Items)" color={ACCENT5}>
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
  const { treatments = [], contraindications = [], thresholds = [] } = data;
  return (
    <>
      <div className="alert border-2 mb-4" style={{ borderColor: ACCENT2, background: '#ffebee' }}>
        <strong style={{ color: ACCENT2 }}>NO APPROVED DISEASE-MODIFYING THERAPY FOR MPS IIIA (2026):</strong>
        <div className="mt-1 small">
          ERT not approved (BBB limits CNS penetration). HSCT NOT recommended in established disease.
          Gene therapy (AAV9-SGSH intrathecal, NCT02716246) is the current best investigational option.
          Refer all eligible patients to trial. AED management is symptomatic only.
        </div>
      </div>

      <SectionCard title="AEDs & Disease-Modifying (Investigational)" color={ACCENT}>
        <div className="row">
          {treatments.map((t, i) => <TreatmentCard key={i} item={t} />)}
        </div>
      </SectionCard>

      <SectionCard title="Drug Safety — Contraindications & Cautions" color={ACCENT2}>
        <div className="row">
          {contraindications.map((c, i) => <CICard key={i} item={c} />)}
        </div>
      </SectionCard>

      {thresholds.length > 0 && (
        <SectionCard title="Clinical Thresholds (12)" color={ACCENT5}>
          <div className="row">
            {thresholds.map((t, i) => (
              <div key={i} className="col-md-6 mb-2 small">
                <strong style={{ color: ACCENT }}>{t.parameter}:</strong>{' '}
                <Badge text={t.threshold} color={ACCENT3} />{' '}
                → {t.action}
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
  const { diagnostic_algorithm = [], definitions = [], key_concepts = [], standards = [], differential_diagnosis = [], pharmacological_distinctions = [] } = data;
  return (
    <>
      <SectionCard title="Diagnostic Algorithm (8 Steps)" color={ACCENT4}>
        <ol className="mb-0 small">
          {diagnostic_algorithm.map((step, i) => <li key={i} className="mb-2">{step}</li>)}
        </ol>
      </SectionCard>

      {pharmacological_distinctions.length > 0 && (
        <SectionCard title="Key Pharmacological Distinctions (10)" color={ACCENT2}>
          <ol className="mb-0 small">
            {pharmacological_distinctions.map((d, i) => <li key={i} className="mb-2">{d}</li>)}
          </ol>
        </SectionCard>
      )}

      {definitions.length > 0 && (
        <SectionCard title="SGSH / MPS IIIA Glossary (12 Terms)" color={ACCENT}>
          {definitions.map((d, i) => (
            <div key={i} className="mb-3 p-2 rounded" style={{ background: '#e0f2f1' }}>
              <div className="fw-bold small" style={{ color: ACCENT }}>{d.term}</div>
              <p className="small mb-0 mt-1">{d.definition}</p>
            </div>
          ))}
        </SectionCard>
      )}

      {differential_diagnosis.length > 0 && (
        <SectionCard title="Differential Diagnosis (7 Conditions)" color={ACCENT3}>
          {differential_diagnosis.map((d, i) => (
            <div key={i} className="mb-3 p-2 rounded" style={{ background: '#fff3e0' }}>
              <div className="fw-bold small" style={{ color: ACCENT3 }}>{d.condition}</div>
              <p className="small mb-0 mt-1">{d.distinguishing}</p>
            </div>
          ))}
        </SectionCard>
      )}

      {key_concepts.length > 0 && (
        <SectionCard title="Key Concepts" color={ACCENT5}>
          <ul className="mb-0 small">
            {key_concepts.map((c, i) => <li key={i} className="mb-1">{c}</li>)}
          </ul>
        </SectionCard>
      )}

      {standards.length > 0 && (
        <SectionCard title="Standards & References (12)" color="#546e7a">
          <div className="row">
            {standards.map((s, i) => (
              <div key={i} className="col-md-6 mb-2 small">
                <strong style={{ color: '#546e7a' }}>{s.code}:</strong> {s.title}
              </div>
            ))}
          </div>
        </SectionCard>
      )}
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function SgshPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/sgsh/overview`).then(r => r.json()).then(setOverview).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2 || tab === 3) {
      fetch(`${API}/api/sgsh/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/sgsh/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
        🧬 SGSH — MPS IIIA / Sanfilippo Syndrome A (Heparan Sulfate Sulfamidase Deficiency)
      </h2>
      <p className="text-muted small mb-3">
        OMIM 252900 / 605270 · 17q25.3 · AR biallelic LOF · No approved ERT (2026) ·
        OAV-101 (tralesinagene aparvovec) intracranial AAV10 Phase II/III ACMENA (NCT02716246) · HSCT NOT recommended · 40-patient cohort
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
