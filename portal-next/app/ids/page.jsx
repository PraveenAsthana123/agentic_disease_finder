'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#004d40';   // deep teal — MPS II X-linked / Hunter / IDS
const ACCENT2 = '#b71c1c';   // dark red — HIGH RISK / contraindications / airway hazard
const ACCENT3 = '#e65100';   // deep orange — CAUTION / relative CI
const ACCENT4 = '#1a237e';   // deep indigo — ERT / investigational
const ACCENT5 = '#4a148c';   // deep purple — OSA / sleep / BBB-crossing
const ACCENT6 = '#2e7d32';   // green — SAFE alternatives / LEV first-line

const ETIOLOGY_COLORS = {
  'Severe — IDS large deletion/rearrangement (IDS/IDSP1 pseudogene)': '#b71c1c',
  'Severe — IDS nonsense / frameshift (LOF, null)': '#c62828',
  'Attenuated — IDS missense (partial LOF, residual activity >5%)': '#1a237e',
  'Severe — IDS splice-site variant': '#e65100',
  'Intermediate / deep-intronic — RNA-seq required': '#00695c',
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

function CICard({ item }) {
  const riskColor = (item.risk || '').includes('HIGH RISK') ? ACCENT2
    : (item.risk || '').includes('RELATIVE CI') ? ACCENT3
    : (item.risk || '').includes('ABSOLUTE CI') ? '#b71c1c'
    : (item.risk || '').includes('CAUTION') ? '#f57c00'
    : (item.risk || '').includes('AVOID') ? '#880e4f'
    : (item.risk || '').includes('EXTREME HAZARD') ? '#4a0000'
    : (item.risk || '').includes('NOT INDICATED') ? '#546e7a'
    : '#546e7a';
  return (
    <div className="col-md-6 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-header text-white small fw-bold" style={{ background: riskColor }}>
          {item.drug} — {item.risk}
        </div>
        <div className="card-body small">
          <p className="mb-1"><strong>Reason:</strong> {item.reason}</p>
          <p className="mb-0 text-success"><strong>Alternative:</strong> {item.alternative}</p>
        </div>
      </div>
    </div>
  );
}

function TreatmentCard({ item }) {
  const levelColor = (item.level || '').includes('Level A') ? ACCENT6
    : (item.level || '').includes('Level B') ? ACCENT
    : (item.level || '').includes('Level C') ? ACCENT3
    : ACCENT4;
  return (
    <div className="col-md-6 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-header text-white small fw-bold" style={{ background: levelColor }}>
          {item.treatment}
        </div>
        <div className="card-body small">
          <p className="mb-1"><strong>Dose:</strong> {item.dose}</p>
          <p className="mb-1"><strong>Efficacy:</strong> {item.efficacy}</p>
          <p className="mb-1"><strong>Safety:</strong> {item.safety}</p>
          <p className="mb-0 text-muted"><em>{item.level}</em></p>
        </div>
      </div>
    </div>
  );
}

// ── Overview Tab ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-5 text-muted">Loading…</div>;
  const seizureTypes = data.by_seizure_type || {};
  const topTriggers  = data.top_triggers || {};
  return (
    <>
      <div className="alert mb-4 text-white fw-bold" style={{ background: ACCENT }}>
        🧬 IDS / MPS II — Hunter Syndrome (Iduronate-2-Sulfatase Deficiency)
        &nbsp;|&nbsp; {data.locus} &nbsp;|&nbsp; {data.inheritance}
      </div>

      {/* Airway extreme hazard */}
      <div className="alert border-2 mb-4" style={{ borderColor: '#4a0000', background: '#ffcdd2' }}>
        <strong style={{ color: '#4a0000' }}>🚨 AIRWAY + ANESTHESIA EXTREME HAZARD (MPS II PRIORITY #1):</strong>
        <div className="mt-1 small">
          GAG-narrowed airway (tongue, trachea, epiglottis, supraglottis) + atlantoaxial instability (C1/C2
          odontoid GAG) → DIFFICULT / FAILED INTUBATION + CERVICAL CORD INJURY RISK during any general
          anesthesia. <strong>Alert anesthesia team BEFORE every procedure</strong>. C-spine radiograph
          (lateral flexion/extension) mandatory pre-GA. Video laryngoscopy / awake fiber-optic preferred.
          Avoid neck hyperflexion. Peri-anesthetic hypoxia → post-ictal seizure cluster.
        </div>
      </div>

      {/* ERT available box */}
      <div className="alert border-2 mb-4" style={{ borderColor: ACCENT4, background: '#e8eaf6' }}>
        <strong style={{ color: ACCENT4 }}>💉 ERT AVAILABLE — TWO APPROVED AGENTS (unlike MPS III):</strong>
        <div className="mt-1 small">
          <strong>Idursulfase (Elaprase, Takeda) FDA 2006:</strong> 0.5 mg/kg/week IV; reduces somatic HS+DS;
          LIMITED BBB PENETRATION — does not prevent CNS progression in severe. &nbsp;|&nbsp;
          <strong>Pabinafusp alfa (JR-141, IZCARGO, JCR Pharma) Japan MHLW 2021:</strong>
          anti-hTfR1 antibody fusion → transferrin-receptor BBB transcytosis → superior CSF/brain I2S;
          Phase III globally ongoing (NCT04251026). &nbsp;|&nbsp;
          <strong>Intrathecal Idursulfase (HGT-2310, COMPASS):</strong> Phase I/II CSF-directed.
        </div>
      </div>

      {/* OSA seizure trigger */}
      <div className="alert border-2 mb-4" style={{ borderColor: ACCENT5, background: '#f3e5f5' }}>
        <strong style={{ color: ACCENT5 }}>😴 OSA IS THE DOMINANT SEIZURE TRIGGER IN MPS II (60–70%):</strong>
        <div className="mt-1 small">
          GAG-narrowed upper airway → obstructive sleep apnea (OSA) in 60–70% → hypoxia → seizure threshold
          reduction → cluster seizures. Annual polysomnography mandatory. CPAP titration first-line.
          ENT evaluation (tonsillar/adenoidal GAG). Treating OSA is seizure management — not just comfort.
        </div>
      </div>

      {/* KPI row */}
      <div className="row g-2 mb-4">
        <KPI label="Cohort" value={data.cohort_size} color={ACCENT} />
        <KPI label="Epilepsy" value={`${data.epilepsy_prevalence_pct}%`} color={ACCENT2} />
        <KPI label="Drug-resistant" value={`${data.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="OSA trigger" value="60–70%" color={ACCENT5} />
        <KPI label="ERT available" value="Yes (2)" color={ACCENT4} />
        <KPI label="X-linked" value="Males" color={ACCENT} />
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Disease Overview" color={ACCENT}>
            <p className="small mb-1"><strong>Mechanism:</strong> {data.disease_mechanism}</p>
            <p className="small mb-1"><strong>Epilepsy onset:</strong> {data.epilepsy_onset}</p>
            <p className="small mb-1"><strong>Cardinal seizure types:</strong> {(data.cardinal_seizure_types || []).join(', ')}</p>
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
              <PctBar key={i} label={trigger.split('(')[0].trim().substring(0, 55)} pct={pct} color={ACCENT3} />
            ))}
          </SectionCard>
          <SectionCard title="MPS II vs MPS III" color={ACCENT4}>
            <p className="small mb-0">{data.distinguishing_from_mps_iii}</p>
          </SectionCard>
        </div>
      </div>

      {/* Lifecycle */}
      {data.lifecycle && (
        <SectionCard title="Disease Lifecycle (6 Stages)" color={ACCENT}>
          <div className="row">
            {data.lifecycle.map((l, i) => (
              <div key={i} className="col-md-6 mb-3">
                <div className="card h-100 border-0" style={{ background: '#e0f2f1' }}>
                  <div className="card-body small">
                    <div className="fw-bold" style={{ color: ACCENT }}>{l.stage}</div>
                    <p className="mb-1 mt-1"><em>{l.features}</em></p>
                    <p className="mb-1"><strong>Epilepsy:</strong> {l.epilepsy}</p>
                    <p className="mb-0 text-muted"><strong>Management:</strong> {l.mgmt}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Key concepts */}
      {data.key_concepts && (
        <SectionCard title="Key Concepts (17)" color={ACCENT4}>
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
          <div className="col-md-3 mb-2 text-center">
            <div className="fw-bold fs-5" style={{ color: ACCENT2 }}>{cohort_summary.drug_resistant_pct}%</div>
            <div className="small text-muted">Drug resistant ({cohort_summary.drug_resistant_n}/{cohort_summary.total_patients})</div>
          </div>
          <div className="col-md-3 mb-2 text-center">
            <div className="fw-bold fs-5" style={{ color: ACCENT5 }}>{cohort_summary.osa_pct}%</div>
            <div className="small text-muted">OSA ({cohort_summary.osa_n}/{cohort_summary.total_patients})</div>
          </div>
          <div className="col-md-3 mb-2 text-center">
            <div className="fw-bold fs-5" style={{ color: ACCENT4 }}>{cohort_summary.on_ert_pct}%</div>
            <div className="small text-muted">On ERT ({cohort_summary.on_ert_n}/{cohort_summary.total_patients})</div>
          </div>
          <div className="col-md-3 mb-2 text-center">
            <div className="fw-bold fs-5" style={{ color: ACCENT3 }}>{cohort_summary.hydrocephalus_pct}%</div>
            <div className="small text-muted">Hydrocephalus ({cohort_summary.hydrocephalus_n}/{cohort_summary.total_patients})</div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Variant / Phenotype Distribution — 40 Patients (5 Etiologies)" color={ACCENT}>
        {etiologies.map((e, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#e0f2f1' }}>
            <div className="d-flex justify-content-between">
              <span className="fw-bold small" style={{ color: ETIOLOGY_COLORS[e.name] || ACCENT }}>{e.name}</span>
              <Badge text={`${e.pct}%`} color={ETIOLOGY_COLORS[e.name] || ACCENT} />
            </div>
            <div className="progress my-1" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, background: ETIOLOGY_COLORS[e.name] || ACCENT }} />
            </div>
            <p className="small mb-1 text-muted">{e.mechanism}</p>
            <p className="small mb-0">
              <Badge text={`Epilepsy risk: ${e.epilepsy_risk}`}
                color={e.epilepsy_risk === 'High' ? ACCENT2 : e.epilepsy_risk === 'Low' ? ACCENT6 : ACCENT3} />
              <span className="text-muted ms-2">EEG: {e.eeg_pattern}</span>
            </p>
          </div>
        ))}
      </SectionCard>

      {patients.length > 0 && (
        <SectionCard title="Per-Patient Table (40 Patients)" color={ACCENT}>
          <div style={{ overflowX: 'auto' }}>
            <table className="table table-sm table-striped small mb-0">
              <thead>
                <tr>
                  <th>ID</th><th>Phenotype</th><th>Etiology</th><th>Onset (yrs)</th>
                  <th>Seizure Types</th><th>Primary AED</th><th>Response</th>
                  <th>OSA</th><th>ERT</th><th>Hydrocephalus</th><th>Stage</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i}>
                    <td className="fw-bold" style={{ color: ACCENT }}>{p.patient_id}</td>
                    <td>
                      <Badge text={p.phenotype}
                        color={p.phenotype === 'Severe' ? ACCENT2 : ACCENT4} />
                    </td>
                    <td style={{ maxWidth: 120, whiteSpace: 'normal', fontSize: '0.7rem' }}>
                      {p.etiology.split('—')[0].trim()}
                    </td>
                    <td>{p.age_onset_seizures_yrs || '—'}</td>
                    <td>{(p.seizure_types || []).join(', ') || '—'}</td>
                    <td>{(p.primary_aed || '—').split('(')[0].trim()}</td>
                    <td>
                      <Badge text={p.drug_response || '—'}
                        color={p.drug_resistant ? ACCENT2 : p.drug_response === 'Controlled' ? ACCENT6 : ACCENT3} />
                    </td>
                    <td>{p.osa ? '✓' : '—'}</td>
                    <td>{p.on_ert ? (p.on_pabinafusp ? 'JR-141' : 'Elaprase') : '—'}</td>
                    <td>{p.hydrocephalus ? '✓' : '—'}</td>
                    <td style={{ maxWidth: 110, whiteSpace: 'normal', fontSize: '0.7rem' }}>
                      {(p.lifecycle_stage || '').split('(')[0].split('—')[0].trim()}
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
      <SectionCard title="Seizure Types (6)" color={ACCENT2}>
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
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (8) — OSA Dominant" color={ACCENT3}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between align-items-center">
              <span className="fw-bold small">{t.trigger.split('/')[0].trim()}</span>
              <Badge text={`${t.pct}%`} color={t.pct >= 60 ? ACCENT2 : ACCENT3} />
            </div>
            <div className="progress my-1" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${t.pct}%`, background: t.pct >= 60 ? ACCENT2 : ACCENT3 }} />
            </div>
            <p className="small text-muted mb-0">{t.mechanism}</p>
          </div>
        ))}
      </SectionCard>

      {monitoring.length > 0 && (
        <SectionCard title="Monitoring Protocol (12 Items)" color={ACCENT5}>
          {monitoring.map((m, i) => (
            <div key={i} className="mb-2 p-2 rounded" style={{ background: '#f3e5f5' }}>
              <div className="fw-bold small" style={{ color: ACCENT5 }}>{m.item}</div>
              <div className="small text-muted">{m.schedule} — {m.rationale}</div>
            </div>
          ))}
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
      <div className="alert border-2 mb-4" style={{ borderColor: ACCENT4, background: '#e8eaf6' }}>
        <strong style={{ color: ACCENT4 }}>💉 ERT AVAILABLE (unlike MPS III) — TWO AGENTS + INTRATHECAL TRIAL:</strong>
        <div className="mt-1 small">
          <strong>Elaprase (Idursulfase) FDA 2006:</strong> weekly IV, somatic reduction, limited CNS.&nbsp;|&nbsp;
          <strong>Pabinafusp alfa (JR-141) Japan 2021:</strong> anti-hTfR1 fusion, BBB-crossing, superior CNS — Phase III globally.&nbsp;|&nbsp;
          <strong>HGT-2310 (IT-ERT) COMPASS:</strong> Phase I/II intrathecal CSF-directed ERT.&nbsp;|&nbsp;
          ERT does NOT replace AED management in CNS-severe phenotype.
        </div>
      </div>

      <SectionCard title="AEDs & ERT Treatments (9)" color={ACCENT}>
        <div className="row">
          {treatments.map((t, i) => <TreatmentCard key={i} item={t} />)}
        </div>
      </SectionCard>

      <SectionCard title="Drug Safety — Contraindications & Cautions (6)" color={ACCENT2}>
        <div className="row">
          {contraindications.map((c, i) => <CICard key={i} item={c} />)}
        </div>
      </SectionCard>

      {thresholds.length > 0 && (
        <SectionCard title="Clinical Thresholds (10)" color={ACCENT5}>
          {thresholds.map((t, i) => (
            <div key={i} className="mb-2 p-2 rounded" style={{ background: '#f3e5f5' }}>
              <div className="fw-bold small" style={{ color: ACCENT5 }}>{t.threshold}</div>
              <div className="small text-muted">{t.action}</div>
            </div>
          ))}
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
      <SectionCard title="Diagnostic Algorithm (10 Steps)" color={ACCENT4}>
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
        <SectionCard title="IDS / MPS II Glossary (14 Terms)" color={ACCENT}>
          {definitions.map((d, i) => (
            <div key={i} className="mb-3 p-2 rounded" style={{ background: '#e0f2f1' }}>
              <div className="fw-bold small" style={{ color: ACCENT }}>{d.term}</div>
              <p className="small mb-0 mt-1">{d.definition}</p>
            </div>
          ))}
        </SectionCard>
      )}

      {differential_diagnosis.length > 0 && (
        <SectionCard title="Differential Diagnosis (8 Conditions)" color={ACCENT3}>
          {differential_diagnosis.map((d, i) => (
            <div key={i} className="mb-3 p-2 rounded" style={{ background: '#fff3e0' }}>
              <div className="fw-bold small" style={{ color: ACCENT3 }}>{d.condition}</div>
              <p className="small mb-0 mt-1">{d.distinguisher}</p>
            </div>
          ))}
        </SectionCard>
      )}

      {key_concepts.length > 0 && (
        <SectionCard title="Key Concepts (17)" color={ACCENT5}>
          <ul className="mb-0 small">
            {key_concepts.map((c, i) => <li key={i} className="mb-1">{c}</li>)}
          </ul>
        </SectionCard>
      )}

      {standards.length > 0 && (
        <SectionCard title="Standards & References (10)" color="#546e7a">
          <ul className="mb-0 small">
            {standards.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
          </ul>
        </SectionCard>
      )}
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function IDSPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/ids/overview`).then(r => r.json()).then(setOverview).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/ids/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/ids/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
        🧬 IDS — MPS II / Hunter Syndrome (Iduronate-2-Sulfatase Deficiency)
      </h2>
      <p className="text-muted small mb-3">
        OMIM 309900 / 300823 · Xq28 · X-Linked Recessive · Males predominantly ·
        ERT: Idursulfase (Elaprase FDA 2006) + Pabinafusp alfa (JR-141 Japan 2021 BBB-crossing) ·
        HSCT controversial · OSA dominant seizure trigger (60–70%) · Airway extreme hazard ·
        40-patient cohort · ~1:100,000–1:170,000 males · Most common MPS in Asia
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
