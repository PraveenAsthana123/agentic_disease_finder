'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — postsynaptic density scaffold protein
const ACCENT2 = '#b71c1c';   // deep red — ABSOLUTE CI / HIGH RISK
const ACCENT3 = '#1b5e20';   // deep green — precision therapy / IGF-1
const ACCENT4 = '#e65100';   // deep orange — regression / SRS warning

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
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header fw-bold" style={{ backgroundColor: '#e8eaf6', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: 11 }}>{text}</span>
  );
}

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading overview…</div>;
  const d = data;
  return (
    <>
      <Alert
        text="⚠️ SHANK3 (22q13.33) / PMS: VGB ABSOLUTE AVOID — visual field monitoring impossible in non-verbal patients. PHT/CBZ HIGH RISK (behaviour worsening). VPA+IGF-1 AVOID (opposing mTOR signals). Abrupt AED withdrawal ABSOLUTE CI — regression trigger. CSWS during regression = urgent overnight EEG + corticosteroids."
        variant="danger"
      />
      <Alert
        text="🧬 PRECISION Rx: IGF-1 (Mecasermin/Increlex) → IGF-1R → PI3K → mTOR → ↑SHANK3 translation from intact allele → restores mGluR5-PSD coupling → E/I rebalance. Phase I (Kolevzon 2014, n=9): social + motor + seizure improvement. Avoid VPA during IGF-1 trial. Post-anaesthetic regression risk 50-65% — pre-op neurology mandatory."
        variant="info"
      />

      <div className="row g-2 mb-4">
        <KPI label="Cohort" value={d.cohort_size} color={ACCENT} />
        <KPI label="With Seizures" value={d.with_seizures} color={ACCENT2} />
        <KPI label="Seizure Prev." value={`${d.seizure_prevalence_pct}%`} color={ACCENT2} />
        <KPI label="Drug-Resistant" value={`${d.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Regression (SRS)" value={`${d.regression_pct}%`} color={ACCENT4} />
        <KPI label="IGF-1 Trial" value={`${d.igf1_trial_pct}%`} color={ACCENT3} />
        <KPI label="On KD" value={`${d.on_kd_pct}%`} color={ACCENT} />
        <KPI label="Mean AEDs" value={d.mean_aed_count} color={ACCENT} />
        <KPI label="Mean Deletion (Mb)" value={d.mean_deletion_mb} color={ACCENT4} />
        <KPI label="Etiology Classes" value={d.etiology_classes} color={ACCENT} />
        <KPI label="Seizure Types" value={d.seizure_types} color={ACCENT} />
      </div>

      <SectionCard title="SHANK3 Postsynaptic Density Scaffold — Biology" borderColor={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <h6 style={{ color: ACCENT }}>SHANK3 PSD Architecture</h6>
            <ul className="small">
              <li><strong>1740 aa, 198 kDa</strong> — largest SHANK family member</li>
              <li>N-terminal Ankyrin Repeat Domain → scaffold assembly</li>
              <li>SH3 + PDZ → binds Homer (mGluR5), GKAP/SAPAP (NMDAR)</li>
              <li>Proline-rich → IRSp53 → actin cytoskeleton → spine morphology</li>
              <li>C-terminal SAM → self-multimerisation → PSD nanodomains</li>
            </ul>
            <h6 style={{ color: ACCENT4 }} className="mt-2">Haploinsufficiency Effect</h6>
            <ul className="small">
              <li>↓40% dendritic spine density (Bozdagi 2010 mouse)</li>
              <li>↓mGluR5-LTD → LTP/LTD imbalance → net excitability ↑</li>
              <li>NMDAR under-anchoring → extra-synaptic NR2B upregulation</li>
              <li>E/I imbalance → epilepsy (30-40%) + ASD (near 100%)</li>
            </ul>
          </div>
          <div className="col-md-6">
            <h6 style={{ color: ACCENT3 }}>IGF-1 Precision Therapy Mechanism</h6>
            <ul className="small">
              <li>IGF-1 → IGF-1R → PI3K → Akt → mTOR-C1</li>
              <li>↑S6K1 + 4EBP1 release → ↑protein translation</li>
              <li>↑SHANK3 protein from intact allele (compensates haploinsufficiency)</li>
              <li>Restores Homer-mGluR5-SHANK3 PSD complex</li>
              <li>↑mGluR5-LTD → normalises E/I balance</li>
            </ul>
            <h6 style={{ color: ACCENT2 }} className="mt-2">VPA + IGF-1 Incompatibility</h6>
            <ul className="small">
              <li>VPA inhibits mTOR (HDAC/TSC pathway)</li>
              <li>→ Blocks IGF-1-driven SHANK3 synthesis</li>
              <li>Switch to LTG or LEV before IGF-1 trial</li>
              <li>4-week VPA washout before IGF-1 initiation</li>
            </ul>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Deletion Size & Severity" borderColor={ACCENT4}>
        <div className="row">
          <div className="col-md-6">
            <h6>Deletion Size → Severity Correlation</h6>
            <table className="table table-sm small">
              <thead><tr><th>Deletion Size</th><th>Key Genes Lost</th><th>Seizure Risk</th></tr></thead>
              <tbody>
                <tr><td>&lt;1 Mb (SHANK3-only)</td><td>SHANK3 ± RABL2B</td><td>~25%</td></tr>
                <tr><td>1–3 Mb</td><td>+ ARSA, ACR</td><td>~35%</td></tr>
                <tr><td>&gt;3 Mb</td><td>+ IB2/MAPK8IP2, ADSL, ACTN1</td><td>~55%</td></tr>
                <tr><td>&gt;5 Mb</td><td>+ multiple additional</td><td>~65%</td></tr>
              </tbody>
            </table>
          </div>
          <div className="col-md-6">
            <h6>Additional Gene Contributions</h6>
            <ul className="small">
              <li><strong>MAPK8IP2 / IB2</strong>: cerebellar coordination disorder</li>
              <li><strong>ADSL</strong>: purine synthesis defect → regression episodes more severe</li>
              <li><strong>ARSA</strong>: metachromatic leukodystrophy (rare second hit)</li>
              <li><strong>SHANK3</strong>: dominant driver for ASD + epilepsy in ALL sizes</li>
            </ul>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="SHANK3 Regression Syndrome (SRS)" borderColor={ACCENT4}>
        <div className="row">
          <div className="col-md-6">
            <h6 style={{ color: ACCENT4 }}>What is SRS?</h6>
            <p className="small">
              50-70% of PMS patients lose previously acquired language and/or motor skills.
              Partially reversible over 6-12 months. Distinct from seizure-related regression.
            </p>
            <p className="small"><strong>Common triggers:</strong> fever, anaesthesia, illness, sleep disruption, AED changes.</p>
            <p className="small"><strong>Clinical signs:</strong> loss of words, motor regression, toilet training loss, ASD behaviour worsening.</p>
          </div>
          <div className="col-md-6">
            <h6>Regression Management Protocol</h6>
            <ul className="small">
              <li>Urgent overnight EEG → rule out CSWS</li>
              <li>CSWS → prednisolone 2 mg/kg × 4 weeks</li>
              <li>Non-CSWS → intensive speech/OT/ABA therapy</li>
              <li>IGF-1 trial if not on it (mTOR upregulation may arrest regression)</li>
              <li>Bayley-4 baseline + 3-month reassessment</li>
            </ul>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Key Standards" borderColor={ACCENT}>
        <div className="row">
          {(d.standards || []).map(s => <Badge key={s} text={s} color={ACCENT} />)}
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Etiology ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiologies, patients } = data;
  const [filter, setFilter] = useState('all');

  const filtered = filter === 'all' ? patients
    : filter === 'seizures' ? patients.filter(p => p.has_seizures)
    : filter === 'igf1' ? patients.filter(p => p.igf1_trial)
    : filter === 'regression' ? patients.filter(p => p.regression_episodes > 0)
    : patients;

  return (
    <>
      <SectionCard title="5 Etiology Classes (40-patient cohort)" borderColor={ACCENT}>
        {(etiologies || []).map((e, i) => (
          <div key={i} className="mb-4 border-bottom pb-3">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{ color: ACCENT }}>{e.etiology}</strong>
              <Badge text={`${e.pct}% (n=${e.n})`} color={ACCENT} />
            </div>
            <PctBar label={e.category} pct={e.pct} color={ACCENT} />
            <p className="small mb-1">{e.mechanism}</p>
            <div className="d-flex flex-wrap gap-2 small">
              <span><strong>Age onset:</strong> {e.age_onset_range}</span>
              <span><strong>AED response:</strong> {e.drug_response}</span>
              <span><strong>Typical AED:</strong> {e.typical_aed}</span>
              <span><strong>Regression risk:</strong> {e.regression_risk}</span>
              <span><strong>IGF-1 candidate:</strong> {e.igf1_candidate ? '✅ Yes' : '❌ No'}</span>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Roster" borderColor={ACCENT}>
        <div className="mb-2 d-flex gap-2 flex-wrap">
          {[['all','All'],['seizures','Has Seizures'],['igf1','IGF-1 Trial'],['regression','Has Regression']].map(([v,l]) => (
            <button key={v} className={`btn btn-sm ${filter===v?'btn-primary':'btn-outline-primary'}`} onClick={() => setFilter(v)}>{l}</button>
          ))}
          <span className="ms-2 text-muted small align-self-center">Showing {filtered.length} / {patients.length}</span>
        </div>
        <div style={{ maxHeight: 380, overflowY: 'auto' }}>
          <table className="table table-sm table-hover small">
            <thead className="table-light sticky-top">
              <tr>
                <th>#</th><th>Name</th><th>Age dx</th><th>Etiology</th>
                <th>Del (Mb)</th><th>Seizures</th><th>AEDs</th>
                <th>Regression</th><th>IGF-1</th><th>DRE</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.name}</td>
                  <td>{p.age_dx}y</td>
                  <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.etiology}</td>
                  <td>{p.deletion_size_mb ?? '—'}</td>
                  <td>{p.has_seizures ? '✅' : '—'}</td>
                  <td>{p.n_aed || '—'}</td>
                  <td>{p.regression_episodes > 0 ? `${p.regression_episodes}×` : '—'}</td>
                  <td>{p.igf1_trial ? '✅' : '—'}</td>
                  <td>{p.drug_resistant ? <span style={{ color: ACCENT2 }}>DRE</span> : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Seizures & Triggers ──────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types, triggers } = data;
  return (
    <>
      <SectionCard title="5 Seizure Types" borderColor={ACCENT}>
        {(seizure_types || []).map((s, i) => (
          <div key={i} className="mb-4 border-bottom pb-3">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{ color: ACCENT }}>{s.type}</strong>
              <Badge text={`${s.prevalence_pct}% prevalence`} color={ACCENT2} />
            </div>
            <PctBar label="Prevalence" pct={s.prevalence_pct} color={ACCENT} />
            <div className="row small mt-1">
              <div className="col-md-4"><strong>Onset:</strong> {s.onset}</div>
              <div className="col-md-4"><strong>Duration:</strong> {s.duration}</div>
              <div className="col-md-4"><strong>EEG:</strong> {s.eeg}</div>
            </div>
            <p className="small mt-1 mb-1"><strong>Semiology:</strong> {s.semiology}</p>
            <div className="alert alert-info py-1 mb-0" style={{ fontSize: 12 }}>
              💡 <strong>Clinical tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="8 Seizure Triggers" borderColor={ACCENT4}>
        {(triggers || []).map((t, i) => (
          <div key={i} className="mb-2 d-flex gap-3 align-items-start border-bottom pb-2">
            <div style={{ minWidth: 120 }}>
              <Badge text={`${t.pct}%`} color={i < 3 ? ACCENT2 : ACCENT4} />
              <div className="small fw-bold mt-1">{t.trigger}</div>
            </div>
            <div className="small flex-grow-1">{t.notes}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments ───────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments, contraindications } = data;
  return (
    <>
      <Alert
        text="🚨 CONTRAINDICATIONS FIRST: VGB ABSOLUTE AVOID (visual monitoring impossible in non-verbal PMS). PHT/CBZ/OXC HIGH RISK (behaviour worsening). VPA+IGF-1 AVOID (opposing mTOR). Abrupt AED withdrawal ABSOLUTE CI (regression trigger). Chronic BZD HIGH RISK (behavioural disinhibition)."
        variant="danger"
      />

      <SectionCard title="Contraindications" borderColor={ACCENT2}>
        {(contraindications || []).map((c, i) => (
          <div key={i} className="mb-3 border-bottom pb-2">
            <div className="d-flex justify-content-between align-items-start">
              <strong style={{ color: ACCENT2 }}>{c.drug}</strong>
              <Badge text={c.risk.split('—')[0].trim()} color={ACCENT2} />
            </div>
            <p className="small mb-1">{c.mechanism}</p>
            <div className="alert alert-warning py-1 mb-0" style={{ fontSize: 12 }}>
              ⚡ <strong>Action:</strong> {c.action}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="8 Treatments" borderColor={ACCENT3}>
        {(treatments || []).map((t, i) => (
          <div key={i} className="mb-4 border-bottom pb-3">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{ color: t.drug.includes('IGF-1') ? ACCENT3 : ACCENT }}>{t.drug}</strong>
              <Badge text={t.level} color={t.drug.includes('IGF-1') ? ACCENT3 : ACCENT} />
            </div>
            <div className="row small">
              <div className="col-md-6">
                <div><strong>Role:</strong> {t.role}</div>
                <div><strong>Dose:</strong> {t.dose}</div>
                <div><strong>MOA:</strong> {t.moa}</div>
              </div>
              <div className="col-md-6">
                <div><strong>Efficacy:</strong> {t.efficacy}</div>
                <div><strong>Monitoring:</strong> {t.monitoring}</div>
              </div>
            </div>
            {t.shank3_note && (
              <div className="alert alert-info py-1 mt-2 mb-0" style={{ fontSize: 12 }}>
                🧬 <strong>SHANK3-specific:</strong> {t.shank3_note}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Lifecycle Management" borderColor={ACCENT}>
        {(data.lifecycle || []).map((stage, i) => (
          <div key={i} className="mb-3 border-bottom pb-2">
            <strong style={{ color: ACCENT }}>{stage.stage}</strong>
            <div className="small mt-1">
              <div><strong>Focus:</strong> {stage.focus}</div>
              <div><strong>Management:</strong> {stage.management}</div>
              <div><strong>Seizure risk:</strong> {stage.seizure_risk}</div>
              <div><strong>Precision:</strong> {stage.precision}</div>
            </div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { concepts, thresholds, references, standards } = data;
  return (
    <>
      <SectionCard title="15 Key Concepts" borderColor={ACCENT}>
        {(concepts || []).map((c, i) => (
          <div key={i} className="mb-3 border-bottom pb-2">
            <strong style={{ color: ACCENT }}>{c.concept}</strong>
            <p className="small mb-0 mt-1">{c.definition}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="12 Clinical Thresholds" borderColor={ACCENT4}>
        <table className="table table-sm small">
          <thead><tr><th>Threshold</th><th>Value</th><th>Action</th></tr></thead>
          <tbody>
            {(thresholds || []).map((t, i) => (
              <tr key={i}>
                <td><strong>{t.threshold}</strong></td>
                <td style={{ color: ACCENT4 }}>{t.value}</td>
                <td>{t.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </SectionCard>

      <SectionCard title="12 Evidence Standards" borderColor={ACCENT}>
        <div className="row">
          {(standards || []).map((s, i) => (
            <div key={i} className="col-md-6 mb-2 small">
              <Badge text={s.standard} color={ACCENT} />
              <span className="ms-2 text-muted">{s.scope}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="6 Key References" borderColor={ACCENT}>
        {(references || []).map((r, i) => (
          <div key={i} className="mb-2 border-bottom pb-1 small">
            <strong style={{ color: ACCENT }}>{r.ref}</strong>
            <p className="mb-0 text-muted">{r.summary}</p>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function SHANK3Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/shank3/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(() => setError('Failed to load overview'));
    fetch(`${API}/api/shank3/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(() => setError('Failed to load breakdown'));
    fetch(`${API}/api/shank3/definitions`)
      .then(r => r.json()).then(setDefinitions)
      .catch(() => setError('Failed to load definitions'));
  }, []);

  const tabContent = [
    <OverviewTab key="ov" data={overview} />,
    <PatientsTab key="pt" data={breakdown} />,
    <SeizuresTab key="sz" data={breakdown} />,
    <TreatmentsTab key="tx" data={breakdown} />,
    <DefinitionsTab key="df" data={definitions} />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h2 style={{ color: ACCENT }}>
          🧬 SHANK3 Epilepsy — Phelan-McDermid Syndrome (22q13.33)
        </h2>
        <p className="text-muted small mb-0">
          Postsynaptic Density Scaffold · mGluR5-AMPA-NMDA Scaffolding · IGF-1 Precision Therapy ·
          SHANK3 Regression Syndrome (SRS) · VGB AVOID · PMS #606232 · 40-patient cohort
        </p>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tabContent[tab]}
    </div>
  );
}
