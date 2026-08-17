'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a3a5c';   // deep navy — voltage-gated K+ channelopathy / Kv1 family
const ACCENT2 = '#7b1c1c';   // deep crimson — contraindications / danger
const ACCENT3 = '#1a4a2e';   // forest green — 4-AP precision therapy / EA1 response
const ACCENT4 = '#4a1a5c';   // deep purple — genetics / myokymia / neuromyotonia

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#eef2f7', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function TabBtn({ label, active, onClick }) {
  return (
    <button
      className={`btn btn-sm me-2 mb-2 ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

function Badge({ text, color = ACCENT }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: 11 }}>
      {text}
    </span>
  );
}

// ── Tab 1: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const {
    gene, locus, protein, mechanism, key_aha, inheritance,
    n_patients, epilepsy_pct, drug_resistant_pct, seizure_free_pct,
    on_4ap_pct, on_cbz_pct, on_kd_pct, polg_done_pct, hla_done_pct,
    ecg_done_pct, avg_sara_score, avg_ea1_attacks_per_month,
    tiagabine_alert, fourAP_alert, cbz_alert, polg_alert, ecg_alert,
    contraindications_summary, thresholds, references,
  } = data;

  return (
    <div>
      <div className="alert alert-primary mb-3" style={{ fontSize: 14 }}>
        <strong>⚡ KCNA1 Epilepsy / Episodic Ataxia Type 1 (EA1 / Kv1.1 / 12p13.32)</strong><br />
        <span className="text-muted">{gene} · {locus}</span><br />
        <span style={{ fontSize: 13 }}>{protein}</span>
      </div>

      <Alert text={`🚫 ${tiagabine_alert}`} variant="danger" />
      <Alert text={`⚡ ${fourAP_alert}`} variant="warning" />
      <Alert text={`⚠️ ${cbz_alert}`} variant="warning" />
      <Alert text={`🧬 ${polg_alert}`} variant="warning" />
      <Alert text={`🫀 ${ecg_alert}`} variant="info" />

      <SectionCard title="🧬 Pathophysiology — Kv1.1 LOF → EA1 + Myokymia + Epilepsy (25%)" borderColor={ACCENT4}>
        <p style={{ fontSize: 13 }}>{mechanism}</p>
        <div className="row g-2 mt-1">
          <div className="col-md-6">
            <div className="card border-success">
              <div className="card-header fw-bold text-success" style={{ fontSize: 12 }}>✅ EA1-Pure (75%) — Precision 4-AP Therapy</div>
              <div className="card-body py-2" style={{ fontSize: 12 }}>
                Brief startle-triggered ataxia (seconds) + interictal myokymia<br />
                <strong>Hallmark:</strong> periorbital + hand rippling (myokymia)<br />
                <strong>Precision Rx:</strong> 4-aminopyridine (4-AP) Level A<br />
                <Badge text="4-AP FIRST-LINE" color="#198754" />
                <Badge text="SARA score monitoring" color="#198754" />
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card border-danger">
              <div className="card-header fw-bold text-danger" style={{ fontSize: 12 }}>⚠️ EA1+Epilepsy (25%) — Dual Therapy</div>
              <div className="card-body py-2" style={{ fontSize: 12 }}>
                Focal TLE / GTCS + EA1 attacks + myokymia<br />
                <strong>Rule:</strong> CBZ for seizures FIRST, then 4-AP for EA1<br />
                <strong>Risk:</strong> SUDEP if uncontrolled GTCS<br />
                <Badge text="SUDEP counselling" color="#dc3545" />
                <Badge text="HLA-B*15:02 before CBZ" color="#dc3545" />
              </div>
            </div>
          </div>
        </div>
      </SectionCard>

      <div className="row g-2 mb-4">
        <KPI label="Cohort (N)" value={n_patients} color={ACCENT} />
        <KPI label="With Epilepsy" value={`${epilepsy_pct}%`} color={ACCENT2} />
        <KPI label="Drug-Resistant" value={`${drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Seizure-Free" value={`${seizure_free_pct}%`} color={ACCENT3} />
        <KPI label="On 4-AP" value={`${on_4ap_pct}%`} color={ACCENT3} />
        <KPI label="On CBZ/OXC" value={`${on_cbz_pct}%`} color={ACCENT} />
        <KPI label="On KD" value={`${on_kd_pct}%`} color={ACCENT4} />
        <KPI label="POLG Done" value={`${polg_done_pct}%`} color={ACCENT4} />
        <KPI label="HLA Done" value={`${hla_done_pct}%`} color={ACCENT} />
        <KPI label="ECG Done" value={`${ecg_done_pct}%`} color={ACCENT3} />
        <KPI label="Avg SARA" value={avg_sara_score} color={ACCENT2} />
        <KPI label="EA1 Attacks/Mo" value={avg_ea1_attacks_per_month} color={ACCENT4} />
      </div>

      <SectionCard title="🚫 Contraindications Summary" borderColor={ACCENT2}>
        {contraindications_summary && contraindications_summary.map((c, i) => (
          <div key={i} className="d-flex align-items-start gap-2 mb-2">
            <span className="badge" style={{ backgroundColor: i === 0 ? '#dc3545' : '#fd7e14', fontSize: 10, minWidth: 70 }}>
              {i === 0 ? 'ABSOLUTE CI' : 'HIGH RISK'}
            </span>
            <span style={{ fontSize: 12 }}>{c}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Clinical Thresholds" borderColor={ACCENT}>
        <ul className="mb-0" style={{ fontSize: 12 }}>
          {thresholds && thresholds.map((t, i) => <li key={i} className="mb-1">{t}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="📚 Key References" borderColor={ACCENT3}>
        <div style={{ fontSize: 12 }}>
          {references && references.map((r, i) => (
            <div key={i} className="mb-1">
              <Badge text={`Ref ${i + 1}`} color={ACCENT3} />
              {r}
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🧬 Gene & Inheritance" borderColor={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <ul className="list-unstyled mb-0" style={{ fontSize: 13 }}>
              <li><strong>Gene:</strong> {gene} ({locus})</li>
              <li><strong>Protein:</strong> {protein}</li>
            </ul>
          </div>
          <div className="col-md-6">
            <ul className="list-unstyled mb-0" style={{ fontSize: 13 }}>
              <li><strong>Inheritance:</strong> {inheritance}</li>
              <li><strong>Key AHA:</strong> {key_aha && key_aha.substring(0, 120)}…</li>
            </ul>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Etiology ────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiology_catalog, patients_sample, summary } = data;
  const colors = [ACCENT3, ACCENT2, ACCENT4, ACCENT, '#8b5a00'];

  return (
    <div>
      {summary && (
        <div className="row g-2 mb-3">
          <KPI label="Cohort (N)" value={summary.n} color={ACCENT} />
          <KPI label="Epilepsy" value={`${summary.epilepsy_pct}%`} color={ACCENT2} />
          <KPI label="Drug-Resistant" value={`${summary.drug_resistant_pct}%`} color={ACCENT2} />
          <KPI label="On 4-AP" value={`${summary.on_4ap_pct}%`} color={ACCENT3} />
          <KPI label="POLG Done" value={`${summary.polg_done_pct}%`} color={ACCENT4} />
          <KPI label="HLA Done" value={`${summary.hla_done_pct}%`} color={ACCENT} />
        </div>
      )}

      <SectionCard title="🧬 Etiology Catalog — KCNA1 (40 Patients, 5 Classes)" borderColor={ACCENT4}>
        <div className="row g-2 mb-3">
          {etiology_catalog && etiology_catalog.map((e, i) => (
            <div className="col-md-4" key={i}>
              <div className="card h-100 border-0 shadow-sm">
                <div className="card-body py-2" style={{ borderLeft: `4px solid ${colors[i % colors.length]}` }}>
                  <div className="fw-bold" style={{ fontSize: 12, color: colors[i % colors.length] }}>
                    {e.category} ({e.pct}%, N={e.n})
                  </div>
                  <div style={{ fontSize: 11 }} className="text-muted mt-1">{e.etiology}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
        {etiology_catalog && etiology_catalog.map((e, i) => (
          <PctBar key={i} label={`${e.category} (N=${e.n})`} pct={e.pct} color={colors[i % colors.length]} />
        ))}
      </SectionCard>

      <SectionCard title="🔬 Mechanistic Detail by Etiology Class" borderColor={ACCENT}>
        {etiology_catalog && etiology_catalog.map((e, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: i < etiology_catalog.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="fw-bold mb-1" style={{ color: colors[i % colors.length], fontSize: 13 }}>
              Class {i + 1}: {e.etiology} — {e.pct}% (N={e.n})
            </div>
            <div className="mb-1"><strong style={{ fontSize: 12 }}>Mechanism:</strong>
              <span style={{ fontSize: 12 }}> {e.mechanism}</span></div>
            <div className="mb-1"><strong style={{ fontSize: 12 }}>EEG/EMG Signature:</strong>
              <span style={{ fontSize: 12 }}> {e.eeg_correlate}</span></div>
            <div className="mb-1"><strong style={{ fontSize: 12 }}>MRI:</strong>
              <span style={{ fontSize: 12 }}> {e.mri_finding}</span></div>
            <div className="alert alert-light py-1 px-2 mb-0" style={{ fontSize: 12 }}>
              <strong>Clinical Note:</strong> {e.clinical_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="👥 Patient Sample (15 / 40)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-hover" style={{ fontSize: 11 }}>
            <thead>
              <tr>
                <th>ID</th><th>Name</th><th>Age</th><th>Etiology</th>
                <th>Epilepsy</th><th>SARA</th><th>Attacks/Mo</th>
                <th>4-AP</th><th>CBZ</th><th>POLG</th><th>ECG</th>
              </tr>
            </thead>
            <tbody>
              {patients_sample && patients_sample.map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td>{p.name}</td>
                  <td>{p.age_years}y</td>
                  <td style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.etiology}</td>
                  <td><span className={`badge ${p.has_epilepsy ? 'bg-danger' : 'bg-success'}`}>{p.has_epilepsy ? 'Y' : 'N'}</span></td>
                  <td>{p.sara_score}</td>
                  <td>{p.ea1_attacks_per_month}</td>
                  <td><span className={`badge ${p.on_4ap ? 'bg-success' : 'bg-secondary'}`}>{p.on_4ap ? 'Y' : 'N'}</span></td>
                  <td><span className={`badge ${p.on_cbz ? 'bg-primary' : 'bg-secondary'}`}>{p.on_cbz ? 'Y' : 'N'}</span></td>
                  <td><span className={`badge ${p.polg_tested === 'Y' ? 'bg-success' : 'bg-warning text-dark'}`}>{p.polg_tested}</span></td>
                  <td><span className={`badge ${p.ecg_done ? 'bg-success' : 'bg-warning text-dark'}`}>{p.ecg_done ? 'Y' : 'N'}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Seizure Types & Triggers ──────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_detail, trigger_detail } = data;
  const trigColors = [ACCENT2, ACCENT, ACCENT3, ACCENT4, '#8b5a00', '#1a5c4a', '#5c1a4a', '#2e2e7a'];

  return (
    <div>
      <Alert text="⚡ EA1 ATTACK vs SEIZURE: EA1 = brief (seconds), diffuse EEG theta slowing, no postictal state. Seizure = structured ictal discharge, minutes duration, postictal state. Video-EEG is essential to distinguish both in EA1+epilepsy patients." variant="info" />
      <Alert text="🔬 MYOKYMIA HALLMARK: Periorbital and hand rippling visible on inspection. EMG shows doublets/triplets/multiplets at 50-150 Hz. Absent in EA2/CACNA1A — key differentiating finding." variant="warning" />

      <SectionCard title="⚡ Seizure / Episodic Event Types (4)" borderColor={ACCENT}>
        {seizure_detail && seizure_detail.map((s, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: i < seizure_detail.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-bold" style={{ color: ACCENT, fontSize: 13 }}>{s.type}</span>
              <Badge text={`${s.prevalence_pct}%`} color={ACCENT} />
            </div>
            <div className="text-muted small mb-1">Age window: {s.age_window}</div>
            <PctBar label="Prevalence in cohort" pct={s.prevalence_pct} color={ACCENT} />
            <div className="mb-1"><strong style={{ fontSize: 12 }}>EEG/EMG Correlate:</strong>
              <span style={{ fontSize: 12 }}> {s.eeg_correlate}</span></div>
            <div className="alert alert-light py-1 px-2 mb-0" style={{ fontSize: 12 }}>
              <strong>Clinical Tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🌡️ EA1 / Seizure Triggers (8)" borderColor={ACCENT2}>
        <div className="mb-3">
          {trigger_detail && trigger_detail.map((t, i) => (
            <PctBar key={i} label={t.trigger} pct={t.prevalence_pct} color={trigColors[i % trigColors.length]} />
          ))}
        </div>
        {trigger_detail && trigger_detail.map((t, i) => (
          <div key={i} className="mb-3 pb-3" style={{ borderBottom: i < trigger_detail.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="fw-bold mb-1" style={{ color: trigColors[i % trigColors.length], fontSize: 13 }}>
              {t.trigger} — {t.prevalence_pct}%
            </div>
            <div style={{ fontSize: 12 }}><strong>Mechanism:</strong> {t.mechanism}</div>
            <div style={{ fontSize: 12 }} className="mt-1 alert alert-light py-1 px-2 mb-0">
              <strong>Management:</strong> {t.clinical_management}
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ─────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatment_detail, contraindication_detail, monitoring, lifecycle } = data;
  const txColors = [ACCENT3, ACCENT, ACCENT4, ACCENT2, '#198754', '#6f42c1', '#0dcaf0'];

  return (
    <div>
      <Alert text="⚡ 4-AP RULE: Start CBZ/LEV for seizure control FIRST (in EA1+epilepsy). Add 4-AP ONLY after ≥3 months seizure-free. 4-AP without seizure control risks acute seizure escalation in EA1+epilepsy." variant="danger" />
      <Alert text="🫀 ECG MANDATORY before 4-AP and mexiletine. QTc >450ms (men) / >470ms (women) → STOP. Monitor 2 weeks after each dose increase, then 3-monthly." variant="warning" />
      <Alert text="🧬 POLG MANDATORY before VPA — Alpers-Huttenlocher fatal hepatic failure." variant="danger" />

      <SectionCard title="💊 Treatments (7)" borderColor={ACCENT3}>
        {treatment_detail && treatment_detail.map((t, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: i < treatment_detail.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="d-flex flex-wrap align-items-center gap-2 mb-1">
              <span className="fw-bold" style={{ color: txColors[i % txColors.length], fontSize: 14 }}>{t.drug}</span>
              <Badge text={t.level.split('—')[0].trim()} color={txColors[i % txColors.length]} />
            </div>
            <div className="text-muted small mb-2">{t.level}</div>
            <div className="row g-2" style={{ fontSize: 12 }}>
              <div className="col-md-6">
                <strong>Dose:</strong> {t.dose}<br />
                <strong>MOA:</strong> {t.moa}
              </div>
              <div className="col-md-6">
                <strong>Efficacy:</strong> {t.efficacy}<br />
                <strong>Safety:</strong> {t.safety}
              </div>
            </div>
            <div className="alert alert-light py-1 px-2 mt-2 mb-1" style={{ fontSize: 12 }}>
              <strong>Monitoring:</strong> {t.monitoring}
            </div>
            <div className="alert alert-info py-1 px-2 mb-0" style={{ fontSize: 12 }}>
              <strong>KCNA1 Note:</strong> {t.kcna1_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications (5)" borderColor={ACCENT2}>
        {contraindication_detail && contraindication_detail.map((c, i) => (
          <div key={i} className="mb-3 pb-3" style={{ borderBottom: i < contraindication_detail.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="d-flex align-items-center gap-2 mb-1">
              <Badge
                text={c.severity.includes('ABSOLUTE') ? 'ABSOLUTE CI' : c.severity.includes('HIGH') ? 'HIGH RISK' : 'MODERATE'}
                color={c.severity.includes('ABSOLUTE') ? '#dc3545' : c.severity.includes('HIGH') ? '#fd7e14' : '#ffc107'}
              />
              <span className="fw-bold" style={{ fontSize: 13 }}>{c.drug}</span>
            </div>
            <div className="text-danger small mb-1"><strong>Risk:</strong> {c.risk}</div>
            <div style={{ fontSize: 12 }}>{c.mechanism}</div>
            <div className="alert alert-danger py-1 px-2 mt-1 mb-0" style={{ fontSize: 12 }}>
              <strong>Action:</strong> {c.action}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📋 Monitoring (10 Items)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm" style={{ fontSize: 12 }}>
            <thead><tr><th>Item</th><th>Timing</th><th>Rationale</th></tr></thead>
            <tbody>
              {monitoring && monitoring.map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold">{m.item}</td>
                  <td className="text-muted">{m.timing}</td>
                  <td>{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🗓️ Lifecycle Windows (6)" borderColor={ACCENT}>
        {lifecycle && lifecycle.map((lc, i) => (
          <div key={i} className="mb-3 pb-2" style={{ borderBottom: i < lifecycle.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="fw-bold" style={{ color: ACCENT, fontSize: 13 }}>{lc.window}</div>
            <div className="text-muted small mb-1">{lc.age}</div>
            <div className="d-flex flex-wrap gap-1 mb-1">
              {lc.key_events && lc.key_events.map((ev, j) => <Badge key={j} text={ev} color={ACCENT4} />)}
            </div>
            <div style={{ fontSize: 12 }} className="text-muted">{lc.note}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 5: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { concepts, thresholds, standards, references } = data;

  return (
    <div>
      <SectionCard title="📖 Key Concepts (14)" borderColor={ACCENT}>
        {concepts && concepts.map((c, i) => (
          <div key={i} className="mb-3 pb-2" style={{ borderBottom: i < concepts.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="fw-bold" style={{ color: ACCENT, fontSize: 13 }}>{c.term}</div>
            <div style={{ fontSize: 12 }} className="mt-1">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Clinical Thresholds (10)" borderColor={ACCENT2}>
        <ul className="mb-0" style={{ fontSize: 12 }}>
          {thresholds && thresholds.map((t, i) => <li key={i} className="mb-2">{t}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="📋 Clinical Standards (8)" borderColor={ACCENT3}>
        <ul className="mb-0" style={{ fontSize: 12 }}>
          {standards && standards.map((s, i) => <li key={i} className="mb-2">{s}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="📚 References (6)" borderColor={ACCENT4}>
        <ol className="mb-0" style={{ fontSize: 12 }}>
          {references && references.map((r, i) => <li key={i} className="mb-2">{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function KCNA1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/kcna1/overview`).then(r => r.json()),
      fetch(`${API}/api/kcna1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/kcna1/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setError(String(e)));
  }, []);

  if (error) return <div className="alert alert-danger m-3">Error: {error}</div>;

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-3 mb-3">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            ⚡ KCNA1 Epilepsy / Episodic Ataxia Type 1 (EA1)
          </h4>
          <div className="text-muted small">
            Kv1.1 Shaker-Type K⁺ Channel · 12p13.32 · EA1 + Myokymia + Epilepsy Spectrum
          </div>
        </div>
        <div className="ms-auto">
          <span className="badge" style={{ backgroundColor: ACCENT3, fontSize: 11 }}>
            4-AP Precision — Level A
          </span>
        </div>
      </div>

      <div className="mb-3">
        {TABS.map((t, i) => (
          <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
