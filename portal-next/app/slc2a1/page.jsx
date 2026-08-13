'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a3a5c';   // deep navy — SLC2A1 / GLUT1-DS
const ACCENT2 = '#7b3f00';   // dark amber — methylxanthine CI / danger
const ACCENT3 = '#1a5c2a';   // dark green — KD precision therapy / safe
const ACCENT4 = '#4a235a';   // purple — PED / movement disorder

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#eaf3fb', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function TabBtn({ label, active, onClick }) {
  return (
    <button
      className={`btn btn-sm me-1 mb-1 ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
      style={active ? { backgroundColor: ACCENT, borderColor: ACCENT } : {}}
      onClick={onClick}
    >{label}</button>
  );
}

// ── Tab: Overview ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const k = data.kpis || {};

  return (
    <>
      <div className="alert alert-primary mb-4" style={{ borderLeft: `5px solid ${ACCENT}`, fontSize: 14 }}>
        <strong>🧬 {data.syndrome}</strong><br />
        Gene: <strong>{data.gene}</strong> · {data.inheritance}<br />
        <strong>EEG hallmark:</strong> {data.eeg_hallmark}<br />
        <strong>Key biomarker:</strong> {data.key_biomarker}
      </div>

      {/* METHYLXANTHINE DANGER Banner */}
      <div className="alert alert-danger mb-4" style={{ borderLeft: `5px solid #dc3545`, fontSize: 14 }}>
        <strong>🚨 METHYLXANTHINES ABSOLUTE CONTRAINDICATION — ALL GLUT1-DS PATIENTS</strong><br />
        Caffeine (coffee/cola/energy drinks/chocolate) + theophylline + aminophylline = <strong>competitive GLUT1 inhibition</strong> → acute seizure exacerbation.<br />
        Document as <strong>ALLERGY</strong> in EMR. Medical ID bracelet. <strong>No caffeine in any form.</strong><br />
        A&amp;E / anaesthetics: <strong>NO aminophylline</strong> — use salbutamol for bronchospasm.
      </div>

      {/* KD FIRST-LINE Banner */}
      <div className="alert alert-success mb-4" style={{ borderLeft: `5px solid ${ACCENT3}`, fontSize: 14 }}>
        <strong>✅ PRECISION THERAPY: {data.precision_therapy}</strong><br />
        <span className="text-muted">KD bypasses GLUT1 via MCT1 (intact) → ketones as alternative CNS fuel. &gt;90% seizure-free. Start at DIAGNOSIS — do NOT wait for 2-AED failures. PED resolves in &gt;85%.</span>
      </div>

      {/* LP FIRST Banner */}
      <div className="alert alert-warning mb-4" style={{ borderLeft: `5px solid ${ACCENT2}`, fontSize: 13 }}>
        <strong>⚠️ LP BEFORE KD — MANDATORY:</strong> CSF glucose normalises on KD → false negative if lumbar puncture delayed.<br />
        Fasting LP (≥4h) before first dose of ketogenic diet. Target: CSF:plasma glucose ratio &lt;0.45 + CSF glucose &lt;2.2 mmol/L.
      </div>

      {/* PED Banner */}
      <div className="alert alert-info mb-4" style={{ borderLeft: `5px solid ${ACCENT4}`, fontSize: 13 }}>
        <strong>🏃 PAROXYSMAL EXERCISE-INDUCED DYSKINESIA (PED) PATHOGNOMONIC:</strong> Involuntary movements triggered by 5-20 min exercise, relieved by rest. EEG normal during PED (not epileptic — do NOT add AEDs for PED). KD resolves PED in &gt;85%.
      </div>

      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Patients (N)" value={k.total_patients} color={ACCENT} />
        <KPI label="PED Present" value={`${k.ped_present} (${k.ped_pct}%)`} color={ACCENT4} />
        <KPI label="On KD" value={`${k.on_kd} (${k.kd_pct}%)`} color={ACCENT3} />
        <KPI label="KD-Controlled" value={`${k.kd_controlled} (${k.kd_controlled_pct}%)`} color={ACCENT3} />
        <KPI label="Methylxanthine Hx" value={`${k.methylxanthine_exposure_hx} (${k.methylxanthine_pct}%)`} color="#dc3545" />
        <KPI label="Avg CSF:Plasma" value={k.avg_csf_plasma_ratio} color={ACCENT2} />
      </div>

      {/* Critical Alerts */}
      <SectionCard title="⚠️ Critical Clinical Alerts — SLC2A1 / GLUT1-DS" borderColor="#dc3545">
        {(data.critical_alerts || []).map((a, i) => (
          <div key={i} className={`alert alert-${a.color} mb-2 py-2`} style={{ fontSize: 13 }}>
            <strong>[{a.severity}]</strong> {a.alert}<br />
            <span className="text-muted">{a.action}</span>
          </div>
        ))}
      </SectionCard>

      {/* Management Pathway */}
      <SectionCard title="🗺️ GLUT1-DS Management Pathway" borderColor={ACCENT3}>
        <p style={{ fontSize: 13 }}>{data.pathway_summary}</p>
      </SectionCard>

      {/* Standards */}
      <SectionCard title="📋 Clinical Standards & References" borderColor={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <strong>Standards (8):</strong>
            <ul className="small mb-2">
              {(data.standards || []).map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>
          <div className="col-md-6">
            <strong>Key References (6):</strong>
            <ul className="small">
              {(data.references || []).map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </div>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Etiology ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiology_catalog = [], patients = [] } = data;

  return (
    <>
      {/* Etiology Bars */}
      <SectionCard title="🔬 Etiology Distribution (5 Classes — N=41 GLUT1-DS Cohort)" borderColor={ACCENT}>
        {etiology_catalog.map((e, i) => (
          <div key={i} className="mb-4">
            <div className="d-flex align-items-center mb-1">
              <span className="badge me-2" style={{ backgroundColor: [ACCENT, ACCENT3, '#d35400', '#2980b9', '#8e44ad'][i], fontSize: 11 }}>
                {e.n} pts ({e.pct}%)
              </span>
              <strong style={{ fontSize: 13 }}>{e.etiology}</strong>
            </div>
            <div className="progress mb-2" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: [ACCENT, ACCENT3, '#d35400', '#2980b9', '#8e44ad'][i] }} />
            </div>
            <div style={{ fontSize: 12, color: '#444' }}><strong>Mechanism:</strong> {e.mechanism}</div>
            <div style={{ fontSize: 12, color: '#444' }} className="mt-1"><strong>EEG:</strong> {e.eeg_signature}</div>
            <div style={{ fontSize: 12, color: '#555', backgroundColor: '#f0f8ff', padding: '6px 10px', borderRadius: 4, marginTop: 6 }}>
              <strong>Clinical note:</strong> {e.clinical_note}
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Patient Table */}
      <SectionCard title={`👤 Patient Roster — ${patients.length} patients`} borderColor={ACCENT}>
        <div style={{ overflowX: 'auto' }}>
          <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Age (mo)</th><th>Sex</th><th>Onset (y)</th>
                <th>Category</th><th>Phase</th><th>CSF:Plasma</th>
                <th>β-OHB</th><th>KD</th><th>PED</th><th>Treatment</th><th>Control</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.id}>
                  <td><strong>{p.id}</strong></td>
                  <td>{p.age_months}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_years}</td>
                  <td style={{ maxWidth: 180, whiteSpace: 'normal', fontSize: 11 }}>{p.category}</td>
                  <td style={{ fontSize: 11 }}>{p.disease_phase}</td>
                  <td>
                    <span style={{ color: p.csf_plasma_ratio < 0.45 ? '#dc3545' : '#198754', fontWeight: 'bold' }}>
                      {p.csf_plasma_ratio}
                    </span>
                  </td>
                  <td style={{ color: p.beta_ohb_mmol >= 2.0 && p.beta_ohb_mmol <= 4.0 ? '#198754' : '#dc3545' }}>
                    {p.beta_ohb_mmol}
                  </td>
                  <td>{p.on_kd ? <span className="badge bg-success">{p.kd_ratio}</span> : <span className="badge bg-secondary">No</span>}</td>
                  <td>{p.ped_present ? <span className="badge bg-warning text-dark">Yes</span> : '—'}</td>
                  <td>{p.current_treatment}</td>
                  <td>
                    <span className="badge" style={{ backgroundColor: p.seizure_control_color, fontSize: 10 }}>
                      {p.seizure_control}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Seizure Types & Triggers ─────────────────────────────────────────────
function SeizureTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types = [], triggers = [] } = data;

  return (
    <>
      <SectionCard title="⚡ Seizure Types (N=41 GLUT1-DS Cohort)" borderColor={ACCENT2}>
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-4">
            <div className="d-flex align-items-center mb-1">
              <span className="badge me-2" style={{ backgroundColor: [ACCENT, ACCENT4, ACCENT2, '#2980b9'][i], fontSize: 11 }}>
                {s.prevalence_pct}%
              </span>
              <strong>{s.type}</strong>
              <span className="text-muted small ms-2">Onset: {s.onset_age}</span>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, backgroundColor: [ACCENT, ACCENT4, ACCENT2, '#2980b9'][i] }} />
            </div>
            <div style={{ fontSize: 12, color: '#444' }}><strong>EEG correlate:</strong> {s.eeg_correlate}</div>
            <div style={{ fontSize: 12, backgroundColor: '#fff8e1', padding: '6px 10px', borderRadius: 4, marginTop: 4 }}>
              <strong>Clinical tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🎯 Seizure Triggers (8 Core — GLUT1-DS Specific)" borderColor={ACCENT3}>
        <div className="row">
          {triggers.map((t, i) => (
            <div key={i} className="col-md-6 mb-3">
              <div className="card h-100" style={{ borderLeft: `3px solid ${i < 3 ? '#dc3545' : i < 5 ? '#fd7e14' : ACCENT}` }}>
                <div className="card-body p-2">
                  <div className="d-flex justify-content-between align-items-center mb-1">
                    <strong style={{ fontSize: 12 }}>{t.trigger}</strong>
                    <span className="badge" style={{
                      backgroundColor: i < 3 ? '#dc3545' : i < 5 ? '#fd7e14' : ACCENT,
                      fontSize: 10
                    }}>{t.prevalence_pct}%</span>
                  </div>
                  <div className="progress mb-1" style={{ height: 5 }}>
                    <div className="progress-bar" style={{
                      width: `${t.prevalence_pct}%`,
                      backgroundColor: i < 3 ? '#dc3545' : i < 5 ? '#fd7e14' : ACCENT
                    }} />
                  </div>
                  <div style={{ fontSize: 11, color: '#555' }}><strong>Mechanism:</strong> {t.mechanism}</div>
                  <div style={{ fontSize: 11, color: '#333', backgroundColor: '#f0fff0', padding: '4px 6px', borderRadius: 3, marginTop: 4 }}>
                    <strong>Management:</strong> {t.management}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments ──────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments = [], contraindications = [] } = data;

  const evidenceColor = {
    'Level A': '#198754',
    'Level B': '#0d6efd',
    'Level C': '#6c757d',
  };

  return (
    <>
      <SectionCard title="💊 Treatments — 8 AEDs / Interventions (SLC2A1 / GLUT1-DS)" borderColor={ACCENT3}>
        {treatments.map((t, i) => (
          <div key={i} className="mb-4 p-3" style={{ borderLeft: `3px solid ${evidenceColor[t.evidence] || '#6c757d'}`, backgroundColor: i === 0 ? '#f0fff4' : '#fafafa' }}>
            <div className="d-flex align-items-center mb-1">
              <span className="badge me-2" style={{ backgroundColor: evidenceColor[t.evidence] || '#6c757d', fontSize: 11 }}>{t.evidence}</span>
              <strong>{t.name}</strong>
              <span className="text-muted small ms-2">— {t.line}</span>
            </div>
            <div className="row mt-2" style={{ fontSize: 12 }}>
              <div className="col-md-6">
                <div><strong>Dose:</strong> {t.dose}</div>
                <div className="mt-1"><strong>MOA:</strong> {t.moa}</div>
              </div>
              <div className="col-md-6">
                <div><strong>Efficacy:</strong> {t.efficacy}</div>
                <div className="mt-1"><strong>Safety:</strong> {t.safety}</div>
                <div className="mt-1"><strong>Monitoring:</strong> <span className="text-muted">{t.monitoring}</span></div>
              </div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications (4 GLUT1-DS-Specific)" borderColor="#dc3545">
        {contraindications.map((c, i) => (
          <div key={i} className="mb-4 p-3" style={{ borderLeft: `4px solid ${c.color}`, backgroundColor: '#fff8f8' }}>
            <div className="d-flex align-items-center mb-1">
              <span className="badge me-2" style={{ backgroundColor: c.color, fontSize: 11 }}>{c.severity}</span>
              <strong style={{ fontSize: 13 }}>{c.name}</strong>
            </div>
            <div style={{ fontSize: 12, color: '#444' }} className="mt-1">
              <strong>Mechanism:</strong> {c.mechanism}
            </div>
            <div style={{ fontSize: 12, backgroundColor: '#fff0f0', padding: '6px 10px', borderRadius: 4, marginTop: 6 }}>
              <strong>Clinical action:</strong> {c.clinical_action}
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
  const { definitions = [], thresholds = [], monitoring = [], lifecycle = [] } = data;

  const [open, setOpen] = useState(null);

  return (
    <>
      <SectionCard title="📖 Key Concepts / Definitions (14 — GLUT1-DS)" borderColor={ACCENT}>
        {definitions.map((d, i) => (
          <div key={i} className="mb-2">
            <button
              className="btn btn-outline-secondary btn-sm w-100 text-start"
              style={{ fontSize: 12 }}
              onClick={() => setOpen(open === i ? null : i)}
            >
              <strong>{d.term}</strong>
            </button>
            {open === i && (
              <div className="p-2 border-start border-primary ps-3 mt-1" style={{ fontSize: 12 }}>
                {d.definition}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Thresholds (10 — GLUT1-DS Clinical Decision Points)" borderColor={ACCENT2}>
        <div className="row">
          {thresholds.map((t, i) => (
            <div key={i} className="col-md-6 mb-2">
              <div className="p-2 rounded" style={{ backgroundColor: '#fff8e1', fontSize: 12 }}>
                <span className="badge me-2" style={{ backgroundColor: ACCENT2 }}>{t.value}</span>
                {t.threshold}
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🔭 Monitoring Plan (8 Items)" borderColor={ACCENT3}>
        {monitoring.map((m, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between">
              <strong style={{ fontSize: 12 }}>{m.item}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT, fontSize: 10 }}>{m.frequency}</span>
            </div>
            <div style={{ fontSize: 12, color: '#555', marginTop: 4 }}>{m.rationale}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🕐 Disease Lifecycle (6 Windows)" borderColor={ACCENT4}>
        {lifecycle.map((l, i) => (
          <div key={i} className="mb-3 p-2" style={{ borderLeft: `3px solid ${ACCENT4}`, backgroundColor: i % 2 === 0 ? '#f8f0ff' : '#fff' }}>
            <div className="fw-bold" style={{ fontSize: 12, color: ACCENT4 }}>{l.phase} <span className="text-muted fw-normal">({l.window})</span></div>
            <div style={{ fontSize: 12, color: '#444' }}>{l.focus}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function SLC2A1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/slc2a1/overview`)
      .then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/slc2a1/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/slc2a1/definitions`)
      .then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="mb-3">
        <h4 className="mb-0 fw-bold">🧬 SLC2A1 / GLUT1-DS — De Vivo Disease</h4>
        <small className="text-muted">
          GLUT1 Deficiency Syndrome · 1p34.2 · Facilitative BBB Glucose Transport · 41-patient cohort · Dashboard #188
        </small>
      </div>

      <div className="mb-3">
        {TABS.map((t, i) => (
          <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizureTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={{ ...defs, monitoring: breakdown?.monitoring, lifecycle: breakdown?.lifecycle }} />}
    </div>
  );
}
