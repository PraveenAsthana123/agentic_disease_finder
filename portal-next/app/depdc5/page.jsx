'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a3a5c';   // deep navy — DEPDC5 / GATOR1
const ACCENT2 = '#8b0000';   // dark red — SUDEP danger / contraindications
const ACCENT3 = '#155724';   // dark green — surgery success / mTOR therapy
const ACCENT4 = '#4a235a';   // deep purple — mTOR pathway / precision medicine

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
      className={`btn btn-sm me-1 mb-1 ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
      onClick={onClick}
    >{label}</button>
  );
}

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading overview…</div>;
  const {
    syndrome, gene, complex, pathway, lof_consequence, inheritance, prevalence_focal_epilepsy,
    cohort, etiology_classes, seizure_types: nSeizure, triggers: nTrigger,
    treatments: nTx, sudep_risk, precision_medicine, key_surgery_option, top_alerts = []
  } = data;

  const kpis = [
    { label: 'Cohort (N)', value: cohort, color: ACCENT },
    { label: 'Etiology Classes', value: etiology_classes, color: ACCENT4 },
    { label: 'Seizure Types', value: nSeizure, color: ACCENT2 },
    { label: 'Triggers', value: nTrigger, color: '#c0392b' },
    { label: 'Treatments', value: nTx, color: ACCENT3 },
    { label: 'SUDEP Risk', value: '3×', color: ACCENT2 },
  ];

  return (
    <div>
      {/* Gene badge */}
      <div className="alert alert-primary py-2 mb-3" style={{ fontSize: 13 }}>
        <strong>🧬 {gene}</strong><br />
        <span>Complex: {complex}</span><br />
        <span className="text-muted">Pathway: {pathway}</span><br />
        <span className="text-muted">LOF consequence: {lof_consequence}</span><br />
        <span className="text-muted">Inheritance: {inheritance} · Prevalence: {prevalence_focal_epilepsy}</span>
      </div>

      {/* Clinical alerts */}
      <SectionCard title="⚠️ Critical Clinical Alerts — DEPDC5" borderColor={ACCENT2}>
        {top_alerts.map((a, i) => (
          <Alert key={i} text={a.text} variant={a.variant || (i < 2 ? 'danger' : i < 4 ? 'warning' : 'info')} />
        ))}
      </SectionCard>

      {/* KPIs */}
      <SectionCard title="📊 Cohort KPIs">
        <div className="row">
          {kpis.map((k, i) => (
            <KPI key={i} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      {/* SUDEP + precision medicine row */}
      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
            <div className="card-header fw-bold small" style={{ color: ACCENT2 }}>⚡ SUDEP Risk</div>
            <div className="card-body">
              <div className="fw-bold text-danger mb-1">{sudep_risk}</div>
              <div style={{ fontSize: 13 }}>
                Nocturnal FBTCS + prone sleeping = highest SUDEP dyad.
                Mandatory counselling at diagnosis. Prescribe: bed alarm, supine sleep, nocturnal monitor.
              </div>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>🔬 Precision Medicine</div>
            <div className="card-body">
              <div className="fw-bold mb-1" style={{ color: ACCENT4 }}>{precision_medicine}</div>
              <div style={{ fontSize: 13 }}>
                mTOR pathway — same target as TSC Everolimus therapy.
                {key_surgery_option}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Pathway diagram */}
      <SectionCard title="🔗 GATOR1 → mTOR Pathway" borderColor={ACCENT4}>
        <div className="d-flex flex-wrap align-items-center gap-2" style={{ fontSize: 13 }}>
          {[
            { label: 'Low amino acids', color: '#aed6f1' },
            { label: '→ GATOR1 activated', color: '#a9cce3' },
            { label: '→ RagA/B GTPase inactivated', color: '#85c1e9' },
            { label: '→ mTORC1 OFF', color: '#5dade2' },
            { label: '✓ Normal neurodevelopment', color: '#27ae60', textColor: '#fff' },
          ].map((step, i) => (
            <span key={i} className="badge" style={{ backgroundColor: step.color, color: step.textColor || '#1a3a5c', padding: '6px 10px' }}>
              {step.label}
            </span>
          ))}
        </div>
        <div className="mt-2 d-flex flex-wrap align-items-center gap-2" style={{ fontSize: 13 }}>
          {[
            { label: 'DEPDC5-LOF', color: '#e74c3c', textColor: '#fff' },
            { label: '→ GATOR1 non-functional', color: '#cb4335', textColor: '#fff' },
            { label: '→ Rag GTPase always ON', color: '#b03a2e', textColor: '#fff' },
            { label: '→ mTORC1 constitutively active', color: '#922b21', textColor: '#fff' },
            { label: '→ FCD + Focal Epilepsy', color: '#641e16', textColor: '#fff' },
          ].map((step, i) => (
            <span key={i} className="badge" style={{ backgroundColor: step.color, color: step.textColor, padding: '6px 10px' }}>
              {step.label}
            </span>
          ))}
        </div>
        <div className="mt-2 text-muted small">
          Therapeutic target: Everolimus inhibits mTORC1 (same mechanism as TSC therapy)
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Patients & Etiology ──────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading breakdown…</div>;
  const { etiology_catalog = [] } = data;

  return (
    <div>
      <SectionCard title="🧬 Etiology Catalog — 5 Classes (N=41)" borderColor={ACCENT}>
        {etiology_catalog.map((e, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{
            background: '#f8f9fa',
            borderLeft: `4px solid ${[ACCENT, ACCENT2, ACCENT3, ACCENT4, '#7f8c8d'][i]}`
          }}>
            <div className="fw-bold mb-1">
              {e.etiology}
              <span className="badge bg-primary ms-2">{e.pct}% (n={e.n})</span>
            </div>
            <div className="mb-2"><strong>Mechanism:</strong> <span style={{ fontSize: 13 }}>{e.mechanism}</span></div>
            <div className="mb-1"><strong>EEG Signature:</strong> <span style={{ fontSize: 13 }}>{e.eeg_signature}</span></div>
            <div className="mb-1"><strong>MRI:</strong> <span style={{ fontSize: 13 }}>{e.mri}</span></div>
            <div className="mt-2 p-2 rounded" style={{ background: '#fff3cd', fontSize: 13 }}>
              <strong>📋 Clinical Note:</strong> {e.clinical_note}
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Etiology distribution bar */}
      <SectionCard title="📊 Etiology Distribution" borderColor={ACCENT4}>
        {etiology_catalog.map((e, i) => (
          <PctBar key={i}
            label={`${e.category.replace(/-/g, ' ')} (n=${e.n})`}
            pct={e.pct}
            color={[ACCENT, ACCENT2, ACCENT3, ACCENT4, '#7f8c8d'][i]} />
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Seizure Types & Triggers ─────────────────────────────────────────────
function SeizureTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading breakdown…</div>;
  const { seizure_types = [], triggers = [], alerts = [] } = data;

  return (
    <div>
      {/* Safety alerts */}
      <SectionCard title="⚠️ Safety Alerts" borderColor={ACCENT2}>
        {alerts.slice(0, 4).map((a, i) => (
          <Alert key={i} text={a.text} variant={a.variant} />
        ))}
      </SectionCard>

      <SectionCard title="⚡ Seizure Types (N=41 DEPDC5 cohort)" borderColor={ACCENT2}>
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{
            background: '#fdf2f8',
            borderLeft: `4px solid ${[ACCENT2, '#8e44ad', '#2980b9', '#27ae60'][i % 4]}`
          }}>
            <div className="d-flex justify-content-between align-items-center mb-2">
              <div className="fw-bold">{s.type}</div>
              <span className="badge" style={{ backgroundColor: ACCENT2, fontSize: 13 }}>{s.pct}%</span>
            </div>
            <PctBar label="" pct={s.pct} color={[ACCENT2, '#8e44ad', '#2980b9', '#27ae60'][i % 4]} />
            <div className="mt-2">
              <strong>EEG Correlate:</strong>
              <div style={{ fontSize: 13, marginTop: 4 }}>{s.eeg_correlate}</div>
            </div>
            <div className="mt-2 p-2 rounded" style={{ background: '#fef9e7', fontSize: 13 }}>
              <strong>🩺 Clinical Tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚨 Seizure Triggers & Precipitants" borderColor={ACCENT4}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{
            background: '#f4f6f9',
            borderLeft: `4px solid ${['#c0392b', '#922b21', '#7d3c98', '#2471a3', '#1e8449', '#d35400', '#7f8c8d', '#6c3483'][i % 8]}`
          }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <div className="fw-bold small">{t.trigger}</div>
              <span className="badge bg-danger">{t.pct}%</span>
            </div>
            <PctBar label="" pct={t.pct} color={['#c0392b', '#922b21', '#7d3c98', '#2471a3', '#1e8449', '#d35400', '#7f8c8d', '#6c3483'][i % 8]} />
            <div style={{ fontSize: 13, marginTop: 4 }} className="text-muted">{t.mechanism}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Treatments ───────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading breakdown…</div>;
  const { treatments = [], contraindications = [], monitoring = [], lifecycle = [] } = data;

  const txColors = [ACCENT, '#1a6e9a', '#1e8449', '#d68910', ACCENT4, '#784212', '#2e4057', ACCENT3];

  return (
    <div>
      <SectionCard title="💊 Treatments — 8 AEDs / Interventions (DEPDC5-FE)" borderColor={ACCENT3}>
        {treatments.map((t, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{
            background: '#f0fdf4',
            borderLeft: `4px solid ${txColors[i % txColors.length]}`
          }}>
            <div className="fw-bold mb-1" style={{ color: txColors[i % txColors.length] }}>{t.drug}</div>
            <div className="mb-1">
              <span className="badge" style={{ backgroundColor: txColors[i % txColors.length] }}>{t.level}</span>
            </div>
            <div className="mb-1"><strong>Dose:</strong> <span style={{ fontSize: 13 }}>{t.dose}</span></div>
            <div className="mb-1"><strong>MOA:</strong> <span style={{ fontSize: 13 }}>{t.moa}</span></div>
            <div className="mb-1"><strong>Efficacy:</strong> <span style={{ fontSize: 13 }}>{t.efficacy}</span></div>
            <div className="mb-1 p-2 rounded" style={{ background: '#fff3cd', fontSize: 13 }}>
              <strong>⚠️ Safety:</strong> {t.safety}
            </div>
            <div style={{ fontSize: 13 }}><strong>Monitoring:</strong> {t.monitoring}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications (4 DEPDC5-specific)" borderColor={ACCENT2}>
        {contraindications.map((c, i) => (
          <div key={i} className="alert alert-danger mb-2" style={{ fontSize: 13 }}>
            <strong>{c.item}</strong><br />
            <span className="text-muted">{c.reason}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📋 Monitoring Protocol (8 Items)" borderColor={ACCENT}>
        <div className="row">
          {monitoring.map((m, i) => (
            <div key={i} className="col-md-6 mb-3">
              <div className="card h-100 shadow-sm">
                <div className="card-header small fw-bold" style={{ backgroundColor: '#eef2f7', color: ACCENT }}>
                  {m.item}
                </div>
                <div className="card-body py-2">
                  <div className="text-muted small mb-1">📅 {m.schedule}</div>
                  <div style={{ fontSize: 12 }}>{m.detail}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="⏳ Clinical Lifecycle — 6 Windows" borderColor={ACCENT4}>
        {lifecycle.map((w, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{
            background: '#f8f0fb',
            borderLeft: `4px solid ${ACCENT4}`
          }}>
            <div className="fw-bold mb-1">{w.window}
              <span className="badge bg-secondary ms-2" style={{ fontSize: 11 }}>{w.age}</span>
            </div>
            <div style={{ fontSize: 13 }}>{w.focus}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted p-3">Loading definitions…</div>;
  const { definitions = [], contraindications = [], thresholds = [], standards = [], references = [] } = data;

  return (
    <div>
      <SectionCard title="📖 Key Concepts — 14 Definitions" borderColor={ACCENT}>
        <div className="accordion" id="defAccordion">
          {definitions.map((d, i) => (
            <div key={i} className="accordion-item mb-2 border rounded">
              <h2 className="accordion-header">
                <button className="accordion-button collapsed fw-bold small py-2"
                  style={{ backgroundColor: '#eef2f7', color: ACCENT }}
                  type="button" data-bs-toggle="collapse"
                  data-bs-target={`#def${i}`}>
                  {d.term}
                </button>
              </h2>
              <div id={`def${i}`} className="accordion-collapse collapse" data-bs-parent="#defAccordion">
                <div className="accordion-body" style={{ fontSize: 13 }}>{d.definition}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🚫 Contraindications" borderColor={ACCENT2}>
        {contraindications.map((c, i) => (
          <div key={i} className="alert alert-danger mb-2" style={{ fontSize: 13 }}>
            <strong>{c.item}</strong><br />{c.reason}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Clinical Thresholds (10)" borderColor={ACCENT4}>
        <ul className="list-group">
          {thresholds.map((t, i) => (
            <li key={i} className="list-group-item list-group-item-action" style={{ fontSize: 13 }}>
              <span className="badge bg-secondary me-2">{i + 1}</span>{t}
            </li>
          ))}
        </ul>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="📚 Clinical Standards (8)" borderColor={ACCENT3}>
            <ul className="list-unstyled mb-0">
              {standards.map((s, i) => (
                <li key={i} className="mb-1" style={{ fontSize: 13 }}>
                  <span className="badge bg-success me-1">✓</span>{s}
                </li>
              ))}
            </ul>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="🔗 Key References (6)" borderColor={ACCENT}>
            <ul className="list-unstyled mb-0">
              {references.map((r, i) => (
                <li key={i} className="mb-1" style={{ fontSize: 13 }}>
                  <span className="badge me-1" style={{ backgroundColor: ACCENT }}>📄</span>{r}
                </li>
              ))}
            </ul>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

// ── Main Page Component ───────────────────────────────────────────────────────
export default function DEPDC5Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/depdc5/overview`)
      .then(r => r.json()).then(setOverview).catch(() => setErr('Overview fetch failed'));
    fetch(`${API}/api/depdc5/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => setErr('Breakdown fetch failed'));
    fetch(`${API}/api/depdc5/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => setErr('Definitions fetch failed'));
  }, []);

  return (
    <main className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT4} 100%)` }}>
        <h4 className="mb-0 fw-bold">🧬 DEPDC5 Focal Epilepsy — GATOR1 Complex</h4>
        <div style={{ fontSize: 13, opacity: 0.9 }}>
          FFEVF / ADNFLE-DEPDC5 · 22q12.3 · mTORC1 pathway · 41-patient cohort · Dashboard #182
        </div>
        <div style={{ fontSize: 12, opacity: 0.75 }}>
          Most common genetic cause of familial focal epilepsy · SUDEP risk 3× elevated ·
          Precision therapy: Everolimus (GMTD trial NCT04203940)
        </div>
      </div>

      {err && <div className="alert alert-danger">{err}</div>}

      {/* Tabs */}
      <div className="mb-3">
        {TABS.map((t, i) => (
          <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {/* Tab content */}
      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizureTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </main>
  );
}
