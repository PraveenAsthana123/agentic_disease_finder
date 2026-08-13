'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a4a2e';   // dark forest green — GABAergic / inhibitory
const ACCENT2 = '#7b2d00';   // deep rust — drop attacks / danger
const ACCENT3 = '#1a3a5c';   // deep navy — clinical precision
const ACCENT4 = '#4a235a';   // deep purple — genetics / DEE

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
    syndrome, gene, protein_function, lof_consequence, inheritance, most_common_gene_mea,
    cohort, etiology_classes, seizure_types, triggers, treatments,
    contraindications, monitoring_items, lifecycle_windows, concepts, standards,
    thresholds, references, kd_efficacy_drop_attacks, hallmark_eeg, key_safety, top_alerts,
  } = data;
  return (
    <div>
      {(top_alerts || []).map((a, i) => (
        <Alert key={i} text={a} variant={i === 0 ? 'danger' : i === 1 ? 'danger' : 'warning'} />
      ))}

      <div className="row mb-3">
        <KPI label="Patients" value={cohort} color={ACCENT} />
        <KPI label="Etiology Classes" value={etiology_classes} color={ACCENT2} />
        <KPI label="Seizure Types" value={seizure_types} color={ACCENT3} />
        <KPI label="Triggers" value={triggers} color={ACCENT} />
        <KPI label="Treatments" value={treatments} color={ACCENT3} />
        <KPI label="Monitoring Items" value={monitoring_items} color={ACCENT4} />
      </div>
      <div className="row mb-3">
        <KPI label="Lifecycle Windows" value={lifecycle_windows} color={ACCENT2} />
        <KPI label="Key Concepts" value={concepts} color={ACCENT4} />
        <KPI label="Standards" value={standards} color={ACCENT3} />
        <KPI label="Thresholds" value={thresholds} color={ACCENT} />
        <KPI label="References" value={references} color={ACCENT2} />
        <KPI label="Contraindications" value={contraindications} color={ACCENT4} />
      </div>

      <SectionCard title="Gene & Biology" borderColor={ACCENT}>
        <p className="mb-1"><strong>Syndrome:</strong> {syndrome}</p>
        <p className="mb-1"><strong>Gene:</strong> {gene}</p>
        <p className="mb-1"><strong>Protein Function:</strong> {protein_function}</p>
        <p className="mb-1"><strong>LOF Consequence:</strong> {lof_consequence}</p>
        <p className="mb-1"><strong>Inheritance:</strong> {inheritance}</p>
        <p className="mb-0"><strong>MAE Significance:</strong> {most_common_gene_mea}</p>
      </SectionCard>

      <SectionCard title="Key Clinical Benchmarks" borderColor={ACCENT3}>
        <p className="mb-1">
          <Badge text="KD Level A" color={ACCENT} />
          <strong> Drop Attack Efficacy:</strong> {kd_efficacy_drop_attacks}
        </p>
        <p className="mb-1">
          <Badge text="EEG Hallmark" color={ACCENT3} />
          <strong> Doose Theta:</strong> {hallmark_eeg}
        </p>
        <p className="mb-0">
          <Badge text="Safety" color={ACCENT2} />
          <strong> Key Safety:</strong> {key_safety}
        </p>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Etiology ───────────────────────────────────────────────
function EtiologyTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiology_catalog } = data;
  return (
    <div>
      <SectionCard title="5-Class Etiology Catalog (N=41 patients)" borderColor={ACCENT}>
        {(etiology_catalog || []).map((e, i) => (
          <PctBar key={i} label={e.etiology} pct={e.pct}
            color={i === 0 ? ACCENT : i === 1 ? ACCENT3 : i === 2 ? ACCENT2 : i === 3 ? ACCENT4 : '#6c757d'} />
        ))}
      </SectionCard>

      {(etiology_catalog || []).map((e, i) => (
        <SectionCard key={i} title={`Class ${i + 1}: ${e.etiology} (n=${e.n}, ${e.pct}%)`}
          borderColor={i === 0 ? ACCENT : i === 1 ? ACCENT3 : i === 2 ? ACCENT2 : i === 3 ? ACCENT4 : '#6c757d'}>
          <p className="small mb-2"><strong>Mechanism:</strong> {e.mechanism}</p>
          <p className="small mb-2"><strong>EEG Signature:</strong> {e.eeg_signature}</p>
          <p className="small mb-2"><strong>MRI:</strong> {e.mri}</p>
          <p className="small mb-0"><strong>Clinical Note:</strong> {e.clinical_note}</p>
        </SectionCard>
      ))}
    </div>
  );
}

// ── Tab 3: Seizure Types & Triggers ──────────────────────────────────────────
function SeizureTriggersTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types, triggers } = data;
  return (
    <div>
      <SectionCard title="Seizure Types (4)" borderColor={ACCENT2}>
        {(seizure_types || []).map((s, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: '1px solid #eee' }}>
            <div className="d-flex align-items-center mb-1">
              <strong>{s.type}</strong>
              <span className="badge ms-2" style={{ backgroundColor: ACCENT2 }}>{s.prevalence_pct}%</span>
            </div>
            <PctBar label="Prevalence" pct={s.prevalence_pct} color={ACCENT2} />
            <p className="small mb-1"><strong>EEG Correlate:</strong> {s.eeg_correlate}</p>
            <p className="small mb-1"><strong>Semiology:</strong> {s.semiology}</p>
            <p className="small mb-1"><strong>Frequency:</strong> {s.frequency}</p>
            <p className="small mb-1 text-warning-emphasis"><strong>Clinical Tip:</strong> {s.clinical_tip}</p>
            <p className="small mb-0"><strong>Treatment Priority:</strong> {s.treatment_priority}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Triggers (8)" borderColor={ACCENT3}>
        {(triggers || []).map((t, i) => (
          <div key={i} className="mb-3 pb-2" style={{ borderBottom: '1px solid #eee' }}>
            <div className="d-flex align-items-center mb-1">
              <strong>{t.trigger}</strong>
              <span className="badge ms-2"
                style={{ backgroundColor: t.prevalence_pct >= 80 ? ACCENT2 : ACCENT3 }}>
                {t.prevalence_pct}%
              </span>
            </div>
            <PctBar label="" pct={t.prevalence_pct}
              color={t.prevalence_pct >= 80 ? ACCENT2 : ACCENT3} />
            <p className="small mb-1"><strong>Mechanism:</strong> {t.mechanism}</p>
            <p className="small mb-0"><strong>Management:</strong> {t.management}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Treatments ────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments, contraindications, alerts } = data;
  return (
    <div>
      {(alerts || []).map((a, i) => (
        <Alert key={i} text={a} variant={i <= 1 ? 'danger' : 'warning'} />
      ))}

      <SectionCard title="Treatments (8)" borderColor={ACCENT}>
        {(treatments || []).map((t, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: '1px solid #eee' }}>
            <div className="d-flex align-items-center flex-wrap mb-1">
              <strong className="me-2">{t.drug}</strong>
              <Badge text={t.level}
                color={t.level === 'Level A' ? '#155724' : t.level === 'Level B' ? ACCENT3 : '#6c757d'} />
              <Badge text={t.role} color={ACCENT4} />
            </div>
            <p className="small mb-1"><strong>Dose:</strong> {t.dose}</p>
            <p className="small mb-1"><strong>MOA:</strong> {t.moa}</p>
            <p className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</p>
            <p className="small mb-1"><strong>Safety:</strong> {t.safety}</p>
            <p className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</p>
            {t.contraindication_note && (
              <p className="small mb-0 text-danger"><strong>CI Note:</strong> {t.contraindication_note}</p>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications (4)" borderColor={ACCENT2}>
        {(contraindications || []).map((c, i) => (
          <div key={i} className="mb-3 pb-2" style={{ borderBottom: '1px solid #eee' }}>
            <div className="d-flex align-items-center mb-1">
              <strong className="me-2">{c.item}</strong>
              <Badge text={c.severity}
                color={c.severity === 'ABSOLUTE' ? '#721c24' : '#856404'} />
            </div>
            <p className="small mb-0">{c.detail}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 5: Definitions ───────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { definitions, thresholds, standards, references } = data;
  return (
    <div>
      <SectionCard title="Key Concepts (14)" borderColor={ACCENT4}>
        {(definitions || []).map((d, i) => (
          <div key={i} className="mb-3 pb-2" style={{ borderBottom: '1px solid #eee' }}>
            <strong className="d-block mb-1" style={{ color: ACCENT4 }}>{d.term}</strong>
            <p className="small mb-0">{d.definition}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds (10)" borderColor={ACCENT3}>
        <ul className="mb-0 ps-3">
          {(thresholds || []).map((t, i) => (
            <li key={i} className="small mb-1">{t}</li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Guidelines & Standards (8)" borderColor={ACCENT}>
        <ul className="mb-0 ps-3">
          {(standards || []).map((s, i) => (
            <li key={i} className="small mb-1">{s}</li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Key References (6)" borderColor={ACCENT2}>
        {(references || []).map((r, i) => (
          <div key={i} className="mb-2 pb-1" style={{ borderBottom: '1px solid #eee' }}>
            <p className="small fw-bold mb-0">{r.citation}</p>
            <p className="small text-muted mb-0"><em>{r.title}</em></p>
            <p className="small mb-0">{r.relevance}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────
export default function SLC6A1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/slc6a1/overview`).then(r => r.json()),
      fetch(`${API}/api/slc6a1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/slc6a1/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(() => setError('Failed to load SLC6A1 data. Ensure backend is running on port 8010.'));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-1">
        <span style={{ fontSize: 28, marginRight: 10 }}>🧬</span>
        <div>
          <h4 className="mb-0" style={{ color: ACCENT }}>
            SLC6A1 Epilepsy — Myoclonic-Atonic Epilepsy (MAE / Doose Syndrome / SLC6A1-DEE)
          </h4>
          <p className="text-muted mb-0" style={{ fontSize: 13 }}>
            SLC6A1 · 3p25.3 · GABA Transporter 1 (GAT-1) · Most Common Single-Gene MAE Cause
          </p>
        </div>
      </div>

      {error && <Alert text={error} variant="danger" />}

      <div className="mb-3">
        {TABS.map((t, i) => (
          <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />
        ))}
      </div>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <EtiologyTab data={breakdown} />}
      {tab === 2 && <SeizureTriggersTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
