'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a2e5a';   // deep navy — voltage-gated K+ channel / precision
const ACCENT2 = '#7b1c1c';   // deep crimson — contraindications / danger
const ACCENT3 = '#1a4a2e';   // forest green — GOF / precision therapy
const ACCENT4 = '#4a1a5c';   // deep purple — genetics / DEE

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
    syndrome, gene, protein_function, dual_mechanism, precision_therapy, inheritance,
    cohort, gof_patients, lof_patients, etiology_classes, seizure_types, triggers,
    treatments, contraindications, monitoring_items, lifecycle_windows,
    concepts, standards, thresholds, references,
    gof_hallmark_mri, lof_hallmark_eeg, key_safety, top_alerts,
  } = data;
  return (
    <div>
      <div className="alert alert-primary mb-3" style={{ fontSize: 14 }}>
        <strong>⚡ {syndrome}</strong><br />
        <span className="text-muted">{gene}</span><br />
        <span style={{ fontSize: 13 }}>{protein_function}</span>
      </div>
      {top_alerts && top_alerts.map((a, i) => <Alert key={i} text={a} variant={i === 0 ? 'danger' : 'warning'} />)}

      <SectionCard title="🔬 Dual-Mechanism Channelopathy" borderColor={ACCENT4}>
        <p style={{ fontSize: 13 }}>{dual_mechanism}</p>
        <div className="row g-2 mt-1">
          <div className="col-md-6">
            <div className="card border-danger">
              <div className="card-header fw-bold text-danger" style={{ fontSize: 12 }}>⚠️ LOF Variants</div>
              <div className="card-body py-2" style={{ fontSize: 12 }}>
                Dominant-negative/haploinsufficiency → reduced Kv1.2 repolarisation → focal/multifocal severe DEE<br/>
                <strong>EEG:</strong> {lof_hallmark_eeg}<br/>
                <strong>MRI:</strong> Normal cerebellum (LOF signal)<br/>
                <Badge text="4-AP ABSOLUTE CI" color="#dc3545" />
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card border-success">
              <div className="card-header fw-bold text-success" style={{ fontSize: 12 }}>✅ GOF Variants — Precision Therapy Eligible</div>
              <div className="card-body py-2" style={{ fontSize: 12 }}>
                K+ leak in interneurons → disinhibition → DEE + cerebellar ataxia<br/>
                <strong>MRI:</strong> {gof_hallmark_mri}<br/>
                <strong>Precision Rx:</strong> {precision_therapy}<br/>
                <Badge text="4-AP after GOF confirmed" color="#198754" />
              </div>
            </div>
          </div>
        </div>
      </SectionCard>

      <div className="row g-2 mb-4">
        <KPI label="Cohort (N)" value={cohort} color={ACCENT} />
        <KPI label="GOF Patients" value={gof_patients} color={ACCENT3} />
        <KPI label="LOF Patients" value={lof_patients} color={ACCENT2} />
        <KPI label="Etiology Classes" value={etiology_classes} color={ACCENT4} />
        <KPI label="Seizure Types" value={seizure_types} color={ACCENT} />
        <KPI label="Triggers" value={triggers} color={ACCENT2} />
        <KPI label="Treatments" value={treatments} color={ACCENT3} />
        <KPI label="Monitoring Items" value={monitoring_items} color={ACCENT4} />
        <KPI label="Lifecycle Windows" value={lifecycle_windows} color={ACCENT} />
        <KPI label="Key Concepts" value={concepts} color={ACCENT2} />
        <KPI label="Standards" value={standards} color={ACCENT3} />
        <KPI label="Thresholds" value={thresholds} color={ACCENT4} />
      </div>

      <SectionCard title="🧬 Gene & Inheritance" borderColor={ACCENT}>
        <div className="row">
          <div className="col-md-6">
            <ul className="list-unstyled mb-0" style={{ fontSize: 13 }}>
              <li><strong>Gene:</strong> {gene}</li>
              <li><strong>Protein:</strong> {protein_function}</li>
              <li><strong>Inheritance:</strong> {inheritance}</li>
            </ul>
          </div>
          <div className="col-md-6">
            <ul className="list-unstyled mb-0" style={{ fontSize: 13 }}>
              <li><strong>Key Safety:</strong> {key_safety}</li>
              <li><strong>References:</strong> {references} key papers</li>
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
  const { etiology_catalog } = data;
  const colors = [ACCENT, ACCENT3, ACCENT4, ACCENT2, '#5a6a2e'];
  return (
    <div>
      <SectionCard title="🧬 Etiology Catalog — KCNA2-DEE (41 Patients, 5 Classes)" borderColor={ACCENT4}>
        <div className="row g-2 mb-3">
          {etiology_catalog.map((e, i) => (
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
        {etiology_catalog.map((e, i) => (
          <PctBar key={i} label={`${e.category} (${e.n} patients)`} pct={e.pct} color={colors[i % colors.length]} />
        ))}
      </SectionCard>

      <SectionCard title="🔬 Mechanistic Detail by Class" borderColor={ACCENT}>
        {etiology_catalog.map((e, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: i < etiology_catalog.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="fw-bold mb-1" style={{ color: colors[i % colors.length], fontSize: 13 }}>
              Class {i + 1}: {e.etiology} — {e.pct}% (N={e.n})
            </div>
            <div className="mb-1"><strong style={{ fontSize: 12 }}>Mechanism:</strong>
              <span style={{ fontSize: 12 }}> {e.mechanism}</span></div>
            <div className="mb-1"><strong style={{ fontSize: 12 }}>EEG Signature:</strong>
              <span style={{ fontSize: 12 }}> {e.eeg_signature}</span></div>
            <div className="mb-1"><strong style={{ fontSize: 12 }}>MRI:</strong>
              <span style={{ fontSize: 12 }}> {e.mri}</span></div>
            <div className="alert alert-light py-1 px-2 mb-0" style={{ fontSize: 12 }}>
              <strong>Clinical Note:</strong> {e.clinical_note}
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Seizure Types & Triggers ──────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types, triggers, alerts } = data;
  const trigColors = [ACCENT2, ACCENT, ACCENT3, ACCENT4, '#8b5a00', '#1a5c4a', '#5c1a4a', '#2e2e7a'];
  return (
    <div>
      {alerts && alerts.map((a, i) => <Alert key={i} text={a} variant={i < 2 ? 'danger' : 'warning'} />)}

      <SectionCard title="⚡ Seizure Types (4)" borderColor={ACCENT}>
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: i < seizure_types.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-bold" style={{ color: ACCENT, fontSize: 13 }}>{s.type}</span>
              <Badge text={`${s.frequency_pct}%`} color={ACCENT} />
            </div>
            <div className="text-muted small mb-1">Age window: {s.age_window}</div>
            <PctBar label="Prevalence in cohort" pct={s.frequency_pct} color={ACCENT} />
            <div className="mb-1"><strong style={{ fontSize: 12 }}>EEG Correlate:</strong>
              <span style={{ fontSize: 12 }}> {s.eeg_correlate}</span></div>
            <div className="alert alert-light py-1 px-2 mb-0" style={{ fontSize: 12 }}>
              <strong>Clinical Tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🌡️ Seizure Triggers (8)" borderColor={ACCENT2}>
        <div className="mb-3">
          {triggers.map((t, i) => (
            <PctBar key={i} label={t.trigger} pct={t.prevalence_pct} color={trigColors[i % trigColors.length]} />
          ))}
        </div>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3 pb-3" style={{ borderBottom: i < triggers.length - 1 ? '1px solid #dee2e6' : 'none' }}>
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
  const { treatments, contraindications, monitoring } = data;
  const txColors = [ACCENT, ACCENT3, ACCENT4, ACCENT2, '#198754', '#6f42c1', '#0dcaf0', '#fd7e14'];
  return (
    <div>
      <Alert text="⚠️ 4-AP ABSOLUTE CONTRAINDICATION in KCNA2-LOF — GOF functional assay must be documented before prescribing. 4-AP in LOF risks acute seizure escalation." variant="danger" />
      <Alert text="⚠️ POLG1 exclusion MANDATORY before VPA — biallelic POLG1 + VPA = Alpers-Huttenlocher syndrome (fatal hepatic failure)." variant="danger" />

      <SectionCard title="💊 Treatments (8)" borderColor={ACCENT3}>
        {treatments.map((t, i) => (
          <div key={i} className="mb-4 pb-3" style={{ borderBottom: i < treatments.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="d-flex flex-wrap align-items-center gap-2 mb-1">
              <span className="fw-bold" style={{ color: txColors[i % txColors.length], fontSize: 14 }}>{t.drug}</span>
              <Badge text={t.evidence_level.split('—')[0].trim()} color={txColors[i % txColors.length]} />
            </div>
            <div className="text-muted small mb-2">{t.evidence_level}</div>
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
            <div className="alert alert-light py-1 px-2 mt-2 mb-0" style={{ fontSize: 12 }}>
              <strong>Monitoring:</strong> {t.monitoring}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications (4)" borderColor={ACCENT2}>
        {contraindications.map((c, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex align-items-center gap-2 mb-1">
              <Badge text={c.risk_level} color={c.risk_level.includes('ABSOLUTE') ? '#dc3545' : '#fd7e14'} />
              <span className="fw-bold" style={{ fontSize: 13 }}>{c.drug_or_action}</span>
            </div>
            <div style={{ fontSize: 12 }}>{c.reason}</div>
            <div className="text-success small mt-1"><strong>Alternative:</strong> {c.alternative}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔬 Monitoring (8)" borderColor={ACCENT4}>
        {monitoring.map((m, i) => (
          <div key={i} className="mb-3 pb-2" style={{ borderBottom: i < monitoring.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="fw-bold" style={{ fontSize: 13, color: ACCENT4 }}>{m.item}</div>
            <div className="text-muted small mb-1">Frequency: {m.frequency}</div>
            <div style={{ fontSize: 12 }}>{m.rationale}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 5: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { definitions, thresholds, standards, references } = data;
  return (
    <div>
      <SectionCard title="📖 Key Concepts (14)" borderColor={ACCENT4}>
        {definitions.map((d, i) => (
          <div key={i} className="mb-3 pb-2" style={{ borderBottom: i < definitions.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="fw-bold mb-1" style={{ color: ACCENT4, fontSize: 13 }}>{d.term}</div>
            <div style={{ fontSize: 12 }}>{d.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Thresholds (10)" borderColor={ACCENT3}>
        {thresholds.map((t, i) => (
          <div key={i} className="mb-2 pb-2" style={{ borderBottom: i < thresholds.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="d-flex flex-wrap gap-2 align-items-center mb-1">
              <span className="fw-bold" style={{ fontSize: 12 }}>{t.threshold}</span>
              <Badge text={t.value} color={ACCENT3} />
            </div>
            <div className="text-muted" style={{ fontSize: 11 }}>{t.rationale}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📋 Standards (8)" borderColor={ACCENT}>
        {standards.map((s, i) => (
          <div key={i} className="mb-2">
            <Badge text={s.standard} color={ACCENT} />
            <span style={{ fontSize: 12 }}> {s.full} — {s.relevance}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📚 References (6)" borderColor={ACCENT2}>
        {references.map((r, i) => (
          <div key={i} className="mb-2 pb-2" style={{ borderBottom: i < references.length - 1 ? '1px solid #dee2e6' : 'none' }}>
            <div className="fw-bold" style={{ fontSize: 12, color: ACCENT2 }}>{r.citation}</div>
            <div style={{ fontSize: 12 }}><em>{r.title}</em></div>
            <div className="text-muted" style={{ fontSize: 11 }}>{r.relevance}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function KCNA2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/kcna2/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab >= 1 && !breakdown) {
      fetch(`${API}/api/kcna2/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
    if (tab === 4 && !definitions) {
      fetch(`${API}/api/kcna2/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-3 mb-3">
        <span style={{ fontSize: 32 }}>⚡</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            KCNA2 Epilepsy — Kv1.2 Channelopathy (KCNA2-DEE)
          </h4>
          <div className="text-muted small">
            KCNA2 · 1p13.3 · Kv1.2 Voltage-Gated K⁺ Channel · 41-Patient Cohort · Dual GOF/LOF Phenotype · 4-AP Precision Therapy (GOF)
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

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

      <div className="text-muted small mt-4 border-top pt-2">
        KCNA2-DEE Dashboard · ILAE 2022 · ClinGen DEFINITIVE · NICE NG217 · MHRA PREVENT 2024 ·
        References: Syrbe 2015 Nat Genet · Masnada 2017 BRAIN · Semmler 2020 EJPN ·
        Precision therapy: 4-AP (GOF-confirmed only) · POLG1 exclusion before VPA
      </div>
    </div>
  );
}
