'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a3a5c';   // deep navy — PRRT2 / BFIE
const ACCENT2 = '#8b4513';   // saddle brown — PKD / movement disorder
const ACCENT3 = '#155724';   // dark green — benign prognosis / CBZ response
const ACCENT4 = '#4a235a';   // deep purple — genetic counselling / precision

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
  const { syndrome, gene, protein_function, lof_consequence, inheritance, hotspot_variant,
          phenotypes, cohort, etiology_classes, seizure_types, triggers, treatments,
          contraindications, monitoring_items, lifecycle_windows, concepts, standards,
          thresholds, references, cbz_efficacy_pkd, bfie_prognosis, key_safety, top_alerts } = data;
  return (
    <div>
      {/* Alerts */}
      {(top_alerts || []).map((a, i) => (
        <Alert key={i} text={a} variant={i === 0 ? 'danger' : i === 1 ? 'warning' : 'info'} />
      ))}

      {/* Header KPIs */}
      <div className="row mb-3">
        <KPI label="Patients" value={cohort} color={ACCENT} />
        <KPI label="Etiology Classes" value={etiology_classes} color={ACCENT2} />
        <KPI label="Seizure/Attack Types" value={seizure_types} color={ACCENT3} />
        <KPI label="Triggers" value={triggers} color={ACCENT} />
        <KPI label="Treatments" value={treatments} color={ACCENT3} />
        <KPI label="Monitoring Items" value={monitoring_items} color={ACCENT4} />
      </div>
      <div className="row mb-3">
        <KPI label="Contraindications" value={contraindications} color={ACCENT2} />
        <KPI label="Lifecycle Windows" value={lifecycle_windows} color={ACCENT} />
        <KPI label="Key Concepts" value={concepts} color={ACCENT4} />
        <KPI label="Standards" value={standards} color={ACCENT3} />
        <KPI label="Thresholds" value={thresholds} color={ACCENT2} />
        <KPI label="References" value={references} color={ACCENT} />
      </div>

      {/* Gene / Biology */}
      <SectionCard title="PRRT2 Gene & Mechanism" borderColor={ACCENT}>
        <table className="table table-sm table-bordered mb-0" style={{ fontSize: 13 }}>
          <tbody>
            <tr><th style={{ width: '30%' }}>Syndrome</th><td>{syndrome}</td></tr>
            <tr><th>Gene / Locus</th><td>{gene}</td></tr>
            <tr><th>Protein Function</th><td>{protein_function}</td></tr>
            <tr><th>LOF Consequence</th><td>{lof_consequence}</td></tr>
            <tr><th>Inheritance</th><td>{inheritance}</td></tr>
            <tr><th>Hotspot Variant</th><td><code>{hotspot_variant}</code></td></tr>
          </tbody>
        </table>
      </SectionCard>

      {/* Phenotypic Spectrum */}
      {phenotypes && (
        <SectionCard title="Phenotypic Spectrum" borderColor={ACCENT2}>
          <div className="row">
            {Object.entries(phenotypes).map(([name, desc]) => (
              <div className="col-md-4 mb-3" key={name}>
                <div className="card h-100" style={{ borderTop: `3px solid ${ACCENT2}` }}>
                  <div className="card-body">
                    <div className="fw-bold mb-1" style={{ color: ACCENT2 }}>{name}</div>
                    <div style={{ fontSize: 13 }}>{desc}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Prognosis & Safety */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="BFIE Prognosis" borderColor={ACCENT3}>
            <div style={{ fontSize: 13 }}>{bfie_prognosis}</div>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="PKD — CBZ Efficacy" borderColor={ACCENT3}>
            <div style={{ fontSize: 13 }}>{cbz_efficacy_pkd}</div>
          </SectionCard>
        </div>
      </div>
      <SectionCard title="Key Safety Alerts" borderColor={ACCENT2}>
        <div style={{ fontSize: 13 }}>{key_safety}</div>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Etiology ────────────────────────────────────────────────
function EtiologyTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiology_catalog } = data;
  return (
    <div>
      <SectionCard title={`Etiology Distribution (N=${(etiology_catalog || []).reduce((s, e) => s + e.n, 0)} patients)`} borderColor={ACCENT}>
        {(etiology_catalog || []).map((e, i) => (
          <PctBar key={i} label={e.etiology} pct={e.pct}
            color={[ACCENT, ACCENT2, ACCENT3, ACCENT4, '#6c757d'][i % 5]} />
        ))}
      </SectionCard>
      {(etiology_catalog || []).map((e, i) => (
        <SectionCard key={i} title={`${e.category} — ${e.n} patients (${e.pct}%)`}
          borderColor={[ACCENT, ACCENT2, ACCENT3, ACCENT4, '#6c757d'][i % 5]}>
          <div className="mb-2">
            <span className="fw-bold small">Mechanism: </span>
            <span style={{ fontSize: 13 }}>{e.mechanism}</span>
          </div>
          <div className="mb-2">
            <span className="fw-bold small">EEG Signature: </span>
            <span style={{ fontSize: 13 }}>{e.eeg_signature}</span>
          </div>
          <div className="mb-2">
            <span className="fw-bold small">MRI: </span>
            <span style={{ fontSize: 13 }}>{e.mri}</span>
          </div>
          <div>
            <span className="fw-bold small">Clinical Note: </span>
            <span style={{ fontSize: 13 }}>{e.clinical_note}</span>
          </div>
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
      <SectionCard title="Seizure Types & Attack Morphology" borderColor={ACCENT}>
        {(seizure_types || []).map((s, i) => (
          <div key={i} className="card mb-3" style={{ borderLeft: `4px solid ${[ACCENT, ACCENT2, ACCENT3, ACCENT4][i % 4]}` }}>
            <div className="card-body">
              <div className="d-flex justify-content-between align-items-start mb-2">
                <div className="fw-bold" style={{ color: [ACCENT, ACCENT2, ACCENT3, ACCENT4][i % 4], fontSize: 14 }}>
                  {s.type}
                </div>
                <span className="badge" style={{ backgroundColor: [ACCENT, ACCENT2, ACCENT3, ACCENT4][i % 4] }}>
                  {s.frequency_pct}%
                </span>
              </div>
              <div className="mb-1 small text-muted">{s.age_window}</div>
              <div className="mb-2" style={{ fontSize: 13 }}>
                <span className="fw-bold">EEG: </span>{s.eeg_correlate}
              </div>
              <div style={{ fontSize: 13 }}>
                <span className="fw-bold">Clinical Tip: </span>{s.clinical_tip}
              </div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure & Attack Triggers" borderColor={ACCENT2}>
        {(triggers || []).map((t, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-bold small">{t.trigger}</span>
              <span className="badge" style={{ backgroundColor: ACCENT2 }}>{t.pct}%</span>
            </div>
            <div className="progress mb-1" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: ACCENT2 }} />
            </div>
            <div style={{ fontSize: 12, color: '#555' }}>{t.detail}</div>
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
  return (
    <div>
      <SectionCard title="Treatment Protocols" borderColor={ACCENT3}>
        {(treatments || []).map((t, i) => (
          <div key={i} className="card mb-3" style={{ borderLeft: `4px solid ${ACCENT3}` }}>
            <div className="card-body">
              <div className="d-flex justify-content-between align-items-start mb-1">
                <div className="fw-bold" style={{ color: ACCENT3 }}>{t.drug}</div>
                <Badge text={t.evidence} color={ACCENT3} />
              </div>
              <div className="small text-muted mb-2">{t.brand}</div>
              <div className="row small">
                {t.dose_pediatric && t.dose_pediatric !== 'N/A' && (
                  <div className="col-md-6 mb-1">
                    <span className="fw-bold">Paediatric: </span>{t.dose_pediatric}
                  </div>
                )}
                {t.dose_adult && t.dose_adult !== 'N/A' && (
                  <div className="col-md-6 mb-1">
                    <span className="fw-bold">Adult: </span>{t.dose_adult}
                  </div>
                )}
              </div>
              {t.titration && t.titration !== 'N/A' && (
                <div className="small mb-1"><span className="fw-bold">Titration: </span>{t.titration}</div>
              )}
              <div className="small mb-1"><span className="fw-bold">MOA: </span>{t.moa}</div>
              <div className="small mb-1"><span className="fw-bold">Efficacy: </span>{t.efficacy}</div>
              <div className="small mb-1"><span className="fw-bold">Safety: </span>{t.safety}</div>
              {t.monitoring && (
                <div className="small mb-1"><span className="fw-bold">Monitoring: </span>{t.monitoring}</div>
              )}
              {t.notes && (
                <div className="small text-muted mt-1"><span className="fw-bold">Notes: </span>{t.notes}</div>
              )}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications" borderColor={ACCENT2}>
        {(contraindications || []).map((c, i) => (
          <div key={i} className="card mb-3 border-danger">
            <div className="card-body">
              <div className="fw-bold text-danger mb-1" style={{ fontSize: 13 }}>{c.item}</div>
              <div style={{ fontSize: 13 }}>{c.reason}</div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol" borderColor={ACCENT4}>
        {(monitoring || []).map((m, i) => (
          <div key={i} className="card mb-2" style={{ borderLeft: `3px solid ${ACCENT4}` }}>
            <div className="card-body py-2">
              <div className="fw-bold small" style={{ color: ACCENT4 }}>{m.item}</div>
              <div className="small text-muted mb-1">{m.schedule}</div>
              <div style={{ fontSize: 12 }}>{m.detail}</div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Lifecycle Windows" borderColor={ACCENT}>
        {(data.lifecycle || []).map((l, i) => (
          <div key={i} className="card mb-3" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="card-header py-2" style={{ backgroundColor: '#eef2f7' }}>
              <span className="fw-bold" style={{ color: ACCENT }}>{l.window}</span>
              <span className="ms-2 badge bg-secondary">{l.age}</span>
            </div>
            <div className="card-body">
              <div style={{ fontSize: 13 }} className="mb-2">{l.focus}</div>
              {l.key_actions && (
                <ul className="mb-0" style={{ fontSize: 12 }}>
                  {l.key_actions.map((a, j) => <li key={j}>{a}</li>)}
                </ul>
              )}
            </div>
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
      <SectionCard title="Key Concepts (14)" borderColor={ACCENT4}>
        {(definitions || []).map((d, i) => (
          <div key={i} className="card mb-2" style={{ borderLeft: `3px solid ${ACCENT4}` }}>
            <div className="card-body py-2">
              <div className="fw-bold small" style={{ color: ACCENT4 }}>{d.term}</div>
              <div style={{ fontSize: 12 }}>{d.definition}</div>
            </div>
          </div>
        ))}
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Thresholds" borderColor={ACCENT2}>
            <ul className="mb-0" style={{ fontSize: 13 }}>
              {(thresholds || []).map((t, i) => <li key={i}>{t}</li>)}
            </ul>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Standards & Guidelines" borderColor={ACCENT3}>
            <ul className="mb-0" style={{ fontSize: 13 }}>
              {(standards || []).map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Key References" borderColor={ACCENT}>
        {(references || []).map((r, i) => (
          <div key={i} className="mb-2">
            <div className="fw-bold small">{r.citation}</div>
            <div className="small text-muted">{r.title}</div>
            <div style={{ fontSize: 12 }}>{r.relevance}</div>
            {i < (references || []).length - 1 && <hr className="my-2" />}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function PRRT2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/prrt2/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(() => setError('Failed to load overview'));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 4 && !breakdown) {
      fetch(`${API}/api/prrt2/breakdown`)
        .then(r => r.json()).then(setBreakdown)
        .catch(() => setError('Failed to load breakdown'));
    }
    if (tab === 4 && !definitions) {
      fetch(`${API}/api/prrt2/definitions`)
        .then(r => r.json()).then(setDefinitions)
        .catch(() => setError('Failed to load definitions'));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4">
      <div className="mb-3">
        <h2 style={{ color: ACCENT }}>
          PRRT2 Epilepsy Spectrum
          <small className="ms-2 fs-6 text-muted">BFIE / PKD / ICCA — 16p11.2</small>
        </h2>
        <div className="text-muted small mb-2">
          PRRT2 (Proline-Rich Transmembrane Protein 2) · SNAP25-Nav axis · Self-limited infantile
          epilepsy (BFIE) + movement-triggered kinesiogenic dyskinesia (PKD) · Dashboard #183
        </div>
        <div>
          <Badge text="Autosomal Dominant" color={ACCENT} />
          <Badge text="60-80% Penetrance" color={ACCENT} />
          <Badge text="c.649dupC Hotspot ~80%" color={ACCENT2} />
          <Badge text="BFIE Self-Limited by 24M" color={ACCENT3} />
          <Badge text="PKD: CBZ Near-100% Response" color={ACCENT3} />
          <Badge text="HLA-B*15:02 Mandatory" color={ACCENT2} />
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      <div className="mb-3">
        {TABS.map((label, i) => (
          <TabBtn key={i} label={label} active={tab === i} onClick={() => setTab(i)} />
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
