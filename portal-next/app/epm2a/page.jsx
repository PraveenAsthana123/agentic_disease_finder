'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo-navy — EPM2A laforin phosphatase
const ACCENT2 = '#b71c1c';   // deep red — ABSOLUTE CI
const ACCENT3 = '#e65100';   // deep orange — disease-modifying
const ACCENT4 = '#1b5e20';   // deep green — disease-modifying (metformin AMPK)

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

// ── TAB 1: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <div className="alert mb-3" style={{ backgroundColor: '#e8eaf6', border: `2px solid ${ACCENT}`, borderRadius: 8 }}>
        <strong style={{ color: ACCENT }}>&#x1f9ec; EPM2A (6q24.3) — Laforin Dual-Specificity Glucan Phosphatase · Lafora Disease Type 1</strong>
        <div className="small mt-1">{ov.gene}</div>
        <div className="small mt-1"><strong>Inheritance:</strong> {ov.inheritance}</div>
        <div className="small mt-1"><strong>OMIM:</strong> {ov.omim}</div>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-3">
        <KPI label="Cohort" value={ov.cohort_size} color={ACCENT} />
        <KPI label="Female" value={`${ov.female_pct}%`} color="#1565c0" />
        <KPI label="Avg Onset" value={`${ov.mean_onset_months}m`} color="#6a1b9a" />
        <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Consanguineous" value={`${ov.consanguineous_pct}%`} color="#4e342e" />
        <KPI label="Biopsy+" value={`${ov.biopsy_positive_pct}%`} color={ACCENT} />
        <KPI label="Cognitive Decline" value={`${ov.cognitive_decline_pct}%`} color={ACCENT2} />
        <KPI label="On VPA" value={`${ov.on_vpa_pct}%`} color="#f57f17" />
        <KPI label="On LEV" value={`${ov.on_lev_pct}%`} color="#0277bd" />
        <KPI label="On Metformin" value={`${ov.on_metformin_pct}%`} color={ACCENT4} />
        <KPI label="On KD" value={`${ov.on_kd_pct}%`} color="#33691e" />
        <KPI label="POLG1 Screened" value={`${ov.polg1_screened_pct}%`} color="#00695c" />
      </div>

      {/* Fatal disease note */}
      <div className="alert mb-3" style={{ backgroundColor: '#fce4ec', border: `2px solid ${ACCENT2}`, fontSize: 13 }}>
        <strong style={{ color: ACCENT2 }}>&#x26a0;&#xfe0f; Fatal Progressive Disease — Lafora Disease Type 1 (EPM2A):</strong>{' '}
        {ov.prognosis_note}
      </div>

      {/* Laforin note */}
      <div className="alert mb-3" style={{ backgroundColor: '#e8eaf6', border: `1px solid ${ACCENT}`, fontSize: 13 }}>
        <strong style={{ color: ACCENT }}>&#x1f9ec; LAFORIN (EPM2A) Glucan Phosphatase Biology:</strong>{' '}
        {ov.laforin_note}
      </div>

      {/* Disease-modifying note */}
      <div className="alert mb-3" style={{ backgroundColor: '#e8f5e9', border: `1px solid ${ACCENT4}`, fontSize: 13 }}>
        <strong style={{ color: ACCENT4 }}>&#x1f4a1; Metformin + KD Disease-Modifying Strategy (AMPK→GYS1):</strong>{' '}
        {ov.disease_modifying_note}
      </div>

      {/* Contraindications */}
      <SectionCard title="&#x1f6ab; Key Contraindications" borderColor={ACCENT2}>
        {(ov.key_contraindications || []).map((ci, i) => (
          <Alert key={i} text={ci} variant={ci.includes('ABSOLUTE') ? 'danger' : 'warning'} />
        ))}
      </SectionCard>

      {/* Etiology distribution */}
      <SectionCard title="Etiology Distribution (5 classes)" borderColor={ACCENT}>
        {Object.entries(ov.etiology_distribution || {}).map(([cat, val]) => (
          <PctBar key={cat} label={cat.split('(')[0].trim()} pct={val.pct} color={ACCENT} />
        ))}
      </SectionCard>

      {/* Seizure distribution */}
      <SectionCard title="Seizure Type Frequency" borderColor="#e65100">
        {(ov.seizure_type_distribution || []).map((s, i) => (
          <PctBar key={i} label={s.type} pct={s.pct} color="#e65100" />
        ))}
      </SectionCard>

      {/* Trigger distribution */}
      <SectionCard title="Seizure Triggers" borderColor="#0277bd">
        {(ov.trigger_distribution || []).map((t, i) => (
          <PctBar key={i} label={t.trigger} pct={t.pct} color="#0277bd" />
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 2: Patients & Etiology ────────────────────────────────────────────────
function PatientsTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const { etiology_catalog = [], patient_sample = [] } = bd;
  return (
    <div>
      <SectionCard title="Etiology Catalog (5 classes)" borderColor={ACCENT}>
        {etiology_catalog.map((et, i) => (
          <div key={i} className="mb-4 p-3" style={{ backgroundColor: '#e8eaf6', borderRadius: 8, border: `1px solid ${ACCENT}` }}>
            <div className="fw-bold mb-1" style={{ color: ACCENT }}>{et.category}</div>
            <div className="row small">
              <div className="col-md-6">
                <div><strong>n={et.n} ({et.pct}%)</strong></div>
                <div><strong>Onset:</strong> {et.typical_onset}</div>
                <div><strong>EEG:</strong> {et.eeg_pattern}</div>
              </div>
              <div className="col-md-6">
                <div><strong>Mechanism:</strong> {et.mechanism}</div>
                <div><strong>Prognosis:</strong> {et.prognosis}</div>
              </div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title={`Patient Sample (first ${patient_sample.length} of 40)`} borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead style={{ backgroundColor: ACCENT, color: '#fff' }}>
              <tr>
                <th>ID</th><th>Subtype</th><th>Sex</th><th>Onset(m)</th>
                <th>Stage</th><th>DRE</th><th>Biopsy+</th>
                <th>VPA</th><th>LEV</th><th>Peramp</th><th>Piracetam</th><th>Metformin</th>
              </tr>
            </thead>
            <tbody>
              {patient_sample.map((p) => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.subtype}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_onset_months}</td>
                  <td>{p.disease_stage}</td>
                  <td style={{ color: p.drug_resistant ? ACCENT2 : '#2e7d32' }}>{p.drug_resistant ? 'Y' : 'N'}</td>
                  <td style={{ color: p.biopsy_positive ? ACCENT : '#777' }}>{p.biopsy_positive ? 'Y' : 'N'}</td>
                  <td style={{ color: p.on_vpa ? '#f57f17' : '#777' }}>{p.on_vpa ? 'Y' : 'N'}</td>
                  <td style={{ color: p.on_lev ? '#0277bd' : '#777' }}>{p.on_lev ? 'Y' : 'N'}</td>
                  <td style={{ color: p.on_perampanel ? '#6a1b9a' : '#777' }}>{p.on_perampanel ? 'Y' : 'N'}</td>
                  <td style={{ color: p.on_piracetam ? '#00695c' : '#777' }}>{p.on_piracetam ? 'Y' : 'N'}</td>
                  <td style={{ color: p.on_metformin ? ACCENT4 : '#777' }}>{p.on_metformin ? 'Y' : 'N'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── TAB 3: Seizures & Triggers ────────────────────────────────────────────────
function SeizuresTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const { seizure_types = [], triggers = [] } = bd;
  return (
    <div>
      <SectionCard title="Seizure Types (5)" borderColor="#e65100">
        {seizure_types.map((s, i) => (
          <div key={i} className="mb-4 p-3" style={{ backgroundColor: '#fff8e1', borderRadius: 8, border: '1px solid #f9a825' }}>
            <div className="d-flex justify-content-between mb-2">
              <span className="fw-bold" style={{ color: '#e65100' }}>{s.type}</span>
              <span className="badge" style={{ backgroundColor: '#e65100', color: '#fff' }}>{s.frequency_pct}%</span>
            </div>
            <div className="small">
              <div><strong>EEG:</strong> {s.eeg}</div>
              <div className="mt-1"><strong>Semiology:</strong> {s.semiology}</div>
              <div className="mt-1 p-2" style={{ backgroundColor: '#e8eaf6', borderRadius: 4, borderLeft: `3px solid ${ACCENT}` }}>
                <strong>&#x1f4a1; Clinical Tip:</strong> {s.clinical_tip}
              </div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (8)" borderColor="#0277bd">
        {triggers.map((t, i) => (
          <div key={i} className="mb-3 p-3" style={{ backgroundColor: '#e3f2fd', borderRadius: 8, border: '1px solid #1565c0' }}>
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-bold" style={{ color: '#0277bd' }}>{t.trigger}</span>
              <span className="badge bg-primary">{t.pct}%</span>
            </div>
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 4: Treatments ────────────────────────────────────────────────────────
function TreatmentsTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const { treatments = [], contraindications = [], monitoring = [], lifecycle = [] } = bd;
  return (
    <div>
      <SectionCard title="Treatments (8)" borderColor={ACCENT}>
        {treatments.map((tx, i) => (
          <div key={i} className="mb-4 p-3" style={{ backgroundColor: '#e8eaf6', borderRadius: 8, border: `1px solid ${ACCENT}` }}>
            <div className="d-flex justify-content-between mb-2">
              <span className="fw-bold" style={{ color: ACCENT }}>{tx.drug}</span>
              <span className="badge" style={{ backgroundColor: ACCENT, color: '#fff', fontSize: 10 }}>{tx.level}</span>
            </div>
            <div className="small">
              <div><strong>Dose:</strong> {tx.dose}</div>
              <div className="mt-1"><strong>MOA:</strong> {tx.moa}</div>
              <div className="mt-1"><strong>Efficacy:</strong> {tx.efficacy}</div>
              <div className="mt-1"><strong>Monitoring:</strong> {tx.monitoring}</div>
              {tx.epm2a_note && (
                <div className="mt-2 p-2" style={{ backgroundColor: '#e8f5e9', borderRadius: 4, borderLeft: `3px solid ${ACCENT4}`, color: ACCENT4 }}>
                  <strong>&#x1f9ec; EPM2A-Specific:</strong> {tx.epm2a_note}
                </div>
              )}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications (6)" borderColor={ACCENT2}>
        {contraindications.map((ci, i) => (
          <div key={i} className="mb-3 p-3" style={{ backgroundColor: '#ffebee', borderRadius: 8, border: `2px solid ${ACCENT2}` }}>
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-bold" style={{ color: ACCENT2 }}>{ci.drug}</span>
              <span className="badge" style={{
                backgroundColor: (ci.severity || '').includes('ABSOLUTE') ? ACCENT2 : '#f57f17',
                color: '#fff'
              }}>
                {(ci.severity || '').split(' ')[0]}
              </span>
            </div>
            <div className="small">
              <div><strong>Severity:</strong> {ci.severity}</div>
              <div className="mt-1"><strong>Mechanism:</strong> {ci.mechanism}</div>
              {ci.alternative && (
                <div className="mt-2 p-2" style={{ backgroundColor: '#e8f5e9', borderRadius: 4, borderLeft: `3px solid ${ACCENT4}` }}>
                  <strong>&#x2714; Alternative:</strong> {ci.alternative}
                </div>
              )}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring (14 items)" borderColor="#00695c">
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead style={{ backgroundColor: '#00695c', color: '#fff' }}>
              <tr><th style={{ width: '30%' }}>Item</th><th style={{ width: '25%' }}>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {monitoring.map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold">{m.item}</td>
                  <td className="text-muted">{m.frequency}</td>
                  <td>{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Lifecycle Stages (6)" borderColor="#01579b">
        {lifecycle.map((lc, i) => (
          <div key={i} className="mb-3 p-3" style={{ backgroundColor: '#e1f5fe', borderRadius: 8, border: '1px solid #0277bd' }}>
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-bold" style={{ color: '#01579b' }}>{lc.stage}</span>
              <span className="text-muted small">{lc.age}</span>
            </div>
            <div className="small">{lc.focus}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 5: Definitions ───────────────────────────────────────────────────────
function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  const { concepts = [], thresholds = [], standards = [], references = [] } = defs;
  return (
    <div>
      <SectionCard title="Key Concepts (15)" borderColor={ACCENT}>
        {concepts.map((c, i) => (
          <div key={i} className="mb-3 p-3" style={{ backgroundColor: '#e8eaf6', borderRadius: 8, border: `1px solid ${ACCENT}` }}>
            <div className="fw-bold mb-1" style={{ color: ACCENT }}>{c.concept}</div>
            <div className="small">{c.explanation}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds (12)" borderColor="#f57f17">
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead style={{ backgroundColor: '#f57f17', color: '#fff' }}>
              <tr><th style={{ width: '40%' }}>Threshold</th><th>Value / Action</th></tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.threshold}</td>
                  <td>{t.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Evidence Standards (12)" borderColor="#1a237e">
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead style={{ backgroundColor: '#1a237e', color: '#fff' }}>
              <tr><th style={{ width: '30%' }}>Standard</th><th>Description</th></tr>
            </thead>
            <tbody>
              {standards.map((s, i) => (
                <tr key={i}>
                  <td className="fw-bold">{s.standard}</td>
                  <td>{s.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="References (6)" borderColor="#01579b">
        {references.map((r, i) => (
          <div key={i} className="mb-2 p-2" style={{ backgroundColor: '#e3f2fd', borderRadius: 6 }}>
            <div className="fw-bold small" style={{ color: '#01579b' }}>{r.ref}</div>
            <div className="small text-muted">{r.full}</div>
            <div className="small mt-1"><strong>Key finding:</strong> {r.key_finding}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── MAIN PAGE ────────────────────────────────────────────────────────────────
export default function EPM2APage() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    setError('');
    Promise.all([
      fetch(`${API}/api/epm2a/overview`).then(r => r.json()),
      fetch(`${API}/api/epm2a/breakdown`).then(r => r.json()),
      fetch(`${API}/api/epm2a/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="p-3 mb-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #283593 100%)`, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">&#x1f9ec; EPM2A Epilepsy Dashboard</h4>
        <div className="small opacity-90">
          Lafora Disease Type 1 · LAFORIN Dual-Specificity Glucan Phosphatase · CBM Domain (W32G Basque Founder) · DSP Domain (C266 Catalytic) ·
          Progressive Myoclonic Epilepsy · Polyglucosan Lafora Bodies · Autosomal Recessive · Fatal Disease · 6q24.3 · 40-Patient Cohort
        </div>
        <div className="small opacity-75 mt-1">
          Metformin AMPK→GYS1 disease-modifying (Level C) + KD dual-mechanism · CBZ/OXC/PHT ABSOLUTE CI (paradoxical myoclonic worsening) ·
          Perampanel Level B (AMPA antagonist) · Piracetam Level B (action myoclonus specific) ·
          Skin biopsy PAS gold standard · TGB ABSOLUTE CI · VGB AVOID (occipital visual) · POLG1 mandatory pre-VPA ·
          W32G homozygous = fastest disease progression — aggressive early disease-modifying therapy
        </div>
      </div>

      {loading && <div className="alert alert-info">Loading EPM2A data…</div>}
      {error && <div className="alert alert-danger">Error: {error}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${activeTab === i ? 'active fw-bold' : ''}`}
              style={activeTab === i ? { color: ACCENT, borderBottom: `3px solid ${ACCENT}` } : {}}
              onClick={() => setActiveTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {activeTab === 0 && <OverviewTab ov={overview} />}
      {activeTab === 1 && <PatientsTab bd={breakdown} />}
      {activeTab === 2 && <SeizuresTab bd={breakdown} />}
      {activeTab === 3 && <TreatmentsTab bd={breakdown} />}
      {activeTab === 4 && <DefinitionsTab defs={definitions} />}
    </div>
  );
}
