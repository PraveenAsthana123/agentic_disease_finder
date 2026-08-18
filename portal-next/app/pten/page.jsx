'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1b5e20';   // deep forest green — PTEN tumour suppressor (growth inhibitory)
const ACCENT2 = '#b71c1c';   // deep red — ABSOLUTE CI / HIGH RISK
const ACCENT3 = '#e65100';   // deep orange — cancer surveillance urgency
const ACCENT4 = '#4a148c';   // deep purple — precision mTOR therapy

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#e8f5e9', color: borderColor }}>
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
      <div className="alert mb-3" style={{ backgroundColor: '#e8f5e9', border: `2px solid ${ACCENT}`, borderRadius: 8 }}>
        <strong style={{ color: ACCENT }}>&#x1f3af; PTEN (10q23.31) — Phosphatase and Tensin Homolog</strong>
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
        <KPI label="ASD Features" value={`${ov.asd_features_pct}%`} color="#0277bd" />
        <KPI label="Macrocephaly" value={`${ov.macrocephaly_pct}%`} color={ACCENT} />
        <KPI label="Infantile Spasms" value={`${ov.infantile_spasms_pct}%`} color={ACCENT2} />
        <KPI label="On Everolimus" value={`${ov.on_everolimus_pct}%`} color={ACCENT4} />
        <KPI label="On KD" value={`${ov.on_kd_pct}%`} color="#33691e" />
        <KPI label="Cancer Surv." value={`${ov.cancer_surveillance_pct}%`} color={ACCENT3} />
        <KPI label="POLG1 Screened" value={`${ov.polg1_screened_pct}%`} color="#00695c" />
        <KPI label="LDD MRI" value={`${ov.ldd_pct}%`} color="#4e342e" />
      </div>

      {/* mTOR pathway note */}
      <div className="alert mb-3" style={{ backgroundColor: '#f3e5f5', border: `1px solid ${ACCENT4}`, fontSize: 13 }}>
        <strong style={{ color: ACCENT4 }}>&#x1f9ec; mTOR Pathway Mechanism:</strong>{' '}
        {ov.mtor_pathway_note}
      </div>

      {/* Cancer surveillance note */}
      <div className="alert mb-3" style={{ backgroundColor: '#fff3e0', border: `1px solid ${ACCENT3}`, fontSize: 13 }}>
        <strong style={{ color: ACCENT3 }}>&#x26a0;&#xfe0f; Cancer Surveillance (NCCN PHTS):</strong>{' '}
        {ov.cancer_surveillance_note}
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
          <div key={i} className="mb-4 p-3" style={{ backgroundColor: '#f9fbe7', borderRadius: 8, border: `1px solid ${ACCENT}` }}>
            <div className="fw-bold mb-1" style={{ color: ACCENT }}>{et.category}</div>
            <div className="row small">
              <div className="col-md-6">
                <div><strong>n={et.n} ({et.pct}%)</strong></div>
                <div><strong>Variants:</strong> {et.variants}</div>
                <div><strong>Phenotype:</strong> {et.phenotype}</div>
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
                <th>ID</th><th>Syndrome</th><th>Sex</th><th>Onset(m)</th>
                <th>DRE</th><th>Macrocephaly</th><th>ASD</th><th>IS</th>
                <th>Everolimus</th><th>KD</th><th>VPA</th><th>Cancer Surv.</th>
              </tr>
            </thead>
            <tbody>
              {patient_sample.map((p) => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.syndrome}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_onset_months}</td>
                  <td style={{ color: p.drug_resistant ? ACCENT2 : '#2e7d32' }}>{p.drug_resistant ? 'Y' : 'N'}</td>
                  <td style={{ color: p.macrocephaly ? '#0277bd' : '#777' }}>{p.macrocephaly ? 'Y' : 'N'}</td>
                  <td style={{ color: p.asd_features ? '#4a148c' : '#777' }}>{p.asd_features ? 'Y' : 'N'}</td>
                  <td style={{ color: p.infantile_spasms ? ACCENT2 : '#777' }}>{p.infantile_spasms ? 'Y' : 'N'}</td>
                  <td style={{ color: p.on_everolimus ? ACCENT4 : '#777' }}>{p.on_everolimus ? 'Y' : 'N'}</td>
                  <td style={{ color: p.on_kd ? '#33691e' : '#777' }}>{p.on_kd ? 'Y' : 'N'}</td>
                  <td style={{ color: p.on_vpa ? '#f57f17' : '#777' }}>{p.on_vpa ? 'Y' : 'N'}</td>
                  <td style={{ color: p.cancer_surveillance_enrolled ? ACCENT3 : '#777' }}>{p.cancer_surveillance_enrolled ? 'Y' : 'N'}</td>
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
              <div className="mt-1 p-2" style={{ backgroundColor: '#e8f5e9', borderRadius: 4, borderLeft: `3px solid ${ACCENT}` }}>
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
            <div className="small">
              <div><strong>Mechanism:</strong> {t.mechanism}</div>
              <div className="mt-1"><strong>Management:</strong> {t.management}</div>
            </div>
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
          <div key={i} className="mb-4 p-3" style={{ backgroundColor: '#f9fbe7', borderRadius: 8, border: `1px solid ${ACCENT}` }}>
            <div className="d-flex justify-content-between mb-2">
              <span className="fw-bold" style={{ color: ACCENT }}>{tx.drug}</span>
              <span className="badge" style={{ backgroundColor: ACCENT, color: '#fff', fontSize: 10 }}>{tx.evidence}</span>
            </div>
            <div className="small">
              <div><strong>Dose:</strong> {tx.dose}</div>
              <div className="mt-1"><strong>MOA:</strong> {tx.moa}</div>
              <div className="mt-1"><strong>Efficacy:</strong> {tx.efficacy}</div>
              <div className="mt-1"><strong>Monitoring:</strong> {tx.monitoring}</div>
              {tx.pten_note && (
                <div className="mt-2 p-2" style={{ backgroundColor: '#e8f5e9', borderRadius: 4, borderLeft: `3px solid ${ACCENT4}`, color: ACCENT4 }}>
                  <strong>&#x1f9ec; PTEN-Specific:</strong> {tx.pten_note}
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
              <span className="badge" style={{ backgroundColor: ci.risk_level === 'ABSOLUTE' ? ACCENT2 : '#f57f17', color: '#fff' }}>
                {ci.risk_level}
              </span>
            </div>
            <div className="small">
              <div><strong>Mechanism:</strong> {ci.mechanism}</div>
              <div className="mt-1"><strong>Management:</strong> {ci.management}</div>
              {ci.pten_specific && (
                <div className="mt-2 p-2" style={{ backgroundColor: '#fff3e0', borderRadius: 4, borderLeft: `3px solid ${ACCENT3}` }}>
                  <strong>&#x26a0; PTEN-Specific:</strong> {ci.pten_specific}
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
            <div className="fw-bold mb-1" style={{ color: '#01579b' }}>{lc.stage}</div>
            <div className="small">
              <div><strong>Key Issues:</strong> {lc.key_issues}</div>
              <div className="mt-1"><strong>Management:</strong> {lc.management}</div>
            </div>
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
          <div key={i} className="mb-3 p-3" style={{ backgroundColor: '#f9fbe7', borderRadius: 8, border: `1px solid ${ACCENT}` }}>
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

      <SectionCard title="Evidence Standards (12)" borderColor="#4a148c">
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead style={{ backgroundColor: '#4a148c', color: '#fff' }}>
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
export default function PTENPage() {
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
      fetch(`${API}/api/pten/overview`).then(r => r.json()),
      fetch(`${API}/api/pten/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pten/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="p-3 mb-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #2e7d32 100%)`, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">&#x1f3af; PTEN Epilepsy Dashboard</h4>
        <div className="small opacity-90">
          PTEN Hamartoma Tumour Syndrome (PHTS) · mTORopathy · Cowden Syndrome · Bannayan-Riley-Ruvalcaba ·
          ASD-Macrocephaly-Epilepsy Triad · Everolimus mTOR Precision · Cancer Surveillance (NCCN) ·
          Phosphatase and Tensin Homolog · 10q23.31 · 40-Patient Cohort
        </div>
        <div className="small opacity-75 mt-1">
          Precision therapy: Everolimus (mTORC1 inhibitor — directly addresses PTEN LOF via AKT-TSC1/2-Rheb-mTOR axis) ·
          Cancer surveillance mandatory (NCCN PHTS protocol) ·
          Key CI: Everolimus+live vaccines ABSOLUTE · PHT/CBZ dual risk (CYP3A4 + absence worsening) ·
          TGB ABSOLUTE NCSE · VPA POLG1 screen mandatory · Macrocephaly+ASD = PTEN trigger (ASHG 2013)
        </div>
      </div>

      {loading && <div className="alert alert-info">Loading PTEN data…</div>}
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
