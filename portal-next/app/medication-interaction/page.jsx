'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',      label: 'Overview' },
  { id: 'interactions',  label: 'Interactions' },
  { id: 'prescriptions', label: 'Prescriptions' },
  { id: 'adr',           label: 'ADR Alerts' },
  { id: 'definitions',   label: 'Definitions' },
];

const SEV_COLOR = { major: 'danger', moderate: 'warning', minor: 'info', none: 'success' };
const ADR_COLOR = { behavioral: 'warning', dermatologic: 'danger', metabolic: 'info', hematologic: 'danger', cognitive: 'warning' };

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function SevBadge({ sev }) {
  return <span className={`badge bg-${SEV_COLOR[sev] || 'secondary'}`}>{sev}</span>;
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  if (ov.error) return <div className="alert alert-warning">{ov.error}</div>;
  const k = ov.kpis || {};
  const drugs = ov.drug_frequency || [];
  const sevs = ov.severity_distribution || [];
  const poly = ov.polytherapy_distribution || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Prescriptions" value={k.total_prescriptions} color="primary" sub="active ASM records" />
        <KPI label="Unique Drugs" value={k.unique_drugs} color="secondary" sub="distinct AEDs" />
        <KPI label="Patients on Meds" value={k.patients_on_meds} color="info" sub="with prescriptions" />
        <KPI label="Interactions Detected" value={k.interactions_detected} color={k.interactions_detected > 0 ? 'warning' : 'success'} sub="drug pairs screened" />
      </div>
      <div className="row mb-4">
        <KPI label="Major Interactions" value={k.major_interactions} color={k.major_interactions > 0 ? 'danger' : 'success'} sub="require action" />
        <KPI label="Patients At Risk" value={k.patients_at_risk} color={k.patients_at_risk > 0 ? 'danger' : 'success'} sub="on interacting drugs" />
        <KPI label="Polytherapy Patients" value={k.polytherapy_patients} color="warning" sub="on ≥2 AEDs" />
        <KPI label="ADR Flags" value={k.adr_flags} color="secondary" sub="adverse drug reaction risk" />
      </div>

      <div className="row g-3 mb-3">
        <div className="col-md-4">
          <div className="card h-100">
            <div className="card-header fw-semibold small">Drug Frequency</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Drug (AED)</th><th>Prescriptions</th></tr></thead>
                <tbody>
                  {drugs.map((d, i) => (
                    <tr key={i}>
                      <td className="fw-semibold small">{d.drug}</td>
                      <td><span className="badge bg-primary">{d.count}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-4">
          <div className="card h-100">
            <div className="card-header fw-semibold small">Interaction Severity</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Severity</th><th>Count</th></tr></thead>
                <tbody>
                  {sevs.map((s, i) => (
                    <tr key={i}>
                      <td><SevBadge sev={s.severity} /></td>
                      <td className="fw-bold">{s.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-4">
          <div className="card h-100">
            <div className="card-header fw-semibold small">Polytherapy Tiers</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Tier</th><th>Patients</th></tr></thead>
                <tbody>
                  {poly.map((p, i) => (
                    <tr key={i}>
                      <td className="text-capitalize small fw-semibold">{p.tier}</td>
                      <td><span className={`badge bg-${p.tier === 'tritherapy' ? 'danger' : p.tier === 'duotherapy' ? 'warning' : 'success'}`}>{p.count}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {k.major_interactions > 0 && (
        <div className="alert alert-danger small d-flex align-items-start gap-2">
          <span style={{ fontSize: '1.2rem' }}>&#x26a0;&#xfe0f;</span>
          <div>
            <strong>Major interaction alert:</strong> {k.major_interactions} major drug-drug interaction(s) detected.
            See the <strong>Interactions</strong> tab for mechanism, required action, and clinical reference.
            Polytherapy patients ({k.polytherapy_patients}) require pharmacist review.
          </div>
        </div>
      )}
    </div>
  );
}

function InteractionsPanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  if (bd.error) return <div className="alert alert-warning">{bd.error}</div>;

  const results = bd.interaction_results || [];
  const kb = bd.interaction_knowledge_base || [];
  const [sevFilter, setSevFilter] = useState('all');

  const filtered = sevFilter === 'all' ? results : results.filter(r => r.severity === sevFilter);

  return (
    <div>
      <div className="card mb-3">
        <div className="card-header fw-semibold small d-flex align-items-center justify-content-between">
          <span>Patient Interaction Screening Results ({results.length} pairs)</span>
          <div className="d-flex gap-1">
            {['all', 'major', 'moderate', 'minor', 'none'].map(s => (
              <button
                key={s}
                className={`btn btn-sm ${sevFilter === s ? `btn-${s === 'all' ? 'primary' : SEV_COLOR[s] || 'secondary'}` : 'btn-outline-secondary'}`}
                onClick={() => setSevFilter(s)}
              >
                {s.charAt(0).toUpperCase() + s.slice(1)} ({s === 'all' ? results.length : results.filter(r => r.severity === s).length})
              </button>
            ))}
          </div>
        </div>
        <div className="card-body p-0">
          <div className="table-responsive" style={{ maxHeight: 340 }}>
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Drug A</th>
                  <th>Drug B</th>
                  <th>Severity</th>
                  <th>Mechanism</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => (
                  <tr key={i} className={r.severity === 'major' ? 'table-danger' : r.severity === 'moderate' ? 'table-warning' : ''}>
                    <td><span className="badge bg-secondary">{r.patient_id}</span></td>
                    <td className="fw-semibold small">{r.drug_a}</td>
                    <td className="fw-semibold small">{r.drug_b}</td>
                    <td><SevBadge sev={r.severity} /></td>
                    <td className="small text-muted" style={{ maxWidth: 280 }}>{r.mechanism}</td>
                    <td className="small fw-semibold" style={{ maxWidth: 200 }}>{r.action}</td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr><td colSpan={6} className="text-center text-muted small py-3">No interactions at this severity level.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header fw-semibold small">AED Interaction Knowledge Base ({kb.length} known pairs)</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Drug A</th><th>Drug B</th><th>Severity</th><th>Mechanism</th><th>Action Required</th><th>Reference</th></tr>
              </thead>
              <tbody>
                {kb.map((r, i) => (
                  <tr key={i} className={r.severity === 'major' ? 'table-danger' : r.severity === 'moderate' ? 'table-warning' : ''}>
                    <td className="fw-semibold small">{r.drug_a}</td>
                    <td className="fw-semibold small">{r.drug_b}</td>
                    <td><SevBadge sev={r.severity} /></td>
                    <td className="small text-muted" style={{ maxWidth: 260 }}>{r.mechanism}</td>
                    <td className="small fw-semibold" style={{ maxWidth: 200 }}>{r.action}</td>
                    <td className="small text-muted">{r.reference}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function PrescriptionsPanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  if (bd.error) return <div className="alert alert-warning">{bd.error}</div>;

  const meds = bd.medication_inventory || [];
  const [sortCol, setSortCol] = useState('patient_id');
  const [sortDir, setSortDir] = useState(1);

  const sorted = [...meds].sort((a, b) => {
    const av = a[sortCol] ?? '', bv = b[sortCol] ?? '';
    return sortDir * (av < bv ? -1 : av > bv ? 1 : 0);
  });

  function toggleSort(col) {
    if (sortCol === col) setSortDir(d => -d);
    else { setSortCol(col); setSortDir(1); }
  }
  function th(col, label) {
    const active = sortCol === col;
    return (
      <th onClick={() => toggleSort(col)} style={{ cursor: 'pointer', userSelect: 'none' }} className={active ? 'text-primary' : ''}>
        {label} {active ? (sortDir === 1 ? '▲' : '▼') : '↕'}
      </th>
    );
  }

  return (
    <div className="card">
      <div className="card-header fw-semibold small">Prescription Inventory ({meds.length} records)</div>
      <div className="card-body p-0">
        <div className="table-responsive" style={{ maxHeight: 500 }}>
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                {th('patient_id', 'Patient')}
                {th('drug_name', 'Drug (AED)')}
                {th('dose_mg', 'Dose (mg)')}
                {th('frequency', 'Frequency')}
                <th>ADR Risk</th>
                {th('adr_type', 'ADR Type')}
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((m, i) => (
                <tr key={i}>
                  <td><span className="badge bg-secondary">{m.patient_id}</span></td>
                  <td className="fw-semibold small">{m.drug_name}</td>
                  <td>{m.dose_mg} mg</td>
                  <td className="small">{m.frequency}</td>
                  <td>
                    <span className={`badge bg-${m.has_adr_risk ? 'danger' : 'success'}`}>
                      {m.has_adr_risk ? 'Yes' : 'No'}
                    </span>
                  </td>
                  <td>
                    {m.adr_type
                      ? <span className={`badge bg-${ADR_COLOR[m.adr_type] || 'secondary'}`}>{m.adr_type}</span>
                      : <span className="text-muted small">—</span>}
                  </td>
                  <td className="small text-muted">{m.created_at?.split('T')[0] || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ADRPanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  if (bd.error) return <div className="alert alert-warning">{bd.error}</div>;

  const meds = (bd.medication_inventory || []).filter(m => m.has_adr_risk);
  const adrGroups = meds.reduce((acc, m) => {
    acc[m.adr_type] = acc[m.adr_type] || [];
    acc[m.adr_type].push(m);
    return acc;
  }, {});

  return (
    <div>
      <div className="alert alert-warning small mb-3">
        <strong>ADR Monitoring:</strong> {meds.length} prescriptions flagged for adverse drug reaction risk across {Object.keys(adrGroups).length} categories.
        All patients on high-ADR-risk AEDs require follow-up assessments.
      </div>

      <div className="row g-3 mb-3">
        {Object.entries(adrGroups).map(([adrType, items]) => (
          <div key={adrType} className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold small d-flex align-items-center gap-2">
                <span className={`badge bg-${ADR_COLOR[adrType] || 'secondary'}`}>{adrType}</span>
                <span>{items.length} prescription{items.length > 1 ? 's' : ''}</span>
              </div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Patient</th><th>Drug</th><th>Dose</th><th>Freq</th></tr></thead>
                  <tbody>
                    {items.map((m, i) => (
                      <tr key={i}>
                        <td><span className="badge bg-secondary">{m.patient_id}</span></td>
                        <td className="small fw-semibold">{m.drug_name}</td>
                        <td className="small">{m.dose_mg} mg</td>
                        <td className="small">{m.frequency}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card">
        <div className="card-header fw-semibold small">ADR Type Reference</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th>Type</th><th>Clinical Impact</th><th>Monitoring</th></tr></thead>
            <tbody>
              {[
                ['behavioral', 'Mood changes, irritability, psychosis risk', 'PHQ-9, GAD-7, psychiatric review'],
                ['dermatologic', 'Rash, Stevens-Johnson Syndrome risk (SJS)', 'Skin checks, halt on rash, dermatology referral'],
                ['metabolic', 'Weight gain, liver toxicity, hyperammonemia', 'LFT, ammonia, lipid panel'],
                ['hematologic', 'Aplastic anemia, thrombocytopenia', 'CBC with differential, platelet count'],
                ['cognitive', 'Memory impairment, word-finding, processing speed', 'MoCA/neuropsych battery, dose review'],
              ].map(([type, impact, monitoring]) => (
                <tr key={type}>
                  <td><span className={`badge bg-${ADR_COLOR[type] || 'secondary'}`}>{type}</span></td>
                  <td className="small">{impact}</td>
                  <td className="small text-muted">{monitoring}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  if (defs.error) return <div className="alert alert-warning">{defs.error}</div>;

  const concepts = defs.concepts || [];
  const metrics = defs.quality_metrics || [];
  const classes = defs.drug_classes || [];
  const refs = defs.compliance_references || [];

  return (
    <div>
      <div className="card mb-3">
        <div className="card-header fw-semibold small">Glossary ({concepts.length} terms)</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th style={{ width: '25%' }}>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {concepts.map((c, i) => (
                <tr key={i}>
                  <td className="fw-semibold small align-top">{c.term}</td>
                  <td className="small">{c.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {classes.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold small">AED Drug Classes</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-light"><tr><th>Class</th><th>Examples</th></tr></thead>
              <tbody>
                {classes.map((c, i) => (
                  <tr key={i}>
                    <td className="fw-semibold small">{c.class}</td>
                    <td className="small text-muted">{c.examples}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {metrics.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold small">Quality Metrics</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-light"><tr><th>Metric</th><th>Target</th><th>Description</th></tr></thead>
              <tbody>
                {metrics.map((m, i) => (
                  <tr key={i}>
                    <td className="fw-semibold small">{m.metric}</td>
                    <td><span className="badge bg-success">{m.target}</span></td>
                    <td className="small text-muted">{m.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {refs.length > 0 && (
        <div className="card">
          <div className="card-header fw-semibold small">Compliance References</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-light"><tr><th>Reference</th><th>Topic</th></tr></thead>
              <tbody>
                {refs.map((r, i) => (
                  <tr key={i}>
                    <td className="small fw-semibold">{r.citation || r.reference}</td>
                    <td className="small text-muted">{r.topic || r.standard}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default function MedicationInteractionPage() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv]   = useState(null);
  const [bd, setBd]   = useState(null);
  const [defs, setDefs] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/medication-interaction/overview`)
      .then(r => r.json()).then(setOv)
      .catch(() => setOv({ error: 'Failed to load overview' }));
    fetch(`${API}/api/medication-interaction/breakdown`)
      .then(r => r.json()).then(setBd)
      .catch(() => setBd({ error: 'Failed to load breakdown' }));
    fetch(`${API}/api/medication-interaction/definitions`)
      .then(r => r.json()).then(setDefs)
      .catch(() => setDefs({ error: 'Failed to load definitions' }));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: '1.6rem' }}>&#x26a0;&#xfe0f;</span>
        <div>
          <h4 className="mb-0 fw-bold">Medication Interaction Checker</h4>
          <div className="text-muted small">
            AED drug-drug interaction screening · polytherapy risk · ADR alerts · interaction knowledge base
            — 9 prescriptions, 5 AEDs, 4 interaction pairs detected
          </div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'      && <OverviewPanel ov={ov} />}
      {tab === 'interactions'  && <InteractionsPanel bd={bd} />}
      {tab === 'prescriptions' && <PrescriptionsPanel bd={bd} />}
      {tab === 'adr'           && <ADRPanel bd={bd} />}
      {tab === 'definitions'   && <DefinitionsPanel defs={defs} />}
    </div>
  );
}
