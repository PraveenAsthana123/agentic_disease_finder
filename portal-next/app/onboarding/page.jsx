'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-2">
      <div className="card text-center shadow-sm border-0">
        <div className="card-body py-2 px-1">
          <div className={`h4 mb-0 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted" style={{ fontSize: '0.75rem' }}>{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function SectionBar({ name, pct, color }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small fw-semibold">
        <span>{name}</span><span>{pct}%</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const k = data.kpis || {};
  const dist = data.completion_distribution || {};
  const monthly = data.onboarding_by_month || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Patients" value={k.total_patients} color="primary" />
        <KPI label="Onboarded" value={k.onboarded_count} color="success" sub="all sections complete" />
        <KPI label="Completion %" value={`${k.onboarding_completion_pct}%`} color={k.onboarding_completion_pct >= 50 ? 'success' : 'warning'} />
        <KPI label="Has Demographics" value={k.patients_with_demographics} color="info" />
        <KPI label="Has Emergency Contact" value={k.patients_with_emergency_contacts} color="secondary" />
        <KPI label="Has Medications" value={k.patients_with_medications} color="warning" />
        <KPI label="Has Documents" value={k.patients_with_documents_uploaded} color="info" />
        <KPI label="Avg Docs/Patient" value={k.avg_documents_per_patient} color="primary" />
      </div>

      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Completion Distribution</div>
            <div className="card-body">
              {Object.entries(dist).map(([bucket, count]) => {
                const colors = { '0-24': 'danger', '25-49': 'warning', '50-74': 'info', '75-99': 'primary', '100': 'success' };
                return (
                  <div key={bucket} className="d-flex align-items-center mb-1">
                    <span className="small fw-semibold me-2" style={{ width: 50 }}>{bucket}%</span>
                    <div className="progress flex-grow-1" style={{ height: 14 }}>
                      <div className={`progress-bar bg-${colors[bucket] || 'secondary'}`}
                           style={{ width: `${(count / (k.total_patients || 1)) * 100}%` }}>
                        {count}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Onboarding by Month</div>
            <div className="card-body">
              {monthly.length === 0 ? <span className="text-muted">No monthly data</span> :
                <table className="table table-sm mb-0">
                  <thead><tr><th>Month</th><th>New Patients</th></tr></thead>
                  <tbody>
                    {monthly.map(m => (
                      <tr key={m.month}><td>{m.month}</td><td>{m.count}</td></tr>
                    ))}
                  </tbody>
                </table>
              }
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const patients = data.patients || [];
  const stats = data.section_stats || {};
  const [filter, setFilter] = useState('all');

  const filtered = filter === 'all' ? patients :
    filter === 'complete' ? patients.filter(p => p.onboarded) :
    patients.filter(p => !p.onboarded);

  return (
    <div>
      <div className="row mb-3">
        {Object.entries(stats).map(([sec, pct]) => (
          <div key={sec} className="col-md-4 col-lg mb-2">
            <SectionBar name={sec.replace(/_/g, ' ')} pct={pct}
              color={pct >= 75 ? 'success' : pct >= 50 ? 'warning' : 'danger'} />
          </div>
        ))}
      </div>

      <div className="mb-2">
        {['all', 'incomplete', 'complete'].map(f => (
          <button key={f} className={`btn btn-sm me-1 ${filter === f ? 'btn-primary' : 'btn-outline-secondary'}`}
                  onClick={() => setFilter(f)}>
            {f === 'all' ? `All (${patients.length})` :
             f === 'complete' ? `Complete (${patients.filter(p => p.onboarded).length})` :
             `Incomplete (${patients.filter(p => !p.onboarded).length})`}
          </button>
        ))}
      </div>

      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-dark">
            <tr>
              <th>Patient</th>
              <th>Demographics</th>
              <th>Emergency</th>
              <th>Medications</th>
              <th>Documents</th>
              <th>Comorbidities</th>
              <th>Completion</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(p => (
              <tr key={p.patient_id}>
                <td className="fw-semibold small">{p.patient_id}</td>
                {['demographics', 'emergency_contact', 'medications', 'documents', 'comorbidities'].map(sec => (
                  <td key={sec} className="text-center">
                    {p.sections[sec]?.complete
                      ? <span className="badge bg-success">✓</span>
                      : <span className="badge bg-secondary">✗</span>}
                  </td>
                ))}
                <td>
                  <div className="progress" style={{ height: 10, minWidth: 60 }}>
                    <div className={`progress-bar bg-${p.completion_pct >= 80 ? 'success' : p.completion_pct >= 40 ? 'warning' : 'danger'}`}
                         style={{ width: `${p.completion_pct}%` }} />
                  </div>
                  <span className="small">{p.completion_pct}%</span>
                </td>
                <td>
                  {p.onboarded
                    ? <span className="badge bg-success">Onboarded</span>
                    : <span className="badge bg-warning text-dark">In Progress</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const sections = data.sections || [];
  const steps = data.process_steps || [];
  const glossary = data.glossary || [];

  return (
    <div>
      <div className="row mb-3">
        <div className="col-md-4">
          <div className="card border-primary mb-3">
            <div className="card-body text-center">
              <div className="h3 text-primary">{data.required_fields}</div>
              <div className="small fw-semibold">Required Fields</div>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-secondary mb-3">
            <div className="card-body text-center">
              <div className="h3 text-secondary">{data.deferred_fields}</div>
              <div className="small fw-semibold">Deferred Fields</div>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-success mb-3">
            <div className="card-body text-center">
              <div className="h3 text-success">{steps.length}</div>
              <div className="small fw-semibold">Process Steps</div>
            </div>
          </div>
        </div>
      </div>

      {/* 3-Step Process */}
      <h6>Onboarding Process</h6>
      <div className="d-flex mb-4 gap-2">
        {steps.map((s, i) => (
          <div key={i} className="card flex-fill text-center border-primary">
            <div className="card-body py-2">
              <div className="badge bg-primary rounded-pill mb-1">Step {s.step}</div>
              <div className="fw-semibold small">{s.name}</div>
              <div className="text-muted" style={{ fontSize: '0.7rem' }}>{s.description}</div>
              <div className="text-info small">{s.target_time}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Sections */}
      <h6>Onboarding Sections</h6>
      <table className="table table-sm table-bordered mb-4">
        <thead className="table-light">
          <tr><th>Section</th><th>Table</th><th>Required</th><th>Deferred</th><th>Weight</th></tr>
        </thead>
        <tbody>
          {sections.map(s => (
            <tr key={s.name}>
              <td className="fw-semibold">{s.name}</td>
              <td className="small text-muted">{s.table}</td>
              <td>{s.required_fields}</td>
              <td>{s.deferred_fields}</td>
              <td>{s.weight_pct}%</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Glossary */}
      <h6>Glossary</h6>
      <dl className="row small">
        {glossary.map(g => (
          <div key={g.term} className="col-md-6 mb-1">
            <dt className="d-inline">{g.term}: </dt>
            <dd className="d-inline text-muted">{g.definition}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export default function OnboardingPage() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/onboarding/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/onboarding/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/onboarding/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'breakdown', label: 'Per Patient' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>📋 Patient Onboarding Dashboard</h3>
      <p className="text-muted small">
        Track patient intake completeness across 5 sections: demographics, emergency contacts,
        medications, documents, and comorbidities. Goal: onboard in 8–10 min, not 2–3 hours.
      </p>

      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && <OverviewPanel data={ov} />}
      {tab === 'breakdown' && <BreakdownPanel data={bd} />}
      {tab === 'definitions' && <DefinitionsPanel data={defs} />}
    </div>
  );
}
