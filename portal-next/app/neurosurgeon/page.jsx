'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const LESION_COLOR = (type) => {
  const m = { HS: 'danger', FCD: 'warning', NL: 'secondary', CAV: 'info', AVM: 'primary', ENC: 'dark', TUM: 'danger', NRM: 'success' };
  return m[type] || 'secondary';
};

export default function NeurosurgeonPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/neurosurgeon/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/neurosurgeon/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/neurosurgeon/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'mri-inventory', label: 'MRI Inventory' },
    { id: 'candidates', label: 'Surgical Candidates' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const maxLesion = overview.lesion_type_distribution?.length
    ? Math.max(...overview.lesion_type_distribution.map(l => l.count))
    : 1;
  const maxLat = overview.laterality_distribution?.length
    ? Math.max(...overview.laterality_distribution.map(l => l.count))
    : 1;
  const maxLoc = overview.lesion_location_distribution?.length
    ? Math.max(...overview.lesion_location_distribution.map(l => l.count))
    : 1;

  const lesionalPct = overview.total_mri_scans
    ? Math.round((overview.lesional_count / overview.total_mri_scans) * 100)
    : 0;
  const surgicalPct = overview.total_patients
    ? Math.round((overview.surgical_candidates / overview.total_patients) * 100)
    : 0;

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">&#x1f9e0; Neurosurgeon / Epilepsy Surgery Dashboard</h4>
      <p className="text-muted small mb-3">
        Surgical candidacy evaluation — {overview.total_patients} patients · {overview.total_mri_scans} MRI scans ·{' '}
        {overview.surgical_candidates} surgical candidates ({surgicalPct}%) · {overview.eeg_analyses_count} EEG analyses
      </p>

      {/* KPI cards */}
      <div className="row g-2 mb-3">
        {[
          { label: 'Total Patients', value: overview.total_patients, sub: 'in cohort', color: 'primary' },
          { label: 'Lesional Cases', value: `${overview.lesional_count} (${lesionalPct}%)`, sub: 'structural abnormality on MRI', color: 'danger' },
          { label: 'Non-Lesional', value: overview.non_lesional_count, sub: 'no visible MRI lesion', color: 'secondary' },
          { label: 'Surgical Candidates', value: `${overview.surgical_candidates} (${surgicalPct}%)`, sub: 'DRE + concordant workup', color: 'success' },
        ].map((card) => (
          <div key={card.label} className="col-6 col-md-3">
            <div className={`card border-${card.color} h-100`}>
              <div className="card-body p-2 text-center">
                <div className={`fs-4 fw-bold text-${card.color}`}>{card.value ?? '—'}</div>
                <div className="small fw-semibold">{card.label}</div>
                <div className="text-muted" style={{ fontSize: '0.7rem' }}>{card.sub}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Lesion type distribution */}
          <div className="col-md-4">
            <div className="card h-100">
              <div className="card-header fw-semibold">MRI Lesion Types</div>
              <div className="card-body">
                {(overview.lesion_type_distribution || []).map((l) => (
                  <div key={l.lesion_type} className="mb-2">
                    <div className="d-flex justify-content-between small">
                      <span>
                        <span className={`badge bg-${LESION_COLOR(l.lesion_type)} me-1`}>{l.lesion_type}</span>
                        {l.label}
                      </span>
                      <span className="fw-bold">{l.count}</span>
                    </div>
                    <div className="progress" style={{ height: 8 }}>
                      <div
                        className={`progress-bar bg-${LESION_COLOR(l.lesion_type)}`}
                        style={{ width: `${(l.count / maxLesion) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Laterality */}
          <div className="col-md-4">
            <div className="card h-100">
              <div className="card-header fw-semibold">Seizure Laterality</div>
              <div className="card-body">
                {(overview.laterality_distribution || []).map((l) => (
                  <div key={l.laterality} className="mb-2">
                    <div className="d-flex justify-content-between small">
                      <span className="fw-semibold">{l.laterality || 'Unknown'}</span>
                      <span className="fw-bold">{l.count}</span>
                    </div>
                    <div className="progress" style={{ height: 8 }}>
                      <div
                        className="progress-bar bg-info"
                        style={{ width: `${(l.count / maxLat) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Lesion location */}
          <div className="col-md-4">
            <div className="card h-100">
              <div className="card-header fw-semibold">Lesion Location</div>
              <div className="card-body">
                {(overview.lesion_location_distribution || []).map((l) => (
                  <div key={l.location} className="mb-2">
                    <div className="d-flex justify-content-between small">
                      <span className="fw-semibold">{l.location}</span>
                      <span className="fw-bold">{l.count}</span>
                    </div>
                    <div className="progress" style={{ height: 8 }}>
                      <div
                        className="progress-bar bg-warning"
                        style={{ width: `${(l.count / maxLoc) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Seizure severity distribution */}
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">Seizure Severity Distribution (avg score: {overview.avg_seizure_severity_score})</div>
              <div className="card-body">
                <div className="row g-1 text-center">
                  {(overview.seizure_severity_distribution || []).map((s) => (
                    <div key={s.severity} className="col">
                      <div className={`p-2 rounded ${s.severity === 'Severe' ? 'bg-danger text-white' : s.severity === 'Moderate' ? 'bg-warning text-dark' : 'bg-success text-white'}`}>
                        <div className="fs-5 fw-bold">{s.count}</div>
                        <div style={{ fontSize: '0.75rem' }}>{s.severity}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* MRI INVENTORY */}
      {tab === 'mri-inventory' && breakdown && (
        <div className="card">
          <div className="card-header fw-semibold">MRI Scan Inventory — All Patients</div>
          <div className="card-body p-0">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Lesion Type</th>
                  <th>Label</th>
                  <th>Location</th>
                  <th>Laterality</th>
                  <th>Classification</th>
                  <th>HS</th>
                  <th>Vol Asymmetry</th>
                  <th>Confidence</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.mri_inventory || []).map((m, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{m.patient_id}</td>
                    <td>
                      <span className={`badge bg-${LESION_COLOR(m.lesion_type)}`}>{m.lesion_type}</span>
                    </td>
                    <td><small>{m.lesion_label}</small></td>
                    <td><small>{m.lesion_location || '—'}</small></td>
                    <td><small>{m.laterality || '—'}</small></td>
                    <td>
                      <span className={`badge ${m.classification === 'LESIONAL' ? 'bg-danger' : 'bg-secondary'}`}>
                        {m.classification}
                      </span>
                    </td>
                    <td>{m.hippocampal_sclerosis === 'Yes' ? '✅' : '—'}</td>
                    <td><small>{m.hippocampal_volume_asymmetry}</small></td>
                    <td>
                      <span className={`badge ${m.radiologist_confidence === 'High' ? 'bg-success' : m.radiologist_confidence === 'Low' ? 'bg-danger' : 'bg-warning text-dark'}`}>
                        {m.radiologist_confidence}
                      </span>
                    </td>
                    <td><small className="text-muted">{m.created_at?.slice(0, 10)}</small></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* SURGICAL CANDIDATES */}
      {tab === 'candidates' && breakdown && (
        <div className="row g-3">
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">Surgical Candidate Assessment</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Patient</th>
                      <th>Lesion</th>
                      <th>Location</th>
                      <th>Laterality</th>
                      <th>Seizure Severity</th>
                      <th>EEG Studies</th>
                      <th>AED Trials</th>
                      <th>Candidate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.surgical_candidates || []).map((c, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{c.patient_id}</td>
                        <td>
                          <span className={`badge bg-${LESION_COLOR(c.lesion_type)}`}>
                            {c.lesion_type || 'NL'}
                          </span>
                        </td>
                        <td><small>{c.lesion_location || '—'}</small></td>
                        <td><small>{c.laterality || '—'}</small></td>
                        <td>
                          <span className={`badge ${c.avg_seizure_severity >= 3 ? 'bg-danger' : c.avg_seizure_severity >= 2 ? 'bg-warning text-dark' : 'bg-success'}`}>
                            {c.avg_seizure_severity}
                          </span>
                        </td>
                        <td>{c.eeg_study_count}</td>
                        <td>{c.aed_trials}</td>
                        <td>
                          <span className={`badge ${c.surgical_candidate ? 'bg-success' : 'bg-secondary'}`}>
                            {c.surgical_candidate ? 'Yes' : 'No'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DEFINITIONS */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">Key Concepts</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th style={{ width: '25%' }}>Concept</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.concepts || []).map((c) => (
                      <tr key={c.name}>
                        <td className="fw-semibold small">{c.name}</td>
                        <td><small className="text-muted">{c.description}</small></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">Quality Metrics</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Metric</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.quality_metrics || []).map((q) => (
                      <tr key={q.name}>
                        <td className="fw-semibold small">{q.name}</td>
                        <td><small className="text-muted">{q.description}</small></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">Surgical Procedures</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Procedure</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.surgical_procedures || []).map((p) => (
                      <tr key={p.name}>
                        <td className="fw-semibold small">{p.name}</td>
                        <td><small className="text-muted">{p.description}</small></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
