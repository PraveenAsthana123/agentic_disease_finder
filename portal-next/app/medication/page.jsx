'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const ADH_COLORS = { good: 'success', moderate: 'warning', needs_attention: 'danger', concern_flagged: 'dark' };
const SEV_COLORS = { high: 'danger', moderate: 'warning', low: 'info' };

export default function MedicationPage() {
  const [overview, setOverview] = useState(null);
  const [schedule, setSchedule] = useState(null);
  const [adherence, setAdherence] = useState(null);
  const [recs, setRecs] = useState(null);
  const [sideEffects, setSideEffects] = useState(null);
  const [tab, setTab] = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/medication`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/medication/schedule`).then(r => r.json()).then(setSchedule).catch(() => {});
    fetch(`${API}/api/medication/adherence`).then(r => r.json()).then(setAdherence).catch(() => {});
    fetch(`${API}/api/medication/recommendations`).then(r => r.json()).then(setRecs).catch(() => {});
    fetch(`${API}/api/medication/side-effects`).then(r => r.json()).then(setSideEffects).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = overview.summary || {};
  const patients = overview.my_medications?.patients || [];
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'schedule', label: 'Schedule' },
    { id: 'adherence', label: 'Adherence' },
    { id: 'warnings', label: 'Warnings' },
    { id: 'side-effects', label: 'Side Effects' },
  ];

  return (
    <div>
      <h3>&#x1f48a; Medication Dashboard</h3>
      <p className="text-muted">Anti-epileptic drug management — prescriptions, schedules, adherence, and safety alerts from clinical.db</p>

      {/* Summary cards */}
      <div className="row mb-3">
        {[
          { label: 'Patients on Meds', value: s.total_patients_on_meds, color: 'primary' },
          { label: 'Total Prescriptions', value: s.total_prescriptions, color: 'info' },
          { label: 'Unique Drugs', value: s.unique_drugs, color: 'success' },
          { label: 'Most Common', value: s.most_common_drug, color: 'warning', small: true },
          { label: 'Polypharmacy', value: s.polypharmacy_count, color: 'danger' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`${c.small ? 'h6' : 'h3'} mb-0 text-${c.color}`}>{c.value ?? '—'}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div>
          <h5>All Patients &amp; Medications</h5>
          <div className="table-responsive">
            <table className="table table-hover table-sm">
              <thead className="table-dark">
                <tr><th>Patient</th><th># Meds</th><th>Drugs</th><th>Details</th></tr>
              </thead>
              <tbody>
                {patients.map(pt => (
                  <tr key={pt.patient_id} style={{cursor:'pointer'}} onClick={() => setExpandedPt(expandedPt === pt.patient_id ? null : pt.patient_id)}>
                    <td><strong>{pt.patient_id}</strong></td>
                    <td><span className={`badge bg-${pt.medication_count > 2 ? 'danger' : pt.medication_count > 1 ? 'warning' : 'success'}`}>{pt.medication_count}</span></td>
                    <td>{pt.medications.map(m => m.drug_name).join(', ')}</td>
                    <td>{expandedPt === pt.patient_id ? '▼' : '▶'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {expandedPt && (() => {
            const pt = patients.find(p => p.patient_id === expandedPt);
            if (!pt) return null;
            return (
              <div className="card mb-3 border-primary">
                <div className="card-header bg-primary text-white">{pt.patient_id} — Medication Detail</div>
                <div className="card-body">
                  <div className="row">
                    {pt.medications.map(m => (
                      <div key={m.drug_name} className="col-md-6 mb-2">
                        <div className="card h-100">
                          <div className="card-body">
                            <h6 className="card-title">{m.drug_name} <small className="text-muted">({m.brand})</small></h6>
                            <div><strong>Dose:</strong> {m.dose_mg} mg {m.frequency}</div>
                            <div><strong>Class:</strong> {m.drug_class}</div>
                            <div className="mt-1">
                              <strong>Side Effects:</strong>{' '}
                              {m.common_side_effects?.map(se => (
                                <span key={se} className="badge bg-light text-dark me-1">{se}</span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* Schedule tab */}
      {tab === 'schedule' && schedule && (
        <div>
          <h5>Daily Medication Schedules</h5>
          {(schedule.patients || []).map(pt => (
            <div key={pt.patient_id} className="card mb-3">
              <div className="card-header"><strong>{pt.patient_id}</strong></div>
              <div className="card-body">
                <div className="row">
                  {['morning', 'noon', 'evening', 'bedtime'].map(slot => {
                    const meds = pt.daily_schedule?.[slot] || [];
                    return (
                      <div key={slot} className="col-md-3 mb-2">
                        <div className={`p-2 rounded ${meds.length > 0 ? 'bg-light' : ''}`}>
                          <div className="fw-bold text-capitalize mb-1">
                            {slot === 'morning' ? '🌅' : slot === 'noon' ? '☀️' : slot === 'evening' ? '🌆' : '🌙'} {slot}
                          </div>
                          {meds.length === 0 && <small className="text-muted">No meds</small>}
                          {meds.map(m => (
                            <div key={m.drug_name} className="small">
                              <span className="badge bg-primary me-1">{m.drug_name}</span>
                              {m.dose_mg}mg
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Adherence tab */}
      {tab === 'adherence' && adherence && (
        <div>
          <h5>Adherence Tracking</h5>
          <div className="row mb-3">
            {[
              { label: 'Good', value: adherence.summary?.good_adherence, color: 'success' },
              { label: 'Moderate', value: adherence.summary?.moderate_adherence, color: 'warning' },
              { label: 'Needs Attention', value: adherence.summary?.needs_attention, color: 'danger' },
              { label: 'Avg Adherence', value: `${adherence.summary?.avg_adherence_pct ?? 0}%`, color: 'info' },
            ].map(c => (
              <div key={c.label} className="col-6 col-md-3 mb-2">
                <div className="card text-center border-0 shadow-sm">
                  <div className="card-body py-2">
                    <div className={`h4 mb-0 text-${c.color}`}>{c.value}</div>
                    <div className="text-muted small">{c.label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr><th>Patient</th><th>Medications</th><th>Adherence</th><th>Level</th><th>Seizures</th><th>Concern</th></tr>
              </thead>
              <tbody>
                {(adherence.patients || []).map(pt => (
                  <tr key={pt.patient_id}>
                    <td><strong>{pt.patient_id}</strong></td>
                    <td>{pt.medications?.join(', ')}</td>
                    <td>
                      <div className="progress" style={{height:'20px', minWidth:'80px'}}>
                        <div className={`progress-bar bg-${ADH_COLORS[pt.adherence_level] || 'secondary'}`}
                             style={{width:`${pt.adherence_score_pct}%`}}>
                          {pt.adherence_score_pct}%
                        </div>
                      </div>
                    </td>
                    <td><span className={`badge bg-${ADH_COLORS[pt.adherence_level] || 'secondary'}`}>{(pt.adherence_level || '').replace('_', ' ')}</span></td>
                    <td>{pt.seizure_count}</td>
                    <td>{pt.concern_flag ? '⚠️' : '✓'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Warnings tab */}
      {tab === 'warnings' && recs && (
        <div>
          <h5>Safety Warnings &amp; Recommendations</h5>
          {(recs.patients || []).map(pt => (
            <div key={pt.patient_id} className="mb-3">
              <h6>{pt.patient_id} — {pt.drugs_assessed?.join(', ')}
                {pt.critical_warnings > 0 && <span className="badge bg-danger ms-2">{pt.critical_warnings} critical</span>}
                {pt.total_warnings === 0 && <span className="badge bg-success ms-2">No warnings</span>}
              </h6>
              {pt.warnings?.length > 0 && (
                <div className="list-group">
                  {pt.warnings.map((w, i) => (
                    <div key={i} className={`list-group-item list-group-item-${SEV_COLORS[w.severity] || 'secondary'}`}>
                      <div className="d-flex justify-content-between">
                        <strong>{(w.type || '').replace(/_/g, ' ')}</strong>
                        <span className={`badge bg-${SEV_COLORS[w.severity] || 'secondary'}`}>{w.severity}</span>
                      </div>
                      <div className="mt-1">{w.message}</div>
                      {w.drug && <small className="text-muted">Drug: {w.drug}</small>}
                      {w.side_effect && <small className="text-muted ms-2">Side effect: {w.side_effect}</small>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Side Effects tab */}
      {tab === 'side-effects' && sideEffects && (
        <div>
          <h5>Side Effect Profiles</h5>
          {(sideEffects.patients || []).map(pt => (
            <div key={pt.patient_id} className="card mb-3">
              <div className="card-header">
                <strong>{pt.patient_id}</strong>
                <span className="badge bg-info ms-2">{pt.total_unique_side_effects} unique side effects</span>
              </div>
              <div className="card-body">
                <div className="row">
                  {(pt.per_drug_profile || []).map(d => (
                    <div key={d.drug} className="col-md-4 mb-2">
                      <div className="card h-100 border-secondary">
                        <div className="card-body">
                          <h6>{d.drug} <small className="text-muted">({d.brand})</small></h6>
                          {d.side_effects?.map(se => (
                            <span key={se} className={`badge ${se.includes('risk') || se.includes('danger') ? 'bg-danger' : 'bg-warning text-dark'} me-1 mb-1`}>{se}</span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                {pt.overlapping_side_effects?.length > 0 && (
                  <div className="mt-2">
                    <strong>Overlapping (multi-drug):</strong>{' '}
                    {pt.overlapping_side_effects.map(o => (
                      <span key={o.side_effect} className="badge bg-danger me-1">{o.side_effect} ({o.count} drugs)</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
