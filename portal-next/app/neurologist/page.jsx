'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const confColor = v =>
  v >= 80 ? 'success' : v >= 60 ? 'primary' : v >= 40 ? 'warning' : 'danger';

export default function NeurologistDashboardPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [sel,  setSel]  = useState(null);

  useEffect(() => {
    fetch(`${API}/api/neurologist/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/neurologist/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/neurologist/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'patients',    label: 'Patient List' },
    { id: 'confidence',  label: 'Confidence Analysis' },
    { id: 'definitions', label: 'Clinical Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f9e0; Neurologist Dashboard</h3>
      <p className="text-muted small">
        EEG interpretation workstation — pending reads, seizure-positive rate, AI confidence,
        HITL overrides, and per-patient seizure/medication tracking.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {ov.kpis.map(k => (
          <div key={k.label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${
                  k.label.includes('Pending') ? (k.value > 50 ? 'danger' : 'warning') :
                  k.label.includes('Confidence') ? confColor(k.value) :
                  k.label.includes('Override') ? 'info' : 'primary'
                }`}>
                  {typeof k.value === 'number' && k.unit === '%' ? `${k.value.toFixed(1)}%` :
                   typeof k.value === 'number' && k.unit === 'hours' ? `${k.value.toFixed(1)}h` :
                   k.value ?? '—'}
                </div>
                <div className="text-muted" style={{fontSize: '0.72rem'}}>{k.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div className="row">
          {/* Prediction distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-primary text-white py-2 small fw-bold">Prediction Distribution</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Diagnosis</th><th className="text-end">Count</th><th style={{width:'40%'}}>Bar</th></tr></thead>
                  <tbody>
                    {Object.entries(ov.prediction_distribution || {})
                      .sort((a,b) => b[1] - a[1])
                      .map(([dx, cnt]) => {
                        const max = Math.max(...Object.values(ov.prediction_distribution));
                        return (
                          <tr key={dx}>
                            <td className="small">{dx}</td>
                            <td className="text-end small fw-bold">{cnt}</td>
                            <td>
                              <div className="progress" style={{height: '14px'}}>
                                <div className={`progress-bar bg-${dx === 'Epilepsy' ? 'danger' : dx === 'Normal' ? 'success' : 'info'}`}
                                     style={{width: `${(cnt / max * 100).toFixed(0)}%`}} />
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Signal quality distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-dark text-white py-2 small fw-bold">Signal Quality Distribution</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Quality</th><th className="text-end">Count</th><th style={{width:'40%'}}>Bar</th></tr></thead>
                  <tbody>
                    {Object.entries(ov.severity_distribution || {})
                      .sort((a,b) => b[1] - a[1])
                      .map(([q, cnt]) => {
                        const max = Math.max(...Object.values(ov.severity_distribution));
                        const color = q === 'Excellent' ? 'success' : q === 'Good' ? 'primary' : q === 'Fair' ? 'warning' : 'danger';
                        return (
                          <tr key={q}>
                            <td className="small"><span className={`badge bg-${color}`}>{q}</span></td>
                            <td className="text-end small fw-bold">{cnt}</td>
                            <td>
                              <div className="progress" style={{height: '14px'}}>
                                <div className={`progress-bar bg-${color}`}
                                     style={{width: `${(cnt / max * 100).toFixed(0)}%`}} />
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Confidence histogram */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-info text-white py-2 small fw-bold">AI Confidence Histogram</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Bucket</th><th className="text-end">Count</th><th style={{width:'50%'}}>Bar</th></tr></thead>
                  <tbody>
                    {Object.entries(ov.confidence_histogram || {}).map(([bucket, cnt]) => {
                      const max = Math.max(...Object.values(ov.confidence_histogram).filter(v => v > 0), 1);
                      return (
                        <tr key={bucket}>
                          <td className="small">{bucket}</td>
                          <td className="text-end small fw-bold">{cnt}</td>
                          <td>
                            <div className="progress" style={{height: '14px'}}>
                              <div className="progress-bar bg-info" style={{width: `${(cnt / max * 100).toFixed(0)}%`}} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Turnaround histogram */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-secondary text-white py-2 small fw-bold">Turnaround Time Distribution</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Window</th><th className="text-end">Count</th><th style={{width:'50%'}}>Bar</th></tr></thead>
                  <tbody>
                    {Object.entries(ov.turnaround_histogram || {}).map(([window, cnt]) => {
                      const max = Math.max(...Object.values(ov.turnaround_histogram).filter(v => v > 0), 1);
                      return (
                        <tr key={window}>
                          <td className="small">{window}</td>
                          <td className="text-end small fw-bold">{cnt}</td>
                          <td>
                            <div className="progress" style={{height: '14px'}}>
                              <div className="progress-bar bg-secondary" style={{width: `${(cnt / max * 100).toFixed(0)}%`}} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Patients tab */}
      {tab === 'patients' && bd && (
        <div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Age</th>
                  <th>Gender</th>
                  <th className="text-end">Analyses</th>
                  <th className="text-end">Pending</th>
                  <th className="text-end">Seizure+</th>
                  <th className="text-end">Confidence</th>
                  <th className="text-end">Sz/month</th>
                  <th>Med Response</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {(bd.patients || []).map(p => (
                  <tr key={p.patient_id} className={sel === p.patient_id ? 'table-active' : ''}>
                    <td className="small fw-bold">{p.patient_id}</td>
                    <td className="small">{p.age ?? '—'}</td>
                    <td className="small">{p.gender || '—'}</td>
                    <td className="text-end small">{p.analyses_count}</td>
                    <td className="text-end small">
                      {p.pending_count > 0
                        ? <span className="badge bg-warning text-dark">{p.pending_count}</span>
                        : '0'}
                    </td>
                    <td className="text-end small">{p.seizure_positive_count}</td>
                    <td className="text-end small">
                      <span className={`text-${confColor(p.avg_confidence)}`}>
                        {p.avg_confidence > 0 ? `${p.avg_confidence.toFixed(1)}%` : '—'}
                      </span>
                    </td>
                    <td className="text-end small">
                      {p.seizure_frequency_per_month != null
                        ? <span className={p.seizure_frequency_per_month > 4 ? 'text-danger fw-bold' : ''}>
                            {p.seizure_frequency_per_month.toFixed(1)}
                          </span>
                        : '—'}
                    </td>
                    <td className="small">
                      {p.medication_response_grade && (
                        <span className={`badge bg-${
                          p.medication_response_grade === 'Well controlled' ? 'success' :
                          p.medication_response_grade === 'Partially controlled' ? 'warning' :
                          p.medication_response_grade === 'Poorly controlled' ? 'danger' : 'secondary'
                        }`}>
                          {p.medication_response_grade}
                        </span>
                      )}
                    </td>
                    <td>
                      <button className="btn btn-outline-primary btn-sm py-0 px-1"
                              style={{fontSize:'0.7rem'}}
                              onClick={() => setSel(sel === p.patient_id ? null : p.patient_id)}>
                        {sel === p.patient_id ? 'Hide' : 'Detail'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Patient detail panel */}
          {sel && bd.patients && (() => {
            const p = bd.patients.find(x => x.patient_id === sel);
            if (!p) return null;
            return (
              <div className="card shadow-sm border-0 mb-3">
                <div className="card-header bg-primary text-white py-2 small fw-bold">
                  Patient Detail: {p.patient_id} — {p.name}
                </div>
                <div className="card-body p-3">
                  <div className="row">
                    {/* Medications */}
                    <div className="col-md-4 mb-2">
                      <h6 className="small fw-bold">Current Medications</h6>
                      {(p.medications || []).length === 0
                        ? <span className="text-muted small">None recorded</span>
                        : <ul className="list-unstyled mb-0">
                            {p.medications.map((m, i) => (
                              <li key={i} className="small">
                                <strong>{m.drug_name}</strong> {m.dose_mg}mg {m.frequency}
                              </li>
                            ))}
                          </ul>}
                    </div>
                    {/* Seizure diary */}
                    <div className="col-md-4 mb-2">
                      <h6 className="small fw-bold">Recent Seizure Events</h6>
                      {(p.seizure_diary || []).length === 0
                        ? <span className="text-muted small">No events</span>
                        : <ul className="list-unstyled mb-0">
                            {p.seizure_diary.slice(0, 5).map((e, i) => (
                              <li key={i} className="small">
                                {e.event_date} — <span className={`text-${e.severity === 'Severe' ? 'danger' : e.severity === 'Moderate' ? 'warning' : 'info'}`}>
                                  {e.severity}
                                </span> ({e.duration_sec}s) {e.trigger && <em>trigger: {e.trigger}</em>}
                              </li>
                            ))}
                          </ul>}
                    </div>
                    {/* HITL reviews */}
                    <div className="col-md-4 mb-2">
                      <h6 className="small fw-bold">HITL Reviews</h6>
                      {(p.hitl_reviews || []).length === 0
                        ? <span className="text-muted small">No overrides</span>
                        : <ul className="list-unstyled mb-0">
                            {p.hitl_reviews.map((r, i) => (
                              <li key={i} className="small">
                                AI: <code>{r.original_label}</code> &#x2192; Dr: <code>{r.override_label}</code>
                                <br/><span className="text-muted">{r.reason}</span>
                              </li>
                            ))}
                          </ul>}
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* Confidence Analysis tab */}
      {tab === 'confidence' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-info text-white py-2 small fw-bold">Confidence Distribution</div>
              <div className="card-body p-2">
                {Object.entries(ov.confidence_histogram || {}).map(([bucket, cnt]) => {
                  const total = Object.values(ov.confidence_histogram).reduce((a,b) => a + b, 0) || 1;
                  return (
                    <div key={bucket} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{minWidth:'55px'}}>{bucket}</span>
                      <div className="progress flex-grow-1" style={{height:'18px'}}>
                        <div className={`progress-bar bg-${bucket.startsWith('9') || bucket.startsWith('8') ? 'success' : bucket.startsWith('7') || bucket.startsWith('6') ? 'primary' : bucket.startsWith('5') ? 'warning' : 'danger'}`}
                             style={{width: `${(cnt / total * 100).toFixed(0)}%`}}>
                          {cnt > 0 && <span style={{fontSize:'0.65rem'}}>{cnt}</span>}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">Interpretation Notes</div>
              <div className="card-body small">
                <ul className="mb-0">
                  <li><strong>&ge;80%</strong> — High confidence; suitable for auto-reporting with physician sign-off</li>
                  <li><strong>60-80%</strong> — Moderate confidence; review recommended before finalizing</li>
                  <li><strong>40-60%</strong> — Low confidence; requires full manual interpretation</li>
                  <li><strong>&lt;40%</strong> — Very low; treat as inconclusive, repeat study may be needed</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          {(defs.sections || []).map(s => (
            <div key={s.heading} className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-dark text-white py-2 small fw-bold">{s.heading}</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {(s.definitions || []).map(d => (
                      <tr key={d.term}>
                        <td className="small fw-bold" style={{width:'30%'}}>{d.term}</td>
                        <td className="small">{d.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
