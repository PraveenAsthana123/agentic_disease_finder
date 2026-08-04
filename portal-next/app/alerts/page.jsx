'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevBadge = s =>
  s === 'critical' ? 'danger' : s === 'high' ? 'warning' : s === 'medium' ? 'info' : 'secondary';

const catIcon = c =>
  c === 'seizure' ? '\u26a1' : c === 'assessment' ? '\ud83d\udccb' : c === 'vitals' ? '\u2764\ufe0f' : c === 'medication' ? '\ud83d\udc8a' : '\ud83d\udd14';

export default function AlertsDashboardPage() {
  const [data,  setData]  = useState(null);
  const [defs,  setDefs]  = useState(null);
  const [tab,   setTab]   = useState('all');
  const [sevF,  setSevF]  = useState('all');
  const [catF,  setCatF]  = useState('all');
  const [sel,   setSel]   = useState(null);

  useEffect(() => {
    fetch(`${API}/api/alerts`).then(r => r.json()).then(setData).catch(() => {});
    fetch(`${API}/api/alerts/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!data) return <div className="p-4"><div className="spinner-border text-danger" /></div>;

  const alerts = data.alerts || [];
  const filtered = alerts.filter(a =>
    (sevF === 'all' || a.severity === sevF) &&
    (catF === 'all' || a.category === catF)
  );
  const critical = alerts.filter(a => a.severity === 'critical');

  const tabs = [
    { id: 'all',         label: 'All Alerts' },
    { id: 'critical',    label: `Critical (${critical.length})` },
    { id: 'by-patient',  label: 'By Patient' },
    { id: 'definitions', label: 'Definitions' },
  ];

  /* group by patient */
  const byPatient = {};
  alerts.forEach(a => {
    if (!byPatient[a.patient_id]) byPatient[a.patient_id] = [];
    byPatient[a.patient_id].push(a);
  });

  return (
    <div>
      <h3>&#x1f6a8; Clinical Alerts Dashboard</h3>
      <p className="text-muted small">
        Real-time clinical alerts — seizure events, assessment threshold breaches,
        medication risks, and vital-sign anomalies. Sourced from seizure_diary,
        assessments, medications, and wearable_readings.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        <div className="col-6 col-md-3 mb-2">
          <div className="card text-center shadow-sm border-0">
            <div className="card-body py-2">
              <div className="h3 mb-0 text-danger">{data.total_alerts}</div>
              <div className="text-muted" style={{fontSize:'0.72rem'}}>Total Alerts</div>
            </div>
          </div>
        </div>
        <div className="col-6 col-md-3 mb-2">
          <div className="card text-center shadow-sm border-0">
            <div className="card-body py-2">
              <div className="h3 mb-0 text-danger">{data.by_severity?.critical || 0}</div>
              <div className="text-muted" style={{fontSize:'0.72rem'}}>Critical</div>
            </div>
          </div>
        </div>
        <div className="col-6 col-md-3 mb-2">
          <div className="card text-center shadow-sm border-0">
            <div className="card-body py-2">
              <div className="h3 mb-0 text-warning">{data.by_severity?.high || 0}</div>
              <div className="text-muted" style={{fontSize:'0.72rem'}}>High</div>
            </div>
          </div>
        </div>
        <div className="col-6 col-md-3 mb-2">
          <div className="card text-center shadow-sm border-0">
            <div className="card-body py-2">
              <div className="h3 mb-0 text-primary">{data.patients_affected}</div>
              <div className="text-muted" style={{fontSize:'0.72rem'}}>Patients Affected</div>
            </div>
          </div>
        </div>
      </div>

      {/* Severity & Category breakdown row */}
      <div className="row mb-3">
        <div className="col-md-6 mb-2">
          <div className="card shadow-sm border-0">
            <div className="card-header bg-danger text-white py-2 small fw-bold">By Severity</div>
            <div className="card-body p-2">
              {Object.entries(data.by_severity || {}).map(([sev, count]) => (
                <div key={sev} className="d-flex align-items-center mb-1">
                  <span className={`badge bg-${sevBadge(sev)} me-2`} style={{width:70}}>{sev}</span>
                  <div className="flex-grow-1">
                    <div className="progress" style={{height:16}}>
                      <div className={`progress-bar bg-${sevBadge(sev)}`}
                        style={{width:`${(count/data.total_alerts*100).toFixed(0)}%`}}>
                        {count}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-2">
          <div className="card shadow-sm border-0">
            <div className="card-header bg-dark text-white py-2 small fw-bold">By Category</div>
            <div className="card-body p-2">
              {Object.entries(data.by_category || {}).map(([cat, count]) => (
                <div key={cat} className="d-flex align-items-center mb-1">
                  <span className="me-2" style={{width:90,fontSize:'0.8rem'}}>{catIcon(cat)} {cat}</span>
                  <div className="flex-grow-1">
                    <div className="progress" style={{height:16}}>
                      <div className="progress-bar bg-primary" style={{width:`${(count/data.total_alerts*100).toFixed(0)}%`}}>
                        {count}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
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

      {/* All / Critical tab */}
      {(tab === 'all' || tab === 'critical') && (
        <div>
          {/* Filters */}
          <div className="d-flex gap-2 mb-2">
            <select className="form-select form-select-sm" style={{width:140}} value={sevF} onChange={e => setSevF(e.target.value)}>
              <option value="all">All Severities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
            </select>
            <select className="form-select form-select-sm" style={{width:140}} value={catF} onChange={e => setCatF(e.target.value)}>
              <option value="all">All Categories</option>
              <option value="seizure">Seizure</option>
              <option value="assessment">Assessment</option>
              <option value="vitals">Vitals</option>
              <option value="medication">Medication</option>
            </select>
            <span className="text-muted small align-self-center">{filtered.length} shown</span>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead>
                <tr>
                  <th>Severity</th>
                  <th>Category</th>
                  <th>Title</th>
                  <th>Patient</th>
                  <th>Action</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {(tab === 'critical' ? critical : filtered).map(a => (
                  <tr key={a.id} className={a.severity === 'critical' ? 'table-danger' : ''}>
                    <td><span className={`badge bg-${sevBadge(a.severity)}`}>{a.severity}</span></td>
                    <td>{catIcon(a.category)} {a.category}</td>
                    <td className="small">{a.title}</td>
                    <td><code className="small">{a.patient_id}</code></td>
                    <td className="small text-muted">{a.action_required}</td>
                    <td>
                      <button className="btn btn-outline-secondary btn-sm py-0 px-1"
                        onClick={() => setSel(sel?.id === a.id ? null : a)}>
                        {sel?.id === a.id ? '−' : '+'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Detail panel */}
          {sel && (
            <div className="card border-danger shadow-sm mb-3">
              <div className="card-header bg-danger text-white py-2 small fw-bold d-flex justify-content-between">
                <span>{sel.title}</span>
                <button className="btn btn-sm btn-outline-light py-0" onClick={() => setSel(null)}>×</button>
              </div>
              <div className="card-body small">
                <div className="row">
                  <div className="col-md-6">
                    <strong>ID:</strong> {sel.id}<br/>
                    <strong>Patient:</strong> {sel.patient_id}<br/>
                    <strong>Severity:</strong> <span className={`badge bg-${sevBadge(sel.severity)}`}>{sel.severity}</span><br/>
                    <strong>Category:</strong> {catIcon(sel.category)} {sel.category}<br/>
                    <strong>Type:</strong> {sel.type}
                  </div>
                  <div className="col-md-6">
                    <strong>Timestamp:</strong> {sel.timestamp}<br/>
                    {sel.event_date && <><strong>Event Date:</strong> {sel.event_date}<br/></>}
                    {sel.duration_sec != null && <><strong>Duration:</strong> {sel.duration_sec}s<br/></>}
                    {sel.instrument && <><strong>Instrument:</strong> {sel.instrument}<br/></>}
                    {sel.score != null && <><strong>Score:</strong> {sel.score}/{sel.max_score}<br/></>}
                    <strong>Source:</strong> {sel.source_table} #{sel.source_id}
                  </div>
                </div>
                <div className="mt-2">
                  <strong>Body:</strong> {sel.body}
                </div>
                <div className="mt-1 text-danger fw-bold">
                  Action: {sel.action_required}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* By Patient tab */}
      {tab === 'by-patient' && (
        <div className="row">
          {Object.entries(byPatient).sort((a,b) => b[1].length - a[1].length).map(([pid, pAlerts]) => (
            <div key={pid} className="col-md-6 col-lg-4 mb-3">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-dark text-white py-2 small fw-bold d-flex justify-content-between">
                  <span>{pid}</span>
                  <span className="badge bg-light text-dark">{pAlerts.length} alerts</span>
                </div>
                <div className="card-body p-2">
                  {pAlerts.slice(0, 5).map(a => (
                    <div key={a.id} className="d-flex align-items-center mb-1 small">
                      <span className={`badge bg-${sevBadge(a.severity)} me-1`} style={{fontSize:'0.65rem'}}>{a.severity}</span>
                      <span className="text-truncate">{catIcon(a.category)} {a.title}</span>
                    </div>
                  ))}
                  {pAlerts.length > 5 && <div className="text-muted small">+{pAlerts.length - 5} more</div>}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs?.definitions && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-danger text-white py-2 small fw-bold">Severity Levels</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {Object.entries(defs.definitions.severity_levels || {}).map(([k,v]) => (
                      <tr key={k}><td><span className={`badge bg-${sevBadge(k)}`}>{k}</span></td><td className="small">{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-dark text-white py-2 small fw-bold">Alert Categories</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {Object.entries(defs.definitions.categories || {}).map(([k,v]) => (
                      <tr key={k}><td>{catIcon(k)} <strong>{k}</strong></td><td className="small">{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          {defs.definitions.thresholds && (
            <div className="col-12">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-warning text-dark py-2 small fw-bold">Alert Thresholds</div>
                <div className="card-body p-2">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Instrument</th><th>Threshold</th><th>Direction</th><th>Label</th></tr></thead>
                    <tbody>
                      {Object.entries(defs.definitions.thresholds).map(([k,v]) => (
                        <tr key={k}><td><strong>{k}</strong></td><td>{v.threshold}</td><td>{v.direction}</td><td className="small">{v.label}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
