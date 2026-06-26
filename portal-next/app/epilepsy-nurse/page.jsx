'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8000';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'seizure-diary', label: 'Seizure Diary' },
  { id: 'adherence', label: 'Adherence' },
  { id: 'sudep-risk', label: 'SUDEP Risk' },
  { id: 'action-plan', label: 'Action Plan' },
  { id: 'education', label: 'Education' },
  { id: 'definitions', label: 'Definitions' },
];

const ENDPOINTS = {
  overview: '/api/epilepsy-nurse',
  'seizure-diary': '/api/epilepsy-nurse/seizure-diary',
  adherence: '/api/epilepsy-nurse/adherence',
  'sudep-risk': '/api/epilepsy-nurse/sudep-risk',
  'action-plan': '/api/epilepsy-nurse/action-plan',
  education: '/api/epilepsy-nurse/education',
  definitions: '/api/epilepsy-nurse/definitions',
};

function Badge({ level, text }) {
  const m = { High: 'danger', Moderate: 'warning', Low: 'success', red: 'danger', amber: 'warning', green: 'success', urgent: 'danger', recommended: 'info', standard: 'secondary' };
  return <span className={`badge bg-${m[level] || m[text] || 'secondary'}`}>{text || level}</span>;
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-3 mb-2">
      <div className="card shadow-sm h-100"><div className="card-body text-center">
        <div className={`h4 mb-1 text-${color || 'primary'}`}>{value ?? '—'}</div>
        <div className="text-muted small">{label}</div>
      </div></div>
    </div>
  );
}

// ─── Sub-tab renderers ──────────────────────────────────────────────

function OverviewPanel({ data }) {
  if (!data || data.error) return <div className="alert alert-warning">{data?.error || 'No data'}</div>;
  const s = data.summary || {};
  return (<div>
    <div className="row mb-3">
      <KPI label="Total Patients" value={s.total_patients} color="primary" />
      <KPI label="Seizure Diary Entries" value={s.total_seizure_diary_entries} color="info" />
      <KPI label="High SUDEP Risk" value={s.high_sudep_risk} color="danger" />
      <KPI label="High Adherence Risk" value={s.high_adherence_risk} color="warning" />
    </div>
    <div className="row mb-3">
      <KPI label="Patients with Diary" value={s.patients_with_diary} color="info" />
      <KPI label="Status Epilepticus Risk" value={s.patients_with_status_risk} color="danger" />
      <KPI label="Injury History" value={s.patients_with_injury_history} color="warning" />
      <KPI label="ER Visits" value={s.total_er_visits} color="danger" />
    </div>
    <p className="text-muted small">{data.subtitle}</p>
  </div>);
}

function SeizureDiaryPanel({ data }) {
  if (!data || data.error) return <div className="alert alert-warning">{data?.error || 'No data'}</div>;
  const pts = data.patients || [];
  if (!pts.length) return <div className="alert alert-info">No seizure diary entries found in clinical.db.</div>;
  return (<div>
    <div className="row mb-3">
      <KPI label="Total Entries" value={data.total_entries} color="primary" />
      <KPI label="Patients" value={data.patients_with_diary} color="info" />
      <KPI label="ER Visits" value={data.total_er_visits} color="danger" />
      <KPI label="Nocturnal Events" value={data.total_nocturnal_events} color="warning" />
    </div>
    {data.aggregate_severity && Object.keys(data.aggregate_severity).length > 0 && (
      <div className="card mb-3"><div className="card-body">
        <h6>Severity Distribution (all patients)</h6>
        <div className="d-flex gap-3 flex-wrap">
          {Object.entries(data.aggregate_severity).map(([k, v]) => (
            <span key={k} className={`badge bg-${k === 'Severe' ? 'danger' : k === 'Moderate' ? 'warning' : k === 'Mild' ? 'success' : 'secondary'} fs-6`}>
              {k}: {v}
            </span>
          ))}
        </div>
      </div></div>
    )}
    <div className="table-responsive"><table className="table table-sm table-striped">
      <thead><tr>
        <th>Patient</th><th>Seizures</th><th>Monthly Freq</th><th>Avg Duration (s)</th>
        <th>Injuries</th><th>ER Visits</th><th>Nocturnal</th><th>Days Since Last</th><th>Free 90d?</th>
      </tr></thead>
      <tbody>{pts.map(p => (
        <tr key={p.patient_id}>
          <td className="fw-bold">{p.patient_id}</td>
          <td>{p.total_seizures}</td>
          <td>{p.monthly_frequency}</td>
          <td>{p.average_duration_sec}</td>
          <td>{p.injury_rate_pct}%</td>
          <td>{p.er_visits}</td>
          <td>{p.nocturnal_events}</td>
          <td>{p.days_since_last_seizure}</td>
          <td>{p.seizure_free_90_days ? <span className="badge bg-success">Yes</span> : <span className="badge bg-secondary">No</span>}</td>
        </tr>
      ))}</tbody>
    </table></div>
  </div>);
}

function AdherencePanel({ data }) {
  if (!data || data.error) return <div className="alert alert-warning">{data?.error || 'No data'}</div>;
  const pts = data.patients || [];
  if (!pts.length) return <div className="alert alert-info">No medication data found.</div>;
  return (<div>
    <div className="row mb-3">
      <KPI label="Total Patients" value={data.total_patients} color="primary" />
      <KPI label="High Risk" value={data.high_risk_count} color="danger" />
      <KPI label="Moderate Risk" value={data.moderate_risk_count} color="warning" />
      <KPI label="Low Risk" value={data.low_risk_count} color="success" />
    </div>
    <div className="table-responsive"><table className="table table-sm table-striped">
      <thead><tr>
        <th>Patient</th><th>AEDs</th><th>Drug Names</th><th>Daily Doses</th>
        <th>Complexity</th><th>Adherence Risk</th><th>Seizures</th>
      </tr></thead>
      <tbody>{pts.map(p => (
        <tr key={p.patient_id}>
          <td className="fw-bold">{p.patient_id}</td>
          <td>{p.aed_count}</td>
          <td><small>{(p.aed_names || []).join(', ') || '—'}</small></td>
          <td>{p.total_daily_doses}</td>
          <td>{p.complexity_score}/10</td>
          <td><Badge level={p.risk_color} text={p.adherence_risk} /></td>
          <td>{p.seizure_count}</td>
        </tr>
      ))}</tbody>
    </table></div>
    {pts.filter(p => p.recommendations?.length).map(p => (
      <div key={p.patient_id} className="card mb-2"><div className="card-body">
        <h6>{p.patient_id} — Recommendations</h6>
        <ul className="mb-0">{p.recommendations.map((r, i) => <li key={i} className="small">{r}</li>)}</ul>
      </div></div>
    ))}
  </div>);
}

function SUDEPPanel({ data }) {
  if (!data || data.error) return <div className="alert alert-warning">{data?.error || 'No data'}</div>;
  const pts = data.patients || [];
  if (!pts.length) return <div className="alert alert-info">No patient data for SUDEP assessment.</div>;
  return (<div>
    <div className="row mb-3">
      <KPI label="Total Patients" value={data.total_patients} color="primary" />
      <KPI label="High Risk" value={data.high_risk_count} color="danger" />
      <KPI label="Moderate Risk" value={data.moderate_risk_count} color="warning" />
      <KPI label="Low Risk" value={data.low_risk_count} color="success" />
    </div>
    <div className="table-responsive"><table className="table table-sm table-striped">
      <thead><tr>
        <th>Patient</th><th>Age</th><th>Gender</th><th>SUDEP Score</th>
        <th>Risk Level</th><th>AEDs</th><th>Seizures</th><th>Action</th>
      </tr></thead>
      <tbody>{pts.map(p => (
        <tr key={p.patient_id}>
          <td className="fw-bold">{p.patient_id}</td>
          <td>{p.age ?? '—'}</td>
          <td>{p.gender || '—'}</td>
          <td><strong>{p.sudep_score}</strong>/10</td>
          <td><Badge level={p.risk_color} text={p.risk_level} /></td>
          <td>{p.aed_count}</td>
          <td>{p.seizure_count}</td>
          <td><small>{p.recommended_action}</small></td>
        </tr>
      ))}</tbody>
    </table></div>
    {/* Risk factors legend */}
    {data.risk_factors_legend && (
      <div className="card mt-3"><div className="card-body">
        <h6>SUDEP-7 Risk Factor Weights</h6>
        <div className="table-responsive"><table className="table table-sm">
          <thead><tr><th>Factor</th><th>Weight</th><th>Reference</th></tr></thead>
          <tbody>{Object.entries(data.risk_factors_legend).map(([k, v]) => (
            <tr key={k}>
              <td>{v.description}</td>
              <td><strong>{v.weight}</strong></td>
              <td><small className="text-muted">{v.reference}</small></td>
            </tr>
          ))}</tbody>
        </table></div>
      </div></div>
    )}
  </div>);
}

function ActionPlanPanel({ data }) {
  if (!data || data.error) return <div className="alert alert-warning">{data?.error || 'No data'}</div>;
  const pts = data.patients || [];
  if (!pts.length) return <div className="alert alert-info">No patient data for action plans.</div>;
  return (<div>
    <div className="row mb-3">
      <KPI label="Patients" value={data.total_patients} color="primary" />
      <KPI label="Status Risk" value={data.patients_with_status_risk} color="danger" />
      <KPI label="Injury History" value={data.patients_with_injury_history} color="warning" />
    </div>
    {pts.map(p => (
      <div key={p.patient_id} className="card mb-3"><div className="card-body">
        <h5>{p.patient_id} {p.patient_name ? `— ${p.patient_name}` : ''}</h5>
        <div className="d-flex gap-2 mb-2 flex-wrap">
          <span className="badge bg-info">Age: {p.age ?? '—'}</span>
          <span className="badge bg-info">AEDs: {(p.current_aeds || []).join(', ') || 'None'}</span>
          <span className="badge bg-info">Seizures: {p.total_recorded_seizures}</span>
          {p.status_epilepticus_risk && <span className="badge bg-danger">STATUS EPILEPTICUS RISK</span>}
          {p.has_injury_history && <span className="badge bg-warning text-dark">INJURY HISTORY</span>}
        </div>
        {p.safety_alerts?.length > 0 && (
          <div className="alert alert-danger py-1 mb-2">
            {p.safety_alerts.map((a, i) => <div key={i} className="small fw-bold">{a}</div>)}
          </div>
        )}
        <h6>First Aid Steps</h6>
        <ol className="small mb-2">{p.first_aid_steps?.map((s, i) => <li key={i}>{s}</li>)}</ol>
        <h6>When to Call Emergency Services</h6>
        <ul className="small mb-2">{p.emergency_criteria?.map((c, i) => <li key={i}>{c}</li>)}</ul>
        {p.rescue_medications?.length > 0 && (<>
          <h6>Rescue Medications</h6>
          <div className="table-responsive"><table className="table table-sm">
            <thead><tr><th>Medication</th><th>Dose</th><th>When</th></tr></thead>
            <tbody>{p.rescue_medications.map((m, i) => (
              <tr key={i}><td className="fw-bold">{m.medication}</td><td>{m.dose_guidance}</td><td>{m.when_to_give}</td></tr>
            ))}</tbody>
          </table></div>
        </>)}
        {p.recovery_guidance && (<>
          <h6>Recovery</h6>
          <ul className="small mb-0">
            <li><strong>Position:</strong> {p.recovery_guidance.recovery_position}</li>
            <li><strong>Monitoring:</strong> {p.recovery_guidance.post_ictal_monitoring}</li>
            <li><strong>Resume activity:</strong> {p.recovery_guidance.when_to_resume_normal_activity}</li>
          </ul>
        </>)}
      </div></div>
    ))}
  </div>);
}

function EducationPanel({ data }) {
  if (!data || data.error) return <div className="alert alert-warning">{data?.error || 'No data'}</div>;
  const pts = data.patients || [];
  if (!pts.length) return <div className="alert alert-info">No patient data for education assessment.</div>;
  return (<div>
    <div className="row mb-3">
      <KPI label="Patients" value={data.total_patients} color="primary" />
      <KPI label="Education Domains" value={data.education_domains_catalog?.length || 12} color="info" />
    </div>
    {pts.map(p => (
      <div key={p.patient_id} className="card mb-3"><div className="card-body">
        <h5>{p.patient_id}
          <span className="badge bg-danger ms-2">{p.urgent_domains} urgent</span>
          <span className="badge bg-info ms-1">{p.recommended_domains} recommended</span>
          <span className="badge bg-secondary ms-1">{p.standard_domains} standard</span>
        </h5>
        <div className="table-responsive"><table className="table table-sm">
          <thead><tr><th>Domain</th><th>Priority</th><th>Risk</th><th>Notes</th></tr></thead>
          <tbody>{(p.education_domains || []).map((d, i) => (
            <tr key={i}>
              <td className="fw-bold">{d.domain}</td>
              <td><Badge level={d.priority_for_patient} text={d.priority_for_patient} /></td>
              <td><Badge level={d.risk_level} text={d.risk_level} /></td>
              <td><small>{d.patient_specific_notes?.join('; ') || '—'}</small></td>
            </tr>
          ))}</tbody>
        </table></div>
      </div></div>
    ))}
  </div>);
}

function DefinitionsPanel({ data }) {
  if (!data || data.error) return <div className="alert alert-warning">{data?.error || 'No data'}</div>;
  const mods = data.modules || {};
  return (<div>
    <h5>{data.title}</h5>
    {Object.entries(mods).map(([key, mod]) => (
      <div key={key} className="card mb-3"><div className="card-body">
        <h6 className="text-primary">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h6>
        <p className="small text-muted mb-1">{mod.purpose}</p>
        <p className="small mb-1"><strong>Data source:</strong> {mod.data_source}</p>
        {mod.key_metrics && (
          <ul className="small mb-1">{Object.entries(mod.key_metrics).map(([mk, mv]) => (
            <li key={mk}><strong>{mk}:</strong> {mv}</li>
          ))}</ul>
        )}
        {mod.references && (
          <div className="small text-muted">
            <strong>References:</strong>
            <ul className="mb-0">{(Array.isArray(mod.references) ? mod.references : [mod.reference]).filter(Boolean).map((r, i) => <li key={i}>{r}</li>)}</ul>
          </div>
        )}
        {mod.reference && !mod.references && (
          <p className="small text-muted mb-0"><strong>Reference:</strong> {mod.reference}</p>
        )}
      </div></div>
    ))}
  </div>);
}

// ─── Main page ──────────────────────────────────────────────────────

export default function EpilepsyNursePage() {
  const [tab, setTab] = useState('overview');
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [patientId, setPatientId] = useState('');
  const [filterPid, setFilterPid] = useState('');

  function loadTab(t, pid) {
    const ep = ENDPOINTS[t];
    if (!ep) return;
    const url = pid ? `${API}${ep}?patient_id=${pid}` : `${API}${ep}`;
    setLoading(true);
    setErr(null);
    fetch(url).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(d => { setData(prev => ({ ...prev, [t]: d })); setLoading(false); })
      .catch(e => { setErr(`${t}: ${e.message}`); setLoading(false); });
  }

  useEffect(() => { loadTab('overview', filterPid); }, [filterPid]);

  function switchTab(t) {
    setTab(t);
    if (!data[t]) loadTab(t, filterPid);
  }

  function handleFilter(e) {
    e.preventDefault();
    setFilterPid(patientId.trim());
    setData({});
    setTab('overview');
  }

  return (
    <div className="container-fluid p-3">
      <h3 className="mb-1">Epilepsy Specialist Nurse</h3>
      <p className="text-muted mb-3">Seizure diary, AED adherence, SUDEP risk, action plans, patient education — all from real clinical.db data</p>

      {/* Patient filter */}
      <form className="row g-2 mb-3 align-items-end" onSubmit={handleFilter}>
        <div className="col-auto">
          <label className="form-label small mb-0">Filter by Patient ID</label>
          <input className="form-control form-control-sm" placeholder="e.g. P0001" value={patientId} onChange={e => setPatientId(e.target.value)} />
        </div>
        <div className="col-auto">
          <button className="btn btn-sm btn-primary" type="submit">Filter</button>
          {filterPid && <button className="btn btn-sm btn-outline-secondary ms-1" type="button" onClick={() => { setPatientId(''); setFilterPid(''); setData({}); }}>Clear</button>}
        </div>
        {filterPid && <div className="col-auto"><span className="badge bg-info">Filtered: {filterPid}</span></div>}
      </form>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => switchTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {err && <div className="alert alert-danger">{err}</div>}
      {loading && <div className="text-muted">Loading...</div>}

      {/* Tab content */}
      {!loading && tab === 'overview' && <OverviewPanel data={data.overview} />}
      {!loading && tab === 'seizure-diary' && <SeizureDiaryPanel data={data['seizure-diary']} />}
      {!loading && tab === 'adherence' && <AdherencePanel data={data.adherence} />}
      {!loading && tab === 'sudep-risk' && <SUDEPPanel data={data['sudep-risk']} />}
      {!loading && tab === 'action-plan' && <ActionPlanPanel data={data['action-plan']} />}
      {!loading && tab === 'education' && <EducationPanel data={data.education} />}
      {!loading && tab === 'definitions' && <DefinitionsPanel data={data.definitions} />}
    </div>
  );
}
