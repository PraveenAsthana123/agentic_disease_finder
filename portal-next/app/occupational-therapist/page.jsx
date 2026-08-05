'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

// Barthel Index items with max scores (standard 0-10 each, total 100)
const BARTHEL_ITEMS = [
  { key: 'Feeding',          max: 10 },
  { key: 'Bathing',          max: 5  },
  { key: 'Grooming',         max: 5  },
  { key: 'Dressing',         max: 10 },
  { key: 'Bowel Control',    max: 10 },
  { key: 'Bladder Control',  max: 10 },
  { key: 'Toilet Use',       max: 10 },
  { key: 'Transfers',        max: 15 },
  { key: 'Mobility',         max: 15 },
  { key: 'Stairs',           max: 10 },
];
const BARTHEL_MAX = BARTHEL_ITEMS.reduce((s, i) => s + i.max, 0); // 100

const QOLIE_DOMAINS = [
  { key: 'Seizure Worry',        max: 100 },
  { key: 'Overall QoL',          max: 100 },
  { key: 'Emotional Well-being', max: 100 },
  { key: 'Energy/Fatigue',       max: 100 },
  { key: 'Cognitive Function',   max: 100 },
  { key: 'Medication Effects',   max: 100 },
  { key: 'Social Function',      max: 100 },
];
const QOLIE_MAX = 100;

function barthelInterpret(score) {
  if (score >= 91) return { interp: 'Slight Dependence', level: 'success' };
  if (score >= 61) return { interp: 'Moderate Dependence', level: 'warning' };
  if (score >= 21) return { interp: 'Severe Dependence', level: 'warning' };
  return { interp: 'Total Dependence', level: 'danger' };
}
function qolieInterpret(score) {
  if (score >= 70) return { interp: 'Good QoL', level: 'success' };
  if (score >= 50) return { interp: 'Moderate QoL', level: 'primary' };
  if (score >= 30) return { interp: 'Poor QoL', level: 'warning' };
  return { interp: 'Very Poor QoL', level: 'danger' };
}

const barthelColor = v =>
  v >= 91 ? 'success' : v >= 61 ? 'primary' : v >= 21 ? 'warning' : 'danger';
const qolColor = v =>
  v >= 70 ? 'success' : v >= 50 ? 'primary' : v >= 30 ? 'warning' : 'danger';
const sevColor = s =>
  s === 'low' ? 'success' : s === 'moderate' ? 'warning' : 'danger';

const EMPTY_FORM = {
  patient_id: '',
  instrument: 'BARTHEL',
  examiner: 'OT',
  barthel: Object.fromEntries(BARTHEL_ITEMS.map(i => [i.key, 0])),
  qolie: Object.fromEntries(QOLIE_DOMAINS.map(d => [d.key, 50])),
};

export default function OccupationalTherapistDashboardPage() {
  const [ov,      setOv]      = useState(null);
  const [bd,      setBd]      = useState(null);
  const [defs,    setDefs]    = useState(null);
  const [history, setHistory] = useState(null);
  const [tab,     setTab]     = useState('overview');
  const [sel,     setSel]     = useState(null);
  const [form,    setForm]    = useState(EMPTY_FORM);
  const [saving,  setSaving]  = useState(false);
  const [saveMsg, setSaveMsg] = useState(null);

  const loadHistory = () =>
    fetch(`${API}/api/occupational-therapist/history?limit=40`)
      .then(r => r.json()).then(setHistory).catch(() => {});

  useEffect(() => {
    fetch(`${API}/api/occupational-therapist/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/occupational-therapist/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/occupational-therapist/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
    loadHistory();
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpi = ov.kpis || {};
  const kpiCards = [
    { label: 'Total Patients',      value: kpi.total_patients,      color: 'primary' },
    { label: 'Barthel Assessments', value: kpi.barthel_assessments, color: 'info' },
    { label: 'QoLIE Assessments',   value: kpi.qolie_assessments,   color: 'info' },
    { label: 'Barthel Impaired',    value: kpi.barthel_impaired,    color: 'warning' },
    { label: 'Avg Barthel',         value: kpi.avg_barthel,         color: barthelColor(kpi.avg_barthel), unit: '/100' },
    { label: 'Avg QoLIE',           value: kpi.avg_qolie,           color: qolColor(kpi.avg_qolie),      unit: '/100' },
    { label: 'ESS Elevated',        value: kpi.ess_elevated,        color: kpi.ess_elevated > 5 ? 'danger' : 'warning' },
    { label: 'QoLIE Poor',          value: kpi.qolie_poor,          color: kpi.qolie_poor > 5 ? 'danger' : 'warning' },
  ];

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'patients',    label: 'Patient Profiles' },
    { id: 'rehab',       label: 'Rehab Candidates' },
    { id: 'medications', label: 'AED Risk' },
    { id: 'record',      label: 'Record Assessment' },
    { id: 'history',     label: 'Score History' },
    { id: 'definitions', label: 'Clinical Definitions' },
  ];

  const handleSave = async e => {
    e.preventDefault();
    if (!form.patient_id.trim()) { setSaveMsg({ ok: false, msg: 'Patient ID is required.' }); return; }
    setSaving(true); setSaveMsg(null);
    let answers_json = {};
    let score = 0;
    let max_score = 0;
    let interpretation = '';
    let level = '';
    if (form.instrument === 'BARTHEL') {
      answers_json = form.barthel;
      score = Object.values(form.barthel).reduce((s, v) => s + Number(v), 0);
      max_score = BARTHEL_MAX;
      const interp = barthelInterpret(score);
      interpretation = interp.interp; level = interp.level;
    } else {
      answers_json = form.qolie;
      score = Math.round(Object.values(form.qolie).reduce((s, v) => s + Number(v), 0) / QOLIE_DOMAINS.length);
      max_score = QOLIE_MAX;
      const interp = qolieInterpret(score);
      interpretation = interp.interp; level = interp.level;
    }
    try {
      const res = await fetch(`${API}/api/occupational-therapist/save-assessment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: form.patient_id.trim(),
          instrument: form.instrument,
          answers_json, score, max_score, interpretation, level,
          examiner: form.examiner || 'OT',
        }),
      });
      const data = await res.json();
      if (data.ok) {
        setSaveMsg({ ok: true, msg: `Saved: ${data.instrument} = ${data.score}/${max_score} (${interpretation})` });
        setForm(EMPTY_FORM);
        loadHistory();
      } else {
        setSaveMsg({ ok: false, msg: data.detail || 'Save failed.' });
      }
    } catch {
      setSaveMsg({ ok: false, msg: 'Network error.' });
    } finally {
      setSaving(false);
    }
  };

  /* helper: distribution bar table */
  const DistTable = ({ title, bg, data, labelKey }) => {
    const max = Math.max(...data.map(d => d.count), 1);
    return (
      <div className="card shadow-sm border-0 mb-3">
        <div className={`card-header bg-${bg} text-white py-2 small fw-bold`}>{title}</div>
        <div className="card-body p-2">
          <table className="table table-sm table-hover mb-0">
            <thead><tr><th>Level</th><th className="text-end">Count</th><th style={{width:'40%'}}>Bar</th></tr></thead>
            <tbody>
              {data.map(d => (
                <tr key={d[labelKey]}>
                  <td className="small">{d[labelKey]}</td>
                  <td className="text-end small fw-bold">{d.count}</td>
                  <td>
                    <div className="progress" style={{height:'14px'}}>
                      <div className={`progress-bar bg-${bg}`} style={{width:`${(d.count/max*100).toFixed(0)}%`}} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  /* helper: patient table row */
  const PatientTable = ({ patients, showGoals }) => (
    <div className="table-responsive">
      <table className="table table-sm table-hover">
        <thead className="table-dark">
          <tr>
            <th>Patient</th><th>Age</th><th>Gender</th><th>Disease</th>
            <th className="text-end">Barthel</th><th>Level</th>
            <th className="text-end">QoLIE</th><th>Level</th>
            <th className="text-end">ESS</th><th className="text-end">Seizures</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {patients.map(p => (
            <tr key={p.patient_id} className={sel === p.patient_id ? 'table-active' : ''}>
              <td className="small fw-bold">{p.patient_id}</td>
              <td className="small">{p.age ?? '—'}</td>
              <td className="small">{p.gender || '—'}</td>
              <td className="small">{p.disease || '—'}</td>
              <td className={`text-end small text-${barthelColor(p.barthel_score)}`}>
                {p.barthel_score != null ? p.barthel_score : '—'}
              </td>
              <td className="small">
                {p.barthel_level && (
                  <span className={`badge bg-${barthelColor(p.barthel_score)}`}>{p.barthel_level}</span>
                )}
              </td>
              <td className={`text-end small text-${qolColor(p.qolie_score)}`}>
                {p.qolie_score != null ? p.qolie_score : '—'}
              </td>
              <td className="small">
                {p.qolie_level && (
                  <span className={`badge bg-${qolColor(p.qolie_score)}`}>{p.qolie_level}</span>
                )}
              </td>
              <td className="text-end small">{p.epworth_score ?? '—'}</td>
              <td className="text-end small">{p.seizure_count ?? '—'}</td>
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

      {/* Patient detail panel */}
      {sel && (() => {
        const p = patients.find(x => x.patient_id === sel);
        if (!p) return null;
        return (
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-header bg-primary text-white py-2 small fw-bold">
              Patient Detail: {p.patient_id} — {p.name}
            </div>
            <div className="card-body p-3">
              <div className="row">
                {/* Barthel items */}
                <div className="col-md-4 mb-2">
                  <h6 className="small fw-bold">Barthel ADL Items</h6>
                  {(p.barthel_items || []).length === 0
                    ? <span className="text-muted small">No data</span>
                    : <ul className="list-unstyled mb-0">
                        {p.barthel_items.map((it, i) => (
                          <li key={i} className="small">
                            <strong>{it.item || it.name}</strong>: {it.score ?? it.value ?? '—'}
                          </li>
                        ))}
                      </ul>}
                </div>
                {/* QoLIE domains */}
                <div className="col-md-4 mb-2">
                  <h6 className="small fw-bold">QoLIE-31 Domains</h6>
                  {(p.qolie_domains || []).length === 0
                    ? <span className="text-muted small">No data</span>
                    : <ul className="list-unstyled mb-0">
                        {p.qolie_domains.map((d, i) => (
                          <li key={i} className="small">
                            <strong>{d.domain || d.name}</strong>: {d.score ?? d.value ?? '—'}
                          </li>
                        ))}
                      </ul>}
                </div>
                {/* Suggested goals + risk factors */}
                <div className="col-md-4 mb-2">
                  <h6 className="small fw-bold">Suggested Rehab Goals</h6>
                  {(p.suggested_goals || []).length === 0
                    ? <span className="text-muted small">None</span>
                    : <ul className="list-unstyled mb-0">
                        {p.suggested_goals.map((g, i) => (
                          <li key={i} className="small">{g}</li>
                        ))}
                      </ul>}
                  {(p.risk_factors || []).length > 0 && (
                    <>
                      <h6 className="small fw-bold mt-2">Risk Factors</h6>
                      <ul className="list-unstyled mb-0">
                        {p.risk_factors.map((r, i) => (
                          <li key={i} className="small text-danger">{r}</li>
                        ))}
                      </ul>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );

  return (
    <div>
      <h3>&#x1f590;&#xfe0f; Occupational Therapist Dashboard</h3>
      <p className="text-muted small">
        Functional independence assessment — Barthel Index, QoLIE-31, Epworth Sleepiness Scale,
        Liverpool Seizure Severity Scale, AED functional risk, and rehabilitation candidates.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {kpiCards.map(k => (
          <div key={k.label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${k.color}`}>
                  {k.value != null ? (typeof k.value === 'number' && !Number.isInteger(k.value)
                    ? k.value.toFixed(1) : k.value) : '—'}{k.unit || ''}
                </div>
                <div className="text-muted" style={{fontSize:'0.72rem'}}>{k.label}</div>
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
          <div className="col-md-6">
            <DistTable title="Barthel Index Distribution" bg="primary" data={ov.barthel_distribution || []} labelKey="level" />
          </div>
          <div className="col-md-6">
            <DistTable title="QoLIE-31 Distribution" bg="info" data={ov.qolie_distribution || []} labelKey="level" />
          </div>
          <div className="col-md-6">
            <DistTable title="Epworth Sleepiness Scale" bg="warning" data={ov.ess_distribution || []} labelKey="level" />
          </div>
          <div className="col-md-6">
            <DistTable title="Liverpool Seizure Severity" bg="danger" data={ov.lsss_distribution || []} labelKey="level" />
          </div>

          {/* Barthel ADL items */}
          {(ov.barthel_items || []).length > 0 && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-dark text-white py-2 small fw-bold">Barthel ADL Item Averages</div>
                <div className="card-body p-2">
                  {ov.barthel_items.map(it => {
                    const pct = (it.avg_score / 10 * 100).toFixed(0);
                    return (
                      <div key={it.item} className="d-flex align-items-center mb-1">
                        <span className="small me-2" style={{minWidth:'90px'}}>{it.item}</span>
                        <div className="progress flex-grow-1" style={{height:'16px'}}>
                          <div className={`progress-bar bg-${barthelColor(it.avg_score * 10)}`}
                               style={{width:`${pct}%`}}>
                            <span style={{fontSize:'0.65rem'}}>{it.avg_score.toFixed(1)}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* QoLIE domains */}
          {(ov.qolie_domains || []).length > 0 && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-secondary text-white py-2 small fw-bold">QoLIE-31 Domain Averages</div>
                <div className="card-body p-2">
                  {ov.qolie_domains.map(d => (
                    <div key={d.domain} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{minWidth:'120px'}}>{d.domain}</span>
                      <div className="progress flex-grow-1" style={{height:'16px'}}>
                        <div className={`progress-bar bg-${qolColor(d.avg_score)}`}
                             style={{width:`${d.avg_score}%`}}>
                          <span style={{fontSize:'0.65rem'}}>{d.avg_score.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Patient Profiles tab */}
      {tab === 'patients' && bd && (
        <PatientTable patients={bd.profiles || []} showGoals={false} />
      )}

      {/* Rehab Candidates tab */}
      {tab === 'rehab' && bd && (
        <div>
          <div className="alert alert-warning small mb-2">
            <strong>Rehab Candidates:</strong> Patients with Barthel &lt; 91 (functional impairment) flagged for OT evaluation.
            {bd.goal_domains && bd.goal_domains.length > 0 && (
              <span> Goal domains: {bd.goal_domains.join(', ')}.</span>
            )}
          </div>
          <PatientTable patients={bd.rehab_candidates || []} showGoals={true} />
        </div>
      )}

      {/* AED Functional Risk tab */}
      {tab === 'medications' && (
        <div className="row">
          <div className="col-md-8">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-dark text-white py-2 small fw-bold">AED Functional Side-Effect Risk</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Drug</th><th>Severity</th><th>Functional Effects</th></tr></thead>
                  <tbody>
                    {(ov.aed_functional_risk || []).map(d => (
                      <tr key={d.drug}>
                        <td className="small fw-bold">{d.drug}</td>
                        <td><span className={`badge bg-${sevColor(d.severity)}`}>{d.severity}</span></td>
                        <td className="small">{(d.effects || []).join(', ')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">OT Interpretation</div>
              <div className="card-body small">
                <ul className="mb-0">
                  <li><span className="badge bg-success">low</span> — Minimal functional impact; standard rehab protocols</li>
                  <li><span className="badge bg-warning">moderate</span> — May affect balance, coordination, or energy; adapt plan</li>
                  <li><span className="badge bg-danger">high</span> — Significant functional impairment risk; close monitoring</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Record Assessment tab */}
      {tab === 'record' && (
        <div className="row">
          <div className="col-md-8">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-success text-white py-2 small fw-bold">
                Record Outcome Assessment
              </div>
              <div className="card-body p-3">
                <form onSubmit={handleSave}>
                  <div className="row mb-2">
                    <div className="col-md-4">
                      <label className="form-label small fw-bold">Patient ID</label>
                      <input className="form-control form-control-sm" placeholder="e.g. P0001"
                             value={form.patient_id}
                             onChange={e => setForm(f => ({ ...f, patient_id: e.target.value }))} />
                    </div>
                    <div className="col-md-4">
                      <label className="form-label small fw-bold">Instrument</label>
                      <select className="form-select form-select-sm"
                              value={form.instrument}
                              onChange={e => setForm(f => ({ ...f, instrument: e.target.value }))}>
                        <option value="BARTHEL">Barthel Index (ADL)</option>
                        <option value="QOLIE31">QOLIE-31 (Quality of Life)</option>
                      </select>
                    </div>
                    <div className="col-md-4">
                      <label className="form-label small fw-bold">Examiner</label>
                      <input className="form-control form-control-sm" placeholder="OT"
                             value={form.examiner}
                             onChange={e => setForm(f => ({ ...f, examiner: e.target.value }))} />
                    </div>
                  </div>

                  {/* Barthel items */}
                  {form.instrument === 'BARTHEL' && (
                    <div className="mb-3">
                      <p className="small text-muted mb-1">Score each ADL item (0 to max):</p>
                      {BARTHEL_ITEMS.map(item => (
                        <div key={item.key} className="d-flex align-items-center mb-1">
                          <span className="small me-2" style={{ minWidth: '130px' }}>{item.key}</span>
                          <input type="number" min="0" max={item.max}
                                 className="form-control form-control-sm me-2" style={{ width: '70px' }}
                                 value={form.barthel[item.key]}
                                 onChange={e => setForm(f => ({
                                   ...f, barthel: { ...f.barthel, [item.key]: Math.min(item.max, Math.max(0, Number(e.target.value))) }
                                 }))} />
                          <span className="text-muted small">/ {item.max}</span>
                        </div>
                      ))}
                      <div className="mt-2 fw-bold small">
                        Total: {Object.values(form.barthel).reduce((s, v) => s + Number(v), 0)} / {BARTHEL_MAX}
                        {' '}— {barthelInterpret(Object.values(form.barthel).reduce((s, v) => s + Number(v), 0)).interp}
                      </div>
                    </div>
                  )}

                  {/* QOLIE-31 domains */}
                  {form.instrument === 'QOLIE31' && (
                    <div className="mb-3">
                      <p className="small text-muted mb-1">Score each domain (0–100):</p>
                      {QOLIE_DOMAINS.map(d => (
                        <div key={d.key} className="d-flex align-items-center mb-1">
                          <span className="small me-2" style={{ minWidth: '160px' }}>{d.key}</span>
                          <input type="number" min="0" max="100"
                                 className="form-control form-control-sm me-2" style={{ width: '80px' }}
                                 value={form.qolie[d.key]}
                                 onChange={e => setForm(f => ({
                                   ...f, qolie: { ...f.qolie, [d.key]: Math.min(100, Math.max(0, Number(e.target.value))) }
                                 }))} />
                          <div className="progress flex-grow-1" style={{ height: '12px' }}>
                            <div className="progress-bar bg-info"
                                 style={{ width: `${form.qolie[d.key]}%` }} />
                          </div>
                        </div>
                      ))}
                      <div className="mt-2 fw-bold small">
                        Average: {Math.round(Object.values(form.qolie).reduce((s, v) => s + Number(v), 0) / QOLIE_DOMAINS.length)} / 100
                        {' '}— {qolieInterpret(Math.round(Object.values(form.qolie).reduce((s, v) => s + Number(v), 0) / QOLIE_DOMAINS.length)).interp}
                      </div>
                    </div>
                  )}

                  {saveMsg && (
                    <div className={`alert alert-${saveMsg.ok ? 'success' : 'danger'} py-2 small mb-2`}>
                      {saveMsg.msg}
                    </div>
                  )}
                  <button type="submit" className="btn btn-success btn-sm" disabled={saving}>
                    {saving ? 'Saving…' : 'Save Assessment'}
                  </button>
                </form>
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-info text-white py-2 small fw-bold">Scoring Reference</div>
              <div className="card-body small">
                <p className="fw-bold mb-1">Barthel Index (0–100)</p>
                <ul className="mb-2">
                  <li>91–100: Slight Dependence</li>
                  <li>61–90: Moderate Dependence</li>
                  <li>21–60: Severe Dependence</li>
                  <li>0–20: Total Dependence</li>
                </ul>
                <p className="fw-bold mb-1">QOLIE-31 (0–100)</p>
                <ul className="mb-0">
                  <li>≥70: Good QoL</li>
                  <li>50–69: Moderate QoL</li>
                  <li>30–49: Poor QoL</li>
                  <li>&lt;30: Very Poor QoL</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Score History tab */}
      {tab === 'history' && (
        <div>
          <div className="d-flex justify-content-between align-items-center mb-2">
            <span className="small text-muted">
              {history ? `${history.count} records` : 'Loading…'}
            </span>
            <button className="btn btn-outline-secondary btn-sm" onClick={loadHistory}>Refresh</button>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th><th>Instrument</th>
                  <th className="text-end">Score</th><th className="text-end">Max</th>
                  <th>Interpretation</th><th>Examiner</th><th>Date</th>
                </tr>
              </thead>
              <tbody>
                {(history?.records || []).map(r => (
                  <tr key={r.id}>
                    <td className="small fw-bold">{r.patient_id}</td>
                    <td><span className="badge bg-secondary">{r.instrument}</span></td>
                    <td className="text-end small fw-bold">{r.score}</td>
                    <td className="text-end small text-muted">{r.max_score}</td>
                    <td className="small">{r.interpretation || '—'}</td>
                    <td className="small">{r.examiner || '—'}</td>
                    <td className="small text-muted">{r.created_at ? r.created_at.slice(0, 16).replace('T', ' ') : '—'}</td>
                  </tr>
                ))}
                {(history?.records || []).length === 0 && (
                  <tr><td colSpan={7} className="text-center text-muted small py-3">No records yet</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          {/* Concepts */}
          {(defs.concepts || []).length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-dark text-white py-2 small fw-bold">Clinical Concepts</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {defs.concepts.map(c => (
                      <tr key={c.term}>
                        <td className="small fw-bold" style={{width:'25%'}}>{c.term}</td>
                        <td className="small">{c.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {/* Quality metrics */}
          {(defs.quality_metrics || []).length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-info text-white py-2 small fw-bold">Quality Metrics</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Metric</th><th>Target</th></tr></thead>
                  <tbody>
                    {defs.quality_metrics.map(m => (
                      <tr key={m.metric}>
                        <td className="small">{m.metric}</td>
                        <td className="small fw-bold">{m.target}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {/* Compliance */}
          {(defs.compliance || []).length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-success text-white py-2 small fw-bold">Compliance Standards</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Standard</th><th>Scope</th></tr></thead>
                  <tbody>
                    {defs.compliance.map(c => (
                      <tr key={c.standard}>
                        <td className="small fw-bold">{c.standard}</td>
                        <td className="small">{c.scope}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {/* Remediation */}
          {(defs.remediation || []).length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">Remediation Rules</div>
              <div className="card-body p-2">
                <ul className="mb-0">
                  {defs.remediation.map((r, i) => (
                    <li key={i} className="small">{typeof r === 'string' ? r : r.rule || JSON.stringify(r)}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
