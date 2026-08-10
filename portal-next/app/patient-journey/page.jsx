'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',   label: 'Overview' },
  { id: 'breakdown',  label: 'Per Patient' },
  { id: 'definitions', label: 'Definitions' },
];

const URGENCY_COLOR = { emergent: 'danger', urgent: 'warning', routine: 'info', elective: 'secondary' };
const STATUS_COLOR  = { completed: 'success', scheduled: 'primary', in_progress: 'info', triaged: 'info', pending_triage: 'warning', cancelled: 'secondary' };
const INTENSITY_COLOR = { High: 'danger', Medium: 'warning', Low: 'success' };
const QUALITY_COLOR = { excellent: 'success', good: 'info', fair: 'warning', poor: 'danger' };

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-2 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-3">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function BarList({ items, keyField, valueField, colorFn }) {
  if (!items?.length) return <p className="text-muted small">No data.</p>;
  const max = Math.max(...items.map(i => i[valueField]));
  return (
    <table className="table table-sm mb-0">
      <tbody>
        {items.map((item, i) => {
          const pct = max > 0 ? ((item[valueField] / max) * 100).toFixed(0) : 0;
          const col = colorFn ? colorFn(item[keyField]) : 'primary';
          return (
            <tr key={i}>
              <td className="small fw-semibold text-capitalize" style={{ width: '40%' }}>
                {(item[keyField] || '').replace(/_/g, ' ')}
              </td>
              <td style={{ width: '45%' }}>
                <div className="progress" style={{ height: 13 }}>
                  <div className={`progress-bar bg-${col}`} style={{ width: `${pct}%` }}>
                    <span className="small">{item[valueField]}</span>
                  </div>
                </div>
              </td>
              <td className="small text-end text-muted">{item[valueField]}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function FunnelStage({ stage, patients, events, index, totalPatients }) {
  const colors = ['primary', 'info', 'warning', 'danger'];
  const col = colors[index] || 'secondary';
  const pct = totalPatients > 0 ? Math.round(patients / totalPatients * 100) : 0;
  const width = 100 - index * 12;
  return (
    <div className="text-center mb-2">
      <div
        className={`bg-${col} text-white rounded py-3 mx-auto d-flex align-items-center justify-content-between px-4`}
        style={{ width: `${width}%`, minWidth: 280 }}
      >
        <span className="fw-bold">{stage}</span>
        <span className="badge bg-white text-dark">{patients} pts · {events} events</span>
      </div>
      {index < 3 && (
        <div className="text-muted small my-1">
          ▼ {pct}% coverage
        </div>
      )}
    </div>
  );
}

export default function PatientJourneyPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('total_touchpoints');

  useEffect(() => {
    fetch(`${API}/api/patient-journey/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/patient-journey/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/patient-journey/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return (
    <div className="p-4 text-center">
      <div className="spinner-border text-primary" />
      <div className="mt-2 text-muted">Loading Patient Care Journey…</div>
    </div>
  );

  const patients = bd?.per_patient || [];
  const filtered = patients
    .filter(p => !search || p.patient_id?.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => (b[sortBy] ?? 0) - (a[sortBy] ?? 0));

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 d-flex align-items-center gap-2">
        <span style={{ fontSize: '1.4rem' }}>🛤️</span>
        <div>
          <h4 className="mb-0 fw-bold">Patient Care Journey</h4>
          <div className="text-muted small">
            {ov.total_patients} patients · {ov.total_touchpoints} touchpoints ·
            Referral → Appointment → Telehealth → Hospitalisation
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          {/* Top KPIs */}
          <div className="row g-2 mb-4">
            <KPI label="Unique Patients"  value={ov.total_patients}       color="primary" />
            <KPI label="Total Touchpoints" value={ov.total_touchpoints}   color="info" />
            <KPI label="Avg Triage Score"  value={`${ov.avg_triage_score}/100`} color="warning" />
            <KPI label="Appt Completion"   value={`${ov.appt_completion_rate}%`} color="success" />
            <KPI label="Avg Satisfaction"  value={`${ov.avg_patient_satisfaction}/5`} color="info" sub="telehealth" />
            <KPI label="Avg LOS"           value={`${ov.avg_length_of_stay_days}d`} color="secondary" sub="hospitalisation" />
          </div>

          {/* Care Funnel */}
          <div className="row g-3 mb-4">
            <div className="col-md-5">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold py-2">🛤️ Care Pathway Funnel</div>
                <div className="card-body">
                  {(ov.care_funnel || []).map((stage, i) => (
                    <FunnelStage
                      key={stage.stage}
                      stage={stage.stage}
                      patients={stage.patients}
                      events={stage.events}
                      index={i}
                      totalPatients={ov.total_patients}
                    />
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-7">
              <div className="row g-3 h-100">
                {/* Referral Urgency */}
                <div className="col-6">
                  <div className="card shadow-sm h-100">
                    <div className="card-header fw-semibold py-2 small">⚡ Referral Urgency</div>
                    <div className="card-body p-2">
                      <BarList
                        items={ov.referral_urgency}
                        keyField="urgency"
                        valueField="count"
                        colorFn={k => URGENCY_COLOR[k] || 'secondary'}
                      />
                    </div>
                  </div>
                </div>
                {/* Referral Source */}
                <div className="col-6">
                  <div className="card shadow-sm h-100">
                    <div className="card-header fw-semibold py-2 small">📋 Referral Source</div>
                    <div className="card-body p-2">
                      <BarList
                        items={ov.referral_source}
                        keyField="source"
                        valueField="count"
                      />
                    </div>
                  </div>
                </div>
                {/* Appt by Dept */}
                <div className="col-6">
                  <div className="card shadow-sm h-100">
                    <div className="card-header fw-semibold py-2 small">🏥 Appts by Dept</div>
                    <div className="card-body p-2">
                      <BarList
                        items={ov.appt_department}
                        keyField="dept"
                        valueField="count"
                        colorFn={() => 'primary'}
                      />
                    </div>
                  </div>
                </div>
                {/* Telehealth Quality */}
                <div className="col-6">
                  <div className="card shadow-sm h-100">
                    <div className="card-header fw-semibold py-2 small">📶 Telehealth Quality</div>
                    <div className="card-body p-2">
                      <BarList
                        items={ov.telehealth_quality}
                        keyField="quality"
                        valueField="count"
                        colorFn={k => QUALITY_COLOR[k] || 'secondary'}
                      />
                      <div className="text-muted small mt-2">Excellent: {ov.excellent_quality_pct}%</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Row 2 */}
          <div className="row g-3 mb-4">
            {/* Hospitalisation */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold py-2">🏨 Hospitalisation KPIs</div>
                <div className="card-body">
                  <table className="table table-sm mb-3">
                    <tbody>
                      <tr><td className="text-muted small">Total stays</td><td className="fw-bold small text-end">{ov.n_hospitalizations}</td></tr>
                      <tr><td className="text-muted small">Avg LOS</td><td className="fw-bold small text-end">{ov.avg_length_of_stay_days} days</td></tr>
                      <tr><td className="text-muted small">Readmission ≤30d</td><td className="fw-bold small text-end text-warning">{ov.readmission_rate_pct}%</td></tr>
                      <tr><td className="text-muted small">Seizure-free at d/c</td><td className="fw-bold small text-end text-success">{ov.seizure_free_at_discharge_pct}%</td></tr>
                      <tr><td className="text-muted small">Avg cost (USD)</td><td className="fw-bold small text-end">${(ov.avg_cost_usd || 0).toLocaleString()}</td></tr>
                    </tbody>
                  </table>
                  <BarList
                    items={ov.admission_types}
                    keyField="type"
                    valueField="count"
                    colorFn={k => k === 'emergency' ? 'danger' : k === 'planned' ? 'success' : 'info'}
                  />
                </div>
              </div>
            </div>

            {/* Appointment types */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold py-2">📅 Appointment Types</div>
                <div className="card-body p-2">
                  <div className="small text-muted mb-2">
                    Completion rate: <strong>{ov.appt_completion_rate}%</strong> · Avg duration: <strong>{ov.avg_appt_duration_min} min</strong>
                  </div>
                  <BarList
                    items={ov.appt_type}
                    keyField="type"
                    valueField="count"
                    colorFn={() => 'primary'}
                  />
                </div>
              </div>
            </div>

            {/* Referral reasons + Discharge dispositions */}
            <div className="col-md-4">
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-semibold py-2 small">📥 Top Referral Reasons</div>
                <div className="card-body p-2">
                  <BarList
                    items={ov.referral_reason}
                    keyField="reason"
                    valueField="count"
                    colorFn={() => 'warning'}
                  />
                </div>
              </div>
              <div className="card shadow-sm">
                <div className="card-header fw-semibold py-2 small">🏠 Discharge Dispositions</div>
                <div className="card-body p-2">
                  <BarList
                    items={ov.discharge_dispositions}
                    keyField="disposition"
                    valueField="count"
                    colorFn={k => k === 'home' ? 'success' : k === 'facility' ? 'warning' : 'info'}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Telehealth platform + session types */}
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold py-2">📱 Telehealth Platforms</div>
                <div className="card-body p-2">
                  <BarList
                    items={ov.telehealth_platform}
                    keyField="platform"
                    valueField="count"
                    colorFn={() => 'info'}
                  />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold py-2">🎥 Telehealth Session Types</div>
                <div className="card-body p-2">
                  <BarList
                    items={ov.telehealth_type}
                    keyField="type"
                    valueField="count"
                    colorFn={() => 'info'}
                  />
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'breakdown' && (
        <>
          <div className="row g-2 mb-3">
            {(bd?.intensity_distribution || []).map(item => (
              <div key={item.intensity} className="col-auto">
                <span className={`badge bg-${INTENSITY_COLOR[item.intensity] || 'secondary'} fs-6`}>
                  {item.intensity}: {item.count} patients
                </span>
              </div>
            ))}
            <div className="col-auto">
              <span className="badge bg-secondary fs-6">
                Avg {bd?.avg_touchpoints_per_patient} touchpoints/patient
              </span>
            </div>
          </div>

          {/* Stages reached distribution */}
          <div className="row g-2 mb-3">
            {(bd?.stages_reached_distribution || []).map(item => (
              <div key={item.stages} className="col-auto">
                <span className="badge bg-primary bg-opacity-75">
                  {item.stages} stage{item.stages !== 1 ? 's' : ''}: {item.count} pts
                </span>
              </div>
            ))}
          </div>

          <div className="row g-2 mb-3">
            <div className="col-md-4">
              <input
                className="form-control form-control-sm"
                placeholder="Search patient ID…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
            <div className="col-md-4">
              <select className="form-select form-select-sm" value={sortBy} onChange={e => setSortBy(e.target.value)}>
                <option value="total_touchpoints">Sort: Total Touchpoints</option>
                <option value="referrals">Sort: Referrals</option>
                <option value="appointments">Sort: Appointments</option>
                <option value="telehealth">Sort: Telehealth</option>
                <option value="hospitalizations">Sort: Hospitalizations</option>
                <option value="stages_reached">Sort: Stages Reached</option>
              </select>
            </div>
            <div className="col-auto d-flex align-items-center">
              <span className="text-muted small">{filtered.length} patients</span>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-body p-0">
              <div className="table-responsive" style={{ maxHeight: 600 }}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light sticky-top">
                    <tr>
                      <th>Patient ID</th>
                      <th className="text-center">Referrals</th>
                      <th className="text-center">Appointments</th>
                      <th className="text-center">Telehealth</th>
                      <th className="text-center">Hospitalizations</th>
                      <th className="text-center">Total</th>
                      <th className="text-center">Stages</th>
                      <th>Intensity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map(p => (
                      <tr key={p.patient_id}>
                        <td className="small fw-semibold">{p.patient_id}</td>
                        <td className="text-center small">{p.referrals}</td>
                        <td className="text-center small">{p.appointments}</td>
                        <td className="text-center small">{p.telehealth}</td>
                        <td className="text-center small">{p.hospitalizations}</td>
                        <td className="text-center small fw-bold">{p.total_touchpoints}</td>
                        <td className="text-center">
                          <span className="badge bg-primary bg-opacity-75">{p.stages_reached}/4</span>
                        </td>
                        <td>
                          <span className={`badge bg-${INTENSITY_COLOR[p.intensity] || 'secondary'}`}>
                            {p.intensity}
                          </span>
                        </td>
                      </tr>
                    ))}
                    {filtered.length === 0 && (
                      <tr><td colSpan={8} className="text-center text-muted py-4">No patients match.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold py-2">📋 Data Sources</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Table</th><th className="text-end">Rows</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {(defs.sources || []).map(s => (
                      <tr key={s.table}>
                        <td className="small font-monospace fw-semibold">{s.table}</td>
                        <td className="small text-end">{s.rows}</td>
                        <td className="small text-muted">{s.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold py-2">⚡ Urgency Levels</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <tbody>
                    {Object.entries(defs.urgency_levels || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td><span className={`badge bg-${URGENCY_COLOR[k] || 'secondary'}`}>{k}</span></td>
                        <td className="small text-muted">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold py-2">📖 Field Definitions</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Field</th><th>Meaning</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(defs.fields || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td className="small font-monospace fw-semibold">{k}</td>
                        <td className="small text-muted">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {(defs.clinical_notes || []).length > 0 && (
              <div className="card shadow-sm">
                <div className="card-header fw-semibold py-2">📝 Clinical Notes</div>
                <div className="card-body">
                  <ul className="mb-0 ps-3">
                    {defs.clinical_notes.map((n, i) => (
                      <li key={i} className="small text-muted mb-1">{n}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
