'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TYPE_COLORS = {
  Seizure:        '#ef4444',
  Appointment:    '#3b82f6',
  Telehealth:     '#8b5cf6',
  Assessment:     '#10b981',
  Hospitalization:'#f59e0b',
};

const badge = type => (
  <span className="badge me-1" style={{ background: TYPE_COLORS[type] || '#6b7280', color: '#fff' }}>{type}</span>
);

export default function LongitudinalTimelinePage() {
  const [ov, setOv]   = useState(null);
  const [bd, setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/longitudinal-timeline/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/longitudinal-timeline/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/longitudinal-timeline/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpis = ov.kpis || {};
  const typeDist = ov.event_type_distribution || {};
  const totalEvts = kpis.total_events || 0;
  const monthly = ov.monthly_trend || [];
  const topPts  = ov.top_patients || [];
  const patients = bd?.patients || [];

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'trend',      label: 'Monthly Trend' },
    { id: 'patients',   label: 'Per Patient' },
    { id: 'definitions',label: 'Definitions' },
  ];

  return (
    <div>
      <h3>📅 Longitudinal Patient Timeline</h3>
      <p className="text-muted">
        Unified clinical event stream — Seizures · Appointments · Telehealth · Assessments · Hospitalisations
        &nbsp;({kpis.unique_patients} patients · {kpis.months_of_data} months · {kpis.date_range_start} → {kpis.date_range_end})
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Events',          value: kpis.total_events,            color: 'primary' },
          { label: 'Unique Patients',        value: kpis.unique_patients,         color: 'info' },
          { label: 'Months of Data',         value: kpis.months_of_data,          color: 'secondary' },
          { label: 'Avg Events / Patient',   value: kpis.avg_events_per_patient,  color: 'success' },
        ].map(({ label, value, color }) => (
          <div key={label} className="col-6 col-md-3 mb-2">
            <div className="card shadow-sm text-center h-100">
              <div className="card-body py-2">
                <div className={`h4 mb-0 text-${color}`}>{value ?? '—'}</div>
                <div className="text-muted small">{label}</div>
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
        <div className="row">
          <div className="col-md-5 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Event Type Distribution</div>
              <div className="card-body">
                {Object.entries(typeDist).sort((a,b) => b[1]-a[1]).map(([type, cnt]) => (
                  <div key={type} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{badge(type)}{type}</span>
                      <span className="fw-semibold">{cnt} <span className="text-muted">({totalEvts ? Math.round(cnt/totalEvts*100) : 0}%)</span></span>
                    </div>
                    <div className="progress" style={{ height: 8 }}>
                      <div
                        className="progress-bar"
                        style={{ width: `${totalEvts ? cnt/totalEvts*100 : 0}%`, background: TYPE_COLORS[type] || '#6b7280' }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-7 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Top 10 Patients by Event Volume</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light"><tr><th>Patient</th><th className="text-end">Events</th></tr></thead>
                  <tbody>
                    {topPts.map(p => (
                      <tr key={p.patient_id}>
                        <td className="font-monospace small">{p.patient_id}</td>
                        <td className="text-end fw-semibold">{p.event_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Monthly Trend tab */}
      {tab === 'trend' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Monthly Event Volume ({monthly.length} months)</div>
          <div className="card-body p-0">
            <div className="table-responsive" style={{ maxHeight: 500 }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light sticky-top">
                  <tr>
                    <th>Month</th>
                    <th className="text-end">Total</th>
                    {Object.keys(TYPE_COLORS).map(t => <th key={t} className="text-end">{t}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {[...monthly].reverse().map(m => (
                    <tr key={m.month}>
                      <td className="font-monospace small">{m.month}</td>
                      <td className="text-end fw-semibold">{m.total}</td>
                      {Object.keys(TYPE_COLORS).map(t => (
                        <td key={t} className="text-end">
                          {m[t] ? <span className="badge" style={{ background: TYPE_COLORS[t], color: '#fff' }}>{m[t]}</span> : <span className="text-muted">—</span>}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Per Patient tab */}
      {tab === 'patients' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Per-Patient Event Breakdown ({patients.length} patients)</div>
          <div className="card-body p-0">
            <div className="table-responsive" style={{ maxHeight: 500 }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light sticky-top">
                  <tr>
                    <th>Patient</th>
                    <th className="text-end">Total</th>
                    {Object.keys(TYPE_COLORS).map(t => <th key={t} className="text-end">{t}</th>)}
                    <th>First</th>
                    <th>Last</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr key={p.patient_id}>
                      <td className="font-monospace small">{p.patient_id}</td>
                      <td className="text-end fw-bold">{p.total}</td>
                      {Object.keys(TYPE_COLORS).map(t => (
                        <td key={t} className="text-end">
                          {p[t] ? <span className="badge" style={{ background: TYPE_COLORS[t], color:'#fff' }}>{p[t]}</span> : <span className="text-muted">—</span>}
                        </td>
                      ))}
                      <td className="small text-muted">{p.first_date || '—'}</td>
                      <td className="small text-muted">{p.last_date || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          <p className="text-muted">{defs.description}</p>
          <h6>Event Types</h6>
          <table className="table table-sm table-bordered mb-3">
            <thead className="table-light"><tr><th>Type</th><th>Description</th></tr></thead>
            <tbody>
              {Object.entries(defs.event_types || {}).map(([k, v]) => (
                <tr key={k}><td>{badge(k)}{k}</td><td className="small">{v}</td></tr>
              ))}
            </tbody>
          </table>
          <h6>Metrics</h6>
          <table className="table table-sm table-bordered mb-3">
            <thead className="table-light"><tr><th>Metric</th><th>Definition</th></tr></thead>
            <tbody>
              {(defs.metrics || []).map(m => (
                <tr key={m.name}><td className="fw-semibold small">{m.name}</td><td className="small">{m.description}</td></tr>
              ))}
            </tbody>
          </table>
          <div className="alert alert-info small">{defs.clinical_context}</div>
        </div>
      )}
    </div>
  );
}
