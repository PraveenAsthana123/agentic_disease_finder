'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function EmergencySosDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/emergency-sos/overview`).then(r => r.json()),
      fetch(`${API}/api/emergency-sos/breakdown`).then(r => r.json()),
      fetch(`${API}/api/emergency-sos/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading emergency SOS data...</div>;

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'events', label: 'Events (' + ov.total_events + ')' },
    { id: 'patients', label: 'Per Patient' },
    { id: 'contacts', label: 'Contacts' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div className="p-3">
      <h3>&#x1f6a8; Emergency SOS Dashboard</h3>
      <p className="text-muted">
        Emergency alert monitoring &mdash; {ov.total_events} SOS events, {ov.patients_with_events} patients,
        {' '}{ov.responder_notified_pct}% responder-notified, avg response {Math.round(ov.response_time_stats.avg_seconds)}s
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={'nav-link ' + (tab === t.id ? 'active' : '')} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && <OverviewTab ov={ov} />}
      {tab === 'events' && <EventsTab ov={ov} />}
      {tab === 'patients' && <PatientsTab bd={bd} />}
      {tab === 'contacts' && <ContactsTab ov={ov} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  );
}

function KpiCard({ label, value, color }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className={'h5 mb-0 text-' + color}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function OverviewTab({ ov }) {
  const rt = ov.response_time_stats;
  const kpis = [
    ['Total Events', ov.total_events, 'danger'],
    ['Patients', ov.patients_with_events, 'primary'],
    ['Notified %', ov.responder_notified_pct + '%', 'success'],
    ['Location Shared', ov.location_sharing_pct + '%', 'info'],
    ['False Alarm %', ov.false_alarm_rate_pct + '%', 'warning'],
    ['Avg Response', Math.round(rt.avg_seconds) + 's', 'secondary'],
    ['Min Response', rt.min_seconds + 's', 'success'],
    ['Under 2 min', rt.pct_under_2min + '%', 'info'],
  ];

  const eventTypes = Object.entries(ov.event_type_distribution || {});
  const totalEvt = eventTypes.reduce((s, [, v]) => s + v, 0);
  const triggerMethods = Object.entries(ov.trigger_method_distribution || {});
  const totalTrig = triggerMethods.reduce((s, [, v]) => s + v, 0);
  const outcomes = Object.entries(ov.outcome_distribution || {});
  const totalOut = outcomes.reduce((s, [, v]) => s + v, 0);

  const outcomeColors = {
    'ems-dispatched': 'danger',
    'er-visit': 'danger',
    'caregiver-responded': 'success',
    'resolved-home': 'success',
    'false-alarm': 'warning',
  };

  const monthly = ov.monthly_trend || [];

  return (
    <div>
      <div className="row mb-3">
        {kpis.map(([l, v, c]) => <KpiCard key={l} label={l} value={v} color={c} />)}
      </div>

      <div className="row mb-3">
        {/* Event Types */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">&#x26a0;&#xfe0f; Event Types</div>
            <div className="card-body p-2">
              {eventTypes.map(([k, v]) => (
                <div key={k} className="mb-1">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="text-capitalize">{k.replace(/-/g, ' ')}</span>
                    <span>{v} ({((v / totalEvt) * 100).toFixed(1)}%)</span>
                  </div>
                  <div className="progress" style={{ height: 6 }}>
                    <div className="progress-bar bg-danger" style={{ width: ((v / totalEvt) * 100) + '%' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Trigger Methods */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">&#x1f4f1; Trigger Methods</div>
            <div className="card-body p-2">
              {triggerMethods.map(([k, v]) => (
                <div key={k} className="mb-1">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="text-capitalize">{k.replace(/-/g, ' ')}</span>
                    <span>{v} ({((v / totalTrig) * 100).toFixed(1)}%)</span>
                  </div>
                  <div className="progress" style={{ height: 6 }}>
                    <div className="progress-bar bg-primary" style={{ width: ((v / totalTrig) * 100) + '%' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Outcomes */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">&#x2705; Outcomes</div>
            <div className="card-body p-2">
              {outcomes.map(([k, v]) => (
                <div key={k} className="mb-1">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="text-capitalize">{k.replace(/-/g, ' ')}</span>
                    <span className={'badge bg-' + (outcomeColors[k] || 'secondary')}>{v}</span>
                  </div>
                  <div className="progress" style={{ height: 6 }}>
                    <div className={'progress-bar bg-' + (outcomeColors[k] || 'secondary')} style={{ width: ((v / totalOut) * 100) + '%' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Monthly Trend */}
      {monthly.length > 0 && (
        <div className="card shadow-sm mb-3">
          <div className="card-header fw-semibold">&#x1f4c5; Monthly SOS Trend</div>
          <div className="card-body p-2">
            <div className="d-flex align-items-end gap-2" style={{ height: 80 }}>
              {monthly.map(m => {
                const maxCnt = Math.max(...monthly.map(x => x.cnt));
                const pct = maxCnt > 0 ? (m.cnt / maxCnt) * 100 : 0;
                return (
                  <div key={m.month} className="d-flex flex-column align-items-center" style={{ flex: 1 }}>
                    <div className="small text-muted mb-1">{m.cnt}</div>
                    <div className="bg-danger rounded" style={{ width: '100%', height: pct * 0.6 + 'px', minHeight: 4 }} />
                    <div className="small text-muted mt-1" style={{ fontSize: '0.65rem' }}>{m.month.slice(5)}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function EventsTab({ ov }) {
  const eventTypes = Object.entries(ov.event_type_distribution || {});
  const triggerMethods = Object.entries(ov.trigger_method_distribution || {});
  const outcomes = Object.entries(ov.outcome_distribution || {});
  const rt = ov.response_time_stats;

  return (
    <div>
      <div className="row mb-3">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Event Type Summary</div>
            <table className="table table-sm table-hover mb-0">
              <thead><tr><th>Type</th><th>Count</th></tr></thead>
              <tbody>
                {eventTypes.map(([k, v]) => (
                  <tr key={k}>
                    <td className="text-capitalize">{k.replace(/-/g, ' ')}</td>
                    <td><span className="badge bg-danger">{v}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Response Time Analysis</div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <tbody>
                  <tr><td>Average</td><td><strong>{rt.avg_seconds}s</strong> ({Math.round(rt.avg_seconds / 60)} min)</td></tr>
                  <tr><td>Fastest</td><td><strong>{rt.min_seconds}s</strong></td></tr>
                  <tr><td>Slowest</td><td><strong>{rt.max_seconds}s</strong></td></tr>
                  <tr><td>Under 2 min</td><td><strong>{rt.pct_under_2min}%</strong></td></tr>
                  <tr><td>Responder notified</td><td><strong>{ov.responder_notified_pct}%</strong></td></tr>
                  <tr><td>Location shared</td><td><strong>{ov.location_sharing_pct}%</strong></td></tr>
                  <tr><td>False alarm rate</td><td><strong>{ov.false_alarm_rate_pct}%</strong></td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Trigger Methods</div>
            <table className="table table-sm table-hover mb-0">
              <thead><tr><th>Method</th><th>Count</th></tr></thead>
              <tbody>
                {triggerMethods.map(([k, v]) => (
                  <tr key={k}><td className="text-capitalize">{k.replace(/-/g, ' ')}</td><td><span className="badge bg-primary">{v}</span></td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Outcomes</div>
            <table className="table table-sm table-hover mb-0">
              <thead><tr><th>Outcome</th><th>Count</th></tr></thead>
              <tbody>
                {outcomes.map(([k, v]) => (
                  <tr key={k}><td className="text-capitalize">{k.replace(/-/g, ' ')}</td><td><span className="badge bg-secondary">{v}</span></td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function PatientsTab({ bd }) {
  const [sort, setSort] = useState('total_events');
  if (!bd || !bd.patient_events) return <div className="text-muted">No patient data.</div>;

  const rows = [...bd.patient_events].sort((a, b) => (b[sort] || 0) - (a[sort] || 0));

  return (
    <div className="card shadow-sm">
      <div className="card-header fw-semibold d-flex justify-content-between align-items-center">
        <span>Per-Patient SOS Summary</span>
        <select className="form-select form-select-sm w-auto" value={sort} onChange={e => setSort(e.target.value)}>
          <option value="total_events">Events</option>
          <option value="avg_response">Avg Response</option>
          <option value="ems_dispatched">EMS</option>
          <option value="er_visits">ER Visits</option>
          <option value="false_alarms">False Alarms</option>
        </select>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="table table-sm table-hover mb-0">
          <thead>
            <tr>
              <th>Patient</th>
              <th>Events</th>
              <th>Avg Response (s)</th>
              <th>EMS</th>
              <th>ER Visits</th>
              <th>False Alarms</th>
              <th>Location Shared</th>
              <th>Contacts</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.patient_id}>
                <td><code>{r.patient_id}</code></td>
                <td><span className="badge bg-danger">{r.total_events}</span></td>
                <td>{r.avg_response}s</td>
                <td>{r.ems_dispatched > 0 ? <span className="badge bg-danger">{r.ems_dispatched}</span> : 0}</td>
                <td>{r.er_visits > 0 ? <span className="badge bg-warning text-dark">{r.er_visits}</span> : 0}</td>
                <td>{r.false_alarms > 0 ? <span className="badge bg-secondary">{r.false_alarms}</span> : 0}</td>
                <td>{r.location_shared_count}</td>
                <td>{r.contact_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ContactsTab({ ov }) {
  const contacts = ov.contacts || {};
  const relDist = Object.entries(contacts.relationship_distribution || {});
  const total = relDist.reduce((s, [, v]) => s + v, 0);

  return (
    <div className="row">
      <div className="col-md-6 mb-3">
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">&#x1f465; Emergency Contacts</div>
          <div className="card-body">
            <div className="row text-center mb-3">
              <div className="col-4">
                <div className="h4 text-primary">{contacts.total}</div>
                <div className="small text-muted">Total Contacts</div>
              </div>
              <div className="col-4">
                <div className="h4 text-success">{contacts.patients_covered}</div>
                <div className="small text-muted">Patients Covered</div>
              </div>
              <div className="col-4">
                <div className="h4 text-info">{contacts.seizure_notify_pct}%</div>
                <div className="small text-muted">Seizure Notify</div>
              </div>
            </div>
            <hr />
            <h6>Relationship Distribution</h6>
            {relDist.map(([k, v]) => (
              <div key={k} className="mb-1">
                <div className="d-flex justify-content-between small mb-1">
                  <span className="text-capitalize">{k.replace(/_/g, ' ')}</span>
                  <span>{v} ({((v / total) * 100).toFixed(0)}%)</span>
                </div>
                <div className="progress" style={{ height: 5 }}>
                  <div className="progress-bar bg-info" style={{ width: ((v / total) * 100) + '%' }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="col-md-6 mb-3">
        <div className="card shadow-sm h-100">
          <div className="card-header fw-semibold">&#x1f4e1; Alert Coverage</div>
          <div className="card-body">
            <p className="small text-muted">
              Each registered patient has at least one emergency contact. Contacts are notified
              immediately via the alert pipeline when an SOS event is triggered.
            </p>
            <table className="table table-sm mb-0">
              <tbody>
                <tr><td>Responder notified rate</td><td><strong>{ov.responder_notified_pct}%</strong></td></tr>
                <tr><td>Location shared rate</td><td><strong>{ov.location_sharing_pct}%</strong></td></tr>
                <tr><td>Seizure-specific notify</td><td><strong>{contacts.seizure_notify_pct}%</strong></td></tr>
                <tr><td>Contacts per patient</td><td><strong>1:1</strong> (each patient → 1 primary)</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      {defs.glossary && (
        <div className="card shadow-sm mb-3">
          <div className="card-header fw-semibold">Glossary</div>
          <div className="card-body p-2">
            <table className="table table-sm mb-0">
              <thead><tr><th>Term</th><th>Definition</th></tr></thead>
              <tbody>
                {Object.entries(defs.glossary).map(([k, v]) => (
                  <tr key={k}><td><strong>{k}</strong></td><td>{v}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {defs.event_types && (
        <div className="card shadow-sm mb-3">
          <div className="card-header fw-semibold">Event Types</div>
          <div className="card-body p-2">
            <table className="table table-sm mb-0">
              <thead><tr><th>Type</th><th>Description</th></tr></thead>
              <tbody>
                {Object.entries(defs.event_types).map(([k, v]) => (
                  <tr key={k}><td><code>{k}</code></td><td>{v}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {defs.outcomes && (
        <div className="card shadow-sm mb-3">
          <div className="card-header fw-semibold">Outcomes</div>
          <div className="card-body p-2">
            <table className="table table-sm mb-0">
              <thead><tr><th>Outcome</th><th>Description</th></tr></thead>
              <tbody>
                {Object.entries(defs.outcomes).map(([k, v]) => (
                  <tr key={k}><td><code>{k}</code></td><td>{v}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {defs.preparedness_metrics && (
        <div className="card shadow-sm mb-3">
          <div className="card-header fw-semibold">Preparedness Metrics</div>
          <div className="card-body p-2">
            {Array.isArray(defs.preparedness_metrics)
              ? <ul>{defs.preparedness_metrics.map((m, i) => <li key={i}>{m}</li>)}</ul>
              : <pre className="small">{JSON.stringify(defs.preparedness_metrics, null, 2)}</pre>
            }
          </div>
        </div>
      )}
    </div>
  );
}
