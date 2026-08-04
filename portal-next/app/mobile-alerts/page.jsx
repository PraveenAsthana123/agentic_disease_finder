'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s => ({ critical:'danger', high:'warning', medium:'info', low:'secondary', info:'primary' }[s] || 'secondary');

export default function MobileAlertsDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mobile-alerts/overview`).then(r => r.json()),
      fetch(`${API}/api/mobile-alerts/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mobile-alerts/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading mobile alerts data...</div>;

  const s = ov.summary;
  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'rules', label: 'Alert Rules' },
    { id: 'escalation', label: 'Escalation' },
    { id: 'events', label: 'Events (' + s.total_events + ')' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div className="p-3">
      <h3>&#x1f4f1; Mobile Alerts / SOS Dashboard</h3>
      <p className="text-muted">
        Real-time mobile SOS alerts &mdash; {s.total_events} events, {s.critical_events} critical,
        {' '}{s.patients_monitored} patients monitored, {s.active_rules} active rules
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={'nav-link ' + (tab === t.id ? 'active' : '')} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && <OverviewTab s={s} ov={ov} bd={bd} />}
      {tab === 'rules' && <RulesTab bd={bd} />}
      {tab === 'escalation' && <EscalationTab bd={bd} />}
      {tab === 'events' && <EventsTab bd={bd} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  );
}

function OverviewTab({ s, ov, bd }) {
  const kpis = [
    ['Total Events', s.total_events, 'primary'],
    ['Critical', s.critical_events, 'danger'],
    ['Acknowledged', s.acknowledged, 'success'],
    ['Resolved', s.resolved, 'success'],
    ['Unresolved', s.unresolved, 'warning'],
    ['Ack Rate', s.ack_rate_pct + '%', 'info'],
    ['Avg Response', s.avg_response_sec + 's', 'info'],
    ['Health Score', s.health_score, s.health_score >= 70 ? 'success' : 'warning'],
  ];
  return (
    <div>
      <div className="row mb-3">
        {kpis.map(([label, val, c]) => (
          <div key={label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card shadow-sm h-100">
              <div className="card-body text-center py-2">
                <div className={'h5 mb-0 text-' + c}>{val}</div>
                <div className="text-muted small">{label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-body">
              <h6>Severity Distribution</h6>
              {ov.severity_distribution.map(({ severity, count, pct }) => (
                <div key={severity} className="d-flex align-items-center mb-2">
                  <span className={'badge bg-' + sevColor(severity) + ' me-2'} style={{ minWidth: '70px' }}>{severity}</span>
                  <div className="flex-grow-1 me-2">
                    <div className="progress" style={{ height: '20px' }}>
                      <div className={'progress-bar bg-' + sevColor(severity)} style={{ width: pct + '%' }}>{count}</div>
                    </div>
                  </div>
                  <span className="text-muted small">{pct}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-body">
              <h6>Notification Channels</h6>
              <table className="table table-sm table-striped">
                <thead><tr><th>Channel</th><th>Provider</th><th>Delivered</th><th>Ack Rate</th></tr></thead>
                <tbody>
                  {bd.channel_stats.map(ch => (
                    <tr key={ch.channel}>
                      <td>{ch.channel}</td>
                      <td><span className="badge bg-secondary">{ch.provider}</span></td>
                      <td>{ch.delivered}</td>
                      <td>{ch.ack_rate_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-body">
          <h6>Daily Alert Trend</h6>
          <div className="d-flex align-items-end" style={{ height: '120px', gap: '4px' }}>
            {bd.daily_trend.map(d => {
              const maxE = Math.max(...bd.daily_trend.map(x => x.events), 1);
              const h = Math.max((d.events / maxE) * 100, 5);
              return (
                <div key={d.date} className="d-flex flex-column align-items-center flex-fill">
                  <div className="bg-primary rounded-top" style={{ width: '100%', height: h + '%', minHeight: '4px' }} title={d.events + ' events'} />
                  <small className="text-muted" style={{ fontSize: '0.65rem' }}>{d.date}</small>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-body">
          <h6>Top 10 Patients by Alert Volume</h6>
          <table className="table table-sm table-striped">
            <thead><tr><th>Patient</th><th>Total</th><th>Critical</th><th>Acked</th></tr></thead>
            <tbody>
              {bd.top_patients.map(p => (
                <tr key={p.patient_id}>
                  <td>{p.patient_id}</td>
                  <td>{p.total}</td>
                  <td>{p.critical > 0 ? <span className="badge bg-danger">{p.critical}</span> : 0}</td>
                  <td>{p.acked}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function RulesTab({ bd }) {
  return (
    <div className="card shadow-sm">
      <div className="card-body">
        <h6>Alert Rules ({bd.rule_stats.length})</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead>
              <tr><th>Rule</th><th>Category</th><th>Severity</th><th>Trigger</th><th>Cooldown</th><th>Channels</th><th>Fired</th><th>Ack Rate</th></tr>
            </thead>
            <tbody>
              {bd.rule_stats.map(r => (
                <tr key={r.rule_id}>
                  <td>{r.name}</td>
                  <td><span className="badge bg-secondary">{r.category}</span></td>
                  <td><span className={'badge bg-' + sevColor(r.severity)}>{r.severity}</span></td>
                  <td><code className="small">{r.trigger}</code></td>
                  <td>{r.cooldown_min}m</td>
                  <td>{r.channels.join(', ')}</td>
                  <td>{r.times_fired}</td>
                  <td>{r.ack_rate_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function EscalationTab({ bd }) {
  return (
    <div>
      <h6>Escalation Chain</h6>
      <div className="row">
        {bd.escalation_detail.map(e => (
          <div key={e.tier} className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-body">
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <span className="badge bg-primary">Tier {e.tier}</span>
                  <span className="text-muted small">+{e.delay_sec}s</span>
                </div>
                <h6>{e.name}</h6>
                <p className="small text-muted mb-2">{e.description}</p>
                <div className="small mb-1"><strong>Contacts:</strong> {e.contacts.join(', ')}</div>
                <div className="small"><strong>Events reached:</strong> {e.events_reached} ({e.pct_of_total}%)</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EventsTab({ bd }) {
  const [sevF, setSevF] = useState('all');
  const events = bd.all_events || [];
  const filtered = sevF === 'all' ? events : events.filter(e => e.severity === sevF);

  return (
    <div>
      <div className="mb-3">
        <select className="form-select form-select-sm w-auto d-inline" value={sevF} onChange={e => setSevF(e.target.value)}>
          <option value="all">All severities</option>
          {['critical', 'high', 'medium', 'low', 'info'].map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <span className="text-muted small ms-2">{filtered.length} events</span>
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-striped">
          <thead>
            <tr><th>ID</th><th>Time</th><th>Rule</th><th>Severity</th><th>Patient</th><th>Description</th><th>Status</th></tr>
          </thead>
          <tbody>
            {filtered.map(e => (
              <tr key={e.id}>
                <td className="small">{e.id}</td>
                <td className="small">{new Date(e.timestamp).toLocaleString()}</td>
                <td>{e.rule_id}</td>
                <td><span className={'badge bg-' + sevColor(e.severity)}>{e.severity}</span></td>
                <td>{e.patient_id}</td>
                <td className="small">{e.description}</td>
                <td>
                  {e.acknowledged && <span className="badge bg-success me-1">Acked</span>}
                  {e.resolved && <span className="badge bg-primary">Resolved</span>}
                  {!e.acknowledged && !e.resolved && <span className="badge bg-warning">Open</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading definitions...</div>;
  return (
    <div className="card shadow-sm">
      <div className="card-body">
        <h6>Definitions &amp; Concepts</h6>
        <dl>
          {(defs.concepts || []).map(c => (
            <div key={c.term} className="mb-2">
              <dt>{c.term}</dt>
              <dd className="text-muted small">{c.definition}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
