'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const ACTOR_COLORS = {
  middleware: '#6366f1',
  system:     '#10b981',
  psychiatrist:'#f59e0b',
  neurologist: '#3b82f6',
  compliance_agent: '#ef4444',
  security_agent:   '#8b5cf6',
};

const actorBadge = actor => (
  <span className="badge me-1" style={{ background: ACTOR_COLORS[actor] || '#6b7280', color: '#fff', fontSize: '0.7rem' }}>{actor}</span>
);

export default function TransactionLogPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [filter, setFilter] = useState('');

  useEffect(() => {
    fetch(`${API}/api/transaction-log/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/transaction-log/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/transaction-log/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const byComponent = ov.by_component || {};
  const byActor     = ov.by_actor || {};
  const byPatient   = ov.by_patient || {};
  const byAction    = ov.by_action || {};
  const hourly      = ov.hourly_activity || [];
  const recent      = ov.recent || [];
  const total       = ov.total || 0;

  const topComponents = Object.entries(byComponent).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const topActions    = Object.entries(byAction).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const topActors     = Object.entries(byActor).sort((a, b) => b[1] - a[1]);
  const maxHour       = Math.max(...hourly.map(h => h.count), 1);

  const items = bd?.items || [];
  const filtered = filter
    ? items.filter(i =>
        (i.component || '').toLowerCase().includes(filter.toLowerCase()) ||
        (i.action || '').toLowerCase().includes(filter.toLowerCase()) ||
        (i.actor || '').toLowerCase().includes(filter.toLowerCase())
      )
    : items;

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'activity',   label: 'Hourly Activity' },
    { id: 'log',        label: 'Event Log' },
    { id: 'definitions',label: 'Definitions' },
  ];

  return (
    <div>
      <h3>📋 Transaction Log</h3>
      <p className="text-muted">
        Unified audit trail of all system actions — {total.toLocaleString()} transactions across {Object.keys(byComponent).length} components · {Object.keys(byActor).length} actors
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Transactions', value: total.toLocaleString(),              color: 'primary' },
          { label: 'Components',         value: Object.keys(byComponent).length,     color: 'info' },
          { label: 'Distinct Actions',   value: Object.keys(byAction).length,        color: 'secondary' },
          { label: 'Actors',             value: Object.keys(byActor).length,         color: 'success' },
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
          {/* By Component */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">By Component (top 10)</div>
              <div className="card-body">
                {topComponents.map(([comp, cnt]) => (
                  <div key={comp} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className="text-truncate" style={{ maxWidth: '65%' }}>{comp}</span>
                      <span className="fw-semibold">{cnt.toLocaleString()} <span className="text-muted">({total ? Math.round(cnt / total * 100) : 0}%)</span></span>
                    </div>
                    <div className="progress" style={{ height: 7 }}>
                      <div className="progress-bar bg-primary" style={{ width: `${total ? cnt / total * 100 : 0}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* By Actor + Top Actions */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">By Actor</div>
              <div className="card-body py-2">
                {topActors.map(([actor, cnt]) => (
                  <div key={actor} className="d-flex justify-content-between align-items-center mb-1 small">
                    <span>{actorBadge(actor)}{actor}</span>
                    <span className="fw-semibold">{cnt.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Top Actions (top 10)</div>
              <div className="card-body py-2">
                {topActions.map(([action, cnt]) => (
                  <div key={action} className="d-flex justify-content-between small mb-1">
                    <span className="text-truncate me-2" style={{ maxWidth: '70%' }}>{action}</span>
                    <span className="fw-semibold text-nowrap">{cnt.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Recent transactions */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Recent Transactions (last 10)</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead>
                      <tr>
                        <th>#</th><th>Component</th><th>Action</th><th>Actor</th><th>Timestamp (local)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recent.slice(0, 10).map(r => (
                        <tr key={r.id}>
                          <td className="text-muted small">{r.id}</td>
                          <td><span className="badge bg-light text-dark border small">{r.component}</span></td>
                          <td className="small text-truncate" style={{ maxWidth: 220 }}>{r.action}</td>
                          <td>{actorBadge(r.actor)}</td>
                          <td className="text-muted small text-nowrap">{r.ts_local || r.ts_utc}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Hourly Activity tab */}
      {tab === 'activity' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Hourly Activity Distribution (24h)</div>
          <div className="card-body">
            <div className="d-flex align-items-end gap-1" style={{ height: 160 }}>
              {hourly.map(h => (
                <div key={h.hour} className="d-flex flex-column align-items-center" style={{ flex: 1, minWidth: 18 }}>
                  <div
                    className="bg-primary rounded-top"
                    style={{ width: '100%', height: `${Math.round((h.count / maxHour) * 140)}px`, minHeight: 2 }}
                    title={`${h.hour}:00 — ${h.count.toLocaleString()} txns`}
                  />
                  <div className="text-muted" style={{ fontSize: '0.6rem', writingMode: 'vertical-rl', transform: 'rotate(180deg)', marginTop: 2 }}>{h.hour}</div>
                </div>
              ))}
            </div>
            <div className="mt-2 text-muted small text-center">Hour of day (UTC) — hover bar for count</div>

            <div className="row mt-3">
              {hourly.map(h => (
                <div key={h.hour} className="col-6 col-md-4 col-lg-3 mb-1 small">
                  <span className="text-muted me-1">{h.hour}:00</span>
                  <span className="fw-semibold">{h.count.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Event Log tab */}
      {tab === 'log' && (
        <div>
          <div className="mb-2">
            <input
              type="text"
              className="form-control"
              placeholder="Filter by component, action, or actor..."
              value={filter}
              onChange={e => setFilter(e.target.value)}
            />
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold d-flex justify-content-between align-items-center">
              <span>Transaction Events</span>
              <span className="badge bg-secondary">{filtered.length.toLocaleString()} shown</span>
            </div>
            <div className="card-body p-0">
              {!bd ? (
                <div className="p-3"><div className="spinner-border spinner-border-sm text-primary" /></div>
              ) : (
                <div className="table-responsive" style={{ maxHeight: 500, overflowY: 'auto' }}>
                  <table className="table table-sm table-striped mb-0">
                    <thead className="sticky-top bg-white">
                      <tr>
                        <th>#</th><th>Component</th><th>Action</th><th>Actor</th><th>Patient</th><th>Timestamp</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.slice(0, 500).map(r => (
                        <tr key={r.id}>
                          <td className="text-muted small">{r.id}</td>
                          <td><span className="badge bg-light text-dark border small">{r.component}</span></td>
                          <td className="small text-truncate" style={{ maxWidth: 200 }}>{r.action}</td>
                          <td>{actorBadge(r.actor)}</td>
                          <td className="text-muted small">{r.patient_id || '—'}</td>
                          <td className="text-muted small text-nowrap">{(r.ts_local || r.ts_utc || '').slice(0, 19)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {filtered.length > 500 && (
                    <div className="text-muted small p-2 text-center">Showing first 500 of {filtered.length.toLocaleString()} — use filter to narrow</div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && (
        <div className="row">
          {defs ? (
            <>
              <div className="col-md-8 mb-3">
                <div className="card shadow-sm h-100">
                  <div className="card-header fw-semibold">Field Definitions</div>
                  <div className="card-body">
                    <table className="table table-sm">
                      <thead><tr><th>Field</th><th>Description</th></tr></thead>
                      <tbody>
                        {Object.entries(defs.fields || {}).map(([field, desc]) => (
                          <tr key={field}>
                            <td><code>{field}</code></td>
                            <td className="small">{desc}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
              <div className="col-md-4 mb-3">
                <div className="card shadow-sm">
                  <div className="card-header fw-semibold">Data Source</div>
                  <div className="card-body small">
                    <p><strong>Table:</strong> <code>transaction_log</code></p>
                    <p><strong>DB:</strong> <code>clinical.db</code></p>
                    <p className="text-muted">{defs.description || 'Transaction audit log of all system actions.'}</p>
                  </div>
                </div>
                <div className="card shadow-sm mt-3">
                  <div className="card-header fw-semibold">Component Registry</div>
                  <div className="card-body py-2">
                    {Object.entries(byComponent).sort((a, b) => b[1] - a[1]).map(([comp, cnt]) => (
                      <div key={comp} className="d-flex justify-content-between small mb-1">
                        <code>{comp}</code>
                        <span className="text-muted">{cnt.toLocaleString()}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="p-3"><div className="spinner-border spinner-border-sm text-primary" /></div>
          )}
        </div>
      )}
    </div>
  );
}
