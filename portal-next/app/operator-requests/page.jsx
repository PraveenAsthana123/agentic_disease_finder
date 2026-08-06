'use client';
import { useEffect, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KpiCard({ label, value, color, sub }) {
  return (
    <div className="card text-center shadow-sm h-100">
      <div className="card-body py-3">
        <div className="fs-3 fw-bold" style={{ color: color || '#6366f1' }}>{value ?? '—'}</div>
        <div className="small fw-semibold text-muted">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
      </div>
    </div>
  );
}

const STATUS_COLORS = {
  open: '#f59e0b', logged: '#6366f1', addressed: '#10b981',
  pending: '#06b6d4', 'not-implemented': '#9ca3af', rejected: '#ef4444',
};

const CAT_COLORS = {
  general: '#6366f1', session: '#10b981', history: '#f59e0b', meta: '#06b6d4',
};

const statusBadge = s => (
  <span className="badge" style={{ background: STATUS_COLORS[s] || '#9ca3af', fontSize: 11 }}>{s}</span>
);

const catBadge = c => (
  <span className="badge" style={{ background: CAT_COLORS[c] || '#9ca3af', fontSize: 11 }}>{c}</span>
);

function BarRow({ label, value, max, color }) {
  const pct = max ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <span className="small fw-semibold">{label}</span>
        <span className="small text-muted">{value} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color || '#6366f1' }} />
      </div>
    </div>
  );
}

function RequestRow({ req }) {
  const [open, setOpen] = useState(false);
  const txt = (req.request_text || '').slice(0, 120).replace(/\n/g, ' ');
  const full = req.request_text || '';
  return (
    <tr style={{ cursor: 'pointer' }} onClick={() => setOpen(o => !o)}>
      <td><span className="text-muted small">#{req.id}</span></td>
      <td>{catBadge(req.category)}</td>
      <td>{statusBadge(req.status)}</td>
      <td>
        <span className="small">{txt}{full.length > 120 ? '…' : ''}</span>
        {open && (
          <div className="mt-1 p-2 bg-light rounded small" style={{ whiteSpace: 'pre-wrap', fontSize: 11, maxHeight: 200, overflowY: 'auto' }}>
            {full}
          </div>
        )}
      </td>
      <td><span className="badge bg-secondary" style={{ fontSize: 10 }}>{req.source}</span></td>
      <td className="text-muted small">{(req.ts_utc || '').slice(0, 10)}</td>
    </tr>
  );
}

export default function OperatorRequestsDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [catFilter, setCatFilter] = useState('all');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/operator-requests/overview`).then(r => r.json()),
      fetch(`${API}/api/operator-requests/breakdown`).then(r => r.json()),
      fetch(`${API}/api/operator-requests/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov) return (
    <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}>
      <div className="spinner-border text-primary" />
    </div>
  );

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'requests', label: '📋 Requests' },
    { id: 'definitions', label: '📚 Definitions' },
  ];

  const statusDist = ov.status_distribution || [];
  const catDist = ov.category_distribution || [];
  const srcDist = ov.source_distribution || [];
  const dailyVol = ov.daily_volume || [];
  const crossTab = ov.category_status_cross || [];

  const maxStatus = Math.max(...statusDist.map(s => s.count), 1);
  const maxCat = Math.max(...catDist.map(c => c.count), 1);
  const maxSrc = Math.max(...srcDist.map(s => s.count), 1);

  // For requests tab: flatten per_category dict
  const perCat = bd?.per_category || {};
  const allCats = Object.keys(perCat);
  const filteredReqs = catFilter === 'all'
    ? allCats.flatMap(c => (perCat[c] || []).map(r => ({ ...r, category: c })))
    : (perCat[catFilter] || []).map(r => ({ ...r, category: catFilter }));
  const recentReqs = filteredReqs.slice(0, 50);

  return (
    <div className="container-fluid py-4">
      <h2 className="fw-bold mb-1">📥 Operator Requests Dashboard</h2>
      <p className="text-muted mb-3">
        {ov.total_requests} total requests · {ov.actionable} actionable ·{' '}
        {ov.resolution_rate}% resolution rate · {ov.addressed} addressed
      </p>

      {/* KPI row */}
      <div className="row g-2 mb-4">
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Total Requests" value={ov.total_requests} color="#6366f1" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Open" value={ov.open} color="#f59e0b" sub="needs triage" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Logged" value={ov.logged} color="#8b5cf6" sub="from transcripts" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Addressed" value={ov.addressed} color="#10b981" sub="completed" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Actionable" value={ov.actionable} color="#ef4444" sub="open + pending" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Resolution Rate" value={`${ov.resolution_rate}%`} color="#06b6d4" sub="addressed/total" />
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div className="row g-4">
          {/* Status Distribution */}
          <div className="col-md-4">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">📊 Status Distribution</div>
              <div className="card-body">
                {statusDist.map(s => (
                  <BarRow key={s.status} label={s.status} value={s.count}
                    max={maxStatus} color={STATUS_COLORS[s.status]} />
                ))}
              </div>
            </div>
          </div>

          {/* Category Distribution */}
          <div className="col-md-4">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">🏷️ Category Distribution</div>
              <div className="card-body">
                {catDist.map(c => (
                  <BarRow key={c.category} label={c.category} value={c.count}
                    max={maxCat} color={CAT_COLORS[c.category]} />
                ))}
              </div>
            </div>
          </div>

          {/* Source Distribution */}
          <div className="col-md-4">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">📡 Source Distribution</div>
              <div className="card-body">
                {srcDist.map(s => (
                  <BarRow key={s.source} label={s.source} value={s.count}
                    max={maxSrc} color="#06b6d4" />
                ))}
              </div>
            </div>
          </div>

          {/* Daily Volume Trend */}
          <div className="col-md-8">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">📈 Daily Request Volume (last 30 days)</div>
              <div className="card-body">
                {dailyVol.length === 0 ? (
                  <div className="text-muted text-center py-3">No data</div>
                ) : (
                  <div style={{ overflowX: 'auto' }}>
                    <div className="d-flex align-items-end gap-1" style={{ minWidth: 600, height: 120 }}>
                      {dailyVol.map((d, i) => {
                        const maxVol = Math.max(...dailyVol.map(x => x.count), 1);
                        const h = Math.max(4, Math.round((d.count / maxVol) * 100));
                        return (
                          <div key={i} className="d-flex flex-column align-items-center flex-fill" title={`${d.date}: ${d.count}`}>
                            <div className="small text-muted" style={{ fontSize: 9 }}>{d.count}</div>
                            <div style={{ width: '100%', height: h, background: '#6366f1', borderRadius: 2 }} />
                            <div className="text-muted" style={{ fontSize: 8, writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
                              {(d.date || '').slice(5)}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Category × Status Cross-tab */}
          <div className="col-md-4">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">🔀 Category × Status</div>
              <div className="card-body p-0">
                <div style={{ overflowX: 'auto' }}>
                  <table className="table table-sm table-striped mb-0" style={{ fontSize: 12 }}>
                    <thead className="table-dark">
                      <tr>
                        <th>Category</th>
                        <th>Status</th>
                        <th>Count</th>
                      </tr>
                    </thead>
                    <tbody>
                      {crossTab.length === 0
                        ? <tr><td colSpan={3} className="text-center text-muted">—</td></tr>
                        : crossTab.map((r, i) => (
                          <tr key={i}>
                            <td>{catBadge(r.category)}</td>
                            <td>{statusBadge(r.status)}</td>
                            <td className="fw-semibold">{r.count}</td>
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

      {/* REQUESTS TAB */}
      {tab === 'requests' && (
        <div>
          <div className="d-flex gap-2 mb-3 flex-wrap">
            {['all', ...allCats].map(c => (
              <button
                key={c}
                className={`btn btn-sm ${catFilter === c ? 'btn-primary' : 'btn-outline-secondary'}`}
                onClick={() => setCatFilter(c)}
              >{c} {c === 'all' ? `(${filteredReqs.length || ov.total_requests})` : `(${(perCat[c] || []).length})`}</button>
            ))}
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">
              Recent Requests — {catFilter === 'all' ? 'All Categories' : catFilter}
              <span className="text-muted ms-2 small">(showing {recentReqs.length} of {filteredReqs.length})</span>
            </div>
            <div className="card-body p-0">
              <div style={{ overflowX: 'auto' }}>
                <table className="table table-sm table-hover mb-0" style={{ fontSize: 12 }}>
                  <thead className="table-dark">
                    <tr>
                      <th>#</th>
                      <th>Category</th>
                      <th>Status</th>
                      <th>Request (click to expand)</th>
                      <th>Source</th>
                      <th>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentReqs.length === 0
                      ? <tr><td colSpan={6} className="text-center text-muted py-3">No requests</td></tr>
                      : recentReqs.map((r, i) => <RequestRow key={i} req={r} />)}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <div className="row g-4">
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">📌 Request Statuses</div>
              <div className="card-body">
                {Object.entries(defs.request_statuses || {}).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    {statusBadge(k)}
                    <span className="ms-2 small">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">🏷️ Categories</div>
              <div className="card-body">
                {Object.entries(defs.categories || {}).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    {catBadge(k)}
                    <span className="ms-2 small">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">📡 Sources</div>
              <div className="card-body">
                {Object.entries(defs.sources || {}).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <span className="badge bg-secondary">{k}</span>
                    <span className="ms-2 small">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">⚙️ Implementation Fields</div>
              <div className="card-body">
                {Object.entries(defs.implementation_fields || {}).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <span className="fw-semibold small text-primary">{k}</span>
                    <span className="ms-2 text-muted small">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">📖 Glossary</div>
              <div className="card-body">
                <div className="row">
                  {Object.entries(defs.glossary || {}).map(([k, v]) => (
                    <div key={k} className="col-md-6 mb-2">
                      <span className="fw-semibold small">{k}</span>
                      <p className="text-muted small mb-0">{v}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
