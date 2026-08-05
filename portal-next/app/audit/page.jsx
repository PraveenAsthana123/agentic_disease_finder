'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const catColor = c => ({
  Administrative: 'primary', Regulatory: 'success', Technical: 'info',
  Clinical: 'warning', Quality: 'secondary'
}[c] || 'dark');

export default function AuditTrailPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/regulatory-audit-trail/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/regulatory-audit-trail/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/regulatory-audit-trail/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="container py-4"><p>Loading Audit Trail...</p></div>;

  const k = ov.kpis || {};
  const cats = ov.category_distribution || [];
  const recentEvents = ov.recent_events || [];
  const tabs = ['overview', 'by submission', 'by actor', 'event log', 'definitions'];

  return (
    <div className="container-fluid py-4" style={{ background: '#0b1120', minHeight: '100vh', color: '#e0e0e0' }}>
      <h2 className="mb-1" style={{ color: '#00e5ff' }}>&#x1f4dc; Regulatory Audit Trail</h2>
      <p className="text-secondary mb-3">
        Immutable record of all regulatory submission actions — {k.total_actions} actions across {k.total_submissions} submissions
      </p>

      {/* KPI cards */}
      <div className="row g-3 mb-4">
        {[
          { label: 'Total Actions',    val: k.total_actions,     icon: '&#x1f4cb;', color: '#00e5ff' },
          { label: 'Submissions',      val: k.total_submissions, icon: '&#x1f4c4;', color: '#76ff03' },
          { label: 'Actors',           val: k.total_actors,      icon: '&#x1f465;', color: '#ffab40' },
          { label: 'Categories',       val: k.total_categories,  icon: '&#x1f3f7;&#xfe0f;', color: '#ce93d8' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3">
            <div className="card h-100" style={{ background: '#162032', border: '1px solid #1e3a5f' }}>
              <div className="card-body text-center">
                <div style={{ fontSize: 28 }} dangerouslySetInnerHTML={{ __html: c.icon }} />
                <div style={{ fontSize: 28, fontWeight: 700, color: c.color }}>{c.val}</div>
                <div className="text-secondary small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3" style={{ borderColor: '#1e3a5f' }}>
        {tabs.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active' : ''}`}
              style={tab === t
                ? { background: '#1a3a5c', color: '#00e5ff', borderColor: '#1e3a5f #1e3a5f #1a3a5c' }
                : { color: '#8899aa', background: 'transparent', border: '1px solid transparent' }}
              onClick={() => setTab(t)}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card" style={{ background: '#162032', border: '1px solid #1e3a5f' }}>
              <div className="card-header" style={{ background: '#1a3a5c', color: '#00e5ff' }}>Category Distribution</div>
              <div className="card-body">
                {cats.map(c => {
                  const pct = Math.round((c.count / k.total_actions) * 100);
                  return (
                    <div key={c.category} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{c.category}</span>
                        <span className="text-secondary">{c.count} ({pct}%)</span>
                      </div>
                      <div style={{ background: '#0b1120', borderRadius: 4, height: 8 }}>
                        <div style={{ width: `${pct}%`, background: '#00e5ff', height: 8, borderRadius: 4 }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card" style={{ background: '#162032', border: '1px solid #1e3a5f' }}>
              <div className="card-header" style={{ background: '#1a3a5c', color: '#00e5ff' }}>Action Type Breakdown</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0" style={{ color: '#e0e0e0' }}>
                  <thead style={{ background: '#1a3a5c' }}>
                    <tr><th>Action</th><th className="text-end">Count</th></tr>
                  </thead>
                  <tbody>
                    {(ov.action_breakdown || []).slice(0, 10).map((a, i) => (
                      <tr key={i} style={{ borderColor: '#1e3a5f' }}>
                        <td className="small">{a.action}</td>
                        <td className="text-end small">{a.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-12">
            <div className="card" style={{ background: '#162032', border: '1px solid #1e3a5f' }}>
              <div className="card-header" style={{ background: '#1a3a5c', color: '#00e5ff' }}>
                Most Active Submission: <span style={{ color: '#76ff03' }}>{k.most_active_submission}</span>
                <span className="ms-2 text-secondary small">({k.most_active_submission_count} actions)</span>
              </div>
              <div className="card-body">
                <p className="text-secondary small mb-0">
                  Source: <code style={{ color: '#76ff03' }}>regulatory_audit_trail</code> table —
                  {k.total_actions} immutable audit events spanning {k.total_submissions} regulatory submissions
                  by {k.total_actors} actors across {k.total_categories} categories.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* By Submission tab */}
      {tab === 'by submission' && (
        <div className="card" style={{ background: '#162032', border: '1px solid #1e3a5f' }}>
          <div className="card-header" style={{ background: '#1a3a5c', color: '#00e5ff' }}>Per-Submission Breakdown</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0" style={{ color: '#e0e0e0' }}>
              <thead style={{ background: '#1a3a5c' }}>
                <tr>
                  <th>Submission ID</th>
                  <th className="text-center">Actions</th>
                  <th className="text-center">Actors</th>
                  <th className="text-center">Categories</th>
                  <th>First Action</th>
                  <th>Last Action</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.per_submission || []).map((s, i) => (
                  <tr key={i} style={{ borderColor: '#1e3a5f' }}>
                    <td><code style={{ color: '#76ff03', fontSize: 12 }}>{s.submission_id}</code></td>
                    <td className="text-center">{s.action_count}</td>
                    <td className="text-center">{s.actor_count}</td>
                    <td className="text-center">{s.category_count}</td>
                    <td className="small text-secondary">{s.first_action?.slice(0, 10)}</td>
                    <td className="small text-secondary">{s.last_action?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* By Actor tab */}
      {tab === 'by actor' && (
        <div className="card" style={{ background: '#162032', border: '1px solid #1e3a5f' }}>
          <div className="card-header" style={{ background: '#1a3a5c', color: '#00e5ff' }}>Per-Actor Activity</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0" style={{ color: '#e0e0e0' }}>
              <thead style={{ background: '#1a3a5c' }}>
                <tr>
                  <th>Actor</th>
                  <th className="text-center">Actions</th>
                  <th className="text-center">Submissions</th>
                  <th className="text-center">Action Types</th>
                  <th>Last Activity</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.per_actor || []).map((a, i) => (
                  <tr key={i} style={{ borderColor: '#1e3a5f' }}>
                    <td style={{ color: '#ffab40' }}>{a.actor}</td>
                    <td className="text-center">{a.action_count}</td>
                    <td className="text-center">{a.submission_count}</td>
                    <td className="text-center">{a.action_types}</td>
                    <td className="small text-secondary">{a.last_activity?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Event Log tab */}
      {tab === 'event log' && (
        <div className="card" style={{ background: '#162032', border: '1px solid #1e3a5f' }}>
          <div className="card-header" style={{ background: '#1a3a5c', color: '#00e5ff' }}>
            Recent Audit Events <span className="text-secondary small ms-2">(most recent first)</span>
          </div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0" style={{ color: '#e0e0e0' }}>
              <thead style={{ background: '#1a3a5c' }}>
                <tr>
                  <th>Timestamp</th>
                  <th>Submission</th>
                  <th>Action</th>
                  <th>Actor</th>
                  <th>Category</th>
                </tr>
              </thead>
              <tbody>
                {recentEvents.map((e, i) => (
                  <tr key={i} style={{ borderColor: '#1e3a5f' }}>
                    <td className="small text-secondary" style={{ whiteSpace: 'nowrap' }}>
                      {e.timestamp?.slice(0, 16) || e.ts_utc?.slice(0, 16)}
                    </td>
                    <td><code style={{ color: '#76ff03', fontSize: 11 }}>{e.submission_id}</code></td>
                    <td className="small">{e.action}</td>
                    <td className="small" style={{ color: '#ffab40' }}>{e.actor}</td>
                    <td>
                      <span className={`badge bg-${catColor(e.category)}`} style={{ fontSize: 10 }}>
                        {e.category}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card" style={{ background: '#162032', border: '1px solid #1e3a5f' }}>
              <div className="card-header" style={{ background: '#1a3a5c', color: '#00e5ff' }}>Action Types</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0" style={{ color: '#e0e0e0' }}>
                  <thead style={{ background: '#1a3a5c' }}>
                    <tr><th>Action</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {(defs.action_types || []).map((a, i) => (
                      <tr key={i} style={{ borderColor: '#1e3a5f' }}>
                        <td className="small fw-bold" style={{ color: '#76ff03' }}>{a.action}</td>
                        <td className="small text-secondary">{a.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card" style={{ background: '#162032', border: '1px solid #1e3a5f' }}>
              <div className="card-header" style={{ background: '#1a3a5c', color: '#00e5ff' }}>Categories</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0" style={{ color: '#e0e0e0' }}>
                  <thead style={{ background: '#1a3a5c' }}>
                    <tr><th>Category</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {(defs.categories || []).map((c, i) => (
                      <tr key={i} style={{ borderColor: '#1e3a5f' }}>
                        <td>
                          <span className={`badge bg-${catColor(c.category)}`}>{c.category}</span>
                        </td>
                        <td className="small text-secondary">{c.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
