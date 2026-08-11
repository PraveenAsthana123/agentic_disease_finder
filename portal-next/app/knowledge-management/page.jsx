'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STAGE_COLORS = {
  published: 'success',
  approved: 'info',
  created: 'secondary',
  expired: 'warning',
  archived: 'dark',
};

const TYPE_ICONS = {
  'Clinical Analysis': '🔬',
  'Assessment Instrument': '📋',
  'Expert Review': '👨‍⚕️',
  'EEG Upload': '🧠',
  'Medication Record': '💊',
  'Imaging Finding': '🖼️',
  'Patient Diary': '📔',
  'Conversation Knowledge': '💬',
};

function KPI({ label, value, color = 'primary', sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className={`card border-${color} h-100`}>
        <div className="card-body p-3 text-center">
          <div className={`fs-3 fw-bold text-${color}`}>{value}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.72rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function StageBadge({ stage }) {
  const color = STAGE_COLORS[stage] || 'secondary';
  return <span className={`badge bg-${color} me-1`}>{stage}</span>;
}

export default function KnowledgeManagementDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [regSearch, setRegSearch] = useState('');
  const [regFilter, setRegFilter] = useState('all');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/knowledge-management/overview`).then(r => r.json()),
      fetch(`${API}/api/knowledge-management/breakdown`).then(r => r.json()),
      fetch(`${API}/api/knowledge-management/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading knowledge management data…</div>;

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'register', label: '📚 Knowledge Register' },
    { id: 'patients', label: '👤 Per Patient' },
    { id: 'lifecycle', label: '🔄 Lifecycle Events' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  // Filter knowledge register
  const register = (bd?.knowledge_register || []).filter(item => {
    const matchSearch = !regSearch ||
      (item.title || '').toLowerCase().includes(regSearch.toLowerCase()) ||
      (item.patient_id || '').toLowerCase().includes(regSearch.toLowerCase());
    const matchFilter = regFilter === 'all' || item.stage === regFilter;
    return matchSearch && matchFilter;
  });

  return (
    <div className="p-3">
      <h3>🗂️ Knowledge Management Dashboard</h3>
      <p className="text-muted small">
        Knowledge lifecycle tracking — {ov.total_knowledge_items} items across {ov.knowledge_types_count} types ·
        {' '}{ov.publish_rate_pct}% publish rate · {ov.patients_with_knowledge} patients covered
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div>
          <div className="row">
            <KPI label="Total Knowledge Items" value={ov.total_knowledge_items} color="primary" />
            <KPI label="Published" value={ov.published_count} color="success" sub={`${ov.publish_rate_pct}% rate`} />
            <KPI label="Approved" value={ov.approved_count} color="info" sub={`${ov.approval_rate_pct}% approval`} />
            <KPI label="Patients Covered" value={ov.patients_with_knowledge} color="warning" />
          </div>
          <div className="row">
            <KPI label="Created (Pending)" value={ov.created_count} color="secondary" />
            <KPI label="Avg Confidence" value={ov.avg_confidence != null ? (ov.avg_confidence * 100).toFixed(1) + '%' : 'N/A'} color="primary" />
            <KPI label="Lifecycle Events" value={ov.total_lifecycle_events} color="dark" />
            <KPI label="Knowledge Types" value={ov.knowledge_types_count} color="info" />
          </div>

          {/* Stage distribution */}
          <div className="row mt-3">
            <div className="col-md-6 mb-3">
              <div className="card">
                <div className="card-header fw-semibold">Lifecycle Stage Distribution</div>
                <div className="card-body">
                  {(ov.stage_distribution || []).map(s => (
                    <div key={s.stage} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span><StageBadge stage={s.stage} /> {s.stage}</span>
                        <span>{s.count} ({Math.round(s.count / ov.total_knowledge_items * 100)}%)</span>
                      </div>
                      <div className="progress" style={{ height: '10px' }}>
                        <div
                          className={`progress-bar bg-${STAGE_COLORS[s.stage] || 'secondary'}`}
                          style={{ width: `${Math.round(s.count / ov.total_knowledge_items * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-6 mb-3">
              <div className="card">
                <div className="card-header fw-semibold">Knowledge Type Distribution</div>
                <div className="card-body">
                  {(ov.type_distribution || []).map(t => (
                    <div key={t.type} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{TYPE_ICONS[t.type] || '📄'} {t.type}</span>
                        <span className="fw-bold">{t.count}</span>
                      </div>
                      <div className="progress" style={{ height: '8px' }}>
                        <div
                          className="progress-bar bg-primary"
                          style={{ width: `${Math.round(t.count / ov.total_knowledge_items * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Source breakdown */}
          <div className="card mb-3">
            <div className="card-header fw-semibold">Source Table Breakdown</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr>
                      <th>Source Table</th>
                      <th className="text-end">Items</th>
                      <th className="text-end">Share</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.source_breakdown || []).map(s => (
                      <tr key={s.source}>
                        <td><code>{s.source}</code></td>
                        <td className="text-end fw-bold">{s.count}</td>
                        <td className="text-end">{Math.round(s.count / ov.total_knowledge_items * 100)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Activity trend */}
          {(ov.activity_trend || []).length > 0 && (
            <div className="card mb-3">
              <div className="card-header fw-semibold">Daily Knowledge Activity (last 20 days)</div>
              <div className="card-body">
                <div className="d-flex align-items-end gap-1" style={{ height: '80px', overflowX: 'auto' }}>
                  {(ov.activity_trend || []).slice(-20).map(d => {
                    const max = Math.max(...(ov.activity_trend || []).map(x => x.events), 1);
                    const h = Math.max(4, Math.round((d.events / max) * 72));
                    return (
                      <div key={d.date} title={`${d.date}: ${d.events} events`}
                        className="bg-info rounded" style={{ width: '14px', height: `${h}px`, flexShrink: 0 }} />
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* REGISTER TAB */}
      {tab === 'register' && (
        <div>
          <div className="d-flex gap-2 mb-3 flex-wrap">
            <input
              type="text"
              className="form-control form-control-sm"
              placeholder="Search title or patient…"
              value={regSearch}
              onChange={e => setRegSearch(e.target.value)}
              style={{ maxWidth: 280 }}
            />
            <select className="form-select form-select-sm" style={{ maxWidth: 180 }}
              value={regFilter} onChange={e => setRegFilter(e.target.value)}>
              <option value="all">All stages</option>
              {['created', 'approved', 'published', 'expired', 'archived'].map(s =>
                <option key={s} value={s}>{s}</option>
              )}
            </select>
            <span className="text-muted small align-self-center">
              {register.length} / {bd?.total_register_items} items
            </span>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-striped">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Type</th>
                  <th>Title</th>
                  <th>Patient</th>
                  <th>Stage</th>
                  <th>Confidence</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {register.slice(0, 50).map(item => (
                  <tr key={item.id}>
                    <td><code className="small">{item.id}</code></td>
                    <td className="small">{TYPE_ICONS[item.type] || '📄'} {item.type}</td>
                    <td className="small" style={{ maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {item.title}
                    </td>
                    <td className="small">{item.patient_id || '—'}</td>
                    <td><StageBadge stage={item.stage} /></td>
                    <td className="small">
                      {item.confidence != null ? `${(item.confidence * 100).toFixed(0)}%` : '—'}
                    </td>
                    <td className="small text-muted">{(item.created_at || '').slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {register.length > 50 && (
            <p className="text-muted small">Showing 50 of {register.length} matching items.</p>
          )}
        </div>
      )}

      {/* PER PATIENT TAB */}
      {tab === 'patients' && (
        <div>
          <p className="text-muted small">{bd?.patient_profiles?.length} patients with knowledge items</p>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-striped">
              <thead>
                <tr>
                  <th>Patient</th>
                  <th className="text-end">Items</th>
                  <th>Types</th>
                  <th>Stages</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.patient_profiles || []).slice(0, 60).map(p => (
                  <tr key={p.patient_id}>
                    <td><code className="small">{p.patient_id}</code></td>
                    <td className="text-end fw-bold">{p.total_items}</td>
                    <td className="small">
                      {(p.types || []).map(t => (
                        <span key={t} className="badge bg-light text-dark me-1 border">
                          {TYPE_ICONS[t] || '📄'} {t}
                        </span>
                      ))}
                    </td>
                    <td>
                      {(p.stages || []).map(s => <StageBadge key={s} stage={s} />)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* LIFECYCLE EVENTS TAB */}
      {tab === 'lifecycle' && (
        <div>
          <p className="text-muted small">{bd?.total_lifecycle_events} total lifecycle events from transaction_log</p>

          {(bd?.stage_flow || []).length > 0 && (
            <div className="card mb-3">
              <div className="card-header fw-semibold">Top Stage Transitions</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Transition</th><th className="text-end">Count</th></tr></thead>
                    <tbody>
                      {(bd?.stage_flow || []).slice(0, 15).map(f => (
                        <tr key={f.transition}>
                          <td className="small"><code>{f.transition}</code></td>
                          <td className="text-end fw-bold">{f.count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead>
                <tr>
                  <th>Component</th>
                  <th>Action</th>
                  <th>Stage</th>
                  <th>Actor</th>
                  <th>Patient</th>
                  <th>Detail</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.lifecycle_events || []).slice(0, 40).map((e, i) => (
                  <tr key={i}>
                    <td className="small">{e.component}</td>
                    <td className="small"><code>{e.action}</code></td>
                    <td><StageBadge stage={e.stage} /></td>
                    <td className="small">{e.actor || '—'}</td>
                    <td className="small">{e.patient_id || '—'}</td>
                    <td className="small text-muted" style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {e.detail}
                    </td>
                    <td className="small text-muted">{(e.timestamp || '').slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="row">
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header fw-semibold">Concepts</div>
                <div className="card-body p-0">
                  <ul className="list-group list-group-flush">
                    {(defs.concepts || []).map(c => (
                      <li key={c.term} className="list-group-item">
                        <div className="fw-semibold small">{c.term}</div>
                        <div className="text-muted small">{c.definition}</div>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card mb-3">
                <div className="card-header fw-semibold">Metrics</div>
                <div className="card-body p-0">
                  <ul className="list-group list-group-flush">
                    {(defs.metrics || []).map(m => (
                      <li key={m.name} className="list-group-item">
                        <div className="fw-semibold small">{m.name}</div>
                        <div className="text-muted small">{m.description}</div>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="card">
                <div className="card-header fw-semibold">Compliance References</div>
                <div className="card-body p-0">
                  <ul className="list-group list-group-flush">
                    {(defs.compliance || []).map((c, i) => (
                      <li key={i} className="list-group-item small text-muted">{c}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
          <div className="card mt-2">
            <div className="card-header fw-semibold">Remediation Guidance</div>
            <ul className="list-group list-group-flush">
              {(defs.remediation || []).map((r, i) => (
                <li key={i} className="list-group-item small">{r}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
