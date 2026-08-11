'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'messages',    label: 'Message Log' },
  { id: 'patients',    label: 'Per Patient' },
  { id: 'categories',  label: 'By Category' },
  { id: 'definitions', label: 'Definitions' },
];

const PRIORITY_COLOR = {
  urgent: 'danger',
  high:   'warning',
  normal: 'primary',
  low:    'secondary',
};

const CAT_EMOJI = {
  'urgent':              '🚨',
  'side-effect-report':  '⚠️',
  'symptom-report':      '🩺',
  'medication-question': '💊',
  'prescription-refill': '🔄',
  'test-results':        '🔬',
  'appointment-request': '📅',
  'general-inquiry':     '💬',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ label, value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <small>{label}</small>
        <small className="fw-bold">{value}</small>
      </div>
      <div className="progress" style={{ height: '8px' }}>
        <div className={`progress-bar bg-${color || 'primary'}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function SecureMessagesPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [catFilter,  setCatFilter]  = useState('all');
  const [dirFilter,  setDirFilter]  = useState('all');
  const [priFilter,  setPriFilter]  = useState('all');
  const [sortPt,     setSortPt]     = useState('messages_desc');

  useEffect(() => {
    fetch(`${API}/api/secure-messages/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/secure-messages/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/secure-messages/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return (
    <div className="p-4">
      <div className="spinner-border text-primary" />
      <span className="ms-2">Loading Secure Messages…</span>
    </div>
  );

  const kpi       = ov.kpis || {};
  const catDist   = ov.category_dist || [];
  const priDist   = ov.priority_dist || [];
  const dirDist   = ov.direction_dist || [];
  const monthly   = ov.monthly_trend || [];
  const avgResp   = ov.avg_response_by_priority || [];
  const catByDir  = ov.category_by_direction || [];

  const messages  = bd?.messages || [];
  const byPatient = bd?.by_patient || [];
  const byCat     = bd?.by_category || [];

  // Filter messages
  const filtered = messages.filter(m => {
    if (catFilter !== 'all' && m.category !== catFilter) return false;
    if (dirFilter !== 'all' && m.direction !== dirFilter) return false;
    if (priFilter !== 'all' && m.priority !== priFilter) return false;
    return true;
  });

  // Sort patients
  const sortedPt = [...byPatient].sort((a, b) => {
    if (sortPt === 'messages_desc') return b.messages - a.messages;
    if (sortPt === 'unread_desc')   return b.unread - a.unread;
    if (sortPt === 'urgent_desc')   return b.urgent_count - a.urgent_count;
    return a.patient_id.localeCompare(b.patient_id);
  });

  const maxCat = Math.max(...catDist.map(c => c.count), 1);
  const maxMon = Math.max(...monthly.map(m => m.count), 1);

  const cats    = [...new Set(messages.map(m => m.category))].sort();
  const urgentPts = byPatient.filter(p => p.urgent_count > 0)
    .sort((a, b) => b.urgent_count - a.urgent_count);

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-3">
        <span style={{ fontSize: '1.8rem' }} className="me-2">💬</span>
        <div>
          <h4 className="mb-0 fw-bold">Secure Messages</h4>
          <small className="text-muted">
            {kpi.total_messages} messages · {kpi.total_patients} patients ·
            {' '}{kpi.inbound_count} inbound / {kpi.outbound_count} outbound ·
            {' '}{kpi.unread_count} unread ({kpi.unread_rate}%) ·
            avg response {kpi.avg_response_time_hours}h
          </small>
        </div>
      </div>

      {/* Urgent alert */}
      {kpi.urgent_count > 0 && (
        <div className="alert alert-danger d-flex align-items-center mb-3" role="alert">
          <span className="me-2" style={{ fontSize: '1.2rem' }}>🚨</span>
          <div>
            <strong>{kpi.urgent_count} urgent</strong> and{' '}
            <strong>{kpi.high_priority_count} high-priority</strong> messages require attention.
          </div>
        </div>
      )}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          <div className="row mb-2">
            <KPI label="Total Messages"   value={kpi.total_messages}          color="primary" />
            <KPI label="Unread"           value={`${kpi.unread_count} (${kpi.unread_rate}%)`} color="warning" />
            <KPI label="Urgent Messages"  value={kpi.urgent_count}            color="danger"  sub="Priority: urgent" />
            <KPI label="Avg Response"     value={`${kpi.avg_response_time_hours}h`} color="info" sub="inbound messages" />
          </div>
          <div className="row mb-4">
            <KPI label="Patients Messaging" value={kpi.total_patients}        color="primary" />
            <KPI label="Inbound"            value={kpi.inbound_count}         color="success" sub="patient → clinic" />
            <KPI label="Outbound"           value={kpi.outbound_count}        color="info"    sub="clinic → patient" />
            <KPI label="High Priority"      value={kpi.high_priority_count}   color="warning" />
          </div>

          <div className="row">
            {/* Category distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Messages by Category</div>
                <div className="card-body">
                  {catDist.map(c => (
                    <Bar
                      key={c.category}
                      label={`${CAT_EMOJI[c.category] || '📨'} ${c.category}`}
                      value={c.count}
                      max={maxCat}
                      color="primary"
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Priority breakdown */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Priority Distribution</div>
                <div className="card-body">
                  {priDist.map(p => (
                    <div key={p.priority} className="d-flex justify-content-between align-items-center mb-2">
                      <span className={`badge bg-${PRIORITY_COLOR[p.priority] || 'secondary'}`}>
                        {p.priority}
                      </span>
                      <div className="d-flex align-items-center gap-2">
                        <div className="progress" style={{ width: '80px', height: '8px' }}>
                          <div
                            className={`progress-bar bg-${PRIORITY_COLOR[p.priority] || 'secondary'}`}
                            style={{ width: `${Math.round(p.count / kpi.total_messages * 100)}%` }}
                          />
                        </div>
                        <span className="fw-bold small">{p.count}</span>
                      </div>
                    </div>
                  ))}
                  <hr className="my-2" />
                  <small className="text-muted fw-semibold d-block mb-1">Avg Response Time by Priority</small>
                  {avgResp.map(r => (
                    <div key={r.priority} className="d-flex justify-content-between small">
                      <span className={`badge bg-${PRIORITY_COLOR[r.priority] || 'secondary'} me-2`}>{r.priority}</span>
                      <span className="fw-bold">{r.avg_response_time ? `${r.avg_response_time}h` : '—'}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Direction split */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Inbound vs Outbound</div>
                <div className="card-body">
                  {dirDist.map(d => (
                    <div key={d.direction} className="mb-3">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="fw-semibold">
                          {d.direction === 'inbound' ? '📥 Patient → Clinic' : '📤 Clinic → Patient'}
                        </span>
                        <span className="fw-bold">{d.count}</span>
                      </div>
                      <div className="progress" style={{ height: '12px' }}>
                        <div
                          className={`progress-bar bg-${d.direction === 'inbound' ? 'success' : 'info'}`}
                          style={{ width: `${Math.round(d.count / kpi.total_messages * 100)}%` }}
                        />
                      </div>
                      <small className="text-muted">{Math.round(d.count / kpi.total_messages * 100)}% of total</small>
                    </div>
                  ))}
                  <hr className="my-2" />
                  <small className="text-muted fw-semibold d-block mb-1">Category × Direction</small>
                  <div className="table-responsive">
                    <table className="table table-sm mb-0" style={{ fontSize: '0.75rem' }}>
                      <thead className="table-light">
                        <tr><th>Category</th><th className="text-center">In</th><th className="text-center">Out</th></tr>
                      </thead>
                      <tbody>
                        {catByDir.map(c => (
                          <tr key={c.category}>
                            <td>{CAT_EMOJI[c.category] || '📨'} {c.category}</td>
                            <td className="text-center">{c.inbound}</td>
                            <td className="text-center">{c.outbound}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Monthly trend */}
          {monthly.length > 0 && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">Monthly Message Volume</div>
              <div className="card-body">
                <div className="d-flex align-items-end gap-1" style={{ height: '80px' }}>
                  {monthly.map(m => (
                    <div key={m.month} className="d-flex flex-column align-items-center flex-grow-1">
                      <div
                        className="bg-primary rounded-top w-100"
                        style={{ height: `${Math.round((m.count / maxMon) * 60)}px`, minHeight: '4px' }}
                        title={`${m.month}: ${m.count}`}
                      />
                      <small className="text-muted" style={{ fontSize: '0.6rem', transform: 'rotate(-30deg)', marginTop: '2px' }}>
                        {m.month?.slice(2)}
                      </small>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Urgent patient list */}
          {urgentPts.length > 0 && (
            <div className="card shadow-sm">
              <div className="card-header fw-semibold text-danger">🚨 Patients with Urgent Messages</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Patient</th>
                      <th className="text-center">Urgent</th>
                      <th className="text-center">Total</th>
                      <th className="text-center">Unread</th>
                      <th className="text-end">Avg Response</th>
                    </tr>
                  </thead>
                  <tbody>
                    {urgentPts.map(p => (
                      <tr key={p.patient_id}>
                        <td className="fw-semibold">{p.patient_id}</td>
                        <td className="text-center"><span className="badge bg-danger">{p.urgent_count}</span></td>
                        <td className="text-center">{p.messages}</td>
                        <td className="text-center">
                          {p.unread > 0 ? <span className="badge bg-warning text-dark">{p.unread}</span> : '—'}
                        </td>
                        <td className="text-end small">{p.avg_response_time ? `${p.avg_response_time}h` : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── MESSAGE LOG ── */}
      {tab === 'messages' && (
        <>
          <div className="d-flex gap-2 mb-3 flex-wrap">
            <select className="form-select form-select-sm w-auto" value={catFilter} onChange={e => setCatFilter(e.target.value)}>
              <option value="all">All Categories</option>
              {cats.map(c => <option key={c} value={c}>{CAT_EMOJI[c] || ''} {c}</option>)}
            </select>
            <select className="form-select form-select-sm w-auto" value={dirFilter} onChange={e => setDirFilter(e.target.value)}>
              <option value="all">All Directions</option>
              <option value="inbound">Inbound</option>
              <option value="outbound">Outbound</option>
            </select>
            <select className="form-select form-select-sm w-auto" value={priFilter} onChange={e => setPriFilter(e.target.value)}>
              <option value="all">All Priorities</option>
              <option value="urgent">Urgent</option>
              <option value="high">High</option>
              <option value="normal">Normal</option>
              <option value="low">Low</option>
            </select>
            <span className="text-muted small align-self-center">{filtered.length} / {messages.length} messages</span>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-hover align-middle">
              <thead className="table-dark sticky-top">
                <tr>
                  <th>#</th>
                  <th>Patient</th>
                  <th>Direction</th>
                  <th>Category</th>
                  <th>Subject</th>
                  <th>Priority</th>
                  <th>Read</th>
                  <th>Response</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(m => (
                  <tr key={m.id}>
                    <td className="text-muted small">{m.id}</td>
                    <td className="fw-semibold small">{m.patient_id}</td>
                    <td>
                      <span className={`badge bg-${m.direction === 'inbound' ? 'success' : 'info'}`}>
                        {m.direction === 'inbound' ? '📥' : '📤'} {m.direction}
                      </span>
                    </td>
                    <td><small>{CAT_EMOJI[m.category] || ''} {m.category}</small></td>
                    <td style={{ maxWidth: '180px' }}>
                      <small className="fw-semibold d-block">{m.subject}</small>
                      <small className="text-muted" style={{ fontSize: '0.7rem' }}>{m.message_preview}</small>
                    </td>
                    <td>
                      <span className={`badge bg-${PRIORITY_COLOR[m.priority] || 'secondary'}`}>{m.priority}</span>
                    </td>
                    <td className="text-center">
                      {m.read_status === 'unread'
                        ? <span className="badge bg-warning text-dark">unread</span>
                        : <span className="text-muted small">✓</span>}
                    </td>
                    <td className="small">{m.response_time_hours ? `${m.response_time_hours}h` : '—'}</td>
                    <td className="small text-muted">{m.created_at?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && (
        <>
          <div className="d-flex gap-2 mb-3">
            <select className="form-select form-select-sm w-auto" value={sortPt} onChange={e => setSortPt(e.target.value)}>
              <option value="messages_desc">Most Messages</option>
              <option value="unread_desc">Most Unread</option>
              <option value="urgent_desc">Most Urgent</option>
              <option value="id">Patient ID</option>
            </select>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-hover align-middle">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th className="text-center">Total</th>
                  <th className="text-center">Inbound</th>
                  <th className="text-center">Outbound</th>
                  <th className="text-center">Unread</th>
                  <th className="text-center">Urgent</th>
                  <th className="text-end">Avg Response</th>
                </tr>
              </thead>
              <tbody>
                {sortedPt.map(p => (
                  <tr key={p.patient_id}>
                    <td className="fw-semibold">{p.patient_id}</td>
                    <td className="text-center">{p.messages}</td>
                    <td className="text-center"><span className="badge bg-success">{p.inbound}</span></td>
                    <td className="text-center"><span className="badge bg-info">{p.outbound}</span></td>
                    <td className="text-center">
                      {p.unread > 0 ? <span className="badge bg-warning text-dark">{p.unread}</span> : '—'}
                    </td>
                    <td className="text-center">
                      {p.urgent_count > 0 ? <span className="badge bg-danger">{p.urgent_count}</span> : '—'}
                    </td>
                    <td className="text-end small">{p.avg_response_time != null ? `${p.avg_response_time}h` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── BY CATEGORY ── */}
      {tab === 'categories' && (
        <>
          <div className="table-responsive mb-3">
            <table className="table table-sm table-hover align-middle">
              <thead className="table-dark">
                <tr>
                  <th>Category</th>
                  <th className="text-center">Total</th>
                  <th className="text-center">Inbound</th>
                  <th className="text-center">Outbound</th>
                  <th className="text-center">Unread</th>
                  <th className="text-end">Avg Response (h)</th>
                </tr>
              </thead>
              <tbody>
                {byCat
                  .sort((a, b) => b.total - a.total)
                  .map(c => (
                    <tr key={c.category}>
                      <td className="fw-semibold">
                        {CAT_EMOJI[c.category] || '📨'} {c.category}
                      </td>
                      <td className="text-center">{c.total}</td>
                      <td className="text-center"><span className="badge bg-success">{c.inbound}</span></td>
                      <td className="text-center"><span className="badge bg-info">{c.outbound}</span></td>
                      <td className="text-center">
                        {c.unread_count > 0 ? <span className="badge bg-warning text-dark">{c.unread_count}</span> : '—'}
                      </td>
                      <td className="text-end">{c.avg_response_time != null ? `${c.avg_response_time}h` : '—'}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {/* Category × Direction breakdown */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Inbound vs Outbound by Category</div>
            <div className="card-body">
              {catByDir.map(c => {
                const total = c.inbound + c.outbound;
                const inPct = total ? Math.round(c.inbound / total * 100) : 0;
                return (
                  <div key={c.category} className="mb-2">
                    <div className="d-flex justify-content-between mb-1">
                      <small>{CAT_EMOJI[c.category] || ''} {c.category}</small>
                      <small className="text-muted">{c.inbound} in / {c.outbound} out</small>
                    </div>
                    <div className="progress" style={{ height: '10px' }}>
                      <div className="progress-bar bg-success" style={{ width: `${inPct}%` }} />
                      <div className="progress-bar bg-info" style={{ width: `${100 - inPct}%` }} />
                    </div>
                  </div>
                );
              })}
              <div className="d-flex gap-3 mt-2">
                <small><span className="badge bg-success me-1">■</span>Inbound</small>
                <small><span className="badge bg-info me-1">■</span>Outbound</small>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">💬 {defs.title}</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-dark">
                  <tr><th>Concept</th><th>Description</th></tr>
                </thead>
                <tbody>
                  {(defs.concepts || []).map(c => (
                    <tr key={c.name}>
                      <td className="fw-semibold" style={{ whiteSpace: 'nowrap' }}>{c.name}</td>
                      <td><small>{c.description}</small></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Data Sources &amp; Endpoints</div>
            <div className="card-body">
              <ul className="mb-2 small">
                {(defs.data_sources || []).map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
              <ul className="mb-0 small">
                <li><code>/api/secure-messages/overview</code> — KPIs, category/priority/direction distribution, monthly trend</li>
                <li><code>/api/secure-messages/breakdown</code> — full message log, per-patient summary, by-category stats</li>
                <li><code>/api/secure-messages/definitions</code> — concept glossary, data sources</li>
              </ul>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
