'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',  label: 'Overview' },
  { id: 'daily',     label: 'Daily Detail' },
  { id: 'messages',  label: 'Recent Messages' },
  { id: 'definitions', label: 'Definitions' },
];

const ROLE_COLOR = { assistant: 'primary', operator: 'success' };
const ROLE_ICON  = { assistant: '🤖', operator: '🧑‍💻' };

const BUCKET_COLOR = (b) => {
  if (b === '< 100') return '#94a3b8';
  if (b === '100-299') return '#3b82f6';
  if (b === '300-499') return '#8b5cf6';
  if (b === '500-999') return '#f59e0b';
  if (b === '1000-1999') return '#ef4444';
  return '#dc2626';
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

function MiniBar({ label, value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-semibold">{value}</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color || '#3b82f6' }} />
      </div>
    </div>
  );
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  const dr = ov.date_range || {};
  const roles = ov.role_distribution || [];
  const stats = ov.text_stats || [];
  const daily = ov.daily_trend || [];
  const hourly = ov.hourly_pattern || [];

  const assistantRow = stats.find(s => s.role === 'assistant') || {};
  const operatorRow  = stats.find(s => s.role === 'operator')  || {};
  const totalKB = ((assistantRow.total_kb || 0) + (operatorRow.total_kb || 0)).toFixed(1);

  const maxDay = Math.max(...daily.map(d => d.total), 1);
  const maxHr  = Math.max(...hourly.map(h => h.cnt), 1);

  return (
    <div>
      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Total Messages" value={ov.total_messages?.toLocaleString()} color="primary" sub="all roles combined" />
        <KPI label="Active Days" value={dr.active_days} color="info" sub={`${dr.first_date} → ${dr.last_date}`} />
        <KPI label="Avg / Day" value={ov.avg_messages_per_day} color="success" sub="messages per active day" />
        <KPI label="AI:Operator Ratio" value={`${ov.assistant_to_operator_ratio}:1`} color="warning" sub="assistant msgs per human msg" />
      </div>
      <div className="row mb-4">
        <KPI label="Operator Messages" value={(roles.find(r => r.role === 'operator')?.cnt || 0).toLocaleString()} color="success" sub="human prompts" />
        <KPI label="Assistant Messages" value={(roles.find(r => r.role === 'assistant')?.cnt || 0).toLocaleString()} color="primary" sub="AI responses" />
        <KPI label="Total Size" value={`${totalKB} KB`} color="secondary" sub={`${(ov.total_characters || 0).toLocaleString()} characters`} />
        <KPI label="Avg Operator Length" value={`${operatorRow.avg_len || 0} ch`} color="info" sub={`max ${operatorRow.max_len || 0} ch`} />
      </div>

      <div className="row">
        {/* Daily trend */}
        <div className="col-md-8 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">📅 Daily Message Volume (all active days)</div>
            <div className="card-body" style={{ overflowX: 'auto' }}>
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 120, minWidth: daily.length * 16 }}>
                {daily.map(d => {
                  const h = Math.max(4, Math.round((d.total / maxDay) * 110));
                  return (
                    <div key={d.day} title={`${d.day}: ${d.total} msgs (op: ${d.operator_msgs}, ai: ${d.assistant_msgs})`}
                         style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1 }}>
                      <div style={{ width: '100%', height: h, backgroundColor: '#3b82f6', borderRadius: 2 }} />
                      <div style={{ fontSize: 8, transform: 'rotate(-60deg)', marginTop: 4, whiteSpace: 'nowrap', color: '#64748b' }}>
                        {d.day.slice(5)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Role breakdown */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">👥 Role Breakdown</div>
            <div className="card-body">
              {roles.map(r => (
                <MiniBar key={r.role}
                  label={`${ROLE_ICON[r.role] || '❓'} ${r.role}`}
                  value={r.cnt.toLocaleString()}
                  max={ov.total_messages || 1}
                  color={ROLE_COLOR[r.role] === 'primary' ? '#3b82f6' : '#22c55e'}
                />
              ))}
              <hr />
              <div className="small text-muted">
                <strong>Text stats</strong>
                <div className="table-responsive mt-1">
                  <table className="table table-sm table-borderless mb-0">
                    <thead><tr><th>Role</th><th>Avg</th><th>Max</th><th>KB</th></tr></thead>
                    <tbody>
                      {stats.map(s => (
                        <tr key={s.role}>
                          <td>{s.role}</td>
                          <td>{s.avg_len}</td>
                          <td>{s.max_len}</td>
                          <td>{s.total_kb}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Hourly heatmap */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-semibold">🕐 Hourly Message Pattern (UTC)</div>
        <div className="card-body">
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {hourly.map(h => {
              const intensity = Math.round((h.cnt / maxHr) * 255);
              const bg = `rgb(${255 - intensity}, ${255 - intensity}, 255)`;
              return (
                <div key={h.hour} title={`Hour ${h.hour}:00 — ${h.cnt} msgs`}
                     style={{ width: 36, height: 36, background: bg, border: '1px solid #e2e8f0',
                              borderRadius: 4, display: 'flex', flexDirection: 'column',
                              alignItems: 'center', justifyContent: 'center', fontSize: 10 }}>
                  <div style={{ fontWeight: 600, color: intensity > 128 ? '#fff' : '#1e293b' }}>{h.hour}</div>
                  <div style={{ color: intensity > 128 ? '#dbeafe' : '#64748b' }}>{h.cnt}</div>
                </div>
              );
            })}
          </div>
          <div className="text-muted small mt-2">Each cell = hour of day (0–23 UTC). Darker = more messages.</div>
        </div>
      </div>
    </div>
  );
}

function DailyPanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const daily = bd.daily_detail || [];
  const lenDist = bd.length_distribution || [];
  const longest = bd.longest_messages || [];

  const maxChars = Math.max(...daily.map(d => d.total_chars), 1);

  return (
    <div>
      <div className="row mb-3">
        {/* Length distribution */}
        <div className="col-md-5 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">📏 Message Length Distribution</div>
            <div className="card-body">
              {lenDist.map(l => (
                <MiniBar key={l.bucket}
                  label={`${l.bucket} chars`}
                  value={l.cnt.toLocaleString()}
                  max={Math.max(...lenDist.map(x => x.cnt), 1)}
                  color={BUCKET_COLOR(l.bucket)}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Longest messages */}
        <div className="col-md-7 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">📖 Longest Messages</div>
            <div className="card-body" style={{ overflowY: 'auto', maxHeight: 280 }}>
              <table className="table table-sm table-hover mb-0">
                <thead><tr><th>Role</th><th>Length</th><th>Date</th></tr></thead>
                <tbody>
                  {longest.map((m, i) => (
                    <tr key={i}>
                      <td><span className={`badge bg-${ROLE_COLOR[m.role] || 'secondary'}`}>{ROLE_ICON[m.role]} {m.role}</span></td>
                      <td>{m.text_length?.toLocaleString()} ch</td>
                      <td className="text-muted small">{m.ts_utc?.slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Day-by-day table */}
      <div className="card shadow-sm">
        <div className="card-header fw-semibold">📋 Per-Day Breakdown</div>
        <div className="card-body p-0" style={{ overflowY: 'auto', maxHeight: 400 }}>
          <table className="table table-sm table-hover table-striped mb-0">
            <thead className="table-dark">
              <tr>
                <th>Date</th>
                <th className="text-center">Total</th>
                <th className="text-center">🧑‍💻 Operator</th>
                <th className="text-center">🤖 Assistant</th>
                <th className="text-center">Avg Len</th>
                <th className="text-end">Total Chars</th>
              </tr>
            </thead>
            <tbody>
              {[...daily].reverse().map(d => (
                <tr key={d.day}>
                  <td className="fw-semibold">{d.day}</td>
                  <td className="text-center">
                    <span className="badge bg-dark">{d.total}</span>
                  </td>
                  <td className="text-center">
                    {d.operator_msgs > 0
                      ? <span className="badge bg-success">{d.operator_msgs}</span>
                      : <span className="text-muted">—</span>}
                  </td>
                  <td className="text-center">
                    <span className="badge bg-primary">{d.assistant_msgs}</span>
                  </td>
                  <td className="text-center text-muted small">{d.avg_text_len}</td>
                  <td className="text-end text-muted small">
                    <div className="d-flex align-items-center justify-content-end gap-2">
                      <div className="progress flex-grow-1" style={{ height: 6, maxWidth: 60 }}>
                        <div className="progress-bar bg-info"
                             style={{ width: `${Math.round((d.total_chars / maxChars) * 100)}%` }} />
                      </div>
                      {d.total_chars?.toLocaleString()}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function MessagesPanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const recent = bd.recent_messages || [];
  const opRecent = bd.operator_recent || [];

  return (
    <div>
      <div className="row">
        {/* Recent all messages */}
        <div className="col-md-7 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">💬 Recent Messages (all roles)</div>
            <div className="card-body p-0" style={{ overflowY: 'auto', maxHeight: 450 }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr><th>#</th><th>Role</th><th>Preview</th><th>Len</th><th>Time (UTC)</th></tr>
                </thead>
                <tbody>
                  {recent.map(m => (
                    <tr key={m.id}>
                      <td className="text-muted small">{m.id}</td>
                      <td>
                        <span className={`badge bg-${ROLE_COLOR[m.role] || 'secondary'}`}>
                          {ROLE_ICON[m.role]} {m.role}
                        </span>
                      </td>
                      <td className="small text-truncate" style={{ maxWidth: 260 }}
                          title={m.text_preview}>{m.text_preview}</td>
                      <td className="text-muted small">{m.text_length}</td>
                      <td className="text-muted small" style={{ whiteSpace: 'nowrap' }}>
                        {m.ts_utc?.slice(0, 16)?.replace('T', ' ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Recent operator messages */}
        <div className="col-md-5 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">🧑‍💻 Recent Operator Messages</div>
            <div className="card-body p-0" style={{ overflowY: 'auto', maxHeight: 450 }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr><th>#</th><th>Preview</th><th>Time</th></tr>
                </thead>
                <tbody>
                  {opRecent.map(m => (
                    <tr key={m.id}>
                      <td className="text-muted small">{m.id}</td>
                      <td className="small" title={m.text_preview}
                          style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {m.text_preview}
                      </td>
                      <td className="text-muted small" style={{ whiteSpace: 'nowrap' }}>
                        {m.ts_utc?.slice(0, 10)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">📐 Field Definitions</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
                <tbody>
                  {(defs.fields || []).map(f => (
                    <tr key={f.field}>
                      <td className="fw-semibold"><code>{f.field}</code></td>
                      <td><span className="badge bg-secondary">{f.type}</span></td>
                      <td className="small text-muted">{f.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">👥 Roles</div>
            <div className="card-body">
              {Object.entries(defs.roles || {}).map(([role, desc]) => (
                <div key={role} className="mb-2">
                  <span className={`badge bg-${ROLE_COLOR[role] || 'secondary'} me-2`}>
                    {ROLE_ICON[role]} {role}
                  </span>
                  <span className="small text-muted">{desc}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">📊 Metrics</div>
            <div className="card-body">
              <dl className="mb-0">
                {Object.entries(defs.metrics || {}).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <dt className="small fw-semibold"><code>{k}</code></dt>
                    <dd className="small text-muted mb-0">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </div>
      </div>
      <div className="card shadow-sm">
        <div className="card-header fw-semibold">🗄️ Data Sources</div>
        <div className="card-body">
          <ul className="mb-0 small">
            {(defs.data_sources || []).map((s, i) => <li key={i}>{s}</li>)}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default function ConversationLogPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/conversation-log/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/conversation-log/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/conversation-log/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  return (
    <div>
      <h3>💬 Conversation Log Dashboard</h3>
      <p className="text-muted small">
        Analytics over the AI–operator conversation history logged in <code>conversation_log</code>.
        {' '}{(ov.total_messages || 0).toLocaleString()} messages across {ov.date_range?.active_days} active days
        ({ov.date_range?.first_date} → {ov.date_range?.last_date}).
        AI-to-operator ratio: {ov.assistant_to_operator_ratio}:1.
      </p>

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

      {tab === 'overview'     && <OverviewPanel    ov={ov} />}
      {tab === 'daily'        && <DailyPanel       bd={bd} />}
      {tab === 'messages'     && <MessagesPanel    bd={bd} />}
      {tab === 'definitions'  && <DefinitionsPanel defs={defs} />}
    </div>
  );
}
