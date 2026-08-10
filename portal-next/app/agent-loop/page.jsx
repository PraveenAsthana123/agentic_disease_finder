'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

/* ── colour helpers ─────────────────────────────────────────────────────── */
const driftColor = s =>
  s === 'low'    ? '#22c55e' :
  s === 'medium' ? '#f59e0b' :
  s === 'high'   ? '#ef4444' : '#6b7280';

const ratioColor = r =>
  r < 5  ? '#22c55e' :
  r < 15 ? '#f59e0b' : '#ef4444';

/* ── shared stat card ───────────────────────────────────────────────────── */
function StatCard({ label, value, unit = '', sub = '', color = '#3b82f6' }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className="fw-bold" style={{ fontSize: '1.6rem', color }}>{value ?? '—'}{unit}</div>
          <div className="text-muted small mt-1">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

/* ── mini bar ───────────────────────────────────────────────────────────── */
function MiniBar({ value, max, color = '#3b82f6' }) {
  const pct = max > 0 ? Math.min(100, Math.round((value / max) * 100)) : 0;
  return (
    <div className="progress" style={{ height: 10 }}>
      <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   TAB: OVERVIEW
═══════════════════════════════════════════════════════════════════════════ */
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;
  if (!ov.available) return <div className="alert alert-info">Agent loop monitor not available.</div>;

  const s   = ov.summary || {};
  const byC = ov.actions_by_component || [];
  const trend = ov.daily_trend || [];
  const maxAct = Math.max(...byC.map(r => r.actions), 1);

  return (
    <div>
      {/* KPI row 1 */}
      <div className="row mb-2">
        <StatCard label="Total Agent Actions" value={s.total_agent_actions?.toLocaleString()}
          sub="all transaction_log rows" color="#3b82f6" />
        <StatCard label="Active Components"  value={s.active_components}
          sub="distinct pipeline nodes" color="#8b5cf6" />
        <StatCard label="Conversation Turns" value={s.conversation_turns?.toLocaleString()}
          sub="operator + assistant" color="#0ea5e9" />
        <StatCard label="Loop Ratio"         value={s.loop_ratio?.toFixed(1)}
          sub="assistant:operator turns" color={ratioColor(s.loop_ratio)} />
      </div>

      {/* KPI row 2 */}
      <div className="row mb-3">
        <StatCard label="Assistant Turns"   value={s.assistant_turns?.toLocaleString()}
          sub="AI-generated messages" color="#22c55e" />
        <StatCard label="Operator Turns"    value={s.operator_turns?.toLocaleString()}
          sub="human-directed prompts" color="#64748b" />
        <StatCard label="Blocked Actions"   value={s.blocked_actions}
          sub={`block rate ${s.block_rate_pct?.toFixed(1)}%`}
          color={s.blocked_actions > 0 ? '#ef4444' : '#22c55e'} />
        <StatCard label="Goal Drift Score"  value={`${s.goal_drift_score?.toFixed(0)}/100`}
          sub={`severity: ${s.drift_severity}`}
          color={driftColor(s.drift_severity)} />
      </div>

      {/* KPI row 3 */}
      <div className="row mb-4">
        <StatCard label="Avg AI Confidence"  value={`${((s.avg_ai_confidence||0)*100).toFixed(0)}%`}
          sub="clinical decision confidence" color="#f59e0b" />
        <StatCard label="High-Conf Decisions" value={s.high_confidence_decisions}
          sub="≥70% confidence" color="#22c55e" />
        <StatCard label="Low-Conf Decisions"  value={s.low_confidence_decisions}
          sub="<70% confidence"
          color={s.low_confidence_decisions > 0 ? '#ef4444' : '#22c55e'} />
        <StatCard label="HITL Override Rate"  value={`${s.hitl_override_rate_pct?.toFixed(1)}%`}
          sub="human overrides vs reviews" color="#6366f1" />
      </div>

      {/* Drift severity alert */}
      {s.drift_severity === 'high' && (
        <div className="alert alert-danger d-flex align-items-center mb-4">
          <span className="me-2 fs-5">⚠️</span>
          <div>
            <strong>High Goal Drift Detected</strong> — Loop ratio {s.loop_ratio?.toFixed(1)}:1
            (assistant:operator). Review conversation patterns to ensure the agent remains
            aligned with operator intent.
          </div>
        </div>
      )}

      <div className="row">
        {/* Component activity */}
        <div className="col-md-7 mb-4">
          <div className="card h-100">
            <div className="card-header fw-semibold">🔧 Actions by Component</div>
            <div className="card-body p-0">
              <div style={{ maxHeight: 340, overflowY: 'auto' }}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light sticky-top">
                    <tr><th>Component</th><th>Actions</th><th style={{ width: 120 }}>Load</th></tr>
                  </thead>
                  <tbody>
                    {byC.map(r => (
                      <tr key={r.component}>
                        <td className="font-monospace small">{r.component}</td>
                        <td className="fw-bold">{r.actions.toLocaleString()}</td>
                        <td><MiniBar value={r.actions} max={maxAct} color="#3b82f6" /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        {/* Daily trend */}
        <div className="col-md-5 mb-4">
          <div className="card h-100">
            <div className="card-header fw-semibold">📅 Daily Action Trend (last 30 days)</div>
            <div className="card-body p-2">
              {trend.length === 0
                ? <div className="text-muted small">No trend data.</div>
                : (
                  <div style={{ maxHeight: 320, overflowY: 'auto' }}>
                    <table className="table table-sm mb-0">
                      <thead className="table-light sticky-top">
                        <tr><th>Date</th><th>Actions</th><th>Feedback</th></tr>
                      </thead>
                      <tbody>
                        {trend.slice(-30).map(r => (
                          <tr key={r.date}>
                            <td className="small">{r.date}</td>
                            <td className="fw-bold">{r.actions.toLocaleString()}</td>
                            <td>{r.feedback}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
            </div>
          </div>
        </div>
      </div>

      {/* Turn breakdown */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">💬 Conversation Turn Analysis</div>
        <div className="card-body">
          <div className="row">
            <div className="col-md-4 mb-3">
              <div className="text-muted small mb-1">Turn Distribution</div>
              <div className="progress" style={{ height: 22 }}>
                <div
                  className="progress-bar bg-success"
                  style={{ width: `${Math.round(s.assistant_turns / Math.max(s.conversation_turns,1) * 100)}%` }}
                  title="Assistant turns"
                >
                  AI {Math.round(s.assistant_turns / Math.max(s.conversation_turns,1) * 100)}%
                </div>
                <div
                  className="progress-bar bg-secondary"
                  style={{ width: `${Math.round(s.operator_turns / Math.max(s.conversation_turns,1) * 100)}%` }}
                  title="Operator turns"
                >
                  Op {Math.round(s.operator_turns / Math.max(s.conversation_turns,1) * 100)}%
                </div>
              </div>
            </div>
            <div className="col-md-4 mb-3">
              <div className="text-muted small mb-1">Loop ratio interpretation</div>
              <div className="small">
                {s.loop_ratio < 5
                  ? <span className="text-success">✅ Healthy — agent responding proportionally</span>
                  : s.loop_ratio < 15
                  ? <span className="text-warning">⚠️ Moderate — review if tasks are completing</span>
                  : <span className="text-danger">🔴 High — possible runaway loop pattern</span>}
              </div>
            </div>
            <div className="col-md-4 mb-3">
              <div className="text-muted small mb-1">Alignment metrics</div>
              <div className="small">
                Agreement rate: <strong>{s.agreement_rate_pct?.toFixed(1)}%</strong><br/>
                Component alignment: <strong>{s.component_alignment_pct?.toFixed(1)}%</strong><br/>
                Correction rate: <strong>{s.correction_rate_pct?.toFixed(1)}%</strong>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   TAB: COMPONENT BREAKDOWN
═══════════════════════════════════════════════════════════════════════════ */
function BreakdownTab({ bd }) {
  const [filter, setFilter] = useState('');
  if (!bd) return <div className="text-muted p-3">Loading…</div>;
  if (!bd.available) return <div className="alert alert-info">Breakdown not available.</div>;

  const comps = (bd.components || []).filter(c =>
    !filter || c.component.toLowerCase().includes(filter.toLowerCase()));

  return (
    <div>
      <div className="mb-3">
        <input
          className="form-control form-control-sm w-50"
          placeholder="Filter by component name…"
          value={filter}
          onChange={e => setFilter(e.target.value)}
        />
      </div>
      {comps.map(comp => {
        const topActions = [...(comp.actions || [])].sort((a,b) => b.count - a.count).slice(0, 10);
        const maxCount   = Math.max(...topActions.map(a => a.count), 1);
        return (
          <div className="card mb-3" key={comp.component}>
            <div className="card-header d-flex justify-content-between align-items-center">
              <span className="fw-semibold font-monospace">{comp.component}</span>
              <span className="badge bg-primary">{comp.total?.toLocaleString()} actions</span>
            </div>
            {topActions.length > 0 && (
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>Action</th><th>Count</th><th style={{ width: 140 }}>Load</th></tr>
                  </thead>
                  <tbody>
                    {topActions.map((a, i) => (
                      <tr key={i}>
                        <td className="font-monospace small text-truncate" style={{ maxWidth: 380 }}
                          title={a.action}>{a.action}</td>
                        <td>{a.count.toLocaleString()}</td>
                        <td><MiniBar value={a.count} max={maxCount} color="#6366f1" /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );
      })}
      {comps.length === 0 && (
        <div className="text-muted">No components match the filter.</div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   TAB: DEFINITIONS
═══════════════════════════════════════════════════════════════════════════ */
function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted p-3">Loading…</div>;
  if (!defs.available) return <div className="alert alert-info">Definitions not available.</div>;

  const items = defs.definitions || [];
  return (
    <div>
      <p className="text-muted small mb-3">
        Metrics, sources, and interpretations for the Agent Loop Monitor dashboard.
        All data is read live from <code>transaction_log</code> and <code>conversation_log</code>.
      </p>
      <table className="table table-bordered table-sm">
        <thead className="table-dark">
          <tr><th style={{ width: 220 }}>Metric</th><th>Description</th><th style={{ width: 280 }}>Source</th></tr>
        </thead>
        <tbody>
          {items.map((d, i) => (
            <tr key={i}>
              <td className="fw-semibold">{d.metric}</td>
              <td className="small">{d.description}</td>
              <td className="font-monospace small text-muted">{d.source}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   PAGE ROOT
═══════════════════════════════════════════════════════════════════════════ */
const TABS = [
  { id: 'overview',   label: '📊 Overview' },
  { id: 'breakdown',  label: '🔧 Component Breakdown' },
  { id: 'definitions', label: '📖 Definitions' },
];

export default function AgentLoopPage() {
  const [tab, setTab]   = useState('overview');
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/agent-loop/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/agent-loop/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/agent-loop/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <h4 className="mb-0 me-3">🔁 Agent Loop Monitor</h4>
        <span className="badge bg-primary me-2">
          {ov?.summary?.total_agent_actions?.toLocaleString() ?? '—'} actions
        </span>
        <span className="badge bg-secondary me-2">
          {ov?.summary?.active_components ?? '—'} components
        </span>
        {ov?.summary?.drift_severity && (
          <span className={`badge bg-${
            ov.summary.drift_severity === 'high'   ? 'danger'  :
            ov.summary.drift_severity === 'medium' ? 'warning' : 'success'
          }`}>
            drift: {ov.summary.drift_severity}
          </span>
        )}
      </div>
      <p className="text-muted small mb-3">
        Monitors agent loop behaviour — action volume, component load, conversation
        turn ratios, goal drift, confidence, and HITL override rate.
        Source: <code>transaction_log</code> ({ov?.summary?.total_agent_actions?.toLocaleString() ?? '…'} rows)
        + <code>conversation_log</code> ({ov?.summary?.conversation_turns?.toLocaleString() ?? '…'} turns).
      </p>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'     && <OverviewTab    ov={ov} />}
      {tab === 'breakdown'    && <BreakdownTab   bd={bd} />}
      {tab === 'definitions'  && <DefinitionsTab defs={defs} />}
    </div>
  );
}
