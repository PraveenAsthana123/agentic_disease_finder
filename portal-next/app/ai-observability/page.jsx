'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

/* ─── colours ─────────────────────────────────────────────────────────────── */
const COMP_COLORS = {
  'http-trace':   '#3b82f6',
  'video_frames': '#8b5cf6',
  'cv_pipeline':  '#a78bfa',
  'assessment':   '#22c55e',
  'drift':        '#f59e0b',
  'training':     '#ef4444',
  'graph_db':     '#06b6d4',
  'consistency':  '#10b981',
  'fairness':     '#ec4899',
  'referral':     '#f97316',
  'seizure_diary':'#84cc16',
  'eeg_upload':   '#14b8a6',
};
const compColor = c => COMP_COLORS[c] || '#6b7280';

const actorColor = a =>
  a === 'middleware' ? '#3b82f6' :
  a === 'system'     ? '#6b7280' :
  a === 'psychiatrist'? '#8b5cf6' :
  a === 'neurologist' ? '#22c55e' :
  '#f59e0b';

const costColor = c =>
  c === 'gpu_compute'  ? '#ef4444' :
  c === 'cloud_infra'  ? '#f59e0b' :
  c === 'llm_inference'? '#3b82f6' : '#6b7280';

const confColor = v =>
  v >= 0.8 ? '#22c55e' :
  v >= 0.6 ? '#f59e0b' :
  v >= 0.4 ? '#f97316' : '#ef4444';

/* ─── stat card ───────────────────────────────────────────────────────────── */
function StatCard({ label, value, color = '#3b82f6', sub }) {
  return (
    <div className="col-6 col-md mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h5 mb-0 fw-bold" style={{ color }}>{value}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

/* ─── horizontal bar ─────────────────────────────────────────────────────── */
function HBar({ label, count, total, color }) {
  const pct = total ? Math.round((count / total) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span className="text-truncate" style={{ maxWidth: '65%' }}>{label}</span>
        <span className="fw-bold">{count.toLocaleString()} <span className="text-muted">({pct}%)</span></span>
      </div>
      <div style={{ background: '#e5e7eb', borderRadius: 4, height: 10 }}>
        <div style={{ width: `${pct}%`, background: color || '#3b82f6', borderRadius: 4, height: 10 }} />
      </div>
    </div>
  );
}

/* ─── mini badge ─────────────────────────────────────────────────────────── */
function Chip({ label, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 12,
      fontSize: '0.7rem', fontWeight: 600, color: '#fff',
      background: color || '#6b7280', marginRight: 4, marginBottom: 4,
    }}>{label}</span>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   OVERVIEW TAB
═══════════════════════════════════════════════════════════════════════════ */
function OverviewTab({ ov }) {
  const kpis = ov.kpis || {};
  const components = ov.component_distribution || [];
  const actors    = ov.actor_activity || [];
  const actions   = ov.action_distribution || [];
  const costByCat = ov.cost_by_category || [];
  const convRoles = ov.conversation_role_distribution || [];
  const totalTx   = kpis.total_transactions || 0;
  const totalCost = (kpis.total_cost_usd || 0).toFixed(2);
  const conf      = kpis.avg_confidence != null ? (kpis.avg_confidence * 100).toFixed(1) + '%' : '—';
  const totalConv = kpis.total_conversations || 0;

  return (
    <div>
      {/* KPI row */}
      <div className="row g-2 mb-3">
        <StatCard label="Transactions" value={totalTx.toLocaleString()} color="#3b82f6" />
        <StatCard label="Components" value={kpis.total_components || 0} color="#8b5cf6" />
        <StatCard label="Actors" value={kpis.total_actors || 0} color="#22c55e" />
        <StatCard label="Analyses" value={kpis.total_analyses || 0} color="#f59e0b" />
        <StatCard label="Conversations" value={totalConv.toLocaleString()} color="#06b6d4" />
        <StatCard label="Total Cost" value={`$${totalCost}`} color="#ef4444" sub="all categories" />
        <StatCard label="Avg Confidence" value={conf} color={confColor(kpis.avg_confidence || 0)} />
      </div>

      <div className="row g-3">
        {/* Component distribution */}
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-body">
              <h6 className="card-title mb-3">Component Distribution</h6>
              {components.slice(0, 12).map(c => (
                <HBar key={c.component} label={c.component} count={c.count} total={totalTx} color={compColor(c.component)} />
              ))}
            </div>
          </div>
        </div>

        {/* Actor activity + conversation roles */}
        <div className="col-md-3">
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="card-title mb-3">Actor Activity</h6>
              {actors.map(a => (
                <HBar key={a.actor} label={a.actor} count={a.count} total={totalTx} color={actorColor(a.actor)} />
              ))}
            </div>
          </div>
        </div>

        {/* Cost by category + conversation roles */}
        <div className="col-md-3">
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="card-title mb-2">Cost by Category</h6>
              {costByCat.map(c => (
                <div key={c.category} className="mb-2">
                  <div className="d-flex justify-content-between small">
                    <span>{c.category}</span>
                    <span className="fw-bold" style={{ color: costColor(c.category) }}>${c.total_cost.toFixed(2)}</span>
                  </div>
                  {c.requests > 0 && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{c.requests.toLocaleString()} requests</div>}
                </div>
              ))}
              <hr className="my-2" />
              <h6 className="mb-2 small">Conversation Roles</h6>
              {convRoles.map(r => (
                <div key={r.role} className="d-flex justify-content-between small mb-1">
                  <span>{r.role}</span>
                  <Chip label={r.count.toLocaleString()} color="#3b82f6" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Top actions */}
      <div className="card shadow-sm mt-3">
        <div className="card-body">
          <h6 className="card-title mb-3">Top Actions (by frequency)</h6>
          <div className="row">
            {actions.slice(0, 10).map(a => (
              <div key={a.action} className="col-md-6 mb-2">
                <HBar label={a.action} count={a.count} total={totalTx} color="#3b82f6" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   COMPONENTS TAB
═══════════════════════════════════════════════════════════════════════════ */
function ComponentsTab({ bk }) {
  const perComp   = bk.per_component_actions || [];
  const perActor  = bk.per_actor_components  || [];

  return (
    <div className="row g-3">
      <div className="col-md-7">
        <div className="card shadow-sm">
          <div className="card-body">
            <h6 className="card-title mb-3">Per-Component Actions</h6>
            <div style={{ overflowY: 'auto', maxHeight: 500 }}>
              <table className="table table-sm table-hover">
                <thead className="table-light sticky-top">
                  <tr>
                    <th>Component</th>
                    <th>Action</th>
                    <th className="text-end">Count</th>
                  </tr>
                </thead>
                <tbody>
                  {perComp.map(c =>
                    c.actions.map((a, i) => (
                      <tr key={`${c.component}-${a.action}`}>
                        {i === 0 && (
                          <td rowSpan={c.actions.length} className="fw-bold" style={{ color: compColor(c.component) }}>
                            {c.component}
                          </td>
                        )}
                        <td><code style={{ fontSize: '0.75rem' }}>{a.action}</code></td>
                        <td className="text-end">{a.count}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="col-md-5">
        <div className="card shadow-sm">
          <div className="card-body">
            <h6 className="card-title mb-3">Per-Actor Component Usage</h6>
            {perActor.map(a => (
              <div key={a.actor} className="mb-3">
                <div className="fw-bold small mb-1" style={{ color: actorColor(a.actor) }}>{a.actor}</div>
                <div className="ms-2">
                  {a.components.map(c => (
                    <div key={c.component} className="d-flex justify-content-between small">
                      <span style={{ color: compColor(c.component) }}>{c.component}</span>
                      <span className="fw-bold">{c.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   TRANSACTIONS TAB
═══════════════════════════════════════════════════════════════════════════ */
function TransactionsTab({ ov, bk }) {
  const daily     = ov.daily_transaction_volume || [];
  const heatmap   = ov.hourly_heatmap           || [];
  const timeline  = bk.transaction_timeline     || [];
  const errorActs = bk.error_actions            || [];
  const convTL    = bk.conversation_timeline    || [];

  // Get last 30 days
  const recent30 = daily.slice(-30);
  const maxDay   = Math.max(...recent30.map(d => d.count), 1);

  return (
    <div>
      {/* Daily volume */}
      <div className="card shadow-sm mb-3">
        <div className="card-body">
          <h6 className="card-title mb-3">Daily Transaction Volume (last 30 days)</h6>
          <div className="d-flex align-items-end gap-1" style={{ height: 80 }}>
            {recent30.map(d => (
              <div key={d.date} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div
                  title={`${d.date}: ${d.count}`}
                  style={{
                    width: '100%', height: `${Math.round((d.count / maxDay) * 70)}px`,
                    background: '#3b82f6', borderRadius: '2px 2px 0 0', minHeight: 2,
                  }}
                />
              </div>
            ))}
          </div>
          <div className="d-flex justify-content-between text-muted mt-1" style={{ fontSize: '0.65rem' }}>
            <span>{recent30[0]?.date}</span>
            <span>{recent30[recent30.length - 1]?.date}</span>
          </div>
        </div>
      </div>

      <div className="row g-3">
        {/* Recent timeline */}
        <div className="col-md-7">
          <div className="card shadow-sm">
            <div className="card-body">
              <h6 className="card-title mb-3">Transaction Timeline (recent, by component)</h6>
              <div style={{ overflowY: 'auto', maxHeight: 300 }}>
                <table className="table table-sm">
                  <thead className="table-light sticky-top">
                    <tr><th>Date</th><th>Component</th><th className="text-end">Count</th></tr>
                  </thead>
                  <tbody>
                    {timeline.slice(-30).reverse().map((t, i) => (
                      <tr key={i}>
                        <td className="text-muted small">{t.date}</td>
                        <td style={{ color: compColor(t.component) }}>{t.component}</td>
                        <td className="text-end fw-bold">{t.count.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        {/* Error actions + conversation timeline */}
        <div className="col-md-5">
          {errorActs.length > 0 && (
            <div className="card shadow-sm mb-3 border-danger">
              <div className="card-body">
                <h6 className="card-title text-danger mb-3">Error Actions</h6>
                <table className="table table-sm">
                  <thead><tr><th>Component</th><th>Action</th><th className="text-end">Count</th></tr></thead>
                  <tbody>
                    {errorActs.map((e, i) => (
                      <tr key={i}>
                        <td style={{ color: compColor(e.component) }}>{e.component}</td>
                        <td><code style={{ fontSize: '0.75rem', color: '#ef4444' }}>{e.action}</code></td>
                        <td className="text-end">{e.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="card shadow-sm">
            <div className="card-body">
              <h6 className="card-title mb-3">Conversation Timeline (recent)</h6>
              <div style={{ overflowY: 'auto', maxHeight: 220 }}>
                <table className="table table-sm">
                  <thead className="table-light sticky-top"><tr><th>Date</th><th>Role</th><th className="text-end">Msgs</th></tr></thead>
                  <tbody>
                    {convTL.slice(-15).reverse().map((c, i) => (
                      <tr key={i}>
                        <td className="text-muted small">{c.date}</td>
                        <td><Chip label={c.role} color={c.role === 'assistant' ? '#3b82f6' : '#6b7280'} /></td>
                        <td className="text-end fw-bold">{c.count}</td>
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
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   COSTS & ANALYSES TAB
═══════════════════════════════════════════════════════════════════════════ */
function CostsAnalysesTab({ bk }) {
  const costTL   = bk.cost_timeline      || [];
  const costSvc  = bk.cost_by_service    || [];
  const analyses = bk.analysis_summary   || [];
  const patients = bk.patient_transaction_profiles || [];

  const recent = costTL.slice(-14);
  const maxCost = Math.max(...recent.map(d => d.total_cost), 0.01);

  return (
    <div>
      {/* Cost timeline sparkline */}
      <div className="card shadow-sm mb-3">
        <div className="card-body">
          <h6 className="card-title mb-3">Cost Timeline — Daily LLM Cost (last 14 days with data)</h6>
          <div className="d-flex align-items-end gap-1" style={{ height: 70 }}>
            {recent.map(d => (
              <div key={d.date} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div
                  title={`${d.date}: $${d.total_cost.toFixed(2)} | ${d.requests} req | ${(d.tokens_in + d.tokens_out).toLocaleString()} tok`}
                  style={{
                    width: '100%', height: `${Math.round((d.total_cost / maxCost) * 60)}px`,
                    background: '#ef4444', borderRadius: '2px 2px 0 0', minHeight: 2,
                  }}
                />
                <div className="text-muted text-center" style={{ fontSize: '0.55rem', marginTop: 2 }}>
                  {d.date.slice(5)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="row g-3">
        {/* Cost by service */}
        <div className="col-md-5">
          {costSvc.length > 0 && (
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="card-title mb-3">Cost by Service</h6>
                {costSvc.map(s => (
                  <div key={s.service} className="mb-2">
                    <div className="d-flex justify-content-between small">
                      <span>{s.service}</span>
                      <span className="fw-bold">${(s.total_cost || 0).toFixed(3)}</span>
                    </div>
                    {s.requests > 0 && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{s.requests} req · {((s.tokens_in || 0) + (s.tokens_out || 0)).toLocaleString()} tok</div>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Patient transaction profiles */}
          <div className="card shadow-sm">
            <div className="card-body">
              <h6 className="card-title mb-3">Patient Activity Profiles (top 10)</h6>
              <div style={{ overflowY: 'auto', maxHeight: 250 }}>
                <table className="table table-sm">
                  <thead className="table-light sticky-top"><tr><th>Patient</th><th className="text-end">Tx</th></tr></thead>
                  <tbody>
                    {patients.slice(0, 10).map(p => (
                      <tr key={p.patient_id}>
                        <td><code style={{ fontSize: '0.75rem' }}>{p.patient_id}</code></td>
                        <td className="text-end fw-bold">{(p.transaction_count || 0).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis summary table */}
        <div className="col-md-7">
          <div className="card shadow-sm">
            <div className="card-body">
              <h6 className="card-title mb-3">EEG Analysis Summary ({analyses.length} records)</h6>
              <div style={{ overflowY: 'auto', maxHeight: 420 }}>
                <table className="table table-sm table-hover">
                  <thead className="table-light sticky-top">
                    <tr>
                      <th>Patient</th>
                      <th>Disease</th>
                      <th>Signal</th>
                      <th className="text-end">Conf</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analyses.map((a, i) => (
                      <tr key={i}>
                        <td><code style={{ fontSize: '0.75rem' }}>{a.patient_id}</code></td>
                        <td>{a.disease}</td>
                        <td>
                          <Chip
                            label={a.signal_quality}
                            color={a.signal_quality === 'Excellent' ? '#22c55e' : a.signal_quality === 'Good' ? '#3b82f6' : a.signal_quality === 'Fair' ? '#f59e0b' : '#ef4444'}
                          />
                        </td>
                        <td className="text-end fw-bold" style={{ color: confColor(a.confidence) }}>
                          {(a.confidence * 100).toFixed(0)}%
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
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEFINITIONS TAB
═══════════════════════════════════════════════════════════════════════════ */
const DEFS = [
  { term: 'Transaction Log', def: 'Append-only audit trail of every AI component action: actor, component, action, patient context, timestamp.' },
  { term: 'Component', def: 'Identifiable subsystem (http-trace, cv_pipeline, drift, fairness, etc.) that generates transaction events.' },
  { term: 'Actor', def: 'Agent or human role responsible for an action: middleware (automated), system (scheduled), clinician roles.' },
  { term: 'http-trace', def: 'HTTP request traces from FastAPI middleware — forms the bulk of transactions (system activity monitoring).' },
  { term: 'cv_pipeline / video_frames', def: 'Computer-vision pipeline stages: frame extraction + processing of patient video sessions.' },
  { term: 'Drift', def: 'Automated data/model drift checks (PSI, KS-test, accuracy delta) run on each scheduled cycle.' },
  { term: 'Fairness', def: 'Fairness analysis runs (AIF360 / demographic parity / equalized odds) logged as component events.' },
  { term: 'Council', def: 'AI Agent Council decisions: multi-agent consensus answers or blocked queries requiring human review.' },
  { term: 'HITL Override', def: 'Human-in-the-loop review event where a clinician overrides or confirms an AI prediction.' },
  { term: 'Avg Confidence', def: 'Mean model prediction confidence across all EEG analyses. Ranges 0–1; ≥0.8 = high, <0.4 = low.' },
  { term: 'Cost Categories', def: 'GPU compute (training/inference), cloud infra (storage/network), LLM inference (Claude/GPT API calls).' },
  { term: 'Conversation Log', def: 'Multi-turn chat messages (assistant + operator roles) stored for audit and RAG training.' },
  { term: 'Hourly Heatmap', def: 'Transaction count by hour-of-day across all components — identifies peak usage windows.' },
  { term: 'Observability', def: 'End-to-end visibility into AI system behaviour: what ran, when, by whom, at what cost, with what outcome.' },
];

function DefinitionsTab() {
  return (
    <div className="card shadow-sm">
      <div className="card-body">
        <h6 className="card-title mb-3">AI Observability — Key Definitions</h6>
        <table className="table table-sm table-hover">
          <thead className="table-light"><tr><th style={{ width: '30%' }}>Term</th><th>Definition</th></tr></thead>
          <tbody>
            {DEFS.map(d => (
              <tr key={d.term}>
                <td className="fw-bold" style={{ color: '#3b82f6' }}>{d.term}</td>
                <td className="small">{d.def}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   ROOT PAGE
═══════════════════════════════════════════════════════════════════════════ */
const TABS = ['Overview', 'Components', 'Transactions', 'Costs & Analyses', 'Definitions'];

export default function AIObservabilityPage() {
  const [tab, setTab]   = useState(0);
  const [ov,  setOv]    = useState(null);
  const [bk,  setBk]    = useState(null);
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ai-observability/overview`).then(r => r.json()),
      fetch(`${API}/api/ai-observability/breakdown`).then(r => r.json()),
    ]).then(([o, b]) => { setOv(o); setBk(b); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov || !bk) return (
    <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 200 }}>
      <div className="spinner-border text-primary" /><span className="ms-3">Loading AI Observability…</span>
    </div>
  );

  const kpis = ov.kpis || {};

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: '1.6rem' }}>🔭</span>
        <div>
          <h4 className="mb-0 fw-bold">AI Observability Dashboard</h4>
          <div className="text-muted small">
            {(kpis.total_transactions || 0).toLocaleString()} transactions · {kpis.total_components || 0} components · {kpis.total_actors || 0} actors · ${(kpis.total_cost_usd || 0).toFixed(2)} total cost
          </div>
        </div>
      </div>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <ComponentsTab bk={bk} />}
      {tab === 2 && <TransactionsTab ov={ov} bk={bk} />}
      {tab === 3 && <CostsAnalysesTab bk={bk} />}
      {tab === 4 && <DefinitionsTab />}
    </div>
  );
}
