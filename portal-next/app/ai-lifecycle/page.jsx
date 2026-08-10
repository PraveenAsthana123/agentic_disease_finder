'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'assets',      label: 'Asset Breakdown' },
  { id: 'events',      label: 'Lifecycle Events' },
  { id: 'definitions', label: 'Definitions' },
];

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

function StageBadge({ stage }) {
  const colors = {
    deployed:    'success',
    monitoring:  'info',
    validation:  'warning',
    development: 'secondary',
    ideation:    'light',
    retired:     'danger',
  };
  return <span className={`badge bg-${colors[stage] || 'secondary'}`}>{stage}</span>;
}

function TypeBadge({ type }) {
  const colors = { agents: 'primary', pipelines: 'info', models: 'success' };
  return <span className={`badge bg-${colors[type] || 'secondary'}`}>{type}</span>;
}

function ActorBadge({ actor }) {
  return <span className={`badge bg-${actor === 'operator' ? 'warning text-dark' : 'secondary'}`}>{actor}</span>;
}

function Bar({ value, max, color }) {
  const pct = Math.min(100, Math.round(((value || 0) / (max || 1)) * 100));
  return (
    <div className="progress" style={{ height: 8 }}>
      <div className={`progress-bar bg-${color || 'primary'}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export default function AILifecyclePage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/ai-lifecycle/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/ai-lifecycle/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/ai-lifecycle/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpis = ov.kpis || {};
  const stageItems = ov.lifecycle_stage_distribution || [];
  const assetTypes = ov.asset_type_distribution || [];
  const dailyEvents = ov.daily_lifecycle_events || [];
  const healthInfo = ov.lifecycle_health || {};

  const maxStage = Math.max(...stageItems.map(s => s.count), 1);
  const maxDaily = Math.max(...dailyEvents.map(d => d.events), 1);

  return (
    <div>
      <h3>🔄 AI Lifecycle Management</h3>
      <p className="text-muted">
        Real transaction_log + agent/pipeline/model registries — {kpis.total_ai_assets} AI assets,{' '}
        {kpis.agents_operational} agents, {kpis.pipelines_active} pipelines, {kpis.models_deployed} models
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

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <div className="row mb-4">
            <KPI label="Total AI Assets"      value={kpis.total_ai_assets}    color="primary" />
            <KPI label="Lifecycle Coverage"   value={`${kpis.lifecycle_coverage}%`} color="success" />
            <KPI label="Models Deployed"      value={kpis.models_deployed}    color="info" />
            <KPI label="Agents Operational"   value={kpis.agents_operational}  color="warning" />
          </div>
          <div className="row mb-4">
            <KPI label="Pipelines Active"   value={kpis.pipelines_active}   color="primary" />
            <KPI label="Validation Events"  value={kpis.validation_events}  color="info" />
            <KPI label="Monitoring Events"  value={kpis.monitoring_events}  color="success" />
            <KPI label="Training Runs"      value={kpis.training_runs}      color="secondary" />
          </div>

          {/* Lifecycle Stage Distribution */}
          <div className="row mb-4">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Lifecycle Stage Distribution</div>
                <div className="card-body">
                  {stageItems.length === 0
                    ? <span className="text-muted small">No data</span>
                    : stageItems.map(s => (
                      <div key={s.stage} className="mb-2">
                        <div className="d-flex justify-content-between mb-1">
                          <StageBadge stage={s.stage} />
                          <span className="fw-semibold">{s.count}</span>
                        </div>
                        <Bar value={s.count} max={maxStage} color={
                          s.stage === 'deployed' ? 'success' :
                          s.stage === 'monitoring' ? 'info' :
                          s.stage === 'validation' ? 'warning' : 'secondary'
                        } />
                      </div>
                    ))
                  }
                </div>
              </div>
            </div>

            {/* Asset Type Distribution */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Asset Type Breakdown</div>
                <div className="card-body">
                  {assetTypes.map(t => (
                    <div key={t.type} className="mb-3">
                      <div className="d-flex justify-content-between align-items-center mb-1">
                        <TypeBadge type={t.type} />
                        <span className="fw-semibold">{t.total}</span>
                      </div>
                      <Bar value={t.total} max={Math.max(...assetTypes.map(x => x.total), 1)} color={
                        t.type === 'agents' ? 'primary' : t.type === 'pipelines' ? 'info' : 'success'
                      } />
                      <div className="mt-1 d-flex gap-2 flex-wrap" style={{ fontSize: '0.75rem' }}>
                        {Object.entries(t).filter(([k]) => !['type','total'].includes(k)).map(([k,v]) => (
                          <span key={k} className="text-muted">{k}: <strong>{v}</strong></span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Daily Lifecycle Events */}
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Daily Lifecycle Event Volume</div>
            <div className="card-body">
              {dailyEvents.length === 0
                ? <span className="text-muted small">No events</span>
                : (
                  <div className="table-responsive" style={{ maxHeight: 280, overflowY: 'auto' }}>
                    <table className="table table-sm table-hover mb-0">
                      <thead className="table-light sticky-top">
                        <tr>
                          <th>Date</th>
                          <th className="text-end">Events</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...dailyEvents].reverse().map((d, i) => (
                          <tr key={i}>
                            <td className="small">{d.date}</td>
                            <td className="text-end fw-semibold">{d.events}</td>
                            <td>
                              <div className="d-flex gap-1 flex-wrap">
                                {d.breakdown && Object.entries(d.breakdown).slice(0, 4).map(([k, v]) => (
                                  <span key={k} className="badge bg-light text-dark" style={{ fontSize: '0.65rem' }}>
                                    {k.replace(/_/g, ' ')}: {v}
                                  </span>
                                ))}
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )
              }
            </div>
          </div>

          {/* Lifecycle Health */}
          {healthInfo && Object.keys(healthInfo).length > 0 && (
            <div className="card shadow-sm mb-4">
              <div className="card-header fw-semibold">Lifecycle Health</div>
              <div className="card-body">
                <div className="row">
                  {Object.entries(healthInfo).map(([k, v]) => (
                    <div key={k} className="col-md-4 mb-2">
                      <div className="text-muted small text-capitalize">{k.replace(/_/g, ' ')}</div>
                      <div className="fw-semibold">
                        {typeof v === 'number' ? v.toFixed(1) : String(v)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── ASSET BREAKDOWN ── */}
      {tab === 'assets' && bd && (
        <>
          {/* Agents */}
          {bd.agent_lifecycle && (
            <div className="card shadow-sm mb-4">
              <div className="card-header fw-semibold">Agents ({bd.agent_lifecycle.length})</div>
              <div className="card-body p-0">
                <div className="table-responsive" style={{ maxHeight: 360, overflowY: 'auto' }}>
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light sticky-top">
                      <tr><th>ID</th><th>Task</th><th>Status</th><th>Stage</th><th>Module</th></tr>
                    </thead>
                    <tbody>
                      {bd.agent_lifecycle.map((a, i) => (
                        <tr key={i}>
                          <td className="small fw-semibold text-nowrap">{a.id}</td>
                          <td className="small">{a.name}</td>
                          <td><span className={`badge bg-${a.status === 'built' ? 'success' : 'secondary'}`}>{a.status}</span></td>
                          <td><StageBadge stage={a.lifecycle_stage || 'deployed'} /></td>
                          <td className="small text-muted text-truncate" style={{ maxWidth: 160 }}>{a.module}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Pipelines */}
          {bd.pipeline_lifecycle && (
            <div className="card shadow-sm mb-4">
              <div className="card-header fw-semibold">Pipelines ({bd.pipeline_lifecycle.length})</div>
              <div className="card-body p-0">
                <div className="table-responsive" style={{ maxHeight: 320, overflowY: 'auto' }}>
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light sticky-top">
                      <tr><th>ID</th><th>Name</th><th>Status</th><th>Stage</th></tr>
                    </thead>
                    <tbody>
                      {bd.pipeline_lifecycle.map((p, i) => (
                        <tr key={i}>
                          <td className="small fw-semibold">{p.id}</td>
                          <td className="small">{p.name}</td>
                          <td><span className={`badge bg-${p.status === 'built' ? 'success' : p.status === 'partial' ? 'warning' : 'secondary'}`}>{p.status}</span></td>
                          <td><StageBadge stage={p.lifecycle_stage || 'deployed'} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Models */}
          {bd.model_lifecycle && (
            <div className="card shadow-sm mb-4">
              <div className="card-header fw-semibold">Models ({bd.model_lifecycle.length})</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr><th>File</th><th>Size</th><th>Modified</th><th>Stage</th></tr>
                    </thead>
                    <tbody>
                      {bd.model_lifecycle.map((m, i) => (
                        <tr key={i}>
                          <td className="small fw-semibold">{m.file}</td>
                          <td className="small">{m.size_mb ? `${m.size_mb} MB` : '—'}</td>
                          <td className="small text-muted">{m.modified_utc ? m.modified_utc.slice(0,10) : '—'}</td>
                          <td><StageBadge stage={m.lifecycle_stage || 'deployed'} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── LIFECYCLE EVENTS ── */}
      {tab === 'events' && bd && bd.recent_lifecycle_events && (
        <div className="card shadow-sm mb-4">
          <div className="card-header fw-semibold">Recent Lifecycle Events ({bd.recent_lifecycle_events.length})</div>
          <div className="card-body p-0">
            <div className="table-responsive" style={{ maxHeight: 500, overflowY: 'auto' }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light sticky-top">
                  <tr>
                    <th>ID</th>
                    <th>Component</th>
                    <th>Action</th>
                    <th>Transition</th>
                    <th>Actor</th>
                    <th>Detail</th>
                    <th>Timestamp (UTC)</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.recent_lifecycle_events.map((ev, i) => (
                    <tr key={i}>
                      <td className="small text-muted">{ev.id}</td>
                      <td className="small fw-semibold">{ev.component}</td>
                      <td><span className="badge bg-light text-dark" style={{ fontSize: '0.7rem' }}>{ev.action}</span></td>
                      <td className="small text-muted">{ev.transition}</td>
                      <td><ActorBadge actor={ev.actor} /></td>
                      <td className="small text-muted text-truncate" style={{ maxWidth: 200 }}>{ev.detail}</td>
                      <td className="small text-muted text-nowrap">{ev.ts_utc ? ev.ts_utc.slice(0,16) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <>
          {(defs.sections || []).map((sec, si) => (
            <div key={si} className="card shadow-sm mb-4">
              <div className="card-header fw-semibold">{sec.title}</div>
              <div className="card-body">
                {(sec.items || []).map((item, ii) => (
                  <div key={ii} className="mb-3">
                    <div className="fw-semibold">{item.term}</div>
                    <div className="text-muted small">{item.definition}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
          {defs.standards && (
            <div className="card shadow-sm mb-4">
              <div className="card-header fw-semibold">Standards & References</div>
              <div className="card-body">
                {(defs.standards || []).map((s, i) => (
                  <div key={i} className="mb-2">
                    <div className="fw-semibold small">{s.name}</div>
                    <div className="text-muted" style={{ fontSize: '0.75rem' }}>{s.description}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
