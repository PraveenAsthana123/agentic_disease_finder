'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusColor = s =>
  s === 'healthy'  ? 'success' :
  s === 'degraded' ? 'warning' :
  s === 'down'     ? 'danger'  : 'secondary';

export default function AIControlTowerPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/ai-control-tower/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/ai-control-tower/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/ai-control-tower/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const aic   = ov.ai_components || {};
  const cost  = ov.cost_summary  || {};
  const health = ov.system_health || {};
  const drift  = ov.drift_status  || {};
  const dq     = ov.data_quality  || {};
  const cons   = ov.consistency   || {};
  const over   = ov.oversight     || {};

  const kpis = [
    { label: 'AI Components',    value: aic.total,             color: 'primary' },
    { label: 'Built',            value: aic.built,             color: 'success' },
    { label: 'Total Tx',         value: (ov.total_transactions || 0).toLocaleString(), color: 'info' },
    { label: 'Blocked Ops',      value: (ov.error_action_total || 0), color: ov.error_action_total > 0 ? 'warning' : 'success' },
    { label: 'Total Cost',       value: `$${cost.total_cost_usd ?? '—'}`, color: 'primary' },
    { label: 'HITL Reviews',     value: over.hitl_reviews ?? '—', color: 'info' },
    { label: 'Drift Status',     value: drift.verdict || 'n/a',   color: drift.verdict === 'stable' ? 'success' : 'warning' },
    { label: 'AI Readiness',     value: dq.ai_readiness_grade || '—', color: 'success' },
  ];

  const tabs = [
    { id: 'overview',     label: 'Overview'          },
    { id: 'activity',     label: 'Activity Log'      },
    { id: 'cost',         label: 'Cost Breakdown'    },
    { id: 'components',   label: 'Component Registry'},
    { id: 'definitions',  label: 'Concepts'          },
  ];

  const categoryColor = c =>
    c === 'gpu_compute'    ? 'danger'  :
    c === 'llm_inference'  ? 'primary' :
    c === 'cloud_infra'    ? 'info'    : 'secondary';

  return (
    <div>
      <h3>&#x1f5fc; AI Control Tower</h3>
      <p className="text-muted small">
        Centralized AI system oversight — component registry, transaction audit,
        system health, FinOps costs, drift status, and human-oversight metrics.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {kpis.map(k => (
          <div key={k.label} className="col-6 col-md-3 col-lg mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h4 mb-0 text-${k.color}`}>{k.value ?? '—'}</div>
                <div className="text-muted" style={{ fontSize: '0.72rem' }}>{k.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div className="row">

          {/* Component registry summary */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header bg-primary text-white py-2 small fw-bold">Component Registry</div>
              <div className="card-body p-2">
                {[
                  { label: 'Total registered', value: aic.total,       color: 'primary' },
                  { label: 'Built',            value: aic.built,       color: 'success' },
                  { label: 'Scaffold',         value: aic.scaffold,    color: 'warning' },
                  { label: 'Planned',          value: aic.planned,     color: 'info'    },
                  { label: 'Not pulled',       value: aic.not_pulled,  color: 'secondary'},
                ].map(r => (
                  <div key={r.label} className="d-flex justify-content-between align-items-center mb-1">
                    <span className="small">{r.label}</span>
                    <span className={`badge bg-${r.color}`}>{r.value ?? 0}</span>
                  </div>
                ))}
                <hr className="my-2" />
                <div className="text-center">
                  <span className="small text-muted">Readiness </span>
                  <span className="fw-bold text-success">
                    {aic.total > 0 ? `${Math.round((aic.built / aic.total) * 100)}%` : '—'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Oversight metrics */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header bg-info text-white py-2 small fw-bold">Human Oversight</div>
              <div className="card-body p-2">
                {[
                  { label: 'HITL Reviews',        value: over.hitl_reviews        },
                  { label: 'Clinical Decisions',  value: over.clinical_decisions   },
                  { label: 'Feedback Entries',    value: over.feedback_entries     },
                  { label: 'Component Findings',  value: over.component_findings   },
                ].map(r => (
                  <div key={r.label} className="d-flex justify-content-between align-items-center mb-1">
                    <span className="small">{r.label}</span>
                    <span className="badge bg-info text-dark">{r.value ?? 0}</span>
                  </div>
                ))}
                <hr className="my-2" />
                <div className="text-center small text-muted">
                  Error action rate:{' '}
                  <span className={`fw-bold text-${(ov.error_action_rate || 0) > 0.05 ? 'danger' : 'success'}`}>
                    {((ov.error_action_rate || 0) * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* System + drift + quality status */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header bg-dark text-white py-2 small fw-bold">System Status</div>
              <div className="card-body p-2">
                {[
                  { label: 'Backend HTTP',    value: health.backend_http ?? '—',            ok: health.backend_http === 200 },
                  { label: 'API Errors',      value: health.api_errors ?? '—',              ok: health.api_errors === 0     },
                  { label: 'DB Status',       value: health.db_status ?? '—',               ok: health.db_status === 'ok'   },
                  { label: 'Total Errors',    value: health.total_errors ?? '—',            ok: health.total_errors === 0   },
                  { label: 'Drift',           value: drift.verdict ?? '—',                  ok: drift.verdict === 'stable'  },
                  { label: 'Consistency',     value: cons.verdict ?? '—',                   ok: cons.verdict === 'consistent'},
                  { label: 'AI Readiness',    value: dq.ai_readiness_grade ?? '—',          ok: true                        },
                ].map(r => (
                  <div key={r.label} className="d-flex justify-content-between align-items-center mb-1">
                    <span className="small">{r.label}</span>
                    <span className={`badge bg-${r.ok ? 'success' : 'warning'}`}
                          style={{ fontSize: '0.7rem' }}>
                      {r.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Cost by category */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">FinOps — Cost by Category</div>
              <div className="card-body p-2">
                {(cost.by_category || []).map(c => {
                  const max = Math.max(...(cost.by_category || []).map(x => x.total_cost), 1);
                  return (
                    <div key={c.category} className="d-flex align-items-center mb-2">
                      <span className="small me-2" style={{ minWidth: '130px' }}>
                        <span className={`badge bg-${categoryColor(c.category)}`}>{c.category}</span>
                      </span>
                      <div className="progress flex-grow-1 me-2" style={{ height: '18px' }}>
                        <div className={`progress-bar bg-${categoryColor(c.category)}`}
                             style={{ width: `${(c.total_cost / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>${c.total_cost.toFixed(2)}</span>
                        </div>
                      </div>
                      <span className="small text-muted">{c.records}r</span>
                    </div>
                  );
                })}
                <div className="text-end small mt-2">
                  <strong>Total: ${cost.total_cost_usd ?? '—'}</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Top component activity */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-secondary text-white py-2 small fw-bold">Top Component Activity</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Component</th><th className="text-end">Tx Count</th><th style={{ width: '40%' }}>Bar</th></tr></thead>
                  <tbody>
                    {(ov.component_activity || []).map(c => {
                      const max = Math.max(...(ov.component_activity || []).map(x => x.cnt), 1);
                      return (
                        <tr key={c.component}>
                          <td className="small">{c.component}</td>
                          <td className="text-end small">{c.cnt}</td>
                          <td>
                            <div className="progress" style={{ height: '12px' }}>
                              <div className="progress-bar bg-secondary"
                                   style={{ width: `${(c.cnt / max * 100).toFixed(0)}%` }} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Action distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-dark text-white py-2 small fw-bold">Action Distribution</div>
              <div className="card-body p-2">
                {(ov.action_distribution || []).map(a => {
                  const max = Math.max(...(ov.action_distribution || []).map(x => x.cnt), 1);
                  const isError = ['blocked', 'error', 'delete'].includes(a.action);
                  return (
                    <div key={a.action} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{ minWidth: '110px' }}>
                        <span className={`badge bg-${isError ? 'danger' : 'primary'}`}>{a.action}</span>
                      </span>
                      <div className="progress flex-grow-1" style={{ height: '14px' }}>
                        <div className={`progress-bar bg-${isError ? 'danger' : 'primary'}`}
                             style={{ width: `${(a.cnt / max * 100).toFixed(0)}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{a.cnt}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Analyses summary */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-success text-white py-2 small fw-bold">Analyses Engine</div>
              <div className="card-body p-2">
                {ov.analyses && (
                  <>
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small">Total Analyses</span>
                      <span className="badge bg-success">{ov.analyses.total}</span>
                    </div>
                    <div className="d-flex justify-content-between mb-2">
                      <span className="small">Avg Confidence</span>
                      <span className="badge bg-info">
                        {ov.analyses.avg_confidence != null
                          ? `${(ov.analyses.avg_confidence * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                    <div className="small fw-bold mb-1 text-muted">Per Disease</div>
                    {Object.entries(ov.analyses.per_disease || {}).map(([disease, cnt]) => (
                      <div key={disease} className="d-flex justify-content-between mb-1">
                        <span className="small">{disease}</span>
                        <span className="badge bg-secondary">{cnt}</span>
                      </div>
                    ))}
                  </>
                )}
              </div>
            </div>
          </div>

        </div>
      )}

      {/* ── Activity Log ── */}
      {tab === 'activity' && bd && (
        <div>
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-header bg-dark text-white py-2 small fw-bold">
              Recent Transactions (last 50)
            </div>
            <div className="card-body p-2">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>#</th>
                      <th>Patient</th>
                      <th>Component</th>
                      <th>Action</th>
                      <th>Actor</th>
                      <th>Detail</th>
                      <th>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.recent_transactions || []).map(tx => {
                      const isError = ['blocked', 'error'].includes(tx.action);
                      return (
                        <tr key={tx.id} className={isError ? 'table-warning' : ''}>
                          <td className="small text-muted">{tx.id}</td>
                          <td className="small">{tx.patient_id || '—'}</td>
                          <td className="small fw-bold">{tx.component}</td>
                          <td>
                            <span className={`badge bg-${isError ? 'danger' : 'primary'}`}
                                  style={{ fontSize: '0.65rem' }}>{tx.action}</span>
                          </td>
                          <td className="small">{tx.actor || '—'}</td>
                          <td className="small text-truncate" style={{ maxWidth: '200px' }}>
                            {tx.detail || '—'}
                          </td>
                          <td className="small text-muted">
                            {tx.ts_local ? tx.ts_local.slice(0, 16) : '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* HITL reviews */}
          {(bd.hitl_reviews || []).length > 0 && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-info text-white py-2 small fw-bold">
                HITL Reviews ({(bd.hitl_reviews || []).length})
              </div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead><tr><th>#</th><th>Patient</th><th>Analysis</th><th>Fields</th><th>Created</th></tr></thead>
                  <tbody>
                    {(bd.hitl_reviews || []).map(r => (
                      <tr key={r.id}>
                        <td className="small">{r.id}</td>
                        <td className="small">{r.patient_id}</td>
                        <td className="small">{r.analysis_id}</td>
                        <td className="small text-muted">
                          {Object.keys(r.fields || {}).join(', ') || '—'}
                        </td>
                        <td className="small">{(r.created_at || '').slice(0, 10)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Clinical decisions */}
          {(bd.clinical_decisions || []).length > 0 && (
            <div className="card shadow-sm border-0">
              <div className="card-header bg-success text-white py-2 small fw-bold">
                Clinical Decisions ({(bd.clinical_decisions || []).length})
              </div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead>
                    <tr>
                      <th>Patient</th><th>AI Prediction</th><th>AI Conf</th>
                      <th>Agree</th><th>Final</th><th>Reviewer</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.clinical_decisions || []).map(d => (
                      <tr key={d.id}>
                        <td className="small">{d.patient_id}</td>
                        <td className="small">{d.ai_prediction || '—'}</td>
                        <td className="small">
                          {d.ai_confidence != null ? `${(d.ai_confidence * 100).toFixed(0)}%` : '—'}
                        </td>
                        <td>
                          <span className={`badge bg-${d.neurologist_agreement ? 'success' : 'warning'}`}
                                style={{ fontSize: '0.65rem' }}>
                            {d.neurologist_agreement ? 'Yes' : 'No'}
                          </span>
                        </td>
                        <td className="small">{d.final_decision || '—'}</td>
                        <td className="small">{d.reviewer || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Cost Breakdown ── */}
      {tab === 'cost' && bd && (
        <div>
          {/* Cost timeline */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-header bg-warning text-dark py-2 small fw-bold">Daily Cost Trend (last 30 days)</div>
            <div className="card-body p-2">
              <div className="table-responsive">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Date</th><th className="text-end">Cost (USD)</th><th>Tokens In</th><th>Tokens Out</th><th style={{ width: '35%' }}>Bar</th></tr></thead>
                  <tbody>
                    {(bd.cost_timeline || []).slice().reverse().map(d => {
                      const max = Math.max(...(bd.cost_timeline || []).map(x => x.daily_cost), 1);
                      return (
                        <tr key={d.cost_date}>
                          <td className="small">{d.cost_date}</td>
                          <td className="text-end small">${(d.daily_cost || 0).toFixed(2)}</td>
                          <td className="small">{(d.tokens_in || 0).toLocaleString()}</td>
                          <td className="small">{(d.tokens_out || 0).toLocaleString()}</td>
                          <td>
                            <div className="progress" style={{ height: '12px' }}>
                              <div className="progress-bar bg-warning"
                                   style={{ width: `${((d.daily_cost || 0) / max * 100).toFixed(0)}%` }} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Cost by service */}
          <div className="card shadow-sm border-0">
            <div className="card-header bg-danger text-white py-2 small fw-bold">Cost by Model / Service</div>
            <div className="card-body p-2">
              <table className="table table-sm mb-0">
                <thead>
                  <tr>
                    <th>Service</th><th>Category</th><th className="text-end">Cost ($)</th>
                    <th className="text-end">Requests</th><th className="text-end">Tokens In</th>
                    <th className="text-end">Tokens Out</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.cost_by_service || []).map((s, i) => (
                    <tr key={i}>
                      <td className="small fw-bold">{s.model_or_service}</td>
                      <td>
                        <span className={`badge bg-${categoryColor(s.category)}`}
                              style={{ fontSize: '0.65rem' }}>{s.category}</span>
                      </td>
                      <td className="text-end small">${(s.total_cost || 0).toFixed(4)}</td>
                      <td className="text-end small">{(s.total_requests || 0).toLocaleString()}</td>
                      <td className="text-end small">{(s.total_tokens_in || 0).toLocaleString()}</td>
                      <td className="text-end small">{(s.total_tokens_out || 0).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Component Registry ── */}
      {tab === 'components' && bd && (
        <div>
          <div className="card shadow-sm border-0">
            <div className="card-header bg-primary text-white py-2 small fw-bold">
              AI Component Registry ({(bd.component_status_matrix || []).length} tracked)
            </div>
            <div className="card-body p-2">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr><th>Component</th><th>Status</th><th>Note</th></tr>
                  </thead>
                  <tbody>
                    {(bd.component_status_matrix || []).map((c, i) => (
                      <tr key={i}>
                        <td className="small">{c.type}</td>
                        <td>
                          <span className={`badge bg-${c.status === 'built' ? 'success' : c.status === 'scaffold' ? 'warning' : 'secondary'}`}
                                style={{ fontSize: '0.65rem' }}>{c.status}</span>
                        </td>
                        <td className="small text-muted">{c.note || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Concepts ── */}
      {tab === 'definitions' && defs && (
        <div>
          {(defs.control_tower_concept || []).map(c => (
            <div key={c.name} className="card shadow-sm border-0 mb-2">
              <div className="card-header bg-dark text-white py-1 small fw-bold">{c.name}</div>
              <div className="card-body p-2 small">{c.description}</div>
            </div>
          ))}

          {defs.system_components && (
            <div className="card shadow-sm border-0 mb-2">
              <div className="card-header bg-info text-white py-2 small fw-bold">System Components</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {defs.system_components.map(c => (
                      <tr key={c.name}>
                        <td className="small fw-bold" style={{ width: '25%', verticalAlign: 'top' }}>{c.name}</td>
                        <td className="small">{c.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {defs.metrics_and_kpis && (
            <div className="card shadow-sm border-0 mb-2">
              <div className="card-header bg-primary text-white py-2 small fw-bold">Metrics & KPIs</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {defs.metrics_and_kpis.map(m => (
                      <tr key={m.name}>
                        <td className="small fw-bold" style={{ width: '28%', verticalAlign: 'top' }}>{m.name}</td>
                        <td className="small">{m.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {defs.clinical_relevance && (
            <div className="card shadow-sm border-0 mb-2">
              <div className="card-header bg-success text-white py-2 small fw-bold">Clinical & Regulatory Relevance</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {defs.clinical_relevance.map(r => (
                      <tr key={r.standard}>
                        <td className="small fw-bold" style={{ width: '15%', verticalAlign: 'top' }}>{r.standard}</td>
                        <td className="small">{r.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {defs.remediation_strategies && (
            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">Remediation Strategies</div>
              <div className="card-body p-2">
                {defs.remediation_strategies.map((s, i) => (
                  <div key={i} className="mb-3">
                    <div className="small fw-bold text-danger mb-1">Trigger: {s.trigger}</div>
                    <div className="small text-muted" style={{ whiteSpace: 'pre-line' }}>{s.strategy}</div>
                    {i < defs.remediation_strategies.length - 1 && <hr className="my-2" />}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
