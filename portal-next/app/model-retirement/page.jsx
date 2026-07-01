'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STAGE_COLORS = { active: 'success', flagged: 'warning', approved: 'info', archived: 'secondary', audit_closed: 'danger' };
const accColor = v => v === null || v === undefined ? 'secondary' : v >= 0.9 ? 'success' : v >= 0.8 ? 'warning' : 'danger';
const driftColor = s => s === 'not_monitored' ? 'secondary' : /none|low/i.test(s) ? 'success' : /moderate/i.test(s) ? 'warning' : 'danger';

export default function ModelRetirementPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/model-retirement/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/model-retirement/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/model-retirement/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;
  if (!ov.available) return <div className="p-4 alert alert-warning">Model retirement data unavailable</div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'lifecycle',   label: 'Lifecycle & Trends' },
    { id: 'training',    label: 'Training History' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>AI Model Retirement</h3>
      <p className="text-muted">Real model lifecycle tracking from models/*.joblib, accuracy reports, drift analysis, and git history</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Models',    value: ov.total_models,           color: 'primary' },
          { label: 'Active',          value: ov.active_models,          color: 'success' },
          { label: 'Flagged',         value: ov.flagged_for_retirement, color: 'warning' },
          { label: 'Retirement Rate', value: `${ov.retirement_rate}%`,  color: ov.retirement_rate > 50 ? 'danger' : 'info' },
          { label: 'Avg Age (days)',  value: ov.avg_model_age_days,     color: 'dark' },
          { label: 'Oldest Model',    value: `${ov.oldest_model.age_days}d`, color: ov.oldest_model.age_days > 30 ? 'danger' : 'success' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ──────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Model Inventory Table */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Model Inventory</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr>
                      <th>Model</th><th>Disease</th><th>Size (KB)</th><th>Accuracy</th>
                      <th>Drift</th><th>Age (days)</th><th>Stage</th><th>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.models || []).map((m, i) => (
                      <tr key={i}>
                        <td className="fw-semibold small">{m.name}</td>
                        <td>{m.disease}</td>
                        <td>{m.file_size_kb}</td>
                        <td>
                          {m.accuracy !== null && m.accuracy !== undefined
                            ? <span className={`badge bg-${accColor(m.accuracy)}`}>{(m.accuracy * 100).toFixed(1)}%</span>
                            : <span className="badge bg-secondary">N/A</span>}
                        </td>
                        <td><span className={`badge bg-${driftColor(m.drift_status)}`}>{m.drift_status}</span></td>
                        <td className={m.age_days > 30 ? 'text-danger fw-bold' : ''}>{m.age_days}</td>
                        <td><span className={`badge bg-${STAGE_COLORS[m.retirement_stage] || 'secondary'}`}>{m.retirement_stage}</span></td>
                        <td className="small text-muted" style={{maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>{m.retirement_reason || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Stage Summary */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Pipeline Stage Summary</div>
              <div className="card-body">
                {(ov.stage_summary || []).map(s => {
                  const maxC = Math.max(...(ov.stage_summary || []).map(x => x.count), 1);
                  return (
                    <div key={s.stage} className="d-flex align-items-center mb-2">
                      <span className={`badge bg-${STAGE_COLORS[s.stage] || 'secondary'}`} style={{minWidth: 100}}>{s.stage}</span>
                      <div className="progress flex-grow-1 mx-2" style={{height: '20px'}}>
                        <div className={`progress-bar bg-${STAGE_COLORS[s.stage] || 'secondary'}`}
                             style={{width: `${s.count / maxC * 100}%`}} />
                      </div>
                      <span className="fw-bold">{s.count}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Accuracy Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Accuracy Distribution</div>
              <div className="card-body">
                {(() => {
                  const maxA = Math.max(...(ov.accuracy_distribution || []).map(b => b.count), 1);
                  return (ov.accuracy_distribution || []).map(b => (
                    <div key={b.bucket} className="d-flex align-items-center mb-2">
                      <span className="fw-semibold" style={{minWidth: 80}}>{b.bucket}</span>
                      <div className="progress flex-grow-1 mx-2" style={{height: '18px'}}>
                        <div className="progress-bar bg-info" style={{width: `${b.count / maxA * 100}%`}} />
                      </div>
                      <span className="fw-bold">{b.count}</span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Lifecycle & Trends Tab ────────────────────────────── */}
      {tab === 'lifecycle' && bd && (
        <div className="row">
          {/* Accuracy vs Drift Scatter (CSS-based) */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Accuracy vs Drift</div>
              <div className="card-body">
                <div style={{position: 'relative', width: '100%', height: 250, border: '1px solid #dee2e6', borderRadius: 4, background: '#f8f9fa'}}>
                  {/* Axis labels */}
                  <div style={{position: 'absolute', bottom: -20, left: '50%', transform: 'translateX(-50)'}} className="small text-muted">Drift Fraction &rarr;</div>
                  <div style={{position: 'absolute', top: '50%', left: -30, transform: 'rotate(-90deg) translateX(-50%)'}} className="small text-muted">Accuracy &uarr;</div>
                  {/* Threshold lines */}
                  <div style={{position: 'absolute', bottom: `${0.80 * 100}%`, left: 0, right: 0, borderTop: '1px dashed #dc3545', opacity: 0.5}} />
                  <div style={{position: 'absolute', left: `${0.50 * 100}%`, top: 0, bottom: 0, borderLeft: '1px dashed #dc3545', opacity: 0.5}} />
                  {/* Data points */}
                  {(bd.accuracy_vs_drift || []).map((m, i) => {
                    const acc = m.accuracy !== null && m.accuracy !== undefined ? m.accuracy : 0.5;
                    const dr = m.drift_frac || 0;
                    const left = `${Math.min(dr * 100, 95)}%`;
                    const bottom = `${Math.min(acc * 100, 95)}%`;
                    const color = STAGE_COLORS[m.stage] || 'secondary';
                    return (
                      <div key={i} title={`${m.name}: acc=${m.accuracy}, drift=${m.drift_frac}`}
                           style={{position: 'absolute', left, bottom, width: 14, height: 14, borderRadius: '50%',
                                   transform: 'translate(-50%, 50%)', cursor: 'pointer', border: '2px solid #fff',
                                   boxShadow: '0 1px 3px rgba(0,0,0,0.3)'}}
                           className={`bg-${color}`} />
                    );
                  })}
                </div>
                <div className="mt-2 small text-muted">
                  Dots: models. Dashed lines: retirement thresholds (acc=0.80, drift=0.50).
                </div>
              </div>
            </div>
          </div>

          {/* Model Age Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Model Age Distribution</div>
              <div className="card-body">
                {(() => {
                  const maxAD = Math.max(...(bd.age_distribution || []).map(b => b.count), 1);
                  return (bd.age_distribution || []).map(b => (
                    <div key={b.bucket} className="d-flex align-items-center mb-2">
                      <span className="fw-semibold" style={{minWidth: 70}}>{b.bucket}</span>
                      <div className="progress flex-grow-1 mx-2" style={{height: '20px'}}>
                        <div className={`progress-bar ${b.bucket.includes('180') || b.bucket.includes('90') ? 'bg-danger' : 'bg-primary'}`}
                             style={{width: `${b.count / maxAD * 100}%`}} />
                      </div>
                      <span className="fw-bold">{b.count}</span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>

          {/* Model Size Comparison */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Model Size Comparison (KB)</div>
              <div className="card-body">
                {(() => {
                  const maxS = Math.max(...(bd.model_size_comparison || []).map(m => m.size_kb), 1);
                  return (bd.model_size_comparison || []).map(m => (
                    <div key={m.name} className="d-flex align-items-center mb-2">
                      <span className="fw-semibold small" style={{minWidth: 130}}>{m.disease}</span>
                      <div className="progress flex-grow-1 mx-2" style={{height: '20px'}}>
                        <div className="progress-bar bg-warning" style={{width: `${m.size_kb / maxS * 100}%`}} />
                      </div>
                      <span className="fw-bold small">{m.size_kb} KB</span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>

          {/* Stage Progression Table */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Stage Progression</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Model</th><th>Disease</th><th>Age</th><th>Accuracy</th><th>Stage</th></tr></thead>
                  <tbody>
                    {(bd.stage_progression || []).map((m, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold">{m.name}</td>
                        <td>{m.disease}</td>
                        <td>{m.age_days}d</td>
                        <td>{m.accuracy !== null && m.accuracy !== undefined ? `${(m.accuracy * 100).toFixed(1)}%` : 'N/A'}</td>
                        <td><span className={`badge bg-${STAGE_COLORS[m.stage] || 'secondary'}`}>{m.stage}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Training History Tab ──────────────────────────────── */}
      {tab === 'training' && bd && (
        <div className="row">
          {/* Retirement Timeline */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Retirement Timeline (Priority Order)</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>#</th><th>Model</th><th>Disease</th><th>Accuracy</th><th>Drift</th><th>Age</th><th>Stage</th><th>Reason</th></tr></thead>
                  <tbody>
                    {(bd.retirement_timeline || []).length === 0
                      ? <tr><td colSpan={8} className="text-muted text-center">No models currently flagged for retirement</td></tr>
                      : (bd.retirement_timeline || []).map((m, i) => (
                      <tr key={i}>
                        <td className="fw-bold">{i + 1}</td>
                        <td className="small fw-semibold">{m.name}</td>
                        <td>{m.disease}</td>
                        <td>{m.accuracy !== null && m.accuracy !== undefined ? <span className={`badge bg-${accColor(m.accuracy)}`}>{(m.accuracy * 100).toFixed(1)}%</span> : <span className="badge bg-secondary">N/A</span>}</td>
                        <td>{m.drift_frac ? `${(m.drift_frac * 100).toFixed(0)}%` : '0%'}</td>
                        <td className={m.age_days > 30 ? 'text-danger fw-bold' : ''}>{m.age_days}d</td>
                        <td><span className={`badge bg-${STAGE_COLORS[m.retirement_stage] || 'secondary'}`}>{m.retirement_stage}</span></td>
                        <td className="small">{m.retirement_reason || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Training Events from track.jsonl */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Training Events (from track.jsonl)</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Timestamp</th><th>Level</th><th>Event</th></tr></thead>
                  <tbody>
                    {(bd.training_history || []).length === 0
                      ? <tr><td colSpan={3} className="text-muted text-center">No training events found in track log</td></tr>
                      : (bd.training_history || []).map((e, i) => (
                      <tr key={i}>
                        <td className="small">{e.ts}</td>
                        <td><span className="badge bg-info">{e.level}</span></td>
                        <td className="small" style={{maxWidth: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>{e.event}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Git Model History */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Git History (models/ &amp; scripts/train*)</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Hash</th><th>Date</th><th>Author</th><th>Message</th></tr></thead>
                  <tbody>
                    {(bd.git_model_history || []).length === 0
                      ? <tr><td colSpan={4} className="text-muted text-center">No git commits found for model files</td></tr>
                      : (bd.git_model_history || []).map((c, i) => (
                      <tr key={i}>
                        <td><code className="small">{c.hash}</code></td>
                        <td className="small">{c.date}</td>
                        <td className="small">{c.author}</td>
                        <td className="small" style={{maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>{c.message}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ───────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Retirement Pipeline Stages</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th style={{width: '30%'}}>Stage</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.stages || []).map((s, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{s.stage}</td>
                        <td>{s.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Metric Definitions</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th style={{width: '25%'}}>Term</th><th>Definition</th></tr></thead>
                  <tbody>
                    {(defs.metrics || []).map((m, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{m.term}</td>
                        <td>{m.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Retirement Criteria</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Criterion</th><th>Threshold</th><th>Source</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.retirement_criteria || []).map((c, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{c.criterion}</td>
                        <td><code>{c.threshold}</code></td>
                        <td className="small text-muted">{c.source}</td>
                        <td>{c.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Clinical Significance</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th style={{width: '25%'}}>Aspect</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.clinical_significance || []).map((c, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{c.aspect}</td>
                        <td>{c.description}</td>
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
