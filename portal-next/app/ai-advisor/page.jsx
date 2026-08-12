'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

/* ─── helpers ────────────────────────────────────────────────────────────── */
const pct = (n, d) => (d ? ((n / d) * 100).toFixed(1) : '0.0');

function StatCard({ label, value, color = '#3b82f6', sub }) {
  return (
    <div className="col-6 col-md mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h5 mb-0 fw-bold" style={{ color }}>{value ?? '—'}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function HBar({ label, count, total, color }) {
  const p = total ? Math.round((count / total) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{count} <span className="text-muted">({p}%)</span></span>
      </div>
      <div style={{ background: '#e5e7eb', borderRadius: 4, height: 10 }}>
        <div style={{ width: `${p}%`, background: color || '#3b82f6', borderRadius: 4, height: 10 }} />
      </div>
    </div>
  );
}

const TABS = ['Overview', 'Model Leaderboard', 'Validation Studies', 'Fusion & Decisions', 'Definitions'];

/* ─── main page ──────────────────────────────────────────────────────────── */
export default function AIAdvisorPage() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ai-advisor/overview`).then(r => r.json()),
      fetch(`${API}/api/ai-advisor/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ai-advisor/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov) return <div className="p-4 text-muted">Loading AI/ML Advisor…</div>;

  const k = ov.kpis;

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">🤖 AI / ML Advisor Dashboard</h4>
      <p className="text-muted small mb-3">
        Model benchmarks · Clinical AI decisions · Validation studies · Multimodal fusion ·
        Responsible AI — {ov.generated_at}
      </p>

      {ov.alert && (
        <div className="alert alert-warning py-2 small mb-3">⚠ {ov.alert}</div>
      )}

      {/* tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 0 && (
        <>
          <div className="row g-2 mb-3">
            <StatCard label="AI Analyses" value={k.total_ai_analyses} color="#6366f1" />
            <StatCard label="Avg Confidence" value={k.avg_ai_confidence ? `${(k.avg_ai_confidence * 100).toFixed(1)}%` : '—'} color="#3b82f6" />
            <StatCard label="Decisions Reviewed" value={k.clinical_decisions_reviewed} color="#0891b2" />
            <StatCard label="Agreement Rate" value={`${k.neurologist_agreement_rate_pct}%`} color="#22c55e" />
            <StatCard label="Override Rate" value={`${k.override_rate_pct}%`}
              color={k.override_rate_pct > 20 ? '#ef4444' : '#f59e0b'} sub="alert >20%" />
          </div>
          <div className="row g-2 mb-4">
            <StatCard label="Model Experiments" value={k.model_experiments} color="#8b5cf6" />
            <StatCard label="Best AUC-ROC" value={k.best_model_auc} color="#22c55e" sub={k.best_model_name} />
            <StatCard label="Avg AUC-ROC" value={k.avg_model_auc} color="#3b82f6" />
            <StatCard label="Validation Studies" value={k.validation_studies} color="#0891b2" />
            <StatCard label="Fusion Sessions" value={k.multimodal_fusion_sessions} color="#6366f1"
              sub={`${k.multimodal_patients} patients`} />
          </div>

          <div className="row g-3">
            {/* Disease distribution */}
            <div className="col-md-4">
              <div className="card shadow-sm">
                <div className="card-header py-2 fw-bold small">Analysis by Disease</div>
                <div className="card-body">
                  {ov.disease_distribution.map((d, i) => (
                    <HBar key={i} label={d.disease} count={d.count} total={k.total_ai_analyses}
                      color={['#6366f1','#3b82f6','#22c55e','#f59e0b','#ef4444'][i % 5]} />
                  ))}
                </div>
              </div>
            </div>

            {/* Signal quality */}
            <div className="col-md-4">
              <div className="card shadow-sm">
                <div className="card-header py-2 fw-bold small">Signal Quality Distribution</div>
                <div className="card-body">
                  {ov.signal_quality_distribution.map((d, i) => (
                    <HBar key={i} label={d.quality || 'Unknown'} count={d.count}
                      total={k.total_ai_analyses}
                      color={d.quality === 'Good' ? '#22c55e' : d.quality === 'Fair' ? '#f59e0b' : '#ef4444'} />
                  ))}
                </div>
              </div>
            </div>

            {/* Task distribution */}
            <div className="col-md-4">
              <div className="card shadow-sm">
                <div className="card-header py-2 fw-bold small">Model Experiments by Task</div>
                <div className="card-body">
                  {ov.task_distribution.map((d, i) => (
                    <HBar key={i} label={d.task} count={d.count} total={k.model_experiments}
                      color={['#6366f1','#3b82f6','#22c55e','#f59e0b','#8b5cf6','#ef4444'][i % 6]} />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Validation summary */}
          <div className="row g-2 mt-3">
            <div className="col-md-4">
              <div className="card shadow-sm text-center py-3">
                <div className="h3 fw-bold text-success">{k.validation_passed}</div>
                <div className="text-muted small">Passed</div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card shadow-sm text-center py-3">
                <div className="h3 fw-bold text-danger">{k.validation_failed}</div>
                <div className="text-muted small">Failed / Remediation</div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card shadow-sm text-center py-3">
                <div className="h3 fw-bold text-warning">{k.validation_ongoing}</div>
                <div className="text-muted small">In Progress</div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── MODEL LEADERBOARD ── */}
      {tab === 1 && bk && (
        <>
          {/* By model type */}
          <div className="row g-3 mb-4">
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header py-2 fw-bold small">Performance by Model Type</div>
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.8rem' }}>
                    <thead className="table-light">
                      <tr>
                        <th>Type</th><th>N</th><th>Avg AUC</th><th>Best AUC</th>
                        <th>Avg F1</th><th>Infer ms</th>
                      </tr>
                    </thead>
                    <tbody>
                      {bk.performance_by_type.map((r, i) => (
                        <tr key={i}>
                          <td className="fw-semibold">{r.model_type}</td>
                          <td>{r.n}</td>
                          <td><span style={{ color: r.avg_auc >= 0.9 ? '#22c55e' : r.avg_auc >= 0.8 ? '#3b82f6' : '#f59e0b' }}>{r.avg_auc ?? '—'}</span></td>
                          <td>{r.best_auc ?? '—'}</td>
                          <td>{r.avg_f1 ?? '—'}</td>
                          <td>{r.avg_infer_ms ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header py-2 fw-bold small">Performance by Task</div>
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.8rem' }}>
                    <thead className="table-light">
                      <tr><th>Task</th><th>N</th><th>Avg AUC</th><th>Best AUC</th></tr>
                    </thead>
                    <tbody>
                      {bk.performance_by_task.map((r, i) => (
                        <tr key={i}>
                          <td className="fw-semibold">{r.task}</td>
                          <td>{r.n}</td>
                          <td>{r.avg_auc ?? '—'}</td>
                          <td>{r.best_auc ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Full leaderboard */}
          <div className="card shadow-sm">
            <div className="card-header py-2 fw-bold small">Full Model Leaderboard (top 30 by AUC-ROC)</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.75rem' }}>
                <thead className="table-light">
                  <tr>
                    <th>Model</th><th>Type</th><th>Task</th><th>Acc</th>
                    <th>Prec</th><th>Recall</th><th>F1</th><th>AUC</th>
                    <th>Train s</th><th>Inf ms</th><th>Status</th><th>Ver</th>
                  </tr>
                </thead>
                <tbody>
                  {bk.model_leaderboard.map((r, i) => (
                    <tr key={i}>
                      <td className="fw-semibold" style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.model_name}</td>
                      <td>{r.model_type}</td>
                      <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.task}</td>
                      <td>{r.accuracy != null ? (r.accuracy * 100).toFixed(1) + '%' : '—'}</td>
                      <td>{r.precision != null ? (r.precision * 100).toFixed(1) + '%' : '—'}</td>
                      <td>{r.recall != null ? (r.recall * 100).toFixed(1) + '%' : '—'}</td>
                      <td>{r.f1 != null ? (r.f1 * 100).toFixed(1) + '%' : '—'}</td>
                      <td>
                        <span style={{ color: r.auc_roc >= 0.9 ? '#22c55e' : r.auc_roc >= 0.8 ? '#3b82f6' : '#f59e0b', fontWeight: 600 }}>
                          {r.auc_roc ?? '—'}
                        </span>
                      </td>
                      <td>{r.train_sec ?? '—'}</td>
                      <td>{r.infer_ms ?? '—'}</td>
                      <td><span className={`badge ${r.status === 'completed' ? 'bg-success' : 'bg-secondary'}`} style={{ fontSize: '0.65rem' }}>{r.status}</span></td>
                      <td>{r.version}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── VALIDATION STUDIES ── */}
      {tab === 2 && bk && (
        <div className="card shadow-sm">
          <div className="card-header py-2 fw-bold small">Validation Studies ({bk.validation_studies.length})</div>
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.78rem' }}>
              <thead className="table-light">
                <tr>
                  <th>Study ID</th><th>Type</th><th>Title</th><th>Status</th>
                  <th>N</th><th>Sens</th><th>Spec</th><th>AUC</th>
                  <th>PI</th><th>Site</th><th>Start</th><th>End</th>
                </tr>
              </thead>
              <tbody>
                {bk.validation_studies.map((r, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{r.study_id}</td>
                    <td>{r.study_type}</td>
                    <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.title}</td>
                    <td>
                      <span className={`badge ${r.status?.includes('Pass') || r.status === 'Completed' ? 'bg-success' : r.status?.includes('Fail') ? 'bg-danger' : r.status?.includes('Progress') || r.status === 'Active' ? 'bg-warning text-dark' : 'bg-secondary'}`} style={{ fontSize: '0.65rem' }}>
                        {r.status}
                      </span>
                    </td>
                    <td>{r.sample_size ?? '—'}</td>
                    <td>{r.sensitivity != null ? (r.sensitivity * 100).toFixed(1) + '%' : '—'}</td>
                    <td>{r.specificity != null ? (r.specificity * 100).toFixed(1) + '%' : '—'}</td>
                    <td>{r.auc_roc ?? '—'}</td>
                    <td>{r.pi}</td>
                    <td>{r.site}</td>
                    <td>{r.start_date}</td>
                    <td>{r.end_date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── FUSION & DECISIONS ── */}
      {tab === 3 && bk && (
        <>
          <div className="row g-3 mb-4">
            {/* Confidence distribution */}
            <div className="col-md-4">
              <div className="card shadow-sm">
                <div className="card-header py-2 fw-bold small">AI Confidence Distribution</div>
                <div className="card-body">
                  {bk.confidence_distribution.map((d, i) => {
                    const total = bk.confidence_distribution.reduce((s, x) => s + x.count, 0);
                    const COLORS = ['#ef4444', '#f97316', '#f59e0b', '#3b82f6', '#22c55e', '#6366f1'];
                    return <HBar key={i} label={d.range} count={d.count} total={total} color={COLORS[i]} />;
                  })}
                </div>
              </div>
            </div>

            {/* Agreement by class */}
            <div className="col-md-8">
              <div className="card shadow-sm">
                <div className="card-header py-2 fw-bold small">AI Agreement by Prediction Class</div>
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.8rem' }}>
                    <thead className="table-light">
                      <tr><th>Prediction</th><th>Total</th><th>Agreed</th><th>Overridden</th><th>Avg Conf</th></tr>
                    </thead>
                    <tbody>
                      {bk.agreement_by_class.map((r, i) => (
                        <tr key={i}>
                          <td className="fw-semibold">{r.prediction}</td>
                          <td>{r.total}</td>
                          <td><span className="text-success fw-bold">{r.agreed}</span> ({pct(r.agreed, r.total)}%)</td>
                          <td><span className={r.overridden / r.total > 0.2 ? 'text-danger fw-bold' : 'text-warning fw-bold'}>{r.overridden}</span> ({pct(r.overridden, r.total)}%)</td>
                          <td>{r.avg_conf != null ? (r.avg_conf * 100).toFixed(1) + '%' : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Fusion by method */}
          <div className="row g-3">
            <div className="col-md-5">
              <div className="card shadow-sm">
                <div className="card-header py-2 fw-bold small">Multimodal Fusion by Method</div>
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.8rem' }}>
                    <thead className="table-light">
                      <tr><th>Method</th><th>N</th><th>Avg Conf</th><th>Concordance</th><th>Avg Mods</th><th>Avg s</th></tr>
                    </thead>
                    <tbody>
                      {bk.fusion_by_method.map((r, i) => (
                        <tr key={i}>
                          <td className="fw-semibold">{r.method}</td>
                          <td>{r.n}</td>
                          <td>{r.avg_conf != null ? (r.avg_conf * 100).toFixed(1) + '%' : '—'}</td>
                          <td>{r.avg_concordance ?? '—'}</td>
                          <td>{r.avg_modalities ?? '—'}</td>
                          <td>{r.avg_sec ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-7">
              <div className="card shadow-sm">
                <div className="card-header py-2 fw-bold small">Fusion by Predicted Subtype</div>
                <div className="card-body">
                  {(() => {
                    const total = bk.fusion_by_subtype.reduce((s, x) => s + x.n, 0);
                    const COLORS = ['#6366f1', '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#0891b2', '#f97316'];
                    return bk.fusion_by_subtype.map((d, i) => (
                      <HBar key={i} label={d.subtype} count={d.n} total={total} color={COLORS[i % COLORS.length]} />
                    ));
                  })()}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 4 && df && (
        <>
          {/* Role card */}
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-bold small">{df.role.title}</div>
            <div className="card-body small">
              <p>{df.role.scope}</p>
              <strong>Responsibilities:</strong>
              <ul className="mb-0 mt-1">
                {df.role.responsibilities.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </div>
          </div>

          {/* Performance thresholds */}
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-bold small">Performance Thresholds (SaMD Regulatory)</div>
            <div className="table-responsive">
              <table className="table table-sm mb-0" style={{ fontSize: '0.8rem' }}>
                <thead className="table-light">
                  <tr><th>Metric</th><th>Target</th><th>Rationale</th></tr>
                </thead>
                <tbody>
                  {df.performance_thresholds.map((r, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{r.metric}</td>
                      <td><span className="badge bg-primary">{r.target}</span></td>
                      <td>{r.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Concepts */}
          <div className="row g-2 mb-3">
            {df.concepts.map((c, i) => (
              <div key={i} className="col-md-6">
                <div className="card shadow-sm h-100">
                  <div className="card-header py-1 fw-bold small">{c.term}</div>
                  <div className="card-body py-2 small">{c.definition}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Regulatory context */}
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-bold small">Regulatory Context</div>
            <div className="table-responsive">
              <table className="table table-sm mb-0" style={{ fontSize: '0.8rem' }}>
                <thead className="table-light"><tr><th>Standard</th><th>Applicability</th></tr></thead>
                <tbody>
                  {df.regulatory_context.map((r, i) => (
                    <tr key={i}><td className="fw-semibold">{r.standard}</td><td>{r.note}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* AI governance notes */}
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-bold small">AI Governance Notes</div>
            <ul className="list-group list-group-flush">
              {df.ai_governance_notes.map((n, i) => (
                <li key={i} className="list-group-item small">{n}</li>
              ))}
            </ul>
          </div>

          {/* References */}
          <div className="card shadow-sm">
            <div className="card-header py-2 fw-bold small">References</div>
            <ol className="list-group list-group-flush list-group-numbered">
              {df.references.map((r, i) => (
                <li key={i} className="list-group-item small">{r}</li>
              ))}
            </ol>
          </div>
        </>
      )}
    </div>
  );
}
