'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const typeColor = t =>
  t === 'XGBoost'           ? 'success'  :
  t === 'LightGBM'          ? 'primary'  :
  t === 'RandomForest'      ? 'info'     :
  t === 'MLP'               ? 'warning'  :
  t === 'SVM'               ? 'secondary':
  'dark';

const taskColor = t =>
  t === 'seizure_detection'  ? 'danger'   :
  t === 'eeg_classification' ? 'primary'  :
  t === 'seizure_prediction' ? 'warning'  :
  'secondary';

const statusBadge = s => s === 'completed' ? 'success' : 'danger';

const pct = (v, total) => total ? ((v / total) * 100).toFixed(1) : 0;

export default function ModelComparisonPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [sort, setSort] = useState('accuracy');

  useEffect(() => {
    fetch(`${API}/api/model-comparison/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/model-comparison/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/model-comparison/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpi = ov.kpis || {};

  const tabs = [
    { id: 'overview',  label: 'Overview' },
    { id: 'leaderboard', label: 'Leaderboard' },
    { id: 'by-type',   label: 'By Model Type' },
    { id: 'by-task',   label: 'By Task' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const sortedModels = ((bd || {}).models || []).slice().sort((a, b) => {
    if (sort === 'accuracy') return (b.accuracy || 0) - (a.accuracy || 0);
    if (sort === 'auc')      return (b.auc_roc || 0) - (a.auc_roc || 0);
    if (sort === 'f1')       return (b.f1_score || 0) - (a.f1_score || 0);
    if (sort === 'speed')    return (a.inference_time_ms || 999) - (b.inference_time_ms || 999);
    return 0;
  });

  return (
    <div>
      <h3>&#x1f4ca; Model Comparison Dashboard</h3>
      <p className="text-muted small">
        Cross-model performance leaderboard — {kpi.total_models} training runs across {kpi.distinct_model_types} model
        types, {kpi.distinct_tasks} clinical tasks, {kpi.distinct_datasets} EEG datasets.
        Best accuracy: <strong>{((kpi.best_accuracy || 0) * 100).toFixed(0)}%</strong> ({kpi.best_accuracy_model}).
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Runs',        value: kpi.total_models,                                color: 'primary' },
          { label: 'Completed',         value: kpi.completed_count,                             color: 'success' },
          { label: 'Failed',            value: kpi.failed_count,                                color: 'danger'  },
          { label: 'Model Types',       value: kpi.distinct_model_types,                        color: 'info'    },
          { label: 'Tasks',             value: kpi.distinct_tasks,                              color: 'warning' },
          { label: 'Datasets',          value: kpi.distinct_datasets,                           color: 'secondary'},
          { label: 'Avg Accuracy',      value: `${((kpi.avg_accuracy || 0) * 100).toFixed(1)}%`, color: 'success'},
          { label: 'Avg AUC-ROC',       value: `${((kpi.avg_auc_roc  || 0) * 100).toFixed(1)}%`, color: 'primary'},
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value ?? '—'}</div>
                <div className="text-muted" style={{fontSize: '0.72rem'}}>{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Model type distribution */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Model Type Distribution</div>
              <div className="card-body p-2">
                {(ov.model_type_dist || []).map(d => {
                  const p = pct(d.count, kpi.total_models);
                  return (
                    <div key={d.model_type} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span><span className={`badge bg-${typeColor(d.model_type)} me-2`}>{d.model_type}</span></span>
                        <span className="fw-bold">{d.count} <span className="text-muted">({p}%)</span></span>
                      </div>
                      <div className="progress" style={{height: '8px'}}>
                        <div className={`progress-bar bg-${typeColor(d.model_type)}`} style={{width: `${p}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Task distribution */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Task Distribution</div>
              <div className="card-body p-2">
                {(ov.task_dist || []).map(d => {
                  const p = pct(d.count, kpi.total_models);
                  const label = d.task.replace(/_/g, ' ');
                  return (
                    <div key={d.task} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span><span className={`badge bg-${taskColor(d.task)} me-2`}>{label}</span></span>
                        <span className="fw-bold">{d.count}</span>
                      </div>
                      <div className="progress" style={{height: '8px'}}>
                        <div className={`progress-bar bg-${taskColor(d.task)}`} style={{width: `${p}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Dataset distribution */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Dataset Distribution</div>
              <div className="card-body p-2">
                {(ov.dataset_dist || []).map(d => {
                  const p = pct(d.count, kpi.total_models);
                  return (
                    <div key={d.dataset} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize">{d.dataset.replace(/_/g, ' ')}</span>
                        <span className="fw-bold">{d.count}</span>
                      </div>
                      <div className="progress" style={{height: '8px'}}>
                        <div className="progress-bar bg-dark" style={{width: `${p}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Accuracy by model type */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Avg Accuracy by Model Type</div>
              <div className="card-body p-2">
                {(ov.accuracy_by_model_type || []).sort((a, b) => b.avg_accuracy - a.avg_accuracy).map(d => {
                  const accPct = (d.avg_accuracy * 100).toFixed(1);
                  return (
                    <div key={d.model_type} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold">{d.model_type}</span>
                        <span className="text-muted">
                          Acc <strong className={`text-${typeColor(d.model_type)}`}>{accPct}%</strong>
                          {' '}· F1 {(d.avg_f1 * 100).toFixed(1)}%
                          {' '}· AUC {(d.avg_auc_roc * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="progress" style={{height: '12px'}}>
                        <div className={`progress-bar bg-${typeColor(d.model_type)}`} style={{width: `${accPct}%`}}>
                          {accPct}%
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Accuracy by task */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Avg Accuracy by Task</div>
              <div className="card-body p-2">
                {(ov.accuracy_by_task || []).sort((a, b) => b.avg_accuracy - a.avg_accuracy).map(d => {
                  const accPct = (d.avg_accuracy * 100).toFixed(1);
                  return (
                    <div key={d.task} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold text-capitalize">{d.task.replace(/_/g, ' ')}</span>
                        <span className="text-muted">
                          Acc <strong className={`text-${taskColor(d.task)}`}>{accPct}%</strong>
                          {' '}· AUC {(d.avg_auc_roc * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="progress" style={{height: '12px'}}>
                        <div className={`progress-bar bg-${taskColor(d.task)}`} style={{width: `${accPct}%`}}>
                          {accPct}%
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Monthly trend */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Monthly Training Volume &amp; Avg Accuracy</div>
              <div className="card-body p-2" style={{overflowX: 'auto'}}>
                <table className="table table-sm table-hover mb-0" style={{fontSize:'0.8rem'}}>
                  <thead className="table-light">
                    <tr>
                      <th>Month</th>
                      <th>Runs</th>
                      <th>Avg Accuracy</th>
                      <th>Accuracy Bar</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.monthly_trend || []).map(m => (
                      <tr key={m.month}>
                        <td>{m.month}</td>
                        <td><span className="badge bg-secondary">{m.count}</span></td>
                        <td className="fw-bold">{((m.avg_accuracy || 0) * 100).toFixed(1)}%</td>
                        <td style={{minWidth: '120px'}}>
                          <div className="progress" style={{height: '8px'}}>
                            <div className="progress-bar bg-success" style={{width: `${((m.avg_accuracy || 0) * 100).toFixed(0)}%`}} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Leaderboard Tab ── */}
      {tab === 'leaderboard' && bd && (
        <div>
          <div className="d-flex gap-2 mb-3 align-items-center flex-wrap">
            <span className="small text-muted">Sort by:</span>
            {[
              { key: 'accuracy', label: 'Accuracy' },
              { key: 'auc',      label: 'AUC-ROC'  },
              { key: 'f1',       label: 'F1 Score'  },
              { key: 'speed',    label: 'Speed (ms)'},
            ].map(s => (
              <button
                key={s.key}
                className={`btn btn-sm ${sort === s.key ? 'btn-primary' : 'btn-outline-secondary'}`}
                onClick={() => setSort(s.key)}
              >
                {s.label}
              </button>
            ))}
            <span className="text-muted small ms-2">Showing top {Math.min(50, sortedModels.length)} of {sortedModels.length}</span>
          </div>
          <div style={{overflowX: 'auto'}}>
            <table className="table table-sm table-hover table-bordered" style={{fontSize: '0.78rem'}}>
              <thead className="table-dark">
                <tr>
                  <th>#</th>
                  <th>Model</th>
                  <th>Type</th>
                  <th>Task</th>
                  <th>Dataset</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>AUC-ROC</th>
                  <th>Infer ms</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {sortedModels.slice(0, 50).map((m, idx) => (
                  <tr key={m.id}>
                    <td className="text-muted">{idx + 1}</td>
                    <td className="fw-semibold">{m.model_name}</td>
                    <td><span className={`badge bg-${typeColor(m.model_type)}`}>{m.model_type}</span></td>
                    <td><span className="badge bg-secondary text-capitalize" style={{fontSize:'0.68rem'}}>{(m.task || '').replace(/_/g, ' ')}</span></td>
                    <td className="text-muted small">{(m.dataset || '').replace(/_/g, ' ')}</td>
                    <td className={`fw-bold text-${m.accuracy >= 0.9 ? 'success' : m.accuracy >= 0.8 ? 'primary' : 'warning'}`}>
                      {((m.accuracy || 0) * 100).toFixed(1)}%
                    </td>
                    <td>{((m.precision_score || 0) * 100).toFixed(1)}%</td>
                    <td>{((m.recall || 0) * 100).toFixed(1)}%</td>
                    <td>{((m.f1_score || 0) * 100).toFixed(1)}%</td>
                    <td className={`fw-bold text-${(m.auc_roc || 0) >= 0.95 ? 'success' : 'primary'}`}>
                      {((m.auc_roc || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="text-muted">{(m.inference_time_ms || 0).toFixed(2)}</td>
                    <td><span className={`badge bg-${statusBadge(m.status)}`}>{m.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── By Model Type Tab ── */}
      {tab === 'by-type' && (
        <div className="row g-3">
          {(ov.accuracy_by_model_type || []).sort((a, b) => b.avg_accuracy - a.avg_accuracy).map(d => (
            <div key={d.model_type} className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header d-flex justify-content-between align-items-center">
                  <span className="fw-semibold">{d.model_type}</span>
                  <span className={`badge bg-${typeColor(d.model_type)}`}>
                    {((d.avg_accuracy || 0) * 100).toFixed(1)}% acc
                  </span>
                </div>
                <div className="card-body p-3">
                  {[
                    { label: 'Avg Accuracy',  value: `${((d.avg_accuracy || 0) * 100).toFixed(2)}%`,  color: typeColor(d.model_type) },
                    { label: 'Avg F1 Score',  value: `${((d.avg_f1      || 0) * 100).toFixed(2)}%`,  color: typeColor(d.model_type) },
                    { label: 'Avg AUC-ROC',   value: `${((d.avg_auc_roc || 0) * 100).toFixed(2)}%`,  color: typeColor(d.model_type) },
                  ].map(row => (
                    <div key={row.label} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-muted">{row.label}</span>
                        <span className={`fw-bold text-${row.color}`}>{row.value}</span>
                      </div>
                      <div className="progress" style={{height: '10px'}}>
                        <div className={`progress-bar bg-${row.color}`} style={{width: row.value}} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── By Task Tab ── */}
      {tab === 'by-task' && (
        <div className="row g-3">
          {(ov.accuracy_by_task || []).sort((a, b) => b.avg_accuracy - a.avg_accuracy).map(d => (
            <div key={d.task} className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header d-flex justify-content-between align-items-center">
                  <span className="fw-semibold text-capitalize">{(d.task || '').replace(/_/g, ' ')}</span>
                  <span className={`badge bg-${taskColor(d.task)}`}>
                    {((d.avg_accuracy || 0) * 100).toFixed(1)}% avg acc
                  </span>
                </div>
                <div className="card-body p-3">
                  {[
                    { label: 'Avg Accuracy', val: d.avg_accuracy },
                    { label: 'Avg F1 Score', val: d.avg_f1 },
                    { label: 'Avg AUC-ROC',  val: d.avg_auc_roc },
                  ].map(row => (
                    <div key={row.label} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-muted">{row.label}</span>
                        <span className={`fw-bold text-${taskColor(d.task)}`}>{((row.val || 0) * 100).toFixed(2)}%</span>
                      </div>
                      <div className="progress" style={{height: '10px'}}>
                        <div className={`progress-bar bg-${taskColor(d.task)}`} style={{width: `${((row.val || 0) * 100).toFixed(1)}%`}} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-md-7">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Metric Definitions</div>
              <div className="card-body p-2" style={{maxHeight: '520px', overflowY: 'auto'}}>
                {(defs.concepts || []).map(c => (
                  <div key={c.name} className="mb-3 pb-3 border-bottom">
                    <div className="fw-semibold small">{c.name}</div>
                    <div className="text-muted" style={{fontSize: '0.8rem'}}>{c.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-5">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">Supported Model Types</div>
              <div className="card-body p-2">
                {(ov.model_type_dist || []).map(d => (
                  <div key={d.model_type} className="d-flex align-items-center mb-2">
                    <span className={`badge bg-${typeColor(d.model_type)} me-2`}>{d.model_type}</span>
                    <span className="small text-muted">{d.count} training runs</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Clinical Tasks</div>
              <div className="card-body p-2">
                {(ov.task_dist || []).map(d => (
                  <div key={d.task} className="d-flex align-items-center mb-2">
                    <span className={`badge bg-${taskColor(d.task)} me-2`}>{(d.task || '').replace(/_/g, ' ')}</span>
                    <span className="small text-muted">{d.count} runs</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
