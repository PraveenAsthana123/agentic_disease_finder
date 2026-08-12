'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',     label: 'Overview' },
  { id: 'architecture', label: 'Architectures' },
  { id: 'comparison',   label: 'Baseline vs DL' },
  { id: 'training',     label: 'Training Design' },
  { id: 'definitions',  label: 'Definitions' },
];

const CAT_COLOR = {
  'Hybrid':      'primary',
  'Baseline DL': 'info',
  'Baseline ML': 'secondary',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-2 mb-2">
      <div className="card text-center shadow-sm border-0 h-100">
        <div className="card-body py-2 px-1">
          <div className={`h3 mb-0 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted" style={{ fontSize: '0.72rem' }}>{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function AucBar({ val, max = 1.0, color }) {
  const pct = Math.min((val / max) * 100, 100);
  const c = color || (pct >= 97 ? 'success' : pct >= 92 ? 'primary' : pct >= 88 ? 'info' : 'warning');
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 14, borderRadius: 6 }}>
        <div className={`progress-bar bg-${c}`} style={{ width: `${pct}%`, borderRadius: 6, transition: 'width 0.6s' }} />
      </div>
      <small className="text-muted" style={{ width: 44, textAlign: 'right' }}>{(val * 100).toFixed(1)}%</small>
    </div>
  );
}

/* ── Overview Tab ──────────────────────────────────────────────── */
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;
  const k = ov.kpis || {};
  const tbl = ov.comparison_table || [];
  const pipe = ov.pipeline_stages || [];
  const proj = ov.dataset_projections || [];

  return (
    <>
      <div className="row g-2 mb-3">
        <KPI label="Architectures Designed" value={k.architectures_designed}       color="primary" />
        <KPI label="Hybrid Models"           value={k.hybrid_architectures}         color="info" />
        <KPI label="Best DL AUC"             value={k.best_dl_auc}                  color="success" sub="CNN-Transformer" />
        <KPI label="Baseline Best AUC"       value={k.best_baseline_auc}            color="secondary" sub="XGBoost" />
        <KPI label="AUC Lift (DL vs ML)"     value={k.projected_auc_lift_pct != null ? `+${k.projected_auc_lift_pct}%` : '—'} color="success" />
        <KPI label="Datasets Covered"        value={k.datasets_covered}             color="warning" />
      </div>

      {/* Architecture AUC lift bar */}
      <div className="card shadow-sm border-0 mb-3">
        <div className="card-header py-2 fw-semibold">Architecture Performance — Expected AUC-ROC</div>
        <div className="card-body py-3">
          <div className="table-responsive">
            <table className="table table-sm table-hover align-middle mb-0">
              <thead className="table-light">
                <tr>
                  <th>Architecture</th>
                  <th>Category</th>
                  <th style={{ width: '35%' }}>Expected AUC-ROC</th>
                  <th>Sensitivity</th>
                  <th>Params (M)</th>
                  <th>Latency (ms)</th>
                </tr>
              </thead>
              <tbody>
                {tbl.map((row, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{row.architecture}</td>
                    <td>
                      <span className={`badge bg-${CAT_COLOR[row.category] || 'secondary'}`}>
                        {row.category}
                      </span>
                    </td>
                    <td><AucBar val={row.expected_auc} /></td>
                    <td>{row.expected_sensitivity != null ? `${(row.expected_sensitivity * 100).toFixed(1)}%` : '—'}</td>
                    <td>{row.params_M != null ? row.params_M : '—'}</td>
                    <td>{row.inference_ms != null ? row.inference_ms : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Dataset projections */}
      <div className="card shadow-sm border-0 mb-3">
        <div className="card-header py-2 fw-semibold">CNN-Transformer Projected Performance by Dataset</div>
        <div className="card-body py-2">
          <div className="table-responsive">
            <table className="table table-sm align-middle mb-0">
              <thead className="table-light">
                <tr><th>Dataset</th><th>Subjects</th><th>Samples</th><th>Proj. AUC</th><th>Proj. Sensitivity</th><th>Proj. Specificity</th></tr>
              </thead>
              <tbody>
                {proj.map((p, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{p.dataset}</td>
                    <td>{p.n_subjects}</td>
                    <td>{p.n_samples.toLocaleString()}</td>
                    <td><span className="text-success fw-bold">{p.projected_auc}</span></td>
                    <td>{(p.projected_sensitivity * 100).toFixed(1)}%</td>
                    <td>{(p.projected_specificity * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Pipeline stages */}
      <div className="card shadow-sm border-0">
        <div className="card-header py-2 fw-semibold">End-to-End Pipeline — {pipe.length} Stages</div>
        <div className="card-body py-2">
          <div className="table-responsive">
            <table className="table table-sm align-middle mb-0">
              <thead className="table-light">
                <tr><th>#</th><th>Stage</th><th>Tool / Method</th><th>Output</th></tr>
              </thead>
              <tbody>
                {pipe.map((s) => (
                  <tr key={s.stage}>
                    <td><span className="badge bg-primary rounded-pill">{s.stage}</span></td>
                    <td className="fw-semibold">{s.name}</td>
                    <td><code className="text-secondary" style={{ fontSize: '0.78rem' }}>{s.tool}</code></td>
                    <td className="text-muted" style={{ fontSize: '0.82rem' }}>{s.output}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  );
}

/* ── Architecture Tab ──────────────────────────────────────────── */
function ArchitectureTab({ bd }) {
  const [selected, setSelected] = useState(null);
  if (!bd) return <div className="text-muted p-3">Loading…</div>;
  const cards = bd.architecture_cards || [];
  const sel = selected != null ? cards[selected] : null;

  return (
    <div className="row g-3">
      {/* Arch selector */}
      <div className="col-md-4">
        {cards.map((a, i) => (
          <div
            key={a.id}
            className={`card shadow-sm border-0 mb-2 ${selected === i ? 'border border-primary' : ''}`}
            style={{ cursor: 'pointer', borderLeft: selected === i ? '4px solid var(--bs-primary)' : '4px solid transparent' }}
            onClick={() => setSelected(i === selected ? null : i)}
          >
            <div className="card-body py-2 px-3">
              <div className="d-flex justify-content-between align-items-center">
                <span className="fw-semibold">{a.name}</span>
                <span className={`badge bg-${CAT_COLOR[a.category] || 'secondary'}`}>{a.category}</span>
              </div>
              <div className="d-flex gap-3 mt-1">
                <small className="text-success">AUC {a.expected_auc}</small>
                <small className="text-muted">{a.params_M}M params</small>
                <small className="text-info">{a.inference_ms}ms</small>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Detail panel */}
      <div className="col-md-8">
        {sel ? (
          <div className="card shadow-sm border-0 h-100">
            <div className="card-header py-2">
              <span className="fw-semibold">{sel.full_name}</span>
              <span className={`badge bg-${CAT_COLOR[sel.category] || 'secondary'} ms-2`}>{sel.category}</span>
            </div>
            <div className="card-body">
              <p className="text-muted" style={{ fontSize: '0.88rem' }}>{sel.description}</p>

              <div className="row g-2 mb-3">
                {[
                  ['Expected AUC', sel.expected_auc, 'success'],
                  ['Sensitivity', `${(sel.expected_sensitivity * 100).toFixed(1)}%`, 'primary'],
                  ['Specificity', `${(sel.expected_specificity * 100).toFixed(1)}%`, 'info'],
                  ['Parameters', `${sel.params_M}M`, 'secondary'],
                  ['Inference', `${sel.inference_ms}ms`, 'warning'],
                  ['Training', `${sel.training_time_min}min`, 'secondary'],
                ].map(([label, val, color]) => (
                  <div key={label} className="col-6 col-md-4">
                    <div className="card border-0 bg-light text-center py-1">
                      <div className={`fw-bold text-${color}`}>{val}</div>
                      <div className="text-muted" style={{ fontSize: '0.7rem' }}>{label}</div>
                    </div>
                  </div>
                ))}
              </div>

              <h6 className="fw-semibold mb-2">Layer Architecture</h6>
              <div className="table-responsive mb-3">
                <table className="table table-sm table-bordered align-middle mb-0">
                  <thead className="table-light"><tr><th>Layer</th><th>Detail</th></tr></thead>
                  <tbody>
                    {(sel.stages || []).map((s, i) => (
                      <tr key={i}>
                        <td><code className="text-primary">{s.layer}</code></td>
                        <td className="text-muted" style={{ fontSize: '0.82rem' }}>{s.detail}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="row g-2 mb-2">
                <div className="col-md-6">
                  <div className="alert alert-success py-2 mb-0" style={{ fontSize: '0.82rem' }}>
                    <strong>Advantage:</strong> {sel.advantage}
                  </div>
                </div>
                <div className="col-md-6">
                  <div className="alert alert-warning py-2 mb-0" style={{ fontSize: '0.82rem' }}>
                    <strong>Limitation:</strong> {sel.limitation}
                  </div>
                </div>
              </div>
              <div className="text-muted mt-2" style={{ fontSize: '0.78rem' }}>
                <strong>Tasks:</strong> {(sel.tasks || []).join(' · ')}<br />
                <strong>Reference:</strong> {sel.reference}
              </div>
            </div>
          </div>
        ) : (
          <div className="card shadow-sm border-0 h-100 d-flex align-items-center justify-content-center text-muted">
            Select an architecture on the left to view details.
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Comparison Tab ────────────────────────────────────────────── */
function ComparisonTab({ bd }) {
  if (!bd) return <div className="text-muted p-3">Loading…</div>;
  const taskCmp = bd.task_comparison || [];
  const ablation = bd.ablation_study || [];

  return (
    <>
      {/* Per-task comparison */}
      <div className="card shadow-sm border-0 mb-3">
        <div className="card-header py-2 fw-semibold">Baseline ML vs Deep Learning — Per Task (AUC-ROC)</div>
        <div className="card-body py-2">
          <div className="table-responsive">
            <table className="table table-sm table-hover align-middle mb-0">
              <thead className="table-light">
                <tr>
                  <th>Task</th>
                  <th>Baseline Avg AUC</th>
                  <th>Baseline Best AUC</th>
                  <th>EEGNet-LSTM</th>
                  <th>CNN-LSTM</th>
                  <th>CNN-Transformer</th>
                  <th>N Runs</th>
                </tr>
              </thead>
              <tbody>
                {taskCmp.map((r, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{r.task}</td>
                    <td>{r.baseline_auc}</td>
                    <td>{r.baseline_best}</td>
                    <td>{r.eegnet_lstm_auc ? <span className="text-info">{r.eegnet_lstm_auc}</span> : '—'}</td>
                    <td>{r.cnn_lstm_auc    ? <span className="text-primary">{r.cnn_lstm_auc}</span>    : '—'}</td>
                    <td>{r.cnn_transformer_auc ? <span className="text-success fw-bold">{r.cnn_transformer_auc}</span> : '—'}</td>
                    <td className="text-muted">{r.n_runs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Ablation */}
      <div className="card shadow-sm border-0 mb-3">
        <div className="card-header py-2 fw-semibold">Ablation Study — Component Contribution to AUC</div>
        <div className="card-body py-2">
          <div className="table-responsive">
            <table className="table table-sm align-middle mb-0">
              <thead className="table-light">
                <tr><th>Variant</th><th style={{ width: '35%' }}>AUC-ROC</th><th>Sensitivity</th><th>Specificity</th><th>Δ AUC</th></tr>
              </thead>
              <tbody>
                {ablation.map((r, i) => (
                  <tr key={i} className={i === ablation.length - 1 ? 'table-success fw-semibold' : ''}>
                    <td>{r.variant}</td>
                    <td><AucBar val={r.auc} /></td>
                    <td>{(r.sens * 100).toFixed(1)}%</td>
                    <td>{(r.spec * 100).toFixed(1)}%</td>
                    <td>
                      <span className={r.delta_auc.startsWith('+') ? 'text-success' : r.delta_auc.startsWith('−') ? 'text-danger' : 'text-muted'}>
                        {r.delta_auc}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-muted mt-2" style={{ fontSize: '0.78rem' }}>
            Ablation baseline = XGBoost (best current ML). Each row adds one component to the pipeline.
          </div>
        </div>
      </div>

      {/* Disease breakdown */}
      {bd.disease_breakdown && (
        <div className="card shadow-sm border-0">
          <div className="card-header py-2 fw-semibold">Clinical.db Analysis Coverage by Disease</div>
          <div className="card-body py-2">
            <div className="table-responsive">
              <table className="table table-sm align-middle mb-0">
                <thead className="table-light"><tr><th>Disease</th><th>Analyses</th><th>Avg Confidence</th></tr></thead>
                <tbody>
                  {bd.disease_breakdown.map((r, i) => (
                    <tr key={i}>
                      <td className="fw-semibold text-capitalize">{r.disease}</td>
                      <td>{r.n}</td>
                      <td>{r.avg_conf ? `${(r.avg_conf * 100).toFixed(1)}%` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/* ── Training Design Tab ───────────────────────────────────────── */
function TrainingTab({ bd }) {
  if (!bd) return <div className="text-muted p-3">Loading…</div>;
  const td = bd.training_design || {};
  const hg = bd.hyperparam_grid || {};

  return (
    <div className="row g-3">
      <div className="col-md-6">
        <div className="card shadow-sm border-0 h-100">
          <div className="card-header py-2 fw-semibold">Training Configuration</div>
          <div className="card-body py-2">
            {[
              ['Data Split',       td.split],
              ['Loss Function',    td.loss],
              ['Optimizer',        td.optimizer],
              ['LR Scheduler',     td.scheduler],
              ['Early Stopping',   td.early_stopping],
              ['Framework',        td.framework],
              ['Hardware',         td.hardware],
            ].map(([label, val]) => val && (
              <div key={label} className="mb-2">
                <strong className="text-muted" style={{ fontSize: '0.8rem' }}>{label}:</strong>
                <div style={{ fontSize: '0.85rem' }}>{val}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="col-md-6">
        <div className="card shadow-sm border-0 mb-3">
          <div className="card-header py-2 fw-semibold">Regularisation Techniques</div>
          <div className="card-body py-2">
            {(td.regularisation || []).map((r, i) => (
              <span key={i} className="badge bg-secondary me-1 mb-1">{r}</span>
            ))}
          </div>
        </div>
        <div className="card shadow-sm border-0 mb-3">
          <div className="card-header py-2 fw-semibold">Data Augmentation</div>
          <div className="card-body py-2">
            {(td.augmentation || []).map((a, i) => (
              <span key={i} className="badge bg-info text-dark me-1 mb-1">{a}</span>
            ))}
          </div>
        </div>
        <div className="card shadow-sm border-0">
          <div className="card-header py-2 fw-semibold">Evaluation Metrics</div>
          <div className="card-body py-2">
            {(td.evaluation_metrics || []).map((m, i) => (
              <span key={i} className="badge bg-success me-1 mb-1">{m}</span>
            ))}
          </div>
        </div>
      </div>

      {/* Hyperparam grid */}
      <div className="col-12">
        <div className="card shadow-sm border-0">
          <div className="card-header py-2 fw-semibold">Hyperparameter Search Grid (CNN-LSTM)</div>
          <div className="card-body py-2">
            <div className="table-responsive">
              <table className="table table-sm align-middle mb-0">
                <thead className="table-light"><tr><th>Parameter</th><th>Values</th></tr></thead>
                <tbody>
                  {Object.entries(hg).map(([param, vals]) => (
                    <tr key={param}>
                      <td><code className="text-primary">{param}</code></td>
                      <td>{Array.isArray(vals) ? vals.map((v, i) => (
                        <span key={i} className="badge bg-light text-dark border me-1">{v}</span>
                      )) : vals}</td>
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

/* ── Definitions Tab ───────────────────────────────────────────── */
function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted p-3">Loading…</div>;
  const concepts = defs.concepts || [];
  const reg = defs.regulatory_context || [];
  const thresholds = defs.performance_thresholds || [];
  const refs = defs.references || [];

  return (
    <div className="row g-3">
      <div className="col-md-6">
        <div className="card shadow-sm border-0 mb-3">
          <div className="card-header py-2 fw-semibold">Key Concepts ({concepts.length})</div>
          <div className="card-body py-2" style={{ maxHeight: 420, overflowY: 'auto' }}>
            {concepts.map((c, i) => (
              <div key={i} className="mb-2">
                <strong style={{ fontSize: '0.85rem' }}>{c.term}</strong>
                <div className="text-muted" style={{ fontSize: '0.8rem' }}>{c.definition}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="card shadow-sm border-0">
          <div className="card-header py-2 fw-semibold">References</div>
          <div className="card-body py-2">
            <ol className="mb-0 ps-3">
              {refs.map((r, i) => (
                <li key={i} className="text-muted mb-1" style={{ fontSize: '0.78rem' }}>{r}</li>
              ))}
            </ol>
          </div>
        </div>
      </div>

      <div className="col-md-6">
        <div className="card shadow-sm border-0 mb-3">
          <div className="card-header py-2 fw-semibold">Regulatory Context</div>
          <div className="card-body py-2">
            {reg.map((r, i) => (
              <div key={i} className="mb-2">
                <span className="badge bg-dark me-2">{r.framework}</span>
                <span className="text-muted" style={{ fontSize: '0.8rem' }}>{r.relevance}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="card shadow-sm border-0">
          <div className="card-header py-2 fw-semibold">Performance Thresholds</div>
          <div className="card-body py-2">
            <table className="table table-sm mb-0">
              <thead className="table-light"><tr><th>Metric</th><th>Required</th></tr></thead>
              <tbody>
                {thresholds.map((t, i) => (
                  <tr key={i}>
                    <td style={{ fontSize: '0.82rem' }}>{t.metric}</td>
                    <td><span className="badge bg-success">{t.threshold}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Main Page ─────────────────────────────────────────────────── */
export default function HybridCNNPage() {
  const [tab, setTab]     = useState('overview');
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [defs, setDefs]   = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hybrid-cnn/overview`)
      .then(r => r.json()).then(setOv).catch(e => setError(e.message));
    fetch(`${API}/api/hybrid-cnn/breakdown`)
      .then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/hybrid-cnn/definitions`)
      .then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: '1.6rem' }}>🧠</span>
        <div>
          <h4 className="mb-0 fw-bold">Hybrid CNN-LSTM / CNN-Transformer</h4>
          <div className="text-muted" style={{ fontSize: '0.82rem' }}>
            Architecture design · Baseline vs DL comparison · Training pipeline · EEG seizure detection
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger py-2">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'     && <OverviewTab      ov={ov} />}
      {tab === 'architecture' && <ArchitectureTab  bd={bd} />}
      {tab === 'comparison'   && <ComparisonTab    bd={bd} />}
      {tab === 'training'     && <TrainingTab      bd={bd} />}
      {tab === 'definitions'  && <DefinitionsTab   defs={defs} />}
    </div>
  );
}
