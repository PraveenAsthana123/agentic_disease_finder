'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'sites', label: 'Sites' },
  { id: 'privacy', label: 'Privacy & DP' },
  { id: 'convergence', label: 'Convergence' },
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

function Bar({ items, labelKey, valueKey, max, color }) {
  if (!items || !items.length) return null;
  const mx = max || Math.max(...items.map(i => i[valueKey]));
  return (
    <div>
      {items.map((item, idx) => (
        <div key={idx} className="mb-2">
          <div className="d-flex justify-content-between small mb-1">
            <span>{item[labelKey]}</span>
            <span className="fw-bold">{typeof item[valueKey] === 'number' ? item[valueKey].toFixed(3) : item[valueKey]}</span>
          </div>
          <div className="progress" style={{ height: 10 }}>
            <div
              className={`progress-bar bg-${color || 'primary'}`}
              style={{ width: `${(item[valueKey] / mx) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function StatusBadge({ status }) {
  const map = { converged: 'success', converging: 'warning', diverged: 'danger', initializing: 'secondary' };
  return <span className={`badge bg-${map[status] || 'secondary'}`}>{status}</span>;
}

export default function FederatedLearningDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/federated-learning/overview`).then(r => r.json()),
      fetch(`${API}/api/federated-learning/breakdown`).then(r => r.json()),
      fetch(`${API}/api/federated-learning/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov) return <div className="text-center p-5"><div className="spinner-border text-primary" /></div>;

  const convergenceStatus = ov.convergence_status;
  const epsilonPct = ov.privacy_budget_epsilon > 0
    ? Math.round((ov.epsilon_spent / ov.privacy_budget_epsilon) * 100)
    : 0;

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <span style={{ fontSize: 28 }} className="me-2">🔗</span>
        <div>
          <h4 className="mb-0">Federated Learning</h4>
          <div className="text-muted small">Multi-site privacy-preserving EEG model training · Differential Privacy · {ov.total_sites} sites</div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Global Model Accuracy" value={`${(ov.global_model_accuracy * 100).toFixed(1)}%`} color="success" />
        <KPI label="Participating Sites" value={ov.total_sites} color="primary" />
        <KPI label="Communication Rounds" value={ov.communication_rounds} color="info" sub="completed" />
        <KPI label="Convergence" value={<StatusBadge status={convergenceStatus} />} />
      </div>
      <div className="row mb-4">
        <KPI label="ε Spent" value={ov.epsilon_spent?.toFixed(4)} color="warning" sub={`of ${ov.privacy_budget_epsilon} budget`} />
        <KPI label="Budget Used" value={`${epsilonPct}%`} color={epsilonPct > 80 ? 'danger' : 'success'} sub="privacy budget" />
        <KPI label="Noise Multiplier" value={ov.noise_multiplier} color="secondary" sub="σ (DP noise)" />
        <KPI label="Gradient Clip Norm" value={ov.gradient_clipping_norm} color="secondary" sub="max gradient norm" />
      </div>

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

      {/* Overview Tab */}
      {tab === 'overview' && bd && (
        <div>
          <div className="row">
            {/* Round History */}
            <div className="col-md-8 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">📈 Global Accuracy per Round</div>
                <div className="card-body" style={{ overflowX: 'auto' }}>
                  <table className="table table-sm table-hover">
                    <thead><tr><th>Round</th><th>Global Accuracy</th><th>Progress</th></tr></thead>
                    <tbody>
                      {(ov.round_history || []).map(r => (
                        <tr key={r.round}>
                          <td>{r.round}</td>
                          <td><span className="fw-bold">{(r.global_accuracy * 100).toFixed(1)}%</span></td>
                          <td style={{ width: 160 }}>
                            <div className="progress" style={{ height: 8 }}>
                              <div className="progress-bar bg-success" style={{ width: `${r.global_accuracy * 100}%` }} />
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Aggregation Comparison */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">⚖️ Aggregation Strategy Comparison</div>
                <div className="card-body">
                  {(bd.aggregation_comparison || []).map((a, i) => (
                    <div key={i} className="mb-3 p-2 border rounded">
                      <div className="fw-bold">{a.strategy}</div>
                      <div className="small text-muted">Accuracy: <span className="text-success fw-bold">{(a.accuracy * 100).toFixed(1)}%</span></div>
                      <div className="small text-muted">Comm cost: {a.communication_cost}</div>
                      <div className="small text-muted">Rounds to converge: {a.rounds_to_converge || '—'}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Heterogeneity Metrics */}
          {bd.heterogeneity_metrics && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">📊 Data Heterogeneity Metrics</div>
              <div className="card-body">
                <div className="row">
                  {Object.entries(bd.heterogeneity_metrics).map(([k, v]) => (
                    <div key={k} className="col-md-4 mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize">{k.replace(/_/g, ' ')}</span>
                        <span className="fw-bold">{typeof v === 'number' ? v.toFixed(3) : v}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${v > 0.3 ? 'warning' : 'success'}`}
                          style={{ width: `${Math.min(v * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="small text-muted mt-2">
                  Non-IID score &gt; 0.3 → recommend FedProx or Scaffold over FedAvg
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Sites Tab */}
      {tab === 'sites' && bd && (
        <div>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">🏥 Site-Level Performance</div>
            <div className="card-body" style={{ overflowX: 'auto' }}>
              <table className="table table-sm table-hover">
                <thead>
                  <tr>
                    <th>Site</th><th>Patients</th><th>Accuracy</th><th>Sensitivity</th>
                    <th>Specificity</th><th>F1</th><th>Weight</th><th>Divergence</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.site_details || []).sort((a, b) => b.accuracy - a.accuracy).map((s, i) => (
                    <tr key={i}>
                      <td>{s.name}</td>
                      <td>{s.n_patients}</td>
                      <td><span className="fw-bold text-success">{(s.accuracy * 100).toFixed(1)}%</span></td>
                      <td>{(s.sensitivity * 100).toFixed(1)}%</td>
                      <td>{(s.specificity * 100).toFixed(1)}%</td>
                      <td>{(s.f1 * 100).toFixed(1)}%</td>
                      <td>{(s.contribution_weight * 100).toFixed(1)}%</td>
                      <td>
                        <span className={`badge bg-${s.weight_divergence_from_global > 0.1 ? 'warning' : 'success'}`}>
                          {s.weight_divergence_from_global?.toFixed(3)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="row">
            {/* Bandwidth */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">📡 Bandwidth Usage (MB)</div>
                <div className="card-body">
                  <Bar
                    items={bd.bandwidth_usage || []}
                    labelKey="site"
                    valueKey="bandwidth_mb"
                    color="info"
                  />
                </div>
              </div>
            </div>

            {/* Gradient norms */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">📐 Gradient Norms & Clipping</div>
                <div className="card-body">
                  <table className="table table-sm">
                    <thead><tr><th>Site</th><th>Gradient Norm</th><th>Clip Rate</th></tr></thead>
                    <tbody>
                      {(bd.gradient_norms || []).map((g, i) => (
                        <tr key={i}>
                          <td>{g.site}</td>
                          <td>{g.gradient_norm?.toFixed(3)}</td>
                          <td>
                            <span className={`badge bg-${g.clipping_rate > 0.1 ? 'warning' : 'secondary'}`}>
                              {(g.clipping_rate * 100).toFixed(1)}%
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Seizure type distribution */}
          {bd.seizure_type_distribution && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">⚡ Seizure Type Distribution by Site (Non-IID indicator)</div>
              <div className="card-body" style={{ overflowX: 'auto' }}>
                <table className="table table-sm">
                  <thead><tr><th>Site</th><th>Focal</th><th>Generalized</th><th>Unknown</th></tr></thead>
                  <tbody>
                    {bd.seizure_type_distribution.map((s, i) => (
                      <tr key={i}>
                        <td>{s.site}</td>
                        <td><span className="badge bg-primary">{s.focal}</span></td>
                        <td><span className="badge bg-success">{s.generalized}</span></td>
                        <td><span className="badge bg-secondary">{s.unknown}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Privacy Tab */}
      {tab === 'privacy' && bd && (
        <div>
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">🔒 DP Parameters</div>
                <div className="card-body">
                  <table className="table table-sm">
                    <tbody>
                      <tr><td>Privacy Budget (ε total)</td><td className="fw-bold">{ov.privacy_budget_epsilon}</td></tr>
                      <tr><td>ε Spent</td><td className="fw-bold text-warning">{ov.epsilon_spent?.toFixed(4)}</td></tr>
                      <tr><td>Delta (δ)</td><td className="fw-bold">{ov.delta?.toExponential(0)}</td></tr>
                      <tr><td>Noise Multiplier (σ)</td><td className="fw-bold">{ov.noise_multiplier}</td></tr>
                      <tr><td>Gradient Clipping Norm</td><td className="fw-bold">{ov.gradient_clipping_norm}</td></tr>
                    </tbody>
                  </table>
                  <div className="mt-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>Privacy Budget Used</span>
                      <span className="fw-bold">{epsilonPct}%</span>
                    </div>
                    <div className="progress" style={{ height: 14 }}>
                      <div
                        className={`progress-bar bg-${epsilonPct > 80 ? 'danger' : epsilonPct > 60 ? 'warning' : 'success'}`}
                        style={{ width: `${epsilonPct}%` }}
                      />
                    </div>
                    <div className="text-muted small mt-1">Clinical target: ε &lt; 10.0 over full training</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">📉 ε Budget History per Round</div>
                <div className="card-body" style={{ maxHeight: 280, overflowY: 'auto' }}>
                  <table className="table table-sm">
                    <thead><tr><th>Round</th><th>ε Spent</th><th>Cumulative ε</th><th>Budget</th></tr></thead>
                    <tbody>
                      {(bd.epsilon_budget_history || []).map((r, i) => (
                        <tr key={i}>
                          <td>{r.round}</td>
                          <td>{r.epsilon_spent?.toFixed(4)}</td>
                          <td className="fw-bold">{r.cumulative_epsilon?.toFixed(4)}</td>
                          <td>{r.budget_limit}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Privacy audit */}
          {bd.privacy_audit && (
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">🛡️ Privacy Audit Trail</div>
              <div className="card-body" style={{ overflowX: 'auto' }}>
                <table className="table table-sm">
                  <thead><tr><th>Round</th><th>ε Spent</th><th>Cumulative ε</th><th>Guarantee Status</th></tr></thead>
                  <tbody>
                    {bd.privacy_audit.map((r, i) => (
                      <tr key={i}>
                        <td>{r.round}</td>
                        <td>{r.epsilon_spent?.toFixed(4)}</td>
                        <td>{r.cumulative?.toFixed(4)}</td>
                        <td>
                          <span className={`badge bg-${r.guarantee_status === 'converged' ? 'success' : 'warning'}`}>
                            {r.guarantee_status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Convergence Tab */}
      {tab === 'convergence' && bd && (
        <div>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">📈 Convergence Curve — Loss & Accuracy</div>
            <div className="card-body" style={{ overflowX: 'auto' }}>
              <table className="table table-sm table-hover">
                <thead>
                  <tr><th>Round</th><th>Global Loss</th><th>Global Accuracy</th><th>Loss Bar</th><th>Accuracy Bar</th></tr>
                </thead>
                <tbody>
                  {(bd.convergence_curve || []).map((r, i) => (
                    <tr key={i}>
                      <td>{r.round}</td>
                      <td className="text-danger fw-bold">{r.global_loss?.toFixed(4)}</td>
                      <td className="text-success fw-bold">{(r.global_accuracy * 100).toFixed(1)}%</td>
                      <td style={{ width: 120 }}>
                        <div className="progress" style={{ height: 8 }}>
                          <div className="progress-bar bg-danger" style={{ width: `${Math.min(r.global_loss * 300, 100)}%` }} />
                        </div>
                      </td>
                      <td style={{ width: 120 }}>
                        <div className="progress" style={{ height: 8 }}>
                          <div className="progress-bar bg-success" style={{ width: `${r.global_accuracy * 100}%` }} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-semibold">⚖️ Strategy Comparison Summary</div>
            <div className="card-body">
              <div className="row">
                {(bd.aggregation_comparison || []).map((a, i) => (
                  <div key={i} className="col-md-3 mb-3">
                    <div className="card border-primary h-100">
                      <div className="card-body text-center">
                        <div className="h5 fw-bold text-primary">{a.strategy}</div>
                        <div className="h4 text-success fw-bold">{(a.accuracy * 100).toFixed(1)}%</div>
                        <div className="text-muted small">accuracy</div>
                        <hr className="my-2" />
                        <div className="small">Rounds: <strong>{a.rounds_to_converge || 'N/A'}</strong></div>
                        <div className="small">Cost: <strong>{a.communication_cost}</strong></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div className="row">
          {(defs.definitions || []).map((d, i) => (
            <div key={i} className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">{d.term}</div>
                <div className="card-body">
                  <p className="small mb-2">{d.definition}</p>
                  {d.unit && d.unit !== 'N/A' && (
                    <div className="small text-muted mb-1"><strong>Unit:</strong> {d.unit}</div>
                  )}
                  {d.interpretation && (
                    <div className="small text-info"><strong>Interpretation:</strong> {d.interpretation}</div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
