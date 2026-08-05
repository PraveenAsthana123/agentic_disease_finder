'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const gateColor = g => g === 'PASS' ? 'success' : g === 'WARN' ? 'warning' : 'danger';
const fmt = v => typeof v === 'number' ? v.toFixed(4) : v ?? '—';

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

export default function FairnessDashboard() {
  const [fl,   setFl]   = useState(null);
  const [ov,   setOv]   = useState(null);
  const [grps, setGrps] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [err,  setErr]  = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/fairness`).then(r => r.json()),
      fetch(`${API}/api/aif360/overview`).then(r => r.json()),
      fetch(`${API}/api/aif360/groups`).then(r => r.json()),
      fetch(`${API}/api/aif360/definitions`).then(r => r.json()),
    ]).then(([f, o, g, d]) => { setFl(f); setOv(o); setGrps(g); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!fl || !ov || !grps) return <div className="text-muted p-4">Loading fairness data…</div>;

  const dm  = ov.dataset_metrics        || {};
  const cm  = ov.classification_metrics || {};
  const mit = ov.mitigation_results     || {};
  const gc  = grps.group_comparison     || {};

  const TABS = [
    { id: 'overview',    label: '📊 Overview' },
    { id: 'groups',      label: '👥 Group Comparison' },
    { id: 'aif360',      label: '🔬 AIF360 Metrics' },
    { id: 'mitigation',  label: '🔧 Mitigation' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  return (
    <div>
      <h3>⚖️ Fairness &amp; Bias Detection Dashboard</h3>
      <p className="text-muted small">
        Dual-library fairness audit — <strong>Fairlearn 0.14</strong> (demographic parity by sex,
        real assessment outcomes, n={fl.n}) + <strong>AIF360 {ov.library_version}</strong>
        (disparate impact, equal opportunity, Reweighing mitigation on EEG classification).
      </p>

      {/* KPI Cards */}
      <div className="row mb-3">
        <KPI label="Fairness Gate (sex)"
             value={fl.fairness_gate}
             color={gateColor(fl.fairness_gate)}
             sub="Fairlearn DPD < 0.2" />
        <KPI label="Dem. Parity Diff (sex)"
             value={fl.demographic_parity_difference?.toFixed(4)}
             color={Math.abs(fl.demographic_parity_difference) < 0.1 ? 'success' : Math.abs(fl.demographic_parity_difference) < 0.2 ? 'warning' : 'danger'}
             sub="|Female − Male| adverse rate" />
        <KPI label="Disparate Impact (AIF360)"
             value={dm.disparate_impact?.toFixed(3)}
             color={dm.disparate_impact >= 0.8 && dm.disparate_impact <= 1.25 ? 'success' : 'warning'}
             sub="≥0.8 = pass (80% rule)" />
        <KPI label="Equal Opportunity Diff"
             value={cm.equal_opportunity_difference?.toFixed(4)}
             color={Math.abs(cm.equal_opportunity_difference ?? 1) < 0.05 ? 'success' : 'warning'}
             sub="TPR(unpriv) − TPR(priv)" />
      </div>
      <div className="row mb-3">
        <KPI label="Samples Analysed"
             value={ov.n_samples}
             color="primary"
             sub={`${ov.n_privileged} priv / ${ov.n_unprivileged} unpriv`} />
        <KPI label="Statistical Parity Diff"
             value={dm.statistical_parity_difference?.toFixed(4)}
             color={Math.abs(dm.statistical_parity_difference ?? 1) < 0.1 ? 'success' : 'warning'}
             sub="ideal = 0" />
        <KPI label="Consistency Score"
             value={dm.consistency?.toFixed(3)}
             color={dm.consistency >= 0.8 ? 'success' : dm.consistency >= 0.6 ? 'warning' : 'danger'}
             sub="individual fairness ≥ 0.8" />
        <KPI label="Model Accuracy"
             value={`${((cm.accuracy ?? 0) * 100).toFixed(1)}%`}
             color="info"
             sub="post-Reweighing" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div>
          <div className="row">
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header fw-bold">📐 Fairlearn — Demographic Parity (by sex)</div>
                <div className="card-body">
                  <p className="small text-muted mb-2">
                    Outcome: <em>{fl.outcome}</em> · n = {fl.n} · Library: {fl.library}
                  </p>
                  <table className="table table-sm table-bordered mb-2">
                    <thead className="table-light">
                      <tr><th>Group</th><th>Count</th><th>Adverse Rate</th></tr>
                    </thead>
                    <tbody>
                      {Object.entries(fl.by_group || {}).map(([grp, info]) => (
                        <tr key={grp}>
                          <td>{grp}</td>
                          <td>{info.count}</td>
                          <td>{(info.selection_rate * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="d-flex justify-content-between align-items-center small">
                    <span>Overall adverse rate: <strong>{(fl.overall_selection_rate * 100).toFixed(1)}%</strong></span>
                    <span className={`badge bg-${gateColor(fl.fairness_gate)}`}>{fl.fairness_gate}</span>
                  </div>
                  <p className="text-muted mt-2 mb-0" style={{ fontSize: '0.8rem' }}>{fl.interpretation}</p>
                </div>
              </div>
            </div>

            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header fw-bold">🔬 AIF360 — Dataset-Level Bias (by age_group)</div>
                <div className="card-body">
                  <p className="small text-muted mb-2">
                    Model: <em>{ov.model_type}</em><br />
                    Attribute source: {ov.protected_attribute_source}
                  </p>
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-light">
                      <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                    </thead>
                    <tbody>
                      {[
                        ['Disparate Impact',      dm.disparate_impact,      v => v >= 0.8 && v <= 1.25],
                        ['Stat. Parity Diff',      dm.statistical_parity_difference, v => Math.abs(v) < 0.1],
                        ['Consistency',            dm.consistency,           v => v >= 0.8],
                        ['Equal Opp. Diff',        cm.equal_opportunity_difference, v => Math.abs(v) < 0.05],
                        ['Avg Odds Diff',           cm.average_odds_difference, v => Math.abs(v) < 0.05],
                      ].map(([label, val, pass]) => (
                        <tr key={label}>
                          <td className="small">{label}</td>
                          <td className="fw-bold">{fmt(val)}</td>
                          <td><span className={`badge bg-${pass(val) ? 'success' : 'warning'}`}>
                            {pass(val) ? 'PASS' : 'REVIEW'}
                          </span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {ov.label_distribution && (
            <div className="card mb-3">
              <div className="card-header fw-bold">🏷️ Label Distribution</div>
              <div className="card-body">
                <div className="row">
                  {Object.entries(ov.label_distribution).map(([label, count]) => (
                    <div key={label} className="col-md-4 mb-2">
                      <div className="d-flex justify-content-between align-items-center">
                        <span className="small">{label}</span>
                        <span className="badge bg-secondary">{count}</span>
                      </div>
                      <div className="progress mt-1" style={{ height: '6px' }}>
                        <div className="progress-bar"
                          style={{ width: `${(count / ov.n_samples) * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* GROUPS TAB */}
      {tab === 'groups' && (
        <div>
          <div className="row mb-3">
            {Object.entries(grps.groups || {}).map(([gname, gdata]) => (
              <div key={gname} className="col-md-6 mb-3">
                <div className={`card border-${gname === 'privileged' ? 'primary' : 'secondary'}`}>
                  <div className={`card-header fw-bold ${gname === 'privileged' ? 'bg-primary' : 'bg-secondary'} text-white`}>
                    {gname === 'privileged' ? '🔵 Privileged' : '🟠 Unprivileged'} — {gdata.label}
                  </div>
                  <div className="card-body">
                    <p className="small text-muted mb-2">{gdata.description}</p>
                    <table className="table table-sm mb-2">
                      <tbody>
                        <tr><td>Samples</td><td className="fw-bold">{gdata.n_samples}</td></tr>
                        <tr><td>Base Rate</td><td className="fw-bold">{(gdata.base_rate * 100).toFixed(1)}%</td></tr>
                        <tr><td>Positive Prediction Rate</td><td className="fw-bold">{(gdata.positive_prediction_rate * 100).toFixed(1)}%</td></tr>
                        <tr><td>True Positive Rate</td><td className="fw-bold">{((gdata.rates?.true_positive_rate ?? 0) * 100).toFixed(1)}%</td></tr>
                        <tr><td>False Positive Rate</td><td className="fw-bold">{((gdata.rates?.false_positive_rate ?? 0) * 100).toFixed(1)}%</td></tr>
                        <tr><td>Accuracy</td><td className="fw-bold">{((gdata.rates?.accuracy ?? 0) * 100).toFixed(1)}%</td></tr>
                      </tbody>
                    </table>
                    <div className="small">
                      <strong>Confusion Matrix:</strong>
                      <span className="text-success ms-2">TP={gdata.confusion_matrix?.true_positives}</span>
                      <span className="text-danger ms-2">FP={gdata.confusion_matrix?.false_positives}</span>
                      <span className="text-danger ms-2">FN={gdata.confusion_matrix?.false_negatives}</span>
                      <span className="text-success ms-2">TN={gdata.confusion_matrix?.true_negatives}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {Object.keys(gc).length > 0 && (
            <div className="card mb-3">
              <div className="card-header fw-bold">📐 Group Comparison (Δ = Unprivileged − Privileged)</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr><th>Metric</th><th>Privileged</th><th>Unprivileged</th><th>Δ</th><th>Status</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(gc).map(([metric, vals]) => {
                      const delta = vals.delta ?? ((vals.unprivileged ?? 0) - (vals.privileged ?? 0));
                      const pass = Math.abs(delta) < 0.05;
                      return (
                        <tr key={metric}>
                          <td className="small">{metric.replace(/_/g, ' ')}</td>
                          <td>{fmt(vals.privileged)}</td>
                          <td>{fmt(vals.unprivileged)}</td>
                          <td className={`fw-bold text-${pass ? 'success' : 'warning'}`}>{fmt(delta)}</td>
                          <td><span className={`badge bg-${pass ? 'success' : 'warning'}`}>{pass ? 'OK' : 'REVIEW'}</span></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {grps.clinical_note && (
            <div className="alert alert-info small">{grps.clinical_note}</div>
          )}
        </div>
      )}

      {/* AIF360 TAB */}
      {tab === 'aif360' && (
        <div>
          <div className="row mb-3">
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header fw-bold">📊 Dataset-Level Metrics</div>
                <div className="card-body p-0">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr><th>Metric</th><th>Value</th></tr>
                    </thead>
                    <tbody>
                      {Object.entries(dm).map(([k, v]) => (
                        <tr key={k}>
                          <td className="small text-muted">{k.replace(/_/g, ' ')}</td>
                          <td className="fw-bold">{typeof v === 'number' ? fmt(v) : JSON.stringify(v)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header fw-bold">🎯 Classification-Level Metrics</div>
                <div className="card-body p-0">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr><th>Metric</th><th>Value</th></tr>
                    </thead>
                    <tbody>
                      {Object.entries(cm).map(([k, v]) => (
                        <tr key={k}>
                          <td className="small text-muted">{k.replace(/_/g, ' ')}</td>
                          <td className="fw-bold">{typeof v === 'number' ? fmt(v) : JSON.stringify(v)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {ov.top_feature_importance && (
            <div className="card mb-3">
              <div className="card-header fw-bold">🔑 Top Feature Importances (RandomForest)</div>
              <div className="card-body">
                <div className="row">
                  {ov.top_feature_importance.slice(0, 10).map(([feat, imp]) => (
                    <div key={feat} className="col-md-6 mb-2">
                      <div className="d-flex justify-content-between align-items-center">
                        <span className="small text-truncate" style={{ maxWidth: '72%' }}>{feat}</span>
                        <span className="small fw-bold">{imp.toFixed(4)}</span>
                      </div>
                      <div className="progress mt-1" style={{ height: '5px' }}>
                        <div className="progress-bar bg-info"
                          style={{ width: `${(imp / ov.top_feature_importance[0][1]) * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          <div className="alert alert-secondary small">
            <strong>Model:</strong> {ov.model_type} &nbsp;·&nbsp;
            <strong>Features:</strong> {ov.n_features} &nbsp;·&nbsp;
            <strong>Protected attr:</strong> {ov.protected_attribute} ({ov.protected_attribute_source})
          </div>
        </div>
      )}

      {/* MITIGATION TAB */}
      {tab === 'mitigation' && (
        <div>
          <div className="card mb-3">
            <div className="card-header fw-bold">🔧 {mit.method}</div>
            <div className="card-body">
              <p className="text-muted small mb-3">{mit.description}</p>
              <div className="row">
                {['before', 'after'].map(phase => {
                  const d = mit[phase];
                  if (!d) return null;
                  const metrics = { ...(d.dataset_metrics || {}), ...(d.classification_metrics || {}) };
                  return (
                    <div key={phase} className="col-md-6 mb-3">
                      <h6 className={`text-${phase === 'before' ? 'danger' : 'success'}`}>
                        {phase === 'before' ? '❌ Before Reweighing' : '✅ After Reweighing'}
                      </h6>
                      <table className="table table-sm table-bordered">
                        <thead className="table-light">
                          <tr><th>Metric</th><th>Value</th></tr>
                        </thead>
                        <tbody>
                          {Object.entries(metrics)
                            .filter(([, v]) => typeof v === 'number')
                            .map(([k, v]) => (
                              <tr key={k}>
                                <td className="small">{k.replace(/_/g, ' ')}</td>
                                <td className="fw-bold">{fmt(v)}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          <div className="alert alert-warning small">
            <strong>Clinical Note:</strong> Reweighing mitigates data-level bias by adjusting sample weights.
            With n={ov.n_samples} (limited sample), metrics reach near-perfect values — real-world deployment
            requires larger, demographically representative EEG datasets. Use these results to validate
            fairness pipeline mechanics, not as clinical performance guarantees.
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <div>
          {defs.library_info && (
            <div className="alert alert-secondary small mb-3">
              <strong>{defs.library_info.name}</strong> — {defs.library_info.citation}
            </div>
          )}
          {['dataset_metrics', 'classification_metrics', 'mitigation_methods'].map(section => {
            const items = defs[section];
            if (!items || !items.length) return null;
            return (
              <div key={section} className="card mb-3">
                <div className="card-header fw-bold text-capitalize">
                  {section.replace(/_/g, ' ')}
                </div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr>
                        <th style={{ width: '18%' }}>Metric</th>
                        <th style={{ width: '26%' }}>Formula</th>
                        <th>Description</th>
                        <th style={{ width: '14%' }}>Ideal</th>
                      </tr>
                    </thead>
                    <tbody>
                      {items.map(m => (
                        <tr key={m.name}>
                          <td className="fw-bold small">{m.name}</td>
                          <td style={{ fontSize: '0.72rem' }}><code>{m.formula}</code></td>
                          <td className="small">{m.description}</td>
                          <td className="small">{m.ideal_value ?? m.acceptable_range ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
