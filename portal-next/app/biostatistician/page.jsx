'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-2">
      <div className={`card border-${color || 'primary'} text-center h-100`}>
        <div className="card-body py-2 px-1">
          <div className={`h4 fw-bold mb-0 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.68rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ items, colorFn }) {
  const mx = Math.max(...(items || []).map(i => i.count || 0), 1);
  return (
    <div>
      {(items || []).map((it, i) => {
        const val = it.count ?? 0;
        const label = it.label || '?';
        const pct = Math.round((val / mx) * 100);
        const color = colorFn ? colorFn(it) : 'primary';
        return (
          <div key={i} className="d-flex align-items-center mb-1 gap-2">
            <div className="text-end small text-muted" style={{ width: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '0.75rem' }}>
              {label}
            </div>
            <div className="flex-grow-1">
              <div className="progress" style={{ height: 16 }}>
                <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }}>
                  <span className="small px-1">{val}</span>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function StatusBadge({ status }) {
  const colorMap = {
    'Completed': 'success',
    'In Progress': 'primary',
    'Failed - Remediation': 'danger',
    'Planned': 'secondary',
  };
  const color = colorMap[status] || 'secondary';
  return <span className={`badge bg-${color}`}>{status || '—'}</span>;
}

function MagnitudeBadge({ mag }) {
  if (!mag) return <span className="badge bg-secondary">N/A</span>;
  if (mag.includes('Large')) return <span className="badge bg-danger">{mag}</span>;
  if (mag.includes('Medium')) return <span className="badge bg-warning text-dark">{mag}</span>;
  return <span className="badge bg-info text-dark">{mag}</span>;
}

export default function BiostatisticianPage() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/biostatistician/overview`).then(r => r.json()),
      fetch(`${API}/api/biostatistician/breakdown`).then(r => r.json()),
      fetch(`${API}/api/biostatistician/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;
  if (!ov) return <div className="p-4 text-muted">Loading Biostatistician dashboard…</div>;

  const kpis = ov.kpis || {};
  const power = ov.power_analysis || {};
  const dur = ov.disease_duration_stats || {};

  return (
    <div className="container-fluid py-3">
      <h2 className="mb-1">📊 Biostatistician Dashboard</h2>
      <p className="text-muted mb-3" style={{ fontSize: '0.85rem' }}>
        Statistical validity &amp; rigour — sample size, power analysis, class balance,
        model evaluation metrics. Tier-1 mandatory consultant (ICH-GCP E9 / ICMR).
      </p>

      {/* KPI row */}
      <div className="row g-2 mb-3">
        <KPI label="Total Patients" value={kpis.total_patients} color="primary" />
        <KPI label="Seizure Records" value={kpis.seizure_metadata_records} color="info" />
        <KPI label="Assessment Records" value={kpis.assessment_records} color="secondary" />
        <KPI label="Validation Studies" value={kpis.validation_studies} color="dark" sub={`${kpis.completed_studies} completed`} />
        <KPI label="DRE Patients" value={kpis.dre_patients} color="danger" sub={`${kpis.dre_prevalence_pct}% prevalence`} />
        <KPI label="Class Gini" value={kpis.class_balance_gini} color="warning" sub="0=pure imbalance" />
        <KPI label="Mean Sensitivity" value={kpis.mean_sensitivity} color="success" sub="completed studies" />
        <KPI label="Mean Specificity" value={kpis.mean_specificity} color="success" sub="completed studies" />
        <KPI label="Mean AUC-ROC" value={kpis.mean_auc_roc} color="primary" sub="completed studies" />
        <KPI label="Mean Sample n" value={kpis.mean_sample_size} color="secondary" sub="val. studies" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {['overview', 'breakdown', 'definitions'].map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === t ? ' active' : ''}`}
              onClick={() => setTab(t)}
            >
              {t === 'overview' ? '📋 Overview' : t === 'breakdown' ? '🔬 Breakdown' : '📚 Definitions'}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Power Analysis */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">⚡ Power Analysis</div>
              <div className="card-body small">
                <table className="table table-sm table-bordered mb-2">
                  <tbody>
                    <tr><td>Effect size assumed</td><td>{power.effect_size_label}</td></tr>
                    <tr><td>α (significance level)</td><td>{power.alpha}</td></tr>
                    <tr><td>Target power (1−β)</td><td>{power.target_power}</td></tr>
                    <tr><td>n required per group</td><td className="fw-bold text-danger">{power.n_required_for_80pct_power}</td></tr>
                    <tr>
                      <td>Prospective (n={power.prospective_n})</td>
                      <td>
                        <div className="progress" style={{ height: 18 }}>
                          <div className="progress-bar bg-warning" style={{ width: `${Math.round((power.power_prospective || 0) * 100)}%` }}>
                            {Math.round((power.power_prospective || 0) * 100)}%
                          </div>
                        </div>
                      </td>
                    </tr>
                    <tr>
                      <td>Retrospective (n={power.retrospective_n})</td>
                      <td>
                        <div className="progress" style={{ height: 18 }}>
                          <div className="progress-bar bg-primary" style={{ width: `${Math.round((power.power_retrospective || 0) * 100)}%` }}>
                            {Math.round((power.power_retrospective || 0) * 100)}%
                          </div>
                        </div>
                      </td>
                    </tr>
                  </tbody>
                </table>
                <div className="alert alert-warning py-1 px-2 mb-0" style={{ fontSize: '0.78rem' }}>
                  {power.note}
                </div>
              </div>
            </div>
          </div>

          {/* Disease Duration Stats */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">📏 Disease Duration (years)</div>
              <div className="card-body small">
                <table className="table table-sm table-bordered mb-3">
                  <tbody>
                    <tr><td>Mean</td><td className="fw-bold">{dur.mean_years} yr</td></tr>
                    <tr><td>SD</td><td>{dur.std_years} yr</td></tr>
                    <tr><td>Range</td><td>{dur.min_years} – {dur.max_years} yr</td></tr>
                    <tr><td>Records with data</td><td>{dur.n_with_data}</td></tr>
                  </tbody>
                </table>
                <div className="fw-bold mb-1">Gender Distribution</div>
                <Bar items={ov.gender_chart || []} colorFn={it => it.label === 'Female' ? 'danger' : it.label === 'Male' ? 'primary' : 'secondary'} />
              </div>
            </div>
          </div>

          {/* AED Trial Distribution */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">💊 AED Trial Distribution</div>
              <div className="card-body small">
                <Bar items={ov.aed_trial_chart || []} colorFn={() => 'primary'} />
              </div>
            </div>
          </div>

          {/* Onset Zone Distribution */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">🧠 Onset Zone Distribution</div>
              <div className="card-body small">
                <Bar items={ov.onset_zone_chart || []} colorFn={() => 'info'} />
              </div>
            </div>
          </div>

          {/* Study Type & Status */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">🔬 Validation Study Types</div>
              <div className="card-body small">
                <Bar items={ov.study_type_chart || []} colorFn={() => 'dark'} />
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">📊 Study Status Distribution</div>
              <div className="card-body small">
                <Bar items={ov.study_status_chart || []}
                  colorFn={it => it.label === 'Completed' ? 'success' : it.label?.includes('Failed') ? 'danger' : 'warning'} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Breakdown tab */}
      {tab === 'breakdown' && bd && (
        <div className="row g-3">
          {/* Metric Summary */}
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-bold">📈 Model Metric Summary (Completed Studies)</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark">
                      <tr>
                        <th>Metric</th><th>N</th><th>Mean</th><th>SD</th><th>Min</th><th>Max</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd.metric_summary || []).map((m, i) => (
                        <tr key={i}>
                          <td className="fw-bold">{m.metric}</td>
                          <td>{m.n}</td>
                          <td className="fw-bold text-success">{m.mean}</td>
                          <td>{m.std}</td>
                          <td>{m.min}</td>
                          <td>{m.max}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Effect Sizes */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">📐 Effect Sizes (Cohen's d)</div>
              <div className="card-body small">
                {(bd.effect_sizes || []).map((ef, i) => (
                  <div key={i} className="border rounded p-2 mb-2">
                    <div className="fw-bold mb-1">{ef.comparison}</div>
                    <div className="d-flex flex-wrap gap-2">
                      <span>Cohen's d: <strong>{ef.cohens_d ?? '—'}</strong></span>
                      <MagnitudeBadge mag={ef.magnitude} />
                    </div>
                    <div className="text-muted mt-1">
                      DRE: n={ef.dre_n}, mean={ef.dre_mean_dur ?? ef.dre_mean_aed} &nbsp;|&nbsp;
                      Responsive: n={ef.resp_n}, mean={ef.resp_mean_dur ?? ef.resp_mean_aed}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Multiple Comparisons */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">🔢 Multiple Comparison Correction</div>
              <div className="card-body small">
                <div className="mb-2">
                  <strong>Method:</strong> {bd.multiple_comparisons?.method} &nbsp;|&nbsp;
                  <strong>k:</strong> {bd.multiple_comparisons?.k_hypotheses} hypotheses &nbsp;|&nbsp;
                  <strong>Bonferroni α:</strong> {bd.multiple_comparisons?.bonferroni_alpha}
                </div>
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-light">
                      <tr><th>#</th><th>Hypothesis</th><th>Bonferroni</th><th>FDR</th></tr>
                    </thead>
                    <tbody>
                      {(bd.multiple_comparisons?.table || []).map((row, i) => (
                        <tr key={i}>
                          <td>{row.rank}</td>
                          <td style={{ fontSize: '0.72rem' }}>{row.hypothesis}</td>
                          <td>{row.bonferroni_adjusted}</td>
                          <td>{row.fdr_adjusted}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Validation Studies Table */}
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-bold">
                🧪 Validation Studies ({bd.total_studies} total · {bd.completed_studies} completed)
              </div>
              <div className="card-body p-0">
                <div className="table-responsive" style={{ maxHeight: 400 }}>
                  <table className="table table-sm table-striped table-hover mb-0">
                    <thead className="table-dark sticky-top">
                      <tr>
                        <th>Study ID</th><th>Type</th><th>Status</th><th>n</th>
                        <th>Sensitivity</th><th>Specificity</th><th>AUC-ROC</th>
                        <th>Site</th><th>Findings</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd.studies || []).map((s, i) => (
                        <tr key={i}>
                          <td className="small fw-bold">{s.study_id || '—'}</td>
                          <td className="small">{s.study_type}</td>
                          <td><StatusBadge status={s.status} /></td>
                          <td>{s.sample_size ?? '—'}</td>
                          <td>{s.sensitivity ?? '—'}</td>
                          <td>{s.specificity ?? '—'}</td>
                          <td>{s.auc_roc ? <strong className="text-success">{s.auc_roc}</strong> : '—'}</td>
                          <td className="small">{s.site}</td>
                          <td className="small text-muted" style={{ fontSize: '0.72rem' }}>{s.findings_short}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          {/* Role summary */}
          <div className="col-12">
            <div className="alert alert-info mb-0">
              <strong>Role:</strong> {defs.role_summary}
            </div>
          </div>

          {/* Core metrics */}
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-bold">📏 Core Statistical Metrics</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark">
                      <tr><th>Metric</th><th>Formula</th><th>Interpretation</th></tr>
                    </thead>
                    <tbody>
                      {(defs.core_metrics || []).map((m, i) => (
                        <tr key={i}>
                          <td className="fw-bold">{m.metric}</td>
                          <td><code>{m.formula}</code></td>
                          <td className="small text-muted">{m.interpretation}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Power framework + class imbalance */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">⚡ Power Analysis Framework</div>
              <div className="card-body small">
                <table className="table table-sm table-bordered mb-2">
                  <tbody>
                    {Object.entries(defs.power_framework || {}).filter(([k]) => k !== 'note' && k !== 'prospective_plan').map(([k, v]) => (
                      <tr key={k}><td className="fw-bold">{k.replace(/_/g, ' ')}</td><td>{String(v)}</td></tr>
                    ))}
                  </tbody>
                </table>
                <div className="alert alert-warning py-1 px-2" style={{ fontSize: '0.78rem' }}>
                  {defs.power_framework?.interpretation}
                </div>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">⚖️ Class Imbalance Strategies</div>
              <div className="card-body small">
                <ul className="list-unstyled mb-0">
                  {(defs.class_imbalance_strategies || []).map((s, i) => (
                    <li key={i} className="mb-2">
                      <span className="badge bg-primary me-1">{s.strategy}</span>
                      <span className="text-muted">{s.when}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* Multiple comparison + split validation */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">🔢 Multiple Comparison Control</div>
              <div className="card-body small">
                <div className="mb-2">
                  {(defs.multiple_comparison_control?.methods || []).map((m, i) => (
                    <span key={i} className="badge bg-secondary me-1">{m}</span>
                  ))}
                </div>
                <p>{defs.multiple_comparison_control?.recommendation}</p>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">🔄 Split Validation (LOSO)</div>
              <div className="card-body small">
                <p><strong>Method:</strong> {defs.split_validation?.method}</p>
                <div className="alert alert-info py-1 px-2" style={{ fontSize: '0.78rem' }}>
                  {defs.split_validation?.rationale}
                </div>
                {defs.split_validation?.implemented && (
                  <span className="badge bg-success">Implemented</span>
                )}
              </div>
            </div>
          </div>

          {/* Compliance mapping */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">✅ Compliance Mapping</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead className="table-light">
                    <tr><th>Standard</th><th>Requirement</th></tr>
                  </thead>
                  <tbody>
                    {(defs.compliance_mapping || []).map((c, i) => (
                      <tr key={i}>
                        <td className="fw-bold small">{c.standard}</td>
                        <td className="small text-muted">{c.requirement}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Glossary */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold">📖 Glossary</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead className="table-light">
                    <tr><th>Term</th><th>Definition</th></tr>
                  </thead>
                  <tbody>
                    {(defs.glossary || []).map((g, i) => (
                      <tr key={i}>
                        <td className="fw-bold small">{g.term}</td>
                        <td className="small text-muted">{g.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* References */}
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-bold">📚 References</div>
              <div className="card-body small">
                <ol className="mb-0">
                  {(defs.references || []).map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ol>
                <div className="text-muted mt-2" style={{ fontSize: '0.78rem' }}>
                  Data source: {defs.data_source}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="text-muted mt-3" style={{ fontSize: '0.75rem' }}>
        Source: {ov.source} · Updated: {ov.updated_at} · For research &amp; prototype demonstration only — not for clinical use.
      </div>
    </div>
  );
}
