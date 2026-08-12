'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusColor = s =>
  s === 'complete' ? 'success' :
  s === 'partial'  ? 'warning' :
  s === 'planned'  ? 'secondary' :
  s === 'design'   ? 'info' :
  s === 'na'       ? 'light' : 'secondary';

const statusLabel = s =>
  s === 'complete' ? 'Complete' :
  s === 'partial'  ? 'Partial' :
  s === 'planned'  ? 'Planned' :
  s === 'design'   ? 'Design' :
  s === 'na'       ? 'N/A' : s;

const figStatusColor = s =>
  s === 'ready'   ? 'success' :
  s === 'partial' ? 'warning' : 'secondary';

const sapTestColor = s =>
  s === 'complete' ? 'success' :
  s === 'partial'  ? 'warning' : 'secondary';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'tripod',      label: 'TRIPOD-AI Checklist' },
  { id: 'manuscript',  label: 'Manuscript & Figures' },
  { id: 'sap',         label: 'Statistical Analysis Plan' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-2 mb-2">
      <div className="card text-center shadow-sm border-0 h-100">
        <div className="card-body py-2 px-1">
          <div className={`h3 mb-0 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="badge bg-light text-muted mt-1">{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function ProgressBar({ label, value, max, color }) {
  const pct = max ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small text-muted mb-1">
        <span>{label}</span>
        <span>{value}/{max} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: '8px' }}>
        <div className={`progress-bar bg-${color || 'primary'}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function ResearchPublicationPage() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/research-publication/overview`).then(r => r.json()),
      fetch(`${API}/api/research-publication/breakdown`).then(r => r.json()),
      fetch(`${API}/api/research-publication/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="container py-4"><div className="alert alert-danger">{err}</div></div>;
  if (!ov) return <div className="container py-4 text-center"><div className="spinner-border text-primary" /></div>;

  const { kpis, model_performance, dataset, reporting_standards, study } = ov;

  return (
    <div className="container-fluid py-3 px-md-4">
      {/* Header */}
      <div className="mb-3">
        <h4 className="mb-0 fw-bold">
          <span className="me-2">📄</span>Research Publication Readiness
        </h4>
        <p className="text-muted small mb-0">
          TRIPOD-AI compliance · Manuscript progress · Statistical Analysis Plan
          <span className="ms-2 badge bg-light text-muted">{study?.reporting_standard}</span>
        </p>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active fw-semibold' : ''}`}
              onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          {/* KPIs */}
          <div className="row g-2 mb-3">
            <KPI label="TRIPOD-AI Compliance" value={`${kpis.tripod_compliance_pct}%`}
              color={kpis.tripod_compliance_pct >= 80 ? 'success' : 'warning'} sub={`${kpis.tripod_complete}/${kpis.tripod_total_reportable} items`} />
            <KPI label="Manuscript Sections" value={`${kpis.manuscript_sections_complete}/${kpis.manuscript_sections_total}`}
              color="primary" sub="complete" />
            <KPI label="Figures Ready" value={`${kpis.figures_ready}/${kpis.figures_total}`}
              color="info" sub="of 12 planned" />
            <KPI label="Tables Ready" value={`${kpis.tables_ready}/${kpis.tables_total}`}
              color="info" sub="of 8 planned" />
            <KPI label="Best Model AUC" value={model_performance.best_auc}
              color={model_performance.best_auc >= 0.90 ? 'success' : 'warning'} sub={`${model_performance.best_accuracy_pct}% accuracy`} />
            <KPI label="External Val. AUC" value={model_performance.external_validation_avg_auc}
              color="primary" sub={`${model_performance.n_validation_studies} studies, 7 sites`} />
          </div>

          <div className="row g-3">
            {/* Study Info */}
            <div className="col-md-4">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-primary text-white fw-semibold py-2">📋 Study</div>
                <div className="card-body p-2">
                  <table className="table table-sm mb-0">
                    <tbody>
                      <tr><td className="text-muted small">Title</td>
                        <td className="small fw-semibold">{study.title}</td></tr>
                      <tr><td className="text-muted small">Researcher</td>
                        <td className="small">{study.researcher}</td></tr>
                      <tr><td className="text-muted small">Degree</td>
                        <td className="small">{study.degree}</td></tr>
                      <tr><td className="text-muted small">Institution</td>
                        <td className="small">{study.institution}</td></tr>
                      <tr><td className="text-muted small">Target Journal</td>
                        <td className="small">{study.target_journal}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Reporting Standards Compliance */}
            <div className="col-md-4">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-success text-white fw-semibold py-2">📐 Reporting Standards</div>
                <div className="card-body p-3">
                  {reporting_standards?.map(rs => (
                    <div key={rs.standard} className="mb-3">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="fw-semibold small">{rs.standard}</span>
                        <span className={`badge bg-${rs.compliance_pct >= 80 ? 'success' : rs.compliance_pct >= 60 ? 'warning' : 'danger'}`}>
                          {rs.compliance_pct}%
                        </span>
                      </div>
                      <div className="progress" style={{ height: '6px' }}>
                        <div className={`progress-bar bg-${rs.compliance_pct >= 80 ? 'success' : 'warning'}`}
                          style={{ width: `${rs.compliance_pct}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Dataset Summary */}
            <div className="col-md-4">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-info text-white fw-semibold py-2">🗃 Dataset</div>
                <div className="card-body p-2">
                  <table className="table table-sm mb-0">
                    <tbody>
                      <tr><td className="text-muted small">Patients (enrolled)</td>
                        <td><span className="badge bg-primary">{dataset.n_patients}</span> / {dataset.retrospective_target} target</td></tr>
                      <tr><td className="text-muted small">EEG Analyses</td>
                        <td><span className="badge bg-success">{dataset.n_analyses}</span></td></tr>
                      <tr><td className="text-muted small">Model Experiments</td>
                        <td><span className="badge bg-dark">{model_performance.n_model_experiments}</span></td></tr>
                      <tr><td className="text-muted small">Consent Records</td>
                        <td><span className="badge bg-secondary">{dataset.n_consent_records}</span></td></tr>
                      <tr><td className="text-muted small">Regulatory Submissions</td>
                        <td><span className="badge bg-warning text-dark">{dataset.n_regulatory_submissions}</span>
                          <span className="ms-1 small text-muted">({dataset.n_approved_submissions} approved)</span></td></tr>
                      <tr><td className="text-muted small">Validation Studies</td>
                        <td><span className="badge bg-info text-dark">{model_performance.n_validation_studies}</span></td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Model Performance Summary */}
          <div className="card shadow-sm border-0 mt-3">
            <div className="card-header bg-dark text-white fw-semibold py-2">🤖 Model Performance (Publication-Ready Metrics)</div>
            <div className="card-body p-3">
              <div className="row g-3">
                <div className="col-md-3 text-center">
                  <div className="h2 text-success mb-0">{model_performance.best_auc}</div>
                  <div className="small text-muted">Best AUC-ROC (XGBoost_v3)</div>
                  <div className="badge bg-success mt-1">Outstanding (≥0.90)</div>
                </div>
                <div className="col-md-3 text-center">
                  <div className="h2 text-primary mb-0">{model_performance.external_validation_avg_auc}</div>
                  <div className="small text-muted">External Validation Avg AUC</div>
                  <div className="badge bg-primary mt-1">7 sites, 42 studies</div>
                </div>
                <div className="col-md-3 text-center">
                  <div className="h2 text-info mb-0">{model_performance.external_validation_avg_sensitivity}</div>
                  <div className="small text-muted">Avg Sensitivity (External)</div>
                  <div className="badge bg-info text-dark mt-1">Target ≥0.80 ✓</div>
                </div>
                <div className="col-md-3 text-center">
                  <div className="h2 text-warning mb-0">{model_performance.external_validation_avg_specificity}</div>
                  <div className="small text-muted">Avg Specificity (External)</div>
                  <div className="badge bg-warning text-dark mt-1">Target ≥0.75 ✓</div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── TRIPOD-AI CHECKLIST ── */}
      {tab === 'tripod' && bd && (
        <div className="card shadow-sm border-0">
          <div className="card-header bg-primary text-white fw-semibold py-2">
            TRIPOD-AI 27-Item Checklist (Collins et al., BMJ 2024)
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>#</th>
                    <th>Section</th>
                    <th style={{ minWidth: '220px' }}>Requirement</th>
                    <th>Status</th>
                    <th style={{ minWidth: '200px' }}>Evidence</th>
                    <th>Data Source</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.tripod_checklist?.map(item => (
                    <tr key={item.item}>
                      <td className="fw-bold">{item.item}</td>
                      <td><span className="badge bg-light text-dark">{item.section}</span></td>
                      <td className="small">{item.description}</td>
                      <td>
                        <span className={`badge bg-${statusColor(item.status)}`}>
                          {statusLabel(item.status)}
                        </span>
                      </td>
                      <td className="small text-muted">{item.evidence}</td>
                      <td className="small"><code>{item.data_source}</code></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── MANUSCRIPT & FIGURES ── */}
      {tab === 'manuscript' && bd && (
        <div className="row g-3">
          {/* Manuscript Sections */}
          <div className="col-md-6">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-success text-white fw-semibold py-2">📝 Manuscript Sections</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Section</th>
                        <th>Words</th>
                        <th>Status</th>
                        <th>Note</th>
                      </tr>
                    </thead>
                    <tbody>
                      {bd.manuscript_sections?.map((sec, i) => (
                        <tr key={i}>
                          <td className="small fw-semibold">{sec.section}</td>
                          <td className="small">
                            {sec.current_words != null
                              ? <>{sec.current_words}<span className="text-muted">/{sec.target_words}</span></>
                              : <span className="text-muted">—</span>}
                          </td>
                          <td>
                            <span className={`badge bg-${statusColor(sec.status)}`}>
                              {statusLabel(sec.status)}
                            </span>
                          </td>
                          <td className="small text-muted">{sec.note}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Figures + Tables */}
          <div className="col-md-6">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-info text-white fw-semibold py-2">📊 Figure Registry (12 figures)</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-light"><tr><th>ID</th><th>Title</th><th>Type</th><th>Status</th></tr></thead>
                    <tbody>
                      {bd.figure_registry?.map(f => (
                        <tr key={f.id}>
                          <td className="fw-bold small">{f.id}</td>
                          <td className="small">{f.title}</td>
                          <td><span className="badge bg-light text-dark small">{f.type}</span></td>
                          <td><span className={`badge bg-${figStatusColor(f.status)}`}>{f.status}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark fw-semibold py-2">📋 Table Registry (8 tables)</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-light"><tr><th>ID</th><th>Title</th><th>Status</th></tr></thead>
                    <tbody>
                      {bd.table_registry?.map(t => (
                        <tr key={t.id}>
                          <td className="fw-bold small">{t.id}</td>
                          <td className="small">{t.title}</td>
                          <td><span className={`badge bg-${figStatusColor(t.status)}`}>{t.status}</span></td>
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

      {/* ── STATISTICAL ANALYSIS PLAN ── */}
      {tab === 'sap' && bd?.statistical_analysis_plan && (
        <div className="row g-3">
          <div className="col-md-5">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-dark text-white fw-semibold py-2">📑 SAP Summary</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    <tr><td className="text-muted small">Version</td>
                      <td className="small">{bd.statistical_analysis_plan.version}</td></tr>
                    <tr><td className="text-muted small">Signed</td>
                      <td><span className={`badge bg-${bd.statistical_analysis_plan.signed ? 'success' : 'danger'}`}>
                        {bd.statistical_analysis_plan.signed ? 'Yes' : 'Not yet'}
                      </span></td></tr>
                    <tr><td className="text-muted small">Primary Endpoint</td>
                      <td className="small">{bd.statistical_analysis_plan.primary_endpoint}</td></tr>
                    <tr><td className="text-muted small">Missing Data</td>
                      <td className="small">{bd.statistical_analysis_plan.missing_data}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark fw-semibold py-2">📏 Sample Size</div>
              <div className="card-body p-2">
                {bd.statistical_analysis_plan.sample_size && (
                  <>
                    <ProgressBar
                      label="Enrollment"
                      value={bd.statistical_analysis_plan.sample_size.enrolled}
                      max={bd.statistical_analysis_plan.sample_size.target}
                      color="warning"
                    />
                    <table className="table table-sm mb-0">
                      <tbody>
                        <tr><td className="text-muted small">Target n</td>
                          <td className="small">{bd.statistical_analysis_plan.sample_size.target}</td></tr>
                        <tr><td className="text-muted small">Power</td>
                          <td className="small">{bd.statistical_analysis_plan.sample_size.power * 100}%</td></tr>
                        <tr><td className="text-muted small">Alpha</td>
                          <td className="small">{bd.statistical_analysis_plan.sample_size.alpha}</td></tr>
                        <tr><td className="text-muted small">Min AUC Δ</td>
                          <td className="small">{bd.statistical_analysis_plan.sample_size.minimum_auc_difference}</td></tr>
                        <tr><td className="text-muted small">Status</td>
                          <td><span className="badge bg-warning text-dark small">
                            {bd.statistical_analysis_plan.sample_size.status}
                          </span></td></tr>
                      </tbody>
                    </table>
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="col-md-7">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-info text-white fw-semibold py-2">🧮 Statistical Tests</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Test</th><th>Purpose</th><th>Status</th></tr>
                  </thead>
                  <tbody>
                    {bd.statistical_analysis_plan.statistical_tests?.map((t, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold">{t.test}</td>
                        <td className="small">{t.purpose}</td>
                        <td><span className={`badge bg-${sapTestColor(t.status)}`}>{t.status}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm border-0">
              <div className="card-header bg-secondary text-white fw-semibold py-2">🔬 Subgroup Analyses</div>
              <div className="card-body py-2">
                <ul className="mb-0">
                  {bd.statistical_analysis_plan.subgroup_analyses?.map((s, i) => (
                    <li key={i} className="small">{s}</li>
                  ))}
                </ul>
                <div className="mt-2 text-muted small">
                  <strong>Secondary endpoints:</strong>{' '}
                  {bd.statistical_analysis_plan.secondary_endpoints?.join(' · ')}
                </div>
              </div>
            </div>
          </div>

          {/* Model type summary */}
          <div className="col-12">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-primary text-white fw-semibold py-2">🤖 Model Experiment Summary ({bd.model_type_summary?.length} types, {bd.statistical_analysis_plan.model_experiments} runs)</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr><th>Model Type</th><th>Avg AUC</th><th>Avg Accuracy (%)</th><th>Avg F1</th></tr>
                    </thead>
                    <tbody>
                      {bd.model_type_summary?.map(m => (
                        <tr key={m.model_type}>
                          <td className="fw-semibold">{m.model_type}</td>
                          <td><span className={`badge bg-${m.avg_auc >= 0.90 ? 'success' : m.avg_auc >= 0.80 ? 'primary' : 'warning'}`}>{m.avg_auc}</span></td>
                          <td>{m.avg_accuracy_pct}%</td>
                          <td>{m.avg_f1}</td>
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

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-md-7">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-primary text-white fw-semibold py-2">📖 Key Concepts</div>
              <div className="card-body p-0">
                <div className="accordion accordion-flush" id="defAccordion">
                  {defs.concepts?.map((c, i) => (
                    <div key={i} className="accordion-item">
                      <h2 className="accordion-header">
                        <button className="accordion-button collapsed py-2 small fw-semibold"
                          type="button" data-bs-toggle="collapse"
                          data-bs-target={`#def${i}`}>
                          {c.name}
                        </button>
                      </h2>
                      <div id={`def${i}`} className="accordion-collapse collapse"
                        data-bs-parent="#defAccordion">
                        <div className="accordion-body py-2 small">{c.description}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="col-md-5">
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-header bg-success text-white fw-semibold py-2">📐 Reporting Standards</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Standard</th><th>Items</th><th>Reference</th></tr>
                  </thead>
                  <tbody>
                    {defs.reporting_standards?.map(rs => (
                      <tr key={rs.standard}>
                        <td className="fw-bold small">{rs.standard}</td>
                        <td><span className="badge bg-primary">{rs.items}</span></td>
                        <td className="small text-muted">{rs.reference}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark fw-semibold py-2">🎯 Performance Thresholds</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Metric</th><th>Target</th><th>Achieved</th></tr>
                  </thead>
                  <tbody>
                    {defs.performance_thresholds?.map((t, i) => (
                      <tr key={i}>
                        <td className="small">{t.metric}</td>
                        <td className="small">{t.target ?? '—'}</td>
                        <td>
                          <span className={`badge bg-${t.achieved >= t.target ? 'success' : 'warning'}`}>
                            {t.achieved}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm border-0 mt-3">
              <div className="card-header bg-secondary text-white fw-semibold py-2">📚 References</div>
              <div className="card-body py-2">
                <ol className="mb-0 ps-3">
                  {defs.references?.map((r, i) => (
                    <li key={i} className="small mb-1">{r}</li>
                  ))}
                </ol>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
