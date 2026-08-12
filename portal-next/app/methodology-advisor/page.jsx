'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusColor = s =>
  s === 'complete'     ? 'success' :
  s === 'in_progress'  ? 'warning' :
  s === 'planned'      ? 'secondary' : 'info';

const studyStatusColor = s =>
  s === 'Passed'               ? 'success' :
  s === 'Completed'            ? 'primary' :
  s === 'In Progress'          ? 'warning' :
  s === 'Failed - Remediation' ? 'danger'  :
  s === 'Planned'              ? 'secondary' : 'info';

const thresholdColor = s =>
  s === 'pass'   ? 'success' :
  s === 'review' ? 'warning' : 'danger';

const evidenceBadge = lvl =>
  lvl.startsWith('1') ? 'success' :
  lvl.startsWith('2') ? 'primary' :
  lvl === '3'         ? 'info'    :
  lvl === '4'         ? 'warning' : 'secondary';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'studies',     label: 'Validation Studies' },
  { id: 'models',      label: 'Model Experiments' },
  { id: 'dissertation',label: 'Dissertation Map' },
  { id: 'definitions', label: 'Definitions' },
];

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

/* ── Overview Tab ─────────────────────────────────────────────────── */
function OverviewTab({ ov, bd }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  const s   = ov.summary || {};
  const st  = ov.study_status || {};
  const pk  = ov.performance_kpis || {};
  const me  = ov.model_experiments || {};
  const pd  = ov.patient_data || {};
  const ci  = ov.consent_irb || {};
  const ms  = ov.multi_site || {};

  const pct = v => v != null ? `${v}%` : '—';

  return (
    <>
      <div className="row g-2 mb-3">
        <KPI label="Validation Studies"    value={s.total_validation_studies}          color="primary" />
        <KPI label="Avg AUC-ROC (studies)" value={pk.avg_auc_roc}                      color="success" />
        <KPI label="Best Model AUC"        value={me.best_auc}                         color="success" />
        <KPI label="Avg Sensitivity"       value={pk.avg_sensitivity}                  color="info" />
        <KPI label="Model Experiments"     value={me.total}                             color="primary" />
        <KPI label="IRB Consent Records"   value={ci.total_consent_records}            color="secondary" sub={`${ci.granted} granted`} />
      </div>

      <div className="row g-3 mb-3">
        {/* Study Status */}
        <div className="col-md-4">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-header py-2 fw-semibold">Study Status Breakdown</div>
            <div className="card-body py-2">
              {[
                ['Passed',               st.passed,             'success'],
                ['Completed',            st.completed,          'primary'],
                ['In Progress',          st.in_progress,        'warning'],
                ['Failed – Remediation', st.failed_remediation, 'danger'],
                ['Planned',              st.planned,            'secondary'],
              ].map(([label, n, color]) => (
                <div key={label} className="d-flex justify-content-between align-items-center mb-1">
                  <span className="text-muted" style={{ fontSize: '0.82rem' }}>{label}</span>
                  <span className={`badge bg-${color}`}>{n ?? 0}</span>
                </div>
              ))}
              <hr className="my-1" />
              <div className="d-flex justify-content-between">
                <small className="text-muted">Pass+Complete rate</small>
                <strong className="text-success">{pct(st.pass_complete_rate_pct)}</strong>
              </div>
            </div>
          </div>
        </div>

        {/* Study Type Distribution */}
        <div className="col-md-4">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-header py-2 fw-semibold">Study Design Distribution</div>
            <div className="card-body py-2">
              {Object.entries(s.study_type_breakdown || {}).map(([type, n]) => (
                <div key={type} className="d-flex justify-content-between align-items-center mb-1">
                  <span className="text-muted" style={{ fontSize: '0.82rem' }}>{type}</span>
                  <span className="badge bg-info text-dark">{n}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Performance KPIs */}
        <div className="col-md-4">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-header py-2 fw-semibold">Performance vs Thresholds</div>
            <div className="card-body py-2">
              {[
                ['Avg Sensitivity',   pk.avg_sensitivity,  `≥ ${pk.sens_threshold}`, pk.avg_sensitivity >= pk.sens_threshold ? 'pass' : 'fail'],
                ['Avg Specificity',   pk.avg_specificity,  `≥ ${pk.spec_threshold}`, pk.avg_specificity >= pk.spec_threshold ? 'pass' : 'fail'],
                ['Avg AUC-ROC',       pk.avg_auc_roc,      `≥ ${pk.auc_threshold}`, pk.avg_auc_roc >= pk.auc_threshold ? 'pass' : 'fail'],
                ['Best AUC-ROC',      pk.best_auc_roc,     '≥ 0.90',                pk.best_auc_roc >= 0.90 ? 'pass' : 'fail'],
              ].map(([label, val, threshold, status]) => (
                <div key={label} className="d-flex justify-content-between align-items-center mb-1">
                  <span className="text-muted" style={{ fontSize: '0.82rem' }}>{label}</span>
                  <span>
                    <span className="me-1">{val ?? '—'}</span>
                    <span className={`badge bg-${thresholdColor(status)}`}>{threshold}</span>
                  </span>
                </div>
              ))}
              <hr className="my-1" />
              <small className="text-muted">
                Multi-site: {ms.sites} sites · {ms.principal_investigators} PIs
              </small>
            </div>
          </div>
        </div>
      </div>

      {/* Data context */}
      <div className="row g-3">
        <div className="col-md-6">
          <div className="card shadow-sm border-0">
            <div className="card-header py-2 fw-semibold">Dataset Overview</div>
            <div className="card-body py-2">
              <table className="table table-sm table-borderless mb-0">
                <tbody>
                  <tr><td className="text-muted">Patients enrolled</td><td className="fw-semibold">{pd.total_patients}</td></tr>
                  <tr><td className="text-muted">Total AI analyses</td><td className="fw-semibold">{pd.total_analyses}</td></tr>
                  <tr><td className="text-muted">Epilepsy analyses</td><td className="fw-semibold">{pd.epilepsy_analyses} ({pd.epilepsy_pct}%)</td></tr>
                  <tr><td className="text-muted">Avg sample size (studies)</td><td className="fw-semibold">{pk.avg_sample_size}</td></tr>
                  <tr><td className="text-muted">Consent records</td><td className="fw-semibold">{ci.total_consent_records} ({ci.consent_rate_pct}% granted)</td></tr>
                  <tr><td className="text-muted">Regulatory submissions</td><td className="fw-semibold">{ov.regulatory?.total_submissions}</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm border-0">
            <div className="card-header py-2 fw-semibold">Study Type Performance</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th style={{ fontSize: '0.78rem' }}>Study Type</th>
                    <th style={{ fontSize: '0.78rem' }}>n</th>
                    <th style={{ fontSize: '0.78rem' }}>Avg AUC</th>
                    <th style={{ fontSize: '0.78rem' }}>Avg Sens</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.study_type_performance || []).map(r => (
                    <tr key={r.study_type}>
                      <td style={{ fontSize: '0.78rem' }}>{r.study_type}</td>
                      <td style={{ fontSize: '0.78rem' }}>{r.count}</td>
                      <td style={{ fontSize: '0.78rem' }}>{r.avg_auc_roc ?? '—'}</td>
                      <td style={{ fontSize: '0.78rem' }}>{r.avg_sensitivity ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

/* ── Validation Studies Tab ─────────────────────────────────────── */
function StudiesTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const studies = bd.all_studies || [];
  return (
    <>
      <div className="mb-2">
        <small className="text-muted">{studies.length} validation studies — sorted by AUC-ROC descending</small>
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-light">
            <tr>
              <th style={{ fontSize: '0.78rem' }}>Study ID</th>
              <th style={{ fontSize: '0.78rem' }}>Type</th>
              <th style={{ fontSize: '0.78rem' }}>Status</th>
              <th style={{ fontSize: '0.78rem' }}>N</th>
              <th style={{ fontSize: '0.78rem' }}>Sensitivity</th>
              <th style={{ fontSize: '0.78rem' }}>Specificity</th>
              <th style={{ fontSize: '0.78rem' }}>AUC-ROC</th>
              <th style={{ fontSize: '0.78rem' }}>Site</th>
              <th style={{ fontSize: '0.78rem' }}>PI</th>
            </tr>
          </thead>
          <tbody>
            {studies.map((r, i) => (
              <tr key={i}>
                <td style={{ fontSize: '0.78rem' }}><code>{r.study_id}</code></td>
                <td style={{ fontSize: '0.78rem' }}>{r.study_type}</td>
                <td style={{ fontSize: '0.78rem' }}>
                  <span className={`badge bg-${studyStatusColor(r.status)}`}>{r.status}</span>
                </td>
                <td style={{ fontSize: '0.78rem' }}>{r.sample_size ?? '—'}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.sensitivity ?? '—'}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.specificity ?? '—'}</td>
                <td style={{ fontSize: '0.78rem' }}>
                  {r.auc_roc != null
                    ? <span className={`fw-semibold text-${r.auc_roc >= 0.95 ? 'success' : r.auc_roc >= 0.85 ? 'primary' : 'warning'}`}>{r.auc_roc}</span>
                    : '—'}
                </td>
                <td style={{ fontSize: '0.78rem' }}>{r.site ?? '—'}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.principal_investigator ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Site breakdown */}
      <h6 className="mt-3 mb-2">Multi-Site Summary</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered mb-0">
          <thead className="table-light">
            <tr>
              <th style={{ fontSize: '0.78rem' }}>Site</th>
              <th style={{ fontSize: '0.78rem' }}>Studies</th>
              <th style={{ fontSize: '0.78rem' }}>Avg AUC</th>
              <th style={{ fontSize: '0.78rem' }}>Avg Sens</th>
            </tr>
          </thead>
          <tbody>
            {(bd.site_breakdown || []).map((r, i) => (
              <tr key={i}>
                <td style={{ fontSize: '0.78rem' }}>{r.site}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.studies}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.avg_auc ?? '—'}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.avg_sensitivity ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

/* ── Model Experiments Tab ──────────────────────────────────────── */
function ModelsTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  return (
    <>
      {/* Model type summary */}
      <h6 className="mb-2">Model Type Summary</h6>
      <div className="table-responsive mb-3">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-light">
            <tr>
              <th style={{ fontSize: '0.78rem' }}>Model Type</th>
              <th style={{ fontSize: '0.78rem' }}>Experiments</th>
              <th style={{ fontSize: '0.78rem' }}>Best AUC</th>
              <th style={{ fontSize: '0.78rem' }}>Avg AUC</th>
              <th style={{ fontSize: '0.78rem' }}>Avg F1</th>
            </tr>
          </thead>
          <tbody>
            {(bd.model_type_summary || []).map((r, i) => (
              <tr key={i}>
                <td style={{ fontSize: '0.78rem' }}><strong>{r.model_type}</strong></td>
                <td style={{ fontSize: '0.78rem' }}>{r.count}</td>
                <td style={{ fontSize: '0.78rem' }}>
                  <span className={`fw-semibold text-${r.best_auc >= 0.98 ? 'success' : 'primary'}`}>{r.best_auc}</span>
                </td>
                <td style={{ fontSize: '0.78rem' }}>{r.avg_auc}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.avg_f1}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Disease breakdown */}
      <h6 className="mb-2">Analysis by Disease</h6>
      <div className="row g-2 mb-3">
        {(bd.disease_analysis || []).map(r => (
          <div key={r.disease} className="col-6 col-md-2">
            <div className="card text-center border-0 shadow-sm h-100">
              <div className="card-body py-2 px-1">
                <div className="h5 mb-0 text-primary">{r.analyses}</div>
                <div className="text-muted" style={{ fontSize: '0.7rem' }}>{r.disease}</div>
                <div className="text-muted" style={{ fontSize: '0.65rem' }}>conf {r.avg_confidence ?? '—'}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Leaderboard */}
      <h6 className="mb-2">Top 25 Model Leaderboard</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-light">
            <tr>
              <th style={{ fontSize: '0.78rem' }}>#</th>
              <th style={{ fontSize: '0.78rem' }}>Model</th>
              <th style={{ fontSize: '0.78rem' }}>Type</th>
              <th style={{ fontSize: '0.78rem' }}>Task</th>
              <th style={{ fontSize: '0.78rem' }}>AUC</th>
              <th style={{ fontSize: '0.78rem' }}>F1</th>
              <th style={{ fontSize: '0.78rem' }}>Precision</th>
              <th style={{ fontSize: '0.78rem' }}>Recall</th>
              <th style={{ fontSize: '0.78rem' }}>N</th>
              <th style={{ fontSize: '0.78rem' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {(bd.model_leaderboard || []).map((r, i) => (
              <tr key={i}>
                <td style={{ fontSize: '0.78rem' }}>{i + 1}</td>
                <td style={{ fontSize: '0.78rem' }}><code>{r.model_name}</code></td>
                <td style={{ fontSize: '0.78rem' }}>{r.model_type}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.task}</td>
                <td style={{ fontSize: '0.78rem' }}>
                  <span className={`fw-semibold text-${r.auc_roc >= 0.98 ? 'success' : 'primary'}`}>{r.auc_roc}</span>
                </td>
                <td style={{ fontSize: '0.78rem' }}>{r.f1_score}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.precision}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.recall}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.n_samples?.toLocaleString()}</td>
                <td style={{ fontSize: '0.78rem' }}>
                  <span className={`badge bg-${r.status === 'completed' ? 'success' : 'secondary'}`}>{r.status}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

/* ── Dissertation Map Tab ────────────────────────────────────────── */
function DissertationTab({ bd, df }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const chapters = bd.dissertation_chapter_map || [];
  const ev = df?.evidence_hierarchy || [];
  const refs = df?.references || [];
  const fw = df?.methodology_frameworks || [];
  return (
    <>
      <h6 className="mb-2">Dissertation Chapter Readiness — DBA (Golden Gate University)</h6>
      <div className="row g-2 mb-4">
        {chapters.map((ch, i) => (
          <div key={i} className="col-12 col-md-6">
            <div className={`card border-${statusColor(ch.status)} shadow-sm h-100`}>
              <div className={`card-header py-1 bg-${statusColor(ch.status)} bg-opacity-10 fw-semibold`} style={{ fontSize: '0.82rem' }}>
                <span className={`badge bg-${statusColor(ch.status)} me-2`}>{ch.status.replace('_', ' ')}</span>
                {ch.chapter}
              </div>
              <div className="card-body py-2">
                <div className="mb-1"><small className="text-muted fw-semibold">Elements: </small>
                  <small>{ch.elements.join(' · ')}</small>
                </div>
                <div><small className="text-muted fw-semibold">Data: </small>
                  <small>{ch.data_sources.join(' · ')}</small>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Evidence hierarchy */}
      <h6 className="mb-2">Evidence Hierarchy</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-bordered mb-0">
          <thead className="table-light">
            <tr>
              <th style={{ fontSize: '0.78rem' }}>Level</th>
              <th style={{ fontSize: '0.78rem' }}>Evidence Type</th>
              <th style={{ fontSize: '0.78rem' }}>Studies (n)</th>
            </tr>
          </thead>
          <tbody>
            {ev.map((r, i) => (
              <tr key={i}>
                <td style={{ fontSize: '0.78rem' }}>
                  <span className={`badge bg-${evidenceBadge(r.level)}`}>{r.level}</span>
                </td>
                <td style={{ fontSize: '0.78rem' }}>{r.type}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.studies}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Methodology frameworks */}
      <h6 className="mb-2">Reporting & Methodology Frameworks</h6>
      <div className="table-responsive mb-3">
        <table className="table table-sm table-hover mb-0">
          <thead className="table-light">
            <tr>
              <th style={{ fontSize: '0.78rem' }}>Framework</th>
              <th style={{ fontSize: '0.78rem' }}>Scope</th>
              <th style={{ fontSize: '0.78rem' }}>Application</th>
            </tr>
          </thead>
          <tbody>
            {fw.map((r, i) => (
              <tr key={i}>
                <td style={{ fontSize: '0.78rem' }}><strong>{r.framework}</strong></td>
                <td style={{ fontSize: '0.78rem' }}>{r.scope}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.application}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* References */}
      <h6 className="mb-2">Key References</h6>
      <ol className="list-group list-group-numbered list-group-flush mb-0">
        {refs.map((r, i) => (
          <li key={i} className="list-group-item px-0 py-1" style={{ fontSize: '0.78rem' }}>{r}</li>
        ))}
      </ol>
    </>
  );
}

/* ── Definitions Tab ─────────────────────────────────────────────── */
function DefinitionsTab({ df }) {
  if (!df) return <div className="text-muted">Loading…</div>;
  const concepts   = df.concepts || [];
  const thresholds = df.performance_thresholds || [];
  return (
    <>
      {/* Thresholds */}
      <h6 className="mb-2">Performance Thresholds</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-bordered">
          <thead className="table-light">
            <tr>
              <th style={{ fontSize: '0.78rem' }}>Metric</th>
              <th style={{ fontSize: '0.78rem' }}>Threshold</th>
              <th style={{ fontSize: '0.78rem' }}>Current</th>
              <th style={{ fontSize: '0.78rem' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {thresholds.map((r, i) => (
              <tr key={i}>
                <td style={{ fontSize: '0.78rem' }}>{r.metric}</td>
                <td style={{ fontSize: '0.78rem' }}>{r.threshold}</td>
                <td style={{ fontSize: '0.78rem' }}><strong>{r.current}</strong></td>
                <td style={{ fontSize: '0.78rem' }}>
                  <span className={`badge bg-${thresholdColor(r.status)}`}>{r.status}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Concepts */}
      <h6 className="mb-2">Methodology Concepts</h6>
      <div className="row g-2">
        {concepts.map((c, i) => (
          <div key={i} className="col-12 col-md-6">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-body py-2">
                <h6 className="card-title mb-1" style={{ fontSize: '0.85rem' }}>{c.term}</h6>
                <p className="card-text text-muted mb-1" style={{ fontSize: '0.78rem' }}>{c.definition}</p>
                <small className="text-info"><em>{c.context}</em></small>
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  );
}

/* ── Main Page ────────────────────────────────────────────────────── */
export default function MethodologyAdvisorPage() {
  const [tab, setTab] = useState('overview');
  const [ov,  setOv]  = useState(null);
  const [bd,  setBd]  = useState(null);
  const [df,  setDf]  = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/methodology-advisor/overview`)
      .then(r => r.json()).then(setOv).catch(() => setErr('Backend error'));
    fetch(`${API}/api/methodology-advisor/breakdown`)
      .then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/methodology-advisor/definitions`)
      .then(r => r.json()).then(setDf).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: '1.5rem' }}>🎓</span>
        <div>
          <h4 className="mb-0">Research Methodology / Dissertation Advisor</h4>
          <small className="text-muted">Doctoral-level scientific rigor · DBA — Golden Gate University · EEG AI Epilepsy Platform</small>
        </div>
      </div>

      {err && <div className="alert alert-danger">{err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewTab    ov={ov} bd={bd} />}
      {tab === 'studies'     && <StudiesTab     bd={bd} />}
      {tab === 'models'      && <ModelsTab      bd={bd} />}
      {tab === 'dissertation' && <DissertationTab bd={bd} df={df} />}
      {tab === 'definitions' && <DefinitionsTab df={df} />}
    </div>
  );
}
