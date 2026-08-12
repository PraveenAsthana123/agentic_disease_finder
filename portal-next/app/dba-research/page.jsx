'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLOR = {
  Approved:        'success',
  Passed:          'success',
  Completed:       'success',
  'Pre-submission':'secondary',
  Planned:         'secondary',
  Submitted:       'primary',
  'Under Review':  'warning',
  'In Progress':   'warning',
  'Failed - Remediation': 'danger',
  'Additional Info Requested': 'info',
};

const CONSENT_LABEL = {
  data_sharing:    'Data Sharing',
  genetic_testing: 'Genetic Testing',
  imaging_sharing: 'Imaging Sharing',
  research:        'Research Participation',
  treatment:       'Treatment',
  video_eeg:       'Video-EEG Recording',
};

function KpiCard({ label, value, sub, color = 'primary', small }) {
  return (
    <div className="col">
      <div className={`card border-${color} h-100`}>
        <div className="card-body text-center py-2">
          <div className={`fs-4 fw-bold text-${color}`}>{value ?? '—'}</div>
          {sub && <div className="small text-muted">{sub}</div>}
          <div className={`small${small ? ' text-muted' : ' fw-semibold'}`}>{label}</div>
        </div>
      </div>
    </div>
  );
}

function ProgressBar({ label, value, max, color = 'primary' }) {
  const pct = max ? Math.min(100, Math.round(value / max * 100)) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{value} / {max} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: '10px' }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function DBAResearchPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [tab, setTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/dba-research/overview`).then(r => r.json()),
      fetch(`${API}/api/dba-research/breakdown`).then(r => r.json()),
      fetch(`${API}/api/dba-research/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, def]) => { setOverview(ov); setBreakdown(bk); setDefinitions(def); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border text-primary" /></div>;
  if (error || !overview?.available) return (
    <div className="container py-4">
      <div className="alert alert-danger">Failed to load DBA Research Dashboard: {error || overview?.error}</div>
    </div>
  );

  const { study, recruitment, ethics_consent, regulatory, validation, ai_performance, data_richness } = overview;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-start justify-content-between mb-3 flex-wrap gap-2">
        <div>
          <h4 className="mb-0 fw-bold">🎓 DBA Research KPI Dashboard</h4>
          <div className="text-muted small mt-1">
            <span className="fw-semibold">{study.researcher}</span> · {study.program} · {study.institution} · {study.location}
          </div>
          <div className="text-muted small">{study.title}</div>
        </div>
        <div className="text-end">
          <span className="badge bg-primary me-1">GGU IRB</span>
          <span className="badge bg-info me-1">IEC India</span>
          <span className="badge bg-secondary">ICMR 2017</span>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {[
          ['overview',    '📊 Overview'],
          ['ethics',      '⚖️ Ethics & Consent'],
          ['regulatory',  '🏛️ Regulatory & Validation'],
          ['definitions', '📖 Study Design & References'],
        ].map(([key, label]) => (
          <li className="nav-item" key={key}>
            <button className={`nav-link${tab === key ? ' active' : ''}`} onClick={() => setTab(key)}>
              {label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview tab ───────────────────────────────────────────── */}
      {tab === 'overview' && (
        <>
          {/* KPI strip */}
          <div className="row row-cols-2 row-cols-md-4 g-2 mb-3">
            <KpiCard label="Patients Enrolled" value={recruitment.enrolled} sub={`Target: ${recruitment.target_total}`} color="primary" />
            <KpiCard label="EEG Analyses" value={recruitment.n_analyses} sub="AI-processed" color="info" />
            <KpiCard label="Research Consent" value={`${ethics_consent.research_consent_pct}%`} sub={`${ethics_consent.research_consent_granted}/${ethics_consent.research_consent_target} patients`} color={ethics_consent.research_consent_pct >= 50 ? 'success' : 'warning'} />
            <KpiCard label="Validation Pass Rate" value={`${validation.pass_rate_pct}%`} sub={`${validation.passed}/${validation.total_studies} studies`} color={validation.pass_rate_pct >= 50 ? 'success' : 'warning'} />
          </div>

          <div className="row row-cols-2 row-cols-md-4 g-2 mb-4">
            <KpiCard label="Best AI Model" value={`${ai_performance.best_model.accuracy}%`} sub={`AUC ${ai_performance.best_model.auc} · ${ai_performance.best_model.type}`} color="success" />
            <KpiCard label="Training Runs" value={ai_performance.total_training_runs} sub="model comparison" color="secondary" />
            <KpiCard label="Overall Consent Rate" value={`${ethics_consent.consent_rate_pct}%`} sub={`${ethics_consent.granted}/${ethics_consent.total_consent_records} records`} color="info" />
            <KpiCard label="Avg Validation AUC" value={validation.avg_auc} sub={`Sens ${validation.avg_sensitivity} · Spec ${validation.avg_specificity}`} color="primary" />
          </div>

          {/* Recruitment progress */}
          <div className="card mb-3">
            <div className="card-header fw-semibold">📋 Patient Recruitment Progress</div>
            <div className="card-body">
              <ProgressBar label="Total (Retro + Prospective)" value={recruitment.enrolled} max={recruitment.target_total} color="primary" />
              <ProgressBar label="Retrospective Target" value={Math.min(recruitment.enrolled, recruitment.target_retrospective)} max={recruitment.target_retrospective} color="info" />
              <ProgressBar label="Prospective Target" value={Math.max(0, recruitment.enrolled - recruitment.target_retrospective)} max={recruitment.target_prospective} color="warning" />
              <div className="text-muted small mt-2">
                Study design: {study.design} · IRB: {study.irb}
              </div>
            </div>
          </div>

          {/* Summary cards row */}
          <div className="row g-2">
            {/* Model best */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">🤖 Best AI Model</div>
                <div className="card-body">
                  <div className="fw-bold fs-5">{ai_performance.best_model.name}</div>
                  <div className="text-muted small">{ai_performance.best_model.type} · {ai_performance.best_model.task}</div>
                  <hr className="my-2" />
                  <div className="row text-center">
                    <div className="col">
                      <div className="fw-bold text-success">{ai_performance.best_model.accuracy}%</div>
                      <div className="small text-muted">Accuracy</div>
                    </div>
                    <div className="col">
                      <div className="fw-bold text-primary">{ai_performance.best_model.auc}</div>
                      <div className="small text-muted">AUC-ROC</div>
                    </div>
                  </div>
                  <div className="text-muted small mt-2">{ai_performance.total_training_runs} total training runs</div>
                </div>
              </div>
            </div>

            {/* Regulatory snapshot */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">🏛️ Regulatory Snapshot</div>
                <div className="card-body">
                  <div className="row text-center">
                    <div className="col"><div className="fw-bold text-success">{regulatory.approved}</div><div className="small">Approved</div></div>
                    <div className="col"><div className="fw-bold text-warning">{regulatory.under_review}</div><div className="small">Under Review</div></div>
                    <div className="col"><div className="fw-bold text-secondary">{regulatory.pre_submission}</div><div className="small">Pre-Submission</div></div>
                  </div>
                  <hr className="my-2" />
                  <div className="small text-muted">
                    {regulatory.total_submissions} total submissions · {regulatory.approval_rate_pct}% approval rate
                  </div>
                  <div className="small text-muted mt-1">
                    IEC/IRB doc target: {regulatory.iec_irb_doc_target} documents
                  </div>
                </div>
              </div>
            </div>

            {/* Validation snapshot */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">✅ Validation Studies</div>
                <div className="card-body">
                  <div className="row text-center">
                    <div className="col"><div className="fw-bold text-success">{validation.passed}</div><div className="small">Passed</div></div>
                    <div className="col"><div className="fw-bold text-primary">{validation.total_studies - validation.passed}</div><div className="small">Other</div></div>
                    <div className="col"><div className="fw-bold text-primary">{validation.total_studies}</div><div className="small">Total</div></div>
                  </div>
                  <hr className="my-2" />
                  <div className="small">Avg AUC: <span className="fw-bold text-success">{validation.avg_auc}</span></div>
                  <div className="small">Sensitivity: <span className="fw-bold">{(validation.avg_sensitivity * 100).toFixed(1)}%</span> · Specificity: <span className="fw-bold">{(validation.avg_specificity * 100).toFixed(1)}%</span></div>
                </div>
              </div>
            </div>
          </div>

          {/* Jurisdictions */}
          <div className="card mt-3">
            <div className="card-header fw-semibold">🌐 Regulatory Jurisdictions</div>
            <div className="card-body d-flex flex-wrap gap-2">
              {study.jurisdictions.map(j => (
                <span key={j} className="badge bg-light text-dark border">{j}</span>
              ))}
            </div>
          </div>
        </>
      )}

      {/* ── Ethics & Consent tab ────────────────────────────────────── */}
      {tab === 'ethics' && breakdown && (
        <>
          <div className="card mb-3">
            <div className="card-header fw-semibold">⚖️ Informed Consent by Type</div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead className="table-light">
                    <tr>
                      <th>Consent Type</th>
                      <th className="text-center">Total</th>
                      <th className="text-center text-success">Granted</th>
                      <th className="text-center text-warning">Pending</th>
                      <th className="text-center text-danger">Declined</th>
                      <th className="text-center text-secondary">Expired</th>
                      <th className="text-center text-info">Withdrawn</th>
                      <th className="text-center">Grant Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.consent_by_type.map(c => (
                      <tr key={c.type}>
                        <td className="fw-semibold">{CONSENT_LABEL[c.type] || c.type}</td>
                        <td className="text-center">{c.total}</td>
                        <td className="text-center text-success fw-bold">{c.granted}</td>
                        <td className="text-center text-warning">{c.pending}</td>
                        <td className="text-center text-danger">{c.declined}</td>
                        <td className="text-center text-secondary">{c.expired}</td>
                        <td className="text-center text-info">{c.withdrawn}</td>
                        <td className="text-center">
                          <span className={`badge bg-${c.granted_pct >= 60 ? 'success' : c.granted_pct >= 40 ? 'warning' : 'danger'}`}>
                            {c.granted_pct}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="text-muted small">
                IRB prerequisite: Research Participation consent must be granted before data use.
                ICMR 2017 §5.3 — written informed consent mandatory for all research participants.
              </div>
            </div>
          </div>

          {/* Monthly analyses trend */}
          {breakdown.monthly_analyses && breakdown.monthly_analyses.length > 0 && (
            <div className="card mb-3">
              <div className="card-header fw-semibold">📈 Monthly Data Collection (EEG Analyses)</div>
              <div className="card-body">
                <div className="d-flex align-items-end gap-1" style={{ height: '80px', overflowX: 'auto' }}>
                  {breakdown.monthly_analyses.map(m => {
                    const maxN = Math.max(...breakdown.monthly_analyses.map(x => x.count), 1);
                    const h = Math.round((m.count / maxN) * 70);
                    return (
                      <div key={m.month} className="text-center" style={{ minWidth: '36px' }}>
                        <div className="bg-primary rounded-top mx-auto" style={{ width: '24px', height: `${h}px` }} title={`${m.month}: ${m.count}`} />
                        <div className="small text-muted" style={{ fontSize: '9px', writingMode: 'vertical-rl', transform: 'rotate(180deg)', height: '24px' }}>{m.month}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── Regulatory & Validation tab ─────────────────────────────── */}
      {tab === 'regulatory' && breakdown && (
        <>
          {/* Regulatory submissions */}
          <div className="card mb-3">
            <div className="card-header fw-semibold d-flex justify-content-between">
              <span>🏛️ Regulatory Submissions ({breakdown.regulatory.list.length})</span>
              <div>
                {Object.entries(breakdown.regulatory.status_distribution).map(([s, n]) => (
                  <span key={s} className={`badge bg-${STATUS_COLOR[s] || 'secondary'} ms-1`}>{s}: {n}</span>
                ))}
              </div>
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Pathway</th>
                      <th>Phase</th>
                      <th>Risk Class</th>
                      <th>Status</th>
                      <th>Validation Score</th>
                      <th>Submitted</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.regulatory.list.map((r, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{r.pathway}</td>
                        <td>{r.phase}</td>
                        <td>{r.risk_class}</td>
                        <td><span className={`badge bg-${STATUS_COLOR[r.status] || 'secondary'}`}>{r.status}</span></td>
                        <td>{r.validation_score != null ? `${r.validation_score}%` : '—'}</td>
                        <td>{r.submitted_date?.slice(0,10) || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Validation studies */}
          <div className="card mb-3">
            <div className="card-header fw-semibold">✅ Validation Studies by Type</div>
            <div className="card-body">
              <div className="table-responsive mb-3">
                <table className="table table-sm">
                  <thead className="table-light">
                    <tr>
                      <th>Study Type</th>
                      <th className="text-center">Count</th>
                      <th className="text-center">Passed</th>
                      <th className="text-center">Avg AUC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.validation.by_type.map(v => (
                      <tr key={v.type}>
                        <td className="fw-semibold">{v.type}</td>
                        <td className="text-center">{v.count}</td>
                        <td className="text-center">
                          <span className={`badge bg-${v.passed === v.count ? 'success' : 'warning'}`}>{v.passed}</span>
                        </td>
                        <td className="text-center">{v.avg_auc ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="fw-semibold small mb-2">All Validation Studies</div>
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead className="table-light">
                    <tr>
                      <th>Type</th>
                      <th>Site</th>
                      <th>PI</th>
                      <th className="text-center">Status</th>
                      <th className="text-center">AUC</th>
                      <th className="text-center">Sens</th>
                      <th className="text-center">Spec</th>
                      <th className="text-center">N</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.validation.studies.map((v, i) => (
                      <tr key={i}>
                        <td className="fw-semibold small">{v.study_type}</td>
                        <td className="small">{v.site || '—'}</td>
                        <td className="small">{v.pi || '—'}</td>
                        <td className="text-center">
                          <span className={`badge bg-${STATUS_COLOR[v.status] || 'secondary'} small`}>{v.status}</span>
                        </td>
                        <td className="text-center">{v.auc ?? '—'}</td>
                        <td className="text-center">{v.sensitivity != null ? `${(v.sensitivity*100).toFixed(1)}%` : '—'}</td>
                        <td className="text-center">{v.specificity != null ? `${(v.specificity*100).toFixed(1)}%` : '—'}</td>
                        <td className="text-center">{v.sample_size ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Model comparison summary */}
          <div className="card mb-3">
            <div className="card-header fw-semibold">🤖 AI Model Performance Comparison</div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead className="table-light">
                    <tr>
                      <th>Model Type</th>
                      <th className="text-center">Runs</th>
                      <th className="text-center">Best Acc</th>
                      <th className="text-center">Best AUC</th>
                      <th className="text-center">Avg Acc</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.model_comparison.map((m, i) => (
                      <tr key={i} className={i === 0 ? 'table-success' : ''}>
                        <td className="fw-semibold">{m.type}{i === 0 ? ' 🏆' : ''}</td>
                        <td className="text-center">{m.runs}</td>
                        <td className="text-center fw-bold">{m.best_acc}%</td>
                        <td className="text-center">{m.best_auc}</td>
                        <td className="text-center text-muted">{m.avg_acc}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── Definitions tab ─────────────────────────────────────────── */}
      {tab === 'definitions' && definitions && (() => {
        const sd = definitions.study_design;
        const irb = definitions.irb_framework;
        return (
          <>
            {/* Study design */}
            <div className="card mb-3">
              <div className="card-header fw-semibold">📋 Study Design</div>
              <div className="card-body">
                <div className="fw-bold mb-1">{sd.title}</div>
                <div className="text-muted small mb-2">{sd.type}</div>
                <div className="row g-3">
                  <div className="col-md-6">
                    <div className="card bg-light">
                      <div className="card-body py-2">
                        <div className="fw-semibold">Retrospective (n={sd.retrospective.target_n})</div>
                        <div className="small text-muted">Source: {sd.retrospective.source}</div>
                        <div className="small">Inclusion: {sd.retrospective.inclusion}</div>
                        <div className="small text-danger">Exclusion: {sd.retrospective.exclusion}</div>
                      </div>
                    </div>
                  </div>
                  <div className="col-md-6">
                    <div className="card bg-light">
                      <div className="card-body py-2">
                        <div className="fw-semibold">Prospective (n={sd.prospective.target_n})</div>
                        <div className="small text-muted">Source: {sd.prospective.source}</div>
                        <div className="small">Duration: {sd.prospective.duration}</div>
                        <div className="small text-primary">{sd.prospective.consent}</div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="mt-3">
                  <div className="fw-semibold small">Primary Outcome:</div>
                  <div className="small">{sd.primary_outcome}</div>
                  <div className="fw-semibold small mt-2">Secondary Outcomes:</div>
                  <ul className="small mb-0">
                    {sd.secondary_outcomes.map((o, i) => <li key={i}>{o}</li>)}
                  </ul>
                </div>
              </div>
            </div>

            {/* IRB framework */}
            <div className="card mb-3">
              <div className="card-header fw-semibold">⚖️ IRB / IEC Framework ({irb.document_master_list}-Document Master List)</div>
              <div className="card-body">
                <div className="row g-2 mb-3">
                  <div className="col-md-6">
                    <div className="badge bg-primary p-2 d-block text-start">
                      Primary: {irb.irb_primary}
                    </div>
                  </div>
                  <div className="col-md-6">
                    <div className="badge bg-info p-2 d-block text-start">
                      Secondary: {irb.irb_secondary}
                    </div>
                  </div>
                </div>
                <div className="row g-2">
                  {Object.entries(irb.jurisdictions).map(([country, regs]) => (
                    <div className="col-md-3" key={country}>
                      <div className="card h-100">
                        <div className="card-header py-1 small fw-semibold">{country}</div>
                        <div className="card-body py-1">
                          {regs.map(r => <div key={r} className="small text-muted">• {r}</div>)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-3">
                  <div className="fw-semibold small mb-1">Document Categories ({irb.document_master_list} total):</div>
                  <div className="d-flex flex-wrap gap-1">
                    {irb.categories.map(c => (
                      <span key={c} className="badge bg-light text-dark border small">{c}</span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Glossary */}
            <div className="card mb-3">
              <div className="card-header fw-semibold">📖 Glossary</div>
              <div className="card-body">
                <div className="table-responsive">
                  <table className="table table-sm">
                    <thead className="table-light">
                      <tr><th>Term</th><th>Definition</th></tr>
                    </thead>
                    <tbody>
                      {definitions.glossary.map(g => (
                        <tr key={g.term}>
                          <td className="fw-bold text-nowrap">{g.term}</td>
                          <td className="small">{g.definition}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* References */}
            <div className="card mb-3">
              <div className="card-header fw-semibold">📚 Key References</div>
              <div className="card-body">
                <ol className="small mb-0">
                  {definitions.references.map((r, i) => (
                    <li key={i} className="mb-1">{r}</li>
                  ))}
                </ol>
              </div>
            </div>
          </>
        );
      })()}
    </div>
  );
}
