'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusColor = s => {
  const v = (s || '').toLowerCase();
  if (v === 'passed') return 'success';
  if (v === 'completed') return 'primary';
  if (v.includes('failed') || v.includes('remediation')) return 'danger';
  if (v === 'in progress') return 'warning';
  if (v === 'planned') return 'secondary';
  return 'secondary';
};

const fmtPct = v => v != null ? `${(+v).toFixed(1)}%` : '—';
const fmtNum = v => v != null ? (+v).toFixed(3) : '—';
const fmtInt = v => v != null ? Math.round(+v).toLocaleString() : '—';

export default function ValidationStudiesDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('auc_roc');
  const [sortDir, setSortDir] = useState('desc');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/validation-studies/overview`).then(r => r.json()),
      fetch(`${API}/api/validation-studies/breakdown`).then(r => r.json()),
      fetch(`${API}/api/validation-studies/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading Validation Studies data…</div>;

  const kpis = ov.kpis || {};
  const typeDist = ov.study_type_distribution || [];
  const statusDist = ov.status_distribution || [];
  const siteDist = ov.site_distribution || [];
  const allStudies = bd?.all_studies || [];
  const topPerforming = bd?.top_performing || [];
  const piWorkload = bd?.pi_workload || [];
  const perSubmission = bd?.per_submission || [];
  const failedStudies = bd?.failed_studies || [];
  const inProgressStudies = bd?.in_progress_studies || [];

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'studies', label: '📋 All Studies' },
    { id: 'submissions', label: '📁 Per Submission' },
    { id: 'pi', label: '👤 PI Workload' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  // Filter + sort for all studies tab
  const filteredStudies = allStudies
    .filter(s => !search || (
      (s.study_id || '').toLowerCase().includes(search.toLowerCase()) ||
      (s.study_type || '').toLowerCase().includes(search.toLowerCase()) ||
      (s.status || '').toLowerCase().includes(search.toLowerCase()) ||
      (s.site || '').toLowerCase().includes(search.toLowerCase()) ||
      (s.principal_investigator || '').toLowerCase().includes(search.toLowerCase())
    ))
    .sort((a, b) => {
      const dir = sortDir === 'asc' ? 1 : -1;
      const av = a[sortBy], bv = b[sortBy];
      if (av == null && bv == null) return 0;
      if (av == null) return dir;
      if (bv == null) return -dir;
      if (typeof av === 'number') return dir * (av - bv);
      return dir * String(av).localeCompare(String(bv));
    });

  const toggleSort = col => {
    if (sortBy === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortBy(col); setSortDir('desc'); }
  };
  const sortArrow = col => sortBy === col ? (sortDir === 'asc' ? ' ↑' : ' ↓') : '';

  return (
    <div className="p-3">
      <h3>🔬 Validation Studies Dashboard</h3>
      <p className="text-muted">
        {kpis.total_studies} studies · {kpis.total_submissions} regulatory submissions · {kpis.total_sites} sites ·
        {' '}{kpis.total_pis} PIs · {fmtPct(kpis.pass_rate_pct)} pass rate ·
        avg AUC {fmtNum(kpis.avg_auc_roc)} · avg sensitivity {fmtNum(kpis.avg_sensitivity)}
      </p>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          {/* KPI cards */}
          <div className="row g-3 mb-4">
            {[
              { label: 'Total Studies', val: kpis.total_studies, color: 'primary' },
              { label: 'Submissions', val: kpis.total_submissions, color: 'info' },
              { label: 'Pass Rate', val: fmtPct(kpis.pass_rate_pct), color: 'success' },
              { label: 'Avg AUC', val: fmtNum(kpis.avg_auc_roc), color: 'warning' },
              { label: 'Avg Sensitivity', val: fmtNum(kpis.avg_sensitivity), color: 'secondary' },
              { label: 'Avg Specificity', val: fmtNum(kpis.avg_specificity), color: 'secondary' },
              { label: 'Avg Sample Size', val: fmtInt(kpis.avg_sample_size), color: 'dark' },
              { label: 'Sites', val: kpis.total_sites, color: 'info' },
            ].map(k => (
              <div className="col-6 col-md-3" key={k.label}>
                <div className={`card border-${k.color} shadow-sm h-100`}>
                  <div className="card-body py-2 px-3">
                    <div className="small text-muted">{k.label}</div>
                    <div className={`fw-bold fs-5 text-${k.color}`}>{k.val}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Status distribution */}
          <div className="row g-3 mb-4">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Status Distribution</div>
                <div className="card-body">
                  {statusDist.map(s => (
                    <div key={s.status} className="mb-2">
                      <div className="d-flex justify-content-between mb-1">
                        <span><span className={`badge bg-${statusColor(s.status)} me-2`}>{s.status}</span></span>
                        <span className="small text-muted">{s.count} ({fmtPct(s.pct)})</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className={`progress-bar bg-${statusColor(s.status)}`} style={{ width: `${s.pct}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Study type distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Study Type Distribution</div>
                <div className="card-body">
                  {typeDist.map((t, i) => (
                    <div key={t.type} className="mb-2">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="small">{t.type}</span>
                        <span className="small text-muted">{t.count} ({fmtPct(t.pct)})</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-info" style={{ width: `${t.pct}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Site distribution */}
          <div className="row g-3 mb-4">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Site Distribution</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Site</th><th className="text-end">Studies</th></tr></thead>
                    <tbody>
                      {siteDist.map(s => (
                        <tr key={s.site}>
                          <td>{s.site}</td>
                          <td className="text-end">
                            <span className="badge bg-primary">{s.count}</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Top performing */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Top Performing Studies (by AUC)</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead>
                      <tr>
                        <th>Study ID</th><th>Type</th><th>AUC</th><th>Sens</th><th>Spec</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topPerforming.map(s => (
                        <tr key={s.study_id}>
                          <td><code className="small">{s.study_id}</code></td>
                          <td className="small">{s.study_type}</td>
                          <td><span className="badge bg-success">{fmtNum(s.auc_roc)}</span></td>
                          <td className="small">{fmtNum(s.sensitivity)}</td>
                          <td className="small">{fmtNum(s.specificity)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Alerts: failed + in-progress */}
          {failedStudies.length > 0 && (
            <div className="alert alert-danger py-2 mb-3">
              <strong>⚠ {failedStudies.length} studies in Failed/Remediation</strong>
              <div className="mt-1 d-flex flex-wrap gap-2">
                {failedStudies.map(s => (
                  <span key={s.study_id} className="badge bg-danger text-white">{s.study_id}</span>
                ))}
              </div>
            </div>
          )}
          {inProgressStudies.length > 0 && (
            <div className="alert alert-warning py-2 mb-3">
              <strong>🔄 {inProgressStudies.length} studies In Progress</strong>
              <div className="mt-1 d-flex flex-wrap gap-2">
                {inProgressStudies.map(s => (
                  <span key={s.study_id} className="badge bg-warning text-dark">{s.study_id}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── ALL STUDIES ── */}
      {tab === 'studies' && (
        <div>
          <div className="mb-3 d-flex gap-2 align-items-center">
            <input
              className="form-control form-control-sm w-auto"
              placeholder="Search study ID, type, site, PI…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{ minWidth: 260 }}
            />
            <span className="text-muted small">{filteredStudies.length} of {allStudies.length} studies</span>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr>
                  {[
                    { key: 'study_id', label: 'Study ID' },
                    { key: 'study_type', label: 'Type' },
                    { key: 'status', label: 'Status' },
                    { key: 'site', label: 'Site' },
                    { key: 'principal_investigator', label: 'PI' },
                    { key: 'sample_size', label: 'N' },
                    { key: 'sensitivity', label: 'Sens' },
                    { key: 'specificity', label: 'Spec' },
                    { key: 'auc_roc', label: 'AUC' },
                    { key: 'start_date', label: 'Start' },
                  ].map(col => (
                    <th
                      key={col.key}
                      style={{ cursor: 'pointer', userSelect: 'none' }}
                      onClick={() => toggleSort(col.key)}
                    >
                      {col.label}{sortArrow(col.key)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredStudies.map(s => (
                  <tr key={s.study_id}>
                    <td><code className="small">{s.study_id}</code></td>
                    <td className="small">{s.study_type}</td>
                    <td>
                      <span className={`badge bg-${statusColor(s.status)}`}>{s.status}</span>
                    </td>
                    <td className="small">{s.site}</td>
                    <td className="small">{s.principal_investigator}</td>
                    <td className="small text-end">{s.sample_size != null ? s.sample_size.toLocaleString() : '—'}</td>
                    <td className="small text-end">{fmtNum(s.sensitivity)}</td>
                    <td className="small text-end">{fmtNum(s.specificity)}</td>
                    <td className="small text-end">
                      {s.auc_roc != null
                        ? <span className={`badge bg-${s.auc_roc >= 0.9 ? 'success' : s.auc_roc >= 0.8 ? 'primary' : 'warning'}`}>{fmtNum(s.auc_roc)}</span>
                        : '—'}
                    </td>
                    <td className="small">{s.start_date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── PER SUBMISSION ── */}
      {tab === 'submissions' && (
        <div>
          <p className="text-muted small mb-3">Studies grouped by regulatory submission ID.</p>
          {perSubmission.length === 0
            ? <div className="alert alert-info">No submission data available.</div>
            : (
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead className="table-dark">
                    <tr>
                      <th>Submission ID</th>
                      <th>Studies</th>
                      <th>Passed</th>
                      <th>Failed</th>
                      <th>In Progress</th>
                      <th>Avg AUC</th>
                      <th>Avg N</th>
                    </tr>
                  </thead>
                  <tbody>
                    {perSubmission.map(sub => (
                      <tr key={sub.submission_id}>
                        <td><code className="small">{sub.submission_id}</code></td>
                        <td><span className="badge bg-primary">{sub.total_studies}</span></td>
                        <td><span className="badge bg-success">{sub.passed}</span></td>
                        <td>
                          {sub.failed > 0
                            ? <span className="badge bg-danger">{sub.failed}</span>
                            : <span className="text-muted">0</span>}
                        </td>
                        <td>
                          {sub.in_progress > 0
                            ? <span className="badge bg-warning text-dark">{sub.in_progress}</span>
                            : <span className="text-muted">0</span>}
                        </td>
                        <td className="small text-end">{fmtNum(sub.avg_auc_roc)}</td>
                        <td className="small text-end">{sub.avg_sample_size != null ? Math.round(sub.avg_sample_size).toLocaleString() : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
        </div>
      )}

      {/* ── PI WORKLOAD ── */}
      {tab === 'pi' && (
        <div>
          <p className="text-muted small mb-3">Principal Investigator workload across all studies.</p>
          <div className="row g-3">
            {piWorkload.map(pi => {
              const passRate = pi.studies > 0 ? ((pi.passed / pi.studies) * 100).toFixed(0) : 0;
              return (
                <div className="col-md-6 col-lg-4" key={pi.principal_investigator}>
                  <div className="card shadow-sm h-100">
                    <div className="card-header fw-semibold d-flex justify-content-between">
                      <span>{pi.principal_investigator}</span>
                      <span className="badge bg-primary">{pi.studies} studies</span>
                    </div>
                    <div className="card-body py-2">
                      <div className="row row-cols-3 g-2 text-center mb-2">
                        <div className="col">
                          <div className="text-success fw-bold">{pi.passed}</div>
                          <div className="small text-muted">Passed</div>
                        </div>
                        <div className="col">
                          <div className="text-danger fw-bold">{pi.failed}</div>
                          <div className="small text-muted">Failed</div>
                        </div>
                        <div className="col">
                          <div className="text-dark fw-bold">{fmtInt(pi.avg_sample_size)}</div>
                          <div className="small text-muted">Avg N</div>
                        </div>
                      </div>
                      <div className="small text-muted mb-1">Pass rate: {passRate}%</div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-success" style={{ width: `${passRate}%` }} />
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          {/* Study types */}
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Study Types</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Type</th><th>Purpose</th></tr></thead>
                  <tbody>
                    {(defs.study_types || []).map(t => (
                      <tr key={t.type}>
                        <td className="fw-semibold small">{t.type}</td>
                        <td className="small text-muted">{t.purpose}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Metrics */}
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Performance Metrics</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Metric</th><th>Definition</th><th>Threshold</th></tr></thead>
                  <tbody>
                    {(defs.metrics || []).map(m => (
                      <tr key={m.metric}>
                        <td className="fw-semibold small">{m.metric}</td>
                        <td className="small text-muted">{m.definition}</td>
                        <td className="small">
                          {m.target_threshold && <span className="badge bg-info">{m.target_threshold}</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Statuses */}
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Status Meanings</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Status</th><th>Meaning</th></tr></thead>
                  <tbody>
                    {(defs.statuses || []).map(s => (
                      <tr key={s.status}>
                        <td>
                          <span className={`badge bg-${statusColor(s.status)}`}>{s.status}</span>
                        </td>
                        <td className="small text-muted">{s.meaning}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Regulatory context */}
          {defs.regulatory_context && (
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Regulatory Context</div>
                <div className="card-body">
                  {Object.entries(defs.regulatory_context).map(([k, v]) => (
                    <div key={k} className="mb-2">
                      <div className="fw-semibold small">{k}</div>
                      <div className="text-muted small">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Glossary */}
          {(defs.glossary || []).length > 0 && (
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Glossary</div>
                <div className="card-body">
                  <div className="row">
                    {(defs.glossary || []).map(g => (
                      <div className="col-md-4 mb-2" key={g.term}>
                        <strong className="small">{g.term}</strong>
                        <div className="text-muted small">{g.definition}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
