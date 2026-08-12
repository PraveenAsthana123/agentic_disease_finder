'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLOR = {
  real:    'success',
  partial: 'warning',
  design:  'info',
  pending: 'secondary',
};
const STATUS_LABEL = {
  real:    'Real',
  partial: 'Partial',
  design:  'Design',
  pending: 'Pending',
};
const IEC_COLOR = {
  submitted:      'success',
  in_progress:    'warning',
  drafted:        'info',
  pending:        'secondary',
  not_applicable: 'light',
};

const TABS = [
  { id: 'overview',     label: 'Overview' },
  { id: 'categories',   label: 'Categories A–I' },
  { id: 'documents',    label: 'Document List' },
  { id: 'jurisdiction', label: 'Jurisdiction Map' },
  { id: 'definitions',  label: 'Definitions' },
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

function ProgressBar({ label, value, max, color, pct: pctProp }) {
  const pct = pctProp !== undefined ? pctProp : (max ? Math.round((value / max) * 100) : 0);
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small text-muted mb-1">
        <span>{label}</span>
        <span>{pctProp !== undefined ? `${pct}%` : `${value}/${max} (${pct}%)`}</span>
      </div>
      <div className="progress" style={{ height: '8px' }}>
        <div className={`progress-bar bg-${color || 'primary'}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  return (
    <span className={`badge bg-${STATUS_COLOR[status] || 'secondary'}`}>
      {STATUS_LABEL[status] || status}
    </span>
  );
}

function useApi(path) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  useEffect(() => {
    setLoading(true);
    fetch(`${API}${path}`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [path]);
  return { data, loading, error };
}

export default function IecIrbTracker() {
  const [tab, setTab] = useState('overview');
  const [catFilter, setCatFilter] = useState('ALL');
  const [statusFilter, setStatusFilter] = useState('ALL');

  const { data: ovData, loading: ovLoad } = useApi('/api/iec-irb-tracker/overview');
  const { data: bkData, loading: bkLoad } = useApi('/api/iec-irb-tracker/breakdown');
  const { data: defData, loading: defLoad } = useApi('/api/iec-irb-tracker/definitions');

  const ov = ovData || {};
  const bk = bkData || {};
  const def = defData || {};

  return (
    <div>
      <div className="d-flex align-items-center mb-3">
        <h4 className="mb-0 me-3">📋 IEC / IRB 173-Document Submission Tracker</h4>
        <span className="badge bg-primary me-2">DBA Research</span>
        <span className="badge bg-success">Phase-Tracked</span>
      </div>
      <p className="text-muted small mb-3">
        Tracks the 173-document master list (categories A–I) for phased IEC (India) + IRB (GGU)
        submission. Multi-jurisdiction compliance: ICMR · DPDP Act 2023 · HIPAA · PIPEDA · TCPS 2 ·
        ICH-GCP · Helsinki. Study: Praveen Asthana DBA — Golden Gate University.
      </p>

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

      {/* ── Overview ── */}
      {tab === 'overview' && (
        ovLoad ? <div className="spinner-border text-primary" /> :
        <div>
          {/* KPIs */}
          <div className="row g-2 mb-4">
            <KPI label="Total Documents"  value={ov.kpis?.total_documents}     color="primary" />
            <KPI label="Real Complete"    value={ov.kpis?.real_complete}        color="success" sub="fully drafted" />
            <KPI label="Partial"          value={ov.kpis?.partial}              color="warning" sub="0.5× weight" />
            <KPI label="Design"           value={ov.kpis?.design}               color="info"    sub="0.25× weight" />
            <KPI label="Pending"          value={ov.kpis?.pending}              color="secondary" sub="not started" />
            <KPI label="Completion"       value={`${ov.kpis?.completion_pct}%`} color="dark" sub="weighted" />
          </div>

          {/* Submission readiness */}
          <div className="row mb-4">
            <div className="col-md-6">
              <div className="card shadow-sm border-0 h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3">🏛️ IEC Phase-1 Readiness (India)</h6>
                  <ProgressBar label="IEC Phase 1 (Retrospective EEG Classification)"
                    pct={ov.phases?.iec?.phase1?.pct} color="primary" />
                  <div className="small text-muted mb-2">{ov.phases?.iec?.phase1?.target}</div>
                  <div className="table-responsive">
                    <table className="table table-sm table-borderless">
                      <tbody>
                        {['phase1','phase2','phase3'].map((ph, i) => {
                          const p = ov.phases?.iec?.[ph];
                          if (!p) return null;
                          return (
                            <tr key={ph}>
                              <td className="fw-bold small">Phase {i+1}</td>
                              <td className="small">{p.name}</td>
                              <td className="small text-muted">{p.target}</td>
                              {p.pct !== undefined && <td><span className={`badge bg-${p.pct >= 80 ? 'success' : p.pct >= 50 ? 'warning' : 'secondary'}`}>{p.pct}%</span></td>}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm border-0 h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3">🎓 IRB Phase-1 Readiness (GGU)</h6>
                  <ProgressBar label="IRB Phase 1 (Core Approval)"
                    pct={ov.phases?.irb?.phase1?.pct} color="success" />
                  <div className="small text-muted mb-2">{ov.phases?.irb?.phase1?.target}</div>
                  <div className="table-responsive">
                    <table className="table table-sm table-borderless">
                      <tbody>
                        {['phase1','phase2','phase3'].map((ph, i) => {
                          const p = ov.phases?.irb?.[ph];
                          if (!p) return null;
                          return (
                            <tr key={ph}>
                              <td className="fw-bold small">Phase {i+1}</td>
                              <td className="small">{p.name}</td>
                              <td className="small text-muted">{p.target}</td>
                              {p.pct !== undefined && <td><span className={`badge bg-${p.pct >= 80 ? 'success' : p.pct >= 50 ? 'warning' : 'secondary'}`}>{p.pct}%</span></td>}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Category summary bars */}
          <div className="card shadow-sm border-0 mb-4">
            <div className="card-body">
              <h6 className="fw-bold mb-3">📂 Category Completion (A–I)</h6>
              {(ov.category_summary || []).map(cat => (
                <ProgressBar key={cat.category}
                  label={`Cat ${cat.category}: ${cat.label} (${cat.total} docs)`}
                  pct={cat.completion_pct}
                  color={cat.completion_pct >= 70 ? 'success' : cat.completion_pct >= 40 ? 'warning' : 'danger'}
                />
              ))}
            </div>
          </div>

          {/* Jurisdiction coverage */}
          <div className="card shadow-sm border-0 mb-4">
            <div className="card-body">
              <h6 className="fw-bold mb-3">🌍 Jurisdiction Coverage</h6>
              <div className="row">
                {(ov.jurisdiction_coverage || []).map(j => (
                  <div key={j.jurisdiction} className="col-6 col-md-3 mb-3">
                    <div className={`card border-${j.pct >= 70 ? 'success' : j.pct >= 40 ? 'warning' : 'danger'} h-100`}>
                      <div className="card-body text-center py-2">
                        <div className={`h4 fw-bold text-${j.pct >= 70 ? 'success' : j.pct >= 40 ? 'warning' : 'danger'}`}>{j.pct}%</div>
                        <div className="fw-semibold small">{j.jurisdiction}</div>
                        <div className="text-muted" style={{ fontSize: '0.7rem' }}>{j.done}/{j.total} done</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Standards */}
          <div className="card shadow-sm border-0">
            <div className="card-body">
              <h6 className="fw-bold mb-2">📜 Applicable Standards</h6>
              <div className="table-responsive">
                <table className="table table-sm">
                  <thead><tr><th>Standard</th><th>Jurisdiction</th><th>Status</th></tr></thead>
                  <tbody>
                    {(ov.standards || []).map(s => (
                      <tr key={s.name}>
                        <td className="small fw-semibold">{s.name}</td>
                        <td className="small">{s.jurisdiction}</td>
                        <td><StatusBadge status={s.status} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Categories A–I ── */}
      {tab === 'categories' && (
        bkLoad ? <div className="spinner-border text-primary" /> :
        <div>
          <div className="card shadow-sm border-0 mb-4">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Phase-Wise Document Summary</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead className="table-light">
                    <tr>
                      <th>System</th><th>Phase</th><th>Total</th>
                      <th>Real</th><th>Partial</th><th>Design</th><th>Pending</th><th>Completion</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bk.phase_summary || []).map(ph => (
                      <tr key={`${ph.system}-${ph.phase}`}>
                        <td className="fw-bold">{ph.system}</td>
                        <td>Phase {ph.phase}</td>
                        <td>{ph.total}</td>
                        <td><span className="badge bg-success">{ph.real}</span></td>
                        <td><span className="badge bg-warning text-dark">{ph.partial}</span></td>
                        <td><span className="badge bg-info text-dark">{ph.design}</span></td>
                        <td><span className="badge bg-secondary">{ph.pending}</span></td>
                        <td>
                          <div className="d-flex align-items-center gap-2">
                            <div className="progress flex-grow-1" style={{ height: 6 }}>
                              <div className={`progress-bar bg-${ph.completion_pct >= 70 ? 'success' : ph.completion_pct >= 40 ? 'warning' : 'secondary'}`}
                                style={{ width: `${ph.completion_pct}%` }} />
                            </div>
                            <span className="small">{ph.completion_pct}%</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="card shadow-sm border-0">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Priority Phase-1 Documents</h6>
              <p className="text-muted small">Documents required for IEC Phase 1 or IRB Phase 1 submission (highest priority).</p>
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead className="table-light">
                    <tr>
                      <th>ID</th><th>Document</th><th>Status</th>
                      <th>IEC Ph</th><th>IRB Ph</th><th>IEC Status</th><th>IRB Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bk.priority_documents || []).map(d => (
                      <tr key={d.id}>
                        <td className="small text-muted">{d.id}</td>
                        <td className="small fw-semibold">{d.name}</td>
                        <td><StatusBadge status={d.status} /></td>
                        <td><span className="badge bg-light text-dark border">Ph {d.iec_phase}</span></td>
                        <td><span className="badge bg-light text-dark border">Ph {d.irb_phase}</span></td>
                        <td><span className={`badge bg-${IEC_COLOR[d.iec_submission_status] || 'secondary'} text-dark`}>{d.iec_submission_status?.replace('_',' ')}</span></td>
                        <td><span className={`badge bg-${IEC_COLOR[d.irb_submission_status] || 'secondary'} text-dark`}>{d.irb_submission_status?.replace('_',' ')}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Document List ── */}
      {tab === 'documents' && (
        bkLoad ? <div className="spinner-border text-primary" /> :
        <div>
          <div className="d-flex gap-2 mb-3 flex-wrap">
            <select className="form-select form-select-sm" style={{ width: 'auto' }}
              value={catFilter} onChange={e => setCatFilter(e.target.value)}>
              <option value="ALL">All Categories</option>
              {['A','B','C','D','E','F','G','H','I'].map(c => (
                <option key={c} value={c}>Category {c}</option>
              ))}
            </select>
            <select className="form-select form-select-sm" style={{ width: 'auto' }}
              value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
              <option value="ALL">All Statuses</option>
              <option value="real">Real</option>
              <option value="partial">Partial</option>
              <option value="design">Design</option>
              <option value="pending">Pending</option>
            </select>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark sticky-top">
                <tr>
                  <th>ID</th><th>Document Name</th><th>Status</th>
                  <th>IEC Ph</th><th>IRB Ph</th><th>Jurisdictions</th>
                </tr>
              </thead>
              <tbody>
                {(bk.document_list || [])
                  .filter(d => catFilter === 'ALL' || d.category === catFilter)
                  .filter(d => statusFilter === 'ALL' || d.status === statusFilter)
                  .map(d => (
                    <tr key={d.id}>
                      <td className="text-muted small">{d.id}</td>
                      <td className="small">{d.name}</td>
                      <td><StatusBadge status={d.status} /></td>
                      <td className="small text-center">
                        <span className="badge bg-light text-dark border">Ph {d.iec_phase}</span>
                      </td>
                      <td className="small text-center">
                        <span className="badge bg-light text-dark border">Ph {d.irb_phase}</span>
                      </td>
                      <td className="small text-muted">
                        {(d.jurisdiction || []).join(' · ')}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          <div className="alert alert-secondary small mt-3">
            Showing {
              (bk.document_list || [])
                .filter(d => catFilter === 'ALL' || d.category === catFilter)
                .filter(d => statusFilter === 'ALL' || d.status === statusFilter).length
            } of {(bk.document_list || []).length} documents.
            Status legend: <b>Real</b>=fully drafted · <b>Partial</b>=incomplete (0.5×) ·
            <b>Design</b>=planned (0.25×) · <b>Pending</b>=not started (0×). Honest §57.7.
          </div>
        </div>
      )}

      {/* ── Jurisdiction Map ── */}
      {tab === 'jurisdiction' && (
        bkLoad ? <div className="spinner-border text-primary" /> :
        <div>
          <div className="row mb-4">
            {(bk.jurisdiction_map || []).map(j => (
              <div key={j.jurisdiction} className="col-md-6 mb-3">
                <div className="card shadow-sm border-0 h-100">
                  <div className="card-body">
                    <h6 className="fw-bold mb-2">
                      {j.jurisdiction === 'India'         ? '🇮🇳' :
                       j.jurisdiction === 'USA'           ? '🇺🇸' :
                       j.jurisdiction === 'Canada'        ? '🇨🇦' : '🌐'} {j.jurisdiction}
                    </h6>
                    <ProgressBar label="Document Completion" pct={j.completion_pct}
                      color={j.completion_pct >= 70 ? 'success' : j.completion_pct >= 40 ? 'warning' : 'danger'} />
                    <div className="row text-center mt-2">
                      <div className="col-3">
                        <div className="small fw-bold text-success">{j.real}</div>
                        <div className="small text-muted">Real</div>
                      </div>
                      <div className="col-3">
                        <div className="small fw-bold text-warning">{j.partial}</div>
                        <div className="small text-muted">Partial</div>
                      </div>
                      <div className="col-3">
                        <div className="small fw-bold text-info">{j.design}</div>
                        <div className="small text-muted">Design</div>
                      </div>
                      <div className="col-3">
                        <div className="small fw-bold text-secondary">{j.pending}</div>
                        <div className="small text-muted">Pending</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="card shadow-sm border-0 mb-4">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Consent Records (Live DB)</h6>
              {bk.consent_breakdown?.length > 0 ? (
                <div className="table-responsive">
                  <table className="table table-sm">
                    <thead><tr><th>Consent Type</th><th>Status</th><th>Count</th></tr></thead>
                    <tbody>
                      {(bk.consent_breakdown || []).map(cb =>
                        Object.entries(cb).filter(([k]) => k !== 'type').map(([st, cnt]) => (
                          <tr key={`${cb.type}-${st}`}>
                            <td className="small">{cb.type}</td>
                            <td><span className={`badge bg-${st === 'granted' ? 'success' : st === 'pending' ? 'warning' : 'secondary'}`}>{st}</span></td>
                            <td className="small fw-bold">{cnt}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              ) : <div className="text-muted small">No consent records found.</div>}
            </div>
          </div>

          <div className="card shadow-sm border-0">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Regulatory Submissions (Live DB)</h6>
              <div className="row">
                {Object.entries(bk.regulatory_submissions?.by_status || {}).map(([st, cnt]) => (
                  <div key={st} className="col-6 col-md-3 mb-2">
                    <div className={`card border-${st === 'Approved' ? 'success' : st === 'Under Review' ? 'warning' : 'secondary'} h-100`}>
                      <div className="card-body text-center py-2">
                        <div className={`h4 fw-bold text-${st === 'Approved' ? 'success' : 'secondary'}`}>{cnt}</div>
                        <div className="small">{st}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && (
        defLoad ? <div className="spinner-border text-primary" /> :
        <div>
          <div className="card shadow-sm border-0 mb-4">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Key Concepts</h6>
              <div className="accordion" id="conceptAccordion">
                {(def.concepts || []).map((c, i) => (
                  <div key={c.term} className="accordion-item">
                    <h2 className="accordion-header">
                      <button className={`accordion-button${i > 0 ? ' collapsed' : ''} py-2`}
                        type="button" data-bs-toggle="collapse"
                        data-bs-target={`#concept-${i}`}>
                        <span className="fw-semibold small">{c.term}</span>
                      </button>
                    </h2>
                    <div id={`concept-${i}`} className={`accordion-collapse collapse${i === 0 ? ' show' : ''}`}>
                      <div className="accordion-body small text-muted">{c.def}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card shadow-sm border-0 mb-4">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Standards & Jurisdictions</h6>
              <div className="table-responsive">
                <table className="table table-sm">
                  <thead><tr><th>Standard</th><th>Jurisdiction</th><th>Scope</th></tr></thead>
                  <tbody>
                    {(def.standards || []).map(s => (
                      <tr key={s.name}>
                        <td className="small fw-semibold">{s.name}</td>
                        <td><span className="badge bg-light text-dark border">{s.jurisdiction}</span></td>
                        <td className="small text-muted">{s.scope}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="card shadow-sm border-0 mb-4">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Performance Thresholds</h6>
              <div className="table-responsive">
                <table className="table table-sm">
                  <thead><tr><th>Metric</th><th>Target</th><th>Rationale</th></tr></thead>
                  <tbody>
                    {(def.thresholds || []).map(t => (
                      <tr key={t.metric}>
                        <td className="small fw-semibold">{t.metric}</td>
                        <td><span className="badge bg-primary">{t.target}</span></td>
                        <td className="small text-muted">{t.rationale}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="card shadow-sm border-0">
            <div className="card-body">
              <h6 className="fw-bold mb-3">References</h6>
              <ol className="small text-muted">
                {(def.references || []).map((r, i) => <li key={i} className="mb-1">{r}</li>)}
              </ol>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
