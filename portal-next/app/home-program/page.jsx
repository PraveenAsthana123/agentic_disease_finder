'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const CAT_COLOR = {
  adl_restoration: 'primary',
  vocational_rehab: 'success',
  cognitive_rehab: 'info',
  social_skills: 'warning',
  fine_motor: 'secondary',
  mobility_training: 'danger',
};

const GRADE_COLOR = { A: 'success', B: 'primary', C: 'warning', D: 'danger' };

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'exercises', label: 'Exercise Library' },
  { id: 'patients', label: 'Per Patient' },
  { id: 'education', label: 'Education Modules' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, sub, color = 'primary' }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className={`card border-${color} h-100`}>
        <div className="card-body text-center p-3">
          <div className={`display-6 fw-bold text-${color}`}>{value ?? '—'}</div>
          <div className="small fw-semibold text-muted">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.72rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function ProgressBar({ pct, color = 'primary', height = 12 }) {
  return (
    <div className="progress" style={{ height }}>
      <div
        className={`progress-bar bg-${color}`}
        style={{ width: `${Math.min(100, pct || 0)}%` }}
      />
    </div>
  );
}

export default function HomeProgramPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState({});

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/home-program/overview`).then(r => r.json()),
      fetch(`${API}/api/home-program/breakdown`).then(r => r.json()),
      fetch(`${API}/api/home-program/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border text-primary" /><div className="mt-2">Loading Home Program data…</div></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  const s = overview?.summary || {};
  const cats = overview?.category_breakdown || [];
  const edu = overview?.education || {};
  const dp = overview?.daily_plan || {};

  const filteredPatients = (breakdown?.patients || []).filter(p =>
    !search || p.patient_id.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div>
          <h2 className="mb-0 fw-bold">🏠 OT Home Program Builder</h2>
          <div className="text-muted small">
            Occupational Therapy home exercise programs — {s.total_patients} patients · {s.total_plans} plans · AOTA 2020 / ILAE 2021
          </div>
        </div>
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

      {/* Overview Tab */}
      {tab === 'overview' && (
        <>
          <div className="row mb-3">
            <KPI label="Total Patients" value={s.total_patients} color="primary" />
            <KPI label="Total Plans" value={s.total_plans} sub={`${s.active_plans} active`} color="success" />
            <KPI label="Avg Progress" value={`${s.avg_progress_pct}%`} color="info" />
            <KPI label="Session Adherence" value={`${s.session_adherence_pct}%`} sub={`${s.sessions_completed}/${s.sessions_planned} sessions`} color="warning" />
          </div>

          <div className="row mb-3">
            <KPI label="Completed Plans" value={s.completed_plans} color="success" />
            <KPI label="On Hold" value={s.on_hold_plans} color="warning" />
            <KPI label="Discontinued" value={s.discontinued_plans} color="danger" />
            <KPI label="Education Modules" value={edu.total_modules} sub={`${edu.patients_enrolled} patients · avg ${edu.avg_completion_pct}% complete`} color="secondary" />
          </div>

          {/* Category bars */}
          <div className="card mb-3">
            <div className="card-header fw-semibold">Goal Category Breakdown</div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead className="table-light">
                    <tr>
                      <th>Category</th>
                      <th>Plans</th>
                      <th>Avg Progress</th>
                      <th>Sessions</th>
                      <th>Adherence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cats.map(c => (
                      <tr key={c.category}>
                        <td><span className={`badge bg-${CAT_COLOR[c.category] || 'secondary'}`}>{c.label}</span></td>
                        <td>{c.plan_count}</td>
                        <td>
                          <ProgressBar pct={c.avg_progress_pct} color={CAT_COLOR[c.category] || 'primary'} />
                          <small className="text-muted">{c.avg_progress_pct}%</small>
                        </td>
                        <td>{c.sessions_completed}/{c.sessions_planned}</td>
                        <td>
                          <ProgressBar pct={c.session_adherence_pct} color="success" />
                          <small className="text-muted">{c.session_adherence_pct}%</small>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Daily plan and education summary */}
          <div className="row">
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header fw-semibold">Daily Plan Adherence</div>
                <div className="card-body">
                  <ul className="list-unstyled mb-0">
                    <li><strong>Total days logged:</strong> {dp.total_days}</li>
                    <li><strong>Exercise logged rate:</strong> {dp.exercise_logged_rate_pct}%</li>
                    <li><strong>Avg daily completion:</strong> {dp.avg_completion_pct}%</li>
                  </ul>
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header fw-semibold">Education Component</div>
                <div className="card-body">
                  <ul className="list-unstyled mb-0">
                    <li><strong>Total modules:</strong> {edu.total_modules}</li>
                    <li><strong>Patients enrolled:</strong> {edu.patients_enrolled}</li>
                    <li><strong>Avg completion:</strong> {edu.avg_completion_pct}%</li>
                    <li><strong>Avg quiz score:</strong> {edu.avg_quiz_score}/100</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="alert alert-info mt-2 small">{overview?.clinical_note}</div>
        </>
      )}

      {/* Exercise Library Tab */}
      {tab === 'exercises' && (
        <>
          <p className="text-muted small mb-3">Prescribed home exercises per goal category — 6 categories, seizure-safety modified per ILAE 2021</p>
          <div className="row">
            {cats.map(c => (
              <div key={c.category} className="col-md-4 mb-3">
                <div className={`card border-${CAT_COLOR[c.category] || 'secondary'} h-100`}>
                  <div className={`card-header bg-${CAT_COLOR[c.category] || 'secondary'} text-white fw-semibold`}>
                    {c.label}
                  </div>
                  <div className="card-body">
                    <div className="mb-2 small text-muted">
                      {c.plan_count} plans · {c.avg_progress_pct}% avg progress · {c.session_adherence_pct}% adherence
                    </div>
                    <ul className="mb-0 ps-3">
                      {(c.sample_exercises || []).map((ex, i) => (
                        <li key={i} className="small">{ex}</li>
                      ))}
                    </ul>
                    {/* Show all exercises from breakdown */}
                    {(() => {
                      const allExs = (breakdown?.patients || [])
                        .flatMap(p => p.prescribed_program || [])
                        .find(pr => pr.category === c.category);
                      const extras = (allExs?.exercises || []).slice(3);
                      return extras.length > 0 ? (
                        <ul className="mb-0 ps-3 mt-1">
                          {extras.map((ex, i) => (
                            <li key={i} className="small text-muted">{ex}</li>
                          ))}
                        </ul>
                      ) : null;
                    })()}
                    <div className="mt-2 text-muted small">
                      <strong>Frequency:</strong> {c.plan_count > 0 ? '5× / week (active) · 2× / week (maintenance)' : 'N/A'}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Per Patient Tab */}
      {tab === 'patients' && (
        <>
          <div className="mb-3">
            <input
              className="form-control"
              placeholder="Search patient ID…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{ maxWidth: 280 }}
            />
          </div>
          <div className="table-responsive mb-3">
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Plans</th>
                  <th>Avg Progress</th>
                  <th>Sessions</th>
                  <th>Adherence</th>
                  <th>Grade</th>
                  <th>Education</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {filteredPatients.map(p => (
                  <>
                    <tr key={p.patient_id}>
                      <td><strong>{p.patient_id}</strong></td>
                      <td>{p.plan_count}</td>
                      <td>
                        <ProgressBar pct={p.avg_progress_pct} />
                        <small className="text-muted">{p.avg_progress_pct}%</small>
                      </td>
                      <td>{p.sessions_completed}/{p.sessions_planned}</td>
                      <td>
                        <ProgressBar pct={p.session_adherence_pct} color="success" />
                        <small className="text-muted">{p.session_adherence_pct}%</small>
                      </td>
                      <td>
                        <span className={`badge bg-${GRADE_COLOR[p.overall_adherence_grade] || 'secondary'} fs-6`}>
                          {p.overall_adherence_grade}
                        </span>
                      </td>
                      <td>
                        <small>{p.education?.modules || 0} modules · {p.education?.avg_completion || 0}%</small>
                      </td>
                      <td>
                        <button
                          className="btn btn-sm btn-outline-primary py-0"
                          onClick={() => setExpanded(prev => ({ ...prev, [p.patient_id]: !prev[p.patient_id] }))}
                        >
                          {expanded[p.patient_id] ? '▲' : '▼'}
                        </button>
                      </td>
                    </tr>
                    {expanded[p.patient_id] && (
                      <tr key={`${p.patient_id}-detail`}>
                        <td colSpan={8}>
                          <div className="px-3 pb-2">
                            <strong>Prescribed Home Program:</strong>
                            <div className="row mt-2">
                              {(p.prescribed_program || []).map(pr => (
                                <div key={pr.category} className="col-md-4 mb-2">
                                  <div className={`card border-${CAT_COLOR[pr.category] || 'secondary'}`}>
                                    <div className={`card-header py-1 bg-${CAT_COLOR[pr.category] || 'secondary'} text-white small fw-semibold`}>
                                      {pr.label}
                                    </div>
                                    <div className="card-body p-2">
                                      <div className="small text-muted mb-1">
                                        {pr.frequency} · {pr.intensity}
                                      </div>
                                      <ul className="mb-0 ps-3">
                                        {pr.exercises.slice(0, 3).map((ex, i) => (
                                          <li key={i} className="small">{ex}</li>
                                        ))}
                                      </ul>
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-muted small">Showing {filteredPatients.length} of {breakdown?.total_patients} patients</div>
        </>
      )}

      {/* Education Modules Tab */}
      {tab === 'education' && (
        <>
          <p className="text-muted small mb-3">Patient education library — 12 modules covering epilepsy self-management, AED adherence, SUDEP awareness, lifestyle</p>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  <th>Module</th>
                  <th>Enrolled Patients</th>
                  <th>Avg Completion</th>
                  <th>Avg Quiz Score</th>
                  <th>Formats</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.education_library || []).map(m => (
                  <tr key={m.module}>
                    <td><strong>{m.module}</strong></td>
                    <td>{m.enrolled_patients}</td>
                    <td>
                      <ProgressBar pct={m.avg_completion_pct} color={m.avg_completion_pct >= 70 ? 'success' : m.avg_completion_pct >= 50 ? 'warning' : 'danger'} />
                      <small className="text-muted">{m.avg_completion_pct}%</small>
                    </td>
                    <td>
                      <ProgressBar pct={m.avg_quiz_score} color="info" />
                      <small className="text-muted">{m.avg_quiz_score}/100</small>
                    </td>
                    <td><span className="badge bg-secondary">{m.formats}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && (
        <>
          <p className="text-muted small mb-3">Clinical glossary for OT Home Program Builder — AOTA 2020 / ILAE 2021 / WHO ICF 2001 aligned</p>
          <div className="row">
            {(definitions?.terms || []).map(t => (
              <div key={t.term} className="col-md-6 mb-3">
                <div className="card h-100">
                  <div className="card-header fw-semibold">{t.term}</div>
                  <div className="card-body">
                    <p className="small mb-1">{t.definition}</p>
                    <div className="text-muted" style={{ fontSize: '0.72rem' }}>
                      <em>{t.source}</em>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="card mt-2">
            <div className="card-header fw-semibold">References</div>
            <div className="card-body">
              <ol className="mb-0">
                {(definitions?.references || []).map((r, i) => (
                  <li key={i} className="small text-muted">{r}</li>
                ))}
              </ol>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
