'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const fmtIcon = f => ({ video: '🎬', article: '📄', quiz: '📝', interactive: '🖥️' }[f] || '📚');
const compColor = pct => {
  if (pct >= 100) return 'success';
  if (pct >= 75) return 'info';
  if (pct >= 25) return 'warning';
  if (pct > 0) return 'secondary';
  return 'light';
};
const compLabel = pct => {
  if (pct >= 100) return 'Completed';
  if (pct > 0) return 'In Progress';
  return 'Not Started';
};

export default function EducationModulesDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('avg_completion');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/education-modules/overview`).then(r => r.json()),
      fetch(`${API}/api/education-modules/breakdown`).then(r => r.json()),
      fetch(`${API}/api/education-modules/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading education modules data…</div>;

  const TABS = [
    { id: 'overview',   label: '📊 Overview' },
    { id: 'modules',    label: '📚 By Module' },
    { id: 'patients',   label: '👤 Per Patient' },
    { id: 'log',        label: '📋 Enrollment Log' },
    { id: 'definitions',label: '📖 Definitions' },
  ];

  const patients = bd?.per_patient || [];
  const enrollmentLog = bd?.enrollment_log || [];
  const moduleDist = ov.module_distribution || [];
  const formatDist = ov.format_distribution || [];
  const compDist = ov.completion_distribution || [];
  const monthlyTrend = ov.monthly_trend || [];
  const quizScoreDist = ov.quiz_score_distribution || [];

  const filteredPatients = patients
    .filter(p =>
      !search ||
      p.patient_id.toLowerCase().includes(search.toLowerCase())
    )
    .sort((a, b) => {
      if (sortBy === 'avg_completion') return b.avg_completion - a.avg_completion;
      if (sortBy === 'completed') return b.completed - a.completed;
      if (sortBy === 'avg_quiz') return (b.avg_quiz || 0) - (a.avg_quiz || 0);
      if (sortBy === 'total_time') return b.total_time - a.total_time;
      return a.patient_id.localeCompare(b.patient_id);
    });

  const filteredLog = enrollmentLog.filter(r =>
    !search ||
    r.patient_id.toLowerCase().includes(search.toLowerCase()) ||
    r.module_name.toLowerCase().includes(search.toLowerCase())
  );

  const kpis = [
    { label: 'Enrollments',      value: ov.total_enrollments, color: 'primary' },
    { label: 'Patients',          value: ov.total_patients,    color: 'info' },
    { label: 'Modules',           value: ov.total_modules,     color: 'secondary' },
    { label: 'Completion Rate',   value: `${ov.completion_rate}%`, color: 'success' },
    { label: 'Avg Completion',    value: `${ov.avg_completion}%`,  color: 'warning' },
    { label: 'Quiz Pass Rate',    value: `${ov.quiz_pass_rate}%`,  color: 'success' },
    { label: 'Avg Quiz Score',    value: `${ov.avg_quiz_score}`,   color: 'info' },
    { label: 'Total Hours',       value: `${ov.total_time_hours}h`,color: 'dark' },
  ];

  const maxModCnt = Math.max(...moduleDist.map(m => m.cnt), 1);

  return (
    <div className="p-3">
      <h3>📚 Patient Education Modules</h3>
      <p className="text-muted small">
        {ov.total_enrollments} enrollments · {ov.total_patients} patients · {ov.total_modules} modules ·
        {ov.completion_rate}% completion rate · avg quiz {ov.avg_quiz_score} · {ov.total_time_hours}h total learning
      </p>

      {/* KPI strip */}
      <div className="row g-2 mb-3">
        {kpis.map(k => (
          <div key={k.label} className="col-6 col-md-3 col-lg-2">
            <div className={`card border-${k.color} shadow-sm h-100`}>
              <div className="card-body p-2 text-center">
                <div className={`fw-bold fs-5 text-${k.color}`}>{k.value}</div>
                <div className="text-muted" style={{ fontSize: '0.7rem' }}>{k.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Status summary row */}
      <div className="row g-2 mb-3">
        <div className="col-md-4">
          <div className="card shadow-sm h-100">
            <div className="card-body p-2">
              <div className="small fw-semibold mb-1">Completion Status</div>
              {compDist.map(c => {
                const color = c.bucket === 'Complete (100%)' ? 'success' : c.bucket === 'Not Started (0%)' ? 'secondary' : 'warning';
                return (
                  <div key={c.bucket} className="d-flex justify-content-between align-items-center mb-1">
                    <span className="small">{c.bucket}</span>
                    <div className="d-flex align-items-center gap-2">
                      <div className="progress" style={{ width: 80, height: 8 }}>
                        <div className={`progress-bar bg-${color}`} style={{ width: `${(c.cnt / ov.total_enrollments) * 100}%` }} />
                      </div>
                      <span className={`badge bg-${color}`}>{c.cnt}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm h-100">
            <div className="card-body p-2">
              <div className="small fw-semibold mb-1">Format Distribution</div>
              {formatDist.map(f => (
                <div key={f.format} className="d-flex justify-content-between align-items-center mb-1">
                  <span className="small">{fmtIcon(f.format)} {f.format}</span>
                  <div className="d-flex align-items-center gap-2">
                    <div className="progress" style={{ width: 80, height: 8 }}>
                      <div className="progress-bar bg-primary" style={{ width: `${(f.cnt / ov.total_enrollments) * 100}%` }} />
                    </div>
                    <span className="small text-muted">{f.cnt} · {f.avg_completion?.toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm h-100">
            <div className="card-body p-2">
              <div className="small fw-semibold mb-1">Quiz Score Distribution (n={ov.quiz_taken})</div>
              {quizScoreDist.map(q => (
                <div key={q.bucket} className="d-flex justify-content-between align-items-center mb-1">
                  <span className="small">{q.bucket}</span>
                  <div className="d-flex align-items-center gap-2">
                    <div className="progress" style={{ width: 80, height: 8 }}>
                      <div className="progress-bar bg-success" style={{ width: `${(q.cnt / ov.quiz_taken) * 100}%` }} />
                    </div>
                    <span className="badge bg-success">{q.cnt}</span>
                  </div>
                </div>
              ))}
              <div className="text-muted mt-1" style={{ fontSize: '0.7rem' }}>
                Pass rate: {ov.quiz_pass_rate}% (≥70 threshold)
              </div>
            </div>
          </div>
        </div>
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

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          <div className="card shadow-sm mb-3">
            <div className="card-body p-2">
              <div className="small fw-semibold mb-2">Monthly Enrollment Trend</div>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Month</th>
                    <th className="text-end">Enrollments</th>
                    <th className="text-end">Completed</th>
                    <th className="text-end">Avg Completion</th>
                  </tr>
                </thead>
                <tbody>
                  {monthlyTrend.map(m => (
                    <tr key={m.month}>
                      <td>{m.month}</td>
                      <td className="text-end">{m.enrollments}</td>
                      <td className="text-end">{m.completed}</td>
                      <td className="text-end">
                        <div className="d-flex align-items-center justify-content-end gap-2">
                          <div className="progress" style={{ width: 60, height: 6 }}>
                            <div className={`progress-bar bg-${m.avg_completion >= 60 ? 'success' : 'warning'}`} style={{ width: `${m.avg_completion}%` }} />
                          </div>
                          <span>{m.avg_completion?.toFixed(1)}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Status counts */}
          <div className="row g-2">
            {[
              { label: 'Completed', value: ov.completed_count, color: 'success', note: `${((ov.completed_count / ov.total_enrollments) * 100).toFixed(1)}% of enrollments` },
              { label: 'In Progress', value: ov.in_progress, color: 'warning', note: 'started but incomplete' },
              { label: 'Not Started', value: ov.not_started, color: 'secondary', note: 'enrolled, 0% progress' },
              { label: 'Quiz Taken', value: ov.quiz_taken, color: 'info', note: `avg ${ov.avg_time_minutes} min/module` },
            ].map(s => (
              <div key={s.label} className="col-6 col-md-3">
                <div className={`card border-${s.color} shadow-sm`}>
                  <div className="card-body p-2 text-center">
                    <div className={`fw-bold fs-4 text-${s.color}`}>{s.value}</div>
                    <div className="fw-semibold small">{s.label}</div>
                    <div className="text-muted" style={{ fontSize: '0.7rem' }}>{s.note}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── BY MODULE ── */}
      {tab === 'modules' && (
        <div className="card shadow-sm">
          <div className="card-body p-2">
            <div className="small fw-semibold mb-2">Module Performance ({moduleDist.length} modules)</div>
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Module</th>
                  <th className="text-end">Enrolled</th>
                  <th className="text-end">Completed</th>
                  <th className="text-end">Avg Completion</th>
                  <th className="text-end">Avg Time (min)</th>
                </tr>
              </thead>
              <tbody>
                {moduleDist.map(m => (
                  <tr key={m.module_name}>
                    <td>{m.module_name}</td>
                    <td className="text-end">{m.cnt}</td>
                    <td className="text-end">{m.completed}</td>
                    <td className="text-end">
                      <div className="d-flex align-items-center justify-content-end gap-2">
                        <div className="progress" style={{ width: 70, height: 6 }}>
                          <div
                            className={`progress-bar bg-${m.avg_comp >= 70 ? 'success' : m.avg_comp >= 50 ? 'warning' : 'danger'}`}
                            style={{ width: `${m.avg_comp}%` }}
                          />
                        </div>
                        <span>{m.avg_comp?.toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="text-end">{m.avg_time?.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && (
        <div>
          <div className="d-flex gap-2 mb-2 flex-wrap">
            <input
              className="form-control form-control-sm"
              style={{ maxWidth: 200 }}
              placeholder="Search patient…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            <select className="form-select form-select-sm" style={{ maxWidth: 180 }} value={sortBy} onChange={e => setSortBy(e.target.value)}>
              <option value="avg_completion">Sort: Avg Completion</option>
              <option value="completed">Sort: Completed</option>
              <option value="avg_quiz">Sort: Quiz Score</option>
              <option value="total_time">Sort: Total Time</option>
              <option value="patient_id">Sort: Patient ID</option>
            </select>
          </div>
          <div className="card shadow-sm">
            <div className="card-body p-2">
              <div className="small text-muted mb-1">{filteredPatients.length} patients</div>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th className="text-end">Modules</th>
                    <th className="text-end">Completed</th>
                    <th className="text-end">Avg Completion</th>
                    <th className="text-end">Avg Quiz</th>
                    <th className="text-end">Total Time (min)</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPatients.map(p => (
                    <tr key={p.patient_id}>
                      <td><code>{p.patient_id}</code></td>
                      <td className="text-end">{p.total_modules}</td>
                      <td className="text-end">
                        <span className={`badge bg-${p.completed === p.total_modules ? 'success' : 'warning'}`}>
                          {p.completed}/{p.total_modules}
                        </span>
                      </td>
                      <td className="text-end">
                        <div className="d-flex align-items-center justify-content-end gap-2">
                          <div className="progress" style={{ width: 60, height: 6 }}>
                            <div
                              className={`progress-bar bg-${compColor(p.avg_completion)}`}
                              style={{ width: `${p.avg_completion}%` }}
                            />
                          </div>
                          <span>{p.avg_completion?.toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="text-end">
                        {p.avg_quiz
                          ? <span className={`badge bg-${p.avg_quiz >= 70 ? 'success' : 'danger'}`}>{p.avg_quiz?.toFixed(1)}</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td className="text-end">{p.total_time}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── ENROLLMENT LOG ── */}
      {tab === 'log' && (
        <div>
          <div className="d-flex gap-2 mb-2">
            <input
              className="form-control form-control-sm"
              style={{ maxWidth: 240 }}
              placeholder="Search patient or module…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            <span className="small text-muted align-self-center">{filteredLog.length} records</span>
          </div>
          <div className="card shadow-sm">
            <div className="card-body p-0">
              <div style={{ maxHeight: 480, overflowY: 'auto' }}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light sticky-top">
                    <tr>
                      <th>Patient</th>
                      <th>Module</th>
                      <th>Format</th>
                      <th className="text-end">Completion</th>
                      <th className="text-end">Quiz</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredLog.map((r, i) => (
                      <tr key={i}>
                        <td><code className="small">{r.patient_id}</code></td>
                        <td className="small">{r.module_name}</td>
                        <td className="small">{fmtIcon(r.format)} {r.format}</td>
                        <td className="text-end">
                          <span className={`badge bg-${compColor(r.completion_pct)}`}>
                            {r.completion_pct}%
                          </span>
                        </td>
                        <td className="text-end small">
                          {r.quiz_score != null
                            ? <span className={r.quiz_score >= 70 ? 'text-success' : 'text-danger'}>{r.quiz_score?.toFixed(1)}</span>
                            : <span className="text-muted">—</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-2">
          {(defs.glossary || []).map(g => (
            <div key={g.term} className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body p-2">
                  <div className="fw-semibold small">{g.term}</div>
                  <div className="text-muted small">{g.definition}</div>
                </div>
              </div>
            </div>
          ))}
          {defs.modules && (
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-body p-2">
                  <div className="fw-semibold small mb-1">Module Catalog</div>
                  <div className="row">
                    {defs.modules.map(m => (
                      <div key={m.name} className="col-md-6 col-lg-4 mb-1">
                        <div className="border rounded p-1 small">
                          <div className="fw-semibold">{m.name}</div>
                          <div className="text-muted">{m.description}</div>
                          <span className="badge bg-secondary mt-1">{m.format}</span>
                        </div>
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
