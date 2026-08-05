'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const severityColor = lvl => {
  const l = (lvl || '').toLowerCase();
  if (l === 'critical') return 'danger';
  if (l === 'severe') return 'warning';
  if (l === 'moderate') return 'info';
  if (l === 'mild') return 'success';
  return 'secondary';
};

const trendIcon = t => ({ improving: '↓ Improving', worsening: '↑ Worsening', stable: '→ Stable' }[t] || t);
const trendColor = t => ({ improving: 'success', worsening: 'danger', stable: 'secondary' }[t] || 'secondary');

export default function LSSSDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('avg_score');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/lsss/overview`).then(r => r.json()),
      fetch(`${API}/api/lsss/breakdown`).then(r => r.json()),
      fetch(`${API}/api/lsss/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading LSSS data...</div>;

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'patients', label: '👤 Per Patient' },
    { id: 'items', label: '📋 Item Analysis' },
    { id: 'log', label: '📝 Assessment Log' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const patients = bd?.patient_summary || [];
  const filteredPatients = patients
    .filter(p => !search || p.patient_id.toLowerCase().includes(search.toLowerCase()) || (p.latest_level || '').toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === 'avg_score') return b.avg_score - a.avg_score;
      if (sortBy === 'latest_score') return b.latest_score - a.latest_score;
      if (sortBy === 'assessments') return b.assessments - a.assessments;
      if (sortBy === 'trend') return (a.trend || '').localeCompare(b.trend || '');
      return a.patient_id.localeCompare(b.patient_id);
    });

  const assessmentLog = bd?.assessment_log || [];
  const itemAverages = bd?.item_averages || [];

  // Split item averages into ictal (item1-10) and post-ictal (item11-20)
  const ictalItems = itemAverages.filter(i => {
    const n = parseInt((i.item || '').replace('item', ''), 10);
    return n >= 1 && n <= 10;
  });
  const postIctalItems = itemAverages.filter(i => {
    const n = parseInt((i.item || '').replace('item', ''), 10);
    return n >= 11 && n <= 20;
  });

  const sevDist = ov.severity_distribution || {};
  const totalAssessments = ov.total_assessments || 0;

  return (
    <div className="p-3">
      <h3>📊 LSSS Dashboard</h3>
      <p className="text-muted">
        Liverpool Seizure Severity Scale — {ov.total_assessments} assessments · {ov.unique_patients} patients ·
        avg score {ov.avg_score?.toFixed(1)} / 80 · {ov.high_risk_patient_count} high-risk patients
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          {/* KPI row */}
          <div className="row mb-3">
            {[
              ['Total Assessments', ov.total_assessments, 'primary'],
              ['Unique Patients', ov.unique_patients, 'info'],
              ['Avg Score', ov.avg_score?.toFixed(1), 'warning'],
              ['High Risk', ov.high_risk_patient_count, 'danger'],
              ['Min Score', ov.min_score, 'success'],
              ['Max Score', ov.max_score, 'secondary'],
            ].map(([label, val, c]) => (
              <div key={label} className="col-6 col-md-2 mb-2">
                <div className="card shadow-sm h-100">
                  <div className="card-body text-center py-2">
                    <div className={`h5 mb-0 text-${c}`}>{val}</div>
                    <div className="text-muted small">{label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="row mb-3">
            {/* Severity distribution */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Severity Distribution</h6>
                  {['Mild', 'Moderate', 'Severe', 'Critical'].map(lvl => {
                    const count = sevDist[lvl] || 0;
                    const pct = totalAssessments > 0 ? ((count / totalAssessments) * 100).toFixed(0) : 0;
                    return (
                      <div key={lvl} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>{lvl}</span>
                          <span>{count} ({pct}%)</span>
                        </div>
                        <div className="progress" style={{ height: '10px' }}>
                          <div className={`progress-bar bg-${severityColor(lvl)}`} style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Score histogram */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Score Distribution</h6>
                  {(ov.score_histogram || []).map(({ bin, count }) => {
                    const pct = totalAssessments > 0 ? ((count / totalAssessments) * 100).toFixed(0) : 0;
                    return (
                      <div key={bin} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span className="font-monospace">{bin}</span>
                          <span>{count}</span>
                        </div>
                        <div className="progress" style={{ height: '10px' }}>
                          <div className="progress-bar bg-primary" style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Monthly trend */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Monthly Assessments</h6>
                  {(ov.monthly_trend || []).map(({ month, assessments, avg_score }) => (
                    <div key={month} className="d-flex justify-content-between border-bottom py-2">
                      <span className="small">{month}</span>
                      <div className="text-end">
                        <span className="badge bg-primary me-1">{assessments}</span>
                        <span className="text-muted small">avg {avg_score?.toFixed(1)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Score gauge */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6>Average Score — {ov.avg_score?.toFixed(1)} / 80 (higher = more severe)</h6>
              <div className="progress" style={{ height: '28px' }}>
                <div
                  className={`progress-bar bg-${severityColor(
                    ov.avg_score >= 70 ? 'critical' : ov.avg_score >= 55 ? 'severe' : ov.avg_score >= 40 ? 'moderate' : 'mild'
                  )}`}
                  style={{ width: `${((ov.avg_score || 0) / 80) * 100}%` }}
                >
                  {ov.avg_score?.toFixed(1)}
                </div>
              </div>
              <div className="d-flex justify-content-between small text-muted mt-1">
                <span>20 — Mild</span>
                <span>40 — Moderate</span>
                <span>55 — Severe</span>
                <span>70 — Critical — 80</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && (
        <div>
          <div className="d-flex gap-2 mb-3">
            <input
              className="form-control"
              style={{ maxWidth: '240px' }}
              placeholder="Search patient / level…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            <select className="form-select" style={{ maxWidth: '180px' }} value={sortBy} onChange={e => setSortBy(e.target.value)}>
              <option value="avg_score">Sort: Avg Score ↓</option>
              <option value="latest_score">Sort: Latest Score ↓</option>
              <option value="assessments">Sort: # Assessments ↓</option>
              <option value="trend">Sort: Trend</option>
              <option value="patient_id">Sort: Patient ID</option>
            </select>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-hover table-bordered">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Assessments</th>
                  <th>Avg Score</th>
                  <th>Latest Score</th>
                  <th>Severity</th>
                  <th>Trend</th>
                  <th>First Date</th>
                  <th>Latest Date</th>
                </tr>
              </thead>
              <tbody>
                {filteredPatients.map(p => (
                  <tr key={p.patient_id}>
                    <td><span className="badge bg-secondary">{p.patient_id}</span></td>
                    <td className="text-center">{p.assessments}</td>
                    <td>
                      <div className="d-flex align-items-center gap-2">
                        <div className="progress flex-grow-1" style={{ height: '10px', minWidth: '50px' }}>
                          <div
                            className={`progress-bar bg-${severityColor(p.latest_level)}`}
                            style={{ width: `${((p.avg_score || 0) / 80) * 100}%` }}
                          />
                        </div>
                        <span className="small">{p.avg_score?.toFixed(1)}</span>
                      </div>
                    </td>
                    <td className="text-center">{p.latest_score}</td>
                    <td>
                      <span className={`badge bg-${severityColor(p.latest_level)}`}>
                        {p.latest_level}
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${trendColor(p.trend)} text-capitalize`}>
                        {trendIcon(p.trend)}
                      </span>
                    </td>
                    <td className="small text-muted">{p.first_date ? p.first_date.slice(0, 10) : '—'}</td>
                    <td className="small text-muted">{p.latest_date ? p.latest_date.slice(0, 10) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-muted small">{filteredPatients.length} of {patients.length} patients shown</div>
          </div>
        </div>
      )}

      {/* ── ITEM ANALYSIS ── */}
      {tab === 'items' && (
        <div>
          <div className="row">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-body">
                  <h6>Ictal Subscale (Items 1–10)</h6>
                  <p className="text-muted small mb-2">Seizure characteristics during the ictus</p>
                  {ictalItems.map(({ item, avg_score }) => {
                    const label = defs?.item_labels?.[item] || item;
                    const pct = ((avg_score / 4) * 100).toFixed(0);
                    return (
                      <div key={item} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>{label}</span>
                          <span className="text-muted">{avg_score?.toFixed(2)} / 4</span>
                        </div>
                        <div className="progress" style={{ height: '10px' }}>
                          <div
                            className={`progress-bar ${avg_score >= 3 ? 'bg-danger' : avg_score >= 2 ? 'bg-warning' : 'bg-success'}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-body">
                  <h6>Post-ictal Subscale (Items 11–20)</h6>
                  <p className="text-muted small mb-2">Post-ictal burden and recovery</p>
                  {postIctalItems.map(({ item, avg_score }) => {
                    const label = defs?.item_labels?.[item] || item;
                    const pct = ((avg_score / 4) * 100).toFixed(0);
                    return (
                      <div key={item} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>{label}</span>
                          <span className="text-muted">{avg_score?.toFixed(2)} / 4</span>
                        </div>
                        <div className="progress" style={{ height: '10px' }}>
                          <div
                            className={`progress-bar ${avg_score >= 3 ? 'bg-danger' : avg_score >= 2 ? 'bg-warning' : 'bg-success'}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── ASSESSMENT LOG ── */}
      {tab === 'log' && (
        <div>
          <h6 className="mb-3">Recent Assessments ({assessmentLog.length} records)</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-bordered">
              <thead className="table-dark">
                <tr>
                  <th>ID</th>
                  <th>Patient</th>
                  <th>Score</th>
                  <th>Max</th>
                  <th>Severity</th>
                  <th>Interpretation</th>
                  <th>Examiner</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {assessmentLog.map(a => (
                  <tr key={a.id}>
                    <td className="small text-muted">{a.id}</td>
                    <td><span className="badge bg-secondary">{a.patient_id}</span></td>
                    <td className="fw-bold">{a.score}</td>
                    <td className="text-muted small">{a.max_score}</td>
                    <td>
                      <span className={`badge bg-${severityColor(a.level)}`}>
                        {a.level}
                      </span>
                    </td>
                    <td className="small">{a.interpretation}</td>
                    <td className="small text-muted">{a.examiner}</td>
                    <td className="small text-muted">{a.date ? a.date.slice(0, 10) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6>{defs.title}</h6>
              <p className="text-muted small">{defs.description}</p>
              <div className="d-flex gap-3 small">
                <span>Score range: <strong>{defs.score_range?.min}–{defs.score_range?.max}</strong></span>
                <span className="text-danger">Higher = More Severe</span>
              </div>
            </div>
          </div>

          <div className="row mb-3">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Severity Thresholds</h6>
                  {(defs.severity_thresholds || []).map(({ level, min, max, description }) => (
                    <div key={level} className="mb-3 pb-2 border-bottom">
                      <div className="d-flex align-items-center gap-2 mb-1">
                        <span className={`badge bg-${severityColor(level)}`}>{level}</span>
                        <span className="small text-muted">{min}–{max}</span>
                      </div>
                      <div className="text-muted small">{description}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6>Subscales</h6>
                  {(defs.subscales || []).map(({ name, items, description }) => (
                    <div key={name} className="mb-3 pb-2 border-bottom">
                      <div className="fw-semibold small mb-1">{name} Subscale ({items?.length} items)</div>
                      <div className="text-muted small">{description}</div>
                    </div>
                  ))}
                  <h6 className="mt-3">Clinical Uses</h6>
                  {(defs.clinical_use || []).map((use, i) => (
                    <div key={i} className="small text-muted mb-1">• {use}</div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6>All 20 Items</h6>
              <div className="row">
                {Object.entries(defs.item_labels || {})
                  .sort(([a], [b]) => parseInt(a.replace('item', '')) - parseInt(b.replace('item', '')))
                  .map(([key, label]) => {
                    const n = parseInt(key.replace('item', ''), 10);
                    const subscale = n <= 10 ? 'Ictal' : 'Post-ictal';
                    return (
                      <div key={key} className="col-md-6 mb-1">
                        <div className="d-flex align-items-center gap-2 small">
                          <span className={`badge bg-${n <= 10 ? 'primary' : 'secondary'}`}>{key}</span>
                          <span>{label}</span>
                          <span className="text-muted">({subscale})</span>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>

          {defs.references && defs.references.length > 0 && (
            <div className="card shadow-sm">
              <div className="card-body">
                <h6>References</h6>
                {defs.references.map((ref, i) => (
                  <div key={i} className="small text-muted mb-1">• {ref}</div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
