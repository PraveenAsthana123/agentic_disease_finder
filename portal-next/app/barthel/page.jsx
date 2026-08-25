'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const levelColor = lvl => {
  const l = (lvl || '').toLowerCase();
  if (l.includes('very') || l.includes('severe')) return 'danger';
  if (l.includes('partial') || l.includes('moderate')) return 'warning';
  if (l.includes('minimal') || l.includes('mild')) return 'info';
  if (l.includes('independent') || l.includes('normal')) return 'success';
  return 'secondary';
};

const trendIcon = t => ({ improving: '↑ Improving', worsening: '↓ Worsening', stable: '→ Stable' }[t] || t);
const trendColor = t => ({ improving: 'success', worsening: 'danger', stable: 'secondary' }[t] || 'secondary');

export default function BarthelDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('avg_score');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/barthel/overview`).then(r => r.json()),
      fetch(`${API}/api/barthel/breakdown`).then(r => r.json()),
      fetch(`${API}/api/barthel/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading Barthel ADL data...</div>;

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

  const indepDist = ov.independence_distribution || {};
  const totalAssessments = ov.total_assessments || 0;

  return (
    <div className="p-3">
      <h3>🦽 Barthel Index (ADL) Dashboard</h3>
      <p className="text-muted">
        Activities of Daily Living — {ov.total_assessments} assessments · {ov.unique_patients} patients ·
        avg score {ov.avg_score?.toFixed(1)} / 100 · {ov.independent_patient_count} independent · {ov.dependent_patient_count} needing support
      </p>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ─────────────────────────────────────────── */}
      {tab === 'overview' && (
        <div>
          {/* KPI cards */}
          <div className="row g-3 mb-4">
            {[
              { label: 'Total Assessments', value: ov.total_assessments, color: 'primary' },
              { label: 'Unique Patients', value: ov.unique_patients, color: 'info' },
              { label: 'Avg Score / 100', value: ov.avg_score?.toFixed(1), color: 'success' },
              { label: 'Min Score', value: ov.min_score, color: 'warning' },
              { label: 'Max Score', value: ov.max_score, color: 'success' },
              { label: 'Fully Independent', value: ov.independent_patient_count, color: 'success' },
            ].map(k => (
              <div className="col-6 col-md-4 col-lg-2" key={k.label}>
                <div className={`card border-${k.color} h-100`}>
                  <div className="card-body text-center p-2">
                    <div className={`fs-4 fw-bold text-${k.color}`}>{k.value ?? '—'}</div>
                    <div className="small text-muted">{k.label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Independence distribution */}
          <div className="row g-3 mb-4">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header">Independence Distribution</div>
                <div className="card-body">
                  {Object.entries(indepDist).map(([lvl, cnt]) => {
                    const pct = totalAssessments ? Math.round((cnt / totalAssessments) * 100) : 0;
                    return (
                      <div key={lvl} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span className={`badge bg-${levelColor(lvl)}`}>{lvl}</span>
                          <span>{cnt} ({pct}%)</span>
                        </div>
                        <div className="progress" style={{ height: 10 }}>
                          <div className={`progress-bar bg-${levelColor(lvl)}`} style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Score histogram */}
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header">Score Distribution (bins of 10)</div>
                <div className="card-body">
                  {(ov.score_histogram || []).map(h => {
                    const pct = totalAssessments ? Math.round((h.count / totalAssessments) * 100) : 0;
                    return (
                      <div key={h.bin} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span className="text-muted">{h.bin}</span>
                          <span>{h.count}</span>
                        </div>
                        <div className="progress" style={{ height: 8 }}>
                          <div className="progress-bar bg-primary" style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* Monthly trend */}
          <div className="card">
            <div className="card-header">Monthly Assessment Trend</div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead><tr><th>Month</th><th>Assessments</th><th>Avg Score</th></tr></thead>
                  <tbody>
                    {(ov.monthly_trend || []).map(m => (
                      <tr key={m.month}>
                        <td>{m.month}</td>
                        <td>{m.count}</td>
                        <td>
                          <span className={`badge bg-${m.avg_score >= 80 ? 'success' : m.avg_score >= 60 ? 'info' : m.avg_score >= 40 ? 'warning' : 'danger'}`}>
                            {m.avg_score}
                          </span>
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

      {/* ── PER PATIENT ───────────────────────────────────────── */}
      {tab === 'patients' && (
        <div>
          <div className="row g-2 mb-3">
            <div className="col-md-6">
              <input className="form-control" placeholder="Search patient ID or level..." value={search} onChange={e => setSearch(e.target.value)} />
            </div>
            <div className="col-md-4">
              <select className="form-select" value={sortBy} onChange={e => setSortBy(e.target.value)}>
                <option value="avg_score">Sort: Avg Score (desc)</option>
                <option value="latest_score">Sort: Latest Score</option>
                <option value="assessments">Sort: # Assessments</option>
                <option value="trend">Sort: Trend</option>
                <option value="patient_id">Sort: Patient ID</option>
              </select>
            </div>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead>
                <tr>
                  <th>Patient</th><th>Assessments</th><th>Avg Score</th>
                  <th>Latest Score</th><th>Level</th><th>Trend</th><th>Last Assessment</th>
                </tr>
              </thead>
              <tbody>
                {filteredPatients.map(p => (
                  <tr key={p.patient_id}>
                    <td><code>{p.patient_id}</code></td>
                    <td>{p.assessments}</td>
                    <td>{p.avg_score?.toFixed(1)}</td>
                    <td>{p.latest_score}</td>
                    <td><span className={`badge bg-${levelColor(p.latest_level)}`}>{p.latest_level}</span></td>
                    <td><span className={`badge bg-${trendColor(p.trend)}`}>{trendIcon(p.trend)}</span></td>
                    <td className="text-muted small">{p.latest_date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── ITEM ANALYSIS ─────────────────────────────────────── */}
      {tab === 'items' && (
        <div>
          <p className="text-muted small">Average score per ADL item across all assessments (higher = more independent).</p>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead><tr><th>#</th><th>ADL Activity</th><th>Avg Score</th><th>N</th><th>Bar</th></tr></thead>
              <tbody>
                {itemAverages.map((it, idx) => {
                  const pct = it.avg != null ? Math.round((it.avg / 15) * 100) : 0;
                  return (
                    <tr key={it.item}>
                      <td className="text-muted">{idx + 1}</td>
                      <td>{it.label}</td>
                      <td>{it.avg != null ? it.avg.toFixed(2) : '—'}</td>
                      <td className="text-muted">{it.n}</td>
                      <td style={{ width: '40%' }}>
                        <div className="progress" style={{ height: 8 }}>
                          <div className="progress-bar bg-success" style={{ width: `${pct}%` }} />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── ASSESSMENT LOG ────────────────────────────────────── */}
      {tab === 'log' && (
        <div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead>
                <tr><th>Date</th><th>Patient</th><th>Score</th><th>Level</th><th>Interpretation</th><th>Examiner</th></tr>
              </thead>
              <tbody>
                {assessmentLog.map(a => (
                  <tr key={a.id}>
                    <td className="text-muted small">{a.date}</td>
                    <td><code>{a.patient_id}</code></td>
                    <td><strong>{a.score}</strong> / {a.max_score || 100}</td>
                    <td><span className={`badge bg-${levelColor(a.level)}`}>{a.level}</span></td>
                    <td className="small">{a.interpretation}</td>
                    <td className="text-muted small">{a.examiner}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ───────────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="card mb-3">
            <div className="card-header"><strong>{defs.full_name}</strong></div>
            <div className="card-body">
              <p>{defs.purpose}</p>
              <dl className="row small">
                <dt className="col-sm-3">Population</dt><dd className="col-sm-9">{defs.population}</dd>
                <dt className="col-sm-3">Administered by</dt><dd className="col-sm-9">{defs.role}</dd>
                <dt className="col-sm-3">Scale</dt><dd className="col-sm-9">0–{defs.scale?.max} (higher = more independent)</dd>
                <dt className="col-sm-3">Admin time</dt><dd className="col-sm-9">{defs.administration?.time}</dd>
                <dt className="col-sm-3">Method</dt><dd className="col-sm-9">{defs.administration?.method}</dd>
              </dl>
              <hr />
              <p className="small"><strong>Epilepsy relevance:</strong> {defs.epilepsy_relevance}</p>
            </div>
          </div>

          {/* Severity bands */}
          <div className="card mb-3">
            <div className="card-header">Independence Bands</div>
            <div className="card-body">
              <table className="table table-sm">
                <thead><tr><th>Score Range</th><th>Level</th><th>Description</th></tr></thead>
                <tbody>
                  {(defs.bands || []).map(b => (
                    <tr key={b.label}>
                      <td>{b.min}–{b.max}</td>
                      <td><span className={`badge bg-${levelColor(b.label)}`}>{b.label}</span></td>
                      <td className="small">{b.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Scoring guide */}
          <div className="card mb-3">
            <div className="card-header">10-Item Scoring Guide</div>
            <div className="card-body">
              {(defs.scoring_guide || []).map(item => (
                <div key={item.item} className="mb-2">
                  <strong className="small">{item.item}</strong>
                  <div className="d-flex flex-wrap gap-1 mt-1">
                    {(item.options || []).map(o => (
                      <span key={o.score} className="badge bg-secondary text-wrap text-start">
                        {o.score}: {o.label}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* References */}
          <div className="card">
            <div className="card-header">References</div>
            <div className="card-body">
              <ol className="small">
                {(defs.references || []).map((r, i) => <li key={i}>{r}</li>)}
              </ol>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
