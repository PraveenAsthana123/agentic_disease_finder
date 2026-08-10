'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const levelColor = l =>
  l === 'high'      ? 'success' :
  l === 'moderate'  ? 'primary' :
  l === 'low'       ? 'warning' :
  l === 'very_low'  ? 'danger'  : 'secondary';

const changeColor = c =>
  c >= 2  ? 'success' :
  c >= 1  ? 'primary' :
  c >= 0  ? 'secondary' : 'danger';

const domainColor = d =>
  d === 'self_care'    ? 'primary' :
  d === 'productivity' ? 'success' :
  d === 'leisure'      ? 'info'    : 'secondary';

function KpiCard({ label, value, unit = '', sub = '', color = 'primary' }) {
  return (
    <div className={`card border-${color} mb-3`}>
      <div className="card-body py-2 px-3">
        <div className={`fw-bold text-${color} fs-5`}>
          {value}{unit && <small className="fs-6 ms-1">{unit}</small>}
        </div>
        <div className="text-muted small">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: '0.72rem' }}>{sub}</div>}
      </div>
    </div>
  );
}

function ScoreBar({ label, value, max = 10, color = 'primary' }) {
  const pct = Math.round((value / max) * 100);
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className={`text-${color} fw-semibold`}>{value.toFixed(1)} / {max}</span>
      </div>
      <div className="progress" style={{ height: 14 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function CopmDashboardPage() {
  const [ov,  setOv]  = useState(null);
  const [bk,  setBk]  = useState(null);
  const [df,  setDf]  = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/copm-dashboard/overview`).then(r => r.json()),
      fetch(`${API}/api/copm-dashboard/breakdown`).then(r => r.json()),
      fetch(`${API}/api/copm-dashboard/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov) return <div className="text-center mt-5 text-muted">Loading COPM Dashboard…</div>;

  const distEntries = [
    { key: 'high',     label: 'High Performance',    color: 'success' },
    { key: 'moderate', label: 'Moderate Performance', color: 'primary' },
    { key: 'low',      label: 'Low Performance',      color: 'warning' },
    { key: 'very_low', label: 'Very Low Performance', color: 'danger'  },
  ];
  const dist = ov.performance_distribution || {};
  const totalDist = Object.values(dist).reduce((a, b) => a + b, 0) || 1;

  const TABS = ['overview', 'breakdown', 'patients', 'definitions'];

  return (
    <div className="container-fluid p-4">
      <h2 className="mb-1">🎯 COPM — Canadian Occupational Performance Measure</h2>
      <p className="text-muted small mb-3">
        Client-centred outcome measure — self-perceived occupational performance &amp; satisfaction (1–10 scale)
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active' : ''}`}
              onClick={() => setTab(t)}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          <div className="row g-3 mb-3">
            <div className="col-6 col-md-2">
              <KpiCard label="Total Assessments" value={ov.total_assessments} color="primary" />
            </div>
            <div className="col-6 col-md-2">
              <KpiCard label="Unique Patients" value={ov.unique_patients} color="info" />
            </div>
            <div className="col-6 col-md-2">
              <KpiCard label="Avg Performance" value={ov.avg_performance?.toFixed(1)} color="success" sub="1–10 scale" />
            </div>
            <div className="col-6 col-md-2">
              <KpiCard label="Avg Satisfaction" value={ov.avg_satisfaction?.toFixed(1)} color="primary" sub="1–10 scale" />
            </div>
            <div className="col-6 col-md-2">
              <KpiCard label="Avg Perf. Change" value={`+${ov.avg_perf_change?.toFixed(1)}`} color="success" sub="initial→reassess" />
            </div>
            <div className="col-6 col-md-2">
              <KpiCard label="Clinically Significant" value={`${ov.clinically_significant_count} (${ov.clinically_significant_pct?.toFixed(0)}%)`} color="warning" sub="≥2pt change" />
            </div>
          </div>

          <div className="row g-3">
            {/* Performance Distribution */}
            <div className="col-md-5">
              <div className="card h-100">
                <div className="card-header bg-primary text-white py-2 small fw-bold">
                  Performance Level Distribution
                </div>
                <div className="card-body">
                  {distEntries.map(e => {
                    const count = dist[e.key] || 0;
                    const pct   = Math.round((count / totalDist) * 100);
                    return (
                      <div key={e.key} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>{e.label}</span>
                          <span>{count} ({pct}%)</span>
                        </div>
                        <div className="progress" style={{ height: 14 }}>
                          <div className={`progress-bar bg-${e.color}`} style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                  <div className="alert alert-info py-2 mt-3 small mb-0">
                    <strong>Clinically significant change</strong> = ≥2 points improvement on performance or satisfaction between assessments.
                  </div>
                </div>
              </div>
            </div>

            {/* Top clinically significant patients */}
            <div className="col-md-7">
              <div className="card h-100">
                <div className="card-header bg-success text-white py-2 small fw-bold">
                  Patient Summary (sorted by performance)
                </div>
                <div className="card-body p-0">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Patient</th>
                        <th>Perf Initial</th><th>Perf Re-assess</th>
                        <th>Δ Perf</th><th>Δ Sat</th>
                        <th>Significant</th><th>Level</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(ov.patient_summary || []).slice(0, 8).map(p => (
                        <tr key={p.patient_id}>
                          <td className="fw-semibold small">{p.patient_id}</td>
                          <td className="small">{p.perf_initial?.toFixed(1)}</td>
                          <td className="small">{p.perf_reassess?.toFixed(1)}</td>
                          <td>
                            <span className={`badge bg-${changeColor(p.perf_change)}`}>
                              +{p.perf_change?.toFixed(1)}
                            </span>
                          </td>
                          <td>
                            <span className={`badge bg-${changeColor(p.sat_change)}`}>
                              +{p.sat_change?.toFixed(1)}
                            </span>
                          </td>
                          <td>
                            {p.significant
                              ? <span className="badge bg-success">Yes</span>
                              : <span className="badge bg-secondary">No</span>}
                          </td>
                          <td>
                            <span className={`badge bg-${levelColor(p.level)} small`}>
                              {p.level?.replace(/_/g, ' ')}
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
        </>
      )}

      {/* ── BREAKDOWN ── */}
      {tab === 'breakdown' && bk && (
        <>
          {/* Domain Summary */}
          <div className="card mb-3">
            <div className="card-header bg-primary text-white py-2 small fw-bold">
              Domain Summary — Performance &amp; Satisfaction
            </div>
            <div className="card-body">
              <div className="row g-3">
                {(bk.domain_summary || []).map(d => (
                  <div key={d.domain} className="col-md-4">
                    <div className={`card border-${domainColor(d.domain)} h-100`}>
                      <div className={`card-header bg-${domainColor(d.domain)} text-white py-2 small fw-bold`}>
                        {d.label}
                      </div>
                      <div className="card-body">
                        <ScoreBar label="Avg Performance" value={d.avg_performance} color={domainColor(d.domain)} />
                        <ScoreBar label="Avg Satisfaction" value={d.avg_satisfaction} color="secondary" />
                        <div className="small mt-2 text-muted">
                          Avg Change: <span className="text-success fw-semibold">+{d.avg_change?.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Problem Heatmap */}
          <div className="card">
            <div className="card-header bg-secondary text-white py-2 small fw-bold">
              Problem Heatmap — All Identified Occupational Problems
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Problem</th><th>Domain</th>
                    <th>Avg Perf</th><th>Avg Sat</th><th>Avg Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.problem_heatmap || []).map(item => (
                    <tr key={item.id}>
                      <td className="small">{item.label}</td>
                      <td>
                        <span className={`badge bg-${domainColor(item.domain)}`}>
                          {item.domain_label}
                        </span>
                      </td>
                      <td>
                        <div className="d-flex align-items-center gap-2">
                          <span>{item.avg_performance?.toFixed(1)}</span>
                          <div className="progress flex-grow-1" style={{ height: 10 }}>
                            <div
                              className={`progress-bar bg-${domainColor(item.domain)}`}
                              style={{ width: `${(item.avg_performance / 10) * 100}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="small text-muted">{item.avg_satisfaction?.toFixed(1)}</td>
                      <td>
                        <span className={`badge bg-${changeColor(item.avg_change)}`}>
                          +{item.avg_change?.toFixed(1)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── PATIENTS ── */}
      {tab === 'patients' && ov && (
        <div className="card">
          <div className="card-header bg-primary text-white py-2 small fw-bold">
            All Patients — COPM Scores
          </div>
          <div className="card-body p-0">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Patient ID</th><th>Name</th><th>Age</th><th>Gender</th>
                  <th>Perf Init</th><th>Sat Init</th>
                  <th>Perf Re-assess</th><th>Sat Re-assess</th>
                  <th>Δ Perf</th><th>Δ Sat</th>
                  <th>Significant</th><th>Level</th>
                </tr>
              </thead>
              <tbody>
                {(ov.patient_summary || []).map(p => (
                  <tr key={p.patient_id}>
                    <td className="fw-semibold small">{p.patient_id}</td>
                    <td className="small">{p.name}</td>
                    <td className="small">{p.age}</td>
                    <td className="small">{p.gender}</td>
                    <td className="small">{p.perf_initial?.toFixed(1)}</td>
                    <td className="small">{p.sat_initial?.toFixed(1)}</td>
                    <td className="small">{p.perf_reassess?.toFixed(1)}</td>
                    <td className="small">{p.sat_reassess?.toFixed(1)}</td>
                    <td>
                      <span className={`badge bg-${changeColor(p.perf_change)}`}>+{p.perf_change?.toFixed(1)}</span>
                    </td>
                    <td>
                      <span className={`badge bg-${changeColor(p.sat_change)}`}>+{p.sat_change?.toFixed(1)}</span>
                    </td>
                    <td>
                      {p.significant
                        ? <span className="badge bg-success">Yes</span>
                        : <span className="badge bg-secondary">No</span>}
                    </td>
                    <td>
                      <span className={`badge bg-${levelColor(p.level)} small`}>
                        {p.level?.replace(/_/g, ' ')}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && df && (
        <div className="card">
          <div className="card-header bg-secondary text-white py-2 small fw-bold">
            {df.title}
          </div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr><th style={{ width: '32%' }}>Term</th><th>Definition</th></tr>
              </thead>
              <tbody>
                {(df.definitions || []).map((d, i) => (
                  <tr key={i}>
                    <td className="fw-semibold small">{d.term}</td>
                    <td className="small text-muted">{d.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
