'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const levelColor = l =>
  l === 'independent'          ? 'success' :
  l === 'modified_independent' ? 'primary' :
  l === 'low_moderate'         ? 'warning' :
  l === 'moderate_dependence'  ? 'warning' :
  l === 'total_dependence'     ? 'danger'  : 'secondary';

const pctColor = p =>
  p >= 85 ? 'success' : p >= 70 ? 'primary' : p >= 50 ? 'warning' : 'danger';

const domainBadge = d =>
  d === 'motor' ? 'primary' : 'info';

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

function BarRow({ label, value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span>{value.toFixed(1)} / {max} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 14 }}>
        <div
          className={`progress-bar bg-${pctColor(pct)}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export default function FimDashboardPage() {
  const [ov,  setOv]  = useState(null);
  const [bk,  setBk]  = useState(null);
  const [df,  setDf]  = useState(null);
  const [tab, setTab] = useState('overview');
  const [sel, setSel] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/fim-dashboard/overview`).then(r => r.json()),
      fetch(`${API}/api/fim-dashboard/breakdown`).then(r => r.json()),
      fetch(`${API}/api/fim-dashboard/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov) return <div className="text-center mt-5 text-muted">Loading FIM Dashboard…</div>;

  const dist = ov.independence_distribution || {};
  const distEntries = [
    { key: 'independent',          label: 'Independent',           color: 'success'   },
    { key: 'modified_independent', label: 'Modified Independent',  color: 'primary'   },
    { key: 'low_moderate',         label: 'Low-Moderate Depend.',  color: 'warning'   },
    { key: 'moderate_dependence',  label: 'Moderate Dependence',   color: 'warning'   },
    { key: 'total_dependence',     label: 'Total Dependence',      color: 'danger'    },
  ];
  const totalDist = Object.values(dist).reduce((a, b) => a + b, 0) || 1;

  const TABS = ['overview', 'breakdown', 'patients', 'definitions'];

  return (
    <div className="container-fluid p-4">
      <h2 className="mb-1">🏋️ FIM — Functional Independence Measure</h2>
      <p className="text-muted small mb-3">
        18-item standardised measure of functional independence (motor + cognitive) — 23 patients assessed
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
            <div className="col-6 col-md-2"><KpiCard label="Total Assessments"  value={ov.total_assessments}            color="primary" /></div>
            <div className="col-6 col-md-2"><KpiCard label="Unique Patients"    value={ov.unique_patients}              color="info" /></div>
            <div className="col-6 col-md-2"><KpiCard label="Avg FIM Total"      value={ov.avg_total?.toFixed(1)}        color="success"   sub="18–126 range" /></div>
            <div className="col-6 col-md-2"><KpiCard label="Avg Motor"          value={ov.avg_motor?.toFixed(1)}        color="primary"   sub="13–91 range" /></div>
            <div className="col-6 col-md-2"><KpiCard label="Avg Cognitive"      value={ov.avg_cognitive?.toFixed(1)}    color="info"      sub="5–35 range" /></div>
            <div className="col-6 col-md-2"><KpiCard label="Min / Max Total"    value={`${ov.min_total} / ${ov.max_total}`} color="secondary" /></div>
          </div>

          <div className="row g-3">
            {/* Independence Distribution */}
            <div className="col-md-5">
              <div className="card h-100">
                <div className="card-header bg-primary text-white py-2 small fw-bold">
                  Independence Level Distribution
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
                </div>
              </div>
            </div>

            {/* Top 5 Patients (lowest FIM = most need) */}
            <div className="col-md-7">
              <div className="card h-100">
                <div className="card-header bg-warning text-dark py-2 small fw-bold">
                  Highest Need Patients (lowest FIM Total)
                </div>
                <div className="card-body p-0">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Patient</th><th>Age</th><th>Total</th><th>Motor</th><th>Cognitive</th><th>Level</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(ov.patient_summary || []).slice(0, 8).map(p => (
                        <tr
                          key={p.patient_id}
                          className={sel === p.patient_id ? 'table-active' : ''}
                          style={{ cursor: 'pointer' }}
                          onClick={() => setSel(sel === p.patient_id ? null : p.patient_id)}
                        >
                          <td><span className="fw-semibold small">{p.patient_id}</span></td>
                          <td className="small">{p.age}</td>
                          <td>
                            <span className={`badge bg-${levelColor(p.level)}`}>{p.total}</span>
                          </td>
                          <td className="small">{p.motor}</td>
                          <td className="small">{p.cognitive}</td>
                          <td className="small text-muted">{p.interpretation}</td>
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
          {/* Subdomain bars */}
          <div className="row g-3 mb-3">
            {['motor', 'cognitive'].map(dom => {
              const subs = (bk.subdomain_summary || []).filter(s => s.domain === dom);
              return (
                <div key={dom} className="col-md-6">
                  <div className="card h-100">
                    <div className={`card-header bg-${domainBadge(dom)} text-white py-2 small fw-bold`}>
                      {dom.charAt(0).toUpperCase() + dom.slice(1)} Subdomains
                    </div>
                    <div className="card-body">
                      {subs.map(s => (
                        <BarRow key={s.subdomain} label={s.label} value={s.avg_score} max={s.max_score} />
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Item Heatmap */}
          <div className="card">
            <div className="card-header bg-secondary text-white py-2 small fw-bold">
              Item-Level Heatmap (avg score / max 7)
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Item</th><th>Domain</th><th>Subdomain</th><th>Avg Score</th><th>/ Max</th><th>%</th>
                  </tr>
                </thead>
                <tbody>
                  {(bk.item_heatmap || []).map(item => {
                    const pct = item.max_score > 0 ? Math.round((item.avg_score / item.max_score) * 100) : 0;
                    return (
                      <tr key={item.id}>
                        <td className="fw-semibold small">{item.label}</td>
                        <td><span className={`badge bg-${domainBadge(item.domain)}`}>{item.domain}</span></td>
                        <td className="small text-muted">{item.subdomain?.replace(/_/g, ' ')}</td>
                        <td>{item.avg_score.toFixed(1)}</td>
                        <td className="small text-muted">{item.max_score}</td>
                        <td>
                          <div className="progress" style={{ height: 10, width: 80 }}>
                            <div
                              className={`progress-bar bg-${pctColor(pct)}`}
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
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
            All Patients — FIM Scores
          </div>
          <div className="card-body p-0">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Patient ID</th><th>Name</th><th>Age</th><th>Gender</th>
                  <th>Total</th><th>Motor</th><th>Cognitive</th><th>Level</th><th>Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {(ov.patient_summary || []).map(p => (
                  <tr key={p.patient_id}>
                    <td className="fw-semibold small">{p.patient_id}</td>
                    <td className="small">{p.name}</td>
                    <td className="small">{p.age}</td>
                    <td className="small">{p.gender}</td>
                    <td><span className={`badge bg-${levelColor(p.level)}`}>{p.total}</span></td>
                    <td className="small">{p.motor}</td>
                    <td className="small">{p.cognitive}</td>
                    <td><span className={`badge bg-${levelColor(p.level)} text-capitalize small`}>{p.level?.replace(/_/g, ' ')}</span></td>
                    <td className="small text-muted">{p.interpretation}</td>
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
                <tr><th style={{ width: '30%' }}>Term</th><th>Definition</th></tr>
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
