'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const scoreColor = s =>
  s >= 90 ? 'success' : s >= 80 ? 'primary' : s >= 70 ? 'warning' : 'danger';

const scoreLabel = s =>
  s >= 90 ? 'Excellent' : s >= 80 ? 'Good' : s >= 70 ? 'Needs Work' : 'Critical';

const statusBadge = status => ({
  excellent: 'success',
  good: 'primary',
  needs_work: 'warning',
  critical: 'danger',
}[status] || 'secondary');

function ScoreGauge({ score, size = 80 }) {
  const r = size / 2 - 8;
  const circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  const color = score >= 90 ? '#22c55e' : score >= 80 ? '#3b82f6' : score >= 70 ? '#f59e0b' : '#ef4444';
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#e5e7eb" strokeWidth="8"/>
      <circle
        cx={size/2} cy={size/2} r={r} fill="none"
        stroke={color} strokeWidth="8"
        strokeDasharray={`${dash} ${circ}`}
        strokeLinecap="round"
        transform={`rotate(-90 ${size/2} ${size/2})`}
      />
      <text x="50%" y="50%" dominantBaseline="middle" textAnchor="middle"
        fontSize={size * 0.2} fontWeight="bold" fill={color}>
        {score.toFixed(0)}
      </text>
    </svg>
  );
}

export default function ResponsibleAIDashboard() {
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [selFw, setSelFw] = useState(null);
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/responsible-ai-dashboard/overview`).then(r => r.json()),
      fetch(`${API}/api/responsible-ai-dashboard/breakdown`).then(r => r.json()),
      fetch(`${API}/api/responsible-ai-dashboard/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err)  return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-3">Loading Responsible AI Dashboard…</div>;

  const frameworks = ov.framework_cards || [];
  const details    = bk?.framework_details || [];
  const excellent  = frameworks.filter(f => f.score >= 90).length;
  const good       = frameworks.filter(f => f.score >= 80 && f.score < 90).length;
  const needsWork  = frameworks.filter(f => f.score < 80).length;

  const selDetail = selFw ? details.find(d => d.id === selFw) : null;

  const TABS = [
    { id: 'overview',    label: '📊 Overview' },
    { id: 'frameworks',  label: '🧩 Frameworks' },
    { id: 'breakdown',   label: '🔍 Breakdown' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-4 flex-wrap">
        <div>
          <h2 className="mb-0">⚖️ Responsible AI Dashboard</h2>
          <small className="text-muted">
            {ov.applicable_frameworks}/{ov.total_frameworks} frameworks applicable · analysis {ov.analysis_date}
          </small>
        </div>
        <div className="ms-auto d-flex align-items-center gap-3">
          <ScoreGauge score={ov.overall_score} size={72} />
          <div>
            <div className="fw-bold fs-5">{ov.overall_score.toFixed(1)}/100</div>
            <span className={`badge bg-${scoreColor(ov.overall_score)}`}>
              {scoreLabel(ov.overall_score)}
            </span>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="row g-3 mb-4">
        {[
          { label: 'Overall Score',       value: `${ov.overall_score.toFixed(1)}%`, icon: '🎯', color: scoreColor(ov.overall_score) },
          { label: 'Total Frameworks',    value: ov.total_frameworks,  icon: '🧩', color: 'dark' },
          { label: 'Applicable',          value: ov.applicable_frameworks, icon: '✅', color: 'primary' },
          { label: 'Excellent (≥90)',      value: excellent, icon: '⭐', color: 'success' },
          { label: 'Good (80–89)',         value: good,      icon: '👍', color: 'info' },
          { label: 'Needs Attention (<80)',value: needsWork,  icon: '⚠️', color: 'warning' },
        ].map(k => (
          <div key={k.label} className="col-6 col-md-4 col-xl-2">
            <div className={`card border-${k.color} h-100 shadow-sm`}>
              <div className="card-body text-center py-3">
                <div style={{ fontSize: '1.6rem' }}>{k.icon}</div>
                <div className={`fw-bold fs-5 text-${k.color}`}>{k.value}</div>
                <small className="text-muted">{k.label}</small>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
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

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div>
          <h5 className="mb-3">Score Distribution</h5>
          <div className="row g-2">
            {frameworks.map(f => (
              <div key={f.id} className="col-12 col-md-6 col-xl-4">
                <div
                  className="card shadow-sm h-100"
                  style={{ cursor: 'pointer', borderLeft: `4px solid ${f.score >= 90 ? '#22c55e' : f.score >= 80 ? '#3b82f6' : f.score >= 70 ? '#f59e0b' : '#ef4444'}` }}
                  onClick={() => { setTab('breakdown'); setSelFw(f.id); }}
                >
                  <div className="card-body py-2 px-3 d-flex align-items-center gap-3">
                    <ScoreGauge score={f.score} size={52} />
                    <div className="flex-grow-1">
                      <div className="fw-semibold small">{f.label.replace(/_/g,' ')}</div>
                      <span className={`badge bg-${statusBadge(f.status)} small`}>{f.status}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Frameworks Tab — sortable table */}
      {tab === 'frameworks' && (
        <div className="table-responsive">
          <table className="table table-hover table-sm align-middle">
            <thead className="table-dark">
              <tr>
                <th>#</th>
                <th>Framework</th>
                <th>Score</th>
                <th>Status</th>
                <th>Bar</th>
              </tr>
            </thead>
            <tbody>
              {[...frameworks].sort((a,b) => b.score - a.score).map((f, i) => (
                <tr key={f.id} style={{ cursor: 'pointer' }}
                    onClick={() => { setTab('breakdown'); setSelFw(f.id); }}>
                  <td className="text-muted small">{i + 1}</td>
                  <td className="fw-semibold">{f.label.replace(/_/g,' ')}</td>
                  <td>
                    <span className={`badge bg-${scoreColor(f.score)}`}>{f.score.toFixed(1)}</span>
                  </td>
                  <td>
                    <span className={`badge bg-${statusBadge(f.status)} text-capitalize`}>{f.status}</span>
                  </td>
                  <td style={{ width: 160 }}>
                    <div className="progress" style={{ height: 8 }}>
                      <div
                        className={`progress-bar bg-${scoreColor(f.score)}`}
                        style={{ width: `${f.score}%` }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Breakdown Tab */}
      {tab === 'breakdown' && (
        <div className="row g-3">
          <div className="col-md-3">
            <div className="list-group list-group-flush border rounded" style={{ maxHeight: 600, overflowY: 'auto' }}>
              {details.map(d => (
                <button
                  key={d.id}
                  className={`list-group-item list-group-item-action py-2 d-flex justify-content-between align-items-center${selFw === d.id ? ' active' : ''}`}
                  onClick={() => setSelFw(d.id)}
                >
                  <span className="small">{d.label.replace(/_/g,' ')}</span>
                  <span className={`badge bg-${scoreColor(d.score)}`}>{d.score.toFixed(0)}</span>
                </button>
              ))}
            </div>
          </div>
          <div className="col-md-9">
            {selDetail ? (
              <div>
                <div className="d-flex align-items-center gap-3 mb-3">
                  <ScoreGauge score={selDetail.score} size={80} />
                  <div>
                    <h5 className="mb-1">{selDetail.label.replace(/_/g,' ')}</h5>
                    <span className={`badge bg-${statusBadge(selDetail.status)}`}>{selDetail.status}</span>
                  </div>
                </div>
                {selDetail.analyses?.length > 0 ? (
                  <div className="table-responsive">
                    <table className="table table-sm table-striped align-middle">
                      <thead className="table-dark">
                        <tr>
                          <th>Analysis</th>
                          <th>Score</th>
                          <th>Method</th>
                          <th>Justification</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selDetail.analyses.map(a => (
                          <tr key={a.id}>
                            <td className="fw-semibold small">{a.label.replace(/_/g,' ')}</td>
                            <td>
                              <span className={`badge bg-${scoreColor(a.score)}`}>{a.score.toFixed(1)}</span>
                            </td>
                            <td className="small text-muted">{a.method || '—'}</td>
                            <td className="small">{a.justification || '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-muted">No sub-analyses available for this framework.</p>
                )}
              </div>
            ) : (
              <div className="text-muted p-4 text-center">
                ← Select a framework to view its detailed breakdown
              </div>
            )}
          </div>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          {Object.entries(defs).map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body py-2">
                  <div className="fw-semibold text-primary small mb-1">
                    {k.replace(/_/g,' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </div>
                  <p className="small text-muted mb-0">{v}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
