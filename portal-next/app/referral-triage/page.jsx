'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const urgencyColor = u => ({ emergent: 'danger', urgent: 'warning', routine: 'primary', elective: 'secondary' }[u] || 'secondary');
const statusColor = s => ({ completed: 'success', scheduled: 'info', in_progress: 'primary', pending_triage: 'warning', cancelled: 'secondary', triaged: 'info' }[s] || 'secondary');
const scoreColor = v => v >= 70 ? 'danger' : v >= 40 ? 'warning' : 'success';

const KPI = ({ label, value, sub, color = 'primary' }) => (
  <div className="col-6 col-md-3 mb-3">
    <div className={`card border-${color} h-100`}>
      <div className="card-body p-3 text-center">
        <div className={`fs-3 fw-bold text-${color}`}>{value}</div>
        <div className="small text-muted">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: '0.72rem' }}>{sub}</div>}
      </div>
    </div>
  </div>
);

const Bar = ({ label, value, max, color = 'primary' }) => (
  <div className="mb-2">
    <div className="d-flex justify-content-between small mb-1">
      <span className="text-capitalize">{label.replace(/_/g, ' ')}</span>
      <span className="fw-semibold">{value}</span>
    </div>
    <div className="progress" style={{ height: 10 }}>
      <div className={`progress-bar bg-${color}`} style={{ width: `${Math.round((value / max) * 100)}%` }} />
    </div>
  </div>
);

export default function ReferralTriageDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('triage_score');
  const [sortDir, setSortDir] = useState('desc');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/referral-triage/overview`).then(r => r.json()),
      fetch(`${API}/api/referral-triage/breakdown`).then(r => r.json()),
      fetch(`${API}/api/referral-triage/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading Referral Triage data…</div>;

  const TABS = ['overview', 'reasons', 'providers', 'referrals', 'definitions'];
  const tabLabel = { overview: '📋 Overview', reasons: '🔍 By Reason', providers: '👨‍⚕️ Providers', referrals: '📄 Referrals', definitions: '📖 Definitions' };

  // Sorted/filtered recent_referrals
  const refs = (bd?.recent_referrals || []).filter(r =>
    !search || Object.values(r).join(' ').toLowerCase().includes(search.toLowerCase())
  ).sort((a, b) => {
    const av = a[sortBy] ?? '';
    const bv = b[sortBy] ?? '';
    if (typeof av === 'number') return sortDir === 'asc' ? av - bv : bv - av;
    return sortDir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
  });

  const toggleSort = col => {
    if (sortBy === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortBy(col); setSortDir('desc'); }
  };
  const sortArrow = col => sortBy === col ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '';

  // Urgency distribution max
  const urgencyMax = Math.max(...Object.values(ov.urgency_distribution || {}));
  const sourceMax = Math.max(...Object.values(ov.source_distribution || {}));
  const reasonMax = Math.max(...Object.values(bd?.reason_distribution || {}));

  return (
    <div>
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <h3 className="mb-0">🏥 Referral Triage Dashboard</h3>
        <span className="badge bg-dark fs-6">{ov.total_referrals} referrals</span>
        {ov.urgent_pending > 0 && (
          <span className="badge bg-danger">{ov.urgent_pending} urgent pending</span>
        )}
        <span className="text-muted small ms-auto">Real data · referral_records {ov.total_referrals} rows · clinical.db</span>
      </div>

      {/* KPI Row */}
      <div className="row mb-3">
        <KPI label="Total Referrals" value={ov.total_referrals} color="primary" />
        <KPI label="Pending Triage" value={ov.pending_triage} sub="awaiting review" color="warning" />
        <KPI label="Urgent Pending" value={ov.urgent_pending} sub="high priority" color="danger" />
        <KPI label="Completion Rate" value={`${ov.completion_rate_pct}%`} sub={`Avg ${ov.avg_triage_time_hrs}h turnaround`} color="info" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>
              {tabLabel[t]}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-6 mb-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Urgency Distribution</div>
              <div className="card-body">
                {Object.entries(ov.urgency_distribution || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={urgencyMax} color={urgencyColor(k)} />
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Referral Source Distribution</div>
              <div className="card-body">
                {Object.entries(ov.source_distribution || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={sourceMax} color="info" />
                ))}
              </div>
            </div>
          </div>
          <div className="col-12 mb-4">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Monthly Triage Timeline</div>
              <div className="card-body">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Month</th>
                        <th>Referrals</th>
                        <th>Triaged</th>
                        <th>Triage Rate</th>
                        <th>Progress</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(ov.triage_timeline || []).map(row => {
                        const rate = row.referrals > 0 ? Math.round((row.triaged / row.referrals) * 100) : 0;
                        return (
                          <tr key={row.date}>
                            <td>{row.date}</td>
                            <td>{row.referrals}</td>
                            <td>{row.triaged}</td>
                            <td><span className={`badge bg-${rate >= 90 ? 'success' : rate >= 70 ? 'info' : 'warning'}`}>{rate}%</span></td>
                            <td style={{ minWidth: 100 }}>
                              <div className="progress" style={{ height: 8 }}>
                                <div className={`progress-bar bg-${rate >= 90 ? 'success' : rate >= 70 ? 'info' : 'warning'}`} style={{ width: `${rate}%` }} />
                              </div>
                            </td>
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
      )}

      {/* By Reason Tab */}
      {tab === 'reasons' && (
        <div className="row">
          <div className="col-md-5 mb-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Referral Reason Distribution</div>
              <div className="card-body">
                {Object.entries(bd?.reason_distribution || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={reasonMax} color="primary" />
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-7 mb-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Urgency by Source</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Source</th>
                        {['emergent', 'urgent', 'routine', 'elective'].map(u => (
                          <th key={u}><span className={`badge bg-${urgencyColor(u)}`}>{u}</span></th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(bd?.urgency_by_source || {}).sort((a, b) => a[0].localeCompare(b[0])).map(([src, urgencies]) => (
                        <tr key={src}>
                          <td className="text-capitalize">{src.replace(/_/g, ' ')}</td>
                          {['emergent', 'urgent', 'routine', 'elective'].map(u => (
                            <td key={u} className="text-center">{urgencies[u] ?? '—'}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
          <div className="col-12 mb-4">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Status Summary</div>
              <div className="card-body">
                <div className="d-flex flex-wrap gap-3">
                  {Object.entries(bd?.status_summary || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                    <div key={k} className="text-center">
                      <div className={`badge bg-${statusColor(k)} fs-6 px-3 py-2`}>{v}</div>
                      <div className="small text-muted mt-1 text-capitalize">{k.replace(/_/g, ' ')}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Providers Tab */}
      {tab === 'providers' && (
        <div className="row">
          <div className="col-md-7 mb-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Provider Workload</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Provider</th>
                        <th>Total</th>
                        <th>Completed</th>
                        <th>Active</th>
                        <th>Completion</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd?.provider_workload || []).sort((a, b) => b.total - a.total).map((p, i) => {
                        const rate = p.total > 0 ? Math.round((p.completed / p.total) * 100) : 0;
                        return (
                          <tr key={i}>
                            <td>{p.assigned_to || '—'}</td>
                            <td>{p.total}</td>
                            <td><span className="badge bg-success">{p.completed}</span></td>
                            <td><span className="badge bg-info">{p.active}</span></td>
                            <td>
                              <div className="d-flex align-items-center gap-2">
                                <div className="progress flex-grow-1" style={{ height: 8 }}>
                                  <div className={`progress-bar bg-${rate >= 60 ? 'success' : rate >= 30 ? 'info' : 'warning'}`} style={{ width: `${rate}%` }} />
                                </div>
                                <span className="small">{rate}%</span>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
          <div className="col-md-5 mb-4">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">Turnaround by Urgency (hours)</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Urgency</th><th>Avg Hours</th><th>Count</th></tr></thead>
                  <tbody>
                    {(bd?.turnaround_by_urgency || []).sort((a, b) => a.avg_hours - b.avg_hours).map(r => (
                      <tr key={r.urgency}>
                        <td><span className={`badge bg-${urgencyColor(r.urgency)}`}>{r.urgency}</span></td>
                        <td>{r.avg_hours}h</td>
                        <td>{r.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Triage Score by Source</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Source</th><th>Avg Score</th><th>Count</th></tr></thead>
                  <tbody>
                    {(bd?.score_by_source || []).sort((a, b) => b.avg_score - a.avg_score).map(r => (
                      <tr key={r.source}>
                        <td className="text-capitalize">{r.source.replace(/_/g, ' ')}</td>
                        <td><span className={`badge bg-${scoreColor(r.avg_score)}`}>{r.avg_score}</span></td>
                        <td>{r.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Referrals Tab */}
      {tab === 'referrals' && (
        <div>
          <div className="d-flex gap-2 mb-3 flex-wrap">
            <input className="form-control form-control-sm" style={{ maxWidth: 280 }}
              placeholder="Search patient, source, reason…"
              value={search} onChange={e => setSearch(e.target.value)} />
            <span className="text-muted small align-self-center">{refs.length} of {bd?.recent_referrals?.length ?? 0} shown</span>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-bordered">
              <thead className="table-dark">
                <tr>
                  {[['patient_id', 'Patient'], ['source', 'Source'], ['reason', 'Reason'],
                    ['urgency', 'Urgency'], ['status', 'Status'], ['assigned_to', 'Assigned To'],
                    ['referred_date', 'Referred'], ['triage_score', 'Score']].map(([col, lbl]) => (
                    <th key={col} style={{ cursor: 'pointer', whiteSpace: 'nowrap' }} onClick={() => toggleSort(col)}>
                      {lbl}{sortArrow(col)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {refs.map((r, i) => (
                  <tr key={i}>
                    <td><code>{r.patient_id}</code></td>
                    <td className="text-capitalize">{(r.source || '').replace(/_/g, ' ')}</td>
                    <td className="text-capitalize">{(r.reason || '').replace(/_/g, ' ')}</td>
                    <td><span className={`badge bg-${urgencyColor(r.urgency)}`}>{r.urgency}</span></td>
                    <td><span className={`badge bg-${statusColor(r.status)}`}>{(r.status || '').replace(/_/g, ' ')}</span></td>
                    <td>{r.assigned_to || <span className="text-muted">Unassigned</span>}</td>
                    <td>{r.referred_date}</td>
                    <td>
                      {r.triage_score != null
                        ? <span className={`badge bg-${scoreColor(r.triage_score)}`}>{r.triage_score}</span>
                        : <span className="text-muted">—</span>}
                    </td>
                  </tr>
                ))}
                {refs.length === 0 && (
                  <tr><td colSpan={8} className="text-center text-muted py-3">No referrals match your search.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div className="row">
          {[
            ['Urgency Levels', defs.urgency_levels],
            ['Triage Statuses', defs.triage_statuses],
            ['Referral Sources', defs.referral_sources],
            ['Referral Reasons', defs.referral_reasons],
          ].map(([title, items]) => (
            <div key={title} className="col-md-6 mb-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">{title}</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <tbody>
                      {(items || []).map((item, i) => (
                        <tr key={i}>
                          <td className="fw-semibold text-capitalize" style={{ width: '35%' }}>
                            {(item.level || item.status || item.source || item.reason || '').replace(/_/g, ' ')}
                          </td>
                          <td className="small text-muted">{item.description || item.definition || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
          {defs.field_descriptions && (
            <div className="col-12 mb-4">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Field Descriptions</div>
                <div className="card-body">
                  <div className="row">
                    {Object.entries(defs.field_descriptions).map(([k, v]) => (
                      <div key={k} className="col-md-4 mb-2">
                        <code>{k}</code>
                        <div className="small text-muted">{v}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
          {defs.clinical_notes && (
            <div className="col-12 mb-4">
              <div className="card shadow-sm border-info">
                <div className="card-header bg-info text-white fw-semibold">Clinical Notes</div>
                <ul className="list-group list-group-flush">
                  {defs.clinical_notes.map((n, i) => <li key={i} className="list-group-item small">{n}</li>)}
                </ul>
              </div>
            </div>
          )}
          {defs.glossary && (
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Glossary</div>
                <div className="card-body">
                  <div className="row">
                    {Object.entries(defs.glossary).map(([k, v]) => (
                      <div key={k} className="col-md-4 mb-2">
                        <strong>{k}</strong>
                        <div className="small text-muted">{v}</div>
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
