'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'triage',      label: 'Triage Scores' },
  { id: 'referrals',   label: 'All Referrals' },
  { id: 'patients',    label: 'Per Patient' },
  { id: 'definitions', label: 'Definitions' },
];

const URGENCY_COLOR = { emergent: 'danger', urgent: 'warning', routine: 'info', elective: 'secondary' };
const STATUS_COLOR  = {
  completed: 'success', scheduled: 'primary', in_progress: 'info',
  triaged: 'info', pending_triage: 'warning', cancelled: 'secondary',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-2 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-3">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function BarList({ items, keyField, valueField, colorFn, maxVal }) {
  if (!items?.length) return <p className="text-muted small">No data.</p>;
  const max = maxVal ?? Math.max(...items.map(i => i[valueField]));
  return (
    <table className="table table-sm mb-0">
      <tbody>
        {items.map((item, i) => {
          const pct = max > 0 ? ((item[valueField] / max) * 100).toFixed(0) : 0;
          const col = colorFn ? colorFn(item[keyField]) : 'primary';
          return (
            <tr key={i}>
              <td className="small fw-semibold text-capitalize" style={{ width: '40%' }}>
                {(item[keyField] || '').replace(/_/g, ' ')}
              </td>
              <td style={{ width: '45%' }}>
                <div className="progress" style={{ height: 14 }}>
                  <div className={`progress-bar bg-${col}`} style={{ width: `${pct}%` }}>
                    <span className="small">{item[valueField]}</span>
                  </div>
                </div>
              </td>
              <td className="small text-end text-muted">{item[valueField]}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function Badge({ val, map }) {
  const col = map?.[val] || 'secondary';
  return <span className={`badge bg-${col} text-capitalize`}>{(val || '').replace(/_/g, ' ')}</span>;
}

export default function ReferralRecordsPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [search,  setSearch]  = useState('');
  const [sortBy,  setSortBy]  = useState('total_referrals');
  const [refSearch, setRefSearch] = useState('');
  const [urgFilter, setUrgFilter] = useState('');

  useEffect(() => {
    fetch(`${API}/api/referral-records/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/referral-records/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/referral-records/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return (
    <div className="p-4 text-center">
      <div className="spinner-border text-primary" />
      <div className="mt-2 text-muted">Loading Referral Records…</div>
    </div>
  );

  const referrals = bd?.referrals || [];
  const patients  = bd?.patient_summary || [];
  const bySource  = bd?.by_source || [];

  const filteredRefs = referrals
    .filter(r => {
      const q = refSearch.toLowerCase();
      if (q && !r.patient_id?.toLowerCase().includes(q) && !r.referral_reason?.toLowerCase().includes(q) && !r.referral_source?.toLowerCase().includes(q)) return false;
      if (urgFilter && r.urgency !== urgFilter) return false;
      return true;
    })
    .sort((a, b) => (b.triage_score ?? 0) - (a.triage_score ?? 0));

  const filteredPts = patients
    .filter(p => !search || p.patient_id?.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => (b[sortBy] ?? 0) - (a[sortBy] ?? 0));

  const concepts = defs?.concepts || [];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 d-flex align-items-center gap-2">
        <span style={{ fontSize: '1.4rem' }}>🏥</span>
        <div>
          <h4 className="mb-0 fw-bold">Referral Records</h4>
          <div className="text-muted small">
            {ov.total_referrals} referrals · {ov.total_patients} patients ·
            avg triage {ov.avg_triage_score}/100 · {ov.urgent_emergent_count} urgent/emergent
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
        <>
          <div className="row g-2 mb-4">
            <KPI label="Total Referrals"    value={ov.total_referrals}      color="primary" />
            <KPI label="Unique Patients"    value={ov.total_patients}       color="info" />
            <KPI label="Avg Triage Score"   value={`${ov.avg_triage_score}/100`} color="warning" />
            <KPI label="Urgent / Emergent"  value={ov.urgent_emergent_count} color="danger" />
            <KPI label="Completion Rate"    value={`${ov.completion_rate}%`} color="success" />
            <KPI label="Pending Triage"     value={ov.pending_count}        color="warning" />
          </div>

          <div className="row g-3 mb-4">
            {/* Source distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header bg-primary text-white small fw-semibold">Referral Source</div>
                <div className="card-body p-2">
                  <BarList items={ov.source_distribution} keyField="source" valueField="count" colorFn={() => 'primary'} />
                </div>
              </div>
            </div>

            {/* Reason distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header bg-info text-white small fw-semibold">Referral Reason</div>
                <div className="card-body p-2">
                  <BarList items={ov.reason_distribution} keyField="reason" valueField="count" colorFn={() => 'info'} />
                </div>
              </div>
            </div>
          </div>

          <div className="row g-3 mb-4">
            {/* Urgency */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header bg-danger text-white small fw-semibold">Urgency Level</div>
                <div className="card-body p-2">
                  <BarList items={ov.urgency_distribution} keyField="urgency" valueField="count"
                    colorFn={u => URGENCY_COLOR[u] || 'secondary'} />
                </div>
              </div>
            </div>

            {/* Triage Status */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header bg-secondary text-white small fw-semibold">Triage Status</div>
                <div className="card-body p-2">
                  <BarList items={ov.triage_status_distribution} keyField="status" valueField="count"
                    colorFn={s => STATUS_COLOR[s] || 'secondary'} />
                </div>
              </div>
            </div>

            {/* Assignee Workload */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header bg-dark text-white small fw-semibold">Assigned Clinician</div>
                <div className="card-body p-2">
                  <BarList items={ov.assigned_to_distribution} keyField="assigned_to" valueField="count"
                    colorFn={() => 'dark'} />
                </div>
              </div>
            </div>
          </div>

          {/* Monthly Trend */}
          <div className="card shadow-sm">
            <div className="card-header bg-warning text-dark small fw-semibold">Monthly Referral Trend</div>
            <div className="card-body p-2">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Month</th>
                    <th>Total</th>
                    <th>Urgent / Emergent</th>
                    <th>Urgent Rate</th>
                    <th>Volume Bar</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.monthly_trend || []).map((row, i) => {
                    const maxTotal = Math.max(...(ov.monthly_trend || []).map(r => r.total));
                    const pct = maxTotal > 0 ? ((row.total / maxTotal) * 100).toFixed(0) : 0;
                    const urgRate = row.total > 0 ? ((row.urgent_emergent / row.total) * 100).toFixed(0) : 0;
                    return (
                      <tr key={i}>
                        <td className="small fw-semibold">{row.month}</td>
                        <td className="small">{row.total}</td>
                        <td className="small">
                          <span className="badge bg-danger">{row.urgent_emergent}</span>
                        </td>
                        <td className="small">{urgRate}%</td>
                        <td style={{ width: '35%' }}>
                          <div className="progress" style={{ height: 12 }}>
                            <div className="progress-bar bg-warning" style={{ width: `${pct}%` }}>
                              <span className="small">{row.total}</span>
                            </div>
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

      {/* ── TRIAGE SCORES ── */}
      {tab === 'triage' && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header bg-danger text-white small fw-semibold">Avg Triage Score by Urgency</div>
              <div className="card-body p-2">
                <BarList items={ov.avg_triage_score_by_urgency} keyField="urgency" valueField="avg_score"
                  colorFn={u => URGENCY_COLOR[u] || 'secondary'} maxVal={100} />
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header bg-info text-white small fw-semibold">Avg Triage Score by Source</div>
              <div className="card-body p-2">
                <BarList items={ov.avg_triage_score_by_source} keyField="source" valueField="avg_score"
                  colorFn={() => 'info'} maxVal={100} />
              </div>
            </div>
          </div>

          {/* By Source table */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header bg-primary text-white small fw-semibold">Source Performance Matrix</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Source</th>
                      <th>Count</th>
                      <th>Avg Triage Score</th>
                      <th>Completion Rate</th>
                      <th>Top Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bySource.sort((a, b) => b.count - a.count).map((row, i) => (
                      <tr key={i}>
                        <td className="small text-capitalize fw-semibold">{row.source?.replace(/_/g, ' ')}</td>
                        <td><span className="badge bg-primary">{row.count}</span></td>
                        <td>
                          <div className="d-flex align-items-center gap-2">
                            <div className="progress flex-grow-1" style={{ height: 10 }}>
                              <div className="progress-bar bg-warning" style={{ width: `${row.avg_triage_score}%` }} />
                            </div>
                            <span className="small text-muted">{row.avg_triage_score}</span>
                          </div>
                        </td>
                        <td><span className={`badge bg-${row.completion_rate > 30 ? 'success' : 'secondary'}`}>{row.completion_rate}%</span></td>
                        <td className="small text-capitalize">{row.top_reason?.replace(/_/g, ' ')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── ALL REFERRALS ── */}
      {tab === 'referrals' && (
        <>
          <div className="row g-2 mb-3">
            <div className="col-md-5">
              <input
                className="form-control form-control-sm"
                placeholder="Search patient ID, reason, source…"
                value={refSearch}
                onChange={e => setRefSearch(e.target.value)}
              />
            </div>
            <div className="col-md-3">
              <select className="form-select form-select-sm" value={urgFilter} onChange={e => setUrgFilter(e.target.value)}>
                <option value="">All urgencies</option>
                {['emergent', 'urgent', 'routine', 'elective'].map(u => (
                  <option key={u} value={u}>{u}</option>
                ))}
              </select>
            </div>
            <div className="col-md-2 d-flex align-items-center">
              <span className="text-muted small">{filteredRefs.length} referrals</span>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-body p-0">
              <div style={{ maxHeight: 520, overflowY: 'auto' }}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark sticky-top">
                    <tr>
                      <th>ID</th>
                      <th>Patient</th>
                      <th>Date</th>
                      <th>Source</th>
                      <th>Reason</th>
                      <th>Urgency</th>
                      <th>Status</th>
                      <th>Score</th>
                      <th>Assigned To</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRefs.map((r, i) => (
                      <tr key={i}>
                        <td className="small text-muted">{r.id}</td>
                        <td className="small fw-semibold">{r.patient_id}</td>
                        <td className="small">{r.referral_date}</td>
                        <td className="small text-capitalize">{r.referral_source?.replace(/_/g, ' ')}</td>
                        <td className="small text-capitalize">{r.referral_reason?.replace(/_/g, ' ')}</td>
                        <td><Badge val={r.urgency} map={URGENCY_COLOR} /></td>
                        <td><Badge val={r.triage_status} map={STATUS_COLOR} /></td>
                        <td>
                          <span className={`badge bg-${r.triage_score >= 70 ? 'danger' : r.triage_score >= 40 ? 'warning' : 'success'}`}>
                            {r.triage_score}
                          </span>
                        </td>
                        <td className="small">{r.assigned_to}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && (
        <>
          <div className="row g-2 mb-3">
            <div className="col-md-4">
              <input
                className="form-control form-control-sm"
                placeholder="Search patient ID…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
            <div className="col-md-3">
              <select className="form-select form-select-sm" value={sortBy} onChange={e => setSortBy(e.target.value)}>
                <option value="total_referrals">Sort: Total Referrals</option>
                <option value="avg_triage_score">Sort: Avg Triage Score</option>
              </select>
            </div>
            <div className="col-md-2 d-flex align-items-center">
              <span className="text-muted small">{filteredPts.length} patients</span>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-body p-0">
              <div style={{ maxHeight: 520, overflowY: 'auto' }}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark sticky-top">
                    <tr>
                      <th>Patient</th>
                      <th>Referrals</th>
                      <th>Avg Triage</th>
                      <th>Latest Date</th>
                      <th>Top Source</th>
                      <th>Top Urgency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPts.map((p, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold">{p.patient_id}</td>
                        <td><span className="badge bg-primary">{p.total_referrals}</span></td>
                        <td>
                          <div className="d-flex align-items-center gap-2">
                            <div className="progress flex-grow-1" style={{ height: 10 }}>
                              <div
                                className={`progress-bar bg-${p.avg_triage_score >= 70 ? 'danger' : p.avg_triage_score >= 40 ? 'warning' : 'success'}`}
                                style={{ width: `${p.avg_triage_score}%` }}
                              />
                            </div>
                            <span className="small text-muted">{p.avg_triage_score}</span>
                          </div>
                        </td>
                        <td className="small">{p.latest_referral_date}</td>
                        <td className="small text-capitalize">{p.top_source?.replace(/_/g, ' ')}</td>
                        <td><Badge val={p.top_urgency} map={URGENCY_COLOR} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && (
        <div className="row g-3">
          {concepts.map((c, i) => (
            <div key={i} className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body py-2 px-3">
                  <div className="small fw-semibold text-primary mb-1">{c.name}</div>
                  <div className="small text-muted">{c.description}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
