'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLORS = {
  'Approved':               '#22c55e',
  'Conditionally Approved': '#10b981',
  'Pending Review':         '#f59e0b',
  'Under Appeal':           '#6366f1',
  'Denied':                 '#ef4444',
  'Withdrawn':              '#94a3b8',
};

const PRIORITY_COLORS = {
  high:   '#ef4444',
  medium: '#f59e0b',
  low:    '#22c55e',
};

function StatCard({ label, value, sub, color = '#6366f1' }) {
  return (
    <div className="col-6 col-md mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h4 mb-0 fw-bold" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function MiniBar({ value, total, color }) {
  const pct = total ? Math.min(100, (value / total) * 100) : 0;
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 10, minWidth: 80 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color || '#6366f1' }} />
      </div>
      <span className="small fw-bold">{value}</span>
    </div>
  );
}

function RateBadge({ rate }) {
  const color = rate >= 70 ? '#22c55e' : rate >= 50 ? '#f59e0b' : '#ef4444';
  return <span className="badge" style={{ backgroundColor: color, fontSize: '0.75rem' }}>{rate}%</span>;
}

// ---------------------------------------------------------------------------
// Tab: Overview
// ---------------------------------------------------------------------------
function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" /></div>;
  if (!data.available) return <div className="alert alert-warning">{data.message}</div>;

  const s = data.summary || {};
  const statusDist = data.status_distribution || [];
  const totalStatus = statusDist.reduce((a, b) => a + b.count, 0);

  return (
    <div>
      {/* KPI row */}
      <div className="row g-2 mb-3">
        <StatCard label="Pre-Auth Requests"  value={s.total_preauth_requests}  color="#6366f1" />
        <StatCard label="Unique Patients"    value={s.unique_patients}          color="#3b82f6" />
        <StatCard label="Approval Rate"      value={`${s.approval_rate_pct}%`}  color="#22c55e"
                  sub="approved + conditional" />
        <StatCard label="Denial Rate"        value={`${s.denial_rate_pct}%`}    color="#ef4444" />
        <StatCard label="Avg Decision (days)" value={s.avg_decision_days ?? 'N/A'} color="#f59e0b" />
        <StatCard label="Under Appeal"       value={s.under_appeal_count}       color="#8b5cf6" />
      </div>

      <div className="row g-3 mb-3">
        {/* Status distribution */}
        <div className="col-md-5">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small">Pre-Auth Status Distribution</div>
            <div className="card-body p-2">
              {statusDist.map(row => (
                <div key={row.status} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span style={{ color: STATUS_COLORS[row.status] || '#6366f1' }}>
                      {row.status}
                    </span>
                    <span className="text-muted">{row.count} ({totalStatus ? Math.round(row.count / totalStatus * 100) : 0}%)</span>
                  </div>
                  <MiniBar value={row.count} total={totalStatus} color={STATUS_COLORS[row.status]} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Payer summary */}
        <div className="col-md-7">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small">Approval Rate by Payer</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Payer</th>
                    <th className="text-center">Total</th>
                    <th className="text-center">Approved</th>
                    <th className="text-center">Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {(data.payer_summary || []).map(p => (
                    <tr key={p.payer}>
                      <td className="small">{p.payer}</td>
                      <td className="text-center small">{p.total}</td>
                      <td className="text-center small">{p.approved}</td>
                      <td className="text-center"><RateBadge rate={p.approval_rate} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3">
        {/* Service type summary */}
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-bold small">Pre-Auth by Service Type</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Service</th>
                    <th className="text-center">Priority</th>
                    <th className="text-center">Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(data.service_summary || []).map(s => (
                    <tr key={s.service_type}>
                      <td className="small">{s.label}</td>
                      <td className="text-center">
                        <span className="badge" style={{ backgroundColor: PRIORITY_COLORS[s.priority] || '#6366f1', fontSize: '0.7rem' }}>
                          {s.priority}
                        </span>
                      </td>
                      <td className="text-center small fw-bold">{s.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Top denial reasons */}
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-bold small">Top Denial Reasons</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>Reason</th><th className="text-center">Count</th></tr>
                </thead>
                <tbody>
                  {(data.top_denial_reasons || []).map((r, i) => (
                    <tr key={i}>
                      <td className="small">{r.reason}</td>
                      <td className="text-center">
                        <span className="badge bg-danger">{r.count}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Financial impact */}
      <div className="row g-2 mt-2">
        <StatCard label="Total Billed"
                  value={`$${(s.total_billed || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                  color="#6366f1" />
        <StatCard label="Total Approved Amt"
                  value={`$${(s.total_approved_amt || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                  color="#22c55e" />
        <StatCard label="Pending Count" value={s.pending_count} color="#f59e0b" />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab: Breakdown
// ---------------------------------------------------------------------------
function BreakdownTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" /></div>;
  if (!data.available) return <div className="alert alert-warning">{data.message}</div>;

  return (
    <div>
      {/* Per-payer detail */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold small">Per-Payer Pre-Auth Detail</div>
        <div className="card-body p-0" style={{ overflowX: 'auto' }}>
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Payer</th>
                <th className="text-center">Total</th>
                <th className="text-center">Approved</th>
                <th className="text-center">Denied</th>
                <th className="text-center">Pending</th>
                <th className="text-center">Appeal</th>
                <th className="text-center">Rate</th>
                <th className="text-center">Avg Days</th>
                <th className="text-right">Billed</th>
              </tr>
            </thead>
            <tbody>
              {(data.payer_detail || []).map(p => (
                <tr key={p.payer}>
                  <td className="small fw-bold">{p.payer}</td>
                  <td className="text-center small">{p.total}</td>
                  <td className="text-center"><span className="badge bg-success">{p.approved}</span></td>
                  <td className="text-center"><span className="badge bg-danger">{p.denied}</span></td>
                  <td className="text-center"><span className="badge bg-warning text-dark">{p.pending}</span></td>
                  <td className="text-center"><span className="badge bg-secondary">{p.under_appeal}</span></td>
                  <td className="text-center"><RateBadge rate={p.approval_rate} /></td>
                  <td className="text-center small">{p.avg_decision_days ?? '—'}</td>
                  <td className="text-end small">${(p.total_billed || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-service type detail */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold small">Per-Service Type Pre-Auth Outcomes</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Service Type</th>
                <th className="text-center">Priority</th>
                <th className="text-center">Total</th>
                <th className="text-center">Approved</th>
                <th className="text-center">Denied</th>
                <th className="text-center">Pending</th>
                <th className="text-center">Rate</th>
              </tr>
            </thead>
            <tbody>
              {(data.service_detail || []).map(s => (
                <tr key={s.service_type}>
                  <td className="small">{s.label}</td>
                  <td className="text-center">
                    <span className="badge" style={{ backgroundColor: PRIORITY_COLORS[s.priority] || '#6366f1', fontSize: '0.7rem' }}>
                      {s.priority}
                    </span>
                  </td>
                  <td className="text-center small">{s.total}</td>
                  <td className="text-center"><span className="badge bg-success">{s.approved}</span></td>
                  <td className="text-center"><span className="badge bg-danger">{s.denied}</span></td>
                  <td className="text-center"><span className="badge bg-warning text-dark">{s.pending}</span></td>
                  <td className="text-center"><RateBadge rate={s.approval_rate} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Denial reason breakdown */}
      <div className="row g-3 mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small">Denial Reason Breakdown</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>Reason</th><th className="text-center">Count</th><th className="text-center">%</th></tr>
                </thead>
                <tbody>
                  {(data.denial_breakdown || []).map((r, i) => (
                    <tr key={i}>
                      <td className="small">{r.reason}</td>
                      <td className="text-center"><span className="badge bg-danger">{r.count}</span></td>
                      <td className="text-center small">{r.pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Recent pre-auth activity (mini) */}
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small">Recent Pre-Auth Requests (latest 10)</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th>Service</th>
                    <th>Status</th>
                    <th className="text-center">Days</th>
                  </tr>
                </thead>
                <tbody>
                  {(data.recent_preauths || []).slice(0, 10).map((r, i) => (
                    <tr key={i}>
                      <td className="small">{r.patient_name || r.patient_id}</td>
                      <td className="small" style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {r.service_type}
                      </td>
                      <td>
                        <span className="badge" style={{
                          backgroundColor: STATUS_COLORS[r.preauth_status] || '#6366f1',
                          fontSize: '0.65rem'
                        }}>
                          {r.preauth_status}
                        </span>
                      </td>
                      <td className="text-center small">{r.decision_days ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Full per-patient pre-auth log */}
      <div className="card shadow-sm">
        <div className="card-header fw-bold small">Pre-Auth Request Log (most recent 40)</div>
        <div className="card-body p-0" style={{ overflowX: 'auto' }}>
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Claim ID</th>
                <th>Patient</th>
                <th>Disease</th>
                <th>Service</th>
                <th>Payer</th>
                <th>Status</th>
                <th className="text-right">Billed</th>
                <th className="text-center">Days</th>
                <th>Submitted</th>
                <th>Denial Reason</th>
              </tr>
            </thead>
            <tbody>
              {(data.recent_preauths || []).map((r, i) => (
                <tr key={i}>
                  <td className="small text-muted">{r.claim_id}</td>
                  <td className="small">{r.patient_name || r.patient_id}</td>
                  <td className="small">{r.disease || '—'}</td>
                  <td className="small">{r.service_type}</td>
                  <td className="small">{r.payer}</td>
                  <td>
                    <span className="badge" style={{
                      backgroundColor: STATUS_COLORS[r.preauth_status] || '#6366f1',
                      fontSize: '0.65rem'
                    }}>
                      {r.preauth_status}
                    </span>
                  </td>
                  <td className="text-end small">${r.amount_billed?.toLocaleString()}</td>
                  <td className="text-center small">{r.decision_days ?? '—'}</td>
                  <td className="small">{r.submitted_at}</td>
                  <td className="small text-muted" style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {r.denial_reason || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab: Definitions
// ---------------------------------------------------------------------------
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" /></div>;
  if (!data.available) return <div className="alert alert-warning">{data.message}</div>;

  return (
    <div>
      <div className="alert alert-info mb-3">
        <strong>What is Prior Authorization?</strong> {data.what_is_prior_auth}
      </div>

      {/* PA Statuses */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold small">Pre-Authorization Statuses</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr><th style={{ width: 200 }}>Status</th><th>Description</th></tr>
            </thead>
            <tbody>
              {(data.preauth_statuses || []).map(s => (
                <tr key={s.status}>
                  <td>
                    <span className="badge" style={{ backgroundColor: STATUS_COLORS[s.status] || '#6366f1' }}>
                      {s.status}
                    </span>
                  </td>
                  <td className="small">{s.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Services requiring pre-auth */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold small">Epilepsy Services Requiring Prior Authorization</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr>
                <th>Service</th>
                <th>CPT Codes</th>
                <th>Priority</th>
                <th>Typical TAT</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {(data.services_requiring_preauth || []).map(s => (
                <tr key={s.service}>
                  <td className="small fw-bold">{s.service}</td>
                  <td className="small text-muted">{(s.cpt_codes || []).join(', ')}</td>
                  <td className="small">{s.priority}</td>
                  <td className="small">{s.typical_tat}</td>
                  <td className="small text-muted">{s.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Payer policies */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold small">Payer PA Policy Summary</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr><th>Payer</th><th>PA Required For</th><th>Turnaround</th></tr>
            </thead>
            <tbody>
              {(data.payer_policies || []).map(p => (
                <tr key={p.payer}>
                  <td className="small fw-bold">{p.payer}</td>
                  <td className="small">{p.pa_required}</td>
                  <td className="small">{p.turnaround}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Denial reasons + actions */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold small">Common Denial Reasons &amp; Remediation Actions</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr><th style={{ width: '40%' }}>Reason</th><th>Recommended Action</th></tr>
            </thead>
            <tbody>
              {(data.common_denial_reasons || []).map(r => (
                <tr key={r.reason}>
                  <td className="small text-danger">{r.reason}</td>
                  <td className="small">{r.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Regulatory references */}
      <div className="card shadow-sm">
        <div className="card-header fw-bold small">Regulatory References</div>
        <ul className="list-group list-group-flush">
          {(data.regulatory_references || []).map((ref, i) => (
            <li key={i} className="list-group-item small">{ref}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------
export default function InsurancePreAuthPage() {
  const [tab,    setTab]    = useState('overview');
  const [ovData, setOvData] = useState(null);
  const [bkData, setBkData] = useState(null);
  const [dfData, setDfData] = useState(null);
  const [error,  setError]  = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [ov, bk, df] = await Promise.all([
          fetch(`${API}/api/insurance-preauth/overview`).then(r => r.json()),
          fetch(`${API}/api/insurance-preauth/breakdown`).then(r => r.json()),
          fetch(`${API}/api/insurance-preauth/definitions`).then(r => r.json()),
        ]);
        setOvData(ov);
        setBkData(bk);
        setDfData(df);
      } catch (e) {
        setError(String(e));
      }
    }
    load();
  }, []);

  if (error) return (
    <div className="container py-5">
      <div className="alert alert-danger">Failed to load: {error}</div>
    </div>
  );

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1 fw-bold">
        <span className="me-2">🏥</span>Insurance Pre-Authorization Dashboard
      </h4>
      <p className="text-muted small mb-3">
        Prior authorization workflow — 114 pre-auth-requiring claims · 7 payers · 62% approval rate ·
        Admin / Compliance / Billing
      </p>

      <ul className="nav nav-tabs mb-3">
        {[
          { id: 'overview',    label: 'Overview' },
          { id: 'breakdown',   label: 'Breakdown' },
          { id: 'definitions', label: 'Definitions' },
        ].map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewTab    data={ovData} />}
      {tab === 'breakdown'   && <BreakdownTab   data={bkData} />}
      {tab === 'definitions' && <DefinitionsTab data={dfData} />}
    </div>
  );
}
