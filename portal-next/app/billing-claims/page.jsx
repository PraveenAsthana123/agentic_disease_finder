'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Per-Patient' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  const map = {
    paid: 'success',
    approved: 'primary',
    submitted: 'info',
    denied: 'danger',
    partially_approved: 'warning',
    appealed: 'secondary',
    write_off: 'dark',
  };
  return (
    <span className={`badge bg-${map[status] || 'secondary'}`}>
      {status?.replace(/_/g, ' ')}
    </span>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const k = data.kpis || {};
  const fmt$ = v => v != null ? `$${Number(v).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}` : '—';
  const fmtPct = v => v != null ? `${v}%` : '—';

  return (
    <div>
      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Total Claims" value={k.total_claims} color="primary" />
        <KPI label="Total Billed" value={fmt$(k.total_billed)} color="info" />
        <KPI label="Total Collected" value={fmt$(k.total_collected)} color="success" />
        <KPI label="Collection Rate" value={fmtPct(k.collection_rate_pct)} color={k.collection_rate_pct >= 85 ? 'success' : 'warning'} sub="Target: 92-96%" />
      </div>
      <div className="row mb-4">
        <KPI label="Outstanding Balance" value={fmt$(k.outstanding_balance)} color="danger" />
        <KPI label="Denial Rate" value={fmtPct(k.denial_rate_pct)} color={k.denial_rate_pct <= 5 ? 'success' : 'danger'} sub="Target: <5%" />
        <KPI label="Avg Days to Payment" value={k.avg_days_to_payment != null ? `${k.avg_days_to_payment}d` : '—'} color="secondary" />
        <KPI label="Unique Patients" value={k.unique_patients} color="primary" />
      </div>

      <div className="row">
        {/* Claim status distribution */}
        <div className="col-md-5 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Claim Status Distribution</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead><tr><th>Status</th><th>Count</th><th>%</th></tr></thead>
                <tbody>
                  {(data.status_distribution || []).map(s => (
                    <tr key={s.status}>
                      <td><StatusBadge status={s.status} /></td>
                      <td>{s.count}</td>
                      <td>
                        <div className="d-flex align-items-center gap-1">
                          <div className="progress flex-grow-1" style={{ height: 8 }}>
                            <div className="progress-bar bg-primary" style={{ width: `${s.pct}%` }} />
                          </div>
                          <small>{s.pct}%</small>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Insurer breakdown */}
        <div className="col-md-7 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Insurance Payer Breakdown</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead><tr><th>Insurer</th><th>Claims</th><th>Billed</th><th>Collected</th><th>Coll%</th></tr></thead>
                <tbody>
                  {(data.insurance_breakdown || []).map(i => {
                    const pct = i.billed > 0 ? ((i.collected / i.billed) * 100).toFixed(1) : 0;
                    return (
                      <tr key={i.insurer}>
                        <td className="fw-semibold">{i.insurer}</td>
                        <td>{i.claims}</td>
                        <td>{fmt$(i.billed)}</td>
                        <td>{fmt$(i.collected)}</td>
                        <td>
                          <span className={`badge bg-${pct >= 40 ? 'success' : pct >= 20 ? 'warning' : 'danger'}`}>{pct}%</span>
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

      {/* Monthly trend */}
      {data.monthly_trend && data.monthly_trend.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Monthly Trend</div>
          <div className="card-body">
            <div className="table-responsive">
              <table className="table table-sm">
                <thead><tr><th>Month</th><th>Claims</th><th>Billed</th><th>Collected</th><th>Collection Rate</th></tr></thead>
                <tbody>
                  {data.monthly_trend.map(m => {
                    const pct = m.billed > 0 ? ((m.collected / m.billed) * 100).toFixed(1) : 0;
                    return (
                      <tr key={m.month}>
                        <td className="fw-semibold">{m.month}</td>
                        <td>{m.claims}</td>
                        <td>{fmt$(m.billed)}</td>
                        <td>{fmt$(m.collected)}</td>
                        <td>
                          <div className="d-flex align-items-center gap-1">
                            <div className="progress flex-grow-1" style={{ height: 8, minWidth: 80 }}>
                              <div className="progress-bar bg-success" style={{ width: `${Math.min(pct, 100)}%` }} />
                            </div>
                            <small>{pct}%</small>
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
      )}

      {/* Top services */}
      {data.top_services && data.top_services.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Top Services by Claims</div>
          <div className="card-body">
            <div className="row">
              {data.top_services.map(s => (
                <div key={s.service} className="col-6 col-md-3 mb-2">
                  <div className="card text-center p-2">
                    <div className="fw-bold">{s.claims}</div>
                    <div className="small text-muted">{s.service?.replace(/_/g, ' ')}</div>
                    <div className="small text-success">${Number(s.avg_billed || 0).toFixed(0)} avg</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const fmt$ = v => v != null ? `$${Number(v).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}` : '—';
  const patients = data.per_patient || [];

  return (
    <div>
      <p className="text-muted small mb-3">Per-patient billing summary — {patients.length} patients with claims in clinical.db</p>
      <div className="table-responsive">
        <table className="table table-sm table-striped table-hover">
          <thead className="table-dark">
            <tr>
              <th>Patient ID</th>
              <th>Claims</th>
              <th>Total Billed</th>
              <th>Total Paid</th>
              <th>Balance</th>
              <th>Coll%</th>
            </tr>
          </thead>
          <tbody>
            {patients.map(p => {
              const pct = p.total_billed > 0 ? ((p.total_paid / p.total_billed) * 100).toFixed(1) : 0;
              return (
                <tr key={p.patient_id}>
                  <td className="fw-bold">{p.patient_id}</td>
                  <td>{p.claims}</td>
                  <td>{fmt$(p.total_billed)}</td>
                  <td className="text-success">{fmt$(p.total_paid)}</td>
                  <td className={p.balance > 0 ? 'text-danger' : 'text-muted'}>{fmt$(p.balance)}</td>
                  <td>
                    <span className={`badge bg-${pct >= 50 ? 'success' : pct >= 20 ? 'warning' : 'danger'}`}>{pct}%</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  return (
    <div>
      {data.statuses && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Claim Statuses</div>
          <div className="card-body">
            <dl className="row mb-0">
              {Object.entries(data.statuses).map(([k, v]) => (
                <div key={k} className="d-flex gap-2 mb-1">
                  <dt className="col-3"><StatusBadge status={k} /></dt>
                  <dd className="col-9 mb-0 small text-muted">{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      )}
      {data.service_types && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Service Types</div>
          <div className="card-body">
            <dl className="row mb-0">
              {Object.entries(data.service_types).map(([k, v]) => (
                <div key={k} className="d-flex gap-2 mb-1">
                  <dt className="col-3 small fw-semibold">{k.replace(/_/g, ' ')}</dt>
                  <dd className="col-9 mb-0 small text-muted">{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      )}
      {data.aging_note && (
        <div className="alert alert-info small">{data.aging_note}</div>
      )}
      {data.icd10_note && (
        <div className="alert alert-light small"><strong>ICD-10:</strong> {data.icd10_note}</div>
      )}
      {data.cpt_note && (
        <div className="alert alert-light small"><strong>CPT:</strong> {data.cpt_note}</div>
      )}
      {data.collection_rate_formula && (
        <div className="alert alert-secondary small"><strong>Collection Rate:</strong> {data.collection_rate_formula}</div>
      )}
    </div>
  );
}

export default function BillingClaimsPage() {
  const [tab, setTab] = useState('overview');
  const [cache, setCache] = useState({});
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  const EP = {
    overview: `${API}/api/billing-claims/overview`,
    breakdown: `${API}/api/billing-claims/breakdown`,
    definitions: `${API}/api/billing-claims/definitions`,
  };

  function loadTab(t) {
    if (cache[t]) return;
    setLoading(true);
    setErr(null);
    fetch(EP[t])
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(d => { setCache(prev => ({ ...prev, [t]: d })); setLoading(false); })
      .catch(e => { setErr(`${t}: ${e.message}`); setLoading(false); });
  }

  useEffect(() => { loadTab('overview'); }, []);

  function switchTab(t) {
    setTab(t);
    loadTab(t);
  }

  return (
    <div className="container-fluid p-3">
      <h3 className="mb-1">&#x1f4b3; Billing &amp; Claims</h3>
      <p className="text-muted mb-3 small">
        Medical billing analytics — 150 claims, 40 patients, 7 claim statuses, 6 insurers, 8 service types. Real clinical.db data.
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => switchTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {err && <div className="alert alert-danger">{err}</div>}
      {loading && <div className="text-muted py-2">Loading…</div>}

      {!loading && tab === 'overview' && <OverviewPanel data={cache.overview} />}
      {!loading && tab === 'breakdown' && <BreakdownPanel data={cache.breakdown} />}
      {!loading && tab === 'definitions' && <DefinitionsPanel data={cache.definitions} />}
    </div>
  );
}
