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

function AdmTypeBadge({ type }) {
  const map = { emergency: 'danger', planned: 'primary', observation: 'info', transfer: 'warning' };
  return <span className={`badge bg-${map[type] || 'secondary'}`}>{type}</span>;
}

function DistBar({ items, colorFn }) {
  const total = Object.values(items || {}).reduce((a, b) => a + b, 0);
  return (
    <table className="table table-sm mb-0">
      <tbody>
        {Object.entries(items || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => {
          const pct = total > 0 ? ((v / total) * 100).toFixed(1) : 0;
          const color = colorFn ? colorFn(k) : 'primary';
          return (
            <tr key={k}>
              <td className="text-nowrap small fw-semibold" style={{ width: '40%' }}>{k.replace(/_/g, ' ')}</td>
              <td style={{ width: '45%' }}>
                <div className="progress" style={{ height: 10 }}>
                  <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
                </div>
              </td>
              <td className="small text-end">{v} <span className="text-muted">({pct}%)</span></td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const fmt$ = v => v != null ? `$${Number(v).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}` : '—';

  const wardColor = w => ({ 'Epilepsy Monitoring Unit': 'info', 'Emergency': 'danger', 'Neurology Ward': 'primary', 'ICU': 'warning', 'Surgical Recovery': 'success' }[w] || 'secondary');
  const typeColor = t => ({ emergency: 'danger', planned: 'primary', observation: 'info' }[t] || 'secondary');
  const dispColor = d => ({ home: 'success', transferred: 'info', ama: 'warning', rehabilitation: 'primary', currently_admitted: 'dark' }[d] || 'secondary');

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Admissions" value={data.total_admissions} color="primary" />
        <KPI label="Currently Admitted" value={data.currently_admitted} color="warning" sub="active inpatients" />
        <KPI label="Avg Length of Stay" value={data.avg_length_of_stay_days != null ? `${data.avg_length_of_stay_days}d` : '—'} color="info" sub="days per admission" />
        <KPI label="Readmission Rate" value={data.readmission_rate_pct != null ? `${data.readmission_rate_pct}%` : '—'} color={data.readmission_rate_pct <= 15 ? 'success' : 'danger'} sub="within 30 days" />
      </div>
      <div className="row mb-4">
        <KPI label="Seizure-Free Discharge" value={data.seizure_free_discharge_rate_pct != null ? `${data.seizure_free_discharge_rate_pct}%` : '—'} color="success" sub="% seizure-free at d/c" />
        <KPI label="Avg Cost / Admission" value={fmt$(data.avg_cost_per_admission)} color="secondary" />
        <KPI label="Private Insurance" value={data.insurance_distribution?.private != null ? `${data.insurance_distribution.private}` : '—'} color="primary" sub="private payer" />
        <KPI label="Public Insurance" value={data.insurance_distribution?.public != null ? `${data.insurance_distribution.public}` : '—'} color="info" sub="public/government" />
      </div>

      <div className="row mb-3">
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Admission Type</div>
            <div className="card-body p-2">
              <DistBar items={data.admission_type_distribution} colorFn={typeColor} />
            </div>
          </div>
        </div>
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Ward Distribution</div>
            <div className="card-body p-2">
              <DistBar items={data.ward_distribution} colorFn={wardColor} />
            </div>
          </div>
        </div>
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Discharge Disposition</div>
            <div className="card-body p-2">
              <DistBar items={data.disposition_distribution} colorFn={dispColor} />
            </div>
          </div>
        </div>
      </div>

      <div className="card mb-3">
        <div className="card-header fw-semibold">Admission Reason Distribution</div>
        <div className="card-body p-2">
          <div className="row">
            {Object.entries(data.admission_reason_distribution || {}).sort((a, b) => b[1] - a[1]).map(([reason, count]) => {
              const total = Object.values(data.admission_reason_distribution || {}).reduce((a, b) => a + b, 0);
              const pct = total > 0 ? ((count / total) * 100).toFixed(1) : 0;
              return (
                <div key={reason} className="col-6 col-md-3 mb-2">
                  <div className="card text-center p-2 h-100">
                    <div className="fw-bold h5 mb-1">{count}</div>
                    <div className="small text-muted">{reason.replace(/_/g, ' ')}</div>
                    <div className="small text-primary">{pct}%</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {data.monthly_timeline && data.monthly_timeline.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Monthly Admissions &amp; Discharges</div>
          <div className="card-body">
            <div className="table-responsive">
              <table className="table table-sm">
                <thead className="table-light">
                  <tr><th>Month</th><th>Admissions</th><th>Discharges</th><th>Avg LOS</th></tr>
                </thead>
                <tbody>
                  {data.monthly_timeline.map(m => (
                    <tr key={m.month}>
                      <td className="fw-semibold">{m.month}</td>
                      <td>{m.admissions}</td>
                      <td>{m.discharges}</td>
                      <td>{m.avg_los != null ? `${m.avg_los}d` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {data.insurance_distribution && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Insurance Type</div>
          <div className="card-body p-2">
            <DistBar items={data.insurance_distribution} />
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
      <p className="text-muted small mb-3">Per-patient hospitalization summary — {patients.length} patients with admissions in clinical.db</p>
      <div className="table-responsive">
        <table className="table table-sm table-striped table-hover">
          <thead className="table-dark">
            <tr>
              <th>Patient</th>
              <th>Admissions</th>
              <th>Total Days</th>
              <th>Avg LOS</th>
              <th>Readmissions</th>
              <th>Seizure-Free %</th>
              <th>Total Cost</th>
            </tr>
          </thead>
          <tbody>
            {patients.map(p => (
              <tr key={p.patient_id}>
                <td className="fw-bold">{p.patient_id}</td>
                <td>{p.total_admissions}</td>
                <td>{p.total_days}</td>
                <td>{p.avg_los != null ? `${p.avg_los}d` : '—'}</td>
                <td>
                  <span className={`badge bg-${p.readmissions > 0 ? 'danger' : 'success'}`}>{p.readmissions}</span>
                </td>
                <td>
                  <div className="d-flex align-items-center gap-1">
                    <div className="progress flex-grow-1" style={{ height: 8, minWidth: 50 }}>
                      <div className={`progress-bar bg-${p.seizure_free_rate >= 75 ? 'success' : p.seizure_free_rate >= 50 ? 'warning' : 'danger'}`} style={{ width: `${p.seizure_free_rate ?? 0}%` }} />
                    </div>
                    <small>{p.seizure_free_rate ?? 0}%</small>
                  </div>
                </td>
                <td className="text-secondary">{fmt$(p.total_cost)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const Section = ({ title, items }) => items ? (
    <div className="card mb-3">
      <div className="card-header fw-semibold">{title}</div>
      <div className="card-body">
        <dl className="row mb-0">
          {Object.entries(items).map(([k, v]) => (
            <div key={k} className="d-flex gap-2 mb-2">
              <dt className="col-4 small fw-semibold">{k.replace(/_/g, ' ')}</dt>
              <dd className="col-8 mb-0 small text-muted">{v}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  ) : null;

  return (
    <div>
      <Section title="Admission Types" items={data.admission_types} />
      <Section title="Admission Reasons" items={data.admission_reasons} />
      <Section title="Wards" items={data.wards} />
      <Section title="Discharge Dispositions" items={data.disposition_types} />
      {data.glossary && data.glossary.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Glossary</div>
          <div className="card-body">
            <dl className="row mb-0">
              {data.glossary.map(g => (
                <div key={g.term} className="d-flex gap-2 mb-2">
                  <dt className="col-3 small fw-semibold">{g.term}</dt>
                  <dd className="col-9 mb-0 small text-muted">{g.definition}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}

export default function HospitalizationPage() {
  const [tab, setTab] = useState('overview');
  const [cache, setCache] = useState({});
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  const EP = {
    overview: `${API}/api/hospitalization/overview`,
    breakdown: `${API}/api/hospitalization/breakdown`,
    definitions: `${API}/api/hospitalization/definitions`,
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
      <h3 className="mb-1">&#x1f3e5; Hospitalization</h3>
      <p className="text-muted mb-3 small">
        Inpatient admission analytics — 115 admissions, 40 patients, 3 admission types, 5 wards, 9 admission reasons. Real clinical.db data.
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
