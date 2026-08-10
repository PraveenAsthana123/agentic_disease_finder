'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Caregivers' },
  { id: 'definitions', label: 'Definitions' },
];

const BURNOUT_COLOR = {
  'Low (0-25)': 'success',
  'Moderate (26-50)': 'warning',
  'High (51-75)': 'danger',
  'Critical (76-100)': 'dark',
};

function burnoutColor(score) {
  if (score >= 76) return 'dark';
  if (score >= 51) return 'danger';
  if (score >= 26) return 'warning';
  return 'success';
}

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

function BarChart({ data, labelKey, valueKey, color = 'primary', title, colorMap }) {
  if (!data || !data.length) return null;
  const max = Math.max(...data.map(d => d[valueKey]));
  return (
    <div className="mb-4">
      {title && <div className="fw-semibold mb-2 small text-muted">{title}</div>}
      {data.map((d, i) => {
        const barColor = colorMap ? (colorMap[d[labelKey]] || color) : color;
        return (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span>{d[labelKey]}</span>
              <span className="fw-semibold">{d[valueKey]}</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div
                className={`progress-bar bg-${barColor}`}
                style={{ width: `${max > 0 ? (d[valueKey] / max) * 100 : 0}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function TrainingBar({ item }) {
  const pct = item.total > 0 ? Math.round((item.count / item.total) * 100) : 0;
  const color = pct >= 80 ? 'success' : pct >= 60 ? 'warning' : 'danger';
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{item.name}</span>
        <span className="fw-semibold">{item.count}/{item.total} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const k = data.kpis || {};
  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Caregivers" value={k.total_caregivers} color="primary" sub="registered" />
        <KPI label="Patients Covered" value={k.patients_covered} color="info" sub="unique patients" />
        <KPI label="Avg Experience" value={`${k.avg_experience_years}y`} color="secondary" sub="years caregiving" />
        <KPI
          label="Avg Burnout"
          value={k.avg_burnout}
          color={k.avg_burnout >= 60 ? 'danger' : k.avg_burnout >= 40 ? 'warning' : 'success'}
          sub="score / 100"
        />
      </div>
      <div className="row mb-3">
        <KPI
          label="Epilepsy Trained"
          value={`${k.epilepsy_trained_pct}%`}
          color={k.epilepsy_trained_pct >= 80 ? 'success' : 'warning'}
          sub="of caregivers"
        />
        <KPI
          label="First Aid Certified"
          value={`${k.first_aid_pct}%`}
          color={k.first_aid_pct >= 80 ? 'success' : 'warning'}
          sub="of caregivers"
        />
        <KPI
          label="Rescue Med Trained"
          value={`${k.rescue_med_pct}%`}
          color={k.rescue_med_pct >= 70 ? 'success' : 'danger'}
          sub="of caregivers"
        />
        <KPI label="Avg Confidence" value={`${k.avg_confidence}/10`} color="info" sub="seizure first-aid" />
      </div>

      <div className="row">
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold small">Burnout Distribution</div>
            <div className="card-body">
              <BarChart
                data={data.burnout_distribution || []}
                labelKey="tier"
                valueKey="cnt"
                colorMap={BURNOUT_COLOR}
              />
            </div>
          </div>
        </div>
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold small">Role Distribution</div>
            <div className="card-body">
              <BarChart
                data={data.role_distribution || []}
                labelKey="role"
                valueKey="cnt"
                color="primary"
              />
            </div>
          </div>
        </div>
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold small">Availability</div>
            <div className="card-body">
              <BarChart
                data={data.availability_distribution || []}
                labelKey="availability"
                valueKey="cnt"
                color="info"
              />
            </div>
          </div>
        </div>
      </div>

      <div className="card mb-3">
        <div className="card-header fw-semibold small">Training Completion</div>
        <div className="card-body">
          {(data.training_counts || []).map((t, i) => (
            <TrainingBar key={i} item={t} />
          ))}
        </div>
      </div>

      <div className="card mb-3">
        <div className="card-header fw-semibold small">Wellness by Role</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Role</th>
                  <th>Count</th>
                  <th>Avg Stress</th>
                  <th>Avg Burnout</th>
                  <th>Avg Confidence</th>
                </tr>
              </thead>
              <tbody>
                {(data.role_wellness || []).map((r, i) => (
                  <tr key={i}>
                    <td className="fw-semibold text-capitalize">{r.role}</td>
                    <td>{r.cnt}</td>
                    <td>
                      <span className={`badge bg-${r.avg_stress >= 7 ? 'danger' : r.avg_stress >= 5 ? 'warning' : 'success'}`}>
                        {r.avg_stress?.toFixed(1)}/10
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${burnoutColor(r.avg_burnout)}`}>
                        {r.avg_burnout?.toFixed(1)}
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${r.avg_confidence >= 7 ? 'success' : r.avg_confidence >= 5 ? 'warning' : 'danger'}`}>
                        {r.avg_confidence?.toFixed(1)}/10
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
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const all = data.all_caregivers || [];
  const highBurnout = data.high_burnout || [];

  return (
    <div>
      {highBurnout.length > 0 && (
        <div className="alert alert-danger mb-3">
          <strong>&#x1f6a8; {highBurnout.length} caregiver{highBurnout.length > 1 ? 's' : ''} with critical/high burnout</strong> — intervention recommended.
        </div>
      )}

      <div className="card mb-4">
        <div className="card-header fw-semibold small">By Role Summary</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Role</th>
                  <th>Count</th>
                  <th>Avg Exp (y)</th>
                  <th>Avg Burnout</th>
                  <th>Avg Stress</th>
                  <th>Trained</th>
                  <th>First Aid</th>
                </tr>
              </thead>
              <tbody>
                {(data.by_role || []).map((r, i) => (
                  <tr key={i}>
                    <td className="fw-semibold text-capitalize">{r.role}</td>
                    <td>{r.total}</td>
                    <td>{r.avg_experience?.toFixed(1)}</td>
                    <td>
                      <span className={`badge bg-${burnoutColor(r.avg_burnout)}`}>
                        {r.avg_burnout?.toFixed(1)}
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${r.avg_stress >= 7 ? 'danger' : r.avg_stress >= 5 ? 'warning' : 'success'}`}>
                        {r.avg_stress?.toFixed(1)}
                      </span>
                    </td>
                    <td>{r.trained_count}/{r.total}</td>
                    <td>{r.first_aid_count}/{r.total}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header fw-semibold small">All Caregivers ({all.length})</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Name</th>
                  <th>Patient</th>
                  <th>Role</th>
                  <th>Avail.</th>
                  <th>Exp (y)</th>
                  <th>Burnout</th>
                  <th>Stress</th>
                  <th>Confid.</th>
                  <th>Epilepsy Trn</th>
                  <th>First Aid</th>
                  <th>Rescue Med</th>
                  <th>Safety Plan</th>
                  <th>Action Plan</th>
                </tr>
              </thead>
              <tbody>
                {all.map((c, i) => (
                  <tr key={i}>
                    <td className="fw-semibold" style={{ whiteSpace: 'nowrap' }}>{c.name}</td>
                    <td><span className="badge bg-secondary">{c.patient_id}</span></td>
                    <td className="text-capitalize">{c.role}</td>
                    <td className="small">{c.availability}</td>
                    <td>{c.experience_years}</td>
                    <td>
                      <span className={`badge bg-${burnoutColor(c.burnout_score)}`}>
                        {c.burnout_score}
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${c.caregiver_stress >= 7 ? 'danger' : c.caregiver_stress >= 5 ? 'warning' : 'success'}`}>
                        {c.caregiver_stress}/10
                      </span>
                    </td>
                    <td>{c.seizure_first_aid_confidence}/10</td>
                    <td>{c.epilepsy_training_completed ? '✅' : '❌'}</td>
                    <td>{c.first_aid_certified ? '✅' : '❌'}</td>
                    <td>{c.rescue_med_trained ? '✅' : '❌'}</td>
                    <td>{c.safety_plan_exists ? '✅' : '❌'}</td>
                    <td>{c.seizure_action_plan_exists ? '✅' : '❌'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  return (
    <div>
      <div className="card mb-3">
        <div className="card-header fw-semibold small">Glossary</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr><th>Term</th><th>Definition</th></tr>
            </thead>
            <tbody>
              {(data.glossary || []).map((g, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ whiteSpace: 'nowrap' }}>{g.term}</td>
                  <td className="small">{g.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {(data.roles || []).length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold small">Caregiver Roles</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr><th>Role</th><th>Description</th></tr>
              </thead>
              <tbody>
                {data.roles.map((r, i) => (
                  <tr key={i}>
                    <td className="fw-semibold text-capitalize">{r.role}</td>
                    <td className="small">{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {(data.wellness_thresholds || []).length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold small">Wellness Thresholds</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr><th>Metric</th><th>Low</th><th>Moderate</th><th>High</th><th>Action</th></tr>
              </thead>
              <tbody>
                {data.wellness_thresholds.map((t, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{t.metric}</td>
                    <td className="small">{t.low}</td>
                    <td className="small">{t.moderate}</td>
                    <td className="small">{t.high}</td>
                    <td className="small">{t.action}</td>
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

export default function CaregiversPage() {
  const [tab, setTab] = useState('overview');
  const [data, setData] = useState({});

  useEffect(() => {
    if (!data[tab]) {
      fetch(`${API}/api/caregivers/${tab}`)
        .then(r => r.json())
        .then(d => setData(prev => ({ ...prev, [tab]: d })))
        .catch(() => setData(prev => ({ ...prev, [tab]: { error: 'Failed to load' } })));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: '1.6rem' }}>🫂</span>
        <div>
          <h4 className="mb-0 fw-bold">Caregivers Dashboard</h4>
          <div className="text-muted small">Caregiver wellness · burnout · training · role distribution — 30 caregivers across 30 patients</div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && <OverviewPanel data={data.overview} />}
      {tab === 'breakdown' && <BreakdownPanel data={data.breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={data.definitions} />}
    </div>
  );
}
