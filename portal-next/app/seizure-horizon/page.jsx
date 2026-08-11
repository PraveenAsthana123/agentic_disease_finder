'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',   label: 'Overview' },
  { id: 'breakdown',  label: 'Horizon Analysis' },
  { id: 'patients',   label: 'Per Patient' },
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

function ViabilityBadge({ viable }) {
  return viable
    ? <span className="badge bg-success">&#x2714; Viable</span>
    : <span className="badge bg-danger">&#x2718; Not Viable</span>;
}

function MetricBar({ value, max = 1, color = 'primary' }) {
  const pct = Math.round((value / max) * 100);
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 10 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="small fw-semibold" style={{ minWidth: 44 }}>{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

function FARBar({ far, target = 0.15 }) {
  const max = 0.35;
  const pct = Math.round((Math.min(far, max) / max) * 100);
  const color = far <= target ? 'success' : far <= 0.25 ? 'warning' : 'danger';
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 10 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`small fw-semibold text-${color}`} style={{ minWidth: 54 }}>{far.toFixed(3)}/hr</span>
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const s = data.summary || {};
  const sweep = data.horizon_sweep || [];
  const roc = data.roc_by_horizon || [];

  return (
    <div>
      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Patients" value={s.n_patients} color="info" sub="with seizure diary" />
        <KPI label="Horizons Tested" value={s.horizons_tested} color="primary" sub="30 min → 24 hr" />
        <KPI label="Viable Horizons" value={s.viable_horizons} color={s.viable_horizons > 0 ? 'success' : 'danger'} sub={`FAR<${s.far_target}/hr & Sens≥${(s.sens_target*100).toFixed(0)}%`} />
        <KPI label="Optimal Horizon" value={s.optimal_label} color="success" sub="longest viable window" />
      </div>
      <div className="row mb-4">
        <KPI label="Sensitivity @Optimal" value={s.optimal_sensitivity != null ? `${(s.optimal_sensitivity*100).toFixed(1)}%` : '—'} color={s.optimal_sensitivity >= s.sens_target ? 'success' : 'warning'} sub="true positive rate" />
        <KPI label="FAR @Optimal" value={s.optimal_far != null ? `${s.optimal_far}/hr` : '—'} color={s.optimal_far <= s.far_target ? 'success' : 'danger'} sub="false alarms/hr" />
        <KPI label="AUC @Optimal" value={s.optimal_auc != null ? s.optimal_auc.toFixed(3) : '—'} color={s.optimal_auc >= 0.80 ? 'success' : 'warning'} sub="ROC area" />
        <KPI label="Seizure Events" value={s.n_seizure_events} color="secondary" sub="total diary events" />
      </div>

      {/* Horizon sweep table */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">&#x1f4ca; Horizon Performance Sweep</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Horizon</th>
                <th>Sensitivity</th>
                <th>FAR / hr</th>
                <th>Specificity</th>
                <th>AUC</th>
                <th>PPV</th>
                <th>Viable?</th>
              </tr>
            </thead>
            <tbody>
              {sweep.map(h => (
                <tr key={h.horizon_label} className={h.clinically_viable ? 'table-success' : ''}>
                  <td><strong>{h.horizon_label}</strong></td>
                  <td>
                    <MetricBar value={h.sensitivity} color={h.meets_sens_target ? 'success' : 'warning'} />
                  </td>
                  <td>
                    <FARBar far={h.far_per_hour} target={s.far_target || 0.15} />
                  </td>
                  <td>{(h.specificity * 100).toFixed(1)}%</td>
                  <td>
                    <span className={`fw-semibold text-${h.auc >= 0.80 ? 'success' : 'warning'}`}>{h.auc.toFixed(3)}</span>
                  </td>
                  <td>{(h.ppv * 100).toFixed(1)}%</td>
                  <td><ViabilityBadge viable={h.clinically_viable} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ROC by horizon */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">&#x1f4c9; Sensitivity vs 1-Specificity (per horizon)</div>
        <div className="card-body">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr>
                <th>Horizon</th>
                <th>Sensitivity</th>
                <th>1 − Specificity</th>
                <th>AUC</th>
                <th>FAR / hr</th>
              </tr>
            </thead>
            <tbody>
              {roc.map(r => (
                <tr key={r.horizon_label}>
                  <td>{r.horizon_label}</td>
                  <td>{(r.sensitivity * 100).toFixed(1)}%</td>
                  <td>{(r.one_minus_specificity * 100).toFixed(1)}%</td>
                  <td>{r.auc.toFixed(3)}</td>
                  <td>{r.far_per_hour.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* References */}
      {data.references && (
        <div className="card mb-2">
          <div className="card-header fw-semibold">&#x1f4da; Clinical References</div>
          <div className="card-body">
            <ul className="mb-0">
              {data.references.map((r, i) => <li key={i}><span className="small">{r}</span></li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const tradeoff = data.sensitivity_far_tradeoff || [];
  const viability = data.viability_matrix || [];
  const descs = data.metric_descriptions || {};

  return (
    <div>
      {/* Sensitivity vs FAR trade-off */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">&#x26a1; Sensitivity–FAR Trade-Off Across Horizons</div>
        <div className="card-body">
          <p className="text-muted small mb-3">
            As horizon grows, sensitivity drops and FAR rises. The clinical sweet-spot (FAR&lt;0.15/hr, Sens≥75%) is shaded green.
          </p>
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr><th>Horizon</th><th>Sensitivity</th><th>FAR / hr</th><th>FAR Target Met?</th></tr>
            </thead>
            <tbody>
              {tradeoff.map(t => (
                <tr key={t.horizon_label}>
                  <td><strong>{t.horizon_label}</strong></td>
                  <td><span className={`fw-semibold text-${t.sensitivity >= 0.75 ? 'success' : 'warning'}`}>{(t.sensitivity*100).toFixed(1)}%</span></td>
                  <td><span className={`fw-semibold text-${t.far_per_hour <= 0.15 ? 'success' : 'danger'}`}>{t.far_per_hour.toFixed(3)}/hr</span></td>
                  <td>{t.far_per_hour <= 0.15 ? <span className="badge bg-success">Yes</span> : <span className="badge bg-danger">No</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Viability matrix */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">&#x2705; Clinical Viability Matrix</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr>
                <th>Horizon</th>
                <th>Sens ≥75%?</th>
                <th>FAR &lt;0.15/hr?</th>
                <th>PPV</th>
                <th>NPV</th>
                <th>AUC</th>
                <th>Clinically Viable</th>
              </tr>
            </thead>
            <tbody>
              {viability.map(v => (
                <tr key={v.horizon_label} className={v.viable ? 'table-success' : ''}>
                  <td><strong>{v.horizon_label}</strong></td>
                  <td>{v.sens_ok ? '✅' : '❌'}</td>
                  <td>{v.far_ok ? '✅' : '❌'}</td>
                  <td>{(v.ppv*100).toFixed(1)}%</td>
                  <td>{(v.npv*100).toFixed(1)}%</td>
                  <td>{v.auc.toFixed(3)}</td>
                  <td><ViabilityBadge viable={v.viable} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Metric descriptions */}
      <div className="card mb-2">
        <div className="card-header fw-semibold">&#x1f4cb; Metric Reference</div>
        <div className="card-body">
          <dl className="row mb-0">
            {Object.entries(descs).map(([term, def]) => (
              <span key={term}>
                <dt className="col-sm-3 text-capitalize">{term.replace(/_/g,' ')}</dt>
                <dd className="col-sm-9 small">{def}</dd>
              </span>
            ))}
          </dl>
        </div>
      </div>
    </div>
  );
}

function PatientPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const profiles = data.patient_profiles || [];
  if (!profiles.length) return <div className="text-muted">No patient data.</div>;

  return (
    <div className="card">
      <div className="card-header fw-semibold">&#x1f9d1;‍&#x2695;&#xfe0f; Per-Patient Best Horizon</div>
      <div className="card-body p-0">
        <table className="table table-sm table-hover mb-0">
          <thead className="table-light">
            <tr>
              <th>Patient</th>
              <th>Seizures</th>
              <th>Avg Severity</th>
              <th>Best Horizon</th>
              <th>Sensitivity</th>
              <th>FAR / hr</th>
              <th>Viable?</th>
            </tr>
          </thead>
          <tbody>
            {profiles.map(p => (
              <tr key={p.patient_id}>
                <td>P{p.patient_id}</td>
                <td>{p.n_seizures}</td>
                <td>{p.avg_severity?.toFixed(1)}</td>
                <td>{p.best_horizon_hr < 1 ? `${Math.round(p.best_horizon_hr*60)} min` : `${p.best_horizon_hr} hr`}</td>
                <td>
                  <span className={`fw-semibold text-${p.sensitivity_at_best >= 0.75 ? 'success' : 'warning'}`}>
                    {(p.sensitivity_at_best*100).toFixed(1)}%
                  </span>
                </td>
                <td>
                  <span className={`fw-semibold text-${p.far_at_best <= 0.15 ? 'success' : 'danger'}`}>
                    {p.far_at_best.toFixed(3)}/hr
                  </span>
                </td>
                <td><ViabilityBadge viable={p.viable} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const terms = data.terms || [];
  const standards = data.clinical_standards || [];

  return (
    <div>
      <div className="card mb-4">
        <div className="card-header fw-semibold">&#x1f4d6; Definitions</div>
        <div className="card-body">
          {terms.map(t => (
            <div key={t.term} className="mb-3">
              <div className="fw-semibold">{t.term}</div>
              <div className="text-muted small">{t.definition}</div>
            </div>
          ))}
        </div>
      </div>
      {standards.length > 0 && (
        <div className="card">
          <div className="card-header fw-semibold">&#x1f4da; Clinical Standards &amp; References</div>
          <div className="card-body">
            <ul className="mb-0">
              {standards.map((s, i) => <li key={i} className="small mb-1">{s}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default function SeizureHorizonPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/seizure-horizon/overview`).then(r => r.json()).then(setOverview).catch(e => setOverview({ error: String(e) }));
    fetch(`${API}/api/seizure-horizon/breakdown`).then(r => r.json()).then(setBreakdown).catch(e => setBreakdown({ error: String(e) }));
    fetch(`${API}/api/seizure-horizon/definitions`).then(r => r.json()).then(setDefinitions).catch(e => setDefinitions({ error: String(e) }));
  }, []);

  const patientData = breakdown
    ? { patient_profiles: breakdown.patient_profiles, error: breakdown.error }
    : null;

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: '1.6rem' }}>&#x23f1;&#xfe0f;</span>
        <div>
          <h4 className="mb-0 fw-bold">Seizure Prediction Horizon Analysis</h4>
          <div className="text-muted small">Sensitivity &amp; FAR/hr across 30 min → 24 hr prediction windows</div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-4">
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

      {tab === 'overview'    && <OverviewPanel    data={overview} />}
      {tab === 'breakdown'   && <BreakdownPanel   data={breakdown} />}
      {tab === 'patients'    && <PatientPanel     data={patientData} />}
      {tab === 'definitions' && <DefinitionsPanel data={definitions} />}
    </div>
  );
}
