'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '\u2014'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function RiskBar({ score }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.65 ? 'danger' : score >= 0.45 ? 'warning' : 'success';
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 12 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="small fw-semibold" style={{ minWidth: 40 }}>{pct}%</span>
    </div>
  );
}

function RiskBadge({ level }) {
  const color = level === 'High' ? 'danger' : level === 'Medium' ? 'warning' : 'success';
  return <span className={`badge bg-${color}`}>{level}</span>;
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const k = data.kpis || {};
  const sevDist = data.severity_distribution || {};
  const trigDist = data.trigger_distribution || {};
  const risks = data.risk_scores || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Patients" value={k.n_patients} color="info" sub="with seizure diary" />
        <KPI label="Seizure Events" value={k.n_seizure_events} color="primary" sub="total recorded" />
        <KPI label="Sensitivity" value={k.sensitivity != null ? `${(k.sensitivity * 100).toFixed(1)}%` : '\u2014'} color={k.sensitivity >= 0.8 ? 'success' : 'warning'} sub={`${k.horizon_hours}h horizon`} />
        <KPI label="FAR" value={k.far_per_hour != null ? `${k.far_per_hour}/hr` : '\u2014'} color={k.far_per_hour <= 0.15 ? 'success' : 'danger'} sub="false alarm rate" />
      </div>
      <div className="row mb-3">
        <KPI label="Specificity" value={k.specificity != null ? `${(k.specificity * 100).toFixed(1)}%` : '\u2014'} color={k.specificity >= 0.85 ? 'success' : 'warning'} sub="true negative rate" />
        <KPI label="AUC-ROC" value={k.auc_roc != null ? k.auc_roc.toFixed(3) : '\u2014'} color={k.auc_roc >= 0.8 ? 'success' : 'warning'} sub="discriminative ability" />
        <KPI label="Mean ISI" value={k.mean_isi_days != null ? `${k.mean_isi_days}d` : '\u2014'} color="secondary" sub="inter-seizure interval" />
        <KPI label="Median ISI" value={k.median_isi_days != null ? `${k.median_isi_days}d` : '\u2014'} color="secondary" sub="inter-seizure interval" />
      </div>

      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Severity Distribution</div>
            <div className="card-body">
              {Object.entries(sevDist).map(([sev, count]) => (
                <div key={sev} className="d-flex justify-content-between mb-2">
                  <span className={`badge bg-${sev === 'Severe' ? 'danger' : sev === 'Mild' ? 'success' : 'secondary'}`}>{sev}</span>
                  <span className="fw-semibold">{count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Trigger Distribution</div>
            <div className="card-body">
              {Object.entries(trigDist).sort((a, b) => b[1] - a[1]).map(([trig, count]) => (
                <div key={trig} className="d-flex justify-content-between mb-2">
                  <span>{trig}</span>
                  <span className="fw-semibold">{count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <div className="card-header fw-semibold">Patient Risk Scores (highest first)</div>
        <div className="card-body">
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead><tr><th>Patient</th><th>Risk</th><th>Seizures</th><th>Severe</th></tr></thead>
              <tbody>
                {risks.slice(0, 15).map(r => (
                  <tr key={r.patient_id}>
                    <td className="fw-semibold">{r.patient_id}</td>
                    <td style={{ minWidth: 160 }}><RiskBar score={r.risk_score} /></td>
                    <td>{r.seizure_count}</td>
                    <td>{r.severe_count}</td>
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

function PatientsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const profiles = data.patient_profiles || [];

  return (
    <div>
      <p className="text-muted mb-3">{data.total_patients} patients with seizure forecasting profiles</p>
      <div className="row">
        {profiles.map(p => (
          <div key={p.patient_id} className="col-md-6 col-lg-4 mb-3">
            <div className="card h-100 shadow-sm">
              <div className="card-header d-flex justify-content-between align-items-center">
                <span className="fw-semibold">{p.patient_id}</span>
                <RiskBadge level={p.risk_level} />
              </div>
              <div className="card-body">
                <div className="mb-2"><RiskBar score={p.risk_score} /></div>
                <div className="row small mb-2">
                  <div className="col-6"><strong>Seizures:</strong> {p.total_seizures}</div>
                  <div className="col-6"><strong>Severe:</strong> {p.severe_count}</div>
                </div>
                <div className="row small mb-2">
                  <div className="col-6"><strong>Mean Dur:</strong> {p.mean_duration_sec != null ? `${p.mean_duration_sec}s` : '\u2014'}</div>
                  <div className="col-6"><strong>Mean ISI:</strong> {p.mean_isi_days != null ? `${p.mean_isi_days}d` : '\u2014'}</div>
                </div>
                {p.temporal_pattern && (
                  <div className="small mb-2"><strong>Pattern:</strong> {p.temporal_pattern}</div>
                )}
                {p.medications && p.medications.length > 0 && (
                  <div className="small mb-2"><strong>Meds:</strong> {p.medications.join(', ')}</div>
                )}
                <div className="small">
                  <strong>Triggers:</strong>{' '}
                  {Object.entries(p.trigger_profile || {}).map(([t, c]) => `${t} (${c})`).join(', ')}
                </div>
                {p.last_event && (
                  <div className="small text-muted mt-1">Last event: {p.last_event}</div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <div className="card mb-4">
        <div className="card-header fw-semibold">Terms &amp; Definitions</div>
        <div className="card-body">
          {(data.terms || []).map((t, i) => (
            <div key={i} className="mb-3">
              <strong>{t.term}</strong>
              <p className="mb-0 text-muted small">{t.definition}</p>
            </div>
          ))}
        </div>
      </div>
      <div className="card mb-4">
        <div className="card-header fw-semibold">References</div>
        <ul className="list-group list-group-flush">
          {(data.references || []).map((r, i) => (
            <li key={i} className="list-group-item small">{r}</li>
          ))}
        </ul>
      </div>
      <div className="card mb-4">
        <div className="card-header fw-semibold">Clinical Notes</div>
        <ul className="list-group list-group-flush">
          {(data.clinical_notes || []).map((n, i) => (
            <li key={i} className="list-group-item small">{n}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default function SeizureForecastingPage() {
  const [tab, setTab] = useState('overview');
  const [data, setData] = useState({});

  useEffect(() => {
    fetch(`${API}/api/seizure-forecasting/${tab}`).then(r => r.json()).then(d => setData(prev => ({ ...prev, [tab]: d }))).catch(() => {});
  }, [tab]);

  return (
    <div className="container-fluid py-4">
      <h2 className="mb-1">Seizure Forecasting Dashboard</h2>
      <p className="text-muted mb-3">Seizure risk prediction from diary history, temporal patterns, and clinical variables</p>

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && <OverviewPanel data={data.overview} />}
      {tab === 'patients' && <PatientsPanel data={data.patients} />}
      {tab === 'definitions' && <DefinitionsPanel data={data.definitions} />}
    </div>
  );
}
