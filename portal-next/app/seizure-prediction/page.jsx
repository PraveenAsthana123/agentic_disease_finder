'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Breakdown' },
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

function ConfusionMatrix({ tp, fp, tn, fn }) {
  return (
    <div className="card mb-3">
      <div className="card-header fw-semibold">Confusion Matrix</div>
      <div className="card-body">
        <table className="table table-bordered text-center mb-0" style={{ maxWidth: 350 }}>
          <thead className="table-light">
            <tr><th></th><th>Predicted +</th><th>Predicted &minus;</th></tr>
          </thead>
          <tbody>
            <tr><td className="fw-semibold">Actual +</td><td className="table-success">{tp}</td><td className="table-danger">{fn}</td></tr>
            <tr><td className="fw-semibold">Actual &minus;</td><td className="table-warning">{fp}</td><td className="table-success">{tn}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (!data.available) return <div className="alert alert-warning">{data.error || 'No data available'}</div>;

  const k = data.kpis || {};
  const riskHist = data.risk_score_distribution || [];
  const riskStats = data.risk_score_stats || {};
  const patients = data.patient_level_performance || [];
  const temporal = data.temporal_daily_pattern || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Prediction Windows" value={k.total_prediction_windows} color="info" sub="total monitoring windows" />
        <KPI label="Patients Monitored" value={k.unique_patients_monitored} color="primary" sub="unique patients" />
        <KPI label="Diary Events" value={k.diary_events_total} color="secondary" sub="confirmed seizures" />
        <KPI label="Prediction Horizon" value={k.avg_prediction_horizon_hours != null ? `${k.avg_prediction_horizon_hours}h` : '\u2014'} color="info" sub={`${k.predictions_with_horizon || 0} patients`} />
      </div>
      <div className="row mb-3">
        <KPI label="Sensitivity" value={k.sensitivity != null ? `${(k.sensitivity * 100).toFixed(1)}%` : '\u2014'} color={k.sensitivity >= 0.7 ? 'success' : 'warning'} sub="TP / (TP + FN)" />
        <KPI label="Specificity" value={k.specificity != null ? `${(k.specificity * 100).toFixed(1)}%` : '\u2014'} color={k.specificity >= 0.9 ? 'success' : 'warning'} sub="TN / (TN + FP)" />
        <KPI label="PPV" value={k.positive_predictive_value != null ? `${(k.positive_predictive_value * 100).toFixed(1)}%` : '\u2014'} color="primary" sub="positive predictive value" />
        <KPI label="FAR/hr" value={k.false_alarm_rate_per_hour != null ? k.false_alarm_rate_per_hour.toFixed(4) : '\u2014'} color={k.false_alarm_rate_per_hour <= 0.05 ? 'success' : 'danger'} sub="false alarms per hour" />
      </div>

      <ConfusionMatrix tp={k.true_positives} fp={k.false_positives} tn={k.true_negatives} fn={k.false_negatives} />

      {riskHist.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Risk Score Distribution</div>
          <div className="card-body">
            <div className="d-flex align-items-end gap-1" style={{ height: 120 }}>
              {riskHist.map((bin, i) => {
                const maxCount = Math.max(...riskHist.map(b => b.count), 1);
                const h = Math.max((bin.count / maxCount) * 100, 2);
                const color = i >= 7 ? '#dc3545' : i >= 5 ? '#ffc107' : '#198754';
                return (
                  <div key={bin.range} className="d-flex flex-column align-items-center flex-grow-1">
                    <div className="small text-muted mb-1">{bin.count}</div>
                    <div style={{ width: '100%', height: `${h}%`, backgroundColor: color, borderRadius: '3px 3px 0 0' }} />
                    <div className="small text-muted mt-1" style={{ fontSize: '0.65rem' }}>{bin.range}</div>
                  </div>
                );
              })}
            </div>
            <div className="mt-2 small text-muted">
              Mean: {riskStats.mean} | Std: {riskStats.std} | Min: {riskStats.min} | Max: {riskStats.max}
            </div>
          </div>
        </div>
      )}

      {temporal.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Day-of-Week Risk Pattern</div>
          <div className="card-body">
            <table className="table table-sm table-striped mb-0">
              <thead><tr><th>Day</th><th>Mean Risk</th><th>Readings</th><th>Seizures Detected</th></tr></thead>
              <tbody>
                {temporal.map(d => (
                  <tr key={d.day}>
                    <td>{d.day}</td>
                    <td>{d.mean_risk_score}</td>
                    <td>{d.readings}</td>
                    <td>{d.seizures_detected}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {patients.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Patient-Level Performance (top 20)</div>
          <div className="card-body" style={{ overflowX: 'auto' }}>
            <table className="table table-sm table-striped mb-0">
              <thead>
                <tr><th>Patient</th><th>Readings</th><th>Diary Events</th><th>TP</th><th>FP</th><th>Sensitivity</th><th>Specificity</th></tr>
              </thead>
              <tbody>
                {patients.map(p => (
                  <tr key={p.patient_id}>
                    <td className="font-monospace small">{p.patient_id}</td>
                    <td>{p.total_readings}</td>
                    <td>{p.diary_events}</td>
                    <td>{p.true_positives}</td>
                    <td>{p.false_positives}</td>
                    <td>{(p.sensitivity * 100).toFixed(1)}%</td>
                    <td>{(p.specificity * 100).toFixed(1)}%</td>
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

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (!data.available) return <div className="alert alert-warning">{data.error || 'No data available'}</div>;

  const patients = data.patient_breakdown || [];
  const corrs = data.feature_correlations || [];
  const preictal = data.preictal_biomarkers || [];
  const thresholds = data.threshold_analysis || [];

  return (
    <div>
      {corrs.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Feature-Risk Correlations</div>
          <div className="card-body">
            <table className="table table-sm table-striped mb-0">
              <thead><tr><th>Feature</th><th>Correlation</th><th>Interpretation</th><th>Samples</th></tr></thead>
              <tbody>
                {corrs.map(c => (
                  <tr key={c.key}>
                    <td>{c.feature}</td>
                    <td className={`fw-semibold ${c.correlation < 0 ? 'text-danger' : 'text-success'}`}>{c.correlation.toFixed(4)}</td>
                    <td><span className={`badge bg-${c.abs_correlation >= 0.5 ? 'danger' : c.abs_correlation >= 0.3 ? 'warning' : 'secondary'}`}>{c.interpretation}</span></td>
                    <td>{c.n_samples}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {preictal.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Pre-Ictal Biomarker Shifts (High-Risk vs Baseline)</div>
          <div className="card-body" style={{ overflowX: 'auto' }}>
            <table className="table table-sm table-striped mb-0">
              <thead>
                <tr><th>Biomarker</th><th>High-Risk Mean</th><th>Baseline Mean</th><th>% Change</th><th>Cohen's d</th><th>Effect Size</th></tr>
              </thead>
              <tbody>
                {preictal.map(b => (
                  <tr key={b.key}>
                    <td>{b.biomarker}</td>
                    <td>{b.seizure_day != null ? b.seizure_day.toFixed(2) : '\u2014'}</td>
                    <td>{b.non_seizure_day != null ? b.non_seizure_day.toFixed(2) : '\u2014'}</td>
                    <td className={b.percent_change > 0 ? 'text-danger' : 'text-success'}>{b.percent_change != null ? `${b.percent_change.toFixed(1)}%` : '\u2014'}</td>
                    <td>{b.cohens_d != null ? b.cohens_d.toFixed(3) : '\u2014'}</td>
                    <td><span className={`badge bg-${b.effect_size === 'large' ? 'danger' : b.effect_size === 'medium' ? 'warning' : 'secondary'}`}>{b.effect_size || '\u2014'}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {thresholds.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Threshold Operating Characteristics</div>
          <div className="card-body">
            <table className="table table-sm table-striped mb-0">
              <thead>
                <tr><th>Threshold</th><th>Sensitivity</th><th>Specificity</th><th>PPV</th><th>F1</th><th>Alarms</th><th>FAR/hr</th></tr>
              </thead>
              <tbody>
                {thresholds.map(t => (
                  <tr key={t.threshold}>
                    <td className="fw-semibold">{(t.threshold * 100).toFixed(0)}%</td>
                    <td>{(t.sensitivity * 100).toFixed(1)}%</td>
                    <td>{(t.specificity * 100).toFixed(1)}%</td>
                    <td>{(t.ppv * 100).toFixed(1)}%</td>
                    <td>{t.f1_score.toFixed(3)}</td>
                    <td>{t.total_alarms}</td>
                    <td>{t.false_alarm_rate_per_hour.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {patients.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Per-Patient Breakdown</div>
          <div className="card-body" style={{ overflowX: 'auto' }}>
            <table className="table table-sm table-striped mb-0">
              <thead>
                <tr><th>Patient</th><th>Diary Sz</th><th>Detections</th><th>TP</th><th>FP</th><th>FN</th><th>Accuracy</th><th>Mean Risk</th><th>Confidence</th></tr>
              </thead>
              <tbody>
                {patients.map(p => (
                  <tr key={p.patient_id}>
                    <td className="font-monospace small">{p.patient_id}</td>
                    <td>{p.diary_seizures}</td>
                    <td>{p.wearable_detections}</td>
                    <td>{p.true_positives}</td>
                    <td>{p.false_positives}</td>
                    <td>{p.false_negatives}</td>
                    <td>{(p.prediction_accuracy * 100).toFixed(1)}%</td>
                    <td>{p.mean_risk_score}</td>
                    <td>{(p.mean_detection_confidence * 100).toFixed(1)}%</td>
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

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (!data.available) return <div className="alert alert-warning">No definitions available</div>;
  const defs = data.definitions || [];
  return (
    <div>
      {defs.map((d, i) => (
        <div key={i} className="card mb-3">
          <div className="card-header fw-semibold">{d.title}</div>
          <div className="card-body"><p className="mb-0">{d.description}</p></div>
        </div>
      ))}
    </div>
  );
}

export default function SeizurePredictionPage() {
  const [tab, setTab] = useState('overview');
  const [data, setData] = useState({});

  useEffect(() => {
    if (data[tab]) return;
    fetch(`${API}/api/seizure-prediction/${tab}`)
      .then(r => r.json())
      .then(d => setData(prev => ({ ...prev, [tab]: d })))
      .catch(() => {});
  }, [tab]);

  return (
    <div className="container-fluid py-4">
      <h3 className="mb-3">Seizure Prediction Dashboard</h3>
      <p className="text-muted mb-4">Wearable-based seizure prediction analytics: sensitivity, specificity, risk scoring, pre-ictal biomarker analysis, and threshold tuning.</p>

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && <OverviewPanel data={data.overview} />}
      {tab === 'breakdown' && <BreakdownPanel data={data.breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={data.definitions} />}
    </div>
  );
}
