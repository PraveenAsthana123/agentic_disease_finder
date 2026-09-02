'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const VERDICT_META = {
  excellent:   { color: '#22c55e', label: 'Excellent' },
  acceptable:  { color: '#3b82f6', label: 'Acceptable' },
  high:        { color: '#f97316', label: 'High FA Rate' },
  unacceptable:{ color: '#ef4444', label: 'Unacceptable' },
};

function KPI({ label, value, sub, color }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div style={{ fontSize: 26, fontWeight: 700, color: color || '#3b82f6' }}>{value ?? '—'}</div>
          <div className="text-muted small fw-semibold">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.68rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

export default function EegVizFalseAlarmPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/eeg-viz/false-alarm`)
      .then(r => r.json())
      .then(setData)
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!data) return <div className="text-muted p-4 text-center"><div className="spinner-border text-primary mb-2" /><div>Loading false alarm data…</div></div>;

  const verdictMeta = VERDICT_META[data.verdict] || { color: '#94a3b8', label: data.verdict };
  const sensitivity_pct = (data.sensitivity * 100).toFixed(1);
  const precision = data.true_positive_windows
    ? (data.true_positive_windows / (data.true_positive_windows + data.false_alarms) * 100).toFixed(1)
    : '—';

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: 28 }}>🚨</span>
        <div>
          <h4 className="mb-0 fw-bold">False Alarm Review</h4>
          <div className="text-muted small">
            Seizure detector FP/FN analysis · {data.file} · {data.recording_hours}h recording
          </div>
          <div className="text-muted" style={{ fontSize: '0.68rem' }}>{data.note}</div>
        </div>
        <span className="ms-auto badge" style={{ background: verdictMeta.color, fontSize: 11 }}>
          {verdictMeta.label}
        </span>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Sensitivity" value={`${sensitivity_pct}%`} sub="seizures detected" color="#22c55e" />
        <KPI label="False Alarms" value={data.false_alarms} sub={`${data.false_alarms_per_hour}/hr`} color="#ef4444" />
        <KPI label="True Positives" value={data.true_positive_windows} sub="windows" color="#3b82f6" />
        <KPI label="Precision" value={`${precision}%`} sub="TP/(TP+FP)" color="#8b5cf6" />
      </div>

      {/* Detector method */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-semibold">Detector Configuration</div>
        <div className="card-body small">
          <div className="row">
            <div className="col-md-6">
              <table className="table table-sm mb-0">
                <tbody>
                  <tr><td className="text-muted">Method</td><td>{data.detector?.method}</td></tr>
                  <tr><td className="text-muted">Threshold K</td><td>{data.detector?.threshold_k}</td></tr>
                  <tr><td className="text-muted">Seizures annotated</td><td>{data.n_seizures_annotated}</td></tr>
                  <tr><td className="text-muted">Seizures detected</td><td>{data.seizures_detected}</td></tr>
                  <tr><td className="text-muted">Recording duration</td><td>{data.recording_hours}h</td></tr>
                  <tr><td className="text-muted">Source</td><td>{data.source}</td></tr>
                </tbody>
              </table>
            </div>
            <div className="col-md-6">
              {/* Sensitivity vs FA tradeoff visual */}
              <div className="p-3 rounded" style={{ background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                <div className="fw-semibold small mb-2">Performance Summary</div>
                <div className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>Sensitivity</span><span className="fw-bold text-success">{sensitivity_pct}%</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar bg-success" style={{ width: `${sensitivity_pct}%` }} />
                  </div>
                </div>
                <div className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>Precision</span><span className="fw-bold text-primary">{precision}%</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar bg-primary" style={{ width: `${precision}%` }} />
                  </div>
                </div>
                <div className="small text-muted mt-2">
                  Verdict: <span className="fw-bold" style={{ color: verdictMeta.color }}>{verdictMeta.label}</span>
                  {' '}({data.false_alarms_per_hour} FA/hr · clinical target: &lt;1/hr)
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* False alarm windows */}
      <div className="card shadow-sm">
        <div className="card-header fw-semibold">
          False Alarm Windows ({data.false_alarm_windows?.length || 0})
        </div>
        {data.false_alarm_windows?.length > 0 ? (
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr><th>#</th><th>Window index</th><th>Time (s)</th><th>Time (mm:ss)</th></tr>
              </thead>
              <tbody>
                {(data.false_alarm_windows || []).map((w, i) => (
                  <tr key={i}>
                    <td className="small text-muted">{i + 1}</td>
                    <td className="fw-semibold small">{w.window}</td>
                    <td className="small">{w.time_s}s</td>
                    <td className="small text-muted">
                      {Math.floor(w.time_s / 60)}:{String(Math.round(w.time_s % 60)).padStart(2, '0')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="card-body text-muted small">No false alarms detected.</div>
        )}
        <div className="card-footer small text-muted">
          Each window = 5s · False alarm = detector triggered with no annotated seizure overlap.
          True positive windows: {data.true_positive_windows} · False alarms: {data.false_alarms}
        </div>
      </div>
    </div>
  );
}
