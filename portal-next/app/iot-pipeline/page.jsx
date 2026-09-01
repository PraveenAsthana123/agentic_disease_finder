'use client';
import { useState, useEffect, useCallback } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const decisionColor = d =>
  d === 'seizure' ? 'danger' : d === 'borderline' ? 'warning' : d === 'normal' ? 'success' : 'secondary';

const stageIcon = s =>
  ({ device: '📡', gateway: '🔗', ingest: '📥', features: '🧮', model: '🤖', decision: '⚖️', sos_alert: '🚨' })[s] || '▸';

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function PipelineTrace({ stages }) {
  if (!stages) return null;
  const order = ['device', 'gateway', 'ingest', 'features', 'model', 'decision', 'sos_alert'];
  return (
    <div className="d-flex flex-wrap align-items-center gap-1 my-3">
      {order.map((s, i) => {
        const val = stages[s];
        const color = val === 'ok' || val === 'not_triggered' ? 'success'
          : val === 'triggered' ? 'danger'
          : val === 'warn' ? 'warning'
          : 'info';
        return (
          <span key={s}>
            <span className={`badge bg-${color} px-2 py-1`} style={{ fontSize: '0.8rem' }}>
              {stageIcon(s)} {s.replace('_', ' ')} · {val || '—'}
            </span>
            {i < order.length - 1 && <span className="text-muted mx-1">→</span>}
          </span>
        );
      })}
    </div>
  );
}

function RunResult({ result, onClear }) {
  if (!result) return null;
  const isSeizure = result.decision === 'seizure';
  const isBorderline = result.decision === 'borderline';
  return (
    <div className={`alert alert-${decisionColor(result.decision)} mb-3`}>
      <div className="d-flex justify-content-between align-items-start">
        <div>
          <strong>
            {isSeizure ? '🚨 SEIZURE DETECTED' : isBorderline ? '⚠️ BORDERLINE ACTIVITY' : '✅ NORMAL EEG'}
          </strong>
          <div className="small mt-1">
            run_id: <code>{result.run_id?.slice(0, 8)}</code> ·
            device: <code>{result.device_id}</code> ·
            patient: <code>{result.patient_id || '—'}</code>
          </div>
          <div className="mt-1">
            Seizure probability: <strong>{result.seizure_prob != null ? `${(result.seizure_prob * 100).toFixed(1)}%` : '—'}</strong> ·
            Pipeline: <strong>{result.pipeline_ms} ms</strong>
            {result.sos_triggered && (
              <span className="ms-2 badge bg-danger">🚨 SOS triggered · alert: {result.alert_id}</span>
            )}
          </div>
          <PipelineTrace stages={result.stages} />
        </div>
        <button className="btn btn-sm btn-outline-secondary ms-2" onClick={onClear}>×</button>
      </div>
      {result.status === 'error' && (
        <div className="mt-2 small"><strong>Error at stage:</strong> {result.stage_failed} — {result.reason}</div>
      )}
    </div>
  );
}

function StatusPanel({ data }) {
  if (!data) return <div className="text-muted small">Loading pipeline status…</div>;
  const dd = data.decision_distribution || {};
  const stages = data.pipeline_stages || [];
  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Runs"     value={data.total_runs}    color="primary" sub="pipeline executions" />
        <KPI label="SOS Triggered"  value={data.sos_triggered} color="danger"  sub="seizure alerts fired" />
        <KPI label="SOS Rate"       value={`${data.sos_rate_pct ?? 0}%`} color="warning" sub="of all runs" />
        <KPI label="Avg Latency"    value={data.avg_pipeline_ms ? `${data.avg_pipeline_ms} ms` : '—'} color="info" sub="end-to-end" />
      </div>

      {data.total_runs === 0 && (
        <div className="alert alert-info">{data.note || 'No runs yet — use the Simulate or Ingest tabs to run the pipeline.'}</div>
      )}

      {Object.keys(dd).length > 0 && (
        <div className="mb-3">
          <h6 className="text-muted">Decision Distribution</h6>
          <div className="d-flex gap-2 flex-wrap">
            {Object.entries(dd).map(([d, n]) => (
              <span key={d} className={`badge bg-${decisionColor(d)} fs-6 px-3 py-2`}>{d}: {n}</span>
            ))}
          </div>
        </div>
      )}

      <div className="mb-3">
        <h6 className="text-muted">Pipeline Stages</h6>
        <div className="d-flex flex-wrap align-items-center gap-1">
          {stages.map((s, i) => (
            <span key={s}>
              <span className="badge bg-success px-2">{stageIcon(s)} {s.replace('_', ' ')}</span>
              {i < stages.length - 1 && <span className="text-muted mx-1">→</span>}
            </span>
          ))}
        </div>
      </div>

      {data.thresholds && (
        <div className="mb-3">
          <h6 className="text-muted">Decision Thresholds</h6>
          <table className="table table-sm table-bordered" style={{ maxWidth: 400 }}>
            <tbody>
              <tr><td>SOS / Seizure Alert</td><td className="fw-bold text-danger">≥ {(data.thresholds.sos * 100).toFixed(0)}%</td></tr>
              <tr><td>Borderline Warning</td><td className="fw-bold text-warning">≥ {(data.thresholds.borderline * 100).toFixed(0)}%</td></tr>
              <tr><td>Normal</td><td className="fw-bold text-success">&lt; {(data.thresholds.borderline * 100).toFixed(0)}%</td></tr>
            </tbody>
          </table>
        </div>
      )}

      {(data.recent_decisions || []).length > 0 && (
        <div>
          <h6 className="text-muted">Recent Pipeline Runs</h6>
          <table className="table table-sm table-striped table-bordered">
            <thead><tr><th>Run ID</th><th>Device</th><th>Decision</th><th>Prob</th><th>SOS</th><th>Timestamp</th></tr></thead>
            <tbody>
              {data.recent_decisions.map((r, i) => (
                <tr key={i}>
                  <td><code>{r.run_id}</code></td>
                  <td>{r.device_id}</td>
                  <td><span className={`badge bg-${decisionColor(r.decision)}`}>{r.decision}</span></td>
                  <td>{r.seizure_prob != null ? `${(r.seizure_prob * 100).toFixed(1)}%` : '—'}</td>
                  <td>{r.sos ? <span className="badge bg-danger">SOS</span> : <span className="badge bg-secondary">—</span>}</td>
                  <td className="small text-muted">{r.ts ? new Date(r.ts).toLocaleTimeString() : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function LogPanel({ data }) {
  if (!data) return <div className="text-muted small">Loading log…</div>;
  const entries = data.entries || [];
  if (entries.length === 0) return <div className="alert alert-info">No log entries yet.</div>;
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="table table-sm table-striped table-bordered small">
        <thead>
          <tr>
            <th>Run</th><th>Device</th><th>Stage</th><th>Status</th>
            <th>Prob</th><th>Decision</th><th>SOS</th><th>ms</th><th>Time</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e, i) => (
            <tr key={i}>
              <td><code>{e.run_id?.slice(0, 8)}</code></td>
              <td>{e.device_id}</td>
              <td><code>{e.stage}</code></td>
              <td><span className={`badge bg-${e.status === 'ok' || e.status === 'not_triggered' ? 'success' : e.status === 'triggered' ? 'danger' : e.status === 'error' ? 'danger' : 'warning'}`}>{e.status}</span></td>
              <td>{e.seizure_prob != null ? `${(e.seizure_prob * 100).toFixed(1)}%` : '—'}</td>
              <td>{e.decision ? <span className={`badge bg-${decisionColor(e.decision)}`}>{e.decision}</span> : '—'}</td>
              <td>{e.sos_triggered ? <span className="badge bg-danger">SOS</span> : '—'}</td>
              <td>{e.elapsed_ms ?? '—'}</td>
              <td className="text-muted">{e.received_at ? new Date(e.received_at).toLocaleTimeString() : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function IngestPanel() {
  const [deviceId,   setDeviceId]   = useState('DEV-001');
  const [patientId,  setPatientId]  = useState('P001');
  const [signalJson, setSignalJson] = useState('');
  const [running,    setRunning]    = useState(false);
  const [result,     setResult]     = useState(null);
  const [err,        setErr]        = useState('');

  const handleIngest = async () => {
    setErr(''); setResult(null); setRunning(true);
    try {
      let raw_signal;
      try { raw_signal = JSON.parse(signalJson); } catch { setErr('raw_signal must be valid JSON'); setRunning(false); return; }
      const resp = await fetch(`${API}/api/iot-pipeline/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device_id: deviceId, patient_id: patientId, raw_signal }),
      });
      const data = await resp.json();
      setResult(data);
    } catch (e) { setErr(String(e)); }
    setRunning(false);
  };

  return (
    <div>
      <p className="text-muted small">
        Send a real device EEG packet through the full 7-stage pipeline.
        raw_signal should be a JSON array of shape [14_channels × 1024_samples] or flat list.
      </p>
      {err && <div className="alert alert-danger small">{err}</div>}
      <RunResult result={result} onClear={() => setResult(null)} />
      <div className="row g-2 mb-3">
        <div className="col-md-4">
          <label className="form-label small fw-bold">Device ID</label>
          <input className="form-control form-control-sm" value={deviceId} onChange={e => setDeviceId(e.target.value)} />
        </div>
        <div className="col-md-4">
          <label className="form-label small fw-bold">Patient ID</label>
          <input className="form-control form-control-sm" value={patientId} onChange={e => setPatientId(e.target.value)} />
        </div>
      </div>
      <div className="mb-3">
        <label className="form-label small fw-bold">raw_signal (JSON array)</label>
        <textarea
          className="form-control form-control-sm font-monospace"
          rows={5}
          placeholder='[[0.1, 0.2, ...], [0.3, 0.4, ...], ...]  ← 14 channels × 1024 samples'
          value={signalJson}
          onChange={e => setSignalJson(e.target.value)}
        />
      </div>
      <button className="btn btn-primary btn-sm" onClick={handleIngest} disabled={running || !signalJson.trim()}>
        {running ? <><span className="spinner-border spinner-border-sm me-1" /> Running pipeline…</> : '▶ Run Pipeline'}
      </button>
    </div>
  );
}

function SimulatePanel() {
  const [deviceId,  setDeviceId]  = useState('DEV-SIM-001');
  const [patientId, setPatientId] = useState('P001');
  const [mode,      setMode]      = useState('normal');
  const [running,   setRunning]   = useState(false);
  const [result,    setResult]    = useState(null);
  const [err,       setErr]       = useState('');

  const handleSim = async () => {
    setErr(''); setResult(null); setRunning(true);
    try {
      const seizure = mode === 'seizure';
      const resp = await fetch(`${API}/api/iot-pipeline/simulate?seizure=${seizure}&device_id=${encodeURIComponent(deviceId)}&patient_id=${encodeURIComponent(patientId)}`);
      const data = await resp.json();
      setResult(data);
    } catch (e) { setErr(String(e)); }
    setRunning(false);
  };

  return (
    <div>
      <p className="text-muted small">
        Generate a synthetic EEG packet and run it through all 7 pipeline stages.
        Normal mode simulates alpha-dominant background; Seizure mode simulates
        ictal gamma + spike-wave activity.
      </p>
      {err && <div className="alert alert-danger small">{err}</div>}
      <RunResult result={result} onClear={() => setResult(null)} />
      <div className="row g-2 mb-3">
        <div className="col-md-3">
          <label className="form-label small fw-bold">Device ID</label>
          <input className="form-control form-control-sm" value={deviceId} onChange={e => setDeviceId(e.target.value)} />
        </div>
        <div className="col-md-3">
          <label className="form-label small fw-bold">Patient ID</label>
          <input className="form-control form-control-sm" value={patientId} onChange={e => setPatientId(e.target.value)} />
        </div>
        <div className="col-md-4">
          <label className="form-label small fw-bold">EEG Mode</label>
          <select className="form-select form-select-sm" value={mode} onChange={e => setMode(e.target.value)}>
            <option value="normal">Normal (alpha-dominant background)</option>
            <option value="seizure">Seizure (ictal gamma + spike-wave)</option>
          </select>
        </div>
      </div>
      <button className="btn btn-primary btn-sm" onClick={handleSim} disabled={running}>
        {running ? <><span className="spinner-border spinner-border-sm me-1" /> Simulating…</> : '▶ Simulate + Run Pipeline'}
      </button>
      <p className="text-muted mt-2" style={{ fontSize: '0.72rem' }}>
        Seizure mode → 14-ch gamma burst 35 Hz + spike-wave → expect seizure_prob ≥ 0.70 → SOS triggered.
        Normal mode → alpha 10 Hz → expect seizure_prob ≤ 0.10 → no alert.
      </p>
    </div>
  );
}

const TABS = [
  { id: 'status',   label: 'Status' },
  { id: 'simulate', label: 'Simulate' },
  { id: 'ingest',   label: 'Ingest (POST)' },
  { id: 'log',      label: 'Pipeline Log' },
];

export default function IoTPipelinePage() {
  const [status, setStatus] = useState(null);
  const [log,    setLog]    = useState(null);
  const [tab,    setTab]    = useState('status');

  const refresh = useCallback(() => {
    fetch(`${API}/api/iot-pipeline/status`).then(r => r.json()).then(setStatus).catch(() => {});
    fetch(`${API}/api/iot-pipeline/log?limit=100`).then(r => r.json()).then(setLog).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 30000);  // auto-refresh every 30 s
    return () => clearInterval(iv);
  }, [refresh]);

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-1">
        <h3>📡 IoT Continuous Monitoring Pipeline</h3>
        <button className="btn btn-outline-secondary btn-sm" onClick={refresh}>↺ Refresh</button>
      </div>
      <p className="text-muted small mb-3">
        End-to-end device → gateway → ingest → features → model → decision → SOS alert pipeline.
        14-channel wearable EEG | 256 Hz | 4-second windows | 47-feature extraction |
        heuristic seizure scorer | IEC 62304 class B.
      </p>

      {/* Stage pipeline diagram */}
      <div className="d-flex flex-wrap align-items-center gap-1 mb-3 p-2 bg-light rounded">
        {['device', 'gateway', 'ingest', 'features', 'model', 'decision', 'sos_alert'].map((s, i, arr) => (
          <span key={s}>
            <span className="badge bg-primary px-2 py-1">{stageIcon(s)} {s.replace('_', ' ')}</span>
            {i < arr.length - 1 && <span className="text-muted">→</span>}
          </span>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => { setTab(t.id); if (t.id === 'status' || t.id === 'log') refresh(); }}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'status'   && <StatusPanel   data={status} />}
      {tab === 'simulate' && <SimulatePanel />}
      {tab === 'ingest'   && <IngestPanel />}
      {tab === 'log'      && <LogPanel      data={log} />}
    </div>
  );
}
