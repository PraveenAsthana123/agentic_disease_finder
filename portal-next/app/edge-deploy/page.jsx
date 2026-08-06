'use client';
import { useEffect, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KpiCard({ label, value, color, sub }) {
  return (
    <div className="card text-center shadow-sm h-100">
      <div className="card-body py-3">
        <div className="fs-3 fw-bold" style={{ color: color || '#6366f1' }}>{value}</div>
        <div className="small fw-semibold text-muted">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
      </div>
    </div>
  );
}

const STATUS_COLOR = {
  validated: '#10b981', complete: '#10b981', baseline: '#6366f1',
  beta: '#06b6d4', experimental: '#f59e0b', ready: '#f59e0b',
  partial: '#f97316', planned: '#9ca3af', pending: '#9ca3af',
};
const statusBadge = s => (
  <span className="badge" style={{ background: STATUS_COLOR[s] || '#6b7280', fontSize: 11 }}>{s}</span>
);

export default function EdgeDeployDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/edge-deploy/overview`).then(r => r.json()),
      fetch(`${API}/api/edge-deploy/breakdown`).then(r => r.json()),
      fetch(`${API}/api/edge-deploy/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov) return (
    <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}>
      <div className="spinner-border text-primary" />
    </div>
  );

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'devices', label: '📟 Target Devices' },
    { id: 'models', label: '🧠 Models' },
    { id: 'quantization', label: '⚡ Quantization' },
    { id: 'definitions', label: '📚 Definitions' },
  ];

  const devices = ov.target_devices || [];
  const quant = ov.quantization_modes || [];
  const pipeline = ov.export_pipeline || [];
  const models = bd?.models || [];
  const terms = defs?.terms || [];

  const supportedDevices = devices.filter(d => d.supported).length;
  const validatedDevices = devices.filter(d => d.status === 'validated').length;

  return (
    <div className="container-fluid py-4">
      <h2 className="fw-bold mb-1">🚀 Edge Deployment Dashboard</h2>
      <p className="text-muted mb-3">
        ONNX export · quantization · edge device targeting ·{' '}
        {ov.total_sklearn_models} sklearn models · {ov.total_onnx_models} ONNX exported ·{' '}
        {supportedDevices} devices supported
      </p>

      {/* KPI row */}
      <div className="row g-3 mb-4">
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Sklearn Models" value={ov.total_sklearn_models} color="#6366f1" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="ONNX Exported" value={ov.total_onnx_models}
            color={ov.total_onnx_models > 0 ? '#10b981' : '#9ca3af'}
            sub={`${ov.onnx_coverage_pct}% coverage`} />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Target Devices" value={devices.length} color="#06b6d4"
            sub={`${supportedDevices} supported`} />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Validated Devices" value={validatedDevices} color="#10b981" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Quant Modes" value={quant.length} color="#f59e0b" />
        </div>
        <div className="col-6 col-md-3 col-xl-2">
          <KpiCard label="Pipeline Steps" value={pipeline.length} color="#8b5cf6"
            sub={`${pipeline.filter(s => s.status === 'complete').length} complete`} />
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div>
          <div className="row g-4">
            {/* Export Pipeline */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">⚙️ Export Pipeline</div>
                <div className="card-body p-0">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr><th>Step</th><th>Output</th><th>Status</th></tr>
                    </thead>
                    <tbody>
                      {pipeline.map((s, i) => (
                        <tr key={i}>
                          <td className="small fw-semibold">{s.step}</td>
                          <td className="small text-muted">{s.output}</td>
                          <td>{statusBadge(s.status)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Sklearn Models list */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">🧠 Sklearn Models on Disk</div>
                <div className="card-body">
                  {ov.sklearn_models?.length > 0 ? (
                    <ul className="list-group list-group-flush">
                      {ov.sklearn_models.map((m, i) => (
                        <li key={i} className="list-group-item d-flex justify-content-between align-items-center py-2">
                          <span className="small font-monospace">{m}</span>
                          {statusBadge(ov.onnx_models?.includes(m.replace('.joblib', '.onnx')) ? 'validated' : 'pending')}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-muted small">No sklearn models found in models/</p>
                  )}
                  {ov.total_onnx_models === 0 && (
                    <div className="alert alert-warning small mt-3 mb-0">
                      No ONNX models exported yet. Run <code>python3 scripts/export_onnx.py</code> to convert sklearn models.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Devices tab */}
      {tab === 'devices' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">📟 Target Edge Devices</div>
          <div className="card-body p-0">
            <table className="table table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Device</th>
                  <th>Architecture</th>
                  <th>RAM</th>
                  <th>Runtime</th>
                  <th>Latency</th>
                  <th>Supported</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {devices.map((d, i) => (
                  <tr key={i}>
                    <td className="fw-semibold small">{d.name}</td>
                    <td className="small text-muted">{d.arch}</td>
                    <td className="small">{d.ram_mb >= 1024 ? `${d.ram_mb / 1024} GB` : `${d.ram_mb} MB`}</td>
                    <td><code className="small">{d.runtime}</code></td>
                    <td className="small">
                      {d.latency_ms != null
                        ? <span className="fw-semibold" style={{ color: d.latency_ms < 20 ? '#10b981' : d.latency_ms < 50 ? '#f59e0b' : '#ef4444' }}>{d.latency_ms} ms</span>
                        : <span className="text-muted">—</span>}
                    </td>
                    <td>
                      {d.supported
                        ? <span className="badge bg-success">✓ Yes</span>
                        : <span className="badge bg-secondary">No</span>}
                    </td>
                    <td>{statusBadge(d.status)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Models tab */}
      {tab === 'models' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">🧠 Per-Model ONNX Status</div>
          <div className="card-body p-0">
            <table className="table table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Model</th>
                  <th>Sklearn Size</th>
                  <th>ONNX Exported</th>
                  <th>ONNX Size</th>
                  <th>Size Reduction</th>
                  <th>Edge Compatible</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <tr key={i}>
                    <td className="fw-semibold small">{m.name}</td>
                    <td className="small">{m.sklearn_size_kb?.toFixed(1)} KB</td>
                    <td>
                      {m.onnx_exported
                        ? <span className="badge bg-success">✓ Yes</span>
                        : <span className="badge bg-secondary">No</span>}
                    </td>
                    <td className="small">{m.onnx_size_kb > 0 ? `${m.onnx_size_kb?.toFixed(1)} KB` : '—'}</td>
                    <td className="small">
                      {m.size_reduction_pct != null
                        ? <span style={{ color: '#10b981' }}>{m.size_reduction_pct}%</span>
                        : <span className="text-muted">—</span>}
                    </td>
                    <td>
                      {m.edge_compatible
                        ? <span className="badge bg-success">✓</span>
                        : <span className="badge bg-warning text-dark">Pending ONNX</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="card-footer text-muted small">
            Run <code>python3 scripts/export_onnx.py</code> to export sklearn→ONNX. Models: {models.length} total.
          </div>
        </div>
      )}

      {/* Quantization tab */}
      {tab === 'quantization' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">⚡ Quantization Modes</div>
          <div className="card-body p-0">
            <table className="table table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Mode</th>
                  <th>Size Reduction</th>
                  <th>Accuracy Delta</th>
                  <th>Latency Factor</th>
                  <th>Speed-up</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {quant.map((q, i) => {
                  const speedup = q.latency_factor > 0 ? (1 / q.latency_factor).toFixed(1) : '—';
                  return (
                    <tr key={i}>
                      <td className="fw-semibold small">{q.mode}</td>
                      <td className="small">{q.size_reduction}</td>
                      <td className="small" style={{ color: q.accuracy_delta === '0.0%' ? '#10b981' : q.accuracy_delta?.startsWith('-') ? '#ef4444' : '#6b7280' }}>
                        {q.accuracy_delta}
                      </td>
                      <td className="small">{q.latency_factor}×</td>
                      <td className="small">
                        {speedup !== '—' && (
                          <span style={{ color: '#10b981', fontWeight: 600 }}>{speedup}× faster</span>
                        )}
                      </td>
                      <td>{statusBadge(q.status)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="card-footer text-muted small">
            INT8 dynamic quantization gives 75% size reduction with only −0.8% accuracy. Recommended for Raspberry Pi / Jetson targets.
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && (
        <div className="row g-3">
          {terms.map((t, i) => (
            <div key={i} className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold text-primary mb-1">{t.term}</h6>
                  <p className="text-muted small mb-0">{t.definition}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
