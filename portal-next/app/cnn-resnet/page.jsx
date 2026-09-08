'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',      label: 'Overview' },
  { id: 'spectrograms',  label: 'Spectrogram Inventory' },
  { id: 'architectures', label: 'Architectures' },
  { id: 'definitions',   label: 'Definitions' },
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

function PctBar({ pct, color }) {
  const c = color || (pct >= 80 ? 'success' : pct >= 60 ? 'info' : pct >= 40 ? 'warning' : 'danger');
  return (
    <div className="progress" style={{ height: 13, borderRadius: 6 }}>
      <div
        className={`progress-bar bg-${c}`}
        style={{ width: `${Math.min(pct, 100)}%`, borderRadius: 6, transition: 'width 0.5s' }}
      />
    </div>
  );
}

function SeverityBadge({ status }) {
  const map = { 'Severe imbalance': 'danger', 'Moderate imbalance': 'warning', 'Balanced': 'success', ok: 'success', warning: 'warning', error: 'danger' };
  return <span className={`badge bg-${map[status] || 'secondary'}`}>{status}</span>;
}

/* ── Overview Tab ─────────────────────────────────────────────────────── */
function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;

  const kpis      = ov.kpis || [];
  const qualDist  = ov.quality_distribution || [];
  const classDist = ov.classification_chart || [];
  const bands     = ov.band_power_chart || [];
  const daily     = ov.daily_activity || [];
  const benches   = ov.literature_benchmarks || [];
  const freqRadar = ov.freq_radar || [];

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-4">
        {kpis.slice(0, 8).map((k, i) => (
          <KPI
            key={i}
            label={k.label}
            value={k.value}
            color={['primary','success','info','warning','secondary','dark','danger','primary'][i]}
            sub={k.sub}
          />
        ))}
      </div>

      <div className="row mb-4">
        {/* Signal Quality */}
        <div className="col-md-5 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>Signal Quality Distribution</strong>
            </div>
            <div className="card-body">
              {qualDist.map((q, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-semibold">{q.quality}</span>
                    <span className="small text-muted">{q.count}</span>
                  </div>
                  <div className="progress" style={{ height: 12, borderRadius: 6 }}>
                    <div
                      className="progress-bar"
                      style={{
                        width: `${Math.round(q.count / Math.max(ov.total_analyses || 1, 1) * 100)}%`,
                        backgroundColor: q.color,
                        borderRadius: 6,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Classification chart */}
        <div className="col-md-7 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>CNN/ResNet Classification Distribution</strong>
            </div>
            <div className="card-body">
              {classDist.map((c, i) => (
                <div key={i} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-semibold">{c.predicted_label}</span>
                    <span className="small text-muted">{c.count} · {(c.mean_confidence * 100).toFixed(1)}% conf</span>
                  </div>
                  <PctBar
                    pct={Math.round(c.count / Math.max(ov.total_analyses || 1, 1) * 100)}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Band Power & Spectrogram info */}
      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>EEG Frequency Band Power (mean)</strong>
            </div>
            <div className="card-body">
              {bands.map((b, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-semibold">{b.band}</span>
                    <span className="small text-muted">{b.range} · mean {b.mean_power}</span>
                  </div>
                  <div className="progress" style={{ height: 11, borderRadius: 6 }}>
                    <div
                      className="progress-bar"
                      style={{
                        width: `${Math.min(b.mean_power * 500, 100)}%`,
                        backgroundColor: b.color,
                        borderRadius: 6,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Freq Radar / Summary */}
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>Spectrogram Config Summary</strong>
            </div>
            <div className="card-body">
              <table className="table table-sm table-bordered mb-0">
                <tbody>
                  {[
                    ['Total Analyses',        ov.total_analyses],
                    ['Spectrograms Generated', ov.total_spectrograms],
                    ['Spectrogram Resolution', ov.spec_resolution],
                    ['Sampling Rate (Hz)',     ov.mean_sampling_rate],
                    ['Mean Confidence',        `${((ov.mean_confidence || 0) * 100).toFixed(1)}%`],
                    ['Patients Analyzed',      ov.patients_analyzed],
                    ['Seizure Events',         ov.seizure_events],
                    ['Pipeline Events',        ov.pipeline_events],
                  ].map(([k, v], i) => (
                    <tr key={i}>
                      <td className="small fw-semibold text-muted">{k}</td>
                      <td className="small">{String(v ?? '—')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Literature benchmarks */}
      {benches.length > 0 && (
        <div className="card shadow-sm mb-4">
          <div className="card-header py-2 bg-dark text-white">
            <strong>Literature Benchmarks — CHB-MIT / TUH Dataset</strong>
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>Dataset</th>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Sensitivity</th>
                    <th>Specificity</th>
                    <th>Reference</th>
                  </tr>
                </thead>
                <tbody>
                  {benches.map((b, i) => (
                    <tr key={i}>
                      <td className="small">{b.dataset}</td>
                      <td className="small fw-semibold">{b.model}</td>
                      <td className="small text-success fw-bold">{(b.accuracy * 100).toFixed(1)}%</td>
                      <td className="small text-primary">{(b.sensitivity * 100).toFixed(1)}%</td>
                      <td className="small text-info">{(b.specificity * 100).toFixed(1)}%</td>
                      <td className="small text-muted">{b.ref}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Daily activity */}
      {daily.length > 0 && (
        <div className="card shadow-sm">
          <div className="card-header py-2 bg-dark text-white">
            <strong>Pipeline Activity (spectrogram generation)</strong>
          </div>
          <div className="card-body">
            {daily.map((d, i) => (
              <div key={i} className="d-flex align-items-center mb-2 gap-2">
                <span className="small text-muted" style={{ minWidth: 90 }}>{d.date}</span>
                <div className="progress flex-grow-1" style={{ height: 12, borderRadius: 6 }}>
                  <div
                    className="progress-bar bg-primary"
                    style={{
                      width: `${Math.min(d.spectrograms / Math.max(...daily.map(x => x.spectrograms || 1)) * 100, 100)}%`,
                      borderRadius: 6,
                    }}
                  />
                </div>
                <span className="small text-muted" style={{ minWidth: 30 }}>{d.spectrograms}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Spectrogram Inventory Tab ────────────────────────────────────────── */
function SpectrogramsPanel({ bk }) {
  if (!bk) return <div className="text-muted p-3">Loading…</div>;

  const specs    = bk.spectrogram_inventory || [];
  const patients = bk.patient_profiles || [];
  const tr       = bk.training_readiness || {};
  const perClass = tr.per_disease || [];
  const pl       = bk.pipeline_log || [];

  return (
    <div>
      {/* Training Readiness */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>CNN/ResNet Training Readiness</strong>
        </div>
        <div className="card-body">
          <div className="row mb-3">
            {[
              ['Total Spectrograms',  tr.total_spectrograms],
              ['Usable (Good)',       tr.usable_spectrograms],
              ['Usable Ratio',        tr.usable_ratio ? `${(tr.usable_ratio * 100).toFixed(0)}%` : '—'],
              ['Classes',             tr.n_classes],
              ['Balance Ratio',       tr.class_balance_ratio],
            ].map(([k, v], i) => (
              <div key={i} className="col-6 col-md-2 mb-2 text-center">
                <div className="small text-muted">{k}</div>
                <div className="fw-bold">{v ?? '—'}</div>
              </div>
            ))}
            <div className="col-6 col-md-2 mb-2 text-center">
              <div className="small text-muted">Balance Status</div>
              <SeverityBadge status={tr.balance_status} />
            </div>
          </div>
          {/* Per-class distribution */}
          {perClass.length > 0 && (
            <div>
              <div className="small fw-semibold text-muted mb-2">Per-class distribution:</div>
              {perClass.map((c, i) => (
                <div key={i} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-semibold">{c.disease}</span>
                    <span className="small text-muted">{c.spectrograms} · {c.pct}%</span>
                  </div>
                  <PctBar pct={c.pct} />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Spectrogram inventory table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Spectrogram Inventory ({specs.length} recordings)</strong>
        </div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped mb-0">
              <thead className="table-dark">
                <tr>
                  <th>#</th>
                  <th>File</th>
                  <th>Patient</th>
                  <th>Disease</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                  <th>Quality</th>
                  <th>SR (Hz)</th>
                  <th>Ch</th>
                </tr>
              </thead>
              <tbody>
                {specs.map((s, i) => (
                  <tr key={i}>
                    <td className="small text-muted">{s.id}</td>
                    <td className="small">{s.file_name}</td>
                    <td className="small">{s.patient_id}</td>
                    <td className="small">{s.disease}</td>
                    <td className="small fw-semibold">{s.predicted_label}</td>
                    <td className="small">
                      <PctBar pct={s.confidence * 100} />
                      <span className="text-muted">{(s.confidence * 100).toFixed(1)}%</span>
                    </td>
                    <td className="small">
                      <span
                        className="badge"
                        style={{
                          backgroundColor:
                            s.signal_quality === 'Good' ? '#10b981' :
                            s.signal_quality === 'Fair' ? '#f59e0b' :
                            s.signal_quality === 'Excellent' ? '#6b7280' : '#ef4444',
                        }}
                      >
                        {s.signal_quality}
                      </span>
                    </td>
                    <td className="small">{s.sampling_rate || '—'}</td>
                    <td className="small">{s.n_channels || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Patient profiles */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Per-Patient CNN/ResNet Profiles ({patients.length})</strong>
        </div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped mb-0">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Age</th>
                  <th>Sex</th>
                  <th>Diagnosis</th>
                  <th>Spectrograms</th>
                  <th>Confidence</th>
                  <th>Seizures</th>
                  <th>AEDs</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i}>
                    <td className="small fw-semibold">{p.patient_id}</td>
                    <td className="small">{p.age}</td>
                    <td className="small">{p.sex}</td>
                    <td className="small">{p.diagnosis}</td>
                    <td className="small">{p.n_spectrograms}</td>
                    <td className="small">
                      <span className={`text-${(p.mean_confidence || 0) >= 0.8 ? 'success' : (p.mean_confidence || 0) >= 0.6 ? 'warning' : 'danger'} fw-bold`}>
                        {((p.mean_confidence || 0) * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="small">{p.n_seizures}</td>
                    <td className="small">{p.aed_names?.join(', ') || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Pipeline log */}
      {pl.length > 0 && (
        <div className="card shadow-sm">
          <div className="card-header py-2 bg-dark text-white">
            <strong>Pipeline Log (recent events)</strong>
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>Timestamp</th>
                    <th>Event</th>
                    <th>Patient</th>
                    <th>Stage</th>
                  </tr>
                </thead>
                <tbody>
                  {pl.slice(0, 20).map((e, i) => (
                    <tr key={i}>
                      <td className="small text-muted">{e.created_at?.slice(0, 16)}</td>
                      <td className="small">{e.event_type}</td>
                      <td className="small">{e.patient_id}</td>
                      <td className="small">{e.pipeline_stage}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Architectures Tab ────────────────────────────────────────────────── */
function ArchitecturesPanel({ bk }) {
  if (!bk) return <div className="text-muted p-3">Loading…</div>;
  const archs = bk.model_architecture || [];
  const [selected, setSelected] = useState(0);
  const arch = archs[selected];

  return (
    <div>
      {/* Variant selector */}
      <div className="d-flex flex-wrap gap-2 mb-3">
        {archs.map((a, i) => (
          <button
            key={i}
            className={`btn btn-sm ${selected === i ? 'btn-primary' : 'btn-outline-primary'}`}
            onClick={() => setSelected(i)}
          >
            {a.key}
          </button>
        ))}
      </div>

      {/* Comparison table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Architecture Comparison — CNN / ResNet Variants</strong>
        </div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped mb-0">
              <thead className="table-dark">
                <tr>
                  <th>Model</th>
                  <th>Params</th>
                  <th>Trainable</th>
                  <th>Layers</th>
                  <th>Input Shape</th>
                  <th>Classes</th>
                  <th>Optimizer</th>
                  <th>Loss</th>
                </tr>
              </thead>
              <tbody>
                {archs.map((a, i) => (
                  <tr
                    key={i}
                    className={selected === i ? 'table-primary' : ''}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setSelected(i)}
                  >
                    <td className="small fw-semibold">{a.key}</td>
                    <td className="small">{(a.total_params || 0).toLocaleString()}</td>
                    <td className="small">{(a.trainable_params || 0).toLocaleString()}</td>
                    <td className="small">{a.n_layers}</td>
                    <td className="small font-monospace">{a.input_shape}</td>
                    <td className="small">{a.output_classes}</td>
                    <td className="small text-muted">{a.optimizer}</td>
                    <td className="small text-muted">{a.loss}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Selected arch detail */}
      {arch && (
        <div className="row">
          <div className="col-md-5 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header py-2 bg-dark text-white">
                <strong>{arch.name}</strong>
              </div>
              <div className="card-body">
                <p className="small text-muted">{arch.description}</p>
                <table className="table table-sm table-bordered mb-0">
                  <tbody>
                    {[
                      ['Total Params',     (arch.total_params || 0).toLocaleString()],
                      ['Trainable Params', (arch.trainable_params || 0).toLocaleString()],
                      ['Layers',           arch.n_layers],
                      ['Input Channels',   arch.input_channels],
                      ['Output Classes',   arch.output_classes],
                      ['Optimizer',        arch.optimizer],
                      ['Loss',             arch.loss],
                      ['Batch Size',       arch.batch_size],
                    ].map(([k, v], i) => (
                      <tr key={i}>
                        <td className="small text-muted fw-semibold">{k}</td>
                        <td className="small">{v ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-7 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header py-2 bg-dark text-white">
                <strong>Layer Stack — {arch.key}</strong>
              </div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-secondary">
                      <tr>
                        <th>#</th>
                        <th>Layer Type</th>
                        <th>Details</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(arch.layers || []).map((l, i) => (
                        <tr key={i}>
                          <td className="small text-muted">{i + 1}</td>
                          <td className="small fw-semibold">{l.type}</td>
                          <td className="small text-muted font-monospace">
                            {Object.entries(l)
                              .filter(([k]) => k !== 'type')
                              .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                              .join(' · ') || '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Definitions Tab ──────────────────────────────────────────────────── */
function DefinitionsPanel({ df }) {
  if (!df) return <div className="text-muted p-3">Loading…</div>;

  const concepts  = df.concepts || [];
  const metrics   = df.quality_metrics || [];
  const variants  = df.model_variants || [];
  const comp      = df.compliance || [];
  const remed     = df.remediation || [];

  return (
    <div>
      <div className="row mb-3">
        <div className="col-lg-7 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>CNN / ResNet Concepts ({concepts.length})</strong>
            </div>
            <div className="card-body" style={{ maxHeight: 420, overflowY: 'auto' }}>
              {concepts.map((c, i) => (
                <div key={i} className="mb-3 pb-3 border-bottom">
                  <div className="fw-semibold small mb-1">{c.term}</div>
                  <div className="text-muted small">{c.definition}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-lg-5 mb-3">
          {/* Model variants */}
          {variants.length > 0 && (
            <div className="card shadow-sm mb-3">
              <div className="card-header py-2 bg-dark text-white">
                <strong>Model Variants</strong>
              </div>
              <div className="card-body">
                {variants.map((v, i) => (
                  <div key={i} className="mb-3 pb-2 border-bottom">
                    <div className="d-flex justify-content-between">
                      <span className="fw-semibold small">{v.name}</span>
                      <span className="badge bg-secondary">{v.params}</span>
                    </div>
                    <div className="text-muted small mt-1">{v.description}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Quality metrics */}
          {metrics.length > 0 && (
            <div className="card shadow-sm mb-3">
              <div className="card-header py-2 bg-dark text-white">
                <strong>Quality Metrics</strong>
              </div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-secondary">
                    <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                  </thead>
                  <tbody>
                    {metrics.map((m, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold">{m.metric}</td>
                        <td className="small">{m.value}</td>
                        <td className="small">
                          <span className={`badge bg-${m.status === 'ok' ? 'success' : m.status === 'warning' ? 'warning' : 'danger'}`}>
                            {m.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Compliance */}
      {comp.length > 0 && (
        <div className="card shadow-sm mb-3">
          <div className="card-header py-2 bg-dark text-white">
            <strong>Regulatory Compliance Context</strong>
          </div>
          <div className="card-body">
            <div className="row">
              {comp.map((c, i) => (
                <div key={i} className="col-md-4 mb-2">
                  <div className="fw-semibold small text-primary">{c.framework}</div>
                  <div className="text-muted small">{c.applicability}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Remediation */}
      {remed.length > 0 && (
        <div className="card shadow-sm">
          <div className="card-header py-2 bg-dark text-white">
            <strong>Remediation / Next Steps</strong>
          </div>
          <div className="card-body">
            {remed.map((r, i) => (
              <div key={i} className="mb-2 d-flex gap-2 align-items-start">
                <span className={`badge bg-${r.severity === 'critical' ? 'danger' : r.severity === 'warning' ? 'warning' : 'info'}`}>
                  {r.severity || 'info'}
                </span>
                <div>
                  <div className="small fw-semibold">{r.issue}</div>
                  <div className="small text-muted">{r.action}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Page root ────────────────────────────────────────────────────────── */
export default function CNNResNetPage() {
  const [tab, setTab] = useState('overview');
  const [ov,  setOv]  = useState(null);
  const [bk,  setBk]  = useState(null);
  const [df,  setDf]  = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/cnn-resnet/overview`).then(r => r.json()),
      fetch(`${API}/api/cnn-resnet/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cnn-resnet/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3">
        <div>
          <h4 className="mb-0 fw-bold">CNN / ResNet — Spectrogram EEG Classifier</h4>
          <div className="text-muted small">
            1D-CNN · 2D-CNN · ResNet-18 · ResNet-34 · Spectrogram-based Epilepsy Detection
          </div>
        </div>
        <div className="ms-auto d-flex gap-2 flex-wrap">
          <span className="badge bg-primary">1D-CNN</span>
          <span className="badge bg-info text-dark">2D-CNN</span>
          <span className="badge bg-secondary">ResNet-18</span>
          <span className="badge bg-dark">ResNet-34</span>
        </div>
      </div>

      {err && <div className="alert alert-danger">Error: {err}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
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

      {tab === 'overview'      && <OverviewPanel      ov={ov} />}
      {tab === 'spectrograms'  && <SpectrogramsPanel  bk={bk} />}
      {tab === 'architectures' && <ArchitecturesPanel bk={bk} />}
      {tab === 'definitions'   && <DefinitionsPanel   df={df} />}
    </div>
  );
}
