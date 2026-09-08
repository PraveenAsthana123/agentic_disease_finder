'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',      label: 'Overview' },
  { id: 'windows',       label: 'Window Inventory' },
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

function SeverityBadge({ severity }) {
  const map = { ok: 'success', warning: 'warning', error: 'danger', info: 'info' };
  return <span className={`badge bg-${map[severity] || 'secondary'}`}>{severity}</span>;
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
  const cfg       = ov.window_config || {};

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
              <strong>EEG Classification Distribution</strong>
            </div>
            <div className="card-body">
              {classDist.map((c, i) => (
                <div key={i} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-semibold">{c.label}</span>
                    <span className="small text-muted">{c.count}</span>
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

      {/* Band Power */}
      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>EEG Frequency Band Power</strong>
            </div>
            <div className="card-body">
              {bands.map((b, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small fw-semibold">{b.band}</span>
                    <span className="small text-muted">{b.mean_power_db} dB · dominant in {b.n_dominant}</span>
                  </div>
                  <div className="progress" style={{ height: 11, borderRadius: 6 }}>
                    <div
                      className="progress-bar"
                      style={{
                        width: `${Math.round(b.n_dominant / Math.max(ov.total_analyses || 1, 1) * 100)}%`,
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

        {/* Window Config */}
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>EEGNet Preprocessing Config</strong>
            </div>
            <div className="card-body">
              <table className="table table-sm table-bordered mb-0">
                <tbody>
                  {[
                    ['Window (s)',         cfg.window_seconds],
                    ['Stride (s)',         cfg.stride_seconds],
                    ['Sampling rate (Hz)', cfg.sampling_rate_hz],
                    ['Channels',           cfg.n_channels],
                    ['Time points',        cfg.n_times],
                    ['Bandpass (Hz)',      cfg.bandpass_hz?.join('–')],
                    ['Notch (Hz)',         cfg.notch_hz],
                    ['Normalisation',      cfg.normalization],
                  ].map(([k, v], i) => (
                    <tr key={i}>
                      <td className="small fw-semibold text-muted">{k}</td>
                      <td className="small">{String(v ?? '—')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-2">
                <span className="text-muted small fw-semibold">Augmentation: </span>
                {(cfg.augmentation || []).map((a, i) => (
                  <span key={i} className="badge bg-secondary me-1" style={{ fontSize: '0.7rem' }}>{a}</span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Literature benchmarks */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Literature Benchmarks — CHB-MIT Dataset</strong>
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

      {/* Daily activity */}
      {daily.length > 0 && (
        <div className="card shadow-sm">
          <div className="card-header py-2 bg-dark text-white">
            <strong>Pipeline Activity (last 14 days)</strong>
          </div>
          <div className="card-body">
            {daily.map((d, i) => (
              <div key={i} className="d-flex align-items-center mb-2 gap-2">
                <span className="small text-muted" style={{ minWidth: 90 }}>{d.date}</span>
                <div className="progress flex-grow-1" style={{ height: 12, borderRadius: 6 }}>
                  <div
                    className="progress-bar bg-primary"
                    style={{
                      width: `${Math.min(d.events / Math.max(...daily.map(x => x.events)) * 100, 100)}%`,
                      borderRadius: 6,
                    }}
                  />
                </div>
                <span className="small text-muted" style={{ minWidth: 30 }}>{d.events}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Window Inventory Tab ─────────────────────────────────────────────── */
function WindowsPanel({ bk }) {
  if (!bk) return <div className="text-muted p-3">Loading…</div>;

  const windows  = bk.window_inventory || [];
  const patients = bk.patient_profiles || [];
  const pr       = bk.preprocessing_readiness || {};
  const flags    = pr.flags || [];

  return (
    <div>
      {/* Preprocessing readiness */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>EEGNet Preprocessing Readiness</strong>
        </div>
        <div className="card-body">
          <div className="row mb-3">
            {[
              ['Total Analyses',      pr.total_analyses],
              ['Usable (Good)',       pr.usable_analyses],
              ['Ictal Windows',       pr.ictal_windows],
              ['Interictal Windows',  pr.interictal_windows],
              ['Balance Ratio',       pr.balance_ratio],
              ['Mean Duration (s)',   pr.mean_duration_s],
            ].map(([k, v], i) => (
              <div key={i} className="col-6 col-md-2 mb-2 text-center">
                <div className="small text-muted">{k}</div>
                <div className="fw-bold">{v ?? '—'}</div>
              </div>
            ))}
          </div>
          {flags.map((f, i) => (
            <div key={i} className="d-flex align-items-start gap-2 mb-2">
              <SeverityBadge severity={f.severity} />
              <div>
                <span className="small fw-semibold">{f.flag}:</span>{' '}
                <span className="small text-muted">{f.detail}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Window inventory table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>EEG Window Inventory ({windows.length} recordings)</strong>
        </div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped mb-0">
              <thead className="table-dark">
                <tr>
                  <th>#</th>
                  <th>File</th>
                  <th>Patient</th>
                  <th>Duration (s)</th>
                  <th>SR (Hz)</th>
                  <th>Channels</th>
                  <th>Windows</th>
                  <th>Confidence</th>
                  <th>Quality</th>
                </tr>
              </thead>
              <tbody>
                {windows.map((w, i) => (
                  <tr key={i}>
                    <td className="small text-muted">{w.id}</td>
                    <td className="small">{w.filename}</td>
                    <td className="small">{w.patient_id}</td>
                    <td className="small">{w.duration_s}</td>
                    <td className="small">{w.sampling_rate}</td>
                    <td className="small">{w.n_channels}</td>
                    <td className="small fw-semibold">{w.n_windows}</td>
                    <td className="small">
                      <PctBar pct={w.confidence} />
                      <span className="text-muted">{w.confidence}%</span>
                    </td>
                    <td className="small">
                      <span
                        className="badge"
                        style={{ backgroundColor: w.quality_color }}
                      >
                        {w.signal_quality}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Patient profiles */}
      <div className="card shadow-sm">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Per-Patient EEGNet Profiles ({patients.length})</strong>
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
                  <th>Windows</th>
                  <th>Confidence</th>
                  <th>Good</th>
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
                    <td className="small">{p.n_windows}</td>
                    <td className="small">
                      <span className={`text-${p.mean_confidence >= 80 ? 'success' : p.mean_confidence >= 60 ? 'warning' : 'danger'} fw-bold`}>
                        {p.mean_confidence}%
                      </span>
                    </td>
                    <td className="small text-success">{p.good_quality}</td>
                    <td className="small">{p.n_seizures}</td>
                    <td className="small">{p.aed_names?.join(', ') || '—'}</td>
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

/* ── Architectures Tab ────────────────────────────────────────────────── */
function ArchitecturesPanel({ bk }) {
  if (!bk) return <div className="text-muted p-3">Loading…</div>;
  const archs = bk.architecture_comparison || [];
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
            {a.variant}
          </button>
        ))}
      </div>

      {/* Comparison table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Architecture Comparison — All Variants</strong>
        </div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped mb-0">
              <thead className="table-dark">
                <tr>
                  <th>Variant</th>
                  <th>Params</th>
                  <th>F1</th>
                  <th>D</th>
                  <th>F2</th>
                  <th>Accuracy</th>
                  <th>Sensitivity</th>
                  <th>Specificity</th>
                  <th>Optimizer</th>
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
                    <td className="small fw-semibold">{a.variant}</td>
                    <td className="small">{a.total_params?.toLocaleString()}</td>
                    <td className="small">{a.F1}</td>
                    <td className="small">{a.D}</td>
                    <td className="small">{a.F2}</td>
                    <td className="small text-success fw-bold">{(a.accuracy * 100).toFixed(1)}%</td>
                    <td className="small text-primary">{(a.sensitivity * 100).toFixed(1)}%</td>
                    <td className="small text-info">{(a.specificity * 100).toFixed(1)}%</td>
                    <td className="small text-muted">{a.optimizer}</td>
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
                      ['Total Params', arch.total_params?.toLocaleString()],
                      ['F1', arch.F1],
                      ['D',  arch.D],
                      ['F2', arch.F2],
                      ['Optimizer', arch.optimizer],
                      ['Dropout', arch.dropout],
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
                <strong>Layer Stack</strong>
              </div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-secondary">
                      <tr>
                        <th>Block</th>
                        <th>Layer Type</th>
                        <th>Details</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(arch.layers || []).map((l, i) => (
                        <tr key={i}>
                          <td className="small text-muted">{l.block}</td>
                          <td className="small fw-semibold">{l.type}</td>
                          <td className="small text-muted">
                            {Object.entries(l)
                              .filter(([k]) => !['block', 'type'].includes(k))
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

  const concepts   = df.concepts || [];
  const regulatory = df.regulatory_context || [];
  const refs       = df.references || [];

  return (
    <div>
      <div className="row">
        <div className="col-lg-7 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white">
              <strong>EEGNet Concepts ({concepts.length})</strong>
            </div>
            <div className="card-body">
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
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 bg-dark text-white">
              <strong>Regulatory Context</strong>
            </div>
            <div className="card-body">
              {regulatory.map((r, i) => (
                <div key={i} className="mb-3 pb-2 border-bottom">
                  <div className="fw-semibold small text-primary mb-1">{r.framework}</div>
                  <div className="text-muted small">{r.applicability}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header py-2 bg-dark text-white">
              <strong>Key References</strong>
            </div>
            <div className="card-body">
              {refs.map((r, i) => (
                <div key={i} className="mb-3 pb-2 border-bottom">
                  <div className="small text-muted">{r.citation}</div>
                  {r.doi && (
                    <div className="small text-primary mt-1">DOI: {r.doi}</div>
                  )}
                  <div className="small text-secondary mt-1">{r.relevance}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Page root ────────────────────────────────────────────────────────── */
export default function EEGNetPage() {
  const [tab, setTab]  = useState('overview');
  const [ov,  setOv]   = useState(null);
  const [bk,  setBk]   = useState(null);
  const [df,  setDf]   = useState(null);
  const [err, setErr]  = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/eegnet/overview`).then(r => r.json()),
      fetch(`${API}/api/eegnet/breakdown`).then(r => r.json()),
      fetch(`${API}/api/eegnet/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3">
        <div>
          <h4 className="mb-0 fw-bold">EEGNet — Compact CNN for Raw EEG</h4>
          <div className="text-muted small">
            Depthwise Conv · Separable Conv · Temporal + Spatial Filters · Field-Standard BCI/Clinical Architecture
          </div>
        </div>
        <div className="ms-auto d-flex gap-2 flex-wrap">
          <span className="badge bg-primary">EEGNet-8,2</span>
          <span className="badge bg-info text-dark">EEGNet-4,2</span>
          <span className="badge bg-secondary">+ Attention</span>
          <span className="badge bg-dark">+ SMOTE</span>
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
      {tab === 'windows'       && <WindowsPanel       bk={bk} />}
      {tab === 'architectures' && <ArchitecturesPanel bk={bk} />}
      {tab === 'definitions'   && <DefinitionsPanel   df={df} />}
    </div>
  );
}
