'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',       label: 'Overview' },
  { id: 'patient-perf',   label: 'Patient Performance' },
  { id: 'model-compare',  label: 'Method Comparison' },
  { id: 'architectures',  label: 'Architectures' },
  { id: 'definitions',    label: 'Definitions' },
];

const ARCH_COLOR = {
  '1D CNN': 'primary', '3D CNN': 'info', EEGNet: 'success',
  Ensemble: 'warning', 'Graph Neural Network': 'danger',
  LSTM: 'secondary', Transformer: 'dark', Utility: 'light',
};

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

function AccBar({ pct, color }) {
  const c = color || (pct >= 95 ? 'success' : pct >= 80 ? 'info' : pct >= 60 ? 'warning' : 'danger');
  return (
    <div className="progress" style={{ height: 18, borderRadius: 8 }}>
      <div
        className={`progress-bar bg-${c}`}
        style={{ width: `${Math.min(pct, 100)}%`, borderRadius: 8, transition: 'width 0.6s ease' }}
      />
    </div>
  );
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  const k = ov.kpis || {};
  const mc = ov.methods_comparison || [];
  const pp = ov.per_patient_accuracy || [];
  const mf = ov.model_files || [];
  const dc = ov.disease_chart || [];
  const tt = ov.training_timeline || [];
  const tm = ov.training_meta || {};

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Total DL Models"        value={k.total_models}              color="primary"   sub="Architectures built" />
        <KPI label="Best Accuracy"          value={`${k.best_accuracy_pct}%`}   color="success"   sub="CHB-MIT patient-specific" />
        <KPI label="Mean Accuracy"          value={`${k.mean_accuracy_pct}%`}   color="info"      sub="Across subjects" />
        <KPI label="Mean Sensitivity"       value={`${k.mean_sensitivity_pct}%`} color="warning"  sub="Seizure detection rate" />
      </div>
      <div className="row mb-4">
        <KPI label="Patients Trained"       value={k.total_patients_trained}    color="primary"   sub="CHB-MIT subjects" />
        <KPI label="Total Analyses"         value={k.total_analyses}            color="secondary" sub="DB predictions" />
        <KPI label="Avg Confidence"         value={`${(k.avg_confidence * 100).toFixed(0)}%`} color="info" sub="Model certainty" />
        <KPI label="Model Files"            value={`${k.total_model_size_mb} MB`} color="dark"   sub="Total .joblib size" />
      </div>

      {/* Methods comparison */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Method Comparison — Mean Accuracy</strong>
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark">
              <tr><th>Method</th><th>Mean Accuracy</th><th>Bar</th></tr>
            </thead>
            <tbody>
              {mc.map(m => (
                <tr key={m.key}>
                  <td className="text-capitalize">{m.method}</td>
                  <td className="fw-bold">{m.mean_accuracy_pct}%</td>
                  <td style={{ width: '40%' }}>
                    <AccBar pct={m.mean_accuracy_pct} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-patient accuracy summary */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white">
          <strong>Per-Subject Accuracy (Deep Learning — CHB-MIT)</strong>
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark">
              <tr><th>Subject</th><th>Accuracy</th><th>Sensitivity</th><th>F1</th><th>Windows</th><th>Seizure %</th></tr>
            </thead>
            <tbody>
              {pp.map(s => (
                <tr key={s.subject}>
                  <td className="fw-bold">{s.subject}</td>
                  <td><span className={`badge bg-${s.accuracy_pct >= 95 ? 'success' : s.accuracy_pct >= 80 ? 'info' : 'warning'}`}>{s.accuracy_pct}%</span></td>
                  <td>{s.sensitivity_pct}%</td>
                  <td>{s.f1_pct}%</td>
                  <td>{s.n_total}</td>
                  <td>{(s.seizure_ratio * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Architecture type distribution */}
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white"><strong>Architecture Types</strong></div>
            <div className="card-body">
              {(ov.arch_type_chart || []).map(a => (
                <div key={a.name} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span><span className={`badge bg-${ARCH_COLOR[a.name] || 'secondary'} me-1`}>&nbsp;</span>{a.name}</span>
                    <span>{a.value}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2 bg-dark text-white"><strong>Model Files</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-secondary">
                  <tr><th>Disease</th><th>Size</th></tr>
                </thead>
                <tbody>
                  {mf.map(f => (
                    <tr key={f.filename}>
                      <td>{f.disease}</td>
                      <td>{f.size_kb} KB</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Training meta */}
      <div className="card shadow-sm mb-4 border-success">
        <div className="card-header py-2 bg-success text-white">
          <strong>Training Run Summary</strong>
        </div>
        <div className="card-body">
          <div className="row">
            <div className="col-md-4"><span className="text-muted">Dataset:</span> <strong>{tm.dataset}</strong></div>
            <div className="col-md-4"><span className="text-muted">Run at:</span> {tm.run_at?.slice(0, 16).replace('T', ' ')}</div>
            <div className="col-md-4"><span className="text-muted">Result:</span> <span className="badge bg-success">{tm.summary}</span></div>
          </div>
          <div className="mt-3">
            {tt.map((r, i) => (
              <div key={i} className="d-flex align-items-center gap-3 mb-1">
                <span className={`badge bg-${r.ok ? 'success' : 'danger'}`}>{r.ok ? 'OK' : 'FAIL'}</span>
                <span className="small text-monospace">{r.script}</span>
                <span className="text-muted small">{r.seconds}s</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Disease distribution */}
      <div className="card shadow-sm mb-4">
        <div className="card-header py-2 bg-dark text-white"><strong>Patient Diagnoses in Clinical DB</strong></div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark">
              <tr><th>Diagnosis</th><th>Count</th><th>Distribution</th></tr>
            </thead>
            <tbody>
              {dc.map(d => {
                const total = dc.reduce((s, x) => s + x.value, 0);
                const pct = total > 0 ? (d.value / total * 100) : 0;
                return (
                  <tr key={d.name}>
                    <td>{d.name}</td>
                    <td>{d.value}</td>
                    <td style={{ width: '45%' }}>
                      <AccBar pct={pct} color="primary" />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function PatientPerfPanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const pd = bd.patient_details || [];

  return (
    <div>
      <h5 className="mb-3">Per-Subject Deep Learning Performance (CHB-MIT)</h5>
      <div className="row">
        {pd.map(p => (
          <div key={p.subject} className="col-md-6 col-lg-3 mb-4">
            <div className="card shadow-sm h-100">
              <div className="card-header bg-dark text-white py-2">
                <strong>{p.subject}</strong>
              </div>
              <div className="card-body">
                <div className="mb-3 text-center">
                  <div className={`display-6 fw-bold text-${p.accuracy_pct >= 95 ? 'success' : 'info'}`}>
                    {p.accuracy_pct}%
                  </div>
                  <div className="text-muted small">Accuracy</div>
                </div>
                <table className="table table-sm mb-0">
                  <tbody>
                    <tr><td className="text-muted">Sensitivity</td><td className="fw-bold">{p.sensitivity_pct}%</td></tr>
                    <tr><td className="text-muted">F1 Score</td><td className="fw-bold">{p.f1_pct}%</td></tr>
                    <tr><td className="text-muted">Total Windows</td><td>{p.n_total}</td></tr>
                    <tr><td className="text-muted">Seizure Windows</td><td>{p.n_seizure}</td></tr>
                    <tr><td className="text-muted">Test Set</td><td>{p.n_test}</td></tr>
                    <tr><td className="text-muted">Seizure Ratio</td><td>{(p.seizure_ratio * 100).toFixed(1)}%</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Benchmark info */}
      {bd.benchmark_info && (
        <div className="card shadow-sm border-info mt-2">
          <div className="card-header py-2 bg-info text-white"><strong>EEGNet Benchmark (Raw Signal)</strong></div>
          <div className="card-body">
            <div className="row">
              <div className="col-md-3"><span className="text-muted">Model:</span> {bd.benchmark_info.model}</div>
              <div className="col-md-3"><span className="text-muted">Mean Accuracy:</span> <strong>{(bd.benchmark_info.mean_accuracy * 100).toFixed(1)}%</strong></div>
              <div className="col-md-3"><span className="text-muted">Mean Sensitivity:</span> <strong>{(bd.benchmark_info.mean_sensitivity * 100).toFixed(1)}%</strong></div>
              <div className="col-md-3"><span className="text-muted">Architecture:</span> {bd.benchmark_info.architecture}</div>
            </div>
            <div className="mt-2 small text-muted">{bd.benchmark_info.note}</div>
            {bd.benchmark_info.per_subject && (
              <div className="mt-3">
                <table className="table table-sm mb-0">
                  <thead className="table-secondary">
                    <tr><th>Subject</th><th>Accuracy</th><th>Sensitivity</th><th>Train</th><th>Test</th></tr>
                  </thead>
                  <tbody>
                    {bd.benchmark_info.per_subject.map(s => (
                      <tr key={s.subject}>
                        <td>{s.subject}</td>
                        <td>{(s.accuracy * 100).toFixed(1)}%</td>
                        <td>{(s.sensitivity * 100).toFixed(1)}%</td>
                        <td>{s.n_train}</td>
                        <td>{s.n_test}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MethodComparePanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const mc = bd.model_comparison || [];
  const [expanded, setExpanded] = useState(null);

  return (
    <div>
      <h5 className="mb-3">Evaluation Method Comparison</h5>
      {mc.map(m => (
        <div key={m.method_key} className="card shadow-sm mb-3">
          <div
            className="card-header d-flex justify-content-between align-items-center py-2 bg-dark text-white"
            style={{ cursor: 'pointer' }}
            onClick={() => setExpanded(expanded === m.method_key ? null : m.method_key)}
          >
            <strong className="text-capitalize">{m.method_label}</strong>
            <div className="d-flex gap-3 align-items-center">
              <span className="badge bg-success">Mean: {m.mean_accuracy_pct}%</span>
              <span className="text-muted small">Min: {(m.min_accuracy * 100).toFixed(1)}% / Max: {(m.max_accuracy * 100).toFixed(1)}%</span>
              <span>{expanded === m.method_key ? '▲' : '▼'}</span>
            </div>
          </div>
          {expanded === m.method_key && (
            <div className="card-body">
              <p className="text-muted small mb-3">{m.method_description}</p>
              <AccBar pct={m.mean_accuracy_pct} />
              <div className="mt-3">
                <table className="table table-sm">
                  <thead className="table-secondary">
                    <tr><th>Subject</th><th>Accuracy</th><th>F1</th></tr>
                  </thead>
                  <tbody>
                    {(m.folds || []).map(f => (
                      <tr key={f.subject}>
                        <td>{f.subject}</td>
                        <td><span className={`badge bg-${f.accuracy_pct >= 95 ? 'success' : f.accuracy_pct >= 80 ? 'info' : 'warning'}`}>{f.accuracy_pct}%</span></td>
                        <td>{(f.f1 * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function ArchitecturesPanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const archs = bd.architectures || [];

  return (
    <div>
      <h5 className="mb-3">Deep Learning Architectures ({archs.length})</h5>
      <div className="row">
        {archs.map(a => (
          <div key={a.class_name} className="col-md-6 col-lg-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className={`card-header py-2 bg-${ARCH_COLOR[a.architecture_type] || 'secondary'} text-${a.architecture_type === 'Utility' ? 'dark' : 'white'}`}>
                <span className="small fw-bold">{a.architecture_type}</span>
                {a.target_disease && a.target_disease !== 'N/A' && (
                  <span className="ms-2 badge bg-dark">{a.target_disease}</span>
                )}
              </div>
              <div className="card-body">
                <h6 className="card-title text-monospace" style={{ fontSize: '0.85rem' }}>{a.class_name}</h6>
                <p className="card-text small text-muted">{a.title}</p>
                {a.description && (
                  <p className="card-text" style={{ fontSize: '0.78rem' }}>{a.description.slice(0, 180)}{a.description.length > 180 ? '…' : ''}</p>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DefinitionsPanel({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  const sections = defs.sections || [];

  return (
    <div>
      {sections.map(sec => (
        <div key={sec.title} className="mb-4">
          <h5 className="border-bottom pb-2 mb-3">{sec.title}</h5>
          <div className="row">
            {(sec.items || []).map(item => (
              <div key={item.term} className="col-md-6 mb-3">
                <div className="card shadow-sm h-100">
                  <div className="card-body">
                    <h6 className="card-title text-primary">{item.term}</h6>
                    <p className="card-text small text-muted mb-0">{item.definition}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function DeepLearningPage() {
  const [tab, setTab]   = useState('overview');
  const [ov,  setOv]   = useState(null);
  const [bd,  setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [err,  setErr]  = useState(null);

  useEffect(() => {
    fetch(`${API}/api/deep-learning/overview`)
      .then(r => r.json())
      .then(setOv)
      .catch(e => setErr(e.message));
  }, []);

  useEffect(() => {
    if (tab === 'patient-perf' || tab === 'model-compare' || tab === 'architectures') {
      if (!bd) {
        fetch(`${API}/api/deep-learning/breakdown`)
          .then(r => r.json())
          .then(setBd)
          .catch(e => setErr(e.message));
      }
    }
    if (tab === 'definitions' && !defs) {
      fetch(`${API}/api/deep-learning/definitions`)
        .then(r => r.json())
        .then(setDefs)
        .catch(e => setErr(e.message));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center gap-3 mb-4">
        <div>
          <h2 className="mb-0">🧠 Deep Learning Dashboard</h2>
          <div className="text-muted small">
            DL architectures · EEGNet benchmark · per-subject performance · method comparison · CHB-MIT real EEG
          </div>
        </div>
        {ov?.kpis && (
          <div className="ms-auto d-flex gap-2">
            <span className="badge bg-success fs-6">{ov.kpis.best_accuracy_pct}% best</span>
            <span className="badge bg-primary fs-6">{ov.kpis.architecture_count} architectures</span>
          </div>
        )}
      </div>

      {err && <div className="alert alert-danger">Error: {err}</div>}

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

      {tab === 'overview'      && <OverviewPanel    ov={ov} />}
      {tab === 'patient-perf'  && <PatientPerfPanel bd={bd} />}
      {tab === 'model-compare' && <MethodComparePanel bd={bd} />}
      {tab === 'architectures' && <ArchitecturesPanel bd={bd} />}
      {tab === 'definitions'   && <DefinitionsPanel defs={defs} />}
    </div>
  );
}
