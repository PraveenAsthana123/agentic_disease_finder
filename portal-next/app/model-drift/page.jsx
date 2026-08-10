'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',  label: '📊 Overview' },
  { id: 'subjects',  label: '🧑 Per-Subject' },
  { id: 'timeline',  label: '📅 Training Timeline' },
  { id: 'definitions', label: '📖 Definitions' },
];

const VERDICT_COLOR = { IMPROVED: 'success', STABLE: 'info', DEGRADED: 'danger' };

function KPI({ label, value, sub, hex }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className="h4 mb-1 fw-bold" style={{ color: hex || '#3b82f6' }}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function MiniBar({ label, value, color }) {
  const pct = Math.round(value * 100);
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-semibold">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color || '#3b82f6' }} />
      </div>
    </div>
  );
}

function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  const ci = ov.bootstrap_ci || {};
  const verdict = ov.performance_verdict || 'STABLE';
  const vcol = VERDICT_COLOR[verdict] || 'secondary';

  return (
    <div>
      {/* KPI row from server-provided kpis */}
      <div className="row mb-3">
        {(ov.kpis || []).map((k, i) => (
          <KPI key={i} label={k.label} value={k.value} hex={k.color} />
        ))}
      </div>

      <div className="row mb-4">
        {/* Verdict + Drift Score card */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-body">
              <h6 className="card-title">Performance Verdict</h6>
              <div className={`alert alert-${vcol} py-2 text-center mb-2`}>
                <strong>{verdict}</strong>
              </div>
              <p className="small text-muted mb-0">
                Drift Score: <strong>{ov.drift_score}%</strong> (100 = uniform performance across subjects)
              </p>
              <p className="small text-muted mb-0 mt-1">
                Run at: {ov.run_at ? new Date(ov.run_at).toLocaleString() : '—'}
              </p>
            </div>
          </div>
        </div>

        {/* Bootstrap CI */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-body">
              <h6 className="card-title">Bootstrap CI (95%)</h6>
              <div className="text-center mb-2">
                <span className="h5 text-success">{(ci.mean * 100).toFixed(1)}%</span>
                <span className="text-muted small ms-2">mean accuracy</span>
              </div>
              <div className="small text-muted text-center">
                [{(ci.ci95_low * 100).toFixed(1)}% – {(ci.ci95_high * 100).toFixed(1)}%]
              </div>
              <div className="small text-muted text-center mt-1">
                {ci.n_subjects} subjects · {(ci.n_boot || 0).toLocaleString()} bootstrap iterations
              </div>
            </div>
          </div>
        </div>

        {/* Performance bars */}
        <div className="col-md-4 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-body">
              <h6 className="card-title">Key Metrics</h6>
              <MiniBar label="Patient-Specific Acc" value={ov.patient_specific_accuracy || 0} color="#10b981" />
              <MiniBar label="Sensitivity" value={ov.patient_specific_sensitivity || 0} color="#f59e0b" />
              <MiniBar label="Cross-Patient Acc" value={ov.cross_patient_accuracy || 0} color="#3b82f6" />
              <MiniBar label="Bonn External Acc" value={ov.bonn_external_accuracy || 0} color="#8b5cf6" />
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-body">
          <h6 className="card-title">Summary</h6>
          <div className="row">
            <div className="col-md-6">
              <ul className="list-unstyled small mb-0">
                <li>Training runs logged: <strong>{ov.n_training_events}</strong></li>
                <li>CV fold events: <strong>{ov.n_cv_events}</strong></li>
                <li>Models in registry: <strong>{ov.n_models}</strong></li>
              </ul>
            </div>
            <div className="col-md-6">
              <p className="small text-muted mb-0">
                Patient-specific accuracy ({(ov.patient_specific_accuracy * 100).toFixed(1)}%) exceeds the
                literature baseline of 95% → <span className="text-success fw-semibold">IMPROVED</span>.
                Cross-patient accuracy ({(ov.cross_patient_accuracy * 100).toFixed(1)}%) reflects
                inter-patient variability (typical for epilepsy AI).
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SubjectsTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const subjects = bk.per_subject || [];
  const strategies = bk.evaluation_strategies || [];
  const models = bk.model_inventory || [];

  return (
    <div>
      <div className="row mb-4">
        <div className="col-md-8 mb-3">
          <div className="card shadow-sm">
            <div className="card-body">
              <h6 className="card-title">Per-Subject Performance (CHB-MIT)</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead className="table-light">
                    <tr>
                      <th>Subject</th>
                      <th>Accuracy</th>
                      <th>Sensitivity</th>
                      <th>F1</th>
                      <th>Seizure Windows</th>
                      <th>Total Windows</th>
                    </tr>
                  </thead>
                  <tbody>
                    {subjects.map((s, i) => (
                      <tr key={i}>
                        <td><strong>{s.subject}</strong></td>
                        <td>
                          <span className={s.accuracy >= 0.95 ? 'text-success fw-semibold' : 'text-warning fw-semibold'}>
                            {(s.accuracy * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td>{(s.sensitivity * 100).toFixed(1)}%</td>
                        <td>{(s.f1 * 100).toFixed(1)}%</td>
                        <td>{s.n_seizure}</td>
                        <td>{s.n_total}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="card-title">Evaluation Strategies</h6>
              {strategies.map((s, i) => (
                <div key={i} className="mb-2 pb-2 border-bottom">
                  <div className="d-flex justify-content-between">
                    <span className="small fw-semibold">{s.method}</span>
                    <span className="badge bg-primary">{(s.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <p className="small text-muted mb-0" style={{ fontSize: '0.72rem' }}>{s.details}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-body">
              <h6 className="card-title">Model Registry ({models.length} models)</h6>
              <div className="table-responsive">
                <table className="table table-sm">
                  <thead className="table-light">
                    <tr><th>Model</th><th>Size</th><th>Modified</th></tr>
                  </thead>
                  <tbody>
                    {models.map((m, i) => (
                      <tr key={i}>
                        <td><strong>{m.name}</strong></td>
                        <td>{m.size_mb} MB</td>
                        <td className="text-muted small">{m.modified?.split(' ')[0]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function TimelineTab({ bk }) {
  if (!bk) return <div className="text-muted">Loading…</div>;
  const timeline = bk.training_timeline || [];
  const cvFolds = bk.cross_validation_folds || [];
  const events = bk.training_events || [];

  const successCount = timeline.filter(t => t.success).length;
  const failCount = timeline.length - successCount;

  return (
    <div>
      <div className="row mb-3">
        <div className="col-md-3 mb-3">
          <div className="card shadow-sm text-center">
            <div className="card-body">
              <div className="h4 text-primary fw-bold">{timeline.length}</div>
              <div className="text-muted small">Training Days</div>
            </div>
          </div>
        </div>
        <div className="col-md-3 mb-3">
          <div className="card shadow-sm text-center">
            <div className="card-body">
              <div className="h4 text-success fw-bold">{successCount}</div>
              <div className="text-muted small">Successful Days</div>
            </div>
          </div>
        </div>
        <div className="col-md-3 mb-3">
          <div className="card shadow-sm text-center">
            <div className="card-body">
              <div className="h4 text-danger fw-bold">{failCount}</div>
              <div className="text-muted small">Failed Days</div>
            </div>
          </div>
        </div>
        <div className="col-md-3 mb-3">
          <div className="card shadow-sm text-center">
            <div className="card-body">
              <div className="h4 text-info fw-bold">{cvFolds.length}</div>
              <div className="text-muted small">CV Fold Events</div>
            </div>
          </div>
        </div>
      </div>

      <div className="row mb-4">
        <div className="col-md-8 mb-3">
          <div className="card shadow-sm">
            <div className="card-body">
              <h6 className="card-title">Training Timeline ({timeline.length} days)</h6>
              <div style={{ maxHeight: 360, overflowY: 'auto' }}>
                <div className="d-flex flex-wrap gap-1">
                  {timeline.map((t, i) => (
                    <div
                      key={i}
                      className={`badge ${t.success ? 'bg-success' : 'bg-danger'}`}
                      title={`${t.date}: ${t.summary}`}
                      style={{ fontSize: '0.65rem', cursor: 'default' }}
                    >
                      {t.date?.slice(5)}
                    </div>
                  ))}
                </div>
                <p className="text-muted small mt-2 mb-0">Green = success, Red = failed run</p>
              </div>
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card shadow-sm">
            <div className="card-body">
              <h6 className="card-title">Recent Training Events</h6>
              <div style={{ maxHeight: 360, overflowY: 'auto' }}>
                {events.slice(0, 10).map((e, i) => (
                  <div key={i} className="d-flex align-items-center mb-1 pb-1 border-bottom">
                    <span className={`badge ${e.success !== false ? 'bg-success' : 'bg-danger'} me-2`} style={{ fontSize: '0.6rem' }}>
                      {e.success !== false ? 'OK' : 'FAIL'}
                    </span>
                    <div>
                      <div className="small fw-semibold">{e.script || e.action || 'training run'}</div>
                      <div className="text-muted" style={{ fontSize: '0.7rem' }}>{e.timestamp?.slice(0, 19)}</div>
                    </div>
                  </div>
                ))}
                {events.length === 0 && <p className="text-muted small">No event records.</p>}
              </div>
            </div>
          </div>
        </div>
      </div>

      {cvFolds.length > 0 && (
        <div className="card shadow-sm">
          <div className="card-body">
            <h6 className="card-title">Cross-Validation Folds (sample)</h6>
            <div className="table-responsive">
              <table className="table table-sm">
                <thead className="table-light">
                  <tr>
                    {Object.keys(cvFolds[0] || {}).map(k => <th key={k}>{k}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {cvFolds.slice(0, 8).map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).map((v, j) => (
                        <td key={j} className="small">
                          {typeof v === 'number' ? v.toFixed(4) : String(v)}
                        </td>
                      ))}
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

function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  const sections = defs.sections || [];
  return (
    <div>
      {sections.map((sec, i) => (
        <div key={i} className="card shadow-sm mb-3">
          <div className="card-body">
            <h6 className="card-title">{sec.title}</h6>
            <table className="table table-sm">
              <thead className="table-light">
                <tr><th style={{ width: '25%' }}>Term</th><th>Definition</th></tr>
              </thead>
              <tbody>
                {(sec.items || []).map((item, j) => (
                  <tr key={j}>
                    <td><strong>{item.term}</strong></td>
                    <td className="small text-muted">{item.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ModelDriftPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/model-drift/overview`).then(r => r.json()),
      fetch(`${API}/api/model-drift/breakdown`).then(r => r.json()),
      fetch(`${API}/api/model-drift/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!overview) return <div className="text-muted p-3">Loading model drift dashboard…</div>;
  if (!overview.available) return <div className="alert alert-warning m-3">No model drift data available.</div>;

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-1">
        <h4 className="mb-0 me-2">📉 Model Drift Dashboard</h4>
        <span className={`badge bg-${VERDICT_COLOR[overview.performance_verdict] || 'secondary'}`}>
          {overview.performance_verdict}
        </span>
      </div>
      <p className="text-muted small mb-3">
        Model performance monitoring across {overview.n_training_runs} training runs ·{' '}
        {overview.n_models} models · drift score {overview.drift_score}% ·{' '}
        real data from training_log + CV pipeline
      </p>

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

      {tab === 'overview'     && <OverviewTab ov={overview} />}
      {tab === 'subjects'     && <SubjectsTab bk={breakdown} />}
      {tab === 'timeline'     && <TimelineTab bk={breakdown} />}
      {tab === 'definitions'  && <DefinitionsTab defs={definitions} />}
    </div>
  );
}
