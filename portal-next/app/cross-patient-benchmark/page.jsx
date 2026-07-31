'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Fold Breakdown' },
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

function AccuracyBar({ value, max }) {
  const pct = Math.round((value / (max || 1)) * 100);
  const color = value >= 0.8 ? 'success' : value >= 0.6 ? 'warning' : 'danger';
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 12 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="small fw-semibold" style={{ minWidth: 50 }}>{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const kpis = data.kpis || {};
  const folds = data.fold_performance || [];
  const dist = data.accuracy_distribution || [];
  const comp = data.in_sample_comparison || {};
  const features = data.feature_set || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Mean Accuracy" value={kpis.mean_accuracy != null ? `${(kpis.mean_accuracy * 100).toFixed(1)}%` : '\u2014'} color={kpis.mean_accuracy >= 0.7 ? 'success' : 'warning'} sub="cross-patient LOSO" />
        <KPI label="Mean F1" value={kpis.mean_f1 != null ? `${(kpis.mean_f1 * 100).toFixed(1)}%` : '\u2014'} color={kpis.mean_f1 >= 0.7 ? 'success' : 'warning'} sub="harmonic mean precision/recall" />
        <KPI label="Subjects" value={kpis.n_subjects} color="info" sub={`${kpis.n_folds} LOSO folds`} />
        <KPI label="Features" value={kpis.feature_count} color="secondary" sub={`${kpis.window_seconds}s windows`} />
      </div>

      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Per-Subject Accuracy</div>
            <div className="card-body">
              {folds.map(f => (
                <div key={f.subject} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="fw-semibold">{f.subject}</span>
                    <span className="text-muted">{f.n_test} test windows</span>
                  </div>
                  <AccuracyBar value={f.accuracy} max={1} />
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">In-Sample vs Cross-Patient</div>
            <div className="card-body">
              <div className="mb-3">
                <div className="small fw-semibold mb-1">In-Sample Accuracy</div>
                <AccuracyBar value={comp.in_sample_accuracy || 0} max={1} />
              </div>
              <div className="mb-3">
                <div className="small fw-semibold mb-1">Cross-Patient Accuracy</div>
                <AccuracyBar value={comp.cross_patient_accuracy || 0} max={1} />
              </div>
              <div className="alert alert-info small mb-0">
                <strong>Generalization Gap:</strong> {comp.gap != null ? `${(comp.gap * 100).toFixed(1)}%` : '\u2014'} drop when evaluating on unseen patients.
                This gap quantifies overfitting to patient-specific EEG patterns.
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Accuracy Distribution</div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0">
                <thead><tr><th>Bin</th><th className="text-end">Count</th><th>Bar</th></tr></thead>
                <tbody>
                  {dist.map((b, i) => (
                    <tr key={i}>
                      <td className="small">{(b.bin_start * 100).toFixed(0)}\u2013{(b.bin_end * 100).toFixed(0)}%</td>
                      <td className="text-end fw-semibold">{b.count}</td>
                      <td>
                        <div className="progress" style={{ height: 10 }}>
                          <div className={`progress-bar bg-${b.count > 0 ? 'primary' : 'light'}`} style={{ width: `${(b.count / Math.max(...dist.map(d => d.count), 1)) * 100}%` }} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Feature Set ({kpis.feature_count} features)</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead><tr><th>Category</th><th>Features</th></tr></thead>
                <tbody>
                  {features.map((fg, i) => (
                    <tr key={i}>
                      <td className="fw-semibold text-capitalize">{fg.category}</td>
                      <td className="small">{fg.features.join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const folds = data.folds_detail || [];
  const gaps = data.generalization_gap || [];
  const difficulty = data.subject_difficulty || [];
  const bands = data.band_power_contribution || [];
  const spatial = data.spatial_patterns || [];

  return (
    <div>
      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">LOSO Fold Details</div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0">
                <thead>
                  <tr><th>Held Out</th><th>Train On</th><th className="text-end">Accuracy</th><th className="text-end">F1</th><th className="text-end">N Test</th></tr>
                </thead>
                <tbody>
                  {folds.map(f => (
                    <tr key={f.held_out_subject} className={f.accuracy < 0.5 ? 'table-danger' : f.accuracy >= 0.8 ? 'table-success' : ''}>
                      <td className="fw-semibold">{f.held_out_subject}</td>
                      <td className="small">{(f.train_subjects || []).join(', ')}</td>
                      <td className="text-end">{(f.accuracy * 100).toFixed(1)}%</td>
                      <td className="text-end">{(f.f1 * 100).toFixed(1)}%</td>
                      <td className="text-end">{f.n_test}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Generalization Gap per Subject</div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0">
                <thead>
                  <tr><th>Subject</th><th className="text-end">In-Sample</th><th className="text-end">Cross-Patient</th><th className="text-end">Gap</th></tr>
                </thead>
                <tbody>
                  {gaps.map(g => (
                    <tr key={g.held_out_subject}>
                      <td className="fw-semibold">{g.held_out_subject}</td>
                      <td className="text-end">{(g.in_sample_accuracy * 100).toFixed(1)}%</td>
                      <td className="text-end">{(g.cross_patient_accuracy * 100).toFixed(1)}%</td>
                      <td className={`text-end fw-bold ${g.gap > 0.3 ? 'text-danger' : 'text-warning'}`}>{(g.gap * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="row mb-4">
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Subject Difficulty Ranking</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead><tr><th>#</th><th>Subject</th><th>Difficulty</th><th className="text-end">Acc</th></tr></thead>
                <tbody>
                  {difficulty.map(d => (
                    <tr key={d.subject}>
                      <td>{d.rank}</td>
                      <td className="fw-semibold">{d.subject}</td>
                      <td><span className={`badge bg-${d.difficulty === 'hard' ? 'danger' : d.difficulty === 'medium' ? 'warning' : 'success'}`}>{d.difficulty}</span></td>
                      <td className="text-end">{(d.accuracy * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Band Power Importance</div>
            <div className="card-body">
              {bands.map(b => (
                <div key={b.band} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="fw-semibold text-capitalize">{b.band}</span>
                    <span>{(b.importance_normalized * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar bg-info" style={{ width: `${b.importance_normalized * 100 * 3}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Top Electrodes per Subject</div>
            <div className="card-body p-0">
              {spatial.map(s => (
                <div key={s.held_out_subject} className="p-2 border-bottom">
                  <div className="small fw-semibold mb-1">{s.held_out_subject}</div>
                  <div className="d-flex flex-wrap gap-1">
                    {(s.dominant_electrodes || []).map((e, i) => (
                      <span key={i} className={`badge ${i === 0 ? 'bg-primary' : 'bg-secondary'}`}>{e.electrode} ({(e.contribution * 100).toFixed(0)}%)</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  return (
    <div>
      {data.caveat && (
        <div className="alert alert-warning small mb-3">
          <strong>Caveat:</strong> {data.caveat}
        </div>
      )}
      <div className="card">
        <div className="card-header fw-semibold">{data.dashboard_name || 'Cross-Patient Benchmark'} \u2014 Definitions</div>
        <div className="card-body p-0">
          <table className="table table-sm table-striped mb-0">
            <thead><tr><th>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {(data.terms || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold text-nowrap">{t.term}</td>
                  <td className="small">{t.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function CrossPatientBenchmarkDashboard() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/cross-patient-benchmark/overview`).then(r => r.json()).then(setOverview).catch(() => setOverview({ error: 'Failed to load overview' }));
    fetch(`${API}/api/cross-patient-benchmark/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setBreakdown({ error: 'Failed to load breakdown' }));
    fetch(`${API}/api/cross-patient-benchmark/definitions`).then(r => r.json()).then(setDefinitions).catch(() => setDefinitions({ error: 'Failed to load definitions' }));
  }, []);

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1">Cross-Patient Benchmark Dashboard</h4>
      <p className="text-muted small mb-3">
        Leave-one-subject-out (LOSO) cross-validation on real CHB-MIT EEG recordings.
        Measures how well the seizure detection model generalizes to unseen patients.
      </p>
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>
      {tab === 'overview' && <OverviewPanel data={overview} />}
      {tab === 'breakdown' && <BreakdownPanel data={breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={definitions} />}
    </div>
  );
}
