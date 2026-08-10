'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'breakdown',   label: 'Per-Subject Detail' },
  { id: 'definitions', label: 'Methodology Guide' },
];

const pct = v => v != null ? `${(+v * 100).toFixed(1)}%` : '—';
const fmtN = v => v != null ? (+v).toFixed(4) : '—';

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-3">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function AccBar({ val, color }) {
  const pctNum = val != null ? Math.min(+(val * 100).toFixed(1), 100) : 0;
  const c = color || (pctNum >= 95 ? 'success' : pctNum >= 80 ? 'info' : pctNum >= 60 ? 'warning' : 'danger');
  return (
    <div className="progress" style={{ height: 16 }}>
      <div className={`progress-bar bg-${c}`} style={{ width: `${pctNum}%` }}>
        {pctNum >= 15 ? `${pctNum}%` : ''}
      </div>
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return null;
  const kpis = data.kpis || {};
  const methods = data.methods || [];

  return (
    <div>
      {/* KPI Row */}
      <div className="row mb-4">
        <KPI label="Evaluation Methods" value={kpis.n_methods} color="dark" />
        <KPI label="Best Method Accuracy" value={pct(kpis.best_accuracy)} color="success" sub="patient-specific" />
        <KPI label="Leakage-Free Accuracy" value={pct(kpis.leakage_free_acc)} color="primary" sub="GroupKFold" />
        <KPI label="External Validation" value={pct(kpis.external_val_acc)} color="danger" sub="Bonn dataset" />
      </div>

      {/* Method Comparison Table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold">Accuracy Method Comparison</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Method</th>
                  <th>Accuracy</th>
                  <th>Bar</th>
                  <th>Bias</th>
                  <th>Best For</th>
                </tr>
              </thead>
              <tbody>
                {methods.map(m => (
                  <tr key={m.id}>
                    <td>
                      <span className={`badge bg-${m.color} me-2`}>{m.id.replace(/_/g, ' ')}</span>
                      <span className="fw-semibold">{m.label}</span>
                      <div className="text-muted small">{m.note}</div>
                    </td>
                    <td className="fw-bold">{pct(m.mean_accuracy)}</td>
                    <td style={{ minWidth: 120 }}><AccBar val={m.mean_accuracy} color={m.color} /></td>
                    <td><span className="badge bg-secondary small">{m.bias}</span></td>
                    <td className="text-muted small">{m.best_for}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="alert alert-info small">
        <strong>Thesis guidance:</strong> Lead with Bonn external validation + GroupKFold leakage-free results.
        Patient-specific overlapping accuracy (highest) is supplementary — not a generalization claim.
        Per §57.7 honesty — all methods surfaced, biases documented.
      </div>
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return null;

  const psOverlap = data.patient_specific_overlap || [];
  const psBasic   = data.patient_specific_basic || [];
  const cpRf      = data.cross_patient_rf || [];
  const cpEns     = data.cross_patient_ensemble || [];
  const swDis     = data.subjectwise_by_disease || [];
  const bonn      = data.bonn_external || [];
  const bonnMeta  = data.bonn_meta || {};

  return (
    <div>
      {/* Patient-specific overlapping */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold">Patient-Specific — Overlapping Windows (per subject)</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th>Subject</th><th>Accuracy</th><th>F1</th><th>Sensitivity</th><th>N Test</th></tr></thead>
            <tbody>
              {psOverlap.map(r => (
                <tr key={r.subject}>
                  <td>{r.subject}</td>
                  <td><span className="badge bg-success">{pct(r.accuracy)}</span></td>
                  <td>{pct(r.f1)}</td>
                  <td>{pct(r.sensitivity)}</td>
                  <td>{r.n_test ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Cross-patient RF vs Ensemble side by side */}
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">Cross-Patient RF LOSO (per fold)</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Held-Out</th><th>Accuracy</th><th>F1</th></tr></thead>
                <tbody>
                  {cpRf.map(r => (
                    <tr key={r.held_out}>
                      <td>{r.held_out}</td>
                      <td><span className={`badge bg-${+r.accuracy >= 0.8 ? 'success' : +r.accuracy >= 0.6 ? 'warning' : 'danger'}`}>{pct(r.accuracy)}</span></td>
                      <td>{pct(r.f1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">Cross-Patient Ensemble LOSO (per fold)</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Held-Out</th><th>Accuracy</th><th>F1</th></tr></thead>
                <tbody>
                  {cpEns.map(r => (
                    <tr key={r.held_out}>
                      <td>{r.held_out}</td>
                      <td><span className={`badge bg-${+r.accuracy >= 0.8 ? 'success' : +r.accuracy >= 0.6 ? 'warning' : 'danger'}`}>{pct(r.accuracy)}</span></td>
                      <td>{pct(r.f1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Subject-wise by disease */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold">Subject-Wise GroupKFold — Per Disease</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th>Disease</th><th>Subjects</th><th>Accuracy</th><th>F1</th><th>Leakage Gap</th></tr></thead>
            <tbody>
              {swDis.map(r => (
                <tr key={r.disease}>
                  <td className="text-capitalize">{r.disease}</td>
                  <td>{r.n_subjects ?? '—'}</td>
                  <td><span className="badge bg-primary">{pct(r.accuracy)}</span></td>
                  <td>{pct(r.f1)}</td>
                  <td><span className={`badge bg-${+(r.leakage_gap || 0) < 0.05 ? 'success' : 'warning'}`}>{pct(r.leakage_gap)}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bonn external validation */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold">Bonn External Validation — {bonnMeta.dataset}</div>
        <div className="card-body">
          <div className="row mb-3">
            <div className="col-md-4"><strong>Samples:</strong> {bonnMeta.n_samples}</div>
            <div className="col-md-4"><strong>Balance:</strong> {bonnMeta.balance}</div>
            <div className="col-md-4"><strong>CV:</strong> {bonnMeta.cv}</div>
          </div>
          <p className="text-muted small">{bonnMeta.purpose}</p>
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th>Model</th><th>Accuracy</th><th>F1</th><th>AUC</th><th>Fold Accuracies</th></tr></thead>
            <tbody>
              {bonn.map(r => (
                <tr key={r.model}>
                  <td className="fw-semibold text-uppercase">{r.model}</td>
                  <td><span className="badge bg-danger">{pct(r.accuracy)}</span></td>
                  <td>{pct(r.f1)}</td>
                  <td>{fmtN(r.auc)}</td>
                  <td className="small text-muted">{(r.fold_acc || []).map(v => pct(v)).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return null;
  const methods = data.methodologies || [];
  const glossary = data.glossary || [];
  const thesis = data.thesis_context || {};

  return (
    <div>
      {/* Methodology Descriptions */}
      <div className="row mb-4">
        {methods.map(m => (
          <div key={m.id} className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small">{m.title}</div>
              <div className="card-body small">
                <p>{m.description}</p>
                <p><strong>Bias:</strong> {m.bias}</p>
                <p><strong>When to cite:</strong> {m.when_to_cite}</p>
                <div className="d-flex flex-wrap gap-1">
                  {(m.references || []).map(r => <span key={r} className="badge bg-light text-dark border">{r}</span>)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Thesis context */}
      <div className="card shadow-sm mb-4 border-primary">
        <div className="card-header fw-bold bg-primary text-white">Thesis Citation Order</div>
        <div className="card-body">
          <ol className="mb-2">
            {(thesis.recommended_cite_order || []).map((c, i) => <li key={i}>{c}</li>)}
          </ol>
          <p className="text-muted small mb-0"><strong>Q1 Reviewer Tip:</strong> {thesis.q1_reviewer_tip}</p>
          <p className="text-muted small mt-1">Source: <code>{thesis.data_source}</code></p>
        </div>
      </div>

      {/* Glossary */}
      <div className="card shadow-sm">
        <div className="card-header fw-bold">Glossary</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {glossary.map(g => (
                <tr key={g.term}><td className="fw-semibold">{g.term}</td><td>{g.definition}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function AccuracyOptionsDashboard() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [err,  setErr]  = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/accuracy-options/overview`).then(r => r.json()),
      fetch(`${API}/api/accuracy-options/breakdown`).then(r => r.json()),
      fetch(`${API}/api/accuracy-options/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err)  return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-3">Loading Accuracy Options data…</div>;

  const kpis = ov.kpis || {};

  return (
    <div>
      <div className="d-flex align-items-center mb-3 gap-3">
        <h3 className="mb-0">📊 Accuracy Methods Comparison</h3>
        <span className="badge bg-dark">{kpis.n_methods} methods</span>
        <span className="badge bg-success">Best: {pct(kpis.best_accuracy)}</span>
      </div>
      <p className="text-muted small mb-3">
        5 evaluation methodologies side-by-side: patient-specific, cross-patient LOSO, leakage-free GroupKFold,
        multi-disease in-sample, and Bonn external validation. Each with documented bias and thesis guidance.
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel    data={ov} />}
      {tab === 'breakdown'   && <BreakdownPanel   data={bd} />}
      {tab === 'definitions' && <DefinitionsPanel data={defs} />}
    </div>
  );
}
