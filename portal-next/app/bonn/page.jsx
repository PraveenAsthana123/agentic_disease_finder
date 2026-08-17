'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'breakdown',   label: 'Fold Breakdown' },
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
  const color = value >= 0.95 ? 'success' : value >= 0.80 ? 'warning' : 'danger';
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
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const k      = data.kpis || {};
  const folds  = data.fold_performance || [];
  const classes= data.class_stats || [];
  const comp   = data.comparison || [];
  const feats  = data.feature_summary || [];
  const bars   = data.bar_chart || [];

  return (
    <div>
      {/* Purpose banner */}
      {data.purpose && (
        <div className="alert alert-info small mb-3">
          <strong>Purpose:</strong> {data.purpose}
        </div>
      )}

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="RF Accuracy"   value={k.rf_accuracy  != null ? `${(k.rf_accuracy  * 100).toFixed(1)}%` : '\u2014'} color="success" sub="5-fold CV, Bonn" />
        <KPI label="RF AUC"        value={k.rf_auc        != null ? `${(k.rf_auc        * 100).toFixed(1)}%` : '\u2014'} color="success" sub="area under ROC" />
        <KPI label="Ensemble Acc." value={k.ensemble_accuracy != null ? `${(k.ensemble_accuracy * 100).toFixed(1)}%` : '\u2014'} color="success" sub="RF + Ensemble" />
        <KPI label="Samples"       value={k.n_samples ? `${k.n_samples} (${k.balance})` : '\u2014'} color="info"    sub={`${k.n_features} features · ${k.cv}`} />
      </div>

      <div className="row mb-4">
        {/* Model comparison bar */}
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Model Performance (Bonn)</div>
            <div className="card-body">
              {bars.map(b => (
                <div key={b.model} className="mb-3">
                  <div className="d-flex justify-content-between small mb-1">
                    <span className="fw-semibold">{b.model}</span>
                    <span className="text-muted">AUC {(b.auc * 100).toFixed(1)}%  F1 {(b.f1 * 100).toFixed(1)}%</span>
                  </div>
                  <AccuracyBar value={b.accuracy} max={1} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Per-fold performance */}
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Per-Fold CV Performance</div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0">
                <thead>
                  <tr><th>Fold</th><th className="text-end">RF Acc.</th><th className="text-end">Ens. Acc.</th><th className="text-end">N Test</th></tr>
                </thead>
                <tbody>
                  {folds.map(f => (
                    <tr key={f.fold}>
                      <td className="fw-semibold">{f.fold}</td>
                      <td className="text-end">{(f.rf_accuracy * 100).toFixed(1)}%</td>
                      <td className="text-end">{(f.ens_accuracy * 100).toFixed(1)}%</td>
                      <td className="text-end text-muted">{f.n_test}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="row mb-4">
        {/* Cross-dataset comparison */}
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Cross-Dataset Comparison</div>
            <div className="card-body">
              {comp.map((c, i) => (
                <div key={i} className={`mb-3 ${c.highlight ? 'p-2 rounded border border-success bg-light' : ''}`}>
                  <div className="d-flex justify-content-between small mb-1">
                    <span className={c.highlight ? 'fw-bold text-success' : 'fw-semibold'}>{c.dataset}</span>
                  </div>
                  <AccuracyBar value={c.accuracy} max={1} />
                  {c.note && <div className="text-muted mt-1" style={{ fontSize: '0.7rem' }}>{c.note}</div>}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Class breakdown */}
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Bonn Dataset Classes (5-Class)</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead><tr><th>ID</th><th>Class</th><th>Category</th><th className="text-end">N</th></tr></thead>
                <tbody>
                  {classes.map(c => (
                    <tr key={c.class_id}>
                      <td><span className="badge bg-secondary">{c.class_id}</span></td>
                      <td className="fw-semibold">{c.label}</td>
                      <td>
                        <span className={`badge bg-${c.category === 'ictal' ? 'danger' : 'primary'}`}>
                          {c.category}
                        </span>
                      </td>
                      <td className="text-end">{c.n_samples}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Feature set */}
      <div className="row mb-4">
        <div className="col-12">
          <div className="card">
            <div className="card-header fw-semibold">Feature Pipeline ({(data.kpis || {}).n_features} features — identical to CHB-MIT pipeline)</div>
            <div className="card-body d-flex flex-wrap gap-3">
              {feats.map(fg => (
                <div key={fg.group} className="border rounded p-2" style={{ minWidth: 160 }}>
                  <div className="small fw-bold mb-1 text-muted">{fg.group} ({fg.count})</div>
                  <div className="d-flex flex-wrap gap-1">
                    {fg.features.map(f => (
                      <span key={f} className="badge bg-light text-dark border">{f}</span>
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

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const folds = data.folds_detail || [];
  const fi    = data.feature_importances || [];
  const cc    = data.class_confusion || [];
  const roc   = data.roc_points || [];
  const gen   = data.generalisation || {};

  return (
    <div>
      <div className="row mb-4">
        {/* Per-fold confusion matrices */}
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Per-Fold Confusion Summary</div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0">
                <thead>
                  <tr><th>Fold</th><th className="text-end">Acc.</th><th className="text-end">Prec.</th><th className="text-end">Recall</th><th className="text-end">F1</th><th className="text-end">TP/TN/FP/FN</th></tr>
                </thead>
                <tbody>
                  {folds.map(f => (
                    <tr key={f.fold} className={f.accuracy < 0.9 ? 'table-warning' : 'table-success'}>
                      <td className="fw-semibold">{f.fold}</td>
                      <td className="text-end">{(f.accuracy * 100).toFixed(0)}%</td>
                      <td className="text-end">{(f.precision * 100).toFixed(0)}%</td>
                      <td className="text-end">{(f.recall * 100).toFixed(0)}%</td>
                      <td className="text-end">{(f.f1 * 100).toFixed(0)}%</td>
                      <td className="text-end small text-muted">
                        {f.confusion.TP}/{f.confusion.TN}/{f.confusion.FP}/{f.confusion.FN}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Feature importances */}
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Feature Importances (RF Gini)</div>
            <div className="card-body" style={{ maxHeight: 300, overflowY: 'auto' }}>
              {fi.map(f => (
                <div key={f.feature} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>
                      <span className="fw-semibold">{f.feature}</span>
                      <span className="text-muted ms-2">[{f.group}]</span>
                    </span>
                    <span>{(f.importance * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress" style={{ height: 8 }}>
                    <div className="progress-bar bg-info" style={{ width: `${f.importance * 600}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row mb-4">
        {/* Class-level confusion */}
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Per-Class Accuracy</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead><tr><th>Class</th><th className="text-end">Correct</th><th className="text-end">Wrong</th><th className="text-end">Acc.</th></tr></thead>
                <tbody>
                  {cc.map(c => (
                    <tr key={c.class_id}>
                      <td><span className="fw-semibold">{c.label}</span></td>
                      <td className="text-end text-success">{c.predicted_correct}</td>
                      <td className="text-end text-danger">{c.predicted_wrong}</td>
                      <td className="text-end fw-bold">{(c.accuracy * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* ROC curve */}
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">ROC Curve (RF, Bonn)</div>
            <div className="card-body">
              <div className="small text-muted mb-2">AUC = 1.00 (all points at TPR ≈ 1.0)</div>
              <div className="p-0">
                {roc.slice(0, 8).map((pt, i) => (
                  <div key={i} className="d-flex justify-content-between small mb-1">
                    <span className="text-muted">FPR {(pt.fpr * 100).toFixed(1)}%</span>
                    <div className="progress flex-grow-1 mx-2" style={{ height: 8, marginTop: 4 }}>
                      <div className="progress-bar bg-success" style={{ width: `${pt.tpr * 100}%` }} />
                    </div>
                    <span>TPR {(pt.tpr * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Generalisation analysis */}
        <div className="col-md-4 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Generalisation Analysis</div>
            <div className="card-body">
              <table className="table table-sm mb-3">
                <tbody>
                  <tr><td className="text-muted small">Bonn Accuracy</td><td className="fw-bold text-success">{gen.bonn_accuracy != null ? `${(gen.bonn_accuracy * 100).toFixed(1)}%` : '\u2014'}</td></tr>
                  <tr><td className="text-muted small">CHB-MIT In-Sample</td><td className="fw-bold">{gen.chbmit_insample != null ? `${(gen.chbmit_insample * 100).toFixed(1)}%` : '\u2014'}</td></tr>
                  <tr><td className="text-muted small">CHB-MIT Cross-Patient</td><td className="fw-bold text-warning">{gen.chbmit_crosspatient != null ? `${(gen.chbmit_crosspatient * 100).toFixed(1)}%` : '\u2014'}</td></tr>
                  <tr><td className="text-muted small">Gap (In-Sample vs Bonn)</td><td className={`fw-bold ${gen.gap_insample_bonn > 0 ? 'text-warning' : 'text-success'}`}>{gen.gap_insample_bonn != null ? `${(gen.gap_insample_bonn * 100).toFixed(1)}%` : '\u2014'}</td></tr>
                </tbody>
              </table>
              {gen.interpretation && (
                <div className="small text-muted" style={{ fontSize: '0.72rem' }}>{gen.interpretation}</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const terms = data.terms || [];
  const refs  = data.references || [];
  const interp = data.interpretation || {};

  return (
    <div>
      {/* Dataset description */}
      {data.dataset_description && (
        <div className="alert alert-secondary small mb-3">
          <strong>Dataset:</strong> {data.dataset_description}
        </div>
      )}

      {/* Why Bonn */}
      {data.why_bonn && (
        <div className="alert alert-info small mb-3">
          <strong>Why external validation on Bonn?</strong> {data.why_bonn}
        </div>
      )}

      {/* Interpretation */}
      {(interp.headline || interp.caveat || interp.thesis_impact) && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Clinical &amp; Thesis Interpretation</div>
          <div className="card-body">
            {interp.headline && <p className="small mb-2"><strong>Headline:</strong> {interp.headline}</p>}
            {interp.caveat   && <p className="small mb-2 text-warning"><strong>Caveat:</strong> {interp.caveat}</p>}
            {interp.thesis_impact && <p className="small mb-0 text-success"><strong>DBA Thesis Impact:</strong> {interp.thesis_impact}</p>}
          </div>
        </div>
      )}

      {/* Terms */}
      <div className="card mb-3">
        <div className="card-header fw-semibold">Term Definitions</div>
        <div className="card-body p-0">
          <table className="table table-sm table-striped mb-0">
            <thead><tr><th style={{ minWidth: 160 }}>Term</th><th>Definition</th><th>Standard</th></tr></thead>
            <tbody>
              {terms.map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold text-nowrap">{t.term}</td>
                  <td className="small">{t.definition}</td>
                  <td className="small text-muted">{t.standard}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* References */}
      <div className="card">
        <div className="card-header fw-semibold">References ({refs.length})</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead><tr><th>#</th><th>Citation</th><th>Relevance</th></tr></thead>
            <tbody>
              {refs.map((r, i) => (
                <tr key={i}>
                  <td className="text-muted small">{i + 1}</td>
                  <td className="small">{r.citation}</td>
                  <td className="small text-muted">{r.relevance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function BonnValidationDashboard() {
  const [tab, setTab]             = useState('overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/bonn/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(() => setOverview({ error: 'Failed to load overview' }));
    fetch(`${API}/api/bonn/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(() => setBreakdown({ error: 'Failed to load breakdown' }));
    fetch(`${API}/api/bonn/definitions`)
      .then(r => r.json()).then(setDefinitions)
      .catch(() => setDefinitions({ error: 'Failed to load definitions' }));
  }, []);

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1">Bonn External Validation Dashboard</h4>
      <p className="text-muted small mb-3">
        Cross-dataset generalisation evidence: CHB-MIT-trained feature pipeline evaluated on the
        Bonn University epilepsy EEG benchmark (Andrzejak et al., 2001/2012) — 200 samples,
        5 classes, 14 features, stratified 5-fold CV. Directly addresses the &ldquo;does it
        generalise beyond CHB-MIT?&rdquo; reviewer objection.
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
      {tab === 'overview'    && <OverviewPanel    data={overview}    />}
      {tab === 'breakdown'   && <BreakdownPanel   data={breakdown}   />}
      {tab === 'definitions' && <DefinitionsPanel data={definitions} />}
    </div>
  );
}
