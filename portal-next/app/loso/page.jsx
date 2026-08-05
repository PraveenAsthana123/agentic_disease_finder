'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',   label: 'Overview' },
  { id: 'breakdown',  label: 'Per-Subject' },
  { id: 'definitions', label: 'Definitions' },
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

function SensBar({ value }) {
  const pct = Math.round(value * 100);
  const color = value >= 0.6 ? 'success' : value >= 0.2 ? 'warning' : 'danger';
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 10 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="small fw-semibold" style={{ minWidth: 44 }}>{pct.toFixed(1)}%</span>
    </div>
  );
}

function tierBadge(tier) {
  const map = { high: 'success', moderate: 'warning', low: 'secondary', zero: 'danger' };
  const label = { high: 'High', moderate: 'Moderate', low: 'Low', zero: 'Zero' };
  return <span className={`badge bg-${map[tier] || 'secondary'}`}>{label[tier] ?? tier}</span>;
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const kpis  = data.kpis || {};
  const tiers = data.sensitivity_tiers || {};
  const bench = data.in_sample_benchmark || {};
  const ds    = data.dataset || {};
  const gap   = data.loso_vs_insample_gap;

  return (
    <div>
      {/* KPI Row */}
      <div className="row mb-4">
        <KPI label="Subjects" value={kpis.subjects} color="info" sub="CHB-MIT LOSO folds" />
        <KPI label="Mean Sensitivity"
          value={kpis.mean_sensitivity != null ? `${(kpis.mean_sensitivity * 100).toFixed(1)}%` : '—'}
          color={kpis.mean_sensitivity >= 0.5 ? 'success' : 'warning'}
          sub="across 24 subjects" />
        <KPI label="Mean Specificity"
          value={kpis.mean_specificity != null ? `${(kpis.mean_specificity * 100).toFixed(1)}%` : '—'}
          color="success" sub="non-seizure rejection" />
        <KPI label="Mean AUC"
          value={kpis.mean_auc != null ? kpis.mean_auc.toFixed(3) : '—'}
          color={kpis.mean_auc >= 0.8 ? 'success' : 'warning'}
          sub="ROC area under curve" />
      </div>
      <div className="row mb-4">
        <KPI label="High-Sensitivity Subjects" value={kpis.high_sens_subjects} color="success" sub="≥60% sensitivity" />
        <KPI label="Zero-Sensitivity Subjects" value={kpis.zero_sens_subjects} color="danger" sub="0% — no cross-patient detect" />
        <KPI label="Seizure Epochs" value={kpis.total_seizure_epochs?.toLocaleString()} color="secondary" sub="total ictal windows" />
        <KPI label="In-Sample AUC"
          value={bench.auc != null ? bench.auc.toFixed(3) : '—'}
          color="primary"
          sub={bench.model ? bench.model : 'best CHB-MIT model'} />
      </div>

      {/* Key finding alert */}
      <div className="alert alert-info mb-4">
        <strong>Key Finding:</strong> Wide per-subject spread (chb09 89.7% → chb14/15 0% sensitivity).
        Seizure morphology is highly patient-specific — a universal cross-patient detector performs
        fundamentally differently than a patient-specific model.
        {gap != null && (
          <span className="ms-2 text-muted">
            Generalization gap: <strong>{(gap * 100).toFixed(1)}% AUC drop</strong> (in-sample → LOSO).
          </span>
        )}
      </div>

      {/* Sensitivity tiers */}
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Sensitivity Tiers</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Tier</th><th>Count</th><th>Subjects</th></tr>
                </thead>
                <tbody>
                  {Object.entries(tiers).map(([k, v]) => (
                    <tr key={k}>
                      <td>{tierBadge(k)} {v.label}</td>
                      <td className="fw-semibold">{v.count}</td>
                      <td className="small text-muted">{(v.subjects || []).join(', ') || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Dataset — CHB-MIT</div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <tbody>
                  {Object.entries(ds).map(([k, v]) => (
                    <tr key={k}>
                      <td className="text-muted small" style={{ width: 140 }}>{k.replace(/_/g,' ')}</td>
                      <td className="small">{String(v)}</td>
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
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const subjects   = data.subjects || [];
  const aucDist    = data.auc_distribution || [];
  const inSample   = data.in_sample_models || [];

  return (
    <div>
      {/* Per-subject table */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Per-Subject LOSO Results — {data.total_subjects} subjects (sorted by sensitivity)</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-dark">
                <tr>
                  <th>Subject</th>
                  <th>Seizure Epochs</th>
                  <th style={{ minWidth: 180 }}>Sensitivity</th>
                  <th>Specificity</th>
                  <th>AUC</th>
                  <th>Tier</th>
                </tr>
              </thead>
              <tbody>
                {subjects.map(r => (
                  <tr key={r.subject}>
                    <td className="fw-semibold">{r.subject}</td>
                    <td>{r.seizure_epochs}</td>
                    <td><SensBar value={r.sensitivity} /></td>
                    <td>{r.specificity_pct}%</td>
                    <td className={r.auc >= 0.9 ? 'text-success fw-semibold' : r.auc < 0.7 ? 'text-danger' : ''}>{r.auc_pct}%</td>
                    <td>{tierBadge(r.tier)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="row mb-4">
        {/* AUC distribution */}
        <div className="col-md-5 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">AUC Distribution</div>
            <div className="card-body">
              {aucDist.map(d => (
                <div key={d.bucket} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{d.bucket}</span><span className="fw-semibold">{d.count} subjects</span>
                  </div>
                  <div className="progress" style={{ height: 12 }}>
                    <div
                      className={`progress-bar ${d.bucket === '>=0.95' ? 'bg-success' : d.bucket === '0.85-0.95' ? 'bg-primary' : d.bucket === '0.70-0.85' ? 'bg-warning' : 'bg-danger'}`}
                      style={{ width: `${Math.round(d.count / 24 * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* In-sample benchmark */}
        {inSample.length > 0 && (
          <div className="col-md-7 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">In-Sample CHB-MIT Models (patient-specific CV)</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Model</th><th>Type</th><th>Accuracy</th><th>F1</th><th>AUC</th></tr>
                  </thead>
                  <tbody>
                    {inSample.map((m, i) => (
                      <tr key={i}>
                        <td className="small">{m.model_name}</td>
                        <td className="small text-muted">{m.model_type}</td>
                        <td>{(m.accuracy * 100).toFixed(1)}%</td>
                        <td>{(m.f1_score * 100).toFixed(1)}%</td>
                        <td className="fw-semibold text-success">{(m.auc_roc * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;

  const sections = [
    { title: 'CV Methods',         key: 'cv_method' },
    { title: 'Metrics',            key: 'metrics' },
    { title: 'Sensitivity Tiers',  key: 'sensitivity_tiers' },
    { title: 'Glossary',           key: 'glossary' },
  ];

  return (
    <div>
      {data.key_finding && (
        <div className="alert alert-primary mb-4">
          <strong>Key Research Finding:</strong> {data.key_finding}
        </div>
      )}
      {data.dataset && (
        <div className="card shadow-sm mb-4">
          <div className="card-header fw-semibold">Dataset — {data.dataset.name}</div>
          <div className="card-body">
            <table className="table table-sm mb-0">
              <tbody>
                {Object.entries(data.dataset).map(([k, v]) => (
                  <tr key={k}>
                    <td className="text-muted small" style={{ width: 160 }}>{k.replace(/_/g,' ')}</td>
                    <td className="small">{String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {sections.map(({ title, key }) =>
        data[key] ? (
          <div className="card shadow-sm mb-3" key={key}>
            <div className="card-header fw-semibold">{title}</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <tbody>
                  {Object.entries(data[key]).map(([k, v]) => (
                    <tr key={k}>
                      <td className="text-muted small fw-semibold" style={{ width: 180 }}>{k.replace(/_/g,' ')}</td>
                      <td className="small">{String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null
      )}
    </div>
  );
}

export default function LOSODashboard() {
  const [tab,  setTab]  = useState('overview');
  const [ov,   setOv]   = useState(null);
  const [bk,   setBk]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [err,  setErr]  = useState(null);

  useEffect(() => {
    const hdr = { 'Content-Type': 'application/json' };
    Promise.all([
      fetch(`${API}/api/loso/overview`,    { headers: hdr }).then(r => r.json()),
      fetch(`${API}/api/loso/breakdown`,   { headers: hdr }).then(r => r.json()),
      fetch(`${API}/api/loso/definitions`, { headers: hdr }).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1">🧠 LOSO Cross-Validation — CHB-MIT</h4>
        <p className="text-muted small mb-0">
          Leave-one-subject-out cross-validation on 24 CHB-MIT subjects · patient-independent generalization metric ·
          Mean Sens 35.1% · Spec 96.9% · AUC 0.846
        </p>
      </div>

      {err && <div className="alert alert-danger small">API error: {err}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel    data={ov}   />}
      {tab === 'breakdown'   && <BreakdownPanel   data={bk}   />}
      {tab === 'definitions' && <DefinitionsPanel data={defs} />}
    </div>
  );
}
