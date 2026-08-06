'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const kappaColor = k =>
  k >= 0.81 ? 'success' : k >= 0.61 ? 'primary' : k >= 0.41 ? 'warning' : 'danger';
const kappaLabel = k =>
  k >= 0.81 ? 'Almost perfect' : k >= 0.61 ? 'Substantial' : k >= 0.41 ? 'Moderate' : 'Fair';

const LABEL_COLORS = {
  seizure: '#ef4444',
  spike: '#f59e0b',
  artifact: '#94a3b8',
  normal: '#22c55e',
  slowing: '#6366f1',
  burst_suppression: '#ec4899',
};

export default function AnnotationQCPage() {
  const [ov, setOv]       = useState(null);
  const [agr, setAgr]     = useState(null);
  const [defs, setDefs]   = useState(null);
  const [tab, setTab]     = useState('overview');
  const [err, setErr]     = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/annotation/overview`).then(r => r.json()),
      fetch(`${API}/api/annotation/agreement`).then(r => r.json()),
      fetch(`${API}/api/annotation/definitions`).then(r => r.json()),
    ])
      .then(([o, a, d]) => { setOv(o); setAgr(a); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err)  return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-3">Loading annotation QC…</div>;

  const TABS = [
    { id: 'overview',    label: '📊 Overview' },
    { id: 'agreement',   label: '🤝 Agreement' },
    { id: 'subjects',    label: '👤 Subjects' },
    { id: 'annotators',  label: '✏️ Annotators' },
    { id: 'labels',      label: '🏷️ Labels' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const ag = ov.agreement || {};
  const pairwise = agr?.pairwise_kappas || ag.pairwise || [];
  const subjects = ov.subject_stats || [];
  const annotators = ov.annotator_stats || [];
  const labelDist = ov.label_distribution || [];
  const annotationLabels = ov.annotation_labels || [];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-4">
        <div>
          <h2 className="mb-0">🏷️ Annotation Quality Control</h2>
          <small className="text-muted">
            Inter-annotator agreement · Label Studio / CVAT · CHB-MIT PhysioNet seizure annotations
          </small>
        </div>
        <div className="ms-auto">
          <span className={`badge bg-${kappaColor(ag.cohens_kappa_mean || 0)} fs-6`}>
            κ = {(ag.cohens_kappa_mean || 0).toFixed(3)} — {kappaLabel(ag.cohens_kappa_mean || 0)}
          </span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="row g-3 mb-4">
        {[
          { label: 'Total Annotations', value: ov.total_annotations, icon: '📝', color: 'primary' },
          { label: 'Subjects (CHB-MIT)', value: ov.total_subjects, icon: '👤', color: 'info' },
          { label: 'Gold Seizures', value: ov.total_seizures_gold, icon: '⚡', color: 'warning' },
          { label: 'Seizure Duration', value: `${(ov.total_seizure_seconds || 0).toLocaleString()}s`, icon: '⏱️', color: 'secondary' },
          { label: "Cohen's κ (mean)", value: (ag.cohens_kappa_mean || 0).toFixed(3), icon: '🤝', color: kappaColor(ag.cohens_kappa_mean || 0) },
          { label: "Krippendorff's α", value: (ag.krippendorff_alpha || ov.agreement?.krippendorff_alpha || 0).toFixed(3), icon: '📐', color: kappaColor(ag.krippendorff_alpha || ov.agreement?.krippendorff_alpha || 0) },
        ].map(({ label, value, icon, color }) => (
          <div key={label} className="col-6 col-md-4 col-lg-2">
            <div className={`card border-${color} h-100`}>
              <div className="card-body text-center p-2">
                <div className="fs-4">{icon}</div>
                <div className={`fs-5 fw-bold text-${color}`}>{value}</div>
                <small className="text-muted">{label}</small>
              </div>
            </div>
          </div>
        ))}
      </div>

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

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div className="row g-4">
          {/* Agreement Summary */}
          <div className="col-md-6">
            <div className="card">
              <div className="card-header fw-semibold">🤝 Agreement Summary</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <tbody>
                    <tr>
                      <td>Cohen's κ (mean)</td>
                      <td>
                        <span className={`badge bg-${kappaColor(ag.cohens_kappa_mean || 0)}`}>
                          {(ag.cohens_kappa_mean || 0).toFixed(3)}
                        </span>
                      </td>
                      <td className="text-muted small">{kappaLabel(ag.cohens_kappa_mean || 0)}</td>
                    </tr>
                    <tr>
                      <td>Krippendorff's α</td>
                      <td>
                        <span className={`badge bg-${kappaColor(ag.krippendorff_alpha || 0)}`}>
                          {(ag.krippendorff_alpha || 0).toFixed(3)}
                        </span>
                      </td>
                      <td className="text-muted small">{agr?.alpha_interpretation || ''}</td>
                    </tr>
                    <tr>
                      <td>Annotators</td>
                      <td colSpan={2}><strong>{ag.n_annotators || 0}</strong> (incl. gold standard)</td>
                    </tr>
                    <tr>
                      <td>Items Rated</td>
                      <td colSpan={2}><strong>{(ag.n_items || 0).toLocaleString()}</strong> windows</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Label Distribution */}
          <div className="col-md-6">
            <div className="card">
              <div className="card-header fw-semibold">🏷️ Label Distribution</div>
              <div className="card-body">
                {labelDist.map(({ label, count, percent }) => (
                  <div key={label} className="mb-2">
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small text-capitalize">{label.replace(/_/g, ' ')}</span>
                      <span className="small text-muted">{count} ({percent}%)</span>
                    </div>
                    <div className="progress" style={{ height: 8 }}>
                      <div
                        className="progress-bar"
                        style={{
                          width: `${percent}%`,
                          backgroundColor: LABEL_COLORS[label] || '#64748b',
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Kappa Interpretation Scale */}
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">📊 Kappa Interpretation Scale</div>
              <div className="card-body">
                <div className="row g-2">
                  {(agr?.scale || [
                    { range: '0.81 – 1.00', label: 'Almost perfect' },
                    { range: '0.61 – 0.80', label: 'Substantial' },
                    { range: '0.41 – 0.60', label: 'Moderate' },
                    { range: '0.21 – 0.40', label: 'Fair' },
                    { range: '0.00 – 0.20', label: 'Slight' },
                    { range: '< 0.00',      label: 'Poor' },
                  ]).map(({ range, label }) => (
                    <div key={range} className="col-6 col-md-4 col-lg-2">
                      <div className="border rounded p-2 text-center">
                        <div className="fw-semibold small">{label}</div>
                        <div className="text-muted" style={{ fontSize: '0.75rem' }}>{range}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Agreement Tab */}
      {tab === 'agreement' && (
        <div className="row g-4">
          <div className="col-md-8">
            <div className="card">
              <div className="card-header fw-semibold">📐 Pairwise Cohen's Kappa</div>
              <div className="card-body p-0">
                <table className="table table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Annotator A</th>
                      <th>Annotator B</th>
                      <th>κ</th>
                      <th>Interpretation</th>
                      <th>Bar</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pairwise.map((p, i) => (
                      <tr key={i}>
                        <td><code>{p.annotator_a}</code></td>
                        <td><code>{p.annotator_b}</code></td>
                        <td>
                          <span className={`badge bg-${kappaColor(p.kappa)}`}>
                            {p.kappa.toFixed(4)}
                          </span>
                        </td>
                        <td className="small text-muted">{kappaLabel(p.kappa)}</td>
                        <td style={{ width: 120 }}>
                          <div className="progress" style={{ height: 10 }}>
                            <div
                              className={`progress-bar bg-${kappaColor(p.kappa)}`}
                              style={{ width: `${Math.max(0, p.kappa * 100).toFixed(0)}%` }}
                            />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-4">
            <div className="card">
              <div className="card-header fw-semibold">📏 Krippendorff's α</div>
              <div className="card-body">
                <div className="text-center mb-3">
                  <div className={`display-6 fw-bold text-${kappaColor(agr?.krippendorff_alpha || 0)}`}>
                    {(agr?.krippendorff_alpha || 0).toFixed(4)}
                  </div>
                  <div className="text-muted small">{agr?.alpha_interpretation}</div>
                </div>
                <div className="progress mb-3" style={{ height: 18 }}>
                  <div
                    className={`progress-bar bg-${kappaColor(agr?.krippendorff_alpha || 0)}`}
                    style={{ width: `${Math.max(0, (agr?.krippendorff_alpha || 0) * 100).toFixed(0)}%` }}
                  >
                    α = {(agr?.krippendorff_alpha || 0).toFixed(3)}
                  </div>
                </div>
                <p className="small text-muted mb-0">
                  Recommended threshold for reliable annotation: <strong>α ≥ 0.80</strong>
                  (Krippendorff, 2011). N = {agr?.n_annotators} annotators,{' '}
                  {(agr?.n_items || 0).toLocaleString()} items.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Subjects Tab */}
      {tab === 'subjects' && (
        <div className="card">
          <div className="card-header fw-semibold">👤 Per-Subject Statistics</div>
          <div className="card-body p-0">
            <table className="table table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Subject</th>
                  <th>Annotations</th>
                  <th>Seizures (gold)</th>
                  <th>Labels Used</th>
                  <th>Annotators</th>
                </tr>
              </thead>
              <tbody>
                {subjects.length > 0 ? subjects.map((s, i) => (
                  <tr key={i}>
                    <td><code>{s.subject}</code></td>
                    <td>{s.n_annotations}</td>
                    <td><span className="badge bg-warning text-dark">{s.n_seizures_gold}</span></td>
                    <td>{s.labels_used}</td>
                    <td>{s.annotators_count}</td>
                  </tr>
                )) : (
                  <tr>
                    <td colSpan={5} className="text-center text-muted py-4">
                      No per-subject data available
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Annotators Tab */}
      {tab === 'annotators' && (
        <div className="card">
          <div className="card-header fw-semibold">✏️ Per-Annotator Statistics</div>
          <div className="card-body p-0">
            <table className="table table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Annotator</th>
                  <th>Annotations</th>
                  <th>Labels Used</th>
                  <th>Subjects Covered</th>
                  <th>Role</th>
                </tr>
              </thead>
              <tbody>
                {annotators.map((a, i) => (
                  <tr key={i}>
                    <td>
                      <code>{a.annotator}</code>
                      {a.annotator === 'annotator_gold' && (
                        <span className="badge bg-warning text-dark ms-2">Gold</span>
                      )}
                    </td>
                    <td>{a.n_annotations}</td>
                    <td>{a.labels_used}</td>
                    <td>{a.subjects_covered}</td>
                    <td className="text-muted small">
                      {a.annotator === 'annotator_gold' ? 'Reference standard' : 'Clinical annotator'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Labels Tab */}
      {tab === 'labels' && (
        <div className="row g-3">
          {annotationLabels.map(({ id, name, color, hotkey }) => (
            <div key={id} className="col-6 col-md-4 col-lg-3">
              <div className="card h-100" style={{ borderLeft: `4px solid ${color || '#64748b'}` }}>
                <div className="card-body">
                  <div className="d-flex justify-content-between align-items-start mb-2">
                    <h6 className="card-title mb-0">{name}</h6>
                    <kbd>{hotkey}</kbd>
                  </div>
                  <div className="d-flex align-items-center gap-2">
                    <div
                      style={{
                        width: 16,
                        height: 16,
                        borderRadius: 3,
                        backgroundColor: color || '#64748b',
                      }}
                    />
                    <code className="small text-muted">{id}</code>
                  </div>
                  {labelDist.find(l => l.label === id) && (
                    <div className="mt-2">
                      <div className="text-muted small">
                        {labelDist.find(l => l.label === id).count} annotations (
                        {labelDist.find(l => l.label === id).percent}%)
                      </div>
                      <div className="progress mt-1" style={{ height: 6 }}>
                        <div
                          className="progress-bar"
                          style={{
                            width: `${labelDist.find(l => l.label === id).percent}%`,
                            backgroundColor: color || '#64748b',
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div className="row g-4">
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">ℹ️ About</div>
              <div className="card-body">
                <h6>{defs.title}</h6>
                <p className="text-muted small mb-0">{defs.description}</p>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">📐 Metrics</div>
              <ul className="list-group list-group-flush">
                {(defs.metrics || []).map((m, i) => (
                  <li key={i} className="list-group-item">
                    <div className="fw-semibold small">{m.name}</div>
                    <div className="text-muted small">{m.description}</div>
                    {m.reference && (
                      <div className="text-muted" style={{ fontSize: '0.72rem' }}>
                        Ref: {m.reference}
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">🛠️ Tools</div>
              <ul className="list-group list-group-flush">
                {Object.entries(defs.tools || {}).map(([key, tool]) => (
                  <li key={key} className="list-group-item">
                    <div className="d-flex justify-content-between align-items-start">
                      <div className="fw-semibold small">{tool.name}</div>
                      <span className="badge bg-secondary">{tool.version}</span>
                    </div>
                    <div className="text-muted small">{tool.role}</div>
                    <ul className="mt-1 mb-0 ps-3">
                      {(tool.features || []).slice(0, 3).map((f, i) => (
                        <li key={i} className="text-muted" style={{ fontSize: '0.72rem' }}>{f}</li>
                      ))}
                    </ul>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">📡 Data Source</div>
              <div className="card-body text-muted small">
                {defs.data_source}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
