'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',   label: 'Overview' },
  { id: 'breakdown',  label: 'Technique Sweep' },
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

function DeltaBadge({ delta }) {
  if (delta == null) return <span className="badge bg-secondary">—</span>;
  if (delta > 0) return <span className="badge bg-success">+{(delta * 100).toFixed(1)}%</span>;
  if (delta < 0) return <span className="badge bg-danger">{(delta * 100).toFixed(1)}%</span>;
  return <span className="badge bg-warning text-dark">±0%</span>;
}

function BoolBadge({ val, trueLabel = 'Yes', falseLabel = 'No' }) {
  return val
    ? <span className="badge bg-success">{trueLabel}</span>
    : <span className="badge bg-danger">{falseLabel}</span>;
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const k = data.kpis || {};
  const techs = data.techniques || [];
  const dist = data.class_distribution || {};

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Original Samples" value={k.original_samples} color="info" sub="from clinical.db" />
        <KPI label="EEG Features" value={k.n_features} color="primary" sub="47-feature pipeline" />
        <KPI label="Best Technique" value={k.best_technique} color="success" sub="by accuracy gain" />
        <KPI label="Baseline Accuracy" value={k.baseline_accuracy != null ? `${(k.baseline_accuracy * 100).toFixed(1)}%` : '—'} color="secondary" sub="pre-augmentation" />
      </div>
      <div className="row mb-4">
        <KPI label="Best Aug. Accuracy" value={k.best_augmented_accuracy != null ? `${(k.best_augmented_accuracy * 100).toFixed(1)}%` : '—'} color="success" sub="post-augmentation" />
        <KPI label="Imbalance Before" value={k.imbalance_ratio_before != null ? `1:${k.imbalance_ratio_before.toFixed(1)}` : '—'} color="danger" sub="ictal:inter-ictal" />
        <KPI label="Imbalance After SMOTE" value={k.imbalance_ratio_after_smote != null ? `1:${k.imbalance_ratio_after_smote.toFixed(1)}` : '—'} color="success" sub="after rebalancing" />
        <KPI label="Augmentation Methods" value={techs.length} color="primary" sub="evaluated" />
      </div>

      {/* Class distribution */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">Class Distribution (Original)</div>
        <div className="card-body">
          <div className="row">
            {Object.entries(dist).map(([cls, count]) => (
              <div key={cls} className="col-6 col-md-3 mb-2 text-center">
                <div className="h5 fw-bold text-info">{count}</div>
                <div className="small text-muted">{cls}</div>
              </div>
            ))}
          </div>
          {Object.keys(dist).length === 1 && (
            <div className="alert alert-info mt-2 mb-0 small">
              Single-disease dataset — augmentation focus is on intra-class variability to improve robustness, not class rebalancing.
            </div>
          )}
        </div>
      </div>

      {/* Techniques summary table */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">Augmentation Techniques — Summary</div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Technique</th>
                <th>Samples Generated</th>
                <th>Total Samples</th>
                <th>Accuracy</th>
                <th>Δ Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {techs.map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{t.name}</td>
                  <td>{t.samples_generated}</td>
                  <td>{t.total_samples}</td>
                  <td>{t.accuracy != null ? `${(t.accuracy * 100).toFixed(1)}%` : '—'}</td>
                  <td><DeltaBadge delta={t.accuracy_delta} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const results = data.results || [];
  const baseline = data.baseline_accuracy;

  // Group by technique
  const grouped = {};
  for (const r of results) {
    if (!grouped[r.technique]) grouped[r.technique] = [];
    grouped[r.technique].push(r);
  }

  return (
    <div>
      <div className="alert alert-secondary mb-3 small">
        Baseline accuracy (no augmentation): <strong>{baseline != null ? `${(baseline * 100).toFixed(1)}%` : '—'}</strong>.
        Each row shows one parameter configuration. Best results highlighted in green.
      </div>

      {Object.entries(grouped).map(([tech, rows]) => {
        const bestAcc = Math.max(...rows.map(r => r.accuracy || 0));
        return (
          <div key={tech} className="card mb-4">
            <div className="card-header fw-semibold">{tech}</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Parameter</th>
                    <th>Augmented</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                    <th>Δ Accuracy</th>
                    <th>Label Safe</th>
                    <th>Alters Dist.</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i} className={r.accuracy === bestAcc ? 'table-success' : ''}>
                      <td className="font-monospace small">{r.parameter}</td>
                      <td>{r.augmented_count}</td>
                      <td>{r.total_count}</td>
                      <td>{r.accuracy != null ? `${(r.accuracy * 100).toFixed(1)}%` : '—'}</td>
                      <td><DeltaBadge delta={r.delta} /></td>
                      <td><BoolBadge val={r.preserves_label} trueLabel="✔ Yes" falseLabel="✗ No" /></td>
                      <td><BoolBadge val={r.alters_distribution} trueLabel="Yes" falseLabel="No" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const techs = data.techniques || [];
  const bp = data.best_practices || [];
  const notes = data.eeg_specific_notes || [];

  return (
    <div>
      <h6 className="fw-semibold mb-3">Augmentation Techniques — Explained</h6>
      {techs.map((t, i) => (
        <div key={i} className="card mb-3">
          <div className="card-header fw-semibold">{t.name}</div>
          <div className="card-body">
            <p className="mb-2 small">{t.description}</p>
            <div className="row">
              <div className="col-md-4">
                <div className="text-muted small fw-semibold mb-1">Parameter</div>
                <div className="small">{t.parameter}</div>
              </div>
              <div className="col-md-4">
                <div className="text-success small fw-semibold mb-1">Strengths</div>
                <ul className="mb-0 ps-3 small">
                  {(t.strengths || []).map((s, j) => <li key={j}>{s}</li>)}
                </ul>
              </div>
              <div className="col-md-4">
                <div className="text-danger small fw-semibold mb-1">Weaknesses</div>
                <ul className="mb-0 ps-3 small">
                  {(t.weaknesses || []).map((w, j) => <li key={j}>{w}</li>)}
                </ul>
              </div>
            </div>
            {t.clinical_relevance && (
              <div className="alert alert-info mt-2 mb-0 small">
                <strong>Clinical relevance:</strong> {t.clinical_relevance}
              </div>
            )}
            {t.reference && (
              <div className="text-muted small mt-2"><em>Ref: {t.reference}</em></div>
            )}
          </div>
        </div>
      ))}

      {bp.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Best Practices</div>
          <ul className="list-group list-group-flush">
            {bp.map((p, i) => <li key={i} className="list-group-item small">{p}</li>)}
          </ul>
        </div>
      )}

      {notes.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">EEG-Specific Notes</div>
          <ul className="list-group list-group-flush">
            {notes.map((n, i) => <li key={i} className="list-group-item small">{n}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function DataAugmentationPage() {
  const [tab, setTab] = useState('overview');
  const [panels, setPanels] = useState({});
  const [loading, setLoading] = useState({});

  const load = async (id) => {
    if (panels[id] || loading[id]) return;
    setLoading(l => ({ ...l, [id]: true }));
    try {
      const r = await fetch(`${API}/api/data-augmentation/${id}`);
      const d = await r.json();
      setPanels(p => ({ ...p, [id]: d }));
    } catch (e) {
      setPanels(p => ({ ...p, [id]: { error: String(e) } }));
    } finally {
      setLoading(l => ({ ...l, [id]: false }));
    }
  };

  useEffect(() => { load('overview'); }, []);
  useEffect(() => { load(tab); }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: '1.6rem' }}>🧬</span>
        <div>
          <h4 className="mb-0 fw-bold">Data Augmentation Dashboard</h4>
          <div className="text-muted small">GAN · Time-Warp · Mixup · Jittering · SMOTE — EEG Seizure Classification</div>
        </div>
      </div>

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

      <div>
        {loading[tab] && <div className="text-muted p-3"><span className="spinner-border spinner-border-sm me-2" />Loading…</div>}
        {!loading[tab] && tab === 'overview'    && <OverviewPanel data={panels['overview']} />}
        {!loading[tab] && tab === 'breakdown'   && <BreakdownPanel data={panels['breakdown']} />}
        {!loading[tab] && tab === 'definitions' && <DefinitionsPanel data={panels['definitions']} />}
      </div>
    </div>
  );
}
