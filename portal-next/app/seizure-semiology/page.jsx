'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'breakdown',   label: 'Per-Patient' },
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

function FallRiskBadge({ level }) {
  const map = { high: 'danger', medium: 'warning', low: 'success' };
  return <span className={`badge bg-${map[level] || 'secondary'}`}>{level || '—'}</span>;
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const typeDist = data.type_distribution || [];
  const zoneDist = data.zone_distribution || [];
  const lat = data.lateralisation || {};
  const hist = data.confidence_histogram || [];
  const modelPerf = data.model_performance || {};
  const perClass = data.per_class_metrics || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Patients" value={data.total_patients} color="info" sub="with semiology events" />
        <KPI label="Events Classified" value={data.total_events_classified} color="primary" sub="AI semiology calls" />
        <KPI label="Semiology Types" value={data.semiology_types_detected} color="success" sub="ILAE categories" />
        <KPI label="Avg Confidence" value={data.average_confidence != null ? `${(data.average_confidence * 100).toFixed(1)}%` : '—'} color="secondary" sub="AI classifier" />
      </div>
      <div className="row mb-4">
        <KPI label="Fall-Risk Events" value={data.fall_risk_events} color="danger" sub="high/medium risk" />
        <KPI label="Fall-Risk %" value={data.fall_risk_pct != null ? `${data.fall_risk_pct.toFixed(1)}%` : '—'} color="danger" sub="of all events" />
        {modelPerf.overall_accuracy != null && (
          <KPI label="Model Accuracy" value={`${(modelPerf.overall_accuracy * 100).toFixed(1)}%`} color="success" sub="semiology classifier" />
        )}
        {modelPerf.macro_f1 != null && (
          <KPI label="Macro F1" value={modelPerf.macro_f1.toFixed(3)} color="primary" sub="across all types" />
        )}
      </div>

      {/* Semiology Type Distribution */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">Semiology Type Distribution</div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Semiology Type</th>
                <th>Events (n)</th>
                <th>Share</th>
                <th>Bar</th>
              </tr>
            </thead>
            <tbody>
              {typeDist.map((t, i) => {
                const total = typeDist.reduce((s, x) => s + x.count, 0);
                const pct = total > 0 ? (t.count / total) * 100 : 0;
                return (
                  <tr key={i}>
                    <td className="fw-semibold">{t.type}</td>
                    <td>{t.count}</td>
                    <td>{pct.toFixed(1)}%</td>
                    <td style={{ width: '30%' }}>
                      <div className="progress" style={{ height: '10px' }}>
                        <div className="progress-bar bg-primary" style={{ width: `${pct}%` }} />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Inferred Localisation Zones */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">Inferred Localisation Zones (Semiology → Focus)</div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr><th>Zone</th><th>Events (n)</th><th>Distribution</th></tr>
            </thead>
            <tbody>
              {zoneDist.map((z, i) => {
                const total = zoneDist.reduce((s, x) => s + x.count, 0);
                const pct = total > 0 ? (z.count / total) * 100 : 0;
                return (
                  <tr key={i}>
                    <td>{z.zone}</td>
                    <td>{z.count}</td>
                    <td style={{ width: '35%' }}>
                      <div className="progress" style={{ height: '10px' }}>
                        <div className="progress-bar bg-info" style={{ width: `${pct}%` }} />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Lateralisation */}
      {Object.keys(lat).length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">Lateralisation</div>
          <div className="card-body">
            <div className="row">
              {Object.entries(lat).map(([k, v]) => (
                <div key={k} className="col-4 col-md-2 text-center mb-2">
                  <div className="h5 fw-bold text-primary">{v}</div>
                  <div className="small text-muted text-capitalize">{k.replace(/_/g, ' ')}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Confidence Histogram */}
      {hist.length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">Confidence Distribution</div>
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr><th>Confidence Bin</th><th>Events</th><th>Bar</th></tr>
              </thead>
              <tbody>
                {hist.map((h, i) => {
                  const maxCount = Math.max(...hist.map(x => x.count));
                  const pct = maxCount > 0 ? (h.count / maxCount) * 100 : 0;
                  return (
                    <tr key={i}>
                      <td className="font-monospace small">{h.bin}</td>
                      <td>{h.count}</td>
                      <td style={{ width: '40%' }}>
                        <div className="progress" style={{ height: '10px' }}>
                          <div className="progress-bar bg-success" style={{ width: `${pct}%` }} />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Per-Class Classifier Metrics */}
      {perClass.length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">Per-Class Classifier Metrics</div>
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Type</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
              </thead>
              <tbody>
                {perClass.map((m, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{m.type}</td>
                    <td>{m.precision != null ? m.precision.toFixed(3) : '—'}</td>
                    <td>{m.recall != null ? m.recall.toFixed(3) : '—'}</td>
                    <td>{m.f1 != null ? m.f1.toFixed(3) : '—'}</td>
                    <td>{m.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const profiles = data.patient_profiles || [];
  const cmLabels = data.confusion_labels || [];
  const cm = data.confusion_matrix || [];

  return (
    <div>
      <div className="alert alert-secondary mb-3 small">
        {data.total_patients_with_events} patients have semiology-classified events. Each card shows their detected types,
        fall-risk level, AI–clinician agreement, and individual event details.
      </div>

      {/* Confusion Matrix */}
      {cm.length > 0 && cmLabels.length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">Confusion Matrix — Semiology Classifier</div>
          <div className="table-responsive">
            <table className="table table-sm table-bordered mb-0" style={{ fontSize: '0.72rem' }}>
              <thead className="table-light">
                <tr>
                  <th>Actual \ Predicted</th>
                  {cmLabels.map((l, i) => <th key={i} className="text-center">{l}</th>)}
                </tr>
              </thead>
              <tbody>
                {cm.map((row, ri) => (
                  <tr key={ri}>
                    <th className="table-light">{cmLabels[ri]}</th>
                    {row.map((val, ci) => (
                      <td key={ci} className={`text-center ${ri === ci ? 'table-success fw-bold' : val > 0 ? 'table-warning' : ''}`}>{val}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Patient Profiles */}
      {profiles.map((p, i) => (
        <div key={i} className="card mb-3">
          <div className="card-header d-flex justify-content-between align-items-center">
            <span className="fw-semibold">{p.patient_id}{p.name ? ` — ${p.name}` : ''}</span>
            <div className="d-flex gap-2 align-items-center">
              <FallRiskBadge level={p.fall_risk_level} />
              {p.ai_clinician_agreement_pct != null && (
                <span className="badge bg-info">AI–Clinician: {p.ai_clinician_agreement_pct.toFixed(0)}%</span>
              )}
            </div>
          </div>
          <div className="card-body">
            <div className="row mb-2">
              <div className="col-sm-3 small text-muted">Age / Sex</div>
              <div className="col-sm-9 small">{p.age ?? '—'} / {p.sex || '—'}</div>
            </div>
            <div className="row mb-2">
              <div className="col-sm-3 small text-muted">Detected Types</div>
              <div className="col-sm-9 small">{(p.types_detected || []).join(', ') || '—'}</div>
            </div>
            <div className="row mb-3">
              <div className="col-sm-3 small text-muted">Avg Confidence</div>
              <div className="col-sm-9 small">{p.avg_confidence != null ? `${(p.avg_confidence * 100).toFixed(1)}%` : '—'}</div>
            </div>

            {/* Individual events */}
            {(p.events || []).length > 0 && (
              <div className="table-responsive">
                <table className="table table-xs table-sm table-hover mb-0" style={{ fontSize: '0.75rem' }}>
                  <thead className="table-light">
                    <tr>
                      <th>#</th>
                      <th>Semiology Type</th>
                      <th>Confidence</th>
                      <th>Lateralisation</th>
                      <th>Inferred Zone</th>
                      <th>Fall Risk Wt.</th>
                      <th>AI–MD Agree</th>
                    </tr>
                  </thead>
                  <tbody>
                    {p.events.map((ev, j) => (
                      <tr key={j}>
                        <td className="text-muted">{ev.event_id}</td>
                        <td className="fw-semibold">{ev.semiology_type}</td>
                        <td>{ev.confidence != null ? `${(ev.confidence * 100).toFixed(1)}%` : '—'}</td>
                        <td className="text-capitalize">{ev.lateralisation || '—'}</td>
                        <td>{ev.inferred_zone || '—'}</td>
                        <td>{ev.fall_risk_weight != null ? ev.fall_risk_weight.toFixed(2) : '—'}</td>
                        <td>{ev.ai_clinician_agree ? <span className="badge bg-success">✔ Yes</span> : <span className="badge bg-danger">✗ No</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const types = data.semiology_types || [];
  const method = data.classification_methodology || {};
  const fallRisk = data.fall_risk_scoring || {};
  const ilae = data.ilae_classification_mapping || [];
  const refs = data.references || [];

  return (
    <div>
      <h6 className="fw-semibold mb-3">Seizure Semiology Types — Clinical Definitions</h6>
      {types.map((t, i) => (
        <div key={i} className="card mb-3">
          <div className="card-header d-flex justify-content-between align-items-center">
            <span className="fw-semibold">{t.type}</span>
            <div className="d-flex gap-2">
              {t.lateralising && <span className="badge bg-primary">Lateralising</span>}
              <span className={`badge bg-${t.fall_risk_weight >= 0.6 ? 'danger' : t.fall_risk_weight >= 0.4 ? 'warning' : 'success'}`}>
                Fall risk: {t.fall_risk_weight}
              </span>
            </div>
          </div>
          <div className="card-body">
            <p className="small mb-2">{t.description}</p>
            <div className="text-muted small">
              <strong>Localisation zone:</strong> {t.localisation_zone}
            </div>
          </div>
        </div>
      ))}

      {/* Classification methodology */}
      {Object.keys(method).length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Classification Methodology</div>
          <div className="card-body">
            {Object.entries(method).map(([k, v]) => (
              <div key={k} className="mb-2">
                <div className="small fw-semibold text-capitalize mb-1">{k.replace(/_/g, ' ')}</div>
                <div className="small text-muted">{typeof v === 'string' ? v : JSON.stringify(v)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Fall risk scoring */}
      {Object.keys(fallRisk).length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Fall Risk Scoring</div>
          <div className="card-body small">
            {Object.entries(fallRisk).map(([k, v]) => (
              <div key={k} className="mb-1">
                <span className="fw-semibold text-capitalize">{k.replace(/_/g, ' ')}: </span>
                <span className="text-muted">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ILAE mapping */}
      {ilae.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">ILAE Classification Mapping</div>
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr>
                  {Object.keys(ilae[0] || {}).map(k => <th key={k} className="text-capitalize">{k.replace(/_/g, ' ')}</th>)}
                </tr>
              </thead>
              <tbody>
                {ilae.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((v, j) => (
                      <td key={j} className="small">{typeof v === 'boolean' ? (v ? '✔' : '✗') : String(v)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* References */}
      {refs.length > 0 && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">References</div>
          <ul className="list-group list-group-flush">
            {refs.map((r, i) => <li key={i} className="list-group-item small">{r}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function SeizureSemiologyPage() {
  const [tab, setTab] = useState('overview');
  const [panels, setPanels] = useState({});
  const [loading, setLoading] = useState({});

  const load = async (id) => {
    if (panels[id] || loading[id]) return;
    setLoading(l => ({ ...l, [id]: true }));
    try {
      const r = await fetch(`${API}/api/seizure-semiology/${id}`);
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
        <span style={{ fontSize: '1.6rem' }}>🧩</span>
        <div>
          <h4 className="mb-0 fw-bold">Seizure Semiology Dashboard</h4>
          <div className="text-muted small">AI-classified motor signs · Localisation zones · Fall risk · AI–Clinician agreement — 41 patients, 133 events, 9 ILAE types</div>
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
