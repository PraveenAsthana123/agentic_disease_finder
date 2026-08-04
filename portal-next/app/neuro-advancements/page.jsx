'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusBadge = s =>
  s === 'built'   ? 'success' :
  s === 'partial' ? 'warning' : 'secondary';

export default function NeuroAdvancementsPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/neuro-advancements/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/neuro-advancements/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/neuro-advancements/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = ov.summary || {};
  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'modalities',  label: 'All Modalities' },
    { id: 'models',      label: 'AI Model Index' },
    { id: 'crossmodal',  label: 'Cross-Modal' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f9e0; Neuro AI Advancement Opportunities</h3>
      <p className="text-muted small">
        Per-modality AI advancement opportunities across 12 neurophysiology tests.
        AI model coverage, biomarker extraction, and cross-modal fusion research directions.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Modalities',        value: s.total_modalities,  color: 'primary' },
          { label: 'Built',             value: s.built,             color: 'success' },
          { label: 'Partial',           value: s.partial,           color: 'warning' },
          { label: 'Planned',           value: s.planned,           color: 'secondary' },
          { label: 'Unique AI Models',  value: s.unique_ai_models,  color: 'info' },
          { label: 'Biomarkers',        value: s.total_biomarkers,  color: 'dark' },
          { label: 'Cross-Modal Ideas', value: s.cross_modal_count, color: 'purple' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 col-lg mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value ?? '—'}</div>
                <div className="text-muted" style={{fontSize: '0.72rem'}}>{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ─────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Status Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Status Distribution</div>
              <div className="card-body">
                {(ov.status_distribution || []).map(d => (
                  <div key={d.name} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${statusBadge(d.name.toLowerCase())}`}>{d.name}</span>
                    <span className="fw-bold">{d.value}</span>
                    <div className="progress flex-grow-1 mx-2" style={{height: '8px'}}>
                      <div className={`progress-bar bg-${statusBadge(d.name.toLowerCase())}`}
                           style={{width: `${(d.value / s.total_modalities * 100)}%`}} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* AI Models per Modality */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">AI Models per Modality</div>
              <div className="card-body">
                {(ov.models_per_modality || []).map(m => (
                  <div key={m.name} className="d-flex justify-content-between align-items-center mb-1">
                    <code className="small">{m.name}</code>
                    <div className="progress flex-grow-1 mx-2" style={{height: '10px'}}>
                      <div className="progress-bar bg-info" style={{width: `${m.value / 4 * 100}%`}} />
                    </div>
                    <span className="badge bg-info">{m.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Top AI Models */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">AI Model Frequency</div>
              <div className="card-body">
                {(ov.model_distribution || []).map(m => (
                  <div key={m.name} className="d-flex justify-content-between align-items-center mb-1">
                    <span className="small">{m.name}</span>
                    <span className="badge bg-primary">{m.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Modality Summary Table */}
          <div className="col-12 mt-2">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Modality Summary</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark">
                      <tr><th>Code</th><th>Name</th><th>Advancement</th><th>AI Models</th><th>Biomarker</th><th>Status</th></tr>
                    </thead>
                    <tbody>
                      {(ov.modalities_table || []).map(m => (
                        <tr key={m.code}>
                          <td><code>{m.code}</code></td>
                          <td>{m.name}</td>
                          <td className="small">{m.advancement}</td>
                          <td className="small">{m.ai_models}</td>
                          <td className="small">{m.biomarker}</td>
                          <td><span className={`badge bg-${statusBadge(m.status)}`}>{m.status}</span></td>
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

      {/* ── All Modalities Tab ───────────────────────────── */}
      {tab === 'modalities' && bd && (
        <div className="row">
          {(bd.per_modality || []).map(m => (
            <div key={m.code} className="col-md-6 col-lg-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header d-flex justify-content-between align-items-center">
                  <span className="fw-bold">{m.name}</span>
                  <span className={`badge bg-${statusBadge(m.status)}`}>{m.status}</span>
                </div>
                <div className="card-body">
                  <p className="small text-muted mb-2">{m.advancement}</p>
                  <div className="mb-2">
                    <strong className="small">AI Models:</strong>
                    <div>{(m.ai_models || []).map(a => (
                      <span key={a} className="badge bg-info me-1 mb-1">{a}</span>
                    ))}</div>
                  </div>
                  <div className="mb-1">
                    <strong className="small">Biomarker:</strong>{' '}
                    <span className="small">{m.biomarker || '—'}</span>
                  </div>
                  {m.note && <p className="small text-success mt-2 mb-0">{m.note}</p>}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── AI Model Index Tab ───────────────────────────── */}
      {tab === 'models' && bd && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold">AI Model to Modality Mapping</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0">
                <thead className="table-dark">
                  <tr><th>AI Model</th><th>Used By</th><th>Count</th></tr>
                </thead>
                <tbody>
                  {(bd.ai_model_index || []).map(m => (
                    <tr key={m.model}>
                      <td className="fw-bold">{m.model}</td>
                      <td>{(m.used_by || []).map(c => <span key={c} className="badge bg-secondary me-1">{c}</span>)}</td>
                      <td><span className="badge bg-primary">{m.count}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          {/* Biomarker Index */}
          <div className="card-header fw-bold border-top">Biomarker Index</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0">
                <thead className="table-dark">
                  <tr><th>Modality</th><th>Name</th><th>Biomarker</th></tr>
                </thead>
                <tbody>
                  {(bd.biomarker_index || []).map(b => (
                    <tr key={b.modality}>
                      <td><code>{b.modality}</code></td>
                      <td>{b.name}</td>
                      <td>{b.biomarker}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Cross-Modal Tab ──────────────────────────────── */}
      {tab === 'crossmodal' && bd && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold">Cross-Modal Advancement Directions</div>
          <div className="card-body">
            <p className="text-muted small mb-3">
              Research frontiers combining multiple neurophysiology modalities for richer clinical insight —
              aligned with the DBA thesis on multimodal EEG-AI fusion.
            </p>
            <ol className="list-group list-group-numbered">
              {(bd.cross_modal_advancements || []).map((a, i) => (
                <li key={i} className="list-group-item">{a}</li>
              ))}
            </ol>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ──────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          {/* Status Legend */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Status Legend</div>
              <div className="card-body">
                {(defs.status_legend || []).map(s => (
                  <div key={s.status} className="d-flex align-items-start mb-2">
                    <span className="badge me-2" style={{background: s.color, minWidth: '60px'}}>{s.status}</span>
                    <span className="small">{s.description}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Modality Categories */}
          <div className="col-md-8 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Modality Categories</div>
              <div className="card-body">
                {(defs.modality_categories || []).map(c => (
                  <div key={c.category} className="mb-3">
                    <h6>{c.category}</h6>
                    <p className="small text-muted mb-1">{c.description}</p>
                    <div>{(c.modalities || []).map(m => <span key={m} className="badge bg-secondary me-1">{m}</span>)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Glossary */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Glossary</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-dark"><tr><th>Term</th><th>Definition</th></tr></thead>
                    <tbody>
                      {(defs.glossary || []).map(g => (
                        <tr key={g.term}><td className="fw-bold">{g.term}</td><td className="small">{g.definition}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Clinical Notes */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Clinical Notes</div>
              <div className="card-body">
                <ul className="mb-3">
                  {(defs.clinical_notes || []).map((n, i) => <li key={i} className="small mb-1">{n}</li>)}
                </ul>
                <h6>References</h6>
                <ol className="small">
                  {(defs.references || []).map((r, i) => <li key={i} className="mb-1 text-muted">{r}</li>)}
                </ol>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
