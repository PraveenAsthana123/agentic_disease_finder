'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'breakdown',   label: 'Patient Breakdown' },
  { id: 'internals',   label: 'Architecture Internals' },
  { id: 'features',    label: 'Feature Selection' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-2 mb-2">
      <div className="card text-center shadow-sm border-0 h-100">
        <div className="card-body py-2 px-1">
          <div className={`h3 mb-0 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted" style={{ fontSize: '0.72rem' }}>{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ val, max, color }) {
  const pct = Math.min(((val || 0) / (max || 1)) * 100, 100);
  const c = color || (pct >= 90 ? 'success' : pct >= 70 ? 'primary' : pct >= 50 ? 'info' : 'warning');
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 12, borderRadius: 6 }}>
        <div className={`progress-bar bg-${c}`} style={{ width: `${pct}%`, borderRadius: 6, transition: 'width 0.6s' }} />
      </div>
      <small className="text-muted" style={{ width: 38, textAlign: 'right' }}>{val != null ? val.toFixed ? (val * 100).toFixed(1) + '%' : val : '—'}</small>
    </div>
  );
}

/* ── Overview Tab ─────────────────────────────────────────── */
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;
  const k = ov.kpis || {};
  const arch = ov.architecture_comparison || [];
  const conf = ov.confidence_distribution || [];
  const tp   = ov.temporal_performance || [];

  return (
    <>
      <div className="row g-2 mb-3">
        <KPI label="EEG Analyses"       value={k.total_analyses}           color="primary" />
        <KPI label="Mean Confidence"    value={k.mean_confidence != null ? (k.mean_confidence * 100).toFixed(1) + '%' : '—'} color="info" />
        <KPI label="Architectures"      value={k.architectures_compared}   color="secondary" />
        <KPI label="Best Architecture"  value={k.best_architecture}        color="success" />
        <KPI label="Temporal Score"     value={k.temporal_median_score != null ? k.temporal_median_score.toFixed(3) : '—'} color="warning" sub="median" />
        <KPI label="Spectral Score"     value={k.spectral_median_score != null ? k.spectral_median_score.toFixed(3) : '—'} color="danger"  sub="median" />
      </div>

      {k.best_architecture_reason && (
        <div className="alert alert-success py-2 mb-3">
          <strong>Selection rationale:</strong> {k.best_architecture_reason}
        </div>
      )}

      <div className="row g-3 mb-3">
        {/* Architecture comparison */}
        <div className="col-12 col-lg-7">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-header bg-primary text-white py-2">Architecture Comparison</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover align-middle mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Architecture</th>
                    <th>Samples</th>
                    <th>Accuracy</th>
                    <th>F1</th>
                    <th>Confidence</th>
                    <th>Latency (ms)</th>
                    <th>Params (M)</th>
                  </tr>
                </thead>
                <tbody>
                  {arch.map((a, i) => (
                    <tr key={i} className={i === 0 ? 'table-success fw-semibold' : ''}>
                      <td>{a.architecture}</td>
                      <td>{a.n_samples}</td>
                      <td>{a.accuracy != null ? (a.accuracy * 100).toFixed(1) + '%' : '—'}</td>
                      <td>{a.f1_score != null ? a.f1_score.toFixed(3) : '—'}</td>
                      <td>{a.mean_confidence != null ? (a.mean_confidence * 100).toFixed(1) + '% ±' + (a.std_confidence * 100).toFixed(1) : '—'}</td>
                      <td>{a.latency_ms}</td>
                      <td>{a.params_M}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Confidence distribution */}
        <div className="col-12 col-lg-5">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-header bg-info text-white py-2">Confidence Distribution</div>
            <div className="card-body py-2">
              {conf.map((c, i) => (
                <div key={i} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <small className="fw-semibold">{c.range}</small>
                    <small className="text-muted">{c.count} ({c.pct}%)</small>
                  </div>
                  <div className="progress" style={{ height: 14, borderRadius: 6 }}>
                    <div
                      className={`progress-bar bg-${c.pct >= 25 ? 'primary' : 'secondary'}`}
                      style={{ width: `${c.pct}%`, borderRadius: 6, transition: 'width 0.6s' }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Disease temporal performance */}
      <div className="card shadow-sm border-0">
        <div className="card-header bg-secondary text-white py-2">Temporal Performance by Disease</div>
        <div className="table-responsive">
          <table className="table table-sm align-middle mb-0">
            <thead className="table-light">
              <tr>
                <th>Disease</th>
                <th>Analyses</th>
                <th>Mean Confidence</th>
                <th>Std</th>
                <th>Preferred</th>
              </tr>
            </thead>
            <tbody>
              {tp.map((r, i) => (
                <tr key={i}>
                  <td className="text-capitalize">{r.disease?.replace(/_/g, ' ')}</td>
                  <td>{r.n_analyses}</td>
                  <td>
                    <Bar val={r.mean_confidence} max={1} />
                  </td>
                  <td><small className="text-muted">±{(r.std_confidence * 100).toFixed(1)}%</small></td>
                  <td>
                    <span className={`badge bg-${r.cnn_lstm_preferred ? 'success' : 'primary'}`}>
                      {r.cnn_lstm_preferred ? 'CNN-LSTM' : 'CNN-Transformer'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

/* ── Patient Breakdown Tab ────────────────────────────────── */
function BreakdownTab({ bd }) {
  const [search, setSearch] = useState('');
  const [archFilter, setArchFilter] = useState('all');
  if (!bd) return <div className="text-muted p-3">Loading…</div>;
  const rows = bd.per_patient || [];
  const filtered = rows.filter(r => {
    const matchSearch = !search || r.patient_id?.toLowerCase().includes(search.toLowerCase()) || r.disease?.toLowerCase().includes(search.toLowerCase());
    const matchArch = archFilter === 'all' || r.dominant_architecture === archFilter;
    return matchSearch && matchArch;
  });

  return (
    <>
      <div className="row g-2 mb-3">
        <div className="col-md-4">
          <input
            className="form-control form-control-sm"
            placeholder="Search patient / disease…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
        <div className="col-md-3">
          <select className="form-select form-select-sm" value={archFilter} onChange={e => setArchFilter(e.target.value)}>
            <option value="all">All architectures</option>
            <option value="CNN-LSTM">CNN-LSTM only</option>
            <option value="CNN-Transformer">CNN-Transformer only</option>
          </select>
        </div>
        <div className="col-md-5 d-flex align-items-center">
          <small className="text-muted">Showing {filtered.length} / {rows.length} patients</small>
        </div>
      </div>

      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-dark">
            <tr>
              <th>Patient</th>
              <th>Disease</th>
              <th>Analyses</th>
              <th>Architecture</th>
              <th>LSTM</th>
              <th>Trans.</th>
              <th>Confidence</th>
              <th>Accuracy</th>
              <th>Temporal</th>
              <th>Spectral</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => (
              <tr key={i}>
                <td><code className="small">{r.patient_id}</code></td>
                <td className="text-capitalize">{r.disease?.replace(/_/g,' ')}</td>
                <td>{r.n_analyses}</td>
                <td>
                  <span className={`badge bg-${r.dominant_architecture === 'CNN-LSTM' ? 'success' : 'primary'}`}>
                    {r.dominant_architecture}
                  </span>
                </td>
                <td><small>{r.cnn_lstm_count}</small></td>
                <td><small>{r.cnn_transformer_count}</small></td>
                <td>
                  <div className="d-flex align-items-center gap-1">
                    <div className="progress flex-grow-1" style={{ height: 10, minWidth: 50, borderRadius: 4 }}>
                      <div className="progress-bar bg-info" style={{ width: `${(r.mean_confidence || 0) * 100}%`, borderRadius: 4 }} />
                    </div>
                    <small>{((r.mean_confidence || 0) * 100).toFixed(0)}%</small>
                  </div>
                </td>
                <td><small>{r.accuracy_proxy != null ? (r.accuracy_proxy * 100).toFixed(0) + '%' : '—'}</small></td>
                <td><small>{r.mean_temporal_score != null ? r.mean_temporal_score.toFixed(3) : '—'}</small></td>
                <td><small>{r.mean_spectral_score != null ? r.mean_spectral_score.toFixed(3) : '—'}</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

/* ── Architecture Internals Tab ──────────────────────────── */
function InternalsTab({ bd }) {
  const [view, setView] = useState('layers');
  if (!bd) return <div className="text-muted p-3">Loading…</div>;
  const layers  = bd.layer_analysis || [];
  const attn    = bd.attention_weights || [];
  const gates   = bd.lstm_gates || [];
  const curves  = bd.training_curves || [];

  return (
    <>
      <div className="btn-group mb-3" role="group">
        {[['layers','Layer Analysis'],['attention','Attention Weights'],['gates','LSTM Gates'],['curves','Training Curves']].map(([k,l]) => (
          <button key={k} className={`btn btn-sm btn-${view===k?'primary':'outline-primary'}`} onClick={()=>setView(k)}>{l}</button>
        ))}
      </div>

      {view === 'layers' && (
        <div className="table-responsive">
          <table className="table table-sm align-middle">
            <thead className="table-light">
              <tr><th>Layer</th><th>Description</th><th>Mean Activation</th><th>Std</th><th>Source Features</th></tr>
            </thead>
            <tbody>
              {layers.map((l, i) => (
                <tr key={i}>
                  <td><code>{l.layer}</code></td>
                  <td><small>{l.description}</small></td>
                  <td>
                    <div className="d-flex align-items-center gap-1">
                      <div className="progress flex-grow-1" style={{ height: 10, minWidth: 60, borderRadius: 4 }}>
                        <div className="progress-bar bg-primary" style={{ width: `${Math.min((l.mean_activation||0)*300,100)}%`, borderRadius: 4 }} />
                      </div>
                      <small>{(l.mean_activation||0).toFixed(3)}</small>
                    </div>
                  </td>
                  <td><small>{(l.std_activation||0).toFixed(3)}</small></td>
                  <td><small className="text-muted">{(l.source_features||[]).join(', ')}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {view === 'attention' && (
        <div className="table-responsive">
          <table className="table table-sm align-middle">
            <thead className="table-light">
              <tr><th>Attention Head</th><th>Band</th><th>Weight</th><th>Mean Band Power</th><th>Description</th></tr>
            </thead>
            <tbody>
              {attn.map((a, i) => (
                <tr key={i}>
                  <td><code>{a.head}</code></td>
                  <td><span className="badge bg-info text-dark">{a.band}</span></td>
                  <td>
                    <div className="d-flex align-items-center gap-1">
                      <div className="progress flex-grow-1" style={{ height: 10, minWidth: 60, borderRadius: 4 }}>
                        <div className="progress-bar bg-warning text-dark" style={{ width: `${(a.weight||0)*100}%`, borderRadius: 4 }} />
                      </div>
                      <small>{(a.weight||0).toFixed(3)}</small>
                    </div>
                  </td>
                  <td><small>{(a.mean_band_power_relative||0).toFixed(4)}</small></td>
                  <td><small className="text-muted">{a.description}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {view === 'gates' && (
        <div className="table-responsive">
          <table className="table table-sm align-middle">
            <thead className="table-light">
              <tr><th>Gate</th><th>Activation</th><th>Std</th><th>Source Feature</th><th>Description</th></tr>
            </thead>
            <tbody>
              {gates.map((g, i) => (
                <tr key={i}>
                  <td><span className="badge bg-secondary">{g.gate?.replace(/_/g,' ')}</span></td>
                  <td>
                    <div className="d-flex align-items-center gap-1">
                      <div className="progress flex-grow-1" style={{ height: 10, minWidth: 60, borderRadius: 4 }}>
                        <div className="progress-bar bg-success" style={{ width: `${(g.mean_activation||0)*100}%`, borderRadius: 4 }} />
                      </div>
                      <small>{(g.mean_activation||0).toFixed(3)}</small>
                    </div>
                  </td>
                  <td><small>{(g.std_activation||0).toFixed(3)}</small></td>
                  <td><small className="text-muted">{g.source_feature}</small></td>
                  <td><small className="text-muted">{g.description}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {view === 'curves' && (
        <div className="row g-3">
          {curves.map((c, i) => (
            <div key={i} className="col-12 col-md-6">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-dark text-white py-2 small">{c.architecture} — {c.phase}</div>
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Epoch</th><th>Loss</th><th>Accuracy</th></tr>
                    </thead>
                    <tbody>
                      {(c.epochs||[]).slice(0,8).map((ep, j) => (
                        <tr key={j}>
                          <td>{ep.epoch}</td>
                          <td>{ep.loss?.toFixed(4)}</td>
                          <td>{ep.accuracy != null ? (ep.accuracy*100).toFixed(1)+'%' : '—'}</td>
                        </tr>
                      ))}
                      {(c.epochs||[]).length > 8 && (
                        <tr><td colSpan={3} className="text-center text-muted small">…{c.epochs.length - 8} more epochs</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
          {curves.length === 0 && <div className="col-12 text-muted">No training curve data available.</div>}
        </div>
      )}
    </>
  );
}

/* ── Feature Selection Tab ────────────────────────────────── */
function FeaturesTab({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;
  const feats = ov.feature_importance || [];
  const maxVar = Math.max(...feats.map(f => f.variance || 0), 1);

  return (
    <>
      <h6 className="mb-3">Feature Importance (by Variance)</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-light">
            <tr><th>#</th><th>Feature</th><th>Variance</th><th>Mean</th><th>Std</th><th>Samples</th></tr>
          </thead>
          <tbody>
            {feats.map((f, i) => (
              <tr key={i}>
                <td><small className="text-muted">{i+1}</small></td>
                <td className="fw-semibold">{f.feature}</td>
                <td>
                  <div className="d-flex align-items-center gap-2">
                    <div className="progress flex-grow-1" style={{ height: 12, minWidth: 80, borderRadius: 4 }}>
                      <div className="progress-bar bg-danger" style={{ width: `${((f.variance||0)/maxVar)*100}%`, borderRadius: 4 }} />
                    </div>
                    <small>{(f.variance||0).toFixed(2)}</small>
                  </div>
                </td>
                <td><small>{(f.mean||0).toFixed(4)}</small></td>
                <td><small>{(f.std||0).toFixed(4)}</small></td>
                <td><small>{f.n_samples}</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="mb-3">Architecture Selection Logic</h6>
      <div className="row g-3">
        <div className="col-12 col-md-6">
          <div className="card border-success h-100">
            <div className="card-header bg-success text-white py-2">CNN-LSTM preferred when…</div>
            <ul className="list-group list-group-flush">
              <li className="list-group-item small">Hurst exponent &gt; 0.45</li>
              <li className="list-group-item small">Autocorrelation &gt; −0.09</li>
              <li className="list-group-item small">Long recordings (&gt; 60 s)</li>
              <li className="list-group-item small">Clear temporal ictal progression</li>
              <li className="list-group-item small">Lower latency requirement</li>
            </ul>
          </div>
        </div>
        <div className="col-12 col-md-6">
          <div className="card border-primary h-100">
            <div className="card-header bg-primary text-white py-2">CNN-Transformer preferred when…</div>
            <ul className="list-group list-group-flush">
              <li className="list-group-item small">High spectral entropy</li>
              <li className="list-group-item small">High Lempel-Ziv complexity</li>
              <li className="list-group-item small">Cross-channel attention needed</li>
              <li className="list-group-item small">Generalised / non-focal networks</li>
              <li className="list-group-item small">Parallel training compute available</li>
            </ul>
          </div>
        </div>
      </div>
    </>
  );
}

/* ── Definitions Tab ─────────────────────────────────────── */
function DefinitionsTab({ defs }) {
  const [section, setSection] = useState('architectures');
  if (!defs) return <div className="text-muted p-3">Loading…</div>;
  const archs  = defs.architectures || [];
  const feats  = defs.features || [];
  const wtp    = defs.when_to_prefer || {};
  const refs   = defs.clinical_references || [];

  return (
    <>
      <div className="btn-group mb-3" role="group">
        {[['architectures','Architectures'],['features','EEG Features'],['when','When to Use'],['refs','References']].map(([k,l]) => (
          <button key={k} className={`btn btn-sm btn-${section===k?'dark':'outline-dark'}`} onClick={()=>setSection(k)}>{l}</button>
        ))}
      </div>

      {section === 'architectures' && archs.map((a, i) => (
        <div key={i} className="card mb-3 shadow-sm border-0">
          <div className="card-header bg-dark text-white py-2">
            {a.name} — <small className="text-light">{a.full_name}</small>
          </div>
          <div className="card-body">
            <p className="small mb-2"><strong>Pipeline:</strong> <code style={{ fontSize: '0.7rem' }}>{a.pipeline}</code></p>
            {a.cnn_role   && <p className="small mb-1"><strong>CNN role:</strong> {a.cnn_role}</p>}
            {a.lstm_role  && <p className="small mb-1"><strong>LSTM role:</strong> {a.lstm_role}</p>}
            {a.transformer_role && <p className="small mb-1"><strong>Transformer role:</strong> {a.transformer_role}</p>}
            {a.preferred_when && (
              <>
                <strong className="small">Preferred when:</strong>
                <ul className="mb-2">
                  {a.preferred_when.map((p, j) => <li key={j} className="small">{p}</li>)}
                </ul>
              </>
            )}
            {a.hyperparameters && (
              <div className="mt-2">
                <strong className="small">Key hyperparameters:</strong>
                <div className="row g-1 mt-1">
                  {Object.entries(a.hyperparameters).map(([k, v]) => (
                    <div key={k} className="col-auto">
                      <span className="badge bg-light text-dark border">{k}: {Array.isArray(v) ? v.join('/') : v}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      ))}

      {section === 'features' && (
        <div className="table-responsive">
          <table className="table table-sm align-middle">
            <thead className="table-dark">
              <tr><th>Feature</th><th>Formula</th><th>Interpretation</th><th>Clinical Use</th></tr>
            </thead>
            <tbody>
              {feats.map((f, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{f.feature}</td>
                  <td><code style={{ fontSize: '0.68rem' }}>{f.formula}</code></td>
                  <td><small>{f.interpretation}</small></td>
                  <td><small className="text-muted">{f.clinical_use}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {section === 'when' && (
        <div className="row g-3">
          {Object.entries(wtp).map(([arch, text]) => (
            <div key={arch} className="col-12">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-secondary text-white py-2">{arch}</div>
                <div className="card-body small">{text}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {section === 'refs' && (
        <div className="table-responsive">
          <table className="table table-sm align-middle">
            <thead className="table-light">
              <tr><th>Citation</th><th>Title</th><th>Journal</th><th>Relevance</th></tr>
            </thead>
            <tbody>
              {refs.map((r, i) => (
                <tr key={i}>
                  <td className="small fw-semibold">{r.citation}</td>
                  <td className="small fst-italic">{r.title}</td>
                  <td><small className="text-muted">{r.journal}</small></td>
                  <td><small>{r.relevance}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

/* ── Page Shell ──────────────────────────────────────────── */
export default function HybridPipelinePage() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv]   = useState(null);
  const [bd, setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hybrid-pipeline/overview`).then(r => r.json()),
      fetch(`${API}/api/hybrid-pipeline/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hybrid-pipeline/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-1">
        <h4 className="mb-0">&#x1f9ec; Hybrid CNN Pipeline</h4>
        <span className="badge bg-success ms-2">Live</span>
      </div>
      <p className="text-muted small mb-3">
        CNN-LSTM vs CNN-Transformer — real-time architecture selection based on EEG temporal &amp; spectral features.
        133 patient analyses · 50 patients · 5 diseases.
      </p>

      {err && <div className="alert alert-danger py-2">API error: {err}</div>}

      <ul className="nav nav-tabs mb-3">
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

      <div>
        {tab === 'overview'    && <OverviewTab ov={ov} />}
        {tab === 'breakdown'   && <BreakdownTab bd={bd} />}
        {tab === 'internals'   && <InternalsTab bd={bd} />}
        {tab === 'features'    && <FeaturesTab ov={ov} />}
        {tab === 'definitions' && <DefinitionsTab defs={defs} />}
      </div>
    </div>
  );
}
