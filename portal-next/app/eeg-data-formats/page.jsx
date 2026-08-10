'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',   label: 'Overview' },
  { id: 'formats',    label: 'Format Detail' },
  { id: 'routes',     label: 'Routing' },
  { id: 'request',    label: 'Data Request Guide' },
  { id: 'definitions', label: 'Definitions' },
];

const STAR_COLOR = { 5: 'success', 4: 'primary', 3: 'info', 2: 'warning', 1: 'danger' };
const ROUTE_COLOR = {
  signal: 'success', rag: 'primary', cv: 'info', extract: 'warning', video: 'secondary',
};
const READY_COLOR = { true: 'success', partial: 'warning', false: 'danger' };
const READY_LABEL = { true: 'AI Ready', partial: 'Partial', false: 'Not AI Ready' };

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

function Bar({ label, value, max, color }) {
  const p = Math.min(100, Math.round((value / Math.max(max, 1)) * 100));
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <span className="small">{label}</span>
        <span className="small fw-bold">{value}</span>
      </div>
      <div className="progress" style={{ height: '8px' }}>
        <div className={`progress-bar bg-${color || 'primary'}`} style={{ width: `${p}%` }} />
      </div>
    </div>
  );
}

function Badge({ label, color }) {
  return (
    <span className={`badge bg-${color || 'secondary'} me-1 mb-1`} style={{ fontSize: '0.72rem' }}>
      {label}
    </span>
  );
}

function Stars({ n }) {
  return (
    <span className={`fw-bold text-${STAR_COLOR[n] || 'secondary'}`}>
      {'★'.repeat(n)}{'☆'.repeat(5 - n)}
    </span>
  );
}

export default function EEGDataFormatsDashboard() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/eeg-data-formats/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/eeg-data-formats/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/eeg-data-formats/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return (
    <div className="p-5 text-center">
      <div className="spinner-border text-primary" />
      <div className="mt-2 text-muted small">Loading EEG Data Formats…</div>
    </div>
  );

  const { summary } = ov;
  const formats = ov.formats_table || [];
  const maxRouteVal = Math.max(1, ...(ov.route_distribution || []).map(r => r.value));
  const maxStarVal  = Math.max(1, ...(ov.star_distribution || []).map(s => s.value));

  return (
    <div>
      {/* Header */}
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: '1.8rem' }}>📂</span>
        <div>
          <h4 className="mb-0 fw-bold">EEG Data Formats Dashboard</h4>
          <p className="text-muted mb-0 small">
            AI-readiness, routing &amp; data-request guidance — {summary?.total_formats} formats cataloged
          </p>
        </div>
      </div>

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

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          <div className="row g-3 mb-4">
            <KPI label="Total Formats" value={summary?.total_formats} color="primary" />
            <KPI label="AI Ready" value={summary?.ai_ready_count}
              color="success" sub={`${Math.round((summary?.ai_ready_count / summary?.total_formats) * 100)}% of formats`} />
            <KPI label="Partially Ready" value={summary?.partially_ready} color="warning" />
            <KPI label="Avg Star Rating" value={summary?.avg_stars?.toFixed(1)}
              color="info" sub="out of 5 stars" />
          </div>

          <div className="row g-3">
            {/* AI Readiness */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">AI Readiness Distribution</div>
                <div className="card-body">
                  {(ov.ai_readiness_distribution || []).map(r => {
                    const key = r.name === 'AI Ready' ? 'true' : r.name === 'Partial' ? 'partial' : 'false';
                    return (
                      <div key={r.name} className="d-flex justify-content-between align-items-center mb-3">
                        <Badge label={r.name} color={READY_COLOR[key]} />
                        <div className="d-flex align-items-center gap-2">
                          <div className="progress flex-grow-1" style={{ width: '90px', height: '8px' }}>
                            <div
                              className={`progress-bar bg-${READY_COLOR[key]}`}
                              style={{ width: `${Math.round((r.value / summary.total_formats) * 100)}%` }}
                            />
                          </div>
                          <span className="small fw-bold">{r.value}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Route Distribution */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Route Distribution</div>
                <div className="card-body">
                  {(ov.route_distribution || []).map(r => (
                    <div key={r.name} className="d-flex justify-content-between align-items-center mb-2">
                      <Badge label={r.name} color={ROUTE_COLOR[r.name] || 'secondary'} />
                      <div className="d-flex align-items-center gap-2">
                        <div className="progress flex-grow-1" style={{ width: '90px', height: '8px' }}>
                          <div
                            className={`progress-bar bg-${ROUTE_COLOR[r.name] || 'secondary'}`}
                            style={{ width: `${Math.round((r.value / maxRouteVal) * 100)}%` }}
                          />
                        </div>
                        <span className="small fw-bold">{r.value}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Star Rating Distribution */}
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Star Rating Distribution</div>
                <div className="card-body">
                  {(ov.star_distribution || []).slice().reverse().map(s => {
                    const n = parseInt(s.name);
                    return (
                      <div key={s.name} className="d-flex justify-content-between align-items-center mb-2">
                        <Stars n={n} />
                        <div className="d-flex align-items-center gap-2">
                          <div className="progress flex-grow-1" style={{ width: '80px', height: '8px' }}>
                            <div
                              className={`progress-bar bg-${STAR_COLOR[n] || 'secondary'}`}
                              style={{ width: `${Math.round((s.value / maxStarVal) * 100)}%` }}
                            />
                          </div>
                          <span className="small fw-bold">{s.value}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* Formats quick reference table */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-semibold">Quick Reference — All Formats</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover small mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Extension</th>
                      <th>Name</th>
                      <th>AI Ready</th>
                      <th>Stars</th>
                      <th>Route</th>
                      <th>Supported</th>
                    </tr>
                  </thead>
                  <tbody>
                    {formats.map(f => (
                      <tr key={f.ext}>
                        <td className="fw-semibold text-nowrap font-monospace">{f.ext}</td>
                        <td>{f.name}</td>
                        <td>
                          <Badge
                            label={READY_LABEL[String(f.ai_ready)] || String(f.ai_ready)}
                            color={READY_COLOR[String(f.ai_ready)] || 'secondary'}
                          />
                        </td>
                        <td><Stars n={f.stars} /></td>
                        <td><Badge label={f.route} color={ROUTE_COLOR[f.route] || 'secondary'} /></td>
                        <td>
                          <Badge
                            label={f.supported ? 'Supported' : 'Unsupported'}
                            color={f.supported ? 'success' : 'danger'}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── FORMAT DETAIL ── */}
      {tab === 'formats' && bd && (
        <div>
          {(bd.per_format || []).map(f => (
            <div key={f.ext} className="card shadow-sm mb-3">
              <div className="card-header d-flex justify-content-between align-items-center">
                <div className="d-flex align-items-center gap-2">
                  <span className="fw-bold font-monospace">{f.ext}</span>
                  <span className="text-muted">— {f.name}</span>
                </div>
                <div className="d-flex gap-1">
                  <Badge
                    label={READY_LABEL[String(f.ai_ready)] || String(f.ai_ready)}
                    color={READY_COLOR[String(f.ai_ready)] || 'secondary'}
                  />
                  <Badge label={f.route} color={ROUTE_COLOR[f.route] || 'secondary'} />
                  <Stars n={f.stars} />
                </div>
              </div>
              <div className="card-body">
                <div className="row g-2">
                  <div className="col-md-4">
                    <div className="small fw-semibold text-muted mb-1">Contains</div>
                    <div className="small">{f.contains || '—'}</div>
                  </div>
                  {f.good_for && (
                    <div className="col-md-4">
                      <div className="small fw-semibold text-success mb-1">Good for</div>
                      <div className="small">{f.good_for}</div>
                    </div>
                  )}
                  {f.bad_for && (
                    <div className="col-md-4">
                      <div className="small fw-semibold text-danger mb-1">Bad for</div>
                      <div className="small">{f.bad_for}</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── ROUTING ── */}
      {tab === 'routes' && bd && (
        <div>
          {(bd.per_route || []).map(r => (
            <div key={r.route} className="card shadow-sm mb-3">
              <div className="card-header d-flex align-items-center gap-2">
                <Badge label={r.route} color={ROUTE_COLOR[r.route] || 'secondary'} />
                <span className="small text-muted">{r.description}</span>
              </div>
              <div className="card-body">
                <div className="d-flex flex-wrap gap-2">
                  {(r.formats || []).map(f => (
                    <div key={f.ext} className="card border-0 bg-light" style={{ minWidth: '140px' }}>
                      <div className="card-body py-2 px-3">
                        <div className="fw-bold font-monospace small">{f.ext}</div>
                        <div className="text-muted" style={{ fontSize: '0.7rem' }}>{f.name}</div>
                        <Stars n={f.stars} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── DATA REQUEST GUIDE ── */}
      {tab === 'request' && bd?.data_request && (
        <div>
          <div className="alert alert-warning small mb-3">
            <strong>Rule:</strong> {bd.data_request.rule}
          </div>

          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100 border-danger">
                <div className="card-header fw-semibold text-danger">Must-Have Data</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Data</th><th>Format</th><th>Purpose</th></tr>
                    </thead>
                    <tbody>
                      {(bd.data_request.must_have || []).map(d => (
                        <tr key={d.data}>
                          <td className="fw-semibold">{d.data}</td>
                          <td><span className="font-monospace small">{d.format}</span></td>
                          <td className="small text-muted">{d.purpose}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="col-md-6">
              <div className="card shadow-sm h-100 border-warning">
                <div className="card-header fw-semibold text-warning">Good-to-Have Data</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Data</th><th>Format</th><th>Purpose</th></tr>
                    </thead>
                    <tbody>
                      {(bd.data_request.good_to_have || []).map(d => (
                        <tr key={d.data}>
                          <td className="fw-semibold">{d.data}</td>
                          <td><span className="font-monospace small">{d.format}</span></td>
                          <td className="small text-muted">{d.purpose}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm border-info">
            <div className="card-header fw-semibold text-info">Minimum Dataset for DBA Research</div>
            <div className="card-body">
              <p className="small mb-0">{bd.data_request.minimum_dbA}</p>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div>
          {/* Route Descriptions */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Route Descriptions</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Route</th><th>Description</th></tr></thead>
                <tbody>
                  {(defs.route_descriptions || []).map(r => (
                    <tr key={r.route}>
                      <td><Badge label={r.route} color={ROUTE_COLOR[r.route] || 'secondary'} /></td>
                      <td className="small">{r.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* AI Readiness Legend */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">AI Readiness Legend</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Value</th><th>Color</th><th>Description</th></tr></thead>
                <tbody>
                  {(defs.ai_readiness_legend || []).map(r => (
                    <tr key={r.value}>
                      <td><Badge label={READY_LABEL[r.value] || r.value} color={READY_COLOR[r.value] || 'secondary'} /></td>
                      <td><span className="font-monospace small">{r.color}</span></td>
                      <td className="small">{r.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Star Rating Legend */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Star Rating Legend</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Stars</th><th>Description</th></tr></thead>
                <tbody>
                  {(defs.star_rating_legend || []).map(r => (
                    <tr key={r.stars}>
                      <td><Stars n={r.stars} /></td>
                      <td className="small">{r.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Glossary */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Glossary</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Term</th><th>Definition</th></tr></thead>
                <tbody>
                  {(defs.glossary || []).map(g => (
                    <tr key={g.term}>
                      <td className="fw-semibold text-nowrap">{g.term}</td>
                      <td className="small">{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Clinical Notes */}
          {(defs.clinical_notes || []).length > 0 && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">Clinical Notes</div>
              <div className="card-body">
                <ul className="mb-0 small">
                  {defs.clinical_notes.map((n, i) => <li key={i}>{n}</li>)}
                </ul>
              </div>
            </div>
          )}

          {/* References */}
          {(defs.references || []).length > 0 && (
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">References</div>
              <div className="card-body">
                <ol className="mb-0 small">
                  {defs.references.map((r, i) => <li key={i}>{r}</li>)}
                </ol>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
