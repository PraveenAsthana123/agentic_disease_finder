'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8000';

const COMPLEXITY_COLOR = { high: 'danger', medium: 'warning', linear: 'success' };
const TYPE_COLOR = { Automated: 'primary', 'Human-in-loop': 'warning', Hybrid: 'info' };

export default function ClinicalFlowchartsPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [selected, setSelected] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/clinical-flowcharts/overview`).then(r => r.json()),
      fetch(`${API}/api/clinical-flowcharts/breakdown`).then(r => r.json()),
      fetch(`${API}/api/clinical-flowcharts/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOverview(o); setBreakdown(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!overview) return <div className="text-muted p-3">Loading clinical flowcharts…</div>;

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'flowcharts', label: '🔀 Flowcharts' },
    { id: 'detail', label: '🔍 Detail' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const a = overview.analytics || {};
  const bsum = breakdown?.summary || {};

  return (
    <div className="container-fluid py-3">
      <h3>🔀 Clinical Process Flowcharts</h3>
      <p className="text-muted small">
        {overview.total} governed workflows — EEG reads, agent council, expert review, IoT alerts, onboarding, assessments.
      </p>

      {/* KPI tiles */}
      <div className="row g-2 mb-3">
        {[
          ['Flowcharts', overview.total, 'primary'],
          ['Total Nodes', a.total_nodes, 'info'],
          ['Decision Points', a.total_decisions, 'warning'],
          ['Total Edges', a.total_edges, 'secondary'],
          ['High Complexity', a.complexity_distribution?.high, 'danger'],
          ['Avg Nodes / Flow', a.avg_nodes_per_flow, 'success'],
        ].map(([label, val, color]) => (
          <div key={label} className="col-6 col-md-2">
            <div className={`card border-${color} text-center p-2`}>
              <div className={`fs-4 fw-bold text-${color}`}>{val ?? '—'}</div>
              <div className="small text-muted">{label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Process-type row */}
      <div className="row g-2 mb-3">
        {[
          ['Automated', bsum.automated, 'primary'],
          ['Human-in-loop', bsum.human_in_loop, 'warning'],
          ['Hybrid', bsum.hybrid, 'info'],
        ].map(([label, val, color]) => (
          <div key={label} className="col-4">
            <div className={`card border-${color} text-center p-2`}>
              <div className={`fs-5 fw-bold text-${color}`}>{val ?? '—'}</div>
              <div className="small text-muted">{label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div>
          <h5>Flowchart Summary</h5>
          <div className="row g-3">
            {(overview.flowcharts || []).map(fc => (
              <div key={fc.id} className="col-md-4">
                <div className="card h-100">
                  <div className="card-body">
                    <h6 className="card-title">{fc.title}</h6>
                    <div className="mb-1">
                      <span className={`badge bg-${COMPLEXITY_COLOR[fc.complexity] || 'secondary'} me-1`}>{fc.complexity}</span>
                      <span className="badge bg-light text-dark border">{fc.category}</span>
                    </div>
                    <div className="small text-muted">
                      {fc.node_count} nodes · {fc.decision_count} decisions
                    </div>
                    <button
                      className="btn btn-sm btn-outline-primary mt-2"
                      onClick={() => { setSelected(fc.id); setTab('detail'); }}
                    >View Detail →</button>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <h5 className="mt-4">Complexity Distribution</h5>
          <div className="row g-2">
            {Object.entries(a.complexity_distribution || {}).map(([level, count]) => (
              <div key={level} className="col-4 col-md-2">
                <div className={`card border-${COMPLEXITY_COLOR[level] || 'secondary'} text-center p-2`}>
                  <div className={`fs-5 fw-bold text-${COMPLEXITY_COLOR[level] || 'secondary'}`}>{count}</div>
                  <div className="small text-muted text-capitalize">{level}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Flowcharts tab — sortable table */}
      {tab === 'flowcharts' && breakdown && (
        <div>
          <h5>All Flowcharts</h5>
          <div className="table-responsive">
            <table className="table table-sm table-hover align-middle">
              <thead className="table-light">
                <tr>
                  <th>Title</th>
                  <th>Category</th>
                  <th>Process Type</th>
                  <th>Complexity</th>
                  <th>Nodes</th>
                  <th>Decisions</th>
                  <th>Edges</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.flowcharts || []).map(fc => (
                  <tr key={fc.id}>
                    <td className="fw-semibold">{fc.title}</td>
                    <td><span className="badge bg-light text-dark border">{fc.category}</span></td>
                    <td><span className={`badge bg-${TYPE_COLOR[fc.process_type] || 'secondary'}`}>{fc.process_type}</span></td>
                    <td><span className={`badge bg-${COMPLEXITY_COLOR[fc.complexity] || 'secondary'}`}>{fc.complexity}</span></td>
                    <td>{fc.node_count}</td>
                    <td>{fc.decision_count}</td>
                    <td>{fc.edge_count}</td>
                    <td>
                      <button
                        className="btn btn-sm btn-outline-secondary"
                        onClick={() => { setSelected(fc.id); setTab('detail'); }}
                      >Detail</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h5 className="mt-4">Node Breakdown</h5>
          <div className="row g-2">
            {(breakdown.flowcharts || []).map(fc => (
              <div key={fc.id} className="col-md-6">
                <div className="card p-2">
                  <div className="fw-semibold mb-1">{fc.title}</div>
                  <div className="d-flex gap-2 flex-wrap">
                    <span className="badge bg-primary">{fc.process_steps} steps</span>
                    <span className="badge bg-warning text-dark">{fc.decision_count} decisions</span>
                    <span className="badge bg-secondary">{fc.edge_count} edges</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detail tab — mermaid + node list */}
      {tab === 'detail' && breakdown && (
        <div>
          {/* Flowchart selector */}
          <div className="mb-3">
            <label className="form-label fw-semibold">Select Flowchart</label>
            <select
              className="form-select"
              value={selected || ''}
              onChange={e => setSelected(e.target.value)}
            >
              <option value="">— choose —</option>
              {(breakdown.flowcharts || []).map(fc => (
                <option key={fc.id} value={fc.id}>{fc.title}</option>
              ))}
            </select>
          </div>

          {selected && (() => {
            const fc = (breakdown.flowcharts || []).find(f => f.id === selected);
            if (!fc) return null;
            return (
              <div>
                <div className="d-flex gap-2 mb-2">
                  <span className={`badge bg-${COMPLEXITY_COLOR[fc.complexity] || 'secondary'}`}>{fc.complexity}</span>
                  <span className={`badge bg-${TYPE_COLOR[fc.process_type] || 'secondary'}`}>{fc.process_type}</span>
                  <span className="badge bg-light text-dark border">{fc.category}</span>
                </div>

                <h5>{fc.title}</h5>

                {/* Mermaid source */}
                <div className="mb-3">
                  <div className="fw-semibold small mb-1">Mermaid Source</div>
                  <pre className="bg-light border rounded p-2 small" style={{ maxHeight: 220, overflow: 'auto' }}>
                    {fc.mermaid}
                  </pre>
                </div>

                {/* Node list */}
                <h6>Parsed Nodes ({fc.node_count})</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-bordered">
                    <thead className="table-light">
                      <tr><th>ID</th><th>Label</th><th>Type</th></tr>
                    </thead>
                    <tbody>
                      {(fc.nodes || []).map((n, i) => (
                        <tr key={i}>
                          <td><code>{n.id}</code></td>
                          <td>{n.label}</td>
                          <td>
                            <span className={`badge ${n.type === 'decision' ? 'bg-warning text-dark' : 'bg-primary'}`}>
                              {n.type}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Stats */}
                <div className="row g-2 mt-2">
                  {[
                    ['Total Nodes', fc.node_count, 'primary'],
                    ['Process Steps', fc.process_steps, 'info'],
                    ['Decision Points', fc.decision_count, 'warning'],
                    ['Edges', fc.edge_count, 'secondary'],
                  ].map(([label, val, color]) => (
                    <div key={label} className="col-3">
                      <div className={`card border-${color} text-center p-2`}>
                        <div className={`fw-bold text-${color}`}>{val}</div>
                        <div className="small text-muted">{label}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          <h5>Element Types</h5>
          <table className="table table-sm table-bordered mb-4">
            <thead className="table-light">
              <tr><th>Type</th><th>Symbol</th><th>Description</th></tr>
            </thead>
            <tbody>
              {(defs.element_types || []).map(e => (
                <tr key={e.term}>
                  <td className="fw-semibold">{e.term}</td>
                  <td><code>{e.symbol}</code></td>
                  <td>{e.description}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h5>Complexity Levels</h5>
          <table className="table table-sm table-bordered mb-4">
            <thead className="table-light">
              <tr><th>Level</th><th>Definition</th><th>Example</th></tr>
            </thead>
            <tbody>
              {(defs.complexity_levels || []).map(c => (
                <tr key={c.level}>
                  <td><span className={`badge bg-${COMPLEXITY_COLOR[c.level.toLowerCase()] || 'secondary'}`}>{c.level}</span></td>
                  <td>{c.definition}</td>
                  <td className="small text-muted">{c.example}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h5>Process Types</h5>
          <table className="table table-sm table-bordered mb-4">
            <thead className="table-light">
              <tr><th>Type</th><th>Definition</th><th>Examples</th></tr>
            </thead>
            <tbody>
              {(defs.process_types || []).map(p => (
                <tr key={p.type}>
                  <td><span className={`badge bg-${TYPE_COLOR[p.type] || 'secondary'}`}>{p.type}</span></td>
                  <td>{p.definition}</td>
                  <td className="small">{(p.examples || []).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h5>Flowchart Catalog</h5>
          <table className="table table-sm table-bordered mb-4">
            <thead className="table-light">
              <tr><th>ID</th><th>Title</th><th>Purpose</th></tr>
            </thead>
            <tbody>
              {(defs.flowchart_catalog || []).map(fc => (
                <tr key={fc.id}>
                  <td><code>{fc.id}</code></td>
                  <td className="fw-semibold">{fc.title}</td>
                  <td className="small">{fc.purpose}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {defs.mermaid_syntax && (
            <div className="card p-3">
              <h6>Mermaid Syntax Reference</h6>
              <p className="small mb-2">{defs.mermaid_syntax.description}</p>
              <div className="small">
                <div><strong>Start:</strong> <code>{defs.mermaid_syntax.start_keyword}</code></div>
                <div className="mt-1"><strong>Node formats:</strong>
                  <ul className="mb-0">
                    {(defs.mermaid_syntax.node_formats || []).map((f, i) => <li key={i}><code>{f}</code></li>)}
                  </ul>
                </div>
                <div className="mt-1"><strong>Edge formats:</strong>
                  <ul className="mb-0">
                    {(defs.mermaid_syntax.edge_formats || []).map((f, i) => <li key={i}><code>{f}</code></li>)}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
