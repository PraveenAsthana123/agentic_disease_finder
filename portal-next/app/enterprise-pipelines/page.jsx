'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusBadge = s =>
  s === 'built'   ? 'success' :
  s === 'partial' ? 'warning' : 'secondary';

const groupColor = i => ['primary','info','success','warning','danger','secondary','dark'][i % 7];

export default function EnterprisePipelinesPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/enterprise-pipelines/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/enterprise-pipelines/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/enterprise-pipelines/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = ov.summary || {};
  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'bygroup',     label: 'By Group' },
    { id: 'allpipelines',label: 'All Pipelines' },
    { id: 'flow',        label: 'Pipeline Flow' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f3ed; Enterprise Pipelines Dashboard</h3>
      <p className="text-muted small">
        45 enterprise-grade AI/ML pipelines across 12 groups — data, feature, training, evaluation,
        RAG, agentic, inference, drift, security, governance, deployment, and observability.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Groups',          value: s.total_groups,      color: 'primary' },
          { label: 'Pipelines',       value: s.total_pipelines,   color: 'info' },
          { label: 'Total Stages',    value: s.total_stages,      color: 'dark' },
          { label: 'Avg Stages',      value: s.avg_stages,        color: 'secondary' },
          { label: 'Built',           value: s.built,             color: 'success' },
          { label: 'Completion',      value: `${s.built_pct}%`,   color: s.built_pct === 100 ? 'success' : 'warning' },
          { label: 'Mapped to UI',    value: s.has_maps_to,       color: 'primary' },
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
          {/* Pipelines per Group bar chart */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Pipelines per Group</div>
              <div className="card-body">
                {(ov.pipelines_per_group || []).map((g, i) => (
                  <div key={g.name} className="d-flex align-items-center mb-2">
                    <span className="small" style={{minWidth: '110px'}}>{g.name}</span>
                    <div className="progress flex-grow-1 mx-2" style={{height: '14px'}}>
                      <div className={`progress-bar bg-${groupColor(i)}`}
                           style={{width: `${g.value / 7 * 100}%`}}>
                        <span className="small">{g.value}</span>
                      </div>
                    </div>
                    <span className="badge bg-secondary small">{g.total_stages} stages</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Stats + extremes */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Pipeline Statistics</div>
              <div className="card-body">
                <div className="row">
                  <div className="col-6 mb-3">
                    <div className="card bg-light text-center">
                      <div className="card-body py-2">
                        <div className="h4 text-success mb-0">{s.built}/{s.total_pipelines}</div>
                        <div className="text-muted small">Built / Total</div>
                      </div>
                    </div>
                  </div>
                  <div className="col-6 mb-3">
                    <div className="card bg-light text-center">
                      <div className="card-body py-2">
                        <div className="h4 text-info mb-0">{s.total_stages}</div>
                        <div className="text-muted small">Total Stages</div>
                      </div>
                    </div>
                  </div>
                  <div className="col-6 mb-3">
                    <div className="card bg-light text-center">
                      <div className="card-body py-2">
                        <div className="h5 text-primary mb-0">{s.max_stages_pipeline}</div>
                        <div className="text-muted small">Largest ({s.max_stages_count} stages)</div>
                      </div>
                    </div>
                  </div>
                  <div className="col-6 mb-3">
                    <div className="card bg-light text-center">
                      <div className="card-body py-2">
                        <div className="h5 text-secondary mb-0">{s.min_stages_pipeline}</div>
                        <div className="text-muted small">Smallest ({s.min_stages_count} stage)</div>
                      </div>
                    </div>
                  </div>
                </div>
                {/* Status distribution */}
                <h6 className="mt-2">Status Distribution</h6>
                {(ov.status_distribution || []).map(d => (
                  <div key={d.name} className="d-flex justify-content-between align-items-center mb-1">
                    <span className={`badge bg-${statusBadge(d.name)}`}>{d.name}</span>
                    <div className="progress flex-grow-1 mx-2" style={{height: '10px'}}>
                      <div className={`progress-bar bg-${statusBadge(d.name)}`}
                           style={{width: `${d.value / s.total_pipelines * 100}%`}} />
                    </div>
                    <span className="fw-bold">{d.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── By Group Tab ─────────────────────────────────── */}
      {tab === 'bygroup' && bd && (
        <div className="row">
          {(bd.groups || []).map((g, gi) => (
            <div key={g.group} className="col-md-6 col-lg-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className={`card-header bg-${groupColor(gi)} text-white d-flex justify-content-between`}>
                  <span className="fw-bold">{g.group}</span>
                  <span className="badge bg-light text-dark">{g.pipeline_count} pipelines</span>
                </div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Pipeline</th><th>Stages</th><th>Status</th></tr></thead>
                    <tbody>
                      {(g.pipelines || []).map(p => (
                        <tr key={p.name}>
                          <td className="small">{p.name}</td>
                          <td><span className="badge bg-secondary">{p.stage_count}</span></td>
                          <td><span className={`badge bg-${statusBadge(p.status)}`}>{p.status}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── All Pipelines Tab ────────────────────────────── */}
      {tab === 'allpipelines' && bd && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold">All 45 Enterprise Pipelines</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0">
                <thead className="table-dark">
                  <tr><th>Group</th><th>Pipeline</th><th>Stages</th><th>#</th><th>Status</th><th>Maps To</th></tr>
                </thead>
                <tbody>
                  {(bd.groups || []).flatMap(g =>
                    (g.pipelines || []).map(p => (
                      <tr key={g.group + p.name}>
                        <td><span className="badge bg-secondary">{g.group}</span></td>
                        <td className="fw-bold small">{p.name}</td>
                        <td className="small">{(p.stages || []).join(' → ')}</td>
                        <td><span className="badge bg-info">{p.stage_count}</span></td>
                        <td><span className={`badge bg-${statusBadge(p.status)}`}>{p.status}</span></td>
                        <td className="small text-muted" style={{maxWidth: '250px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>
                          {p.maps_to || '—'}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Pipeline Flow Tab ────────────────────────────── */}
      {tab === 'flow' && bd && (
        <div>
          {(bd.groups || []).map((g, gi) => (
            <div key={g.group} className="card shadow-sm mb-3">
              <div className={`card-header bg-${groupColor(gi)} text-white fw-bold`}>
                {g.group} ({g.pipeline_count} pipelines)
              </div>
              <div className="card-body">
                {(g.pipelines || []).map(p => (
                  <div key={p.name} className="mb-3">
                    <div className="d-flex align-items-center mb-1">
                      <strong className="small me-2">{p.name}</strong>
                      <span className={`badge bg-${statusBadge(p.status)} me-2`}>{p.status}</span>
                    </div>
                    <div className="d-flex flex-wrap align-items-center">
                      {(p.stages || []).map((st, si) => (
                        <span key={si} className="d-flex align-items-center">
                          <span className="badge bg-light text-dark border px-2 py-1">{st}</span>
                          {si < p.stages.length - 1 && <span className="mx-1 text-muted">→</span>}
                        </span>
                      ))}
                    </div>
                    {p.maps_to && (
                      <div className="small text-success mt-1">↳ {p.maps_to}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions Tab ──────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          {/* Group Descriptions */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Group Descriptions</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-dark"><tr><th>Group</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.group_descriptions || []).map(g => (
                      <tr key={g.group}>
                        <td className="fw-bold">{g.group}</td>
                        <td className="small">{g.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Status Legend */}
          <div className="col-md-3 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Status Legend</div>
              <div className="card-body">
                {(defs.status_legend || []).map(s => (
                  <div key={s.status} className="d-flex align-items-start mb-2">
                    <span className={`badge bg-${statusBadge(s.status)} me-2`} style={{minWidth: '60px'}}>{s.status}</span>
                    <span className="small">{s.description}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Glossary */}
          <div className="col-md-3 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Glossary</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-dark"><tr><th>Term</th><th>Definition</th></tr></thead>
                  <tbody>
                    {(defs.glossary || []).map(g => (
                      <tr key={g.term}><td className="fw-bold small">{g.term}</td><td className="small">{g.definition}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Clinical Notes + References */}
          {defs.clinical_notes && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold">Notes</div>
                <div className="card-body">
                  <ul className="small mb-0">
                    {(defs.clinical_notes || []).map((n, i) => <li key={i} className="mb-1">{n}</li>)}
                  </ul>
                </div>
              </div>
            </div>
          )}
          {defs.references && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold">References</div>
                <div className="card-body">
                  <ol className="small mb-0">
                    {(defs.references || []).map((r, i) => <li key={i} className="mb-1 text-muted">{r}</li>)}
                  </ol>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
