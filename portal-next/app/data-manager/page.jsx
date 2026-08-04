'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusBadge = s =>
  s === 'built'   ? 'success' :
  s === 'partial' ? 'warning' : 'secondary';

const catColor = i => ['primary','info','success','warning','danger','secondary','dark'][i % 7];

export default function DataManagerPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/data-manager/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/data-manager/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/data-manager/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = ov.summary || {};
  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'tasks',       label: 'All Tasks' },
    { id: 'dashboards',  label: 'Sub-Dashboards' },
    { id: 'workflow',    label: 'Workflow' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f4cb; Clinical Data Manager Dashboard</h3>
      <p className="text-muted small">
        {ov.mission}
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Tasks',             value: s.total_tasks,          color: 'primary' },
          { label: 'Built',             value: s.built,                color: 'success' },
          { label: 'Partial',           value: s.partial,              color: 'warning' },
          { label: 'Planned',           value: s.planned,              color: 'secondary' },
          { label: 'Completion',        value: `${s.built_pct}%`,      color: s.built_pct === 100 ? 'success' : 'warning' },
          { label: 'Steps',             value: s.total_steps,          color: 'info' },
          { label: 'Challenges',        value: s.total_challenges,     color: 'danger' },
          { label: 'Sub-Dashboards',    value: s.dashboards,           color: 'dark' },
          { label: 'Quality Metrics',   value: s.quality_assessments,  color: 'primary' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 col-lg mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value ?? '\u2014'}</div>
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

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-7 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Task Summary</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark">
                      <tr><th>Task</th><th>AI Feature</th><th>Deliverable</th><th>Steps</th><th>Status</th></tr>
                    </thead>
                    <tbody>
                      {(ov.task_table || []).map(t => (
                        <tr key={t.name}>
                          <td className="fw-bold small">{t.name}</td>
                          <td className="small">{t.ai_feature}</td>
                          <td className="small">{t.deliverable}</td>
                          <td><span className="badge bg-info">{t.steps_count}</span></td>
                          <td><span className={`badge bg-${statusBadge(t.status)}`}>{t.status}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div className="col-md-5 mb-3">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">Status Distribution</div>
              <div className="card-body">
                {(ov.status_distribution || []).map(d => (
                  <div key={d.name} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${statusBadge(d.name)}`} style={{minWidth: '70px'}}>{d.name}</span>
                    <div className="progress flex-grow-1 mx-2" style={{height: '14px'}}>
                      <div className={`progress-bar bg-${statusBadge(d.name)}`}
                           style={{width: `${d.value / s.total_tasks * 100}%`}}>
                        <span className="small">{d.value}</span>
                      </div>
                    </div>
                    <span className="fw-bold">{d.value}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="card shadow-sm">
              <div className="card-header fw-bold">Quality Assessments</div>
              <div className="card-body">
                <div className="d-flex flex-wrap gap-2">
                  {(ov.quality_assessments || []).map((qa, i) => (
                    <span key={i} className={`badge bg-${catColor(i)} py-2 px-3`}>{qa}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* All Tasks Tab */}
      {tab === 'tasks' && bd && (
        <div>
          {(bd.per_task || []).map((t, ti) => (
            <div key={t.name} className="card shadow-sm mb-3">
              <div className={`card-header bg-${catColor(ti)} text-white d-flex justify-content-between align-items-center`}>
                <span className="fw-bold">{t.name}</span>
                <div>
                  <span className="badge bg-light text-dark me-1">{t.ai_feature}</span>
                  <span className={`badge bg-${statusBadge(t.status)}`}>{t.status}</span>
                </div>
              </div>
              <div className="card-body">
                <div className="row">
                  <div className="col-md-6">
                    <h6>Steps</h6>
                    <ol className="small mb-0">
                      {(t.steps || []).map((st, si) => <li key={si} className="mb-1">{st}</li>)}
                    </ol>
                  </div>
                  <div className="col-md-6">
                    <h6>Challenges</h6>
                    <ul className="small text-danger mb-0">
                      {(t.challenges || []).map((ch, ci) => <li key={ci} className="mb-1">{ch}</li>)}
                    </ul>
                    {(t.endpoints || []).length > 0 && (
                      <div className="mt-2">
                        <h6>Endpoints</h6>
                        {t.endpoints.map((ep, ei) => (
                          <span key={ei} className="badge bg-dark me-1">{ep}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
                <div className="mt-2">
                  <span className="small text-muted">Deliverable: </span>
                  <span className="badge bg-info">{t.deliverable}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Sub-Dashboards Tab */}
      {tab === 'dashboards' && (
        <div className="row">
          {(ov.dashboards || []).map((d, di) => (
            <div key={d.name} className="col-md-4 col-lg-3 mb-3">
              <div className="card shadow-sm h-100">
                <div className={`card-header bg-${catColor(di)} text-white fw-bold`}>
                  {d.name}
                </div>
                <div className="card-body">
                  <p className="small mb-2">{d.shows}</p>
                  <span className={`badge bg-${statusBadge(d.status)}`}>{d.status}</span>
                  {d.endpoint && (
                    <div className="mt-2">
                      <span className="badge bg-dark small">{d.endpoint}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Workflow Tab */}
      {tab === 'workflow' && bd && (
        <div>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Data Manager Workflow Pipeline</div>
            <div className="card-body">
              <div className="d-flex flex-wrap align-items-center">
                {(bd.per_task || []).map((t, ti) => (
                  <span key={t.name} className="d-flex align-items-center mb-1">
                    <span className={`badge bg-${statusBadge(t.status)} py-2 px-3`}>
                      {t.name}
                    </span>
                    {ti < (bd.per_task || []).length - 1 && (
                      <span className="mx-1 text-muted fw-bold">&rarr;</span>
                    )}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-bold">Steps per Phase</div>
            <div className="card-body">
              {(bd.per_task || []).map((t, ti) => (
                <div key={t.name} className="d-flex align-items-center mb-2">
                  <span className="small fw-bold" style={{minWidth: '150px'}}>{t.name}</span>
                  <div className="progress flex-grow-1 mx-2" style={{height: '18px'}}>
                    <div className={`progress-bar bg-${catColor(ti)}`}
                         style={{width: `${(t.steps || []).length / 5 * 100}%`}}>
                      <span className="small">{(t.steps || []).length} steps</span>
                    </div>
                  </div>
                  <span className="badge bg-danger small">{(t.challenges || []).length} challenges</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Task Categories</div>
              <div className="card-body">
                {(defs.task_categories || []).map((cat, ci) => (
                  <div key={cat.name} className="mb-3">
                    <h6 className={`text-${catColor(ci)}`}>{cat.name}</h6>
                    <div className="d-flex flex-wrap gap-1">
                      {(cat.tasks || []).map(t => (
                        <span key={t} className="badge bg-light text-dark border">{t}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">Role Description</div>
              <div className="card-body">
                <p className="small mb-0">{defs.role_description}</p>
              </div>
            </div>

            {defs.status_legend && (
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold">Status Legend</div>
                <div className="card-body">
                  {(defs.status_legend || []).map(sl => (
                    <div key={sl.status} className="d-flex align-items-start mb-2">
                      <span className={`badge bg-${statusBadge(sl.status)} me-2`} style={{minWidth: '60px'}}>{sl.status}</span>
                      <span className="small">{sl.description}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {defs.glossary && (
              <div className="card shadow-sm">
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
            )}
          </div>

          {defs.clinical_notes && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Clinical Notes</div>
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
              <div className="card shadow-sm">
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
