'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const qColor = s => s >= 80 ? 'success' : s >= 50 ? 'warning' : 'danger';

export default function DataOpsPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/data-ops/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/data-ops/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/data-ops/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'pipelines',   label: 'Pipelines & Ingestion' },
    { id: 'quality',     label: 'Data Quality' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>DataOps Dashboard</h3>
      <p className="text-muted">Data pipeline monitoring: ingestion metrics, quality scores, storage stats, vector ingest, lineage</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Patients',       value: ov.kpis.total_patients,                    color: 'primary' },
          { label: 'Uploads',        value: ov.kpis.total_uploads,                     color: 'info' },
          { label: 'Analyses',       value: ov.kpis.total_analyses,                    color: 'info' },
          { label: 'AI Readiness',   value: `${ov.kpis.ai_readiness}%`,               color: qColor(ov.kpis.ai_readiness) },
          { label: 'Signal Good',    value: `${ov.kpis.signal_good_pct}%`,            color: qColor(ov.kpis.signal_good_pct) },
          { label: 'Avg Coverage',   value: `${ov.kpis.avg_coverage_pct}%`,           color: qColor(ov.kpis.avg_coverage_pct) },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ─────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Modality Coverage */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Modality Coverage</div>
              <div className="card-body">
                {Object.entries(ov.modality_coverage || {}).map(([mod, pct]) => (
                  <div key={mod} className="mb-2">
                    <div className="d-flex justify-content-between small">
                      <span>{mod}</span><span className="fw-bold">{pct}%</span>
                    </div>
                    <div className="progress" style={{height:'12px'}}>
                      <div className={`progress-bar bg-${qColor(pct)}`} style={{width:`${pct}%`}} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Quality Dimensions */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Quality Dimensions</div>
              <div className="card-body">
                {Object.entries(ov.quality_dimensions_summary || {}).map(([dim, score]) => (
                  <div key={dim} className="mb-2">
                    <div className="d-flex justify-content-between small">
                      <span>{dim}</span><span className="fw-bold">{score}</span>
                    </div>
                    <div className="progress" style={{height:'12px'}}>
                      <div className={`progress-bar bg-${qColor(score)}`} style={{width:`${score}%`}} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Signal Quality */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Signal Quality Distribution</div>
              <div className="card-body">
                {Object.entries(ov.signal_quality_distribution || {}).map(([q, cnt]) => (
                  <div key={q} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${q === 'Good' ? 'success' : 'warning'}`}>{q}</span>
                    <span className="fw-bold">{cnt}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Storage */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Storage</div>
              <div className="card-body">
                <div className="d-flex justify-content-between mb-2">
                  <span>clinical.db</span><span className="fw-bold">{ov.kpis.db_size_mb} MB</span>
                </div>
                <div className="d-flex justify-content-between mb-2">
                  <span>Vector DB (ChromaDB)</span><span className="fw-bold">{ov.kpis.vector_size_mb} MB</span>
                </div>
                <div className="d-flex justify-content-between mb-2">
                  <span>Transaction Events</span><span className="fw-bold">{ov.kpis.total_txn_events}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Vector Ingest */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Vector Ingest</div>
              <div className="card-body">
                <div className="d-flex justify-content-between mb-2">
                  <span>Status</span>
                  <span className={`badge bg-${ov.vector_ingest?.status === 'ok' ? 'success' : 'warning'}`}>
                    {ov.vector_ingest?.status || 'unknown'}
                  </span>
                </div>
                <div className="d-flex justify-content-between mb-2">
                  <span>Last Run</span><span className="small">{ov.vector_ingest?.last_run || 'N/A'}</span>
                </div>
                <div className="d-flex justify-content-between mb-2">
                  <span>Records Embedded</span><span className="fw-bold">{ov.vector_ingest?.records_embedded || 0}</span>
                </div>
                <div className="d-flex justify-content-between mb-2">
                  <span>DB Size</span><span className="fw-bold">{ov.vector_ingest?.db_size_mb || 0} MB</span>
                </div>
              </div>
            </div>
          </div>

          {/* Top Pipeline Activity */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Top Pipeline Activity</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Component</th><th>Action</th><th>Count</th></tr></thead>
                  <tbody>
                    {(ov.pipeline_top5 || []).map((r, i) => (
                      <tr key={i}><td>{r.component}</td><td>{r.action}</td><td className="fw-bold">{r.count}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Pipelines & Ingestion Tab ────────────────────────── */}
      {tab === 'pipelines' && bd && (
        <div className="row">
          {/* Full pipeline activity table */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">All Pipeline Activity</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Component</th><th>Action</th><th>Count</th><th>First Seen</th><th>Last Seen</th></tr></thead>
                  <tbody>
                    {(bd.pipeline_activity || []).map((r, i) => (
                      <tr key={i}>
                        <td>{r.component}</td><td>{r.action}</td>
                        <td className="fw-bold">{r.count}</td>
                        <td className="small text-muted">{r.first_seen?.slice(0, 16)}</td>
                        <td className="small text-muted">{r.last_seen?.slice(0, 16)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Ingestion breakdown */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Ingestion by Component</div>
              <div className="card-body">
                {(bd.ingestion_breakdown || []).map(r => {
                  const maxCnt = Math.max(...(bd.ingestion_breakdown || []).map(x => x.count), 1);
                  return (
                    <div key={r.component} className="mb-2">
                      <div className="d-flex justify-content-between small">
                        <span>{r.component}</span><span className="fw-bold">{r.count}</span>
                      </div>
                      <div className="progress" style={{height:'10px'}}>
                        <div className="progress-bar bg-primary" style={{width:`${r.count/maxCnt*100}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Daily volume */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Daily Transaction Volume</div>
              <div className="card-body">
                {(bd.daily_volume || []).map(r => {
                  const maxD = Math.max(...(bd.daily_volume || []).map(x => x.count), 1);
                  return (
                    <div key={r.date} className="mb-1">
                      <div className="d-flex justify-content-between small">
                        <span>{r.date}</span><span className="fw-bold">{r.count}</span>
                      </div>
                      <div className="progress" style={{height:'8px'}}>
                        <div className="progress-bar bg-info" style={{width:`${r.count/maxD*100}%`}} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Storage inventory */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Storage Inventory (Table Row Counts)</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Table</th><th>Rows</th></tr></thead>
                  <tbody>
                    {(bd.storage_inventory || []).map(r => (
                      <tr key={r.table}><td>{r.table}</td><td className="fw-bold">{r.rows}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Data lineage */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Data Lineage Pipeline</div>
              <div className="card-body">
                <ol className="mb-0">
                  {(bd.data_lineage || []).map((step, i) => (
                    <li key={i} className="mb-1">{step}</li>
                  ))}
                </ol>
                <div className="text-muted small mt-2">Last quality run: {bd.dq_run_at}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Data Quality Tab ─────────────────────────────────── */}
      {tab === 'quality' && bd && (
        <div className="row">
          {/* AI Readiness */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">AI Readiness Components</div>
              <div className="card-body">
                <div className="row">
                  {Object.entries(bd.ai_readiness_components || {}).map(([k, v]) => (
                    <div key={k} className="col-md-2 col-4 text-center mb-2">
                      <div className={`h4 mb-0 text-${qColor(v)}`}>{v}%</div>
                      <div className="small text-muted text-capitalize">{k.replace(/_/g, ' ')}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Quality dimensions full */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Quality Dimensions (ISO 25012)</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Dimension</th><th>Score</th><th>Basis</th><th>Measured</th></tr></thead>
                  <tbody>
                    {(bd.quality_dimensions || []).map(d => (
                      <tr key={d.dimension}>
                        <td className="fw-bold">{d.dimension}</td>
                        <td>{d.score !== null ? <span className={`text-${qColor(d.score)}`}>{d.score}</span> : <span className="text-muted">N/A</span>}</td>
                        <td className="small">{d.basis}</td>
                        <td>{d.measured === true ? <span className="badge bg-success">Yes</span> : <span className="badge bg-secondary">No</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Missing matrix */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Missing Data Matrix</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Modality</th><th>Present</th><th>Missing</th><th>% Missing</th></tr></thead>
                  <tbody>
                    {(bd.missing_matrix || []).map(m => (
                      <tr key={m.modality}>
                        <td>{m.modality}</td>
                        <td className="text-success fw-bold">{m.present}</td>
                        <td className="text-danger fw-bold">{m.missing}</td>
                        <td>
                          <div className="d-flex align-items-center">
                            <div className="progress flex-grow-1 me-2" style={{height:'10px'}}>
                              <div className="progress-bar bg-danger" style={{width:`${m.pct_missing}%`}} />
                            </div>
                            <span className="small">{m.pct_missing}%</span>
                          </div>
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

      {/* ── Definitions Tab ──────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div>
          {(defs.sections || []).map(sec => (
            <div key={sec.title} className="card shadow-sm mb-3">
              <div className="card-header fw-bold">{sec.title}</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th style={{width:'25%'}}>Term</th><th>Definition</th></tr></thead>
                  <tbody>
                    {(sec.items || []).map(it => (
                      <tr key={it.term}><td className="fw-bold">{it.term}</td><td>{it.definition}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
