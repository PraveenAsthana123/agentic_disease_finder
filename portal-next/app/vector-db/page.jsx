'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function VectorDBPage() {
  const [overview, setOverview] = useState(null);
  const [collections, setCollections] = useState(null);
  const [operations, setOperations] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/vector-db/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/vector-db/collections`).then(r => r.json()).then(setCollections).catch(() => {});
    fetch(`${API}/api/vector-db/operations`).then(r => r.json()).then(setOperations).catch(() => {});
    fetch(`${API}/api/vector-db/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = overview.summary || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'collections', label: 'Collections' },
    { id: 'operations', label: 'Operations' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f5c4;&#xfe0f; Vector DB Dashboard</h3>
      <p className="text-muted">ChromaDB vector store: embeddings, collections, operations, and health from data/vector_db</p>

      {/* Summary cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Vectors', value: (s.total_vectors || 0).toLocaleString(), color: 'primary' },
          { label: 'Dimension', value: s.dimension || '—', color: 'info' },
          { label: 'Collections', value: s.total_collections || 0, color: 'success' },
          { label: 'Patients', value: s.unique_patients || 0, color: 'warning' },
          { label: 'Storage', value: s.storage_human || '—', color: 'secondary' },
          { label: 'Status', value: s.health === 'healthy' ? 'Healthy' : s.health || '—', color: s.health === 'healthy' ? 'success' : 'danger' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h4 mb-0 text-${c.color}`}>{c.value}</div>
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
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Document Type Distribution</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Type</th><th className="text-end">Count</th><th>Bar</th></tr></thead>
                  <tbody>
                    {(overview.type_distribution || []).map(t => {
                      const maxCount = Math.max(...(overview.type_distribution || []).map(x => x.count));
                      const pct = maxCount > 0 ? (t.count / maxCount * 100) : 0;
                      return (
                        <tr key={t.type}>
                          <td><code>{t.type}</code></td>
                          <td className="text-end">{t.count}</td>
                          <td style={{width:'40%'}}>
                            <div className="progress" style={{height:'16px'}}>
                              <div className="progress-bar bg-primary" style={{width:`${pct}%`}} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Store Info</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <tbody>
                    <tr><td className="text-muted">DB File Size</td><td>{s.db_size_human}</td></tr>
                    <tr><td className="text-muted">Total Storage</td><td>{s.storage_human}</td></tr>
                    <tr><td className="text-muted">Queue Depth</td><td>{(s.queue_depth || 0).toLocaleString()}</td></tr>
                    <tr><td className="text-muted">FTS Indexed</td><td>{s.fts_indexed}</td></tr>
                    <tr><td className="text-muted">Metadata Keys</td><td>{(s.metadata_keys || []).join(', ')}</td></tr>
                    <tr><td className="text-muted">Earliest Record</td><td>{overview.date_range?.earliest || '—'}</td></tr>
                    <tr><td className="text-muted">Latest Record</td><td>{overview.date_range?.latest || '—'}</td></tr>
                    <tr><td className="text-muted">Source</td><td><code>{overview.source}</code></td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Collections tab */}
      {tab === 'collections' && collections && (
        <div>
          {(collections.collections || []).map(col => (
            <div key={col.id} className="card shadow-sm mb-3">
              <div className="card-header fw-bold">
                Collection: <code>{col.name}</code>
                <span className="badge bg-primary ms-2">{col.vector_count} vectors</span>
                <span className="badge bg-info ms-1">{col.dimension}d</span>
              </div>
              <div className="card-body">
                <div className="row">
                  <div className="col-md-6">
                    <h6>Patient Distribution (top 15)</h6>
                    <table className="table table-sm">
                      <thead><tr><th>Patient</th><th className="text-end">Vectors</th></tr></thead>
                      <tbody>
                        {(col.patient_distribution || []).map(p => (
                          <tr key={p.patient_id}><td><code>{p.patient_id}</code></td><td className="text-end">{p.vectors}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="col-md-6">
                    <h6>Recent Embeddings</h6>
                    <table className="table table-sm">
                      <thead><tr><th>ID</th><th>Patient</th><th>Preview</th></tr></thead>
                      <tbody>
                        {(col.samples || []).map((s, i) => (
                          <tr key={i}>
                            <td><code className="small">{s.embedding_id}</code></td>
                            <td>{s.patient_id || '—'}</td>
                            <td className="small text-muted" style={{maxWidth:'250px',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{s.document_preview || '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Operations tab */}
      {tab === 'operations' && operations && (
        <div className="row">
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Operation Summary</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Operation</th><th className="text-end">Count</th></tr></thead>
                  <tbody>
                    {(operations.operations_summary || []).map(o => (
                      <tr key={o.operation}><td>{o.operation}</td><td className="text-end">{o.count.toLocaleString()}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            {operations.throughput && (
              <div className="card shadow-sm mt-3">
                <div className="card-header fw-bold">Throughput</div>
                <div className="card-body">
                  <table className="table table-sm mb-0">
                    <tbody>
                      <tr><td className="text-muted">Total Ops</td><td>{operations.throughput.total_ops.toLocaleString()}</td></tr>
                      <tr><td className="text-muted">Time Span</td><td>{operations.throughput.days_span} days</td></tr>
                      <tr><td className="text-muted">Ops / Day</td><td>{operations.throughput.ops_per_day}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Ingestion Timeline</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Date</th><th className="text-end">Ops</th><th>Bar</th></tr></thead>
                  <tbody>
                    {(operations.timeline || []).map(t => {
                      const maxOps = Math.max(...(operations.timeline || []).map(x => x.operations));
                      const pct = maxOps > 0 ? (t.operations / maxOps * 100) : 0;
                      return (
                        <tr key={t.date}>
                          <td className="small">{t.date}</td>
                          <td className="text-end">{t.operations}</td>
                          <td style={{width:'40%'}}>
                            <div className="progress" style={{height:'14px'}}>
                              <div className="progress-bar bg-success" style={{width:`${pct}%`}} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Recent Queue Entries</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Seq</th><th>Op</th><th>ID</th><th>Time</th></tr></thead>
                  <tbody>
                    {(operations.recent_queue || []).map(q => (
                      <tr key={q.seq_id}>
                        <td className="small">{q.seq_id}</td>
                        <td><span className={`badge bg-${q.operation === 'INSERT' ? 'success' : q.operation === 'DELETE' ? 'danger' : 'warning'} small`}>{q.operation}</span></td>
                        <td className="small"><code>{q.embedding_id}</code></td>
                        <td className="small text-muted">{q.created_at?.split(' ')[0]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-8 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Metric Definitions</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Metric</th><th>Description</th><th>Unit</th></tr></thead>
                  <tbody>
                    {(defs.definitions || []).map(d => (
                      <tr key={d.metric}><td className="fw-bold">{d.metric}</td><td>{d.description}</td><td><code>{d.unit}</code></td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Technology</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <tbody>
                    {defs.technology && Object.entries(defs.technology).map(([k, v]) => (
                      <tr key={k}><td className="text-muted">{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
