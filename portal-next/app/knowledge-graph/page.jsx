'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function KnowledgeGraphPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/knowledge-graph/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/knowledge-graph/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/knowledge-graph/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = overview.summary || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'patients', label: 'Patient Subgraphs' },
    { id: 'network', label: 'Disease & Med Network' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f578;&#xfe0f; Knowledge Graph Dashboard</h3>
      <p className="text-muted">Entity-relationship graph from clinical.db + ChromaDB: patients, diseases, medications, analyses, MRI, neuropsych, HITL reviews, and embedded documents</p>

      {/* Summary cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Nodes', value: (s.total_nodes || 0).toLocaleString(), color: 'primary' },
          { label: 'Total Edges', value: (s.total_edges || 0).toLocaleString(), color: 'success' },
          { label: 'Entity Types', value: s.entity_types || 0, color: 'info' },
          { label: 'Relation Types', value: s.relation_types || 0, color: 'warning' },
          { label: 'Avg Degree', value: s.avg_degree || 0, color: 'secondary' },
          { label: 'Max Degree', value: s.max_degree || 0, color: 'danger' },
          { label: 'Density', value: s.density || 0, color: 'dark' },
          { label: 'Isolated', value: s.isolated_nodes || 0, color: s.isolated_nodes > 0 ? 'warning' : 'success' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 col-lg mb-2">
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
              <div className="card-header fw-bold">Entity Type Distribution</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Type</th><th className="text-end">Count</th><th>Bar</th></tr></thead>
                  <tbody>
                    {(overview.type_distribution || []).map(t => {
                      const maxC = Math.max(...(overview.type_distribution || []).map(x => x.count));
                      const pct = maxC > 0 ? (t.count / maxC * 100) : 0;
                      return (
                        <tr key={t.type}>
                          <td><code>{t.type}</code></td>
                          <td className="text-end">{t.count}</td>
                          <td><div className="bg-primary" style={{height: 14, width: `${pct}%`, borderRadius: 3}} /></td>
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
              <div className="card-header fw-bold">Relation Type Distribution</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Relation</th><th className="text-end">Count</th><th>Bar</th></tr></thead>
                  <tbody>
                    {(overview.relation_distribution || []).map(r => {
                      const maxC = Math.max(...(overview.relation_distribution || []).map(x => x.count));
                      const pct = maxC > 0 ? (r.count / maxC * 100) : 0;
                      return (
                        <tr key={r.relation}>
                          <td><code>{r.relation}</code></td>
                          <td className="text-end">{r.count}</td>
                          <td><div className="bg-success" style={{height: 14, width: `${pct}%`, borderRadius: 3}} /></td>
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
              <div className="card-header fw-bold">Hub Nodes (Most Connected)</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>#</th><th>Node</th><th>Type</th><th className="text-end">Degree</th></tr></thead>
                  <tbody>
                    {(overview.hub_nodes || []).map((h, i) => (
                      <tr key={h.id}>
                        <td>{i + 1}</td>
                        <td><code>{h.id}</code></td>
                        <td><span className="badge bg-secondary">{h.type}</span></td>
                        <td className="text-end fw-bold">{h.degree}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Patient subgraphs tab */}
      {tab === 'patients' && breakdown && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold">Per-Patient Subgraph Richness</div>
          <div className="card-body p-0">
            <table className="table table-sm table-hover mb-0">
              <thead><tr><th>Patient</th><th className="text-end">Edges</th><th className="text-end">Neighbors</th><th>Relations</th></tr></thead>
              <tbody>
                {(breakdown.patient_subgraphs || []).map(p => (
                  <tr key={p.patient_id}>
                    <td><code>{p.patient_id}</code></td>
                    <td className="text-end">{p.edges}</td>
                    <td className="text-end">{p.neighbors}</td>
                    <td>{Object.entries(p.relations || {}).map(([k, v]) => (
                      <span key={k} className="badge bg-info me-1">{k}: {v}</span>
                    ))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Disease & Medication network tab */}
      {tab === 'network' && breakdown && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Disease Clusters</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Disease</th><th className="text-end">Patients</th></tr></thead>
                  <tbody>
                    {(breakdown.disease_clusters || []).map(d => (
                      <tr key={d.disease}>
                        <td><strong>{d.disease}</strong></td>
                        <td className="text-end">{d.patient_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Medication Network</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Medication</th><th className="text-end">Patients</th><th>Dose</th><th>Freq</th></tr></thead>
                  <tbody>
                    {(breakdown.medication_network || []).map(m => (
                      <tr key={m.medication}>
                        <td><code>{m.medication}</code></td>
                        <td className="text-end">{m.patients}</td>
                        <td>{m.dose_mg ? `${m.dose_mg} mg` : '—'}</td>
                        <td>{m.frequency || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Graph Statistics</div>
              <div className="card-body">
                <div className="row">
                  {Object.entries(breakdown.graph_stats || {}).map(([k, v]) => (
                    <div key={k} className="col-6 col-md-3 mb-2">
                      <div className="text-muted small">{k.replace(/_/g, ' ')}</div>
                      <div className="fw-bold">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Metric Definitions</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Metric</th><th>Description</th><th>Unit</th></tr></thead>
                  <tbody>
                    {(defs.metrics || []).map(m => (
                      <tr key={m.name}><td className="fw-bold">{m.name}</td><td>{m.description}</td><td><code>{m.unit}</code></td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-3 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Entity Types</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Type</th><th>Source</th></tr></thead>
                  <tbody>
                    {(defs.entity_types || []).map(e => (
                      <tr key={e.type}><td><code>{e.type}</code></td><td className="small">{e.source}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-3 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Relation Types</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Relation</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.relation_types || []).map(r => (
                      <tr key={r.relation}><td><code>{r.relation}</code></td><td className="small">{r.description}</td></tr>
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
