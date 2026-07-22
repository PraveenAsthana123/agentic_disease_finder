import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'collections', label: 'Collections' },
  { id: 'operations', label: 'Operations' },
  { id: 'queue', label: 'Queue Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function VectorDBDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [collections, setCollections] = useState(null)
  const [operations, setOperations] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API}/api/vector-db/overview`),
      axios.get(`${API}/api/vector-db/collections`),
      axios.get(`${API}/api/vector-db/operations`),
      axios.get(`${API}/api/vector-db/definitions`),
    ])
      .then(([ov, col, ops, df]) => {
        setOverview(ov.data)
        setCollections(col.data)
        setOperations(ops.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Vector DB data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return null

  const s = overview.summary || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Vector DB Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        ChromaDB vector storage — embedding index, collection management, and queue operations
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab s={s} overview={overview} operations={operations} />}
      {tab === 'collections' && <CollectionsTab collections={collections} />}
      {tab === 'operations' && <OperationsTab operations={operations} />}
      {tab === 'queue' && <QueueTab operations={operations} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ s, overview, operations }) {
  const typeDist = overview.type_distribution || []
  const timeline = operations?.timeline || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card><KPI label="Total Vectors" value={s.total_vectors} sub="embeddings stored" color="#3b82f6" /></Card>
      <Card><KPI label="Collections" value={s.total_collections} sub="active collections" color="#10b981" /></Card>
      <Card><KPI label="Dimension" value={s.dimension} sub="vector dimensionality" color="#8b5cf6" /></Card>
      <Card><KPI label="Unique Patients" value={s.unique_patients} sub="patient embeddings" color="#06b6d4" /></Card>
      <Card><KPI label="Queue Depth" value={s.queue_depth} sub="pending operations" color="#f59e0b" /></Card>
      <Card><KPI label="Storage" value={s.storage_human} sub="total disk usage" color="#ec4899" /></Card>
      <Card><KPI label="DB Size" value={s.db_size_human} sub="SQLite + HNSW" color="#f97316" /></Card>
      <Card><KPI label="Status" value={<Badge text={s.status || 'online'} color={s.health === 'healthy' ? '#10b981' : '#ef4444'} />} sub={s.health} /></Card>

      {/* Type Distribution Pie */}
      <Card title="Embedding Type Distribution" span={2}>
        {typeDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={typeDist} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={90} label={({ type, count }) => `${type} (${count})`}>
                {typeDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No type data</div>}
      </Card>

      {/* Operations Timeline */}
      <Card title="Operations Timeline" span={2}>
        {timeline.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={timeline}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="operations" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No timeline data</div>}
      </Card>

      {/* Type Distribution Bar */}
      <Card title="Vectors by Type" span={2}>
        {typeDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={typeDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="type" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
      </Card>

      {/* Operations Summary */}
      <Card title="Operations Summary" span={2}>
        {(operations?.operations_summary || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={operations.operations_summary}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="operation" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No operations</div>}
      </Card>

      {/* Metadata Keys */}
      <Card title="Metadata Keys" span={4}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {(s.metadata_keys || []).map((k, i) => (
            <Badge key={i} text={k} color="#3b82f6" />
          ))}
        </div>
      </Card>
    </div>
  )
}

function CollectionsTab({ collections }) {
  const colls = collections?.collections || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {colls.map((col, ci) => (
        <Card key={ci} title={`Collection: ${col.name}`}>
          <div style={{ display: 'flex', gap: 24, marginBottom: 16, flexWrap: 'wrap' }}>
            <div><strong style={{ fontSize: 12, color: '#64748b' }}>ID:</strong> <span style={{ fontSize: 12, fontFamily: 'monospace' }}>{col.id}</span></div>
            <div><strong style={{ fontSize: 12, color: '#64748b' }}>Dimension:</strong> <span style={{ fontSize: 13 }}>{col.dimension}</span></div>
            <div><strong style={{ fontSize: 12, color: '#64748b' }}>Vectors:</strong> <span style={{ fontSize: 13, fontWeight: 600, color: '#3b82f6' }}>{col.vector_count}</span></div>
          </div>

          <h4 style={{ fontSize: 13, color: '#334155', marginBottom: 8 }}>Sample Embeddings</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Embedding ID</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Patient</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Document Preview</th>
                </tr>
              </thead>
              <tbody>
                {(col.samples || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{s.embedding_id}</td>
                    <td style={{ padding: '6px 10px' }}><Badge text={s.patient_id} color="#3b82f6" /></td>
                    <td style={{ padding: '6px 10px', color: '#64748b', maxWidth: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.document_preview}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ))}
      {colls.length === 0 && <Card><div style={{ color: '#94a3b8', fontSize: 13 }}>No collections found</div></Card>}
    </div>
  )
}

function OperationsTab({ operations }) {
  const timeline = operations?.timeline || []
  const opsSummary = operations?.operations_summary || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Operations by Type */}
      <Card title="Operations by Type" span={1}>
        {opsSummary.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={opsSummary} dataKey="count" nameKey="operation" cx="50%" cy="50%" outerRadius={90} label={({ operation, count }) => `${operation} (${count})`}>
                {opsSummary.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No operations</div>}
      </Card>

      {/* Daily Operations */}
      <Card title="Daily Operations" span={1}>
        {timeline.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={timeline}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="operations" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No timeline data</div>}
      </Card>

      {/* Operations Summary Table */}
      <Card title="Operations Summary" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Operation</th>
              <th style={{ padding: '8px 12px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Count</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Share</th>
            </tr>
          </thead>
          <tbody>
            {opsSummary.map((op, i) => {
              const total = opsSummary.reduce((a, b) => a + b.count, 0)
              const pct = total > 0 ? ((op.count / total) * 100).toFixed(1) : 0
              return (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}><Badge text={op.operation} color={COLORS[i % COLORS.length]} /></td>
                  <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600 }}>{op.count}</td>
                  <td style={{ padding: '8px 12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ width: 80, height: 6, background: '#f1f5f9', borderRadius: 3 }}>
                        <div style={{ width: `${pct}%`, height: 6, background: COLORS[i % COLORS.length], borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 12, color: '#64748b' }}>{pct}%</span>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Card>
    </div>
  )
}

function QueueTab({ operations }) {
  const queue = operations?.recent_queue || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Recent Queue Operations (${queue.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Seq</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Timestamp</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Operation</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Embedding ID</th>
              </tr>
            </thead>
            <tbody>
              {queue.map((q, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11, color: '#94a3b8' }}>{q.seq_id}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{q.created_at}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={q.operation} color={q.operation === 'INSERT' ? '#10b981' : q.operation === 'DELETE' ? '#ef4444' : '#3b82f6'} /></td>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{q.embedding_id}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {queue.length === 0 && <div style={{ color: '#94a3b8', fontSize: 13, padding: 20 }}>No queue operations</div>}
      </Card>
    </div>
  )
}

function DefinitionsTab({ definitions }) {
  const defs = definitions?.definitions || []
  const tech = definitions?.technology || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Technology */}
      <Card title="Technology Stack">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <div>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Database</div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>{tech.database || 'N/A'}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Distance Metric</div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>{tech.distance_metric || 'N/A'}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Index Type</div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>{tech.index_type || 'N/A'}</div>
          </div>
        </div>
      </Card>

      {/* Metric Definitions */}
      <Card title="Metric Definitions">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Metric</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Unit</th>
            </tr>
          </thead>
          <tbody>
            {defs.map((d, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600 }}>{d.metric}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{d.description}</td>
                <td style={{ padding: '8px 12px' }}><Badge text={d.unit} color="#8b5cf6" /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Compliance */}
      <Card title="Clinical Relevance & Compliance">
        <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.8 }}>
          <p>Vector databases store patient clinical embeddings for semantic search and RAG retrieval.
          Compliance considerations include:</p>
          <ul style={{ marginTop: 8 }}>
            <li><strong>HIPAA:</strong> Patient embeddings contain derived PHI; access must be logged and encrypted at rest</li>
            <li><strong>EU AI Act Art.10:</strong> Data governance for training data used in embedding generation</li>
            <li><strong>FDA AI-ML:</strong> Embedding quality affects downstream clinical decision support accuracy</li>
            <li><strong>ISO 14971:</strong> Vector search failures are a risk to clinical decision reliability</li>
            <li><strong>IEC 62304:</strong> Vector DB is a software component requiring lifecycle management</li>
          </ul>
        </div>
      </Card>
    </div>
  )
}
