import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const SEV_COLORS = { low: '#10b981', medium: '#f59e0b', high: '#ef4444' }
const STATUS_COLORS = { active: '#10b981', archived: '#94a3b8' }

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

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(2)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

function fmtDollars(v) {
  if (v == null) return '--'
  return '$' + v.toFixed(2)
}

const TABS = ['Overview', 'Prompts & Versions', 'Token & Cost', 'Hallucinations', 'RAG Evaluation']

export default function LLMOpsDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/api/llmops/overview`),
      axios.get(`${API}/api/llmops/breakdown`),
      axios.get(`${API}/api/llmops/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading LLMOps data…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const k = ov?.kpis || {}

  /* ── Tab: Overview ── */
  const renderOverview = () => {
    const costByModel = ov?.cost_by_model || []
    const costByProvider = Object.entries(ov?.cost_by_provider || {}).map(([name, cost]) => ({ name, cost }))
    const hallucByCat = Object.entries(ov?.hallucination_by_category || {}).map(([name, count]) => ({
      name: name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      count
    }))
    const promptSummary = ov?.prompt_summary || []

    return (
      <>
        <Card title="Key Metrics" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
            <KPI label="Active Prompts" value={k.active_prompts} />
            <KPI label="Total Versions" value={k.total_versions} />
            <KPI label="Avg Accuracy" value={fmtPct(k.avg_accuracy)} color="#10b981" />
            <KPI label="Avg Halluc Rate" value={fmtPct(k.avg_hallucination_rate)} color={k.avg_hallucination_rate > 0.05 ? '#ef4444' : '#10b981'} />
            <KPI label="30d Cost" value={fmtDollars(k.total_cost_30d_usd)} sub={`${k.budget_pct}% of budget`} color={k.budget_pct > 80 ? '#ef4444' : '#1e293b'} />
            <KPI label="30d Requests" value={fmt(k.total_requests_30d)} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginTop: 16 }}>
            <KPI label="30d Tokens" value={fmt(k.total_tokens_30d)} />
            <KPI label="Latency Violations" value={k.latency_violations} color={k.latency_violations > 0 ? '#ef4444' : '#10b981'} />
            <KPI label="High-Halluc Prompts" value={k.high_halluc_prompts} color={k.high_halluc_prompts > 0 ? '#ef4444' : '#10b981'} />
            <KPI label="RAG Faithfulness" value={fmtPct(k.avg_rag_faithfulness)} color="#3b82f6" />
          </div>
        </Card>

        <Card title="Cost by Provider">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={costByProvider} dataKey="cost" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, cost }) => `${name}: $${cost}`}>
                {costByProvider.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip formatter={v => fmtDollars(v)} />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Cost by Model">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={costByModel}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={v => typeof v === 'number' ? fmtDollars(v) : v} />
              <Bar dataKey="cost_usd" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Cost (USD)" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Hallucinations by Category" span={2}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={hallucByCat} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis dataKey="name" type="category" width={150} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#ef4444" radius={[0, 4, 4, 0]} name="Detections" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Prompt Summary" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Prompt', 'Category', 'Model', 'Versions', 'Accuracy', 'Halluc Rate', 'Latency (ms)'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {promptSummary.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 500 }}>{p.prompt}</td>
                    <td style={{ padding: '8px 10px' }}><Badge text={p.category} color="#3b82f6" /></td>
                    <td style={{ padding: '8px 10px' }}>{p.model}</td>
                    <td style={{ padding: '8px 10px' }}>{p.versions}</td>
                    <td style={{ padding: '8px 10px', color: p.accuracy >= 0.9 ? '#10b981' : '#f59e0b' }}>{fmtPct(p.accuracy)}</td>
                    <td style={{ padding: '8px 10px', color: p.halluc_rate > 0.05 ? '#ef4444' : '#10b981' }}>{fmtPct(p.halluc_rate)}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(p.latency_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    )
  }

  /* ── Tab: Prompts & Versions ── */
  const renderPrompts = () => {
    const prompts = bd?.prompt_versions || []

    return (
      <>
        {prompts.map((p, idx) => (
          <Card key={p.prompt_id} title={`${p.label} (${p.category})`}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>
              Model: <strong>{p.model_id}</strong> &middot; {p.total_versions} versions &middot;
              Current accuracy: <span style={{ color: '#10b981', fontWeight: 600 }}>{fmtPct(p.current_accuracy)}</span>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={p.versions}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="version" tick={{ fontSize: 11 }} label={{ value: 'Version', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} domain={[0.6, 1]} />
                <Tooltip formatter={(v, name) => [name === 'accuracy' ? fmtPct(v) : fmtPct(v), name === 'accuracy' ? 'Accuracy' : 'Halluc Rate']} />
                <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} dot />
                <Line type="monotone" dataKey="hallucination_rate" stroke="#ef4444" strokeWidth={2} dot />
                <Legend />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ overflowX: 'auto', marginTop: 8 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['Ver', 'Tokens', 'Accuracy', 'Halluc', 'Latency', 'Status', 'Age (days)'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', textAlign: 'left', color: '#64748b' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {p.versions.map(v => (
                    <tr key={v.version} style={{ borderBottom: '1px solid #f1f5f9', background: v.status === 'active' ? '#f0fdf4' : undefined }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>v{v.version}</td>
                      <td style={{ padding: '6px 8px' }}>{fmt(v.avg_tokens)}</td>
                      <td style={{ padding: '6px 8px' }}>{fmtPct(v.accuracy)}</td>
                      <td style={{ padding: '6px 8px' }}>{fmtPct(v.hallucination_rate)}</td>
                      <td style={{ padding: '6px 8px' }}>{fmt(v.avg_latency_ms)} ms</td>
                      <td style={{ padding: '6px 8px' }}><Badge text={v.status} color={STATUS_COLORS[v.status] || '#94a3b8'} /></td>
                      <td style={{ padding: '6px 8px' }}>{v.created_days_ago}d</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        ))}
      </>
    )
  }

  /* ── Tab: Token & Cost ── */
  const renderTokenCost = () => {
    const daily = bd?.daily_token_usage || []
    const modelTotals = bd?.model_totals || {}
    const budget = bd?.budget_usd || 0

    // Flatten daily data for chart: each day → total input/output/cost across models
    const dailyChart = daily.map(d => {
      let totalIn = 0, totalOut = 0, totalCost = 0
      const models = d.models || {}
      Object.values(models).forEach(m => {
        totalIn += m.input_tokens || 0
        totalOut += m.output_tokens || 0
        totalCost += m.cost_usd || 0
      })
      return { day: `D-${d.day_offset}`, input: totalIn, output: totalOut, cost: parseFloat(totalCost.toFixed(2)) }
    }).reverse()

    const modelRows = Object.entries(modelTotals).map(([id, m]) => ({ id, ...m }))

    return (
      <>
        <Card title="Budget" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
            <KPI label="30d Spend" value={fmtDollars(bd?.total_cost_30d_usd)} />
            <KPI label="Budget" value={fmtDollars(budget)} />
            <KPI label="Budget Used" value={`${bd?.budget_pct}%`} color={bd?.budget_pct > 80 ? '#ef4444' : '#10b981'} />
            <KPI label="30d Requests" value={fmt(bd?.total_requests_30d)} />
          </div>
        </Card>

        <Card title="Daily Token Usage (14d)" span={2}>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={dailyChart}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="input" stackId="a" fill="#3b82f6" name="Input Tokens" />
              <Bar dataKey="output" stackId="a" fill="#10b981" name="Output Tokens" radius={[4, 4, 0, 0]} />
              <Legend />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Daily Cost (14d)" span={2}>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={dailyChart}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={v => fmtDollars(v)} />
              <Line type="monotone" dataKey="cost" stroke="#8b5cf6" strokeWidth={2} dot name="Cost (USD)" />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Model Totals (30d)" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Model', 'Requests', 'Input Tokens', 'Output Tokens', 'Cost'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {modelRows.map(m => (
                  <tr key={m.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 500 }}>{m.id}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(m.requests)}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(m.input_tokens)}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(m.output_tokens)}</td>
                    <td style={{ padding: '8px 10px' }}>{fmtDollars(m.cost_usd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    )
  }

  /* ── Tab: Hallucinations ── */
  const renderHallucinations = () => {
    const report = bd?.hallucination_report || []

    return (
      <>
        <Card title="Hallucination Summary" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 16 }}>
            <KPI label="Total Evaluations" value={fmt(report.reduce((s, r) => s + r.total_evaluations, 0))} />
            <KPI label="Total Detected" value={fmt(report.reduce((s, r) => s + r.hallucinations_detected, 0))} color="#ef4444" />
            <KPI label="High Severity" value={report.filter(r => r.severity === 'high').length} color="#ef4444" />
          </div>
        </Card>

        {report.map(r => (
          <Card key={r.prompt_id} title={r.label}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>
              {r.total_evaluations} evals &middot; {r.hallucinations_detected} detected &middot;
              Rate: <span style={{ color: SEV_COLORS[r.severity], fontWeight: 600 }}>{fmtPct(r.hallucination_rate)}</span>
              &nbsp;<Badge text={r.severity} color={SEV_COLORS[r.severity]} />
            </div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={Object.entries(r.by_category).map(([cat, count]) => ({
                cat: cat.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
                count
              }))} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="cat" type="category" width={140} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count" fill={SEV_COLORS[r.severity]} radius={[0, 4, 4, 0]} name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        ))}
      </>
    )
  }

  /* ── Tab: RAG Evaluation ── */
  const renderRAG = () => {
    const rag = bd?.rag_evaluation || []
    const metrics = df?.rag_metrics || {}

    const ragChart = rag.map(r => ({
      name: r.label,
      recall: r.recall_at_5,
      precision: r.precision_at_5,
      mrr: r.mrr,
      ndcg: r.ndcg,
      faithfulness: r.faithfulness,
      relevance: r.relevance,
    }))

    return (
      <>
        <Card title="RAG Pipeline Metrics" span={2}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={ragChart}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
              <Tooltip formatter={v => fmtPct(v)} />
              <Bar dataKey="recall" fill="#3b82f6" name="Recall@5" />
              <Bar dataKey="precision" fill="#10b981" name="Precision@5" />
              <Bar dataKey="faithfulness" fill="#8b5cf6" name="Faithfulness" />
              <Bar dataKey="relevance" fill="#f59e0b" name="Relevance" />
              <Legend />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Pipeline Details" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Pipeline', 'Vector DB', 'Chunk', 'Overlap', 'Recall@5', 'Precision@5', 'MRR', 'NDCG', 'Faithfulness', 'Latency', 'Docs', 'Stale?'].map(h => (
                    <th key={h} style={{ padding: '8px 8px', textAlign: 'left', color: '#64748b', fontWeight: 600, fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rag.map(r => (
                  <tr key={r.pipeline_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px', fontWeight: 500 }}>{r.label}</td>
                    <td style={{ padding: '8px' }}>{r.vector_db}</td>
                    <td style={{ padding: '8px' }}>{r.chunk_size}</td>
                    <td style={{ padding: '8px' }}>{r.overlap}</td>
                    <td style={{ padding: '8px' }}>{fmtPct(r.recall_at_5)}</td>
                    <td style={{ padding: '8px' }}>{fmtPct(r.precision_at_5)}</td>
                    <td style={{ padding: '8px' }}>{fmtPct(r.mrr)}</td>
                    <td style={{ padding: '8px' }}>{fmtPct(r.ndcg)}</td>
                    <td style={{ padding: '8px' }}>{fmtPct(r.faithfulness)}</td>
                    <td style={{ padding: '8px' }}>{r.retrieval_latency_ms} ms</td>
                    <td style={{ padding: '8px' }}>{fmt(r.total_documents)}</td>
                    <td style={{ padding: '8px' }}>{r.index_stale ? <Badge text="Stale" color="#ef4444" /> : <Badge text="Fresh" color="#10b981" />}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Metric Definitions" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
            {Object.entries(metrics).map(([key, desc]) => (
              <div key={key} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 4 }}>
                  {key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                </div>
                <div style={{ fontSize: 12, color: '#64748b' }}>{desc}</div>
              </div>
            ))}
          </div>
        </Card>
      </>
    )
  }

  const renderers = [renderOverview, renderPrompts, renderTokenCost, renderHallucinations, renderRAG]

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>LLMOps Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Prompt versioning, token/cost tracking, latency monitoring, hallucination detection &amp; RAG evaluation
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === i ? '#1e293b' : '#f1f5f9', color: tab === i ? '#fff' : '#64748b'
          }}>{t}</button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 20 }}>
        {renderers[tab]()}
      </div>
    </div>
  )
}
