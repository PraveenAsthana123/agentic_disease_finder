import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#16a34a', '#eab308', '#ef4444', '#8b5cf6', '#ec4899', '#f59e0b', '#06b6d4']
const PIE_COLORS = ['#10b981', '#3b82f6', '#eab308', '#ef4444', '#8b5cf6']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(2)) : String(v)
}

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

export default function ReinforcementLearningDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      try {
        const [ov, bd, df] = await Promise.all([
          axios.get(`${API_URL}/api/reinforcement-learning/overview`),
          axios.get(`${API_URL}/api/reinforcement-learning/breakdown`),
          axios.get(`${API_URL}/api/reinforcement-learning/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading RL environment data...</div>
  if (error) return <div style={{ padding: 40, color: '#dc2626' }}>Error: {error}</div>

  const tabs = ['overview', 'breakdown', 'strategies', 'definitions']
  const kpi = overview?.kpi || {}
  const env = overview?.environment || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#0f172a' }}>Reinforcement Learning — Treatment Optimization</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          RL-based clinical decision support: state observations from wearables + medication + seizure diary → reward-driven treatment policy
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: 600, textTransform: 'capitalize',
            background: tab === t ? '#3b82f6' : '#f1f5f9', color: tab === t ? '#fff' : '#475569'
          }}>{t}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview(overview, kpi, env)}
      {tab === 'breakdown' && renderBreakdown(breakdown)}
      {tab === 'strategies' && renderStrategies(breakdown, overview)}
      {tab === 'definitions' && renderDefinitions(defs)}
    </div>
  )
}

function renderOverview(ov, kpi, env) {
  const rewards = ov?.patient_rewards || []
  const bins = ov?.reward_distribution?.bins || []
  const stateObs = ov?.state_observations || {}
  const policy = ov?.policy_performance || {}
  const actions = env?.action_space?.actions || []
  const meds = env?.action_space?.available_medications || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Environment KPIs" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
          <KPI label="Patients Tracked" value={kpi.patients_tracked || 0} color="#3b82f6" />
          <KPI label="Adherence Records" value={fmt(kpi.total_adherence_records)} color="#16a34a" />
          <KPI label="Seizure Events" value={kpi.total_seizure_events || 0} color="#ef4444" />
          <KPI label="Wearable Readings" value={fmt(kpi.total_wearable_readings)} color="#8b5cf6" />
          <KPI label="Mean Reward" value={fmt(kpi.mean_reward)} color="#f59e0b" />
          <KPI label="Medications" value={kpi.medications_in_formulary || 0} color="#06b6d4" />
        </div>
      </Card>

      <Card title="Reward Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={bins}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="range" fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 11, color: '#64748b', marginTop: 8 }}>
          Mean: {fmt(ov?.reward_distribution?.mean_reward)} · Std: {fmt(ov?.reward_distribution?.std_reward)} · Median: {fmt(ov?.reward_distribution?.median_reward)}
        </div>
      </Card>

      <Card title="State Observations (Wearables)">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Heart Rate</div>
            <div style={{ fontSize: 18, fontWeight: 600 }}>{fmt(stateObs.heart_rate?.mean)}</div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>±{fmt(stateObs.heart_rate?.std)}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#64748b' }}>HRV (ms)</div>
            <div style={{ fontSize: 18, fontWeight: 600 }}>{fmt(stateObs.hrv?.mean)}</div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>±{fmt(stateObs.hrv?.std)}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Seizure Risk</div>
            <div style={{ fontSize: 18, fontWeight: 600 }}>{fmt(stateObs.seizure_risk?.mean)}</div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>±{fmt(stateObs.seizure_risk?.std)}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Seizures Detected</div>
            <div style={{ fontSize: 18, fontWeight: 600, color: '#ef4444' }}>{stateObs.seizures_detected || 0}</div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>from {fmt(stateObs.total_readings)} readings</div>
          </div>
        </div>
      </Card>

      <Card title="Patient Reward Scores" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={rewards.slice(0, 20)} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 1]} fontSize={11} />
            <YAxis type="category" dataKey="patient_id" fontSize={10} width={70} />
            <Tooltip formatter={v => fmt(v)} />
            <Legend />
            <Bar dataKey="adherence_component" stackId="a" fill="#16a34a" name="Adherence" />
            <Bar dataKey="seizure_freedom_component" stackId="a" fill="#3b82f6" name="Seizure Freedom" />
            <Bar dataKey="qol_component" stackId="a" fill="#f59e0b" name="QoLIE-31" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Action Space">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {actions.map((a, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 10px', background: '#f8fafc', borderRadius: 6 }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600 }}>{a.action}</div>
                <div style={{ fontSize: 11, color: '#64748b' }}>{a.frequency}</div>
              </div>
              <span style={{
                fontSize: 10, padding: '2px 8px', borderRadius: 10, fontWeight: 600,
                background: a.type === 'conservative' ? '#dcfce7' : a.type === 'escalation' ? '#fee2e2' : a.type === 'de-escalation' ? '#dbeafe' : '#fef9c3',
                color: a.type === 'conservative' ? '#166534' : a.type === 'escalation' ? '#991b1b' : a.type === 'de-escalation' ? '#1e40af' : '#854d0e'
              }}>{a.type}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Available Medications (Formulary)">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {meds.map((m, i) => (
            <span key={i} style={{ fontSize: 12, padding: '4px 10px', background: '#eff6ff', color: '#1e40af', borderRadius: 6, fontWeight: 500 }}>{m}</span>
          ))}
        </div>
        <div style={{ marginTop: 12, fontSize: 12, color: '#64748b' }}>
          Policy decisions: {policy.total_decisions || 0} · AI confidence: {fmt(policy.mean_ai_confidence)} · Agreement: {fmt(policy.neurologist_agreement_rate * 100)}%
        </div>
      </Card>
    </div>
  )
}

function renderBreakdown(bd) {
  const trajectories = bd?.patient_trajectories || []
  const expl = bd?.exploration_vs_exploitation || {}
  const counterfactual = bd?.counterfactual_outcomes || []
  const severity = bd?.severity_penalty || []
  const risk = bd?.risk_transitions || []
  const summary = bd?.summary || {}

  const explData = [
    { name: 'Explorers (>2 meds)', count: expl.explorers || 0, adherence: expl.explorer_mean_adherence || 0, stability: expl.explorer_mean_stability || 0 },
    { name: 'Exploiters (≤2 meds)', count: expl.exploiters || 0, adherence: expl.exploiter_mean_adherence || 0, stability: expl.exploiter_mean_stability || 0 }
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Trajectory Summary" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
          <KPI label="Patients" value={summary.patients_with_trajectories || 0} color="#3b82f6" />
          <KPI label="Improving" value={summary.improving_patients || 0} color="#16a34a" />
          <KPI label="Stable" value={summary.stable_patients || 0} color="#eab308" />
          <KPI label="Declining" value={summary.declining_patients || 0} color="#ef4444" />
          <KPI label="Mean Stability" value={fmt(summary.mean_stability)} color="#8b5cf6" />
        </div>
      </Card>

      <Card title="Exploration vs Exploitation" span={1}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={explData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" fontSize={10} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Legend />
            <Bar dataKey="adherence" fill="#3b82f6" name="Mean Adherence %" radius={[4, 4, 0, 0]} />
            <Bar dataKey="stability" fill="#16a34a" name="Stability" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 12, color: '#475569', marginTop: 8 }}>
          Optimal strategy: <strong style={{ textTransform: 'capitalize' }}>{expl.optimal_strategy || '--'}</strong>
        </div>
      </Card>

      <Card title="Seizure Severity Penalties">
        {severity.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={severity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="severity" fontSize={11} />
              <YAxis fontSize={11} />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#ef4444" name="Count" radius={[4, 4, 0, 0]} />
              <Bar dataKey="reward_penalty" fill="#f59e0b" name="Penalty" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No seizure severity data</div>}
      </Card>

      <Card title="Patient Adherence Trajectories" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Patient</th>
                <th style={{ padding: '6px 8px', textAlign: 'right' }}>Days</th>
                <th style={{ padding: '6px 8px', textAlign: 'right' }}>Adherence %</th>
                <th style={{ padding: '6px 8px', textAlign: 'right' }}>Stability</th>
                <th style={{ padding: '6px 8px', textAlign: 'center' }}>Trend</th>
                <th style={{ padding: '6px 8px', textAlign: 'right' }}>Meds Tried</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Medications</th>
              </tr>
            </thead>
            <tbody>
              {trajectories.slice(0, 20).map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 8px', fontWeight: 600 }}>{t.patient_id}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right' }}>{t.records}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right' }}>{fmt(t.mean_adherence)}%</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right' }}>{fmt(t.stability_score)}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                    <span style={{
                      fontSize: 10, padding: '2px 8px', borderRadius: 10, fontWeight: 600,
                      background: t.trend_direction === 'improving' ? '#dcfce7' : t.trend_direction === 'declining' ? '#fee2e2' : '#fef9c3',
                      color: t.trend_direction === 'improving' ? '#166534' : t.trend_direction === 'declining' ? '#991b1b' : '#854d0e'
                    }}>{t.trend_direction}</span>
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right' }}>{t.medication_count}</td>
                  <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b' }}>{(t.medications_tried || []).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="QoLIE-31 Counterfactual Outcomes" span={1}>
        <ResponsiveContainer width="100%" height={240}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="first_qolie31" name="Baseline QoLIE-31" fontSize={11} />
            <YAxis dataKey="latest_qolie31" name="Latest QoLIE-31" fontSize={11} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter data={counterfactual} fill="#3b82f6" />
          </ScatterChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>X = baseline, Y = latest score. Above diagonal = improved.</div>
      </Card>

      <Card title="Seizure Risk State Transitions" span={1}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={risk.slice(0, 15)} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" fontSize={11} />
            <YAxis type="category" dataKey="patient_id" fontSize={10} width={65} />
            <Tooltip />
            <Legend />
            <Bar dataKey="risk_increasing_transitions" fill="#ef4444" name="Risk ↑" />
            <Bar dataKey="risk_decreasing_transitions" fill="#16a34a" name="Risk ↓" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function renderStrategies(bd, ov) {
  const strategies = bd?.exploration_vs_exploitation || {}
  const env = ov?.environment || {}
  const stateFeatures = env?.state_space?.features || []
  const rewardComponents = env?.reward_function?.components || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="RL Environment Specification" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
          <div>
            <h4 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>State Space ({stateFeatures.length} dimensions)</h4>
            {stateFeatures.map((f, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #f1f5f9', fontSize: 12 }}>
                <span style={{ fontWeight: 500 }}>{f.feature}</span>
                <span style={{ color: '#64748b' }}>{f.source} · {f.type}</span>
              </div>
            ))}
          </div>
          <div>
            <h4 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>Reward Function</h4>
            {rewardComponents.map((c, i) => (
              <div key={i} style={{ padding: '6px 10px', background: '#f8fafc', borderRadius: 6, marginBottom: 6, fontSize: 12 }}>
                {c}
              </div>
            ))}
            <div style={{ marginTop: 12, fontSize: 12 }}>
              <div><strong>Range:</strong> {env?.reward_function?.range || '[0, 1]'}</div>
              <div><strong>Discount (γ):</strong> {env?.reward_function?.discount_factor || 0.95}</div>
            </div>
          </div>
        </div>
      </Card>

      <Card title="Exploration vs Exploitation">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ padding: 12, background: '#eff6ff', borderRadius: 8 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e40af' }}>Explorers ({strategies.explorers || 0} patients)</div>
            <div style={{ fontSize: 12, color: '#3b82f6' }}>Tried &gt;2 medications</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>Mean adherence: {fmt(strategies.explorer_mean_adherence)}%</div>
          </div>
          <div style={{ padding: 12, background: '#f0fdf4', borderRadius: 8 }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#166534' }}>Exploiters ({strategies.exploiters || 0} patients)</div>
            <div style={{ fontSize: 12, color: '#16a34a' }}>Stayed with ≤2 medications</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>Mean adherence: {fmt(strategies.exploiter_mean_adherence)}%</div>
          </div>
          <div style={{ padding: 8, background: '#fefce8', borderRadius: 8, fontSize: 13, fontWeight: 600, textAlign: 'center' }}>
            Optimal: <span style={{ textTransform: 'uppercase', color: '#854d0e' }}>{strategies.optimal_strategy || '--'}</span>
          </div>
        </div>
      </Card>

      <Card title="Safety Constraints">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[
            'Clinician retains full override authority',
            'No abrupt AED withdrawal',
            'Dose within therapeutic ranges',
            'Emergency override for status epilepticus',
            'Min 4-week observation before policy update',
            'Adverse events trigger automatic review'
          ].map((s, i) => (
            <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'flex-start', fontSize: 12 }}>
              <span style={{ color: '#16a34a', fontWeight: 700, flexShrink: 0 }}>✓</span>
              <span>{s}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

function renderDefinitions(defs) {
  if (!defs) return null
  const concepts = defs?.concepts || []
  const refs = defs?.clinical_references || []
  const algos = defs?.rl_algorithms || []
  const constraints = defs?.safety_constraints || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="RL Concepts" span={2}>
        {concepts.map((c, i) => (
          <div key={i} style={{ padding: '10px 0', borderBottom: i < concepts.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b' }}>{c.term}</div>
            <div style={{ fontSize: 12, color: '#475569', marginTop: 2 }}>{c.definition}</div>
          </div>
        ))}
      </Card>

      <Card title="Applicable RL Algorithms">
        {algos.map((a, i) => (
          <div key={i} style={{ padding: '8px 12px', background: i % 2 === 0 ? '#f8fafc' : '#fff', borderRadius: 6, marginBottom: 4 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#1e40af' }}>{a.name}</div>
            <div style={{ fontSize: 12, color: '#475569' }}>{a.suitability}</div>
          </div>
        ))}
      </Card>

      <Card title="Clinical References">
        {refs.map((r, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < refs.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#334155' }}>{r.ref}</div>
            <div style={{ fontSize: 12, color: '#475569', fontStyle: 'italic' }}>{r.title}</div>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{r.relevance}</div>
          </div>
        ))}
      </Card>
    </div>
  )
}
