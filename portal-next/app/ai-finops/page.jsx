'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const CAT_COLORS = {
  'Autobuild (GPU + Agent)': 'primary',
  'Health Monitoring': 'success',
  'Watchdog': 'info',
  'Git/CI Triggers': 'warning',
  'Ops (Restarts)': 'danger',
  'Ollama Inference': 'dark',
  'Model Storage': 'secondary',
};

export default function AiFinopsPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/ai-finops/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/ai-finops/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/ai-finops/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;
  if (!ov.available) return <div className="p-4 alert alert-warning">AI FinOps data unavailable</div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'analysis',    label: 'Cost Analysis' },
    { id: 'builds',      label: 'Build Sessions' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const maxDaily = Math.max(...(ov.daily_costs || []).map(d => d.cost), 0.01);

  return (
    <div>
      <h3>AI FinOps</h3>
      <p className="text-muted">Real cost tracking from track.jsonl, model files, and system events</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Cost',       value: `$${ov.total_cost}`,         color: 'primary' },
          { label: 'Last 24h',         value: `$${ov.cost_last_24h}`,      color: 'info' },
          { label: 'Last 7d',          value: `$${ov.cost_last_7d}`,       color: 'warning' },
          { label: 'Total Builds',     value: ov.total_builds,              color: 'success' },
          { label: 'Cost/Build',       value: `$${ov.cost_per_build}`,     color: ov.cost_per_build > 1 ? 'danger' : 'success' },
          { label: 'Storage',          value: `${ov.total_storage_mb} MB`, color: 'secondary' },
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

      {/* ── Overview Tab ──────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Cost Breakdown by Category */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Cost Breakdown by Category</div>
              <div className="card-body">
                {(ov.cost_breakdown || []).map(c => (
                  <div key={c.category} className="d-flex justify-content-between align-items-center mb-2">
                    <span className="fw-semibold" style={{minWidth:160, fontSize:'0.85rem'}}>{c.category}</span>
                    <div className="d-flex align-items-center" style={{width:'50%'}}>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className={`progress-bar bg-${CAT_COLORS[c.category] || 'secondary'}`}
                             style={{width:`${Math.max(c.percent, 1)}%`}} />
                      </div>
                      <span className="fw-bold small">${c.cost} ({c.percent}%)</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Daily Cost Trend */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Daily Cost (Last 14 Days)</div>
              <div className="card-body">
                {(ov.daily_costs || []).map(d => (
                  <div key={d.date} className="d-flex align-items-center mb-1">
                    <span className="text-muted small" style={{minWidth:85}}>{d.date.slice(5)}</span>
                    <div className="progress flex-grow-1 me-2" style={{height:'16px'}}>
                      <div className="progress-bar bg-primary" style={{width:`${d.cost / maxDaily * 100}%`}} />
                    </div>
                    <span className="fw-bold small">${d.cost}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Model Storage */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Model Storage</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Model</th><th>Size (MB)</th><th>Monthly Cost</th></tr></thead>
                  <tbody>
                    {(ov.model_storage || []).map((m, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{m.name}</td>
                        <td>{m.size_mb}</td>
                        <td className="text-muted">${m.monthly_cost.toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="p-2 text-muted small">
                  Total: {ov.total_storage_mb} MB
                </div>
              </div>
            </div>
          </div>

          {/* Event Counts */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Event Summary</div>
              <div className="card-body">
                <div className="d-flex justify-content-between mb-2">
                  <span>Total Events Tracked</span>
                  <span className="fw-bold text-primary">{ov.total_events}</span>
                </div>
                <div className="d-flex justify-content-between mb-2">
                  <span>Build Sessions</span>
                  <span className="fw-bold">{ov.total_builds}</span>
                </div>
                <div className="d-flex justify-content-between mb-2">
                  <span>Successful Builds</span>
                  <span className="fw-bold text-success">{ov.successful_builds}</span>
                </div>
                <div className="d-flex justify-content-between mb-2">
                  <span>Total Build Minutes</span>
                  <span className="fw-bold">{ov.total_build_minutes} min</span>
                </div>
                <hr />
                {(ov.cost_breakdown || []).map(c => (
                  <div key={c.category} className="d-flex justify-content-between mb-1">
                    <span className="small">{c.category}</span>
                    <span className="small fw-bold">{c.events} events</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Cost Analysis Tab ─────────────────────────────────── */}
      {tab === 'analysis' && bd && (
        <div className="row">
          {/* Cost Velocity */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Cost Velocity (Last 14 Days)</div>
              <div className="card-body">
                {(() => {
                  const maxCum = Math.max(...(bd.cost_velocity || []).map(d => d.cumulative), 0.01);
                  return (bd.cost_velocity || []).map(d => (
                    <div key={d.date} className="d-flex align-items-center mb-1">
                      <span className="text-muted small" style={{minWidth:85}}>{d.date.slice(5)}</span>
                      <div className="progress flex-grow-1 me-2" style={{height:'16px'}}>
                        <div className="progress-bar bg-primary" style={{width:`${d.cumulative / maxCum * 100}%`}} />
                      </div>
                      <span className="small">${d.daily}d / <span className="fw-bold">${d.cumulative}c</span></span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>

          {/* Hourly Cost Heatmap */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Hourly Cost Heatmap (Last 7 Days)</div>
              <div className="card-body" style={{overflowX:'auto'}}>
                <table className="table table-sm table-bordered text-center mb-0" style={{fontSize:'0.75rem'}}>
                  <thead>
                    <tr>
                      <th></th>
                      {(bd.hourly_heatmap.hour_labels || []).map(h => <th key={h}>{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.hourly_heatmap.matrix || []).map((row, ri) => (
                      <tr key={ri}>
                        <td className="fw-bold">{(bd.hourly_heatmap.day_labels || [])[ri]}</td>
                        {row.map((v, ci) => {
                          const maxV = Math.max(...bd.hourly_heatmap.matrix.flat(), 0.001);
                          const intensity = v / maxV;
                          const bg = v === 0 ? '#f8f9fa' : `rgba(220, 53, 69, ${0.15 + intensity * 0.85})`;
                          const fg = intensity > 0.5 ? '#fff' : '#333';
                          return <td key={ci} style={{backgroundColor: bg, color: fg}}>{v > 0 ? `$${v}` : ''}</td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Cost Efficiency */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Cost Efficiency by Category</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Level</th><th>Events</th><th>Cost/Event</th><th>Total</th></tr></thead>
                  <tbody>
                    {(bd.efficiency || []).map((e, i) => (
                      <tr key={i}>
                        <td><span className="badge bg-secondary">{e.level}</span></td>
                        <td className="fw-bold">{e.events}</td>
                        <td className="text-muted">{e.cost_per_event}</td>
                        <td className="fw-bold">{e.total_cost}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Storage Breakdown */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Storage Breakdown</div>
              <div className="card-body">
                {(bd.storage_breakdown || []).map(m => (
                  <div key={m.model} className="d-flex justify-content-between align-items-center mb-2">
                    <span className="fw-semibold" style={{minWidth:100}}>{m.model}</span>
                    <div className="d-flex align-items-center" style={{width:'55%'}}>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className="progress-bar bg-info" style={{width:`${m.percent}%`}} />
                      </div>
                      <span className="small fw-bold">{m.size_mb} MB ({m.percent}%)</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Top Expensive Builds */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Top 5 Most Expensive Builds</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Start</th><th>Duration</th><th>Cost</th></tr></thead>
                  <tbody>
                    {(bd.top_expensive_builds || []).map((b, i) => (
                      <tr key={i}>
                        <td className="small">{b.start}</td>
                        <td>{b.duration_min} min</td>
                        <td className="fw-bold text-danger">{b.cost}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Downtime Cost */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Downtime Cost</div>
              <div className="card-body text-center">
                <div className="h2 text-danger">${bd.downtime_cost}</div>
                <div className="text-muted">{bd.downtime_events} downtime events</div>
                <div className="small text-muted mt-2">Wasted compute from system downtime (avg ~2 min per event)</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Build Sessions Tab ────────────────────────────────── */}
      {tab === 'builds' && bd && (
        <div>
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Build Session Log (Recent 25)</div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0">
                <thead><tr><th>Start</th><th>Duration (min)</th><th>Cost</th></tr></thead>
                <tbody>
                  {(bd.build_log || []).map((b, i) => (
                    <tr key={i}>
                      <td className="small">{b.start}</td>
                      <td>{b.duration_min}</td>
                      <td className="fw-bold">{b.cost}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ───────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Cost Model</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Resource</th><th>Rate</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.cost_model || []).map((c, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{c.resource}</td>
                        <td className="text-primary fw-bold">{c.rate}</td>
                        <td className="small">{c.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Metric Definitions</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th style={{width:'25%'}}>Term</th><th>Definition</th></tr></thead>
                  <tbody>
                    {(defs.metrics || []).map((m, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{m.term}</td>
                        <td>{m.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Optimization Strategies</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Strategy</th><th>Potential Savings</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.optimization_strategies || []).map((s, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{s.strategy}</td>
                        <td><span className="badge bg-success">{s.savings}</span></td>
                        <td>{s.description}</td>
                      </tr>
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
