'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statColor = s => s === 'healthy' ? 'success' : s === 'degraded' ? 'warning' : 'danger';
const utilColor = s => s === 'ok' ? 'success' : s === 'warning' ? 'warning' : 'danger';
const deltaIcon = d => d > 0 ? '▲' : d < 0 ? '▼' : '─';
const deltaColor = d => d > 5 ? 'text-danger' : d < -5 ? 'text-success' : 'text-muted';

export default function CloudOpsPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/cloud-ops/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/cloud-ops/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/cloud-ops/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',      label: 'Overview' },
    { id: 'resources',     label: 'Resources & Cost' },
    { id: 'autoscale',     label: 'Autoscaling' },
    { id: 'definitions',   label: 'Definitions' },
  ];

  return (
    <div>
      <h3>Cloud Ops Dashboard</h3>
      <p className="text-muted">Infrastructure monitoring: region health, cost tracking, autoscaling events, uptime SLA, resource utilisation</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Regions',        value: ov.kpis.total_regions,           color: 'primary' },
          { label: 'Healthy',        value: ov.kpis.healthy_regions,         color: 'success' },
          { label: 'Avg Uptime',     value: `${ov.kpis.avg_uptime_pct}%`,   color: ov.kpis.avg_uptime_pct >= 99.95 ? 'success' : 'warning' },
          { label: 'Monthly Cost',   value: `$${ov.kpis.total_cost_usd}`,   color: ov.kpis.budget_pct > 80 ? 'danger' : 'info' },
          { label: 'Budget Used',    value: `${ov.kpis.budget_pct}%`,       color: ov.kpis.budget_pct > 80 ? 'danger' : ov.kpis.budget_pct > 50 ? 'warning' : 'success' },
          { label: 'Mean Latency',   value: `${ov.kpis.mean_latency_ms}ms`, color: ov.kpis.mean_latency_ms > 100 ? 'danger' : 'success' },
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
          {/* Region Status */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Region Status</div>
              <div className="card-body">
                {Object.entries(ov.region_status_distribution || {}).map(([status, count]) => (
                  <div key={status} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${statColor(status)}`}>{status}</span>
                    <div className="flex-grow-1 mx-2"><div className="progress" style={{height:'16px'}}><div className={`progress-bar bg-${statColor(status)}`} style={{width:`${count/ov.kpis.total_regions*100}%`}} /></div></div>
                    <span className="fw-bold">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Cost by Service */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Cost by Service</div>
              <div className="card-body">
                {(ov.cost_by_service || []).map(c => {
                  const maxCost = Math.max(...(ov.cost_by_service||[]).map(x=>x.cost_usd), 1);
                  return (
                    <div key={c.service} className="mb-2">
                      <div className="d-flex justify-content-between small">
                        <span>{c.service}</span>
                        <span>${c.cost_usd} <span className={deltaColor(c.delta_pct)}>{deltaIcon(c.delta_pct)} {c.delta_pct}%</span></span>
                      </div>
                      <div className="progress" style={{height:'10px'}}><div className="progress-bar bg-info" style={{width:`${c.cost_usd/maxCost*100}%`}} /></div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Autoscale Summary */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Autoscale Summary</div>
              <div className="card-body">
                <div className="d-flex justify-content-around text-center">
                  <div><div className="h4 text-primary">{ov.autoscale_summary?.total_events || 0}</div><div className="text-muted small">Total Events</div></div>
                  <div><div className="h4 text-success">{ov.autoscale_summary?.scale_up || 0}</div><div className="text-muted small">Scale Up</div></div>
                  <div><div className="h4 text-warning">{ov.autoscale_summary?.scale_down || 0}</div><div className="text-muted small">Scale Down</div></div>
                </div>
                <hr />
                <div className="text-center">
                  <span className={`badge bg-${ov.kpis.sla_met ? 'success' : 'danger'} fs-6`}>
                    SLA {ov.kpis.sla_met ? 'Met' : 'Missed'} (target 99.95%)
                  </span>
                </div>
                <div className="text-center mt-2 text-muted small">
                  {ov.kpis.total_incidents_30d} incident(s) in last 30 days
                </div>
              </div>
            </div>
          </div>

          {/* Region Summary Table */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Region Health Summary</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark"><tr><th>Region</th><th>Status</th><th>Uptime</th><th>Latency</th><th>Instances</th><th>Error Rate</th><th>SLA</th></tr></thead>
                    <tbody>
                      {(ov.region_summary || []).map(r => (
                        <tr key={r.region}>
                          <td>{r.region}</td>
                          <td><span className={`badge bg-${statColor(r.status)}`}>{r.status}</span></td>
                          <td>{r.uptime_pct}%</td>
                          <td>{r.latency_ms}ms</td>
                          <td>{r.instances}</td>
                          <td>{r.error_rate_pct}%</td>
                          <td>{r.meets_sla ? <span className="text-success">&#10003;</span> : <span className="text-danger">&#10007;</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Resources & Cost Tab ─────────────────────────────── */}
      {tab === 'resources' && bd && (
        <div className="row">
          {/* Resource Utilisation */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Resource Utilisation by Region</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark"><tr><th>Region</th><th>CPU</th><th>Memory</th><th>Disk</th><th>Network</th></tr></thead>
                    <tbody>
                      {(bd.resource_utilisation || []).map(u => (
                        <tr key={u.region_id}>
                          <td>{u.label}</td>
                          <td>
                            <div className="d-flex align-items-center">
                              <div className="progress flex-grow-1 me-2" style={{height:'14px'}}><div className={`progress-bar bg-${utilColor(u.cpu_status)}`} style={{width:`${u.cpu_pct}%`}} /></div>
                              <span className="small">{u.cpu_pct}%</span>
                            </div>
                          </td>
                          <td>
                            <div className="d-flex align-items-center">
                              <div className="progress flex-grow-1 me-2" style={{height:'14px'}}><div className={`progress-bar bg-${utilColor(u.memory_status)}`} style={{width:`${u.memory_pct}%`}} /></div>
                              <span className="small">{u.memory_pct}%</span>
                            </div>
                          </td>
                          <td>{u.disk_pct}%</td>
                          <td>{u.network_mbps} Mbps</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Cost Detail */}
          <div className="col-md-8 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Cost Breakdown by Service</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark"><tr><th>Service</th><th>Current ($)</th><th>Previous ($)</th><th>Change</th></tr></thead>
                    <tbody>
                      {(bd.cost_breakdown || []).map(c => (
                        <tr key={c.service_id}>
                          <td>{c.service}</td>
                          <td className="fw-bold">${c.cost_usd}</td>
                          <td className="text-muted">${c.prev_month_usd}</td>
                          <td className={deltaColor(c.delta_pct)}>{deltaIcon(c.delta_pct)} {c.delta_pct}%</td>
                        </tr>
                      ))}
                      <tr className="table-info fw-bold">
                        <td>Total</td>
                        <td>${bd.total_cost_usd}</td>
                        <td colSpan="2">Budget: ${bd.budget_usd} ({bd.budget_pct}% used)</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Budget Gauge */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Budget Status</div>
              <div className="card-body text-center">
                <div className="h1 mb-1" style={{color: bd.budget_pct > 80 ? '#dc3545' : bd.budget_pct > 50 ? '#ffc107' : '#198754'}}>{bd.budget_pct}%</div>
                <div className="text-muted">of ${bd.budget_usd}/mo budget</div>
                <div className="progress mt-3" style={{height:'24px'}}>
                  <div className={`progress-bar bg-${bd.budget_pct > 80 ? 'danger' : bd.budget_pct > 50 ? 'warning' : 'success'}`} style={{width:`${Math.min(bd.budget_pct,100)}%`}}>
                    ${bd.total_cost_usd}
                  </div>
                </div>
                <div className="mt-3 small text-muted">
                  Remaining: ${(bd.budget_usd - bd.total_cost_usd).toFixed(2)}
                </div>
              </div>
            </div>

            {/* Uptime History */}
            <div className="card shadow-sm mt-3">
              <div className="card-header fw-bold">30-Day Uptime</div>
              <div className="card-body text-center">
                <div className="h2 mb-1" style={{color: bd.uptime_avg_30d >= 99.95 ? '#198754' : '#ffc107'}}>{bd.uptime_avg_30d}%</div>
                <div className="text-muted small">avg (target: {bd.uptime_sla_target}%)</div>
                <div className="text-muted small mt-1">{bd.incidents_30d} incident(s)</div>
                <div className="d-flex flex-wrap mt-2 justify-content-center">
                  {(bd.uptime_history || []).map((d, i) => (
                    <div key={i} title={`Day -${d.day_offset}: ${d.uptime_pct}%`}
                      style={{width:'8px',height:'8px',margin:'1px',borderRadius:'2px',
                        backgroundColor: d.uptime_pct >= 99.95 ? '#198754' : d.uptime_pct >= 99.5 ? '#ffc107' : '#dc3545'}} />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Autoscaling Tab ──────────────────────────────────── */}
      {tab === 'autoscale' && bd && (
        <div className="row">
          {/* Policies */}
          <div className="col-md-5 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Autoscale Policies</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark"><tr><th>Policy</th><th>Min</th><th>Max</th><th>Trigger</th><th>Cooldown</th></tr></thead>
                    <tbody>
                      {(bd.autoscale_policies || []).map(p => (
                        <tr key={p.name}>
                          <td className="fw-bold">{p.name}</td>
                          <td>{p.min}</td>
                          <td>{p.max}</td>
                          <td><code className="small">{p.metric}</code></td>
                          <td>{p.cooldown_s}s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Events */}
          <div className="col-md-7 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Recent Autoscale Events</div>
              <div className="card-body p-0">
                <div className="table-responsive" style={{maxHeight:'400px',overflowY:'auto'}}>
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark sticky-top"><tr><th>Time (UTC)</th><th>Policy</th><th>Direction</th><th>From</th><th>To</th><th>Trigger</th></tr></thead>
                    <tbody>
                      {(bd.autoscale_events || []).map((e, i) => (
                        <tr key={i}>
                          <td className="small">{e.timestamp_utc}</td>
                          <td>{e.policy}</td>
                          <td><span className={`badge bg-${e.direction === 'scale-up' ? 'success' : 'warning'}`}>{e.direction}</span></td>
                          <td>{e.from_count}</td>
                          <td>{e.to_count}</td>
                          <td><code className="small">{e.trigger_metric}</code></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ──────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Regions</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-dark"><tr><th>Region ID</th><th>Label</th><th>Provider</th></tr></thead>
                  <tbody>{(defs.regions || []).map(r => <tr key={r.id}><td><code>{r.id}</code></td><td>{r.label}</td><td>{r.provider}</td></tr>)}</tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm mt-3">
              <div className="card-header fw-bold">Services</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-dark"><tr><th>Service</th><th>Label</th></tr></thead>
                  <tbody>{(defs.services || []).map(s => <tr key={s.id}><td><code>{s.id}</code></td><td>{s.label}</td></tr>)}</tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm mt-3">
              <div className="card-header fw-bold">Status Levels</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-dark"><tr><th>Status</th><th>Description</th></tr></thead>
                  <tbody>{Object.entries(defs.status_levels || {}).map(([k,v]) => <tr key={k}><td><span className={`badge bg-${statColor(k)}`}>{k}</span></td><td className="small">{v.description}</td></tr>)}</tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Cost Thresholds</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-dark"><tr><th>Level</th><th>Threshold</th><th>Action</th></tr></thead>
                  <tbody>{Object.entries(defs.cost_alert_levels || {}).map(([k,v]) => <tr key={k}><td className="fw-bold">{k}</td><td>${v.threshold_usd}</td><td className="small">{v.action}</td></tr>)}</tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm mt-3">
              <div className="card-header fw-bold">Resource Thresholds</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-dark"><tr><th>Resource</th><th>Warning</th><th>Critical</th></tr></thead>
                  <tbody>{Object.entries(defs.resource_thresholds || {}).map(([k,v]) => <tr key={k}><td className="text-capitalize">{k}</td><td className="text-warning">{v.warning}{v.unit}</td><td className="text-danger">{v.critical}{v.unit}</td></tr>)}</tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm mt-3">
              <div className="card-header fw-bold">Autoscale Policies</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-dark"><tr><th>Policy</th><th>Min/Max</th><th>Trigger</th><th>Cooldown</th></tr></thead>
                  <tbody>{(defs.autoscale_policies || []).map(p => <tr key={p.name}><td>{p.name}</td><td>{p.min}-{p.max}</td><td><code>{p.metric}</code></td><td>{p.cooldown_s}s</td></tr>)}</tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm mt-3">
              <div className="card-header fw-bold">Clinical Relevance</div>
              <div className="card-body small">{defs.clinical_relevance}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
