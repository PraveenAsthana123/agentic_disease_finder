'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const lvlColor = l => l === 'ERROR' || l === 'FATAL' ? 'danger' : l === 'WARN' ? 'warning' : l === 'DEBUG' ? 'secondary' : 'info';
const statusBg = s => s === 'critical' ? 'danger' : s === 'warning' ? 'warning' : 'success';
const pctBar = (val, max) => Math.min(100, Math.round(val / Math.max(max, 1) * 100));

export default function ObservabilityPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/observability/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/observability/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/observability/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="container py-4"><p>Loading Observability Dashboard...</p></div>;

  const k = ov.kpis || {};
  const tabs = ['overview', 'logs & traces', 'component detail', 'definitions'];

  return (
    <div className="container-fluid py-4" style={{background:'#0b1120',minHeight:'100vh',color:'#e0e0e0'}}>
      <h2 className="mb-1" style={{color:'#00e5ff'}}>&#x1f441;&#xfe0f; Observability Dashboard</h2>
      <p className="text-secondary mb-3">Real-time logs, traces, and metrics from transaction_log ({k.total_events} events)</p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {tabs.map(t => (
          <li className="nav-item" key={t}>
            <button className={`nav-link ${tab===t?'active text-white':'text-secondary'}`}
              style={tab===t?{background:'#1a2744',borderColor:'#00e5ff'}:{background:'transparent'}}
              onClick={()=>setTab(t)}>{t.charAt(0).toUpperCase()+t.slice(1)}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && <>
        {/* KPI Cards */}
        <div className="row g-3 mb-4">
          {[
            {label:'Total Events', val:k.total_events, color:'#00e5ff'},
            {label:'Errors', val:k.error_count, color:'#ff5252'},
            {label:'Error Rate', val:`${k.error_rate_pct}%`, color: k.error_rate_pct > 5 ? '#ff5252' : '#4caf50'},
            {label:'Warnings', val:k.warn_count, color:'#ffc107'},
            {label:'P50 Latency', val:`${k.p50_latency_ms}ms`, color:'#00e5ff'},
            {label:'P95 Latency', val:`${k.p95_latency_ms}ms`, color: k.p95_latency_ms > 1000 ? '#ffc107' : '#4caf50'},
            {label:'Components', val:k.active_components, color:'#7c4dff'},
            {label:'Active Alerts', val:k.active_alerts, color: k.active_alerts > 0 ? '#ff5252' : '#4caf50'},
          ].map((c,i) => (
            <div className="col-md-3 col-sm-6" key={i}>
              <div className="card text-center" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
                <div className="card-body py-3">
                  <div className="small text-secondary">{c.label}</div>
                  <div className="fs-3 fw-bold" style={{color:c.color}}>{c.val}</div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Active Alerts */}
        {ov.active_alerts && ov.active_alerts.length > 0 && (
          <div className="card mb-4" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
            <div className="card-header" style={{background:'#0d1b30',color:'#ff5252'}}>Active Alerts</div>
            <div className="card-body p-0">
              <table className="table table-dark table-sm mb-0">
                <thead><tr><th>Rule</th><th>Severity</th><th>Value</th><th>Component</th><th>Status</th></tr></thead>
                <tbody>
                  {ov.active_alerts.map((a,i) => (
                    <tr key={i}>
                      <td>{a.rule}</td>
                      <td><span className={`badge bg-${a.severity==='critical'?'danger':'warning'}`}>{a.severity}</span></td>
                      <td>{a.value}</td>
                      <td>{a.component || '—'}</td>
                      <td><span className="badge bg-danger">{a.status}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Log Level Distribution */}
        <div className="row g-3 mb-4">
          <div className="col-md-6">
            <div className="card" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
              <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Log Level Distribution</div>
              <div className="card-body">
                {ov.log_level_distribution && Object.entries(ov.log_level_distribution).map(([level, count]) => (
                  <div key={level} className="mb-2">
                    <div className="d-flex justify-content-between small">
                      <span className={`badge bg-${lvlColor(level)}`}>{level}</span>
                      <span>{count}</span>
                    </div>
                    <div className="progress" style={{height:'8px',background:'#0d1b30'}}>
                      <div className={`progress-bar bg-${lvlColor(level)}`}
                        style={{width:`${pctBar(count, k.total_events)}%`}}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
              <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Latency Percentiles</div>
              <div className="card-body">
                {[
                  {label:'P50', val:k.p50_latency_ms, max:8000},
                  {label:'P95', val:k.p95_latency_ms, max:8000},
                  {label:'P99', val:k.p99_latency_ms, max:8000},
                  {label:'Mean', val:k.mean_latency_ms, max:8000},
                ].map(p => (
                  <div key={p.label} className="mb-2">
                    <div className="d-flex justify-content-between small">
                      <span>{p.label}</span>
                      <span>{p.val}ms</span>
                    </div>
                    <div className="progress" style={{height:'8px',background:'#0d1b30'}}>
                      <div className="progress-bar" style={{
                        width:`${pctBar(p.val, p.max)}%`,
                        background: p.val > 3000 ? '#ff5252' : p.val > 1000 ? '#ffc107' : '#4caf50'
                      }}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Daily Volume Chart */}
        {ov.daily_volume && ov.daily_volume.length > 0 && (
          <div className="card mb-4" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
            <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Daily Event Volume</div>
            <div className="card-body">
              <div className="d-flex align-items-end" style={{height:'120px',gap:'2px'}}>
                {ov.daily_volume.map((d,i) => {
                  const maxV = Math.max(...ov.daily_volume.map(x=>x.total));
                  const h = Math.max(4, Math.round(d.total / Math.max(maxV,1) * 110));
                  return (
                    <div key={i} style={{flex:1,display:'flex',flexDirection:'column',alignItems:'center'}}>
                      <div style={{width:'100%',height:`${h}px`,background: d.errors > 0 ? '#ff5252' : '#00e5ff',
                        borderRadius:'2px 2px 0 0',minWidth:'4px'}} title={`${d.date}: ${d.total} events, ${d.errors} errors`}></div>
                    </div>
                  );
                })}
              </div>
              <div className="d-flex justify-content-between mt-1">
                <small className="text-secondary">{ov.daily_volume[0]?.date}</small>
                <small className="text-secondary">{ov.daily_volume[ov.daily_volume.length-1]?.date}</small>
              </div>
            </div>
          </div>
        )}

        {/* Component Health Table */}
        <div className="card mb-4" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
          <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Component Health</div>
          <div className="card-body p-0">
            <table className="table table-dark table-sm table-hover mb-0">
              <thead><tr><th>Component</th><th>Events</th><th>Errors</th><th>Error Rate</th><th>P50</th><th>P95</th><th>P99</th><th>Status</th></tr></thead>
              <tbody>
                {(ov.component_health||[]).map((c,i) => (
                  <tr key={i}>
                    <td className="fw-bold">{c.component}</td>
                    <td>{c.total_events}</td>
                    <td>{c.error_count}</td>
                    <td>{c.error_rate_pct}%</td>
                    <td>{c.p50_ms}ms</td>
                    <td>{c.p95_ms}ms</td>
                    <td>{c.p99_ms}ms</td>
                    <td><span className={`badge bg-${statusBg(c.status)}`}>{c.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </>}

      {/* ── Logs & Traces Tab ── */}
      {tab === 'logs & traces' && bd && <>
        {/* Recent Logs */}
        <div className="card mb-4" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
          <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Recent Log Entries (last 50)</div>
          <div className="card-body p-0" style={{maxHeight:'400px',overflowY:'auto'}}>
            <table className="table table-dark table-sm table-hover mb-0" style={{fontSize:'0.8rem'}}>
              <thead style={{position:'sticky',top:0,background:'#1a2744'}}><tr>
                <th>ID</th><th>Timestamp</th><th>Level</th><th>Component</th><th>Action</th><th>Actor</th><th>Latency</th><th>Trace ID</th>
              </tr></thead>
              <tbody>
                {(bd.recent_logs||[]).map((l,i) => (
                  <tr key={i}>
                    <td>{l.id}</td>
                    <td className="text-secondary">{(l.timestamp||'').replace('T',' ').slice(0,19)}</td>
                    <td><span className={`badge bg-${lvlColor(l.level)}`}>{l.level}</span></td>
                    <td>{l.component}</td>
                    <td>{l.action}</td>
                    <td>{l.actor}</td>
                    <td>{l.latency_ms}ms</td>
                    <td className="text-secondary" style={{fontSize:'0.7rem'}}>{(l.trace_id||'').slice(0,13)}...</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Sample Traces */}
        <div className="card mb-4" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
          <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Sample Distributed Traces</div>
          <div className="card-body">
            {(bd.sample_traces||[]).map((t,i) => (
              <div key={i} className="mb-3 p-2" style={{background:'#0d1b30',borderRadius:'6px',border:'1px solid #2a3a5c'}}>
                <div className="d-flex justify-content-between mb-2">
                  <span><strong>Trace:</strong> <code style={{color:'#7c4dff'}}>{(t.trace_id||'').slice(0,20)}...</code></span>
                  <span className="small text-secondary">Patient: {t.patient_id} | {t.span_count} spans | {t.total_duration_ms}ms
                    {t.has_error && <span className="badge bg-danger ms-2">ERROR</span>}
                  </span>
                </div>
                <div className="d-flex" style={{gap:'1px',height:'24px'}}>
                  {t.spans.map((s,j) => {
                    const w = Math.max(3, Math.round(s.duration_ms / Math.max(t.total_duration_ms,1) * 100));
                    return (
                      <div key={j} style={{
                        width:`${w}%`, background: s.status==='error' ? '#ff5252' : '#00e5ff',
                        borderRadius:'3px', opacity: 0.8
                      }} title={`${s.span_type} (${s.component}/${s.action}) — ${s.duration_ms}ms`}></div>
                    );
                  })}
                </div>
                <div className="d-flex flex-wrap mt-1" style={{gap:'4px'}}>
                  {t.spans.map((s,j) => (
                    <small key={j} className="text-secondary">{s.span_type}:{s.duration_ms}ms</small>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Action & Actor Distribution */}
        <div className="row g-3 mb-4">
          <div className="col-md-6">
            <div className="card" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
              <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Action Distribution</div>
              <div className="card-body">
                {bd.action_distribution && Object.entries(bd.action_distribution).map(([action, count]) => (
                  <div key={action} className="mb-1">
                    <div className="d-flex justify-content-between small">
                      <span>{action}</span><span>{count}</span>
                    </div>
                    <div className="progress" style={{height:'6px',background:'#0d1b30'}}>
                      <div className="progress-bar" style={{width:`${pctBar(count, Math.max(...Object.values(bd.action_distribution)))}%`,background:'#7c4dff'}}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
              <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Actor Distribution</div>
              <div className="card-body">
                {bd.actor_distribution && Object.entries(bd.actor_distribution).map(([actor, count]) => (
                  <div key={actor} className="mb-1">
                    <div className="d-flex justify-content-between small">
                      <span>{actor}</span><span>{count}</span>
                    </div>
                    <div className="progress" style={{height:'6px',background:'#0d1b30'}}>
                      <div className="progress-bar" style={{width:`${pctBar(count, Math.max(...Object.values(bd.actor_distribution)))}%`,background:'#00bcd4'}}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </>}

      {/* ── Component Detail Tab ── */}
      {tab === 'component detail' && bd && <>
        <div className="row g-3">
          {(bd.component_health||[]).map((c,i) => (
            <div className="col-md-6 col-lg-4" key={i}>
              <div className="card h-100" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
                <div className="card-header d-flex justify-content-between" style={{background:'#0d1b30'}}>
                  <span className="fw-bold">{c.component}</span>
                  <span className={`badge bg-${statusBg(c.status)}`}>{c.status}</span>
                </div>
                <div className="card-body">
                  <div className="row text-center mb-2">
                    <div className="col-4">
                      <div className="small text-secondary">Events</div>
                      <div className="fw-bold" style={{color:'#00e5ff'}}>{c.total_events}</div>
                    </div>
                    <div className="col-4">
                      <div className="small text-secondary">Errors</div>
                      <div className="fw-bold" style={{color:'#ff5252'}}>{c.error_count}</div>
                    </div>
                    <div className="col-4">
                      <div className="small text-secondary">Err Rate</div>
                      <div className="fw-bold" style={{color: c.error_rate_pct > 5 ? '#ff5252' : '#4caf50'}}>{c.error_rate_pct}%</div>
                    </div>
                  </div>
                  <table className="table table-dark table-sm mb-0" style={{fontSize:'0.8rem'}}>
                    <tbody>
                      <tr><td>P50 Latency</td><td className="text-end">{c.p50_ms}ms</td></tr>
                      <tr><td>P95 Latency</td><td className="text-end">{c.p95_ms}ms</td></tr>
                      <tr><td>P99 Latency</td><td className="text-end">{c.p99_ms}ms</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
        </div>
      </>}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs && <>
        {/* Log Levels */}
        <div className="card mb-4" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
          <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Log Levels</div>
          <div className="card-body p-0">
            <table className="table table-dark table-sm mb-0">
              <thead><tr><th>Level</th><th>Description</th></tr></thead>
              <tbody>
                {defs.log_levels && Object.entries(defs.log_levels).map(([level, desc]) => (
                  <tr key={level}><td><span className={`badge bg-${lvlColor(level)}`}>{level}</span></td><td>{desc}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Trace Span Types */}
        <div className="card mb-4" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
          <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Trace Span Types</div>
          <div className="card-body p-0">
            <table className="table table-dark table-sm mb-0">
              <thead><tr><th>Span Type</th><th>Description</th></tr></thead>
              <tbody>
                {(defs.trace_span_types||[]).map((s,i) => (
                  <tr key={i}><td><code>{s.name}</code></td><td>{s.description}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Metric Thresholds */}
        <div className="card mb-4" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
          <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Metric Thresholds</div>
          <div className="card-body p-0">
            <table className="table table-dark table-sm mb-0">
              <thead><tr><th>Metric</th><th>Warning</th><th>Critical</th></tr></thead>
              <tbody>
                {defs.metric_thresholds && Object.entries(defs.metric_thresholds).map(([metric, t]) => (
                  <tr key={metric}><td>{metric}</td><td className="text-warning">{t.warning}</td><td className="text-danger">{t.critical}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Alert Rules */}
        <div className="card mb-4" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
          <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Alert Rules</div>
          <div className="card-body p-0">
            <table className="table table-dark table-sm mb-0">
              <thead><tr><th>Rule</th><th>Condition</th><th>Severity</th><th>Action</th></tr></thead>
              <tbody>
                {(defs.alert_rules||[]).map((r,i) => (
                  <tr key={i}><td>{r.name}</td><td><code>{r.condition}</code></td>
                    <td><span className={`badge bg-${r.severity==='critical'?'danger':'warning'}`}>{r.severity}</span></td>
                    <td>{r.action}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Instrumentation */}
        {defs.instrumentation && (
          <div className="card mb-4" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
            <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Instrumentation</div>
            <div className="card-body">
              <p><strong>Standard:</strong> {defs.instrumentation.standard}</p>
              <p><strong>Trace Propagation:</strong> {defs.instrumentation.trace_propagation}</p>
              <p><strong>Canonical Fields:</strong> <code>{(defs.instrumentation.canonical_fields||[]).join(', ')}</code></p>
              <h6 className="mt-3">Backends</h6>
              <table className="table table-dark table-sm mb-0">
                <tbody>
                  {defs.instrumentation.backends && Object.entries(defs.instrumentation.backends).map(([k,v]) => (
                    <tr key={k}><td className="fw-bold">{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Data Source */}
        {defs.data_source && (
          <div className="card mb-4" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
            <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Data Source</div>
            <div className="card-body">
              <p><strong>Table:</strong> <code>{defs.data_source.table}</code> in <code>{defs.data_source.database}</code></p>
              <p>{defs.data_source.description}</p>
            </div>
          </div>
        )}

        {/* Clinical Relevance */}
        {defs.clinical_relevance && (
          <div className="card mb-4" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
            <div className="card-header" style={{background:'#0d1b30',color:'#ff9800'}}>Clinical Relevance</div>
            <div className="card-body"><p>{defs.clinical_relevance}</p></div>
          </div>
        )}
      </>}
    </div>
  );
}
