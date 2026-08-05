'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusColor = code => {
  if (!code) return '#9e9e9e';
  if (code < 300) return '#4caf50';
  if (code < 400) return '#ffc107';
  if (code < 500) return '#ff9800';
  return '#ff5252';
};

const latColor = ms => ms > 3000 ? '#ff5252' : ms > 1000 ? '#ffc107' : '#4caf50';

export default function TracesPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [filter, setFilter] = useState('');

  const load = () => {
    fetch(`${API}/api/traces/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/traces/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/traces/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  };

  useEffect(() => { load(); }, []);

  if (!ov) return (
    <div className="container py-4" style={{background:'#0b1120',minHeight:'100vh',color:'#e0e0e0'}}>
      <p>Loading HTTP Traces Dashboard...</p>
    </div>
  );

  const k = ov.kpis || {};
  const tabs = ['overview', 'trace table', 'definitions'];
  const traces = (bd?.traces || []).filter(t =>
    !filter || t.method_path?.toLowerCase().includes(filter.toLowerCase()) ||
    t.trace_id?.includes(filter)
  );

  return (
    <div className="container-fluid py-4" style={{background:'#0b1120',minHeight:'100vh',color:'#e0e0e0'}}>
      <div className="d-flex align-items-center justify-content-between mb-1">
        <h2 style={{color:'#00e5ff'}}>&#x1f4e1; HTTP Trace Dashboard</h2>
        <button className="btn btn-sm btn-outline-info" onClick={load}>&#x21bb; Refresh</button>
      </div>
      <p className="text-secondary mb-3">
        Every API response carries <code style={{color:'#00e5ff'}}>X-Trace-ID</code> + <code style={{color:'#00e5ff'}}>X-Latency-Ms</code> headers.
        Traces are stored in <code>transaction_log</code> (component=http-trace).
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {tabs.map(t => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link ${tab === t ? 'active text-white' : 'text-secondary'}`}
              style={tab === t ? {background:'#1a2744',borderColor:'#00e5ff'} : {background:'transparent'}}
              onClick={() => setTab(t)}
            >{t.charAt(0).toUpperCase() + t.slice(1)}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && <>
        {/* KPI Cards */}
        <div className="row g-3 mb-4">
          {[
            {label:'Total Requests', val: k.total_requests ?? '—', color:'#00e5ff'},
            {label:'Errors', val: k.error_count ?? '—', color:'#ff5252'},
            {label:'Error Rate', val: k.error_rate_pct != null ? `${k.error_rate_pct}%` : '—',
              color: (k.error_rate_pct||0) > 5 ? '#ff5252' : '#4caf50'},
            {label:'P50 Latency', val: k.p50_latency_ms != null ? `${k.p50_latency_ms}ms` : '—',
              color: latColor(k.p50_latency_ms||0)},
            {label:'P95 Latency', val: k.p95_latency_ms != null ? `${k.p95_latency_ms}ms` : '—',
              color: latColor(k.p95_latency_ms||0)},
            {label:'Trace Coverage', val: k.trace_coverage ?? '100%', color:'#4caf50'},
          ].map((c, i) => (
            <div className="col-md-2 col-sm-4" key={i}>
              <div className="card text-center h-100" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
                <div className="card-body py-3">
                  <div className="small text-secondary mb-1">{c.label}</div>
                  <div className="fs-4 fw-bold" style={{color:c.color}}>{c.val}</div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="row g-4 mb-4">
          {/* Top Paths */}
          <div className="col-md-6">
            <div className="card h-100" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
              <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Top Paths</div>
              <div className="card-body p-0">
                <table className="table table-dark table-sm mb-0">
                  <thead><tr><th>Path</th><th className="text-end">Requests</th></tr></thead>
                  <tbody>
                    {(ov.top_paths || []).map((p, i) => (
                      <tr key={i}>
                        <td><code style={{color:'#b0bec5',fontSize:'0.8rem'}}>{p.path}</code></td>
                        <td className="text-end fw-bold" style={{color:'#00e5ff'}}>{p.count}</td>
                      </tr>
                    ))}
                    {!(ov.top_paths?.length) && <tr><td colSpan={2} className="text-secondary text-center">No data yet — make a few API calls first</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Status Distribution */}
          <div className="col-md-3">
            <div className="card h-100" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
              <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>HTTP Status Distribution</div>
              <div className="card-body">
                {(ov.status_distribution || []).map((s, i) => (
                  <div key={i} className="d-flex justify-content-between align-items-center mb-2">
                    <span className="badge" style={{background: statusColor(parseInt(s.status)), fontSize:'0.85rem'}}>{s.status}</span>
                    <span className="fw-bold" style={{color:'#e0e0e0'}}>{s.count}</span>
                  </div>
                ))}
                {!(ov.status_distribution?.length) && <p className="text-secondary small">No data yet</p>}
              </div>
            </div>
          </div>

          {/* Recent Traces */}
          <div className="col-md-3">
            <div className="card h-100" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
              <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Recent Traces (last 5)</div>
              <div className="card-body p-2" style={{fontSize:'0.75rem'}}>
                {(ov.recent || []).slice(0, 5).map((r, i) => (
                  <div key={i} className="mb-2 p-2" style={{background:'#0d1b30',borderRadius:'4px'}}>
                    <div className="d-flex justify-content-between">
                      <span style={{color: statusColor(r.status_code)}}>{r.status_code}</span>
                      <span style={{color: latColor(r.latency_ms)}}>{r.latency_ms}ms</span>
                    </div>
                    <div style={{color:'#b0bec5',wordBreak:'break-all'}}>{r.method_path}</div>
                    <div style={{color:'#546e7a',fontFamily:'monospace'}}>{r.trace_id?.slice(0,18)}…</div>
                  </div>
                ))}
                {!(ov.recent?.length) && <p className="text-secondary small">No traces yet — make API calls to populate</p>}
              </div>
            </div>
          </div>
        </div>

        {/* Propagation Info */}
        <div className="card" style={{background:'#1a2744',border:'1px solid #00e5ff'}}>
          <div className="card-body">
            <h6 style={{color:'#00e5ff'}}>&#x1f517; Trace Propagation</h6>
            <p className="small text-secondary mb-1">
              Every response now includes: <code style={{color:'#00e5ff'}}>X-Trace-ID: &lt;uuid-v4&gt;</code> and <code style={{color:'#00e5ff'}}>X-Latency-Ms: &lt;ms&gt;</code>
            </p>
            <p className="small text-secondary mb-0">
              Logged to <code>transaction_log</code> with <code>component=http-trace</code>.
              Council agent steps additionally carry <code>request_id</code> + <code>tenant_id</code> per <code>council_orchestrator.py</code>.
            </p>
          </div>
        </div>
      </>}

      {/* ── Trace Table Tab ── */}
      {tab === 'trace table' && <>
        <div className="mb-3">
          <input
            className="form-control form-control-sm"
            style={{background:'#1a2744',color:'#e0e0e0',border:'1px solid #2a3a5c',maxWidth:'400px'}}
            placeholder="Filter by path or trace ID…"
            value={filter}
            onChange={e => setFilter(e.target.value)}
          />
        </div>
        <div className="card" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
          <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>
            HTTP Traces ({traces.length} shown)
          </div>
          <div className="card-body p-0" style={{overflowX:'auto'}}>
            <table className="table table-dark table-sm table-hover mb-0" style={{fontSize:'0.8rem'}}>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Trace ID</th>
                  <th>Method &amp; Path</th>
                  <th>Status</th>
                  <th>Latency</th>
                  <th>Timestamp (UTC)</th>
                </tr>
              </thead>
              <tbody>
                {traces.map((r, i) => (
                  <tr key={i}>
                    <td style={{color:'#546e7a'}}>{r.id}</td>
                    <td style={{fontFamily:'monospace',color:'#78909c',fontSize:'0.72rem'}}>{r.trace_id?.slice(0,18)}…</td>
                    <td><code style={{color:'#b0bec5',fontSize:'0.78rem'}}>{r.method_path}</code></td>
                    <td>
                      <span className="badge" style={{background: statusColor(r.status_code)}}>
                        {r.status_code}
                      </span>
                    </td>
                    <td style={{color: latColor(r.latency_ms), fontFamily:'monospace'}}>
                      {r.latency_ms}ms
                    </td>
                    <td style={{color:'#546e7a',fontSize:'0.72rem'}}>{r.ts_utc}</td>
                  </tr>
                ))}
                {!traces.length && (
                  <tr><td colSpan={6} className="text-center text-secondary py-4">
                    {filter ? 'No traces match filter.' : 'No traces yet — restart backend and make API calls to populate.'}
                  </td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </>}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-4">
          <div className="col-md-6">
            <div className="card h-100" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
              <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Trace Propagation</div>
              <div className="card-body" style={{fontSize:'0.88rem'}}>
                <p><strong style={{color:'#00e5ff'}}>Trace ID:</strong> {defs.trace_id}</p>
                <p><strong style={{color:'#00e5ff'}}>X-Trace-ID Header:</strong> {defs.x_trace_id_header}</p>
                <p><strong style={{color:'#00e5ff'}}>Storage:</strong> {defs.storage}</p>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card h-100" style={{background:'#1a2744',border:'1px solid #2a3a5c'}}>
              <div className="card-header" style={{background:'#0d1b30',color:'#00e5ff'}}>Latency SLAs &amp; Span Types</div>
              <div className="card-body" style={{fontSize:'0.88rem'}}>
                <p className="mb-2"><strong>Latency SLAs:</strong></p>
                <ul>
                  {Object.entries(defs.latency_slas || {}).map(([k, v]) => (
                    <li key={k}><code>{k.replace(/_/g,' ')}</code>: {v}ms</li>
                  ))}
                </ul>
                <p className="mt-2 mb-2"><strong>Span Types:</strong></p>
                <ul>
                  {(defs.span_types || []).map((s, i) => (
                    <li key={i}><code style={{color:'#00e5ff'}}>{s.name}</code> — {s.description}</li>
                  ))}
                </ul>
                <p><strong>Propagation Fields:</strong> {(defs.propagation_fields || []).join(', ')}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
