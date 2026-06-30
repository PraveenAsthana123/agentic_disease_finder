'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const RISK_COLORS = { low: 'success', medium: 'warning', high: 'danger' };
const LEVEL_COLORS = { health: 'info', ollama: 'secondary', ops: 'primary', system: 'info', watchdog: 'danger', git: 'success', autobuild: 'warning' };

export default function ShadowAIPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [srcFilter, setSrcFilter] = useState('all');

  useEffect(() => {
    fetch(`${API}/api/shadow-ai/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/shadow-ai/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/shadow-ai/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;
  if (!ov.available) return <div className="p-4 alert alert-warning">Shadow AI data unavailable</div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'analysis',   label: 'Detection Analysis' },
    { id: 'log',        label: 'Event Log' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const riskColor = RISK_COLORS[ov.risk_level] || 'secondary';
  const maxTimeline = Math.max(...(ov.detection_timeline || []).map(d => d.count), 1);
  const maxSource = Math.max(...(ov.top_shadow_sources || []).map(s => s.count), 1);

  return (
    <div>
      <h3>Shadow AI Detection</h3>
      <p className="text-muted">Monitoring for unauthorized/unregistered AI tool usage — real data from track.jsonl</p>

      {/* Risk Level Alert Banner */}
      <div className={`alert alert-${riskColor} d-flex align-items-center mb-3`}>
        <strong className="me-2">Risk Level:</strong>
        <span className={`badge bg-${riskColor} fs-6 me-2`}>{ov.risk_level?.toUpperCase()}</span>
        <span className="text-muted small">
          Shadow rate: {ov.shadow_rate}% &nbsp;|&nbsp;
          {ov.shadow_rate < 5 && 'Below 5% — minimal unauthorized AI activity detected.'}
          {ov.shadow_rate >= 5 && ov.shadow_rate < 15 && 'Between 5–15% — investigation recommended.'}
          {ov.shadow_rate >= 15 && 'Above 15% — immediate governance action required.'}
        </span>
      </div>

      {/* KPI Cards */}
      <div className="row mb-3">
        {[
          { label: 'Events Scanned',      value: ov.total_events_scanned,   color: 'primary' },
          { label: 'Registered Tools',    value: ov.registered_tools_count, color: 'success' },
          { label: 'Shadow Detections',   value: ov.shadow_detections,      color: 'danger' },
          { label: 'Shadow Rate',         value: `${ov.shadow_rate}%`,      color: riskColor },
          { label: 'Detected Last 24h',   value: ov.detections_last_24h,    color: 'warning' },
          { label: 'Detected Last 7d',    value: ov.detections_last_7d,     color: 'info' },
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
          {/* Detection Timeline */}
          <div className="col-md-7 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Shadow Detection Timeline (Last 14 Days)</div>
              <div className="card-body">
                {(ov.detection_timeline || []).map(d => (
                  <div key={d.date} className="d-flex align-items-center mb-1">
                    <span className="text-muted small" style={{minWidth:85}}>{d.date.slice(5)}</span>
                    <div className="progress flex-grow-1 me-2" style={{height:'18px'}}>
                      <div className={`progress-bar bg-${d.count > 0 ? 'danger' : 'secondary'}`}
                           style={{width:`${d.count / maxTimeline * 100}%`}} />
                    </div>
                    <span className="fw-bold small">{d.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Risk Distribution + Top Shadow Sources */}
          <div className="col-md-5 mb-3">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">Risk Distribution</div>
              <div className="card-body">
                {(ov.risk_distribution || []).map(r => (
                  <div key={r.category} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${RISK_COLORS[r.category] || 'secondary'}`} style={{minWidth:70}}>{r.category}</span>
                    <div className="progress flex-grow-1 mx-2" style={{height:'18px'}}>
                      <div className={`progress-bar bg-${RISK_COLORS[r.category] || 'secondary'}`}
                           style={{width:`${ov.shadow_detections ? r.count / ov.shadow_detections * 100 : 0}%`}} />
                    </div>
                    <span className="fw-bold small">{r.count}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="card shadow-sm">
              <div className="card-header fw-bold">Top Shadow Sources</div>
              <div className="card-body">
                {(ov.top_shadow_sources || []).length === 0 && (
                  <p className="text-success mb-0">No shadow sources detected — system is clean.</p>
                )}
                {(ov.top_shadow_sources || []).map(s => (
                  <div key={s.source} className="mb-2">
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small fw-semibold">{s.source}</span>
                      <span className="badge bg-danger">{s.count}</span>
                    </div>
                    <div className="progress" style={{height:'12px'}}>
                      <div className="progress-bar bg-danger" style={{width:`${s.count / maxSource * 100}%`}} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Detection Analysis Tab ────────────────────────────── */}
      {tab === 'analysis' && bd && (
        <div className="row">
          {/* Hourly Heatmap */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Shadow Detection Heatmap (Last 7 Days × 24h)</div>
              <div className="card-body" style={{overflowX:'auto'}}>
                <table className="table table-sm table-bordered text-center mb-0" style={{fontSize:'0.72rem'}}>
                  <thead>
                    <tr>
                      <th style={{minWidth:70}}></th>
                      {(bd.hourly_heatmap.hour_labels || []).map(h => <th key={h}>{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.hourly_heatmap.matrix || []).map((row, ri) => (
                      <tr key={ri}>
                        <td className="fw-bold text-nowrap">{(bd.hourly_heatmap.day_labels || [])[ri]}</td>
                        {row.map((v, ci) => {
                          const maxV = Math.max(...bd.hourly_heatmap.matrix.flat(), 1);
                          const intensity = v / maxV;
                          const bg = v === 0 ? '#f8f9fa' : `rgba(220, 53, 69, ${0.15 + intensity * 0.85})`;
                          const fg = intensity > 0.5 ? '#fff' : '#333';
                          return <td key={ci} style={{backgroundColor: bg, color: fg}}>{v || ''}</td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Temporal Pattern */}
          <div className="col-md-5 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Temporal Pattern (6-Hour Blocks)</div>
              <div className="card-body">
                {(() => {
                  const maxT = Math.max(...(bd.temporal_pattern || []).map(t => t.count), 1);
                  return (bd.temporal_pattern || []).map(t => (
                    <div key={t.block} className="d-flex align-items-center mb-2">
                      <span className="fw-semibold me-2" style={{minWidth:70}}>{t.block}</span>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className={`progress-bar bg-${t.count > 0 ? 'danger' : 'secondary'}`}
                             style={{width:`${t.count / maxT * 100}%`}} />
                      </div>
                      <span className="fw-bold small">{t.count}</span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>

          {/* Level Distribution */}
          <div className="col-md-7 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Shadow Events by Level</div>
              <div className="card-body">
                {(bd.level_distribution || []).length === 0 && (
                  <p className="text-success mb-0">No shadow events detected across any level.</p>
                )}
                {(() => {
                  const maxL = Math.max(...(bd.level_distribution || []).map(l => l.count), 1);
                  return (bd.level_distribution || []).map(l => (
                    <div key={l.level} className="d-flex align-items-center mb-2">
                      <span className={`badge bg-${LEVEL_COLORS[l.level] || 'secondary'} me-2`} style={{minWidth:80}}>{l.level}</span>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className="progress-bar bg-danger" style={{width:`${l.count / maxL * 100}%`}} />
                      </div>
                      <span className="fw-bold small">{l.count}</span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>

          {/* Source Analysis Table */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Source Analysis (All Levels)</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr>
                      <th>Source / Level</th>
                      <th>Status</th>
                      <th>Total Events</th>
                      <th>Shadow Events</th>
                      <th>Shadow %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.source_analysis || []).map((s, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{s.source}</td>
                        <td>
                          <span className={`badge bg-${s.registered ? 'success' : 'danger'}`}>
                            {s.registered ? 'Registered' : 'Unregistered'}
                          </span>
                        </td>
                        <td>{s.total_events}</td>
                        <td>
                          {s.shadow_events > 0
                            ? <span className="text-danger fw-bold">{s.shadow_events}</span>
                            : <span className="text-success">0</span>}
                        </td>
                        <td>
                          {s.total_events > 0
                            ? `${(s.shadow_events / s.total_events * 100).toFixed(1)}%`
                            : '0%'}
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

      {/* ── Event Log Tab ─────────────────────────────────────── */}
      {tab === 'log' && bd && (
        <div>
          <div className="mb-3 d-flex align-items-center gap-2 flex-wrap">
            <strong className="small">Filter by source:</strong>
            {['all', ...new Set((bd.recent_shadow_events || []).map(e => e.shadow_source))].map(src => (
              <button key={src} className={`btn btn-sm ${srcFilter === src ? 'btn-danger' : 'btn-outline-secondary'}`}
                      onClick={() => setSrcFilter(src)}>
                {src === 'all' ? 'All' : src}
              </button>
            ))}
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-bold">
              Recent Shadow Events ({(bd.recent_shadow_events || []).filter(e => srcFilter === 'all' || e.shadow_source === srcFilter).length} shown)
            </div>
            <div className="card-body p-0">
              {(bd.recent_shadow_events || []).length === 0 ? (
                <div className="p-3 text-success">
                  No shadow events detected — all activity matches the authorized tool registry.
                </div>
              ) : (
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr><th>Timestamp</th><th>Level</th><th>Shadow Source</th><th>Event</th><th>Host</th></tr>
                  </thead>
                  <tbody>
                    {(bd.recent_shadow_events || [])
                      .filter(e => srcFilter === 'all' || e.shadow_source === srcFilter)
                      .map((e, i) => (
                      <tr key={i}>
                        <td className="small text-nowrap">{e.ts}</td>
                        <td><span className={`badge bg-${LEVEL_COLORS[e.level] || 'secondary'}`}>{e.level}</span></td>
                        <td><span className="badge bg-danger">{e.shadow_source}</span></td>
                        <td className="small" style={{maxWidth:400, wordBreak:'break-word'}}>{e.event}</td>
                        <td className="text-muted small">{e.host}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ───────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold">Metric Definitions &amp; Detection Methodology</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead><tr><th style={{width:'28%'}}>Term</th><th>Definition</th></tr></thead>
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
      )}
    </div>
  );
}
