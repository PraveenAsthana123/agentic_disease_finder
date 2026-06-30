'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TYPE_COLORS = { feature: 'primary', bugfix: 'danger', refactor: 'info', documentation: 'success', test: 'warning', infrastructure: 'secondary', performance: 'dark', style: 'light', other: 'secondary' };
const RISK_COLORS = { high: 'danger', medium: 'warning', low: 'success' };
const DEPLOY_COLORS = { autobuild: 'warning', deploy: 'success', git: 'info' };

export default function ChangeManagementPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [typeFilter, setTypeFilter] = useState('all');

  useEffect(() => {
    fetch(`${API}/api/change-management/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/change-management/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/change-management/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;
  if (!ov.available) return <div className="p-4 alert alert-warning">Change management data unavailable</div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'impact',      label: 'Impact & Trends' },
    { id: 'deploys',     label: 'Deploy Log' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const maxDaily = Math.max(...(ov.daily_change_counts || []).map(d => d.count), 1);

  return (
    <div>
      <h3>AI Change Management</h3>
      <p className="text-muted">Real change tracking from git history, track.jsonl, and uptime.jsonl</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Changes',     value: ov.total_changes,      color: 'primary' },
          { label: 'Total Deploys',     value: ov.total_deploys,      color: 'success' },
          { label: 'Deploy Success',    value: `${ov.deploy_success_rate}%`, color: ov.deploy_success_rate >= 90 ? 'success' : 'warning' },
          { label: 'Rollbacks',         value: ov.rollback_count,     color: 'danger' },
          { label: 'Last 24h',         value: ov.changes_last_24h,   color: 'info' },
          { label: 'Last 7d',          value: ov.changes_last_7d,    color: 'warning' },
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
          {/* Change Type Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Change Type Distribution</div>
              <div className="card-body">
                {(ov.change_type_distribution || []).map(s => (
                  <div key={s.type} className="d-flex justify-content-between align-items-center mb-2">
                    <span className="fw-semibold text-capitalize" style={{minWidth:120}}>{s.type}</span>
                    <div className="d-flex align-items-center" style={{width:'55%'}}>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className={`progress-bar bg-${TYPE_COLORS[s.type] || 'secondary'}`}
                             style={{width:`${s.percent}%`}} />
                      </div>
                      <span className="fw-bold">{s.count} ({s.percent}%)</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Risk Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Risk Distribution</div>
              <div className="card-body">
                {(ov.risk_distribution || []).map(r => (
                  <div key={r.level} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${RISK_COLORS[r.level] || 'secondary'} text-uppercase`} style={{minWidth:80}}>{r.level}</span>
                    <div className="d-flex align-items-center" style={{width:'60%'}}>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className={`progress-bar bg-${RISK_COLORS[r.level] || 'secondary'}`}
                             style={{width:`${r.percent}%`}} />
                      </div>
                      <span className="fw-bold">{r.count} ({r.percent}%)</span>
                    </div>
                  </div>
                ))}
                <div className="mt-3 small text-muted">
                  Avg {ov.avg_files_per_change} files / {ov.avg_lines_per_change} lines per change
                </div>
              </div>
            </div>
          </div>

          {/* Daily Change Counts */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Daily Changes (Last 14 Days)</div>
              <div className="card-body">
                {(ov.daily_change_counts || []).map(d => (
                  <div key={d.date} className="d-flex align-items-center mb-1">
                    <span className="text-muted small" style={{minWidth:85}}>{d.date.slice(5)}</span>
                    <div className="progress flex-grow-1 me-2" style={{height:'16px'}}>
                      <div className="progress-bar bg-primary" style={{width:`${d.count / maxDaily * 100}%`}} />
                    </div>
                    <span className="fw-bold small">{d.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Top Contributors */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Top Contributors</div>
              <div className="card-body">
                {(() => {
                  const maxC = Math.max(...(ov.top_contributors || []).map(c => c.commits), 1);
                  return (ov.top_contributors || []).map(c => (
                    <div key={c.author} className="d-flex align-items-center mb-2">
                      <span className="fw-semibold" style={{minWidth:140}}>{c.author}</span>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className="progress-bar bg-info" style={{width:`${c.commits / maxC * 100}%`}} />
                      </div>
                      <span className="fw-bold">{c.commits}</span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>

          {/* Recent Changes */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Recent Changes (Last 15)</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Hash</th><th>Date</th><th>Type</th><th>Risk</th><th>Files</th><th>Lines</th><th>Message</th></tr></thead>
                  <tbody>
                    {(ov.recent_changes || []).map((c, i) => (
                      <tr key={i}>
                        <td><code className="small">{c.hash}</code></td>
                        <td className="small">{(c.date || '').slice(0, 10)}</td>
                        <td><span className={`badge bg-${TYPE_COLORS[c.type] || 'secondary'}`}>{c.type}</span></td>
                        <td><span className={`badge bg-${RISK_COLORS[c.risk] || 'secondary'}`}>{c.risk}</span></td>
                        <td className="fw-bold">{c.files_changed}</td>
                        <td>{c.lines_changed}</td>
                        <td className="small" style={{maxWidth:300, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap'}}>{c.message}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Impact & Trends Tab ──────────────────────────────── */}
      {tab === 'impact' && bd && (
        <div className="row">
          {/* Impact by Type */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Impact by Change Type</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Type</th><th>Changes</th><th>Avg Files</th><th>Avg Lines</th><th>Total +/-</th></tr></thead>
                  <tbody>
                    {(bd.impact_by_type || []).map((t, i) => (
                      <tr key={i}>
                        <td><span className={`badge bg-${TYPE_COLORS[t.type] || 'secondary'}`}>{t.type}</span></td>
                        <td className="fw-bold">{t.changes}</td>
                        <td>{t.avg_files}</td>
                        <td>{t.avg_lines}</td>
                        <td className="small">
                          <span className="text-success">+{t.total_insertions}</span>{' / '}
                          <span className="text-danger">-{t.total_deletions}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Risk Trend */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Risk Trend (Last 14 Days)</div>
              <div className="card-body">
                {(() => {
                  const maxT = Math.max(...(bd.risk_trend || []).map(d => d.total), 1);
                  return (bd.risk_trend || []).map(d => (
                    <div key={d.date} className="d-flex align-items-center mb-1">
                      <span className="text-muted small" style={{minWidth:85}}>{d.date.slice(5)}</span>
                      <div className="progress flex-grow-1 me-2" style={{height:'16px'}}>
                        {d.total > 0 && <>
                          <div className="progress-bar bg-danger" style={{width:`${d.high / maxT * 100}%`}} />
                          <div className="progress-bar bg-warning" style={{width:`${d.medium / maxT * 100}%`}} />
                          <div className="progress-bar bg-success" style={{width:`${d.low / maxT * 100}%`}} />
                        </>}
                      </div>
                      <span className="small fw-bold">{d.total}</span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>

          {/* Hourly Heatmap */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Development Activity Heatmap (Last 7 Days)</div>
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
                          const maxV = Math.max(...bd.hourly_heatmap.matrix.flat(), 1);
                          const intensity = v / maxV;
                          const bg = v === 0 ? '#f8f9fa' : `rgba(13, 110, 253, ${0.15 + intensity * 0.85})`;
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

          {/* Change Velocity */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Change Velocity (Last 14 Days)</div>
              <div className="card-body">
                {(() => {
                  const maxCum = Math.max(...(bd.change_velocity || []).map(d => d.cumulative), 1);
                  return (bd.change_velocity || []).map(d => (
                    <div key={d.date} className="d-flex align-items-center mb-1">
                      <span className="text-muted small" style={{minWidth:85}}>{d.date.slice(5)}</span>
                      <div className="progress flex-grow-1 me-2" style={{height:'16px'}}>
                        <div className="progress-bar bg-primary" style={{width:`${d.cumulative / maxCum * 100}%`}} />
                      </div>
                      <span className="small">{d.daily}d / <span className="fw-bold">{d.cumulative}c</span></span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>

          {/* Rollback Events */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Rollback Events (Recent)</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>DOWN At</th><th>HTTP</th><th>Recovery</th></tr></thead>
                  <tbody>
                    {(bd.rollback_events || []).map((r, i) => (
                      <tr key={i}>
                        <td className="small">{r.ts}</td>
                        <td><code>{r.http}</code></td>
                        <td className="small">{r.recovery_ts || <span className="text-muted">pending</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Deploy Log Tab ────────────────────────────────────── */}
      {tab === 'deploys' && bd && (
        <div>
          {/* Level filter buttons */}
          <div className="mb-3">
            {['all', 'autobuild', 'deploy', 'git'].map(lv => (
              <button key={lv} className={`btn btn-sm me-1 ${typeFilter === lv ? 'btn-primary' : 'btn-outline-secondary'}`}
                      onClick={() => setTypeFilter(lv)}>
                {lv}
              </button>
            ))}
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Deploy Event Log (Recent 30)</div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0">
                <thead><tr><th>Timestamp</th><th>Level</th><th>Event</th><th>Host</th></tr></thead>
                <tbody>
                  {(bd.deploy_timeline || [])
                    .filter(e => typeFilter === 'all' || e.level === typeFilter)
                    .map((e, i) => (
                    <tr key={i}>
                      <td className="small">{e.ts}</td>
                      <td><span className={`badge bg-${DEPLOY_COLORS[e.level] || 'secondary'}`}>{e.level}</span></td>
                      <td>{e.event}</td>
                      <td className="text-muted small">{e.host}</td>
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
              <div className="card-header fw-bold">Change Management Stages</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th style={{width:'25%'}}>Stage</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.stages || []).map((s, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{s.stage}</td>
                        <td>{s.description}</td>
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
        </div>
      )}
    </div>
  );
}
