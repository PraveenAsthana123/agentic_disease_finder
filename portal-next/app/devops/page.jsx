'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TYPE_COLORS = { feat: 'success', fix: 'danger', refactor: 'warning', docs: 'info', chore: 'secondary', other: 'secondary' };

export default function DevOpsPage() {
  const [ov, setOv]     = useState(null);
  const [pl, setPl]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/devops/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/devops/pipelines`).then(r => r.json()).then(setPl).catch(() => {});
    fetch(`${API}/api/devops/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;
  if (!ov.available) return <div className="p-4 alert alert-warning">DevOps data unavailable</div>;

  const s = ov.summary || {};
  const tabs = [
    { id: 'overview',  label: 'Overview' },
    { id: 'velocity',  label: 'Commit Velocity' },
    { id: 'pipelines', label: 'Pipelines & Jobs' },
    { id: 'defs',      label: 'Definitions' },
  ];

  const maxCommits = Math.max(...(ov.commit_velocity || []).map(d => d.commits), 1);

  return (
    <div>
      <h3>&#x2699;&#xfe0f; DevOps / CI-CD Dashboard</h3>
      <p className="text-muted">Real git analytics, DORA metrics, and pipeline status — {ov.period_days}-day window</p>

      {/* DORA KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Deploy Freq/day',    value: s.deploy_freq_per_day, color: 'success', suffix: '' },
          { label: 'Deploys (90d)',       value: s.deploy_count_90d,    color: 'primary', suffix: '' },
          { label: 'Change Fail Rate',   value: `${s.change_fail_rate_pct}%`, color: s.change_fail_rate_pct > 10 ? 'danger' : 'success', suffix: '' },
          { label: 'MTTR (min)',          value: s.mttr_minutes,        color: 'info',    suffix: '' },
          { label: 'Avg Daily Commits',  value: s.avg_daily_commits,   color: 'warning', suffix: '' },
          { label: 'Files Changed 30d',  value: s.files_changed_30d,   color: 'secondary', suffix: '' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value}{c.suffix}</div>
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
          {/* Commit Type Breakdown */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Commit Type Breakdown (90d)</div>
              <div className="card-body">
                {(ov.commit_types || []).map(ct => {
                  const pct = Math.round(ct.count / Math.max(s.total_commits_90d, 1) * 100);
                  return (
                    <div key={ct.type} className="d-flex justify-content-between align-items-center mb-2">
                      <span className="fw-semibold" style={{minWidth: 90}}>{ct.type}</span>
                      <div className="d-flex align-items-center" style={{width: '65%'}}>
                        <div className="progress flex-grow-1 me-2" style={{height: '18px'}}>
                          <div
                            className={`progress-bar bg-${TYPE_COLORS[ct.type] || 'secondary'}`}
                            style={{width: `${pct}%`}}
                          />
                        </div>
                        <span className="fw-bold small">{ct.count} ({pct}%)</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Repo Info */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Repository Info</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <tbody>
                    <tr><td>Branch</td><td><code>{s.current_branch}</code></td></tr>
                    <tr><td>Total Branches</td><td>{s.branches}</td></tr>
                    <tr><td>Commits Ahead</td><td>{s.commits_ahead}</td></tr>
                    <tr><td>Total Commits (90d)</td><td>{s.total_commits_90d}</td></tr>
                    <tr><td>Top Author</td><td>{(ov.top_authors || [])[0]?.name || 'N/A'}</td></tr>
                    <tr><td>Generated At</td><td className="text-muted small">{ov.generated_at?.replace('T', ' ').slice(0,19)}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Recent Deploys */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Recent Deployments</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Hash</th><th>Date</th><th>Subject</th></tr></thead>
                  <tbody>
                    {(ov.recent_deploys || []).slice(0, 10).map(d => (
                      <tr key={d.hash}>
                        <td><code className="text-success">{d.hash}</code></td>
                        <td className="text-muted small">{d.date?.slice(0, 10)}</td>
                        <td className="small">{d.subject?.slice(0, 80)}{d.subject?.length > 80 ? '…' : ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Hot Files */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Most Changed Files (30d)</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>File</th><th>Changes</th></tr></thead>
                  <tbody>
                    {(ov.hottest_files || []).slice(0, 10).map(f => (
                      <tr key={f.file}>
                        <td className="small text-break">{f.file}</td>
                        <td><span className="badge bg-warning">{f.changes}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Top Authors */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Top Committers (90d)</div>
              <div className="card-body">
                {(ov.top_authors || []).map(a => (
                  <div key={a.name} className="d-flex justify-content-between align-items-center mb-2">
                    <span className="fw-semibold">{a.name}</span>
                    <span className="badge bg-primary">{a.commits} commits</span>
                  </div>
                ))}
                {!ov.top_authors?.length && <p className="text-muted">No author data</p>}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Commit Velocity Tab ──────────────────────────────── */}
      {tab === 'velocity' && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold">Daily Commit Velocity (Last 30 Days)</div>
          <div className="card-body">
            {(ov.commit_velocity || []).map(d => (
              <div key={d.date} className="d-flex align-items-center mb-1">
                <span className="text-muted small me-2" style={{minWidth: 85}}>{d.date.slice(5)}</span>
                <div className="progress flex-grow-1 me-2" style={{height: '18px'}}>
                  <div
                    className="progress-bar bg-primary"
                    style={{width: `${Math.round(d.commits / maxCommits * 100)}%`}}
                  />
                </div>
                <span className="small fw-bold" style={{minWidth: 40}}>{d.commits}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Pipelines Tab ───────────────────────────────────── */}
      {tab === 'pipelines' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Pipeline Summary</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <tbody>
                    <tr><td>Total Pipelines</td><td>{pl?.total_pipelines ?? 0}</td></tr>
                    <tr><td>Enabled</td><td>{pl?.enabled ?? 0}</td></tr>
                    <tr><td>Disabled</td><td>{pl?.disabled ?? 0}</td></tr>
                    <tr><td>Running Jobs</td><td>{pl?.running_jobs?.length ?? 0}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">System Health Snapshot</div>
              <div className="card-body">
                {pl?.health_snapshot ? (
                  <table className="table table-sm mb-0">
                    <tbody>
                      <tr><td>API Status</td><td><span className={`badge bg-${pl.health_snapshot.api_status === 'up' ? 'success' : 'secondary'}`}>{pl.health_snapshot.api_status}</span></td></tr>
                      <tr><td>DB Status</td><td><span className={`badge bg-${pl.health_snapshot.db_status === 'ok' ? 'success' : 'secondary'}`}>{pl.health_snapshot.db_status}</span></td></tr>
                      <tr><td>Endpoints OK</td><td>{pl.health_snapshot.endpoints_ok} / {pl.health_snapshot.endpoints_total}</td></tr>
                    </tbody>
                  </table>
                ) : <p className="text-muted">No snapshot data</p>}
              </div>
            </div>
          </div>
          {pl?.pipelines?.length > 0 && (
            <div className="col-12 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Pipelines</div>
                <div className="card-body p-0">
                  <table className="table table-sm table-hover mb-0">
                    <thead><tr><th>Name</th><th>Schedule</th><th>Status</th><th>Last Run</th></tr></thead>
                    <tbody>
                      {pl.pipelines.map(p => (
                        <tr key={p.name}>
                          <td>{p.name}</td>
                          <td><code>{p.schedule}</code></td>
                          <td><span className={`badge bg-${p.enabled ? 'success' : 'secondary'}`}>{p.enabled ? 'enabled' : 'disabled'}</span></td>
                          <td className="text-muted small">{p.last_run || 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Definitions Tab ─────────────────────────────────── */}
      {tab === 'defs' && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold">DORA Metric Definitions</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead><tr><th style={{width:'25%'}}>Metric</th><th>Definition</th><th>Source</th></tr></thead>
              <tbody>
                {(defs?.definitions || []).map(d => (
                  <tr key={d.term}>
                    <td className="fw-semibold">{d.term}</td>
                    <td className="small">{d.definition}</td>
                    <td className="text-muted small"><code>{d.source}</code></td>
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
