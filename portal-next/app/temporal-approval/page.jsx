'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLOR = {pending:'warning', approved:'success', rejected:'danger', active:'primary', complete:'success'};
const SLA_COLOR = {ok:'success', warning:'warning', breached:'danger'};
const RISK_COLOR = {low:'success', medium:'warning', high:'danger'};

export default function TemporalApprovalDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('queue');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/temporal-approval/overview`).then(r => r.json()),
      fetch(`${API}/api/temporal-approval/breakdown`).then(r => r.json()),
      fetch(`${API}/api/temporal-approval/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading Temporal Approval Workflow...</div>;

  const kpi = ov.kpis || {};
  const TABS = [
    {id:'queue', label:'Approval Queue'},
    {id:'pipeline', label:'Deployment Pipeline'},
    {id:'states', label:'Workflow States'},
    {id:'definitions', label:'Definitions'},
  ];

  const badge = (s, text) => (
    <span className={`badge bg-${STATUS_COLOR[s] || 'secondary'} me-1`}>{text || s}</span>
  );

  return (
    <div className="p-3">
      <h3>&#x231b; Temporal Approval Workflow</h3>
      <p className="text-muted small">
        AI Dark Factory Stage 9 — Durable human-in-the-loop approval gateway.
        Workflows pause here until an authorized reviewer approves, rejects, or escalates.
      </p>

      {/* KPIs */}
      <div className="row g-2 mb-3">
        {[
          {label:'Queue Depth', val:kpi.queue_depth, color:'primary'},
          {label:'Pending', val:kpi.pending, color:'warning'},
          {label:'Approved', val:kpi.approved, color:'success'},
          {label:'Rejected', val:kpi.rejected, color:'danger'},
          {label:'SLA Breached', val:kpi.sla_breached, color:'dark'},
          {label:'Avg Elapsed (h)', val:kpi.avg_elapsed_hours, color:'info'},
        ].map(k => (
          <div key={k.label} className="col-6 col-md-2">
            <div className={`card border-${k.color} text-center p-2`}>
              <div className={`fs-4 fw-bold text-${k.color}`}>{k.val}</div>
              <div className="small text-muted">{k.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Status distribution */}
      <div className="row g-2 mb-3">
        {(ov.status_distribution || []).map(s => (
          <div key={s.name} className="col-auto">
            <span className={`badge bg-${STATUS_COLOR[s.name.toLowerCase()] || 'secondary'} px-3 py-2`}>
              {s.name}: {s.value}
            </span>
          </div>
        ))}
        <div className="col-auto">|</div>
        {(ov.risk_distribution || []).map(r => (
          <div key={r.name} className="col-auto">
            <span className={`badge bg-${RISK_COLOR[r.name.toLowerCase()] || 'secondary'} px-3 py-2`}>
              Risk {r.name}: {r.value}
            </span>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* Approval Queue */}
      {tab === 'queue' && (
        <div>
          <h6>Approval Queue ({(ov.queue || []).length} items)</h6>
          <div className="table-responsive">
            <table className="table table-sm table-bordered small">
              <thead className="table-dark">
                <tr>
                  <th>ID</th>
                  <th>Workflow</th>
                  <th>Requested By</th>
                  <th>Stage</th>
                  <th>Status</th>
                  <th>Priority</th>
                  <th>Risk</th>
                  <th>Elapsed / SLA</th>
                  <th>Eval Score</th>
                  <th>Artifact</th>
                </tr>
              </thead>
              <tbody>
                {(ov.queue || []).map(w => (
                  <tr key={w.id}>
                    <td><code>{w.id}</code></td>
                    <td>{w.workflow}</td>
                    <td>{w.requested_by}</td>
                    <td>{w.stage}</td>
                    <td>{badge(w.status)}</td>
                    <td>
                      <span className={`badge bg-${w.priority === 'high' ? 'danger' : w.priority === 'medium' ? 'warning' : 'secondary'}`}>
                        {w.priority}
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${RISK_COLOR[w.risk] || 'secondary'}`}>{w.risk}</span>
                    </td>
                    <td>
                      {w.elapsed_hours}h / {w.sla_hours}h
                      {w.elapsed_hours >= w.sla_hours && (
                        <span className="badge bg-danger ms-1">BREACH</span>
                      )}
                    </td>
                    <td>{w.eval_score != null ? (w.eval_score * 100).toFixed(1) + '%' : '—'}</td>
                    <td><code className="small">{w.artifact}</code></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* SLA summary from breakdown */}
          {bd && bd.sla_summary && (
            <div className="row g-2 mt-2">
              <div className="col-auto">
                <span className="badge bg-success px-3 py-2">SLA OK: {bd.sla_summary.ok}</span>
              </div>
              <div className="col-auto">
                <span className="badge bg-warning text-dark px-3 py-2">Warning: {bd.sla_summary.warning}</span>
              </div>
              <div className="col-auto">
                <span className="badge bg-danger px-3 py-2">Breached: {bd.sla_summary.breached}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Deployment Pipeline */}
      {tab === 'pipeline' && (
        <div>
          <h6>AI Dark Factory — Full Deployment Pipeline (Stages 9–10)</h6>
          <p className="text-muted small">
            How an approved workflow moves from Temporal Gate → Harness CI/CD → OTel monitoring.
          </p>

          {/* Top-level pipeline flow */}
          <div className="d-flex flex-wrap gap-2 mb-4 align-items-center">
            {(ov.deployment_pipeline || []).map((s, i) => (
              <div key={s.n} className="d-flex align-items-center gap-2">
                <div className={`card border-${STATUS_COLOR[s.status] || 'secondary'} p-2 text-center`} style={{minWidth:110}}>
                  <div className="fs-5">{s.icon}</div>
                  <div className="fw-bold small">{s.name}</div>
                  <div className="text-muted" style={{fontSize:'0.7rem'}}>{s.tool}</div>
                  <span className={`badge bg-${STATUS_COLOR[s.status] || 'secondary'} mt-1`}>{s.status}</span>
                </div>
                {i < (ov.deployment_pipeline.length - 1) && <span className="text-muted fs-5">→</span>}
              </div>
            ))}
          </div>

          {/* Harness CI pipeline detail */}
          <h6>Harness CI/CD Pipeline Detail (Stage 10)</h6>
          <div className="table-responsive">
            <table className="table table-sm table-bordered small">
              <thead className="table-dark">
                <tr>
                  <th>Stage</th>
                  <th>Action</th>
                  <th>Tool</th>
                  <th>Status</th>
                  <th>Duration</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.harness_pipeline || []).map((s, i) => (
                  <tr key={i} className={s.status === 'active' ? 'table-info' : ''}>
                    <td><strong>{s.stage}</strong></td>
                    <td>{s.action}</td>
                    <td><code>{s.tool}</code></td>
                    <td>
                      <span className={`badge bg-${STATUS_COLOR[s.status] || 'secondary'}`}>
                        {s.status}
                      </span>
                    </td>
                    <td>{s.duration_s != null ? `${s.duration_s}s` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Workflow States */}
      {tab === 'states' && (
        <div>
          <h6>Temporal Workflow State Machine</h6>
          <p className="text-muted small">
            All possible states a workflow can be in during the approval lifecycle.
          </p>
          <div className="row g-2">
            {(ov.workflow_states || []).map(s => (
              <div key={s.state} className="col-md-4">
                <div className={`card border-${s.color} p-2`}>
                  <div className="fw-bold">
                    <span className={`badge bg-${s.color} me-2`}>{s.state}</span>
                  </div>
                  <div className="small text-muted mt-1">{s.desc}</div>
                </div>
              </div>
            ))}
          </div>

          <h6 className="mt-4">SLA Policy</h6>
          {defs && defs.sla_policy && (
            <table className="table table-sm small">
              <thead className="table-dark"><tr><th>Priority</th><th>SLA</th></tr></thead>
              <tbody>
                <tr><td>High</td><td>{defs.sla_policy.high_priority_hours}h</td></tr>
                <tr><td>Medium</td><td>{defs.sla_policy.medium_priority_hours}h</td></tr>
                <tr><td>Low</td><td>{defs.sla_policy.low_priority_hours}h</td></tr>
                <tr className="table-warning"><td colSpan={2}>{defs.sla_policy.escalation}</td></tr>
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Definitions */}
      {tab === 'definitions' && defs && (
        <div>
          <h6>Glossary</h6>
          <table className="table table-sm table-bordered small">
            <thead className="table-dark"><tr><th>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {(defs.glossary || []).map(g => (
                <tr key={g.term}>
                  <td><strong>{g.term}</strong></td>
                  <td>{g.def}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="alert alert-info small mt-3">
            <strong>Integration Note:</strong> {defs.integration_note}
          </div>

          <h6>References</h6>
          <ul className="small">
            {(defs.references || []).map(r => (
              <li key={r.name}>
                <strong>{r.name}</strong>
                {r.url && !r.url.startsWith('http') && (
                  <> — <a href={r.url}>{r.url}</a></>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
