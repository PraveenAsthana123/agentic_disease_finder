'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevBadge = s => s === 'critical' ? 'danger' : s === 'high' ? 'warning' : s === 'medium' ? 'info' : 'secondary';
const riskBadge = r => r === 'high' ? 'danger' : r === 'medium' ? 'warning' : r === 'elevated' ? 'warning' : 'success';
const expBadge = e => e === 'high' ? 'danger' : e === 'medium' ? 'warning' : 'success';
const owaspBadge = s => s === 'monitored' ? 'info' : s === 'mitigated' ? 'success' : 'secondary';

export default function SecOpsPage() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/sec-ops/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/sec-ops/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/sec-ops/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-danger" /></div>;

  const k = ov.kpis || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'threats', label: 'Threats & Scanning' },
    { id: 'access', label: 'Access Audit' },
    { id: 'owasp', label: 'OWASP & Compliance' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#c9d1d9', padding: '1.5rem' }}>
      <h3 style={{ color: '#ff7b72' }}>SecOps Dashboard</h3>
      <p className="text-muted">Threat detection, injection/jailbreak scanning, PII protection, access audit, OWASP LLM Top-10 coverage</p>

      {/* KPI Cards */}
      <div className="row mb-3">
        {[
          { label: 'Transactions', value: k.total_transactions, color: '#58a6ff' },
          { label: 'Conversations', value: k.total_conversations, color: '#58a6ff' },
          { label: 'Guardrail Events', value: k.guardrail_events, color: '#f0883e' },
          { label: 'Blocked', value: k.blocked_events, color: '#ff7b72' },
          { label: 'Threats Detected', value: k.total_threats_detected, color: k.total_threats_detected > 0 ? '#ff7b72' : '#3fb950' },
          { label: 'Critical Threats', value: k.critical_threats, color: k.critical_threats > 0 ? '#ff7b72' : '#3fb950' },
          { label: 'PII Patterns', value: k.pii_patterns_active, color: '#d2a8ff' },
          { label: 'Injection Patterns', value: k.injection_patterns_active, color: '#d2a8ff' },
          { label: 'Oversight Rate', value: `${k.oversight_rate_pct}%`, color: '#3fb950' },
          { label: 'Compliance Score', value: k.compliance_score, color: k.compliance_score >= 75 ? '#3fb950' : k.compliance_score >= 50 ? '#f0883e' : '#ff7b72' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-body py-2 text-center">
                <div className="h4 mb-0" style={{ color: c.color }}>{c.value}</div>
                <div className="small text-muted">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3" style={{ borderColor: '#30363d' }}>
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`}
              style={tab === t.id ? { background: '#161b22', color: '#ff7b72', borderColor: '#30363d' } : { color: '#8b949e', background: 'transparent', border: 'none' }}
              onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Threat Severity Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#ff7b72' }}>Threat Severity Distribution</div>
              <div className="card-body">
                {Object.keys(ov.threat_severity || {}).length === 0
                  ? <p className="text-muted mb-0">No threats detected in conversation logs</p>
                  : Object.entries(ov.threat_severity).map(([sev, count]) => (
                    <div key={sev} className="d-flex justify-content-between mb-2">
                      <span className={`badge bg-${sevBadge(sev)}`}>{sev}</span>
                      <span className="fw-bold">{count}</span>
                    </div>
                  ))}
              </div>
            </div>
          </div>

          {/* Threat Categories */}
          <div className="col-md-4 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#f0883e' }}>Threat Categories</div>
              <div className="card-body">
                {Object.keys(ov.threat_categories || {}).length === 0
                  ? <p className="text-muted mb-0">No threat categories triggered</p>
                  : Object.entries(ov.threat_categories).map(([cat, count]) => (
                    <div key={cat} className="d-flex justify-content-between mb-2">
                      <span className="small">{cat.replace(/_/g, ' ')}</span>
                      <span className="badge bg-secondary">{count}</span>
                    </div>
                  ))}
              </div>
            </div>
          </div>

          {/* Guardrail Enforcement Log */}
          <div className="col-md-4 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#3fb950' }}>Guardrail Enforcement</div>
              <div className="card-body" style={{ maxHeight: '300px', overflow: 'auto' }}>
                {(ov.guardrail_log || []).length === 0
                  ? <p className="text-muted mb-0">No guardrail events</p>
                  : (ov.guardrail_log || []).map((e, i) => (
                    <div key={i} className="border-bottom pb-2 mb-2" style={{ borderColor: '#30363d !important' }}>
                      <div className="d-flex justify-content-between">
                        <span className={`badge bg-${e.action === 'blocked' ? 'danger' : e.action === 'sign-off' ? 'success' : 'warning'}`}>{e.action}</span>
                        <span className="small text-muted">{e.actor}</span>
                      </div>
                      <div className="small text-muted mt-1">{e.component} {e.patient_id ? `| ${e.patient_id}` : ''}</div>
                    </div>
                  ))}
              </div>
            </div>
          </div>

          {/* Security Agent Events */}
          <div className="col-12 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#58a6ff' }}>Security Agent Activity</div>
              <div className="card-body">
                {(ov.security_agent_events || []).length === 0
                  ? <p className="text-muted mb-0">No security agent events recorded</p>
                  : <div className="table-responsive"><table className="table table-sm table-dark table-striped mb-0">
                    <thead><tr><th>Actor</th><th>Action</th><th>Component</th><th>Detail</th><th>Timestamp</th></tr></thead>
                    <tbody>{(ov.security_agent_events || []).map((e, i) => (
                      <tr key={i}><td>{e.actor}</td><td><span className={`badge bg-${e.action === 'blocked' ? 'danger' : 'info'}`}>{e.action}</span></td><td>{e.component}</td><td className="small">{(e.detail || '').slice(0, 80)}</td><td className="small text-muted">{e.timestamp}</td></tr>
                    ))}</tbody>
                  </table></div>}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Threats & Scanning Tab ──────────────────────────── */}
      {tab === 'threats' && (
        <div className="row">
          {/* PII Pattern Inventory */}
          <div className="col-md-6 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#d2a8ff' }}>PII Detection Patterns ({k.pii_patterns_active} active)</div>
              <div className="card-body">
                <div className="table-responsive"><table className="table table-sm table-dark mb-0">
                  <thead><tr><th>Pattern</th><th>Severity</th><th>Regulation</th><th>Detections</th></tr></thead>
                  <tbody>{(ov.pii_inventory || []).map((p, i) => (
                    <tr key={i}><td>{p.name}</td><td><span className={`badge bg-${sevBadge(p.severity)}`}>{p.severity}</span></td><td className="small">{p.regulation}</td><td>{p.detections}</td></tr>
                  ))}</tbody>
                </table></div>
              </div>
            </div>
          </div>

          {/* Injection Pattern Inventory */}
          <div className="col-md-6 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#ff7b72' }}>Injection Detection Patterns ({k.injection_patterns_active} active)</div>
              <div className="card-body">
                <div className="table-responsive"><table className="table table-sm table-dark mb-0">
                  <thead><tr><th>Category</th><th>Severity</th><th>Detections</th></tr></thead>
                  <tbody>{(ov.injection_inventory || []).map((p, i) => (
                    <tr key={i}><td>{p.pattern_label.replace(/_/g, ' ')}</td><td><span className={`badge bg-${sevBadge(p.severity)}`}>{p.severity}</span></td><td>{p.detections}</td></tr>
                  ))}</tbody>
                </table></div>
              </div>
            </div>
          </div>

          {/* Conversation Role Analysis */}
          {bd && (
            <div className="col-md-6 mb-3">
              <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
                <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#58a6ff' }}>Conversation Role Distribution</div>
                <div className="card-body">
                  <p className="small text-muted mb-2">Abnormal role ratios may indicate injection or manipulation attempts</p>
                  {(bd.conversation_roles || []).map((r, i) => {
                    const total = (bd.conversation_roles || []).reduce((s, x) => s + x.count, 0);
                    const pct = total ? Math.round(r.count / total * 100) : 0;
                    return (
                      <div key={i} className="mb-2">
                        <div className="d-flex justify-content-between small">
                          <span>{r.role}</span>
                          <span>{r.count} ({pct}%)</span>
                        </div>
                        <div className="progress" style={{ height: '8px', background: '#30363d' }}>
                          <div className="progress-bar bg-info" style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Action Distribution */}
          {bd && (
            <div className="col-md-6 mb-3">
              <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
                <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#f0883e' }}>Transaction Action Distribution</div>
                <div className="card-body" style={{ maxHeight: '350px', overflow: 'auto' }}>
                  {(bd.action_distribution || []).map((a, i) => {
                    const max = Math.max(...(bd.action_distribution || []).map(x => x.count), 1);
                    const pct = Math.round(a.count / max * 100);
                    const isPriv = ['delete', 'update', 'human_decision', 'sign-off', 'blocked'].includes(a.action);
                    return (
                      <div key={i} className="mb-2">
                        <div className="d-flex justify-content-between small">
                          <span>{isPriv ? '⚠️ ' : ''}{a.action}</span>
                          <span>{a.count}</span>
                        </div>
                        <div className="progress" style={{ height: '6px', background: '#30363d' }}>
                          <div className={`progress-bar ${isPriv ? 'bg-warning' : 'bg-secondary'}`} style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Access Audit Tab ────────────────────────────────── */}
      {tab === 'access' && bd && (
        <div className="row">
          {/* Patient Access Audit */}
          <div className="col-md-6 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#58a6ff' }}>Patient Data Access Audit</div>
              <div className="card-body">
                <div className="table-responsive"><table className="table table-sm table-dark table-striped mb-0">
                  <thead><tr><th>Patient</th><th>Actors</th><th>Components</th><th>Events</th><th>Risk</th></tr></thead>
                  <tbody>{(bd.patient_access_audit || []).map((p, i) => (
                    <tr key={i}><td>{p.patient_id}</td><td>{p.distinct_actors}</td><td>{p.distinct_components}</td><td>{p.total_events}</td><td><span className={`badge bg-${riskBadge(p.risk)}`}>{p.risk}</span></td></tr>
                  ))}</tbody>
                </table></div>
              </div>
            </div>
          </div>

          {/* Actor Privilege Matrix */}
          <div className="col-md-6 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#f0883e' }}>Actor Privilege Matrix</div>
              <div className="card-body">
                <div className="table-responsive"><table className="table table-sm table-dark table-striped mb-0">
                  <thead><tr><th>Actor</th><th>Txns</th><th>Comps</th><th>Patients</th><th>Priv. Actions</th><th>Risk</th></tr></thead>
                  <tbody>{(bd.actor_privilege_matrix || []).map((a, i) => (
                    <tr key={i}><td>{a.actor}</td><td>{a.transactions}</td><td>{a.components}</td><td>{a.patients_accessed}</td><td className="small">{(a.privileged_actions || []).join(', ') || '-'}</td><td><span className={`badge bg-${riskBadge(a.risk_level)}`}>{a.risk_level}</span></td></tr>
                  ))}</tbody>
                </table></div>
              </div>
            </div>
          </div>

          {/* Component Attack Surface */}
          <div className="col-md-6 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#d2a8ff' }}>Component Attack Surface</div>
              <div className="card-body">
                <div className="table-responsive"><table className="table table-sm table-dark table-striped mb-0">
                  <thead><tr><th>Component</th><th>Actors</th><th>Txns</th><th>Actions</th><th>Exposure</th></tr></thead>
                  <tbody>{(bd.attack_surface || []).map((c, i) => (
                    <tr key={i}><td>{c.component}</td><td>{c.distinct_actors}</td><td>{c.transactions}</td><td>{c.distinct_actions}</td><td><span className={`badge bg-${expBadge(c.exposure)}`}>{c.exposure}</span></td></tr>
                  ))}</tbody>
                </table></div>
              </div>
            </div>
          </div>

          {/* Incident Timeline */}
          <div className="col-md-6 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#ff7b72' }}>Incident Timeline (Recent Privileged Events)</div>
              <div className="card-body" style={{ maxHeight: '400px', overflow: 'auto' }}>
                {(bd.incident_timeline || []).map((e, i) => (
                  <div key={i} className="border-bottom pb-2 mb-2" style={{ borderColor: '#30363d' }}>
                    <div className="d-flex justify-content-between">
                      <span className={`badge bg-${e.action === 'blocked' ? 'danger' : e.action === 'delete' ? 'danger' : e.action === 'sign-off' ? 'success' : 'warning'}`}>{e.action}</span>
                      <span className="small text-muted">{e.timestamp}</span>
                    </div>
                    <div className="small mt-1"><strong>{e.actor}</strong> on <em>{e.component}</em>{e.patient_id ? ` | ${e.patient_id}` : ''}</div>
                    {e.detail && <div className="small text-muted mt-1">{(e.detail || '').slice(0, 100)}</div>}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Daily Security Trend */}
          <div className="col-12 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#3fb950' }}>Daily Security Event Trend</div>
              <div className="card-body">
                <div className="table-responsive"><table className="table table-sm table-dark mb-0">
                  <thead><tr><th>Date</th><th>Total</th><th>Blocked</th><th>Sign-offs</th><th>Mutations</th></tr></thead>
                  <tbody>{(bd.daily_security || []).map((d, i) => (
                    <tr key={i}><td>{d.date}</td><td>{d.total}</td><td className={d.blocked > 0 ? 'text-danger' : ''}>{d.blocked}</td><td className={d.signoffs > 0 ? 'text-success' : ''}>{d.signoffs}</td><td className={d.mutations > 0 ? 'text-warning' : ''}>{d.mutations}</td></tr>
                  ))}</tbody>
                </table></div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── OWASP & Compliance Tab ──────────────────────────── */}
      {tab === 'owasp' && bd && (
        <div className="row">
          <div className="col-md-8 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#ff7b72' }}>OWASP LLM Top-10 Coverage</div>
              <div className="card-body">
                <div className="table-responsive"><table className="table table-sm table-dark mb-0">
                  <thead><tr><th>ID</th><th>Vulnerability</th><th>Status</th><th>Controls</th></tr></thead>
                  <tbody>{(bd.owasp_coverage || []).map((o, i) => (
                    <tr key={i}><td className="fw-bold">{o.id}</td><td>{o.name}</td><td><span className={`badge bg-${owaspBadge(o.status)}`}>{o.status}</span></td><td className="small">{o.controls}</td></tr>
                  ))}</tbody>
                </table></div>
              </div>
            </div>
          </div>
          <div className="col-md-4 mb-3">
            <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
              <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#3fb950' }}>Compliance Posture</div>
              <div className="card-body text-center">
                <div className="display-3 mb-2" style={{ color: k.compliance_score >= 75 ? '#3fb950' : k.compliance_score >= 50 ? '#f0883e' : '#ff7b72' }}>{k.compliance_score}</div>
                <p className="text-muted">Compliance Score / 100</p>
                <hr style={{ borderColor: '#30363d' }} />
                <div className="text-start">
                  <div className="d-flex justify-content-between mb-1 small"><span>PII Coverage</span><span>{k.pii_patterns_active}/6 patterns</span></div>
                  <div className="d-flex justify-content-between mb-1 small"><span>Injection Coverage</span><span>{k.injection_patterns_active}/10 patterns</span></div>
                  <div className="d-flex justify-content-between mb-1 small"><span>Oversight Rate</span><span>{k.oversight_rate_pct}%</span></div>
                  <div className="d-flex justify-content-between mb-1 small"><span>Active Enforcement</span><span>{k.blocked_events > 0 ? 'Yes' : 'No'}</span></div>
                  <div className="d-flex justify-content-between mb-1 small"><span>Privileged Actors</span><span>{k.privileged_actors}/{k.total_actors}</span></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ─────────────────────────────────── */}
      {tab === 'definitions' && defs && defs.sections && (
        <div className="row">
          {defs.sections.map((s, i) => (
            <div key={i} className="col-md-6 mb-3">
              <div className="card border-0 shadow-sm" style={{ background: '#161b22' }}>
                <div className="card-header fw-bold" style={{ background: '#1c2128', color: '#58a6ff' }}>{s.title}</div>
                <div className="card-body">
                  {s.items.map((d, j) => (
                    <div key={j} className="mb-3">
                      <div className="fw-bold small" style={{ color: '#e6edf3' }}>{d.term}</div>
                      <div className="small text-muted">{d.definition}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
