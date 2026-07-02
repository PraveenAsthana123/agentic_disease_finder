'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const RISK_COLOR = { high: 'danger', medium: 'warning', low: 'info' };

export default function AIGovernancePage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/ai-governance/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/ai-governance/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/ai-governance/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const s = overview.summary || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'decisions', label: 'Decision Audit' },
    { id: 'consultants', label: 'Consultant Matrix' },
    { id: 'health', label: 'Governance Health' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f3db;&#xfe0f; AI Governance Dashboard</h3>
      <p className="text-muted">Clinical AI governance from real clinical.db: decision audit trail, expert review panel, HITL oversight, consultant matrix, and use-case risk classification</p>

      {/* Summary cards */}
      <div className="row mb-3">
        {[
          { label: 'Clinical Decisions', value: s.total_decisions || 0, color: 'primary' },
          { label: 'Agreement Rate', value: `${s.agreement_rate || 0}%`, color: s.agreement_rate >= 80 ? 'success' : 'warning' },
          { label: 'Expert Reviews', value: s.expert_reviews || 0, color: 'info' },
          { label: 'Expert Consensus', value: `${s.expert_agreement_pct || 0}%`, color: s.expert_agreement_pct >= 66 ? 'success' : 'warning' },
          { label: 'HITL Reviews', value: s.hitl_reviews || 0, color: 'primary' },
          { label: 'Override Rate', value: `${s.override_rate || 0}%`, color: s.override_rate > 30 ? 'danger' : 'success' },
          { label: 'Avg Feedback', value: `${s.avg_feedback_rating || 0}/5`, color: s.avg_feedback_rating >= 4 ? 'success' : 'warning' },
          { label: 'Consultant Roles', value: s.consultant_roles || 0, color: 'info' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 col-lg mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h4 mb-0 text-${c.color}`}>{c.value}</div>
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
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Decision Audit Trail</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Patient</th><th>AI Prediction</th><th>Confidence</th><th>Agreement</th><th>Final</th></tr></thead>
                  <tbody>
                    {(overview.decision_trail || []).map(d => (
                      <tr key={d.id}>
                        <td><code>{d.patient_id}</code></td>
                        <td>{d.ai_prediction}</td>
                        <td>{d.confidence ? (d.confidence * 100).toFixed(0) + '%' : '-'}</td>
                        <td><span className={`badge bg-${d.agreement === 'Yes' ? 'success' : 'danger'}`}>{d.agreement || '-'}</span></td>
                        <td>{d.final_decision}</td>
                      </tr>
                    ))}
                    {(overview.decision_trail || []).length === 0 && <tr><td colSpan={5} className="text-muted text-center">No decisions recorded</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Expert Review Panel</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Patient</th><th>Role</th><th>Expert</th><th>Finding</th><th>Agrees</th></tr></thead>
                  <tbody>
                    {(overview.review_panel || []).map(r => (
                      <tr key={r.id}>
                        <td><code>{r.patient_id}</code></td>
                        <td>{r.role}</td>
                        <td>{r.expert}</td>
                        <td className="small">{(r.finding || '').substring(0, 50)}</td>
                        <td><span className={`badge bg-${r.agree_with_ai === 'agree' ? 'success' : 'danger'}`}>{r.agree_with_ai}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">HITL Oversight Log</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Patient</th><th>AI Prediction</th><th>Decision</th><th>Human Decision</th><th>Reason</th></tr></thead>
                  <tbody>
                    {(overview.hitl_detail || []).map(h => (
                      <tr key={h.id}>
                        <td><code>{h.patient_id}</code></td>
                        <td>{h.ai_prediction}</td>
                        <td><span className={`badge bg-${h.decision === 'accept' ? 'success' : 'warning'}`}>{h.decision}</span></td>
                        <td>{h.human_decision || '-'}</td>
                        <td><code>{h.reason_code || '-'}</code></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Governance Event Breakdown</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Event Type</th><th>Count</th></tr></thead>
                  <tbody>
                    {(overview.governance_event_breakdown || []).map(e => (
                      <tr key={e.action}>
                        <td>{e.action}</td>
                        <td><span className="badge bg-secondary">{e.cnt}</span></td>
                      </tr>
                    ))}
                    {(overview.governance_event_breakdown || []).length === 0 && <tr><td colSpan={2} className="text-muted text-center">No governance events</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Decision Audit tab */}
      {tab === 'decisions' && breakdown && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Role-Based Expert Breakdown</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Role</th><th>Reviews</th><th>Agreed</th><th>Agreement %</th></tr></thead>
                  <tbody>
                    {(breakdown.role_breakdown || []).map(r => (
                      <tr key={r.role}>
                        <td>{r.role}</td>
                        <td>{r.cnt}</td>
                        <td>{r.agreed}</td>
                        <td>{r.cnt > 0 ? ((r.agreed / r.cnt) * 100).toFixed(0) + '%' : '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Decision Confidence Distribution</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Confidence Band</th><th>Count</th></tr></thead>
                  <tbody>
                    {(breakdown.confidence_distribution || []).map(c => (
                      <tr key={c.band}>
                        <td>{c.band}</td>
                        <td><span className="badge bg-primary">{c.cnt}</span></td>
                      </tr>
                    ))}
                    {(breakdown.confidence_distribution || []).length === 0 && <tr><td colSpan={2} className="text-muted text-center">No decisions</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Use-Case Risk Register</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Role</th><th>Risk Class</th><th>Tier</th><th>Mandatory</th><th>Tasks</th><th>Compliance Docs</th></tr></thead>
                  <tbody>
                    {(breakdown.use_case_register || []).map(u => (
                      <tr key={u.role}>
                        <td>{u.role}</td>
                        <td><span className={`badge bg-${RISK_COLOR[u.risk_class] || 'secondary'}`}>{u.risk_class}</span></td>
                        <td>{u.tier}</td>
                        <td>{u.mandatory ? 'Yes' : 'No'}</td>
                        <td>{u.tasks}</td>
                        <td>{u.compliance_docs}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Consultant Matrix tab */}
      {tab === 'consultants' && breakdown && (
        <div>
          <div className="alert alert-info mb-3">
            Engagement Model: <strong>{breakdown.engagement_model || 'Advisory'}</strong> | Last Updated: <strong>{breakdown.matrix_updated || '-'}</strong>
          </div>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Consultant Engagement Matrix</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead><tr><th>ID</th><th>Name</th><th>Role</th><th>Tier</th><th>Mandatory</th><th>Objective</th><th>Tasks</th><th>Compliance Docs</th><th>Challenges</th></tr></thead>
                <tbody>
                  {(breakdown.consultant_matrix || []).map(c => (
                    <tr key={c.id}>
                      <td><code>{c.id}</code></td>
                      <td className="fw-bold">{c.name}</td>
                      <td>{c.role}</td>
                      <td><span className={`badge bg-${c.tier === 1 ? 'danger' : c.tier === 2 ? 'warning' : 'info'}`}>Tier {c.tier}</span></td>
                      <td>{c.mandatory ? <span className="badge bg-danger">Required</span> : <span className="badge bg-secondary">Optional</span>}</td>
                      <td className="small">{c.objective}</td>
                      <td>{c.task_count}</td>
                      <td>{c.compliance_doc_count}</td>
                      <td>{c.challenge_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Governance Health tab */}
      {tab === 'health' && breakdown && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Governance Health Scores</div>
              <div className="card-body">
                {Object.entries(breakdown.health_scores || {}).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <div className="d-flex justify-content-between">
                      <span className="small text-capitalize">{k.replace(/_/g, ' ')}</span>
                      <span className="fw-bold">{typeof v === 'number' && v <= 100 ? v + '%' : v}</span>
                    </div>
                    {typeof v === 'number' && v <= 100 && (
                      <div className="progress" style={{height: '8px'}}>
                        <div className={`progress-bar bg-${v >= 80 ? 'success' : v >= 50 ? 'warning' : 'danger'}`} style={{width: v + '%'}} />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Governance Event Timeline</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Date</th><th>Action</th><th>Count</th></tr></thead>
                  <tbody>
                    {(breakdown.governance_timeline || []).map((e, i) => (
                      <tr key={i}>
                        <td>{e.day}</td>
                        <td>{e.action}</td>
                        <td><span className="badge bg-secondary">{e.cnt}</span></td>
                      </tr>
                    ))}
                    {(breakdown.governance_timeline || []).length === 0 && <tr><td colSpan={3} className="text-muted text-center">No events</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          {(defs.sections || []).map(sec => (
            <div key={sec.title} className="card shadow-sm mb-3">
              <div className="card-header fw-bold">{sec.title}</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th style={{width:'25%'}}>Term</th><th>Definition</th></tr></thead>
                  <tbody>
                    {(sec.items || []).map(item => (
                      <tr key={item.term}>
                        <td className="fw-bold">{item.term}</td>
                        <td className="small">{item.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
