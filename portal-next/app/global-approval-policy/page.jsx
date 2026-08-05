'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const DECISION_COLOR = {
  allow: '#22c55e',
  require_human_approval: '#f97316',
  deny: '#ef4444',
};
const RISK_COLOR = {
  low: '#22c55e',
  medium: '#f97316',
  high: '#ef4444',
  critical: '#7f1d1d',
};
const ROLE_COLOR = {
  clinical_reviewer: '#3b82f6',
  data_steward: '#14b8a6',
  model_owner: '#8b5cf6',
  security_officer: '#ef4444',
  governance_lead: '#ec4899',
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: `${color}22`, color, border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600,
      textTransform: 'uppercase', whiteSpace: 'nowrap',
    }}>{text}</span>
  );
}

function DecisionBadge({ decision }) {
  const c = DECISION_COLOR[decision] || '#94a3b8';
  const label = decision === 'require_human_approval' ? 'HITL' : decision;
  return <Badge text={label} color={c} />;
}

function RiskBadge({ band }) {
  return <Badge text={band} color={RISK_COLOR[band] || '#94a3b8'} />;
}

function RoleBadge({ role }) {
  if (!role) return <span style={{ color: '#94a3b8' }}>—</span>;
  return <Badge text={role.replace(/_/g, ' ')} color={ROLE_COLOR[role] || '#94a3b8'} />;
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center', padding: '10px 16px', background: '#f8fafc', borderRadius: 8 }}>
      <div style={{ fontSize: 24, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

const TH = { padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600, color: '#64748b', borderBottom: '2px solid #e2e8f0' };
const TD = { padding: '8px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' };

export default function GlobalApprovalPolicyDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/global-approval-policy/overview`).then(r => r.json()),
      fetch(`${API}/api/global-approval-policy/breakdown`).then(r => r.json()),
      fetch(`${API}/api/global-approval-policy/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading global approval policy…</div>;

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'rules', label: 'Rules Table' },
    { id: 'roles', label: 'Role Scopes' },
    { id: 'risk', label: 'Risk Bands' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const sum = ov.summary || {};

  return (
    <div className="p-3">
      <h3 style={{ marginBottom: 4 }}>Global Approval Policy Dashboard</h3>
      <p className="text-muted" style={{ fontSize: 13, marginBottom: 16 }}>
        {ov.policy_name} — v{ov.version} — status: <strong>{ov.status}</strong> —{' '}
        {sum.total_rules} rules · {sum.total_roles} roles · {sum.total_risk_bands} risk bands · {sum.applies_to_domains} domains
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginBottom: 20 }}>
            <KPI label="Rules" value={sum.total_rules} />
            <KPI label="Roles" value={sum.total_roles} />
            <KPI label="Risk Bands" value={sum.total_risk_bands} />
            <KPI label="Decision Types" value={sum.total_decision_types} />
            <KPI label="Approvable Scopes" value={sum.total_approvable_scopes} />
            <KPI label="Audit Fields" value={sum.audit_required_fields} />
            <KPI label="Retention (days)" value={sum.retention_days} sub="~7 years" />
            <KPI label="Applies To Domains" value={sum.applies_to_domains} />
          </div>

          <div style={{ background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,.08)' }}>
            <h5 style={{ marginBottom: 12 }}>Decision Distribution</h5>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              {(ov.decision_distribution || []).map(d => (
                <div key={d.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <DecisionBadge decision={d.name} />
                  <span style={{ fontWeight: 700, fontSize: 20, color: '#1e293b' }}>{d.value}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,.08)' }}>
            <h5 style={{ marginBottom: 12 }}>Applies To Domains</h5>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {(ov.applies_to || []).map(d => (
                <span key={d} style={{
                  background: '#f1f5f9', color: '#475569', borderRadius: 12,
                  padding: '3px 10px', fontSize: 12,
                }}>{d.replace(/_/g, ' ')}</span>
              ))}
            </div>
          </div>

          <div style={{ background: '#fff', borderRadius: 8, padding: 16, boxShadow: '0 1px 3px rgba(0,0,0,.08)' }}>
            <h5 style={{ marginBottom: 12 }}>Default Decision</h5>
            <DecisionBadge decision={ov.default_decision} />
            <span style={{ marginLeft: 10, fontSize: 13, color: '#64748b' }}>Classification: {ov.classification?.replace(/_/g, ' ')}</span>
          </div>
        </div>
      )}

      {tab === 'rules' && (
        <div style={{ background: '#fff', borderRadius: 8, padding: 16, boxShadow: '0 1px 3px rgba(0,0,0,.08)', overflowX: 'auto' }}>
          <h5 style={{ marginBottom: 12 }}>Rules Table ({(ov.rules_table || []).length} rules)</h5>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Rule ID', 'Name', 'Decision', 'Required Role', 'Audit'].map(h => (
                  <th key={h} style={TH}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(ov.rules_table || []).map(r => (
                <tr key={r.rule_id} style={{ background: '#fff' }}
                  onMouseEnter={e => e.currentTarget.style.background = '#f8fafc'}
                  onMouseLeave={e => e.currentTarget.style.background = '#fff'}>
                  <td style={TD}><code style={{ fontSize: 12, color: '#7c3aed' }}>{r.rule_id}</code></td>
                  <td style={TD}>{r.name}</td>
                  <td style={TD}><DecisionBadge decision={r.decision} /></td>
                  <td style={TD}><RoleBadge role={r.required_role} /></td>
                  <td style={TD}>
                    <span style={{ color: r.required_audit ? '#22c55e' : '#94a3b8', fontWeight: 600, fontSize: 12 }}>
                      {r.required_audit ? 'Yes' : 'No'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {tab === 'roles' && bd && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16, marginBottom: 20 }}>
            {(bd.per_role || []).map(r => (
              <div key={r.role} style={{ background: '#fff', borderRadius: 8, padding: 16, boxShadow: '0 1px 3px rgba(0,0,0,.08)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <RoleBadge role={r.role} />
                  <span style={{ fontSize: 20, fontWeight: 700, color: '#1e293b' }}>{r.scope_count} scopes</span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {(r.can_approve || []).map(s => (
                    <span key={s} style={{
                      background: '#f1f5f9', color: '#475569', borderRadius: 10,
                      padding: '2px 8px', fontSize: 11,
                    }}>{s.replace(/_/g, ' ')}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div style={{ background: '#fff', borderRadius: 8, padding: 16, boxShadow: '0 1px 3px rgba(0,0,0,.08)', overflowX: 'auto' }}>
            <h5 style={{ marginBottom: 12 }}>Per Rule Detail</h5>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {['Rule', 'Name', 'Decision', 'Role', 'Risk', 'Domains'].map(h => (
                    <th key={h} style={TH}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.per_rule || []).map(r => (
                  <tr key={r.rule_id}
                    onMouseEnter={e => e.currentTarget.style.background = '#f8fafc'}
                    onMouseLeave={e => e.currentTarget.style.background = ''}>
                    <td style={TD}><code style={{ fontSize: 12, color: '#7c3aed' }}>{r.rule_id}</code></td>
                    <td style={TD}>{r.name}</td>
                    <td style={TD}><DecisionBadge decision={r.decision} /></td>
                    <td style={TD}><RoleBadge role={r.required_role} /></td>
                    <td style={TD}><RiskBadge band={r.risk_band || 'low'} /></td>
                    <td style={TD}>{(r.applies_to || []).map(d => d.replace(/_/g, ' ')).join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === 'risk' && bd && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16, marginBottom: 20 }}>
            {(bd.per_risk_band || []).map(b => (
              <div key={b.band} style={{ background: '#fff', borderRadius: 8, padding: 16, boxShadow: '0 1px 3px rgba(0,0,0,.08)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <RiskBadge band={b.band} />
                  <span style={{ fontSize: 18, fontWeight: 700, color: '#1e293b' }}>{b.rule_count} rules</span>
                </div>
                <div style={{ fontSize: 13, color: '#64748b', marginBottom: 6 }}>{b.description}</div>
                <div><DecisionBadge decision={b.default_decision} /></div>
              </div>
            ))}
          </div>

          <div style={{ background: '#fff', borderRadius: 8, padding: 16, boxShadow: '0 1px 3px rgba(0,0,0,.08)' }}>
            <h5 style={{ marginBottom: 12 }}>HITL Configuration</h5>
            {bd.hitl && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
                <KPI label="Timeout (hours)" value={bd.hitl.timeout_hours} />
                <KPI label="Escalation Role" value={bd.hitl.escalation_role?.replace(/_/g, ' ')} />
                <KPI label="Escalation After (hrs)" value={bd.hitl.escalation_after_hours} />
                <KPI label="Auto-Deny on Timeout" value={bd.hitl.auto_deny_on_timeout ? 'Yes' : 'No'} />
              </div>
            )}
          </div>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div>
          <div style={{ background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,.08)' }}>
            <h5 style={{ marginBottom: 12 }}>Decision Types</h5>
            {(defs.decision_types || []).map(d => (
              <div key={d.decision} style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 10 }}>
                <DecisionBadge decision={d.decision} />
                <span style={{ fontSize: 13, color: '#475569' }}>{d.description}</span>
              </div>
            ))}
          </div>

          <div style={{ background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,.08)' }}>
            <h5 style={{ marginBottom: 12 }}>Role Descriptions</h5>
            {(defs.role_descriptions || []).map(r => (
              <div key={r.role} style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 10 }}>
                <RoleBadge role={r.role} />
                <span style={{ fontSize: 13, color: '#475569' }}>{r.description}</span>
              </div>
            ))}
          </div>

          <div style={{ background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,.08)' }}>
            <h5 style={{ marginBottom: 12 }}>Risk Band Legend</h5>
            {(defs.risk_band_legend || []).map(b => (
              <div key={b.band} style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 10 }}>
                <RiskBadge band={b.band} />
                <span style={{ fontSize: 13, color: '#475569' }}>{b.description}</span>
              </div>
            ))}
          </div>

          <div style={{ background: '#fff', borderRadius: 8, padding: 16, boxShadow: '0 1px 3px rgba(0,0,0,.08)' }}>
            <h5 style={{ marginBottom: 12 }}>Glossary</h5>
            {(defs.glossary || []).map(g => (
              <div key={g.term} style={{ marginBottom: 10 }}>
                <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{g.term}:</span>{' '}
                <span style={{ fontSize: 13, color: '#475569' }}>{g.definition}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
