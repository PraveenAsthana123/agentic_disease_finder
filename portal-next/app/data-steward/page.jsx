'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: '🛡️ Overview' },
  { id: 'phi',         label: '🔐 PHI & De-ID' },
  { id: 'access',      label: '🔑 Access & Audit' },
  { id: 'incidents',   label: '🚨 Incidents' },
  { id: 'datasets',    label: '🗄️ Datasets' },
  { id: 'sharing',     label: '🔗 Data Sharing' },
  { id: 'definitions', label: '📖 Definitions' },
];

const RISK_COLORS = { critical: '#ef4444', high: '#f97316', medium: '#f59e0b', low: '#22c55e' };
function riskColor(level) { return RISK_COLORS[level] || '#6b7280'; }

function KPI({ label, value, color = '#3b82f6', sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h4 mb-0 fw-bold" style={{ color }}>{value ?? '—'}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function RiskBadge({ level }) {
  return (
    <span className="badge" style={{ backgroundColor: riskColor(level), color: '#fff' }}>{level}</span>
  );
}

function Bar({ pct, color = '#3b82f6', label }) {
  return (
    <div className="mb-1">
      {label && <div className="text-muted small mb-0">{label}</div>}
      <div style={{ background: '#e5e7eb', borderRadius: 4, height: 12 }}>
        <div style={{ width: `${Math.min(100, pct || 0)}%`, background: color, borderRadius: 4, height: 12 }} />
      </div>
    </div>
  );
}

/* ── OVERVIEW TAB ───────────────────────────────────────────── */
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  const k = ov.kpis || {};
  const charts = ov.charts || {};
  const riskScore = k.privacy_risk_score ?? 0;
  const riskLevel = riskScore >= 75 ? 'critical' : riskScore >= 50 ? 'high' : riskScore >= 25 ? 'medium' : 'low';

  return (
    <>
      <div className="row">
        <KPI label="Total Datasets" value={k.total_datasets} color="#3b82f6" />
        <KPI label="Patients w/ PHI" value={k.total_patients} color="#8b5cf6" />
        <KPI label="PHI Fields Detected" value={k.phi_fields_detected} color="#ef4444" />
        <KPI label="De-Identified %" value={`${k.deidentified_pct?.toFixed(1)}%`} color={k.deidentified_pct > 0 ? '#22c55e' : '#ef4444'} sub="target: 100%" />
        <KPI label="Unique Actors" value={k.unique_actors} color="#f59e0b" />
        <KPI label="Audit Events" value={k.audit_events?.toLocaleString()} color="#06b6d4" />
        <KPI label="Data Sharing Actions" value={k.data_sharing_actions} color="#a3e635" />
        <KPI label="Privacy Risk Score" value={`${riskScore}/100`} color={riskColor(riskLevel)} sub={riskLevel.toUpperCase()} />
      </div>

      <div className="row mt-2">
        {/* PHI Exposure */}
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">PHI Exposure by Field</div>
            <div className="card-body">
              {(charts.phi_exposure_by_field || []).map(f => (
                <div key={f.field} className="mb-2">
                  <div className="d-flex justify-content-between small">
                    <span className="fw-semibold">{f.field}</span>
                    <span className="text-danger">{f.records_exposed} records</span>
                  </div>
                  <Bar pct={(f.records_exposed / Math.max(k.total_patients, 1)) * 100} color="#ef4444" />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Risk Distribution */}
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">Risk Factor Distribution</div>
            <div className="card-body">
              {(ov.risk_factors || []).map(rf => (
                <div key={rf.factor} className="mb-2">
                  <div className="d-flex justify-content-between small">
                    <span className="fw-semibold">{rf.factor}</span>
                    <span>{rf.score}/{rf.max}</span>
                  </div>
                  <Bar pct={(rf.score / rf.max) * 100} color={rf.score >= rf.max ? '#ef4444' : rf.score >= rf.max * 0.5 ? '#f59e0b' : '#22c55e'} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Access by actor */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">Access Events by Actor</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light"><tr><th>Actor</th><th>Events</th><th>Share</th></tr></thead>
            <tbody>
              {(charts.access_by_actor || []).map(a => (
                <tr key={a.actor}>
                  <td className="fw-semibold">{a.actor}</td>
                  <td>{a.count.toLocaleString()}</td>
                  <td style={{ width: 120 }}>
                    <Bar pct={(a.count / Math.max(k.audit_events, 1)) * 100} color="#3b82f6" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Actions timeline */}
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">Audit Actions Timeline</div>
        <div className="card-body">
          <div className="row">
            {(charts.actions_timeline || []).map(t => (
              <div key={t.month} className="col-auto text-center me-2">
                <div className="fw-bold" style={{ color: '#3b82f6' }}>{t.count.toLocaleString()}</div>
                <div className="text-muted small">{t.month}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

/* ── PHI & DE-ID TAB ────────────────────────────────────────── */
function PhiTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const perPatient = bd.per_patient_phi || [];
  const deid = bd.deidentification_detail || {};

  return (
    <>
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">Per-Patient PHI Exposure</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr><th>Patient ID</th><th>Name</th><th>PHI Fields</th><th>Risk</th></tr>
            </thead>
            <tbody>
              {perPatient.slice(0, 30).map(p => (
                <tr key={p.patient_id}>
                  <td className="text-muted small">{p.patient_id}</td>
                  <td className="fw-semibold">{p.name || '—'}</td>
                  <td>
                    {(p.fields_exposed || []).map(f => (
                      <span key={f} className="badge bg-danger me-1" style={{ fontSize: '0.65rem' }}>{f}</span>
                    ))}
                  </td>
                  <td><RiskBadge level={p.risk_level || 'medium'} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">De-identification Status</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr><th>Patient ID</th><th>Identifiers Present</th><th>De-ID Method</th><th>De-ID %</th></tr>
            </thead>
            <tbody>
              {(deid.per_patient || []).slice(0, 20).map(p => (
                <tr key={p.patient_id}>
                  <td className="text-muted small">{p.patient_id}</td>
                  <td>
                    {(p.identifiers_present || []).map(id => (
                      <span key={id} className="badge bg-warning text-dark me-1" style={{ fontSize: '0.65rem' }}>{id}</span>
                    ))}
                  </td>
                  <td className="small">{p.deid_method || '—'}</td>
                  <td>
                    <Bar pct={p.deid_pct || 0} color={p.deid_pct >= 100 ? '#22c55e' : '#ef4444'} />
                    <span className="small text-muted">{p.deid_pct ?? 0}%</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

/* ── ACCESS & AUDIT TAB ─────────────────────────────────────── */
function AccessTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const log = (bd.access_log || []).slice(0, 50);
  const perms = bd.actor_permissions || {};

  return (
    <>
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">Actor Permissions Matrix</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light"><tr><th>Actor</th><th>Actions</th><th>Event Count</th></tr></thead>
            <tbody>
              {Object.entries(perms).map(([actor, actions]) => (
                <tr key={actor}>
                  <td className="fw-semibold">{actor}</td>
                  <td>
                    {(actions || []).map(a => (
                      <span key={a.action} className="badge bg-primary me-1" style={{ fontSize: '0.65rem' }}>{a.action}</span>
                    ))}
                  </td>
                  <td>{(actions || []).reduce((s, a) => s + (a.count || 0), 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">Recent Access Log (latest 50)</div>
        <div className="card-body p-0" style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr><th>#</th><th>Patient</th><th>Component</th><th>Action</th><th>Actor</th><th>Timestamp</th></tr>
            </thead>
            <tbody>
              {log.map(e => (
                <tr key={e.id}>
                  <td className="text-muted small">{e.id}</td>
                  <td className="small">{e.patient_id || '—'}</td>
                  <td className="small text-truncate" style={{ maxWidth: 140 }}>{e.component}</td>
                  <td className="small">{e.action}</td>
                  <td className="small">{e.actor || '—'}</td>
                  <td className="small text-muted">{e.timestamp ? new Date(e.timestamp).toLocaleString() : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

/* ── INCIDENTS TAB ───────────────────────────────────────────── */
function IncidentsTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const inc = bd.incident_candidates || {};
  const multiActor = inc.multi_actor_access || [];
  const unusual = inc.unusual_patterns || [];

  return (
    <>
      <div className="alert alert-warning d-flex gap-2 align-items-center">
        <span style={{ fontSize: '1.4rem' }}>⚠️</span>
        <span>
          <strong>{multiActor.length + unusual.length} potential incident(s)</strong> detected from access pattern analysis.
          Review and determine if escalation is required.
        </span>
      </div>

      {multiActor.length > 0 && (
        <div className="card shadow-sm mb-3 border-warning">
          <div className="card-header fw-bold text-warning">Multi-Actor Access Events</div>
          <div className="card-body p-0">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Patient ID</th><th>Distinct Actors</th><th>Total Events</th><th>Severity</th></tr>
              </thead>
              <tbody>
                {multiActor.map((inc, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{inc.patient_id || '—'}</td>
                    <td>{inc.distinct_actors}</td>
                    <td>{inc.total_events}</td>
                    <td><RiskBadge level={inc.severity || 'medium'} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {unusual.length > 0 && (
        <div className="card shadow-sm mb-3 border-danger">
          <div className="card-header fw-bold text-danger">Unusual Access Patterns</div>
          <div className="card-body p-0">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Patient ID</th><th>Pattern</th><th>Severity</th></tr>
              </thead>
              <tbody>
                {unusual.map((p, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{p.patient_id || '—'}</td>
                    <td className="small">{p.pattern || '—'}</td>
                    <td><RiskBadge level={p.severity || 'medium'} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {multiActor.length === 0 && unusual.length === 0 && (
        <div className="alert alert-success">✅ No incident candidates detected.</div>
      )}
    </>
  );
}

/* ── DATASETS TAB ────────────────────────────────────────────── */
function DatasetsTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const fileTypes = bd.file_type_analysis || [];
  const eegVideo = bd.eeg_video_privacy || [];
  const retention = bd.retention_analysis || [];
  const dsReg = bd.dataset_registration || {};
  const records = dsReg.records || [];

  return (
    <>
      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">File Type Analysis</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Extension</th><th>Count</th></tr></thead>
                <tbody>
                  {fileTypes.map(f => (
                    <tr key={f.extension}>
                      <td><span className="badge bg-secondary">.{f.extension}</span></td>
                      <td>{f.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold">Retention Analysis</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Period</th><th>Records</th></tr></thead>
                <tbody>
                  {retention.map(r => (
                    <tr key={r.period}>
                      <td className="small">{r.period}</td>
                      <td>{r.record_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">EEG / Video Privacy Risk</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr><th>File</th><th>Type</th><th>Privacy Risk</th><th>Reason</th></tr>
            </thead>
            <tbody>
              {eegVideo.slice(0, 30).map((f, i) => (
                <tr key={i}>
                  <td className="small text-truncate" style={{ maxWidth: 160 }}>{f.file_name}</td>
                  <td><span className="badge bg-secondary">.{f.extension}</span></td>
                  <td>
                    <span className="badge" style={{
                      backgroundColor: f.privacy_risk === 'high' ? '#ef4444' : f.privacy_risk === 'medium' ? '#f59e0b' : '#22c55e',
                      color: '#fff'
                    }}>{f.privacy_risk}</span>
                  </td>
                  <td className="small text-muted">{f.risk_reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">Dataset Registry (latest 20)</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr><th>ID</th><th>Patient</th><th>File</th><th>Type</th><th>Registered</th></tr>
            </thead>
            <tbody>
              {records.slice(-20).reverse().map(r => (
                <tr key={r.id}>
                  <td className="text-muted small">{r.id}</td>
                  <td className="small">{r.patient_id}</td>
                  <td className="small text-truncate" style={{ maxWidth: 140 }}>{r.file_name}</td>
                  <td><span className="badge bg-secondary">{r.file_type}</span></td>
                  <td className="small text-muted">{r.created_at ? new Date(r.created_at).toLocaleDateString() : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

/* ── DATA SHARING TAB ────────────────────────────────────────── */
function SharingTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  const sharing = bd.data_sharing_detail || {};
  const records = sharing.records || [];
  const summary = sharing.summary || {};

  return (
    <>
      <div className="row mb-3">
        <div className="col-md-4">
          <div className="card shadow-sm">
            <div className="card-body text-center">
              <div className="h3 fw-bold text-primary">{summary.total_sharing_actions ?? 0}</div>
              <div className="text-muted small">Total Sharing Actions</div>
            </div>
          </div>
        </div>
      </div>

      {records.length === 0 ? (
        <div className="alert alert-info">✅ No data sharing actions recorded. All data is contained within the system.</div>
      ) : (
        <div className="card shadow-sm mb-3">
          <div className="card-header fw-bold">Data Sharing Records</div>
          <div className="card-body p-0">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Patient</th><th>Actor</th><th>Action</th><th>Recipient</th><th>Timestamp</th></tr>
              </thead>
              <tbody>
                {records.map((r, i) => (
                  <tr key={i}>
                    <td className="small">{r.patient_id}</td>
                    <td className="small">{r.actor}</td>
                    <td className="small">{r.action}</td>
                    <td className="small">{r.recipient || '—'}</td>
                    <td className="small text-muted">{r.timestamp ? new Date(r.timestamp).toLocaleDateString() : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}

/* ── DEFINITIONS TAB ─────────────────────────────────────────── */
function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  return (
    <>
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">Glossary</div>
        <div className="card-body">
          {(defs.terms || []).map(t => (
            <div key={t.term} className="mb-3">
              <div className="fw-semibold">{t.term}</div>
              <div className="text-muted small">{t.definition}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">Quality Metrics</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th>Metric</th><th>Description</th><th>Target</th></tr></thead>
            <tbody>
              {(defs.quality_metrics || []).map(m => (
                <tr key={m.metric}>
                  <td className="fw-semibold small">{m.metric}</td>
                  <td className="small text-muted">{m.description}</td>
                  <td><span className="badge bg-success">{m.target}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">Compliance References</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th>Standard</th><th>Relevance</th><th>Article / Section</th></tr></thead>
            <tbody>
              {(defs.compliance_references || []).map((r, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{r.standard}</td>
                  <td className="small text-muted">{r.relevance}</td>
                  <td className="small">{r.article || r.section || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {(defs.remediation_strategies || []).length > 0 && (
        <div className="card shadow-sm mb-3">
          <div className="card-header fw-bold">Remediation Strategies</div>
          <div className="card-body">
            {(defs.remediation_strategies || []).map((s, i) => (
              <div key={i} className="mb-3">
                <div className="fw-semibold">{s.issue}</div>
                <div className="text-muted small">{s.strategy}</div>
                {s.priority && <span className="badge bg-warning text-dark mt-1">{s.priority}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

/* ── MAIN PAGE ───────────────────────────────────────────────── */
export default function DataStewardPage() {
  const [activeTab, setActiveTab] = useState('overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);

  useEffect(() => {
    fetch(`${API}/api/data-steward/overview`).then(r => r.json()).then(setOverview).catch(() => setOverview({ error: true }));
    fetch(`${API}/api/data-steward/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setBreakdown({ error: true }));
    fetch(`${API}/api/data-steward/definitions`).then(r => r.json()).then(setDefs).catch(() => setDefs({ error: true }));
  }, []);

  const riskScore = overview?.kpis?.privacy_risk_score ?? 0;
  const riskLevel = riskScore >= 75 ? 'critical' : riskScore >= 50 ? 'high' : riskScore >= 25 ? 'medium' : 'low';

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-3 mb-3">
        <div>
          <h4 className="mb-0 fw-bold">🛡️ Data Steward / Privacy Officer</h4>
          <div className="text-muted small">PHI management · De-identification · Access control · Audit · Compliance</div>
        </div>
        <span className="badge ms-auto" style={{ backgroundColor: riskColor(riskLevel), fontSize: '0.85rem' }}>
          Risk: {riskScore}/100 — {riskLevel.toUpperCase()}
        </span>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${activeTab === t.id ? 'active' : ''}`}
              onClick={() => setActiveTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {activeTab === 'overview'    && <OverviewTab    ov={overview} />}
      {activeTab === 'phi'         && <PhiTab         bd={breakdown} />}
      {activeTab === 'access'      && <AccessTab      bd={breakdown} />}
      {activeTab === 'incidents'   && <IncidentsTab   bd={breakdown} />}
      {activeTab === 'datasets'    && <DatasetsTab    bd={breakdown} />}
      {activeTab === 'sharing'     && <SharingTab     bd={breakdown} />}
      {activeTab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  );
}
