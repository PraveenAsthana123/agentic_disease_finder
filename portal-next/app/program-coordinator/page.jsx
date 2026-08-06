'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',          label: 'Overview' },
  { id: 'journey',           label: 'Patient Journey' },
  { id: 'mdt',               label: 'MDT Coordination' },
  { id: 'kpi',               label: 'Operational KPIs' },
  { id: 'resource-planning', label: 'Resource Planning' },
];

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-2 mb-2">
      <div className="card text-center shadow-sm border-0 h-100">
        <div className="card-body py-2 px-1">
          <div className={`h3 mb-0 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted" style={{ fontSize: '0.72rem' }}>{label}</div>
        </div>
      </div>
    </div>
  );
}

function ProgressBar({ pct, color }) {
  const c = color || (pct >= 80 ? 'success' : pct >= 50 ? 'primary' : pct >= 25 ? 'warning' : 'danger');
  return (
    <div className="progress" style={{ height: '8px' }}>
      <div className={`progress-bar bg-${c}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

// ─── Overview Panel ───────────────────────────────────────────────────────

function OverviewPanel({ ov }) {
  if (!ov) return <div className="spinner-border text-primary" />;
  const pj = ov.modules?.patient_journey || {};
  const kp = ov.modules?.kpi_dashboard || {};
  const rp = ov.modules?.resource_planning || {};
  const mdt = ov.modules?.mdt_coordination || {};

  const funnel = pj.stage_funnel || {};
  const stages = ['registered', 'data_uploaded', 'eeg_analyzed', 'clinically_assessed', 'expert_reviewed'];
  const labels = {
    registered: 'Registered',
    data_uploaded: 'Data Uploaded',
    eeg_analyzed: 'EEG Analyzed',
    clinically_assessed: 'Clinically Assessed',
    expert_reviewed: 'Expert Reviewed',
  };

  return (
    <div>
      <div className="alert alert-primary border-0 shadow-sm mb-3">
        <strong>Role:</strong> {ov.role} &nbsp;|&nbsp; {ov.description}
      </div>

      {/* KPI row */}
      <div className="row mb-4">
        <KPI label="Patients Enrolled" value={kp.kpis?.patients_enrolled} color="primary" />
        <KPI label="EEG Analyses" value={kp.kpis?.eeg_analyses_run} color="info" />
        <KPI label="Avg Progress %" value={pj.avg_progress_pct} color="success" />
        <KPI label="Pending MDT Review" value={mdt.pending_review} color="warning" />
        <KPI label="Primary Bottleneck" value={rp.primary_bottleneck?.replace('awaiting_','') || '—'} color="danger" />
        <KPI label="Avg Confidence" value={kp.kpis?.avg_model_confidence?.toFixed(3)} color="secondary" />
      </div>

      {/* Care pipeline funnel */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold">Care Pipeline Funnel</div>
        <div className="card-body">
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <thead><tr><th>Stage</th><th>Patients</th><th>Coverage</th></tr></thead>
              <tbody>
                {stages.map(s => {
                  const n = funnel[s] || 0;
                  const total = funnel['registered'] || 1;
                  const pct = Math.round(100 * n / total);
                  return (
                    <tr key={s}>
                      <td>{labels[s]}</td>
                      <td><span className="badge bg-primary">{n}</span></td>
                      <td style={{ minWidth: '180px' }}>
                        <div className="d-flex align-items-center gap-2">
                          <div style={{ flex: 1 }}><ProgressBar pct={pct} /></div>
                          <small className="text-muted">{pct}%</small>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Flags */}
      {(kp.flags || []).length > 0 && (
        <div className="card border-warning shadow-sm mb-3">
          <div className="card-header fw-bold text-warning">Program Flags</div>
          <div className="card-body">
            <ul className="mb-0">
              {(kp.flags || []).map((f, i) => <li key={i} className="text-warning fw-bold">{f}</li>)}
            </ul>
          </div>
        </div>
      )}

      <p className="text-muted small">{pj.note}</p>
    </div>
  );
}

// ─── Journey Panel ────────────────────────────────────────────────────────

function JourneyPanel({ journey }) {
  if (!journey) return <div className="spinner-border text-primary" />;
  const { journeys = [], stage_funnel = {}, avg_progress_pct } = journey;

  return (
    <div>
      <div className="d-flex gap-3 mb-3 flex-wrap">
        <span className="badge bg-primary fs-6">Patients: {journey.n_patients}</span>
        <span className="badge bg-info fs-6">Avg Progress: {avg_progress_pct}%</span>
        <span className="badge bg-warning text-dark fs-6">
          Stalled: {journeys.filter(j => j.stalled).length}
        </span>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold">Per-Patient Care Pathway</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Age/Gender</th>
                  <th>Dept</th>
                  <th>Current Stage</th>
                  <th>Next Action</th>
                  <th>Progress</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {journeys.map(j => (
                  <tr key={j.patient_id} className={j.stalled ? 'table-warning' : ''}>
                    <td><small className="text-muted">{j.patient_id}</small><br />{j.name}</td>
                    <td>{j.age} / {j.gender}</td>
                    <td>{j.department}</td>
                    <td><span className="badge bg-secondary">{j.current_stage?.replace(/_/g,' ')}</span></td>
                    <td>
                      {j.next_action === 'complete'
                        ? <span className="badge bg-success">complete</span>
                        : <span className="badge bg-primary">{j.next_action?.replace(/_/g,' ')}</span>}
                    </td>
                    <td style={{ minWidth: '120px' }}>
                      <div className="d-flex align-items-center gap-1">
                        <div style={{ flex: 1 }}><ProgressBar pct={j.progress_pct} /></div>
                        <small>{j.progress_pct}%</small>
                      </div>
                    </td>
                    <td>
                      {j.stalled
                        ? <span className="badge bg-warning text-dark">Stalled</span>
                        : j.next_action === 'complete'
                          ? <span className="badge bg-success">Done</span>
                          : <span className="badge bg-info">Active</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <p className="text-muted small">{journey.note}</p>
    </div>
  );
}

// ─── MDT Panel ────────────────────────────────────────────────────────────

function MdtPanel({ mdt }) {
  if (!mdt) return <div className="spinner-border text-primary" />;
  const { pending_queue = [], mdt_role_load = {} } = mdt;

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Analyses Total" value={mdt.analyses_total} color="primary" />
        <KPI label="Reviewed" value={mdt.reviewed} color="success" />
        <KPI label="Pending Review" value={mdt.pending_review} color="warning" />
        <KPI label="HITL Reviews" value={mdt.hitl_reviews} color="info" />
        <KPI label="AI Agreement" value={mdt.ai_agreement_rate != null ? `${(mdt.ai_agreement_rate*100).toFixed(0)}%` : '—'} color="secondary" />
      </div>

      <div className="row">
        <div className="col-md-8">
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Pending Review Queue (top 20, low-confidence first)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>Analysis ID</th><th>Patient</th><th>Prediction</th><th>Confidence</th><th>Priority</th></tr>
                  </thead>
                  <tbody>
                    {pending_queue.length === 0
                      ? <tr><td colSpan={5} className="text-center text-muted">No pending reviews</td></tr>
                      : pending_queue.map(r => (
                          <tr key={r.analysis_id}>
                            <td><code>{r.analysis_id}</code></td>
                            <td>{r.patient_id}</td>
                            <td>{r.predicted_label}</td>
                            <td>{r.confidence != null ? r.confidence.toFixed(3) : '—'}</td>
                            <td>
                              <span className={`badge bg-${r.priority === 'high' ? 'danger' : 'secondary'}`}>
                                {r.priority}
                              </span>
                            </td>
                          </tr>
                        ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">MDT Role Workload</div>
            <div className="card-body">
              {Object.keys(mdt_role_load).length === 0
                ? <p className="text-muted small">No role data</p>
                : Object.entries(mdt_role_load).map(([role, count]) => (
                    <div key={role} className="mb-2">
                      <div className="d-flex justify-content-between mb-1">
                        <small>{role}</small><small>{count} reviews</small>
                      </div>
                      <ProgressBar pct={Math.min(100, (count / (mdt.reviewed || 1)) * 100)} color="info" />
                    </div>
                  ))}
            </div>
          </div>
        </div>
      </div>
      <p className="text-muted small">{mdt.note}</p>
    </div>
  );
}

// ─── KPI Panel ────────────────────────────────────────────────────────────

function KpiPanel({ kpiData }) {
  if (!kpiData) return <div className="spinner-border text-primary" />;
  const { kpis = {}, coverage_rates_pct = {}, patients_by_department = {}, flags = [] } = kpiData;

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Patients Enrolled" value={kpis.patients_enrolled} color="primary" />
        <KPI label="EEG Analyses Run" value={kpis.eeg_analyses_run} color="info" />
        <KPI label="Avg Confidence" value={kpis.avg_model_confidence?.toFixed(3)} color="secondary" />
        <KPI label="Low Confidence" value={kpis.low_confidence_analyses} color="warning" />
        <KPI label="Assessments" value={kpis.assessments_recorded} color="success" />
        <KPI label="Expert Reviews" value={kpis.expert_reviews_done} color="danger" />
      </div>

      <div className="row">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Coverage Rates by Stage</div>
            <div className="card-body">
              {Object.entries(coverage_rates_pct).map(([stage, pct]) => (
                <div key={stage} className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <small>{stage.replace(/_/g,' ')}</small>
                    <small className="fw-bold">{pct}%</small>
                  </div>
                  <ProgressBar pct={pct} />
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Patients by Department</div>
            <div className="card-body">
              {Object.keys(patients_by_department).length === 0
                ? <p className="text-muted small">No department data</p>
                : Object.entries(patients_by_department).map(([dept, count]) => (
                    <div key={dept} className="mb-2">
                      <div className="d-flex justify-content-between mb-1">
                        <small>{dept}</small><small>{count}</small>
                      </div>
                      <ProgressBar pct={Math.min(100, (count / (kpis.patients_enrolled || 1)) * 100)} color="primary" />
                    </div>
                  ))}
            </div>
          </div>
        </div>
      </div>

      {flags.length > 0 && (
        <div className="alert alert-warning">
          <strong>Flags:</strong>
          <ul className="mb-0 mt-1">
            {flags.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      )}
      <p className="text-muted small">{kpiData.note}</p>
    </div>
  );
}

// ─── Resource Planning Panel ──────────────────────────────────────────────

function ResourcePanel({ resource }) {
  if (!resource) return <div className="spinner-border text-primary" />;
  const { backlog_by_stage = {}, primary_bottleneck, bottleneck_count, recommendation } = resource;

  const backlogLabels = {
    awaiting_upload: 'Awaiting Upload',
    awaiting_analysis: 'Awaiting Analysis',
    awaiting_assessment: 'Awaiting Assessment',
    awaiting_expert_review: 'Awaiting Expert Review',
  };

  const maxBacklog = Math.max(...Object.values(backlog_by_stage), 1);

  return (
    <div>
      <div className="row mb-4">
        <div className="col-md-8">
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Backlog by Stage</div>
            <div className="card-body">
              {Object.entries(backlog_by_stage).map(([stage, count]) => {
                const isPrimary = stage === primary_bottleneck;
                return (
                  <div key={stage} className="mb-3">
                    <div className="d-flex justify-content-between mb-1">
                      <small className={isPrimary ? 'fw-bold text-danger' : ''}>
                        {backlogLabels[stage] || stage.replace(/_/g,' ')}
                        {isPrimary && ' 🔴 PRIMARY BOTTLENECK'}
                      </small>
                      <span className={`badge bg-${isPrimary ? 'danger' : 'secondary'}`}>{count}</span>
                    </div>
                    <ProgressBar pct={Math.round(100 * count / maxBacklog)} color={isPrimary ? 'danger' : 'warning'} />
                  </div>
                );
              })}
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className={`card shadow-sm mb-3 border-${primary_bottleneck ? 'danger' : 'success'}`}>
            <div className="card-header fw-bold">
              {primary_bottleneck ? '🔴 Bottleneck Identified' : '✅ Balanced'}
            </div>
            <div className="card-body">
              {primary_bottleneck ? (
                <>
                  <p className="mb-2">
                    <strong>Stage:</strong> {backlogLabels[primary_bottleneck] || primary_bottleneck?.replace(/_/g,' ')}
                  </p>
                  <p className="mb-2">
                    <strong>Patients stuck:</strong>{' '}
                    <span className="badge bg-danger fs-6">{bottleneck_count}</span>
                  </p>
                  <hr />
                  <p className="mb-0 text-muted small">
                    <strong>Recommendation:</strong><br />
                    {recommendation}
                  </p>
                </>
              ) : (
                <p className="text-success mb-0">{recommendation}</p>
              )}
            </div>
          </div>
        </div>
      </div>
      <p className="text-muted small">{resource.note}</p>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────

export default function ProgramCoordinatorPage() {
  const [ov,       setOv]       = useState(null);
  const [journey,  setJourney]  = useState(null);
  const [mdt,      setMdt]      = useState(null);
  const [kpiData,  setKpiData]  = useState(null);
  const [resource, setResource] = useState(null);
  const [tab,      setTab]      = useState('overview');
  const [err,      setErr]      = useState(null);

  useEffect(() => {
    fetch(`${API}/api/coordinator`).then(r => r.json()).then(setOv).catch(e => setErr(e.message));
    fetch(`${API}/api/coordinator/journey`).then(r => r.json()).then(setJourney).catch(() => {});
    fetch(`${API}/api/coordinator/mdt`).then(r => r.json()).then(setMdt).catch(() => {});
    fetch(`${API}/api/coordinator/kpi`).then(r => r.json()).then(setKpiData).catch(() => {});
    fetch(`${API}/api/coordinator/resource-planning`).then(r => r.json()).then(setResource).catch(() => {});
  }, []);

  if (err) return <div className="p-4 alert alert-danger">Error: {err}</div>;
  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  return (
    <div>
      <h3>&#x1f9ed; Epilepsy Program Coordinator Dashboard</h3>
      <p className="text-muted small">
        Patient journey tracking, MDT coordination, operational KPIs, and resource/capacity
        planning across the epilepsy care program.
      </p>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'          && <OverviewPanel  ov={ov} />}
      {tab === 'journey'           && <JourneyPanel   journey={journey} />}
      {tab === 'mdt'               && <MdtPanel       mdt={mdt} />}
      {tab === 'kpi'               && <KpiPanel       kpiData={kpiData} />}
      {tab === 'resource-planning' && <ResourcePanel  resource={resource} />}
    </div>
  );
}
