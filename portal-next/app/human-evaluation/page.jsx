'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card text-center shadow-sm border-0">
        <div className="card-body py-2 px-1">
          <div className={`h4 mb-0 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted" style={{ fontSize: '0.75rem' }}>{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

const REASON_CODES = [
  { code: 'ART', label: 'Artifact — signal contamination' },
  { code: 'NRM', label: 'Normal variant — not pathological' },
  { code: 'CLS', label: 'Different seizure classification' },
  { code: 'MED', label: 'Medication effect' },
  { code: 'AGE', label: 'Age-related EEG pattern' },
  { code: 'CTX', label: 'Clinical context changes diagnosis' },
  { code: 'OTH', label: 'Other (see notes)' },
];

function SubmitPanel({ onSubmitted }) {
  const [form, setForm] = useState({
    patient_id: '',
    analysis_id: '',
    ai_prediction: '',
    decision: 'accept',
    human_decision: '',
    reason_code: '',
    reviewer_id: '',
    notes: '',
  });
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState(null);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const submit = async () => {
    if (!form.patient_id || !form.ai_prediction) {
      setErr('Patient ID and AI Prediction are required.');
      return;
    }
    if (form.decision === 'override' && !form.human_decision) {
      setErr('Human Decision label is required when overriding.');
      return;
    }
    setErr(null);
    setSubmitting(true);
    try {
      const body = {
        patient_id: form.patient_id,
        ai_prediction: form.ai_prediction,
        decision: form.decision,
      };
      if (form.analysis_id) body.analysis_id = parseInt(form.analysis_id, 10);
      if (form.human_decision) body.human_decision = form.human_decision;
      if (form.reason_code) body.reason_code = form.reason_code;
      if (form.reviewer_id) body.reviewer_id = form.reviewer_id;
      if (form.notes) body.notes = form.notes;

      const res = await fetch(`${API}/api/hitl/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Submit failed');
      setResult(data);
      setForm(f => ({ ...f, patient_id: '', analysis_id: '', ai_prediction: '', human_decision: '', reason_code: '', reviewer_id: '', notes: '' }));
      if (onSubmitted) onSubmitted();
    } catch (e) {
      setErr(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="card border-primary mb-4">
      <div className="card-header bg-primary text-white fw-bold">
        Submit HITL Review — Accept or Override AI Prediction
      </div>
      <div className="card-body">
        {result && (
          <div className="alert alert-success py-2 mb-3">
            Saved — ID #{result.id} · Decision: <strong>{result.decision}</strong>
          </div>
        )}
        {err && <div className="alert alert-danger py-2 mb-3">{err}</div>}
        <div className="row g-2">
          <div className="col-md-4">
            <label className="form-label small fw-semibold">Patient ID *</label>
            <input className="form-control form-control-sm" placeholder="e.g. P0001"
              value={form.patient_id} onChange={e => set('patient_id', e.target.value)} />
          </div>
          <div className="col-md-2">
            <label className="form-label small fw-semibold">Analysis ID</label>
            <input className="form-control form-control-sm" placeholder="optional int"
              value={form.analysis_id} onChange={e => set('analysis_id', e.target.value)} />
          </div>
          <div className="col-md-3">
            <label className="form-label small fw-semibold">AI Prediction *</label>
            <input className="form-control form-control-sm" placeholder="e.g. Epilepsy"
              value={form.ai_prediction} onChange={e => set('ai_prediction', e.target.value)} />
          </div>
          <div className="col-md-3">
            <label className="form-label small fw-semibold">Reviewer ID</label>
            <input className="form-control form-control-sm" placeholder="e.g. N001"
              value={form.reviewer_id} onChange={e => set('reviewer_id', e.target.value)} />
          </div>

          <div className="col-md-3">
            <label className="form-label small fw-semibold">Decision *</label>
            <select className="form-select form-select-sm" value={form.decision} onChange={e => set('decision', e.target.value)}>
              <option value="accept">Accept — AI is correct</option>
              <option value="override">Override — change diagnosis</option>
            </select>
          </div>

          {form.decision === 'override' && <>
            <div className="col-md-4">
              <label className="form-label small fw-semibold">Human Decision Label *</label>
              <input className="form-control form-control-sm" placeholder="e.g. Artifact, Normal"
                value={form.human_decision} onChange={e => set('human_decision', e.target.value)} />
            </div>
            <div className="col-md-5">
              <label className="form-label small fw-semibold">Reason Code</label>
              <select className="form-select form-select-sm" value={form.reason_code} onChange={e => set('reason_code', e.target.value)}>
                <option value="">— select reason —</option>
                {REASON_CODES.map(r => <option key={r.code} value={r.code}>{r.code} — {r.label.split('—')[1]?.trim()}</option>)}
              </select>
            </div>
          </>}

          <div className="col-12">
            <label className="form-label small fw-semibold">Notes (optional)</label>
            <textarea className="form-control form-control-sm" rows={2} placeholder="Clinical rationale…"
              value={form.notes} onChange={e => set('notes', e.target.value)} />
          </div>
        </div>
        <div className="mt-3">
          <button className={`btn btn-sm ${form.decision === 'override' ? 'btn-warning' : 'btn-success'}`}
            onClick={submit} disabled={submitting}>
            {submitting ? 'Submitting…' : form.decision === 'override' ? 'Submit Override' : 'Accept AI Decision'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function HumanEvaluationDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [hist, setHist] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [refresh, setRefresh] = useState(0);

  const load = () => {
    Promise.all([
      fetch(`${API}/api/human-evaluation/overview`).then(r => r.json()),
      fetch(`${API}/api/human-evaluation/breakdown`).then(r => r.json()),
      fetch(`${API}/api/human-evaluation/definitions`).then(r => r.json()),
      fetch(`${API}/api/hitl/history`).then(r => r.json()),
    ]).then(([o, b, d, h]) => { setOv(o); setBd(b); setDefs(d); setHist(h); })
      .catch(e => setErr(String(e)));
  };

  useEffect(() => { load(); }, [refresh]);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading human evaluation dashboard…</div>;

  const k = ov.kpis || {};
  const TABS = [
    { id: 'submit', label: 'Submit Review' },
    { id: 'overview', label: 'Overview' },
    { id: 'history', label: 'Review History' },
    { id: 'experts', label: 'Expert Reviews' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const reviews = hist?.reviews || [];
  const hitl = bd?.hitl_details || [];
  const experts = bd?.expert_details || [];

  return (
    <div className="p-3">
      <h3>Human Evaluation (HITL) Dashboard</h3>
      <p className="text-muted">
        Neurologist accept/override decisions for AI predictions — agreement tracking, audit trail, clinical governance.
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

      {/* ── SUBMIT ── */}
      {tab === 'submit' && (
        <div>
          <SubmitPanel onSubmitted={() => setRefresh(r => r + 1)} />
          <div className="alert alert-info py-2 small">
            <strong>Governance note:</strong> All HITL reviews are persisted to <code>hitl_reviews</code> table and
            count toward the AI-human agreement rate metric. Overrides trigger audit log entries per §51 responsible AI policy.
          </div>
        </div>
      )}

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          <div className="row">
            <KPI label="HITL Reviews" value={k.total_hitl_reviews} color="primary" />
            <KPI label="Expert Reviews" value={k.total_expert_reviews} color="info" />
            <KPI label="Clinical Decisions" value={k.total_clinical_decisions} color="secondary" />
            <KPI label="Component Findings" value={k.total_component_findings} color="warning" />
            <KPI label="Agreement Rate" value={`${k.agreement_rate}%`} color={k.agreement_rate >= 70 ? 'success' : 'warning'} sub="AI vs human" />
            <KPI label="Override Rate" value={`${k.override_rate}%`} color={k.override_rate > 40 ? 'danger' : 'success'} sub="of HITL reviews" />
            <KPI label="Avg Feedback" value={k.avg_feedback_rating ? `${k.avg_feedback_rating}/5` : '—'} color="primary" />
          </div>

          <div className="row mt-3">
            <div className="col-md-4 mb-3">
              <div className="card h-100">
                <div className="card-header small fw-bold">Decision Breakdown</div>
                <div className="card-body">
                  {(ov.decision_types || []).map(d => (
                    <div key={d.label} className="mb-2">
                      <div className="d-flex justify-content-between small">
                        <span>{d.label}</span><span>{d.value}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className={`progress-bar ${d.label === 'Override' ? 'bg-warning' : 'bg-success'}`}
                          style={{ width: `${Math.round(d.value / (k.total_hitl_reviews || 1) * 100)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-4 mb-3">
              <div className="card h-100">
                <div className="card-header small fw-bold">Agreement by Role</div>
                <div className="card-body">
                  {(ov.agreement_breakdown || []).map(a => (
                    <div key={a.label} className="mb-2">
                      <div className="d-flex justify-content-between small">
                        <span>{a.label}</span><span>{a.value}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className={`progress-bar ${a.label === 'Agree' ? 'bg-success' : 'bg-danger'}`}
                          style={{ width: `${Math.round(a.value / ((ov.agreement_breakdown || []).reduce((s, x) => s + x.value, 0) || 1) * 100)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-4 mb-3">
              <div className="card h-100">
                <div className="card-header small fw-bold">Reviews by Role</div>
                <div className="card-body">
                  {(ov.role_distribution || []).map(r => (
                    <div key={r.role} className="d-flex justify-content-between py-1 border-bottom small">
                      <span>{r.role}</span><strong>{r.count}</strong>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="card mt-2">
            <div className="card-header small fw-bold">Review Timeline</div>
            <div className="card-body p-2">
              <table className="table table-sm table-hover mb-0">
                <thead><tr><th>Date</th><th>Reviews</th><th>Bar</th></tr></thead>
                <tbody>
                  {(ov.review_timeline || []).map(t => (
                    <tr key={t.date}>
                      <td className="small">{t.date}</td>
                      <td>{t.reviews}</td>
                      <td><div className="progress" style={{ height: 6, minWidth: 60 }}>
                        <div className="progress-bar bg-primary" style={{ width: `${t.reviews * 20}%` }} />
                      </div></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── HISTORY ── */}
      {tab === 'history' && (
        <div>
          <div className="d-flex justify-content-between align-items-center mb-2">
            <h6 className="mb-0">All HITL Reviews ({reviews.length})</h6>
            <button className="btn btn-sm btn-outline-secondary" onClick={() => setRefresh(r => r + 1)}>Refresh</button>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-striped">
              <thead className="table-dark">
                <tr>
                  <th>ID</th><th>Patient</th><th>Analysis</th>
                  <th>AI Prediction</th><th>Decision</th>
                  <th>Human Label</th><th>Reason</th><th>Reviewer</th><th>Date</th>
                </tr>
              </thead>
              <tbody>
                {reviews.length === 0 && (
                  <tr><td colSpan={9} className="text-muted text-center">No reviews yet.</td></tr>
                )}
                {reviews.map(r => {
                  const f = r.fields || {};
                  return (
                    <tr key={r.id}>
                      <td>{r.id}</td>
                      <td><code>{r.patient_id}</code></td>
                      <td>{r.analysis_id || '—'}</td>
                      <td>{f.ai_prediction || '—'}</td>
                      <td>
                        <span className={`badge ${f.decision === 'override' ? 'bg-warning text-dark' : 'bg-success'}`}>
                          {f.decision || '—'}
                        </span>
                      </td>
                      <td>{f.human_decision || '—'}</td>
                      <td>{f.reason_code ? <code>{f.reason_code}</code> : '—'}</td>
                      <td>{f.reviewer_id || '—'}</td>
                      <td className="small text-muted">{r.created_at?.slice(0, 16) || '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── EXPERTS ── */}
      {tab === 'experts' && (
        <div>
          <h6 className="mb-3">Expert Review Log ({experts.length} entries)</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr>
                  <th>ID</th><th>Patient</th><th>Role</th><th>Expert</th>
                  <th>Finding</th><th>Agrees w/ AI</th><th>Note</th><th>Date</th>
                </tr>
              </thead>
              <tbody>
                {experts.map(e => (
                  <tr key={e.id}>
                    <td>{e.id}</td>
                    <td><code>{e.patient_id}</code></td>
                    <td>{e.role}</td>
                    <td>{e.expert}</td>
                    <td className="small">{e.finding}</td>
                    <td>
                      <span className={`badge ${e.agree_with_ai === 'agree' ? 'bg-success' : 'bg-danger'}`}>
                        {e.agree_with_ai}
                      </span>
                    </td>
                    <td className="small text-muted">{e.note || '—'}</td>
                    <td className="small text-muted">{e.created_at?.slice(0, 10) || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h6 className="mt-3 mb-2">Confidence vs Agreement</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead><tr><th>Analysis ID</th><th>AI Confidence</th><th>Agreed?</th></tr></thead>
              <tbody>
                {(ov.confidence_vs_agreement || []).map((c, i) => (
                  <tr key={i}>
                    <td>{c.analysis_id}</td>
                    <td>{(c.confidence * 100).toFixed(1)}%</td>
                    <td>
                      <span className={`badge ${c.agreed ? 'bg-success' : 'bg-danger'}`}>
                        {c.agreed ? 'Yes' : 'No'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && (
        <div>
          {defs && Object.entries(defs).map(([k, v]) => (
            <div key={k} className="mb-3">
              <h6 className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}</h6>
              {typeof v === 'string' && <p className="small text-muted mb-1">{v}</p>}
              {Array.isArray(v) && (
                <ul className="list-group list-group-flush">
                  {v.map((item, i) => (
                    <li key={i} className="list-group-item py-1 small">
                      {typeof item === 'string' ? item : JSON.stringify(item)}
                    </li>
                  ))}
                </ul>
              )}
              {typeof v === 'object' && !Array.isArray(v) && v !== null && (
                <table className="table table-sm table-bordered">
                  <tbody>
                    {Object.entries(v).map(([kk, vv]) => (
                      <tr key={kk}>
                        <td className="fw-semibold small" style={{ width: '30%' }}>{kk}</td>
                        <td className="small">{String(vv)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
