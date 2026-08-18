'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const REASON_CODES = [
  { code: 'ART', label: 'Artifact — signal contamination' },
  { code: 'NRM', label: 'Normal variant — not pathological' },
  { code: 'CLS', label: 'Different seizure classification' },
  { code: 'MED', label: 'Medication effect' },
  { code: 'AGE', label: 'Age-related EEG pattern' },
  { code: 'CTX', label: 'Clinical context changes diagnosis' },
  { code: 'OTH', label: 'Other (see notes)' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card text-center shadow-sm border-0">
        <div className="card-body py-2">
          <div className={`h4 mb-0 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted" style={{ fontSize: '0.75rem' }}>{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

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
    setSubmitting(true); setErr(null); setResult(null);
    try {
      const r = await fetch(`${API}/api/hitl/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          human_decision: form.decision === 'accept' ? form.ai_prediction : form.human_decision,
        }),
      });
      const d = await r.json();
      if (r.ok) { setResult(d); onSubmitted && onSubmitted(); }
      else setErr(d.detail || 'Submission failed.');
    } catch (e) { setErr('Network error.'); }
    finally { setSubmitting(false); }
  };

  if (result) return (
    <div className="alert alert-success">
      <div className="fw-bold mb-1">Evaluation Submitted</div>
      <div className="small">
        ID: <code>{result.id}</code> · Agreement: <strong>{result.agreement ? 'YES' : 'NO'}</strong>
        {result.audit_flag && <span className="badge bg-warning text-dark ms-2">Audit Flag</span>}
      </div>
      <button className="btn btn-sm btn-outline-success mt-2" onClick={() => setResult(null)}>Submit Another</button>
    </div>
  );

  return (
    <div className="card shadow-sm">
      <div className="card-header py-2 bg-primary text-white">
        <strong>Submit Human Evaluation</strong>
      </div>
      <div className="card-body">
        {err && <div className="alert alert-danger small">{err}</div>}
        <div className="row g-3">
          <div className="col-md-4">
            <label className="form-label small fw-semibold">Patient ID *</label>
            <input className="form-control form-control-sm" value={form.patient_id}
              onChange={e => set('patient_id', e.target.value)} placeholder="e.g. PAT-001" />
          </div>
          <div className="col-md-4">
            <label className="form-label small fw-semibold">Analysis ID</label>
            <input className="form-control form-control-sm" value={form.analysis_id}
              onChange={e => set('analysis_id', e.target.value)} placeholder="optional" />
          </div>
          <div className="col-md-4">
            <label className="form-label small fw-semibold">Reviewer ID</label>
            <input className="form-control form-control-sm" value={form.reviewer_id}
              onChange={e => set('reviewer_id', e.target.value)} placeholder="e.g. DR-001" />
          </div>
          <div className="col-md-6">
            <label className="form-label small fw-semibold">AI Prediction *</label>
            <select className="form-select form-select-sm" value={form.ai_prediction}
              onChange={e => set('ai_prediction', e.target.value)}>
              <option value="">— select AI prediction —</option>
              {['seizure', 'normal', 'artifact', 'uncertain'].map(v => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>
          <div className="col-md-6">
            <label className="form-label small fw-semibold">Decision</label>
            <div className="d-flex gap-3 mt-1">
              {['accept', 'override'].map(d => (
                <div key={d} className="form-check">
                  <input className="form-check-input" type="radio" id={`dec_${d}`}
                    checked={form.decision === d} onChange={() => set('decision', d)} />
                  <label className="form-check-label small text-capitalize" htmlFor={`dec_${d}`}>{d}</label>
                </div>
              ))}
            </div>
          </div>
          {form.decision === 'override' && (
            <>
              <div className="col-md-6">
                <label className="form-label small fw-semibold">Human Decision</label>
                <select className="form-select form-select-sm" value={form.human_decision}
                  onChange={e => set('human_decision', e.target.value)}>
                  <option value="">— your classification —</option>
                  {['seizure', 'normal', 'artifact', 'uncertain'].map(v => (
                    <option key={v} value={v}>{v}</option>
                  ))}
                </select>
              </div>
              <div className="col-md-6">
                <label className="form-label small fw-semibold">Reason Code</label>
                <select className="form-select form-select-sm" value={form.reason_code}
                  onChange={e => set('reason_code', e.target.value)}>
                  <option value="">— reason for override —</option>
                  {REASON_CODES.map(r => (
                    <option key={r.code} value={r.code}>{r.code} – {r.label}</option>
                  ))}
                </select>
              </div>
            </>
          )}
          <div className="col-12">
            <label className="form-label small fw-semibold">Notes</label>
            <textarea className="form-control form-control-sm" rows={2} value={form.notes}
              onChange={e => set('notes', e.target.value)} placeholder="Clinical context, observations..." />
          </div>
        </div>
        <div className="mt-3">
          <button className="btn btn-primary btn-sm" onClick={submit} disabled={submitting}>
            {submitting ? <><span className="spinner-border spinner-border-sm me-1" />Submitting…</> : 'Submit Evaluation'}
          </button>
          <span className="text-muted small ms-3">Fields marked * are required</span>
        </div>
      </div>
    </div>
  );
}

function HistoryPanel({ refresh }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);

  const load = () => {
    fetch(`${API}/api/hitl/history`)
      .then(r => r.json()).then(setD).catch(() => setErr('Load error'));
  };

  useEffect(() => { load(); }, [refresh]);

  if (err) return <p className="text-danger">{err}</p>;
  if (!d) return <div className="text-center py-4"><span className="spinner-border text-primary" /></div>;

  const records = d.records || d.evaluations || [];
  const stats = d.stats || {};

  return (
    <div>
      {stats && Object.keys(stats).length > 0 && (
        <div className="row g-3 mb-4">
          <KPI label="Total Evaluations" value={stats.total || records.length} color="primary" />
          <KPI label="Overrides" value={stats.overrides ?? '—'} color="warning" />
          <KPI label="Agreement Rate" value={stats.agreement_rate != null ? `${(stats.agreement_rate * 100).toFixed(0)}%` : '—'} color="success" />
          <KPI label="Audit Flags" value={stats.audit_flags ?? '—'} color="danger" />
        </div>
      )}
      <div className="card shadow-sm">
        <div className="card-header py-2 d-flex align-items-center">
          <strong>Evaluation History</strong>
          <span className="badge bg-secondary ms-2">{records.length}</span>
          <button className="btn btn-sm btn-outline-secondary ms-auto" onClick={load}>↻ Refresh</button>
        </div>
        <div className="card-body p-0">
          {records.length === 0 ? (
            <p className="text-muted text-center py-4 mb-0">No evaluations yet. Submit one above.</p>
          ) : (
            <div style={{ maxHeight: 400, overflowY: 'auto' }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light sticky-top">
                  <tr>
                    <th>ID</th>
                    <th>Patient</th>
                    <th>AI Prediction</th>
                    <th>Decision</th>
                    <th>Agreement</th>
                    <th>Reason</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {records.slice().reverse().slice(0, 50).map((r, i) => (
                    <tr key={r.id || i}>
                      <td className="font-monospace small text-muted">{r.id?.slice(0, 8) || '—'}</td>
                      <td className="small">{r.patient_id}</td>
                      <td><span className="badge bg-secondary">{r.ai_prediction}</span></td>
                      <td><span className={`badge bg-${r.decision === 'accept' ? 'success' : 'warning'}`}>{r.decision}</span></td>
                      <td>
                        {r.agreement != null && (
                          <span className={`badge bg-${r.agreement ? 'success' : 'danger'}`}>
                            {r.agreement ? 'Yes' : 'No'}
                          </span>
                        )}
                      </td>
                      <td className="small text-muted">{r.reason_code || '—'}</td>
                      <td className="small text-muted">{r.timestamp ? new Date(r.timestamp).toLocaleString() : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function HitlPage() {
  const [tab, setTab] = useState('submit');
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <div>
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: '#1b5e20' }}>
            Human-in-the-Loop (HITL) Evaluation
          </h4>
          <div className="small text-muted">
            Clinician review · AI override workflow · Audit trail · Agreement rate tracking
          </div>
        </div>
        <div className="ms-auto">
          <a href="/human-evaluation" className="btn btn-outline-success btn-sm">
            Full HITL Review →
          </a>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {[
          { id: 'submit', label: 'Submit Evaluation' },
          { id: 'history', label: 'Evaluation History' },
        ].map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active fw-semibold' : ''}`}
              onClick={() => setTab(t.id)}
              style={tab === t.id ? { color: '#1b5e20', borderBottomColor: '#1b5e20' } : {}}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'submit' && (
        <SubmitPanel onSubmitted={() => setRefreshKey(k => k + 1)} />
      )}
      {tab === 'history' && (
        <HistoryPanel refresh={refreshKey} />
      )}

      <div className="alert alert-warning small mt-4">
        <strong>Clinical Governance:</strong> All HITL overrides are logged for audit. Human decisions take precedence over AI predictions. Agreement rate &lt; 85% triggers model review. This system supports clinical decision-making and does not replace clinician judgment.
      </div>
    </div>
  );
}
