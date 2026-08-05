'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'channels',    label: 'Per Channel' },
  { id: 'patients',    label: 'Per Patient' },
  { id: 'definitions', label: 'Definitions' },
  { id: 'rerecord',    label: 'Re-record' },
];

const GRADE_COLOR = { Good: '#22c55e', Fair: '#f59e0b', Poor: '#ef4444' };
const GRADE_BG    = { Good: 'success',  Fair: 'warning',  Poor: 'danger'  };

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-3">
          <div className={`h4 fw-bold mb-1 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ pct, color, max }) {
  const w = Math.min(100, Math.max(0, max ? (pct / max) * 100 : pct || 0));
  return (
    <div className="progress" style={{ height: 10, borderRadius: 6 }}>
      <div className="progress-bar" style={{ width: `${w}%`, backgroundColor: color || '#3b82f6', borderRadius: 6 }} />
    </div>
  );
}

function GradeBadge({ grade }) {
  return <span className={`badge bg-${GRADE_BG[grade] || 'secondary'}`}>{grade || '—'}</span>;
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted p-3">Loading…</div>;

  const impDist  = ov.impedance_grade_distribution  || [];
  const qualDist = ov.quality_grade_distribution    || [];
  const monthly  = ov.monthly_trend                 || [];

  const impTotal  = impDist.reduce((s, x) => s + x.count, 0) || 1;
  const qualTotal = qualDist.reduce((s, x) => s + x.count, 0) || 1;
  const maxMonth  = Math.max(...monthly.map(m => m.recordings), 1);

  return (
    <div>
      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Recordings"           value={ov.total_recordings}                            color="primary"   sub="channel quality assessments" />
        <KPI label="Patients"             value={ov.total_patients}                               color="info"      sub="with channel data" />
        <KPI label="Avg Impedance"        value={`${ov.avg_impedance_kohm} kΩ`}                  color="warning"   sub="target <5 kΩ (ACNS)" />
        <KPI label="Avg SNR"              value={`${ov.avg_snr_db} dB`}                          color="success"   sub="target >20 dB" />
      </div>
      <div className="row mb-4">
        <KPI label="Good Impedance"       value={`${ov.overall_good_impedance_pct}%`}            color="success"   sub="<5 kΩ channels" />
        <KPI label="Good Quality"         value={`${ov.overall_good_quality_pct}%`}              color="primary"   sub="clean signal channels" />
        <KPI label="Channels / Recording" value={ov.total_channels}                               color="secondary" sub="10-20 system" />
        <KPI label="Min / Max Impedance"  value={`${ov.min_impedance_kohm}–${ov.max_impedance_kohm} kΩ`} color="secondary" sub="range across all patients" />
      </div>

      {/* Impedance + Quality grade distributions */}
      <div className="row mb-4">
        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Impedance Grade Distribution</div>
            <div className="card-body">
              {impDist.map(g => (
                <div key={g.grade} className="mb-3">
                  <div className="d-flex justify-content-between small mb-1">
                    <span><GradeBadge grade={g.grade} /></span>
                    <span className="fw-semibold">{g.count} channels ({((g.count / impTotal) * 100).toFixed(0)}%)</span>
                  </div>
                  <Bar pct={(g.count / impTotal) * 100} color={GRADE_COLOR[g.grade]} />
                </div>
              ))}
              <div className="mt-3 p-2 rounded" style={{ background: '#f8f9fa', fontSize: '0.78rem' }}>
                <strong>ACNS standard:</strong> Good &lt;5 kΩ · Fair 5–10 kΩ · Poor &gt;10 kΩ
              </div>
            </div>
          </div>
        </div>

        <div className="col-12 col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold">Signal Quality Grade Distribution</div>
            <div className="card-body">
              {qualDist.map(g => (
                <div key={g.grade} className="mb-3">
                  <div className="d-flex justify-content-between small mb-1">
                    <span><GradeBadge grade={g.grade} /></span>
                    <span className="fw-semibold">{g.count} channels ({((g.count / qualTotal) * 100).toFixed(0)}%)</span>
                  </div>
                  <Bar pct={(g.count / qualTotal) * 100} color={GRADE_COLOR[g.grade]} />
                </div>
              ))}
              <div className="mt-3 p-2 rounded" style={{ background: '#f8f9fa', fontSize: '0.78rem' }}>
                <strong>Interpretation:</strong> Good = clean · Fair = artifact risk · Poor = re-acquire
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Monthly trend */}
      {monthly.length > 0 && (
        <div className="card shadow-sm mb-4">
          <div className="card-header fw-semibold">Monthly Recording Volume</div>
          <div className="card-body">
            <div className="d-flex align-items-end gap-2" style={{ height: 80, overflowX: 'auto' }}>
              {monthly.map(m => (
                <div key={m.month} className="text-center flex-shrink-0" style={{ minWidth: 52 }}>
                  <div className="bg-primary rounded-top" style={{ height: Math.max(4, (m.recordings / maxMonth) * 64), width: 36, margin: '0 auto' }} />
                  <div className="text-muted" style={{ fontSize: '0.65rem', marginTop: 2 }}>{m.month}</div>
                  <div className="fw-semibold" style={{ fontSize: '0.7rem' }}>{m.recordings}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ChannelsPanel({ ov }) {
  const [sort, setSort] = useState('avg_impedance_kohm');
  const [dir,  setDir]  = useState(1);

  if (!ov) return <div className="text-muted p-3">Loading…</div>;
  const channels = [...(ov.per_channel_avg || [])].sort((a, b) => dir * ((a[sort] || 0) - (b[sort] || 0)));

  const toggle = col => {
    if (sort === col) setDir(d => -d);
    else { setSort(col); setDir(1); }
  };
  const Hdr = ({ col, label }) => (
    <th className="small" style={{ cursor: 'pointer', userSelect: 'none' }} onClick={() => toggle(col)}>
      {sort === col ? (dir === 1 ? '▼ ' : '▲ ') : ''}{label}
    </th>
  );
  const maxImp = Math.max(...channels.map(c => c.avg_impedance_kohm || 0), 1);

  return (
    <div>
      <div className="mb-2 text-muted small">{channels.length} channels (10-20 system) — click column header to sort</div>

      {/* Visual heatmap-style bar chart */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Average Impedance by Channel (kΩ)</div>
        <div className="card-body">
          <div className="row">
            {channels.map(c => {
              const imp = c.avg_impedance_kohm || 0;
              const color = imp < 5 ? '#22c55e' : imp < 10 ? '#f59e0b' : '#ef4444';
              return (
                <div key={c.channel} className="col-6 col-md-3 col-lg-2 mb-3 text-center">
                  <div className="fw-semibold small">{c.channel}</div>
                  <div style={{ height: 60, display: 'flex', alignItems: 'flex-end', justifyContent: 'center', marginBottom: 4 }}>
                    <div style={{
                      width: 28, backgroundColor: color, borderRadius: '3px 3px 0 0',
                      height: Math.max(4, (imp / maxImp) * 56),
                    }} />
                  </div>
                  <div className="fw-bold" style={{ fontSize: '0.78rem', color }}>{imp.toFixed(1)}</div>
                  <div className="text-muted" style={{ fontSize: '0.65rem' }}>kΩ</div>
                </div>
              );
            })}
          </div>
          <div className="mt-2" style={{ fontSize: '0.72rem' }}>
            <span style={{ color: '#22c55e' }}>■ Good (&lt;5)</span>
            <span className="ms-3" style={{ color: '#f59e0b' }}>■ Fair (5–10)</span>
            <span className="ms-3" style={{ color: '#ef4444' }}>■ Poor (&gt;10)</span>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-light">
            <tr>
              <th className="small">Channel</th>
              <Hdr col="avg_impedance_kohm"   label="Avg Impedance (kΩ)" />
              <Hdr col="avg_snr_db"           label="Avg SNR (dB)" />
              <Hdr col="pct_good_impedance"   label="Good Impedance %" />
              <Hdr col="pct_good_quality"     label="Good Quality %" />
            </tr>
          </thead>
          <tbody>
            {channels.map(c => {
              const imp = c.avg_impedance_kohm || 0;
              const impGrade = imp < 5 ? 'Good' : imp < 10 ? 'Fair' : 'Poor';
              const snr = c.avg_snr_db || 0;
              const snrColor = snr >= 20 ? 'success' : snr >= 10 ? 'warning' : 'danger';
              return (
                <tr key={c.channel}>
                  <td className="fw-semibold">{c.channel}</td>
                  <td>
                    <span className={`badge bg-${GRADE_BG[impGrade]}`}>{imp.toFixed(2)}</span>
                  </td>
                  <td><span className={`text-${snrColor} fw-semibold`}>{snr.toFixed(1)}</span></td>
                  <td>
                    <div className="d-flex align-items-center gap-2">
                      <span className="small">{(c.pct_good_impedance || 0).toFixed(0)}%</span>
                      <div style={{ flex: 1 }}>
                        <Bar pct={c.pct_good_impedance || 0} color="#22c55e" />
                      </div>
                    </div>
                  </td>
                  <td>
                    <div className="d-flex align-items-center gap-2">
                      <span className="small">{(c.pct_good_quality || 0).toFixed(0)}%</span>
                      <div style={{ flex: 1 }}>
                        <Bar pct={c.pct_good_quality || 0} color="#3b82f6" />
                      </div>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PatientsPanel({ bk }) {
  const [sort, setSort] = useState('avg_impedance');
  const [dir,  setDir]  = useState(1);

  if (!bk) return <div className="text-muted p-3">Loading…</div>;
  const patients = [...(bk.per_patient || [])].sort((a, b) => dir * ((a[sort] || 0) - (b[sort] || 0)));

  const toggle = col => {
    if (sort === col) setDir(d => -d);
    else { setSort(col); setDir(1); }
  };
  const Hdr = ({ col, label }) => (
    <th className="small" style={{ cursor: 'pointer', userSelect: 'none' }} onClick={() => toggle(col)}>
      {sort === col ? (dir === 1 ? '▼ ' : '▲ ') : ''}{label}
    </th>
  );

  return (
    <div>
      <div className="mb-2 text-muted small">{patients.length} patients — click column header to sort · Poor grade = re-gelling needed</div>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-light">
            <tr>
              <th className="small">Patient</th>
              <Hdr col="avg_impedance"  label="Avg Impedance (kΩ)" />
              <Hdr col="avg_snr"        label="Avg SNR (dB)" />
              <Hdr col="good_channels"  label="Good Ch" />
              <Hdr col="fair_channels"  label="Fair Ch" />
              <Hdr col="poor_channels"  label="Poor Ch" />
              <th className="small">Overall Grade</th>
              <th className="small">Recorded</th>
            </tr>
          </thead>
          <tbody>
            {patients.map(p => (
              <tr key={p.patient_id}>
                <td><span className="badge bg-secondary">{p.patient_id}</span></td>
                <td className="fw-semibold">{(p.avg_impedance || 0).toFixed(2)}</td>
                <td className={`text-${p.avg_snr >= 20 ? 'success' : p.avg_snr >= 10 ? 'warning' : 'danger'} fw-semibold`}>
                  {(p.avg_snr || 0).toFixed(1)}
                </td>
                <td className="text-success text-center">{p.good_channels ?? 0}</td>
                <td className="text-warning text-center">{p.fair_channels ?? 0}</td>
                <td className="text-danger text-center">{p.poor_channels ?? 0}</td>
                <td><GradeBadge grade={p.overall_grade} /></td>
                <td className="text-muted small">{p.created_at ? p.created_at.slice(0, 10) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Poor-grade alert */}
      {patients.filter(p => p.overall_grade === 'Poor').length > 0 && (
        <div className="alert alert-danger mt-3 small">
          <strong>Re-acquisition needed:</strong>{' '}
          {patients.filter(p => p.overall_grade === 'Poor').length} patient(s) have Poor overall grade —
          re-gel electrodes or repeat recording before EEG interpretation.
          Patients: {patients.filter(p => p.overall_grade === 'Poor').map(p => p.patient_id).join(', ')}
        </div>
      )}
    </div>
  );
}

function DefinitionsPanel({ def }) {
  if (!def) return <div className="text-muted p-3">Loading…</div>;
  const grades   = def.impedance_grades  || {};
  const quality  = def.quality_grades    || {};
  const channels = def.channels          || {};
  const glossary = def.glossary          || {};
  const notes    = def.clinical_notes    || [];

  return (
    <div>
      {/* Impedance grades */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Impedance Grade Thresholds (ACNS)</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th>Grade</th><th>Definition</th></tr></thead>
            <tbody>
              {Object.entries(grades).map(([g, d]) => (
                <tr key={g}>
                  <td><GradeBadge grade={g} /></td>
                  <td className="small">{d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Quality grades */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Signal Quality Grade Definitions</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th>Grade</th><th>Definition</th></tr></thead>
            <tbody>
              {Object.entries(quality).map(([g, d]) => (
                <tr key={g}>
                  <td><GradeBadge grade={g} /></td>
                  <td className="small">{d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Channel positions */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">10-20 Channel Positions</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <thead className="table-light"><tr><th>Channel</th><th>Location &amp; Role</th></tr></thead>
              <tbody>
                {Object.entries(channels).map(([ch, desc]) => (
                  <tr key={ch}>
                    <td className="fw-semibold small">{ch}</td>
                    <td className="small">{desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Glossary */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Glossary</div>
        <div className="card-body">
          <div className="row">
            {Object.entries(glossary).map(([term, def_]) => (
              <div key={term} className="col-12 col-md-6 mb-3">
                <div className="p-2 rounded border" style={{ background: '#f8faff' }}>
                  <div className="fw-semibold small">{term}</div>
                  <div className="text-muted" style={{ fontSize: '0.8rem' }}>{def_}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Clinical notes */}
      {notes.length > 0 && (
        <div className="card shadow-sm mb-4">
          <div className="card-header fw-semibold">Clinical Notes</div>
          <div className="card-body">
            <ul className="mb-0" style={{ fontSize: '0.85rem' }}>
              {notes.map((n, i) => <li key={i} className="mb-2">{n}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

const PRIORITY_COLOR = { urgent: 'danger', routine: 'secondary' };
const STATUS_COLOR   = { pending: 'warning', scheduled: 'info', completed: 'success' };

const CHANNELS_10_20 = [
  'Fp1','Fp2','F3','F4','C3','C4','P3','P4',
  'O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz',
];

function RerecordPanel({ rr, onRefresh }) {
  const [form, setForm] = useState({
    patient_id: '', channels: [], reason: '', priority: 'routine',
    notes: '', requested_by: 'EEG Technician',
  });
  const [submitting, setSubmitting] = useState(false);
  const [submitMsg,  setSubmitMsg]  = useState(null);

  if (!rr) return <div className="text-muted p-3">Loading…</div>;

  const { requests = [], candidates = [], summary = {} } = rr;

  const toggleChannel = ch => {
    setForm(f => ({
      ...f,
      channels: f.channels.includes(ch)
        ? f.channels.filter(c => c !== ch)
        : [...f.channels, ch],
    }));
  };

  const prefillCandidate = c => {
    setForm(f => ({
      ...f,
      patient_id: c.patient_id,
      channels: c.poor_channels,
    }));
    document.getElementById('rerecord-form-top')?.scrollIntoView({ behavior: 'smooth' });
  };

  const submit = async () => {
    if (!form.patient_id || form.channels.length === 0 || !form.reason) {
      setSubmitMsg({ type: 'danger', text: 'Patient ID, at least one channel, and a reason are required.' });
      return;
    }
    setSubmitting(true);
    setSubmitMsg(null);
    try {
      const res = await fetch(`${API}/api/channel-quality/rerecord-request`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');
      setSubmitMsg({ type: 'success', text: `Re-record request #${data.id} submitted for patient ${data.patient_id}.` });
      setForm({ patient_id: '', channels: [], reason: '', priority: 'routine', notes: '', requested_by: 'EEG Technician' });
      onRefresh();
    } catch (e) {
      setSubmitMsg({ type: 'danger', text: e.message });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      {/* Summary KPIs */}
      <div className="row mb-4">
        {[
          { label: 'Total Requests',    value: summary.total_requests,            color: 'primary' },
          { label: 'Pending',           value: summary.pending,                   color: 'warning' },
          { label: 'Scheduled',         value: summary.scheduled,                 color: 'info'    },
          { label: 'Candidates to Flag',value: summary.candidates_awaiting_flag,  color: 'danger'  },
        ].map(k => <KPI key={k.label} label={k.label} value={k.value ?? 0} color={k.color} />)}
      </div>

      {/* Candidates needing a request */}
      {candidates.length > 0 && (
        <div className="card shadow-sm mb-4 border-danger">
          <div className="card-header fw-semibold text-danger">
            Patients with &gt;30% Poor Channels — No Pending Request
          </div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0 align-middle">
              <thead className="table-light">
                <tr>
                  <th className="small">Patient</th>
                  <th className="small">Poor Channels</th>
                  <th className="small">% Poor</th>
                  <th className="small">Recorded</th>
                  <th className="small">Action</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map(c => (
                  <tr key={c.patient_id}>
                    <td><span className="badge bg-secondary">{c.patient_id}</span></td>
                    <td className="small">{c.poor_channels.join(', ')}</td>
                    <td><span className="badge bg-danger">{c.pct_poor}%</span></td>
                    <td className="text-muted small">{c.recorded_at ? c.recorded_at.slice(0, 10) : '—'}</td>
                    <td>
                      <button className="btn btn-sm btn-outline-danger py-0" onClick={() => prefillCandidate(c)}>
                        Flag for Re-record
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Submit Form */}
      <div className="card shadow-sm mb-4" id="rerecord-form-top">
        <div className="card-header fw-semibold">Submit Re-record Request</div>
        <div className="card-body">
          {submitMsg && (
            <div className={`alert alert-${submitMsg.type} py-2 small`}>{submitMsg.text}</div>
          )}
          <div className="row mb-3">
            <div className="col-12 col-md-6 mb-3">
              <label className="form-label small fw-semibold">Patient ID *</label>
              <input
                className="form-control form-control-sm"
                placeholder="e.g. P001"
                value={form.patient_id}
                onChange={e => setForm(f => ({ ...f, patient_id: e.target.value }))}
              />
            </div>
            <div className="col-12 col-md-3 mb-3">
              <label className="form-label small fw-semibold">Priority</label>
              <select
                className="form-select form-select-sm"
                value={form.priority}
                onChange={e => setForm(f => ({ ...f, priority: e.target.value }))}
              >
                <option value="routine">Routine</option>
                <option value="urgent">Urgent</option>
              </select>
            </div>
            <div className="col-12 col-md-3 mb-3">
              <label className="form-label small fw-semibold">Requested By</label>
              <input
                className="form-control form-control-sm"
                value={form.requested_by}
                onChange={e => setForm(f => ({ ...f, requested_by: e.target.value }))}
              />
            </div>
          </div>

          <div className="mb-3">
            <label className="form-label small fw-semibold">Channels to Re-record * (select all that apply)</label>
            <div className="d-flex flex-wrap gap-2">
              {CHANNELS_10_20.map(ch => (
                <button
                  key={ch}
                  type="button"
                  className={`btn btn-sm ${form.channels.includes(ch) ? 'btn-danger' : 'btn-outline-secondary'}`}
                  style={{ minWidth: 46, fontSize: '0.75rem', padding: '2px 8px' }}
                  onClick={() => toggleChannel(ch)}
                >
                  {ch}
                </button>
              ))}
            </div>
            {form.channels.length > 0 && (
              <div className="mt-2 small text-danger">
                Selected: {form.channels.join(', ')}
              </div>
            )}
          </div>

          <div className="mb-3">
            <label className="form-label small fw-semibold">Reason *</label>
            <select
              className="form-select form-select-sm"
              value={form.reason}
              onChange={e => setForm(f => ({ ...f, reason: e.target.value }))}
            >
              <option value="">— Select reason —</option>
              <option value="High impedance (>10 kΩ) — re-gel electrode">High impedance (&gt;10 kΩ) — re-gel electrode</option>
              <option value="Poor SNR (<10 dB) — signal too noisy">Poor SNR (&lt;10 dB) — signal too noisy</option>
              <option value="Electrode pop artifact — reseat electrode">Electrode pop artifact — reseat electrode</option>
              <option value="Gel bridge between channels — clean and re-apply">Gel bridge between channels — clean and re-apply</option>
              <option value="Patient movement artifact — repeat segment">Patient movement artifact — repeat segment</option>
              <option value="Amplifier saturation — check connection">Amplifier saturation — check connection</option>
              <option value="Other">Other</option>
            </select>
          </div>

          <div className="mb-3">
            <label className="form-label small fw-semibold">Notes (optional)</label>
            <textarea
              className="form-control form-control-sm"
              rows={2}
              placeholder="Additional context for the re-recording technician…"
              value={form.notes}
              onChange={e => setForm(f => ({ ...f, notes: e.target.value }))}
            />
          </div>

          <button
            className="btn btn-danger btn-sm"
            onClick={submit}
            disabled={submitting}
          >
            {submitting ? 'Submitting…' : 'Submit Re-record Request'}
          </button>
        </div>
      </div>

      {/* Request Queue */}
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold">Re-record Request Queue ({requests.length})</div>
        {requests.length === 0 ? (
          <div className="card-body text-muted small">No requests submitted yet.</div>
        ) : (
          <div className="table-responsive">
            <table className="table table-sm mb-0 align-middle">
              <thead className="table-light">
                <tr>
                  <th className="small">#</th>
                  <th className="small">Patient</th>
                  <th className="small">Channels</th>
                  <th className="small">Reason</th>
                  <th className="small">Priority</th>
                  <th className="small">Status</th>
                  <th className="small">Requested By</th>
                  <th className="small">Date</th>
                </tr>
              </thead>
              <tbody>
                {requests.map(r => (
                  <tr key={r.id}>
                    <td className="text-muted small">{r.id}</td>
                    <td><span className="badge bg-secondary">{r.patient_id}</span></td>
                    <td className="small">{(r.channels || []).join(', ')}</td>
                    <td className="small" style={{ maxWidth: 200 }}>{r.reason}</td>
                    <td><span className={`badge bg-${PRIORITY_COLOR[r.priority] || 'secondary'}`}>{r.priority}</span></td>
                    <td><span className={`badge bg-${STATUS_COLOR[r.status] || 'secondary'}`}>{r.status}</span></td>
                    <td className="small text-muted">{r.requested_by}</td>
                    <td className="small text-muted">{r.created_at ? r.created_at.slice(0, 10) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChannelQualityPage() {
  const [tab, setTab] = useState('overview');
  const [ov,  setOv]  = useState(null);
  const [bk,  setBk]  = useState(null);
  const [def, setDef] = useState(null);
  const [rr,  setRr]  = useState(null);
  const [err, setErr] = useState(null);

  const loadRr = () =>
    fetch(`${API}/api/channel-quality/rerecord-requests`).then(r => r.json()).then(setRr);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/channel-quality/overview`).then(r => r.json()),
      fetch(`${API}/api/channel-quality/breakdown`).then(r => r.json()),
      fetch(`${API}/api/channel-quality/definitions`).then(r => r.json()),
      fetch(`${API}/api/channel-quality/rerecord-requests`).then(r => r.json()),
    ]).then(([o, b, d, r]) => { setOv(o); setBk(b); setDef(d); setRr(r); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: '1.6rem' }}>&#x1f4f6;</span>
        <div>
          <h4 className="mb-0 fw-bold">EEG Channel Quality Dashboard</h4>
          <div className="text-muted small">
            {ov
              ? `${ov.total_recordings} recordings · ${ov.total_patients} patients · avg impedance ${ov.avg_impedance_kohm} kΩ · avg SNR ${ov.avg_snr_db} dB`
              : 'Loading…'}
          </div>
        </div>
      </div>

      {err && <div className="alert alert-danger">{err}</div>}

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel  ov={ov} />}
      {tab === 'channels'    && <ChannelsPanel  ov={ov} />}
      {tab === 'patients'    && <PatientsPanel  bk={bk} />}
      {tab === 'definitions' && <DefinitionsPanel def={def} />}
      {tab === 'rerecord'    && <RerecordPanel  rr={rr} onRefresh={loadRr} />}
    </div>
  );
}
