'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const respColor = r =>
  r === 'ictal_discharge'        ? 'danger'  :
  r === 'generalized_spike_wave' ? 'warning' :
  r === 'focal_spike_wave'       ? 'warning' :
  r === 'firda'                  ? 'info'    :
  r === 'slowing_only'           ? 'primary' : 'secondary';

const gradeColor = g =>
  g >= 3 ? 'danger' : g === 2 ? 'warning' : 'success';

function KpiCard({ label, value, unit = '', sub = '', color = 'primary' }) {
  return (
    <div className={`card border-${color} mb-3`}>
      <div className="card-body py-2 px-3">
        <div className={`fw-bold text-${color} fs-5`}>
          {value}{unit && <small className="fs-6 ms-1">{unit}</small>}
        </div>
        <div className="text-muted small">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: '0.72rem' }}>{sub}</div>}
      </div>
    </div>
  );
}

function Bar({ label, count, max, color = 'primary' }) {
  const pct = max > 0 ? Math.round((count / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className={`text-${color} fw-semibold`}>{count}</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

/* ── Overview tab ──────────────────────────────────────────────────── */
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const { hv, ips, total_patients, combined_activation_any, protocols } = data;
  const hvMax = Math.max(...Object.values(hv.response_counts));
  return (
    <div>
      <h5 className="mb-3">HV / Photic Stimulation — Population Summary</h5>
      <p className="text-muted small mb-3">
        N={total_patients} patients · Standard: {protocols.standard}
      </p>

      <div className="row g-3 mb-4">
        <div className="col-6 col-md-3"><KpiCard label="HV Performed" value={hv.performed} color="primary" sub={`of ${total_patients} patients`} /></div>
        <div className="col-6 col-md-3"><KpiCard label="HV Activation Rate" value={hv.activation_rate_pct} unit="%" color="warning" sub="spike-wave or ictal" /></div>
        <div className="col-6 col-md-3"><KpiCard label="IPS Performed" value={ips.performed} color="info" sub={`of ${total_patients} patients`} /></div>
        <div className="col-6 col-md-3"><KpiCard label="PPR Rate" value={ips.ppr_rate_pct} unit="%" color={ips.ppr_rate_pct > 20 ? 'warning' : 'success'} sub={`${ips.ppr_count} PPR events`} /></div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-6 col-md-3"><KpiCard label="Epileptiform PPR (III–IV)" value={ips.epileptiform_ppr} color="danger" sub="Grade III or IV" /></div>
        <div className="col-6 col-md-3"><KpiCard label="Avg HV Duration" value={hv.avg_duration_s} unit="s" color="secondary" sub="target 180 s (3 min)" /></div>
        <div className="col-6 col-md-3"><KpiCard label="Avg Normalization" value={hv.avg_normalization_s} unit="s" color="secondary" sub="after HV stop" /></div>
        <div className="col-6 col-md-3"><KpiCard label="HV Early Stops" value={hv.early_stops} color={hv.early_stops > 0 ? 'danger' : 'success'} sub="clinical seizure" /></div>
      </div>

      <div className="row g-3">
        <div className="col-md-6">
          <div className="card">
            <div className="card-header fw-semibold">HV Response Distribution</div>
            <div className="card-body">
              {Object.entries(hv.response_counts).map(([resp, cnt]) => (
                <Bar key={resp} label={resp.replace(/_/g, ' ')} count={cnt}
                  max={hv.performed} color={respColor(resp)} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card">
            <div className="card-header fw-semibold">IPS PPR Grade Distribution</div>
            <div className="card-body">
              {Object.entries(ips.grade_distribution).map(([g, cnt]) => (
                <Bar key={g} label={`Grade ${g}`} count={cnt}
                  max={ips.ppr_count || 1} color={gradeColor(parseInt(g))} />
              ))}
              {ips.ppr_count === 0 && <p className="text-muted small">No PPR events detected.</p>}
            </div>
          </div>
        </div>
      </div>

      <div className="alert alert-info mt-4 small">
        <strong>Combined activation events (HV + IPS):</strong> {combined_activation_any} across {total_patients} patients.
        IPS rates tested: {protocols.ips_rates_hz.join(', ')} Hz.
      </div>
    </div>
  );
}

/* ── Protocol Detail tab ───────────────────────────────────────────── */
function ProtocolTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const { per_patient, hv_response_distribution, ips_hz_distribution,
          hv_contraindications, ips_grade_distribution } = data;
  const [sort, setSort] = useState('patient_id');
  const [filter, setFilter] = useState('all');

  let rows = [...per_patient];
  if (filter === 'hv_active') rows = rows.filter(r => ['generalized_spike_wave','focal_spike_wave','ictal_discharge'].includes(r.hv_response));
  if (filter === 'ppr') rows = rows.filter(r => r.ips_ppr);
  rows.sort((a, b) => {
    if (sort === 'patient_id') return a.patient_id - b.patient_id;
    if (sort === 'hv_response') return (a.hv_response||'').localeCompare(b.hv_response||'');
    if (sort === 'ips_grade') return (b.ips_grade||0) - (a.ips_grade||0);
    if (sort === 'ips_peak_hz') return (b.ips_peak_hz||0) - (a.ips_peak_hz||0);
    return 0;
  });

  const hvMax = Math.max(...hv_response_distribution.map(r => r.count), 1);
  const hzMax = Math.max(...ips_hz_distribution.map(r => r.count), 1);

  return (
    <div>
      <div className="row g-3 mb-3">
        <div className="col-md-6">
          <div className="card">
            <div className="card-header fw-semibold">HV Response Distribution</div>
            <div className="card-body">
              {hv_response_distribution.map(r => (
                <Bar key={r.response} label={r.label} count={r.count}
                  max={hvMax} color={respColor(r.response)} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card">
            <div className="card-header fw-semibold">IPS Peak Activation Rate Histogram (Hz)</div>
            <div className="card-body">
              {ips_hz_distribution.filter(r => r.count > 0).map(r => (
                <Bar key={r.hz} label={`${r.hz} Hz`} count={r.count}
                  max={hzMax} color="info" />
              ))}
              {ips_hz_distribution.every(r => r.count === 0) && <p className="text-muted small">No PPR events.</p>}
            </div>
          </div>
        </div>
      </div>

      {hv_contraindications.length > 0 && (
        <div className="alert alert-warning small mb-3">
          <strong>HV Contraindications ({hv_contraindications.length} patients):</strong>{' '}
          {hv_contraindications.map(c => `P${c.patient_id}: ${c.reason}`).join(' · ')}
        </div>
      )}

      <div className="d-flex gap-2 mb-3">
        <select className="form-select form-select-sm w-auto"
          value={filter} onChange={e => setFilter(e.target.value)}>
          <option value="all">All Patients</option>
          <option value="hv_active">HV Activation (spike-wave / ictal)</option>
          <option value="ppr">IPS PPR Only</option>
        </select>
        <select className="form-select form-select-sm w-auto"
          value={sort} onChange={e => setSort(e.target.value)}>
          <option value="patient_id">Sort: Patient ID</option>
          <option value="hv_response">Sort: HV Response</option>
          <option value="ips_grade">Sort: IPS Grade ↓</option>
          <option value="ips_peak_hz">Sort: Peak Hz ↓</option>
        </select>
      </div>

      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead className="table-dark">
            <tr>
              <th>Patient</th><th>Age/Sex</th>
              <th>HV Done</th><th>HV Response</th><th>HV Dur (s)</th><th>Norm (s)</th>
              <th>IPS Done</th><th>PPR</th><th>Grade</th><th>Peak Hz</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.patient_id}>
                <td>P{r.patient_id}</td>
                <td>{r.age}y {r.sex}</td>
                <td>{r.hv_performed ? <span className="badge bg-success">Yes</span> : <span className="badge bg-secondary">No</span>}</td>
                <td>
                  {r.hv_performed
                    ? <span className={`badge bg-${respColor(r.hv_response)}`}>{r.hv_response.replace(/_/g,' ')}</span>
                    : <span className="text-muted">—</span>}
                  {r.hv_early_stop && <span className="badge bg-danger ms-1">early stop</span>}
                </td>
                <td>{r.hv_duration_s ?? '—'}</td>
                <td>{r.hv_normalization_s ?? '—'}</td>
                <td>{r.ips_performed ? <span className="badge bg-info">Yes</span> : <span className="badge bg-secondary">No</span>}</td>
                <td>{r.ips_ppr ? <span className="badge bg-warning text-dark">PPR</span> : '—'}</td>
                <td>{r.ips_grade ? <span className={`badge bg-${gradeColor(r.ips_grade)}`}>Grade {r.ips_grade}</span> : '—'}</td>
                <td>{r.ips_peak_hz ? `${r.ips_peak_hz} Hz` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Definitions tab ───────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const { hv_protocol, ips_protocol, references, safety_notes } = data;
  return (
    <div>
      <div className="row g-3">
        <div className="col-md-6">
          <div className="card mb-3">
            <div className="card-header fw-semibold text-primary">Hyperventilation (HV) Protocol</div>
            <div className="card-body small">
              <p><strong>Full name:</strong> {hv_protocol.full_name}</p>
              <p><strong>Duration:</strong> {hv_protocol.duration}</p>
              <p><strong>Mechanism:</strong> {hv_protocol.mechanism}</p>
              <p><strong>Target conditions:</strong></p>
              <ul>{hv_protocol.target_conditions.map((c,i) => <li key={i}>{c}</li>)}</ul>
              <p><strong>Protocol steps:</strong></p>
              <ol>{hv_protocol.protocol_steps.map((s,i) => <li key={i}>{s}</li>)}</ol>
              <p><strong>Normalization:</strong> {hv_protocol.normalization_target}</p>
              <p><strong>Standard:</strong> {hv_protocol.standard}</p>

              <p className="fw-semibold mt-2">Contraindications:</p>
              <ul>{hv_protocol.contraindications.map((c,i) => <li key={i} className="text-danger">{c}</li>)}</ul>

              <p className="fw-semibold mt-2">EEG Responses:</p>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Response</th><th>Clinical Significance</th></tr></thead>
                <tbody>
                  {hv_protocol.responses.map((r,i) => (
                    <tr key={i}>
                      <td><span className={`badge bg-${respColor(r.code)}`}>{r.label}</span></td>
                      <td>{r.significance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-6">
          <div className="card mb-3">
            <div className="card-header fw-semibold text-info">Intermittent Photic Stimulation (IPS)</div>
            <div className="card-body small">
              <p><strong>Full name:</strong> {ips_protocol.full_name}</p>
              <p><strong>Rates tested:</strong> {ips_protocol.rates_hz.join(', ')} Hz</p>
              <p><strong>Stimulus:</strong> {ips_protocol.stimulus}</p>
              <p><strong>Sequence:</strong> {ips_protocol.sequence}</p>
              <p><strong>Target conditions:</strong></p>
              <ul>{ips_protocol.target_conditions.map((c,i) => <li key={i}>{c}</li>)}</ul>
              <p><strong>Protocol steps:</strong></p>
              <ol>{ips_protocol.protocol_steps.map((s,i) => <li key={i}>{s}</li>)}</ol>
              <p><strong>Standard:</strong> {ips_protocol.standard}</p>

              <p className="fw-semibold mt-2">PPR Grades (Waltz et al. 1992):</p>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Grade</th><th>Description</th></tr></thead>
                <tbody>
                  {ips_protocol.ppr_grades.map((g,i) => (
                    <tr key={i}>
                      <td><span className={`badge bg-${gradeColor(g.grade)}`}>{g.label}</span></td>
                      <td>{g.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <p className="fw-semibold mt-2">Contraindications:</p>
              <ul>{ips_protocol.contraindications.map((c,i) => <li key={i}>{c}</li>)}</ul>
            </div>
          </div>
        </div>
      </div>

      <div className="card mb-3">
        <div className="card-header fw-semibold text-danger">Safety Notes</div>
        <div className="card-body small">
          <ul>{safety_notes.map((n,i) => <li key={i}>{n}</li>)}</ul>
        </div>
      </div>

      <div className="card">
        <div className="card-header fw-semibold">References</div>
        <div className="card-body small">
          <ol>{references.map((r,i) => <li key={i}>{r}</li>)}</ol>
        </div>
      </div>
    </div>
  );
}

/* ── Main page ─────────────────────────────────────────────────────── */
export default function HvPhoticPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');

  useEffect(() => {
    const ep = tab === 'overview'     ? 'overview'
             : tab === 'protocol'     ? 'breakdown'
             : tab === 'definitions'  ? 'definitions' : null;
    if (!ep) return;
    setLoading(true); setError('');
    fetch(`${API}/api/hv-photic/${ep}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(d => {
        if (tab === 'overview')    setOverview(d);
        if (tab === 'protocol')    setBreakdown(d);
        if (tab === 'definitions') setDefinitions(d);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  const tabs = [
    { id: 'overview',    label: '📊 Overview' },
    { id: 'protocol',    label: '📋 Per Patient' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1">⚡ HV / Photic Stimulation Protocol Dashboard</h4>
      <p className="text-muted small mb-3">
        EEG Activation Procedures — Hyperventilation (HV) + Intermittent Photic Stimulation (IPS)
        · ACNS 2016 · Kasteleijn-Nolst Trenité 2012 · EEG Technician Role
      </p>

      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}>{t.label}
            </button>
          </li>
        ))}
      </ul>

      {loading && <div className="text-center py-4"><div className="spinner-border text-primary" /></div>}
      {error   && <div className="alert alert-danger">{error}</div>}

      {!loading && !error && (
        <>
          {tab === 'overview'    && <OverviewTab    data={overview} />}
          {tab === 'protocol'    && <ProtocolTab    data={breakdown} />}
          {tab === 'definitions' && <DefinitionsTab data={definitions} />}
        </>
      )}
    </div>
  );
}
