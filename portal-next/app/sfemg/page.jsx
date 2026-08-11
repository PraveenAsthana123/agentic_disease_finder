'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const DX_COLOR = dx => ({
  'Normal':                          'success',
  'Myasthenia Gravis':               'danger',
  'Lambert-Eaton MS':                'warning',
  'Congenital Myasthenic Syndrome':  'info',
  'Motor Neuron Disease':            'dark',
  'Myopathic':                       'secondary',
}[dx] || 'secondary');

const LIMB_ICON = l =>
  l === 'cranial' ? '🧠' : l === 'upper' ? '🖐️' : '🦶';

function KPICard({ label, value, sub, color = 'primary' }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className={`card border-${color} h-100`}>
        <div className="card-body text-center py-3">
          <div className={`display-6 fw-bold text-${color}`}>{value}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="xsmall text-muted mt-1" style={{ fontSize: '0.72rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ label, val, max, colorClass = 'primary', unit = '' }) {
  const pct = max > 0 ? Math.round((val / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{val}{unit}</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className={`progress-bar bg-${colorClass}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function RefLine({ label, normal, unit }) {
  return (
    <div className="d-flex justify-content-between small border-bottom py-1">
      <span className="text-muted">{label}</span>
      <span className="fw-semibold">{normal} {unit}</span>
    </div>
  );
}

export default function SFEMGPage() {
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [defs, setDefs]   = useState(null);
  const [tab, setTab]     = useState('overview');
  const [ptSort, setPtSort] = useState('max_mcd_us');
  const [studyFilter, setStudyFilter] = useState('all');

  useEffect(() => {
    fetch(`${API}/api/sfemg/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/sfemg/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/sfemg/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  const kpis = ov?.kpis || {};
  const mcdDist = ov?.mcd_distribution || {};
  const fdDist  = ov?.fd_distribution  || {};
  const dxDist  = ov?.diagnosis_distribution || {};
  const muscles = ov?.muscle_summary || {};
  const patients = (ov?.patient_summary || []).slice().sort((a, b) =>
    (b[ptSort] ?? 0) - (a[ptSort] ?? 0));

  const scatter  = bd?.scatter || [];
  const dxProfs  = bd?.diagnosis_profiles || [];
  const studyLog = bd?.study_log || [];

  const filteredStudies = studyFilter === 'all'
    ? studyLog
    : studyLog.filter(s => s.overall_abnormal === (studyFilter === 'abnormal'));

  const mcdMax = Math.max(...Object.values(mcdDist), 1);
  const fdMax  = Math.max(...Object.values(fdDist), 1);
  const dxMax  = Math.max(...Object.values(dxDist), 1);

  const tabs = ['overview', 'jitter_analysis', 'per_patient', 'definitions'];
  const TAB_LABELS = {
    overview:       'Overview',
    jitter_analysis:'Jitter & FD Analysis',
    per_patient:    'Per Patient',
    definitions:    'Definitions',
  };

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center mb-1">
        <h2 className="fw-bold mb-0">🧬 Single Fiber EMG (SFEMG)</h2>
        <span className="badge bg-info text-dark ms-3 fs-6">NMJ Diagnostic</span>
      </div>
      <p className="text-muted mb-3">
        Most sensitive test for neuromuscular junction (NMJ) dysfunction — jitter (MCD), fiber density, and blocking analysis.
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {tabs.map(t => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link${tab === t ? ' active fw-semibold' : ''}`}
              onClick={() => setTab(t)}
            >{TAB_LABELS[t]}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <>
          {/* KPIs */}
          <div className="row mb-2">
            <KPICard label="Total Studies"   value={kpis.total_studies  ?? '—'} color="primary" />
            <KPICard label="Patients"        value={kpis.total_patients ?? '—'} color="info" />
            <KPICard label="Abnormal Studies" value={kpis.pct_abnormal != null ? `${kpis.pct_abnormal}%` : '—'}
              sub={`${kpis.n_abnormal ?? 0} studies`} color="danger" />
            <KPICard label="Mean MCD (Jitter)" value={kpis.mean_mcd_us != null ? `${kpis.mean_mcd_us} µs` : '—'}
              sub="normal < 35 µs" color={kpis.mean_mcd_us >= 35 ? 'warning' : 'success'} />
          </div>
          <div className="row mb-4">
            <KPICard label="Mean Fiber Density" value={kpis.mean_fiber_density ?? '—'}
              sub="normal 1.3–1.8" color={kpis.mean_fiber_density > 1.8 ? 'warning' : 'success'} />
            <KPICard label="Mean Blocking" value={kpis.mean_blocking_pct != null ? `${kpis.mean_blocking_pct}%` : '—'}
              sub="normal < 10%" color={kpis.mean_blocking_pct >= 10 ? 'danger' : 'success'} />
            <KPICard label="Blocking Abnormal" value={kpis.blocking_abnormal_n ?? '—'}
              sub="≥ 10% blocking" color="warning" />
            <KPICard label="FD Abnormal" value={kpis.fd_abnormal_n ?? '—'}
              sub="> 1.8 fiber density" color="secondary" />
          </div>

          <div className="row">
            {/* MCD Distribution */}
            <div className="col-md-4 mb-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Jitter (MCD) Distribution</div>
                <div className="card-body">
                  <div className="small text-muted mb-2">Normal threshold: &lt; 55 µs per pair</div>
                  {Object.entries(mcdDist).map(([label, n]) => (
                    <Bar key={label} label={label} val={n} max={mcdMax} unit=" studies"
                      colorClass={label.includes('< 35') ? 'success' : label.includes('35') ? 'info' : label.includes('55') ? 'warning' : 'danger'} />
                  ))}
                </div>
              </div>
            </div>

            {/* FD Distribution */}
            <div className="col-md-4 mb-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Fiber Density Distribution</div>
                <div className="card-body">
                  <div className="small text-muted mb-2">Normal range: 1.3–1.8 fiber potentials/MU</div>
                  {Object.entries(fdDist).map(([label, n]) => (
                    <Bar key={label} label={label} val={n} max={fdMax} unit=" studies"
                      colorClass={label.includes('normal') ? 'success' : label.includes('< 1.3') ? 'warning' : 'danger'} />
                  ))}
                </div>
              </div>
            </div>

            {/* Diagnosis Distribution */}
            <div className="col-md-4 mb-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Diagnosis Distribution</div>
                <div className="card-body">
                  {Object.entries(dxDist).map(([dx, n]) => (
                    <div key={dx} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>
                          <span className={`badge bg-${DX_COLOR(dx)} me-1`}>&nbsp;</span>
                          {dx}
                        </span>
                        <span className="fw-bold">{n}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className={`progress-bar bg-${DX_COLOR(dx)}`}
                          style={{ width: `${Math.round(n / dxMax * 100)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Muscle Summary */}
          <div className="card mb-4">
            <div className="card-header fw-semibold">Muscle Summary</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Muscle</th>
                    <th>Studies</th>
                    <th>Abnormal %</th>
                    <th>Mean MCD (µs)</th>
                    <th>Mean FD</th>
                    <th>Limb</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(muscles).map(([muscle, info]) => (
                    <tr key={muscle}>
                      <td className="small fw-semibold">{muscle}</td>
                      <td>{info.n_studies}</td>
                      <td>
                        <span className={`badge bg-${info.pct_abnormal >= 50 ? 'danger' : info.pct_abnormal >= 20 ? 'warning' : 'success'}`}>
                          {info.pct_abnormal}%
                        </span>
                      </td>
                      <td className={info.mean_mcd_us >= 55 ? 'text-danger fw-semibold' : ''}>
                        {info.mean_mcd_us}
                      </td>
                      <td className={info.mean_fd > 1.8 ? 'text-warning fw-semibold' : ''}>
                        {info.mean_fd}
                      </td>
                      <td>
                        {LIMB_ICON(muscle.toLowerCase().includes('orb') || muscle.toLowerCase().includes('front') ? 'cranial' : 'upper')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── Jitter & FD Analysis ── */}
      {tab === 'jitter_analysis' && (
        <>
          {/* Per-Diagnosis Profiles */}
          <div className="card mb-4">
            <div className="card-header fw-semibold">Per-Diagnosis Mean Profiles</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Diagnosis</th>
                    <th>N Studies</th>
                    <th>Mean MCD (µs)</th>
                    <th>Mean Fiber Density</th>
                    <th>Mean Blocking %</th>
                    <th>MCD Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {dxProfs.map(p => (
                    <tr key={p.diagnosis}>
                      <td>
                        <span className={`badge bg-${DX_COLOR(p.diagnosis)} me-2`}>&nbsp;</span>
                        {p.diagnosis}
                      </td>
                      <td>{p.n}</td>
                      <td className={p.mean_mcd_us >= 55 ? 'text-danger fw-bold' : p.mean_mcd_us >= 35 ? 'text-warning fw-semibold' : 'text-success'}>
                        {p.mean_mcd_us}
                      </td>
                      <td className={p.mean_fd > 1.8 ? 'text-warning fw-semibold' : ''}>
                        {p.mean_fd}
                      </td>
                      <td className={p.mean_blocking_pct >= 10 ? 'text-danger fw-semibold' : ''}>
                        {p.mean_blocking_pct}%
                      </td>
                      <td>
                        <span className={`badge bg-${p.mean_mcd_us >= 80 ? 'danger' : p.mean_mcd_us >= 55 ? 'warning' : p.mean_mcd_us >= 35 ? 'info' : 'success'}`}>
                          {p.mean_mcd_us >= 80 ? 'Severe' : p.mean_mcd_us >= 55 ? 'Abnormal' : p.mean_mcd_us >= 35 ? 'Borderline' : 'Normal'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Study Scatter Table */}
          <div className="card mb-4">
            <div className="card-header d-flex align-items-center justify-content-between fw-semibold">
              <span>Study Log — Jitter × Blocking × FD</span>
              <select className="form-select form-select-sm w-auto"
                value={studyFilter} onChange={e => setStudyFilter(e.target.value)}>
                <option value="all">All Studies</option>
                <option value="abnormal">Abnormal Only</option>
                <option value="normal">Normal Only</option>
              </select>
            </div>
            <div className="card-body p-0">
              <div style={{ maxHeight: 480, overflowY: 'auto' }}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light sticky-top">
                    <tr>
                      <th>Study ID</th>
                      <th>Patient</th>
                      <th>Muscle</th>
                      <th>Method</th>
                      <th>Pairs</th>
                      <th>MCD (µs)</th>
                      <th>FD</th>
                      <th>Blocking %</th>
                      <th>Diagnosis</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredStudies.map(s => (
                      <tr key={s.study_id}>
                        <td className="small text-muted">{s.study_id}</td>
                        <td>P-{s.patient_id}</td>
                        <td className="small">{s.muscle.split('(')[0].trim()}</td>
                        <td className="small">{s.stimulation_method.split(' ')[0]}</td>
                        <td>{s.n_pairs}</td>
                        <td className={s.mcd_mean_us >= 55 ? 'text-danger fw-bold' : s.mcd_mean_us >= 35 ? 'text-warning' : 'text-success'}>
                          {s.mcd_mean_us}
                        </td>
                        <td className={s.fd_abnormal ? 'text-warning fw-semibold' : ''}>
                          {s.fiber_density}
                        </td>
                        <td className={s.blocking_abnormal ? 'text-danger fw-semibold' : ''}>
                          {s.blocking_pct}%
                        </td>
                        <td>
                          <span className={`badge bg-${DX_COLOR(s.diagnosis)}`} style={{ fontSize: '0.7rem' }}>
                            {s.diagnosis}
                          </span>
                        </td>
                        <td>
                          <span className={`badge bg-${s.overall_abnormal ? 'danger' : 'success'}`}>
                            {s.overall_abnormal ? 'Abnormal' : 'Normal'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── Per Patient ── */}
      {tab === 'per_patient' && (
        <>
          <div className="d-flex align-items-center gap-3 mb-3">
            <span className="small text-muted">Sort by:</span>
            {['max_mcd_us', 'max_fd', 'max_blocking_pct', 'patient_id'].map(k => (
              <button key={k} onClick={() => setPtSort(k)}
                className={`btn btn-sm ${ptSort === k ? 'btn-primary' : 'btn-outline-secondary'}`}>
                {k === 'max_mcd_us' ? 'Max MCD' : k === 'max_fd' ? 'Max FD' : k === 'max_blocking_pct' ? 'Max Blocking' : 'Patient ID'}
              </button>
            ))}
          </div>
          <div className="card">
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th>Sex</th>
                    <th>Age</th>
                    <th>Studies</th>
                    <th>Max MCD (µs)</th>
                    <th>Max FD</th>
                    <th>Max Blocking %</th>
                    <th>Primary Dx</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr key={p.patient_id}>
                      <td className="fw-semibold">P-{p.patient_id}</td>
                      <td>{p.sex}</td>
                      <td>{p.age}</td>
                      <td>{p.n_studies}</td>
                      <td className={p.max_mcd_us >= 55 ? 'text-danger fw-bold' : p.max_mcd_us >= 35 ? 'text-warning' : 'text-success'}>
                        {p.max_mcd_us}
                      </td>
                      <td className={p.max_fd > 1.8 ? 'text-warning fw-semibold' : ''}>{p.max_fd}</td>
                      <td className={p.max_blocking_pct >= 10 ? 'text-danger fw-semibold' : ''}>{p.max_blocking_pct}%</td>
                      <td>
                        <span className={`badge bg-${DX_COLOR(p.primary_diagnosis)}`} style={{ fontSize: '0.7rem' }}>
                          {p.primary_diagnosis}
                        </span>
                      </td>
                      <td>
                        <span className={`badge bg-${p.overall_abnormal ? 'danger' : 'success'}`}>
                          {p.overall_abnormal ? 'Abnormal' : 'Normal'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && defs && (
        <>
          <div className="alert alert-info mb-4">
            <strong>SFEMG — Clinical Overview: </strong>{defs.overview}
          </div>

          {/* Reference values */}
          <div className="row mb-4">
            <div className="col-md-4 mb-3">
              <div className="card h-100">
                <div className="card-header fw-semibold">Normal Reference Values</div>
                <div className="card-body">
                  <RefLine label="MCD (per pair)" normal="< 55 µs" unit="" />
                  <RefLine label="MCD (mean)" normal="< 35 µs" unit="" />
                  <RefLine label="Fiber Density" normal="1.3–1.8" unit="potentials/MU" />
                  <RefLine label="Blocking" normal="< 10%" unit="" />
                </div>
              </div>
            </div>

            {/* Parameters */}
            <div className="col-md-8 mb-3">
              <div className="card h-100">
                <div className="card-header fw-semibold">Parameter Definitions</div>
                <div className="card-body">
                  {(defs.parameters || []).map(p => (
                    <div key={p.name} className="mb-3">
                      <div className="fw-semibold">{p.name}
                        <span className="badge bg-secondary ms-2">{p.unit}</span>
                        <span className="badge bg-success ms-1">Normal: {p.normal}</span>
                      </div>
                      <div className="small text-muted mt-1">{p.definition}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Diagnostic patterns */}
          <div className="card mb-4">
            <div className="card-header fw-semibold">Diagnostic Patterns</div>
            <div className="card-body">
              <div className="row">
                {(defs.diagnostic_patterns || []).map(dp => (
                  <div key={dp.key} className="col-md-4 mb-3">
                    <div className={`card border-${DX_COLOR(dp.pattern)}`}>
                      <div className="card-body py-2">
                        <div className={`fw-semibold text-${DX_COLOR(dp.pattern)}`}>{dp.pattern}</div>
                        <div className="small text-muted mt-1">{dp.description.split(' — ')[1] || dp.description}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Epilepsy relevance */}
          <div className="card mb-4">
            <div className="card-header fw-semibold">🧠 Epilepsy Relevance</div>
            <div className="card-body">
              <div className="row">
                {(defs.epilepsy_relevance || []).map((e, i) => (
                  <div key={i} className="col-md-6 mb-2">
                    <div className="border rounded p-2 h-100">
                      <div className="fw-semibold small">{e.context}</div>
                      <div className="small text-muted mt-1">{e.detail}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Muscles */}
          <div className="card mb-4">
            <div className="card-header fw-semibold">Standard Muscles Studied</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Muscle</th><th>Nerve</th><th>Limb</th><th>Normal MCD (µs)</th></tr>
                </thead>
                <tbody>
                  {(defs.muscles || []).map(m => (
                    <tr key={m.muscle}>
                      <td className="fw-semibold small">{m.muscle}</td>
                      <td className="small text-muted">{m.nerve}</td>
                      <td>{LIMB_ICON(m.limb)} {m.limb}</td>
                      <td className="text-success">~{m.mcd_normal_mean} ± {m.mcd_normal_sd}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* References */}
          <div className="card">
            <div className="card-header fw-semibold">Key References</div>
            <div className="card-body">
              <ol className="mb-0">
                {(defs.key_references || []).map((r, i) => (
                  <li key={i} className="small text-muted mb-1">{r}</li>
                ))}
              </ol>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
