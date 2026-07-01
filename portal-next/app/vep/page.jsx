'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s => s === 'Normal' ? 'success' : s === 'Mild' ? 'info' : s === 'Moderate' ? 'warning' : 'danger';
const patColor = p => p === 'normal' ? 'success' : p === 'optic_neuritis' ? 'info' : p === 'optic_neuropathy' ? 'warning' : 'danger';
const abnBadge = (val, ref, dir) => {
  const abn = dir === 'upper' ? val > ref : val < ref;
  return <span className={abn ? 'text-danger fw-bold' : 'text-success'}>{val}{abn ? ' !' : ''}</span>;
};

export default function VEPPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/vep/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/vep/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/vep/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'analysis',    label: 'VEP Analysis' },
    { id: 'patients',    label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>Visual Evoked Potentials (VEP)</h3>
      <p className="text-muted">Real clinical.db data: visual pathway integrity, P100 peak analysis, pattern-reversal VEP scoring</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Studies',      value: ov.kpis.total_studies,          color: 'primary' },
          { label: 'Abnormal',           value: ov.kpis.abnormal_count,         color: 'danger' },
          { label: 'Abnormal Rate',      value: `${ov.kpis.abnormal_rate_pct}%`, color: ov.kpis.abnormal_rate_pct > 30 ? 'danger' : 'warning' },
          { label: 'Mean P100 (ms)',     value: ov.kpis.mean_p100_latency_ms,   color: ov.kpis.mean_p100_latency_ms > 115 ? 'danger' : 'success' },
          { label: 'Mean P100 Amp',      value: `${ov.kpis.mean_p100_amplitude_uv} \u00b5V`, color: ov.kpis.mean_p100_amplitude_uv < 3 ? 'danger' : 'success' },
          { label: 'Eyes/Study',         value: ov.kpis.eyes_per_study,         color: 'dark' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value}</div>
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
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ─────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Severity Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Severity Distribution</div>
              <div className="card-body">
                {ov.severity_distribution.map(s => (
                  <div key={s.severity} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${sevColor(s.severity)}`}>{s.severity}</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 18}}>
                        <div className={`progress-bar bg-${sevColor(s.severity)}`}
                             style={{width: `${ov.kpis.total_studies ? (s.count / ov.kpis.total_studies * 100) : 0}%`}}>
                          {s.count}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Diagnostic Pattern Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Diagnostic Patterns</div>
              <div className="card-body">
                {ov.pattern_distribution.map(p => (
                  <div key={p.pattern} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${patColor(p.pattern)} me-2`} style={{minWidth: 90, fontSize: '0.7rem'}}>{p.label}</span>
                    <div className="flex-grow-1 mx-1">
                      <div className="progress" style={{height: 18}}>
                        <div className={`progress-bar bg-${patColor(p.pattern)}`}
                             style={{width: `${ov.kpis.total_studies ? (p.count / ov.kpis.total_studies * 100) : 0}%`}}>
                          {p.count}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Per-Eye Abnormality Rates */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Abnormality by Eye</div>
              <div className="card-body">
                {ov.eye_abnormality_rates.map(e => (
                  <div key={e.eye} className="mb-3">
                    <div className="d-flex justify-content-between small">
                      <span className="fw-semibold">{e.eye} Eye</span>
                      <span>{e.abnormal}/{e.total} ({e.rate_pct}%)</span>
                    </div>
                    <div className="progress" style={{height: 14}}>
                      <div className={`progress-bar bg-${e.rate_pct > 30 ? 'danger' : e.rate_pct > 15 ? 'warning' : 'success'}`}
                           style={{width: `${e.rate_pct}%`}} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Patient Summary Table */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Patient Summary</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr>
                      <th>Patient</th><th>Age</th><th>Disease</th><th>Severity</th>
                      <th>Pattern</th><th>Inter-eye Diff</th><th>Abnormal Eyes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ov.patient_summary.map((p, i) => (
                      <tr key={i}>
                        <td className="fw-semibold small">{p.name}</td>
                        <td>{p.age}</td>
                        <td className="small">{p.disease}</td>
                        <td><span className={`badge bg-${sevColor(p.overall_severity)}`}>{p.overall_severity}</span></td>
                        <td><span className={`badge bg-${patColor(p.diagnostic_pattern)}`} style={{fontSize:'0.65rem'}}>{p.diagnostic_pattern.replace(/_/g, ' ')}</span></td>
                        <td>{p.inter_eye_diff_ms} ms</td>
                        <td>{p.abnormal_eyes}/{p.total_eyes}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── VEP Analysis Tab ─────────────────────────────────── */}
      {tab === 'analysis' && bd && (
        <div className="row">
          {/* Left Eye Summary */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">{bd.left_summary.eye}</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Parameter</th><th>Mean</th><th>Reference</th><th>Status</th></tr></thead>
                  <tbody>
                    <tr><td>N75 Latency</td><td>{bd.left_summary.mean_n75_ms} ms</td><td>&le;{bd.left_summary.refs.n75_latency_upper} ms</td>
                      <td>{bd.left_summary.mean_n75_ms > bd.left_summary.refs.n75_latency_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>P100 Latency</td><td>{bd.left_summary.mean_p100_ms} ms</td><td>&le;{bd.left_summary.refs.p100_latency_upper} ms</td>
                      <td>{bd.left_summary.mean_p100_ms > bd.left_summary.refs.p100_latency_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>N145 Latency</td><td>{bd.left_summary.mean_n145_ms} ms</td><td>&le;{bd.left_summary.refs.n145_latency_upper} ms</td>
                      <td>{bd.left_summary.mean_n145_ms > bd.left_summary.refs.n145_latency_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>P100 Amplitude</td><td>{bd.left_summary.mean_p100_amp_uv} &micro;V</td><td>&ge;{bd.left_summary.refs.p100_amplitude_lower} &micro;V</td>
                      <td>{bd.left_summary.mean_p100_amp_uv < bd.left_summary.refs.p100_amplitude_lower ? <span className="badge bg-danger">Low</span> : <span className="badge bg-success">OK</span>}</td></tr>
                  </tbody>
                </table>
                <div className="mt-2 small text-muted">Abnormal: {bd.left_summary.abnormal_pct}% | n={bd.left_summary.count}</div>
              </div>
            </div>
          </div>

          {/* Right Eye Summary */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">{bd.right_summary.eye}</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Parameter</th><th>Mean</th><th>Reference</th><th>Status</th></tr></thead>
                  <tbody>
                    <tr><td>N75 Latency</td><td>{bd.right_summary.mean_n75_ms} ms</td><td>&le;{bd.right_summary.refs.n75_latency_upper} ms</td>
                      <td>{bd.right_summary.mean_n75_ms > bd.right_summary.refs.n75_latency_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>P100 Latency</td><td>{bd.right_summary.mean_p100_ms} ms</td><td>&le;{bd.right_summary.refs.p100_latency_upper} ms</td>
                      <td>{bd.right_summary.mean_p100_ms > bd.right_summary.refs.p100_latency_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>N145 Latency</td><td>{bd.right_summary.mean_n145_ms} ms</td><td>&le;{bd.right_summary.refs.n145_latency_upper} ms</td>
                      <td>{bd.right_summary.mean_n145_ms > bd.right_summary.refs.n145_latency_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>P100 Amplitude</td><td>{bd.right_summary.mean_p100_amp_uv} &micro;V</td><td>&ge;{bd.right_summary.refs.p100_amplitude_lower} &micro;V</td>
                      <td>{bd.right_summary.mean_p100_amp_uv < bd.right_summary.refs.p100_amplitude_lower ? <span className="badge bg-danger">Low</span> : <span className="badge bg-success">OK</span>}</td></tr>
                  </tbody>
                </table>
                <div className="mt-2 small text-muted">Abnormal: {bd.right_summary.abnormal_pct}% | n={bd.right_summary.count}</div>
              </div>
            </div>
          </div>

          {/* P100 Latency Histogram */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">P100 Latency Distribution</div>
              <div className="card-body">
                {bd.p100_latency_histogram.map(b => {
                  const maxC = Math.max(...bd.p100_latency_histogram.map(x => x.count), 1);
                  const isAbn = b.range.includes('115') || b.range.includes('130') || b.range.includes('150');
                  return (
                    <div key={b.range} className="d-flex align-items-center mb-1">
                      <span className="small" style={{width: 60}}>{b.range} ms</span>
                      <div className="flex-grow-1">
                        <div className="progress" style={{height: 16}}>
                          <div className={`progress-bar bg-${isAbn ? 'danger' : 'primary'}`}
                               style={{width: `${(b.count / maxC) * 100}%`}}>
                            {b.count > 0 ? b.count : ''}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div className="small text-muted mt-1">Reference: P100 &le;115.0 ms (red = abnormal range)</div>
              </div>
            </div>
          </div>

          {/* P100 Amplitude Histogram */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">P100 Amplitude Distribution</div>
              <div className="card-body">
                {bd.p100_amplitude_histogram.map(b => {
                  const maxC = Math.max(...bd.p100_amplitude_histogram.map(x => x.count), 1);
                  const isLow = b.range === '<1' || b.range === '1-2' || b.range === '2-3';
                  return (
                    <div key={b.range} className="d-flex align-items-center mb-1">
                      <span className="small" style={{width: 55}}>{b.range} &micro;V</span>
                      <div className="flex-grow-1">
                        <div className="progress" style={{height: 16}}>
                          <div className={`progress-bar bg-${isLow ? 'warning' : 'success'}`}
                               style={{width: `${(b.count / maxC) * 100}%`}}>
                            {b.count > 0 ? b.count : ''}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div className="small text-muted mt-1">Reference: P100 amplitude &ge;3.0 &micro;V (yellow = low range)</div>
              </div>
            </div>
          </div>

          {/* Inter-Eye Difference Histogram */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Inter-Eye P100 Difference</div>
              <div className="card-body">
                {bd.inter_eye_histogram.map(b => {
                  const maxC = Math.max(...bd.inter_eye_histogram.map(x => x.count), 1);
                  const isAbn = b.range.includes('8-12') || b.range === '12+';
                  return (
                    <div key={b.range} className="d-flex align-items-center mb-1">
                      <span className="small" style={{width: 50}}>{b.range} ms</span>
                      <div className="flex-grow-1">
                        <div className="progress" style={{height: 16}}>
                          <div className={`progress-bar bg-${isAbn ? 'danger' : 'info'}`}
                               style={{width: `${(b.count / maxC) * 100}%`}}>
                            {b.count > 0 ? b.count : ''}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div className="small text-muted mt-1">Abnormal: inter-eye difference &gt;8.0 ms (red = abnormal)</div>
              </div>
            </div>
          </div>

          {/* Left vs Right Eye Comparison */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Left vs Right Eye Comparison</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr><th>Eye</th><th>Total</th><th>Abnormal</th><th>Abnormal %</th><th>Mean P100</th><th>Mean Amp</th></tr>
                  </thead>
                  <tbody>
                    {bd.eye_comparison.map((e, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{e.eye}</td>
                        <td>{e.total}</td>
                        <td className={e.abnormal > 0 ? 'text-danger fw-bold' : ''}>{e.abnormal}</td>
                        <td><span className={`badge bg-${e.abnormal_pct > 30 ? 'danger' : e.abnormal_pct > 15 ? 'warning' : 'success'}`}>{e.abnormal_pct}%</span></td>
                        <td>{e.mean_p100_ms} ms</td>
                        <td>{e.mean_p100_amp_uv} &micro;V</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Patient Detail Tab ───────────────────────────────── */}
      {tab === 'patients' && bd && (
        <div>
          {bd.patient_details.map((pt, i) => (
            <div key={i} className="card shadow-sm mb-2">
              <div className="card-header d-flex justify-content-between align-items-center py-2"
                   style={{cursor: 'pointer'}}
                   onClick={() => setExpandedPt(expandedPt === i ? null : i)}>
                <div>
                  <span className="fw-bold">{pt.name}</span>
                  <span className="text-muted ms-2 small">Age {pt.age} | {pt.disease}</span>
                </div>
                <div>
                  <span className={`badge bg-${sevColor(pt.overall_severity)} me-1`}>{pt.overall_severity}</span>
                  <span className={`badge bg-${patColor(pt.diagnostic_pattern)}`} style={{fontSize:'0.65rem'}}>{pt.diagnostic_pattern.replace(/_/g, ' ')}</span>
                  {pt.inter_eye_abnormal && <span className="badge bg-danger ms-1" style={{fontSize:'0.6rem'}}>Asymmetry</span>}
                  <span className="ms-2 small">{expandedPt === i ? '\u25B2' : '\u25BC'}</span>
                </div>
              </div>
              {expandedPt === i && (
                <div className="card-body">
                  <div className="mb-2 small">
                    <strong>Inter-eye P100 difference:</strong> {pt.inter_eye_diff_ms} ms
                    {pt.inter_eye_abnormal ? <span className="text-danger fw-bold"> (Abnormal &gt;8 ms)</span> : <span className="text-success"> (Normal)</span>}
                  </div>
                  <div className="row">
                    {/* Left Eye */}
                    {pt.left && (
                      <div className="col-md-6">
                        <h6>Left Eye (OS)</h6>
                        <table className="table table-sm table-bordered">
                          <thead><tr><th>Parameter</th><th>Value</th><th>Ref</th></tr></thead>
                          <tbody>
                            <tr><td>N75 Latency</td><td>{abnBadge(pt.left.n75_latency_ms, pt.left.n75_ref, 'upper')}</td><td>&le;{pt.left.n75_ref} ms</td></tr>
                            <tr><td>P100 Latency</td><td>{abnBadge(pt.left.p100_latency_ms, pt.left.p100_ref, 'upper')}</td><td>&le;{pt.left.p100_ref} ms</td></tr>
                            <tr><td>N145 Latency</td><td>{abnBadge(pt.left.n145_latency_ms, pt.left.n145_ref, 'upper')}</td><td>&le;{pt.left.n145_ref} ms</td></tr>
                            <tr><td>P100 Amplitude</td><td>{abnBadge(pt.left.p100_amplitude_uv, pt.left.p100_amp_ref, 'lower')}</td><td>&ge;{pt.left.p100_amp_ref} &micro;V</td></tr>
                          </tbody>
                        </table>
                        <span className={`badge bg-${sevColor(pt.left.severity)}`}>{pt.left.severity}</span>
                      </div>
                    )}
                    {/* Right Eye */}
                    {pt.right && (
                      <div className="col-md-6">
                        <h6>Right Eye (OD)</h6>
                        <table className="table table-sm table-bordered">
                          <thead><tr><th>Parameter</th><th>Value</th><th>Ref</th></tr></thead>
                          <tbody>
                            <tr><td>N75 Latency</td><td>{abnBadge(pt.right.n75_latency_ms, pt.right.n75_ref, 'upper')}</td><td>&le;{pt.right.n75_ref} ms</td></tr>
                            <tr><td>P100 Latency</td><td>{abnBadge(pt.right.p100_latency_ms, pt.right.p100_ref, 'upper')}</td><td>&le;{pt.right.p100_ref} ms</td></tr>
                            <tr><td>N145 Latency</td><td>{abnBadge(pt.right.n145_latency_ms, pt.right.n145_ref, 'upper')}</td><td>&le;{pt.right.n145_ref} ms</td></tr>
                            <tr><td>P100 Amplitude</td><td>{abnBadge(pt.right.p100_amplitude_uv, pt.right.p100_amp_ref, 'lower')}</td><td>&ge;{pt.right.p100_amp_ref} &micro;V</td></tr>
                          </tbody>
                        </table>
                        <span className={`badge bg-${sevColor(pt.right.severity)}`}>{pt.right.severity}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions Tab ──────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Protocol</div>
              <div className="card-body small">
                <p>{defs.protocol.description}</p>
                <h6>Stimulus</h6>
                <ul>
                  <li><strong>Type:</strong> {defs.protocol.stimulus.type}</li>
                  <li><strong>Check size:</strong> {defs.protocol.stimulus.check_size}</li>
                  <li><strong>Reversal rate:</strong> {defs.protocol.stimulus.reversal_rate}</li>
                  <li><strong>Field:</strong> {defs.protocol.stimulus.field}</li>
                  <li><strong>Luminance:</strong> {defs.protocol.stimulus.luminance}</li>
                </ul>
                <h6>Recording</h6>
                <ul>
                  <li><strong>Active:</strong> {defs.protocol.recording.active_electrode}</li>
                  <li><strong>Reference:</strong> {defs.protocol.recording.reference_electrode}</li>
                  <li><strong>Filter:</strong> {defs.protocol.recording.filter}</li>
                  <li><strong>Epoch:</strong> {defs.protocol.recording.epoch}</li>
                  <li><strong>Averages:</strong> {defs.protocol.recording.averages}</li>
                </ul>
                <h6>Indications</h6>
                <ul>{defs.protocol.indications.map((ind, i) => <li key={i}>{ind}</li>)}</ul>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Parameters ({defs.parameters.length})</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Parameter</th><th>Unit</th><th>Description</th></tr></thead>
                  <tbody>
                    {defs.parameters.map((p, i) => (
                      <tr key={i}><td className="fw-semibold small">{p.name}</td><td>{p.unit}</td><td className="small">{p.description}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Reference Ranges</div>
              <div className="card-body small">
                <ul>
                  {Object.entries(defs.reference_ranges).filter(([k]) => k !== 'notes').map(([k, v]) => (
                    <li key={k}><strong>{k.replace(/_/g, ' ')}:</strong> {v}</li>
                  ))}
                </ul>
                <p className="text-muted">{defs.reference_ranges.notes}</p>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Diagnostic Patterns</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <tbody>
                    {defs.diagnostic_patterns.map((p, i) => (
                      <tr key={i}><td><span className={`badge bg-${patColor(p.pattern)}`}>{p.pattern.replace(/_/g, ' ')}</span></td><td className="small">{p.description}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm mt-3">
              <div className="card-header fw-bold">Severity Levels</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <tbody>
                    {defs.severity_levels.map((s, i) => (
                      <tr key={i}><td><span className={`badge bg-${sevColor(s.level)}`}>{s.level}</span></td><td className="small">{s.criteria}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card shadow-sm mt-3">
              <div className="card-header fw-bold">Clinical Significance</div>
              <div className="card-body small">{defs.clinical_significance}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
