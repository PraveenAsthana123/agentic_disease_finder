'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s => s === 'Normal' ? 'success' : s === 'Mild' ? 'info' : s === 'Moderate' ? 'warning' : 'danger';
const patColor = p => p === 'normal' ? 'success' : p === 'peripheral_lesion' ? 'info' : p === 'cervical_cord_lesion' ? 'warning' : 'danger';
const abnBadge = (val, ref, dir) => {
  const abn = dir === 'upper' ? val > ref : val < ref;
  return <span className={abn ? 'text-danger fw-bold' : 'text-success'}>{val}{abn ? ' !' : ''}</span>;
};

export default function SSEPPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/ssep/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/ssep/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/ssep/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'analysis',    label: 'Pathway Analysis' },
    { id: 'patients',    label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>Somatosensory Evoked Potentials (SSEP)</h3>
      <p className="text-muted">Real clinical.db data: dorsal column-medial lemniscal pathway integrity, N20/P37 peak analysis, sensory pathway scoring</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Studies',     value: ov.kpis.total_studies,       color: 'primary' },
          { label: 'Abnormal',          value: ov.kpis.abnormal_count,      color: 'danger' },
          { label: 'Abnormal Rate',     value: `${ov.kpis.abnormal_rate_pct}%`, color: ov.kpis.abnormal_rate_pct > 30 ? 'danger' : 'warning' },
          { label: 'Mean N20 (ms)',     value: ov.kpis.mean_n20_latency_ms, color: ov.kpis.mean_n20_latency_ms > 22 ? 'danger' : 'success' },
          { label: 'Mean P37 (ms)',     value: ov.kpis.mean_p37_latency_ms, color: ov.kpis.mean_p37_latency_ms > 45 ? 'danger' : 'success' },
          { label: 'Limbs/Study',       value: ov.kpis.limbs_per_study,     color: 'dark' },
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

          {/* Per-Limb Abnormality Rates */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Abnormality by Limb</div>
              <div className="card-body">
                {ov.limb_abnormality_rates.map(l => (
                  <div key={l.limb} className="mb-3">
                    <div className="d-flex justify-content-between small">
                      <span className="fw-semibold">{l.limb} Limb</span>
                      <span>{l.abnormal}/{l.total} ({l.rate_pct}%)</span>
                    </div>
                    <div className="progress" style={{height: 14}}>
                      <div className={`progress-bar bg-${l.rate_pct > 30 ? 'danger' : l.rate_pct > 15 ? 'warning' : 'success'}`}
                           style={{width: `${l.rate_pct}%`}} />
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
                      <th>Pattern</th><th>Abnormal Limbs</th>
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
                        <td>{p.abnormal_limbs}/{p.total_limbs}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Pathway Analysis Tab ─────────────────────────────── */}
      {tab === 'analysis' && bd && (
        <div className="row">
          {/* Upper Limb Summary */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Upper Limb — Median Nerve</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Parameter</th><th>Mean</th><th>Reference</th><th>Status</th></tr></thead>
                  <tbody>
                    <tr><td>N9 Latency</td><td>{bd.upper_summary.mean_n9_ms} ms</td><td>≤{bd.upper_summary.refs.n9_latency_upper} ms</td>
                      <td>{bd.upper_summary.mean_n9_ms > bd.upper_summary.refs.n9_latency_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>N13 Latency</td><td>{bd.upper_summary.mean_n13_ms} ms</td><td>≤{bd.upper_summary.refs.n13_latency_upper} ms</td>
                      <td>{bd.upper_summary.mean_n13_ms > bd.upper_summary.refs.n13_latency_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>N20 Latency</td><td>{bd.upper_summary.mean_n20_ms} ms</td><td>≤{bd.upper_summary.refs.n20_latency_upper} ms</td>
                      <td>{bd.upper_summary.mean_n20_ms > bd.upper_summary.refs.n20_latency_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>N20 Amplitude</td><td>{bd.upper_summary.mean_n20_amp_uv} µV</td><td>≥{bd.upper_summary.refs.n20_amplitude_lower} µV</td>
                      <td>{bd.upper_summary.mean_n20_amp_uv < bd.upper_summary.refs.n20_amplitude_lower ? <span className="badge bg-danger">Low</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>N9-N13 IPL</td><td>{bd.upper_summary.mean_n9_n13_ipl_ms} ms</td><td>≤{bd.upper_summary.refs.n9_n13_ipl_upper} ms</td>
                      <td>{bd.upper_summary.mean_n9_n13_ipl_ms > bd.upper_summary.refs.n9_n13_ipl_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>N13-N20 CCT</td><td>{bd.upper_summary.mean_n13_n20_ipl_ms} ms</td><td>≤{bd.upper_summary.refs.n13_n20_ipl_upper} ms</td>
                      <td>{bd.upper_summary.mean_n13_n20_ipl_ms > bd.upper_summary.refs.n13_n20_ipl_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                  </tbody>
                </table>
                <div className="mt-2 small text-muted">Abnormal: {bd.upper_summary.abnormal_pct}% | n={bd.upper_summary.count}</div>
              </div>
            </div>
          </div>

          {/* Lower Limb Summary */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Lower Limb — Posterior Tibial Nerve</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Parameter</th><th>Mean</th><th>Reference</th><th>Status</th></tr></thead>
                  <tbody>
                    <tr><td>N22 Latency</td><td>{bd.lower_summary.mean_n22_ms} ms</td><td>≤{bd.lower_summary.refs.n22_latency_upper} ms</td>
                      <td>{bd.lower_summary.mean_n22_ms > bd.lower_summary.refs.n22_latency_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>P37 Latency</td><td>{bd.lower_summary.mean_p37_ms} ms</td><td>≤{bd.lower_summary.refs.p37_latency_upper} ms</td>
                      <td>{bd.lower_summary.mean_p37_ms > bd.lower_summary.refs.p37_latency_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>P37 Amplitude</td><td>{bd.lower_summary.mean_p37_amp_uv} µV</td><td>≥{bd.lower_summary.refs.p37_amplitude_lower} µV</td>
                      <td>{bd.lower_summary.mean_p37_amp_uv < bd.lower_summary.refs.p37_amplitude_lower ? <span className="badge bg-danger">Low</span> : <span className="badge bg-success">OK</span>}</td></tr>
                    <tr><td>N22-P37 CCT</td><td>{bd.lower_summary.mean_n22_p37_ipl_ms} ms</td><td>≤{bd.lower_summary.refs.n22_p37_ipl_upper} ms</td>
                      <td>{bd.lower_summary.mean_n22_p37_ipl_ms > bd.lower_summary.refs.n22_p37_ipl_upper ? <span className="badge bg-danger">High</span> : <span className="badge bg-success">OK</span>}</td></tr>
                  </tbody>
                </table>
                <div className="mt-2 small text-muted">Abnormal: {bd.lower_summary.abnormal_pct}% | n={bd.lower_summary.count}</div>
              </div>
            </div>
          </div>

          {/* N20 Latency Histogram */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">N20 Latency Distribution (Upper Limb)</div>
              <div className="card-body">
                {bd.n20_latency_histogram.map(b => {
                  const maxC = Math.max(...bd.n20_latency_histogram.map(x => x.count), 1);
                  const isAbn = b.range.includes('22') || b.range.includes('25') || b.range.includes('30');
                  return (
                    <div key={b.range} className="d-flex align-items-center mb-1">
                      <span className="small" style={{width: 55}}>{b.range} ms</span>
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
                <div className="small text-muted mt-1">Reference: N20 ≤22.0 ms (red = abnormal range)</div>
              </div>
            </div>
          </div>

          {/* P37 Latency Histogram */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">P37 Latency Distribution (Lower Limb)</div>
              <div className="card-body">
                {bd.p37_latency_histogram.map(b => {
                  const maxC = Math.max(...bd.p37_latency_histogram.map(x => x.count), 1);
                  const isAbn = b.range.includes('45') || b.range.includes('55');
                  return (
                    <div key={b.range} className="d-flex align-items-center mb-1">
                      <span className="small" style={{width: 55}}>{b.range} ms</span>
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
                <div className="small text-muted mt-1">Reference: P37 ≤45.0 ms (red = abnormal range)</div>
              </div>
            </div>
          </div>

          {/* Upper vs Lower Limb Comparison */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Upper vs Lower Limb Comparison</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr><th>Limb</th><th>Total</th><th>Abnormal</th><th>Abnormal %</th><th>Mean Cortical Latency</th><th>Mean CCT</th></tr>
                  </thead>
                  <tbody>
                    {bd.limb_comparison.map((l, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{l.limb}</td>
                        <td>{l.total}</td>
                        <td className={l.abnormal > 0 ? 'text-danger fw-bold' : ''}>{l.abnormal}</td>
                        <td><span className={`badge bg-${l.abnormal_pct > 30 ? 'danger' : l.abnormal_pct > 15 ? 'warning' : 'success'}`}>{l.abnormal_pct}%</span></td>
                        <td>{l.mean_cortical_latency_ms} ms</td>
                        <td>{l.mean_cct_ms} ms</td>
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
                  <span className="ms-2 small">{expandedPt === i ? '\u25B2' : '\u25BC'}</span>
                </div>
              </div>
              {expandedPt === i && (
                <div className="card-body">
                  <div className="row">
                    {/* Upper Limb */}
                    {pt.upper && (
                      <div className="col-md-6">
                        <h6>Upper Limb (Median Nerve)</h6>
                        <table className="table table-sm table-bordered">
                          <thead><tr><th>Parameter</th><th>Value</th><th>Ref</th></tr></thead>
                          <tbody>
                            <tr><td>N9 Latency</td><td>{abnBadge(pt.upper.n9_latency_ms, pt.upper.n9_ref, 'upper')}</td><td>≤{pt.upper.n9_ref} ms</td></tr>
                            <tr><td>N13 Latency</td><td>{abnBadge(pt.upper.n13_latency_ms, pt.upper.n13_ref, 'upper')}</td><td>≤{pt.upper.n13_ref} ms</td></tr>
                            <tr><td>N20 Latency</td><td>{abnBadge(pt.upper.n20_latency_ms, pt.upper.n20_ref, 'upper')}</td><td>≤{pt.upper.n20_ref} ms</td></tr>
                            <tr><td>N20 Amplitude</td><td>{abnBadge(pt.upper.n20_amplitude_uv, pt.upper.n20_amp_ref, 'lower')}</td><td>≥{pt.upper.n20_amp_ref} µV</td></tr>
                            <tr><td>N9-N13 IPL</td><td>{abnBadge(pt.upper.n9_n13_ipl_ms, pt.upper.n9_n13_ipl_ref, 'upper')}</td><td>≤{pt.upper.n9_n13_ipl_ref} ms</td></tr>
                            <tr><td>N13-N20 CCT</td><td>{abnBadge(pt.upper.n13_n20_ipl_ms, pt.upper.n13_n20_ipl_ref, 'upper')}</td><td>≤{pt.upper.n13_n20_ipl_ref} ms</td></tr>
                          </tbody>
                        </table>
                        <span className={`badge bg-${sevColor(pt.upper.severity)}`}>{pt.upper.severity}</span>
                      </div>
                    )}
                    {/* Lower Limb */}
                    {pt.lower && (
                      <div className="col-md-6">
                        <h6>Lower Limb (Posterior Tibial Nerve)</h6>
                        <table className="table table-sm table-bordered">
                          <thead><tr><th>Parameter</th><th>Value</th><th>Ref</th></tr></thead>
                          <tbody>
                            <tr><td>N22 Latency</td><td>{abnBadge(pt.lower.n22_latency_ms, pt.lower.n22_ref, 'upper')}</td><td>≤{pt.lower.n22_ref} ms</td></tr>
                            <tr><td>P37 Latency</td><td>{abnBadge(pt.lower.p37_latency_ms, pt.lower.p37_ref, 'upper')}</td><td>≤{pt.lower.p37_ref} ms</td></tr>
                            <tr><td>P37 Amplitude</td><td>{abnBadge(pt.lower.p37_amplitude_uv, pt.lower.p37_amp_ref, 'lower')}</td><td>≥{pt.lower.p37_amp_ref} µV</td></tr>
                            <tr><td>N22-P37 CCT</td><td>{abnBadge(pt.lower.n22_p37_ipl_ms, pt.lower.n22_p37_ipl_ref, 'upper')}</td><td>≤{pt.lower.n22_p37_ipl_ref} ms</td></tr>
                          </tbody>
                        </table>
                        <span className={`badge bg-${sevColor(pt.lower.severity)}`}>{pt.lower.severity}</span>
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
                <h6>Upper Limb</h6>
                <p><strong>Stimulus:</strong> {defs.protocol.upper_limb.stimulus_site}</p>
                <p><strong>Recording:</strong></p>
                <ul>{defs.protocol.upper_limb.recording_sites.map((r, i) => <li key={i}>{r}</li>)}</ul>
                <h6>Lower Limb</h6>
                <p><strong>Stimulus:</strong> {defs.protocol.lower_limb.stimulus_site}</p>
                <p><strong>Recording:</strong></p>
                <ul>{defs.protocol.lower_limb.recording_sites.map((r, i) => <li key={i}>{r}</li>)}</ul>
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
                <h6>Upper Limb (Median Nerve)</h6>
                <ul>
                  {Object.entries(defs.reference_ranges.upper_limb).map(([k, v]) => (
                    <li key={k}><strong>{k.replace(/_/g, ' ')}:</strong> {v}</li>
                  ))}
                </ul>
                <h6>Lower Limb (Posterior Tibial Nerve)</h6>
                <ul>
                  {Object.entries(defs.reference_ranges.lower_limb).map(([k, v]) => (
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
