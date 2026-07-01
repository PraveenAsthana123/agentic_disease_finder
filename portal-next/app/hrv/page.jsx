'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s => s === 'Normal' ? 'success' : s === 'Mild' ? 'info' : s === 'Moderate' ? 'warning' : 'danger';
const patColor = p => p === 'normal' ? 'success' : p === 'sympathetic_dominance' ? 'warning' : p === 'parasympathetic_dominance' ? 'info' : 'danger';
const abnBadge = (val, lo, hi) => {
  const abn = val < lo || val > hi;
  return <span className={abn ? 'text-danger fw-bold' : 'text-success'}>{val}{abn ? ' !' : ''}</span>;
};

export default function HRVPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hrv/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/hrv/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/hrv/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'analysis',    label: 'HRV Analysis' },
    { id: 'patients',    label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const sevEntries = ov.severity_distribution ? Object.entries(ov.severity_distribution) : [];
  const patEntries = ov.pattern_distribution ? Object.entries(ov.pattern_distribution) : [];

  return (
    <div>
      <h3>Heart Rate Variability (HRV) / RR Variation</h3>
      <p className="text-muted">Real clinical.db data: autonomic nervous system analysis, time &amp; frequency domain HRV, sympathovagal balance scoring</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Studies',    value: ov.kpis.total_studies,          color: 'primary' },
          { label: 'Abnormal',         value: ov.kpis.abnormal_count,         color: 'danger' },
          { label: 'Abnormal Rate',    value: `${ov.kpis.abnormal_rate_pct}%`, color: ov.kpis.abnormal_rate_pct > 30 ? 'danger' : 'warning' },
          { label: 'Mean SDNN (ms)',   value: ov.kpis.mean_sdnn_ms,           color: ov.kpis.mean_sdnn_ms < 100 ? 'danger' : 'success' },
          { label: 'Mean RMSSD (ms)',  value: ov.kpis.mean_rmssd_ms,          color: ov.kpis.mean_rmssd_ms < 20 ? 'danger' : 'success' },
          { label: 'Mean LF/HF',      value: ov.kpis.mean_lf_hf_ratio,       color: ov.kpis.mean_lf_hf_ratio > 3 ? 'danger' : 'success' },
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
                {sevEntries.map(([sev, count]) => (
                  <div key={sev} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${sevColor(sev)}`}>{sev}</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 18}}>
                        <div className={`progress-bar bg-${sevColor(sev)}`}
                             style={{width: `${ov.kpis.total_studies ? (count / ov.kpis.total_studies * 100) : 0}%`}}>
                          {count}
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
                {patEntries.map(([pat, count]) => (
                  <div key={pat} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${patColor(pat)} me-2`} style={{minWidth: 90, fontSize: '0.7rem'}}>{pat.replace(/_/g, ' ')}</span>
                    <div className="flex-grow-1 mx-1">
                      <div className="progress" style={{height: 18}}>
                        <div className={`progress-bar bg-${patColor(pat)}`}
                             style={{width: `${ov.kpis.total_studies ? (count / ov.kpis.total_studies * 100) : 0}%`}}>
                          {count}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Autonomic Score Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Autonomic Dysfunction Scores</div>
              <div className="card-body">
                <div className="small text-muted mb-2">Score 0 = healthy, 100 = severe dysfunction</div>
                {ov.patient_summary.filter(p => p.autonomic_score > 0).slice(0, 8).map((p, i) => (
                  <div key={i} className="d-flex justify-content-between align-items-center mb-1">
                    <span className="small" style={{maxWidth: 100, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap'}}>{p.name}</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 14}}>
                        <div className={`progress-bar bg-${p.autonomic_score > 50 ? 'danger' : p.autonomic_score > 25 ? 'warning' : 'info'}`}
                             style={{width: `${p.autonomic_score}%`}}>
                          {p.autonomic_score}
                        </div>
                      </div>
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
                      <th>Pattern</th><th>SDNN</th><th>RMSSD</th><th>LF/HF</th><th>Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ov.patient_summary.map((p, i) => (
                      <tr key={i}>
                        <td className="fw-semibold small">{p.name}</td>
                        <td>{p.age}</td>
                        <td className="small">{p.disease}</td>
                        <td><span className={`badge bg-${sevColor(p.severity)}`}>{p.severity}</span></td>
                        <td><span className={`badge bg-${patColor(p.diagnostic_pattern)}`} style={{fontSize:'0.65rem'}}>{p.diagnostic_pattern.replace(/_/g, ' ')}</span></td>
                        <td className={p.sdnn_ms < 100 ? 'text-danger fw-bold' : ''}>{p.sdnn_ms} ms</td>
                        <td className={p.rmssd_ms < 20 ? 'text-danger fw-bold' : ''}>{p.rmssd_ms} ms</td>
                        <td className={p.lf_hf_ratio > 3 ? 'text-danger fw-bold' : ''}>{p.lf_hf_ratio}</td>
                        <td><span className={`badge bg-${p.autonomic_score > 50 ? 'danger' : p.autonomic_score > 25 ? 'warning' : 'success'}`}>{p.autonomic_score}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── HRV Analysis Tab ─────────────────────────────────── */}
      {tab === 'analysis' && bd && (
        <div className="row">
          {/* Time Domain Summary */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Time Domain Parameters</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Parameter</th><th>Mean</th><th>Min</th><th>Max</th><th>Ref</th><th>Abn%</th></tr></thead>
                  <tbody>
                    {bd.time_domain_summary.map((p, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold">{p.parameter}</td>
                        <td>{p.mean} {p.unit}</td>
                        <td className="small text-muted">{p.min}</td>
                        <td className="small text-muted">{p.max}</td>
                        <td className="small">{p.ref_low}-{p.ref_high}</td>
                        <td><span className={p.abnormal_pct > 20 ? 'text-danger fw-bold' : ''}>{p.abnormal_pct}%</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Frequency Domain Summary */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Frequency Domain Parameters</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Parameter</th><th>Mean</th><th>Min</th><th>Max</th><th>Ref</th><th>Abn%</th></tr></thead>
                  <tbody>
                    {bd.freq_domain_summary.map((p, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold">{p.parameter}</td>
                        <td>{p.mean} {p.unit}</td>
                        <td className="small text-muted">{p.min}</td>
                        <td className="small text-muted">{p.max}</td>
                        <td className="small">{p.ref_low}-{p.ref_high}</td>
                        <td><span className={p.abnormal_pct > 20 ? 'text-danger fw-bold' : ''}>{p.abnormal_pct}%</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* SDNN Histogram */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">SDNN Distribution</div>
              <div className="card-body">
                <div className="small text-muted mb-2">Abnormal &lt;100 ms (cardiovascular risk marker)</div>
                {bd.sdnn_histogram.filter(b => b.count > 0).map((b, i) => (
                  <div key={i} className="d-flex align-items-center mb-1">
                    <span className="small" style={{minWidth: 75}}>{b.bin_start}-{b.bin_end} ms</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 16}}>
                        <div className={`progress-bar bg-${b.bin_end <= 100 ? 'danger' : 'success'}`}
                             style={{width: `${Math.max(5, b.count / 30 * 100)}%`}}>
                          {b.count}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* RMSSD Histogram */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">RMSSD Distribution</div>
              <div className="card-body">
                <div className="small text-muted mb-2">Abnormal &lt;20 ms (reduced vagal tone)</div>
                {bd.rmssd_histogram.filter(b => b.count > 0).map((b, i) => (
                  <div key={i} className="d-flex align-items-center mb-1">
                    <span className="small" style={{minWidth: 75}}>{b.bin_start}-{b.bin_end} ms</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 16}}>
                        <div className={`progress-bar bg-${b.bin_end <= 20 ? 'danger' : 'success'}`}
                             style={{width: `${Math.max(5, b.count / 30 * 100)}%`}}>
                          {b.count}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* LF/HF Ratio Histogram */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">LF/HF Ratio Distribution</div>
              <div className="card-body">
                <div className="small text-muted mb-2">Normal 1.0-3.0 | &gt;3.0 sympathetic dominance | &lt;0.5 parasympathetic</div>
                {bd.lf_hf_ratio_histogram.filter(b => b.count > 0).map((b, i) => (
                  <div key={i} className="d-flex align-items-center mb-1">
                    <span className="small" style={{minWidth: 75}}>{b.bin_start}-{b.bin_end}</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 16}}>
                        <div className={`progress-bar bg-${b.bin_start >= 3 ? 'danger' : b.bin_end <= 0.5 ? 'info' : 'success'}`}
                             style={{width: `${Math.max(5, b.count / 30 * 100)}%`}}>
                          {b.count}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Autonomic Score Histogram */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Autonomic Dysfunction Score Distribution</div>
              <div className="card-body">
                <div className="small text-muted mb-2">Score 0-100 | higher = more dysfunction</div>
                {bd.autonomic_score_histogram.filter(b => b.count > 0).map((b, i) => (
                  <div key={i} className="d-flex align-items-center mb-1">
                    <span className="small" style={{minWidth: 60}}>{b.bin_start}-{b.bin_end}</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 16}}>
                        <div className={`progress-bar bg-${b.bin_start >= 50 ? 'danger' : b.bin_start >= 25 ? 'warning' : 'success'}`}
                             style={{width: `${Math.max(5, b.count / 30 * 100)}%`}}>
                          {b.count}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Patient Detail Tab ───────────────────────────────── */}
      {tab === 'patients' && bd && (
        <div>
          {bd.patient_detail_cards.map((pt, i) => (
            <div key={i} className="card mb-2 shadow-sm">
              <div className="card-header d-flex justify-content-between align-items-center py-2"
                   style={{cursor:'pointer'}} onClick={() => setExpandedPt(expandedPt === i ? null : i)}>
                <span>
                  <strong>{pt.name}</strong> <span className="text-muted small">({pt.age}y, {pt.disease})</span>
                  <span className={`badge bg-${sevColor(pt.severity)} ms-2`}>{pt.severity}</span>
                  <span className={`badge bg-${patColor(pt.diagnostic_pattern)} ms-1`} style={{fontSize:'0.65rem'}}>
                    {pt.diagnostic_pattern.replace(/_/g, ' ')}
                  </span>
                  <span className={`badge bg-${pt.autonomic_score > 50 ? 'danger' : pt.autonomic_score > 25 ? 'warning' : 'secondary'} ms-1`}>
                    Score: {pt.autonomic_score}
                  </span>
                </span>
                <span>{expandedPt === i ? '\u25B2' : '\u25BC'}</span>
              </div>
              {expandedPt === i && (
                <div className="card-body">
                  <div className="row">
                    <div className="col-md-6">
                      <h6>Time Domain</h6>
                      <table className="table table-sm">
                        <thead><tr><th>Parameter</th><th>Value</th><th>Ref</th></tr></thead>
                        <tbody>
                          <tr><td>Mean RR</td><td>{pt.time_domain.mean_rr_ms} ms</td><td>700-1100 ms</td></tr>
                          <tr><td>SDNN</td><td className={pt.time_domain.sdnn_ms < 100 ? 'text-danger fw-bold' : ''}>{pt.time_domain.sdnn_ms} ms</td><td>100-200 ms</td></tr>
                          <tr><td>RMSSD</td><td className={pt.time_domain.rmssd_ms < 20 ? 'text-danger fw-bold' : ''}>{pt.time_domain.rmssd_ms} ms</td><td>20-75 ms</td></tr>
                          <tr><td>pNN50</td><td className={pt.time_domain.pnn50_pct < 5 ? 'text-danger fw-bold' : ''}>{pt.time_domain.pnn50_pct}%</td><td>5-40%</td></tr>
                          <tr><td>HR Mean</td><td>{pt.time_domain.hr_mean_bpm} bpm</td><td>60-85 bpm</td></tr>
                          <tr><td>HR Min</td><td>{pt.time_domain.hr_min_bpm} bpm</td><td>-</td></tr>
                          <tr><td>HR Max</td><td>{pt.time_domain.hr_max_bpm} bpm</td><td>-</td></tr>
                        </tbody>
                      </table>
                    </div>
                    <div className="col-md-6">
                      <h6>Frequency Domain</h6>
                      <table className="table table-sm">
                        <thead><tr><th>Parameter</th><th>Value</th><th>Ref</th></tr></thead>
                        <tbody>
                          <tr><td>VLF Power</td><td>{pt.frequency_domain.vlf_power_ms2} ms&sup2;</td><td>300-1500 ms&sup2;</td></tr>
                          <tr><td>LF Power</td><td>{pt.frequency_domain.lf_power_ms2} ms&sup2;</td><td>750-2500 ms&sup2;</td></tr>
                          <tr><td>HF Power</td><td>{pt.frequency_domain.hf_power_ms2} ms&sup2;</td><td>250-1500 ms&sup2;</td></tr>
                          <tr><td>LF/HF Ratio</td><td className={pt.frequency_domain.lf_hf_ratio > 3 ? 'text-danger fw-bold' : ''}>{pt.frequency_domain.lf_hf_ratio}</td><td>1.0-3.0</td></tr>
                          <tr><td>Total Power</td><td>{pt.frequency_domain.total_power_ms2} ms&sup2;</td><td>1500-5000 ms&sup2;</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <div className="small text-muted mt-1">Recording: {pt.recording_type} | Seizures: {pt.seizure_count} | Medications: {pt.med_count}</div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions Tab ──────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">{defs.test_name}</div>
              <div className="card-body">
                <p>{defs.protocol.description}</p>
                <h6>Recording Methods</h6>
                <ul className="small">
                  <li><strong>Short-term:</strong> {defs.protocol.recording_methods.short_term}</li>
                  <li><strong>Long-term:</strong> {defs.protocol.recording_methods.long_term}</li>
                </ul>
                <h6>Indications</h6>
                <ul className="small">
                  {defs.protocol.indications.map((ind, i) => <li key={i}>{ind}</li>)}
                </ul>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Time Domain Parameters</div>
              <div className="card-body">
                {defs.parameters.time_domain.map((p, i) => (
                  <div key={i} className="mb-2">
                    <strong>{p.name}</strong> <span className="text-muted">({p.unit})</span>
                    <div className="small">{p.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Frequency Domain Parameters</div>
              <div className="card-body">
                {defs.parameters.frequency_domain.map((p, i) => (
                  <div key={i} className="mb-2">
                    <strong>{p.name}</strong> <span className="text-muted">({p.unit})</span>
                    <div className="small">{p.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Diagnostic Patterns</div>
              <div className="card-body">
                {defs.diagnostic_patterns.map((p, i) => (
                  <div key={i} className="mb-2">
                    <span className={`badge bg-${patColor(p.pattern)} me-2`}>{p.pattern.replace(/_/g, ' ')}</span>
                    <span className="small">{p.description}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Severity Levels</div>
              <div className="card-body">
                {defs.severity_levels.map((s, i) => (
                  <div key={i} className="mb-2">
                    <span className={`badge bg-${sevColor(s.level)} me-2`}>{s.level}</span>
                    <span className="small">{s.criteria}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Clinical Significance</div>
              <div className="card-body small">{defs.clinical_significance}</div>
            </div>
          </div>

          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Reference Ranges</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Parameter</th><th>Normal Low</th><th>Normal High</th><th>Unit</th></tr></thead>
                  <tbody>
                    {Object.entries(defs.reference_ranges).filter(([k]) => k !== 'notes').map(([k, v]) => (
                      <tr key={k}>
                        <td className="small">{k.replace(/_/g, ' ')}</td>
                        <td>{typeof v === 'object' ? v.low : v}</td>
                        <td>{typeof v === 'object' ? v.high : '-'}</td>
                        <td className="small">{typeof v === 'object' ? v.unit : ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {defs.reference_ranges.notes && <div className="small text-muted p-2">{defs.reference_ranges.notes}</div>}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
