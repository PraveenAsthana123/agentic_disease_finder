'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s => s === 'Normal' ? 'success' : s === 'Mild' ? 'info' : s === 'Moderate' ? 'warning' : 'danger';
const patColor = p => p === 'normal' || p === 'normotensive' ? 'success' : p === 'arrhythmia_burden' || p === 'sustained_hypertension' ? 'danger' : p === 'autonomic_dysregulation' ? 'warning' : 'info';
const dipColor = d => d === 'normal_dipper' ? 'success' : d === 'extreme_dipper' ? 'info' : d === 'non_dipper' ? 'warning' : 'danger';

export default function ABPMHolterPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/abpm-holter/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/abpm-holter/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/abpm-holter/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'analysis',    label: 'Combined Analysis' },
    { id: 'patients',    label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const sevEntries = ov.severity_distribution ? Object.entries(ov.severity_distribution) : [];
  const patEntries = ov.pattern_distribution ? Object.entries(ov.pattern_distribution) : [];
  const dipEntries = ov.dipping_distribution ? Object.entries(ov.dipping_distribution) : [];

  return (
    <div>
      <h3>ABPM / Holter Combined — Cardiac-Autonomic Monitoring</h3>
      <p className="text-muted small">
        24h ambulatory blood pressure + Holter ECG combined analysis · dipping status · arrhythmia burden ·
        QTc monitoring · SUDEP risk stratification · autonomic dysfunction in epilepsy.
        Source: <code>abpm_holter</code> table (clinical.db)
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Studies',    value: ov.kpis.total_studies,              color: 'primary' },
          { label: 'Abnormal',         value: ov.kpis.abnormal_count,             color: 'danger' },
          { label: 'Abnormal Rate',    value: `${ov.kpis.abnormal_rate_pct}%`,    color: ov.kpis.abnormal_rate_pct > 30 ? 'danger' : 'warning' },
          { label: 'Mean 24h SBP',     value: `${ov.kpis.mean_systolic_24h} mmHg`, color: ov.kpis.mean_systolic_24h > 130 ? 'danger' : 'success' },
          { label: 'Mean 24h DBP',     value: `${ov.kpis.mean_diastolic_24h} mmHg`, color: ov.kpis.mean_diastolic_24h > 80 ? 'danger' : 'success' },
          { label: 'Mean QTc',         value: `${ov.kpis.mean_qtc_ms} ms`,        color: ov.kpis.mean_qtc_ms > 450 ? 'danger' : 'success' },
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
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Severity Distribution</div>
              <div className="card-body">
                {sevEntries.map(([sev, count]) => (
                  <div key={sev} className="d-flex align-items-center mb-2">
                    <span className={`badge bg-${sevColor(sev)} me-2`} style={{minWidth: 70}}>{sev}</span>
                    <div className="flex-grow-1 me-2">
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

          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Diagnostic Patterns</div>
              <div className="card-body">
                {patEntries.map(([pat, count]) => count > 0 && (
                  <div key={pat} className="d-flex align-items-center mb-2">
                    <span className={`badge bg-${patColor(pat)} me-1`} style={{minWidth: 90, fontSize:'0.7rem'}}>{pat.replace(/_/g,' ')}</span>
                    <div className="flex-grow-1 me-1">
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

          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Nocturnal Dipping Status</div>
              <div className="card-body">
                {dipEntries.map(([cat, count]) => (
                  <div key={cat} className="d-flex align-items-center mb-2">
                    <span className={`badge bg-${dipColor(cat)} me-1`} style={{minWidth: 90, fontSize:'0.7rem'}}>{cat.replace(/_/g,' ')}</span>
                    <div className="flex-grow-1 me-1">
                      <div className="progress" style={{height: 18}}>
                        <div className={`progress-bar bg-${dipColor(cat)}`}
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

          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Patient Summary</div>
              <div className="card-body p-0" style={{overflowX:'auto'}}>
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr>
                      <th>Patient</th><th>Age</th><th>Severity</th><th>Pattern</th>
                      <th>24h SBP</th><th>24h DBP</th><th>Dipping</th><th>QTc</th><th>PVCs</th><th>Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ov.patient_summary.map((p, i) => (
                      <tr key={i}>
                        <td className="fw-semibold small">{p.name}</td>
                        <td>{p.age}</td>
                        <td><span className={`badge bg-${sevColor(p.severity)}`}>{p.severity}</span></td>
                        <td><span className={`badge bg-${patColor(p.diagnostic_pattern)}`} style={{fontSize:'0.65rem'}}>{p.pattern_label || p.diagnostic_pattern.replace(/_/g,' ')}</span></td>
                        <td className={p.systolic_24h > 130 ? 'text-danger fw-bold' : ''}>{p.systolic_24h}</td>
                        <td className={p.diastolic_24h > 80 ? 'text-danger fw-bold' : ''}>{p.diastolic_24h}</td>
                        <td><span className={`badge bg-${dipColor(p.dipping_category)}`} style={{fontSize:'0.65rem'}}>{p.dipping_pct}%</span></td>
                        <td className={p.qtc_ms > 450 ? 'text-danger fw-bold' : ''}>{p.qtc_ms} ms</td>
                        <td className={p.pvc_count > 500 ? 'text-danger fw-bold' : ''}>{p.pvc_count}</td>
                        <td><span className={`badge bg-${p.cardiac_score > 50 ? 'danger' : p.cardiac_score > 25 ? 'warning' : 'success'}`}>{p.cardiac_score}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Combined Analysis Tab ────────────────────────────── */}
      {tab === 'analysis' && bd && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">ABPM Parameters</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Parameter</th><th>Mean</th><th>Min</th><th>Max</th><th>Ref</th><th>Abn%</th></tr></thead>
                  <tbody>
                    {bd.abpm_summary.map((p, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold">{p.parameter}</td>
                        <td>{p.mean} {p.unit}</td>
                        <td className="small text-muted">{p.min}</td>
                        <td className="small text-muted">{p.max}</td>
                        <td className="small">{p.ref_low}–{p.ref_high}</td>
                        <td><span className={p.abnormal_pct > 20 ? 'text-danger fw-bold' : ''}>{p.abnormal_pct}%</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Holter ECG Parameters</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Parameter</th><th>Mean</th><th>Min</th><th>Max</th><th>Ref</th><th>Abn%</th></tr></thead>
                  <tbody>
                    {bd.holter_summary.map((p, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold">{p.parameter}</td>
                        <td>{p.mean} {p.unit}</td>
                        <td className="small text-muted">{p.min}</td>
                        <td className="small text-muted">{p.max}</td>
                        <td className="small">{p.ref_low}–{p.ref_high}</td>
                        <td><span className={p.abnormal_pct > 20 ? 'text-danger fw-bold' : ''}>{p.abnormal_pct}%</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Histograms */}
          {[
            { key: 'systolic_histogram',      title: '24h Systolic BP Distribution',  unit: 'mmHg', danger: 130, warn: null,   color: (b) => b.bin_start >= 130 ? 'danger' : 'success' },
            { key: 'dipping_histogram',       title: 'Nocturnal Dipping Distribution', unit: '%',   danger: 0,   warn: 10,    color: (b) => b.bin_end <= 0 ? 'danger' : b.bin_end <= 10 ? 'warning' : 'success' },
            { key: 'qtc_histogram',           title: 'QTc Interval Distribution',      unit: 'ms',  danger: 500, warn: 450,   color: (b) => b.bin_start >= 500 ? 'danger' : b.bin_start >= 450 ? 'warning' : 'success' },
            { key: 'pvc_histogram',           title: 'PVC Count Distribution',         unit: '',    danger: 500, warn: null,   color: (b) => b.bin_start >= 500 ? 'danger' : 'success' },
            { key: 'cardiac_score_histogram', title: 'Cardiac-Autonomic Risk Score',   unit: '',    danger: 50,  warn: 25,    color: (b) => b.bin_start >= 50 ? 'danger' : b.bin_start >= 25 ? 'warning' : 'success' },
          ].map(({ key, title, unit, color }) => (
            <div key={key} className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">{title}</div>
                <div className="card-body">
                  {(bd[key] || []).filter(b => b.count > 0).map((b, i) => (
                    <div key={i} className="d-flex align-items-center mb-1">
                      <span className="small" style={{minWidth: 90}}>{b.bin_start}{unit && `${unit}`}–{b.bin_end}{unit && `${unit}`}</span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{height: 16}}>
                          <div className={`progress-bar bg-${color(b)}`}
                               style={{width: `${Math.max(8, b.count / 25 * 100)}%`}}>
                            {b.count}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Patient Detail Tab ───────────────────────────────── */}
      {tab === 'patients' && bd && (
        <div>
          {(bd.patient_detail_cards || []).map((pt, i) => (
            <div key={i} className="card mb-2 shadow-sm">
              <div className="card-header d-flex justify-content-between align-items-center py-2"
                   style={{cursor:'pointer'}} onClick={() => setExpandedPt(expandedPt === i ? null : i)}>
                <span>
                  <strong>{pt.name}</strong>
                  <span className="text-muted small ms-1">({pt.age}y · {pt.disease})</span>
                  <span className={`badge bg-${sevColor(pt.severity)} ms-2`}>{pt.severity}</span>
                  <span className={`badge bg-${patColor(pt.diagnostic_pattern)} ms-1`} style={{fontSize:'0.65rem'}}>
                    {pt.diagnostic_pattern.replace(/_/g,' ')}
                  </span>
                  <span className={`badge bg-${dipColor(pt.dipping_category)} ms-1`} style={{fontSize:'0.65rem'}}>
                    {pt.dipping_category.replace(/_/g,' ')}
                  </span>
                  <span className={`badge bg-${pt.cardiac_score > 50 ? 'danger' : pt.cardiac_score > 25 ? 'warning' : 'secondary'} ms-1`}>
                    Score: {pt.cardiac_score}
                  </span>
                </span>
                <span>{expandedPt === i ? '▲' : '▼'}</span>
              </div>
              {expandedPt === i && (
                <div className="card-body">
                  <div className="row">
                    <div className="col-md-6">
                      <h6>ABPM — Blood Pressure</h6>
                      <table className="table table-sm">
                        <thead><tr><th>Parameter</th><th>Value</th><th>Ref</th><th>Status</th></tr></thead>
                        <tbody>
                          {pt.abpm && Object.entries(pt.abpm).map(([k, v]) => (
                            <tr key={k}>
                              <td className="small">{k.replace(/_/g,' ')}</td>
                              <td className={v.flag !== 'normal' ? 'text-danger fw-bold' : ''}>{v.value} {v.unit}</td>
                              <td className="small text-muted">{v.ref_low}–{v.ref_high}</td>
                              <td><span className={`badge bg-${v.flag === 'normal' ? 'success' : 'danger'}`} style={{fontSize:'0.6rem'}}>{v.flag}</span></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="col-md-6">
                      <h6>Holter — ECG</h6>
                      <table className="table table-sm">
                        <thead><tr><th>Parameter</th><th>Value</th><th>Ref</th><th>Status</th></tr></thead>
                        <tbody>
                          {pt.holter && Object.entries(pt.holter).map(([k, v]) => (
                            <tr key={k}>
                              <td className="small">{k.replace(/_/g,' ')}</td>
                              <td className={v.flag !== 'normal' ? 'text-danger fw-bold' : ''}>{v.value} {v.unit}</td>
                              <td className="small text-muted">{v.ref_low}–{v.ref_high}</td>
                              <td><span className={`badge bg-${v.flag === 'normal' ? 'success' : 'danger'}`} style={{fontSize:'0.6rem'}}>{v.flag}</span></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  {(pt.seizure_count !== undefined || pt.med_count !== undefined) && (
                    <div className="small text-muted mt-1">Seizures: {pt.seizure_count} · Medications: {pt.med_count}</div>
                  )}
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
              <div className="card-header fw-bold">{defs.test_name || 'ABPM / Holter Combined Study'}</div>
              <div className="card-body">
                {defs.protocol && (
                  <>
                    <p>{defs.protocol.description}</p>
                    {defs.protocol.recording_methods && (
                      <>
                        <h6>Recording Methods</h6>
                        <ul className="small">
                          {Object.entries(defs.protocol.recording_methods).map(([k, v]) => (
                            <li key={k}><strong>{k.toUpperCase()}:</strong> {v}</li>
                          ))}
                        </ul>
                      </>
                    )}
                    {defs.protocol.indications && (
                      <>
                        <h6>Indications</h6>
                        <ul className="small">
                          {defs.protocol.indications.map((ind, i) => <li key={i}>{ind}</li>)}
                        </ul>
                      </>
                    )}
                    {defs.protocol.standard && (
                      <p className="small text-muted fst-italic">{defs.protocol.standard}</p>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>

          {defs.parameters && (
            <>
              <div className="col-md-6 mb-3">
                <div className="card shadow-sm">
                  <div className="card-header fw-bold">ABPM Parameters ({(defs.parameters.abpm || []).length})</div>
                  <div className="card-body">
                    {(defs.parameters.abpm || []).map((p, i) => (
                      <div key={i} className="mb-2 border-bottom pb-1">
                        <strong>{p.label}</strong> <span className="text-muted">({p.unit})</span>
                        <div className="small">{p.description}</div>
                        <div className="small text-info">Ref: {p.ref_low}–{p.ref_high} {p.unit}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="col-md-6 mb-3">
                <div className="card shadow-sm">
                  <div className="card-header fw-bold">Holter ECG Parameters ({(defs.parameters.holter || []).length})</div>
                  <div className="card-body">
                    {(defs.parameters.holter || []).map((p, i) => (
                      <div key={i} className="mb-2 border-bottom pb-1">
                        <strong>{p.label}</strong> <span className="text-muted">({p.unit})</span>
                        <div className="small">{p.description}</div>
                        <div className="small text-info">Ref: {p.ref_low}–{p.ref_high} {p.unit}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}

          {defs.dipping_categories && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Dipping Categories</div>
                <div className="card-body">
                  {defs.dipping_categories.map((d, i) => (
                    <div key={i} className="mb-2">
                      <span className={`badge bg-${dipColor(d.category)} me-2`}>{d.label}</span>
                      <span className="small fw-semibold">{d.range}</span>
                      <div className="small text-muted">{d.risk}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {defs.diagnostic_patterns && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Diagnostic Patterns</div>
                <div className="card-body">
                  {defs.diagnostic_patterns.map((p, i) => (
                    <div key={i} className="mb-2">
                      <span className={`badge bg-${patColor(p.pattern)} me-2`}>{p.label}</span>
                      <span className="small">{p.description}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {defs.severity_levels && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Severity Levels</div>
                <div className="card-body">
                  {defs.severity_levels.map((s, i) => (
                    <div key={i} className="mb-2">
                      <span className={`badge bg-${sevColor(s.level)} me-2`}>{s.level}</span>
                      <span className="small fw-semibold">Score {s.score_range}</span>
                      <div className="small text-muted">{s.description}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {defs.reference_ranges && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Reference Ranges</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Parameter</th><th>Low</th><th>High</th><th>Unit</th></tr></thead>
                    <tbody>
                      {Object.entries(defs.reference_ranges).map(([k, v]) => (
                        <tr key={k}>
                          <td className="small">{typeof v === 'object' && v.label ? v.label : k.replace(/_/g,' ')}</td>
                          <td>{typeof v === 'object' ? v.low : v}</td>
                          <td>{typeof v === 'object' ? v.high : '—'}</td>
                          <td className="small">{typeof v === 'object' ? v.unit : ''}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {defs.clinical_significance && (
            <div className="col-12 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Clinical Significance</div>
                <div className="card-body small">{defs.clinical_significance}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
