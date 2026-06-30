'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const CLS_COLORS = { Lesional: 'danger', 'Non-Lesional': 'info', Equivocal: 'warning', Normal: 'success' };
const QUAL_COLORS = { Diagnostic: 'success', Adequate: 'info', Suboptimal: 'warning', 'Non-diagnostic': 'danger' };

export default function MRIReviewPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/mri-review/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/mri-review/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/mri-review/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'findings', label: 'Findings & Volumetrics' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const clsDist = overview.classification_distribution || {};
  const lesionDist = overview.lesion_type_distribution || {};
  const lobeDist = overview.lobe_distribution || {};
  const latDist = overview.laterality_distribution || {};
  const total = overview.total_patients || 0;

  return (
    <div>
      <h3>MRI Brain Review</h3>
      <p className="text-muted">Structural MRI — Epilepsy Pre-Surgical Evaluation — real data from clinical.db</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Scans', value: total, color: 'primary' },
          { label: 'Lesional Rate', value: `${overview.lesional_rate}%`, color: 'danger' },
          { label: 'HS Cases', value: overview.hippocampal_sclerosis_count, color: 'warning' },
          { label: 'Non-Lesional', value: clsDist['Non-Lesional'] || 0, color: 'info' },
          { label: 'Normal', value: clsDist['Normal'] || 0, color: 'success' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value ?? '\u2014'}</div>
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

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div className="row">
          {/* Classification Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">MRI Classification Distribution</div>
              <div className="card-body">
                {Object.entries(clsDist).map(([k, v]) => (
                  <div key={k} className="d-flex justify-content-between align-items-center mb-2">
                    <span><span className={`badge bg-${CLS_COLORS[k] || 'secondary'} me-2`}>{k}</span></span>
                    <div className="d-flex align-items-center" style={{width:'60%'}}>
                      <div className="progress flex-grow-1 me-2" style={{height:'18px'}}>
                        <div className={`progress-bar bg-${CLS_COLORS[k] || 'secondary'}`}
                             style={{width:`${total ? (v/total*100) : 0}%`}} />
                      </div>
                      <span className="fw-bold">{v}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Lesion Type Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Lesion Type Distribution</div>
              <div className="card-body">
                {Object.entries(lesionDist).sort((a,b) => b[1]-a[1]).map(([k, v]) => (
                  <div key={k} className="d-flex justify-content-between align-items-center mb-2">
                    <span className="small">{k}</span>
                    <div className="d-flex align-items-center" style={{width:'50%'}}>
                      <div className="progress flex-grow-1 me-2" style={{height:'14px'}}>
                        <div className="progress-bar bg-primary" style={{width:`${total ? (v/total*100) : 0}%`}} />
                      </div>
                      <span className="fw-bold small">{v}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Lobe Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Lesion Location (Lobe)</div>
              <div className="card-body">
                {Object.entries(lobeDist).sort((a,b) => b[1]-a[1]).map(([k, v]) => (
                  <div key={k} className="d-flex justify-content-between align-items-center mb-2">
                    <span className="small">{k}</span>
                    <div className="d-flex align-items-center" style={{width:'50%'}}>
                      <div className="progress flex-grow-1 me-2" style={{height:'14px'}}>
                        <div className="progress-bar bg-info" style={{width:`${total ? (v/total*100) : 0}%`}} />
                      </div>
                      <span className="fw-bold small">{v}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Laterality Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Laterality</div>
              <div className="card-body">
                {Object.entries(latDist).sort((a,b) => b[1]-a[1]).map(([k, v]) => (
                  <div key={k} className="d-flex justify-content-between align-items-center mb-2">
                    <span>{k}</span>
                    <div className="d-flex align-items-center" style={{width:'50%'}}>
                      <div className="progress flex-grow-1 me-2" style={{height:'18px'}}>
                        <div className="progress-bar bg-warning" style={{width:`${total ? (v/total*100) : 0}%`}} />
                      </div>
                      <span className="fw-bold">{v}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Per-patient summary table */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Per-Patient MRI Summary</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-dark">
                      <tr>
                        <th>Patient</th><th>Name</th><th>Disease</th>
                        <th>Classification</th><th>Lesion Type</th>
                        <th>Location</th><th>Laterality</th>
                        <th>HS</th><th>Quality</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(overview.patients || []).map(p => (
                        <tr key={p.patient_id}>
                          <td className="fw-bold">{p.patient_id}</td>
                          <td>{p.name}</td>
                          <td>{p.disease}</td>
                          <td><span className={`badge bg-${CLS_COLORS[p.classification] || 'secondary'}`}>{p.classification}</span></td>
                          <td>{p.lesion_type}</td>
                          <td>{p.lesion_location || '\u2014'}</td>
                          <td>{p.laterality || '\u2014'}</td>
                          <td>{p.hippocampal_sclerosis === 'Yes' ? <span className="badge bg-danger">Yes</span> : 'No'}</td>
                          <td><span className={`badge bg-${QUAL_COLORS[p.quality] || 'secondary'}`}>{p.quality}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Findings & Volumetrics Tab */}
      {tab === 'findings' && breakdown && (
        <div className="row">
          {/* Volume Asymmetry Stats */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Hippocampal Volume Asymmetry</div>
              <div className="card-body">
                <p className="text-muted small">Asymmetry index = |V<sub>left</sub> - V<sub>right</sub>| / (V<sub>left</sub> + V<sub>right</sub>). Normal &lt; 0.05, Abnormal &ge; 0.08</p>
                {breakdown.volume_asymmetry_stats && (
                  <div className="row text-center">
                    <div className="col-4">
                      <div className="h4 text-primary">{breakdown.volume_asymmetry_stats.mean}</div>
                      <div className="small text-muted">Mean</div>
                    </div>
                    <div className="col-4">
                      <div className="h4 text-danger">{breakdown.volume_asymmetry_stats.max}</div>
                      <div className="small text-muted">Max</div>
                    </div>
                    <div className="col-4">
                      <div className="h4 text-warning">{breakdown.volume_asymmetry_stats.abnormal_count}</div>
                      <div className="small text-muted">Abnormal (&ge;0.08)</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Concordance summary */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">MRI-EEG Concordance</div>
              <div className="card-body">
                {(() => {
                  const concord = {};
                  (breakdown.scans || []).forEach(s => {
                    const c = s.concordant || 'unknown';
                    concord[c] = (concord[c] || 0) + 1;
                  });
                  return Object.entries(concord).sort((a,b) => b[1]-a[1]).map(([k,v]) => (
                    <div key={k} className="d-flex justify-content-between mb-1">
                      <span className="small">{k}</span>
                      <span className="badge bg-secondary">{v}</span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>

          {/* Detailed scan table */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Detailed Scan Findings</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-dark">
                      <tr>
                        <th>Patient</th><th>Lesion</th><th>Location</th>
                        <th>T2/FLAIR</th><th>Enhancing</th>
                        <th>Asymmetry</th><th>Concordance</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(breakdown.scans || []).map(s => (
                        <tr key={s.patient_id}>
                          <td className="fw-bold">{s.patient_id}</td>
                          <td>{s.lesion_type}</td>
                          <td>{s.location ? `${s.laterality} ${s.location}` : '\u2014'}</td>
                          <td>{s.t2_flair_signal}</td>
                          <td>{s.enhancing ? <span className="badge bg-danger">Yes</span> : 'No'}</td>
                          <td>
                            <span className={s.hippocampal_volume_asymmetry >= 0.08 ? 'text-danger fw-bold' : ''}>
                              {s.hippocampal_volume_asymmetry}
                            </span>
                          </td>
                          <td className="small">{s.concordant}</td>
                          <td><span className={`badge bg-${s.confidence === 'High' ? 'success' : 'warning'}`}>{s.confidence}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Patient Detail Tab */}
      {tab === 'patients' && breakdown && (
        <div className="row">
          {(breakdown.scans || []).map(s => (
            <div key={s.patient_id} className="col-md-6 col-lg-4 mb-3">
              <div className={`card shadow-sm ${expandedPt === s.patient_id ? 'border-primary' : ''}`}
                   style={{cursor:'pointer'}} onClick={() => setExpandedPt(expandedPt === s.patient_id ? null : s.patient_id)}>
                <div className="card-header d-flex justify-content-between">
                  <span className="fw-bold">{s.patient_id}</span>
                  <span className={`badge bg-${CLS_COLORS[s.classification] || 'secondary'}`}>{s.classification}</span>
                </div>
                <div className="card-body py-2">
                  <div className="small"><strong>Name:</strong> {s.name}</div>
                  <div className="small"><strong>Lesion:</strong> {s.lesion_type}</div>
                  <div className="small"><strong>Location:</strong> {s.location ? `${s.laterality} ${s.location}` : 'N/A'}</div>
                  <div className="small"><strong>HS:</strong> {s.hippocampal_sclerosis}</div>
                  {expandedPt === s.patient_id && (
                    <div className="mt-2 pt-2 border-top">
                      <div className="small"><strong>T2/FLAIR:</strong> {s.t2_flair_signal}</div>
                      <div className="small"><strong>Enhancing:</strong> {s.enhancing ? 'Yes' : 'No'}</div>
                      <div className="small"><strong>Vol. Asymmetry:</strong> <span className={s.hippocampal_volume_asymmetry >= 0.08 ? 'text-danger fw-bold' : ''}>{s.hippocampal_volume_asymmetry}</span></div>
                      <div className="small"><strong>Quality:</strong> {s.quality}</div>
                      <div className="small"><strong>Protocol:</strong> {s.protocol}</div>
                      <div className="small"><strong>Concordance:</strong> {s.concordant}</div>
                      <div className="small"><strong>Seizure Diary:</strong> {s.seizure_diary_entries} entries</div>
                      <div className="small"><strong>Scan Date:</strong> {s.scan_date?.split('T')[0]}</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-8 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">{defs.name}</div>
              <div className="card-body">
                <p>{defs.purpose}</p>
                <h6>Epilepsy MRI Protocol</h6>
                <p className="small text-muted">{defs.protocol?.field_strength}</p>
                <ul className="small">
                  {(defs.protocol?.sequences || []).map((s, i) => <li key={i}>{s}</li>)}
                </ul>
                <p className="small text-muted">Ref: {defs.protocol?.reference}</p>
              </div>
            </div>
          </div>

          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Volumetric Analysis</div>
              <div className="card-body small">
                <p><strong>Method:</strong> {defs.volumetric_analysis?.method}</p>
                <p><strong>Normal:</strong> {defs.volumetric_analysis?.normal_range}</p>
                <p><strong>Suspicious:</strong> {defs.volumetric_analysis?.suspicious}</p>
                <p><strong>Abnormal:</strong> {defs.volumetric_analysis?.abnormal}</p>
              </div>
            </div>
          </div>

          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Lesion Types</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-dark">
                    <tr><th>Code</th><th>Label</th><th>Prevalence</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {(defs.lesion_types || []).map(lt => (
                      <tr key={lt.code}>
                        <td className="fw-bold">{lt.code}</td>
                        <td>{lt.label}</td>
                        <td>{(lt.prevalence * 100).toFixed(0)}%</td>
                        <td className="small">{lt.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Clinical Significance</div>
              <div className="card-body">
                <ul>
                  {(defs.clinical_significance || []).map((s, i) => <li key={i}>{s}</li>)}
                </ul>
                <h6 className="mt-3">References</h6>
                <ol className="small">
                  {(defs.references || []).map((r, i) => <li key={i}>{r}</li>)}
                </ol>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
