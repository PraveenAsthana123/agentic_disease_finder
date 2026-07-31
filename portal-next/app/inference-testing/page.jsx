'use client';
import {useState, useEffect} from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const kpiColor = c => ({'green':'success','red':'danger','yellow':'warning','blue':'primary','gray':'secondary'}[c]||'info');
const confBar = pct => pct >= 80 ? 'success' : pct >= 60 ? 'warning' : 'danger';

export default function InferenceTestingDashboard(){
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(()=>{
    Promise.all([
      fetch(`${API}/api/inference-testing/overview`).then(r=>r.json()),
      fetch(`${API}/api/inference-testing/breakdown`).then(r=>r.json()),
      fetch(`${API}/api/inference-testing/definitions`).then(r=>r.json()),
    ]).then(([o,b,d])=>{setOv(o);setBd(b);setDefs(d);})
      .catch(e=>setErr(String(e)));
  },[]);

  if(err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if(!ov) return <div className="text-muted p-3">Loading inference testing data...</div>;

  const TABS = [
    {id:'overview', label:'Overview'},
    {id:'models', label:'Model Leaderboard'},
    {id:'analyses', label:'Recent Inferences'},
    {id:'validation', label:'Validation Studies'},
    {id:'definitions', label:'Definitions'},
  ];

  return (<div className="p-3">
    <h3>Inference Testing Dashboard</h3>
    <p className="text-muted">
      Model benchmarks, latency analysis, prediction confidence &amp; validation studies
      &mdash; {ov.kpis?.find(k=>k.label==='Models Benchmarked')?.value} models,
      {' '}{ov.kpis?.find(k=>k.label==='Total Inferences')?.value} inferences recorded
    </p>
    {ov.honest_note && <div className="alert alert-info small py-1 px-2">{ov.honest_note}</div>}

    <ul className="nav nav-tabs mb-3">
      {TABS.map(t=><li key={t.id} className="nav-item">
        <button className={`nav-link ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id)}>{t.label}</button>
      </li>)}
    </ul>

    {/* ── Overview Tab ── */}
    {tab==='overview' && <div>
      <div className="row mb-3">
        {(ov.kpis||[]).map(k=>
          <div key={k.label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card shadow-sm h-100"><div className="card-body text-center py-2">
              <div className={`h5 mb-0 text-${kpiColor(k.color)}`}>{k.value}</div>
              <div className="text-muted small">{k.label}</div>
            </div></div>
          </div>
        )}
      </div>

      <div className="row mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Model Type Distribution</h6>
            {(ov.model_type_distribution||[]).map(d=>{
              const total = ov.kpis?.find(k=>k.label==='Models Benchmarked')?.value||1;
              const pct = ((d.count/total)*100).toFixed(0);
              return <div key={d.type} className="d-flex align-items-center mb-1">
                <span className="me-2 small" style={{minWidth:'100px'}}>{d.type}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-primary" style={{width:`${pct}%`}}>
                      {d.count} ({pct}%)
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Task Distribution</h6>
            {(ov.task_distribution||[]).map(d=>{
              const total = ov.kpis?.find(k=>k.label==='Models Benchmarked')?.value||1;
              const pct = ((d.count/total)*100).toFixed(0);
              return <div key={d.task} className="d-flex align-items-center mb-1">
                <span className="me-2 small" style={{minWidth:'120px'}}>{d.task}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-info" style={{width:`${pct}%`}}>
                      {d.count} ({pct}%)
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
      </div>

      <div className="row mb-3">
        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Inference Latency Buckets</h6>
            {(ov.latency_buckets||[]).map(d=>{
              const max = Math.max(...(ov.latency_buckets||[]).map(b=>b.count),1);
              const pct = ((d.count/max)*100).toFixed(0);
              return <div key={d.bucket} className="d-flex align-items-center mb-1">
                <span className="me-2 small" style={{minWidth:'80px'}}>{d.bucket}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-success" style={{width:`${pct}%`}}>
                      {d.count}
                    </div>
                  </div>
                </div>
              </div>;
            })}
            <div className="text-muted small mt-1">
              Range: {ov.latency_range?.min_ms}ms &ndash; {ov.latency_range?.max_ms}ms (avg {ov.latency_range?.avg_ms}ms)
            </div>
          </div></div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Accuracy Distribution</h6>
            {(ov.accuracy_buckets||[]).map(d=>{
              const max = Math.max(...(ov.accuracy_buckets||[]).map(b=>b.count),1);
              const pct = ((d.count/max)*100).toFixed(0);
              return <div key={d.bucket} className="d-flex align-items-center mb-1">
                <span className="me-2 small" style={{minWidth:'80px'}}>{d.bucket}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className="progress-bar bg-primary" style={{width:`${pct}%`}}>
                      {d.count}
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm"><div className="card-body">
            <h6>Prediction Confidence</h6>
            {(ov.confidence_buckets||[]).map(d=>{
              const max = Math.max(...(ov.confidence_buckets||[]).map(b=>b.count),1);
              const pct = ((d.count/max)*100).toFixed(0);
              return <div key={d.bucket} className="d-flex align-items-center mb-1">
                <span className="me-2 small" style={{minWidth:'80px'}}>{d.bucket}</span>
                <div className="flex-grow-1 me-2">
                  <div className="progress" style={{height:'18px'}}>
                    <div className={`progress-bar bg-${confBar(parseInt(d.bucket))}`} style={{width:`${pct}%`}}>
                      {d.count}
                    </div>
                  </div>
                </div>
              </div>;
            })}
          </div></div>
        </div>
      </div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Per-Disease Inference Performance</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr><th>Disease</th><th>Inferences</th><th>Avg Confidence</th></tr></thead>
            <tbody>
              {(ov.disease_inferences||[]).map(d=>
                <tr key={d.disease}>
                  <td>{d.disease}</td>
                  <td>{d.count}</td>
                  <td><span className={`badge bg-${confBar(d.avg_confidence)}`}>{d.avg_confidence}%</span></td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Models Tab ── */}
    {tab==='models' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Per-Task Performance</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr>
              <th>Task</th><th>Models</th><th>Avg Accuracy</th><th>Avg F1</th>
              <th>Avg AUC</th><th>Avg Latency</th><th>Min/Max Latency</th>
            </tr></thead>
            <tbody>
              {(bd.task_performance||[]).map(t=>
                <tr key={t.task}>
                  <td className="fw-bold">{t.task}</td>
                  <td>{t.model_count}</td>
                  <td><span className={`badge bg-${t.avg_accuracy>=85?'success':'warning'}`}>{t.avg_accuracy}%</span></td>
                  <td>{t.avg_f1}%</td>
                  <td>{t.avg_auc}%</td>
                  <td>{t.avg_latency}ms</td>
                  <td>{t.min_latency}&ndash;{t.max_latency}ms</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Per-Model-Type Performance</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr>
              <th>Model Type</th><th>Count</th><th>Avg Accuracy</th><th>Avg F1</th><th>Avg Latency</th>
            </tr></thead>
            <tbody>
              {(bd.model_type_performance||[]).map(t=>
                <tr key={t.model_type}>
                  <td className="fw-bold">{t.model_type}</td>
                  <td>{t.count}</td>
                  <td><span className={`badge bg-${t.avg_accuracy>=85?'success':'warning'}`}>{t.avg_accuracy}%</span></td>
                  <td>{t.avg_f1}%</td>
                  <td>{t.avg_latency}ms</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Top Models (by accuracy)</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0" style={{fontSize:'0.82rem'}}>
            <thead><tr>
              <th>Model</th><th>Type</th><th>Ver</th><th>Task</th><th>Dataset</th>
              <th>Accuracy</th><th>F1</th><th>AUC</th><th>Latency</th><th>Samples</th><th>Status</th>
            </tr></thead>
            <tbody>
              {(bd.top_models||[]).map((m,i)=>
                <tr key={i}>
                  <td className="fw-bold">{m.model_name}</td>
                  <td>{m.model_type}</td>
                  <td>{m.version}</td>
                  <td>{m.task}</td>
                  <td className="text-truncate" style={{maxWidth:'120px'}}>{m.dataset}</td>
                  <td><span className={`badge bg-${m.accuracy_pct>=90?'success':m.accuracy_pct>=80?'warning':'danger'}`}>{m.accuracy_pct}%</span></td>
                  <td>{m.f1_pct}%</td>
                  <td>{m.auc_pct}%</td>
                  <td>{m.latency_ms}ms</td>
                  <td>{m.n_samples}</td>
                  <td><span className={`badge bg-${m.status==='active'?'success':'secondary'}`}>{m.status}</span></td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Recent Inferences Tab ── */}
    {tab==='analyses' && bd && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Prediction Label Distribution by Disease</h6>
        {(bd.label_distribution||[]).map(d=>
          <div key={d.disease} className="mb-3">
            <strong>{d.disease}</strong>
            <div className="row mt-1">
              {(d.labels||[]).map(l=>
                <div key={l.label} className="col-auto mb-1">
                  <span className="badge bg-secondary me-1">{l.label}</span>
                  <span className="small text-muted">{l.count}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div></div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Recent Analyses</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr>
              <th>Patient</th><th>Disease</th><th>Prediction</th><th>Confidence</th>
              <th>Signal Quality</th><th>Date</th>
            </tr></thead>
            <tbody>
              {(bd.recent_analyses||[]).map((a,i)=>
                <tr key={i}>
                  <td>{a.patient_id}</td>
                  <td>{a.disease}</td>
                  <td className="fw-bold">{a.predicted_label}</td>
                  <td>
                    <div className="progress" style={{height:'18px',minWidth:'80px'}}>
                      <div className={`progress-bar bg-${confBar((a.confidence||0)*100)}`}
                        style={{width:`${((a.confidence||0)*100).toFixed(0)}%`}}>
                        {((a.confidence||0)*100).toFixed(1)}%
                      </div>
                    </div>
                  </td>
                  <td>{a.signal_quality}</td>
                  <td className="text-muted small">{a.created_at}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Validation Studies Tab ── */}
    {tab==='validation' && bd && <div>
      <div className="row mb-3">
        <div className="col-md-4">
          <div className="card shadow-sm h-100"><div className="card-body text-center">
            <div className="h4 text-primary">{ov.validation_summary?.total}</div>
            <div className="text-muted">Total Studies</div>
          </div></div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm h-100"><div className="card-body text-center">
            <div className="h4 text-success">{ov.validation_summary?.avg_sensitivity}%</div>
            <div className="text-muted">Avg Sensitivity</div>
          </div></div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm h-100"><div className="card-body text-center">
            <div className="h4 text-info">{ov.validation_summary?.avg_specificity}%</div>
            <div className="text-muted">Avg Specificity</div>
          </div></div>
        </div>
      </div>

      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>Validation Studies</h6>
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0" style={{fontSize:'0.82rem'}}>
            <thead><tr>
              <th>Study</th><th>Title</th><th>Type</th><th>Status</th>
              <th>N</th><th>Sensitivity</th><th>Specificity</th><th>AUC</th>
              <th>Site</th><th>PI</th>
            </tr></thead>
            <tbody>
              {(bd.validation_studies||[]).map((v,i)=>
                <tr key={i}>
                  <td className="fw-bold">{v.study_id}</td>
                  <td className="text-truncate" style={{maxWidth:'180px'}}>{v.title}</td>
                  <td>{v.study_type}</td>
                  <td><span className={`badge bg-${v.status==='completed'?'success':v.status==='active'?'primary':'secondary'}`}>{v.status}</span></td>
                  <td>{v.sample_size}</td>
                  <td>{v.sensitivity_pct}%</td>
                  <td>{v.specificity_pct}%</td>
                  <td>{v.auc_pct}%</td>
                  <td>{v.site}</td>
                  <td>{v.principal_investigator}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>
    </div>}

    {/* ── Definitions Tab ── */}
    {tab==='definitions' && defs && <div>
      <div className="card shadow-sm mb-3"><div className="card-body">
        <h6>ML / Clinical Metric Definitions</h6>
        <div className="table-responsive">
          <table className="table table-sm mb-0">
            <thead><tr><th style={{width:'200px'}}>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {(defs.terms||[]).map(t=>
                <tr key={t.term}><td className="fw-bold">{t.term}</td><td>{t.definition}</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div></div>

      <div className="row">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Status Legend</h6>
            {(defs.status_legend||[]).map(s=>
              <div key={s.status} className="mb-1">
                <span className="badge bg-secondary me-2">{s.status}</span>
                <span className="small">{s.meaning}</span>
              </div>
            )}
          </div></div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm mb-3"><div className="card-body">
            <h6>Clinical Thresholds</h6>
            {defs.clinical_thresholds && Object.entries(defs.clinical_thresholds).map(([k,v])=>
              <div key={k} className="d-flex justify-content-between mb-1 small">
                <span>{k.replace(/_/g,' ')}</span>
                <span className="fw-bold">{v}</span>
              </div>
            )}
          </div></div>
        </div>
      </div>
    </div>}
  </div>);
}
